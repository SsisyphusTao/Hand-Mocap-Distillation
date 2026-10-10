"""Optional phone replay of independent 2D and raw fixed-scale 3D predictions."""
import json
import math
from pathlib import Path
import subprocess
import cv2
import numpy as np
import torch

from hand_kd.data import MANO_TO_MPII, student_crop
from hand_kd.evaluation import summarize_errors
from hand_kd.model import load_single_stack
from hand_kd.utils import sha256, save_json

ROOT = Path(__file__).resolve().parents[2]
COLORS = [(90,180,245),(85,200,110),(235,160,80),(215,100,190),(125,110,245)]


def draw_skeleton(canvas, points):
    for finger,color in enumerate(COLORS):
        chain = [0] + list(range(1+4*finger,5+4*finger))
        for a,b in zip(chain[:-1],chain[1:]):
            cv2.line(canvas,tuple(points[a]),tuple(points[b]),color,2,cv2.LINE_AA)
        for j in chain[1:]:
            cv2.circle(canvas,tuple(points[j]),4,color,-1,cv2.LINE_AA)


def phone_replay(run, output, result, baseline=None, config=None):
    """Render separate predicted 2D landmarks and raw 3D joints at fixed scale.

    The upper overlay is not a camera projection of the lower 3D geometry.
    Optional inputs let the full38 reporter reuse the historical renderer.
    """
    if baseline is None:
        raise ValueError('Verified phone baseline directory is required')
    baseline = Path(baseline)
    metadata = json.loads((baseline/'provenance.json').read_text())
    receipt = json.loads((baseline/'verification.json').read_text())
    for filename in ('raw-predictions.npz','provenance.json','animation.json'):
        assert sha256(baseline/filename) == receipt['files_sha256'][filename]
    animation = json.loads((baseline/'animation.json').read_text())
    if config is None:
        config = json.loads((run/'configuration.json').read_text())['config']
    assert metadata['checkpoint_sha256'] == sha256(Path(config['teacher_checkpoint']))
    assert sha256(animation['source_video']) == metadata['source_sha256']
    assert sha256(baseline/'detections.npz') == metadata['detections_sha256']
    detections = np.load(baseline/'detections.npz',allow_pickle=False)
    teacher = np.load(baseline/'raw-predictions.npz',allow_pickle=False)
    valid, times = detections['valid'], detections['timestamps_seconds']
    np.testing.assert_array_equal(valid,teacher['valid'])
    np.testing.assert_array_equal(times,teacher['timestamps_seconds'])
    indices = np.flatnonzero(valid)
    images, transforms = [], []
    for i in indices:
        frame = cv2.imread(str(baseline/'frames'/f'{i+1:06d}.jpg'))
        crop, affine = student_crop(frame,detections['boxes_xyxy'][i])
        images.append(crop); transforms.append(affine)
    np.testing.assert_allclose(transforms,teacher['crop_transform'][valid],rtol=0,atol=2e-5)
    images = torch.from_numpy(np.stack(images))
    vertices = [teacher['vertices_m']]
    uv = [teacher['uv_normalized']]
    for arm in ('A','B'):
        model = load_single_stack(ROOT,Path(result['arms'][arm]['final_checkpoint'])).eval().requires_grad_(False)
        mesh = np.full((len(valid),778,3),np.nan,np.float32)
        points = np.full((len(valid),21,2),np.nan,np.float32)
        with torch.no_grad():
            for start in range(0,len(indices),32):
                prediction = model(images[start:start+32].cuda())
                chosen = indices[start:start+32]
                mesh[chosen] = prediction['mesh_pred'].cpu().numpy()*.20
                points[chosen] = prediction['uv_pred'].cpu().numpy()
        vertices.append(mesh); uv.append(points)
        del model
    regressor = np.load(ROOT/'external/mobrecon/template/j_reg.npy').astype(np.float32)
    joints = np.einsum('jv,mtvc->mtjc',regressor,np.stack(vertices))[:,:,MANO_TO_MPII]
    np.savez_compressed(output/'phone-predictions.npz',model_names=['teacher','A','B'],vertices_m=vertices,
                        joints_mpii_m=joints,uv_normalized=uv,valid=valid,timestamps_seconds=times,
                        crop_transform=teacher['crop_transform'])
    assert np.isfinite(joints[:,valid]).all() and np.isnan(joints[:,~valid]).all()
    assert np.isfinite(np.asarray(uv)[:,valid]).all()
    fps = 30
    display_times = np.arange(math.ceil((times[-1]+np.median(np.diff(times)))*fps))/fps
    display_indices = np.clip(np.searchsorted(times,display_times,side='right')-1,0,len(times)-1)
    video = output/'phone-comparison.mp4'
    encoder = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720',
                                '-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    cropped = {i:np.clip((images[k].numpy().transpose(1,2,0)*.5+.5)*255,0,255).astype(np.uint8)[:,:,::-1]
               for k,i in enumerate(indices)}
    labels = ['Source recording','Full MobRecon teacher','A: single stack / GT','B: single stack / feature KD']
    previews = []
    try:
        for t,i in zip(display_times,display_indices):
            canvas = np.full((720,1280,3),22,np.uint8)
            original = cv2.imread(str(baseline/'frames'/f'{i+1:06d}.jpg'))
            ratio = min(310/original.shape[1],490/original.shape[0])
            small = cv2.resize(original,None,fx=ratio,fy=ratio)
            y,x = 60+(490-small.shape[0])//2,(320-small.shape[1])//2
            canvas[y:y+small.shape[0],x:x+small.shape[1]] = small
            for column,label in enumerate(labels):
                cv2.putText(canvas,label,(column*320+8,30),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(canvas,f'{t:.2f} s / source {i}',(10,580),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1,cv2.LINE_AA)
            for model_index in range(3):
                panel = canvas[:,(model_index+1)*320:(model_index+2)*320]
                if not valid[i]:
                    cv2.putText(panel,'MISSING DETECTION',(35,190),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,90,240),2)
                    continue
                panel[55:335,20:300] = cv2.resize(cropped[i],(280,280))
                draw_skeleton(panel,np.rint(np.asarray(uv)[model_index,i]*280+[20,55]).astype(int))
                p = joints[model_index,i]-joints[model_index,i,:1]
                view = np.stack([.8660254*p[:,0]+.5*p[:,2],p[:,1]],axis=-1)
                draw_skeleton(panel,np.rint(view*1250+[160,635]).astype(int))
                cv2.putText(panel,'3D: wrist-relative, fixed view',(12,380),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,200,200),1)
            cv2.putText(canvas,'Top: predicted 2D landmarks. Bottom: raw 3D joints, same scale; no smoothing. No phone ground truth.',
                        (10,704),cv2.FONT_HERSHEY_SIMPLEX,.52,(210,210,210),1,cv2.LINE_AA)
            encoder.stdin.write(canvas.tobytes())
            if any(abs(t-chosen)<.5/fps for chosen in (4.8,7.3666667,12.1)):
                path=output/f'phone-{t:.2f}s.jpg'; cv2.imwrite(str(path),canvas); previews.append(path.name)
    finally:
        encoder.stdin.close()
    if encoder.wait()!=0:
        raise RuntimeError('Video encoding failed')
    cap=cv2.VideoCapture(str(video)); count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release()
    assert count == len(display_times)
    proxy = {arm:summarize_errors(joints[k,valid],joints[0,valid])
             for k,arm in [(1,'A'),(2,'B')]}
    save_json(output/'phone-verification.json',dict(source_frames=len(times),valid_frames=int(valid.sum()),
              missing_frames=int((~valid).sum()),video_frames=count,fps=fps,
              timestamps_preserved_in_raw=True,display_sampling='causal previous-frame sampling at 30 fps',
              teacher_checkpoint_sha256=metadata['checkpoint_sha256'],source_sha256=metadata['source_sha256'],
              raw_sha256=sha256(output/'phone-predictions.npz'),video_sha256=sha256(video),
              teacher_agreement_only=proxy,previews=previews,
              limitations='Existing unlabeled clip; 3D joints shown without IK or inferred camera alignment; not independent accuracy'))
