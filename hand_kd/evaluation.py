"""Root-relative metric definitions and fixed labeled evaluation."""
import numpy as np
import torch
from hand_kd.data import MANO_TO_MPII
from hand_kd.utils import write_json


def similarity_align(predicted, target):
    predicted, target = np.asarray(predicted, np.float64), np.asarray(target, np.float64)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError('Expected matching B x J x 3 arrays')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite evaluation geometry')
    pm, tm = predicted.mean(1, keepdims=True), target.mean(1, keepdims=True)
    x, y = predicted-pm, target-tm
    variance = (x*x).sum((1,2))
    if (variance < 1e-15).any():
        raise ValueError('Degenerate prediction in PA alignment')
    u, singular, vh = np.linalg.svd(x.transpose(0,2,1) @ y)
    signs = np.ones_like(singular)
    signs[:, -1] = np.where(np.linalg.det(u@vh) < 0, -1, 1)
    rotation = (u * signs[:,None,:]) @ vh
    scale = (singular * signs).sum(1) / variance
    return scale[:,None,None] * (x @ rotation) + tm


def summarize_errors(predicted, target):
    predicted, target = np.asarray(predicted), np.asarray(target)
    if predicted.shape != target.shape or predicted.shape[1:] != (21,3) or len(predicted) == 0:
        raise ValueError('Expected nonempty B x 21 x 3 predictions and labels')
    predicted, target = predicted-predicted[:,:1], target-target[:,:1]
    errors = np.linalg.norm(predicted-target,axis=-1)*1000
    pa_errors = np.linalg.norm(similarity_align(predicted,target)-target,axis=-1)*1000
    fingers = {name:float(errors[:,1+4*i:5+4*i].mean()) for i,name in enumerate(['thumb','index','middle','ring','pinky'])}
    return dict(samples=len(errors), mpjpe_mm=float(errors.mean()),
                median_sample_mpjpe_mm=float(np.median(errors.mean(1))),
                p95_joint_error_mm=float(np.percentile(errors,95)),
                pa_mpjpe_mm=float(pa_errors.mean()), per_finger_mpjpe_mm=fingers)


def evaluate(model, loader, regressor, device, path):
    model.eval()
    predicted, truth, ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            vertices = model(batch['image_student'].to(device))['mesh_pred'] * .2
            joints = torch.einsum('jv,bvc->bjc', regressor, vertices)[:, MANO_TO_MPII]
            predicted.append(joints.cpu().numpy())
            truth.append(batch['joints_mpii_m'].numpy()); ids.extend(batch['sample_id'])
    predicted, truth = np.concatenate(predicted), np.concatenate(truth)
    # Retain the raw failure evidence even when alignment/metrics reject geometry.
    with path.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, prediction=predicted, truth=truth, ids=ids)
    metrics = summarize_errors(predicted, truth)
    write_json(path.with_suffix('.json'), metrics)
    return metrics
