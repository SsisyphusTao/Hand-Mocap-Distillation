"""Render MobRecon predictions on your own video, using a fixed box or a YOLO detector."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from hand_kd.data import ROOT, MANO_TO_MPII, student_crop
from hand_kd.model import load_mobrecon, load_single_stack, setup_torch

COLORS = [(90, 180, 245), (85, 200, 110), (235, 160, 80), (215, 100, 190), (125, 110, 245)]


def source_points(uv, affine):
    inverse = cv2.invertAffineTransform(affine)
    return (np.asarray(uv) * 128) @ inverse[:, :2].T + inverse[:, 2]


def validate_box(box, width, height):
    box = np.asarray(box, dtype=np.float32)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
        raise ValueError(f'Expected an in-frame x1 y1 x2 y2 box; frame is {width}x{height}')
    return box


def select_box(boxes, scores, classes, hand_class, confidence, width, height):
    for i in np.argsort(scores)[::-1]:
        if not np.isfinite(scores[i]) or scores[i] < confidence or classes[i] != hand_class:
            continue
        box = np.clip(boxes[i], [0, 0, 0, 0], [width, height, width, height])
        try:
            return validate_box(box, width, height)
        except ValueError:
            continue
    return None


def display_indices(times, duration, fps):
    display_times = np.arange(math.ceil(duration * fps)) / fps
    return np.clip(np.searchsorted(times, display_times, side='right') - 1, 0, len(times) - 1)


def video_timing(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_entries', 'frame=best_effort_timestamp_time,pkt_duration_time',
               '-of', 'json', str(path)]
    frames = json.loads(subprocess.check_output(command))['frames']
    if not frames:
        raise ValueError('The input has no video frames')
    times = np.array([float(frame['best_effort_timestamp_time']) for frame in frames])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Video timestamps must be finite and strictly increasing')
    origin = times[0]
    times -= origin
    last_duration = float(frames[-1].get('pkt_duration_time', 0))
    if not np.isfinite(last_duration) or last_duration <= 0:
        last_duration = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    return times, float(times[-1] + last_duration), float(origin)


def draw_skeleton(canvas, points):
    # Clip extreme off-image predictions before converting to OpenCV integer coordinates.
    points = np.rint(np.clip(points, -100000, 100000)).astype(int)
    for finger, color in enumerate(COLORS):
        chain = [0] + list(range(1 + 4 * finger, 5 + 4 * finger))
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(canvas, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
        for j in chain[1:]:
            cv2.circle(canvas, tuple(points[j]), 3, color, -1, cv2.LINE_AA)


def render_frame(frame, uv_source, joints, crop, timestamp):
    canvas = np.full((640, 960, 3), 22, np.uint8)
    left = canvas[:, :640]
    ratio = min(620 / frame.shape[1], 550 / frame.shape[0])
    resized = cv2.resize(frame, None, fx=ratio, fy=ratio)
    h, w = resized.shape[:2]
    x, y = (640 - w) // 2, 45 + (550 - h) // 2
    left[y:y + h, x:x + w] = resized
    cv2.putText(left, 'Source / predicted 2D landmarks', (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(left, f'{timestamp:.3f} s', (12, 625), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (230, 230, 230), 1, cv2.LINE_AA)
    if joints is None:
        cv2.putText(canvas, 'MISSING DETECTION', (665, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (70, 90, 240), 2, cv2.LINE_AA)
        return canvas
    draw_skeleton(left, uv_source * [w / frame.shape[1], h / frame.shape[0]] + [x, y])
    canvas[45:325, 660:940] = cv2.resize(crop, (280, 280))
    cv2.putText(canvas, 'Model input', (660, 26), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, '3D / wrist-relative / fixed scale', (650, 356),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (230, 230, 230), 1, cv2.LINE_AA)
    relative = joints - joints[:1]
    view = np.stack([.8660254 * relative[:, 0] + .5 * relative[:, 2], relative[:, 1]], axis=-1)
    draw_skeleton(canvas[370:620, 640:960], view * 1250 + [160, 225])
    return canvas


def publish(source, destination):
    """Publish a completed file without replacing anything at the destination."""
    os.link(source, destination)


def run(args):
    for executable in ('ffmpeg', 'ffprobe'):
        if shutil.which(executable) is None:
            raise ValueError(f'{executable} is required on PATH')
    inputs = [args.input, args.checkpoint] + ([args.detector] if args.detector else [])
    for path in inputs:
        if not path.is_file():
            raise ValueError(f'Missing input: {path}')
    destinations = [args.output] + ([args.predictions] if args.predictions else [])
    if len({p.resolve() for p in destinations}) != len(destinations):
        raise ValueError('Video and prediction outputs must be different files')
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Output already exists: {path}')
    if args.output.suffix.lower() != '.mp4':
        raise ValueError('--output must end in .mp4')
    if args.predictions and args.predictions.suffix.lower() != '.npz':
        raise ValueError('--predictions must end in .npz')
    times, duration, origin = video_timing(args.input)
    repetitions = np.bincount(display_indices(times, duration, args.fps), minlength=len(times))
    setup_torch(0)
    loader = load_single_stack if args.model == 'student' else load_mobrecon
    model = loader(ROOT, args.checkpoint, args.device).eval().requires_grad_(False)
    regressor = np.load(ROOT / 'external/mobrecon/template/j_reg.npy').astype(np.float32)
    detector = None
    if args.detector:
        from ultralytics import YOLO
        detector = YOLO(str(args.detector))
        if args.hand_class not in detector.names:
            raise ValueError(f'Unknown --hand-class; detector classes: {detector.names}')
        print(f'Detector classes: {detector.names}; using {args.hand_class}', flush=True)
    # Allocate the optional raw export only when requested. Rendering itself streams frames.
    raw = None
    if args.predictions:
        raw = {key: np.full((len(times), *shape), np.nan, np.float32) for key, shape in
               [('vertices_m', (778, 3)), ('joints_mpii_m', (21, 3)), ('uv_normalized', (21, 2)),
                ('uv_source_pixels', (21, 2)), ('boxes_xyxy', (4,)), ('crop_transform', (2, 3))]}
        raw.update(valid=np.zeros(len(times), bool), timestamps_seconds=times,
                   source_timestamp_origin_seconds=np.array(origin),
                   model=np.array(args.model), detection_mode=np.array('detector' if detector else 'fixed_box'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    with tempfile.TemporaryDirectory(prefix='.hand-demo-', dir=args.output.parent) as tmp:
        video = Path(tmp) / 'demo.mp4'
        capture = cv2.VideoCapture(str(args.input))
        with (Path(tmp) / 'ffmpeg.log').open('w+b') as error_log:
            encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', '960x640', '-r', str(args.fps), '-i', '-', '-an',
                '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(video)],
                stdin=subprocess.PIPE, stderr=error_log)
            try:
                for i, timestamp in enumerate(times):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f'Decode stopped at frame {i}; expected {len(times)} frames')
                    height, width = frame.shape[:2]
                    if detector:
                        boxes = detector.predict(frame, conf=args.confidence, classes=[args.hand_class],
                                                 device=args.device, verbose=False)[0].boxes
                        box = select_box(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                         boxes.cls.cpu().numpy(), args.hand_class, args.confidence, width, height)
                    else:
                        box = validate_box(args.bbox, width, height)
                    uv_source = joints = crop = None
                    if box is not None:
                        image, affine = student_crop(frame, box)
                        with torch.inference_mode():
                            prediction = model(torch.from_numpy(image[None]).to(args.device))
                        vertices = prediction['mesh_pred'][0].cpu().numpy() * .20
                        uv = prediction['uv_pred'][0].cpu().numpy()
                        joints = (regressor @ vertices)[MANO_TO_MPII]
                        uv_source = source_points(uv, affine)
                        if not all(np.isfinite(a).all() for a in (vertices, uv, joints, uv_source)):
                            raise ValueError(f'Non-finite model output at frame {i}')
                        crop = np.clip((image.transpose(1, 2, 0) * .5 + .5) * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
                        valid_count += 1
                        if raw is not None:
                            for key, value in [('vertices_m', vertices), ('uv_normalized', uv),
                                    ('joints_mpii_m', joints), ('uv_source_pixels', uv_source),
                                    ('boxes_xyxy', box), ('crop_transform', affine)]:
                                raw[key][i] = value
                            raw['valid'][i] = True
                    if repetitions[i]:
                        canvas = render_frame(frame, uv_source, joints, crop, timestamp)
                        for _ in range(repetitions[i]):
                            encoder.stdin.write(canvas.tobytes())
                    if (i + 1) % 100 == 0:
                        print(f'{i + 1}/{len(times)} frames; {valid_count} with predictions', flush=True)
                if capture.read()[0]:
                    raise ValueError('Decoder returned more frames than the timestamp stream')
            finally:
                capture.release()
                encoder.stdin.close()
                returncode = encoder.wait()
            if returncode:
                error_log.seek(0)
                raise RuntimeError(error_log.read().decode(errors='replace'))
        if raw is not None:
            args.predictions.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.hand-demo-', dir=args.predictions.parent) as pred_tmp:
                prediction_path = Path(pred_tmp) / 'predictions.npz'
                np.savez_compressed(prediction_path, **raw)
                publish(prediction_path, args.predictions)
        publish(video, args.output)
    print(json.dumps(dict(output=str(args.output), source_frames=len(times), valid_frames=valid_count,
                          video_frames=int(repetitions.sum()), fps=args.fps), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Compatible MobRecon state dict')
    parser.add_argument('--model', choices=['student', 'full'], default='student')
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--bbox', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'))
    location.add_argument('--detector', type=Path, help='Local Ultralytics YOLO hand-detector weights')
    parser.add_argument('--hand-class', type=int, help='Class ID for the right hand in your detector')
    parser.add_argument('--confidence', type=float, default=.5, help='Detector score threshold')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--fps', type=float, default=30, help='Output cadence; source timestamps determine playback')
    parser.add_argument('--predictions', type=Path, help='Optional raw per-source-frame .npz export')
    args = parser.parse_args()
    if args.detector and args.hand_class is None:
        parser.error('--detector requires --hand-class')
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be finite and positive')
    if not 0 < args.confidence <= 1:
        parser.error('--confidence must be in (0, 1]')
    run(args)


if __name__ == '__main__':
    main()
