# Project setup

Run commands from the repository root. Replace placeholder paths with your local asset paths.

## Environment

Prefer an existing compatible environment. Use Python 3.10 and PyTorch 2.5.1; training requires CUDA. For asset-free CPU tests:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-core.txt
```

If `ensurepip` is missing, install `python3.10-venv`, or use `uv venv --python python3.10 --seed .venv`. Training invokes pip to record its environment, so retain pip in the virtual environment.

For training or model inference, install the additional dependencies and clone HandMesh yourself:

```bash
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements-train.txt
git clone https://github.com/SeanChenxy/HandMesh.git external/mobrecon
git -C external/mobrecon checkout 3c87e958d4855f890e3884ec94bcfc0f99422c3d
```

Check revision and local modifications before changing an existing upstream checkout. `openmesh` may need a C++ compiler, CMake and Python development headers. Video commands need `ffmpeg` and `ffprobe` on PATH. The recorded training hardware is an RTX 3060 Ti with 8 GB VRAM; check available memory before GPU runs.

## Code entry points

| Task | Entry point |
| --- | --- |
| Student architecture and weight loading | `hand_kd/model.py` |
| Dataset loading, selection and augmentation | `hand_kd/data.py` |
| Supervision and feature losses | `hand_kd/losses.py` |
| Paired training and checkpoint resume | `python -m hand_kd.train` |
| Joint metrics and prediction export | `hand_kd/evaluation.py` |
| Own-video inference | `python -m hand_kd.demo` |
| Completed-run analysis | `python -m experiments.single_stack.analyze` |
| Optional WiLoR comparison | `experiments/wilor_transfer/{features,support,pilot,long_run}.py` |

## Assets and training

`configs/train.json` contains parameters only. Pass local paths on the CLI; relative paths resolve from the repository root. Old `configs/*.local.json` files are historical references and are not loaded automatically. Do not put machine-specific paths in shared configuration.

Before training, locate these real inputs and validate their referenced paths:

| Input | Required structure |
| --- | --- |
| Training manifest | Schema 1, pinned HandMesh revision, FreiHAND/CompHand ZIP paths and hashes, FreiHAND camera/joint/vertex arrays, and auxiliary-file hashes. |
| FreiHAND arrays | `K.npy`, `gt_joints.npy`, `gt_vertices.npy` for 32,560 base samples shared across four backgrounds; original-author mesh vertices in camera coordinates and meters. |
| CompHand auxiliary files | `comphand-records.json` and `comphand-vertices.npy` beside the manifest, with image/mask/mesh correspondence. |
| Selection | `selection.json` and referenced `admitted-indices.npy`; hashes bind parent manifest and indices. Admitted indices must be the exact complement of declared exclusions. |
| Teacher | This project's supervised full MobRecon epoch-9 checkpoint. |
| Evaluation | Labeled sample manifest and historical WiLoR cache: frozen boxes, target NPZ files, and manifest/regressor/checkpoint hashes bound by `index.json`. |
| Templates | HandMesh `template/j_reg.npy`, `right_faces.npy`, `transform.pkl`; regressor must match the evaluation cache. |

The original selection admits 130,236 FreiHAND and 107,281 CompHand images. Recover exact exclusions from selection, not counts alone: one FreiHAND base sample loses all four backgrounds, and CompHand excludes erroneous `base_pose/…/2` poses. Mirror CompHand OBJ x only once, using the image/coordinate conventions in `data.py`. Keep training ZIPs compressed; the loader opens them per worker.

Teacher/student weights, evaluation caches and raw-download preparation scripts are not distributed. If only raw datasets are available, recover or implement manifest preparation and verify sample correspondence before launching training. Do not substitute synthetic inputs or invent an available preparation command.

Set actual paths, then check input identities and run a bounded smoke:

```bash
train_assets=(
  --dataset-manifest /path/to/training/manifest.json
  --selection /path/to/selection.json
  --teacher /path/to/epoch-008-model.pt
  --validation-manifest /path/to/validation/manifest.json
  --validation-cache /path/to/teacher-cache
)
.venv/bin/python -m hand_kd.train "${train_assets[@]}" --check-inputs

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hand_kd.train \
  --config configs/train.json "${train_assets[@]}" --output outputs/smoke-new \
  --smoke --stop-after-batches 32
```

`--check-inputs` hashes assets, including ZIPs; it does not test decoding or forward passes. Smoke covers both consistency-loss branches without completing an epoch. For requested full training, use a separate new output directory without smoke flags:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hand_kd.train \
  --config configs/train.json "${train_assets[@]}" --output outputs/full38-new

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hand_kd.train \
  --config configs/train.json "${train_assets[@]}" --output outputs/full38-new --resume
```

Preserve config, source and asset identities when resuming. Check a retained `STOP` file's reason before removing it. Historical runs require their original source snapshots; never modify hashes to bypass resume checks. If asset relocation changes a manifest, record the new identity and migration source.

For WiLoR work, additionally install `experiments/wilor_transfer/requirements-teacher.txt` and clone WiLoR at `fcb911312a38fa8badd30d9656a167485d61b8f9` into `external/wilor/`. Its loader requires `pretrained_models/model_config.yaml`, `mano_data/`, and weights at repository-root `models/wilor/wilor_final.ckpt`. Configure ignored same-directory `*.local.json` recipes. The long comparison also needs a separately supplied prefitted head; no head-only prefit generator is included. Do not use the pilot's jointly trained head as a substitute.

## Inference and analysis

With compatible student weights and HandMesh installed, fixed-box inference needs neither training data nor evaluation cache:

```bash
.venv/bin/python -m hand_kd.demo \
  --input /path/to/video.mp4 --checkpoint /path/to/B-student.pt \
  --bbox 200 120 600 520 --output outputs/demo-new.mp4 \
  --predictions outputs/demo-new.npz
```

Replace the box with actual source-pixel coordinates. Default device is CPU; add `--device cuda` for GPU, or `--model full` for full MobRecon weights. For detection, install `ultralytics==8.1.34` and replace `--bbox` with `--detector /path/to/hand-detector.pt --hand-class CLASS_ID`. Use the detector's actual right-hand class; COCO weights have no hand class. Fixed-box mode does not detect hand absence. Use new output filenames.

For a completed 38-epoch run with predictions, weights and logs:

```bash
.venv/bin/python -m experiments.single_stack.analyze \
  --run /path/to/completed-run --output outputs/analysis-new
```

Historical recording replay additionally requires `--phone-baseline /path/to/teacher-replay`, its provenance/verification/animation JSON, raw predictions, detections, frames and referenced private source video. Use the demo for new recordings. Published media provenance and GIF rebuild arguments are in `figures/readme-media.json`.

## Basic checks

For Python changes, run the asset-free checks in a compatible environment:

```bash
.venv/bin/python -m compileall -q hand_kd experiments tests
.venv/bin/python -m hand_kd.train --help
.venv/bin/python -m hand_kd.demo --help
.venv/bin/python -m experiments.single_stack.analyze --help
.venv/bin/python -m unittest discover -s tests -v
```

