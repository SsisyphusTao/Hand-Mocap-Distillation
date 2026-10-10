# WiLoR → full MobRecon feature distillation

This experiment tests whether intermediate features from a frozen WiLoR teacher improve a full MobRecon student under matched supervised training. The student retains all 12.11M parameters.

The short comparison completed 256 updates. A planned ten-epoch comparison stopped at about 2.31 epochs: feature loss decreased, but the completed runs did not improve joint accuracy over supervision alone.

## Method

Teacher and student receive two corresponding augmented views of each image. MobRecon uses `128×128` RGB crops; WiLoR uses the central `256×192` portion of the corresponding higher-resolution crop with its own normalization.

The target is WiLoR's last normalized backbone feature **before pose refinement**, pooled from `1280×16×12` to `1280×8×6`. Four MobRecon stages are sampled onto an `8×6` grid covering the central 75% of their width. An alignment head projects them to 128 channels, adds position embeddings and LayerNorm, then aggregates stages and projects to 1280 channels. The regional version additionally applies four-head self-attention across stages and spatial regions.

This adapts PAT's regional-attention idea without adaptive teacher prompts. Spatial correspondence uses normalized feature-map extents; it does not establish identical CNN receptive fields and ViT token centers.

```text
L = L_native_paired_supervision + 0.1 × L_feature
```

`L_feature` is channel-normalized MSE averaged over both views, with detached teacher targets. The teacher runs in FP32; pooled targets are stored as float16 and compared in FP32. The alignment head is used only during training.

## Experiment design

| Setting | Short comparison | Long comparison |
| --- | --- | --- |
| Groups | Supervised / projected head / regional head | Supervised / fresh regional head / prefitted regional head |
| Training images | 8,192 sampled without replacement | All 237,517 admitted FreiHAND/CompHand images per epoch |
| Planned duration | 256 updates per group | 10 epochs, 7,423 updates per epoch per group |
| Teacher targets | Cached once and reused by all groups | Computed online for each shared batch |
| Batch size | 32 | 32, accumulated in microbatches of 8 |
| Tail handling | Complete batches | Retain the final 13 images; weight microbatch gradients by sample count |
| Evaluation endpoint | Fixed update 256 | Intended fixed epoch 10; not reached |

All groups start from the same supervised MobRecon epoch-9 checkpoint, with seed `20260917`, fresh Adam at `1e-5`, zero weight decay and fixed BatchNorm statistics. They consume identical paired inputs and use native mesh, UV, normal, edge and two-view consistency supervision. The prefitted head has additional training history, so its total compute is not matched to the fresh head.

Evaluation uses the same reused 256-image development set and frozen detection crops. Pose improvement is judged against the supervised control using MPJPE and PA-MPJPE.

## Failure analysis and local compute limits

**The observed failure was a mismatch between feature fit and pose quality.** The feature-trained students did not outperform the matched supervised student in the completed comparisons. A head-only diagnostic could reconstruct teacher features from an unchanged student; a later backbone-only diagnostic reduced feature error while joint error increased. These small, reused-data checks suggest that fitting the target did not reliably teach better geometry.

Our working explanation is that the target sits before WiLoR's MANO-based pose refinement, while pooled feature MSE neither emphasizes joint-critical regions nor directly supervises the student's final decoder output. Sampled feature-loss gradients were also small relative to supervised gradients and nearly orthogonal to them. This supports investigating the target and its influence on training; it does not establish a single cause or show that WiLoR knowledge cannot transfer.

**The ten-epoch experiment remains unfinished because of the local compute budget.** The machine used an **RTX 3060 Ti with 8 GB VRAM**. Student microbatches of eight made the three-arm run feasible, but the measured throughput initially implied about **75 hours** for ten epochs. At roughly 2.31 epochs, the run still estimated about **58 hours remaining**, with no observed accuracy benefit. I stopped it rather than spend that remaining compute on the same objective. This was a deliberate stop, not an out-of-memory termination of this run; the unexecuted epochs could have changed the outcome.

## Reproduction

Run commands from the repository root in a Python 3.10 environment with PyTorch 2.5.1 + CUDA 12.4, torchvision 0.20.1 and [training dependencies](../../requirements-train.txt). HandMesh must be checked out at `3c87e958d4855f890e3884ec94bcfc0f99422c3d` in `external/mobrecon/`.

```bash
.venv/bin/python -m pip install -r experiments/wilor_transfer/requirements-teacher.txt
git clone https://github.com/rolpotamias/WiLoR.git external/wilor
git -C external/wilor checkout fcb911312a38fa8badd30d9656a167485d61b8f9
```

Required assets are the training manifest/selection, initial MobRecon weights, evaluation manifest/cache, WiLoR weights at `models/wilor/wilor_final.ckpt`, and WiLoR's `pretrained_models/model_config.yaml` and `mano_data/`. These are obtained separately under their upstream terms.

Copy the [configuration examples](configs/) to same-directory `*.local.json` files and replace asset paths and hashes, including the pilot's nested `validation` object. For the long run, point `parent_config` and `prefit_config` to the local files. It also requires `<prefit_run>/frozen-head.pt`; no generator for that head-only prefit asset is included, and the pilot's jointly trained head is not a substitute.

Start with smoke runs in new output directories:

```bash
.venv/bin/python -m experiments.wilor_transfer.pilot \
  --config experiments/wilor_transfer/configs/pilot.local.json \
  --output outputs/wilor-pilot-smoke-new --smoke

.venv/bin/python -m experiments.wilor_transfer.long_run \
  --config experiments/wilor_transfer/configs/long.local.json \
  --output outputs/wilor-long-smoke-new --smoke --stop-after-batches 2
```

For the 256-update pilot or ten-epoch comparison, remove the smoke flags. The long runner supports `--resume` for an existing run.
