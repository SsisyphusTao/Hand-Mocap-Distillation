# Agent instructions

Hand Mocap Distillation is a personal research project. The main experiment compares supervised and feature-distilled single-stack MobRecon students. The optional WiLoR transfer experiment is unfinished.

## Read for the task

- For environment preparation, asset discovery, training, inference or analysis, read the relevant section of [SETUP.md](SETUP.md). Do not read every setup section for a prose edit.
- For model/loss/evaluation changes, consult [the single-stack experiment](experiments/single_stack/README.md). For WiLoR changes, consult [its experiment](experiments/wilor_transfer/README.md).
- Work from the repository root. Check `git status --short` and preserve existing edits, staging and private assets. Do not commit review drafts until requested.

## Editing documentation

- Keep documentation in English. The main README is a concise first-person project story: the author's motivation, original plan, visible results, how to participate and chosen next steps. Put the GIF and key results before detailed explanations.
- Experiment READMEs may use report style. Explain the question, method, matched comparison, findings and interpretation. For unfinished work, include the observed failure, diagnostic evidence and actual resource constraint that stopped it.
- Keep setup commands and asset requirements in `SETUP.md`; agent behavior belongs here. Do not put ignored-directory reminders, file-management rules, source-snapshot bookkeeping, code maps or migration narratives into experiment reports unless the user specifically needs them.
- Remove sentences that only announce where other documentation lives. Use a link when it supports the surrounding explanation; do not add navigation prose to fill a section.
- Preserve useful diagrams, GIFs, curves and result tables during prose cleanup. Shortening a report must not erase its technical explanation or reason the run remains unfinished.
- Preserve the author's intent and voice when polishing English. Do not invent personal motivations, choose research priorities on their behalf, or turn a suggestion into “I want to do this next.”
- Separate measured outcomes from hypotheses. Feature loss reduction is not pose improvement; a stopped run does not establish its planned endpoint. Keep relevant limitations near the claim they qualify without repeatedly framing the project around disclaimers.
- Do not introduce new planning/reporting files or restore removed documents without a concrete need. Update existing content and repair links after renaming headings or removing files.

## Changing code and experiments

- Prefer the smallest change that answers the current question. Reuse working code; do not add frameworks, empty base classes or wrappers for directory symmetry.
- Clean up code using callers and actual entry points. Do not delete optional experiment code merely because the main training path does not call it, and do not retain a file solely to make the layout look complete.
- Preserve paired-control and geometry semantics during refactoring. Treat new losses, data, crops, smoothing or sampling policies as new behavior or experiments, not cosmetic cleanup.
- Check code, configuration and recorded evidence before changing a numerical claim. Never fabricate assets, commands, successful runs or a cause for an observed failure.
- If real assets are missing, report the specific missing inputs and complete independent checks that are possible. Synthetic data or a passing CLI help command does not prove real training or inference works.
- Use bounded smoke runs before requested long training. Do not launch full training merely to verify documentation or refactoring, and do not overwrite an existing run to obtain a fresh check.

## Constraints to preserve

- **Paired controls:** identical retained initialization, fresh Adam and exact shared A/B augmented tensors. Teacher stays frozen in evaluation mode; student BatchNorm trains. The main trainer is the fixed 38-epoch/batch-32 recipe, with consistency starting at epoch 6. New recipes require reviewing code, config and analysis together. Preserve shared checkpoint commits after both updates and retain optimizer, RNG and next-batch state.
- **Sampling:** main training drops 13 images per epoch, recording their indices; WiLoR long training retains the tail. Do not make these recipes interchangeable during cleanup.
- **Evaluation:** truth comes from labeled joints, not cached teacher vertices. Preserve group/image split separation and frozen detection masks. `TEACHER_CHECKPOINT_SHA256` in `data.py` identifies the historical WiLoR cache, not the MobRecon teacher. Changing evaluation data/cache defines a new experiment; the main trainer also checks teacher-reference error.
- **Coordinates:** non-mirrored right-hand RGB input, `128×128`, normalized to `[-1,1]`; crop expansion 1.5×. UV is normalized crop-space in MPII order. Multiply native mesh coordinates by `0.20` for meters; regress joints with shared `j_reg.npy`, then reorder MANO → MPII. Camera axes: x right, y down, z forward. MPJPE subtracts the wrist; PA alignment allows uniform scale but no reflection. Preserve the finite zero-edge gradient branch in the edge-length loss.
- **Video:** keep source timestamps and causal previous-frame selection. Missing detections stay NaN and visibly marked; never reuse stale poses. Stored 3D joints are not wrist-subtracted. Do not add smoothing, future-frame input, per-frame scale normalization or IK without treating it as a behavior change. UV overlays are independent predictions, not calibrated 3D projections; detection confidence is not pose confidence.
- **Artifacts:** keep datasets, weights, private recordings, generated outputs and upstream repositories out of Git. Preserve original runs and published JSON. Curated media may be committed with provenance. A reused development set or teacher agreement cannot establish independent accuracy or phone performance.

## Verification and handoff

- For documentation changes, check local links and heading anchors, configuration/command references and factual consistency. Do not rerun training for prose edits.
- For Python changes, run the relevant basic checks in [SETUP.md](SETUP.md#basic-checks). Check WiLoR CLI help when changing its runners.
- Training, geometry or video changes additionally need relevant real-asset smoke, numerical comparison and playback inspection. Verify at the boundary actually changed; report anything not exercised.
- At handoff, state what changed, the commands/environment used, what passed and what remains unverified. Distinguish historical experiment results from this task's checks. Leave requested review drafts uncommitted.
