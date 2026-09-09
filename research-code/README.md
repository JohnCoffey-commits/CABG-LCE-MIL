# Research source release

This directory publishes the accepted, committed CABG-LCE-MIL project source from MEDIC-AD commit `cb5a33ca4d6d976d5f736f3f0a3217fdb5e47b2f`, relative to upstream base `ad62e7c910f4febad7b07030bd1c11796ae064e7`.

## Contents

- `overlay/qwen-vl-finetune/` — data, trainer, argument and checkpoint integration changes.
- `overlay/transformers/models/qwen2_5_vl/` — Qwen2.5-VL configuration and raw A/N attention integration changes.
- `overlay/reproduction/` — committed experiment, training, analysis, test and independent-verification source from the project stages.
- `patches/cabg-lce-mil-source.patch` — the same source contribution as an applyable Git patch.
- `environment/` — the recorded Python dependency and package configuration snapshots.
- `source-manifest.sha256` — SHA-256 identities for every exported overlay file.
- `PROJECT_SOURCE_HEAD` and `UPSTREAM_BASE` — exact source identities.

The stage directories retain the actual evolution of the work:

- `stage2b`–`stage2e`: training feasibility, repeatability and baseline preparation;
- `stage2f`–`stage2g`: FB-MAQ engineering and anomaly-specific pilot;
- `stage2h`: LAD-MIL mechanism and calibration gates;
- `stage2i`–`stage2j`: CABG-MIL diagnostics and v1.2 redesign;
- `stage2k`–`stage2m`: D3R, pilot and full Scout execution;
- `stage2n`–`stage2p`: stability repairs, shared-parent causal branch and fixed cutoff;
- `stage2q`, `stage2s`–`stage2aa`: reproducibility, controller/LCE mechanism, baselines and timing studies;
- `stage2ab`–`stage2ad`: capability-pipeline and independent-cohort readiness code.

`stage2r` is deliberately absent. In the source-of-record repository it remains untracked preparation rather than accepted committed code, so it was not silently promoted into this release.

## Deliberate exclusions

This release contains source and small execution configuration files only. Binary tensor fixtures, dataset/member manifests, MRI images, checkpoints, model weights, optimizer/RNG state, raw traces, private source records and host-specific formal execution protocols are excluded. Consequently, the repository exposes the complete accepted implementation but cannot reproduce numerical results without the separately governed data, weights and immutable artifacts.

## Integrity check

From the repository root:

```sh
sha256sum -c research-code/source-manifest.sha256
```

On systems without `sha256sum`, use an equivalent SHA-256 checker. The patch and manifest are generated from the pinned Git identities above, not from a remote runtime directory.
