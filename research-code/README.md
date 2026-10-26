# Research source release

Release `uts32933-report-20261026-v1` combines the previously published committed contribution with the report-era stage2af/ag/ah implementation. `PROJECT_SOURCE_HEAD` continues to record the historical committed baseline (cb5a33ca4d6d976d5f736f3f0a3217fdb5e47b2f); `SOURCE_RELEASE.json` records the additional source snapshot and the three connection-placeholder substitutions.

## Contents

- `overlay/`: model integration, training, evaluation, diagnostics, numerical checks and their retained historical dependencies.
- `patches/cabg-lce-mil-source.patch`: the same exported contribution as a patch against upstream `ad62e7c910f4febad7b07030bd1c11796ae064e7`.
- `environment/`: historical dependency snapshots; the current reported environment is also specified in `report-configuration.json`.
- `source-manifest.sha256`: identities of every exported overlay file.
- `scripts/apply-overlay.sh`: installs the overlay on the pinned clean upstream checkout.
- `scripts/verify_release.py`: checks the inventory and Python syntax without importing model code.

## Report-era stages

`stage2af` implements the 1,000-update ChestX-Det comparison, preprocessing, fixed LCE calibration, linear-head comparator and independent training/result checks. `driver_v2.py` selects `train.py` for accepted seed43 attempt-v1 and `train_v2.py` for accepted seed44 attempt-v2. Both versions are required; the difference concerns the frozen initialization reference and recorded engineering amendment.

`stage2ag` implements the local mapping, spatial-shuffle, token-swap and fixed prior-odds interventions. `stage2ah` implements spatial value transport, its matched aligned/shuffled training, evaluation and separate results verification. Earlier modules remain because these stages import the LCE, controller, runtime, deterministic execution and linear-classifier implementations from them. Stage2r and the separate historical stage2ae are outside this report release.

## Running and inspecting

Read [the report code map](../REPORT_CODE_MAP.md) before choosing an entry point. The original orchestration helpers are preserved for inspection and expect an established execution environment, original workspace layout and frozen protocol artifacts. They are not one-command launchers for an arbitrary computer. SSH target/socket placeholders must be configured for a separately authorized new execution. No deployment, training or inference is part of source reconstruction or inventory verification.

The scientific implementations are unchanged. Dataset images, prepared tensors, model/checkpoint bodies, private manifests and host-specific frozen protocols are not included. The public source and separately supplied analysis attachment serve different purposes: implementation inspection and CPU numerical recomputation respectively.
