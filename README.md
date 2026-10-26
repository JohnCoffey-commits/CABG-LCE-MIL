# CABG LCE MIL research implementation

Research code supporting Peixian Zheng's UTS 32933 Research Project report, built on [MEDIC-AD](https://github.com/AIDASLab/Medic-AD). This release includes the completed ChestX-Det experiments, mechanism diagnostics and spatial evidence transport study used in the final report.

The implementation combines image-level LCE supervision and a MIL anomaly readout, CABG gradient budgeting, and an additional differentiable spatial value-aggregation route. In the matched ChestX-Det study, fixed LCE improved auxiliary anomaly-score AUROC relative to LM-only in both retained seeds (0.8005/0.8037 versus 0.6033/0.3619). CABG enforced the specified auxiliary-gradient budget. Spatial transport was implemented and evaluated with an aligned/shuffled control.

## Start here

- [Report code map](REPORT_CODE_MAP.md): contributions, source files and experiment entry points.
- [Research source](research-code/README.md): complete source overlay, reconstruction patch and source inventory.
- [Course analysis](course-analysis/README.md): numerical analysis scripts and public aggregate tables.
- [Release notes](RELEASE_NOTES.md): scope and validation for `uts32933-report-20261026-v1`.

## Reconstruct the source tree

```sh
git clone https://github.com/AIDASLab/Medic-AD.git medic-ad
git -C medic-ad checkout ad62e7c910f4febad7b07030bd1c11796ae064e7
./research-code/scripts/apply-overlay.sh medic-ad
```

Alternatively apply `research-code/patches/cabg-lce-mil-source.patch` to the same clean base. Both routes include stage2af/ag/ah. The source inventory is checked with `python3 research-code/scripts/verify_release.py`.

The historical committed project HEAD is `cb5a33ca4d6d976d5f736f3f0a3217fdb5e47b2f`. Later report-era code was maintained as untracked source in the research checkout and is included explicitly in this release. Its identity is recorded in `research-code/SOURCE_RELEASE.json` and `research-code/source-manifest.sha256`; the old HEAD alone does not identify the full report implementation.

## Execution and evidence

The reported experiments used 1,000 training images, 128 development images and 542 final images, with seeds 43 and 44. Stage2af contains three matched conditions at 1,000 updates each; stage2ag provides local inference interventions; stage2ah adds aligned/shuffled transport training under matched settings. Configuration and entry points are documented in the code map.

Auxiliary-score discrimination and generated-answer performance are distinct outcomes. CABG and transport did not establish a consistent additional generated-answer advantage across both seeds. The development and final images were reused, patient identity and exact pretraining overlap remain unknown, and these findings do not establish clinical validity or patient-independent generalization. The earlier MRI independent-cohort readiness block remains a historical result; it is not the completion status of the later ChestX-Det study.

Model re-execution additionally needs the original model, admitted data, execution environment and protocol artifacts. No model training or inference was run to prepare this source release. The local course analysis attachment supports CPU-only numerical recomputation without model or image files.

## Attribution and reuse

MEDIC-AD supplies the inherited architecture, Lingshu supplies the base model and ChestX-Det supplies the images and labels. Inherited source files retain their copyright and licence notices. Dependencies and data remain subject to their providers' terms. This repository does not add a blanket licence for the research contribution.

The public release excludes image/member manifests, per-image source records, image caches, weights, checkpoints, optimizer state, raw gradients and private infrastructure details. SSH connection values in the three newly exported transport helpers are placeholders; the scientific computation is unchanged.
