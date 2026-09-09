# CABG-LCE-MIL

Research code for **CABG-LCE-MIL v1.2**, developed as a University of Technology Sydney Master of IT Research Project on top of [MEDIC-AD](https://github.com/AIDASLab/Medic-AD).

CABG-LCE-MIL adds image-level supervision to the model's normal-versus-abnormal evidence during training. It combines a temperature-free logit contrast, smooth multiple-instance pooling and a gradient-budget controller while leaving the original answer-generation path unchanged at inference time.

## Source release

- `research-code/` — the complete committed source contribution relative to the pinned MEDIC-AD base, including model integration, training code, controlled experiments, CPU tests and independent verifiers.

The research export is a source release rather than an artifact archive. It excludes MRI data, patient/case manifests, protected-test material, checkpoints, optimizer state, model weights, raw execution traces and infrastructure credentials.

## Reconstruct the research source tree

The public source release is pinned to:

- upstream MEDIC-AD base: `ad62e7c910f4febad7b07030bd1c11796ae064e7`
- accepted project source: `cb5a33ca4d6d976d5f736f3f0a3217fdb5e47b2f`

Clone the upstream repository at the pinned base and apply the source overlay:

```sh
git clone https://github.com/AIDASLab/Medic-AD.git medic-ad
git -C medic-ad checkout ad62e7c910f4febad7b07030bd1c11796ae064e7
./research-code/scripts/apply-overlay.sh medic-ad
```

Alternatively, apply `research-code/patches/cabg-lce-mil-source.patch` to a clean checkout of the same upstream commit.

The environment snapshot used by the project is under `research-code/environment/`. Model weights and datasets must be obtained separately from their original providers and used under their own access terms.

## Evidence boundary

The codebase supports a substantial internal result: the corrected execution configuration reproduced two full 24-step trajectories, recovery checks and matched development-set comparisons. It also provides local causal evidence that continuing LCE after a shared checkpoint participates in maintaining late spatial concentration under the tested conditions.

These results do **not** establish patient-independent capability or clinical validity. The latest formal project state is `BLOCKED_INDEPENDENT_COHORT_READINESS`: no qualifying independent same-task cohort met the locked access, grouping, source-isolation, two-dimensional truth, rendering and precision requirements. Development-set score gains therefore remain internal evidence.

## Project briefing

The supervisor-facing research summary is available at [medic-ad-briefing.vercel.app](https://medic-ad-briefing.vercel.app/). It is hosted independently; its website source is not distributed in this repository.

## Attribution and reuse

MEDIC-AD is the upstream work by Park et al. and remains subject to its authors' terms. Individual inherited source files retain their existing copyright and licence notices. Third-party Python and JavaScript dependencies retain their respective licences.

This repository does not add a blanket licence for the research contribution. Public visibility permits inspection but does not by itself grant broader reuse rights.
