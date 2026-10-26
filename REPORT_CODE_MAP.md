# Report code map

Paths below are relative to `research-code/overlay/` and become paths in the reconstructed MEDIC-AD tree.

| Report component | Main implementation | Supporting files |
|---|---|---|
| LCE loss and MIL readout | `reproduction/stage2k/producer_math.py` | stage2k constants and label contract; stage2af metrics |
| CABG gradient budget | `reproduction/stage2l/controller.py` | `stage2i/cabg_math.py`, stage2l constants |
| LM-only, fixed LCE and CABG comparison | `reproduction/stage2af/core.py` | `train.py`, `train_v2.py`, `runtime.py`, `calibrate.py` |
| Data and input contract | `reproduction/stage2af/admit_data.py`, `inputs.py` | input/admission verifiers; inherited official model wrapper |
| Generation and linear-head evaluation | `reproduction/stage2af/evaluate.py`, `linear_head.py`, `metrics.py` | accepted input contract and stage2ac linear solver |
| Appendix C local interventions | `reproduction/stage2ag/interventions.py`, `diagnose_gpu.py`, `diagnose_prior.py` | `reduce_diagnostics.py`, `verify_diagnostics.py`, `verify_prior.py` |
| Spatial transport and shuffle ablation | `reproduction/stage2ah/transport.py` | `train.py`, `evaluate.py`, `verify_training.py`, `verify_results.py` |
| Report numerical reductions and figures | `course-analysis/scripts/` at repository root | source snapshots in the separate course analysis attachment |

## Main entry points in the original execution environment

- Accepted main-study coordinator: `python -m reproduction.stage2af.driver_v2 --root <new-study-root> --name <job-name>`.
- Main-study training: stage2af `train` for seed43 and `train_v2` for seed44; both consume the frozen protocol and initial-state artifacts.
- Mechanism diagnosis: stage2ag `diagnose_gpu` and `diagnose_prior` consume the accepted fixed-LCE endpoint and permitted development inputs.
- Transport training/evaluation: stage2ah `train` and `evaluate` consume the matched stage2af runtime and protocol dependencies; see their argument parsers for required paths.
- Numerical report reproduction: see `course-analysis/README.md`; no GPU is needed for the supplied saved-results analysis.

The commands describe recorded entry points, not a complete model-reproduction recipe. Source reconstruction does not provide model weights, data caches, execution seals or a new GPU authorization. The original main-study engineering amendments select seed43 attempt-v1 and seed44 attempt-v2, with step1000 fixed as the final endpoint.
