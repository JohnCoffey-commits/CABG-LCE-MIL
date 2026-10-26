# Report source release uts32933-report-20261026-v1

Prepared on 26 October 2026 for the UTS 32933 final report.

## Changes

Added all 79 Python files from the completed stage2af/ag/ah study directories, preserving accepted train/train_v2 variants and inherited dependencies. Added the seven course-analysis scripts, 16 public summary/curve/training tables, report configuration and a contribution-to-file map. Updated the reconstruction patch, overlay inventory and project overview to cover the completed ChestX-Det work.

Three local transport helpers replace a private SSH endpoint and socket path with explicit placeholders. Computation, optimization, metrics and experimental controls are unchanged. Original research artifacts and the source-of-record checkout remain unchanged.

## Verification

The release preparation checks Python syntax and inventory coverage, applies both reconstruction routes to the pinned upstream source and compares their outputs, checks local import availability, and reruns the saved-results numerical analysis in a separate local copy. The corresponding validation record is `release-validation.json`. No model, GPU, data admission, training or inference is run for this release.

## Attachments

The course delivery contains a source ZIP, a numbered program-code listing PDF, and the separate existing course analysis ZIP with its 26 saved-results source snapshots. Per-image records and manifests are kept out of this public repository. The code-listing PDF covers the report-era stages, their central LCE/CABG modules and all seven course-analysis scripts; the source ZIP additionally retains inherited integration and runtime dependencies.

Use the published commit URL as the immutable version reference; `main` can change later.
