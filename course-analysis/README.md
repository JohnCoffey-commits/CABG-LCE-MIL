# Course report analysis

These scripts produce the statistical tables and research figures for the completed ChestX-Det study. They operate on saved numerical observations; no image, model or checkpoint loading is needed.

## Public contents

- `scripts/`: the original seven analysis, plotting, verification and packaging scripts.
- `tables/`: 16 aggregate, curve-coordinate and training-step tables copied from the accepted analysis package.
- `requirements.txt`: recorded numerical and plotting dependencies.

The per-image source snapshots, image/member manifests and the two tables containing per-image identifiers (13 and 17) are supplied separately in the local course analysis attachment. They are not published in this repository. The public CSVs can be inspected directly; rerunning the original analysis requires that attachment. The historical scripts `collect_sources.py` and `package.py` expect the original workspace and are not needed in a detached copy.

## Recompute from the course attachment

Use a working copy of the supplied `course-analysis-20260920-v1` folder, install its requirements in a separate Python environment, then run:

```sh
python scripts/analyze.py
python scripts/ba_sensitivity.py
python scripts/plot_figures.py
python scripts/build_guide.py
python scripts/verify_analysis.py
```

`build_guide.py` regenerates the historical guide and README in that working copy. Retain the supplied archive separately. These commands reproduce numerical reductions and research figures, not the layout of the final Word report. The new AUROC and BA sensitivity intervals are retrospective; the original primary outcome remains generated-answer balanced accuracy.
