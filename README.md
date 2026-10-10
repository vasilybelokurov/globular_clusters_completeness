# gc_completeness

Completeness of the Milky Way globular-cluster catalogue via orbit superposition, following
Khoperskov et al. 2025, A&A 695, A220 ([arXiv:2411.15062](https://arxiv.org/abs/2411.15062)).
The method, its assumptions and the current results are described in `CODE_SUMMARY.md`.

## Data

- GC catalogue with 6D phase space: path set in `configs/default.yaml` (`catalog.path`); not included.
- Baumgardt combined table (M_V, sizes), not redistributed here:

```bash
mkdir -p data && curl -sS -o data/baumgardt_combined_table.txt \
  https://people.smp.uq.edu.au/HolgerBaumgardt/globular/combined_table.txt
```

- Dust maps (SFD, Marshall, Bayestar19, DECaPS) for `--extinction map_low|map_high`: stored in
  `~/data/dustmaps` (path in `configs/dustmapsrc.json`, used via `DUSTMAPS_CONFIG_FNAME`); fetch with
  `dustmaps` (`pip install dustmaps`). DECaPS mean map is 8.6 GB.

## Run

```bash
source ~/Work/venvs/.venv/bin/activate        # numpy scipy astropy agama matplotlib pandas pyyaml pytest
cd "/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/Code/globular_clusters_completeness"
python -m pytest -q tests                      # unit tests
python scripts/run_khop_classes.py --mode mock # mock validation   -> outputs/khop_classes/, plots/fig40
python scripts/run_khop_classes.py --mode real # real catalogue    -> outputs/khop_classes/, plots/fig41-42
python scripts/run_adaptive.py --mode mock     # adaptive bins + symmetry, mocks -> outputs/adaptive/, plots/fig50
python scripts/run_adaptive.py --mode real     # adaptive bins + symmetry, real  -> outputs/adaptive/, plots/fig51-53
python scripts/run_cv.py --mode mock           # C(V) phase likelihood, mocks    -> outputs/cv/, plots/fig60
python scripts/run_cv.py --mode real [--extinction present_ebv|map_low|map_high]   # C(V), real -> outputs/cv/, plots/fig61-62
python scripts/make_paper_figures.py           # print-size figures for writeup/ -> plots/paper_*.png
python scripts/make_paper_figures.py --vmax map_high   # only the 1/V_max corrected-distribution figure
```

Write-up: `writeup/gc_completeness.tex` (compiled PDF alongside).

## Layout

| Path | Content |
|---|---|
| `gc_completeness/catalog.py` | Catalogue loading; M_V from Baumgardt's table |
| `gc_completeness/coordinates.py` | Galactocentric transforms (astropy v4.0 solar parameters) |
| `gc_completeness/potential.py`, `orbits.py` | AGAMA potential; orbit integration with caching |
| `gc_completeness/density.py` | Target density shapes |
| `gc_completeness/khoperskov.py` | (R, \|z\|) grid, target counts, orbit occupancy, NNLS weights |
| `gc_completeness/khop_classes.py` | Per-class completeness C(d), relative and anchored |
| `gc_completeness/adaptive.py`, `symmetry.py` | Equal-count (M_V, d) bins; near/far and north/south diagnostics |
| `gc_completeness/cv.py` | C(V) phase likelihood; separable C_V(V)·g(x) proxy model |
| `gc_completeness/dust.py` | A_V along orbits from 3D/2D dust maps; analytic dust disc for mocks |
| `gc_completeness/mocks.py` | Mock populations (AGAMA quasi-spherical DF) |
| `configs/default.yaml` | Catalogue, solar parameters, potential |
| `configs/khop_classes.yaml` | Classes, distance bins, targets, grid, mock settings |
| `data/baumgardt_combined_table.txt` | Baumgardt combined table (downloaded 2026-10-09) |
