# Spatially valid conformal prediction for air-pollution mapping in China: preregistered external validation

Code and frozen results for:

> Wang T.-Y. Preregistered external validation of spatially valid conformal prediction for air-pollution exposure mapping in China.

Preregistration: <https://osf.io/edhzw/>

## What this repo contains

- `src/` — the full analysis pipeline (Python 3.13).
- `data/processed/panel_province_year.csv` — the assembled panel: annual
  population-weighted PM2.5 (ACAG Surface PM2.5 V6.GL.03, CC BY 4.0) and NASA POWER
  meteorology for 31 provinces, 1998–2024 (837 rows, no missing values).
- `results/` — frozen confirmatory outputs, sensitivity analyses, and GP parameter
  diagnostics. The archived first-pass sensitivity file
  (`results/sensitivity/sensitivity_results_v1_pool_all_years.csv`) is retained
  unmodified for the deviation register (D-004); it is not used in the paper.
- `requirements.txt` — pinned environment.

## Reproduce

```
python -m pip install -r requirements.txt

python src/01_build_panel.py              # rebuild the panel from raw sources (optional; panel included)
python src/03_spatial_conformal.py        # implementation self-test on synthetic data
python src/04_run_frozen_analysis.py --dry-run
python src/04_run_frozen_analysis.py --real   # requires PREREG_SUBMITTED.txt (see below)
python src/05_make_figures.py                 # Figures 1 and 3
python src/06_sensitivity_analyses.py
python src/07_contour_maps.py                 # Figures 2 and 4 (contour maps; Figure 4
                                              #   requires the standard base map file)
python src/09_gp_parameter_diagnostics.py
```

`04_run_frozen_analysis.py --real` refuses to run unless a file named
`PREREG_SUBMITTED.txt` (recording the registration link and date) is present, so
the plan-before-results ordering is enforced by the code. For reproduction of the
published numbers, create that file with two lines:

```
registration: https://osf.io/edhzw/
registered: 2026-09-18
```

All outputs are deterministic (seed 20260917 for calibration sampling, seed 7 for
the gradient-boosting benchmark) and must match to bit level.

## Methods summary

- Predictor: Gaussian process (exponential covariance, parameters by marginal
  likelihood), identical for both interval methods.
- Intervals: standard split conformal (GSCP) vs the spatially weighted conformal
  procedure of Mao, Martin & Reich (2024), *JASA* 119(546):904–914 (sLSCP), both at
  nominal 90%.
- Partitions: leave-one-macro-region-out (training/calibration pool restricted to
  1998–2019, as preregistered) and leave-years-out (2020–2024 held out).
- Benchmark: `HistGradientBoostingRegressor(max_iter=200, random_state=7)` with
  split conformal prediction.

## Data sources and licences

| Source | Role | Licence |
|---|---|---|
| ACAG Surface PM2.5 V6.GL.03 | exposure outcome | CC BY 4.0 |
| NASA POWER | meteorological covariates | public domain |
| GADM v4.1 | cartography only (not redistributed) | academic use |
| WorldPop 1 km density (2020) | context in panel build | CC BY 4.0 |

Administrative boundaries are used for cartography only. The Figure 4 base map is
standard map GS(2019)1838 (Ministry of Natural Resources of the People's Republic
of China), reproduced without modification; obtain it from the official standard
map service.

## Licence

MIT — see [LICENSE](LICENSE).
