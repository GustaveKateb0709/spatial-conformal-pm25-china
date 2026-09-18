#!/usr/bin/env python3
"""
09_gp_parameter_diagnostics.py — GP parameter diagnostics (supplementary material; not a confirmatory analysis)

Background: the confirmatory outputs of script 04 do not record the estimated
Gaussian-process parameters. The Discussion explains why the sLSCP and GSCP
intervals coincide by reference to the estimated correlation length relative to
the study domain. This script refits the GP for each leave-one-macro-region-out
training set (partition construction identical to the primary pipeline:
preregistration Section 5, training pool 1998-2019, 30% calibration, seed
20260917) and records the estimated parameters.

Output: results/gp_parameter_diagnostics.csv
Note: this script is diagnostic only; it does not produce any primary result.
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import importlib.util
spec = importlib.util.spec_from_file_location("frozen", os.path.join(ROOT, "code", "04_run_frozen_analysis.py"))
frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
load_panel = frozen.load_panel
REGIONS = frozen.REGIONS
SpatialConformal = frozen.SpatialConformal
_norm_xy = frozen._norm_xy
SEED_CAL = frozen.SEED_CAL


def main():
    df = load_panel()
    rows = []
    # domain scale in normalized coordinates (diagonal), to interpret phi
    S_all = _norm_xy(df)
    span = float(np.sqrt(((S_all.max(0) - S_all.min(0)) ** 2).sum()))
    for reg, provs in REGIONS.items():
        te = df["province_en"].isin(provs)
        pool = ~te & (df["year"] <= 2019)
        rng = np.random.RandomState(SEED_CAL)
        idx = np.where(pool)[0]; rng.shuffle(idx)
        cal_idx = set(idx[:max(5, int(0.3 * len(idx)))])
        tr = df[pool & ~df.index.isin(cal_idx)].reset_index(drop=True)
        S = _norm_xy(tr); y = tr["pm25_popwt"].to_numpy(float)
        m = SpatialConformal().fit(S, y)
        rows.append(dict(
            held_out=reg, n_train=len(tr),
            sigma2=m.sig2, phi=m.phi, tau2=m.tau2, eta=m.eta,
            phi_over_span=m.phi / span,
            eta_over_span=m.eta / span,
            pct_variance_spatial=100.0 * m.sig2 / (m.sig2 + m.tau2),
        ))
        print(f"{reg:16s} n_train={len(tr):3d} phi={m.phi:.3f} eta={m.eta:.3f} "
              f"span={span:.3f} phi/span={m.phi/span:.2f}")

    out = pd.DataFrame(rows)
    out.attrs["domain_span_normalized"] = span
    out.to_csv(os.path.join(ROOT, "results", "gp_parameter_diagnostics.csv"), index=False)
    with open(os.path.join(ROOT, "results", "gp_parameter_diagnostics.json"), "w") as f:
        json.dump({"domain_span_normalized": span, "rows": rows}, f, indent=1)
    print(f"\ndomain span (normalized coordinates) = {span:.3f}")
    print("written: results/gp_parameter_diagnostics.csv / .json")


if __name__ == "__main__":
    main()
