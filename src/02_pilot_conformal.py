#!/usr/bin/env python3
"""
02_pilot_conformal.py - baseline-only pilot run.

Purpose (both parts matter)
  1) to calibrate the pass/fail thresholds frozen in the preregistration;
  2) to confirm the pipeline runs end to end.
  These outputs are NOT study results. Study results are produced only after the
  preregistration is registered, by the frozen script code/04_run_frozen_analysis.py.
  This script and its outputs must not be reported as findings.

Design
  Target       y = population-weighted PM2.5 (ACAG, province level)
  Features     X = year, temperature, precipitation, relative humidity, population
                   (provincial-capital point values as a v1 proxy)
  Model        gradient boosting regressor (scikit-learn)
  Partition A  leave-macro-region-out: each macro-region held out in turn
                 calibration = random 30% of the other regions, 1998-2019
                 training    = the remainder of the other regions, 1998-2019
                 test        = the held-out macro-region (all years; and 2020-2024 separately)
  Partition B  leave-years-out: 2020-2024 held out
                 calibration = random 30% of the remaining years; training = the rest
  Intervals    standard split conformal, nominal 90% (alpha = 0.10)
  Metrics      empirical coverage, mean interval half-width
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, "data", "processed", "panel_province_year.csv")
OUTD = os.path.join(ROOT, "data", "processed")
ALPHA = 0.10
RNG = np.random.RandomState(20260917)  # fixed seed: the pilot must also be reproducible

def qhat(resid, alpha=ALPHA):
    n = len(resid)
    if n < 5:
        return np.nan
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return np.sort(resid)[k - 1]

def split_conformal(train_X, train_y, cal_X, cal_y, test_X, test_y, model=None):
    model = model or HistGradientBoostingRegressor(max_iter=200, random_state=7)
    model.fit(train_X, train_y)
    pc = model.predict(cal_X)
    pt = model.predict(test_X)
    q = qhat(np.abs(cal_y - pc))
    cov = float(np.mean(np.abs(test_y - pt) <= q)) if len(test_y) else np.nan
    half = float(np.mean(q)) if not np.isnan(q) else np.nan
    return cov, half, len(train_X), len(cal_X), len(test_y)

def main():
    df = pd.read_csv(PANEL)
    df = df.dropna(subset=["pm25_popwt"]).copy()
    feats = ["year", "t2m_mean_c", "prec_total_mm", "rh_mean_pct", "pop_million"]
    d = df.dropna(subset=feats).copy()
    print(f"Usable rows: {len(d)} / {len(df)} (dropped {len(df) - len(d)} for missing features)")
    X = d[feats].to_numpy()
    y = d["pm25_popwt"].to_numpy()
    region = d["macro_region"].to_numpy()
    year = d["year"].to_numpy()
    results = []

    print("\n== A. leave-macro-region-out ==")
    print(f"{'region':<18}{'n_test':>7}{'coverage':>10}{'half-width':>12}   (nominal 90%)")
    for reg in sorted(set(region)):
        m_te = region == reg
        m_tr = (~m_te) & (year <= 2019)
        tr_idx = np.where(m_tr)[0]
        cal_idx = np.where((~m_te) & (year <= 2019))[0]
        rng = np.random.RandomState(12345)
        rng.shuffle(cal_idx)
        cal_idx = cal_idx[:max(5, int(0.3 * len(cal_idx)))]
        tr_idx = np.array([i for i in tr_idx if i not in set(cal_idx)])
        te_idx = np.where(m_te)[0]
        cov, half, _, _, _ = split_conformal(X[tr_idx], y[tr_idx], X[cal_idx], y[cal_idx],
                                             X[te_idx], y[te_idx])
        results.append(["A_leave_region", reg, "all_years", len(te_idx), cov, half])
        print(f"{reg:<18}{len(te_idx):>7}{cov:>10.2%}{half:>12.1f}")

    print("\n== A2. leave-macro-region-out, test restricted to 2020-2024 ==")
    for reg in sorted(set(region)):
        te = np.where((region == reg) & (year >= 2020))[0]
        if len(te) < 5:
            print(f"{reg:<18} too few observations, skipped")
            continue
        m_tr = (~(region == reg)) & (year <= 2019)
        tr_idx = np.where(m_tr)[0]
        cal_idx = np.where((~region == reg) & (year <= 2019))[0] if False else np.where((~(region == reg)) & (year <= 2019))[0]
        rng = np.random.RandomState(12345)
        rng.shuffle(cal_idx)
        cal_idx = cal_idx[:max(5, int(0.3 * len(cal_idx)))]
        tr_idx = np.array([i for i in tr_idx if i not in set(cal_idx)])
        cov, half, _, _, _ = split_conformal(X[tr_idx], y[tr_idx], X[cal_idx], y[cal_idx], X[te], y[te])
        results.append(["A2_leave_region_recent", reg, "2020-2024", len(te), cov, half])
        print(f"{reg:<18}{len(te):>7}{cov:>10.2%}{half:>12.1f}")

    print("\n== B. leave-years-out (train 1998-2019, calibrate 30%, test 2020-2024) ==")
    te = np.where(year >= 2020)[0]
    tr_idx = np.where(year <= 2019)[0]
    cal_idx = tr_idx.copy()
    rng = np.random.RandomState(999)
    rng.shuffle(cal_idx)
    cal_idx = cal_idx[:max(5, int(0.3 * len(cal_idx)))]
    tr_idx = np.array([i for i in tr_idx if i not in set(cal_idx)])
    cov, half, _, _, _ = split_conformal(X[tr_idx], y[tr_idx], X[cal_idx], y[cal_idx], X[te], y[te])
    results.append(["B_leave_year", "all", "2020-2024", len(te), cov, half])
    print(f"{'all':<18}{len(te):>7}{cov:>10.2%}{half:>12.1f}")

    out = pd.DataFrame(results, columns=["scheme", "held_out", "test_window", "n_test",
                                         "coverage", "interval_half_width"])
    out.to_csv(os.path.join(OUTD, "pilot_conformal_results.csv"), index=False)
    print(f"\nWritten: {os.path.join(OUTD, 'pilot_conformal_results.csv')}")
    print("\nNote: pilot output only. Not a study result.")

if __name__ == "__main__":
    main()
