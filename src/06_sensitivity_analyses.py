#!/usr/bin/env python3
"""
06_sensitivity_analyses.py — sensitivity analyses (preregistration Section 7, five scenarios)

SA1  secondary outcome: % population >=35 ug/m3
SA2  shortened window 2015-2024
SA3  Yangtze River Delta sub-analysis (already in the primary pipeline; confirmed here)
SA4  calibration fractions 20% and 40%
SA5  excluding 2020-2022

Each scenario changes one design element; everything else follows the primary analysis.
"""
import os, sys, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import importlib.util
spec = importlib.util.spec_from_file_location("scp", os.path.join(ROOT, "code", "03_spatial_conformal.py"))
scp = importlib.util.module_from_spec(spec); spec.loader.exec_module(scp)
SpatialConformal = scp.SpatialConformal

from sklearn.ensemble import HistGradientBoostingRegressor

ALPHA = 0.10
SEED_CAL = 20260917
SEED_MODEL = 7
FEATS = ["year", "t2m_mean_c", "prec_total_mm", "rh_mean_pct", "pop_million"]
COORD = {
 "Beijing":(116.41,39.90),"Tianjin":(117.36,39.34),"Hebei":(114.51,38.04),"Shanxi":(112.55,37.87),
 "Nei Mongol":(111.75,40.84),"Liaoning":(123.43,41.81),"Jilin":(125.32,43.82),"Heilongjiang":(126.53,45.80),
 "Shanghai":(121.47,31.23),"Jiangsu":(118.80,32.06),"Zhejiang":(120.16,30.27),"Anhui":(117.23,31.82),
 "Fujian":(119.30,26.07),"Jiangxi":(115.86,28.68),"Shandong":(117.12,36.65),"Henan":(113.63,34.75),
 "Hubei":(114.31,30.59),"Hunan":(112.94,28.23),"Guangdong":(113.26,23.13),"Guangxi":(108.37,22.82),
 "Hainan":(110.20,20.04),"Chongqing":(106.55,29.56),"Sichuan":(104.07,30.57),"Guizhou":(106.63,26.65),
 "Yunnan":(102.83,24.88),"Xizang":(91.17,29.65),"Shaanxi":(108.94,34.34),"Gansu":(103.83,36.06),
 "Qinghai":(101.78,36.62),"Ningxia Hui":(106.23,38.49),"Xinjiang Uygur":(87.62,43.83),
}
REGIONS = {
 "North China":["Beijing","Tianjin","Hebei","Shanxi","Nei Mongol"],
 "Northeast China":["Liaoning","Jilin","Heilongjiang"],
 "East China":["Shanghai","Jiangsu","Zhejiang","Anhui","Fujian","Jiangxi","Shandong"],
 "Central China":["Henan","Hubei","Hunan"],
 "South China":["Guangdong","Guangxi","Hainan"],
 "Southwest China":["Chongqing","Sichuan","Guizhou","Yunnan","Xizang"],
 "Northwest China":["Shaanxi","Gansu","Qinghai","Ningxia Hui","Xinjiang Uygur"],
}
YRD = ["Shanghai","Jiangsu","Zhejiang","Anhui"]

def load_panel():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "panel_province_year.csv"))
    df = df.dropna(subset=["pm25_popwt"] + FEATS).copy()
    df["region"] = df["province_en"].map({p: r for r, ps in REGIONS.items() for p in ps})
    df["lon"] = df["province_en"].map(lambda p: COORD[p][0])
    df["lat"] = df["province_en"].map(lambda p: COORD[p][1])
    return df

def _norm_xy(d):
    a = d[["lon", "lat"]].to_numpy(float).copy()
    a[:, 0] = (a[:, 0] - 75.0) / 60.0
    a[:, 1] = (a[:, 1] - 18.0) / 35.0
    return a

def gscp_interval(delta, mu, sd, alpha=ALPHA, w=None):
    n = len(delta)
    if w is None:
        w = np.full(n, 1.0 / (n + 1)); w_self = 1.0 / (n + 1)
    else:
        Z = 1.0 + w.sum(); w_self = 1.0 / Z; w = w / Z
    order = np.argsort(delta)
    ds, ws = delta[order], w[order]
    tail = w_self + np.cumsum(ws[::-1])[::-1]
    ok = np.where(tail >= alpha)[0]
    if not len(ok): return mu, mu
    h = sd * ds[ok[-1]]
    return mu - h, mu + h

def run_one(df, outcome_col, tr_mask, cal_mask, te_mask):
    tr, cal, te = df[tr_mask], df[cal_mask], df[te_mask]
    if len(tr) < 30 or len(cal) < 10 or len(te) < 5:
        return None
    tr = tr.reset_index(drop=True)
    S_tr = _norm_xy(tr); y_tr = tr[outcome_col].to_numpy(float)
    m = SpatialConformal(alpha=ALPHA).fit(S_tr, y_tr)
    S_cal = _norm_xy(cal); S_te = _norm_xy(te)
    pred_cal = np.array([m._pred(S_cal[i])[:2] for i in range(len(S_cal))])
    y_cal = cal[outcome_col].to_numpy(float)
    delta_cal = np.abs(y_cal - pred_cal[:, 0]) / pred_cal[:, 1]
    y_te = te[outcome_col].to_numpy(float)
    cov_s = 0; w_s = 0.0
    for i in range(len(te)):
        mu, sd = m._pred(S_te[i])[:2]
        d = np.sqrt(((S_cal - S_te[i]) ** 2).sum(1))
        w = np.exp(-d ** 2 / (2 * m.eta ** 2))
        lo, hi = gscp_interval(delta_cal, mu, sd, w=w)
        cov_s += (lo <= y_te[i] <= hi); w_s += (hi - lo)
    gbm = HistGradientBoostingRegressor(max_iter=200, random_state=SEED_MODEL)
    gbm.fit(tr[FEATS], tr[outcome_col])
    rc = np.abs(cal[outcome_col] - gbm.predict(cal[FEATS]))
    q = np.sort(rc)[min(len(rc) - 1, int(np.ceil((len(rc) + 1) * (1 - ALPHA))) - 1)]
    pt = gbm.predict(te[FEATS])
    cov_g = float(np.mean(np.abs(y_te - pt) <= q))
    return dict(n_test=len(te), cov_sLSCP=cov_s / len(te), cov_GBM=cov_g,
                width_sLSCP=w_s / len(te), width_GBM=2 * q)

def pooled(rows, col):
    return float((rows[col] * rows["n_test"]).sum() / rows["n_test"].sum()) if rows["n_test"].sum() else float("nan")

def run_holdout(df, te_mask, outcome_col, analysis, held_out, pool_year_max=2019, cal_frac=0.3):
    """Leave-region-out style hold-out, shared by every scenario.

    pool_year_max: the training/calibration pool is restricted to years <= this
    value, exactly as in the primary pipeline (preregistration Section 5:
    training and calibration come from the other macro-regions, 1998-2019).
    The held-out units are evaluated over all years available in the scenario
    window. Deviation D-004 in the supplement records the earlier
    implementation in which this restriction was omitted in parts of the
    sensitivity code; the corrected implementation is used here.
    """
    pool = ~te_mask & (df["year"] <= pool_year_max)
    rng = np.random.RandomState(SEED_CAL)
    idx = np.where(pool)[0]; rng.shuffle(idx)
    cal_idx = set(idx[:max(5, int(cal_frac * len(idx)))])
    cal_mask = df.index.isin(cal_idx) & pool
    tr_mask = pool & ~df.index.isin(cal_idx)
    r = run_one(df, outcome_col, tr_mask, cal_mask, te_mask)
    if r is None:
        return []
    return [dict(analysis=analysis, held_out=held_out, **r)]

def run_lr(df, outcome_col):
    """Leave-region-out over the seven macro-regions (primary partition design)."""
    results = []
    for reg, provs in REGIONS.items():
        te_mask = df["province_en"].isin(provs)
        results += run_holdout(df, te_mask, outcome_col,
                               analysis="leave_region", held_out=reg)
    return results

def run_ly(df, outcome_col, test_start=2020):
    """Leave-years-out"""
    te = df["year"] >= test_start
    pool = df["year"] < test_start
    rng = np.random.RandomState(SEED_CAL)
    idx = np.where(pool)[0]; rng.shuffle(idx)
    cal_idx = set(idx[:max(5, int(0.3 * len(idx)))])
    cal_mask = df.index.isin(cal_idx); tr_mask = pool & ~df.index.isin(cal_idx)
    r = run_one(df, outcome_col, tr_mask, cal_mask, te)
    return [dict(analysis="leave_year", held_out=f"{test_start}-2024", **r)] if r else []

def main():
    df = load_panel()
    all_results = []

    def tag(rows, scenario):
        for r in rows:
            r["scenario"] = scenario
        all_results.extend(rows)

    # SA1: secondary outcome
    print("SA1: secondary outcome %pop>=35")
    tag(run_lr(df, "pct_pop_ge35"), "SA1_secondary_outcome")
    tag(run_ly(df, "pct_pop_ge35"), "SA1_secondary_outcome")

    # SA2: shortened window 2015-2024 (pool still restricted to years <= 2019, as frozen)
    # indices must be reset after filtering: the partition code selects the calibration
    # set under the "positional index == index label" convention
    print("SA2: window 2015-2024")
    df2 = df[df["year"] >= 2015].copy()
    df2 = df2.reset_index(drop=True)
    tag(run_lr(df2, "pm25_popwt"), "SA2_window_2015_2024")

    # SA3: Yangtze River Delta (preregistration Section 5: partition as in leave-region-out)
    print("SA3: Yangtze River Delta")
    tag(run_holdout(df, df["province_en"].isin(YRD), "pm25_popwt",
                    analysis="yrd_sub", held_out="YRD"), "SA3_yrd")

    # SA4: calibration fractions 20% / 40%
    print("SA4: calibration fractions 20%/40%")
    for frac in [0.20, 0.40]:
        rows = []
        for reg, provs in REGIONS.items():
            rows += run_holdout(df, df["province_en"].isin(provs), "pm25_popwt",
                                analysis=f"calfrac_{int(frac*100)}", held_out=reg,
                                cal_frac=frac)
        tag(rows, f"SA4_calfrac_{int(frac*100)}")

    # SA5: excluding 2020-2022 (pool 1998-2019; test years include 2023-2024)
    print("SA5: excluding 2020-2022")
    df5 = df[(df["year"] < 2020) | (df["year"] > 2022)].copy()
    df5 = df5.reset_index(drop=True)
    tag(run_lr(df5, "pm25_popwt"), "SA5_exclude_2020_2022")

    # primary-analysis reference rows (must reproduce the 04 outputs bit for bit)
    tag(run_lr(df, "pm25_popwt"), "REF_primary_A")
    tag(run_ly(df, "pm25_popwt"), "REF_primary_B")

    res = pd.DataFrame(all_results)
    outdir = os.path.join(ROOT, "results", "sensitivity")
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, "sensitivity_results.csv"), index=False)
    print(f"\n{len(res)} rows written to {outdir}/sensitivity_results.csv")

    # summary (grouped by scenario to avoid mixing same-named analysis labels)
    print("\n=== summary (pooled sLSCP / GBM) ===")
    for sc in res["scenario"].unique():
        sub = res[res["scenario"] == sc]
        n = int(sub["n_test"].sum())
        cs = (sub["cov_sLSCP"] * sub["n_test"]).sum() / n
        cg = (sub["cov_GBM"] * sub["n_test"]).sum() / n
        print(f"  {sc:24s} n={n:4d}  sLSCP={cs:.4f}  GBM={cg:.4f}")

if __name__ == "__main__":
    main()
