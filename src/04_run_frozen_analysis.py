#!/usr/bin/env python3
"""
04_run_frozen_analysis.py - frozen confirmatory analysis (sole implementation of Sections 5-7 of the preregistration)

IMPORTANT
    Before the preregistration is submitted, this script may only be run with --dry-run (synthetic data).
    The real-data mode (--real) may only be executed after registration is complete, and the date of execution must be recorded.
    This preserves the ordering 'plan frozen first, results produced afterwards', which is the source of the study's credibility.

Usage
    python3 code/04_run_frozen_analysis.py --dry-run    # synthetic data (may be run at any time)
    python3 code/04_run_frozen_analysis.py --real       # real data (only after registration)

Outputs
    results/<mode>/results_bundle.json    summary
    results/<mode>/coverage_by_region.csv block-wise coverage
    results/<mode>/run_log.txt            run log (timestamps and random seeds)
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from importlib import import_module
SC = import_module("03_spatial_conformal".replace("03_", "03_")) if False else None

import importlib.util
spec = importlib.util.spec_from_file_location("scp", os.path.join(ROOT, "code", "03_spatial_conformal.py"))
scp = importlib.util.module_from_spec(spec); spec.loader.exec_module(scp)
SpatialConformal = scp.SpatialConformal

from sklearn.ensemble import HistGradientBoostingRegressor

ALPHA = 0.10
SEED_CAL = 20260917          # calibration-sampling seed frozen in the preregistration
SEED_MODEL = 7               # model seed frozen in the preregistration
FEATS = ["year", "t2m_mean_c", "prec_total_mm", "rh_mean_pct", "pop_million"]

# provincial-capital coordinates (province names as spelled by ACAG); the spatial proxy declared in Section 3
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
THRESH = dict(global_lo=0.85, global_hi=0.95, n_regions_ok=5, region_lo=0.80, region_hi=1.00, worst_min=0.65)

def load_panel():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "panel_province_year.csv"))
    df = df.dropna(subset=["pm25_popwt"] + FEATS).copy()
    df["region"] = df["province_en"].map({p: r for r, ps in REGIONS.items() for p in ps})
    df["lon"] = df["province_en"].map(lambda p: COORD[p][0])
    df["lat"] = df["province_en"].map(lambda p: COORD[p][1])
    return df

def dry_run_panel(df, seed=999):
    """Synthetic outcome: real covariates and spatial structure retained, y replaced by synthetic values."""
    rng = np.random.RandomState(seed)
    # synthetic process with spatial effects, on a scale comparable to real PM2.5 (roughly 5-100)
    base = 35 + 0.02 * (df["year"] - 1998) - 0.6 * df["t2m_mean_c"] + 0.004 * df["prec_total_mm"]
    prov_eff = df["province_en"].map(lambda p: rng.normal(0, 12))
    y = np.clip(base + prov_eff + rng.normal(0, 6, len(df)), 3, 130)
    df = df.copy(); df["pm25_popwt"] = y.values
    return df

# ---------------- interval methods ----------------
def gscp_interval(delta, mu, sd, alpha=ALPHA, w=None):
    """Shared interface for equal-weight (GSCP) and weighted (sLSCP); w=None reduces to equal-weight full-data conformal."""
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

def _norm_xy(df):
    """Provincial-centre longitude/latitude rescaled to [0,1]^2 (75-135 E, 18-53 N)."""
    a = df[["lon", "lat"]].to_numpy(float).copy()
    a[:, 0] = (a[:, 0] - 75.0) / 60.0
    a[:, 1] = (a[:, 1] - 18.0) / 35.0
    return a

def fit_predictor_gp(tr, te, alpha=ALPHA):
    """Gaussian-process (Kriging) predictor; the primary GSCP-versus-sLSCP comparison shares this predictor."""
    S = _norm_xy(tr); y = tr["pm25_popwt"].to_numpy(float)
    m = SpatialConformal(alpha=alpha).fit(S, y)
    Ste = _norm_xy(te)
    out = np.array([m._pred(Ste[i])[:2] for i in range(len(te))])
    return m, out

def eval_split(df, tr_mask, cal_mask, te_mask, tag, rows, bundle, obs_rows):
    tr_raw, cal, te = df[tr_mask], df[cal_mask], df[te_mask]
    if len(tr_raw) < 30 or len(cal) < 10 or len(te) < 5:
        return
    tr = tr_raw.reset_index(drop=True)
    # --- primary comparison: GP predictor, GSCP (equal weight) vs sLSCP (spatially weighted) ---
    m, pred_te = fit_predictor_gp(tr, te)
    s_cal = _norm_xy(cal); s_te = _norm_xy(te)
    # Calibration scores: out-of-sample standardised residuals from the fitted predictor.
    # The calibration set is disjoint from the training set by design (preregistration
    # Section 5), so scores must be computed out of sample rather than as in-fit residuals.
    # See deviation register entries D-001 (wording) and D-002 (this defect and its fix).
    pred_cal = np.array([m._pred(s_cal[i])[:2] for i in range(len(s_cal))])
    y_cal = cal["pm25_popwt"].to_numpy(float)
    delta_cal = np.abs(y_cal - pred_cal[:, 0]) / pred_cal[:, 1]
    y_te = te["pm25_popwt"].to_numpy(float)
    cov_g = cov_s = 0.0; w_g = w_s = 0.0
    for i in range(len(te)):
        mu, sd = float(pred_te[i, 0]), float(pred_te[i, 1])
        lo, hi = gscp_interval(delta_cal, mu, sd, w=None)
        cov_g += (lo <= y_te[i] <= hi); w_g += (hi - lo)
        d = np.sqrt(((s_cal - s_te[i]) ** 2).sum(1))
        w = np.exp(-d ** 2 / (2 * m.eta ** 2))
        lo2, hi2 = gscp_interval(delta_cal, mu, sd, w=w)
        cov_s += (lo2 <= y_te[i] <= hi2); w_s += (hi2 - lo2)
        obs_rows.append(dict(split=tag, province=te["province_en"].iloc[i],
                             macro_region=te["macro_region"].iloc[i], year=int(te["year"].iloc[i]),
                             y=float(y_te[i]), pred=mu,
                             lo_GSCP=float(lo), hi_GSCP=float(hi),
                             lo_sLSCP=float(lo2), hi_sLSCP=float(hi2)))
    # --- secondary comparison: gradient boosting + split conformal (the pilot baseline) ---
    gbm = HistGradientBoostingRegressor(max_iter=200, random_state=SEED_MODEL).fit(tr[FEATS], tr["pm25_popwt"])
    rc = np.abs(cal["pm25_popwt"] - gbm.predict(cal[FEATS]))
    q = np.sort(rc)[min(len(rc) - 1, int(np.ceil((len(rc) + 1) * (1 - ALPHA))) - 1)]
    pred_te = gbm.predict(te[FEATS]); y_te2 = te["pm25_popwt"].to_numpy(float)
    cov_b = float(np.mean(np.abs(y_te2 - pred_te) <= q)); w_b = float(2 * q)
    nte = len(te)
    rows.append(dict(split=tag, n_train=len(tr), n_cal=len(cal), n_test=nte,
                     cov_GSCP=cov_g / nte, cov_sLSCP=cov_s / nte, cov_GBM=cov_b,
                     width_GSCP=w_g / nte, width_sLSCP=w_s / nte, width_GBM=w_b))
    bundle["details"][tag] = dict(n_te=nte,
                                  cov_GSCP=cov_g / nte, cov_sLSCP=cov_s / nte, cov_GBM=cov_b,
                                  width_GSCP=w_g / nte, width_sLSCP=w_s / nte, width_GBM=w_b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.real):
        print("Specify --dry-run or --real."); return
    if a.real and not os.path.exists(os.path.join(ROOT, "PREREG_SUBMITTED.txt")):
        print("Refusing to run: PREREG_SUBMITTED.txt not found.\n"
              "   Under the preregistration protocol, real-data mode requires that registration"
              "   has been completed with its link and date recorded in PREREG_SUBMITTED.txt.")
        return
    mode = "dry_run" if a.dry_run else "real"
    outdir = os.path.join(ROOT, "results", mode); os.makedirs(outdir, exist_ok=True)
    df = load_panel()
    if a.dry_run:
        df = dry_run_panel(df)
    log = [f"mode={mode}", f"start={time.strftime('%Y-%m-%d %H:%M:%S')}",
           f"n_rows={len(df)}", f"seed_cal={SEED_CAL} seed_model={SEED_MODEL} alpha={ALPHA}"]
    rows, obs_rows = [], []
    bundle = {"thresholds": THRESH, "details": {}, "mode": mode}
    # A. leave-macro-region-out
    # Preregistration Section 5 (frozen): training and calibration come from the
    # other macro-regions restricted to 1998-2019; the held-out macro-region is
    # evaluated over all years. Deviation D-004 in the supplement records the
    # earlier execution in which the 1998-2019 restriction was omitted here.
    for reg, provs in REGIONS.items():
        te = df["province_en"].isin(provs)
        pool = ~te & (df["year"] <= 2019)
        rng = np.random.RandomState(SEED_CAL)
        idx = np.where(pool)[0]; rng.shuffle(idx)
        cal_idx = set(idx[:max(5, int(0.3 * len(idx)))])
        cal = df.index.isin(cal_idx) & pool
        tr = pool & ~df.index.isin(cal_idx)
        eval_split(df, tr, cal, te, f"A_leave_region_{reg}", rows, bundle, obs_rows)
    # A2. leave-region-out, recent years
    for reg, provs in REGIONS.items():
        te = df["province_en"].isin(provs) & (df["year"] >= 2020)
        pool = (~df["province_en"].isin(provs)) & (df["year"] <= 2019)
        pool2 = (~df["province_en"].isin(provs)) & (df["year"] <= 2019)
        rng = np.random.RandomState(SEED_CAL)
        idx = np.where(pool2)[0]; rng.shuffle(idx)
        cal_idx = set(idx[:max(5, int(0.3 * len(idx)))])
        cal = df.index.isin(cal_idx)
        tr = pool & ~df.index.isin(cal_idx)
        eval_split(df, tr, cal, te, f"A2_leave_region_recent_{reg}", rows, bundle, obs_rows)
    # B. leave-years-out
    te = df["year"] >= 2020
    pool = df["year"] <= 2019
    rng = np.random.RandomState(SEED_CAL); idx = np.where(pool)[0]; rng.shuffle(idx)
    cal_idx = set(idx[:max(5, int(0.3 * len(idx)))])
    cal = df.index.isin(cal_idx); tr = pool & ~df.index.isin(cal_idx)
    eval_split(df, tr, cal, te, "B_leave_year", rows, bundle, obs_rows)
    # C. Yangtze River Delta sub-analysis (hold out the delta, train on the rest)
    # Preregistration Section 5: "As leave-region-out", i.e. the same 1998-2019
    # restriction on the training/calibration pool applies.
    te = df["province_en"].isin(YRD)
    pool = ~te & (df["year"] <= 2019)
    rng = np.random.RandomState(SEED_CAL); idx = np.where(pool)[0]; rng.shuffle(idx)
    cal_idx = set(idx[:max(5, int(0.3 * len(idx)))])
    cal = df.index.isin(cal_idx); tr = pool & ~df.index.isin(cal_idx)
    eval_split(df, tr, cal, te, "C_Yangtze_Delta", rows, bundle, obs_rows)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "coverage_by_region.csv"), index=False)
    # verdict (pooled global coverage, matching preregistration Section 6; see D-003)
    def pooled(sub, col):
        return float((sub[col] * sub["n_test"]).sum() / sub["n_test"].sum()) if sub["n_test"].sum() else float("nan")
    main_A = res[res["split"].str.startswith("A_leave_region_")]
    main_A2 = res[res["split"].str.startswith("A2_leave_region_recent_")]
    main_B = res[res["split"].str.startswith("B_leave_year")]
    bundle["verdict"] = {
        "A_pooled_global_cov_sLSCP": pooled(main_A, "cov_sLSCP"),
        "A_pooled_global_cov_GSCP": pooled(main_A, "cov_GSCP"),
        "A_pooled_global_cov_GBM": pooled(main_A, "cov_GBM"),
        "A_unweighted_mean_cov_sLSCP": float(main_A["cov_sLSCP"].mean()),
        "A_worst_region_sLSCP": float(main_A["cov_sLSCP"].min()),
        "A_n_regions_ok_sLSCP": int(((main_A["cov_sLSCP"] >= THRESH["region_lo"]) &
                                     (main_A["cov_sLSCP"] <= THRESH["region_hi"])).sum()),
        "A_n_regions_ok_GSCP": int(((main_A["cov_GSCP"] >= THRESH["region_lo"]) &
                                    (main_A["cov_GSCP"] <= THRESH["region_hi"])).sum()),
        "A2_pooled_global_cov_sLSCP": pooled(main_A2, "cov_sLSCP"),
        "B_pooled_global_cov_sLSCP": pooled(main_B, "cov_sLSCP"),
        "B_pooled_global_cov_GSCP": pooled(main_B, "cov_GSCP"),
        "H1_met_sLSCP": bool(THRESH["global_lo"] <= pooled(main_A, "cov_sLSCP") <= THRESH["global_hi"]
                             and THRESH["global_lo"] <= pooled(main_B, "cov_sLSCP") <= THRESH["global_hi"]),
        "H2_met_sLSCP": bool((((main_A["cov_sLSCP"] >= THRESH["region_lo"]) &
                               (main_A["cov_sLSCP"] <= THRESH["region_hi"])).sum() >= THRESH["n_regions_ok"])
                             and (main_A["cov_sLSCP"].min() >= THRESH["worst_min"])),
    }
    pd.DataFrame(obs_rows).to_csv(os.path.join(outdir, "observations_per_partition.csv"), index=False)
    json.dump(bundle, open(os.path.join(outdir, "results_bundle.json"), "w"), ensure_ascii=False, indent=1)
    log.append(f"end={time.strftime('%Y-%m-%d %H:%M:%S')}")
    open(os.path.join(outdir, "run_log.txt"), "w").write("\n".join(log))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nVerdict:", json.dumps(bundle["verdict"], ensure_ascii=False, indent=1))
    print(f"\nOutput directory: {outdir}")

if __name__ == "__main__":
    main()
