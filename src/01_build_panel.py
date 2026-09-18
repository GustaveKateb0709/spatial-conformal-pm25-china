#!/usr/bin/env python3
"""
01_build_panel.py - assemble the province x year panel from ACAG PM2.5 and NASA POWER meteorology.

Inputs
  data/raw/acag/2026-09-17/ChinaPM25-V6GL03-Annual-REGIONAL-1998-2024-wThresFrac.csv
  data/raw/nasa_power/2026-09-17/power_<city>_T2M_PREC_RH_2015_2024.json
  data/raw/nasa_power/2026-09-17_hist/power_<city>_T2M_PREC_RH_1998_2014.json
  data/provinces.csv   (city-to-province lookup table)

Outputs
  data/processed/panel_province_year.csv   province x year panel
  data/processed/panel_build_report.txt    construction report (rows, missingness, range checks)

Notes
  - Meteorology uses provincial-capital point values as a v1 provincial proxy.
    This limitation is declared in the preregistration and in the manuscript.
  - The city-to-province lookup lives in data/provinces.csv. The source weather files are
    named by city, so the lookup table carries those names as data; the code itself
    contains no non-ASCII literals.
  - This script performs assembly and quality control only. It does no modelling.
"""
import json, os, glob
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUTD = os.path.join(ROOT, "data", "processed")
os.makedirs(OUTD, exist_ok=True)

def load_lookup():
    """city name (as used in the source filenames) -> province, coordinates, macro-region"""
    lp = pd.read_csv(os.path.join(ROOT, "data", "provinces.csv"))
    return {r.city_zh: (r.province_en, float(r.lat), float(r.lon), r.macro_region_en)
            for r in lp.itertuples()}

def load_acag():
    f = os.path.join(RAW, "acag", "2026-09-17",
                     "ChinaPM25-V6GL03-Annual-REGIONAL-1998-2024-wThresFrac.csv")
    df = pd.read_csv(f)
    df = df.rename(columns={
        "Region": "province_en", "Year": "year",
        "Population-Weighted PM2.5 [ug/m3]": "pm25_popwt",
        "Geographic-Mean PM2.5 [ug/m3]": "pm25_geomean",
        "Population Coverage [%]": "pop_coverage",
        "Geographic Coverage [%]": "geo_coverage",
        "Total Population [million people]": "pop_million",
        "% pop >= 35 ug/m3 [%]": "pct_pop_ge35",
        "% pop >= 25 ug/m3 [%]": "pct_pop_ge25",
        "% pop >= 15 ug/m3 [%]": "pct_pop_ge15",
        "% pop >= 10 ug/m3 [%]": "pct_pop_ge10",
        "% pop >= 5 ug/m3 [%]": "pct_pop_ge5",
    })
    keep = ["province_en", "year", "pm25_popwt", "pm25_geomean", "pop_coverage", "geo_coverage",
            "pop_million", "pct_pop_ge5", "pct_pop_ge10", "pct_pop_ge15", "pct_pop_ge25",
            "pct_pop_ge35"]
    return df[keep].copy()

def load_power(lookup):
    recs = []
    dirs = [os.path.join(RAW, "nasa_power", "2026-09-17"),
            os.path.join(RAW, "nasa_power", "2026-09-17_hist")]
    for d in dirs:
        for f in glob.glob(os.path.join(d, "power_*_T2M_PREC_RH_*.json")):
            city = os.path.basename(f).split("_")[1]
            if city not in lookup:
                continue
            prov_en, lat, lon, _ = lookup[city]
            d0 = json.load(open(f))
            p = d0.get("properties", {}).get("parameter", {})
            if not p:
                continue
            years = {}
            for k in ("T2M", "PRECTOTCORR", "RH2M"):
                for ymd, v in p.get(k, {}).items():
                    if v is None or v <= -900:
                        continue
                    years.setdefault(int(str(ymd)[:4]), {}).setdefault(k, []).append(float(v))
            for y, dd in years.items():
                recs.append({
                    "province_en": prov_en, "year": y,
                    "t2m_mean_c": np.mean(dd.get("T2M", [])),
                    "prec_total_mm": np.sum(dd.get("PRECTOTCORR", [])),
                    "rh_mean_pct": np.mean(dd.get("RH2M", [])),
                    "n_days_weather": len(dd.get("T2M", [])),
                })
    w = pd.DataFrame(recs)
    return w.groupby(["province_en", "year"], as_index=False).agg(
        t2m_mean_c=("t2m_mean_c", "mean"), prec_total_mm=("prec_total_mm", "mean"),
        rh_mean_pct=("rh_mean_pct", "mean"), n_city_days=("n_days_weather", "sum"))

def main():
    lookup = load_lookup()
    a = load_acag()
    w = load_power(lookup)
    rep = []
    rep.append(f"ACAG rows: {len(a)}  provinces: {a['province_en'].nunique()}  "
               f"years: {a['year'].min()}-{a['year'].max()}")
    rep.append(f"Meteorology aggregated rows: {len(w)}  provinces: {w['province_en'].nunique()}  "
               f"years: {w['year'].min()}-{w['year'].max()}")
    m = a.merge(w, on=["province_en", "year"], how="left")
    prov2region = {v[0]: v[3] for v in lookup.values()}
    m["macro_region"] = m["province_en"].map(prov2region)
    miss = m["t2m_mean_c"].isna().sum()
    rep.append(f"After merge: {len(m)} rows; missing meteorology: {miss} "
               f"({100 * miss / len(m):.1f}%)")
    rep.append("Range checks:")
    for c, lo, hi in [("pm25_popwt", 0, 300), ("t2m_mean_c", -50, 50),
                      ("prec_total_mm", 0, 5000), ("rh_mean_pct", 0, 100)]:
        bad = ((m[c] < lo) | (m[c] > hi)).sum()
        rep.append(f"  {c}: {bad} out-of-range rows")
    m.to_csv(os.path.join(OUTD, "panel_province_year.csv"), index=False)
    open(os.path.join(OUTD, "panel_build_report.txt"), "w").write("\n".join(rep))
    print("\n".join(rep))

if __name__ == "__main__":
    main()
