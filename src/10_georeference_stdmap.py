#!/usr/bin/env python3
"""
10_georeference_stdmap.py — 标准底图地理配准 v2（边界 chamfer 配准）

原理：标准底图 GS(2019)1838 上的省界/国界黑线 = 官方界线；GADM 提供同样的
界线经纬度。将 GADM 界线点经试验变换映射到像素，以"到底图最近暗像元的距离"
为目标做 Nelder-Mead 优化，收敛即得 lon/lat -> 像素 的精确变换。

变换模型：二次多项式（12 参数）px = f(lon,lat), py = g(lon,lat)。
输出：data/stdmap_transform_v2.npz + results/stdmap_georef_report.json
"""
import os, json, zipfile
import numpy as np
from PIL import Image
from scipy import ndimage, optimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "data", "raw", "stdmap_GS2019_1838.jpg")
GADM = os.path.join(ROOT, "data", "raw", "gadm", "2026-09-17", "gadm41_CHN_1.json.zip")

TERMS = lambda x, y: np.column_stack([
    np.ones_like(x), x, y, x * y, x * x, y * y])


def gadm_boundary_points(step=3):
    z = zipfile.ZipFile(GADM)
    gj = json.loads(z.read("gadm41_CHN_1.json"))
    pts = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = []
        if g["type"] == "Polygon":
            polys = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            polys = [p[0] for p in g["coordinates"]]
        for poly in polys:
            arr = np.array(poly)
            pts.append(arr[::step, :2])
    P = np.vstack(pts)
    return P[:, 0], P[:, 1]


def main():
    # ---- 底图暗像元掩膜 + 距离变换 ----
    g = np.array(Image.open(IMG).convert("L"))
    H, W = g.shape
    dark = g < 140
    dt = ndimage.distance_transform_edt(~dark)

    rows = np.where(dark.mean(axis=1) > 0.55)[0]
    cols = np.where(dark.mean(axis=0) > 0.55)[0]
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    print(f"neatline: x {x0}-{x1}, y {y0}-{y1}")

    # ---- GADM 界线点（子采样）----
    lon, lat = gadm_boundary_points(step=3)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(lon), size=min(4000, len(lon)), replace=False)
    lon, lat = lon[idx], lat[idx]
    print(f"GADM boundary sample points: {len(lon)}")

    # ---- 初始参数：图廓线性估计（12 槽，完整二次）----
    # px = a0+a1*lon+a2*lat+a3*lon*lat+a4*lon^2+a5*lat^2 ; py 同构（b 槽）
    sx = (4557 - 193) / (135 - 75)          # px / deg lon
    sy = (3189 - 158) / (20 - 50)           # py / deg lat（负）
    a0 = 193 - 75 * sx
    b2 = sy
    b0 = 3189 - b2 * 20
    p0 = np.array([a0, sx, 0.0, 0.0, 0.0, 0.0, b0, 0.0, b2, 0.0, 0.0, 0.0])

    INSET = dict(x0=int(W * 0.775), y0=int(H * 0.665))     # 南海 insets 区（不绘制）

    def to_px(lons, lats, p):
        a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5 = p
        px = a0 + a1*lons + a2*lats + a3*lons*lats + a4*lons*lons + a5*lats*lats
        py = b0 + b1*lons + b2*lats + b3*lons*lats + b4*lons*lons + b5*lats*lats
        return px, py

    def dists(p):
        px, py = to_px(lon, lat, p)
        out = ((px < x0) | (px > x1) | (py < y0) | (py > y1))
        px = np.clip(px, 0, W - 1); py = np.clip(py, 0, H - 1)
        d = dt[py.astype(int), px.astype(int)].copy()
        d[out] = 60.0
        return np.minimum(d, 60.0)

    def objective(p):
        d = np.sort(dists(p))
        return float(d[:int(len(d) * 0.85)].mean())   # 截尾 15%，抗文字/经纬网噪声

    print("init objective:", round(objective(p0), 2))
    # 三段式：仿射 6 参数 -> 双线性 8 参数 -> 完整二次 12 参数
    stages = [6, 8, 12]
    p = p0.copy()
    for n in stages:
        sub = p0.copy(); sub[:n] = p[:n]
        res = optimize.minimize(objective, sub, method="Nelder-Mead",
                                options=dict(maxiter=30000, maxfev=30000,
                                             xatol=0.005, fatol=0.0005,
                                             adaptive=True))
        p = res.x
        d = dists(p)
        print(f"stage({n} params): obj={res.fun:.2f} mean={d.mean():.1f} "
              f"median={np.median(d):.1f} p90={np.percentile(d, 90):.1f}")
    px, py = to_px(lon, lat, p)
    d = dt[py.clip(0, H - 1).astype(int), px.clip(0, W - 1).astype(int)]
    report = dict(
        mean_px=float(np.mean(np.minimum(d, 60))), median_px=float(np.median(d)),
        p90_px=float(np.percentile(d, 90)), params=[float(v) for v in p],
        n_points=int(len(lon)),
    )
    print(f"align: mean={report['mean_px']:.1f}px median={report['median_px']:.1f}px "
          f"p90={report['p90_px']:.1f}px")

    np.savez(os.path.join(ROOT, "data", "stdmap_transform_v2.npz"),
             params=p, model="px/py = c0+c1*lon+c2*lat+c3*lon*lat+c4*lon^2+c5*lat^2 (12 params)",
             neatline=[int(x0), int(x1), int(y0), int(y1)],
             inset=np.array([INSET["x0"], INSET["y0"]]))
    with open(os.path.join(ROOT, "results", "stdmap_georef_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("saved: data/stdmap_transform_v2.npz + results/stdmap_georef_report.json")


if __name__ == "__main__":
    main()
