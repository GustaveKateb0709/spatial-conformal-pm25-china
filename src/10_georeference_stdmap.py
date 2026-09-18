#!/usr/bin/env python3
"""
10_georeference_stdmap.py — georeference the standard base map (v2, boundary chamfer)

Principle: the province/national boundary lines printed on the standard base map
GS(2019)1838 are the official boundaries; GADM provides the same boundaries as
lon/lat. GADM boundary points are mapped to pixels by a trial transform, and
Nelder-Mead minimises the distance from each mapped point to the nearest dark
image pixel. At convergence the transform maps lon/lat -> pixels accurately.

Model: quadratic polynomial (12 parameters) px = f(lon,lat), py = g(lon,lat).
Output: data/stdmap_transform_v2.npz + results/stdmap_georef_report.json
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
    # ---- dark-pixel mask of the base map + distance transform ----
    g = np.array(Image.open(IMG).convert("L"))
    H, W = g.shape
    dark = g < 140
    dt = ndimage.distance_transform_edt(~dark)

    rows = np.where(dark.mean(axis=1) > 0.55)[0]
    cols = np.where(dark.mean(axis=0) > 0.55)[0]
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    print(f"neatline: x {x0}-{x1}, y {y0}-{y1}")

    # ---- GADM boundary points (subsampled) ----
    lon, lat = gadm_boundary_points(step=3)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(lon), size=min(4000, len(lon)), replace=False)
    lon, lat = lon[idx], lat[idx]
    print(f"GADM boundary sample points: {len(lon)}")

    # ---- initial parameters: linear estimate from the neatline (12 slots, full quadratic) ----
    # px = a0+a1*lon+a2*lat+a3*lon*lat+a4*lon^2+a5*lat^2 ; py likewise (b slots)
    sx = (4557 - 193) / (135 - 75)          # px / deg lon
    sy = (3189 - 158) / (20 - 50)           # py / deg lat (negative)
    a0 = 193 - 75 * sx
    b2 = sy
    b0 = 3189 - b2 * 20
    p0 = np.array([a0, sx, 0.0, 0.0, 0.0, 0.0, b0, 0.0, b2, 0.0, 0.0, 0.0])

    INSET = dict(x0=int(W * 0.775), y0=int(H * 0.665))  # South China Sea inset (never drawn over)

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
        return float(d[:int(len(d) * 0.85)].mean())   # trimmed 15%: robust to text and graticule

    print("init objective:", round(objective(p0), 2))
    # staged: affine (6 params) -> bilinear (8) -> full quadratic (12)
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
