#!/usr/bin/env python3
"""
07_contour_maps.py — contour-filled PM2.5 maps (optional auxiliary figures)

Fig 2  Yangtze River Delta PM2.5 contour map (GADM province boundaries overlay)
Fig 4  national PM2.5 contour map

Usage: python3 code/07_contour_maps.py
"""
import os, json, zipfile
import numpy as np
import netCDF4 as nc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC_FILE = os.path.join(ROOT, "data", "raw", "acag", "2026-09-17", "V6GL03.CNNPM25.AS.201501-201512.nc")
GADM = os.path.join(ROOT, "data", "raw", "gadm", "2026-09-17", "gadm41_CHN_1.json.zip")
TWN = os.path.join(ROOT, "data", "raw", "gadm", "2026-09-17", "gadm41_TWN_0.json.zip")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
TEAL = "#2C6E7E"
INK = "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

def load_gadm_provinces(want=None):
    z = zipfile.ZipFile(GADM)
    gj = json.loads(z.read("gadm41_CHN_1.json"))
    out = {}
    for f in gj["features"]:
        nm = f["properties"].get("NAME_1", "")
        if want and nm not in want: continue
        g = f["geometry"]
        polys = []
        if g["type"] == "Polygon":
            polys.append(np.array(g["coordinates"][0]))
        elif g["type"] == "MultiPolygon":
            for p in g["coordinates"]:
                polys.append(np.array(p[0]))
        if polys:
            out.setdefault(nm, []).extend(polys)
    return out

def read_nc_region(lon_range, lat_range):
    """Extract a lon/lat subset from the NetCDF file."""
    ds = nc.Dataset(NC_FILE)
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    # locate indices
    lat_idx = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lon_idx = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
    # extract
    pm = ds["PM25"][lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    lat_sub = lat[lat_idx]
    lon_sub = lon[lon_idx]
    ds.close()
    return lon_sub, lat_sub, np.array(pm)

def draw_boundaries(ax, provinces, lw=0.6, ec="#444444"):
    for prov, polys in provinces.items():
        for poly in polys:
            if poly.ndim == 2 and poly.shape[1] >= 2:
                ax.plot(poly[:, 0], poly[:, 1], color=ec, lw=lw, zorder=3)

def draw_labels(ax, provinces, fontsize=7, box=True):
    for prov, polys in provinces.items():
        all_pts = np.vstack(polys)
        clon, clat = all_pts[:, 0].mean(), all_pts[:, 1].mean()
        label = prov if len(prov) < 15 else prov[:12] + "…"
        if box:
            ax.text(clon, clat, label, fontsize=fontsize, ha="center", va="center",
                    color=INK, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.85))
        else:
            ax.text(clon, clat, label, fontsize=fontsize, ha="center", va="center",
                    color=INK, fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

def draw_north_arrow(ax, x=0.95, y=0.95):
    ax.annotate("", xy=(x, y), xytext=(x, y - 0.06),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(x, y + 0.015, "N", transform=ax.transAxes, fontsize=9,
            ha="center", va="bottom", fontweight="bold", color=INK)

def draw_scale_bar(ax, x0=0.08, y0=0.05, length_km=200):
    """Scale bar (approximate: 1 degree of longitude ~ 85 km near 30N)."""
    deg = length_km / (85.0)
    ax.annotate("", xy=(x0 + deg * 0.06, y0), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.text(x0 + deg * 0.03, y0 + 0.01, f"{length_km} km", transform=ax.transAxes,
            fontsize=7, ha="center", va="bottom", color=INK)

# ==================== Fig 2: Yangtze River Delta contour map ====================
def fig2():
    lon, lat, pm = read_nc_region((114.0, 123.5), (26.5, 36.0))
    yrd_prov = load_gadm_provinces({"Shanghai", "Jiangsu", "Zhejiang", "Anhui"})
    all_prov = load_gadm_provinces()  # all province polygons (surroundings)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.5))

    # contourf fill
    levels = np.arange(10, 65, 2.5)
    cf = ax.contourf(lon, lat, pm, levels=levels, cmap="Spectral_r", extend="max", zorder=1)

    # YRD province borders (thick)
    draw_boundaries(ax, yrd_prov, lw=1.0, ec=INK)
    # YRD province labels
    draw_labels(ax, yrd_prov, fontsize=7.5, box=True)

    # contour lines (thin white for depth)
    ax.contour(lon, lat, pm, levels=levels[::2], colors="white", linewidths=0.4, alpha=0.5, zorder=2)

    # north arrow
    draw_north_arrow(ax)
    # scale bar
    draw_scale_bar(ax)

    # colour bar
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, aspect=25)
    cbar.set_label("PM2.5 (µg/m³)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.set_xlim(114.5, 123.0); ax.set_ylim(27.0, 35.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig2_yrd_pm25_contour.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(os.path.join(OUT, "Fig2_yrd_pm25_contour.pdf"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  ✅ Fig2_yrd_pm25_contour")

# ==== Fig 4: national contour map (GADM boundaries incl. Taiwan) ====
def load_twn_outline():
    z = zipfile.ZipFile(TWN)
    gj = json.loads(z.read("gadm41_TWN_0.json"))
    polys = []
    for f in gj["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            polys.append(np.array(g["coordinates"][0]))
        elif g["type"] == "MultiPolygon":
            for pp in g["coordinates"]:
                polys.append(np.array(pp[0]))
    return polys

def fig4():
    lon, lat, pm = read_nc_region((73.0, 136.0), (17.0, 54.0))
    all_prov = load_gadm_provinces()
    twn = load_twn_outline()

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 7.0))

    levels = np.arange(5, 85, 2.5)
    cf = ax.contourf(lon, lat, pm, levels=levels, cmap="YlOrRd", extend="both", zorder=1)

    # province borders
    for prov, polys in all_prov.items():
        for poly in polys:
            if poly.ndim == 2 and poly.shape[1] >= 2:
                ax.plot(poly[:, 0], poly[:, 1], color="#888888", lw=0.4, zorder=2)
    # Taiwan boundary (GADM TWN)
    for poly in twn:
        ax.plot(poly[:, 0], poly[:, 1], color="#888888", lw=0.4, zorder=2)

    # label selected provinces only (avoid crowding)
    labels = {"Beijing":"Beijing", "Shanghai":"Shanghai", "Guangdong":"Guangdong", "Sichuan":"Sichuan",
              "Xinjiang Uygur":"Xinjiang", "Xizang":"Xizang", "Heilongjiang":"Heilongjiang",
              "Yunnan":"Yunnan", "Hunan":"Hunan", "Jiangsu":"Jiangsu"}
    for prov, polys in all_prov.items():
        if prov not in labels: continue
        all_pts = np.vstack(polys)
        clon, clat = all_pts[:, 0].mean(), all_pts[:, 1].mean()
        ax.text(clon, clat, labels[prov], fontsize=7, ha="center", va="center",
                color=INK, fontweight="bold", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    twn_pts = np.vstack(twn)
    ax.text(twn_pts[:, 0].mean() + 1.2, twn_pts[:, 1].mean() - 0.4, "Taiwan",
            fontsize=7, ha="center", va="center", color=INK, fontweight="bold", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # north arrow
    draw_north_arrow(ax)
    # scale bar
    draw_scale_bar(ax, length_km=500)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02, aspect=25)
    cbar.set_label("PM2.5 (µg/m³)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.set_facecolor("#D0E0F0")  # light blue for the sea

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig4_national_pm25_contour.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(os.path.join(OUT, "Fig4_national_pm25_contour.pdf"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  OK Fig4_national_pm25_contour (GADM boundaries incl. Taiwan)")

if __name__ == "__main__":
    print("=== contour-filled maps ===")
    fig2()
    fig4()
    print("\ndone")
