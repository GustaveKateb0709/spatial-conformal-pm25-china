import os, json, zipfile
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "real")
PANEL = os.path.join(ROOT, "data", "processed", "panel_province_year.csv")
GADM = os.path.join(ROOT, "data", "raw", "gadm", "2026-09-17", "gadm41_CHN_1.json.zip")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

TEAL="#2C6E7E"; SAGE="#8FBCA8"; RED="#B5442D"; GOLD="#D9A441"; GREY="#808080"; INK="#333333"; LIGHT="#E8EFEA"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":9,
    "axes.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.color":INK,"ytick.color":INK,"axes.labelcolor":INK,
    "figure.facecolor":"white","savefig.facecolor":"white","savefig.dpi":300})

def style_ax(ax):
    ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, zorder=0); ax.set_axisbelow(True)

def save(fig, name):
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(OUT, name + ext), bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig); print(f"  ✅ {name}")

def load_gadm_yrd():
    z=zipfile.ZipFile(GADM); gj=json.loads(z.read("gadm41_CHN_1.json"))
    want={"Shanghai","Jiangsu","Zhejiang","Anhui"}; out={}
    for f in gj["features"]:
        nm=f["properties"].get("NAME_1")
        if nm not in want: continue
        g=f["geometry"]
        if g["type"]=="Polygon": out.setdefault(nm,[]).append(np.array(g["coordinates"][0]))
        elif g["type"]=="MultiPolygon":
            for p in g["coordinates"]: out.setdefault(nm,[]).append(np.array(p[0]))
    return out

def fig1():
    cov=pd.read_csv(os.path.join(RES,"coverage_by_region.csv"))
    bundle=json.load(open(os.path.join(RES,"results_bundle.json")))
    a=cov[cov["split"].str.startswith("A_leave_region_")].copy()
    a["region"]=a["split"].str.replace("A_leave_region_","",regex=False); a=a.sort_values("region")
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.2,3.8),gridspec_kw={"width_ratios":[1,1.3],"wspace":0.35})
    methods=["GSCP\n(equal wt.)","sLSCP\n(spatial wt.)","GBM +\nsplit conf."]
    pc=[bundle["verdict"]["A_pooled_global_cov_GSCP"],bundle["verdict"]["A_pooled_global_cov_sLSCP"],bundle["verdict"]["A_pooled_global_cov_GBM"]]
    bc=[SAGE,TEAL,RED]
    x=np.arange(3); axa.bar(x,pc,0.55,color=bc,edgecolor=INK,lw=0.5,zorder=3)
    axa.axhline(0.90,color=INK,lw=0.8,ls="--",zorder=2)
    axa.axhspan(0.85,0.95,color="#E8EFEA",zorder=0,alpha=0.6)
    for i,v in enumerate(pc):
        axa.text(i,v+0.03,f"{v:.0%}",ha="center",fontsize=8,fontweight="bold",color=INK)
    axa.set_xticks(x); axa.set_xticklabels(methods,fontsize=7.5)
    axa.set_ylabel("Pooled coverage",fontsize=8); axa.set_ylim(0,1.05)
    axa.set_title("Pooled global coverage",fontsize=8,pad=6); style_ax(axa)
    axa.text(0.02,0.98,"(a)",transform=axa.transAxes,fontsize=10,fontweight="bold",va="top")
    regions=a["region"].tolist(); y=np.arange(len(regions))
    for i,(_,row) in enumerate(a.iterrows()):
        axb.plot(row["cov_GSCP"],i,"s",color=SAGE,ms=6,mec=INK,mew=0.5,zorder=4)
        axb.plot(row["cov_sLSCP"],i,"o",color=TEAL,ms=6,mec=INK,mew=0.5,zorder=5)
        axb.plot(row["cov_GBM"],i,"D",color=RED,ms=5,mec=INK,mew=0.5,zorder=4)
    axb.axvline(0.90,color=INK,lw=0.8,ls="--",zorder=1)
    axb.axvspan(0.80,1.00,color="#E8EFEA",zorder=0,alpha=0.5)
    axb.set_yticks(y); axb.set_yticklabels([r.replace(" China","") for r in regions],fontsize=7.5)
    axb.set_xlabel("Empirical coverage",fontsize=8); axb.set_xlim(-0.02,1.08); axb.set_ylim(-0.5,len(regions)-0.5)
    axb.set_title("By macro-region",fontsize=8,pad=6)
    el=[Line2D([0],[0],marker="o",color="w",mfc=TEAL,ms=7,mec=INK,mew=0.5,label="sLSCP"),
        Line2D([0],[0],marker="s",color="w",mfc=SAGE,ms=6,mec=INK,mew=0.5,label="GSCP"),
        Line2D([0],[0],marker="D",color="w",mfc=RED,ms=5,mec=INK,mew=0.5,label="GBM"),
        Line2D([0],[0],color=INK,lw=0.8,ls="--",label="0.90")]
    axb.legend(handles=el,frameon=False,fontsize=6.5,loc="lower left")
    axb.text(0.02,0.98,"(b)",transform=axb.transAxes,fontsize=10,fontweight="bold",va="top")
    style_ax(axb); fig.tight_layout(); save(fig,"Fig1 Coverage By Macro Region")

def fig3():
    panel=pd.read_csv(PANEL); piv=panel.pivot(index="province_en",columns="year",values="pm25_popwt")
    anom=piv.sub(piv.mean(axis=1),axis=0); order=anom.mean(axis=1).sort_values(ascending=False).index; anom=anom.loc[order]
    fig,ax=plt.subplots(figsize=(6.8,5.2)); vmax=np.nanmax(np.abs(anom.to_numpy()))*0.65
    im=ax.imshow(anom.to_numpy(),aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(len(anom.columns))); ax.set_xticklabels([str(c) for c in anom.columns],rotation=90,fontsize=6.5)
    ax.set_yticks(range(len(anom.index))); ax.set_yticklabels(anom.index,fontsize=7)
    cbar=fig.colorbar(im,ax=ax,shrink=0.82,pad=0.015); cbar.set_label("PM2.5 anomaly (µg/m³)",fontsize=8); cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("Year",fontsize=8); fig.tight_layout(); save(fig,"Fig3 PM2.5 Anomaly Heatmap")

if __name__=="__main__":
    print("=== figures v3 (Fig1, Fig3; Fig2/Fig4 are produced by 07_contour_maps.py) ==="); fig1(); fig3(); print("\nall done")
