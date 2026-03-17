"""
routers/visualizaciones.py — Gráficas matplotlib → base64 PNG
"""
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sc
import io, base64
from routers.upload import _df, _infer_type

router = APIRouter(tags=["visualizaciones"])

# ── Paleta (crema ejecutivo) ──────────────────────────────
OFF  = "#F5F4F0"; OFF2 = "#EDECEA"; OFF3 = "#E6E4DF"
NAV  = "#1F2A44"; NAVL = "#263450"
GLD  = "#B79B5E"; GLDD = "#9A8148"; GLDL = "#D4BB8A"
OLV  = "#6B7454"; OLVL = "#8A9670"
TXT  = "#1F2A44"; TXTM = "#4A5568"; TXTD = "#8A9BB0"
RED  = "#C0392B"

def _base_style(ax, fig):
    fig.patch.set_facecolor(OFF)
    ax.set_facecolor(OFF)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(OFF3)
    ax.spines["bottom"].set_color(OFF3)
    ax.tick_params(colors=TXTD, labelsize=7)
    ax.xaxis.label.set_color(TXTM)
    ax.yaxis.label.set_color(TXTM)
    ax.title.set_color(TXTM)
    ax.yaxis.grid(True, color=OFF3, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)

def _encode(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=OFF, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── GET /api/viz/histograma ───────────────────────────────
@router.get("/viz/histograma")
def histograma(col: str, bins: int = 15):
    df = _df()
    s  = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty: raise HTTPException(400, "Sin datos numéricos.")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    edges = np.linspace(float(s.min()), float(s.max()), bins+1)
    n_, _, patches = ax.hist(s.values, bins=edges, color=GLD, edgecolor=GLDD,
                              linewidth=0.5, alpha=0.9)
    for i,p in enumerate(patches):
        p.set_alpha(0.22 + 0.78*(n_[i]/n_.max()))

    mu = s.mean()
    ax.axvline(mu, color=OLV, linewidth=1.4, linestyle="--", alpha=0.85, zorder=5)
    ax.text(mu, ax.get_ylim()[1]*0.88, "  μ", color=OLV, fontsize=7.5, fontweight="600",
            fontfamily="Helvetica Neue")

    ax.set_xlabel(col, labelpad=6, fontsize=8)
    ax.set_ylabel("Frecuencia", labelpad=6, fontsize=8)
    ax.set_title(f"Distribución · {col}", fontsize=9, pad=10)
    _base_style(ax, fig)

    return {
        "img":    _encode(fig),
        "mean":   round(float(mu),4),
        "median": round(float(s.median()),4),
        "std":    round(float(s.std()),4),
        "skew":   round(float(s.skew()),4),
        "n":      len(s),
    }


# ── GET /api/viz/boxplot ──────────────────────────────────
@router.get("/viz/boxplot")
def boxplot(col: str):
    df = _df()
    s  = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty: raise HTTPException(400, "Sin datos numéricos.")

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    bp = ax.boxplot(s.values, vert=True, patch_artist=True,
        boxprops=dict(facecolor=OFF2, color=GLDD, linewidth=1.3),
        medianprops=dict(color=GLDD, linewidth=2.2),
        whiskerprops=dict(color=OFF3, linewidth=1.3),
        capprops=dict(color=OFF3, linewidth=1.3),
        flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor="#9A2010",
                        markersize=4.5, alpha=0.65, linewidth=0.4))
    ax.set_ylabel(col, labelpad=6, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"Boxplot · {col}", fontsize=9, pad=10)
    _base_style(ax, fig)

    q1,q3 = float(s.quantile(.25)), float(s.quantile(.75))
    iqr   = q3-q1
    out_n = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
    return {
        "img": _encode(fig),
        "q1":       round(q1,4),
        "median":   round(float(s.median()),4),
        "q3":       round(q3,4),
        "iqr":      round(iqr,4),
        "outliers": out_n,
        "lo":       round(q1-1.5*iqr,4),
        "hi":       round(q3+1.5*iqr,4),
    }


# ── GET /api/viz/scatter ──────────────────────────────────
@router.get("/viz/scatter")
def scatter(xcol: str, ycol: str, trend: bool = True):
    df = _df()
    tmp = df[[xcol,ycol]].copy()
    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp)<3: raise HTTPException(400,"Datos insuficientes.")

    xv = tmp[xcol].values; yv = tmp[ycol].values
    r,p_r = sc.pearsonr(xv,yv)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.scatter(xv, yv, color=GLD, alpha=0.60, edgecolors=GLDD,
               linewidths=0.4, s=22, zorder=3)
    if trend:
        m,b = np.polyfit(xv,yv,1)
        xs  = np.linspace(xv.min(),xv.max(),200)
        ax.plot(xs, m*xs+b, color=OLV, linewidth=1.6,
                linestyle="--", alpha=0.80, zorder=2)
        ax.set_title(f"ŷ = {m:.4f}x + {b:.4f}   ·   r = {r:.4f}", fontsize=8.5, pad=10)
    else:
        ax.set_title(f"Dispersión · {ycol} vs {xcol}", fontsize=9, pad=10)

    ax.set_xlabel(xcol, labelpad=6, fontsize=8)
    ax.set_ylabel(ycol, labelpad=6, fontsize=8)
    _base_style(ax, fig)

    return {
        "img": _encode(fig),
        "r":   round(float(r),4),
        "p":   round(float(p_r),6),
        "n":   len(tmp),
    }


# ── GET /api/viz/barras ───────────────────────────────────
@router.get("/viz/barras")
def barras(col: str, top: int = 12):
    df = _df()
    s  = df[col].astype(str).replace(["nan","<NA>"],"<nulo>")
    vc = s.value_counts().head(top)
    labels = [str(l)[:16] for l in vc.index]
    counts = vc.values.tolist()

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.bar(range(len(labels)), counts, color=GLD, edgecolor=GLDD, linewidth=0.5)
    for i,b in enumerate(bars):
        b.set_alpha(0.25 + 0.75*(counts[i]/max(counts)))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Frecuencia", labelpad=6, fontsize=8)
    ax.set_title(f"Frecuencia · {col}", fontsize=9, pad=10)
    _base_style(ax, fig)

    total = sum(counts)
    return {
        "img":  _encode(fig),
        "freq": [{"cat":l,"n":c,"pct":round(c/total*100,1)}
                 for l,c in zip(vc.index.tolist(), counts)],
    }


# ── GET /api/viz/grupos ───────────────────────────────────
@router.get("/viz/grupos")
def grupos(col_g: str, col_y: str, max_g: int = 6):
    df = _df()
    top_cats = df[col_g].astype(str).value_counts().head(max_g).index.tolist()
    data  = [pd.to_numeric(df.loc[df[col_g].astype(str)==g,col_y],errors="coerce").dropna().values
             for g in top_cats]
    data  = [d for d in data if len(d)>0]
    labels= [str(l)[:14] for l,d in zip(top_cats,data) if len(d)>0]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bp = ax.boxplot(data, patch_artist=True,
        labels=labels,
        boxprops=dict(facecolor=OFF2, color=GLDD, linewidth=1.2),
        medianprops=dict(color=GLDD, linewidth=2),
        whiskerprops=dict(color=OFF3, linewidth=1.2),
        capprops=dict(color=OFF3, linewidth=1.2),
        flierprops=dict(marker="o",markerfacecolor=RED,markersize=3,alpha=0.5))
    ax.set_ylabel(col_y, labelpad=6, fontsize=8)
    ax.set_title(f"{col_y} por {col_g}", fontsize=9, pad=10)
    plt.xticks(rotation=20, ha="right", fontsize=7.5)
    _base_style(ax, fig)

    medias = [{"grupo":l,"media":round(float(d.mean()),4),"n":len(d)}
              for l,d in zip(labels,data)]
    return {"img": _encode(fig), "medias": medias}
