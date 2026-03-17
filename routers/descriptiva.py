"""
routers/descriptiva.py — Estadística descriptiva
"""
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from routers.upload import _df, _safe, _infer_type

router = APIRouter(tags=["descriptiva"])


@router.get("/descriptiva/{col}")
def descriptiva_numerica(col: str):
    df = _df()
    if col not in df.columns:
        raise HTTPException(400, f"Columna '{col}' no existe.")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise HTTPException(400, "Sin valores numéricos.")

    q1,q3 = s.quantile(.25), s.quantile(.75)
    iqr   = q3-q1
    cv    = s.std()/s.mean()*100 if s.mean() != 0 else None

    stats = [
        ("N",           int(len(s)),              "Observaciones válidas"),
        ("Media (μ)",   round(float(s.mean()),4),  "Tendencia central"),
        ("Mediana",     round(float(s.median()),4),"P50"),
        ("Moda",        round(float(s.mode().iloc[0]),4) if not s.mode().empty else None, "Valor más frecuente"),
        ("Desv. Std (σ)",round(float(s.std()),4),  "Dispersión"),
        ("Varianza",    round(float(s.var()),4),   "σ²"),
        ("CV (%)",      round(cv,2) if cv else None,"Coef. de variación"),
        ("Mínimo",      round(float(s.min()),4),   "Valor mínimo"),
        ("Máximo",      round(float(s.max()),4),   "Valor máximo"),
        ("Rango",       round(float(s.max()-s.min()),4),"Máx − Mín"),
        ("P25 (Q1)",    round(float(q1),4),        "Cuartil inferior"),
        ("P75 (Q3)",    round(float(q3),4),        "Cuartil superior"),
        ("IQR",         round(float(iqr),4),       "Rango intercuartil"),
        ("Asimetría",   round(float(s.skew()),4),  "γ₁ (cola " + ("derecha" if s.skew()>0 else "izquierda") + ")"),
        ("Curtosis",    round(float(s.kurtosis()),4),"Exceso de curtosis"),
    ]

    freq = _freq_agrupada(s, 10)
    return {
        "col":   col,
        "n":     len(s),
        "stats": [{"stat":a,"value":_safe(b),"interp":c} for a,b,c in stats],
        "freq":  freq,
        "kpis":  {
            "mean":   round(float(s.mean()),4),
            "median": round(float(s.median()),4),
            "std":    round(float(s.std()),4),
            "skew":   round(float(s.skew()),4),
        }
    }


@router.get("/descriptiva-cat/{col}")
def descriptiva_categorica(col: str):
    df = _df()
    if col not in df.columns:
        raise HTTPException(400, f"Columna '{col}' no existe.")
    s = df[col].astype(str).replace("nan","<nulo>").replace("<NA>","<nulo>")
    vc = s.value_counts()
    total = len(s)
    return {
        "col":     col,
        "n":       total,
        "unicos":  int(s.nunique()),
        "moda":    str(vc.index[0]) if not vc.empty else "—",
        "freq_moda": int(vc.iloc[0]) if not vc.empty else 0,
        "freq": [
            {
                "categoria": str(cat),
                "conteo":    int(cnt),
                "pct":       round(cnt/total*100, 2),
            }
            for cat, cnt in vc.items()
        ][:30],  # max 30
    }


@router.get("/columns")
def list_columns():
    df = _df()
    cols = []
    for c in df.columns:
        cols.append({"name": c, "type": _infer_type(df[c])})
    return {"columns": cols}


def _freq_agrupada(s: pd.Series, bins: int) -> list:
    try:
        cut, edges = pd.cut(s, bins=bins, retbins=True, include_lowest=True)
        vc   = cut.value_counts().sort_index()
        total = len(s)
        rows = []
        fa = 0
        for interval, cnt in vc.items():
            fa += int(cnt)
            rows.append({
                "clase":    str(interval),
                "fi":       int(cnt),
                "fri":      round(int(cnt)/total*100, 2),
                "Fa":       fa,
                "Fra":      round(fa/total*100, 2),
            })
        return rows
    except Exception:
        return []
