"""
routers/upload.py — Carga, perfil y limpieza de datos
v2: soporte de múltiples hojas Excel + endpoint /api/demo
CORREGIDO: Exportación Excel y CSV con nombres de archivo correctos
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io
import re
import os

router = APIRouter(tags=["upload"])

# ── Sesión en memoria ─────────────────────────────────────
_session: dict = {}

def _df() -> pd.DataFrame:
    if "df" not in _session:
        raise HTTPException(400, "No hay datos cargados. Sube un archivo primero.")
    return _session["df"]

def _safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)):    return bool(v)
    return v

def _row(r):
    return {k: _safe(v) for k,v in r.items()}

def sanitize_filename(filename: str) -> str:
    """
    Elimina caracteres problemáticos y asegura nombre base limpio.
    Maneja cualquier extensión y caracteres especiales.
    """
    if not filename:
        return "datos_limpios"
    
    # Eliminar extensión (cualquier extensión)
    base = re.sub(r'\.[^.]+$', '', filename)
    
    # Eliminar caracteres problemáticos para sistemas de archivos
    base = re.sub(r'[<>:"/\\|?*]', '_', base)
    
    # Eliminar espacios al inicio/final y reemplazar múltiples espacios
    base = re.sub(r'\s+', ' ', base.strip())
    
    # Si queda vacío, usar nombre por defecto
    if not base or base == '.':
        base = "datos_limpios"
    
    return base


# ── POST /api/upload ──────────────────────────────────────
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    sheet_name: str  = Form(None),   # opcional — nombre de hoja Excel
):
    content = await file.read()
    fname   = file.filename
    sheets  = []
    current_sheet = None

    try:
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        elif fname.lower().endswith((".xlsx", ".xls")):
            # Leer lista de hojas
            xf = pd.ExcelFile(io.BytesIO(content))
            sheets = xf.sheet_names

            # Elegir hoja: la pedida, o la primera
            if sheet_name and sheet_name in sheets:
                current_sheet = sheet_name
            else:
                current_sheet = sheets[0]

            df = pd.read_excel(io.BytesIO(content), sheet_name=current_sheet)
        else:
            raise HTTPException(400, "Formato no soportado. Usa CSV o Excel.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Error leyendo archivo: {e}")

    # Normalizar nulos
    df.replace(["", " ", "NA", "N/A", "n/a", "null", "NULL", "None", "none",
                "#N/A", "#NA", "nan", "NaN"], pd.NA, inplace=True)

    _session["df"]            = df
    _session["original"]      = df.copy()
    _session["logs"]          = []
    _session["file_content"]  = content   # guardar para cambio de hoja
    _session["filename"]      = file.filename
    _session["sheets"]        = sheets
    _session["current_sheet"] = current_sheet

    return {
        "filename":      file.filename,
        "rows":          len(df),
        "cols":          len(df.columns),
        "columns":       list(df.columns),
        "preview":       [_row(r) for r in df.head(12).to_dict("records")],
        "profile":       _build_profile(df),
        "sheets":        sheets,
        "current_sheet": current_sheet,
    }


# ── POST /api/switch-sheet ────────────────────────────────
@router.post("/switch-sheet")
async def switch_sheet(payload: dict):
    """Cambia de hoja sin re-subir el archivo (usa el contenido en sesión)."""
    sheet = payload.get("sheet")
    content = _session.get("file_content")
    sheets  = _session.get("sheets", [])
    fname   = _session.get("filename", "")

    if not content:
        raise HTTPException(400, "No hay archivo en sesión. Sube el archivo primero.")
    if sheet not in sheets:
        raise HTTPException(400, f"Hoja '{sheet}' no existe en el archivo.")

    try:
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet)
    except Exception as e:
        raise HTTPException(400, f"Error leyendo hoja: {e}")

    df.replace(["", " ", "NA", "N/A", "n/a", "null", "NULL", "None", "none",
                "#N/A", "#NA", "nan", "NaN"], pd.NA, inplace=True)

    _session["df"]            = df
    _session["original"]      = df.copy()
    _session["logs"]          = []
    _session["current_sheet"] = sheet

    return {
        "filename":      fname,
        "rows":          len(df),
        "cols":          len(df.columns),
        "columns":       list(df.columns),
        "preview":       [_row(r) for r in df.head(12).to_dict("records")],
        "profile":       _build_profile(df),
        "sheets":        sheets,
        "current_sheet": sheet,
    }


# ── POST /api/demo ────────────────────────────────────────
@router.post("/demo")
async def generate_demo():
    """Genera un dataset de demostración con variables numéricas y categóricas."""
    np.random.seed(42)
    n = 200

    edad        = np.random.randint(22, 65, n)
    ingresos    = np.round(np.random.lognormal(8.2, 0.45, n), 2)
    gasto       = np.round(ingresos * np.random.uniform(0.25, 0.75, n), 2)
    antiguedad  = np.round(np.abs(np.random.normal(5, 3, n)), 1)
    satisfaccion= np.random.randint(1, 6, n).astype(float)
    region      = np.random.choice(["Sierra", "Costa", "Oriente", "Insular"], n,
                                    p=[0.40, 0.38, 0.15, 0.07])
    sector      = np.random.choice(["Comercio", "Servicios", "Industria", "Agro"], n,
                                    p=[0.30, 0.35, 0.25, 0.10])

    # Inyectar algunos nulos y outliers
    idx_null = np.random.choice(n, 8, replace=False)
    ingresos[idx_null[:4]] = np.nan
    gasto[idx_null[4:]] = np.nan
    ingresos[np.random.choice(n, 3)] *= 8   # outliers

    df = pd.DataFrame({
        "EDAD":          edad,
        "INGRESOS":      ingresos,
        "GASTO_MENSUAL": gasto,
        "ANTIGUEDAD":    antiguedad,
        "SATISFACCION":  satisfaccion,
        "REGION":        region,
        "SECTOR":        sector,
    })

    _session["df"]            = df
    _session["original"]      = df.copy()
    _session["logs"]          = []
    _session["file_content"]  = None
    _session["filename"]      = "demo_dataset.csv"
    _session["sheets"]        = []
    _session["current_sheet"] = None

    return {
        "filename":      "demo_dataset.csv",
        "rows":          len(df),
        "cols":          len(df.columns),
        "columns":       list(df.columns),
        "preview":       [_row(r) for r in df.head(12).to_dict("records")],
        "profile":       _build_profile(df),
        "sheets":        [],
        "current_sheet": None,
    }


# ── GET /api/profile ──────────────────────────────────────
@router.get("/profile")
def get_profile():
    df = _df()
    return {"profile": _build_profile(df), "rows": len(df), "cols": len(df.columns)}


# ── POST /api/clean ───────────────────────────────────────
@router.post("/clean")
def clean_column(payload: dict):
    df  = _df()
    col = payload.get("column")
    method = payload.get("method", "median_impute")

    if col not in df.columns:
        raise HTTPException(400, f"Columna '{col}' no existe.")

    s = pd.to_numeric(df[col], errors="coerce")
    n_before = s.isna().sum()

    if method == "median_impute":
        df[col] = s.fillna(s.median())
        detail = f"Imputación con mediana ({s.median():.4f})"
    elif method == "mean_impute":
        df[col] = s.fillna(s.mean())
        detail = f"Imputación con media ({s.mean():.4f})"
    elif method == "drop_nulls":
        mask = s.notna()
        df = df[mask].reset_index(drop=True)
        detail = f"Eliminación de {(~mask).sum()} filas con nulos"
    elif method == "drop_outliers":
        q1,q3 = s.quantile(.25), s.quantile(.75)
        iqr   = q3-q1
        mask  = (s >= q1-1.5*iqr) & (s <= q3+1.5*iqr)
        df    = df[mask | s.isna()].reset_index(drop=True)
        detail = f"Eliminación de {(~mask & s.notna()).sum()} outliers IQR"
    elif method == "winsorize":
        q1,q3 = s.quantile(.05), s.quantile(.95)
        df[col] = s.clip(lower=q1, upper=q3)
        detail = f"Winsorización P5-P95 [{q1:.2f}, {q3:.2f}]"
    else:
        raise HTTPException(400, f"Método '{method}' no reconocido.")

    _session["df"] = df
    log = {"column": col, "method": method, "detail": detail}
    _session["logs"].append(log)

    return {
        "ok":      True,
        "detail":  detail,
        "rows":    len(df),
        "logs":    _session["logs"],
        "profile": _build_profile(df),
    }


# ── GET /api/diagnostics/{col} ────────────────────────────
@router.get("/diagnostics/{col}")
def diagnostics(col: str):
    df = _df()
    if col not in df.columns:
        raise HTTPException(400, f"Columna '{col}' no existe.")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise HTTPException(400, "La columna no tiene valores numéricos.")
    q1,q3 = s.quantile(.25), s.quantile(.75)
    iqr   = q3-q1
    lo,hi = q1-1.5*iqr, q3+1.5*iqr
    out_n = int(((s<lo)|(s>hi)).sum())
    return {
        "n":        int(len(s)),
        "n_null":   int(df[col].isna().sum()),
        "pct_null": round(df[col].isna().mean()*100, 2),
        "pct_out":  round(out_n/len(s)*100, 2) if len(s) else 0,
        "outliers": out_n,
        "skew":     round(float(s.skew()), 4),
        "iqr":      round(float(iqr), 4),
        "q1":       round(float(q1), 4),
        "q3":       round(float(q3), 4),
        "lo":       round(float(lo), 4),
        "hi":       round(float(hi), 4),
        "recommendation": _recommend(s, out_n/len(s) if len(s) else 0, df[col].isna().mean()),
    }


# ── GET /api/columns ──────────────────────────────────────
@router.get("/columns")
def list_columns():
    df = _df()
    return {"columns": [{"name": c, "type": _infer_type(df[c])} for c in df.columns]}


# ── GET /api/export/csv ───────────────────────────────────
@router.get("/export/csv")
def export_csv():
    """
    Exporta el dataset limpio como CSV.
    CORREGIDO: Genera nombres de archivo correctos (sin caracteres extraños)
    """
    df  = _df()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    
    # Generar nombre de archivo limpio
    fname = _session.get("filename", "datos.csv")
    base_name = sanitize_filename(fname)
    out_name = f"{base_name}_limpio.csv"
    
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )


# ── GET /api/export/xlsx ──────────────────────────────────
@router.get("/export/xlsx")
def export_xlsx():
    """
    Exporta el dataset limpio como Excel, preservando hojas originales.
    CORREGIDO: Genera nombres de archivo correctos (sin .xlsxx)
    """
    df       = _df()
    fname    = _session.get("filename", "datos.xlsx")
    content  = _session.get("file_content")
    sheets   = _session.get("sheets", [])
    cur_sh   = _session.get("current_sheet")

    buf = io.BytesIO()

    if content and sheets:
        # Preservar hojas no modificadas, reemplazar la hoja actual con datos limpios
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sh in sheets:
                if sh == cur_sh:
                    df.to_excel(writer, sheet_name=sh, index=False)
                else:
                    try:
                        df_other = pd.read_excel(io.BytesIO(content), sheet_name=sh)
                        df_other.to_excel(writer, sheet_name=sh, index=False)
                    except Exception as e:
                        # Si falla, al menos crear una hoja con mensaje de error
                        error_df = pd.DataFrame({"Error": [f"No se pudo cargar la hoja original: {str(e)}"]})
                        error_df.to_excel(writer, sheet_name=f"{sh}_error", index=False)
    else:
        # Si no hay múltiples hojas, exportar solo una
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Datos_Limpios", index=False)

    buf.seek(0)
    
    # 🟢 CORREGIDO: Generar nombre de archivo correctamente
    base_name = sanitize_filename(fname)
    out_name = f"{base_name}_limpio.xlsx"
    
    return StreamingResponse(
        iter([buf.read()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={out_name}"}
    )


# ── HELPERS ───────────────────────────────────────────────
def _infer_type(s: pd.Series) -> str:
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().sum() / max(s.notna().sum(), 1) >= 0.85:
        return "numerica"
    if s.nunique() / max(s.notna().sum(), 1) < 0.20:
        return "categorica"
    return "texto"

def _build_profile(df: pd.DataFrame) -> list:
    rows = []
    for col in df.columns:
        tipo = _infer_type(df[col])
        rows.append({
            "columna":     col,
            "tipo":        tipo,
            "n":           int(df[col].notna().sum()),
            "missing":     int(df[col].isna().sum()),
            "pct_missing": round(df[col].isna().mean()*100, 1),
            "unicos":      int(df[col].nunique()),
            "ejemplo":     str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—",
        })
    return rows

def _recommend(s, pout, pna) -> str:
    skew = abs(s.skew())
    if pna > 0.30:   return "Alta proporción de nulos — considera eliminar la columna o usar imputación avanzada."
    if pna > 0.05 and skew > 1: return "Nulos moderados + asimetría alta → imputar con mediana."
    if pna > 0.05:   return "Nulos moderados → imputar con media o mediana."
    if pout > 0.10:  return "Alta proporción de outliers → winsorización (P5-P95) o eliminación."
    if pout > 0.02:  return "Outliers detectados → winsorización recomendada para análisis paramétrico."
    return "Calidad aceptable — sin transformación necesaria."