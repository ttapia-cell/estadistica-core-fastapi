from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats

def correlacion(x: pd.Series, y: pd.Series, method: str = "pearson") -> dict:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    df = pd.DataFrame({"x": a, "y": b}).dropna()
    if df.shape[0] < 3:
        return {"n": int(df.shape[0]), "r": None, "p": None, "method": method}

    if method == "pearson":
        r, p = stats.pearsonr(df["x"], df["y"])
    elif method == "spearman":
        r, p = stats.spearmanr(df["x"], df["y"])
    else:
        raise ValueError("method debe ser pearson o spearman")
    return {"n": int(df.shape[0]), "r": float(r), "p": float(p), "method": method}

def regresion_lineal(df: pd.DataFrame, y: str, X: list[str]) -> dict:
    """
    Regresión OLS: y ~ X. Devuelve coeficientes, R2, predicciones.
    """
    data = df[[y] + X].copy()
    for c in [y] + X:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    if data.shape[0] < (len(X) + 2):
        return {"n": int(data.shape[0]), "modelo": None}

    yv  = data[y].values.astype(float)
    Xv  = np.column_stack([np.ones(len(yv)), data[X].values.astype(float)])
    k   = Xv.shape[1]   # intercepto + predictores
    n   = len(yv)

    # OLS: β = (XᵀX)⁻¹ Xᵀy
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    params  = XtX_inv @ Xv.T @ yv
    y_hat   = Xv @ params
    resid   = yv - y_hat
    sse     = float(resid @ resid)
    sst     = float(((yv - yv.mean()) ** 2).sum())
    r2      = 1.0 - sse / sst if sst > 0 else 0.0
    mse     = sse / (n - k)

    # Errores estándar, t, p
    se_params = np.sqrt(np.diag(XtX_inv) * mse)
    t_params  = params / se_params
    p_params  = [float(2 * stats.t.sf(abs(t), df=n - k)) for t in t_params]

    coef  = {"const": float(params[0])}
    pvals = {"const": p_params[0]}
    for i, name in enumerate(X, start=1):
        coef[name]  = float(params[i])
        pvals[name] = p_params[i]

    # F-statistic
    ssr    = sst - sse
    f_stat = (ssr / (k - 1)) / mse if mse > 0 else 0.0
    p_f    = float(stats.f.sf(f_stat, k - 1, n - k))

    return {
        "n":            n,
        "r2":           r2,
        "coeficientes": coef,
        "pvalues":      pvals,
        "se_params":    {("const" if i == 0 else X[i-1]): float(se_params[i]) for i in range(k)},
        "t_params":     {("const" if i == 0 else X[i-1]): float(t_params[i])  for i in range(k)},
        "f_stat":       f_stat,
        "p_f":          p_f,
        "mse":          mse,
        "predicciones": y_hat.tolist(),
        "residuales":   resid.tolist(),
        # expose Xv stats needed for CI bands
        "_params_arr":  params.tolist(),
        "_XtX_inv":     XtX_inv.tolist(),
        "modelo":       True,  # flag de éxito
    }

def tabla_contingencia(df: pd.DataFrame, fila: str, columna: str, dropna: bool = False) -> dict:
    """
    Devuelve tablas de contingencia:
    - conteos
    - porcentajes por fila
    - porcentajes por columna
    - porcentajes globales
    Incluye totales.
    """
    a = df[fila]
    b = df[columna]

    # Conteos
    ct = pd.crosstab(a, b, dropna=dropna, margins=True, margins_name="Total")

    # Porcentaje global
    total = ct.loc["Total", "Total"]
    pct_global = (ct / total * 100).round(2)

    # Porcentaje por fila (excluye la fila Total para el cálculo, luego reanexa)
    ct_no_total_row = ct.drop(index="Total")
    row_sums = ct_no_total_row.sum(axis=1)
    pct_fila = (ct_no_total_row.div(row_sums, axis=0) * 100).round(2)
    pct_fila["Total"] = 100.00
    pct_fila.loc["Total"] = (ct.loc["Total"] / total * 100).round(2)

    # Porcentaje por columna (excluye la columna Total para el cálculo, luego reanexa)
    ct_no_total_col = ct.drop(columns="Total")
    col_sums = ct_no_total_col.sum(axis=0)
    pct_col = (ct_no_total_col.div(col_sums, axis=1) * 100).round(2)
    pct_col.loc["Total"] = (ct.loc["Total"].drop("Total") / total * 100).round(2)
    pct_col["Total"] = 100.00

    return {
        "conteos": ct,
        "pct_global": pct_global,
        "pct_fila": pct_fila,
        "pct_col": pct_col
    }

def chi2_independencia(df: pd.DataFrame, fila: str, columna: str, dropna: bool = False) -> dict:
    """
    Prueba Chi-cuadrado de independencia para dos variables categóricas.
    Retorna: chi2, gl, p, tabla esperada (E).
    """
    from scipy.stats import chi2_contingency

    # Tabla sin totales para el test
    tabla = pd.crosstab(df[fila], df[columna], dropna=dropna)

    chi2, p, dof, expected = chi2_contingency(tabla)

    expected_df = pd.DataFrame(expected, index=tabla.index, columns=tabla.columns)

    return {
        "chi2": float(chi2),
        "gl": int(dof),
        "p": float(p),
        "esperados": expected_df
    }

def v_cramer(df: pd.DataFrame, fila: str, columna: str, dropna: bool = False) -> float:
    """
    Calcula V de Cramér como tamaño de efecto para chi-cuadrado.
    """
    import numpy as np
    from scipy.stats import chi2_contingency

    tabla = pd.crosstab(df[fila], df[columna], dropna=dropna)
    chi2, _, _, _ = chi2_contingency(tabla)

    n = tabla.values.sum()
    r, k = tabla.shape

    v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
    return float(v)
