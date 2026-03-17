"""
routers/inferencia.py
Pruebas de hipótesis, IC, regresión, correlación, contingencia.
Usa core/relacional.py y core/sample_size_mean.py directamente.
"""
from fastapi import APIRouter, HTTPException
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io as _io, base64
from scipy import stats as sc
from routers.upload import _df, _safe
from core.paired_tests import prueba_pareada_automatica
from core.sample_size_mean import tamanio_muestra_media
from core.data_validation import get_categorical_variables

router = APIRouter(tags=["inferencia"])

OFF="#F5F4F0"; OFF3="#E6E4DF"
GLD="#B79B5E"; GDD="#9A8148"; OLV="#6B7454"
TXM="#4A5568"; TXD="#8A9BB0"

def _b64(fig):
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=OFF)
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _ax_style(ax, fig):
    fig.patch.set_facecolor(OFF); ax.set_facecolor(OFF)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(OFF3)
    ax.tick_params(colors=TXD, labelsize=7)
    ax.xaxis.label.set_color(TXM); ax.yaxis.label.set_color(TXM)
    ax.yaxis.grid(True, color=OFF3, linewidth=0.8, alpha=0.9); ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)

def _decision(pval, alpha):
    rechaza = pval <= alpha
    return {
        "rechaza": rechaza,
        "texto": f"Se rechaza H\u2080 (p = {pval:.6f} \u2264 \u03b1 = {alpha})" if rechaza
                 else f"No se rechaza H\u2080 (p = {pval:.6f} > \u03b1 = {alpha})",
    }

def r4(v): return round(float(v), 4)
def r6(v): return round(float(v), 6)

@router.post("/ic-media")
def ic_media(payload: dict):
    col=payload.get("col"); conf=float(payload.get("conf",0.95))
    s=pd.to_numeric(_df()[col],errors="coerce").dropna()
    if len(s)<2: raise HTTPException(400,"Al menos 2 observaciones.")
    n=len(s); mean=float(s.mean()); se=float(sc.sem(s))
    t_c=float(sc.t.ppf((1+conf)/2,df=n-1))
    return {"mean":r4(mean),"se":r4(se),"t_crit":r4(t_c),"gl":n-1,"n":n,
            "lo":r4(mean-t_c*se),"hi":r4(mean+t_c*se),"conf":conf}

@router.post("/tamano-muestra")
def tamano_muestra(payload: dict):
    col = payload.get("col")
    conf = float(payload.get("conf", 0.95))
    E = float(payload.get("E", 1.0))
    sigma = payload.get("sigma")

    data_arr = None
    df = _df()

    # Si se seleccionó una columna, extraer los datos numéricos
    if col and col in df.columns:
        data_arr = pd.to_numeric(df[col], errors="coerce").dropna().values

    # Llamar a la función del módulo externo
    result = tamanio_muestra_media(
        conf_level=conf,
        E=E,
        sigma=float(sigma) if sigma is not None else None,
        data=data_arr
    )

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Agregar n redondeado al alza
    result["n_ceil"] = int(np.ceil(result["n"]))
    return result

@router.post("/ttest-1")
def ttest_1m(payload: dict):
    col=payload.get("col"); mu0=float(payload.get("mu0",0))
    alt=payload.get("alt","two-sided"); alpha=float(payload.get("alpha",0.05))
    s=pd.to_numeric(_df()[col],errors="coerce").dropna()
    if len(s)<2: raise HTTPException(400,"Datos insuficientes.")
    t,p=sc.ttest_1samp(s,popmean=mu0,alternative=alt)
    return {"t":r4(t),"p":r6(p),"n":len(s),"mean":r4(float(s.mean())),"mu0":mu0,"alt":alt,"decision":_decision(float(p),alpha)}

@router.post("/ttest-2")
def ttest_2ind(payload: dict):
    col_g=payload.get("col_g"); col_y=payload.get("col_y")
    g1=payload.get("g1"); g2=payload.get("g2")
    alt=payload.get("alt","two-sided"); alpha=float(payload.get("alpha",0.05))
    eq=bool(payload.get("equal_var",False)); df=_df()
    x1=pd.to_numeric(df.loc[df[col_g].astype(str)==str(g1),col_y],errors="coerce").dropna()
    x2=pd.to_numeric(df.loc[df[col_g].astype(str)==str(g2),col_y],errors="coerce").dropna()
    if len(x1)<2 or len(x2)<2: raise HTTPException(400,"Grupos insuficientes.")
    t,p=sc.ttest_ind(x1,x2,equal_var=eq,alternative=alt)
    return {"t":r4(t),"p":r6(p),"n1":len(x1),"n2":len(x2),
            "mean1":r4(float(x1.mean())),"mean2":r4(float(x2.mean())),"decision":_decision(float(p),alpha)}

@router.post("/ztest-prop")
def ztest_prop(payload: dict):
    phat=float(payload.get("phat",0.5)); p0=float(payload.get("p0",0.5))
    n=int(payload.get("n",100)); alt=payload.get("alt","two-sided"); alpha=float(payload.get("alpha",0.05))
    se=np.sqrt(p0*(1-p0)/n); z=(phat-p0)/se
    p=2*(1-sc.norm.cdf(abs(z))) if alt=="two-sided" else (1-sc.norm.cdf(z) if alt=="greater" else sc.norm.cdf(z))
    return {"z":r4(z),"p":r6(p),"phat":phat,"p0":p0,"n":n,"decision":_decision(float(p),alpha)}

@router.post("/paired")
def paired(payload: dict):
    c1 = payload.get("c1")
    c2 = payload.get("c2")
    alpha = float(payload.get("alpha", 0.05))
    df = _df()

    x1 = pd.to_numeric(df[c1], errors="coerce")
    x2 = pd.to_numeric(df[c2], errors="coerce")

    # Llamar a la función del módulo externo
    result = prueba_pareada_automatica(x1, x2)

    if "error" in result:
        raise HTTPException(400, result["error"])

    return {
        "test": result["test"],
        "stat": round(float(result["estadistico"]), 4),
        "p": round(float(result["p_value"]), 6),
        "n": result["n"],
        "mean_diff": round(float((x2 - x1).mean()), 4),
        "shapiro_p": round(float(result["p_normalidad"]), 4),
        "decision": {
            "rechaza": result["p_value"] <= alpha,
            "texto": f"Se rechaza H₀ (p = {result['p_value']:.6f} ≤ α = {alpha})" if result["p_value"] <= alpha
                     else f"No se rechaza H₀ (p = {result['p_value']:.6f} > α = {alpha})"
        }
    }

@router.get("/correlacion")
def correlacion(x: str, y: str, method: str = "pearson"):
    from core.relacional import correlacion as _corr
    df=_df()
    if x not in df.columns or y not in df.columns: raise HTTPException(400,"Columnas no encontradas.")
    result=_corr(df[x],df[y],method=method)
    if result["r"] is None: raise HTTPException(400,"Datos insuficientes.")
    both=pd.DataFrame({"x":pd.to_numeric(df[x],errors="coerce"),"y":pd.to_numeric(df[y],errors="coerce")}).dropna()
    m,b=np.polyfit(both["x"],both["y"],1)
    fig,ax=plt.subplots(figsize=(7,3.2))
    ax.scatter(both["x"],both["y"],color=GLD,alpha=0.6,edgecolors=GDD,linewidths=0.4,s=20,zorder=3)
    xs=np.linspace(both["x"].min(),both["x"].max(),200)
    ax.plot(xs,m*xs+b,color=OLV,linewidth=1.4,linestyle="--",alpha=0.8,zorder=2)
    ax.set_xlabel(x,fontsize=8,labelpad=5); ax.set_ylabel(y,fontsize=8,labelpad=5)
    ax.set_title(f"r ({method}) = {result['r']:.4f}   p = {result['p']:.6f}",fontsize=8.5,color=TXM,pad=10)
    _ax_style(ax,fig)
    result["img"]=_b64(fig); result["r"]=r4(result["r"]); result["p"]=r6(result["p"])
    return result

@router.get("/regresion")
def regresion(y: str, x: str, conf: float = 0.95):
    from core.relacional import regresion_lineal as _reg
    df = _df()
    if y not in df.columns or x not in df.columns: raise HTTPException(400,"Columnas no encontradas.")
    result = _reg(df, y=y, X=[x])
    if result.get("modelo") is None: raise HTTPException(400,"Datos insuficientes.")

    coef  = result["coeficientes"];  pval   = result["pvalues"]
    se    = result["se_params"];     tvals  = result["t_params"]
    b0    = coef["const"];           b1     = coef[x]
    n     = result["n"];             r2     = result["r2"]
    mse   = result["mse"]
    f_s   = result["f_stat"];        p_f    = result["p_f"]

    # IC coeficientes con t crítico
    t_c   = float(sc.t.ppf((1+conf)/2, df=n-2))
    ci_b0_lo = r4(b0 - t_c*se["const"]);  ci_b0_hi = r4(b0 + t_c*se["const"])
    ci_b1_lo = r4(b1 - t_c*se[x]);        ci_b1_hi = r4(b1 + t_c*se[x])

    # Datos para gráfico
    tmp = df[[y,x]].copy()
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp = tmp.dropna()
    xv = tmp[x].values.astype(float); yv = tmp[y].values.astype(float)
    x_bar = xv.mean(); ss_xx = ((xv-x_bar)**2).sum()

    fig,ax = plt.subplots(figsize=(7.5,3.4))
    ax.scatter(xv,yv,color=GLD,alpha=0.58,edgecolors=GDD,linewidths=0.4,s=22,zorder=3)
    xs  = np.linspace(xv.min(),xv.max(),200); ys = b0+b1*xs
    se_m= np.sqrt(mse*(1/n+(xs-x_bar)**2/ss_xx))
    ax.plot(xs,ys,color=OLV,linewidth=1.6,linestyle="--",alpha=0.85,zorder=2)
    ax.fill_between(xs,ys-t_c*se_m,ys+t_c*se_m,alpha=0.10,color=OLV)
    ax.set_title(f"\u0177 = {b0:.4f} + {b1:.4f}\u00b7{x}   \u00b7   R\u00b2 = {r2:.4f}",fontsize=8.5,color=TXM,pad=10)
    ax.set_xlabel(x,fontsize=8,labelpad=5); ax.set_ylabel(y,fontsize=8,labelpad=5)
    _ax_style(ax,fig)

    return {
        "b0":r4(b0),       "b1":r4(b1),
        "se_b0":r4(se["const"]),  "se_b1":r4(se[x]),
        "t_b0":r4(tvals["const"]),"t_b1":r4(tvals[x]),
        "p_b0":r6(pval["const"]), "p_b1":r6(pval[x]),
        "ci_b0_lo":ci_b0_lo, "ci_b0_hi":ci_b0_hi,
        "ci_b1_lo":ci_b1_lo, "ci_b1_hi":ci_b1_hi,
        "r2":round(r2,6), "f_stat":r4(f_s), "p_f":r6(p_f),
        "n":n, "img":_b64(fig),
    }

@router.post("/contingencia")
def contingencia(payload: dict):
    fila = payload.get("fila")
    col = payload.get("col")
    alpha = float(payload.get("alpha", 0.05))
    df = _df()
    
    # Validar que ambas columnas existan
    if fila not in df.columns or col not in df.columns:
        raise HTTPException(400, "Columnas no encontradas.")
    
    # Validar que sean categóricas (usando nuestra nueva función)
    cat_vars = get_categorical_variables(df)
    if fila not in cat_vars:
        raise HTTPException(400, f"'{fila}' no es una variable categórica válida para esta prueba.")
    if col not in cat_vars:
        raise HTTPException(400, f"'{col}' no es una variable categórica válida para esta prueba.")
    
    # Verificar que haya al menos 2 categóricas en total
    if len(cat_vars) < 2:
        raise HTTPException(400, "Se necesitan al menos dos variables categóricas para esta prueba.")

@router.post("/generar-datos")
def generar_datos(payload: dict):
    from core.dataset_generator import (dataset_numerico,dataset_dos_grupos,
                                         dataset_correlacion,dataset_regresion_multiple)
    from routers.upload import _session, _build_profile
    tipo=payload.get("tipo","correlacion"); n=int(payload.get("n",150)); seed=int(payload.get("seed",42))
    if tipo=="numerico":
        df=dataset_numerico(n=n,distribucion=payload.get("dist","normal"),
                             mu=float(payload.get("mu",0)),sigma=float(payload.get("sigma",1)),seed=seed)
    elif tipo=="dos_grupos":
        df=dataset_dos_grupos(n1=n//2,n2=n//2,mu1=float(payload.get("mu1",0)),
                               mu2=float(payload.get("mu2",1)),sigma=float(payload.get("sigma",1)),seed=seed)
    elif tipo=="correlacion":
        df=dataset_correlacion(n=n,beta0=float(payload.get("beta0",0)),
                                beta1=float(payload.get("beta1",1)),ruido=float(payload.get("ruido",1)),seed=seed)
    elif tipo=="regresion_multiple":
        df=dataset_regresion_multiple(n=n,ruido=float(payload.get("ruido",1)),seed=seed)
    else:
        raise HTTPException(400,f"Tipo '{tipo}' no reconocido.")
    _session.update({"df":df,"original":df.copy(),"logs":[],"file_content":None,
                     "filename":f"demo_{tipo}.csv","sheets":[],"current_sheet":None})
    return {"filename":f"demo_{tipo}.csv","rows":len(df),"cols":len(df.columns),
            "columns":list(df.columns),"preview":[{k:_safe(v) for k,v in r.items()} for r in df.head(12).to_dict("records")],
            "profile":_build_profile(df),"sheets":[],"current_sheet":None}
