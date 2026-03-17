import numpy as np
import pandas as pd

# =========================
# SEMANA 1 – DESCRIPTIVA
# =========================

def dataset_numerico(
    n: int = 100,
    distribucion: str = "normal",
    mu: float = 0.0,
    sigma: float = 1.0,
    minimo: float | None = None,
    maximo: float | None = None,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Genera un dataset numérico con una sola variable X.
    """
    if seed is not None:
        np.random.seed(seed)

    if distribucion == "normal":
        x = np.random.normal(mu, sigma, n)
    elif distribucion == "uniforme":
        x = np.random.uniform(minimo if minimo is not None else 0,
                               maximo if maximo is not None else 1, n)
    elif distribucion == "exponencial":
        x = np.random.exponential(scale=1/mu if mu != 0 else 1, size=n)
    else:
        raise ValueError("Distribución no soportada")

    return pd.DataFrame({"X": x})


def dataset_categorico(
    categorias: list[str] = ["A", "B", "C"],
    probabilidades: list[float] | None = None,
    n: int = 100,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Genera un dataset con una variable categórica.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.choice(categorias, size=n, p=probabilidades)
    return pd.DataFrame({"Categoria": x})


# =========================
# SEMANA 3 – HIPÓTESIS
# =========================

def dataset_dos_grupos(
    n1: int = 50,
    n2: int = 50,
    mu1: float = 0.0,
    mu2: float = 1.0,
    sigma: float = 1.0,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Genera dos grupos para pruebas t de medias.
    """
    if seed is not None:
        np.random.seed(seed)

    g1 = np.random.normal(mu1, sigma, n1)
    g2 = np.random.normal(mu2, sigma, n2)

    return pd.DataFrame({
        "Grupo": ["A"] * n1 + ["B"] * n2,
        "Y": np.concatenate([g1, g2])
    })


# =========================
# SEMANA 4 – RELACIONAL
# =========================

def dataset_correlacion(
    n: int = 100,
    beta0: float = 0.0,
    beta1: float = 1.0,
    ruido: float = 1.0,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Genera X e Y con relación lineal.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.normal(0, 1, n)
    e = np.random.normal(0, ruido, n)
    y = beta0 + beta1 * x + e

    return pd.DataFrame({"X": x, "Y": y})


def dataset_regresion_multiple(
    n: int = 100,
    betas: dict[str, float] | None = None,
    ruido: float = 1.0,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Genera un dataset para regresión múltiple.
    """
    if seed is not None:
        np.random.seed(seed)

    if betas is None:
        betas = {"X1": 1.0, "X2": 0.5}

    data = {}
    y = np.zeros(n)

    for var, beta in betas.items():
        x = np.random.normal(0, 1, n)
        data[var] = x
        y += beta * x

    y += np.random.normal(0, ruido, n)
    data["Y"] = y

    return pd.DataFrame(data)
