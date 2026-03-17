import numpy as np
from scipy.stats import norm

def tamanio_muestra_media(conf_level, E, sigma=None, data=None):
    """
    Tamaño de muestra para la media (población infinita).
    Usa sigma conocida o estimada desde data.
    """

    if sigma is None and data is None:
        return {"error": "Debes proporcionar sigma conocida o una variable para estimarla."}

    # Z crítico
    z = norm.ppf(1 - (1 - conf_level) / 2)

    # Estimar sigma si no se proporciona
    if sigma is None:
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]

        if len(data) < 2:
            return {"error": "No hay suficientes datos para estimar la desviación estándar."}

        sigma = np.std(data, ddof=1)

    n = (z * sigma / E) ** 2

    return {
        "z": float(z),
        "sigma": float(sigma),
        "n": float(n)
    }
