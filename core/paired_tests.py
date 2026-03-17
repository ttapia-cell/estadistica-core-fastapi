import numpy as np
from scipy import stats

def prueba_pareada_automatica(x1, x2):
    """
    Decide automáticamente entre t pareada y Wilcoxon
    según la normalidad de las diferencias.
    """

    # Convertir a arrays numéricos y limpiar NaN
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    mask = ~np.isnan(x1) & ~np.isnan(x2)
    x1 = x1[mask]
    x2 = x2[mask]

    if len(x1) < 3:
        return {
            "error": "No hay suficientes pares de datos válidos para realizar la prueba."
        }

    # Diferencias
    d = x2 - x1

    # Test de normalidad (Shapiro-Wilk)
    stat_norm, p_norm = stats.shapiro(d)

    # Decisión automática
    if p_norm >= 0.05:
        # t pareada
        stat, p_value = stats.ttest_rel(x2, x1)
        test_usado = "t pareada"
    else:
        # Wilcoxon
        stat, p_value = stats.wilcoxon(x2, x1)
        test_usado = "Wilcoxon"

    return {
        "test": test_usado,
        "n": len(d),
        "estadistico": float(stat),
        "p_value": float(p_value),
        "p_normalidad": float(p_norm)
    }
