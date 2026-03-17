import pandas as pd

def get_categorical_variables(df, max_unique=15, min_unique=2):
    """Detecta variables categóricas válidas en un DataFrame."""
    categorical_vars = []
    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty:
            continue
        unique_values = serie.nunique()
        if serie.dtype.name in ["object", "category", "bool"]:
            if unique_values >= min_unique:
                categorical_vars.append(col)
        elif pd.api.types.is_numeric_dtype(serie):
            if min_unique <= unique_values <= max_unique:
                categorical_vars.append(col)
    return categorical_vars
