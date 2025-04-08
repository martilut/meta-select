import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer


scalers = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "power": PowerTransformer,
    "quantile": QuantileTransformer
}

def remove_outliers(
        df: pd.DataFrame,
        outlier_modifier: float = 1.0,
) -> pd.DataFrame:
    q1 = df.quantile(0.25, axis="index")
    q3 = df.quantile(0.75, axis="index")
    iqr = q3 - q1

    lower = q1 - outlier_modifier * iqr
    upper = q3 + outlier_modifier * iqr

    for i, feature in enumerate(df.columns):
        feature_col = df[feature]
        feature_col[feature_col < lower[i]] = lower[i]
        feature_col[feature_col > upper[i]] = upper[i]
        df[feature] = feature_col

    return df

def scale(
        df: pd.DataFrame,
        scaler: str,
) -> tuple[pd.DataFrame, BaseEstimator]:
    scaler_init = scalers[scaler]()
    df_scaled = scaler_init.fit_transform(X=df)
    return pd.DataFrame(
        df_scaled,
        columns=df.columns,
        index=df.index
    ), scaler_init

def fill_na(
        df: pd.DataFrame,
        fill_func: str = "median",
) -> pd.DataFrame:
    if fill_func == "median":
        values = df.median(numeric_only=True)
    else:
        values = df.mean(numeric_only=True)
    df.fillna(values, inplace=True)
    return df
