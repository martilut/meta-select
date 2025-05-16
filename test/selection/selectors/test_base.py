import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler


class BaseSelectorTest:
    @staticmethod
    def _assert_valid_output(X_original: pd.DataFrame, X_selected: pd.DataFrame):
        assert isinstance(X_selected, pd.DataFrame), "Output must be a DataFrame"
        assert set(X_selected.columns).issubset(set(X_original.columns)), "Output columns must be a subset of input"
        assert 0 < X_selected.shape[1] <= X_original.shape[1], "Selected features must be non-empty and <= original"

    @staticmethod
    def _generate_classification_data(
            scaled: bool = False,
            n_samples: int = 100,
            n_features: int = 10,
            n_informative: int = 5
    ):
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        if scaled:
            X_df = pd.DataFrame(MinMaxScaler().fit_transform(X_df), columns=X_df.columns)
        return X_df, pd.Series(y)

    @staticmethod
    def _generate_regression_data(
            scaled: bool = False,
            n_samples: int = 100,
            n_features: int = 10,
            n_informative: int = 5
    ):
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        if scaled:
            X_df = pd.DataFrame(MinMaxScaler().fit_transform(X_df), columns=X_df.columns)
        return X_df, pd.Series(y)
