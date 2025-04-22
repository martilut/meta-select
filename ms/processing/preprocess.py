import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            scaler: BaseEstimator = None,
            strategy: str = 'mean',
    ):
        """
        Preprocessing pipeline that imputes missing values and scales numerical features.

        Parameters:
        - scaler: A scikit-learn scaler instance (e.g., StandardScaler(), MinMaxScaler()).
        """
        self.scaler = scaler
        self.strategy = strategy
        self.numeric_columns = None
        self.column_transformer = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.numeric_columns = X.columns.tolist()
        steps = [
            ('imputer', SimpleImputer(strategy=self.strategy))
        ]
        if self.scaler is not None:
            steps.append(('scaler', self.scaler))
        transformer_pipeline = Pipeline(steps=steps)

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', transformer_pipeline, self.numeric_columns)
            ],
            remainder='drop'
        )

        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        if self.column_transformer is None:
            raise RuntimeError("You must call fit before transform.")

        transformed = self.column_transformer.transform(X)
        return pd.DataFrame(transformed, columns=X.columns, index=X.index)

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.transform(X)
