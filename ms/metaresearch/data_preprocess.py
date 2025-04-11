import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target


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

        transformer_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.strategy)),
            ('scaler', self.scaler)
        ])

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


def is_classif(y: pd.Series) -> bool:
    target_type = type_of_target(y)
    return target_type in {"binary", "multiclass"}

def cv_decorator(func):
    def wrapper(
            *args,
            **kwargs
    ):
        x: pd.DataFrame = kwargs.get("x", None)
        y: pd.DataFrame = kwargs.get("y", None)
        split: dict = kwargs.get("split", None)
        subset: dict = kwargs.get("subset", None)
        preprocessor: Preprocessor = kwargs.get("preprocessor", None)
        to_agg: bool = kwargs.get("to_agg", True)

        if x is None or y is None or split is None:
            return func(
                x_train=x,
                y_train=y,
                *args,
                **kwargs
            )
        cv_res = []
        for i in split:
            x_train = x.iloc[split[i]["train"], :]
            y_train = y.iloc[split[i]["train"], :]
            x_test = x.iloc[split[i]["test"], :]
            y_test = y.iloc[split[i]["test"], :]

            if subset is not None:
                x_train = x_train.loc[:, subset[i]]
                x_test = x_test.loc[:, subset[i]]

            inner_split = split[i].get("inner_split", None)

            if preprocessor is not None:
                x_fitted = preprocessor.fit(x_train)
                x_train = x_fitted.transform(x_train)
                x_test = x_fitted.transform(x_test)
                if not is_classif(y=y_train):
                    y_fitted = preprocessor.fit(y_train)
                    y_train = y_fitted.transform(y_train)
                    y_test = y_fitted.transform(y_test)

            y_type = "class" if is_classif(y_train) else "reg"
            print(f"Split {i}, "
                  f"x_train: {x_train.shape}, "
                  f"x_test: {x_test.shape}, "
                  f"y_train: {y_train.shape}, "
                  f"y_test: {y_test.shape}, "
                  f"y type: {y_type}, "
                  f"has inner_split: {inner_split is not None}")

            res = func(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                inner_split=inner_split,
                *args,
                **kwargs
            )
            res.columns = [f"{col}_{i}" for col in res.columns]
            cv_res.append(res)
        cv_res = pd.concat(cv_res, axis=1)
        return cv_res.mean(
            axis=1,
            skipna=False
        ).to_frame(name="value") if to_agg else cv_res
    return wrapper
