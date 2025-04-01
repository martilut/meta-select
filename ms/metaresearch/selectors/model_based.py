import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from xgboost import XGBClassifier, XGBRegressor

from ms.metaresearch.selector import Selector


class XGBSelector(Selector):
    def __init__(
            self,
            importance_threshold: float = 0.0
    ) -> None:
        self.importance_threshold = importance_threshold

    @property
    def name(self) -> str:
        return "xgb"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame | None = None,
            y_test: pd.Series | None = None,
    ) -> pd.DataFrame:
        xgb = XGBClassifier()
        xgb.fit(X=x_train, y=y_train)

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = xgb.feature_importances_
        return res

    def compute_regression(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame | None = None,
            y_test: pd.Series | None = None,
    ) -> pd.DataFrame:
        xgb = XGBRegressor()
        xgb.fit(X=x_train, y=y_train)

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = xgb.feature_importances_
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["value"].abs() <= self.importance_threshold, "value"] = None
        return res


class LassoSelector(Selector):
    def __init__(
            self,
            coef_threshold: float = 0.0
    ) -> None:
        self.coef_threshold = coef_threshold

    @property
    def name(self) -> str:
        return "lasso"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame | None = None,
            y_test: pd.Series | None = None,
    ) -> pd.DataFrame:
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.15,
            random_state=42,
            fit_intercept=True,
        )
        model.fit(X=x_train, y=y_train)

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = model.coef_.flatten()
        return res

    def compute_regression(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame | None = None,
            y_test: pd.Series | None = None,
    ) -> pd.DataFrame:
        model = Lasso(
            alpha=0.15,
            random_state=42
        )
        model.fit(X=x_train, y=y_train)

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = model.coef_
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["value"].abs() <= self.coef_threshold, "value"] = None
        return res
