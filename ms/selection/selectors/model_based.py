import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from xgboost import XGBClassifier, XGBRegressor

from ms.selection.selector import Selector


class XGBSelector(Selector):
    def __init__(
            self,
            importance_threshold: float = 0.0,
            reg_params: dict | None = None,
            class_params: dict | None = None,
            random_state: int | None = None,
            cv: bool = False,
    ) -> None:
        super().__init__(cv=cv)
        self.importance_threshold = importance_threshold
        self.reg_params = reg_params if reg_params is not None else {
            "random_state": random_state
        }
        self.class_params = class_params if class_params is not None else {
            "random_state": random_state
        }

    @property
    def name(self) -> str:
        return "xgb"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        xgb = XGBClassifier(**self.class_params) \
            if task == "class" \
            else XGBRegressor(**self.reg_params)
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
            coef_threshold: float = 0.0,
            reg_params: dict | None = None,
            class_params: dict | None = None,
            random_state: int | None = None,
            cv: bool = False,
    ) -> None:
        super().__init__(cv=cv)
        self.coef_threshold = coef_threshold
        self.reg_params = reg_params if reg_params is not None else {
            "alpha": 0.15,
            "random_state": random_state,
            "fit_intercept": True,

        }
        self.class_params = class_params if class_params is not None else {
            "penalty": "l1",
            "solver": "liblinear",
            "C": 0.15,
            "random_state": random_state,
            "fit_intercept": True
        }

    @property
    def name(self) -> str:
        return "lasso"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        model = LogisticRegression(**self.class_params) \
            if task == "class" \
            else Lasso(**self.reg_params)

        model.fit(X=x_train, y=y_train)

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = model.coef_.flatten()
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["value"].abs() <= self.coef_threshold, "value"] = None
        return res
