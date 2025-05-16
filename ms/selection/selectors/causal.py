from typing import Any

import numpy as np
import pandas as pd
from dowhy import CausalModel
from econml.dml import CausalForestDML
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor

from ms.selection.selector import Selector


class TESelector(Selector):
    @property
    def name(self) -> str:
        return "te"

    def __init__(
        self,
        model_y: Any = None,
        model_t: Any = None,
        to_tune: bool = True,
        quantile_value: float = 0.8,
        max_depth: int = 50,
        min_leaf: int = 5,
        n_splits: int = 2,
        n_estimators: int = 100,
        n_trees: int = 500,
        n_jobs: int = -1,
        mode: str = "individual",
        model_type: str = "cf",  # or "drof"
        random_state: int | None = None,
        cv: bool = True,
    ):
        super().__init__(cv=cv)
        default_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
        )
        self.model_y = default_model if model_y is None else model_y
        self.model_t = default_model if model_t is None else model_t
        self.to_tune = to_tune
        self.quantile_value = quantile_value
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.mode = mode  # "individual" or "joint"
        self.model_type = model_type
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def _get_model(self):
        if self.model_type == "cf":
            return CausalForestDML(
                model_y=self.model_y,
                model_t=self.model_t,
                cv=2,
                criterion="het",
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_leaf,
                max_depth=self.max_depth,
                fit_intercept=True,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif self.model_type == "drof":
            return DROrthoForest(
                n_trees=self.n_trees,
                min_leaf_size=self.min_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Invalid model_type. Choose 'cf' or 'drof'")

    def _compute_effect_individual(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        i: int,
        f_name: str,
    ):
        print(f"Processing feature {f_name}..., {i}/{x_train.shape[1]}")
        t_train, t_test = x_train.iloc[:, i].to_numpy(), x_test.iloc[:, i].to_numpy()
        x_train_cov = np.delete(x_train, i, axis=1)
        x_test_cov = np.delete(x_test, i, axis=1)

        dml = self._get_model()

        if self.to_tune:
            dml.tune(Y=y_train.to_numpy().ravel(), T=t_train, X=x_train_cov)

        dml.fit(Y=y_train.to_numpy().ravel(), T=t_train, X=x_train_cov)
        treatment_effects = dml.effect(x_test_cov)
        return f_name, treatment_effects

    def _compute_effect_joint(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ):
        dml = self._get_model()

        if self.to_tune:
            dml.tune(Y=y_train, T=x_train, X=x_train)

        dml.fit(Y=y_train, T=x_train, X=x_train)
        treatment_effects = dml.effect(x_test)
        return treatment_effects

    def compute_generic(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        task: str = "class",
    ) -> pd.DataFrame:
        x_test = x_train if x_test is None else x_test

        res = pd.DataFrame(index=x_train.columns, columns=["value"])

        effect_results = {f_name: [] for f_name in x_train.columns}

        if self.mode == "individual":
            print("Processing individual features...")
            results = [
                self._compute_effect_individual(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    i=i,
                    f_name=f_name,
                )
                for i, f_name in enumerate(x_train.columns)
            ]
            for f_name, treatment_effects in results:
                effect_results[f_name].extend(treatment_effects)
        elif self.mode == "joint":
            print("Processing joint features...")
            treatment_effects = self._compute_effect_joint(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
            )
            for i, f_name in enumerate(x_train.columns):
                effect_results[f_name].extend(treatment_effects[:, i])
        else:
            raise ValueError("Invalid mode. Choose 'individual' or 'joint'")

        res["value"] = [np.mean(effect_results[f_name]) for f_name in x_train.columns]
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        quantile_eff = res["value"].abs().quantile(self.quantile_value)
        for f_name in res.index:
            if abs(res.loc[f_name, "value"]) < quantile_eff:
                res.loc[f_name, "value"] = None
        return res


class TEDAGSelector(Selector):
    @property
    def name(self) -> str:
        return "te_dag"

    def __init__(
        self,
        cv: bool = True,
        reg_method: str = "backdoor.linear_regression",
        class_method: str = "backdoor.propensity_score_matching",
        method_params: dict | None = None,
    ) -> None:
        super().__init__(cv=cv)
        self.reg_method = reg_method
        self.class_method = class_method
        self.method_params = method_params

    def compute_generic(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        task: str = "class",
    ) -> pd.DataFrame:
        train = x_train.copy()
        train["y"] = y_train.copy()
        test = x_test.copy() if x_test is not None else x_train.copy()
        test["y"] = y_test.copy() if y_test is not None else y_train.copy()
        res = {}
        for feature in x_train.columns:
            model = CausalModel(
                data=train,
                treatment=feature,
                outcome="y",
                common_causes=[i for i in train.columns if i not in ["y", feature]],
            )
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True, method_name="maximal-adjustment"
            )
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name=self.reg_method,
                method_params=self.method_params,
            )
            res[feature] = causal_estimate.value
        res = pd.DataFrame.from_dict(res, orient="index", columns=["value"])
        res.index = x_train.columns
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["value"].abs() == 0.0, "value"] = None
        return res
