from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.orf import DROrthoForest
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from ms.metaresearch.selector import Selector


class CausalForestSelector(Selector):
    @property
    def name(self) -> str:
        return "causal_forest"

    def __init__(
            self,
            model_y: Any = None,
            model_t: Any = None,
            alpha: float = 0.05,
            lambda_reg: float = 0.01,
            quantile_value: float = 0.8,
            n_splits: int = 5,
            n_estimators: int = 1000,
            n_trees: int=500,
            n_jobs: int = -1,
            mode: str = "individual",
            model_type: str = "CausalForestDML"
    ):
        self.model_y = Lasso(alpha=alpha) if model_y is None else model_y
        self.model_t = Lasso(alpha=alpha) if model_t is None else model_t
        self.lambda_reg = lambda_reg
        self.quantile_value = quantile_value
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.mode = mode  # "individual" or "joint"
        self.model_type = model_type  # "CausalForestDML" or "DROrthoForest"

    def _get_model(self):
        if self.model_type == "CausalForestDML":
            return CausalForestDML(
                model_y=self.model_y,
                model_t=self.model_t,
                n_estimators=self.n_estimators,
                min_samples_leaf=5,
                max_depth=50,
                verbose=0,
                random_state=123
            )
        elif self.model_type == "DROrthoForest":
            return DROrthoForest(
                n_trees=self.n_trees,
                min_leaf_size=5,
                max_depth=50,
                verbose=0,
                random_state=123
            )
        else:
            raise ValueError("Invalid model_type. Choose 'CausalForestDML' or 'DROrthoForest'")

    def _compute_effect_individual(self, x_train, y_train, x_val, i, f_name):
        t_train, t_val = x_train[:, i], x_val[:, i]  # Treatment variable
        x_train_cov = np.delete(x_train, i, axis=1)  # Covariates excluding t
        x_val_cov = np.delete(x_val, i, axis=1)

        dml = self._get_model()

        dml.tune(Y=y_train, T=t_train, X=x_train_cov)
        dml.fit(Y=y_train, T=t_train, X=x_train_cov)
        treatment_effects = dml.effect(x_val_cov)
        return f_name, treatment_effects

    def _compute_effect_joint(self, x_train, y_train, x_val):
        dml = self._get_model()

        dml.tune(Y=y_train, T=x_train, X=x_train)
        dml.fit(Y=y_train, T=x_train, X=x_train)  # All features as treatments
        treatment_effects = dml.effect(x_val)  # Compute effects for all features
        return treatment_effects

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        x_train_arr = x_train.to_numpy()
        y_train_arr = y_train.to_numpy()
        res = pd.DataFrame(index=x_train.columns, columns=["eff_mean"])

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=123)
        effect_results = {f_name: [] for f_name in x_train.columns}

        for train_idx, val_idx in kf.split(x_train_arr):
            x_train_fold, x_val_fold = x_train_arr[train_idx], x_train_arr[val_idx]
            y_train_fold, y_val_fold = y_train_arr[train_idx], y_train_arr[val_idx]

            if self.mode == "individual":
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._compute_effect_individual)(
                        x_train_fold,
                        y_train_fold,
                        x_val_fold,
                        i,
                        f_name
                    )
                    for i, f_name in enumerate(x_train.columns)
                )
                for f_name, treatment_effects in results:
                    effect_results[f_name].extend(treatment_effects)
            elif self.mode == "joint":
                treatment_effects = self._compute_effect_joint(x_train_fold, y_train_fold, x_val_fold)
                for i, f_name in enumerate(x_train.columns):
                    effect_results[f_name].extend(treatment_effects[:, i])
            else:
                raise ValueError("Invalid mode. Choose 'individual' or 'joint'")

        res["eff_mean"] = [np.mean(effect_results[f_name]) for f_name in x_train.columns]
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        quantile_eff = res["eff_mean"].abs().quantile(self.quantile_value)
        res["selected"] = res["eff_mean"].abs() >= quantile_eff
        return res
