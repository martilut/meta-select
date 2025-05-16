from typing import Callable, Dict

import optuna
import pandas as pd
from sklearn.base import BaseEstimator

from ms.processing.cv import cv_decorator
from ms.processing.preprocess import Preprocessor
from ms.utils.navigation import rewrite_decorator


class MetaModel:
    """
    Wrapper class for machine learning models with built-in support for
    hyperparameter tuning using Optuna and cross-validation evaluation.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        model: BaseEstimator,
        params: dict | None = None,
        tune: bool = False,
    ):
        """
        Initialize a MetaModel instance.

        Args:
            name (str): Internal identifier for the model.
            display_name (str): Human-readable name for display.
            model (BaseEstimator): Scikit-learn compatible model.
            params (dict | None): Hyperparameter search space.
            tune (bool): Whether to tune hyperparameters.
        """
        self.name = name
        self.display_name = display_name
        self.model = model
        self.params = params
        self.tune = tune
        self.best_params = None

    @rewrite_decorator
    def run(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        split: dict,
        opt_scoring: str,
        model_scoring: Dict[str, Callable],
        n_trials: int,
        preprocessor: Preprocessor | None = None,
        subset: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Entry point for model training and evaluation.

        Args:
            x (pd.DataFrame): Feature data.
            y (pd.DataFrame): Target data.
            split (dict): Train/test split indices.
            opt_scoring (str): Metric name for optimization.
            model_scoring (dict): Metrics for model evaluation.
            n_trials (int): Number of Optuna trials.
            preprocessor (Preprocessor | None): Optional preprocessing pipeline.
            subset (dict | None): Subset of features for each fold.

        Returns:
            pd.DataFrame: Evaluation results.
        """
        print(f"Meta-model: {self.name}")

        return self.train_and_evaluate(
            x=x,
            y=y,
            split=split,
            subset=subset,
            preprocessor=preprocessor,
            to_agg=False,
            opt_scoring=opt_scoring,
            model_scoring=model_scoring,
            n_trials=n_trials,
        )

    @cv_decorator
    def train_and_evaluate(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        inner_split: dict,
        opt_scoring: str,
        model_scoring: Dict[str, Callable],
        n_trials: int,
        **kwargs,
    ) -> pd.DataFrame:
        best_params = (
            self.optimize_hyperparameters(
                x=x_train,
                y=y_train,
                split=inner_split,
                scoring=model_scoring[opt_scoring],
                n_trials=n_trials,
            )
            if self.tune
            else self.model.get_params()
        )
        self.best_params = best_params
        self.model.set_params(**best_params)

        self.model.fit(x_train.values, y_train.values.ravel())
        result = pd.DataFrame(
            index=[name for name in model_scoring.keys()], columns=["train", "test"]
        )
        for name, func in model_scoring.items():
            train_score = func(self.model, x_train.values, y_train.values.ravel())
            test_score = func(self.model, x_test.values, y_test.values.ravel())
            result.loc[name] = [train_score, test_score]
        return result

    def optimize_hyperparameters(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        split: dict,
        scoring: Callable,
        n_trials: int,
    ) -> dict:
        if not self.params:
            return self.model.get_params()

        print("Optimizing hyperparameters using Optuna")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(
                trial=trial, x=x, y=y, split=split, scoring=scoring
            ),
            n_trials=n_trials,
        )
        return study.best_params

    def objective(
        self,
        trial: optuna.Trial,
        x: pd.DataFrame,
        y: pd.DataFrame,
        split: dict,
        scoring: Callable,
    ) -> float:
        param_grid = {
            param: (
                trial.suggest_int(param, min(values), max(values))
                if all(isinstance(v, int) for v in values)
                else trial.suggest_float(param, min(values), max(values))
                if all(isinstance(v, float) for v in values)
                else trial.suggest_categorical(param, values)
            )
            for param, values in self.params.items()
        }

        self.model.set_params(**param_grid)

        @cv_decorator
        def optuna_score(
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_test: pd.DataFrame,
            **kwargs,
        ):
            self.model.fit(x_train.values, y_train.values.ravel())
            score = scoring(self.model, x_test.values, y_test.values.ravel())
            return pd.DataFrame([{"score": score}])

        scores_df = optuna_score(x=x, y=y, split=split, to_agg=True)
        return scores_df.iloc[0, 0]

    @rewrite_decorator
    def save_params(self, *args, **kwargs) -> dict:
        return self.best_params if self.best_params else self.model.get_params()
