from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier, XGBRegressor

from ms.config.features_list import to_loc
from ms.config.navigation_config import NavigationConfig
from ms.metalearning.meta_model import MetaModel
from ms.processing.preprocess import Preprocessor
from ms.selection.mlp_wrapper import MLPWithCoef
from ms.selection.selectors.base import BaseSelector
from ms.selection.selectors.causal import TESelector
from ms.selection.selectors.ensemble import EnsembleSelector
from ms.selection.selectors.model_based import LassoSelector, XGBSelector
from ms.selection.selectors.model_free import (
    CorrelationSelector,
    FValueSelector,
    MutualInfoSelector,
)
from ms.selection.selectors.model_wrapper import RFESelector


class ExperimentConfig:
    SEED = 42
    OUTER_K = 5
    INNER_K = 3

    PREPROCESSOR = Preprocessor(
        scaler=PowerTransformer(method="yeo-johnson", standardize=True),
        strategy="mean",
    )

    CONF = NavigationConfig()

    @staticmethod
    def get_data(
        features_path: Path, metrics_path: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = pd.read_csv(features_path, index_col=0)
        features = features.loc[:, to_loc]
        metrics = pd.read_csv(metrics_path, index_col=0)
        datasets = features.merge(metrics, on="dataset_name").index
        return features.loc[datasets], metrics.loc[datasets]

    # MetaModels
    KNN_CLASS = MetaModel(
        name="knn",
        display_name="KNN",
        model=KNeighborsClassifier(),
        params={
            "n_neighbors": [3, 5, 7, 9, 11, 15, 20],
            "weights": ["uniform", "distance"],
            "leaf_size": (5, 20),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1],
            "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"],
        },
        tune=True,
    )

    KNN_REG = MetaModel(
        name="knn",
        display_name="KNN",
        model=KNeighborsRegressor(),
        params={
            "n_neighbors": [3, 5, 7, 9, 11, 15, 20],
            "weights": ["uniform", "distance"],
            "leaf_size": (5, 20),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2],
            "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"],
        },
        tune=True,
    )

    XGB_CLASS = MetaModel(
        name="xgb",
        display_name="XGBoost",
        model=XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
        ),
        params={
            "n_estimators": [20, 50, 100, 200],
            "learning_rate": (0.01, 0.3),
            "max_depth": [3, 5, 7, 9, 15],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": [0, 0.1, 0.2, 1, 5],
            "scale_pos_weight": [1, 2, 5, 10],
            "booster": ["gbtree", "gblinear", "dart"],
        },
        tune=True,
    )

    XGB_REG = MetaModel(
        name="xgb",
        display_name="XGBoost",
        model=XGBRegressor(),
        params=XGB_CLASS.params,
        tune=True,
    )

    MLP_CLASS = MetaModel(
        name="mlp",
        display_name="MLP",
        model=MLPClassifier(),
        params={
            "hidden_layer_sizes": [
                (10, 10),
                (20, 20),
                (50, 50),
                (5, 5, 5),
                (10, 10, 10),
                (20, 20, 20),
            ],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "lbfgs", "sgd"],
            "alpha": (0.0001, 0.001, 0.01, 0.05, 0.1, 0.5),
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": (0.001, 0.01, 0.05, 0.1),
            "max_iter": [10, 50, 100, 200],
            "momentum": (0.5, 0.7, 0.9),
        },
        tune=True,
    )

    MLP_REG = MetaModel(
        name="mlp",
        display_name="MLP",
        model=MLPRegressor(),
        params=MLP_CLASS.params,
        tune=True,
    )

    # Feature Selectors
    BASE_SELECTOR = BaseSelector()
    CORR = CorrelationSelector(p_threshold=0.1, corr_threshold=0.2)
    MI = MutualInfoSelector(quantile_value=0.5)
    F_VALUE = FValueSelector(p_threshold=0.05, quantile_value=0.5)

    XGB_SELECTOR = XGBSelector(
        importance_threshold=0.02,
        reg_params=None,
        class_params=None,
        random_state=SEED,
        cv=False,
    )

    LASSO_SELECTOR = LassoSelector(
        coef_threshold=0.0,
        reg_params={"alpha": 0.01, "fit_intercept": True, "random_state": SEED},
        class_params=None,
        random_state=SEED,
        cv=False,
    )

    TE = TESelector(
        model_y=None,
        model_t=None,
        to_tune=False,
        quantile_value=0.7,
        max_depth=20,
        min_leaf=10,
        n_splits=2,
        n_estimators=40,
        n_trees=40,
        n_jobs=-1,
        mode="individual",
        model_type="cf",
        cv=True,
    )

    RFE_CLASS_XGB = RFESelector(
        model=XGB_CLASS.model, rank_threshold=1.0, name="rfe_xgb", cv=False
    )
    RFE_REG_XGB = RFESelector(
        model=XGB_REG.model, rank_threshold=1.0, name="rfe_xgb", cv=False
    )

    RFE_CLASS_MLP = RFESelector(
        model=MLPWithCoef(model=MLP_CLASS.model),
        rank_threshold=1.0,
        name="rfe_mlp",
        cv=False,
    )

    RFE_REG_MLP = RFESelector(
        model=MLPWithCoef(model=MLP_REG.model),
        rank_threshold=1.0,
        name="rfe_mlp",
        cv=False,
    )

    ENSEMBLE = EnsembleSelector(
        selectors=[CORR, MI, F_VALUE, XGB_SELECTOR, LASSO_SELECTOR]
    )

    # Scoring
    OPT_SCORING_CLASS = "b_acc"
    MODEL_SCORING_CLASS = {
        "b_acc": make_scorer(balanced_accuracy_score),
        "f1": make_scorer(f1_score, average="weighted"),
    }

    OPT_SCORING_REG = "rmse"
    MODEL_SCORING_REG = {
        "mae": make_scorer(mean_absolute_error),
        "rmse": make_scorer(root_mean_squared_error),
        "r2": make_scorer(r2_score),
    }
