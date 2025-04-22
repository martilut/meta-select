from pathlib import Path

import pandas as pd
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score, roc_auc_score, mean_squared_error, \
    mean_absolute_error
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
from ms.selection.selectors.causal import TESelector, CFSelector
from ms.selection.selectors.model_based import XGBSelector, LassoSelector
from ms.selection.selectors.model_free import CorrelationSelector, MutualInfoSelector, FValueSelector
from ms.selection.selectors.model_wrapper import RFESelector

SEED = 42
PREPROCESSOR = Preprocessor(
    scaler=PowerTransformer(
        method="yeo-johnson",
        standardize=True
    ),
    strategy="mean",
)

CONF = NavigationConfig()

OUTER_K = 5
INNER_K = 3

def get_data(
        features_path: Path,
        metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_path, index_col=0)
    features = features.loc[:, to_loc]
    metrics = pd.read_csv(metrics_path, index_col=0)
    datasets = features.merge(metrics, on="dataset_name").index
    features = features.loc[datasets]
    metrics = metrics.loc[datasets]
    return features, metrics


knn_class_mm = MetaModel(
    name="knn",
    display_name="KNN",
    model=KNeighborsClassifier(),
    params={
        "n_neighbors": 6,
        "weights": "uniform",
        "leaf_size": 40,
        "algorithm": "auto",
        "p": 1,
    }
)
knn_reg_mm = MetaModel(
    name="knn",
    display_name="KNN",
    model=KNeighborsRegressor(),
    params={
        "n_neighbors": [3, 5, 7, 9, 11, 15, 20],
        "weights": ["uniform", "distance"],
        "leaf_size": (20, 50),
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2],
        "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"]
    }
)
xgb_class_mm = MetaModel(
    name="xgb",
    display_name="XGBoost",
    model=XGBClassifier(),
    params={
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 50,
        "eval_metric": "merror",
    },
)
xgb_reg_mm = MetaModel(
    name="xgb",
    display_name="XGBoost",
    model=XGBRegressor(),
    params={
        "n_estimators": [50, 100, 200, 500, 1000],
        "learning_rate": (0.01, 0.3),
        "max_depth": [3, 5, 7, 9, 15],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "gamma": [0, 0.1, 0.2, 1, 5],
        "scale_pos_weight": [1, 2, 5, 10],
        "booster": ["gbtree", "gblinear", "dart"],
    }
)
mlp_class_mm = MetaModel(
    name="mlp",
    display_name="MLP",
    model=MLPClassifier(),
    params={
        "hidden_layer_sizes": 25,
        "activation": "logistic",
        "solver": "lbfgs",
        "alpha": 0.05,
        "batch_size": "auto",
        "learning_rate": "adaptive",
        "learning_rate_init": 0.05,
        "max_iter": 100,
    },
)
mlp_reg_mm = MetaModel(
    name="mlp",
    display_name="MLP",
    model=MLPRegressor(),
    params={
        "hidden_layer_sizes": [(50,), (100,), (150,), (200,)],
        "activation": ["relu", "tanh", "logistic"],
        "solver": ["adam", "lbfgs", "sgd"],
        "alpha": (0.0001, 0.1),
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": (0.001, 0.01),
        "max_iter": [200, 500, 1000, 2000],
        "momentum": (0.5, 0.9),
        "early_stopping": [True, False],
    },
)


base = BaseSelector()
corr = CorrelationSelector(
    p_threshold=0.05,
    corr_threshold=0.3,
)
mi = MutualInfoSelector(
    quantile_value=0.5,
)
f_value = FValueSelector(
    p_threshold=0.05,
    quantile_value=0.5,
)
xgb = XGBSelector(
    importance_threshold=0.0,
    reg_params=None,
    class_params=None,
    random_state=SEED,
    cv=False,
)
lasso = LassoSelector(
    coef_threshold=0.0,
    reg_params={
        "alpha":0.01,
        "fit_intercept": True,
        "random_state": SEED,
    },
    class_params=None,
    random_state=SEED,
    cv=False,
)
te = TESelector(
    model_y=None,
    model_t=None,
    to_tune=False,
    quantile_value=0.5,
    max_depth=10,
    min_leaf=5,
    n_splits=2,
    n_estimators=20,
    n_trees=40,
    n_jobs=-1,
    mode="individual",
    model_type="cf", # or "drof"
    random_state=SEED,
    cv=True,
)
cf = CFSelector(
    cf_steps=100,
    train_epochs=100,
    dc=0.2,
    device="cpu",
    cv=False,
)
rfe_class_xgb = RFESelector(
    model=xgb_class_mm.model,
    rank_threshold=1.0,
    name="rfe_xgb",
    cv=False,
)
rfe_reg_xgb = RFESelector(
    model=xgb_reg_mm.model,
    rank_threshold=1.0,
    name="rfe_xgb",
    cv=False,
)
rfe_class_mlp = RFESelector(
    model=MLPWithCoef(model=mlp_class_mm.model),
    rank_threshold=1.0,
    name="rfe_mlp",
    cv=False,
)
rfe_reg_mlp = RFESelector(
    model=MLPWithCoef(model=mlp_reg_mm.model),
    rank_threshold=1.0,
    name="rfe_mlp",
    cv=False,
)


opt_scoring_class = "b_acc"
model_scoring_class = {
    'b_acc': make_scorer(balanced_accuracy_score),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc': make_scorer(
        roc_auc_score,
        average='weighted',
        max_fpr=None,
        multi_class="ovo",
        response_method="predict_proba"
    ),
}

opt_scoring_reg = "mse"
model_scoring_reg = {
    'mse': make_scorer(
        mean_squared_error,
    ),
    'mae': make_scorer(
        mean_absolute_error,
    ),
}
