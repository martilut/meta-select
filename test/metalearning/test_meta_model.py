import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from ms.metalearning.meta_model import MetaModel


def dummy_scorer(model, x, y):
    return r2_score(y, model.predict(x))


def test_run_without_tuning():
    x = pd.DataFrame(np.random.rand(20, 3))
    y = pd.DataFrame(np.random.rand(20, 1))

    model = MetaModel(
        name="ridge", display_name="Ridge Regression", model=Ridge(), tune=False
    )

    split = {0: {"train": list(range(10)), "test": list(range(10, 20))}}
    results = model.run(
        x=x,
        y=y,
        split=split,
        opt_scoring="r2",
        model_scoring={"r2": dummy_scorer},
        n_trials=1,
    )
    assert isinstance(results, pd.DataFrame)
    assert "train_0" in results.columns
    assert "test_0" in results.columns
    assert "r2" in results.index


def test_optimize_hyperparameters_returns_dict():
    x = pd.DataFrame(np.random.rand(20, 3))
    y = pd.DataFrame(np.random.rand(20, 1))

    model = MetaModel(
        name="ridge",
        display_name="Ridge",
        model=Ridge(),
        params={"alpha": [0.01, 0.1, 1.0, 10.0]},
        tune=True,
    )

    split = {0: {"train": list(range(10)), "test": list(range(10, 20))}}

    best_params = model.optimize_hyperparameters(
        x=x, y=y, split=split, scoring=dummy_scorer, n_trials=2
    )

    assert isinstance(best_params, dict)
    assert "alpha" in best_params
