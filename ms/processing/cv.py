import pandas as pd

from ms.processing.preprocess import Preprocessor
from ms.utils.utils import is_classif


def cv_decorator(func):
    """
    Decorator for applying cross-validation logic to a model evaluation function.

    Handles:
    - Splitting data according to provided `split` dictionary
    - Applying column-wise feature subset if `subset` is provided
    - Fitting and applying a `Preprocessor` if provided
    - Optionally aggregating results across splits

    Expected decorated function signature:
    def func(x_train, y_train, x_test, y_test, inner_split=None, ...)

    Args (via kwargs):
        x (pd.DataFrame): Full feature DataFrame.
        y (pd.DataFrame): Full target DataFrame.
        split (dict): Dictionary containing train/test indices for each fold.
        subset (dict): Dictionary mapping folds to feature subsets.
        preprocessor (Preprocessor): Optional preprocessing object.
        to_agg (bool): Whether to average results across folds (default: True).

    Returns:
        pd.DataFrame: Aggregated or per-split evaluation results.
    """

    def wrapper(*args, **kwargs):
        x: pd.DataFrame = kwargs.get("x", None)
        y: pd.DataFrame = kwargs.get("y", None)
        split: dict = kwargs.get("split", None)
        subset: dict = kwargs.get("subset", None)
        preprocessor: Preprocessor = kwargs.get("preprocessor", None)
        to_agg: bool = kwargs.get("to_agg", True)

        if split is None:
            # No CV; run once
            if x is not None and y is not None:
                return func(x_train=x, y_train=y, *args, **kwargs)
            return func(*args, **kwargs)

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
                # if not is_classif(y=y_train):
                #     y_fitted = preprocessor.fit(y_train)
                #     y_train = y_fitted.transform(y_train)
                #     y_test = y_fitted.transform(y_test)

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
