from pathlib import Path

import numpy as np
import pandas as pd

from ms.processing.cv import cv_decorator
from ms.processing.preprocess import Preprocessor
from ms.selection.selector import Selector
from ms.utils.navigation import pjoin, rewrite_decorator


@rewrite_decorator
def run_selector(
        selector: Selector,
        features: pd.DataFrame,
        metrics: pd.DataFrame,
        split: dict,
        preprocessor: Preprocessor | None = None,
        k_best: int | None = None,
        save_path: Path | None = None,
        to_rewrite: bool = False,
) -> dict:
    result_dict = {}
    errors_dict = {}

    target_res = run_target(
        x=features,
        y=metrics,
        split=split,
        preprocessor=preprocessor,
        to_agg=False,
        selector=selector,
        k_best=k_best,
        save_path=pjoin(
            save_path.parent,
            "data.csv"
        ),
        to_rewrite=to_rewrite,
    )

    for i in split:
        result_dict[i] = target_res.loc[:, f"value_{i}"].dropna().index.to_list()
        k_best_safe = k_best if k_best is not None else 1
        if len(result_dict[i]) < k_best_safe:
            errors_dict[i] = len(result_dict[i])

    if len(errors_dict) > 0:
        save_errors(
            errors_dict=errors_dict,
            save_path=pjoin(
                save_path.parent,
                "errors.json"
            ),
            to_rewrite=to_rewrite,
        )

    return result_dict


@rewrite_decorator
@cv_decorator
def run_target(
    selector: Selector,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    preprocessor: Preprocessor | None = None,
    k_best: int | None = None,
    inner_split: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    return selector.compute_select(
        x=x_train,
        y=y_train,
        split=inner_split if selector.cv else None,
        preprocessor=preprocessor,
        to_agg=True,
        k_best=k_best,
    )


@rewrite_decorator
def save_errors(
        errors_dict: dict,
        **kwargs: dict,
) -> dict:
    return errors_dict


def aggregate_selector_folds(
        df: pd.DataFrame,
) -> list[str]:
    df.index = df.iloc[:, 0]
    value_cols = [col for col in df.columns if col.startswith("value_")]
    values = df[value_cols]

    non_nan_counts = values.notna().sum(axis=1)
    min_required = len(value_cols) // 2 + 1
    selected_mask = non_nan_counts >= min_required
    filtered = values[selected_mask]

    def has_consistent_sign(row):
        signs = np.sign(row.dropna())
        return np.all(signs > 0) or np.all(signs < 0)

    consistent_mask = filtered.apply(has_consistent_sign, axis=1)
    final_df = filtered[consistent_mask]

    feature_scores = final_df.abs().mean(axis=1)
    sorted_features = feature_scores.sort_values(ascending=False).index.tolist()

    return sorted_features


def reduce_features(selector_features: dict) -> dict:
    lengths = [len(feats) for feats in selector_features.values() if feats]
    if not lengths:
        return {selector: [] for selector in selector_features}  # all empty

    M = min(lengths)

    reduced = {
        selector: features[:M] if len(features) >= M else features
        for selector, features in selector_features.items()
    }

    return reduced
