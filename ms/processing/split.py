from random import sample

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ms.utils.navigation import rewrite_decorator
from ms.utils.utils import is_classif


@rewrite_decorator
def split_k_fold(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    outer_k: int = 5,
    inner_k: int | None = 3,
    seed: int = None,
    shuffle: bool = True,
    **kwargs,
) -> dict[int, dict[str, list[int]]]:
    """
    Performs outer and optional inner k-fold cross-validation splitting.

    Args:
        x_df (pd.DataFrame): Feature data.
        y_df (pd.DataFrame): Target data.
        outer_k (int): Number of outer CV folds.
        inner_k (int | None): Number of inner CV folds for nested CV. If None, no inner split is performed.
        seed (int): Random seed for reproducibility.
        shuffle (bool): Whether to shuffle before splitting.

    Returns:
        dict[int, dict[str, list[int]]]: A dictionary of fold indices with optional inner splits.
    """
    splits_dict = {}

    outer_split = (
        StratifiedKFold(n_splits=outer_k, shuffle=shuffle, random_state=seed)
        if is_classif(y_df)
        else KFold(n_splits=outer_k, shuffle=shuffle, random_state=seed)
    )

    for i, (outer_train, outer_test) in enumerate(outer_split.split(x_df, y_df)):
        splits_dict[i] = {
            "train": list(map(int, outer_train)),
            "test": list(map(int, outer_test)),
        }

        if inner_k is not None:
            inner_split = (
                StratifiedKFold(n_splits=inner_k, shuffle=shuffle, random_state=seed)
                if is_classif(y_df)
                else KFold(n_splits=inner_k, shuffle=shuffle, random_state=seed)
            )
            x_train = x_df.iloc[outer_train]
            y_train = y_df.iloc[outer_train]
            splits_dict[i]["inner_split"] = {}

            for j, (inner_train, inner_val) in enumerate(
                inner_split.split(x_train, y_train)
            ):
                splits_dict[i]["inner_split"][j] = {
                    "train": list(map(int, inner_train)),
                    "test": list(map(int, inner_val)),
                }

    return splits_dict


def slice_features(
    x_df: pd.DataFrame,
    slice_sizes: list[int] | None = None,
    n_iter: int = 5,
) -> dict[int, dict[int, list[int]]]:
    """
    Randomly slices features into subsets of specified sizes.

    Args:
        x_df (pd.DataFrame): Feature data.
        slice_sizes (list[int] | None): List of subset sizes. Defaults to all features.
        n_iter (int): Number of slices per size.

    Returns:
        dict[int, dict[int, list[int]]]: A dictionary mapping slice sizes and iteration to selected column indices.
    """
    if slice_sizes is None:
        slice_sizes = [x_df.shape[1]]

    samples_dict = {}
    f_num = x_df.shape[1]
    f_cols = list(range(f_num))

    for size in slice_sizes:
        samples_dict[size] = {}
        for i in range(n_iter):
            slice_indices = sample(f_cols, size)
            samples_dict[size][i] = slice_indices

    return samples_dict


def sample_features(
    suffix: str,
    x_df: pd.DataFrame,
    percents: list[float] | None = None,
    n_iter: int = 1,
) -> dict[int, dict[int, list[int]]]:
    """
    Samples subsets of columns with a specified suffix.

    Args:
        suffix (str): Column name prefix (before `___`) to sample from.
        x_df (pd.DataFrame): Feature data.
        percents (list[float] | None): Percentages of additional features to sample.
        n_iter (int): Number of sampling iterations per percentage.

    Returns:
        dict[int, dict[int, list[int]]]: Dictionary mapping percentage index and iteration to feature indices.
    """
    if percents is None:
        percents = [0.1, 0.5, 1.0]

    samples_dict = {}
    additional_indices = [
        i for i, f in enumerate(x_df.columns) if f.split("___")[0] == suffix
    ]
    original_indices = [
        i for i, f in enumerate(x_df.columns) if f.split("___")[0] != suffix
    ]

    for i, percent in enumerate(percents):
        sample_size = int(len(additional_indices) * percent)
        samples_dict[i] = {}
        for j in range(n_iter):
            sampled = sample(additional_indices, sample_size)
            samples_dict[i][j] = sampled + original_indices

    return samples_dict
