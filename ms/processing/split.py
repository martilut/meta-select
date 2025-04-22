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
    splits_dict = {}

    if is_classif(y=y_df):
        outer_split = StratifiedKFold(
            n_splits=outer_k,
            shuffle=shuffle,
            random_state=seed,
        )
    else:
        outer_split = KFold(
            n_splits=outer_k,
            shuffle=shuffle,
            random_state=seed,
        )

    outer_splits = outer_split.split(x_df, y_df)

    for i, (outer_train, outer_test) in enumerate(outer_splits):
        splits_dict[i] = {
            "train": list(map(int, outer_train)),
            "test": list(map(int, outer_test)),
        }

        if inner_k is not None:
            if is_classif(y=y_df):
                inner_split = StratifiedKFold(
                    n_splits=inner_k,
                    shuffle=shuffle,
                    random_state=seed,
                )
            else:
                inner_split = KFold(
                    n_splits=inner_k,
                    shuffle=shuffle,
                    random_state=seed,
                )
            x_train = x_df.iloc[outer_train]
            y_train = y_df.iloc[outer_train]
            splits_dict[i]["inner_split"] = {}
            inner_splits = inner_split.split(x_train, y_train)
            for j, (inner_train, inner_val) in enumerate(inner_splits):
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
    if slice_sizes is None:
        slice_sizes = [x_df.shape[1]]

    samples_dict = {}

    f_num = x_df.shape[1]
    f_cols = [i for i in range(f_num)]

    for size in slice_sizes:
        samples_dict[size] = {}
        for i in range(n_iter):
            slice_sizes = sample(f_cols, size)
            samples_dict[size][i] = slice_sizes

    return samples_dict


def sample_features(
        suffix: str,
        x_df: pd.DataFrame,
        percents: list[float] | None = None,
        n_iter: int = 1,
) -> dict[int, dict[int, list[int]]]:
    if percents is None:
        percents = [0.1, 0.5, 1.0]

    samples_dict = {}
    additional_indices = []
    original_indices = []
    for i, f in enumerate(list(x_df.columns)):
        if f.split("___")[0] == suffix:
            additional_indices.append(i)
        else:
            original_indices.append(i)

    for i, percent in enumerate(percents):
        sample_size = int(len(additional_indices) * percent)
        samples_dict[i] = {}
        for j in range (n_iter):
            samples_dict[i][j] = sample(additional_indices, sample_size) + original_indices

    return samples_dict
