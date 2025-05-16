import numpy as np
import pandas as pd

from ms.processing.split import sample_features, slice_features, split_k_fold


def test_split_k_fold_classification():
    x = pd.DataFrame(np.random.rand(100, 10))
    y = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)))
    splits = split_k_fold(x, y, outer_k=5, inner_k=2, seed=42)
    assert isinstance(splits, dict)
    assert "train" in splits[0] and "test" in splits[0]
    assert "inner_split" in splits[0]
    assert len(splits) == 5
    assert len(splits[0]["inner_split"]) == 2


def test_split_k_fold_regression():
    x = pd.DataFrame(np.random.rand(100, 5))
    y = pd.DataFrame(np.random.rand(100, 1))
    splits = split_k_fold(x, y, outer_k=3, inner_k=2)
    assert isinstance(splits, dict)
    assert len(splits) == 3


def test_slice_features():
    x = pd.DataFrame(np.random.rand(50, 10))
    slices = slice_features(x, slice_sizes=[3, 5], n_iter=2)
    assert isinstance(slices, dict)
    assert set(slices.keys()) == {3, 5}
    for k, v in slices.items():
        for lst in v.values():
            assert len(lst) == k


def test_sample_features():
    cols = ["A", "B", "C___1", "C___2", "C___3"]
    x = pd.DataFrame(np.random.rand(30, len(cols)), columns=cols)
    result = sample_features("C", x, percents=[0.33, 0.66], n_iter=2)
    assert isinstance(result, dict)
    for perc_idx in result:
        for iter_idx, indices in result[perc_idx].items():
            assert all(isinstance(i, int) for i in indices)
