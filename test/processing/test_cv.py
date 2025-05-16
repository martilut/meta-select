import numpy as np
import pandas as pd
import pytest

from ms.processing.cv import cv_decorator


@pytest.fixture
def dummy_data():
    x = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    y = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)), columns=["target"])
    split = {
        0: {"train": list(range(0, 6)), "test": list(range(6, 10))},
        1: {"train": list(range(4, 10)), "test": list(range(0, 4))},
    }
    return x, y, split


@cv_decorator
def dummy_eval(x_train, y_train, x_test=None, y_test=None, **kwargs):
    """
    Dummy evaluation function that returns accuracy-like values.
    """
    if x_test is None:
        x_test = x_train
    if y_test is None:
        y_test = y_train
    result = pd.DataFrame(
        [{"score": (y_test.values.ravel() == y_test.values.ravel()).mean()}]
    )
    return result


def test_cv_decorator_aggregated(dummy_data):
    x, y, split = dummy_data
    result = dummy_eval(x=x, y=y, split=split)
    assert isinstance(result, pd.DataFrame)
    assert "value" in result.columns
    assert result.shape == (1, 1)


def test_cv_decorator_nonaggregated(dummy_data):
    x, y, split = dummy_data
    result = dummy_eval(x=x, y=y, split=split, to_agg=False)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ["score_0", "score_1"])


def test_cv_decorator_no_split(dummy_data):
    x, y, _ = dummy_data
    result = dummy_eval(x=x, y=y, split=None)
    assert isinstance(result, pd.DataFrame)
    assert "score" in result.columns
