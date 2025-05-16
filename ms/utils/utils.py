import time

import pandas as pd
from sklearn.utils.multiclass import type_of_target

from ms.utils.navigation import rewrite_decorator


def measure_runtime(func, *args, **kwargs):
    """
    Measures and returns the runtime of a function in seconds.

    Parameters:
        func (callable): The function to measure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        result: The return value of the function.
        runtime: Time in seconds it took to execute the function.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    runtime = end - start
    return result, runtime


@rewrite_decorator
def save_runtime(
    runtime: float, row: str, column: str, *args, **kwargs
) -> pd.DataFrame:
    return pd.DataFrame([runtime], index=[row], columns=[column])


def is_classif(y: pd.Series) -> bool:
    target_type = type_of_target(y)
    return target_type in {"binary", "multiclass"}
