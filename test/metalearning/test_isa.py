import numpy as np
import pandas as pd
from ms.metalearning.isa import InstanceSpaceAnalysis, PILOTResult


def test_pilot_shapes():
    np.random.seed(42)
    features = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
    metrics = pd.DataFrame(np.random.rand(50, 3), columns=[f"m{i}" for i in range(3)])

    model = InstanceSpaceAnalysis(n_tries=3)
    result = model.pilot(features, metrics)

    assert isinstance(result, PILOTResult)
    assert result.x_bar.shape[1] == features.shape[1] + metrics.shape[1]
    assert result.z.shape[1] == 2
    assert result.summary.shape[0] == features.shape[1]
    assert result.a.shape == (2, features.shape[1])


def test_error_func_decreasing():
    np.random.seed(0)
    x = np.random.rand(10, 4)
    y = np.random.rand(10, 2)
    x_bar = np.hstack((x, y))
    n, m = x.shape[1], x_bar.shape[1]

    isa = InstanceSpaceAnalysis()
    alpha = np.random.rand(2 * m + 2 * n)
    err = isa.error_func(alpha, x_bar, n, m)

    assert np.isfinite(err)
    assert err >= 0
