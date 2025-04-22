import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ms.processing.preprocess import Preprocessor


# Sample test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8]
    })

def test_fit_and_transform_mean_strategy(sample_data):
    preprocessor = Preprocessor(strategy='mean')
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    # Mean of A: (1+2+4)/3 = 2.33, Mean of B: (5+7+8)/3 = 6.67
    expected = pd.DataFrame({
        'A': [1.0, 2.0, (1+2+4)/3, 4.0],
        'B': [5.0, (5+7+8)/3, 7.0, 8.0]
    })
    pd.testing.assert_frame_equal(transformed, expected, atol=1e-2)

def test_fit_and_transform_with_scaler(sample_data):
    preprocessor = Preprocessor(strategy='mean', scaler=StandardScaler())
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    # Should have mean 0 and std 1
    np.testing.assert_almost_equal(transformed.mean().values, [0, 0], decimal=6)
    np.testing.assert_almost_equal(transformed.std(ddof=0).values, [1, 1], decimal=6)

def test_transform_before_fit_raises(sample_data):
    preprocessor = Preprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_data)

def test_input_not_dataframe():
    preprocessor = Preprocessor()
    with pytest.raises(ValueError):
        preprocessor.fit(np.array([[1, 2], [3, 4]]))

def test_custom_imputation_strategy(sample_data):
    preprocessor = Preprocessor(strategy='median')
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    expected = sample_data.copy()
    expected['A'].fillna(expected['A'].median(), inplace=True)
    expected['B'].fillna(expected['B'].median(), inplace=True)

    pd.testing.assert_frame_equal(transformed, expected, atol=1e-2)

def test_output_shape_and_index(sample_data):
    preprocessor = Preprocessor(strategy='mean')
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    assert transformed.shape == sample_data.shape
    assert all(transformed.index == sample_data.index)
    assert all(transformed.columns == sample_data.columns)
