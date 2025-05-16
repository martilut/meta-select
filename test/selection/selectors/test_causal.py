from test.selection.selectors.test_base import BaseSelectorTest

import numpy as np
import pytest

from ms.selection.selectors.causal import TESelector


class TestTESelector(BaseSelectorTest):
    def test_process_individual_mode(self):
        X, y = self._generate_classification_data(
            n_samples=30, n_features=10, n_informative=5
        )
        selector = TESelector(
            to_tune=False, mode="individual", n_estimators=8, random_state=42
        )

        result = selector.process(X, y)

        self._assert_valid_output(X, result)
        assert result.shape[1] <= X.shape[1]

    def test_process_invalid_mode_raises(self):
        X, y = self._generate_classification_data(
            n_samples=30, n_features=10, n_informative=5
        )
        selector = TESelector(
            to_tune=False, mode="invalid_mode", n_estimators=8, random_state=42
        )

        with pytest.raises(ValueError):
            selector.process(X, y)
