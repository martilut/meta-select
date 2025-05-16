from test.selection.selectors.test_base import BaseSelectorTest

from ms.selection.selectors.model_free import (
    CorrelationInnerSelector,
    CorrelationSelector,
    FValueSelector,
    MutualInfoSelector,
)


class TestModelFree(BaseSelectorTest):
    def test_process_correlation_selector(self):
        X, y = self._generate_classification_data()
        selector = CorrelationSelector()
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_correlation_inner_selector(self):
        X, y = self._generate_regression_data()
        selector = CorrelationInnerSelector(vif_value_threshold=10)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_mutual_info_selector_classification(self):
        X, y = self._generate_classification_data()
        selector = MutualInfoSelector(quantile_value=0.5)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_mutual_info_selector_regression(self):
        X, y = self._generate_regression_data()
        selector = MutualInfoSelector()
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_f_value_selector_classification(self):
        X, y = self._generate_classification_data()
        selector = FValueSelector()
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_f_value_selector_regression(self):
        X, y = self._generate_regression_data()
        selector = FValueSelector()
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)
