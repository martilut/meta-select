from test.selection.selectors.test_base import BaseSelectorTest

from ms.selection.selectors.model_based import LassoSelector, XGBSelector


class TestXGBSelector(BaseSelectorTest):
    def test_process_classification(self):
        X, y = self._generate_classification_data()
        selector = XGBSelector(random_state=42, importance_threshold=0.01)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_regression(self):
        X, y = self._generate_regression_data()
        selector = XGBSelector(random_state=42, importance_threshold=0.01)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)


class TestLassoSelector(BaseSelectorTest):
    def test_process_classification(self):
        X, y = self._generate_classification_data()
        selector = LassoSelector(random_state=42, coef_threshold=0.01)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_regression(self):
        X, y = self._generate_regression_data()
        selector = LassoSelector(random_state=42, coef_threshold=0.01)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)
