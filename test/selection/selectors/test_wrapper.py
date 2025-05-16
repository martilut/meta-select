from test.selection.selectors.test_base import BaseSelectorTest

from sklearn.linear_model import LogisticRegression

from ms.selection.selectors.model_wrapper import RFESelector


class TestRFESelector(BaseSelectorTest):
    def test_process_classification(self):
        X, y = self._generate_classification_data()
        model = LogisticRegression(solver="liblinear", random_state=42)
        selector = RFESelector(model=model, rank_threshold=1)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)

    def test_process_regression(self):
        X, y = self._generate_regression_data()
        from sklearn.linear_model import Lasso

        model = Lasso(alpha=0.1, random_state=42)
        selector = RFESelector(model=model, rank_threshold=1)
        X_selected = selector.process(X, y)
        self._assert_valid_output(X, X_selected)
