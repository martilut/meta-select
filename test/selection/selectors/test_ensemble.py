from ms.selection.selectors.ensemble import EnsembleSelector
from ms.selection.selector import Selector
import pandas as pd

from test.selection.selectors.test_base import BaseSelectorTest


class DummySelector(Selector):
    def __init__(self, selected_features):
        super().__init__(cv=False)
        self._selected_features = selected_features

    @property
    def name(self):
        return "dummy"

    def compute_generic(self, x_train, y_train, x_test=None, y_test=None, task="class"):
        res = pd.DataFrame(index=self._selected_features)
        res["value"] = 1.0
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        return res


class TestEnsembleSelector(BaseSelectorTest):
    def test_majority_vote_selection(self):
        X, y = self._generate_classification_data()

        sel1 = DummySelector(selected_features=["f0", "f1", "f2"])
        sel2 = DummySelector(selected_features=["f1", "f2", "f3"])
        sel3 = DummySelector(selected_features=["f2", "f3", "f4"])

        ensemble = EnsembleSelector(selectors=[sel1, sel2, sel3])
        X_selected = ensemble.process(X, y)

        expected_features = {"f1", "f2", "f3"}
        assert set(X_selected.columns) == expected_features

        self._assert_valid_output(X, X_selected)

    def test_no_majority_features(self):
        X, y = self._generate_classification_data()

        sel1 = DummySelector(selected_features=["f0"])
        sel2 = DummySelector(selected_features=["f1"])
        sel3 = DummySelector(selected_features=["f2"])

        ensemble = EnsembleSelector(selectors=[sel1, sel2, sel3])
        X_selected = ensemble.process(X, y)

        assert X_selected.empty

    def test_empty_selectors_list(self):
        X, y = self._generate_classification_data()

        ensemble = EnsembleSelector(selectors=[])
        X_selected = ensemble.process(X, y)

        assert X_selected.empty
