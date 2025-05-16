from typing import List
import pandas as pd
from collections import Counter

from ms.selection.selector import Selector


class EnsembleSelector(Selector):
    def __init__(self, selectors: List[Selector], cv: bool = False) -> None:
        super().__init__(cv=cv)
        self.selectors = selectors

    @property
    def name(self) -> str:
        return "ensemble"

    def compute_generic(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        task: str = "class",
    ) -> pd.DataFrame:
        feature_counts = Counter()

        for selector in self.selectors:
            result = selector.compute_select(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                task=task,
            )
            feature_counts.update(result.index.tolist())

        if not feature_counts:
            return pd.DataFrame(columns=["value"])

        majority_threshold = len(self.selectors) // 2 + 1

        selected_features = {
            feature: count
            for feature, count in feature_counts.items()
            if count >= majority_threshold
        }

        return pd.DataFrame.from_dict(
            selected_features, orient="index", columns=["value"]
        )

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        return res
