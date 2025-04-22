import pandas as pd

from ms.selection.selector import Selector


class BaseSelector(Selector):
    @property
    def name(self) -> str:
        return "base"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        res = pd.DataFrame(index=x_train.columns)
        res["value"] = [1.0] * len(x_train.columns)
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        return res
