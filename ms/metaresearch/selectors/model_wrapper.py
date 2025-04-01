from abc import ABC

import pandas as pd
from sklearn.feature_selection import RFE

from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selector import Selector


class ModelWrapper(ABC):
    def __init__(
            self,
            model: MetaModel,
    ):
        self.model = model

class RFESelector(Selector, ModelWrapper):
    def __init__(
            self,
            model: MetaModel,
            rank_threshold: float = 1.0,
    ) -> None:
        super().__init__(model=model)
        self.rank_threshold = rank_threshold

    @property
    def name(self) -> str:
        return "rfe"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rfe = RFE(estimator=self.model)
        rfe.fit(X=x_train, y=y_train)

        res = pd.DataFrame(index=x_train.columns)
        res["rfe_fi"] = rfe.ranking_

        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        for i, rank in enumerate(res["rfe_fi"]):
            if rank > self.rank_threshold:
                res.iloc[i, 0] = None
        return res
