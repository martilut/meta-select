from abc import ABC

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
            cv: bool = False,
    ) -> None:
        super().__init__(model=model)
        super().__init__(cv=cv)
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
            task: str = "class",
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


class RK_RFE(Selector, ModelWrapper):
    @property
    def name(self) -> str:
        return "rk_rfe"

    def __init__(
            self,
            model: MetaModel,
            cv: bool = False, 
            ntree: int = 200
    ):
        super().__init__(cv=cv)
        super().__init__(model=model)
        self.ntree = ntree

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        datam = pd.concat(
            [y_train.reset_index(drop=True), x_train.reset_index(drop=True)], 
            axis=1
        )
        y_matrix = datam.iloc[:, 0].values
        remaining_features = list(datam.columns[1:])
        mse = float('inf')
        selected_predictors = None

        while len(remaining_features) >= 3:
            rf = RandomForestRegressor(n_estimators=self.ntree)
            rf.fit(datam[remaining_features], y_matrix)

            predictions = rf.predict(datam[remaining_features])
            test_error = (y_matrix - predictions) ** 2
            mean_squared_error_val = np.mean(test_error)

            importances = rf.feature_importances_
            importance_df = pd.DataFrame({'feature': remaining_features, 'importance': importances})
            importance_df = importance_df.sort_values(by='importance')

            if mean_squared_error_val <= mse:
                mse = mean_squared_error_val
                selected_predictors = importance_df['feature'].tolist()

            elim_feature = importance_df.iloc[0]['feature']
            remaining_features = [f for f in remaining_features if f != elim_feature]

        result_df = pd.DataFrame(index=selected_predictors)
        result_df["value"] = np.arange(len(selected_predictors), 0, -1)

        return result_df

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        return res