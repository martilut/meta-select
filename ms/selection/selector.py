from abc import ABC, abstractmethod
import pandas as pd

from ms.processing.cv import cv_decorator
from ms.utils.utils import is_classif


class Selector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __init__(
            self,
            cv: bool = False,
    ) -> None:
        self.cv = cv

    def compute_classification(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.compute_generic(x_train, y_train, x_test, y_test, task="class")

    def compute_regression(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.compute_generic(x_train, y_train, x_test, y_test, task="reg")

    @abstractmethod
    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        ...

    def compute(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if is_classif(y=y_train):
            res = self.compute_classification(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
        else:
            res = self.compute_regression(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
        return res

    def select(
            self,
            res: pd.DataFrame,
            k_best: int | None = None,
    ) -> pd.DataFrame:
        filtered_df = self.__select__(res=res)
        filtered_df.sort_values(
            by="value",
            ascending=False,
            inplace=True,
            key=abs
        )
        selected_df = filtered_df.dropna(axis="rows", how="any").copy()
        selected_features = selected_df.index.tolist()
        if k_best is not None and k_best < len(selected_features):
            selected_features = selected_features[:k_best]
            selected_df = selected_df.loc[selected_features]
        return selected_df

    @abstractmethod
    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        ...

    @cv_decorator
    def compute_select(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            k_best: int | None = None,
            **kwargs,
    ) -> pd.DataFrame:
        res = self.compute(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        return self.select(res=res, k_best=k_best)
