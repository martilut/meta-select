from abc import ABC, abstractmethod
import pandas as pd
from sklearn.utils.multiclass import type_of_target


class Selector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def compute_classification(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.compute_generic(x_train, y_train, x_test, y_test)

    def compute_regression(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.compute_generic(x_train, y_train, x_test, y_test)

    @abstractmethod
    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...

    def compute(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.is_classif(y=y_train):
            return self.compute_classification(x_train, y_train, x_test, y_test)
        else:
            return self.compute_regression(x_train, y_train, x_test, y_test)

    def select(
            self,
            res: pd.DataFrame,
            k_best: int | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        filtered_df = self.__select__(res=res)
        filtered_df.sort_values(
            by="value",
            ascending=False,
            inplace=True,
            key=abs
        )
        selected_df = filtered_df.dropna(axis="rows")
        selected_features = selected_df.index.tolist()
        if k_best is not None and k_best < len(selected_features):
            selected_features = selected_features[:k_best]

        return selected_df, selected_features

    @abstractmethod
    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        ...

    def compute_select(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            k_best: int | None = None,
    ) -> pd.DataFrame:
        if x_test is None:
            x_test = x_train
        if y_test is None:
            y_test = y_train

        res = self.compute(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        return self.select(res=res, k_best=k_best)

    @staticmethod
    def is_classif(
            y: pd.Series
    ) -> bool:
        target_type = type_of_target(y)
        return target_type in {"binary", "multiclass"}
