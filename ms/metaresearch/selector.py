from abc import ABC, abstractmethod
import pandas as pd
from sklearn.utils.multiclass import type_of_target


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
            x: pd.DataFrame,
            y: pd.DataFrame,
            split: dict | None = None,
    ) -> pd.DataFrame:
        if split is not None and self.cv:
            res_list = []
            for i in split:
                x_train = x.iloc[split[i]["train"], :]
                y_train = y.iloc[split[i]["train"], :]
                x_test = x.iloc[split[i]["test"], :]
                y_test = y.iloc[split[i]["test"], :]

                if self.is_classif(y=y):
                    res = self.compute_classification(
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test
                    )
                else:
                    res = self.compute_regression(
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test
                    )
                res_list.append(res)
            res = pd.concat(res_list).mean(axis=0)
        else:
            if self.is_classif(y=y):
                res = self.compute_classification(
                    x_train=x,
                    y_train=y
                )
            else:
                res = self.compute_regression(
                    x_train=x,
                    y_train=y
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
        selected_df = filtered_df.dropna(axis="rows", how="any")
        selected_features = selected_df.index.tolist()
        if k_best is not None and k_best < len(selected_features):
            selected_features = selected_features[:k_best]
        selected_df["selected"] = [False] * len(selected_df)
        selected_df.loc[selected_features, "selected"] = True
        return selected_df

    @abstractmethod
    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        ...

    def compute_select(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            split: dict | None = None,
            k_best: int | None = None,
    ) -> pd.DataFrame:
        res = self.compute(
            x=x,
            y=y,
            split=split,
        )

        return self.select(res=res, k_best=k_best)

    @staticmethod
    def is_classif(
            y: pd.Series
    ) -> bool:
        target_type = type_of_target(y)
        return target_type in {"binary", "multiclass"}
