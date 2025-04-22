import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, mutual_info_regression, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ms.selection.selector import Selector


class CorrelationSelector(Selector):
    def __init__(
            self,
            corr_type: str = "spearman",
            p_threshold: float = 0.05,
            corr_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.corr_type = corr_type
        self.p_threshold = p_threshold
        self.corr_threshold = corr_threshold

    @property
    def name(self) -> str:
        return "corr"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        x = x_train.to_numpy()
        y = y_train.to_numpy()
        if self.corr_type == "pearson":
            result = pearsonr(x=x, y=y)
            stats, p_values = result.statistic[:-1, -1], result.pvalue[:-1, -1]
        else:
            result = spearmanr(a=x, b=y)
            stats, p_values = result.statistic[:-1, -1], result.pvalue[:-1, -1]

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = stats
        res["p"] = p_values

        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        for i, p_value in enumerate(res["p"]):
            if p_value > self.p_threshold or abs(res.iloc[i, 0]) < self.corr_threshold:
                res.iloc[i, 0] = None
        return res


class CorrelationInnerSelector(Selector):
    @property
    def name(self) -> str:
        return "corr_inner"

    def __init__(
            self,
            corr_type: str = "spearman",
            corr_threshold: float = 0.9,
            vif_count_threshold: int | None = None,
            vif_value_threshold: float | None = None,
    ):
        super().__init__()
        self.corr_type = corr_type
        self.corr_threshold = corr_threshold
        self.vif_count_threshold = vif_count_threshold
        self.vif_value_threshold = vif_value_threshold

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        corr = x_train.corr(method=self.corr_type)
        collinear_pairs = set()

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                if i != j and (abs(corr.iloc[i, j])) >= self.corr_threshold:
                    collinear_pairs.add(tuple(sorted([corr.index[i], corr.columns[j]])))

        corr.drop(set([i[0] for i in collinear_pairs]), inplace=True, axis="index")
        corr.drop(set([i[0] for i in collinear_pairs]), inplace=True, axis="columns")

        res = pd.DataFrame(index=x_train.columns)
        res["value"] = [0.0] * len(x_train.columns)

        if self.vif_count_threshold is None and self.vif_value_threshold is None:
            res = res.loc[corr.index]
            return res

        sorted_vif = self.compute_vif(x_train.loc[:, corr.columns])
        max_iter = self.vif_count_threshold \
            if self.vif_count_threshold is not None \
            else len(sorted_vif.index)

        for i in range(max_iter):
            vif_max = sorted_vif.max()["VIF"]
            if self.vif_value_threshold is not None and vif_max < self.vif_value_threshold:
                break
            sorted_vif = self.compute_vif(x_train.loc[:, sorted_vif.index[1:]])

        res = res.loc[sorted_vif.index]
        res["value"] = sorted_vif["VIF"]
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        return res

    @staticmethod
    def compute_vif(dataset: pd.DataFrame) -> pd.DataFrame:
        vif_data = pd.DataFrame(index=dataset.columns)
        vif_data["VIF"] = [variance_inflation_factor(dataset.values, i)
                           for i in range(len(dataset.columns))]
        return vif_data.sort_values(by="VIF", ascending=False)


class Chi2Selector(Selector):
    def __init__(
            self,
            p_threshold: float = 0.05
    ) -> None:
        super().__init__()
        self.p_threshold = p_threshold

    @property
    def name(self) -> str:
        return "chi2"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        chi2_stats, p_values = chi2(X=x_train, y=y_train)
        res = pd.DataFrame(index=x_train.columns)
        res["value"] = chi2_stats
        res["p"] = p_values
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["p"] > self.p_threshold, "value"] = None
        return res


class MutualInfoSelector(Selector):
    def __init__(
            self,
            quantile_value: float = 0.5
    ) -> None:
        super().__init__()
        self.quantile_value = quantile_value

    @property
    def name(self) -> str:
        return "mi"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        mi = mutual_info_classif(X=x_train, y=y_train) \
            if task == "class" \
            else mutual_info_regression(X=x_train, y=y_train)
        res = pd.DataFrame(index=x_train.columns)
        res["value"] = mi
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        quantile_mi = res["value"].quantile(self.quantile_value)
        res.loc[res["value"] < quantile_mi, "value"] = None
        return res


class FValueSelector(Selector):
    def __init__(
            self,
            p_threshold: float = 0.05,
            quantile_value: float = 0.5
    ) -> None:
        super().__init__()
        self.p_threshold = p_threshold
        self.quantile_value = quantile_value

    @property
    def name(self) -> str:
        return "f_val"

    def compute_generic(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame | None = None,
            y_test: pd.DataFrame | None = None,
            task: str = "class",
    ) -> pd.DataFrame:
        if task == "class":
            f_statistic, p_values = f_classif(X=x_train, y=y_train)
        else:
            f_statistic, p_values = f_regression(X=x_train, y=y_train)
        res = pd.DataFrame(index=x_train.columns)
        res["value"] = f_statistic
        res["p"] = p_values
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        quantile_f = res["value"].quantile(self.quantile_value)
        res.loc[
            (res["p"] > self.p_threshold)
            | (res["value"].abs() < quantile_f), "value"
        ] = None
        return res
