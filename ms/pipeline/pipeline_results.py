import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

from ms.config.experiment_config import ExperimentConfig
from ms.metadataset.model_type import ModelType
from ms.utils.navigation import pjoin


class PlotConfig:
    def __init__(self, task_type: str):
        if task_type == "classification":
            self.metrics = ["f1", "b_acc"]
            self.metric_labels = {"f1": "F1-score", "b_acc": "Balanced Accuracy"}
        elif task_type == "regression":
            self.metrics = ["rmse", "mae"]
            self.metric_labels = {"rmse": "RMSE", "mae": "MAE"}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


class ResultCollector:
    def __init__(self):
        self.m_rename = {
            "rtdl_FTTransformer": "FTT",
            "rtdl_MLP": "MLP",
            "rtdl_ResNet": "ResNet",
        }
        self.selector_full_names = {
            "base": "baseline",
            "corr": "correlation",
            "cf": "counterfactual",
            "f_val": "f_value",
            "lasso": "lasso",
            "mi": "mutual_info",
            "rfe_mlp": "rfe_mlp",
            "rfe_xgb": "rfe_xgb",
            "te": "treatment_effect",
            "xgb": "xgb",
        }
        self.models = {
            "DecisionTree": ModelType.baseline,
            "KNN": ModelType.baseline,
            "LinearModel": ModelType.baseline,
            "RandomForest": ModelType.baseline,
            "CatBoost": ModelType.gbdt,
            "LightGBM": ModelType.gbdt,
            "XGBoost": ModelType.gbdt,
            "DANet": ModelType.nn,
            "rtdl_FTTransformer": ModelType.nn,
            "rtdl_MLP": ModelType.nn,
            "rtdl_ResNet": ModelType.nn,
            "STG": ModelType.nn,
            "TabNet": ModelType.nn,
            "VIME": ModelType.nn,
        }

    def collect(self, root_folder: str, pred_folder: str) -> pd.DataFrame:
        records = []
        for target_model in os.listdir(root_folder):
            target_path = os.path.join(root_folder, target_model)
            if not os.path.isdir(target_path):
                continue
            model_type = self.models.get(target_model)
            display_target_model = self.m_rename.get(target_model, target_model)

            for selector in os.listdir(target_path):
                if selector == "rfe_mlp":
                    continue
                selector_path = os.path.join(target_path, selector)
                pred_path = os.path.join(selector_path, pred_folder)
                if not os.path.exists(pred_path):
                    continue
                display_selector = self.selector_full_names.get(selector, selector)

                for file in os.listdir(pred_path):
                    if not file.endswith(".csv"):
                        continue
                    metamodel = file.replace(".csv", "")
                    df = pd.read_csv(os.path.join(pred_path, file), index_col=0)
                    for metric, row in df.iterrows():
                        fold_data = {"train": [], "test": []}
                        for col_name, value in row.items():
                            try:
                                fold_type, _ = str(col_name).split("_")
                                if fold_type in fold_data:
                                    fold_data[fold_type].append(float(value))
                            except ValueError:
                                continue
                        for fold_type in ["train", "test"]:
                            values = fold_data[fold_type]
                            if values:
                                values_series = pd.Series(values)
                                records.append(
                                    {
                                        "target_model": display_target_model,
                                        "model_type": (
                                            model_type.name if model_type else None
                                        ),
                                        "selector": display_selector,
                                        "metamodel": metamodel,
                                        "metric": metric,
                                        "fold_type": fold_type,
                                        "mean": values_series.mean(),
                                        "std": values_series.std(),
                                    }
                                )
        return pd.DataFrame.from_records(records)


class PlotGenerator:
    def __init__(self):
        self.colors = {"FS": "tab:blue", "FS+ISA": "tab:orange"}
        plt.rc("font", size=10)

    def boxplot_comparison(self, df: pd.DataFrame, metric: str, filename: str):
        df_filtered = df[(df["fold_type"] == "test") & (df["metric"] == metric)]
        metamodels = df_filtered["metamodel"].unique()
        methods = ["FS", "FS+ISA"]
        fig, axes = plt.subplots(
            len(metamodels), 1, figsize=(12, 5 * len(metamodels)), sharey=False
        )
        if len(metamodels) == 1:
            axes = [axes]
        for idx, metamodel in enumerate(metamodels):
            ax = axes[idx]
            metamodel_data = df_filtered[df_filtered["metamodel"] == metamodel]
            selectors = sorted(metamodel_data["selector"].unique())
            positions, data, box_colors = [], [], []
            for i, selector in enumerate(selectors):
                for j, method in enumerate(methods):
                    subset = metamodel_data[
                        (metamodel_data["selector"] == selector)
                        & (metamodel_data["method"] == method)
                    ]["mean"].dropna()
                    if not subset.empty:
                        data.append(subset)
                        positions.append(i * 3 + j)
                        box_colors.append(self.colors[method])
            if data:
                bp = ax.boxplot(
                    data,
                    positions=positions,
                    patch_artist=True,
                    widths=0.6,
                    showfliers=False,
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                for median in bp["medians"]:
                    median.set(color="black", linewidth=2)
                ax.set_xticks([i * 3 + 0.5 for i in range(len(selectors))])
                ax.set_xticklabels(selectors, rotation=45, ha="right")
                ax.set_ylabel(metric.upper())
                ax.set_title(f"FS pipeline comparison for {metamodel}")
                ax.grid(True)
        legend_handles = [
            Patch(facecolor=color, label=method)
            for method, color in self.colors.items()
        ]
        axes[0].legend(handles=legend_handles, title="Method", loc="upper right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def grouped_barplot(self, df: pd.DataFrame, task_type: str, save_dir: str = None):
        config = PlotConfig(task_type)
        selectors = sorted(df["selector"].unique())
        model_types = sorted(df["model_type"].dropna().unique())
        width = 0.8 / len(selectors)
        for metric in config.metrics:
            df_metric = df[(df["metric"] == metric) & (df["fold_type"] == "test")]
            for model_type in model_types:
                type_df = df_metric[df_metric["model_type"] == model_type]
                target_models = sorted(type_df["target_model"].unique())
                x_indices = np.arange(len(target_models))
                for metamodel in type_df["metamodel"].unique():
                    sub_df = type_df[type_df["metamodel"] == metamodel]
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for i, selector in enumerate(selectors):
                        selector_df = sub_df[sub_df["selector"] == selector]
                        means, stds = [], []
                        for tm in target_models:
                            target_data = selector_df[selector_df["target_model"] == tm]
                            if not target_data.empty:
                                means.append(target_data["mean"].values[0])
                                stds.append(target_data["std"].values[0])
                            else:
                                means.append(np.nan)
                                stds.append(np.nan)
                        offset = (i - (len(selectors) - 1) / 2) * width
                        bar_positions = x_indices + offset
                        ax.bar(
                            bar_positions, means, width=width, label=selector, alpha=0.9
                        )
                        ax.errorbar(
                            bar_positions,
                            means,
                            yerr=stds,
                            fmt="none",
                            ecolor="black",
                            elinewidth=1,
                            capsize=4,
                        )
                    ax.set_xticks(x_indices)
                    ax.set_xticklabels(target_models, rotation=45)
                    ax.set_xlabel("Target Model")
                    ax.set_ylabel(f"{config.metric_labels[metric]} (mean ± std)")
                    ax.set_title(
                        f"{model_type} | {metamodel} | {config.metric_labels[metric]}"
                    )
                    ax.legend(title="Selector")
                    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(f"{save_dir}/{metric}_{model_type}_{metamodel}.png")
                    else:
                        plt.show()
                    plt.close()


class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compare_against_baseline(
        self, metamodel_name: str, metric: str, baseline_selector: str = "baseline"
    ) -> pd.DataFrame:
        df_metric = self.df[
            (self.df["metric"] == metric) & (self.df["fold_type"] == "test")
        ]
        df_sub = df_metric[
            (df_metric["metamodel"] == metamodel_name)
            & (df_metric["selector"] != "rfe_mlp")
        ]
        pivot = df_sub.pivot_table(
            index="target_model", columns="selector", values="mean"
        ).dropna()
        results = []
        for selector in pivot.columns:
            if selector == baseline_selector:
                continue
            stat, p_value = mannwhitneyu(
                pivot[baseline_selector], pivot[selector], alternative="greater"
            )
            results.append({"vs_selector": selector, "stat": stat, "p_value": p_value})
        results_df = pd.DataFrame(results)
        n_tests = len(results_df)
        results_df["p_value_corrected"] = results_df["p_value"] * n_tests
        results_df["p_value_corrected"] = results_df["p_value_corrected"].clip(
            upper=1.0
        )
        results_df["baseline_worse"] = pivot[baseline_selector].mean() > results_df[
            "vs_selector"
        ].apply(lambda sel: pivot[sel].mean())
        results_df["significant"] = (results_df["p_value_corrected"] < 0.05) & (
            results_df["baseline_worse"]
        )
        return results_df


if __name__ == "__main__":
    task_type = "regression"  # or "classification"
    collector = ResultCollector()
    root_path = pjoin(ExperimentConfig.CONF.results_path, "tabzilla", "raw")
    df_main = collector.collect(root_path, "pred")
    df_isa = collector.collect(root_path, "isa_pred")

    df_main["method"] = "FS"
    df_isa["method"] = "FS+ISA"
    df_all = pd.concat([df_main, df_isa], ignore_index=True)

    plotter = PlotGenerator()
    plotter.boxplot_comparison(df_all, metric="mae", filename="fs_pipeline_mae.pdf")
    plotter.grouped_barplot(df_main, task_type=task_type, save_dir="plots")

    analyzer = StatisticalAnalyzer(df_main)
    result_table = analyzer.compare_against_baseline(
        metamodel_name="mlp", metric="rmse"
    )
    print(result_table)
