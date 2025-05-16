import os

import numpy as np
import pandas as pd

from ms.config.experiment_config import ExperimentConfig
from ms.utils.navigation import pjoin


class ResultCollector:
    def __init__(self, root_folder: str, metrics: list[str] = None):
        """
        :param root_folder: Path to the root directory of results.
        :param metrics: List of metrics to process. E.g., ['mse', 'mae', 'r2', 'rmse'].
        """
        self.root_folder = root_folder
        self.metrics = [m.lower() for m in metrics] if metrics else None
        self.include_rmse = "rmse" in self.metrics if self.metrics else False
        self.records = []

    def _process_file(
        self, file_path: str, target_model: str, selector: str, metamodel: str
    ):
        df = pd.read_csv(file_path, index_col=0)

        for metric, row in df.iterrows():
            metric_lower = metric.lower()

            if self.metrics and metric_lower not in self.metrics:
                if not (self.include_rmse and metric_lower == "mse"):
                    continue

            fold_data = {"train": [], "test": []}
            rmse_data = (
                {"train": [], "test": []}
                if self.include_rmse and metric_lower == "mse"
                else None
            )

            for col_name, value in row.items():
                try:
                    fold_type, _ = col_name.split("_")
                    value = float(value)
                    if fold_type in fold_data:
                        fold_data[fold_type].append(value)
                        if rmse_data is not None:
                            rmse_data[fold_type].append(np.sqrt(value))
                except ValueError:
                    continue

            for fold_type in ["train", "test"]:
                values = fold_data[fold_type]
                if values:
                    record = {
                        "target_model": target_model,
                        "selector": selector,
                        "metamodel": metamodel,
                        "metric": metric,
                        "fold_type": fold_type,
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }

                    self.records.append(record)

    def collect(self) -> pd.DataFrame:
        for target_model in os.listdir(self.root_folder):
            target_path = os.path.join(self.root_folder, target_model)
            if not os.path.isdir(target_path):
                continue

            for selector in os.listdir(target_path):
                pred_path = os.path.join(target_path, selector, "pred")
                if not os.path.exists(pred_path):
                    continue

                for filename in os.listdir(pred_path):
                    if not filename.endswith(".csv"):
                        continue

                    metamodel = filename.replace(".csv", "")
                    file_path = os.path.join(pred_path, filename)
                    self._process_file(file_path, target_model, selector, metamodel)

        return pd.DataFrame.from_records(self.records)


if __name__ == "__main__":
    root_path = pjoin(ExperimentConfig.CONF.results_path, "tabzilla", "raw")

    collector = ResultCollector(root_folder=root_path, metrics=["mse", "r2", "rmse"])
    results_df = collector.collect()

    print(results_df)
    results_df.to_csv("all_results.csv", index=False)
