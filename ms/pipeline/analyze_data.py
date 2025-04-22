import os

import numpy as np
import pandas as pd

from ms.config.pipeline_constants import CONF
from ms.utils.navigation import pjoin


def collect_results(root_folder):
    records = []

    for target_model in os.listdir(root_folder):
        target_path = os.path.join(root_folder, target_model)
        if not os.path.isdir(target_path):
            continue

        for selector in os.listdir(target_path):
            selector_path = os.path.join(target_path, selector)
            pred_path = os.path.join(selector_path, 'pred')
            if not os.path.exists(pred_path):
                continue

            for metamodel_file in os.listdir(pred_path):
                if not metamodel_file.endswith('.csv'):
                    continue

                metamodel = metamodel_file.replace('.csv', '')
                file_path = os.path.join(pred_path, metamodel_file)

                df = pd.read_csv(file_path, index_col=0)  # rows are metrics
                for metric, row in df.iterrows():
                    fold_data = {'train': [], 'test': []}
                    rmse_data = {'train': [], 'test': []}

                    for col_name, value in row.items():
                        try:
                            fold_type, fold_num = col_name.split('_')
                            value = float(value)
                            if fold_type in fold_data:
                                fold_data[fold_type].append(value)
                                if metric.lower() == 'mse':
                                    rmse_data[fold_type].append(np.sqrt(value))
                        except ValueError:
                            continue  # Skip malformed column names

                    for fold_type in ['train', 'test']:
                        values = fold_data[fold_type]
                        rmse_values = rmse_data[fold_type]
                        if values:
                            values_series = pd.Series(values)
                            record = {
                                'target_model': target_model,
                                'selector': selector,
                                'metamodel': metamodel,
                                'metric': metric,
                                'fold_type': fold_type,
                                'mean': values_series.mean(),
                                'std': values_series.std()
                            }
                            if metric.lower() == 'mse' and rmse_values:
                                rmse_series = pd.Series(rmse_values)
                                record['rmse_mean'] = rmse_series.mean()
                                record['rmse_std'] = rmse_series.std()
                            else:
                                record['rmse_mean'] = np.nan
                                record['rmse_std'] = np.nan

                            records.append(record)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    root_folder = pjoin(CONF.results_path, "tabzilla", "raw")  # Replace with your actual path
    results_df = collect_results(root_folder)
    print(results_df)
    results_df.to_csv('all_results.csv', index=False)
    # print("Results collected and saved to all_results.csv")