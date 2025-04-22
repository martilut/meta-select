import os
import pandas as pd
import numpy as np

from ms.config.pipeline_constants import TZ_RESULTS_ROOT
from ms.utils.navigation import pjoin

# Path to the root folder
root = pjoin(TZ_RESULTS_ROOT, "abs")

# Selectors to consider
selectors = ['base', 'corr', 'cf', 'f_val', 'lasso', 'mi', 'rfe_mlp', 'rfe_xgb', 'te', 'xgb']

# Dictionary to store times for each target and selector
target_selector_times = {target: {selector: [] for selector in selectors} for target in os.listdir(root)}

# Gather time data
targets = sorted(os.listdir(root))

print("Starting time data aggregation...")

for target in targets:
    print(f"Processing target: {target}")
    for selector in selectors:
        time_file_path = os.path.join(root, target, selector, 'time.csv')

        if os.path.exists(time_file_path):
            print(f"  Found time file: {time_file_path}")
            df = pd.read_csv(time_file_path, index_col=0)

            if df.shape[0] == 1 and df.shape[1] == 1:
                time_value = df.iloc[0, 0]
                target_selector_times[target][selector].append(time_value)
                # Log the time for each selector
                print(f"    Time for selector '{selector}' on target '{target}': {time_value} seconds")
        else:
            print(f"  Time file not found: {time_file_path}")

print("Time data aggregation completed.")

# Prepare data for DataFrame
data = {target: [np.mean(target_selector_times[target].get(selector, [np.nan])) for selector in selectors]
        for target in targets}

# Create a DataFrame from the collected data
df_time = pd.DataFrame(data, index=selectors).T

# Optionally, save the DataFrame to a CSV file if needed
df_time.to_csv('selector_times.csv')

print("DataFrame with time data created successfully.")
print(df_time)
