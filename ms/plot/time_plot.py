import os
import pandas as pd
import matplotlib.pyplot as plt

from ms.config.pipeline_constants import CONF
from ms.utils.navigation import pjoin

# Adjust as needed
root_folder = pjoin(CONF.results_path, "tabzilla", "abs")
init_sizes = ['20', '40', '60', '80', '100', '123']

# Full names for selectors
selector_full_names = {
    'base': 'baseline',
    'corr': 'correlation',
    'cf': 'counterfactual',
    'f_val': 'f_value',
    'lasso': 'lasso',
    'mi': 'mutual_info',
    'rfe_mlp': 'rfe_mlp',
    'rfe_xgb': 'rfe_xgb',
    'te': 'treatment_effect',
    'xgb': 'xgb'
}

# Discover targets
target_models = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
num_targets = len(target_models)

# Prepare plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# For shared legend
shared_handles = {}
init_sizes_int = [int(s) for s in init_sizes]

for i, target in enumerate(target_models):
    target_path = os.path.join(root_folder, target)
    selectors = sorted([d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))])

    for selector in selectors:
        if selector == "base":
            continue
        times = []
        selector_path = os.path.join(target_path, selector)
        full_name = selector_full_names.get(selector, selector)  # fallback to key if not found

        for size in init_sizes:
            if size == '123':
                csv_path = os.path.join(selector_path, 'time.csv')
            else:
                csv_path = os.path.join(selector_path, size, 'time.csv')

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0)
                try:
                    time_val = df.loc[target, selector] / 5
                except KeyError:
                    time_val = float('nan')
            else:
                time_val = float('nan')

            times.append(time_val)

        line, = axes[i].plot(init_sizes_int, times, marker='o', label=full_name)

        # Only add one line per selector to legend
        if full_name not in shared_handles:
            shared_handles[full_name] = line

    axes[i].set_title(f"{target}")
    axes[i].set_xlabel("Initial number of features")
    axes[i].set_ylabel("Average selection time (s)")
    axes[i].set_yscale('log')
    axes[i].grid(True, which='both', ls='--', linewidth=0.5)

# Add shared legend
fig.legend(
    handles=shared_handles.values(),
    labels=shared_handles.keys(),
    loc='lower center',
    ncol=min(len(shared_handles), 5),
    fontsize='medium',
    frameon=False
)

plt.tight_layout(rect=[0, 0.07, 1, 1])  # Adjust bottom to make space for legend
plt.savefig("time.pdf")
