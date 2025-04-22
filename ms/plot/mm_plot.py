import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from ms.config.pipeline_constants import CONF
from ms.utils.navigation import pjoin

# Path to the root folder
root = pjoin(CONF.results_path, "tabzilla", "abs")

# Metamodels and selectors (colors will map to these)
metamodels = ['knn', 'xgb', 'mlp']
selectors = ['base', 'corr', 'cf', 'f_val', 'lasso', 'mi', 'rfe_mlp', 'rfe_xgb', 'te', 'xgb']

selector_colors = {
    'base': '#1f77b4',  # Blue
    'corr': '#ff7f0e',  # Orange
    'cf': '#2ca02c',  # Green
    'f_val': '#d62728',  # Red
    'lasso': '#9467bd',  # Purple
    'mi': '#8c564b',  # Brown
    'rfe_mlp': '#e377c2',  # Pink
    'te': '#7f7f7f',  # Gray
    'xgb': '#bcbd22',  # Yellow-green
    'rfe_xgb': '#17becf'  # Cyan
}

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


# Gather data
targets = sorted(os.listdir(root))
targets = [t for t in targets if len(t.split(".")) == 1]
results = {meta: {} for meta in metamodels}

# Start logging
print("Starting data aggregation...")

for target in targets:
    print(f"Processing target: {target}")
    for selector in selectors:
        for meta in metamodels:
            file_path = os.path.join(root, target, selector, 'pred', f'{meta}.csv')

            if os.path.exists(file_path):
                print(f"  Found file: {file_path}")
                df = pd.read_csv(file_path, index_col=0)

                if 'f1' in df.index:
                    test_cols = [col for col in df.columns if col.startswith('test')]
                    f1_scores = df.loc['f1', test_cols].astype(float)
                    mean_f1 = f1_scores.mean()
                    std_f1 = f1_scores.std()

                    # Store results
                    results[meta].setdefault(target, {})[selector] = (mean_f1, std_f1)

                    # Log aggregation
                    print(f"    Aggregated for {selector} (meta: {meta}):")
                    print(f"      Mean F1: {mean_f1:.4f}, Std F1: {std_f1:.4f}")
            else:
                print(f"  File not found: {file_path}")

# Log the end of aggregation
print("Data aggregation completed.")

# Plot per metamodel
width = 0.08
x = np.arange(len(targets))

for meta in metamodels:
    print(f"Generating plot for {meta.upper()}...")
    fig, ax = plt.subplots(figsize=(12, 6))
    offset = -width * (len(selectors) / 2)

    top3_selectors_per_target = {}
    for target in targets:
        selector_scores = []
        for selector in selectors:
            mean = results[meta].get(target, {}).get(selector, (np.nan,))[0]
            if not np.isnan(mean):
                selector_scores.append((selector, mean))
        top3 = sorted(selector_scores, key=lambda x: x[1], reverse=True)[:3]
        top3_selectors_per_target[target] = {s for s, _ in top3}
        print(f"  Top 3 selectors for target '{target}': {top3}")

    for i, selector in enumerate(selectors):
        means = []
        stds = []
        bar_positions = []
        for t_idx, target in enumerate(targets):
            mean_std = results[meta].get(target, {}).get(selector, (np.nan, np.nan))
            means.append(mean_std[0])
            stds.append(mean_std[1])
            bar_positions.append(x[t_idx] + offset + i * width)

        bars = ax.bar(
            bar_positions,
            means,
            width,
            yerr=stds,
            capsize=2,
            ecolor='black',
            error_kw={'elinewidth': 0.8, 'alpha': 0.4},
            label=selector,
            color=selector_colors.get(selector, '#333333'),
            edgecolor='none'  # No edge here
        )

        # Manually draw black rectangles around top 3
        for bar, t_idx in zip(bars, range(len(targets))):
            target = targets[t_idx]
            if selector in top3_selectors_per_target[target]:
                # Accurate rectangle around the bar
                rect = patches.Rectangle(
                    (bar.get_x(), bar.get_y()),
                    bar.get_width(),
                    bar.get_height(),
                    linewidth=1.8,
                    edgecolor='black',
                    facecolor='none',
                    zorder=5  # on top of bars and error bars
                )
                ax.add_patch(rect)

    ax.set_title(f'Metamodel: {meta.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=30)
    ax.set_xlabel('Target')
    ax.set_ylabel('Average F1 Score')

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=selector_colors[selector],
            markersize=10,
            label=selector_full_names.get(selector, selector)
        )
        for selector in selectors
    ]

    ax.legend(handles=legend_handles, title="Selector", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    plt.show()

print("Plot generation completed.")
