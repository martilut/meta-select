from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_score

from ms.metalearning.isa import PILOTResult
from ms.utils.navigation import rewrite_decorator


@rewrite_decorator
def save_isa_data(
    selected_names: list[str],
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    res: PILOTResult,
    selector_name: str,
    save_plot: Path | None = None,
    draw_convex_hull: bool = True,
    discrete_data: bool = True,
    *args,
    **kwargs,
) -> dict:
    data = {"selected": selected_names, "z": res.z.tolist()}
    isa_features = pd.DataFrame(res.z, index=features.index, columns=["z1", "z2"])
    true_labels = metrics.iloc[:, 0].values
    if not discrete_data:
        true_labels = np.digitize(true_labels, bins=np.linspace(0, 1, num=10))

    silhouette = silhouette_score(isa_features, true_labels)
    data["silhouette"] = silhouette

    cmap = ListedColormap(["red", "blue"])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        isa_features["z1"],
        isa_features["z2"],
        c=true_labels,
        cmap=cmap,
        alpha=0.7,
        edgecolors="k",
    )

    if draw_convex_hull:
        unique_labels = np.unique(true_labels)
        for label in unique_labels:
            points = isa_features[true_labels == label]
            if len(points) > 2:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points.iloc[simplex, 0], points.iloc[simplex, 1], "k-")

    plt.title(
        f"{metrics.columns[0]}, {selector_name} | Silhouette Score: {silhouette:.4f}"
    )
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.colorbar(scatter, ticks=[0, 1], label="True Cluster")
    plt.grid(True)

    if save_plot is not None:
        save_plot.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_plot)
    plt.close()
    return data
