from pathlib import Path

import pandas as pd

from ms.metaresearch.data_preprocess import scale, fill_na
from ms.metaresearch.selector import Selector
from ms.utils.navigation import pjoin, rewrite_decorator


@rewrite_decorator
def run_selector(
        selector: Selector,
        features: pd.DataFrame,
        metrics: pd.DataFrame,
        splits: dict,
        scaler: str,
        k_best: int | None = None,
        fill_func: str = "mean",
        save_path: Path | None = None,
        to_rewrite: bool = False,
) -> dict:
    result_dict = {i:[] for i in metrics.columns}
    errors_dict = {}

    for target_model in metrics.columns:
        target_res = run_target(
            selector=selector,
            features=features,
            metrics=metrics,
            splits=splits,
            scaler=scaler,
            target_model=target_model,
            k_best=k_best,
            fill_func=fill_func,
            save_path=pjoin(save_path.parent, f"{target_model}.csv"),
            to_rewrite=to_rewrite,
        )
        result_dict[target_model] = {i:[] for i in splits}
        for i in splits:
            result_dict[target_model][i] = [
                feat for feat in target_res.index if target_res.loc[feat, f"selected_{i}"] is True
            ]
            k_best_safe = k_best if k_best is not None else 1
            print(k_best_safe)
            if len(result_dict[target_model][i]) < k_best_safe:
                errors_dict[f"{target_model}_{i}"] = len(result_dict[target_model][i])

    if len(errors_dict) > 0:
        save_errors(
            errors_dict=errors_dict,
            save_path=pjoin(
                save_path.parent,
                "errors.json"
            ),
            to_rewrite=to_rewrite,
        )

    return result_dict


@rewrite_decorator
def run_target(
    selector: Selector,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    splits: dict,
    scaler: str,
    target_model: str,
    k_best: int | None = None,
    fill_func: str = "mean",
    **kwargs,
) -> pd.DataFrame:
    print(f"Processing target model {target_model}...")
    dfs = []
    for i in splits:
        print(f"Processing split {i}...")
        x_train = fill_na(features.iloc[splits[i]["train"]], fill_func=fill_func)
        x_train, _ = scale(x_train, scaler)
        y_train, _ = scale(metrics.iloc[splits[i]["train"]], scaler)
        selected_df = selector.compute_select(
            x=x_train,
            y=y_train[target_model],
            k_best=k_best,
        )
        selected_df.columns = [f"{col}_{i}" for col in selected_df.columns]
        dfs.append(selected_df)
    merged_df = pd.concat(
        dfs,
        axis=1
    )
    return merged_df


@rewrite_decorator
def save_errors(
        errors_dict: dict,
        **kwargs: dict,
) -> dict:
    return errors_dict
