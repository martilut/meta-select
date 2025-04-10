from pathlib import Path

import pandas as pd

from ms.metaresearch.data_preprocess import Preprocessor, cv_decorator
from ms.metaresearch.selector import Selector
from ms.utils.navigation import pjoin, rewrite_decorator


@rewrite_decorator
def run_selector(
        selector: Selector,
        features: pd.DataFrame,
        metrics: pd.DataFrame,
        split: dict,
        preprocessor: Preprocessor | None = None,
        k_best: int | None = None,
        save_path: Path | None = None,
        to_rewrite: bool = False,
) -> dict:
    result_dict = {i:[] for i in metrics.columns}
    errors_dict = {}

    for target_model in metrics.columns:
        target_res = run_target(
            x=features,
            y=metrics,
            split=split,
            preprocessor=preprocessor,
            to_agg=False,
            selector=selector,
            target_model=target_model,
            k_best=k_best,
            save_path=pjoin(
                save_path.parent,
                f"{target_model}.csv"
            ),
            to_rewrite=to_rewrite,
        )
        result_dict[target_model] = {i:[] for i in split}
        for i in split:
            result_dict[target_model][i] = target_res.loc[:, f"value_{i}"].dropna().index.to_list()
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
@cv_decorator
def run_target(
    selector: Selector,
    target_model: str,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    preprocessor: Preprocessor | None = None,
    k_best: int | None = None,
    inner_split: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    print(f"Processing target model {target_model}...")
    return selector.compute_select(
        x=x_train,
        y=y_train[target_model].to_frame(),
        split=inner_split if selector.cv else None,
        preprocessor=preprocessor,
        to_agg=True,
        k_best=k_best,
    )


@rewrite_decorator
def save_errors(
        errors_dict: dict,
        **kwargs: dict,
) -> dict:
    return errors_dict
