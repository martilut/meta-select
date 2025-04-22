import random

import numpy as np
import pandas as pd

from ms.config.pipeline_constants import SEED, get_data, PREPROCESSOR, opt_scoring_class, model_scoring_class, CONF, \
    OUTER_K, INNER_K, base, corr, mi, f_value, xgb, lasso, te, cf, rfe_reg_xgb, rfe_reg_mlp, knn_reg_mm, xgb_reg_mm, \
    mlp_reg_mm, opt_scoring_reg, model_scoring_reg
from ms.metalearning.meta_model import MetaModel
from ms.pipeline.runner import run_selector
from ms.processing.split import split_k_fold
from ms.selection.selector import Selector
from ms.utils.navigation import pjoin, get_file_name
from ms.utils.utils import measure_runtime, save_runtime, is_classif

np.random.seed(SEED)
random.seed(SEED)


def run_selectors(
        features: pd.DataFrame,
        metrics: pd.DataFrame,
        source: str,
        metrics_suffix: str,
        selectors: list[Selector],
) -> None:
    for target_name in metrics.columns:
        print(f"Running selectors for {target_name}")
        target = metrics.loc[:, target_name].to_frame()
        target_split = split_k_fold(
            x_df=features,
            y_df=target,
            outer_k=OUTER_K,
            inner_k=INNER_K,
            shuffle=True,
            seed=SEED,
            save_path=pjoin(
                CONF.results_path,
                source,
                metrics_suffix,
                f"{target_name}.json"
            ),
        )
        for selector in selectors:
            print(f"Running {selector.name}")
            _, time = measure_runtime(
                func=run_selector,
                selector=selector,
                features=features,
                metrics=target,
                split=target_split,
                preprocessor=PREPROCESSOR,
                k_best=None,
                save_path=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "res.json"
                ),
            )
            save_runtime(
                runtime=time,
                row=target_name,
                column=selector.name,
                save_path=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "time.csv"
                ),
            )


def run_models(
        features: pd.DataFrame,
        metrics: pd.DataFrame,
        source: str,
        metrics_suffix: str,
        selectors: list[Selector],
        models: dict[MetaModel, list[str]],
) -> None:
    for target_name in metrics.columns:
        print(f"Running models for {target_name}")
        for selector in selectors:
            print(f"Running selector {selector.name}")
            target = metrics.loc[:, target_name].to_frame()
            target_split = split_k_fold(
                save_path=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    f"{target_name}.json"
                ),
            )
            selector_data = run_selector(
                save_path=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "res.json"
                ),
            )
            for model in models:
                if selector.name not in models[model]:
                    continue
                print(f"Running model {model.name}")
                model.run(
                    x=features,
                    y=target,
                    split=target_split,
                    opt_scoring=opt_scoring_class if is_classif(target) else opt_scoring_reg,
                    model_scoring=model_scoring_class if is_classif(target) else model_scoring_reg,
                    n_trials=10,
                    preprocessor=PREPROCESSOR,
                    subset=selector_data,
                    save_path=pjoin(
                        CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "pred",
                        f"{model.name}.csv"
                    ),
                    to_rewrite=False,
                )
                model.save_params(
                    save_path=pjoin(
                        CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "pred",
                        f"{model.name}_params.json"
                    ),
                    to_rewrite=False,
                )


if __name__ == "__main__":
    source = "tabzilla"
    feature_suffix = None
    metrics_suffix = "raw"

    non_wrapper = [
        base,
        corr,
        mi,
        f_value,
        xgb,
        lasso,
        te,
    ]
    wrapper = [rfe_reg_xgb, rfe_reg_mlp]
    selectors = wrapper + non_wrapper

    models = {
        knn_reg_mm: [s.name for s in non_wrapper],
        xgb_reg_mm: [s.name for s in non_wrapper + [rfe_reg_xgb]],
        mlp_reg_mm: [s.name for s in non_wrapper + [rfe_reg_mlp]],
    }

    features, metrics = get_data(
        features_path=pjoin(
            CONF.resources_path,
            source,
            "filtered",
            f"{get_file_name(prefix='features', suffix=feature_suffix)}.csv"
        ),
        metrics_path=pjoin(
            CONF.resources_path,
            source,
            "target",
            f"{get_file_name(prefix='metrics', suffix=metrics_suffix)}.csv"
        )
    )
    run_selectors(
        features=features,
        metrics=metrics,
        source=source,
        metrics_suffix=metrics_suffix,
        selectors=selectors,
    )
    run_models(
        features=features,
        metrics=metrics,
        source=source,
        metrics_suffix=metrics_suffix,
        selectors=selectors,
        models=models,
    )
