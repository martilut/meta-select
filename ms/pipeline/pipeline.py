import random

import numpy as np
import pandas as pd

from ms.config.experiment_config import ExperimentConfig
from ms.metalearning.isa import InstanceSpaceAnalysis
from ms.metalearning.meta_model import MetaModel
from ms.pipeline.runner import aggregate_selector_folds, run_selector
from ms.plot.plotting import save_isa_data
from ms.processing.split import split_k_fold
from ms.selection.selector import Selector
from ms.utils.navigation import get_file_name, load, pjoin
from ms.utils.utils import is_classif, measure_runtime, save_runtime

np.random.seed(ExperimentConfig.SEED)
random.seed(ExperimentConfig.SEED)


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
            outer_k=ExperimentConfig.OUTER_K,
            inner_k=ExperimentConfig.INNER_K,
            shuffle=True,
            seed=ExperimentConfig.SEED,
            save_path=pjoin(
                ExperimentConfig.CONF.results_path,
                source,
                metrics_suffix,
                f"{target_name}.json",
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
                preprocessor=ExperimentConfig.PREPROCESSOR,
                k_best=None,
                save_path=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "res.json",
                ),
            )
            save_runtime(
                runtime=time,
                row=target_name,
                column=selector.name,
                save_path=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "time.csv",
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
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    f"{target_name}.json",
                ),
            )
            selector_data = run_selector(
                save_path=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "res.json",
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
                    opt_scoring=(
                        ExperimentConfig.OPT_SCORING_CLASS
                        if is_classif(target)
                        else ExperimentConfig.OPT_SCORING_REG
                    ),
                    model_scoring=(
                        ExperimentConfig.MODEL_SCORING_CLASS
                        if is_classif(target)
                        else ExperimentConfig.MODEL_SCORING_REG
                    ),
                    n_trials=10,
                    preprocessor=ExperimentConfig.PREPROCESSOR,
                    subset=selector_data,
                    save_path=pjoin(
                        ExperimentConfig.CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "pred",
                        f"{model.name}.csv",
                    ),
                    to_rewrite=True,
                )
                model.save_params(
                    save_path=pjoin(
                        ExperimentConfig.CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "pred",
                        f"{model.name}_params.json",
                    ),
                    to_rewrite=True,
                )


def run_isa(
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    source: str,
    metrics_suffix: str,
    selectors: list[Selector],
    models: dict[MetaModel, list[str]],
    isa: InstanceSpaceAnalysis,
) -> None:
    for target_name in metrics.columns:
        print(f"Running models for {target_name}")
        for selector in selectors:
            print(f"Running selector {selector.name}")
            target = metrics.loc[:, target_name].to_frame()
            target_split = split_k_fold(
                save_path=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    f"{target_name}.json",
                ),
            )
            csv_data = load(
                path=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "data.csv",
                ),
            )
            selector_data = aggregate_selector_folds(csv_data)
            if len(selector_data) > 15:
                selector_data = selector_data[:15]
            pilot_res = isa.pilot(
                features=features.loc[:, selector_data],
                metrics=target,
                preprocessor=ExperimentConfig.PREPROCESSOR,
            )
            # summary = pilot_res.summary
            isa_features = pd.DataFrame(
                pilot_res.z, index=features.index, columns=["z1", "z2"]
            )
            save_isa_data(
                selected_names=selector_data,
                features=isa_features,
                metrics=target,
                res=pilot_res,
                selector_name=selector.name,
                save_plot=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "isa_data",
                    target_name,
                    "plot.png",
                ),
                save_path=pjoin(
                    ExperimentConfig.CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "isa_data",
                    target_name,
                    "data.json",
                ),
                to_rewrite=False,
            )
            for model in models:
                if selector.name not in models[model]:
                    continue
                print(f"Running model {model.name}")
                model.run(
                    x=isa_features,
                    y=target,
                    split=target_split,
                    opt_scoring=(
                        ExperimentConfig.OPT_SCORING_CLASS
                        if is_classif(target)
                        else ExperimentConfig.OPT_SCORING_REG
                    ),
                    model_scoring=(
                        ExperimentConfig.MODEL_SCORING_CLASS
                        if is_classif(target)
                        else ExperimentConfig.MODEL_SCORING_REG
                    ),
                    n_trials=10,
                    preprocessor=ExperimentConfig.PREPROCESSOR,
                    subset=selector_data,
                    save_path=pjoin(
                        ExperimentConfig.CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "isa_pred",
                        f"{model.name}.csv",
                    ),
                    to_rewrite=False,
                )
                model.save_params(
                    save_path=pjoin(
                        ExperimentConfig.CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "isa_pred",
                        f"{model.name}_params.json",
                    ),
                    to_rewrite=False,
                )


if __name__ == "__main__":
    source = "tabzilla"
    feature_suffix = None
    metrics_suffix = "type"
    task_type = "class"

    non_wrapper = [
        ExperimentConfig.BASE_SELECTOR,
        ExperimentConfig.CORR,
        ExperimentConfig.MI,
        ExperimentConfig.F_VALUE,
        ExperimentConfig.XGB_SELECTOR,
        ExperimentConfig.LASSO_SELECTOR,
        ExperimentConfig.TE,
    ]
    wrapper = []
    if task_type == "class":
        wrapper.append(ExperimentConfig.RFE_CLASS_XGB)
    elif task_type == "reg":
        wrapper.append(ExperimentConfig.RFE_REG_XGB)
    selectors = wrapper + non_wrapper

    if task_type == "class":
        models = {
            ExperimentConfig.KNN_CLASS: [s.name for s in non_wrapper],
            ExperimentConfig.XGB_CLASS: [s.name for s in non_wrapper + wrapper],
            ExperimentConfig.MLP_CLASS: [s.name for s in non_wrapper],
        }
    else:
        models = {
            ExperimentConfig.KNN_REG: [s.name for s in non_wrapper],
            ExperimentConfig.XGB_REG: [s.name for s in non_wrapper + wrapper],
            ExperimentConfig.MLP_REG: [s.name for s in non_wrapper],
        }

    features, metrics = ExperimentConfig.get_data(
        features_path=pjoin(
            ExperimentConfig.CONF.resources_path,
            source,
            "filtered",
            f"{get_file_name(prefix='features', suffix=feature_suffix)}.csv",
        ),
        metrics_path=pjoin(
            ExperimentConfig.CONF.resources_path,
            source,
            "target",
            f"{get_file_name(prefix='metrics', suffix=metrics_suffix)}.csv",
        ),
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
    run_isa(
        features=features,
        metrics=metrics,
        source=source,
        metrics_suffix=metrics_suffix,
        selectors=selectors,
        models=models,
        isa=InstanceSpaceAnalysis(),
    )
