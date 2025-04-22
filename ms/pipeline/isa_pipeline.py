import random

import numpy as np
import pandas as pd

from ms.config.pipeline_constants import SEED, get_data, PREPROCESSOR, opt_scoring_class, model_scoring_class, CONF, \
    base, corr, mi, f_value, xgb, lasso, te, rfe_reg_xgb, rfe_reg_mlp, knn_reg_mm, xgb_reg_mm, \
    mlp_reg_mm, opt_scoring_reg, model_scoring_reg
from ms.metalearning.isa import InstanceSpaceAnalysis
from ms.metalearning.meta_model import MetaModel
from ms.pipeline.runner import aggregate_selector_folds, save_isa_data
from ms.processing.split import split_k_fold
from ms.selection.selector import Selector
from ms.utils.navigation import pjoin, get_file_name, load
from ms.utils.utils import is_classif

np.random.seed(SEED)
random.seed(SEED)


def run_models(
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
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    f"{target_name}.json"
                ),
            )
            csv_data = load(
                path=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "data.csv"
                ),
            )
            selector_data = aggregate_selector_folds(csv_data)
            if len(selector_data) > 15:
                selector_data = selector_data[:15]
            pilot_res = isa.pilot(
                features=features.loc[:, selector_data],
                metrics=target,
                preprocessor=PREPROCESSOR,
            )
            # summary = pilot_res.summary
            isa_features = pd.DataFrame(pilot_res.z, index=features.index, columns=["z1", "z2"])
            save_isa_data(
                selected_names=selector_data,
                features=isa_features,
                metrics=target,
                res=pilot_res,
                selector_name=selector.name,
                save_plot=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "isa_data",
                    target_name,
                    "plot.png"
                ),
                save_path=pjoin(
                    CONF.results_path,
                    source,
                    metrics_suffix,
                    target_name,
                    selector.name,
                    "isa_data",
                    target_name,
                    "data.json"
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
                    opt_scoring=opt_scoring_class if is_classif(target) else opt_scoring_reg,
                    model_scoring=model_scoring_class if is_classif(target) else model_scoring_reg,
                    n_trials=10,
                    # preprocessor=PREPROCESSOR,
                    # subset=selector_data,
                    save_path=pjoin(
                        CONF.results_path,
                        source,
                        metrics_suffix,
                        target_name,
                        selector.name,
                        "isa_pred",
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
                        "isa_pred",
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
        # cf,
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

    run_models(
        features=features,
        metrics=metrics,
        source=source,
        metrics_suffix=metrics_suffix,
        selectors=selectors,
        models=models,
        isa=InstanceSpaceAnalysis()
    )
