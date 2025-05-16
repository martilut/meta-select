from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import torch
from dowhy import CausalModel
from econml.dml import CausalForestDML
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ms.selection.selector import Selector


class TESelector(Selector):
    @property
    def name(self) -> str:
        return "te"

    def __init__(
        self,
        model_y: Any = None,
        model_t: Any = None,
        to_tune: bool = True,
        quantile_value: float = 0.8,
        max_depth: int = 50,
        min_leaf: int = 5,
        n_splits: int = 2,
        n_estimators: int = 100,
        n_trees: int = 500,
        n_jobs: int = -1,
        mode: str = "individual",
        model_type: str = "cf",  # or "drof"
        random_state: int | None = None,
        cv: bool = True,
    ):
        super().__init__(cv=cv)
        default_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
        )
        self.model_y = default_model if model_y is None else model_y
        self.model_t = default_model if model_t is None else model_t
        self.to_tune = to_tune
        self.quantile_value = quantile_value
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.mode = mode  # "individual" or "joint"
        self.model_type = model_type
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def _get_model(self):
        if self.model_type == "cf":
            return CausalForestDML(
                model_y=self.model_y,
                model_t=self.model_t,
                cv=2,
                criterion="het",
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_leaf,
                max_depth=self.max_depth,
                fit_intercept=True,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        elif self.model_type == "drof":
            return DROrthoForest(
                n_trees=self.n_trees,
                min_leaf_size=self.min_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Invalid model_type. Choose 'cf' or 'drof'")

    def _compute_effect_individual(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        i: int,
        f_name: str,
    ):
        print(f"Processing feature {f_name}..., {i}/{x_train.shape[1]}")
        t_train, t_test = x_train.iloc[:, i].to_numpy(), x_test.iloc[:, i].to_numpy()
        x_train_cov = np.delete(x_train, i, axis=1)
        x_test_cov = np.delete(x_test, i, axis=1)

        dml = self._get_model()

        if self.to_tune:
            dml.tune(Y=y_train.to_numpy().ravel(), T=t_train, X=x_train_cov)

        dml.fit(Y=y_train.to_numpy().ravel(), T=t_train, X=x_train_cov)
        treatment_effects = dml.effect(x_test_cov)
        return f_name, treatment_effects

    def _compute_effect_joint(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ):
        dml = self._get_model()

        if self.to_tune:
            dml.tune(Y=y_train, T=x_train, X=x_train)

        dml.fit(Y=y_train, T=x_train, X=x_train)
        treatment_effects = dml.effect(x_test)
        return treatment_effects

    def compute_generic(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        task: str = "class",
    ) -> pd.DataFrame:
        x_test = x_train if x_test is None else x_test

        res = pd.DataFrame(index=x_train.columns, columns=["value"])

        effect_results = {f_name: [] for f_name in x_train.columns}

        if self.mode == "individual":
            print("Processing individual features...")
            results = [
                self._compute_effect_individual(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    i=i,
                    f_name=f_name,
                )
                for i, f_name in enumerate(x_train.columns)
            ]
            for f_name, treatment_effects in results:
                effect_results[f_name].extend(treatment_effects)
        elif self.mode == "joint":
            print("Processing joint features...")
            treatment_effects = self._compute_effect_joint(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
            )
            for i, f_name in enumerate(x_train.columns):
                effect_results[f_name].extend(treatment_effects[:, i])
        else:
            raise ValueError("Invalid mode. Choose 'individual' or 'joint'")

        res["value"] = [np.mean(effect_results[f_name]) for f_name in x_train.columns]
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        quantile_eff = res["value"].abs().quantile(self.quantile_value)
        for f_name in res.index:
            if abs(res.loc[f_name, "value"]) < quantile_eff:
                res.loc[f_name, "value"] = None
        return res


class TEDAGSelector(Selector):
    @property
    def name(self) -> str:
        return "te_dag"

    def __init__(
        self,
        cv: bool = True,
        reg_method: str = "backdoor.linear_regression",
        class_method: str = "backdoor.propensity_score_matching",
        method_params: dict | None = None,
    ) -> None:
        super().__init__(cv=cv)
        self.reg_method = reg_method
        self.class_method = class_method
        self.method_params = method_params

    def compute_generic(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        task: str = "class",
    ) -> pd.DataFrame:
        train = x_train.copy()
        train["y"] = y_train.copy()
        test = x_test.copy() if x_test is not None else x_train.copy()
        test["y"] = y_test.copy() if y_test is not None else y_train.copy()
        res = {}
        for feature in x_train.columns:
            model = CausalModel(
                data=train,
                treatment=feature,
                outcome="y",
                common_causes=[i for i in train.columns if i not in ["y", feature]],
            )
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True, method_name="maximal-adjustment"
            )
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name=self.reg_method,
                method_params=self.method_params,
            )
            res[feature] = causal_estimate.value
        res = pd.DataFrame.from_dict(res, orient="index", columns=["value"])
        res.index = x_train.columns
        return res

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["value"].abs() == 0.0, "value"] = None
        return res


class ClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class CFSelector(Selector):
    @property
    def name(self) -> str:
        return "cf"

    def __init__(
        self,
        cf_steps: int = 500,
        train_epochs: int = 300,
        dc: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cv: bool = False,
    ) -> None:
        super().__init__(cv=cv)
        self.cf_steps = cf_steps
        self.train_epochs = train_epochs
        self.dc = dc
        self.device = device

    def compute_generic(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        task: str = "class",
    ) -> pd.DataFrame:
        X = np.array(x_train)
        y = np.array(y_train).flatten()
        num_features = X.shape[1]
        fitness_results = {}

        with ThreadPoolExecutor(max_workers=num_features) as executor:
            futures = {}
            for feat_idx in range(num_features):
                mask = np.zeros(num_features, dtype=bool)
                mask[feat_idx] = True
                future = executor.submit(self._evaluate_feature_subset, X, y, mask)
                futures[future] = feat_idx

            for future in as_completed(futures):
                feat_idx = futures[future]
                fitness = future.result()
                fitness_results[feat_idx] = fitness

        inf_df = pd.DataFrame.from_dict(
            fitness_results, orient="index", columns=["value"]
        )
        new_features_names = [x_train.columns[i] for i in fitness_results]
        inf_df.index = new_features_names
        return inf_df

    def __select__(self, res: pd.DataFrame) -> pd.DataFrame:
        res.loc[res["value"].abs() == 0.0, "value"] = None
        return res

    def _evaluate_feature_subset(self, X_train, y_train, feature_mask):
        X_train_masked = X_train[:, feature_mask]
        X_train_tensor = torch.FloatTensor(X_train_masked).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        model = ClassificationModel(feature_mask.sum()).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.train_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        med = torch.median(X_train_tensor, dim=0).values
        mad = torch.median(torch.abs(X_train_tensor - med) + 1e-6, dim=0).values

        model.eval()
        with torch.no_grad():
            logits = model(X_train_tensor)
            original_classes = torch.argmax(logits, dim=-1)

        target_tensor = ((original_classes + 1) % 2).to(self.device)
        cf_batch = self._generate_counterfactual_batch(
            model, X_train_tensor, target_tensor, mad
        )

        with torch.no_grad():
            logits_cf = model(cf_batch)
            cf_classes = torch.argmax(logits_cf, dim=-1)

        cf_accuracy = (cf_classes == target_tensor).float().mean().item()
        fitness = cf_accuracy
        return fitness

    def _generate_counterfactual_batch(self, model, x_original, target_class, mad):
        x_cf = x_original.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([x_cf], lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.cf_steps):
            optimizer.zero_grad()
            logits = model(x_cf)
            pred_loss = criterion(logits, target_class)
            distance_loss = ((x_cf - x_original).abs() / mad).sum(dim=1).mean()
            loss = pred_loss + self.dc * distance_loss
            loss.backward()
            optimizer.step()

        return x_cf.detach()
