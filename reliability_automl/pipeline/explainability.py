from __future__ import annotations

import numpy as np
import shap

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState

_LARGE_DATASET_THRESHOLD = 1000
_SHAP_SAMPLE_SIZE = 100


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    model = state.trained_models[state.best_model_name]
    X = state.processed_df.values
    feature_names = state.processed_df.columns.tolist()
    n_rows, n_features = X.shape
    model_name = state.best_model_name

    background = shap.sample(X, _SHAP_SAMPLE_SIZE) if n_rows > _LARGE_DATASET_THRESHOLD else X

    tree_models = ("RandomForest", "XGBoost")
    linear_models = ("LogisticRegression", "LinearRegression")

    if any(t in model_name for t in tree_models):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
    elif any(t in model_name for t in linear_models):
        explainer = shap.LinearExplainer(model, background)
        shap_vals = explainer.shap_values(X)
    else:
        explainer = shap.KernelExplainer(model.predict, background)
        shap_vals = explainer.shap_values(X)

    # Handle multi-class: list of arrays → mean absolute over classes
    if isinstance(shap_vals, list):
        shap_array = np.mean(np.abs(np.stack(shap_vals, axis=0)), axis=0)
    else:
        shap_array = shap_vals

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)

    state.shap_values = shap_array

    shap_feature_importance = {
        f: float(np.mean(np.abs(shap_array[:, idx])))
        for idx, f in enumerate(feature_names)
    }
    state.shap_feature_importance = shap_feature_importance

    feature_trust = state.feature_trust or {}
    n_top = max(1, int(0.25 * n_features))
    sorted_features = sorted(shap_feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = {f for f, _ in sorted_features[:n_top]}

    state.high_importance_low_trust_features = [
        f for f in feature_names
        if feature_trust.get(f, 1.0) < 0.5 and f in top_features
    ]
    return state
