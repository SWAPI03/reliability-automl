from __future__ import annotations

import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    if state.best_model_name is not None and state.best_model_name in state.trained_models:
        model_name = state.best_model_name
    else:
        model_name = next(iter(state.trained_models))

    model = state.trained_models[model_name]
    X = state.processed_df.values
    y_raw = state.raw_df[state.target_column].values
    # Use encoded labels if a label encoder was applied during training
    if state.task_type == "classification" and getattr(state, "label_encoder", None) is not None:
        y = state.label_encoder.transform(y_raw)
    else:
        y = y_raw
    trust_scores = state.trust_scores.copy()
    n_rows = len(X)

    if state.task_type == "classification":
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        # Build O(1) lookup dict
        class_to_idx = {c: i for i, c in enumerate(classes)}
        errors = np.zeros(n_rows, dtype=float)
        for i in range(n_rows):
            true_label = y[i]
            idx = class_to_idx.get(true_label)
            errors[i] = 1.0 - proba[i, idx] if idx is not None else 1.0
    else:
        preds = model.predict(X)
        y_range = float(np.max(y) - np.min(y))
        errors = np.abs(preds - y) / y_range if y_range > 0 else np.zeros(n_rows, dtype=float)

    errors = np.clip(errors, 0.0, 1.0)
    trust_scores = trust_scores * (1.0 - errors)

    max_trust = float(np.max(trust_scores))
    if max_trust > 0:
        trust_scores = trust_scores / max_trust
    else:
        # All trust collapsed to 0 — reset to uniform 1.0 so weights stage can normalize
        trust_scores = np.ones(n_rows, dtype=float)

    state.trust_scores = trust_scores
    return state
