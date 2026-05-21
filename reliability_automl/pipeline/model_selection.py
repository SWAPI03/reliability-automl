from __future__ import annotations

import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    R = state.reliability_score if state.reliability_score is not None else 0.5

    # y_range only needed for regression; use encoded values if available
    if state.task_type == "regression":
        y_raw = state.raw_df[state.target_column].values
        try:
            y_numeric = y_raw.astype(float)
            y_range = float(np.max(y_numeric) - np.min(y_numeric))
        except (ValueError, TypeError):
            y_range = 1.0
    else:
        y_range = 1.0

    best_name = None
    best_score = -np.inf

    for name, scores in state.model_scores.items():
        cv_score = scores["cv_score"]

        if state.task_type == "classification":
            reliability_score = cv_score * R
        else:
            # cv_score is neg_root_mean_squared_error from sklearn → always negative
            # normalized_rmse = |cv_score| / y_range (uses actual RMSE, not just 1/y_range)
            normalized_rmse = abs(cv_score) / y_range if y_range > 0 else 0.0
            # Clip to [0,1] so extreme RMSE doesn't produce negative reliability score
            normalized_rmse = min(normalized_rmse, 1.0)
            reliability_score = (1.0 - normalized_rmse) * R

        state.model_scores[name]["reliability_adjusted_score"] = float(reliability_score)

        if reliability_score > best_score:
            best_score = reliability_score
            best_name = name

    state.best_model_name = best_name
    return state
