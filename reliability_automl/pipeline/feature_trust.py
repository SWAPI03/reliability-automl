from __future__ import annotations

import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    trust_scores = state.trust_scores
    raw_df = state.raw_df
    target = state.target_column

    feature_cols = [c for c in raw_df.columns if c != target]
    feature_trust = {}

    for f in feature_cols:
        not_nan_mask = raw_df[f].notna().values
        if not_nan_mask.any():
            feature_trust[f] = float(np.mean(trust_scores[not_nan_mask]))
        else:
            feature_trust[f] = 0.0

    # Confidence score: 1 / (1 + variance) — always in (0, 1], no clipping needed
    # More stable than 1 - variance: never negative, smooth response to variance
    raw_var = float(np.var(trust_scores))
    confidence_score = 1.0 / (1.0 + raw_var)

    state.feature_trust = feature_trust
    state.confidence_score = confidence_score
    return state
