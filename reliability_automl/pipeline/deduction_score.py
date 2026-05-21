from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df = state.processed_df
    target = state.target_column

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    n_rows = len(X)

    n_clusters = min(config.n_pseudo_sources, n_rows)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    state.cluster_labels = labels

    # Bootstrap trust scores if not yet set
    # Use per-row structural quality as initial trust (better than uniform 0.5)
    if state.trust_scores is None:
        if state.raw_df is not None:
            missing_per_row = state.raw_df.isnull().mean(axis=1).values
            trust_scores = np.clip(1.0 - missing_per_row, 0.1, 1.0)
        else:
            trust_scores = np.full(n_rows, 0.5)
    else:
        trust_scores = state.trust_scores

    # Compute per-cluster mean trust
    pseudo_source_trust = np.array([
        trust_scores[labels == k].mean() if np.any(labels == k) else 0.5
        for k in range(n_clusters)
    ])

    D = float(np.mean(pseudo_source_trust[labels]))
    state.deduction_score = float(np.clip(D, 0.0, 1.0))
    return state
