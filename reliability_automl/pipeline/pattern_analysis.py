from __future__ import annotations

import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState

_FULL_MATRIX_LIMIT = 5000
_SAMPLE_LIMIT = 10000
_TOP_K = 20


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df = state.processed_df
    target = state.target_column
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    n_rows = len(X)

    # ── Similarity score ──────────────────────────────────────────────────────
    if n_rows <= _FULL_MATRIX_LIMIT:
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(X, X, metric="euclidean")
        sim_matrix = 1.0 / (1.0 + dist_matrix)
        mask = ~np.eye(n_rows, dtype=bool)
        sim_score = float(sim_matrix[mask].mean())
    else:
        # Sample for similarity estimation
        rng = np.random.default_rng(42)
        sample_size = min(_SAMPLE_LIMIT, n_rows)
        idx = rng.choice(n_rows, size=sample_size, replace=False)
        X_sample = X[idx]
        k = min(_TOP_K, sample_size - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
        nbrs.fit(X_sample)
        distances, _ = nbrs.kneighbors(X_sample)
        # distances[:, 0] is self (0), skip it
        neighbor_dists = distances[:, 1:]
        sims = 1.0 / (1.0 + neighbor_dists)
        sim_score = float(sims.mean())

    # ── Conflict detection ────────────────────────────────────────────────────
    raw_target = state.raw_df[target].values if target else None
    conflict_mask = np.zeros(n_rows, dtype=bool)
    total_pairs = n_rows * (n_rows - 1) / 2
    conflicting_pairs = 0

    if raw_target is not None and n_rows > 1:
        # For large datasets, only check within approximate duplicate groups
        # Round features to detect near-duplicates efficiently
        if n_rows > _FULL_MATRIX_LIMIT:
            # Use rounded feature vectors as keys (coarse grouping)
            X_rounded = np.round(X, decimals=3)
            groups: dict = defaultdict(list)
            for i, row in enumerate(X_rounded):
                groups[tuple(row)].append(i)
        else:
            groups = defaultdict(list)
            for i, row in enumerate(X):
                groups[tuple(row)].append(i)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            labels = raw_target[indices]
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                n_g = len(indices)
                for a in range(n_g):
                    for b in range(a + 1, n_g):
                        if labels[a] != labels[b]:
                            conflicting_pairs += 1
                            conflict_mask[indices[a]] = True
                            conflict_mask[indices[b]] = True

    C = conflicting_pairs / total_pairs if total_pairs > 0 else 0.0

    state.similarity_score = float(np.clip(sim_score, 0.0, 1.0))
    state.conflict_score = float(np.clip(C, 0.0, 1.0))
    state.conflict_mask = conflict_mask
    return state
