from __future__ import annotations

import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState

_VECTORIZED_LIMIT = 10000
_MIN_TRUST = 0.05   # floor: no row gets completely zeroed out


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    G = state.reliability_graph
    n_rows = len(state.processed_df)

    init_val = state.reliability_score if state.reliability_score is not None else 0.5
    trust_scores = np.full(n_rows, init_val, dtype=float)

    conflict_mask = (
        state.conflict_mask.astype(float)
        if state.conflict_mask is not None
        else np.zeros(n_rows, dtype=float)
    )
    cluster_labels = state.cluster_labels
    tol = config.trust_convergence_tol
    max_iter = config.trust_max_iterations

    # Pre-build adjacency list once
    adjacency = {i: list(G.neighbors(i)) for i in range(n_rows)}

    # For large graphs, limit iterations
    effective_max_iter = min(max_iter, 20) if n_rows > _VECTORIZED_LIMIT else max_iter

    for _ in range(effective_max_iter):
        new_trust = trust_scores.copy()

        # Pseudo-source trust per cluster
        if cluster_labels is not None:
            unique_clusters = np.unique(cluster_labels)
            pseudo_source_trust = {
                int(k): float(trust_scores[cluster_labels == k].mean())
                for k in unique_clusters
            }
        else:
            pseudo_source_trust = None

        for i in range(n_rows):
            neighbors = adjacency[i]
            if not neighbors:
                # Isolated node: preserve current trust
                new_trust[i] = trust_scores[i]
                continue

            # Neighbor influence term: 1 - prod(1 - t_j * sim_ij)
            product = 1.0
            for j in neighbors:
                product *= 1.0 - trust_scores[j] * G[i][j]["weight"]
            neighbor_term = 1.0 - product

            # Conflict penalty: reduce trust proportionally, not zero it out
            # Use soft penalty: multiply by (1 - 0.5 * conflict_i) instead of (1 - conflict_i)
            # This prevents complete zeroing of conflicting rows
            conflict_i = conflict_mask[i]
            conflict_factor = 1.0 - 0.5 * conflict_i  # 0.5 for conflicting, 1.0 for clean

            # Source trust
            if pseudo_source_trust is not None and cluster_labels is not None:
                source_trust = pseudo_source_trust[int(cluster_labels[i])]
            else:
                source_trust = init_val

            # Blend: 70% propagated + 30% initial value (prevents collapse)
            propagated = neighbor_term * conflict_factor * source_trust
            new_trust[i] = 0.7 * propagated + 0.3 * init_val

        # Apply floor and clip
        new_trust = np.clip(new_trust, _MIN_TRUST, 1.0)

        delta = np.max(np.abs(new_trust - trust_scores))
        trust_scores = new_trust

        if delta < tol:
            break

    # Re-normalize to [MIN_TRUST, 1.0] range
    t_min, t_max = trust_scores.min(), trust_scores.max()
    if t_max - t_min > 1e-6:
        trust_scores = _MIN_TRUST + (1.0 - _MIN_TRUST) * (trust_scores - t_min) / (t_max - t_min)

    state.trust_scores = trust_scores
    return state
