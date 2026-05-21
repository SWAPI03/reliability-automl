from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState

_FULL_MATRIX_LIMIT = 5000   # use full pairwise only below this
_TOP_K_NEIGHBORS = 10       # k-NN edges for large datasets
_SAMPLE_LIMIT = 50000       # hard cap: sample rows if dataset is huge


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df = state.processed_df
    X = df.values
    n_rows = len(X)

    G = nx.Graph()
    G.add_nodes_from(range(n_rows))
    threshold = config.similarity_threshold

    # For very large datasets, sample a representative subset
    if n_rows > _SAMPLE_LIMIT:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_rows, size=_SAMPLE_LIMIT, replace=False)
        X_work = X[sample_idx]
        idx_map = sample_idx          # maps local → original row index
    else:
        X_work = X
        idx_map = np.arange(n_rows)

    n_work = len(X_work)

    if n_work <= _FULL_MATRIX_LIMIT:
        # Full pairwise — safe for small datasets
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(X_work, X_work, metric="euclidean")
        sim_matrix = 1.0 / (1.0 + dist_matrix)
        for i in range(n_work):
            for j in range(i + 1, n_work):
                s = float(sim_matrix[i, j])
                if s > threshold:
                    G.add_edge(int(idx_map[i]), int(idx_map[j]), weight=s)
    else:
        # k-NN approximate — memory-safe for large datasets
        k = min(_TOP_K_NEIGHBORS, n_work - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
        nbrs.fit(X_work)
        distances, indices = nbrs.kneighbors(X_work)

        for i in range(n_work):
            for rank in range(1, k + 1):   # skip rank 0 (self)
                j = int(indices[i, rank])
                if j <= i:
                    continue
                dist = float(distances[i, rank])
                s = 1.0 / (1.0 + dist)
                if s > threshold:
                    G.add_edge(int(idx_map[i]), int(idx_map[j]), weight=s)

    state.reliability_graph = G
    return state
