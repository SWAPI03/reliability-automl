"""Tests for trust_propagation.py — Properties 14, 15."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.trust_propagation import run


def make_state(n: int, graph: nx.Graph, R: float = 0.7,
               conflict_mask=None, cluster_labels=None) -> PipelineState:
    state = PipelineState()
    state.processed_df = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    state.reliability_graph = graph
    state.reliability_score = R
    state.conflict_mask = conflict_mask
    state.cluster_labels = cluster_labels
    return state


def test_trust_scores_in_range():
    n = 10
    G = nx.path_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.8
    state = run(make_state(n, G), PipelineConfig())
    assert np.all(state.trust_scores >= 0.0)
    assert np.all(state.trust_scores <= 1.0)


def test_trust_scores_length():
    n = 15
    G = nx.path_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.6
    state = run(make_state(n, G), PipelineConfig())
    assert len(state.trust_scores) == n


def test_isolated_nodes_keep_initial_trust():
    n = 5
    G = nx.Graph()
    G.add_nodes_from(range(n))  # no edges
    R = 0.75
    state = run(make_state(n, G, R=R), PipelineConfig())
    # All nodes isolated → trust stays at R
    assert np.allclose(state.trust_scores, R)


def test_iteration_cap():
    """Propagation must stop at max_iterations even if not converged."""
    n = 5
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.99
    config = PipelineConfig(trust_max_iterations=3, trust_convergence_tol=1e-10)
    state = run(make_state(n, G), config)
    assert state.trust_scores is not None
    assert len(state.trust_scores) == n


def test_conflict_reduces_trust():
    """Conflicting rows should have lower trust than non-conflicting."""
    n = 4
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.8
    conflict_mask = np.array([True, True, False, False])
    state = run(make_state(n, G, conflict_mask=conflict_mask), PipelineConfig())
    # Conflicting rows (0,1) should have lower trust than non-conflicting (2,3)
    assert state.trust_scores[0] <= state.trust_scores[2] + 1e-9
