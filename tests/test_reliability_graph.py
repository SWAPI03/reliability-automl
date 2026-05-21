"""Tests for reliability_graph.py — Property 13."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import euclidean

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.reliability_graph import run


def make_state(df: pd.DataFrame) -> PipelineState:
    state = PipelineState()
    state.processed_df = df
    return state


def test_graph_has_n_nodes():
    n = 10
    df = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
    state = run(make_state(df), PipelineConfig())
    assert state.reliability_graph.number_of_nodes() == n


def test_edges_only_where_sim_above_threshold():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]])
    df = pd.DataFrame(X, columns=["a", "b"])
    config = PipelineConfig(similarity_threshold=0.5)
    state = run(make_state(df), config)
    G = state.reliability_graph
    # rows 0 and 1 are identical → sim=1.0 > 0.5 → edge exists
    assert G.has_edge(0, 1)
    # rows 0 and 2 are far apart → sim << 0.5 → no edge
    assert not G.has_edge(0, 2)


def test_edge_weight_equals_sim():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])  # dist=5, sim=1/6
    df = pd.DataFrame(X, columns=["a", "b"])
    config = PipelineConfig(similarity_threshold=0.0)
    state = run(make_state(df), config)
    G = state.reliability_graph
    assert G.has_edge(0, 1)
    expected_sim = 1.0 / (1.0 + 5.0)
    assert G[0][1]["weight"] == pytest.approx(expected_sim, abs=1e-9)


def test_no_self_loops():
    df = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
    state = run(make_state(df), PipelineConfig())
    G = state.reliability_graph
    assert not any(G.has_edge(i, i) for i in range(5))


def test_graph_stored_in_state():
    df = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
    state = run(make_state(df), PipelineConfig())
    assert state.reliability_graph is not None
