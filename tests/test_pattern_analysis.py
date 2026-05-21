"""Tests for pattern_analysis.py — Properties 7, 8, 9."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import euclidean

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.pattern_analysis import run


def make_state(processed_df: pd.DataFrame, raw_df: pd.DataFrame, target: str = "target") -> PipelineState:
    state = PipelineState()
    state.processed_df = processed_df
    state.raw_df = raw_df
    state.target_column = target
    return state


# ── sim(i,j) formula ───────────────────────────────────────────────────────────

def test_sim_formula_identical_vectors():
    """Identical vectors → dist=0 → sim=1."""
    v = np.array([1.0, 2.0, 3.0])
    dist = euclidean(v, v)
    sim = 1.0 / (1.0 + dist)
    assert sim == pytest.approx(1.0)


def test_sim_formula_known_distance():
    """sim(i,j) = 1/(1+dist) for known vectors."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([3.0, 4.0])
    dist = 5.0  # 3-4-5 triangle
    expected_sim = 1.0 / (1.0 + 5.0)
    assert 1.0 / (1.0 + euclidean(v1, v2)) == pytest.approx(expected_sim)


def test_sim_always_in_0_1():
    """sim must be in (0, 1] for any two vectors."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        v1 = rng.normal(size=5)
        v2 = rng.normal(size=5)
        sim = 1.0 / (1.0 + euclidean(v1, v2))
        assert 0.0 < sim <= 1.0


# ── Similarity score aggregation ───────────────────────────────────────────────

def test_sim_score_two_identical_rows():
    """Two identical rows → dist=0 → Sim=1."""
    df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0], "target": [0, 0]})
    raw_df = df.copy()
    state = run(make_state(df, raw_df), PipelineConfig())
    assert state.similarity_score == pytest.approx(1.0)


def test_sim_score_range():
    """Similarity score must be in (0, 1]."""
    rng = np.random.default_rng(1)
    n = 20
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "target": np.zeros(n),
    })
    state = run(make_state(df, df.copy()), PipelineConfig())
    assert 0.0 < state.similarity_score <= 1.0


def test_sim_score_manual_computation():
    """Manually compute Sim for 3 rows and verify."""
    X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    df = pd.DataFrame(X, columns=["a", "b"])
    df["target"] = 0
    raw_df = df.copy()

    from scipy.spatial.distance import cdist
    dist_mat = cdist(X, X, metric="euclidean")
    sim_mat = 1.0 / (1.0 + dist_mat)
    n = 3
    off_diag = sim_mat[~np.eye(n, dtype=bool)]
    expected_sim = off_diag.mean()

    state = run(make_state(df, raw_df), PipelineConfig())
    assert state.similarity_score == pytest.approx(expected_sim, abs=1e-9)


# ── Conflict detection ─────────────────────────────────────────────────────────

def test_no_conflicts():
    """Unique feature vectors → no conflicts."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    raw_df = df.copy()
    raw_df["target"] = [0, 1, 0]
    df["target"] = [0, 1, 0]
    state = run(make_state(df, raw_df), PipelineConfig())
    assert state.conflict_score == pytest.approx(0.0)
    assert state.conflict_mask.sum() == 0


def test_conflict_identical_features_different_labels():
    """Two rows with same features but different labels → conflict."""
    df = pd.DataFrame({"a": [1.0, 1.0, 3.0], "b": [2.0, 2.0, 4.0]})
    raw_df = df.copy()
    raw_df["target"] = [0, 1, 0]  # rows 0 and 1 conflict
    df["target"] = [0, 1, 0]
    state = run(make_state(df, raw_df), PipelineConfig())
    # 1 conflicting pair out of 3 total pairs
    total_pairs = 3 * 2 / 2  # = 3
    expected_C = 1.0 / total_pairs
    assert state.conflict_score == pytest.approx(expected_C)
    assert state.conflict_mask[0] == True  # noqa: E712
    assert state.conflict_mask[1] == True  # noqa: E712
    assert state.conflict_mask[2] == False  # noqa: E712


def test_conflict_mask_shape():
    """conflict_mask must have same length as dataframe."""
    n = 15
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "target": np.zeros(n)})
    state = run(make_state(df, df.copy()), PipelineConfig())
    assert len(state.conflict_mask) == n


def test_conflict_score_zero_when_same_labels():
    """Identical feature rows with same label → no conflict."""
    df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]})
    raw_df = df.copy()
    raw_df["target"] = [1, 1]
    df["target"] = [1, 1]
    state = run(make_state(df, raw_df), PipelineConfig())
    assert state.conflict_score == pytest.approx(0.0)


def test_conflict_score_range():
    """Conflict score must be in [0, 1]."""
    rng = np.random.default_rng(5)
    n = 10
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    raw_df = df.copy()
    raw_df["target"] = rng.integers(0, 2, size=n)
    df["target"] = raw_df["target"]
    state = run(make_state(df, raw_df), PipelineConfig())
    assert 0.0 <= state.conflict_score <= 1.0
