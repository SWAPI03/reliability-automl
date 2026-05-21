"""Tests for statistical_score.py — Property 6: Statistical Score Formula Correctness."""
import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.statistical_score import run


def make_state(df: pd.DataFrame, target: str = "target") -> PipelineState:
    state = PipelineState()
    state.processed_df = df
    state.target_column = target
    return state


# ── Formula tests ──────────────────────────────────────────────────────────────

def test_score_range():
    """Score must always be in [0, 1]."""
    df = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50), "target": np.zeros(50)})
    state = run(make_state(df), PipelineConfig())
    assert 0.0 <= state.statistical_score <= 1.0


def test_no_outliers_no_low_variance():
    """Clean data with good variance → T close to 1."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "target": np.zeros(n),
    })
    state = run(make_state(df), PipelineConfig())
    # No outliers expected, no low-variance cols → T should be high
    assert state.statistical_score > 0.5


def test_all_constant_columns_low_variance():
    """All feature columns constant → low_variance_ratio=1 → T = 1 - 0.4 = 0.6."""
    n = 50
    df = pd.DataFrame({
        "a": np.ones(n),
        "b": np.ones(n) * 2,
        "target": np.zeros(n),
    })
    state = run(make_state(df), PipelineConfig())
    # low_variance_ratio = 1.0, outlier_ratio depends on z-score of constant cols
    # T = 1 - (0.6*outlier_ratio + 0.4*1.0) ≤ 0.6
    assert state.statistical_score <= 0.6 + 1e-9


def test_outlier_mask_shape():
    """outlier_mask must have same length as dataframe."""
    n = 30
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "target": np.zeros(n)})
    state = run(make_state(df), PipelineConfig())
    assert state.outlier_mask is not None
    assert len(state.outlier_mask) == n


def test_outlier_mask_dtype():
    """outlier_mask must be boolean."""
    n = 20
    df = pd.DataFrame({"x": np.random.randn(n), "target": np.zeros(n)})
    state = run(make_state(df), PipelineConfig())
    assert state.outlier_mask.dtype == bool


# ── Z-score branch (< 1000 rows) ───────────────────────────────────────────────

def test_zscore_branch_used_for_small_dataset():
    """For n < 1000, outliers are detected via Z-score."""
    rng = np.random.default_rng(1)
    n = 50
    data = rng.normal(0, 1, n).tolist()
    data[0] = 100.0  # extreme outlier
    df = pd.DataFrame({"a": data, "target": np.zeros(n)})
    state = run(make_state(df), PipelineConfig(outlier_zscore_threshold=3.0))
    # Row 0 should be flagged
    assert state.outlier_mask[0] is True or state.outlier_mask[0] == True  # noqa: E712


def test_zscore_no_outliers():
    """Tightly clustered data → no outliers detected."""
    n = 100
    df = pd.DataFrame({"a": np.ones(n) * 5.0, "b": np.ones(n) * 3.0, "target": np.zeros(n)})
    state = run(make_state(df), PipelineConfig(outlier_zscore_threshold=3.0))
    # All values identical → z-score = 0 → no outliers
    assert state.outlier_mask.sum() == 0


# ── IsolationForest branch (≥ 1000 rows) ───────────────────────────────────────

def test_isolation_forest_branch_used_for_large_dataset():
    """For n >= 1000, IsolationForest is used (smoke test)."""
    rng = np.random.default_rng(2)
    n = 1000
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "target": np.zeros(n),
    })
    state = run(make_state(df), PipelineConfig())
    assert state.outlier_mask is not None
    assert len(state.outlier_mask) == n
    assert state.statistical_score is not None


def test_isolation_forest_score_range():
    """Score from IsolationForest path must be in [0, 1]."""
    rng = np.random.default_rng(3)
    n = 1200
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(5, 2, n),
        "target": np.zeros(n),
    })
    state = run(make_state(df), PipelineConfig())
    assert 0.0 <= state.statistical_score <= 1.0


# ── Formula verification ────────────────────────────────────────────────────────

def test_formula_with_known_ratios():
    """Manually inject known outlier/variance ratios and verify T formula."""
    # Build data where we know exactly: outlier_ratio=0.1, low_variance_ratio=0.5
    # Use 20 rows, 2 cols; col 'a' has 2 extreme outliers (10%), col 'b' is constant
    n = 20
    a = np.zeros(n)
    a[0] = 1000.0
    a[1] = -1000.0
    b = np.ones(n)  # constant → low variance
    df = pd.DataFrame({"a": a, "b": b, "target": np.zeros(n)})
    state = run(make_state(df), PipelineConfig(outlier_zscore_threshold=3.0))

    outlier_ratio = state.outlier_mask.sum() / n
    # low_variance_ratio: b is constant (var=0 < 0.01), a has high variance
    low_variance_ratio = 0.5  # 1 out of 2 cols

    expected_T = 1.0 - (0.6 * outlier_ratio + 0.4 * low_variance_ratio)
    expected_T = max(0.0, min(1.0, expected_T))
    assert state.statistical_score == pytest.approx(expected_T, abs=1e-9)
