"""Tests for deduction_score.py — Properties 10, 11."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.deduction_score import run

# Fixed seed for all random data generation — ensures reproducible clustering
RNG = np.random.default_rng(42)


def make_state(
    df: pd.DataFrame,
    target: str = "target",
    trust_scores=None,
    raw_df=None,
) -> PipelineState:
    state = PipelineState()
    state.processed_df = df
    state.target_column = target
    state.trust_scores = trust_scores
    state.raw_df = raw_df  # needed for bootstrap path
    return state


def _df(n: int, n_features: int = 2, seed: int = 0) -> pd.DataFrame:
    """Helper: deterministic feature DataFrame with a target column."""
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(size=n) for i in range(n_features)}
    data["target"] = np.zeros(n)
    return pd.DataFrame(data)


# ── Property 10: Cluster assignment completeness ──────────────────────────────

def test_cluster_labels_length():
    """cluster_labels must have length == n_rows."""
    n = 30
    df = _df(n)
    state = run(make_state(df), PipelineConfig(n_pseudo_sources=5))
    assert len(state.cluster_labels) == n


def test_cluster_labels_range():
    """Every label must be in [0, n_clusters-1]."""
    n, n_clusters = 50, 5
    df = _df(n)
    state = run(make_state(df), PipelineConfig(n_pseudo_sources=n_clusters))
    assert state.cluster_labels.min() >= 0
    assert state.cluster_labels.max() < n_clusters


def test_every_row_assigned():
    """No row should be unassigned (no -1 or NaN labels)."""
    n = 40
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "target": np.zeros(n)})
    state = run(make_state(df), PipelineConfig(n_pseudo_sources=4))
    assert not np.any(np.isnan(state.cluster_labels.astype(float)))
    assert np.all(state.cluster_labels >= 0)


def test_cluster_labels_dtype():
    """cluster_labels should be integer dtype."""
    df = _df(20)
    state = run(make_state(df), PipelineConfig(n_pseudo_sources=3))
    assert np.issubdtype(state.cluster_labels.dtype, np.integer)


def test_single_cluster():
    """With n_pseudo_sources=1, all labels must be 0 and D == trust.mean()."""
    n = 30
    df = _df(n, seed=7)
    trust = RNG.uniform(0, 1, size=n)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=1))
    assert np.all(state.cluster_labels == 0)
    assert state.deduction_score == pytest.approx(float(trust.mean()), abs=1e-9)


def test_n_clusters_capped_at_n_rows():
    """n_pseudo_sources > n_rows should not crash (capped to n_rows)."""
    n = 3
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [0, 0, 0]})
    state = run(make_state(df), PipelineConfig(n_pseudo_sources=10))
    assert state.deduction_score is not None
    assert len(state.cluster_labels) == n


# ── Property 11: Deduction score formula ─────────────────────────────────────

def test_deduction_score_uniform_trust():
    """With uniform trust=0.5, D should equal 0.5."""
    n = 30
    df = _df(n)
    trust = np.full(n, 0.5)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=5))
    assert state.deduction_score == pytest.approx(0.5, abs=1e-9)


def test_deduction_score_all_zero_trust():
    """With all-zero trust, D should be 0.0 (boundary condition)."""
    n = 20
    df = _df(n)
    trust = np.zeros(n)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=3))
    assert state.deduction_score == pytest.approx(0.0, abs=1e-9)


def test_deduction_score_all_one_trust():
    """With all-one trust, D should be 1.0 (boundary condition)."""
    n = 20
    df = _df(n)
    trust = np.ones(n)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=3))
    assert state.deduction_score == pytest.approx(1.0, abs=1e-9)


def test_deduction_score_range():
    """D must always be in [0, 1]."""
    n = 40
    df = _df(n)
    trust = RNG.uniform(0, 1, size=n)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=5))
    assert 0.0 <= state.deduction_score <= 1.0


def test_deduction_score_formula():
    """D = mean(pseudo_source_trust[cluster_labels[i]]) across all rows."""
    n = 30
    df = _df(n, seed=42)
    trust = RNG.uniform(0, 1, size=n)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=3))

    labels = state.cluster_labels
    n_clusters = labels.max() + 1
    pseudo_trust = np.array([trust[labels == k].mean() for k in range(n_clusters)])
    expected_D = float(np.clip(pseudo_trust[labels].mean(), 0, 1))

    assert state.deduction_score == pytest.approx(expected_D, abs=1e-9)


def test_deduction_score_bootstrap_no_raw_df():
    """When trust_scores is None and raw_df is None, bootstrap uses 0.5 → D = 0.5."""
    n = 20
    df = _df(n)
    # raw_df=None forces the uniform 0.5 bootstrap path
    state = run(make_state(df, trust_scores=None, raw_df=None), PipelineConfig(n_pseudo_sources=3))
    assert state.deduction_score == pytest.approx(0.5, abs=1e-9)


def test_deduction_score_bootstrap_uses_structural_quality():
    """When trust_scores is None but raw_df has missing values,
    bootstrap uses per-row missing ratio — a fully-missing dataset should
    produce lower D than a fully-clean dataset."""
    n = 20
    df = _df(n)

    # All rows missing in all feature columns (a, b) + target = 3 cols total
    # missing_ratio per row = 2/3 → trust = clip(1 - 2/3, 0.1, 1.0) = 0.333
    raw_all_missing = pd.DataFrame({
        "a": [np.nan] * n,
        "b": [np.nan] * n,
        "target": np.zeros(n),
    })
    # No raw_df → uniform bootstrap trust = 0.5 → D = 0.5
    state_missing = run(make_state(df, trust_scores=None, raw_df=raw_all_missing), PipelineConfig(n_pseudo_sources=2))
    state_clean   = run(make_state(df, trust_scores=None, raw_df=None),            PipelineConfig(n_pseudo_sources=2))

    # missing_ratio = 2/3 → trust = 1 - 2/3 = 0.333 per row → D = 0.333
    expected_trust = 1.0 - (2 / 3)
    assert state_missing.deduction_score == pytest.approx(expected_trust, abs=1e-9)
    # No raw_df gives uniform 0.5 → D = 0.5
    assert state_clean.deduction_score == pytest.approx(0.5, abs=1e-9)
    # Missing dataset scores lower
    assert state_missing.deduction_score < state_clean.deduction_score


# ── Stability and sensitivity tests ──────────────────────────────────────────

def test_deduction_score_sensitivity_to_noise():
    """System with noisy trust (half rows = 0) should score lower than clean trust."""
    n = 50
    df = _df(n, seed=5)
    trust_clean = np.ones(n) * 0.8
    trust_noisy = np.concatenate([np.ones(n // 2) * 0.8, np.zeros(n // 2)])

    state_clean = run(make_state(df, trust_scores=trust_clean), PipelineConfig(n_pseudo_sources=3))
    state_noisy = run(make_state(df, trust_scores=trust_noisy), PipelineConfig(n_pseudo_sources=3))

    assert state_noisy.deduction_score < state_clean.deduction_score


def test_imbalanced_clusters_stable():
    """Highly imbalanced clusters (45 vs 5 rows) should not crash and D in [0,1]."""
    n = 50
    df = pd.DataFrame({
        "a": np.concatenate([np.zeros(45), np.ones(5)]),
        "target": np.zeros(n),
    })
    trust = RNG.uniform(0, 1, size=n)
    state = run(make_state(df, trust_scores=trust), PipelineConfig(n_pseudo_sources=2))
    assert 0.0 <= state.deduction_score <= 1.0


def test_no_mutation_of_input():
    """run() must not modify the input DataFrame."""
    df = _df(10)
    df_copy = df.copy()
    run(make_state(df), PipelineConfig(n_pseudo_sources=3))
    pd.testing.assert_frame_equal(df, df_copy)


def test_deterministic_with_same_seed():
    """Same data and config must always produce the same cluster labels."""
    df = _df(30, seed=99)
    trust = np.linspace(0, 1, 30)
    config = PipelineConfig(n_pseudo_sources=4)

    state1 = run(make_state(df, trust_scores=trust), config)
    state2 = run(make_state(df, trust_scores=trust), config)

    np.testing.assert_array_equal(state1.cluster_labels, state2.cluster_labels)
    assert state1.deduction_score == pytest.approx(state2.deduction_score, abs=1e-12)
