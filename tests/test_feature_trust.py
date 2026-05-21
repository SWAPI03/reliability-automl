"""Tests for feature_trust.py — Properties 17, 18."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.feature_trust import run


def make_state(
    raw_df: pd.DataFrame,
    target: str,
    trust_scores: np.ndarray | None,
) -> PipelineState:
    state = PipelineState()
    state.raw_df = raw_df
    state.target_column = target
    state.trust_scores = trust_scores
    return state


# ── Property 17: Feature Trust Formula ───────────────────────────────────────

def test_feature_trust_formula():
    """feature_trust[f] = mean(trust_scores[non-NaN rows])."""
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "target": [0, 1, 0, 1]})
    trust = np.array([0.8, 0.6, 0.4, 0.9])
    state = run(make_state(df, "target", trust), PipelineConfig())
    expected = np.mean([0.8, 0.6, 0.9])  # rows 0, 1, 3 are non-NaN
    assert state.feature_trust["a"] == pytest.approx(expected, abs=1e-9)


def test_feature_trust_all_nan():
    """All-NaN feature → feature_trust = 0.0."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "target": [0, 1, 0]})
    trust = np.array([0.5, 0.7, 0.3])
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert state.feature_trust["a"] == 0.0


def test_feature_trust_no_missing():
    """No missing values → feature_trust = mean of all trust scores."""
    trust = np.array([0.4, 0.6, 0.8, 1.0])
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert state.feature_trust["a"] == pytest.approx(float(trust.mean()), abs=1e-9)


def test_feature_trust_range():
    """All feature trust values must be in [0, 1]."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": range(10), "b": range(10), "target": range(10)})
    trust = rng.uniform(0, 1, 10)
    state = run(make_state(df, "target", trust), PipelineConfig())
    for v in state.feature_trust.values():
        assert 0.0 <= v <= 1.0


def test_multiple_features():
    """All feature columns must appear in feature_trust."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [np.nan, 5.0, 6.0, 7.0],
        "target": [0, 1, 0, 1],
    })
    trust = np.array([0.8, 0.6, 0.4, 0.9])
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert "a" in state.feature_trust
    assert "b" in state.feature_trust
    # 'a' non-NaN rows: 0,1,3 → mean(0.8,0.6,0.9)
    assert state.feature_trust["a"] == pytest.approx(np.mean([0.8, 0.6, 0.9]), abs=1e-9)
    # 'b' non-NaN rows: 1,2,3 → mean(0.6,0.4,0.9)
    assert state.feature_trust["b"] == pytest.approx(np.mean([0.6, 0.4, 0.9]), abs=1e-9)


def test_target_excluded_from_feature_trust():
    """Target column must not appear in feature_trust."""
    df = pd.DataFrame({"a": range(5), "target": range(5)})
    trust = np.random.default_rng(0).uniform(0, 1, 5)
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert "target" not in state.feature_trust


def test_constant_feature():
    """Constant feature (no variance) — trust is still computed correctly."""
    df = pd.DataFrame({"a": [1, 1, 1, 1], "target": [0, 1, 0, 1]})
    trust = np.array([0.5, 0.6, 0.7, 0.8])
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert state.feature_trust["a"] == pytest.approx(float(trust.mean()), abs=1e-9)


def test_feature_trust_sensitivity():
    """Higher trust scores must produce higher feature trust."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    trust_high = np.array([0.9, 0.9, 0.9, 0.9])
    trust_low  = np.array([0.1, 0.1, 0.1, 0.1])
    state_high = run(make_state(df, "target", trust_high), PipelineConfig())
    state_low  = run(make_state(df, "target", trust_low),  PipelineConfig())
    assert state_high.feature_trust["a"] > state_low.feature_trust["a"]


def test_trust_length_mismatch_raises():
    """Trust array length != n_rows must raise an error."""
    df = pd.DataFrame({"a": range(5), "target": range(5)})
    trust = np.random.rand(3)  # wrong length
    with pytest.raises(Exception):
        run(make_state(df, "target", trust), PipelineConfig())


# ── Property 18: Confidence Score ────────────────────────────────────────────

def test_confidence_score_zero_variance():
    """Uniform trust → zero variance → confidence = 1/(1+0) = 1.0."""
    trust = np.array([0.8, 0.8, 0.8, 0.8])
    df = pd.DataFrame({"a": range(4), "target": range(4)})
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert state.confidence_score == pytest.approx(1.0, abs=1e-9)


def test_confidence_score_high_variance():
    """Alternating 0/1 trust → high variance → confidence < 1.0."""
    trust = np.array([0.0, 1.0, 0.0, 1.0])
    df = pd.DataFrame({"a": range(4), "target": range(4)})
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert state.confidence_score < 1.0


def test_confidence_score_formula():
    """confidence = 1 / (1 + variance(trust))."""
    trust = np.array([0.2, 0.4, 0.6, 0.8])
    df = pd.DataFrame({"a": range(4), "target": range(4)})
    state = run(make_state(df, "target", trust), PipelineConfig())
    expected = 1.0 / (1.0 + float(np.var(trust)))
    assert state.confidence_score == pytest.approx(expected, abs=1e-9)


def test_confidence_score_range():
    """Confidence score must always be in (0, 1]."""
    rng = np.random.default_rng(5)
    trust = rng.uniform(0, 1, 20)
    df = pd.DataFrame({"a": range(20), "target": range(20)})
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert 0.0 < state.confidence_score <= 1.0


def test_confidence_clipping_extreme_values():
    """Extreme trust values must not produce confidence outside (0, 1]."""
    trust = np.array([100.0, -100.0, 100.0, -100.0])
    df = pd.DataFrame({"a": range(4), "target": range(4)})
    state = run(make_state(df, "target", trust), PipelineConfig())
    assert 0.0 < state.confidence_score <= 1.0


def test_confidence_decreases_with_variance():
    """More variance in trust → lower confidence."""
    df = pd.DataFrame({"a": range(4), "target": range(4)})
    trust_stable = np.array([0.7, 0.75, 0.72, 0.73])   # low variance
    trust_noisy  = np.array([0.1, 0.9, 0.1, 0.9])       # high variance
    state_stable = run(make_state(df, "target", trust_stable), PipelineConfig())
    state_noisy  = run(make_state(df, "target", trust_noisy),  PipelineConfig())
    assert state_stable.confidence_score > state_noisy.confidence_score
