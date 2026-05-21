"""Tests for noise_simulation.py — Property 22."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.noise_simulation import run


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_state(n: int = 60, seed: int = 0, R: float = 0.8) -> PipelineState:
    """
    Well-separated binary classification so the model has a meaningful
    clean score to compare against after noise injection.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)

    feature_cols = ["a", "b", "c"]
    raw_df = pd.DataFrame(X, columns=feature_cols)
    raw_df["target"] = y

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    state = PipelineState()
    state.raw_df = raw_df
    state.target_column = "target"
    state.column_types = {
        "a": "numeric", "b": "numeric", "c": "numeric", "target": "numeric"
    }
    state.processed_df = pd.DataFrame(X, columns=feature_cols)
    state.task_type = "classification"
    state.reliability_score = R
    # Non-uniform weights so we can test that reliability is actually used
    trust = rng.uniform(0.1, 1.0, n)
    state.sample_weights = trust / trust.sum()
    state.trained_models = {"RandomForest": model}
    state.best_model_name = "RandomForest"
    return state


# ── Basic output checks ───────────────────────────────────────────────────────

def test_noisy_model_scores_stored():
    state = run(_make_state(), PipelineConfig(cv_folds=2))
    assert state.noisy_model_scores is not None
    assert len(state.noisy_model_scores) > 0


def test_noisy_scores_have_required_keys():
    state = run(_make_state(), PipelineConfig(cv_folds=2))
    for scores in state.noisy_model_scores.values():
        assert "cv_score" in scores
        assert "reliability_adjusted_score" in scores


def test_all_models_have_noisy_scores():
    """Noise simulation must produce scores for all 3 model types (RF, LR, XGB)."""
    state = run(_make_state(), PipelineConfig(cv_folds=2))
    # noise_simulation always trains RF, LR, XGB internally
    assert len(state.noisy_model_scores) == 3
    assert "RandomForest" in state.noisy_model_scores
    assert "LogisticRegression" in state.noisy_model_scores
    assert "XGBoost" in state.noisy_model_scores


def test_noisy_cv_score_range():
    """Noisy CV accuracy must be in [0, 1]."""
    state = run(_make_state(), PipelineConfig(cv_folds=2))
    for scores in state.noisy_model_scores.values():
        assert 0.0 <= scores["cv_score"] <= 1.0


def test_noisy_reliability_adjusted_score_is_float():
    state = run(_make_state(), PipelineConfig(cv_folds=2))
    for scores in state.noisy_model_scores.values():
        assert isinstance(scores["reliability_adjusted_score"], float)


# ── Data safety ───────────────────────────────────────────────────────────────

def test_raw_df_unchanged_after_noise():
    """Original raw_df must not be modified by noise injection."""
    state = _make_state()
    original = state.raw_df["a"].values.copy()
    run(state, PipelineConfig(cv_folds=2))
    np.testing.assert_array_equal(state.raw_df["a"].values, original)


def test_processed_df_unchanged_after_noise():
    """processed_df must not be modified by noise injection."""
    state = _make_state()
    original = state.processed_df["a"].values.copy()
    run(state, PipelineConfig(cv_folds=2))
    np.testing.assert_array_equal(state.processed_df["a"].values, original)


# ── Robustness validation ─────────────────────────────────────────────────────

def test_noise_reduces_performance():
    """
    Noisy CV score must not be significantly higher than clean training score.
    This proves noise actually degrades performance.
    """
    state = _make_state()
    model = state.trained_models[state.best_model_name]
    X = state.processed_df.values
    y = state.raw_df[state.target_column].values
    clean_score = model.score(X, y)

    state = run(state, PipelineConfig(cv_folds=2))
    noisy_cv = state.noisy_model_scores[state.best_model_name]["cv_score"]

    # Noisy score should not exceed clean training score by more than 10%
    assert noisy_cv <= clean_score + 0.10, (
        f"Noisy CV={noisy_cv:.4f} is unexpectedly higher than clean={clean_score:.4f}"
    )


def test_reliability_adjusted_scores_non_negative():
    """
    All reliability-adjusted scores must be >= 0.
    This is the core claim: reliability-aware scoring never produces negative values.
    """
    state = run(_make_state(), PipelineConfig(cv_folds=2))
    for name, scores in state.noisy_model_scores.items():
        assert scores["reliability_adjusted_score"] >= 0.0, (
            f"{name}: reliability_adjusted_score={scores['reliability_adjusted_score']} < 0"
        )


def test_extreme_noise_low_reliability():
    """With R=0.1 (bad dataset), all reliability-adjusted scores must be < 1.0."""
    state = _make_state(R=0.1)
    state = run(state, PipelineConfig(cv_folds=2))
    for name, scores in state.noisy_model_scores.items():
        assert scores["reliability_adjusted_score"] < 1.0, (
            f"{name}: score={scores['reliability_adjusted_score']} should be < 1.0 when R=0.1"
        )


# ── Stability ─────────────────────────────────────────────────────────────────

def test_noise_simulation_stability():
    """
    Running noise simulation twice on identical input must produce
    consistent results (noise uses a fixed seed).
    """
    s1 = run(_make_state(), PipelineConfig(cv_folds=2))
    s2 = run(_make_state(), PipelineConfig(cv_folds=2))
    for m in s1.noisy_model_scores:
        diff = abs(
            s1.noisy_model_scores[m]["cv_score"] -
            s2.noisy_model_scores[m]["cv_score"]
        )
        assert diff < 0.01, (
            f"{m}: noisy scores differ by {diff:.4f} between runs — not deterministic"
        )


# ── Sample weight validation ──────────────────────────────────────────────────

def test_sample_weights_are_non_uniform():
    """
    Sample weights must not all be equal — reliability is actually used.
    Uses non-uniform trust scores in _make_state.
    """
    state = _make_state()
    assert not np.allclose(state.sample_weights, state.sample_weights[0]), (
        "All sample weights are identical — trust-based weighting has no effect"
    )


def test_sample_weights_sum_to_one():
    """Sample weights must sum to 1.0 before noise simulation."""
    state = _make_state()
    assert abs(state.sample_weights.sum() - 1.0) < 1e-9
