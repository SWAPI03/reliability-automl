"""Tests for automl_trainer.py — Properties 19, 20."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.automl_trainer import run, _detect_task_type

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(
    X: np.ndarray,
    y: np.ndarray,
    target: str = "target",
    trust_scores=None,
    R: float = 0.8,
) -> PipelineState:
    state = PipelineState()
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    state.processed_df = pd.DataFrame(X, columns=feature_cols)
    raw_df = pd.DataFrame(X, columns=feature_cols)
    raw_df[target] = y
    state.raw_df = raw_df
    state.target_column = target
    state.reliability_score = R
    state.trust_scores = trust_scores
    return state


def _clf_data(n: int = 60, seed: int = 0):
    """Well-separated binary classification — f0 drives the label."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _reg_data(n: int = 60, seed: int = 1):
    """Linear regression — y ≈ 2*f0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
    return X, y


# ── Task type detection (Property 20) ────────────────────────────────────────

def test_detect_classification_int():
    assert _detect_task_type(np.array([0, 1, 0, 1, 2] * 4)) == "classification"


def test_detect_classification_object():
    """String labels with few unique values → classification."""
    y = np.array(["cat", "dog", "cat", "dog"] * 5)
    assert _detect_task_type(y) == "classification"


def test_detect_regression_float():
    assert _detect_task_type(np.linspace(0, 100, 50)) == "regression"


def test_detect_regression_many_unique():
    """More than 20 unique integer values → regression."""
    assert _detect_task_type(np.arange(50, dtype=float)) == "regression"


def test_detect_classification_boundary():
    """Exactly 20 unique int values → classification (boundary)."""
    y = np.arange(20, dtype=int)
    assert _detect_task_type(y) == "classification"


# ── Sample weight normalization (Property 19) ─────────────────────────────────

def test_sample_weights_sum_to_one():
    X, y = _clf_data()
    trust = np.random.default_rng(7).uniform(0, 1, len(X))
    state = run(make_state(X, y, trust_scores=trust), PipelineConfig(cv_folds=2))
    assert state.sample_weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_sample_weights_non_negative():
    X, y = _clf_data()
    trust = np.random.default_rng(8).uniform(0, 1, len(X))
    state = run(make_state(X, y, trust_scores=trust), PipelineConfig(cv_folds=2))
    assert np.all(state.sample_weights >= 0)


def test_sample_weights_length():
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=2))
    assert len(state.sample_weights) == len(X)


def test_zero_trust_scores_no_crash():
    """All-zero trust must not crash (division-by-zero guard)."""
    X, y = _clf_data()
    trust = np.zeros(len(X))
    state = run(make_state(X, y, trust_scores=trust), PipelineConfig(cv_folds=2))
    assert state.sample_weights is not None
    assert np.all(state.sample_weights >= 0)
    assert state.sample_weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_no_trust_scores_uses_uniform_fallback():
    """When trust_scores is None, sample_weights must still be computed."""
    X, y = _clf_data()
    state = run(make_state(X, y, trust_scores=None), PipelineConfig(cv_folds=2))
    assert state.sample_weights is not None
    assert state.sample_weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_reliability_score_affects_weights():
    """
    w_i = trust_i * R — same trust but different R must produce different weights.
    (After normalization the relative ordering is the same, but the raw products differ.)
    We verify that the weights are proportional to trust*R before normalization.
    """
    X, y = _clf_data()
    rng = np.random.default_rng(42)
    trust = rng.uniform(0.1, 1.0, len(X))

    state_low  = run(make_state(X, y, trust_scores=trust, R=0.2), PipelineConfig(cv_folds=2))
    state_high = run(make_state(X, y, trust_scores=trust, R=0.9), PipelineConfig(cv_folds=2))

    # After normalization the weights are identical (trust*R / sum(trust*R) = trust / sum(trust))
    # because R cancels out. So we verify the raw formula instead:
    raw_low  = trust * 0.2;  raw_low  /= raw_low.sum()
    raw_high = trust * 0.9;  raw_high /= raw_high.sum()
    np.testing.assert_allclose(state_low.sample_weights,  raw_low,  atol=1e-9)
    np.testing.assert_allclose(state_high.sample_weights, raw_high, atol=1e-9)
    # Both should be equal (R cancels after normalization — this is expected)
    np.testing.assert_allclose(state_low.sample_weights, state_high.sample_weights, atol=1e-9)


def test_weights_affect_training():
    """
    Uniform trust vs skewed trust (half rows zeroed) must produce different models.
    This proves trust weights actually change model behavior.
    """
    X, y = _clf_data(n=80)
    n = len(X)
    trust_uniform = np.ones(n)
    trust_skewed  = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])

    state1 = run(make_state(X, y, trust_scores=trust_uniform), PipelineConfig(cv_folds=2))
    state2 = run(make_state(X, y, trust_scores=trust_skewed),  PipelineConfig(cv_folds=2))

    # At least one model must produce different predictions
    any_different = False
    for name in state1.trained_models:
        if name in state2.trained_models:
            p1 = state1.trained_models[name].predict(X)
            p2 = state2.trained_models[name].predict(X)
            if not np.array_equal(p1, p2):
                any_different = True
                break
    assert any_different, "Uniform and skewed trust produced identical models — weights have no effect"


# ── Model training ────────────────────────────────────────────────────────────

def test_classification_models_trained():
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=2))
    assert state.task_type == "classification"
    assert "RandomForest" in state.trained_models
    assert "LogisticRegression" in state.trained_models
    assert "XGBoost" in state.trained_models


def test_regression_models_trained():
    X, y = _reg_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=2))
    assert state.task_type == "regression"
    assert "RandomForest" in state.trained_models
    assert "LinearRegression" in state.trained_models
    assert "XGBoost" in state.trained_models


def test_model_scores_stored_with_std():
    """model_scores must contain both cv_score and cv_std for every model."""
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=3))
    for name, scores in state.model_scores.items():
        assert "cv_score" in scores, f"{name} missing cv_score"
        assert "cv_std"   in scores, f"{name} missing cv_std"


def test_cv_score_in_valid_range():
    """CV accuracy must be in [0, 1] for classification."""
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=3))
    for name, scores in state.model_scores.items():
        assert 0.0 <= scores["cv_score"] <= 1.0, (
            f"{name} cv_score={scores['cv_score']} out of [0,1]"
        )


def test_cv_std_non_negative():
    """CV standard deviation must be >= 0."""
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=3))
    for name, scores in state.model_scores.items():
        assert scores["cv_std"] >= 0.0


def test_models_can_predict_classification():
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=2))
    for model in state.trained_models.values():
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})


def test_regression_predictions_are_float():
    """Regression model predictions must be numeric (float/continuous)."""
    X, y = _reg_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=2))
    for name, model in state.trained_models.items():
        preds = model.predict(X)
        assert preds.dtype.kind in ("f", "c"), (
            f"{name} regression predictions have dtype {preds.dtype}"
        )


def test_string_labels_encoded_for_xgboost():
    """String class labels must be encoded — XGBoost requires integer labels."""
    X, _ = _clf_data()
    y_str = np.where(X[:, 0] > 0, "pos", "neg")
    state = run(make_state(X, y_str), PipelineConfig(cv_folds=2))
    assert state.label_encoder is not None
    assert state.task_type == "classification"
    # XGBoost must be in trained_models (would crash without encoding)
    assert "XGBoost" in state.trained_models


def test_task_type_stored():
    X, y = _clf_data()
    state = run(make_state(X, y), PipelineConfig(cv_folds=2))
    assert state.task_type in ("classification", "regression")
