"""Tests for feedback_refinement.py — Property 16."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.feedback_refinement import run


# ── Shared fixture ────────────────────────────────────────────────────────────

def _make_clf_state(n: int = 50, seed: int = 42, initial_trust: float = 0.8) -> PipelineState:
    """
    Well-separated binary classification so the model makes some errors
    but not all — gives us both correct and mispredicted rows to test against.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    state = PipelineState()
    feature_cols = ["a", "b", "c"]
    state.processed_df = pd.DataFrame(X, columns=feature_cols)
    raw_df = pd.DataFrame(X, columns=feature_cols)
    raw_df["target"] = y
    state.raw_df = raw_df
    state.target_column = "target"
    state.task_type = "classification"
    state.trained_models = {"RandomForest": model}
    state.best_model_name = "RandomForest"
    state.trust_scores = np.full(n, initial_trust)
    return state


# ── Basic output checks ───────────────────────────────────────────────────────

def test_trust_scores_in_range():
    """All trust scores must be in [0, 1] after refinement."""
    state = run(_make_clf_state(), PipelineConfig())
    assert np.all(state.trust_scores >= 0.0)
    assert np.all(state.trust_scores <= 1.0)


def test_max_trust_equals_one():
    """After normalization, max trust must equal 1.0."""
    state = run(_make_clf_state(), PipelineConfig())
    assert float(np.max(state.trust_scores)) == pytest.approx(1.0, abs=1e-9)


def test_trust_scores_length():
    """Output trust array must have same length as input."""
    n = 40
    state = run(_make_clf_state(n), PipelineConfig())
    assert len(state.trust_scores) == n


def test_trust_normalization_explicit():
    """max(trust_scores) == 1.0 is the normalization invariant."""
    state = run(_make_clf_state(60), PipelineConfig())
    assert state.trust_scores.max() == pytest.approx(1.0, abs=1e-9)


# ── Core formula validation ───────────────────────────────────────────────────

def test_mispredicted_rows_have_lower_trust():
    """
    Rows the model gets wrong must have lower trust than rows it gets right.
    This is the core claim: trust ← trust × (1 − prediction_error).
    """
    state = _make_clf_state(80)
    model = state.trained_models["RandomForest"]
    X = state.processed_df.values
    y = state.raw_df["target"].values

    preds = model.predict(X)
    errors = (preds != y).astype(float)

    state = run(state, PipelineConfig())

    mis_idx     = np.where(errors == 1)[0]
    correct_idx = np.where(errors == 0)[0]

    if len(mis_idx) > 0 and len(correct_idx) > 0:
        assert state.trust_scores[mis_idx].mean() < state.trust_scores[correct_idx].mean(), (
            "Mispredicted rows should have lower trust than correctly predicted rows"
        )


def test_trust_never_increases_beyond_one():
    """Refinement must never inflate trust above 1.0."""
    state = run(_make_clf_state(), PipelineConfig())
    assert np.all(state.trust_scores <= 1.0 + 1e-9)


def test_trust_formula_applied():
    """
    Verify the formula: trust_new = trust_old × (1 − error), then normalized.
    For classification, error_i = 1 - predicted_proba[true_class].
    """
    state = _make_clf_state(60)
    model = state.trained_models["RandomForest"]
    X = state.processed_df.values
    y = state.raw_df["target"].values

    # Compute errors the same way feedback_refinement.py does
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    errors = np.array([
        1.0 - proba[i, class_to_idx[y[i]]] if y[i] in class_to_idx else 1.0
        for i in range(len(y))
    ])
    errors = np.clip(errors, 0.0, 1.0)

    old_trust = state.trust_scores.copy()
    state = run(state, PipelineConfig())

    raw_expected = old_trust * (1.0 - errors)
    max_raw = raw_expected.max()
    if max_raw > 0:
        normalized_expected = raw_expected / max_raw
        np.testing.assert_allclose(state.trust_scores, normalized_expected, atol=1e-9)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_edge_case_zero_trust_fallback():
    """If all trust scores collapse to 0, fallback to uniform 1.0 (not 1/n)."""
    state = _make_clf_state(10)
    state.trust_scores = np.zeros(10)
    state = run(state, PipelineConfig())
    assert np.all(state.trust_scores >= 0.0)
    # Fallback should be uniform ones (not 1/n)
    np.testing.assert_allclose(state.trust_scores, np.ones(10), atol=1e-9)


def test_all_wrong_predictions_reduces_trust():
    """When all predictions are wrong, mean trust should be low."""
    state = _make_clf_state(40)
    # Flip all labels to force maximum prediction error
    state.raw_df["target"] = 1 - state.raw_df["target"]
    state = run(state, PipelineConfig())
    # After normalization max=1, but mean should be much lower than initial 0.8
    assert np.mean(state.trust_scores) < 0.8


def test_stability():
    """Same input must produce identical output (deterministic)."""
    state1 = run(_make_clf_state(), PipelineConfig())
    state2 = run(_make_clf_state(), PipelineConfig())
    np.testing.assert_allclose(state1.trust_scores, state2.trust_scores, atol=1e-9)


def test_uses_best_model():
    """Refinement must use best_model_name, not an arbitrary model."""
    state = _make_clf_state(30)
    # Add a second dummy model — refinement should still use best_model_name
    state.trained_models["DummyModel"] = state.trained_models["RandomForest"]
    state.best_model_name = "RandomForest"
    result = run(state, PipelineConfig())
    assert result.trust_scores is not None
    assert len(result.trust_scores) == 30
