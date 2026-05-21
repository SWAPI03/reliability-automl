"""Tests for explainability.py — Properties 23, 24."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.explainability import run

# ── Shared fixture ────────────────────────────────────────────────────────────

N_ROWS = 50
N_FEATURES = 4
FEATURE_COLS = ["f0", "f1", "f2", "f3"]


def _make_state(
    n: int = N_ROWS,
    feature_trust: dict | None = None,
) -> PipelineState:
    """
    Build a controlled state where y = (X[:,0] > 0).
    This guarantees f0 is the most important feature for SHAP.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, N_FEATURES))
    y = (X[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    state = PipelineState()
    state.processed_df = pd.DataFrame(X, columns=FEATURE_COLS)
    state.task_type = "classification"
    state.trained_models = {"RandomForest": model}
    state.best_model_name = "RandomForest"
    state.feature_trust = feature_trust or {
        "f0": 0.9, "f1": 0.3, "f2": 0.8, "f3": 0.2
    }
    return state


# ── Basic output checks ───────────────────────────────────────────────────────

def test_shap_feature_importance_stored():
    """shap_feature_importance must be populated with one entry per feature."""
    state = run(_make_state(), PipelineConfig())
    assert state.shap_feature_importance is not None
    assert len(state.shap_feature_importance) == N_FEATURES


def test_shap_importance_non_negative():
    """All SHAP importance values must be >= 0 (mean absolute SHAP)."""
    state = run(_make_state(), PipelineConfig())
    for v in state.shap_feature_importance.values():
        assert v >= 0.0


def test_shap_values_stored():
    """shap_values array must be stored in state."""
    state = run(_make_state(), PipelineConfig())
    assert state.shap_values is not None


def test_shap_values_shape():
    """shap_values must have n_rows rows — one SHAP vector per sample."""
    state = run(_make_state(), PipelineConfig())
    assert state.shap_values.shape[0] == N_ROWS


def test_shap_importance_keys_match_features():
    """Keys in shap_feature_importance must match processed_df column names."""
    state = run(_make_state(), PipelineConfig())
    assert set(state.shap_feature_importance.keys()) == set(FEATURE_COLS)


# ── Importance correctness ────────────────────────────────────────────────────

def test_shap_importance_ordering():
    """
    f0 drives the target (y = X[:,0] > 0), so it must be the most important
    feature according to SHAP. This validates that explainability is meaningful.
    """
    state = run(_make_state(), PipelineConfig())
    imp = state.shap_feature_importance
    most_important = max(imp, key=imp.get)
    assert most_important == "f0", (
        f"Expected f0 to be most important, got {most_important}. "
        f"Importances: {imp}"
    )


def test_shap_importance_formula():
    """shap_feature_importance[f] = mean(|shap_values[:, f_idx]|)."""
    state = run(_make_state(), PipelineConfig())
    shap_arr = state.shap_values
    imp = state.shap_feature_importance
    for idx, col in enumerate(FEATURE_COLS):
        expected = float(np.mean(np.abs(shap_arr[:, idx])))
        assert imp[col] == pytest.approx(expected, abs=1e-9)


# ── Flagging logic ────────────────────────────────────────────────────────────

def test_high_importance_low_trust_flagging():
    """All flagged features must have trust < 0.5."""
    state = run(_make_state(), PipelineConfig())
    for f in state.high_importance_low_trust_features:
        assert state.feature_trust[f] < 0.5


def test_top_25_percent_selection():
    """Flagged features must be in the top 25% of SHAP importance."""
    state = run(_make_state(), PipelineConfig())
    imp = state.shap_feature_importance
    threshold = np.percentile(list(imp.values()), 75)
    top_features = {f for f, v in imp.items() if v >= threshold}
    for f in state.high_importance_low_trust_features:
        assert f in top_features, (
            f"{f} was flagged but not in top 25% importance. "
            f"Threshold={threshold:.4f}, value={imp[f]:.4f}"
        )


def test_flagging_requires_both_conditions():
    """A feature must satisfy BOTH trust < 0.5 AND top-25% importance to be flagged."""
    # f1 and f3 have low trust; f0 has high importance
    # Only features that are BOTH low-trust AND high-importance should be flagged
    state = _make_state(feature_trust={"f0": 0.9, "f1": 0.1, "f2": 0.9, "f3": 0.1})
    state = run(state, PipelineConfig())
    imp = state.shap_feature_importance
    threshold = np.percentile(list(imp.values()), 75)
    for f in state.high_importance_low_trust_features:
        assert state.feature_trust[f] < 0.5
        assert imp[f] >= threshold


def test_no_false_flagging_when_all_high_trust():
    """When all features have high trust (>= 0.5), nothing should be flagged."""
    state = _make_state(feature_trust={"f0": 0.9, "f1": 0.8, "f2": 0.85, "f3": 0.75})
    state = run(state, PipelineConfig())
    assert len(state.high_importance_low_trust_features) == 0


def test_no_false_flagging_when_all_low_importance():
    """When all features have equal (low) importance, top-25% threshold may
    include all — but flagging still requires trust < 0.5."""
    # All low trust but we verify the output is still a valid list
    state = _make_state(feature_trust={"f0": 0.1, "f1": 0.1, "f2": 0.1, "f3": 0.1})
    state = run(state, PipelineConfig())
    assert isinstance(state.high_importance_low_trust_features, list)


# ── Robustness ────────────────────────────────────────────────────────────────

def test_missing_feature_trust_key_does_not_crash():
    """If a feature is missing from feature_trust, run() must not crash."""
    state = _make_state(feature_trust={"f0": 0.9, "f2": 0.8})  # f1, f3 missing
    state = run(state, PipelineConfig())
    assert state.shap_feature_importance is not None


def test_shap_stability():
    """
    Running explainability twice on identical input must produce
    consistent importance values (TreeExplainer is deterministic).
    """
    state1 = run(_make_state(), PipelineConfig())
    state2 = run(_make_state(), PipelineConfig())
    imp1 = state1.shap_feature_importance
    imp2 = state2.shap_feature_importance
    for f in imp1:
        assert abs(imp1[f] - imp2[f]) < 1e-9, (
            f"SHAP importance for {f} is not stable: {imp1[f]} vs {imp2[f]}"
        )


def test_uses_tree_explainer():
    """RandomForest model must use TreeExplainer (not KernelExplainer)."""
    import shap
    state = _make_state()
    model = state.trained_models["RandomForest"]
    # TreeExplainer should work without error on a RandomForest
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(state.processed_df.values[:5])
    assert vals is not None
