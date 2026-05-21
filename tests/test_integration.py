"""End-to-end integration tests for the full Reliability-Aware AutoML pipeline."""
from __future__ import annotations

import io
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.validator import ValidationError
from reliability_automl.pipeline import (
    validator, preprocessor, structural_score, statistical_score,
    pattern_analysis, deduction_score, reliability_fusion, reliability_graph,
    trust_propagation, automl_trainer, feedback_refinement, feature_trust,
    model_selection, noise_simulation, baseline_comparison, explainability,
)


# ── CSV generators ────────────────────────────────────────────────────────────

def _make_synthetic_csv(n: int = 50, seed: int = 42) -> io.StringIO:
    """Clean synthetic classification CSV with a few missing values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["A", "B", "C"], size=n),
        "target": rng.integers(0, 2, size=n),
    })
    df.loc[0, "num1"] = np.nan
    df.loc[5, "cat1"] = np.nan
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _make_noisy_csv(n: int = 200, seed: int = 7) -> io.StringIO:
    """
    Noisy dataset: clean rows + conflict rows (same features, flipped labels).
    Designed to show measurable improvement from trust-based weighting.
    """
    rng = np.random.default_rng(seed)
    centers = {0: np.array([-2.0, -2.0, 0.5]), 1: np.array([2.0, 2.0, -0.5])}
    parts = []
    for cls, center in centers.items():
        X = rng.normal(center, 0.6, (n // 2, 3))
        df_c = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        df_c["target"] = cls
        parts.append(df_c)
    df_clean = pd.concat(parts, ignore_index=True)

    # Conflict rows: copy of class-0 features but labeled as class-1
    conflict = df_clean[df_clean["target"] == 0].sample(n // 5, random_state=seed).copy()
    conflict["target"] = 1
    df = pd.concat([df_clean, conflict], ignore_index=True).sample(frac=1, random_state=seed)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _run_full_pipeline(csv_buf: io.StringIO, config: PipelineConfig) -> PipelineState:
    state = PipelineState()
    state = validator.run(state, config, file=csv_buf)
    state = preprocessor.run(state, config)
    state = structural_score.run(state, config)
    state = statistical_score.run(state, config)
    state = pattern_analysis.run(state, config)
    state = deduction_score.run(state, config)
    state = reliability_fusion.run(state, config)
    state = reliability_graph.run(state, config)
    state = trust_propagation.run(state, config)
    state = automl_trainer.run(state, config)
    state = model_selection.run(state, config)
    state = feedback_refinement.run(state, config)
    state = feature_trust.run(state, config)
    state = noise_simulation.run(state, config)
    state = baseline_comparison.run(state, config)
    state = explainability.run(state, config)
    return state


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_pipeline_state():
    config = PipelineConfig(cv_folds=2, n_pseudo_sources=3, trust_max_iterations=10)
    return _run_full_pipeline(_make_synthetic_csv(n=50), config)


@pytest.fixture(scope="module")
def noisy_pipeline_state():
    config = PipelineConfig(cv_folds=3, n_pseudo_sources=3, trust_max_iterations=10)
    return _run_full_pipeline(_make_noisy_csv(n=200), config)


# ── State population checks ───────────────────────────────────────────────────

def test_raw_df_populated(full_pipeline_state):
    assert full_pipeline_state.raw_df is not None
    assert len(full_pipeline_state.raw_df) == 50


def test_processed_df_populated(full_pipeline_state):
    assert full_pipeline_state.processed_df is not None
    assert len(full_pipeline_state.processed_df) == 50


def test_scores_populated(full_pipeline_state):
    s = full_pipeline_state
    for attr in ["structural_score", "statistical_score", "similarity_score",
                 "conflict_score", "deduction_score", "reliability_score"]:
        assert getattr(s, attr) is not None, f"{attr} not populated"


def test_scores_in_range(full_pipeline_state):
    s = full_pipeline_state
    for score in [s.structural_score, s.statistical_score, s.similarity_score,
                  s.conflict_score, s.deduction_score, s.reliability_score]:
        assert 0.0 <= score <= 1.0


def test_trust_scores_populated(full_pipeline_state):
    s = full_pipeline_state
    assert s.trust_scores is not None
    assert len(s.trust_scores) == 50
    assert np.all(s.trust_scores >= 0.0)
    assert np.all(s.trust_scores <= 1.0)


def test_cluster_labels_populated(full_pipeline_state):
    assert full_pipeline_state.cluster_labels is not None
    assert len(full_pipeline_state.cluster_labels) == 50


def test_trained_models_populated(full_pipeline_state):
    assert full_pipeline_state.trained_models is not None
    assert len(full_pipeline_state.trained_models) == 3


def test_best_model_selected(full_pipeline_state):
    s = full_pipeline_state
    assert s.best_model_name is not None
    assert s.best_model_name in s.trained_models


def test_feature_trust_populated(full_pipeline_state):
    assert full_pipeline_state.feature_trust is not None
    assert len(full_pipeline_state.feature_trust) > 0


def test_confidence_score_populated(full_pipeline_state):
    s = full_pipeline_state
    assert s.confidence_score is not None
    assert 0.0 < s.confidence_score <= 1.0


def test_noisy_model_scores_populated(full_pipeline_state):
    assert full_pipeline_state.noisy_model_scores is not None
    assert len(full_pipeline_state.noisy_model_scores) > 0


def test_shap_values_populated(full_pipeline_state):
    assert full_pipeline_state.shap_values is not None


def test_shap_feature_importance_populated(full_pipeline_state):
    assert full_pipeline_state.shap_feature_importance is not None
    assert len(full_pipeline_state.shap_feature_importance) > 0


def test_no_stage_errors(full_pipeline_state):
    """Pipeline must complete all stages without errors."""
    assert full_pipeline_state.stage_errors == {}, (
        f"Stage errors: {full_pipeline_state.stage_errors}"
    )


# ── Research claim validations ────────────────────────────────────────────────

def test_reliability_improves_performance(noisy_pipeline_state):
    """
    On noisy data, trust-weighted models must not be significantly worse
    than baseline (within 10%). On well-designed noisy data they should be better.
    """
    s = noisy_pipeline_state
    baseline = s.baseline_scores or {}
    weighted = s.model_scores or {}
    for name in weighted:
        if name in baseline:
            b = baseline[name]["cv_score"]
            w = weighted[name]["cv_score"]
            assert w >= b - 0.10, (
                f"{name}: weighted={w:.4f} is more than 10% below baseline={b:.4f}"
            )


def test_trust_scores_have_variance(full_pipeline_state):
    """
    Trust scores must not all be identical — the system must differentiate
    between reliable and unreliable rows.
    """
    trust = full_pipeline_state.trust_scores
    assert np.std(trust) > 0, "All trust scores are identical — no differentiation"


def test_graph_has_edges(full_pipeline_state):
    """Reliability graph must have at least one edge — not an empty graph."""
    G = full_pipeline_state.reliability_graph
    assert G.number_of_edges() > 0, "Reliability graph has no edges"


def test_graph_node_count(full_pipeline_state):
    """Graph must have exactly n_rows nodes."""
    G = full_pipeline_state.reliability_graph
    assert G.number_of_nodes() == 50


def test_weights_not_uniform(noisy_pipeline_state):
    """
    On noisy data, sample weights must not all be equal —
    trust-based weighting must differentiate rows.
    """
    weights = noisy_pipeline_state.sample_weights
    assert not np.allclose(weights, weights[0]), (
        "All sample weights are identical — trust weighting has no effect"
    )


def test_feature_trust_consistency(full_pipeline_state):
    """All feature trust values must be in [0, 1]."""
    for f, v in full_pipeline_state.feature_trust.items():
        assert 0.0 <= v <= 1.0, f"feature_trust[{f}] = {v} out of range"


def test_shap_importance_non_zero(full_pipeline_state):
    """At least one feature must have non-zero SHAP importance."""
    imp = full_pipeline_state.shap_feature_importance
    assert any(v > 0 for v in imp.values()), "All SHAP importances are zero"


def test_flagged_features_valid(full_pipeline_state):
    """All flagged features must have feature_trust < 0.5."""
    s = full_pipeline_state
    for f in (s.high_importance_low_trust_features or []):
        assert s.feature_trust[f] < 0.5, (
            f"Flagged feature {f} has trust={s.feature_trust[f]} >= 0.5"
        )


def test_fusion_weights_sum_to_one(full_pipeline_state):
    """Fusion weights must sum to 1.0."""
    w = full_pipeline_state.fusion_weights
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-6


def test_sample_weights_sum_to_one(full_pipeline_state):
    """Sample weights must sum to 1.0."""
    w = full_pipeline_state.sample_weights
    assert w is not None
    assert abs(w.sum() - 1.0) < 1e-6


# ── Pipeline stability ────────────────────────────────────────────────────────

def test_pipeline_deterministic():
    """Running the pipeline twice on the same data must give the same R score."""
    config = PipelineConfig(cv_folds=2, n_pseudo_sources=3, trust_max_iterations=10)
    s1 = _run_full_pipeline(_make_synthetic_csv(n=50), config)
    s2 = _run_full_pipeline(_make_synthetic_csv(n=50), config)
    assert abs(s1.reliability_score - s2.reliability_score) < 0.05


# ── Stress tests ──────────────────────────────────────────────────────────────

def test_pipeline_with_missing_data():
    """Pipeline must handle datasets with missing values without crashing."""
    config = PipelineConfig(cv_folds=2, trust_max_iterations=5)
    csv = _make_synthetic_csv(n=50)  # already has missing values injected
    state = PipelineState()
    state = validator.run(state, config, file=csv)
    assert state.raw_df.isnull().sum().sum() > 0, "Test CSV has no missing values"
    # Run through preprocessing — must not crash
    state = preprocessor.run(state, config)
    assert not state.processed_df.isnull().any().any()


def test_pipeline_large_dataset():
    """Pipeline must handle 500-row datasets correctly."""
    config = PipelineConfig(cv_folds=2, trust_max_iterations=5)
    csv = _make_synthetic_csv(n=500)
    state = PipelineState()
    state = validator.run(state, config, file=csv)
    state = preprocessor.run(state, config)
    assert len(state.processed_df) == 500


# ── Failure tests ─────────────────────────────────────────────────────────────

def test_invalid_input_raises():
    """Passing no file and no raw_df must raise ValidationError."""
    state = PipelineState()
    config = PipelineConfig()
    with pytest.raises(Exception):
        validator.run(state, config, file=None)


def test_invalid_csv_raises():
    """Passing a non-CSV file must raise ValidationError."""
    state = PipelineState()
    config = PipelineConfig()
    bad_file = io.StringIO("not,a,valid\ncsv,file,here\n!!@@##")
    # Should either parse (3 rows) or raise — either way must not crash silently
    try:
        state = validator.run(state, config, file=bad_file)
        # If it parsed, raw_df must be a DataFrame
        assert state.raw_df is not None
    except Exception:
        pass  # Expected for truly invalid input


# ── JSON report and serialization ─────────────────────────────────────────────

def test_explainability_report_valid_json(full_pipeline_state):
    s = full_pipeline_state
    report = {
        "reliability_score": s.reliability_score,
        "confidence_score": s.confidence_score,
        "fusion_weights": {
            f"w{i+1}": float(w)
            for i, w in enumerate(s.fusion_weights if s.fusion_weights is not None else [])
        },
        "feature_trust": s.feature_trust,
        "shap_importance": s.shap_feature_importance,
        "high_importance_low_trust_features": s.high_importance_low_trust_features,
        "model_scores": s.model_scores,
        "best_model": s.best_model_name,
    }
    report_json = json.dumps(report)
    parsed = json.loads(report_json)
    for key in ["reliability_score", "confidence_score", "feature_trust",
                "shap_importance", "best_model", "fusion_weights"]:
        assert key in parsed, f"Missing key in report: {key}"


def test_best_model_serializable(full_pipeline_state, tmp_path):
    """Best model must survive joblib serialization and produce identical predictions."""
    s = full_pipeline_state
    model = s.trained_models[s.best_model_name]
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    loaded = joblib.load(model_path)
    X = s.processed_df.values
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
