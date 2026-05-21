"""Tests for model_selection.py — Property 21."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.model_selection import run


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_clf_state(scores: dict, R: float = 0.8) -> PipelineState:
    state = PipelineState()
    state.task_type = "classification"
    state.reliability_score = R
    state.model_scores = {name: {"cv_score": s} for name, s in scores.items()}
    state.raw_df = pd.DataFrame({"target": [0, 1, 0, 1]})
    state.target_column = "target"
    return state


def make_reg_state(scores: dict, y_range: float = 10.0, R: float = 0.8) -> PipelineState:
    state = PipelineState()
    state.task_type = "regression"
    state.reliability_score = R
    state.model_scores = {name: {"cv_score": s} for name, s in scores.items()}
    state.raw_df = pd.DataFrame({"target": [0.0, y_range]})
    state.target_column = "target"
    return state


# ── Classification selection ──────────────────────────────────────────────────

def test_best_model_selected_classification():
    state = run(make_clf_state({"A": 0.9, "B": 0.7, "C": 0.8}), PipelineConfig())
    assert state.best_model_name == "A"


def test_reliability_adjusted_score_classification():
    """Score = cv_score × R for classification."""
    R, cv = 0.8, 0.9
    state = run(make_clf_state({"A": cv}, R=R), PipelineConfig())
    assert state.model_scores["A"]["reliability_adjusted_score"] == pytest.approx(cv * R, abs=1e-9)


def test_best_model_name_stored():
    state = run(make_clf_state({"X": 0.5, "Y": 0.9}), PipelineConfig())
    assert state.best_model_name is not None
    assert state.best_model_name in state.model_scores


def test_model_ranking_order():
    """The model with the highest reliability_adjusted_score must be selected."""
    state = run(make_clf_state({"A": 0.6, "B": 0.8, "C": 0.7}), PipelineConfig())
    scores = state.model_scores
    sorted_models = sorted(
        scores, key=lambda m: scores[m]["reliability_adjusted_score"], reverse=True
    )
    assert sorted_models[0] == state.best_model_name


def test_tie_breaking():
    """When two models have equal scores, one must still be selected."""
    state = run(make_clf_state({"A": 0.8, "B": 0.8}), PipelineConfig())
    assert state.best_model_name in ["A", "B"]


def test_zero_reliability_score():
    """R=0 → all reliability_adjusted_scores must be 0.0."""
    state = run(make_clf_state({"A": 0.9, "B": 0.8}, R=0.0), PipelineConfig())
    for m in state.model_scores.values():
        assert m["reliability_adjusted_score"] == pytest.approx(0.0, abs=1e-9)


def test_reliability_changes_selection():
    """With R=1.0, the model with highest cv_score must win."""
    state = run(make_clf_state({"A": 0.9, "B": 0.85}, R=1.0), PipelineConfig())
    assert state.best_model_name == "A"


def test_all_scores_stored():
    """reliability_adjusted_score must be stored for every model."""
    state = run(make_clf_state({"A": 0.9, "B": 0.7, "C": 0.8}), PipelineConfig())
    for name in ["A", "B", "C"]:
        assert "reliability_adjusted_score" in state.model_scores[name]


# ── Regression selection ──────────────────────────────────────────────────────

def test_best_model_selected_regression():
    """Less negative neg_rmse = better model."""
    state = run(make_reg_state({"A": -1.0, "B": -0.5, "C": -2.0}), PipelineConfig())
    assert state.best_model_name == "B"


def test_reliability_adjusted_score_regression():
    """Score = (1 - |cv_score|/y_range) × R for regression."""
    R, y_range, cv_score = 0.8, 10.0, -1.0
    state = run(make_reg_state({"A": cv_score}, y_range=y_range, R=R), PipelineConfig())
    normalized_rmse = abs(cv_score) / y_range   # uses actual RMSE, not 1/y_range
    expected = (1.0 - normalized_rmse) * R
    assert state.model_scores["A"]["reliability_adjusted_score"] == pytest.approx(expected, abs=1e-9)


def test_regression_score_range():
    """Regression reliability_adjusted_score must be in [0, 1]."""
    state = run(make_reg_state({"A": -1.0}, y_range=10.0, R=0.8), PipelineConfig())
    score = state.model_scores["A"]["reliability_adjusted_score"]
    assert 0.0 <= score <= 1.0


def test_extreme_rmse_clipped():
    """Extreme RMSE (larger than y_range) must not produce negative score."""
    state = run(make_reg_state({"A": -1000.0}, y_range=10.0, R=0.8), PipelineConfig())
    score = state.model_scores["A"]["reliability_adjusted_score"]
    assert score >= 0.0


def test_regression_uses_actual_rmse():
    """
    Two models with different RMSE magnitudes must get different scores.
    This validates that |cv_score| is used, not a constant 1/y_range.
    """
    state = run(make_reg_state({"A": -1.0, "B": -5.0}, y_range=10.0), PipelineConfig())
    score_a = state.model_scores["A"]["reliability_adjusted_score"]
    score_b = state.model_scores["B"]["reliability_adjusted_score"]
    assert score_a > score_b  # lower RMSE → higher score


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_missing_cv_score_raises():
    """Missing cv_score in model_scores must raise an error."""
    state = PipelineState()
    state.task_type = "classification"
    state.reliability_score = 0.8
    state.model_scores = {"A": {}}  # no cv_score key
    state.raw_df = pd.DataFrame({"target": [0, 1]})
    state.target_column = "target"
    with pytest.raises(Exception):
        run(state, PipelineConfig())


def test_single_model():
    """With only one model, it must be selected as best."""
    state = run(make_clf_state({"OnlyModel": 0.75}), PipelineConfig())
    assert state.best_model_name == "OnlyModel"
