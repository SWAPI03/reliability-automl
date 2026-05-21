"""Tests for reliability_fusion.py — Property 12: R formula and weight normalization."""
import numpy as np
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.reliability_fusion import run


def make_state(S=0.8, T=0.7, Sim=0.6, C=0.1, D=0.75) -> PipelineState:
    state = PipelineState()
    state.structural_score = S
    state.statistical_score = T
    state.similarity_score = Sim
    state.conflict_score = C
    state.deduction_score = D
    return state


# ── Formula correctness ────────────────────────────────────────────────────────

def test_uniform_weights_formula():
    """With uniform weights [0.2]*5, verify R = 0.2*(S+T+Sim+(1-C)+D)."""
    S, T, Sim, C, D = 0.8, 0.7, 0.6, 0.1, 0.75
    expected_R = 0.2 * (S + T + Sim + (1 - C) + D)
    state = run(make_state(S, T, Sim, C, D), PipelineConfig(fusion_weights=None))
    assert state.reliability_score == pytest.approx(expected_R, abs=1e-9)


def test_custom_weights_formula():
    """With provided weights, R = w·[S, T, Sim, 1-C, D] after normalization."""
    S, T, Sim, C, D = 0.9, 0.8, 0.7, 0.2, 0.6
    weights = [1.0, 2.0, 1.0, 1.0, 1.0]  # sum = 6
    w = np.array(weights) / sum(weights)
    expected_R = w[0]*S + w[1]*T + w[2]*Sim + w[3]*(1-C) + w[4]*D
    state = run(make_state(S, T, Sim, C, D), PipelineConfig(fusion_weights=weights))
    assert state.reliability_score == pytest.approx(expected_R, abs=1e-9)


def test_weights_sum_to_one_uniform():
    """Uniform weights must sum to 1.0."""
    state = run(make_state(), PipelineConfig(fusion_weights=None))
    assert state.fusion_weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_weights_sum_to_one_custom():
    """Provided weights are normalized to sum to 1.0."""
    state = run(make_state(), PipelineConfig(fusion_weights=[2.0, 3.0, 1.0, 2.0, 2.0]))
    assert state.fusion_weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_fusion_weights_shape():
    """fusion_weights must be a numpy array of shape (5,)."""
    state = run(make_state(), PipelineConfig())
    assert isinstance(state.fusion_weights, np.ndarray)
    assert state.fusion_weights.shape == (5,)


def test_score_clipped_to_zero():
    """R must not go below 0."""
    # All scores = 0, C = 1 → R = 0.2*(0+0+0+0+0) = 0
    state = run(make_state(S=0, T=0, Sim=0, C=1.0, D=0), PipelineConfig())
    assert state.reliability_score >= 0.0


def test_score_clipped_to_one():
    """R must not exceed 1."""
    state = run(make_state(S=1, T=1, Sim=1, C=0, D=1), PipelineConfig())
    assert state.reliability_score <= 1.0


def test_perfect_scores():
    """All component scores = 1, C = 0 → R = 1.0."""
    state = run(make_state(S=1.0, T=1.0, Sim=1.0, C=0.0, D=1.0), PipelineConfig())
    assert state.reliability_score == pytest.approx(1.0)


def test_worst_scores():
    """All component scores = 0, C = 1 → R = 0.0."""
    state = run(make_state(S=0.0, T=0.0, Sim=0.0, C=1.0, D=0.0), PipelineConfig())
    assert state.reliability_score == pytest.approx(0.0)


def test_weight_normalization_already_summing_to_one():
    """Weights already summing to 1 should remain unchanged."""
    w = [0.2, 0.2, 0.2, 0.2, 0.2]
    state = run(make_state(), PipelineConfig(fusion_weights=w))
    np.testing.assert_allclose(state.fusion_weights, np.array(w), atol=1e-9)


def test_reliability_score_stored():
    """reliability_score must be stored in state."""
    state = run(make_state(), PipelineConfig())
    assert state.reliability_score is not None
    assert isinstance(state.reliability_score, float)
