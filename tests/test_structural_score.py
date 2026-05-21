"""Tests for structural_score.py — Property 5: Structural Score Formula Correctness."""
import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.structural_score import run


def make_state(df: pd.DataFrame) -> PipelineState:
    state = PipelineState()
    state.raw_df = df
    return state


def test_perfect_data():
    """No missing, no duplicates, no constant columns → S = 1.0."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    state = run(make_state(df), PipelineConfig())
    assert state.structural_score == pytest.approx(1.0)


def test_all_missing():
    """All cells NaN, 2 rows → missing_ratio=1, duplicate_ratio=0.5, constant_ratio=1.
    S = 1 - (0.4*1 + 0.3*0.5 + 0.3*1) = 1 - 0.85 = 0.15."""
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    state = run(make_state(df), PipelineConfig())
    expected = 1.0 - (0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 1.0)
    assert state.structural_score == pytest.approx(expected)


def test_all_duplicates():
    """All rows identical → duplicate_ratio = (n-1)/n for n rows.
    With 2 rows: 1 duplicate / 2 rows = 0.5 → S = 1 - 0.3*0.5 = 0.85."""
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    state = run(make_state(df), PipelineConfig())
    # duplicate_ratio = 1/2 = 0.5, constant_ratio = 1.0
    # S = 1 - (0.4*0 + 0.3*0.5 + 0.3*1.0) = 1 - 0.45 = 0.55
    assert state.structural_score == pytest.approx(0.55)


def test_constant_columns():
    """All columns constant → constant_ratio=1 → S = 1 - 0.3 = 0.7."""
    df = pd.DataFrame({"a": [5, 5, 5], "b": [7, 7, 7]})
    state = run(make_state(df), PipelineConfig())
    # constant_ratio=1, duplicate_ratio=2/3, missing_ratio=0
    # S = 1 - (0 + 0.3*(2/3) + 0.3*1) = 1 - (0.2 + 0.3) = 0.5
    expected = 1.0 - (0.4 * 0 + 0.3 * (2 / 3) + 0.3 * 1.0)
    assert state.structural_score == pytest.approx(expected)


def test_known_formula():
    """Manually compute ratios and verify formula."""
    # 4 rows, 2 cols = 8 cells; 2 NaN → missing_ratio = 0.25
    # 1 duplicate row → duplicate_ratio = 1/4 = 0.25
    # 0 constant cols → constant_ratio = 0
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 1.0],
        "b": [np.nan, 2.0, 3.0, 1.0],
    })
    # Row 0: (1, NaN), Row 3: (1, 1) — not duplicates
    # Let's use a cleaner example
    df = pd.DataFrame({
        "a": [1.0, 1.0, 3.0, 4.0],
        "b": [2.0, 2.0, np.nan, 5.0],
    })
    # missing: 1/8 = 0.125
    # duplicates: row 0 == row 1 → 1 duplicate → 1/4 = 0.25
    # constant: neither col is constant → 0
    missing_ratio = 1 / 8
    duplicate_ratio = 1 / 4
    constant_ratio = 0.0
    expected = 1.0 - (0.4 * missing_ratio + 0.3 * duplicate_ratio + 0.3 * constant_ratio)

    state = run(make_state(df), PipelineConfig())
    assert state.structural_score == pytest.approx(expected)


def test_score_clipped_to_zero():
    """Worst-case data should not produce negative score."""
    df = pd.DataFrame({
        "a": [np.nan, np.nan, np.nan],
        "b": [np.nan, np.nan, np.nan],
    })
    state = run(make_state(df), PipelineConfig())
    assert state.structural_score >= 0.0


def test_score_clipped_to_one():
    """Best-case data should not exceed 1.0."""
    df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
    state = run(make_state(df), PipelineConfig())
    assert state.structural_score <= 1.0


def test_partial_missing():
    """Half the cells are NaN, no duplicates, no constant cols."""
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.nan],
        "b": [np.nan, 2.0, np.nan, 4.0],
    })
    missing_ratio = 4 / 8  # 0.5
    duplicate_ratio = 0.0
    constant_ratio = 0.0
    expected = 1.0 - (0.4 * 0.5)
    state = run(make_state(df), PipelineConfig())
    assert state.structural_score == pytest.approx(expected)
