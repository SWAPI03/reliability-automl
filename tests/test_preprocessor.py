"""Unit tests for reliability_automl/pipeline/preprocessor.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.preprocessor import run
from reliability_automl.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(df: pd.DataFrame, target: str, col_types: dict) -> PipelineState:
    state = PipelineState()
    state.raw_df = df.copy(deep=True)
    state.target_column = target
    state.column_types = col_types
    return state


def _numeric_df() -> tuple[pd.DataFrame, str, dict]:
    """All-numeric feature DataFrame with one NaN."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0, 1, 0, 1, 0],
    })
    col_types = {"a": "numeric", "b": "numeric", "target": "numeric"}
    return df, "target", col_types


def _categorical_df() -> tuple[pd.DataFrame, str, dict]:
    """All-categorical feature DataFrame with one NaN."""
    df = pd.DataFrame({
        "color": ["red", "blue", np.nan, "red", "blue"],
        "size": ["S", "M", "L", "S", "M"],
        "target": [0, 1, 0, 1, 0],
    })
    col_types = {"color": "categorical", "size": "categorical", "target": "numeric"}
    return df, "target", col_types


def _mixed_df() -> tuple[pd.DataFrame, str, dict]:
    """Mixed numeric + categorical features."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "num2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "cat1": ["a", "b", "a", np.nan, "b"],
        "target": [0, 1, 0, 1, 0],
    })
    col_types = {
        "num1": "numeric",
        "num2": "numeric",
        "cat1": "categorical",
        "target": "numeric",
    }
    return df, "target", col_types


# ---------------------------------------------------------------------------
# Tests: no NaN values remain
# ---------------------------------------------------------------------------

class TestNoNaNAfterPreprocessing:
    def test_numeric_no_nan(self):
        df, target, col_types = _numeric_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert not state.processed_df.isnull().any().any()

    def test_categorical_no_nan(self):
        df, target, col_types = _categorical_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert not state.processed_df.isnull().any().any()

    def test_mixed_no_nan(self):
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert not state.processed_df.isnull().any().any()


# ---------------------------------------------------------------------------
# Tests: all columns are numeric dtype
# ---------------------------------------------------------------------------

class TestAllColumnsNumeric:
    def test_numeric_only_dtype(self):
        df, target, col_types = _numeric_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        for col in state.processed_df.columns:
            assert pd.api.types.is_numeric_dtype(state.processed_df[col]), (
                f"Column '{col}' is not numeric"
            )

    def test_categorical_only_dtype(self):
        df, target, col_types = _categorical_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        for col in state.processed_df.columns:
            assert pd.api.types.is_numeric_dtype(state.processed_df[col])

    def test_mixed_dtype(self):
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        for col in state.processed_df.columns:
            assert pd.api.types.is_numeric_dtype(state.processed_df[col])


# ---------------------------------------------------------------------------
# Tests: raw_df is unchanged
# ---------------------------------------------------------------------------

class TestRawDfUnchanged:
    def test_raw_df_unchanged_numeric(self):
        df, target, col_types = _numeric_df()
        original = df.copy(deep=True)
        state = run(_make_state(df, target, col_types), PipelineConfig())
        pd.testing.assert_frame_equal(state.raw_df, original)

    def test_raw_df_unchanged_categorical(self):
        df, target, col_types = _categorical_df()
        original = df.copy(deep=True)
        state = run(_make_state(df, target, col_types), PipelineConfig())
        pd.testing.assert_frame_equal(state.raw_df, original)

    def test_raw_df_unchanged_mixed(self):
        df, target, col_types = _mixed_df()
        original = df.copy(deep=True)
        state = run(_make_state(df, target, col_types), PipelineConfig())
        pd.testing.assert_frame_equal(state.raw_df, original)

    def test_raw_df_still_has_target_column(self):
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert target in state.raw_df.columns

    def test_processed_df_excludes_target(self):
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert target not in state.processed_df.columns


# ---------------------------------------------------------------------------
# Tests: numeric columns have mean ≈ 0, std ≈ 1 after scaling
# ---------------------------------------------------------------------------

class TestScalingProperties:
    def test_numeric_mean_approx_zero(self):
        df, target, col_types = _numeric_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        for col in ["a", "b"]:
            mean = state.processed_df[col].mean()
            assert abs(mean) < 1e-10, f"Column '{col}' mean={mean} not ≈ 0"

    def test_numeric_std_approx_one(self):
        df, target, col_types = _numeric_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        for col in ["a", "b"]:
            std = state.processed_df[col].std(ddof=0)
            assert abs(std - 1.0) < 1e-10, f"Column '{col}' std={std} not ≈ 1"

    def test_mixed_numeric_cols_scaled(self):
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        for col in ["num1", "num2"]:
            mean = state.processed_df[col].mean()
            std = state.processed_df[col].std(ddof=0)
            assert abs(mean) < 1e-10, f"Column '{col}' mean={mean} not ≈ 0"
            assert abs(std - 1.0) < 1e-10, f"Column '{col}' std={std} not ≈ 1"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_numeric_no_categorical(self):
        df, target, col_types = _numeric_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert state.processed_df is not None
        assert state.processed_df.shape[1] == 2  # a, b

    def test_all_categorical_no_numeric(self):
        df, target, col_types = _categorical_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert state.processed_df is not None
        # OHE expands columns; just check > 0 columns
        assert state.processed_df.shape[1] > 0

    def test_datetime_treated_as_categorical(self):
        df = pd.DataFrame({
            "dt": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"],
            "target": [0, 1, 0, 1, 0],
        })
        col_types = {"dt": "datetime", "target": "numeric"}
        state = run(_make_state(df, "target", col_types), PipelineConfig())
        assert state.processed_df is not None
        assert not state.processed_df.isnull().any().any()
        for col in state.processed_df.columns:
            assert pd.api.types.is_numeric_dtype(state.processed_df[col])

    def test_preprocessor_pipeline_stored(self):
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        assert state.preprocessor_pipeline is not None

    def test_preprocessor_pipeline_can_transform(self):
        """Fitted pipeline should be able to transform new data."""
        df, target, col_types = _mixed_df()
        state = run(_make_state(df, target, col_types), PipelineConfig())
        new_data = df[["num1", "num2", "cat1"]].iloc[:2]
        result = state.preprocessor_pipeline.transform(new_data)
        assert result.shape[0] == 2
