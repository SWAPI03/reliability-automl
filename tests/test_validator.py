from __future__ import annotations
import io
import pandas as pd
import pytest
from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline.validator import ValidationError, run


def _make_csv(rows):
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _state():
    return PipelineState()


def _config():
    return PipelineConfig()


# --- Unit Tests ---

def test_valid_csv_sets_raw_df_target_column_types():
    csv = _make_csv([
        {"feature1": 1.0, "feature2": "a", "target": 0},
        {"feature1": 2.0, "feature2": "b", "target": 1},
        {"feature1": 3.0, "feature2": "c", "target": 0},
        {"feature1": 4.0, "feature2": "d", "target": 1},
        {"feature1": 5.0, "feature2": "e", "target": 0},
        {"feature1": 6.0, "feature2": "f", "target": 1},
        {"feature1": 7.0, "feature2": "g", "target": 0},
        {"feature1": 8.0, "feature2": "h", "target": 1},
        {"feature1": 9.0, "feature2": "i", "target": 0},
        {"feature1": 10.0, "feature2": "j", "target": 1},
    ])
    state = run(_state(), _config(), file=csv)

    assert isinstance(state.raw_df, pd.DataFrame)
    assert len(state.raw_df) == 10
    assert state.target_column == "target"
    assert state.column_types is not None
    assert state.column_types["feature1"] == "numeric"
    assert state.column_types["feature2"] == "categorical"
    assert state.column_types["target"] == "numeric"


def test_invalid_csv_raises_validation_error():
    # Pass an object that is not a valid file-like or path — pd.read_csv will raise
    with pytest.raises(ValidationError):
        run(_state(), _config(), file=object())


def test_invalid_csv_string_raises_validation_error():
    # Passing a non-file, non-bytes object that will fail pd.read_csv
    with pytest.raises(ValidationError):
        run(_state(), _config(), file=12345)


def test_fewer_than_10_rows_stores_warning_but_continues():
    csv = _make_csv([
        {"a": i, "target": i % 2} for i in range(5)
    ])
    state = run(_state(), _config(), file=csv)

    assert isinstance(state.raw_df, pd.DataFrame)
    assert len(state.raw_df) == 5
    assert "validator_warning" in state.stage_errors
    assert "5" in state.stage_errors["validator_warning"]


def test_target_column_detected_by_name_target():
    csv = _make_csv([{"x": i, "target": i} for i in range(10)])
    state = run(_state(), _config(), file=csv)
    assert state.target_column == "target"


def test_target_column_detected_by_name_label():
    csv = _make_csv([{"x": i, "label": i} for i in range(10)])
    state = run(_state(), _config(), file=csv)
    assert state.target_column == "label"


def test_target_column_detected_by_name_class():
    csv = _make_csv([{"x": i, "class": i} for i in range(10)])
    state = run(_state(), _config(), file=csv)
    assert state.target_column == "class"


def test_target_column_case_insensitive():
    csv = _make_csv([{"x": i, "TARGET": i} for i in range(10)])
    state = run(_state(), _config(), file=csv)
    assert state.target_column == "TARGET"


def test_target_column_fallback_to_last_column():
    csv = _make_csv([{"a": i, "b": i * 2, "outcome": i % 3} for i in range(10)])
    state = run(_state(), _config(), file=csv)
    assert state.target_column == "outcome"


def test_raw_df_already_set_skips_parsing():
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    state = _state()
    state.raw_df = df
    result = run(state, _config())
    assert result.raw_df is df  # same object, not re-parsed


def test_bytes_input_parsed_correctly():
    df = pd.DataFrame({"x": range(10), "target": range(10)})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")
    state = run(_state(), _config(), file=csv_bytes)
    assert isinstance(state.raw_df, pd.DataFrame)
    assert len(state.raw_df) == 10
