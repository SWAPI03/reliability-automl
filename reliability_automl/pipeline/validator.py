from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


class ValidationError(Exception):
    pass


def _detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Detect column types: 'numeric', 'datetime', or 'categorical'."""
    types: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types


def _identify_target_column(df: pd.DataFrame) -> Optional[str]:
    """Identify target column by name heuristics, falling back to last column."""
    target_keywords = {"target", "label", "class"}
    for col in df.columns:
        if col.lower() in target_keywords:
            return col
    # Fallback: last column
    if len(df.columns) > 0:
        return df.columns[-1]
    return None


def run(
    state: PipelineState,
    config: PipelineConfig,
    file=None,
) -> PipelineState:
    """
    Validate and parse the input CSV.

    Parameters
    ----------
    state:
        Shared pipeline state. If ``state.raw_df`` is already a DataFrame,
        parsing is skipped.
    config:
        Pipeline configuration (unused here but kept for interface consistency).
    file:
        A file-like object, bytes, or str path to a CSV file.  If ``None``,
        ``state.raw_df`` must already be a DataFrame.
    """
    # --- 1. Parse CSV if needed ---
    if isinstance(state.raw_df, pd.DataFrame):
        df = state.raw_df
    else:
        # Determine the source to pass to pd.read_csv
        if file is not None:
            source = file
            if isinstance(file, bytes):
                source = io.BytesIO(file)
        elif state.raw_df is not None:
            # state.raw_df holds a file-like object or path
            source = state.raw_df
        else:
            raise ValidationError(
                "No data source provided: pass a file or set state.raw_df."
            )

        try:
            df = pd.read_csv(source)
        except Exception as exc:
            raise ValidationError(f"Failed to parse CSV: {exc}") from exc

    # --- 2. Row count warning ---
    if len(df) < 10:
        state.stage_errors["validator_warning"] = (
            f"Dataset has only {len(df)} row(s), which may be too small for "
            "reliable analysis."
        )

    # --- 3. Detect column types ---
    column_types = _detect_column_types(df)

    # --- 4. Identify target column ---
    target_column = _identify_target_column(df)
    if target_column is None:
        raise ValidationError(
            "Could not identify a target column. "
            "Please ensure the dataset has at least one column."
        )

    # --- 5. Persist results in state ---
    state.raw_df = df
    state.target_column = target_column
    state.column_types = column_types

    return state
