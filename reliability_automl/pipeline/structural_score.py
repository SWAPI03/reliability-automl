from __future__ import annotations

import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df = state.raw_df

    total_cells = df.size
    missing_ratio = df.isna().sum().sum() / total_cells if total_cells > 0 else 0.0

    n_rows = len(df)
    duplicate_ratio = df.duplicated().sum() / n_rows if n_rows > 0 else 0.0

    n_cols = len(df.columns)
    constant_ratio = sum(df[c].nunique(dropna=False) <= 1 for c in df.columns) / n_cols if n_cols > 0 else 0.0

    S = 1.0 - (0.4 * missing_ratio + 0.3 * duplicate_ratio + 0.3 * constant_ratio)
    state.structural_score = float(np.clip(S, 0.0, 1.0))
    return state
