from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df = state.processed_df
    target = state.target_column

    # Numeric columns excluding target
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    n_rows = len(df)

    if not numeric_cols or n_rows == 0:
        state.statistical_score = 1.0
        state.outlier_mask = np.zeros(n_rows, dtype=bool)
        return state

    X = df[numeric_cols].values

    if n_rows < 1000:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z_scores = np.abs(stats.zscore(X, nan_policy="omit"))
        # NaN z-scores (from constant columns) → treat as non-outlier (0)
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        outlier_mask = np.any(z_scores > config.outlier_zscore_threshold, axis=1)
    else:
        # IsolationForest method
        clf = IsolationForest(contamination="auto", random_state=42)
        preds = clf.fit_predict(X)
        outlier_mask = preds == -1

    outlier_ratio = outlier_mask.sum() / n_rows

    variances = np.var(X, axis=0)
    low_variance_ratio = np.sum(variances < config.low_variance_threshold) / len(numeric_cols)

    T = 1.0 - (0.6 * outlier_ratio + 0.4 * low_variance_ratio)
    state.statistical_score = float(np.clip(T, 0.0, 1.0))
    state.outlier_mask = outlier_mask
    return state
