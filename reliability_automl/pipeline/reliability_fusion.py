"""Reliability Fusion Engine — Dynamic Weight Learning.

Weights are learned via variance-based importance:
  w_i = var(component_i_per_row) / sum(all variances)

This gives higher weight to components that vary more across rows,
meaning they carry more discriminative information for this specific dataset.

Sim is a dataset-level scalar so it always gets a small fixed weight.
S, T, C, D are row-level and their variance drives the weights.

Falls back to uniform if all variances are zero.
"""
from __future__ import annotations

import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def _build_row_components(state: PipelineState) -> np.ndarray:
    """Build (n_rows, 5) matrix of per-row component values."""
    n = len(state.processed_df)

    # S proxy: 1 - fraction of missing values per row (computed on raw_df)
    if state.raw_df is not None:
        missing_per_row = state.raw_df.isnull().mean(axis=1).values
        s_row = np.clip(1.0 - missing_per_row, 0, 1)
    else:
        s_row = np.ones(n)

    # T proxy: 1 for non-outlier rows, 0 for outlier rows
    if state.outlier_mask is not None:
        t_row = (~state.outlier_mask).astype(float)
    else:
        t_row = np.ones(n)

    # Sim proxy: dataset-level scalar broadcast to all rows
    # (low variance by design — gets small weight automatically)
    sim_scalar = state.similarity_score if state.similarity_score is not None else 0.5
    sim_row = np.full(n, sim_scalar)

    # C proxy: 1 for non-conflicting rows, 0 for conflicting rows
    if state.conflict_mask is not None:
        c_row = (~state.conflict_mask).astype(float)
    else:
        c_row = np.ones(n)

    # D proxy: per-row deduction score via cluster trust
    if state.cluster_labels is not None:
        labels_arr = state.cluster_labels
        unique = np.unique(labels_arr)
        # Use uniform cluster trust (0.5) since trust_scores not yet computed
        # Weight by cluster size: larger clusters = more representative
        cluster_sizes = np.array([np.sum(labels_arr == k) for k in unique], dtype=float)
        cluster_trust = cluster_sizes / cluster_sizes.sum()
        trust_map = {int(k): float(cluster_trust[i]) for i, k in enumerate(unique)}
        d_row = np.array([trust_map.get(int(l), 0.5) for l in labels_arr])
    else:
        d_row = np.full(n, 0.5)

    return np.column_stack([s_row, t_row, sim_row, c_row, d_row])


def _variance_weights(row_components: np.ndarray) -> np.ndarray:
    """Variance-based weights: more variable = more informative."""
    variances = np.var(row_components, axis=0)

    # Sim (index 2) is always near-zero variance — give it a small floor
    variances[2] = max(variances[2], 1e-4)

    total = variances.sum()
    if total < 1e-12:
        return np.full(5, 0.2)

    return variances / total


def _entropy_weights(row_components: np.ndarray) -> np.ndarray:
    """Entropy-based weights: more spread = more informative.
    Uses normalized entropy of the value distribution per component."""
    weights = np.zeros(5)
    for i in range(5):
        col = row_components[:, i]
        # Bin into 10 buckets and compute entropy
        hist, _ = np.histogram(col, bins=10, range=(0, 1))
        hist = hist + 1e-9  # smoothing
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log(prob))
        weights[i] = entropy

    # Sim gets a floor
    weights[2] = max(weights[2], 0.1)
    total = weights.sum()
    return weights / total if total > 1e-9 else np.full(5, 0.2)


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    S   = state.structural_score  or 0.0
    T   = state.statistical_score or 0.0
    Sim = state.similarity_score  or 0.0
    C   = state.conflict_score    or 0.0
    D   = state.deduction_score   or 0.0

    # ── User-provided weights ─────────────────────────────────────────────────
    if config.fusion_weights is not None:
        w = np.array(config.fusion_weights, dtype=float)
        w = w / w.sum()
        weight_method = "user-provided"

    else:
        row_components = _build_row_components(state)

        # ── Primary: variance-based ───────────────────────────────────────────
        w_var = _variance_weights(row_components)

        # ── Secondary: entropy-based ──────────────────────────────────────────
        w_ent = _entropy_weights(row_components)

        # ── Blend both (average) for robustness ──────────────────────────────
        w = (w_var + w_ent) / 2.0
        w = w / w.sum()

        # Check if result is still suspiciously uniform
        max_dev = np.max(np.abs(w - 0.2))
        if max_dev < 0.01:
            # All components have similar distributions — use variance only
            w = w_var
            weight_method = "variance-based"
        else:
            weight_method = "variance + entropy blend"

    R = w[0]*S + w[1]*T + w[2]*Sim + w[3]*(1.0 - C) + w[4]*D
    state.reliability_score     = float(np.clip(R, 0.0, 1.0))
    state.fusion_weights        = w
    state.fusion_weight_method  = weight_method  # type: ignore[attr-defined]
    return state
