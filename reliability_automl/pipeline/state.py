from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineState:
    # Ingestion
    raw_df: Optional[pd.DataFrame] = None
    target_column: Optional[str] = None
    column_types: Optional[Dict[str, str]] = None  # "numeric" | "categorical" | "datetime"

    # Preprocessing
    processed_df: Optional[pd.DataFrame] = None
    preprocessor_pipeline: Optional[Any] = None  # sklearn Pipeline

    # Scores
    structural_score: Optional[float] = None
    statistical_score: Optional[float] = None
    similarity_score: Optional[float] = None
    conflict_score: Optional[float] = None
    deduction_score: Optional[float] = None
    reliability_score: Optional[float] = None
    fusion_weights: Optional[np.ndarray] = None  # shape (5,)
    fusion_weight_method: Optional[str] = None   # how weights were computed

    # Row-level
    trust_scores: Optional[np.ndarray] = None    # shape (n_rows,)
    outlier_mask: Optional[np.ndarray] = None    # bool, shape (n_rows,)
    conflict_mask: Optional[np.ndarray] = None   # bool, shape (n_rows,)
    cluster_labels: Optional[np.ndarray] = None  # shape (n_rows,)
    sample_weights: Optional[np.ndarray] = None  # shape (n_rows,)

    # Graph
    reliability_graph: Optional[Any] = None  # nx.Graph

    # Feature-level
    feature_trust: Optional[Dict[str, float]] = None
    confidence_score: Optional[float] = None

    # Models
    trained_models: Optional[Dict[str, Any]] = None
    model_scores: Optional[Dict[str, Dict[str, float]]] = None
    best_model_name: Optional[str] = None
    task_type: Optional[str] = None  # "classification" | "regression"
    label_encoder: Optional[Any] = None  # LabelEncoder for string targets

    # Noise simulation
    noisy_model_scores: Optional[Dict[str, Dict[str, float]]] = None

    # Baseline comparison (Before vs After)
    baseline_scores: Optional[Dict[str, Dict[str, float]]] = None  # unweighted model scores
    noise_experiment: Optional[Dict[float, Dict[str, float]]] = None  # multi-level noise results

    # Explainability
    shap_values: Optional[np.ndarray] = None
    shap_feature_importance: Optional[Dict[str, float]] = None
    high_importance_low_trust_features: Optional[List[str]] = None

    # Per-stage error tracking
    stage_errors: Dict[str, str] = field(default_factory=dict)
