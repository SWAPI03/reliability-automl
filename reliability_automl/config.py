from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    similarity_threshold: float = 0.5
    low_variance_threshold: float = 0.01
    outlier_zscore_threshold: float = 3.0
    n_pseudo_sources: int = 5
    trust_convergence_tol: float = 0.001
    trust_max_iterations: int = 1000
    noise_level_fraction: float = 0.10
    cv_folds: int = 5
    fusion_weights: Optional[List[float]] = None
