"""Baseline comparison: train models WITHOUT trust weights (equal weights).
Includes multi-level noise injection experiment for robustness analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState

# Noise levels for the robustness experiment (fraction of feature std)
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]


def _add_noise(X: np.ndarray, level: float, rng) -> np.ndarray:
    if level == 0.0:
        return X.copy()
    noise = rng.normal(0, level * np.std(X, axis=0), X.shape)
    return X + noise


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    X = state.processed_df.values
    y_raw = state.raw_df[state.target_column].values

    # Encode string labels for XGBoost compatibility
    le = getattr(state, "label_encoder", None)
    if state.task_type == "classification" and le is not None:
        y = le.transform(y_raw)
    else:
        y = y_raw

    R = state.reliability_score if state.reliability_score is not None else 0.5
    y_range = float(np.max(y) - np.min(y)) if state.task_type == "regression" else 1.0

    if state.task_type == "classification":
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        }
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "LinearRegression": LinearRegression(),
            "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        }
        cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
        scoring = "neg_root_mean_squared_error"

    rng = np.random.default_rng(99)

    baseline_scores = {}
    noise_experiment = {}  # {noise_level: {model: score}}

    for name, model in models.items():
        # ── Clean baseline (no weights) ────────────────────────────────────────
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_score = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        model.fit(X, y)

        if state.task_type == "classification":
            reliability_adjusted = cv_score * R
        else:
            nrmse = min(abs(cv_score) / y_range, 1.0) if y_range > 0 else 0.0
            reliability_adjusted = (1.0 - nrmse) * R

        # ── Noise robustness at 10% (single level for quick comparison) ────────
        X_noisy_10 = _add_noise(X, 0.1, rng)
        noisy_cv = cross_val_score(
            model.__class__(**model.get_params()), X_noisy_10, y, cv=cv, scoring=scoring
        )
        noisy_score = float(noisy_cv.mean())

        baseline_scores[name] = {
            "cv_score": cv_score,
            "cv_std": cv_std,
            "reliability_adjusted_score": reliability_adjusted,
            "noisy_cv_score": noisy_score,
            "accuracy_drop": cv_score - noisy_score,
        }

    # ── Multi-level noise experiment (best model only for speed) ──────────────
    best = state.best_model_name
    if best and best in models:
        baseline_model_cls = models[best].__class__
        baseline_params = models[best].get_params()

        # Weighted model scores at each noise level
        weighted_model = state.trained_models.get(best)
        fit_weights = state.sample_weights * len(X) if state.sample_weights is not None else None

        for level in NOISE_LEVELS:
            # Use a fixed seed per noise level for reproducibility
            level_rng = np.random.default_rng(int(level * 1000) + 99)
            X_lvl = _add_noise(X, level, level_rng)

            # Baseline score at this noise level
            b_cv = cross_val_score(
                baseline_model_cls(**baseline_params), X_lvl, y, cv=cv, scoring=scoring
            )
            b_score = float(b_cv.mean())

            # Weighted score at this noise level
            if fit_weights is not None:
                try:
                    w_cv = cross_val_score(
                        baseline_model_cls(**baseline_params), X_lvl, y, cv=cv,
                        scoring=scoring, params={"sample_weight": fit_weights}
                    )
                except TypeError:
                    w_cv = cross_val_score(
                        baseline_model_cls(**baseline_params), X_lvl, y, cv=cv,
                        scoring=scoring, fit_params={"sample_weight": fit_weights}
                    )
                w_score = float(w_cv.mean())
            else:
                w_score = b_score

            noise_experiment[level] = {
                "baseline": b_score,
                "weighted": w_score,
            }

    state.baseline_scores = baseline_scores
    state.noise_experiment = noise_experiment  # type: ignore[attr-defined]
    return state
