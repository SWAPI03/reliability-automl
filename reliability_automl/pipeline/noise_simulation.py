from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline import preprocessor as _preprocessor


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    rng = np.random.default_rng(42)
    target = state.target_column
    column_types = state.column_types

    noisy_raw = state.raw_df.copy(deep=True)
    for col, ctype in column_types.items():
        if col == target or ctype != "numeric":
            continue
        std = float(noisy_raw[col].std(skipna=True))
        if std > 0:
            noise = rng.normal(0, config.noise_level_fraction * std, size=len(noisy_raw))
            noisy_raw[col] = noisy_raw[col] + noise

    noisy_state = PipelineState()
    noisy_state.raw_df = noisy_raw
    noisy_state.target_column = target
    noisy_state.column_types = column_types
    noisy_state = _preprocessor.run(noisy_state, config)

    X_noisy = noisy_state.processed_df.values
    y_raw = state.raw_df[target].values

    # Encode string labels for XGBoost compatibility (same as automl_trainer)
    le = getattr(state, "label_encoder", None)
    if state.task_type == "classification" and le is not None:
        y = le.transform(y_raw)
    else:
        y = y_raw

    n_rows = len(X_noisy)
    R = state.reliability_score if state.reliability_score is not None else 0.5
    y_range = float(np.max(y) - np.min(y)) if state.task_type == "regression" else 1.0

    fit_weights = state.sample_weights * n_rows if state.sample_weights is not None else None

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

    noisy_model_scores = {}
    for name, model in models.items():
        if fit_weights is not None:
            try:
                cv_scores = cross_val_score(model, X_noisy, y, cv=cv, scoring=scoring,
                                            params={"sample_weight": fit_weights})
            except TypeError:
                cv_scores = cross_val_score(model, X_noisy, y, cv=cv, scoring=scoring,
                                            fit_params={"sample_weight": fit_weights})
        else:
            cv_scores = cross_val_score(model, X_noisy, y, cv=cv, scoring=scoring)

        cv_score = float(cv_scores.mean())
        if state.task_type == "classification":
            reliability_adjusted = cv_score * R
        else:
            normalized_rmse = min(abs(cv_score) / y_range, 1.0) if y_range > 0 else 0.0
            reliability_adjusted = (1.0 - normalized_rmse) * R

        noisy_model_scores[name] = {
            "cv_score": cv_score,
            "reliability_adjusted_score": float(reliability_adjusted),
        }

    state.noisy_model_scores = noisy_model_scores
    return state
