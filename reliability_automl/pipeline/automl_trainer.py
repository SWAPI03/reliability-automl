from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def _detect_task_type(y: np.ndarray) -> str:
    series = pd.Series(y)
    n_unique = series.nunique()
    dtype = series.dtype
    if n_unique <= 20 and (dtype == int or dtype == object or str(dtype).startswith("int")):
        return "classification"
    return "regression"


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    X = state.processed_df.values
    y_raw = state.raw_df[state.target_column].values
    n_rows = len(X)

    task_type = _detect_task_type(y_raw)  # detect from raw labels before encoding

    # Encode string labels to integers for XGBoost compatibility
    label_encoder = None
    if task_type == "classification" and y_raw.dtype == object:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = y_raw

    # Do NOT re-detect task type after encoding — use the one from raw labels
    state.task_type = task_type
    state.label_encoder = label_encoder

    R = state.reliability_score if state.reliability_score is not None else 0.5
    trust = state.trust_scores if state.trust_scores is not None else np.full(n_rows, 0.5)

    raw_weights = trust * R
    total = raw_weights.sum()
    sample_weights = raw_weights / total if total > 0 else np.full(n_rows, 1.0 / n_rows)
    state.sample_weights = sample_weights

    fit_weights = sample_weights * n_rows

    if task_type == "classification":
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

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(
                model, X, y, cv=cv, scoring=scoring,
                params={"sample_weight": fit_weights},
            )
        except TypeError:
            cv_scores = cross_val_score(
                model, X, y, cv=cv, scoring=scoring,
                fit_params={"sample_weight": fit_weights},
            )
        cv_score = float(cv_scores.mean())
        cv_std   = float(cv_scores.std())
        model.fit(X, y, sample_weight=fit_weights)
        trained_models[name] = model
        model_scores[name] = {"cv_score": cv_score, "cv_std": cv_std}

    state.trained_models = trained_models
    state.model_scores = model_scores
    return state
