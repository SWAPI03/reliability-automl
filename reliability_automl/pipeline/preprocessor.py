from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState


def run(state: PipelineState, config: PipelineConfig) -> PipelineState:
    """
    Preprocess features in state.raw_df.

    - Numeric columns: impute (mean) → scale (StandardScaler)
    - Categorical/datetime columns: impute (most_frequent) → OneHotEncode
    - Target column is excluded from transformation.
    - state.raw_df is never modified (deep copy taken before any work).
    - Results stored in state.processed_df (DataFrame) and
      state.preprocessor_pipeline (fitted sklearn Pipeline).
    """
    raw_df: pd.DataFrame = state.raw_df.copy(deep=True)
    target_col: str = state.target_column
    column_types: dict[str, str] = state.column_types

    # Separate feature columns from target
    feature_cols = [c for c in raw_df.columns if c != target_col]

    numeric_cols = [
        c for c in feature_cols if column_types.get(c) == "numeric"
    ]
    # Treat datetime and categorical the same way (OHE)
    categorical_cols = [
        c for c in feature_cols if column_types.get(c) in ("categorical", "datetime")
    ]

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    sklearn_pipeline = Pipeline([("preprocessor", ct)])

    features_df = raw_df[feature_cols]
    transformed = sklearn_pipeline.fit_transform(features_df)

    # Build column names for the output DataFrame
    output_cols: list[str] = list(numeric_cols)  # numeric cols keep their names

    if categorical_cols:
        ohe: OneHotEncoder = (
            sklearn_pipeline
            .named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        output_cols.extend(ohe_names)

    processed_df = pd.DataFrame(transformed, columns=output_cols, index=raw_df.index)

    # Preserve the original raw_df unchanged
    state.raw_df = raw_df
    state.processed_df = processed_df
    state.preprocessor_pipeline = sklearn_pipeline

    return state
