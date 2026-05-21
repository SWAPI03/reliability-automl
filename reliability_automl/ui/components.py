"""Reusable Streamlit UI components."""
from __future__ import annotations

import io
import json
import joblib
import streamlit as st
import numpy as np


def file_uploader() -> object | None:
    """CSV file upload widget. Returns the uploaded file object or None."""
    return st.file_uploader("Upload a CSV dataset", type=["csv"])


def stage_progress(stage_name: str) -> None:
    """Display a spinner/status message for a running pipeline stage."""
    st.info(f"⏳ Running: {stage_name}...")


def download_model_button(model, model_name: str) -> None:
    """Serialize model with joblib and provide a download button."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    st.download_button(
        label=f"⬇️ Download Best Model ({model_name})",
        data=buf,
        file_name=f"{model_name.lower().replace(' ', '_')}_model.joblib",
        mime="application/octet-stream",
    )


def download_report_button(report: dict) -> None:
    """Serialize explainability report as JSON and provide a download button."""
    report_json = json.dumps(report, indent=2, default=_json_serializer)
    st.download_button(
        label="⬇️ Download Explainability Report (JSON)",
        data=report_json,
        file_name="explainability_report.json",
        mime="application/json",
    )


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
