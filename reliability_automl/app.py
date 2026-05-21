"""Streamlit entry point — Reliability-Aware AutoML."""
from __future__ import annotations

import json
import joblib
import io
import numpy as np
import streamlit as st

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline import (
    validator, preprocessor, structural_score, statistical_score,
    pattern_analysis, deduction_score, reliability_fusion, reliability_graph,
    trust_propagation, automl_trainer, feedback_refinement, feature_trust,
    model_selection, noise_simulation, baseline_comparison, explainability,
)
from reliability_automl.ui import components, dashboard
from reliability_automl.ui.styles import CUSTOM_CSS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reliability-Aware AutoML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Pipeline stages definition ────────────────────────────────────────────────
STAGES = [
    ("Validation",          validator.run,           {}),
    ("Preprocessing",       preprocessor.run,        {}),
    ("Structural Score",    structural_score.run,    {}),
    ("Statistical Score",   statistical_score.run,   {}),
    ("Pattern Analysis",    pattern_analysis.run,    {}),
    ("Deduction Score",     deduction_score.run,     {}),
    ("Reliability Fusion",  reliability_fusion.run,  {}),
    ("Reliability Graph",   reliability_graph.run,   {}),
    ("Trust Propagation",   trust_propagation.run,   {}),
    ("AutoML Training",     automl_trainer.run,      {}),
    ("Model Selection",     model_selection.run,     {}),
    ("Feedback Refinement", feedback_refinement.run, {}),
    ("Feature Trust",       feature_trust.run,       {}),
    ("Noise Simulation",    noise_simulation.run,    {}),
    ("Baseline Comparison", baseline_comparison.run, {}),
    ("Explainability",      explainability.run,      {}),
]

STAGE_ICONS = {
    "Validation": "✅", "Preprocessing": "✅", "Structural Score": "✅",
    "Statistical Score": "✅", "Pattern Analysis": "✅", "Deduction Score": "✅",
    "Reliability Fusion": "✅", "Reliability Graph": "✅", "Trust Propagation": "✅",
    "AutoML Training": "✅", "Model Selection": "✅", "Feedback Refinement": "✅",
    "Feature Trust": "✅", "Noise Simulation": "✅", "Baseline Comparison": "✅",
    "Explainability": "✅",
}


def _run_stage(name, fn, state, config, **kwargs):
    try:
        state = fn(state, config, **kwargs)
    except Exception as exc:
        state.stage_errors[name] = str(exc)
        st.error(f"❌ **{name}** failed: {exc}")
    return state


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
        <div style="font-size:2.5rem;">🔬</div>
        <div style="font-size:1.1rem;font-weight:700;color:#ffffff;">Reliability-Aware</div>
        <div style="font-size:0.85rem;color:#7c83fd;font-weight:600;">AutoML System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    st.markdown('<div style="color:#8b92a5;font-size:0.75rem;text-transform:uppercase;'
                'letter-spacing:0.08em;padding:4px 0 8px 0;">Navigation</div>',
                unsafe_allow_html=True)

    sections = [
        ("Data Overview"),
        ("Reliability Scores"),
        ("Trust Graph"),
        ("Model Performance"),
        ("Explainability"),
        ("Before vs After"),
    ]
    selected = st.radio(
        "Go to",
        sections,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Pipeline status (shown after run)
    if "pipeline_state" in st.session_state:
        state_obj = st.session_state["pipeline_state"]
        st.markdown('<div style="color:#8b92a5;font-size:0.75rem;text-transform:uppercase;'
                    'letter-spacing:0.08em;padding:4px 0 8px 0;">Pipeline Status</div>',
                    unsafe_allow_html=True)
        for name, _, _ in STAGES:
            if name in state_obj.stage_errors:
                icon = "❌"
                color = "#f87171"
            else:
                icon = "✅"
                color = "#4ade80"
            st.markdown(
                f'<div style="font-size:0.78rem;color:{color};padding:2px 0;">'
                f'{icon} {name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div style="color:#4b5563;font-size:0.7rem;text-align:center;">'
                'v1.0 · Built with Streamlit</div>', unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 24px 0;">
    <h1 style="margin:0;font-size:2rem;background:linear-gradient(135deg,#7c83fd,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Reliability-Aware AutoML</h1>
    <p style="color:#8b92a5;margin:4px 0 0 0;font-size:0.95rem;">
    Upload a CSV dataset to compute reliability scores, propagate trust, and train robust models.
    </p>
</div>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your CSV file here",
    type=["csv"],
    help="Upload any tabular CSV dataset. The system will auto-detect the target column.",
)

if uploaded_file is None:
    # Landing page
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem;margin-bottom:8px;">🎯</div>
            <div style="font-weight:700;color:#fff;margin-bottom:6px;">Multi-Dimensional Scoring</div>
            <div style="color:#8b92a5;font-size:0.85rem;">
            Structural, Statistical, Similarity, Conflict, and Deduction scores combined into one Reliability Score R.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem;margin-bottom:8px;">🕸️</div>
            <div style="font-weight:700;color:#fff;margin-bottom:6px;">Graph Trust Propagation</div>
            <div style="color:#8b92a5;font-size:0.85rem;">
            Trust flows through a row-similarity graph. Reliable rows reinforce each other; conflicts are penalized.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem;margin-bottom:8px;">🤖</div>
            <div style="font-weight:700;color:#fff;margin-bottom:6px;">Reliability-Aware Training</div>
            <div style="color:#8b92a5;font-size:0.85rem;">
            Models trained with per-row trust weights. Best model selected by Accuracy × R — not accuracy alone.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
config = PipelineConfig()

# Cache pipeline result in session state to avoid re-running on every interaction
file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.get("file_id") != file_id:
    st.session_state["file_id"] = file_id
    st.session_state.pop("pipeline_state", None)

if "pipeline_state" not in st.session_state:
    state = PipelineState()

    progress_bar = st.progress(0, text="Starting pipeline...")
    status_box = st.empty()

    total = len(STAGES)
    for i, (name, fn, kwargs) in enumerate(STAGES):
        progress_bar.progress((i) / total, text=f"⏳ Running: **{name}**...")
        status_box.info(f"Running stage {i+1}/{total}: **{name}**")

        if name == "Validation":
            state = _run_stage(name, fn, state, config, file=uploaded_file)
        else:
            state = _run_stage(name, fn, state, config, **kwargs)

        if name in state.stage_errors:
            progress_bar.progress((i + 1) / total, text=f"❌ Failed at: {name}")
            status_box.error(f"Pipeline stopped at **{name}**. See error above.")
            break

    else:
        progress_bar.progress(1.0, text="✅ Pipeline complete!")
        status_box.success("All pipeline stages completed successfully!")

    st.session_state["pipeline_state"] = state

state = st.session_state["pipeline_state"]

# ── Render selected section ───────────────────────────────────────────────────
section_name = selected

if section_name == "Data Overview":
    dashboard.render_data_overview(state)

elif section_name == "Reliability Scores":
    dashboard.render_reliability_scores(state)

elif section_name == "Trust Graph":
    dashboard.render_trust_graph(state)

elif section_name == "Model Performance":
    dashboard.render_model_performance(state)

elif section_name == "Explainability":
    dashboard.render_explainability(state)

elif section_name == "Before vs After":
    dashboard.render_before_vs_after(state)

# ── Downloads (always visible at bottom) ─────────────────────────────────────
st.markdown("---")
st.markdown("### ⬇️ Downloads")
dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    if state.trained_models and state.best_model_name:
        buf = io.BytesIO()
        joblib.dump(state.trained_models[state.best_model_name], buf)
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Download Best Model ({state.best_model_name})",
            data=buf,
            file_name=f"{state.best_model_name.lower()}_model.joblib",
            mime="application/octet-stream",
        )

with dl_col2:
    def _safe(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    report = {
        "reliability_score": _safe(state.reliability_score),
        "confidence_score": _safe(state.confidence_score),
        "fusion_weights": {
            f"w{i+1}": _safe(float(w))
            for i, w in enumerate(state.fusion_weights if state.fusion_weights is not None else [])
        },
        "feature_trust": {k: _safe(v) for k, v in (state.feature_trust or {}).items()},
        "shap_importance": {k: _safe(v) for k, v in (state.shap_feature_importance or {}).items()},
        "high_importance_low_trust_features": state.high_importance_low_trust_features,
        "model_scores": state.model_scores,
        "best_model": state.best_model_name,
    }
    st.download_button(
        label="⬇️ Download Explainability Report (JSON)",
        data=json.dumps(report, indent=2, default=str),
        file_name="reliability_report.json",
        mime="application/json",
    )
