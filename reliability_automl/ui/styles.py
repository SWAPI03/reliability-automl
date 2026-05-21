"""Custom CSS styles for the Streamlit dashboard."""

CUSTOM_CSS = """
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: #1a1d27;
    border-right: 1px solid #2d2f3e;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Typography ── */
h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
p, li, label { color: #c0c4d0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #8b92a5 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #7c83fd !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

/* ── Cards (via st.container with border) ── */
.card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    margin: 4px;
}
.badge-green  { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade80; }
.badge-yellow { background: #3a3010; color: #facc15; border: 1px solid #facc15; }
.badge-red    { background: #3a1010; color: #f87171; border: 1px solid #f87171; }
.badge-blue   { background: #101a3a; color: #60a5fa; border: 1px solid #60a5fa; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #8b92a5 !important;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7c83fd !important;
    border-bottom: 2px solid #7c83fd !important;
    background: #1e2130 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2d3148;
}

/* ── Buttons ── */
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c83fd, #5b63f5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stButton"] button:hover { opacity: 0.85; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1e2130;
    border: 2px dashed #3d4168;
    border-radius: 14px;
    padding: 20px;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7c83fd, #a78bfa) !important;
    border-radius: 4px;
}

/* ── Divider ── */
hr { border-color: #2d3148 !important; }

/* ── Sidebar nav items ── */
.nav-item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 4px 0;
    cursor: pointer;
    color: #8b92a5;
    font-weight: 500;
    transition: background 0.15s;
}
.nav-item:hover, .nav-item.active {
    background: #2d3148;
    color: #7c83fd;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #2d3148;
    margin-bottom: 20px;
}
.section-icon {
    font-size: 1.4rem;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
}

/* ── Pipeline stage indicator ── */
.stage-complete { color: #4ade80; }
.stage-running  { color: #facc15; }
.stage-error    { color: #f87171; }
.stage-pending  { color: #4b5563; }
</style>
"""
