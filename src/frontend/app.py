"""
NeuroScan AI — Brain MRI Analysis Application.

Main Streamlit application entry point providing sidebar navigation between:
- Home: System overview, capabilities, and clinical disclaimer
- Upload & Analyze: Interactive single/batch MRI scan upload and real-time inference
- Results History: Session scan tracking, review, and CSV export
- About & Model Cards: Scientific background, limitations, and ethical guidelines
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so src imports work cleanly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from src.frontend.components.disclaimer import show_disclaimer_banner
from src.frontend.pages import about, history, home, results, upload

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI — Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True

# ---------------------------------------------------------------------------
# Global Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Professional Medical AI theme */
    .main-header {
        color: #0077B6;
        font-weight: 700;
    }
    .disclaimer-box {
        background-color: #FFF3CD;
        border-left: 6px solid #FFC107;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-confidence-high {
        background-color: #D1E7DD;
        color: #0F5132;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-confidence-low {
        background-color: #F8D7DA;
        color: #842029;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/visibility/materialicons/48dp/2x/baseline_visibility_black_48dp.png",
        width=48,
    )
    st.title("🧠 NeuroScan AI")
    st.caption("AI-Assisted Brain MRI Triage & Staging")
    st.divider()

    page = st.radio(
        "Navigation",
        options=["Home", "Upload & Analyze", "Results View", "Scan History", "About & Ethics"],
        index=0,
    )

    st.divider()

    # Demo Mode Switcher
    demo_toggle = st.toggle("Demo Mode (Offline Mock)", value=st.session_state.demo_mode)
    st.session_state.demo_mode = demo_toggle
    if demo_toggle:
        st.info("ℹ️ Running in Demo Mode with synthetic heatmaps & offline inference.")
    else:
        st.success("🌐 Connected to live FastAPI backend.")

    st.divider()
    st.markdown(
        "<small style='color: #6c757d;'>"
        "NeuroScan AI v0.1.0<br>"
        "Research & Educational Use Only<br>"
        "Not a diagnostic device."
        "</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------
if page == "Home":
    home.render()
elif page == "Upload & Analyze":
    upload.render()
elif page == "Results View":
    results.render()
elif page == "Scan History":
    history.render()
elif page == "About & Ethics":
    about.render()

