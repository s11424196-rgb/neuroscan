"""
Home page for NeuroScan AI.

Displays a hero section, feature cards for Brain Tumour Detection and
Alzheimer's Screening, quick model statistics, and prominent disclaimer
information.
"""

from __future__ import annotations

import streamlit as st

from src.frontend.components.disclaimer import show_disclaimer_banner


def render() -> None:
    """Render the home / landing page."""
    show_disclaimer_banner()

    # ── Hero section ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 40px 20px 30px;
        ">
            <h1 style="
                color: #0077B6;
                font-size: 2.8rem;
                margin-bottom: 8px;
            ">🧠 NeuroScan AI</h1>
            <p style="
                font-size: 1.25rem;
                color: #4A6FA5;
                max-width: 680px;
                margin: 0 auto;
                line-height: 1.6;
            ">
                AI-assisted brain MRI analysis for <strong>tumour
                detection</strong> and <strong>Alzheimer's screening</strong>.
                Upload a scan, get instant predictions with Grad-CAM
                heatmap explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Feature cards ───────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #E8F4FD 0%, #D4ECFA 100%);
                border-radius: 14px;
                padding: 28px;
                min-height: 280px;
                border: 1px solid #B8DAEF;
            ">
                <h3 style="color: #023E8A; margin-top: 0;">
                    🔬 Brain Tumour Detection
                </h3>
                <p style="color: #1B2A4A; line-height: 1.6;">
                    Classifies brain MRI scans into four categories:
                </p>
                <ul style="color: #1B2A4A;">
                    <li><strong>Glioma</strong></li>
                    <li><strong>Meningioma</strong></li>
                    <li><strong>Pituitary Tumour</strong></li>
                    <li><strong>No Tumour</strong></li>
                </ul>
                <p style="color: #4A6FA5; font-size: 0.88rem;">
                    Powered by EfficientNet-B0 · Trained on 7,023 MRI images
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #E8F8F0 0%, #D0F0E0 100%);
                border-radius: 14px;
                padding: 28px;
                min-height: 280px;
                border: 1px solid #A8D8C0;
            ">
                <h3 style="color: #065F46; margin-top: 0;">
                    🧩 Alzheimer's Screening
                </h3>
                <p style="color: #1B2A4A; line-height: 1.6;">
                    Screens brain MRI scans for signs of cognitive decline:
                </p>
                <ul style="color: #1B2A4A;">
                    <li><strong>Non-Demented</strong></li>
                    <li><strong>Very Mild Dementia</strong></li>
                    <li><strong>Mild Dementia</strong></li>
                    <li><strong>Moderate Dementia</strong></li>
                </ul>
                <p style="color: #4A6FA5; font-size: 0.88rem;">
                    Powered by ResNet-50 · Trained on 6,400 MRI images
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick stats ─────────────────────────────────────────────────────
    st.subheader("📊 Model Quick Stats")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tumour Model Accuracy", "96.2%")
    s2.metric("Alzheimer Model Accuracy", "93.8%")
    s3.metric("Supported Formats", "6+")
    s4.metric("Avg Inference Time", "< 2 s")

    st.divider()

    # ── Get started button ──────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center; margin: 16px 0 8px;'>",
        unsafe_allow_html=True,
    )
    if st.button("🚀  Get Started — Upload a Scan", type="primary", use_container_width=True):
        st.session_state["nav_page"] = "Upload & Analyze"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── What this tool does / does NOT do ───────────────────────────────
    st.subheader("ℹ️ What This Tool Does")
    col_do, col_dont = st.columns(2)

    with col_do:
        st.markdown(
            """
            **✅ This tool CAN:**
            - Provide AI-generated predictions on brain MRI scans
            - Generate Grad-CAM heatmaps showing model attention areas
            - Offer a second-opinion screening aid for researchers
            - Produce downloadable reports for further discussion
            """
        )

    with col_dont:
        st.markdown(
            """
            **❌ This tool CANNOT:**
            - Replace professional medical diagnosis
            - Provide definitive clinical conclusions
            - Account for full patient history and context
            - Guarantee accuracy on all scan types / protocols
            """
        )

    # ── Full disclaimer ─────────────────────────────────────────────────
    with st.expander("📋 Full Disclaimer", expanded=False):
        st.markdown(
            """
            **Research & Educational Use Only**

            NeuroScan AI is an experimental tool developed for research and
            educational purposes. It uses deep-learning models that have been
            trained on publicly available datasets and have **not** been
            validated for clinical use.

            - Predictions may be **inaccurate, biased, or misleading**.
            - Results must **never** be used as the sole basis for medical
              decisions.
            - Always consult a qualified **radiologist or neurologist** for
              clinical interpretation.
            - The developers accept **no liability** for actions taken based
              on this tool's output.

            By using this tool you acknowledge and accept these limitations.
            """
        )
