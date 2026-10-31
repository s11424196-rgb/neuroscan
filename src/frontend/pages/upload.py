"""
Upload & Analyze page for NeuroScan AI.

Provides a three-step workflow:
  1. Select analysis type
  2. Upload MRI scan file(s)
  3. Run analysis (via backend API or local demo-mode mock inference)

Supports batch uploads and displays inline results upon completion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import streamlit as st
from PIL import Image

from src.frontend.components.disclaimer import show_disclaimer_banner
from src.frontend.components.visualizations import (
    show_confidence_bar,
    show_heatmap_overlay,
    show_probability_chart,
    show_scan_viewer,
    show_uncertainty_badge,
)
from src.frontend.components.disclaimer import show_result_disclaimer
from src.frontend.utils.api_client import (
    is_backend_available,
    run_prediction,
    upload_scan,
)
from src.frontend.utils.mock_inference import mock_predict


# ---------------------------------------------------------------------------
# Accepted file extensions
# ---------------------------------------------------------------------------

ACCEPTED_EXTENSIONS = ["dcm", "nii", "nii.gz", "jpg", "jpeg", "png"]
ACCEPTED_MIME = [
    "image/jpeg",
    "image/png",
    "application/dicom",
    "application/octet-stream",
]

ANALYSIS_OPTIONS = {
    "Brain Tumour Detection": "tumour",
    "Alzheimer's Screening": "alzheimer",
    "Both (Tumour + Alzheimer's)": "both",
}


# ---------------------------------------------------------------------------
# Helper: load image from uploaded file
# ---------------------------------------------------------------------------

def _load_image(uploaded_file) -> Image.Image:
    """Convert an uploaded file to a PIL Image.

    For DICOM / NIfTI files a simple grey placeholder is returned since full
    medical-format parsing requires optional heavy dependencies.
    """
    name = uploaded_file.name.lower()
    if name.endswith((".jpg", ".jpeg", ".png")):
        return Image.open(uploaded_file).convert("RGB")

    # For medical formats, return a placeholder (full parsing handled server-side)
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        # Create a grey placeholder indicating the scan was received
        placeholder = Image.fromarray(
            np.full((224, 224), 128, dtype=np.uint8), mode="L"
        )
        return placeholder.convert("RGB")


# ---------------------------------------------------------------------------
# Helper: process single file
# ---------------------------------------------------------------------------

def _process_single(
    uploaded_file,
    analysis_type: str,
    demo_mode: bool,
) -> Dict[str, Any]:
    """Analyse one uploaded file.

    Returns a result dict suitable for session-state history.
    """
    image = _load_image(uploaded_file)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if demo_mode:
        result = mock_predict(image, analysis_type)
    else:
        file_bytes = uploaded_file.getvalue()
        scan_id = upload_scan(file_bytes, uploaded_file.name)
        result = run_prediction(scan_id, analysis_type)

    # Normalise result structure
    if "results" in result:
        # "both" mode returns a list
        for sub in result["results"]:
            sub["filename"] = uploaded_file.name
            sub["timestamp"] = timestamp
            sub["image"] = image
        result["filename"] = uploaded_file.name
        result["timestamp"] = timestamp
        result["image"] = image
    else:
        result["filename"] = uploaded_file.name
        result["timestamp"] = timestamp
        result["image"] = image

    return result


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the upload and analysis page."""
    show_disclaimer_banner()

    st.title("📤 Upload & Analyze")
    st.markdown(
        "Follow the steps below to analyse your brain MRI scan."
    )

    # ── Demo-mode toggle ────────────────────────────────────────────────
    backend_up = is_backend_available()
    if not backend_up:
        st.info(
            "🔌 **Backend not detected** — Demo Mode is enabled automatically. "
            "Predictions are synthetic and for demonstration only.",
            icon="ℹ️",
        )

    demo_mode = st.toggle(
        "🧪 Demo Mode (synthetic predictions)",
        value=not backend_up,
        help="Use local mock inference instead of the backend API.",
    )

    st.divider()

    # ── Step 1: Analysis type ───────────────────────────────────────────
    st.subheader("Step 1 — Select Analysis Type")
    analysis_label = st.radio(
        "Choose the type of analysis:",
        list(ANALYSIS_OPTIONS.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    analysis_type: str = ANALYSIS_OPTIONS[analysis_label]

    st.divider()

    # ── Step 2: File upload ─────────────────────────────────────────────
    st.subheader("Step 2 — Upload MRI Scan(s)")
    uploaded_files = st.file_uploader(
        "Drag and drop or browse files",
        type=ACCEPTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Accepted formats: DICOM (.dcm), NIfTI (.nii, .nii.gz), JPEG, PNG. Max 200 MB.",
    )

    if uploaded_files:
        n_files = len(uploaded_files)
        st.success(f"📁 {n_files} file{'s' if n_files > 1 else ''} uploaded")

        # Show preview + file info
        preview_cols = st.columns(min(n_files, 4))
        for idx, uf in enumerate(uploaded_files[:4]):
            with preview_cols[idx % len(preview_cols)]:
                try:
                    img = _load_image(uf)
                    st.image(img, caption=uf.name, use_container_width=True)
                except Exception:
                    st.warning(f"Cannot preview: {uf.name}")
                size_kb = len(uf.getvalue()) / 1024
                st.caption(f"**{uf.name}** — {size_kb:.1f} KB")
        if n_files > 4:
            st.caption(f"… and {n_files - 4} more file(s)")

    st.divider()

    # ── Step 3: Analyse ─────────────────────────────────────────────────
    st.subheader("Step 3 — Run Analysis")

    if not uploaded_files:
        st.info("Upload one or more MRI scans above to begin analysis.")
        return

    if st.button("🔬  Analyze Scan(s)", type="primary", use_container_width=True):
        all_results: List[Dict[str, Any]] = []
        progress_bar = st.progress(0, text="Preparing…")

        for i, uf in enumerate(uploaded_files):
            progress_text = f"Analysing **{uf.name}** ({i + 1}/{len(uploaded_files)})…"
            progress_bar.progress(
                (i) / len(uploaded_files),
                text=progress_text,
            )

            try:
                result = _process_single(uf, analysis_type, demo_mode)
                all_results.append(result)
            except Exception as exc:
                st.error(
                    f"❌ Error processing **{uf.name}**: {exc}",
                    icon="🚨",
                )

        progress_bar.progress(1.0, text="✅ Analysis complete!")

        if not all_results:
            st.error("No results were produced. Please check the files and try again.")
            return

        # Persist to session state
        if "scan_history" not in st.session_state:
            st.session_state["scan_history"] = []
        st.session_state["scan_history"].extend(all_results)
        st.session_state["latest_results"] = all_results

        st.divider()

        # ── Display inline results ─────────────────────────────────────
        st.subheader("📋 Results")
        show_result_disclaimer()

        for res in all_results:
            _display_single_result(res)


def _display_single_result(result: Dict[str, Any]) -> None:
    """Render the results for one scan analysis inline."""
    filename = result.get("filename", "Unknown")

    # Handle "both" mode
    if "results" in result:
        st.markdown(f"### 📄 {filename}")
        for sub_result in result["results"]:
            _render_result_card(sub_result, result.get("image"))
        return

    st.markdown(f"### 📄 {filename}")
    _render_result_card(result, result.get("image"))


def _render_result_card(result: Dict[str, Any], image=None) -> None:
    """Render a prediction card with heatmap for one analysis type."""
    atype = result.get("analysis_type", "unknown")
    label = "🔬 Tumour Detection" if atype == "tumour" else "🧩 Alzheimer's Screening"
    st.markdown(f"#### {label}")

    col_img, col_heat = st.columns(2)

    # Heatmap overlay
    heatmap_arr = result.get("heatmap")
    overlay_img = None

    with col_img:
        if image is not None:
            show_scan_viewer(image, caption="Original Scan")

    with col_heat:
        if image is not None and heatmap_arr is not None:
            key = f"alpha_{result.get('filename', '')}_{atype}"
            alpha = st.slider(
                "Heatmap Opacity",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                key=key,
            )
            overlay_img = show_heatmap_overlay(image, heatmap_arr, alpha)
            st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)

    # Prediction details
    predicted_class = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence", 0.0)
    probabilities = result.get("probabilities", {})

    st.markdown(
        f"""
        <div style="
            background: #F0F9FF;
            border: 1px solid #B8DAEF;
            border-radius: 10px;
            padding: 18px;
            margin: 10px 0;
        ">
            <h4 style="color: #023E8A; margin-top:0;">
                Predicted: {predicted_class}
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_uncertainty_badge(confidence)

    if probabilities:
        st.markdown("**Class Probabilities:**")
        show_probability_chart(probabilities)

    # Model info
    model_info = result.get("model_info", {})
    if model_info:
        with st.expander("ℹ️ Model Information"):
            st.markdown(
                f"- **Architecture:** {model_info.get('architecture', 'N/A')}\n"
                f"- **Version:** {model_info.get('version', 'N/A')}\n"
                f"- **Mode:** {model_info.get('mode', 'production')}"
            )

    processing_time = result.get("processing_time")
    if processing_time:
        st.caption(f"⏱️ Processing time: {processing_time:.2f}s")

    st.divider()
