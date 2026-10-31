"""
Results display page for NeuroScan AI.

Shows detailed prediction results for the most recent analysis, including
heatmap overlay with adjustable opacity, confidence bars, uncertainty
badges, and download buttons for PDF reports and CSV exports.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import streamlit as st
from PIL import Image

from src.frontend.components.disclaimer import (
    show_disclaimer_banner,
    show_result_disclaimer,
)
from src.frontend.components.report_generator import (
    generate_csv_results,
    generate_pdf_report,
)
from src.frontend.components.visualizations import (
    show_confidence_bar,
    show_heatmap_overlay,
    show_probability_chart,
    show_scan_viewer,
    show_uncertainty_badge,
)


def render() -> None:
    """Render the results page."""
    show_disclaimer_banner()

    st.title("📊 Analysis Results")

    latest: List[Dict[str, Any]] | None = st.session_state.get("latest_results")

    if not latest:
        st.info(
            "No results to display yet. Head over to **Upload & Analyze** "
            "to run your first scan.",
            icon="📂",
        )
        if st.button("Go to Upload & Analyze"):
            st.session_state["nav_page"] = "Upload & Analyze"
            st.rerun()
        return

    show_result_disclaimer()

    is_batch = len(latest) > 1

    # ── Batch summary table ─────────────────────────────────────────────
    if is_batch:
        st.subheader("📑 Batch Summary")
        summary_rows = []
        for r in latest:
            if "results" in r:
                for sub in r["results"]:
                    summary_rows.append(
                        {
                            "File": r.get("filename", ""),
                            "Analysis": sub.get("analysis_type", ""),
                            "Prediction": sub.get("predicted_class", ""),
                            "Confidence": f"{sub.get('confidence', 0) * 100:.1f}%",
                        }
                    )
            else:
                summary_rows.append(
                    {
                        "File": r.get("filename", ""),
                        "Analysis": r.get("analysis_type", ""),
                        "Prediction": r.get("predicted_class", ""),
                        "Confidence": f"{r.get('confidence', 0) * 100:.1f}%",
                    }
                )
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        st.divider()

    # ── Individual results ──────────────────────────────────────────────
    for idx, result in enumerate(latest):
        filename = result.get("filename", f"Scan {idx + 1}")

        if is_batch:
            with st.expander(f"📄 {filename}", expanded=(idx == 0)):
                _render_full_result(result, idx)
        else:
            _render_full_result(result, idx)

    # ── Download buttons ────────────────────────────────────────────────
    st.divider()
    st.subheader("⬇️ Download")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        # PDF for first result (or single result)
        first = latest[0]
        image = first.get("image")
        if "results" in first:
            target = first["results"][0]
        else:
            target = first

        heatmap_arr = target.get("heatmap")
        heatmap_img = None
        if image is not None and heatmap_arr is not None:
            heatmap_img = show_heatmap_overlay(image, heatmap_arr, 0.4)

        target_for_pdf = {**target, "filename": first.get("filename", "scan")}
        try:
            pdf_bytes = generate_pdf_report(
                target_for_pdf,
                image or Image.new("RGB", (224, 224), (128, 128, 128)),
                heatmap_img,
            )
            st.download_button(
                "📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"neuroscan_report_{first.get('filename', 'scan')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.warning(
                "PDF generation requires the `fpdf2` package. "
                "Install with: `pip install fpdf2`"
            )

    with dl_col2:
        if is_batch:
            flat_results = _flatten_results(latest)
            csv_str = generate_csv_results(flat_results)
            st.download_button(
                "📥 Download CSV Results",
                data=csv_str,
                file_name="neuroscan_batch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("CSV export is available for batch analyses.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten nested 'both' results into a simple list."""
    flat: List[Dict[str, Any]] = []
    for r in results:
        if "results" in r:
            for sub in r["results"]:
                flat.append({**sub, "filename": r.get("filename", "")})
        else:
            flat.append(r)
    return flat


def _render_full_result(result: Dict[str, Any], idx: int) -> None:
    """Render full detail for one result (may contain sub-results for 'both')."""
    image = result.get("image")

    if "results" in result:
        for sub_idx, sub in enumerate(result["results"]):
            _render_single_detail(sub, image, f"{idx}_{sub_idx}")
    else:
        _render_single_detail(result, image, str(idx))


def _render_single_detail(
    result: Dict[str, Any],
    image: Image.Image | None,
    key_suffix: str,
) -> None:
    """Render detailed prediction card for a single analysis result."""
    atype = result.get("analysis_type", "unknown")
    label = "🔬 Tumour Detection" if atype == "tumour" else "🧩 Alzheimer's Screening"
    st.markdown(f"#### {label}")

    col_img, col_heat = st.columns(2)

    heatmap_arr = result.get("heatmap")

    with col_img:
        if image is not None:
            show_scan_viewer(image, caption="Original Scan")
        else:
            st.info("Image preview not available.")

    with col_heat:
        if image is not None and heatmap_arr is not None:
            alpha = st.slider(
                "Heatmap Opacity",
                0.0,
                1.0,
                0.4,
                0.05,
                key=f"result_alpha_{key_suffix}",
            )
            overlay = show_heatmap_overlay(image, heatmap_arr, alpha)
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
        else:
            st.info("Heatmap not available.")

    # Prediction card
    predicted = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence", 0.0)
    probabilities = result.get("probabilities", {})

    st.markdown(
        f"""
        <div style="
            background: #F0F9FF;
            border-left: 5px solid #0077B6;
            border-radius: 8px;
            padding: 18px;
            margin: 10px 0;
        ">
            <h4 style="color: #023E8A; margin: 0 0 4px 0;">
                Predicted: {predicted}
            </h4>
            <span style="color: #4A6FA5;">
                Confidence: {confidence * 100:.1f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_uncertainty_badge(confidence)

    if probabilities:
        st.markdown("**All Class Probabilities:**")
        # Show top-2 highlighted
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for i, (cls, prob) in enumerate(sorted_probs):
            show_confidence_bar(cls, prob, is_primary=(i == 0))

    # Model info
    model_info = result.get("model_info", {})
    if model_info:
        with st.expander("ℹ️ Model Information"):
            st.markdown(
                f"- **Architecture:** {model_info.get('architecture', 'N/A')}\n"
                f"- **Version:** {model_info.get('version', 'N/A')}\n"
                f"- **Mode:** {model_info.get('mode', 'production')}"
            )

    st.divider()
