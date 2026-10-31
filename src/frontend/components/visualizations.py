"""
Visualisation components for NeuroScan AI.

Provides reusable Streamlit components for displaying brain MRI scans,
Grad-CAM heatmap overlays, confidence bars, probability charts, and
uncertainty badges.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image


# ---------------------------------------------------------------------------
# Scan viewer
# ---------------------------------------------------------------------------

def show_scan_viewer(image: Image.Image, caption: str = "MRI Scan") -> None:
    """Display an MRI scan image with zoom capability.

    Args:
        image: PIL Image of the scan.
        caption: Caption text displayed below the image.
    """
    st.image(image, caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------

def show_heatmap_overlay(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on top of the original scan image.

    Args:
        original: Original MRI scan as a PIL Image.
        heatmap: 2-D numpy array (values 0-255) representing the heatmap.
        alpha: Opacity of the heatmap overlay (0 = transparent, 1 = opaque).

    Returns:
        Blended PIL Image with heatmap overlay.
    """
    # Ensure original is RGB
    orig_rgb = original.convert("RGB")
    orig_arr = np.array(orig_rgb, dtype=np.float32)

    # Resize heatmap to match original image dimensions
    heatmap_img = Image.fromarray(heatmap.astype(np.uint8)).resize(
        orig_rgb.size, Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_img, dtype=np.float32)

    # Apply a colour map: blue (low) → red (high)
    heatmap_coloured = np.zeros((*heatmap_arr.shape, 3), dtype=np.float32)
    normalised = heatmap_arr / 255.0
    # Red channel increases with heat
    heatmap_coloured[..., 0] = normalised * 255
    # Green channel peaks in middle
    heatmap_coloured[..., 1] = (1.0 - np.abs(normalised - 0.5) * 2) * 180
    # Blue channel decreases with heat
    heatmap_coloured[..., 2] = (1.0 - normalised) * 255

    # Blend
    blended = (1 - alpha) * orig_arr + alpha * heatmap_coloured
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Confidence bar
# ---------------------------------------------------------------------------

def show_confidence_bar(
    class_name: str,
    confidence: float,
    is_primary: bool = False,
) -> None:
    """Display a styled horizontal confidence bar for a prediction class.

    Args:
        class_name: Name of the predicted class.
        confidence: Confidence score between 0 and 1.
        is_primary: If True, use a bolder style to highlight the top
            prediction.
    """
    pct = confidence * 100
    if is_primary:
        bar_colour = "#0077B6"
        bg_colour = "#D4ECFA"
        font_weight = "700"
        height = "28px"
    else:
        bar_colour = "#48CAE4"
        bg_colour = "#EAF6FB"
        font_weight = "400"
        height = "20px"

    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between;
                        font-weight: {font_weight}; font-size: 0.9rem;
                        margin-bottom: 3px;">
                <span>{class_name}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background: {bg_colour}; border-radius: 6px;
                        overflow: hidden; height: {height};">
                <div style="width: {pct}%; background: {bar_colour};
                            height: 100%; border-radius: 6px;
                            transition: width 0.6s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Probability chart
# ---------------------------------------------------------------------------

def show_probability_chart(probabilities: Dict[str, float]) -> None:
    """Display a horizontal bar chart of class probabilities.

    Args:
        probabilities: Mapping of class names to probability scores (0-1).
    """
    # Sort descending
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for idx, (cls_name, prob) in enumerate(sorted_probs):
        show_confidence_bar(cls_name, prob, is_primary=(idx == 0))


# ---------------------------------------------------------------------------
# Uncertainty badge
# ---------------------------------------------------------------------------

def show_uncertainty_badge(confidence: float, threshold: float = 0.7) -> None:
    """Display a confidence/uncertainty badge.

    Shows a green "High Confidence" badge when the confidence exceeds the
    threshold, or an amber "Low Confidence" badge with a human review
    recommendation otherwise.

    Args:
        confidence: Top-class confidence score (0-1).
        threshold: Minimum confidence for the "high" badge.
    """
    if confidence >= threshold:
        st.markdown(
            """
            <div style="
                display: inline-block;
                background: #D4EDDA;
                color: #155724;
                border: 1px solid #C3E6CB;
                border-radius: 20px;
                padding: 6px 18px;
                font-weight: 600;
                font-size: 0.9rem;
            ">✅ High Confidence</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                display: inline-block;
                background: #FFF3CD;
                color: #856404;
                border: 1px solid #FFEEBA;
                border-radius: 20px;
                padding: 6px 18px;
                font-weight: 600;
                font-size: 0.9rem;
            ">⚠️ Low Confidence — Human Review Recommended</div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "The model's confidence is below the recommended threshold. "
            "This result should be reviewed by a qualified medical "
            "professional before any interpretation is drawn."
        )


# ---------------------------------------------------------------------------
# Confusion matrix (evaluation helper)
# ---------------------------------------------------------------------------

def show_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
) -> None:
    """Render a confusion matrix as a styled HTML table.

    Args:
        cm: Square numpy array of shape (n_classes, n_classes).
        class_names: List of class labels corresponding to rows/columns.
    """
    n = len(class_names)
    header = "<th></th>" + "".join(
        f"<th style='padding:6px 10px; color:#0077B6;'>{cn}</th>"
        for cn in class_names
    )
    rows = ""
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(n):
        cells = f"<td style='font-weight:600; color:#0077B6;'>{class_names[i]}</td>"
        for j in range(n):
            intensity = int(200 - (cm[i, j] / max_val) * 160)
            bg = f"rgb({intensity},{intensity+30},{intensity+55})" if i == j else f"rgb(255,{intensity},{intensity})"
            if i == j:
                bg = f"rgb({intensity}, {min(255,intensity+55)}, {min(255,intensity+55)})"
            cells += (
                f"<td style='text-align:center; padding:8px; "
                f"background:{bg}; border-radius:4px;'>{cm[i, j]}</td>"
            )
        rows += f"<tr>{cells}</tr>"
    st.markdown(
        f"""
        <table style="border-collapse: separate; border-spacing: 3px;
                      margin: 10px 0; font-size: 0.85rem;">
            <tr>{header}</tr>
            {rows}
        </table>
        """,
        unsafe_allow_html=True,
    )
