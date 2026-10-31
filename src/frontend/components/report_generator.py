"""
Client-side report generator for NeuroScan AI.

Generates PDF reports and CSV exports from analysis results. Used in demo
mode or as a fallback when the backend report endpoint is unavailable.
Relies on ``fpdf2`` for PDF creation.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from src.frontend.components.disclaimer import get_pdf_disclaimer_text


def _ensure_fpdf():
    """Lazily import fpdf; raise helpful error if missing."""
    try:
        from fpdf import FPDF  # fpdf2 package
        return FPDF
    except ImportError:
        raise ImportError(
            "The 'fpdf2' package is required for PDF generation. "
            "Install it with:  pip install fpdf2"
        )


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def generate_pdf_report(
    result: Dict[str, Any],
    original_image: Image.Image,
    heatmap_image: Optional[Image.Image] = None,
) -> bytes:
    """Generate a single-scan PDF report.

    Args:
        result: Dictionary containing at least:
            - ``filename`` (str)
            - ``analysis_type`` (str)
            - ``predicted_class`` (str)
            - ``confidence`` (float)
            - ``probabilities`` (dict[str, float])
            - ``model_info`` (dict with ``architecture``, ``version``)
            - ``timestamp`` (str)
        original_image: The uploaded MRI scan as a PIL Image.
        heatmap_image: Optional blended heatmap overlay as a PIL Image.

    Returns:
        Raw PDF bytes ready for download.
    """
    FPDF = _ensure_fpdf()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 119, 182)
    pdf.cell(0, 12, "NeuroScan AI", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0, 7,
        f"Brain MRI Analysis Report  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)

    # ── Scan information ────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(27, 42, 74)
    pdf.cell(0, 8, "Scan Information", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    info_lines = [
        f"File: {result.get('filename', 'N/A')}",
        f"Analysis Type: {result.get('analysis_type', 'N/A')}",
        f"Timestamp: {result.get('timestamp', 'N/A')}",
    ]
    for line in info_lines:
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Scan image ──────────────────────────────────────────────────────
    def _save_pil_temp(img: Image.Image, label: str) -> str:
        """Save a PIL image to a temporary in-memory PNG and return path."""
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".png", prefix=f"neuroscan_{label}_")
        os.close(fd)
        img.convert("RGB").save(path)
        return path

    img_path = _save_pil_temp(original_image, "scan")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Original Scan", new_x="LMARGIN", new_y="NEXT")
    pdf.image(img_path, x=10, w=80)
    pdf.ln(3)

    if heatmap_image is not None:
        hm_path = _save_pil_temp(heatmap_image, "heatmap")
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Grad-CAM Heatmap Overlay", new_x="LMARGIN", new_y="NEXT")
        pdf.image(hm_path, x=10, w=80)
        pdf.ln(3)

    # ── Results ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(27, 42, 74)
    pdf.cell(0, 8, "Prediction Results", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 119, 182)
    predicted = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence", 0.0)
    pdf.cell(
        0, 7,
        f"Predicted Class: {predicted}  ({confidence * 100:.1f}%)",
        new_x="LMARGIN", new_y="NEXT",
    )

    probabilities = result.get("probabilities", {})
    if probabilities:
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for cls_name, prob in sorted(
            probabilities.items(), key=lambda x: x[1], reverse=True
        ):
            pdf.cell(0, 6, f"  {cls_name}: {prob * 100:.1f}%", new_x="LMARGIN", new_y="NEXT")

    # ── Model info ──────────────────────────────────────────────────────
    model_info = result.get("model_info", {})
    if model_info:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(27, 42, 74)
        pdf.cell(0, 7, "Model Information", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(
            0, 6,
            f"Architecture: {model_info.get('architecture', 'N/A')}",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.cell(
            0, 6,
            f"Version: {model_info.get('version', 'N/A')}",
            new_x="LMARGIN", new_y="NEXT",
        )

    # ── Disclaimer ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(180, 60, 20)
    pdf.cell(0, 10, "Disclaimer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 60, 20)
    pdf.multi_cell(0, 5, get_pdf_disclaimer_text())

    return pdf.output()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def generate_csv_results(results: List[Dict[str, Any]]) -> str:
    """Generate a CSV string from a list of prediction results.

    Args:
        results: List of result dictionaries. Each must contain at least
            ``filename``, ``analysis_type``, ``predicted_class``,
            ``confidence``, and ``timestamp``.

    Returns:
        CSV-formatted string.
    """
    output = io.StringIO()
    fieldnames = [
        "filename",
        "analysis_type",
        "predicted_class",
        "confidence",
        "timestamp",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow(
            {
                "filename": r.get("filename", ""),
                "analysis_type": r.get("analysis_type", ""),
                "predicted_class": r.get("predicted_class", ""),
                "confidence": f"{r.get('confidence', 0.0):.4f}",
                "timestamp": r.get("timestamp", ""),
            }
        )
    return output.getvalue()
