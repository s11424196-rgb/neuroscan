"""Report generation service (PDF and CSV).

Generates professional clinical-style reports containing scan images,
prediction results, Grad-CAM heatmaps, and mandatory research disclaimers.
"""

import csv
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

from src.api.models.schemas import PredictionResult

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "DISCLAIMER: This tool is for research and educational use only and is "
    "NOT a substitute for professional medical diagnosis. Results should be "
    "reviewed by a qualified healthcare professional before any clinical "
    "decision-making."
)


class ReportService:
    """Generate downloadable PDF and CSV reports."""

    # ── PDF Generation ────────────────────────────────────────────

    async def generate_pdf(
        self,
        scan_id: str,
        filename: str,
        results: Dict[str, PredictionResult],
        heatmap_overlay_b64: Optional[str] = None,
        model_version: str = "unknown",
    ) -> bytes:
        """Build a PDF report and return it as raw bytes.

        Parameters
        ----------
        scan_id:
            Unique scan identifier.
        filename:
            Original scan filename (sanitised).
        results:
            Mapping of task name to ``PredictionResult``.
        heatmap_overlay_b64:
            Optional base64-encoded heatmap overlay PNG.
        model_version:
            Version string for the model used.

        Returns
        -------
        bytes
            PDF file content.
        """
        try:
            from fpdf import FPDF
        except ImportError:
            logger.warning(
                "fpdf2 not installed — falling back to plain-text pseudo-PDF."
            )
            return self._generate_text_report(
                scan_id, filename, results, model_version
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.add_page()

        # ── Header ────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Brain MRI AI Detection Report", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0,
            6,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ln=True,
            align="C",
        )
        pdf.ln(8)

        # ── Scan Metadata ─────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Scan Information", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Scan ID:  {scan_id}", ln=True)
        pdf.cell(0, 6, f"Filename: {filename}", ln=True)
        pdf.cell(0, 6, f"Model Version: {model_version}", ln=True)
        pdf.ln(6)

        # ── Results Table ─────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Prediction Results", ln=True)
        pdf.set_font("Helvetica", "", 10)

        for task, res in results.items():
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, f"Task: {task.title()}", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"  Predicted Class: {res.predicted_class}", ln=True)
            pdf.cell(0, 6, f"  Confidence:      {res.confidence:.2%}", ln=True)
            pdf.cell(
                0, 6, f"  Uncertainty:     {res.uncertainty_score:.4f}", ln=True
            )
            if res.is_low_confidence:
                pdf.set_text_color(200, 0, 0)
                pdf.cell(
                    0,
                    6,
                    "  ⚠ LOW CONFIDENCE — interpret with caution",
                    ln=True,
                )
                pdf.set_text_color(0, 0, 0)

            pdf.cell(0, 6, "  Class Probabilities:", ln=True)
            for cls, prob in res.probabilities.items():
                pdf.cell(0, 6, f"    {cls}: {prob:.4f}", ln=True)
            pdf.ln(3)

        # ── Heatmap (if available) ────────────────────────────────
        if heatmap_overlay_b64:
            import base64
            import tempfile

            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Grad-CAM Heatmap Overlay", ln=True)
            img_data = base64.b64decode(heatmap_overlay_b64)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            pdf.image(tmp_path, x=30, w=150)
            pdf.ln(6)

        # ── Disclaimer ────────────────────────────────────────────
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(180, 0, 0)
        pdf.multi_cell(0, 5, DISCLAIMER)
        pdf.set_text_color(0, 0, 0)

        return pdf.output()

    # ── CSV Generation ────────────────────────────────────────────

    async def generate_csv(
        self,
        rows: List[Dict],
    ) -> str:
        """Build a CSV string from a list of result dictionaries.

        Parameters
        ----------
        rows:
            Each dict should contain keys like ``scan_id``, ``task``,
            ``predicted_class``, ``confidence``, etc.

        Returns
        -------
        str
            CSV-formatted string.
        """
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    # ── Fallback ──────────────────────────────────────────────────

    @staticmethod
    def _generate_text_report(
        scan_id: str,
        filename: str,
        results: Dict[str, PredictionResult],
        model_version: str,
    ) -> bytes:
        """Plain-text fallback when fpdf2 is not installed."""
        lines = [
            "=" * 60,
            "Brain MRI AI Detection Report",
            "=" * 60,
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Scan ID:   {scan_id}",
            f"Filename:  {filename}",
            f"Model:     {model_version}",
            "",
        ]
        for task, res in results.items():
            lines.append(f"--- {task.title()} ---")
            lines.append(f"Predicted: {res.predicted_class}")
            lines.append(f"Confidence: {res.confidence:.2%}")
            lines.append(f"Uncertainty: {res.uncertainty_score:.4f}")
            for cls, prob in res.probabilities.items():
                lines.append(f"  {cls}: {prob:.4f}")
            lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines).encode()
