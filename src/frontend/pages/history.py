"""
Scan history page for NeuroScan AI.

Displays a table of all scans analysed during the current session, allows
exporting results to CSV, and provides navigation to view individual results.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.frontend.components.disclaimer import show_disclaimer_banner
from src.frontend.components.report_generator import generate_csv_results


def _flatten_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten scan history entries (handling 'both' mode sub-results)."""
    flat: List[Dict[str, Any]] = []
    for entry in history:
        if "results" in entry:
            for sub in entry["results"]:
                flat.append(
                    {
                        "filename": entry.get("filename", ""),
                        "timestamp": entry.get("timestamp", sub.get("timestamp", "")),
                        "analysis_type": sub.get("analysis_type", ""),
                        "predicted_class": sub.get("predicted_class", ""),
                        "confidence": sub.get("confidence", 0.0),
                    }
                )
        else:
            flat.append(
                {
                    "filename": entry.get("filename", ""),
                    "timestamp": entry.get("timestamp", ""),
                    "analysis_type": entry.get("analysis_type", ""),
                    "predicted_class": entry.get("predicted_class", ""),
                    "confidence": entry.get("confidence", 0.0),
                }
            )
    return flat


def render() -> None:
    """Render the scan history page."""
    show_disclaimer_banner()

    st.title("📜 Scan History")
    st.markdown("All analyses performed during this session are listed below.")

    history: List[Dict[str, Any]] = st.session_state.get("scan_history", [])

    if not history:
        st.info(
            "No scans have been analysed yet in this session. "
            "Go to **Upload & Analyze** to get started.",
            icon="📂",
        )
        if st.button("Go to Upload & Analyze"):
            st.session_state["nav_page"] = "Upload & Analyze"
            st.rerun()
        return

    flat = _flatten_history(history)

    # ── History table ───────────────────────────────────────────────────
    display_rows = []
    for row in flat:
        conf = row["confidence"]
        display_rows.append(
            {
                "Timestamp": row["timestamp"],
                "Filename": row["filename"],
                "Analysis Type": row["analysis_type"].title(),
                "Predicted Class": row["predicted_class"],
                "Confidence": f"{conf * 100:.1f}%",
            }
        )

    st.dataframe(
        display_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        },
    )

    st.caption(f"Total analyses: **{len(flat)}** across **{len(history)}** scan(s)")

    st.divider()

    # ── View result detail ──────────────────────────────────────────────
    st.subheader("🔍 View Detail")
    scan_names = [f"{i+1}. {h.get('filename', 'Scan')}" for i, h in enumerate(history)]
    selected = st.selectbox("Select a scan to view:", scan_names, index=0)
    if selected and st.button("View Full Results"):
        idx = scan_names.index(selected)
        st.session_state["latest_results"] = [history[idx]]
        st.session_state["nav_page"] = "Results"
        st.rerun()

    st.divider()

    # ── Export / Clear ──────────────────────────────────────────────────
    exp_col, clear_col = st.columns(2)

    with exp_col:
        csv_data = generate_csv_results(flat)
        st.download_button(
            "📥 Export All as CSV",
            data=csv_data,
            file_name="neuroscan_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with clear_col:
        if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
            st.session_state["scan_history"] = []
            st.session_state["latest_results"] = None
            st.rerun()
