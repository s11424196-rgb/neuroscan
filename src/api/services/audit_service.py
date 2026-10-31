"""Audit-logging service backed by SQLite.

Records every significant API action (uploads, predictions, report downloads)
into a lightweight SQLite database for traceability and compliance.
"""

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from src.api.core.config import settings
from src.api.models.schemas import AuditLogEntry

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    scan_id     TEXT,
    user        TEXT,
    details     TEXT
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action    ON audit_log (action);
CREATE INDEX IF NOT EXISTS idx_audit_scan_id   ON audit_log (scan_id);
"""


class AuditService:
    """Thin async wrapper around a SQLite-backed audit log.

    SQLite operations are run in an executor so they don't block the
    event loop.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        raw = db_path or settings.database_url
        # Strip the ``sqlite:///`` prefix if present
        self._db_path = raw.replace("sqlite:///", "")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── DB helpers ────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE_SQL + _CREATE_INDEX_SQL)
        logger.info("Audit database initialised at %s", self._db_path)

    # ── Public API ────────────────────────────────────────────────

    async def log(
        self,
        action: str,
        scan_id: Optional[str] = None,
        user: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Record an audit event.

        Parameters
        ----------
        action:
            Short action tag, e.g. ``upload``, ``predict``, ``download_report``.
        scan_id:
            Associated scan identifier (if applicable).
        user:
            Identifier of the acting user / API-key holder.
        details:
            Free-form details.  **Must not contain PHI.**
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._insert, action, scan_id, user, details
        )

    def _insert(
        self,
        action: str,
        scan_id: Optional[str],
        user: Optional[str],
        details: Optional[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (timestamp, action, scan_id, user, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), action, scan_id, user, details),
            )

    async def query(
        self,
        page: int = 1,
        page_size: int = 50,
        action: Optional[str] = None,
        scan_id: Optional[str] = None,
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[List[AuditLogEntry], int]:
        """Return paginated, filtered audit entries.

        Returns
        -------
        tuple
            ``(entries, total_count)``
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._query,
            page,
            page_size,
            action,
            scan_id,
            user,
            start_date,
            end_date,
        )

    def _query(
        self,
        page: int,
        page_size: int,
        action: Optional[str],
        scan_id: Optional[str],
        user: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Tuple[List[AuditLogEntry], int]:
        clauses: List[str] = []
        params: List[str] = []

        if action:
            clauses.append("action = ?")
            params.append(action)
        if scan_id:
            clauses.append("scan_id = ?")
            params.append(scan_id)
        if user:
            clauses.append("user = ?")
            params.append(user)
        if start_date:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("timestamp <= ?")
            params.append(end_date)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._connect() as conn:
            total: int = conn.execute(
                f"SELECT COUNT(*) FROM audit_log{where}", params
            ).fetchone()[0]

            rows = conn.execute(
                f"SELECT * FROM audit_log{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [page_size, (page - 1) * page_size],
            ).fetchall()

        entries = [
            AuditLogEntry(
                id=r["id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                action=r["action"],
                scan_id=r["scan_id"],
                user=r["user"],
                details=r["details"],
            )
            for r in rows
        ]
        return entries, total

    async def get_stats(self) -> Dict[str, int]:
        """Return aggregate counts useful for the admin dashboard."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_stats)

    def _get_stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            total_uploads = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE action = 'upload'"
            ).fetchone()[0]
            total_predictions = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE action = 'predict'"
            ).fetchone()[0]

            type_rows = conn.execute(
                "SELECT details, COUNT(*) as cnt FROM audit_log "
                "WHERE action = 'predict' GROUP BY details"
            ).fetchall()

        predictions_by_type = {r["details"] or "unknown": r["cnt"] for r in type_rows}

        return {
            "total_scans": total_uploads,
            "total_predictions": total_predictions,
            "predictions_by_type": predictions_by_type,
        }
