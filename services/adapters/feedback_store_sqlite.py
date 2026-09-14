"""
SQLite feedback store implementation.

Backend-agnostic - same schema can be used for HANA adapter later.

PII Rule: technician_id must be an SAP personnel ID, never a display name.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from services.interfaces import FeedbackEvent, FeedbackStore


class SqliteFeedbackStore(FeedbackStore):
    """SQLite-backed feedback store."""

    def __init__(self, db_path: str):
        """
        Initialize the feedback store.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create feedback table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_log (
                    feedback_id TEXT PRIMARY KEY,
                    query_issue_text TEXT NOT NULL,
                    suggested_notification_id TEXT,
                    action TEXT NOT NULL CHECK (action IN ('ACCEPT','OVERRIDE','REJECT')),
                    technician_id TEXT NOT NULL,
                    override_text TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    def log(self, event: FeedbackEvent) -> str:
        """
        Log a feedback event.
        
        Args:
            event: FeedbackEvent to log.
        
        Returns:
            The generated feedback_id.
        """
        feedback_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback_log 
                (feedback_id, query_issue_text, suggested_notification_id, action, technician_id, override_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    event.query_issue_text,
                    event.suggested_notification_id,
                    event.action,
                    event.technician_id,
                    event.override_text,
                    created_at,
                ),
            )
            conn.commit()
        
        return feedback_id

    def recent(self, limit: int = 50) -> list[dict]:
        """
        Retrieve recent feedback events.
        
        Args:
            limit: Maximum number of events to return.
        
        Returns:
            List of feedback event dicts, newest first.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT feedback_id, query_issue_text, suggested_notification_id,
                       action, technician_id, override_text, created_at
                FROM feedback_log
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]