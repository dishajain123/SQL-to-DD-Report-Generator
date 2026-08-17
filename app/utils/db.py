"""Architecture step 19: Persistence / Audit.

SQLite-backed job history, structural/output validation results, and human
decisions. SQLite is deliberately used rather than a heavier DB — this is a
single-process pipeline and the schema is small; swapping to Postgres later
only requires changing the connection string in a real deployment.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.utils.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    platform TEXT NOT NULL,
    intent TEXT NOT NULL,
    status TEXT NOT NULL,
    run_number INTEGER,
    report_path TEXT,
    excel_path TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dd_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    chain_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    entity_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    derivation_option TEXT NOT NULL,
    expression TEXT,
    effective_start_date TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    validation_errors TEXT,
    row_json TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dd_row_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    edited_expression TEXT,
    reviewer TEXT NOT NULL,
    comment TEXT,
    decided_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dd_row_id) REFERENCES dd_rows(id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    detail TEXT NOT NULL,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _ensure_job_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
    }
    for column_name, column_type in (
        ("run_number", "INTEGER"),
        ("report_path", "TEXT"),
        ("excel_path", "TEXT"),
        ("error_message", "TEXT"),
    ):
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}")

    needs_backfill = "run_number" not in existing_columns
    if not needs_backfill:
        null_count = conn.execute("SELECT COUNT(*) AS count FROM jobs WHERE run_number IS NULL").fetchone()["count"]
        needs_backfill = bool(null_count)

    if needs_backfill:
        rows = conn.execute("SELECT job_id FROM jobs ORDER BY created_at, job_id").fetchall()
        for idx, row in enumerate(rows, start=1):
            conn.execute("UPDATE jobs SET run_number = ? WHERE job_id = ?", (idx, row["job_id"]))


@contextmanager
def get_connection(db_path: str | None = None):
    path = db_path or settings.sqlite_db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA)
        _ensure_job_columns(conn)


def record_job(job_id: str, company: str, platform: str, intent: str, status: str, db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        existing = conn.execute("SELECT run_number FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if existing and existing["run_number"] is not None:
            run_number = existing["run_number"]
        else:
            run_number = conn.execute("SELECT COALESCE(MAX(run_number), 0) + 1 AS next_run FROM jobs").fetchone()[
                "next_run"
            ]
        conn.execute(
            "INSERT INTO jobs (job_id, company, platform, intent, status, run_number) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET status=excluded.status, updated_at=CURRENT_TIMESTAMP",
            (job_id, company, platform, intent, status, run_number),
        )


def update_job_status(
    job_id: str,
    status: str,
    db_path: str | None = None,
    *,
    report_path: str | None = None,
    excel_path: str | None = None,
    error_message: str | None = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                report_path = COALESCE(?, report_path),
                excel_path = COALESCE(?, excel_path),
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (status, report_path, excel_path, error_message, job_id),
        )


def record_dd_row(job_id: str, chain_id: str, row_index: int, dd_row_dict: dict, db_path: str | None = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO dd_rows (job_id, chain_id, row_index, entity_name, column_name, "
            "derivation_option, expression, effective_start_date, status, confidence, "
            "validation_errors, row_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                chain_id,
                row_index,
                dd_row_dict["entity_name"],
                dd_row_dict["column_name"],
                dd_row_dict["derivation_option"],
                dd_row_dict.get("display_derivation_expression", ""),
                str(dd_row_dict["effective_start_date"]),
                dd_row_dict["status"],
                dd_row_dict["confidence"],
                json.dumps(dd_row_dict.get("validation_errors", [])),
                json.dumps(dd_row_dict, default=str),
            ),
        )
        return cur.lastrowid


def get_pending_review_rows(db_path: str | None = None) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT * FROM dd_rows WHERE status = 'PENDING_REVIEW' ORDER BY id")
        return cur.fetchall()


def get_dd_rows_for_job(job_id: str, db_path: str | None = None) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM dd_rows WHERE job_id = ? ORDER BY row_index, id",
            (job_id,),
        )
        return cur.fetchall()


def get_job(job_id: str, db_path: str | None = None) -> sqlite3.Row | None:
    with get_connection(db_path) as conn:
        return conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()


def get_job_output_dir(job_id: str, db_path: str | None = None) -> Path:
    job = get_job(job_id, db_path=db_path)
    if not job:
        return Path(settings.output_dir) / job_id

    run_number = job["run_number"]
    if run_number is None:
        return Path(settings.output_dir) / job_id
    return Path(settings.output_dir) / f"{int(run_number):03d}_{job_id}"


def record_review_decision(
    dd_row_id: int, action: str, reviewer: str, edited_expression: str | None = None,
    comment: str = "", db_path: str | None = None
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO review_decisions (dd_row_id, action, edited_expression, reviewer, comment) "
            "VALUES (?, ?, ?, ?, ?)",
            (dd_row_id, action, edited_expression, reviewer, comment),
        )
        new_status = "ACTIVE" if action in ("APPROVE", "EDIT", "OVERRIDE") else "INACTIVE"
        if action == "EDIT" and edited_expression:
            conn.execute(
                "UPDATE dd_rows SET status = ?, expression = ? WHERE id = ?",
                (new_status, edited_expression, dd_row_id),
            )
        else:
            conn.execute("UPDATE dd_rows SET status = ? WHERE id = ?", (new_status, dd_row_id))


def log_audit(job_id: str, stage: str, detail: str, db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO audit_log (job_id, stage, detail) VALUES (?, ?, ?)",
            (job_id, stage, detail),
        )
