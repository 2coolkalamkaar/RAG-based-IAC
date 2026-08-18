"""
Job Store — SQLite persistence for approved Terraform runs.
============================================================
Separate from LangGraph's state.db to avoid schema conflicts.
Database path: jobs.db (project root)
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "jobs.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the jobs table if it doesn't exist. Called at server startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                thread_id   TEXT,
                created_at  TEXT NOT NULL,
                workflow    TEXT NOT NULL,
                prompt      TEXT NOT NULL,
                trust_score REAL,
                trust_label TEXT,
                files_json  TEXT NOT NULL
            )
        """)
        conn.commit()


def save_job(
    thread_id: str,
    prompt: str,
    workflow: str,
    trust_score: float | None,
    trust_label: str | None,
    files: dict[str, str],
) -> str:
    """
    Persist an approved Terraform run.

    Args:
        thread_id:   LangGraph thread_id (for cross-referencing state.db).
        prompt:      Original user request.
        workflow:    Workflow name e.g. 'HitL RAG', 'Advanced RAG'.
        trust_score: 0.0–1.0 trust score (None if not applicable).
        trust_label: Human-readable label e.g. 'High Trust'.
        files:       Dict mapping filename → HCL code.

    Returns:
        The UUID of the newly created job.
    """
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, thread_id, created_at, workflow, prompt, trust_score, trust_label, files_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, thread_id, now, workflow, prompt, trust_score, trust_label, json.dumps(files)),
        )
        conn.commit()
    return job_id


def load_all_jobs(limit: int = 50) -> list[dict]:
    """
    Return the most recent jobs, newest first.
    Files are NOT included to keep the list payload small.
    """
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, thread_id, created_at, workflow, prompt, trust_score, trust_label
            FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_job(job_id: str) -> dict | None:
    """
    Load a single job including the full Terraform files dict.
    Returns None if job_id not found.
    """
    init_db()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["files"] = json.loads(result.pop("files_json"))
    return result


def delete_job(job_id: str) -> bool:
    """Delete a job by ID. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
    return cursor.rowcount > 0
