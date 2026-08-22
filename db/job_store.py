"""
Job Store — SQLite persistence for approved Terraform runs.
============================================================
Separate from LangGraph's state.db to avoid schema conflicts.
Database path: jobs.db (project root)

Schema v3 additions:
  - trust_factors   TEXT   — JSON of {factor: weight_value} for trust breakdown
"""
import json
import sqlite3
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path

DB_PATH        = Path(__file__).parent.parent / "jobs.db"
WORKSPACES_DIR = Path(__file__).parent.parent / "workspaces"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema migration helper ───────────────────────────────────────────────────

def _add_column_if_missing(conn: sqlite3.Connection, col: str, col_type: str):
    """Idempotently add a column to the jobs table."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if col not in existing:
        conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")


def _add_control_table(conn: sqlite3.Connection):
    """Single-row control table for global circuit breaker flag."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS control (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    # Seed APPLY_PAUSED = false if not present
    conn.execute("""
        INSERT OR IGNORE INTO control (key, value) VALUES ('APPLY_PAUSED', 'false')
    """)


def init_db():
    """Create/migrate the jobs table. Called at server startup."""
    WORKSPACES_DIR.mkdir(exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id             TEXT PRIMARY KEY,
                thread_id      TEXT,
                created_at     TEXT NOT NULL,
                workflow       TEXT NOT NULL,
                prompt         TEXT NOT NULL,
                trust_score    REAL,
                trust_label    TEXT,
                trust_factors  TEXT,
                files_json     TEXT NOT NULL,
                workspace_path TEXT,
                apply_status   TEXT,
                apply_outputs  TEXT,
                plan_summary   TEXT,
                cost_estimate  REAL
            )
        """)
        # v2 migrations — idempotent for existing DBs
        for col, col_type in [
            ("workspace_path", "TEXT"),
            ("apply_status",   "TEXT"),
            ("apply_outputs",  "TEXT"),
            ("plan_summary",   "TEXT"),
            ("cost_estimate",  "REAL"),
            ("trust_factors",  "TEXT"),  # v3
        ]:
            _add_column_if_missing(conn, col, col_type)

        _add_control_table(conn)
        conn.commit()


# ── Workspace helpers ─────────────────────────────────────────────────────────

def create_workspace(job_id: str) -> str:
    """
    Create a persistent workspace directory for a job.
    Returns the absolute path as a string.
    The directory is: workspaces/{job_id}/
    """
    ws = WORKSPACES_DIR / job_id
    ws.mkdir(parents=True, exist_ok=True)
    return str(ws)


def get_workspace(job_id: str) -> str | None:
    """Return workspace path from DB for an existing job, or None."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT workspace_path FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return row["workspace_path"] if row else None


def delete_workspace(job_id: str):
    """Remove the workspace directory (call after a confirmed destroy)."""
    ws = WORKSPACES_DIR / job_id
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


# ── Circuit breaker ───────────────────────────────────────────────────────────

def is_apply_paused() -> bool:
    """Return True if the global circuit breaker is engaged."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM control WHERE key = 'APPLY_PAUSED'"
        ).fetchone()
    return row and row["value"].lower() == "true"


def set_apply_paused(paused: bool):
    """Engage (True) or disengage (False) the global circuit breaker."""
    init_db()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE control SET value = ? WHERE key = 'APPLY_PAUSED'",
            ("true" if paused else "false",),
        )
        conn.commit()


# ── Job CRUD ──────────────────────────────────────────────────────────────────

def save_job(
    thread_id: str,
    prompt: str,
    workflow: str,
    trust_score: float | None,
    trust_label: str | None,
    trust_factors: dict | None,
    files: dict[str, str],
    workspace_path: str | None = None,
) -> str:
    """
    Persist an approved Terraform run.

    Args:
        thread_id:      LangGraph thread_id (for cross-referencing state.db).
        prompt:         Original user request.
        workflow:       Workflow name e.g. 'hitl', 'advanced'.
        trust_score:    0.0-1.0 trust score (None if not applicable).
        trust_label:    Human-readable label e.g. 'High Trust'.
        files:          Dict mapping filename -> HCL code.
        workspace_path: Absolute path to persistent workspace dir (optional).

    Returns:
        The UUID of the newly created job.
    """
    init_db()
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    # If no workspace_path provided, create one
    if workspace_path is None:
        workspace_path = create_workspace(job_id)
        # Write files into workspace immediately
        for fname, content in files.items():
            (Path(workspace_path) / fname).write_text(content)

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs
                (id, thread_id, created_at, workflow, prompt, trust_score,
                 trust_label, trust_factors, files_json, workspace_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, thread_id, now, workflow, prompt,
                trust_score, trust_label,
                json.dumps(trust_factors) if trust_factors else None,
                json.dumps(files), workspace_path,
            ),
        )
        conn.commit()
    return job_id


def update_apply_status(
    job_id: str,
    status: str,
    outputs: dict | None = None,
):
    """
    Record the result of a terraform apply or destroy.

    Args:
        job_id:  The job UUID.
        status:  One of: 'applied', 'failed', 'destroyed'
        outputs: Dict from `terraform output -json` (None on failure/destroy).
    """
    with _get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET apply_status = ?, apply_outputs = ? WHERE id = ?",
            (status, json.dumps(outputs) if outputs else None, job_id),
        )
        conn.commit()


def update_plan_summary(
    job_id: str,
    plan_summary: dict,
    cost_estimate: float,
):
    """Store the plan summary and cost estimate on the job record."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET plan_summary = ?, cost_estimate = ? WHERE id = ?",
            (json.dumps(plan_summary), cost_estimate, job_id),
        )
        conn.commit()


def load_all_jobs(limit: int = 50) -> list[dict]:
    """
    Return the most recent jobs, newest first.
    Files are NOT included to keep the list payload small.
    """
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, thread_id, created_at, workflow, prompt, trust_score,
                   trust_label, workspace_path, apply_status, cost_estimate
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
    if result.get("trust_factors"):
        result["trust_factors"] = json.loads(result["trust_factors"])
    else:
        result["trust_factors"] = {}
    if result.get("apply_outputs"):
        result["apply_outputs"] = json.loads(result["apply_outputs"])
    if result.get("plan_summary"):
        result["plan_summary"] = json.loads(result["plan_summary"])
    return result


def delete_job(job_id: str) -> bool:
    """Delete a job by ID. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
    return cursor.rowcount > 0
