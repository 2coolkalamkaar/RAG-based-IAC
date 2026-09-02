"""
FastAPI Backend Server
======================
Wraps all LangGraph workflows and exposes clean REST + SSE endpoints
for the Next.js frontend. Zero changes to existing workflow code.

Run with:
    PYTHONPATH=. venv/bin/uvicorn api.server:app --reload --port 8000
"""
import json
import uuid
import asyncio
import importlib
import queue
import threading
from datetime import datetime, timezone
from typing import AsyncGenerator
from langchain_core.callbacks.base import BaseCallbackHandler

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db.job_store import (
    init_db, save_job, load_all_jobs, load_job, delete_job,
    is_apply_paused, set_apply_paused, update_plan_summary, update_apply_status,
)
from data.custom_doc_injector import inject_document, list_internal_docs
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.types import Command

from pathlib import Path

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Terraform Architect API",
    description="Self-healing agentic RAG pipeline for Terraform generation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Workflow Registry ─────────────────────────────────────────────────────────
WORKFLOW_MODULES = {
    "basic":       "workflows.agent_workflow",
    "rag":         "workflows.agent_workflow_rag",
    "advanced":    "workflows.agent_workflow_advanced_rag",
    "secure":      "workflows.agent_workflow_secure_rag",
    "hitl":        "workflows.agent_workflow_hitl",
}

DB_PATH = str(Path(__file__).parent.parent / "chroma_db_terraform")

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()

    # Checkpoint the LangGraph SQLite WAL so state.db-wal doesn't grow unboundedly.
    # This merges any pending WAL writes into the main DB file and resets the WAL
    # to 0 bytes — keeps state reads fast and disk usage predictable.
    import sqlite3
    from pathlib import Path as _Path
    state_db = _Path(__file__).parent.parent / "state.db"
    if state_db.exists():
        try:
            with sqlite3.connect(str(state_db)) as _conn:
                result = _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                # result = (busy, log, checkpointed)
                print(f"[Startup] state.db WAL checkpoint: busy={result[0]}, log={result[1]}, checkpointed={result[2]}")
        except Exception as _e:
            print(f"[Startup] WAL checkpoint skipped: {_e}")


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vs = Chroma(persist_directory=DB_PATH, embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "cosine"})
        chunk_count = vs._collection.count()
    except Exception as e:
        chunk_count = -1
    return {
        "status": "ok",
        "chunk_count": chunk_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Run Workflow (SSE) ────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    workflow: str       # "basic" | "rag" | "advanced" | "secure" | "hitl"
    prompt: str
    thread_id: str | None = None


def _sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


class StreamingQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, q: queue.Queue):
        self.q = q
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put({"type": "token", "content": token})

async def _stream_workflow(workflow: str, prompt: str, thread_id: str) -> AsyncGenerator[str, None]:
    module_path = WORKFLOW_MODULES.get(workflow)
    if not module_path:
        yield _sse("error", {"message": f"Unknown workflow: {workflow}"})
        return

    try:
        mod = importlib.import_module(module_path)
        agent_app = mod.app
    except Exception as e:
        yield _sse("error", {"message": f"Failed to load workflow: {e}"})
        return

    q = queue.Queue()
    handler = StreamingQueueCallbackHandler(q)

    has_config = hasattr(mod, "memory") or workflow in ("hitl", "advanced", "secure", "rag")
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [handler]} if has_config else {"callbacks": [handler]}

    initial_state = {
        "user_request": prompt,
        "messages": [],
        "terraform_code": {},
        "validation_errors": "",
        "is_valid": False,
        "retry_count": 0,
        "retrieved_context": "",
        "citations": [],
        "cost_estimate": "",
        "avg_retrieval_similarity": 0.0,
        "avg_reranker_score": 0.0,
        "trust_score": 0.0,
        "trust_label": "",
        "trust_factors": {},
        "trust_explanation": "",
        # Phase 1: workspace + job identity
        "job_id":           thread_id,
        "workspace_path":   "",
        # Phase 3/3.5: plan + guard fields
        "plan_json":              {},
        "plan_summary":           {},
        "cost_estimate_monthly":  0.0,
        "cost_breakdown":         [],
        "blast_radius_passed":    True,
        "cost_ceiling_passed":    True,
        # Phase 4: apply
        "apply_status":   "",
        "apply_outputs":  {},
        # HitL fields
        "hitl_action":  "",
        "patch_request": "",
        "upload_mode":   False,
        "resource_integrity_passed": True,
        "docs_retrieved": 0,
    }

    final_state = initial_state.copy()

    def run_graph():
        try:
            stream_gen = agent_app.stream(initial_state, config=config)
            for event in stream_gen:
                q.put({"type": "node_update", "event": event})
            q.put({"type": "done"})
        except Exception as e:
            q.put({"type": "error", "error": str(e)})

    thread = threading.Thread(target=run_graph)
    thread.start()

    try:
        while True:
            item = await asyncio.to_thread(q.get)
            if item["type"] == "done":
                break
            elif item["type"] == "error":
                yield _sse("error", {"message": item["error"]})
                break
            elif item["type"] == "token":
                yield _sse("code_stream", {"chunk": item["content"]})
            elif item["type"] == "node_update":
                event = item["event"]
                for node_name, state_update in event.items():
                    final_state.update(state_update)

                    # Node started
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "running",
                    })

                # Enriched events per node
                if node_name == "Retriever_Node":
                    citations = state_update.get("citations", [])
                    docs_count = state_update.get("docs_retrieved", len(citations))
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done",
                        "citations": citations,
                        "docs_retrieved": docs_count,
                        "avg_retrieval_similarity": state_update.get("avg_retrieval_similarity", 0.0),
                    })

                elif node_name == "Architect_Node":
                    files = state_update.get("terraform_code", {})
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done",
                        "file_count": len(files),
                    })

                elif node_name == "Validator_Node":
                    is_valid = state_update.get("is_valid", False)
                    errors = state_update.get("validation_errors", "")
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done" if is_valid else "warning",
                        "is_valid": is_valid,
                        "validation_errors": errors,
                    })

                elif node_name == "Fixer_Node":
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done",
                        "retry_count": state_update.get("retry_count", 1),
                    })

                elif node_name == "Plan_Node":
                    plan_summary   = state_update.get("plan_summary", {})
                    cost_est       = state_update.get("cost_estimate_monthly", 0.0)
                    cost_breakdown = state_update.get("cost_breakdown", [])
                    blast_ok       = state_update.get("blast_radius_passed", True)
                    cost_ok        = state_update.get("cost_ceiling_passed", True)
                    yield _sse("plan_preview", {
                        "node":                  node_name,
                        "status":                "done" if blast_ok and cost_ok else "warning",
                        "plan_summary":          plan_summary,
                        "cost_estimate_monthly": cost_est,
                        "cost_breakdown":        cost_breakdown,
                        "blast_radius_passed":   blast_ok,
                        "cost_ceiling_passed":   cost_ok,
                    })

                elif node_name == "Apply_Node":
                    apply_status = state_update.get("apply_status", "")
                    yield _sse("apply_result", {
                        "node":         node_name,
                        "status":       apply_status,
                        "apply_outputs": state_update.get("apply_outputs", {}),
                    })

                elif node_name == "Destroy_Node":
                    yield _sse("destroy_result", {
                        "node":   node_name,
                        "status": state_update.get("apply_status", ""),
                    })

                elif node_name == "Trust_Assessor_Node":
                    yield _sse("trust_score", {
                        "node": node_name,
                        "status": "done",
                        "score": state_update.get("trust_score", 0.0),
                        "label": state_update.get("trust_label", ""),
                        "factors": state_update.get("trust_factors", {}),
                        "explanation": state_update.get("trust_explanation", ""),
                    })

                elif node_name == "HitL_Node":
                    yield _sse("hitl_pause", {
                        "node": node_name,
                        "status": "paused",
                        "thread_id": thread_id,
                    })
                    return  # Stop streaming — frontend will poll for resume

                else:
                    yield _sse("node_update", {"node": node_name, "status": "done"})

        # Final state
        if config:
            try:
                state_snapshot = agent_app.get_state(config)
                final_state = state_snapshot.values
                
                # If we are paused before HitL_Node, emit hitl_pause and exit instead of complete
                if "HitL_Node" in state_snapshot.next:
                    yield _sse("hitl_pause", {
                        "node": "HitL_Node",
                        "status": "paused",
                        "thread_id": thread_id,
                        "files": final_state.get("terraform_code", {}),
                        "citations": final_state.get("citations", []),
                        "resource_citations": final_state.get("resource_citations", {}),
                    })
                    return
            except Exception:
                pass

        files = final_state.get("terraform_code", {})
        yield _sse("complete", {
            "files": files,
            "citations": final_state.get("citations", []),
            "resource_citations": final_state.get("resource_citations", {}),
            "trust_score": final_state.get("trust_score", 0.0),
            "trust_label": final_state.get("trust_label", ""),
            "trust_factors": final_state.get("trust_factors", {}),
            "trust_explanation": final_state.get("trust_explanation", ""),
            "cost_estimate": final_state.get("cost_estimate", ""),
        })

    except Exception as e:
        yield _sse("error", {"message": str(e)})


@app.post("/api/run")
async def run_workflow(req: RunRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_workflow(req.workflow, req.prompt, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-ID": thread_id,
        },
    )


# ── HitL Actions ─────────────────────────────────────────────────────────────
class HitLAction(BaseModel):
    thread_id: str
    workflow: str
    action: str          # "approve" | "patch" | "apply" | "destroy"
    patch_request: str = ""
    prompt: str = ""     # original prompt for job saving
    override_confirmed: bool = False  # typed confirmation for flagged applies


@app.post("/api/hitl/action")
def hitl_action(req: HitLAction):
    module_path = WORKFLOW_MODULES.get(req.workflow)
    if not module_path:
        raise HTTPException(status_code=400, detail=f"Unknown workflow: {req.workflow}")

    mod = importlib.import_module(module_path)
    agent_app = mod.app
    config = {"configurable": {"thread_id": req.thread_id}}

    # Check circuit breaker for apply/destroy actions
    if req.action in ("apply", "destroy") and is_apply_paused():
        raise HTTPException(
            status_code=503,
            detail="Apply is globally paused. Use /api/admin/pause to re-enable."
        )

    # Check blast-radius override confirmation for flagged applies
    if req.action == "apply":
        current_state = agent_app.get_state(config).values
        blast_ok = current_state.get("blast_radius_passed", True)
        cost_ok  = current_state.get("cost_ceiling_passed", True)
        if (not blast_ok or not cost_ok) and not req.override_confirmed:
            raise HTTPException(
                status_code=422,
                detail="Blast-radius or cost guard failed. Set override_confirmed=true to proceed."
            )

    agent_app.invoke(
        Command(resume={"hitl_action": req.action, "patch_request": req.patch_request}),
        config=config,
    )

    final_state = agent_app.get_state(config).values

    # Persist job on approve or apply
    if req.action in ("approve", "apply"):
        job_id = save_job(
            thread_id=req.thread_id,
            prompt=req.prompt,
            workflow=req.workflow,
            trust_score=final_state.get("trust_score"),
            trust_label=final_state.get("trust_label"),
            trust_factors=final_state.get("trust_factors"),
            files=final_state.get("terraform_code", {}),
            workspace_path=final_state.get("workspace_path"),
            resource_citations=final_state.get("resource_citations", {}),
        )
        # Persist plan summary if available
        if final_state.get("plan_summary"):
            update_plan_summary(
                job_id,
                final_state.get("plan_summary", {}),
                final_state.get("cost_estimate_monthly", 0.0),
            )
        # If this was a direct apply, persist the apply status immediately
        if req.action == "apply":
            apply_status = final_state.get("apply_status", "")
            if apply_status:
                update_apply_status(job_id, apply_status, final_state.get("apply_outputs"))

    return {
        "status": "ok",
        "action": req.action,
        "apply_status": final_state.get("apply_status", ""),
        "files": final_state.get("terraform_code", {}),
        "resource_citations": final_state.get("resource_citations", {}),
    }


# ── Job History ───────────────────────────────────────────────────────────────
@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    return {"jobs": load_all_jobs(limit=limit)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str):
    deleted = delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted"}


# ── Admin: Circuit Breaker ────────────────────────────────────────────────────
@app.get("/api/admin/status")
def admin_status():
    return {
        "apply_paused": is_apply_paused(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/admin/pause")
def admin_pause(pause: bool = True):
    set_apply_paused(pause)
    return {
        "apply_paused": pause,
        "message": "Apply globally paused." if pause else "Apply re-enabled.",
    }


# ── Settings: AWS Credentials ─────────────────────────────────────────────────
class CredentialsRequest(BaseModel):
    role_arn: str


@app.get("/api/settings/credentials")
def get_credentials():
    from aws.credentials_manager import get_credentials_status
    return get_credentials_status()


@app.post("/api/settings/credentials")
def save_credentials(req: CredentialsRequest):
    from aws.credentials_manager import validate_role_arn, save_role_arn
    if not validate_role_arn(req.role_arn):
        raise HTTPException(status_code=400, detail="Invalid Role ARN format.")
    save_role_arn(req.role_arn)
    return {"status": "saved", "role_arn": req.role_arn}


@app.post("/api/settings/credentials/test")
def test_credentials():
    from aws.credentials_manager import test_assume_role
    try:
        result = test_assume_role()
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Custom Doc Upload ─────────────────────────────────────────────────────────
@app.post("/api/docs/upload")
async def upload_doc(
    file: UploadFile = File(...),
    description: str = Form(default=""),
):
    content = await file.read()
    try:
        result = inject_document(
            file_bytes=content,
            filename=file.filename,
            description=description,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/api/docs/list")
def list_docs(limit: int = 20):
    return {"docs": list_internal_docs(limit=limit)}


# ── ETL Pipeline: Vector DB Sync ──────────────────────────────────────────────

class ETLRequest(BaseModel):
    full_rebuild: bool = False
    skip_pull: bool = False


@app.get("/api/etl/status")
def etl_status():
    """Return the ETL manifest (last run time, total indexed, provider version)."""
    import json
    from pathlib import Path as _P
    manifest_path = _P("chroma_db_terraform") / "etl_manifest.json"
    if not manifest_path.exists():
        return {"status": "never_run", "manifest": None}
    try:
        manifest = json.loads(manifest_path.read_text())
        return {"status": "ok", "manifest": manifest}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/etl/run")
def etl_run_sse(req: ETLRequest):
    """
    Stream ETL pipeline logs over SSE.
    The ETL run happens in a background thread; each print() line is
    captured and streamed as a 'log' event to the UI.
    """
    import sys
    import io
    import threading

    log_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()

    class _QueueWriter(io.TextIOBase):
        """Redirect stdout into the SSE log queue."""
        def write(self, msg: str):
            if msg.strip():
                log_queue.put(msg.rstrip())
            return len(msg)
        def flush(self):
            pass

    def _run_etl():
        old_stdout = sys.stdout
        sys.stdout = _QueueWriter()
        try:
            from data.etl_pipeline import run_etl
            run_etl(
                full_rebuild=req.full_rebuild,
                dry_run=False,
                skip_pull=req.skip_pull,
            )
            log_queue.put("__DONE__")
        except Exception as exc:
            log_queue.put(f"[ERROR] {exc}")
            log_queue.put("__DONE__")
        finally:
            sys.stdout = old_stdout
            done_event.set()

    threading.Thread(target=_run_etl, daemon=True).start()

    async def _stream():
        while True:
            try:
                line = log_queue.get(timeout=0.1)
                if line == "__DONE__":
                    yield f"event: done\ndata: ETL pipeline complete\n\n"
                    break
                yield f"event: log\ndata: {json.dumps(line)}\n\n"
            except queue.Empty:
                yield f"event: ping\ndata: {{}}\n\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
