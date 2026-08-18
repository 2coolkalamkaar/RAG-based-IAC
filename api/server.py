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
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db.job_store import init_db, save_job, load_all_jobs, load_job, delete_job
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

    has_config = hasattr(mod, "memory") or workflow in ("hitl", "advanced", "secure", "rag")
    config = {"configurable": {"thread_id": thread_id}} if has_config else None

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
    }

    final_state = initial_state.copy()

    try:
        stream_gen = agent_app.stream(initial_state, config=config) if config else agent_app.stream(initial_state)

        for event in stream_gen:
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
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done",
                        "citations": citations,
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
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done" if is_valid else "warning",
                        "is_valid": is_valid,
                    })

                elif node_name == "Fixer_Node":
                    yield _sse("node_update", {
                        "node": node_name,
                        "status": "done",
                        "retry_count": state_update.get("retry_count", 1),
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

                await asyncio.sleep(0)  # Yield control for async streaming

        # Final state
        if config:
            try:
                final_state = agent_app.get_state(config).values
            except Exception:
                pass

        files = final_state.get("terraform_code", {})
        yield _sse("complete", {
            "files": files,
            "citations": final_state.get("citations", []),
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
    action: str          # "approve" | "patch"
    patch_request: str = ""
    prompt: str = ""     # original prompt for job saving


@app.post("/api/hitl/action")
def hitl_action(req: HitLAction):
    module_path = WORKFLOW_MODULES.get(req.workflow)
    if not module_path:
        raise HTTPException(status_code=400, detail=f"Unknown workflow: {req.workflow}")

    mod = importlib.import_module(module_path)
    agent_app = mod.app
    config = {"configurable": {"thread_id": req.thread_id}}

    agent_app.invoke(
        Command(resume={"hitl_action": req.action, "patch_request": req.patch_request}),
        config=config,
    )

    # If approved, save to job store
    if req.action == "approve":
        final_state = agent_app.get_state(config).values
        save_job(
            thread_id=req.thread_id,
            prompt=req.prompt,
            workflow=req.workflow,
            trust_score=final_state.get("trust_score"),
            trust_label=final_state.get("trust_label"),
            files=final_state.get("terraform_code", {}),
        )

    return {"status": "ok", "action": req.action}


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
