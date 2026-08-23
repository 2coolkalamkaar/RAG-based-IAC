# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-healing agentic RAG pipeline that turns plain-English infrastructure requests into
production-ready, security-hardened Terraform. Four progressive LangGraph `StateGraph` tiers
(basic → RAG → advanced RAG → secure RAG → HITL), grounded in a ChromaDB knowledge base of the
official `terraform-provider-aws` docs, with a FastAPI backend and a Next.js frontend.

## Commands

### Python backend

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build/refresh the vector store (incremental — safe to re-run)
PYTHONPATH=. venv/bin/python data/etl_pipeline.py

# Run the FastAPI backend (used by the Next.js frontend)
PYTHONPATH=. venv/bin/uvicorn api.server:app --reload --port 8000

# Legacy all-in-one Streamlit UI (all workflow tiers, still functional)
PYTHONPATH=. venv/bin/streamlit run ui/multi_workflow_ui.py

# Benchmark suite (12 scenarios across all tiers, ~30-60 min due to API rate limits)
PYTHONPATH=. venv/bin/python benchmarking/run_benchmark.py
PYTHONPATH=. venv/bin/streamlit run benchmarking/benchmark_dashboard.py
```

Always set `PYTHONPATH=.` — workflows import from `data.*`, `db.*`, `api.*` etc. as top-level
packages relative to the repo root.

There is no formal pytest suite; `test_tokens.py` and `archive/RAG_test.py` are standalone
scripts, not a `pytest`-discoverable suite.

### Frontend (`frontend/`)

```bash
cd frontend
npm run dev      # Next.js dev server on :3000, proxies /api/* to the FastAPI backend on :8000
npm run build
npm run lint
```

The dev proxy is configured in `frontend/next.config.ts` (`rewrites()` → `http://127.0.0.1:8000/api/:path*`),
which is how CORS is avoided — the backend must be running on port 8000 for the frontend to work.

### Docker

`Dockerfile` / `docker-compose.yml` at the repo root run the **legacy Streamlit app** (`RAG.py`,
port 8501) — they predate the FastAPI/Next.js migration and are not wired to `api/server.py` or
`frontend/`. `Terraform-Architect-FullStack/` has its own separate `docker-compose.yml` for the
newer full-stack setup.

### Infra tooling required by the workflows themselves

The agent shells out to real binaries during validation — these must be installed for workflows
above the "basic" tier to work:

```bash
bash scripts/install_terraform.sh
bash scripts/install_tflint.sh
# Checkov is installed via requirements.txt
```

## Architecture

### The four+one workflow tiers (`workflows/`)

Each file is a self-contained LangGraph `StateGraph` — they are intentionally **not** DRY with
each other; each tier is a superset copy of the previous one plus new nodes, so that benchmarking
can compare them independently. When fixing a bug that reproduces across tiers, check whether it
needs fixing in all of `agent_workflow.py`, `agent_workflow_rag.py`,
`agent_workflow_advanced_rag.py`, `agent_workflow_secure_rag.py`, and `agent_workflow_hitl.py`.

- **`agent_workflow.py`** — Tier 1, basic: LLM-only generation, no retrieval.
- **`agent_workflow_rag.py`** — Tier 2: adds a ChromaDB retriever node before generation.
- **`agent_workflow_advanced_rag.py`** — Tier 3: MultiQuery expansion (query → 3 sub-queries) +
  a custom `ScorePreservingReranker` (CrossEncoder) before generation.
- **`agent_workflow_secure_rag.py`** — Tier 4: adds Checkov (CIS AWS benchmark) scanning to the
  validate/fix loop, plus a `Trust_Assessor_Node` that scores retrieval similarity, reranker
  confidence, and validation/security pass rate into a 0-100% trust score.
- **`agent_workflow_hitl.py`** — Tier 5: wraps Secure RAG with LangGraph `interrupt()` for a
  human approval gate, plus a `Patcher_Node` that applies natural-language change requests as
  surgical diffs (not full regeneration). This is the tier the frontend and API primarily target.

Common pipeline shape across tiers: **Retriever → Architect (Gemini 2.5 Pro) → Validator
(`terraform validate` + TFLint [+ Checkov]) → Fixer (up to 3 retries, re-validates) → [Trust
Assessor] → [Human-in-the-Loop] → [Patcher]**.

Validation/security subprocess calls (`terraform`, `tflint`, `checkov`) always run in a
`tempfile.mkdtemp()` cwd with list-form args — never `shell=True` or string-interpolated
commands. Preserve this pattern; it's why `.semgrepignore` excludes these call sites.

State persistence for LangGraph threads uses `SqliteSaver` (LangGraph's own state.db) — kept
deliberately separate from `db/job_store.py`'s `jobs.db` (approved/completed job records) to
avoid schema conflicts. See `db/job_store.py` docstring.

### RAG knowledge base (`data/`)

- **`etl_pipeline.py`** — incremental, SHA-256 hash-based sync of the
  `terraform-provider-aws` docs into ChromaDB (`chroma_db_terraform/`, gitignored). Only
  re-embeds changed files on subsequent runs.
- **`vector_store.py`** — ChromaDB init helper; embeddings are
  `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`, cosine similarity space.
- **`custom_doc_injector.py`** — lets SREs inject internal/custom docs into the same vector
  store outside the ETL pipeline (used by `/api/docs/upload`).
- **`mock_sre_upload/`** — sample `.tf` files (including one with intentional vulnerabilities)
  for exercising "SRE Upload Mode," where existing Terraform is uploaded directly into the
  validate/patch nodes, bypassing generation. Uploaded filenames are sanitized via
  `pathlib.Path.name` to prevent path traversal — preserve this when touching upload code.

### API layer (`api/server.py`)

FastAPI app exposing the LangGraph workflows over REST + SSE for the Next.js frontend, with
**zero changes to workflow code** — it's a thin wrapper. Key points:

- `WORKFLOW_MODULES` maps tier names (`basic`, `rag`, `advanced`, `secure`, `hitl`) to the
  corresponding `workflows.*` module, imported dynamically via `importlib`.
- `POST /api/run` streams a workflow run as Server-Sent Events (`_stream_workflow` /
  `_sse` in `api/server.py`); the frontend consumes this via `frontend/lib/sse.ts`.
- `POST /api/hitl/action` resumes an interrupted HITL thread with a `langgraph.types.Command`
  (approve / patch).
- `/api/jobs*` — CRUD over `db/job_store.py`'s SQLite-backed job history.
- `/api/docs/*` — upload/list custom docs injected into the vector store.
- CORS is restricted to `http://localhost:3000` / `127.0.0.1:3000` — in dev this is bypassed
  entirely via the Next.js rewrite proxy instead (see Frontend commands above), so the browser
  never makes a cross-origin request in the first place.

### Frontend (`frontend/`, Next.js 16 / React 19 / TypeScript)

- `app/` — pages: `/` (main pipeline run view), `/history` (past jobs), `/knowledge` (doc
  upload/list against the vector store).
- `components/` — `PipelineVisualizer` (live stage view driven by SSE events), `LogTerminal`,
  `TrustScoreCard`, `HitLPanel` (approve/patch UI for the interrupt gate), `TerraformViewer`,
  `Sidebar`.
- `lib/api.ts` — REST client; `lib/sse.ts` — SSE stream consumer.
- `frontend/AGENTS.md` is auto-regenerated by `next dev` (Next.js 16 canary agent-rules block) —
  don't hand-edit it; if it shows up as a diff after running `npm run dev`, that's expected.

### Security posture (relevant when touching generation/validation code)

- Architect node system prompts enforce hard rules: no `0.0.0.0/0` SSH ingress, mandatory EBS
  encryption, no unjustified public IPs.
- Shift-left: Checkov CIS AWS checks run before any human sees output (Secure RAG tier+).
- `.github/workflows/` runs `semgrep`, `trivy`, and `promptfoo-pr` (LLM eval/red-team) in CI —
  check these when changing prompt templates (`prompts/`) or subprocess-invoking code.

### Legacy/archive

`archive/` and `ui/` (Streamlit) predate the FastAPI/Next.js split and are kept for reference/
fallback, not actively developed. `Terraform-Architect-FullStack/` is a separate, self-contained
full-stack variant with its own README and docker-compose — don't assume it shares code with the
root-level `api/`/`frontend/`.
