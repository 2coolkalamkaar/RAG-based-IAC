# TerraForge Pipeline Review
> Full analysis of the HitL RAG workflow — optimization opportunities and architecture notes

---

## Architecture Overview

```
User Prompt
    │
    ▼
[Retriever_Node]  ──→  MultiQuery + CrossEncoder Reranker → ChromaDB
    │
    ▼
[Architect_Node]  ──→  Gemini 2.5 Pro (temp=0.2, streaming)
    │
    ▼
[Validator_Node]  ──→  terraform init + validate + tflint (deterministic)
    │  ╔════════╗
    ├─→║ Fixer  ║ (up to N retries, resource integrity check)
    │  ╚════════╝
    ▼
[Plan_Node]       ──→  terraform plan (real or MOCK_AWS)
    │
    ▼
[Trust_Assessor]  ──→  Weighted score: retrieval + reranker + validation
    │
    ▼
[HitL_Node]       ──→  interrupt() — awaits human approval
    │
    ├──→ [Patcher_Node] → back to Validator (surgical changes)
    ├──→ [Apply_Node]   → terraform apply
    └──→ [Destroy_Node] → terraform destroy
```

---

## ✅ What's Working Well

| Aspect | Status |
|--------|--------|
| Self-healing Fixer loop | ✅ Solid — resource integrity check prevents silent drops |
| Trust scoring formula | ✅ Good multi-factor signal (retrieval + reranker + validation) |
| Edge order (Plan → Trust) | ✅ Fixed — blast-radius now reads real plan data |
| Validator determinism | ✅ `terraform validate` + `tflint` — no LLM guessing |
| HitL interrupt/resume | ✅ LangGraph's `interrupt()` + SQLite checkpointing |
| `---` artifact stripping | ✅ `clean_code()` prevents YAML frontmatter from reaching `terraform init` |

---

## ⚠️ Optimization Opportunities

### 1. 🐢 Retriever cold-starts on every run
**Problem:** The retriever re-instantiates `HuggingFaceEmbeddings` (loads model to memory) and `Chroma` on **every single request**. This adds ~3-5s to every run.

**Fix:** Move the embedding model and vector store into **module-level singletons** — load once at startup.
```python
# At module level (line ~163), add:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vector_store    = Chroma(persist_directory=DB_PATH, embedding_function=_embedding_model, ...)
```

### 2. 🔄 CrossEncoder Reranker cold-starts on every run
**Problem:** `HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")` downloads/loads the model on **every request**. This is the source of the "Booting up Score-Preserving CrossEncoder Reranker..." message — it's 80MB each time.

**Fix:** Same pattern — cache as a module-level singleton.

### 3. 🔁 Fixer has no hard retry cap enforced in routing
**Problem:** `validator_routing` checks `retry_count >= 3` to give up, but `retry_count` is only incremented in `fixer_node`. If a run gets stuck in a loop, it can silently run 4+ LLM calls before stopping.

**Fix:** Add a hard cap and explicit `failed` routing:
```python
def validator_routing(state: AgentState) -> str:
    if state.get("is_valid"):
        return "trust_assessor"
    if state.get("retry_count", 0) >= 3:
        return "trust_assessor"  # Force through with low trust
    return "fixer"
```

> [!NOTE]
> Check current routing logic — it may already do this, but confirm the trust score penalty for failed validation is applied correctly.

### 4. 💾 `state.db` WAL file is 4MB and growing
**Problem:** The SQLite WAL file (`state.db-wal`) is 4.1MB and never gets checkpointed. LangGraph's SQLiteSaver doesn't auto-checkpoint. This will keep growing with every run and will eventually slow down state reads.

**Fix:** Add a periodic WAL checkpoint at server startup:
```python
# In api/server.py startup:
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```

### 5. 📨 `messages` list grows unbounded in state
**Problem:** `AgentState.messages` uses `operator.add` (appending). Every node adds messages. After 10+ runs on the same thread, the context sent to Architect/Fixer grows huge (wasting tokens and slowing LLM calls).

**Fix:** Trim the messages list in Architect/Fixer to keep only the last N messages:
```python
history_messages = state.get("messages", [])[-10:]  # Keep last 10 only
```

### 6. 🏗️ Two LLM instances (Gemini 2.5 Pro) loaded at module level
**Problem:** Both `llm` and `mq_llm` are identical model configs (`gemini-2.5-pro`) with just different temperatures. Two separate connections = two auth handshakes on startup.

**Fix:** Use one client with temperature overrides at call time, or at least make it explicit in a config block.

### 7. ⚡ `ScorePreservingReranker` is re-defined inline on every request
**Problem:** The entire `ScorePreservingReranker` class is defined **inside** `retriever_node` on every call. Class definitions aren't free — this should be at module level.

**Fix:** Move the class definition to module level, outside the function.

### 8. 🔍 Stale doc deletion runs on every retrieval
**Problem:** Lines 201-207 query ChromaDB for `iac_eval_dataset` docs and delete them **every single request**. This was a one-time cleanup that got left in the hot path.

**Fix:** Run this cleanup once at server startup, not on every retrieval.

---

## What is `knowledge_base/policy_docs`?

These 6 markdown files are **organizational context docs** — they describe your intended security and compliance policies:

| File | Policy |
|------|--------|
| `01_mandatory_tagging.md` | All resources must have `ManagedBy`, `JobID`, `Environment` tags |
| `02_no_public_access.md` | No public S3 buckets, no open security groups |
| `03_encryption_at_rest.md` | S3/EBS/RDS must have encryption enabled |
| `04_iam_restrictions.md` | No IAM `Action=*` or `Resource=*` wildcards |
| `05_instance_sizes_and_counts.md` | Max 20 resources, no oversized instances without explicit request |
| `06_prompt_injection_refusal.md` | Architect must refuse injection attempts |

> [!IMPORTANT]
> **These files are NOT currently being ingested into ChromaDB or injected into any prompt.** They live in `knowledge_base/policy_docs/` but nothing reads them at runtime. The Architect's system prompt hardcodes the same rules directly (lines 300-321 in `agent_workflow_hitl.py`).

### Options for using them:
1. **Keep as-is (documentation)** — they serve as the human-readable spec that the hardcoded prompt was written from. Good for auditing.
2. **Ingest into ChromaDB** — inject them as retrieval docs so the Architect sees them as grounding context. Good for dynamic policy updates without code changes.
3. **Inject as structured system context** — load them at startup and append to every Architect prompt. Best of both worlds.

**Recommendation:** Option 3 — load them at startup, concatenate into a `### POLICIES ###` section in the Architect prompt. This means you can update a policy by editing a markdown file, not the Python code.

---

## Priority Fix Ranking

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | Retriever/Reranker cold-start on every request | High (3-5s per run) | Low |
| 2 | Stale doc cleanup in hot path | Medium | Trivial |
| 3 | `messages` list unbounded growth | Medium (token waste) | Low |
| 4 | `state.db` WAL not checkpointed | Medium (disk) | Trivial |
| 5 | `ScorePreservingReranker` class inline | Low | Trivial |
| 6 | Policy docs not connected to runtime | Low (feature gap) | Medium |
| 7 | Fixer retry cap verification | Low (safety) | Low |
