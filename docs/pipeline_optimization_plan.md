# Pipeline Optimization & Hackathon-Readiness Plan

> Consolidates the security/correctness fixes already shipped on `feature/apply-pipeline-v2`,
> plus the latency, retrieval-quality, and architecture work discussed but not yet built.
> Companion to `apply_pipeline_plan_v2.md` (the apply/destroy feature plan) and
> `pipeline_review.md` (the original perf review this section 1 implements).

---

## 0. Positioning — why this over Claude/GPT/Gemini directly

Any of those, given shell access and AWS creds, can already write Terraform and run `apply`.
That part is commoditized. This product only earns its place in the gap between *"the LLM said
this is fine"* and *"a human should trust that."* Four things live in that gap and are the only
things worth defending in a pitch:

1. **A non-LLM last line of defense** — the Blast-Radius Guard walks the actual `terraform plan`
   JSON, not the model's account of itself, and hard-blocks anything touching untagged/unowned
   resources. Same instinct as the Fixer's resource-integrity check, one layer up.
2. **Policy grounding with citations**, not house rules baked into a system prompt.
3. **A quantified trust signal** instead of uniform LLM confidence — every chat response reads
   equally sure of itself whether right or hallucinated; a trust score doesn't.
4. **No long-lived secret, ever** — AssumeRole + External ID, the same pattern Datadog/Vercel use.

Competitive reality check: plan/review/apply-with-guardrails alone is *not* novel — Terraform
Cloud, Spacelift, and env0 already do this with Sentinel/OPA + Infracost. The actual pitch is
**natural-language generation grounded in org policy, with that same guardrail rigor** — self-serve
infra for teams without a platform team, not a Spacelift replacement.

**The catch:** items 1 and 4 above were, until this branch's follow-up fixes, not actually wired
into the execution path. Section 1 below is what closed that gap — it is not cosmetic cleanup,
it *is* the product's credibility.

---

## 1. Correctness/security fixes — ✅ Shipped

These were flagged as differentiators that were only stubbed. All now implemented in
`workflows/agent_workflow_hitl.py` and `workflows/blast_radius_guard.py`:

| Fix | What changed |
|---|---|
| AssumeRole actually used | `_get_aws_subprocess_env()` calls `assume_role()` and injects the temporary session into the `plan`/`apply`/`destroy` subprocess env only. Fails closed (guard = `False`) if the Role ARN isn't configured — never silently falls back to ambient credentials. |
| Real blast-radius guard wired in | `plan_node` now calls `run_all_guards()` (ownership-tag check + IAM wildcard/escalation scan + cost ceiling) instead of the placeholder `delete_count == 0` check. |
| Cost estimation is real | Added a static per-resource-type / per-instance-type monthly cost table in `blast_radius_guard.py` (`estimate_monthly_cost`) — no external pricing API, but the cost-ceiling guard now does something instead of always passing. |
| Apply/destroy timeout | `APPLY_TIMEOUT_SECONDS` (default 600s) enforced via `subprocess.run(..., timeout=...)` on both `apply_node` and `destroy_node` — closes the Phase 4.6 circuit-breaker gap. |
| Retriever/reranker caching | Embedding model, Chroma store, `ScorePreservingReranker` class, and `HuggingFaceCrossEncoder` moved to module-level singletons (`_get_vector_store()`, `_get_cross_encoder()`), lazily built once per process instead of once per request. Stale-doc cleanup folded into that same one-time init. |
| Minor cleanup | Stray `return "fixer"""` typo in `validator_routing` fixed. |

**Verification still needed (not done in this sandbox — no venv available):**
- Run one real request through `uvicorn` and confirm the log shows "Booting up..." exactly once,
  then subsequent requests skip straight to "Executing Smart Search Pipeline."
- Confirm a `plan`/`apply` against `MOCK_AWS=false` with a real Role ARN configured actually
  authenticates as the assumed role (check `aws sts get-caller-identity` matches the role, not
  the backend host's own identity).
- Re-run the Claude-Code-vs-this-system benchmark against a **warm** server (not restarted
  between runs) — the original benchmark's roughly-equal timing may have been measuring cold-start
  cost that this fix specifically removes.

---

## 2. Latency optimization — not yet built, ranked by payoff

| # | Fix | Why it helps | Effort |
|---|---|---|---|
| 1 | **Warm-start on server boot**, not on first request | Right now the first real request (possibly a judge) eats the one-time load cost. Trigger `_get_vector_store()` / `_get_cross_encoder()` from a FastAPI startup event instead. | Low |
| 2 | **Parallelize `terraform validate` / `tflint` / `checkov`** | They're independent, currently sequential. Running concurrently (e.g. `concurrent.futures` or async subprocess) saves real wall-clock time every single run. | Low |
| 3 | **Prompt-level cache for repeat/near-duplicate requests** | Cache keyed on request text (or its embedding for near-duplicates) skips retrieval + generation entirely for a repeat ask. Also a legitimate demo trick: pre-warm cache with demo prompts before going on stage. | Medium |
| 4 | **Cheaper/faster model for MultiQuery expansion** | Query expansion currently costs a full Gemini 2.5 Pro round-trip *before* the real generation call. Swap to a lighter model, or skip expansion when the first search already returns strong hits. | Medium |
| 5 | **Trim `messages` history sent to the LLM** | Already flagged in `pipeline_review.md` — history grows unbounded per thread, wasting tokens and slowing every call on long-lived threads. Keep last N messages. | Low |
| 6 | **WAL checkpoint on `state.db`** | `pipeline_review.md` item — SQLite WAL file grows unchecked; periodic `PRAGMA wal_checkpoint(TRUNCATE)` at startup keeps state reads fast over time. | Trivial |
| 7 | **Single shared LLM client instance** | `llm` and `mq_llm` are two separate `ChatVertexAI` connections for the same model, just different temperature. Collapse to one client with per-call temperature override. | Trivial |
| 8 | **Stream pipeline progress, not just final text** | Perceived speed matters as much as real speed on stage — streaming plan/guard progress over the existing SSE connection keeps a judge watching a live trace instead of a blank screen. | Medium |

**Caveat:** singleton caching (already shipped, Section 1) is per-*process*. If this ever runs
behind multiple `uvicorn` workers, each worker pays the load cost once — fine for a single-process
hackathon demo, worth remembering before scaling.

---

## 3. Retriever/reranker quality — not yet built

Speed and *correctness of what gets retrieved* are separate problems:

| # | Improvement | What it fixes |
|---|---|---|
| 1 | **Hybrid search (keyword + embedding)** | Pure embedding search sometimes fuzzes past exact identifiers like `aws_s3_bucket_versioning`. Combining keyword/exact-match with semantic search is the standard fix for technical-docs retrieval. |
| 2 | **Wire in the existing policy-doc knowledge base** | `knowledge_base/policy_docs/` (6 files: tagging, encryption, IAM, instance limits, injection refusal) exists but is never ingested/retrieved — `pipeline_review.md`'s own recommendation (Option 3: load at startup, inject as a `### POLICIES ###` prompt section). This is the strongest differentiation story (Section 0, item 2) and is mostly wiring, not new invention. |
| 3 | **Code-aware reranker** | Current reranker (`ms-marco-MiniLM`) is trained for web search, not code/config docs. Test a code-tuned reranker, or an LLM-based "pick the 2 chunks that actually matter" pass for hard queries. |
| 4 | **Diversity-aware chunk selection (MMR)** | Avoids retrieving 3 near-duplicate chunks that say the same thing, wasting context budget that could carry more distinct information. |
| 5 | **Measure retrieval quality directly** | Extend the existing IaC-Eval benchmark to score whether the *right* chunk was retrieved for a given question, not just whether the final Terraform output was correct — today there's no way to tell if good output came from good retrieval or from the model just knowing the answer. |

---

## 4. Architecture changes worth considering

- **Split retrieval/embedding out of the API process** into its own always-warm service, so
  restarting the FastAPI app during iteration doesn't force a full model reload.
- **Pre-warm the cache for demo prompts specifically** before presenting — legitimate practice,
  not cheating; it's what a warm production system looks like vs. a cold demo.
- **Async LangGraph nodes** where steps are independent (ties into Section 2, item 2).

---

## 5. Partial-apply failure mode — not yet built

**Why this matters:** once a job's `terraform apply` starts touching a real AWS account, "it
failed" is no longer a safe answer. If 2 of 5 planned resources were created before an error, real
billed infrastructure now exists, tagged correctly but invisible to anyone looking at the job
history. This is the first question a security-minded engineer will ask, and right now the honest
answer is "we haven't fully specified it." This section specifies it.

### 5.1 What actually happens today (verified against the current code)

- **The workspace is safe — Phase 1 already got this right.** Each job gets a persistent
  `workspaces/{job_id}/` directory (`db/job_store.py` `create_workspace`), not a `tempfile.mkdtemp()`
  that gets deleted after the run. Terraform writes `terraform.tfstate` incrementally as each
  resource is created, so if resource 3 of 5 fails, resources 1 and 2 are recorded in that state
  file on disk. **Nothing is silently lost.**
- **But `apply_status` collapses everything to one bit.** `apply_node` in
  `workflows/agent_workflow_hitl.py` returns `apply_status: "failed"` whether zero resources were
  touched or four of five succeeded. `db/job_store.update_apply_status()` just persists that
  string. The job history / HitLPanel has no way to distinguish "nothing happened, safe to ignore"
  from "real resources exist, someone needs to act."
- **There is currently no supported retry/cleanup path on the same job.** The graph has
  `Apply_Node → END` as an unconditional edge — once a run ends there, the thread has no pending
  `interrupt()` for a `Command(resume=...)` to attach to. To do anything about a partially-applied
  job today, a human would have to `cd` into `workspaces/{job_id}/` and run `terraform` by hand,
  entirely outside the product. Starting a *new* job instead would regenerate fresh Terraform in a
  fresh workspace with no awareness that the old workspace still has real resources sitting in it.
- **The timeout path is the riskiest moment.** `subprocess.run(..., timeout=APPLY_TIMEOUT_SECONDS)`
  kills the process via `Popen.kill()` (SIGKILL) the instant the timeout fires. Terraform is
  mid-write to its state file when this can happen — a hard kill is the one scenario more dangerous
  than a clean failure, because it risks a **corrupted** state file, not just an incomplete one.

### 5.2 The one-paragraph answer for judges (works today, ship this first)

> "Every job's Terraform state lives in a workspace directory that's never deleted, so even if an
> apply dies halfway through, nothing is silently lost — the state file has exactly what's real.
> We don't currently surface partial-vs-total failure as a distinct status, and there's no
> in-product retry button yet — recovering today means operating that workspace directly. That's
> the next thing we're closing before this goes anywhere near a real account unattended."

This is honest, doesn't overclaim, and correctly identifies the persistent workspace as the thing
that makes the rest of this fixable rather than catastrophic.

### 5.3 Implementation plan

**1. Detect what's actually real after a failure — read state, don't trust the exit code.**
Add a helper in `agent_workflow_hitl.py`, called from `apply_node`'s except/failure branches:

```python
def _reconcile_apply_state(workspace_path: str, env: dict, planned_create_count: int) -> dict:
    """After a failed/timed-out apply, ask Terraform's own state — not the exit code —
    what actually exists. Returns {"status": "failed"|"partial"|"applied", "created_resources": [...]}."""
    res = subprocess.run(
        ["terraform", "state", "list"], cwd=workspace_path, env=env,
        capture_output=True, text=True,
    )
    resources = [r for r in res.stdout.strip().splitlines() if r]
    if not resources:
        return {"status": "failed", "created_resources": []}
    if len(resources) < planned_create_count:
        return {"status": "partial", "created_resources": resources}
    return {"status": "applied", "created_resources": resources}
```

Call this from both the `subprocess.TimeoutExpired` branch and the `returncode != 0` branch of
`apply_node`, using `state.get("plan_summary", {}).get("create", 0)` as `planned_create_count`.

**2. Add a `"partial"` status and a `created_resources` column.**
Extend `db/job_store.py` the same way `workspace_path`/`apply_status`/`plan_summary` were added
(there's already an additive-column migration pattern in that file — follow it):
```python
("created_resources", "TEXT"),  # JSON list of resource addresses actually in state
```
`update_apply_status()` should accept and persist `created_resources`; `apply_status` can now be
one of `"applied" | "partial" | "failed" | "destroyed"`.

**3. Graceful subprocess termination instead of a hard kill on timeout.**
`subprocess.run(timeout=...)` only exposes `.kill()` (SIGKILL) on expiry. Use `Popen` directly so a
timeout sends SIGTERM first and gives Terraform a grace window to finish its current resource
operation and flush state cleanly, only escalating to SIGKILL if it ignores that:
```python
proc = subprocess.Popen(["terraform", "apply", "-auto-approve", "tfplan"], cwd=workspace_path, env=env, ...)
try:
    proc.wait(timeout=APPLY_TIMEOUT_SECONDS)
except subprocess.TimeoutExpired:
    proc.terminate()          # SIGTERM — let Terraform finish its current write
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()            # last resort only
```

**4. A supported reconciliation path — new endpoint, never a silent resume.**
Add `POST /api/jobs/{job_id}/reconcile` in `api/server.py`. Given the job's stored
`workspace_path`, it re-enters the graph **at `Plan_Node`, never directly at `Apply_Node`** — a
fresh plan against the *current real state* is mandatory before any further action, so the human
reviewing it sees exactly what's left to create or what's now safe to tear down. Route the result
back through `HitL_Node` like any other review; never auto-apply or auto-destroy from this path.

**5. Surface it distinctly in the UI, not folded into "failed."**
In `HitLPanel` and the job history list: a job with `apply_status == "partial"` gets its own
visual state (e.g. "⚠️ Partial — 2/5 resources created, action required") with a prominent
"Reconcile" button wired to the new endpoint — visually and behaviorally distinct from the generic
red "failed" badge, which today looks identical whether zero or four resources exist.

**6. Log the detection event even without full audit UI.**
Phase 7 (exportable audit trail) is already cut from hackathon scope — but log
`_reconcile_apply_state()`'s result server-side regardless, so there's at least a record in
application logs of every partial-state detection, independent of whether anyone clicks "Reconcile."

### 5.4 Priority relative to the rest of this plan

Low urgency for the demo itself if the demo stays on `MOCK_AWS=true` (this code path never
executes there). **High urgency before any real, unattended `MOCK_AWS=false` apply** — and worth
having the §5.2 answer memorized regardless, since this is a predictable question from anyone with
production infra experience. Suggest: ship §5.2 as a verbal answer immediately (free), and treat
5.3 items 1–3 (state reconciliation + graceful kill) as a pre-requisite for turning `MOCK_AWS=false`
on in front of anyone, ahead of items 4–6 (full reconcile UI), which can follow after the deadline.

---

## Suggested build order (pre-hackathon)

1. Warm-start on boot (§2.1) — cheap, fixes the worst first-impression risk.
2. Parallelize validate/tflint/checkov (§2.2) — cheap, shows up in every benchmark run.
3. Prompt-level cache for demo prompts (§2.3) — cheap, directly de-risks the live demo.
4. Wire in policy-doc retrieval (§3.2) — medium effort, this *is* the differentiation pitch.
5. Re-run the Claude-Code benchmark warm (§1 verification) to get an honest before/after number
   to put on a slide.
6. Have the §5.2 partial-apply answer ready verbally; implement §5.3 items 1–3 (state
   reconciliation + graceful kill) before ever running `MOCK_AWS=false` unattended.

Everything else in this document is real but lower priority — worth doing, not worth doing before
the deadline.
