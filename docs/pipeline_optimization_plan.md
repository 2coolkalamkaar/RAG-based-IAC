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
| Cost estimation is real | `plan_node` now tries the **Infracost CLI** first (`_estimate_cost_infracost` — real per-resource pricing, fed the plan JSON we already computed, no second `terraform plan`), and falls back to the static per-resource-type table in `blast_radius_guard.py` (`estimate_monthly_cost`) if Infracost isn't installed or `INFRACOST_API_KEY` isn't set. Either way the cost-ceiling guard now does something instead of always passing. Install via `scripts/install_infracost.sh`. Reuses the same tool `agent_workflow_advanced_rag.py`'s `Cost_Estimator_Node` already depends on. |
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
| 1 | ✅ **Warm-start on server boot**, not on first request | Done — `api/server.py`'s `on_startup()` now imports `workflows.agent_workflow_hitl` and calls `_get_vector_store()` / `_get_cross_encoder()` directly at boot, before the server accepts requests. Whoever hits the pipeline first (a demo, a judge) no longer eats the ~4-5s cold-load cost. | Low |
| 2 | **Parallelize `terraform validate` / `tflint` / `checkov`** | They're independent, currently sequential. Running concurrently (e.g. `concurrent.futures` or async subprocess) saves real wall-clock time every single run. See `docs/backlog.md` for the write-up (trade-off: `tflint` would then run even when `validate` already failed). | Low |
| 3 | **Prompt-level cache for repeat/near-duplicate requests** | Cache keyed on request text (or its embedding for near-duplicates) skips retrieval + generation entirely for a repeat ask. Also a legitimate demo trick: pre-warm cache with demo prompts before going on stage. | Medium |
| 4 | **Cheaper/faster model for MultiQuery expansion** | Query expansion currently costs a full Gemini 2.5 Pro round-trip *before* the real generation call. Swap to a lighter model, or skip expansion when the first search already returns strong hits. | Medium |
| 5 | ✅ **Trim `messages` history sent to the LLM** | Done — `architect_node` windows to the last 8 non-Fixer messages (commit `bd6af9c`). | Low |
| 6 | ✅ **WAL checkpoint on `state.db`** | Done — `api/server.py`'s `on_startup()` runs `PRAGMA wal_checkpoint(TRUNCATE)` (commit `bd6af9c`). | Trivial |
| 7 | **Single shared LLM client instance** | Partial — `llm` and `mq_llm` now share a `_VERTEX_CONFIG` dict (commit `bd6af9c`), so config can't drift, but they're still two separate `ChatVertexAI` instances/auth handshakes. Full fix would collapse to one client with a per-call temperature override. | Trivial |
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
history. This is the first question a security-minded engineer will ask.

Full write-up — what happens today, the one-paragraph honest answer for judges, and every approach
considered for detection/safe-kill/tracking/recovery/notification (not just the one chosen) — has
moved to **`docs/partial_apply_recovery_plan.md`**.

**Priority:** low urgency for the demo itself if it stays on `MOCK_AWS=true` (this code path never
executes there). High urgency before any real, unattended `MOCK_AWS=false` apply. Ship the verbal
answer immediately (free); treat state reconciliation + a graceful kill as a pre-requisite for
turning `MOCK_AWS=false` on in front of anyone — see the linked doc for the rest.

---

## Suggested build order (pre-hackathon)

1. Warm-start on boot (§2.1) — cheap, fixes the worst first-impression risk.
2. Parallelize validate/tflint/checkov (§2.2) — cheap, shows up in every benchmark run.
3. Prompt-level cache for demo prompts (§2.3) — cheap, directly de-risks the live demo.
4. Wire in policy-doc retrieval (§3.2) — medium effort, this *is* the differentiation pitch.
5. Re-run the Claude-Code benchmark warm (§1 verification) to get an honest before/after number
   to put on a slide.
6. Have the partial-apply verbal answer ready (`docs/partial_apply_recovery_plan.md`); implement
   state reconciliation + graceful kill before ever running `MOCK_AWS=false` unattended.

Everything else in this document is real but lower priority — worth doing, not worth doing before
the deadline.
