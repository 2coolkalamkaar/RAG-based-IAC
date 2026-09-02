# Backlog

Ideas that are real and worth building eventually, but not right now. Moved out of the active
plan docs (`apply_pipeline_plan_v2.md`, `docs/apply_pipeline_plan.md`) so those stay focused on
what's actually being built. An item lands here when there's no immediate need for it — not
because it's a bad idea.

---

## Drift detection against org policy

*Originally Phase 6. Depends on: persistent workspaces (Phase 1), Apply (Phase 4).*

**Why it's on the backlog:** no immediate need — nothing has been applied to a real account long
enough for drift to be a live problem yet, and it needs real elapsed time to demo meaningfully
(apply something, wait, cause drift, show the detector catch it) — not a good use of build time
under deadline pressure.

**Goal:** Once real infra exists, periodically check whether it still matches what the org's own
docs require — not just whether it matches the last-applied Terraform state.

Two different kinds of drift, both worth catching:
1. **Infra drift** — did reality stop matching what was actually built (e.g. someone manually
   opened a security group in the AWS console after apply)?
2. **Policy drift** — did the *rules* change since the resource was created (e.g. a new cost
   policy now caps instance size below what's already running)? This is the one a generic
   "terraform plan diff" checker doesn't catch, and the one that makes this feature more than a
   commodity drift checker — it's grounded in the same org policy KB as generation (Phase 2.5).

**How it works:**
1. On a schedule, run `terraform plan` against each applied job's workspace to catch drift from
   the live account.
2. Re-run retrieval against the org's knowledge base for each live resource, and flag any
   mismatch against *current* internal policy docs — not just infra drift, policy drift.

**Files touched:** new `workflows/drift_check.py`, `db/job_store.py`.

---

## Exportable audit trail

*Originally Phase 7. Depends on: Apply (Phase 4).*

**Why it's on the backlog:** no immediate need — nobody is consuming job history as a compliance
artifact yet; worth building once there's an actual audience (a regulated org, a security review)
asking for it.

**Goal:** For a regulated org, the record of "who asked, what was generated, what the trust score
was, who approved, what got applied" is itself the product, not a side effect.

**How it works:**
1. Bundle the trust score breakdown, plan diff, reviewer identity, and apply outcome into one job
   report.
2. Add a one-click export (PDF or shareable link) per job, framed as change-management evidence.
3. When built, fold in the Blast-Radius Guard outcomes (Phase 3.5 — ownership/cost/IAM override
   history) alongside trust score, plan diff, reviewer identity, and apply outcome — the guardrail
   trail is part of what "the record of what happened" means.

**Files touched:** `db/job_store.py`, new `frontend/app/history/[job]/report`.

---

## Parallelize independent tasks in the HITL pipeline

*Discussed against `workflows/agent_workflow_hitl.py`. No dependency — can land any time.*

**Why it's on the backlog:** no immediate need — modest, not urgent wins; worth doing but not
ahead of the workspace_path persistence bug or other correctness work.

**Goal:** Two spots in the pipeline currently run two independent things sequentially for no
reason other than code order. Both are I/O-bound (waiting on a network call or a subprocess, not
CPU work), so a `concurrent.futures.ThreadPoolExecutor` overlaps the waiting instead of stacking
it — Python's GIL blocks concurrent *computation*, not concurrent *waiting*.

**Case 1 — `retriever_node`'s second, unrelated search:**
```python
docs = retriever_pipeline.invoke(user_request)              # slow: MultiQuery + rerank
sim_results = vector_store.similarity_search_with_relevance_scores(user_request, k=5)  # independent
```
`sim_results` only feeds `avg_retrieval_similarity` for the trust score — it doesn't depend on
`docs`, and `docs` doesn't depend on it. Run both via a thread pool and `.result()` both instead of
awaiting them one after another; the second call's cost becomes free, hidden behind the first's
wait.

**Case 2 — `validate_terraform_code`'s `terraform validate` and `tflint`:**
```python
val_res    = subprocess.run(["terraform", "validate", ...], ...)
tflint_res = subprocess.run(["tflint", ...], ...)
```
Both are read-only checks against the same already-`init`'d directory; neither depends on the
other's result. Submitting both to a thread pool saves roughly whichever one is faster, on every
validation pass — including every Fixer retry loop, so it compounds.

**Trade-off to decide before building:** today, a `terraform validate` failure returns immediately
and `tflint` never runs (saves a wasted lint pass on code that already failed). Parallelizing means
both run unconditionally every time, even when `validate` is going to fail anyway. Net win in the
common case (most runs pass validate), but not strictly free — worth confirming that's acceptable
before implementing.

**Files touched:** `workflows/agent_workflow_hitl.py` (`retriever_node`, `validate_terraform_code`).
