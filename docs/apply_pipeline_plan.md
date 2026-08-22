# From Prompt to Live Infrastructure — Implementation Plan

A single-user path: someone submits AWS credentials, describes the infra they want, the system
generates and validates Terraform, a human reviews the real plan, and on approval the system
applies it for real — plus the features that make this more than "another AI Terraform generator."

> **Out of scope:** Multi-tenancy and auth are deliberately excluded from this plan. Everything
> below assumes one user, one AWS account, one credential set — that layer gets designed
> separately once this core loop works end to end.

## Pipeline overview

```
1. Workspace  →  2. Credentials  →  3. Plan preview  →  4. Human approval  →  5. Apply
```

---

## Foundation

*Needed before an apply can safely exist.*

### Phase 1 — Persistent Terraform workspace

**Goal:** Right now every validation run happens in a temp folder that gets deleted right after.
Once real resources exist, that folder holds the state file — it can never be thrown away.

**How it works:**
1. When a job is created, make one folder for it: `workspaces/{job_id}/`, instead of a temp
   directory.
2. Every step for that job — generate, validate, fix, plan, apply — reads and writes inside that
   same folder from now on.
3. Save the folder's path on the job record, so it can be found again later (a patch request, a
   re-apply, a review days later).

**Files touched:** `db/job_store.py`, `workflows/agent_workflow_hitl.py`, new `workspaces/`

### Phase 2 — Credential capture & storage

*Depends on: Phase 1*

**Goal:** The user needs to give the system AWS credentials once, and the system needs to use
them only at the moment it runs Terraform — never logged, never handed to the AI model, never
written where a generated file could reach them.

**How it works:**
1. Add one settings screen: paste an AWS Access Key ID and Secret Key.
2. Encrypt them before saving — a symmetric key from an environment variable, stored in a small
   new table, not the jobs table.
3. When (and only when) a `terraform plan` or `apply` subprocess runs, decrypt and pass them in
   as that one process's environment variables. They never touch LangGraph state, logs, or the
   workspace folder itself.

**Files touched:** new `db/credential_store.py`, `api/server.py`, new `frontend/app/settings`

---

## Core apply loop

*Turns your existing review gate into a real "make it happen" button.*

### Phase 3 — Plan preview, before the human sees anything

*Depends on: Phase 1, 2*

**Goal:** Approving generated *code* is not the same as approving *what will actually happen to
the AWS account*. Before the human reviewer sees the job, run a real `terraform plan` and show
them exactly what will be created, changed, or destroyed.

**How it works:**
1. Add a **Plan_Node** to the HITL workflow, right after validation passes and before the review
   interrupt.
2. It runs `terraform init` then `terraform plan` in the job's workspace, using the credentials
   from Phase 2, and captures the output.
3. The plan text streams to the frontend over the existing SSE connection, and shows up next to
   the generated code and trust score in the review panel.

**Files touched:** `workflows/agent_workflow_hitl.py`, `api/server.py`,
`frontend/components/HitLPanel`

### Phase 4 — Apply, gated by approval

*Depends on: Phase 3*

**Goal:** This is the moment the system stops being a code generator and starts being
infrastructure. It only happens after a human has seen the real plan and clicked approve.

**How it works:**
1. Add an **Apply_Node**, reached by resuming the graph's existing interrupt with a new `"apply"`
   action — same mechanism already used for approve/patch.
2. It runs `terraform apply -auto-approve` in the workspace with the same scoped credentials, and
   streams the output live, the same way generation logs stream today.
3. On success, save the real resource IDs and outputs onto the job record — this becomes the
   permanent receipt of what was actually built.
4. On failure, stop and surface the error; never retry an apply automatically the way the Fixer
   retries generation.

**Files touched:** `workflows/agent_workflow_hitl.py`, `api/server.py`, `db/job_store.py`,
`frontend/components/HitLPanel`

---

## What makes it worth switching to

*The features a generic AI-Terraform tool can't copy.*

### Phase 5 — Citations on generated code

*Independent*

**Goal:** The retriever already finds the doc chunks that justify each choice — today that
reasoning is thrown away after generation. Keep it, and show it.

**How it works:**
1. Carry the retrieved chunk's source (doc name, section) alongside each generated resource
   block, instead of discarding it after the Architect node runs.
2. In the review UI, let the reviewer hover a setting — like EBS encryption — and see which
   internal doc required it.

**Files touched:** `workflows/agent_workflow_advanced_rag.py`,
`frontend/components/TerraformViewer`

### Phase 6 — Drift detection against org policy

*Depends on: Phase 1, 4*

**Goal:** Once real infra exists, periodically check whether it still matches what the org's own
docs require — not just whether it matches the last-applied state.

**How it works:**
1. On a schedule, run `terraform plan` against each applied job's workspace to catch drift from
   the live account.
2. Re-run retrieval against the org's knowledge base for each live resource, and flag any
   mismatch against current internal policy docs — not just infra drift, policy drift.

**Files touched:** new `workflows/drift_check.py`, `db/job_store.py`

### Phase 7 — Exportable audit trail

*Depends on: Phase 4*

**Goal:** For a regulated org, the record of "who asked, what was generated, what the trust score
was, who approved, what got applied" is itself the product, not a side effect.

**How it works:**
1. Bundle the trust score breakdown, plan diff, reviewer identity, and apply outcome into one job
   report.
2. Add a one-click export (PDF or shareable link) per job, framed as change-management evidence.

**Files touched:** `db/job_store.py`, new `frontend/app/history/[job]/report`

### Phase 8 — Cost delta on the plan preview

*Depends on: Phase 3*

**Goal:** Next to "what will be created," show "what it will cost per month" — small addition,
disproportionate trust it buys from a reviewer before they click apply.

**How it works:**
1. Parse the resource types and sizes out of the `terraform plan` output from Phase 3.
2. Look up approximate monthly pricing per resource and show a total delta alongside the plan
   diff.

**Files touched:** `workflows/agent_workflow_hitl.py`, `frontend/components/HitLPanel`

---

## Suggested build order

1. **Persistent workspace** — nothing else can be built until state survives between steps.
2. **Credential storage** — needed before any real `plan` or `apply` call can run.
3. **Plan preview node** — makes the existing approval gate actually meaningful.
4. **Apply node** — the core promise of the product: approved code becomes real infrastructure.
5. **Citations, then audit export, then cost delta, then drift detection** — each is independent
   and can land in any order once Apply exists.
