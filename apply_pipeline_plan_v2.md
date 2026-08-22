# From Prompt to Live Infrastructure — Implementation Plan (v2)

A single-user path: someone connects an AWS account, describes the infra they want, the system
generates and validates Terraform, guardrails check it against blast-radius and policy risk, a
human reviews the real plan, and on approval the system applies it for real.

> **Out of scope:** Multi-tenancy and auth are deliberately excluded from this plan. Everything
> below assumes one user, one AWS account, one credential set — that layer gets designed
> separately once this core loop works end to end.

### What changed from v1

- **Phase 2 rewritten** — temporary AWS STS credentials via an assumed IAM role, not stored
  long-lived keys. Nothing to encrypt, nothing to rotate, nothing to leak.
- **New Phase 2.5** — Architect prompt hardening + a policy knowledge base, so the RAG pipeline
  grounds decisions in org policy, not just AWS syntax.
- **New Phase 3.5** — a Blast-Radius & Cost Guard node that inspects the real plan before a human
  ever sees it, and feeds two new hard-override triggers into the existing Trust Assessor.
- **New Phase 4.5** — a destroy path, mirroring apply. You can't demo or safely test a "make real
  infra" tool without a matching "and now clean it up" button.
- **New Phase 4.6** — a circuit breaker: a global pause switch and a per-job apply timeout, so a
  stuck or runaway apply can't run indefinitely.
- Cost delta (previously Phase 8) folded directly into Phase 3, since it now feeds a guardrail
  rather than being a nice-to-have display.

---

## Pipeline overview

```
1. Workspace → 2. Credentials (AssumeRole) → 2.5 Prompt/Policy hardening →
3. Plan preview + cost → 3.5 Blast-Radius & Cost Guard → 4. Human approval →
5. Apply  (or 4.5 Destroy, gated the same way)
```

---

## Foundation

### Phase 1 — Persistent Terraform workspace

**Goal:** Right now every validation run happens in a temp folder that gets deleted right after.
Once real resources exist, that folder holds the state file — it can never be thrown away.

**How it works:**
1. When a job is created, make one folder for it: `workspaces/{job_id}/`, instead of a temp
   directory.
2. Every step for that job — generate, validate, fix, plan, apply, destroy — reads and writes
   inside that same folder.
3. Save the folder's path on the job record so it can be found again later.

**Files touched:** `db/job_store.py`, `workflows/agent_workflow_hitl.py`, new `workspaces/`

---

### Phase 2 — Credentials via temporary AssumeRole session (not stored keys)

**Goal:** Never store a long-lived AWS secret at all. Instead of asking the user to paste an
Access Key + Secret Key, have them give you a **Role ARN** for a role they create in their own
account, scoped to only what the agent needs, with a trust policy pointing at your backend's
identity. At apply-time, call `sts:AssumeRole` and get a session that expires on its own.

**Why this instead of encrypted static keys:** a Role ARN isn't a secret — there's no encryption
table to build, no rotation story, no "what if the DB leaks" question. It's also the standard
pattern real SaaS-to-AWS integrations use (Datadog, Vercel, Snowflake all do this), so it reads
as a deliberate, informed choice rather than a shortcut.

**How it works:**
1. Settings screen: paste a Role ARN, plus an External ID your backend generates per-user (extra
   protection against the "confused deputy" problem — required by AWS's own docs for third-party
   AssumeRole access).
2. Store just the ARN + External ID in the jobs table — no encryption module needed.
3. When a `plan` or `apply` subprocess runs, call `sts:AssumeRole`, get temporary credentials
   (default expiry ~1hr), inject them as that one process's environment variables only. Never
   touch LangGraph state, logs, or the workspace folder.
4. Scope the role's own IAM policy tightly (e.g. deny on IAM/root actions, deny on regions/
   services outside what the hackathon demo needs) — this is a second layer of containment even
   if everything else fails.

**Files touched:** `api/server.py`, new `frontend/app/settings`, `db/job_store.py`
(no `credential_store.py` needed — nothing to encrypt)

---

### Phase 2.5 — Prompt hardening + policy-grounded generation

*Depends on: none, can build in parallel with Phase 2*

**Goal:** Generation-time guardrails today keep the code syntactically and security-scan clean
(no open SSH, mandatory encryption). That's not the same as "safe to actually run against a real
account." This phase upgrades the Architect prompt and the retrieval layer for apply-time risk,
not just code-quality risk.

**Architect prompt additions:**
1. **Mandatory tagging** — every resource must include `ManagedBy=agent`, `JobID={job_id}`.
   This isn't optional flavor — Phase 3.5's guard and Phase 4.5's destroy path both depend on
   being able to tell "resources this agent created" from "everything else in the account."
2. **Explicit deny-list** — no wildcard IAM (`Action: "*"`, `Resource: "*"`), no deletion or
   modification of any resource not created within this job's own state, no resource counts
   that exceed what the user's request implies (stops a `count = 500` typo/hallucination from
   turning into 500 real EC2 instances).
3. **Injection-refusal clause** — the user's free-text request is untrusted input. Add an
   explicit instruction: if the request asks the agent to ignore its rules, disable a security
   check, expose credentials, or perform any action outside generating Terraform for the stated
   infra, the Architect must refuse that part of the request and say why, not attempt it.

**Retrieval upgrade — a second knowledge base for policy, not just AWS syntax:**
1. Add a small **policy document set** (even 5-10 short markdown files to start: encryption
   requirements, no-public-access rules, allowed instance sizes, tagging convention) into a
   second ChromaDB collection, separate from the AWS provider-docs collection.
2. On generation, retrieve from *both* collections — AWS docs for "how do I write this
   correctly," policy docs for "what am I allowed to do here." Surface which policy chunk
   justified a given constraint the same way you already surface AWS-doc citations.
3. This is also your strongest Atlan-relevant upgrade: it moves the RAG pipeline from "grounded
   in vendor documentation" to "grounded in organizational policy" — literally the same shape as
   a context/trust layer that grounds an agent in a company's own governance rules, just scoped
   down to your hackathon's needs.

**Files touched:** wherever your Architect system prompt currently lives (likely inside
`prompts/` or `workflows/agent_workflow_advanced_rag.py`), new `knowledge_base/policy_docs/`,
ETL script extended to index the new collection

---

## Core apply loop

### Phase 3 — Plan preview + cost delta, before the human sees anything

*Depends on: Phase 1, 2*

**Goal:** Approving generated *code* is not the same as approving *what will actually happen to
the AWS account*. Show the human exactly what will be created, changed, or destroyed — and what
it will cost — before they review anything else.

**How it works:**
1. Add a **Plan_Node** to the HITL workflow, right after validation passes and before the review
   interrupt.
2. Run `terraform init` then `terraform plan -out=tfplan`, and also `terraform show -json tfplan`
   to get a structured, parseable version (needed by Phase 3.5, not just for display).
3. Parse resource types/counts out of the JSON plan and estimate monthly cost per resource;
   compute a total delta.
4. Stream the human-readable plan text plus the cost delta to the frontend over the existing SSE
   connection, next to the generated code and trust score in the review panel.

**Files touched:** `workflows/agent_workflow_hitl.py`, `api/server.py`,
`frontend/components/HitLPanel`

---

### Phase 3.5 — Blast-Radius & Cost Guard

*Depends on: Phase 3*

**Goal:** A human reviewer can misjudge a wall of Terraform diff text, especially under demo-day
time pressure. This node does a deterministic pre-check on the structured plan — the same
"don't just trust the LLM's account of itself" instinct behind your Fixer integrity check,
applied one layer up, at the infrastructure level instead of the code level.

**How it works:**
1. Walk the JSON plan from Phase 3. For every resource marked `delete` or `update` (not
   `create`), check its tags — if it isn't tagged `ManagedBy=agent, JobID={this job}`, that means
   the plan touches something the agent didn't create. **Hard block**, regardless of how clean
   everything else looks — this is the single most important check in this phase.
2. Compare the estimated cost delta from Phase 3 against a configurable ceiling. Over the
   ceiling → block apply pending explicit override, don't just display a warning.
3. Scan generated IAM policy documents in the plan for wildcard `Action`/`Resource` or known
   privilege-escalation combinations (e.g. `iam:PassRole` + `lambda:CreateFunction`). Flag as a
   hard override even if Checkov didn't already catch it.
4. Feed all three outcomes into the existing Trust Assessor as two new override categories:
   `blast_radius_passed` and `cost_ceiling_passed`, each capable of hard-capping the trust score
   the same way `resource_integrity_passed` already does.

**Files touched:** new `workflows/blast_radius_guard.py`, extend the Trust Assessor scoring
logic in `workflows/agent_workflow_advanced_rag.py` / `agent_workflow_secure_rag.py`

---

### Phase 4 — Apply, gated by approval

*Depends on: Phase 3, 3.5*

**Goal:** This is the moment the system stops being a code generator and starts being
infrastructure. It only happens after a human has seen the real plan, the real cost, and passed
the blast-radius guard.

**How it works:**
1. Add an **Apply_Node**, reached by resuming the graph's existing interrupt with a new
   `"apply"` action — same mechanism already used for approve/patch.
2. If Phase 3.5 flagged a hard override (blast-radius or cost), require a distinct, explicit
   confirmation step from the reviewer before this node can even be reached — not just the
   normal approve click. A good pattern: make them type the resource count or job ID to confirm,
   the same way destructive actions on GitHub/Terraform Cloud require typing the repo name.
3. Run `terraform apply -auto-approve` in the workspace with the temporary AssumeRole
   credentials from Phase 2, streaming output live the same way generation logs stream today.
4. On success, save the real resource IDs and outputs onto the job record — the permanent
   receipt of what was actually built.
5. On failure, stop and surface the error; never retry an apply automatically the way the Fixer
   retries generation — an apply failure needs eyes on it, not another autonomous attempt.

**Files touched:** `workflows/agent_workflow_hitl.py`, `api/server.py`, `db/job_store.py`,
`frontend/components/HitLPanel`

---

### Phase 4.5 — Destroy path

*Depends on: Phase 4*

**Goal:** A tool that can only create real infrastructure, never remove it, is half a product —
and for a hackathon specifically, you need a reliable way to tear down anything you apply during
dev/demo or you'll be paying for orphaned resources.

**How it works:**
1. Mirror the Apply_Node: a **Destroy_Node**, reached the same way, in the same workspace, with
   the same scoped credentials.
2. Runs `terraform plan -destroy` first (reuse Phase 3's plan-preview UI) so the human sees
   exactly what's being torn down before confirming — same typed-confirmation pattern as a
   flagged apply.
3. On success, mark the job record's resources as destroyed but keep the job's history intact —
   this becomes part of the audit trail in Phase 7, not something that gets deleted.

**Files touched:** `workflows/agent_workflow_hitl.py`, `api/server.py`, `db/job_store.py`

---

### Phase 4.6 — Circuit breaker

*Depends on: Phase 4*

**Goal:** Anything that can touch real cloud resources needs an emergency stop that doesn't
depend on the agent behaving correctly.

**How it works:**
1. A single global flag (env var or DB row) checked at the very top of Apply_Node and
   Destroy_Node — `APPLY_PAUSED=true` blocks every apply/destroy immediately, no matter what
   state any individual job is in.
2. A hard timeout on the `terraform apply`/`destroy` subprocess itself (e.g. 10 minutes) — if
   it's hung, kill it and surface a failure rather than let it run indefinitely.

**Files touched:** `api/server.py`, `workflows/agent_workflow_hitl.py`

---

## What makes it worth switching to

### Phase 5 — Citations on generated code

*Independent*

**Goal:** The retriever already finds the doc chunks (AWS docs and, after Phase 2.5, policy
docs) that justify each choice — today that reasoning is thrown away after generation. Keep it,
and show it.

**How it works:**
1. Carry each retrieved chunk's source (doc name, section, and which collection — AWS docs vs.
   policy docs) alongside each generated resource block.
2. In the review UI, let the reviewer hover a setting — like EBS encryption — and see whether it
   came from AWS best-practice docs or an org policy requirement.

**Files touched:** `workflows/agent_workflow_advanced_rag.py`,
`frontend/components/TerraformViewer`

### Phase 6 — Drift detection against org policy

*Depends on: Phase 1, 4* — **cut from hackathon scope, keep as roadmap slide**

Needs elapsed real-world time to demo meaningfully; not worth build time before the deadline.

### Phase 7 — Exportable audit trail

*Depends on: Phase 4* — **cut from hackathon scope, keep as roadmap slide**

When you do build it, fold in the Phase 3.5 guardrail outcomes (blast-radius/cost override
history) alongside trust score, plan diff, reviewer identity, and apply outcome — the guardrail
trail is now part of what "the record of what happened" means.

---

## Guardrail summary — what could go wrong, and what catches it

| Risk | Guardrail | Where enforced |
|---|---|---|
| Long-lived AWS secret leaks | No static keys stored — temporary AssumeRole session only | Phase 2 |
| Generated code violates org policy, not just AWS syntax | Policy knowledge base grounds generation | Phase 2.5 |
| User prompt tries to override safety rules | Explicit injection-refusal clause in Architect prompt | Phase 2.5 |
| Fixer silently drops resources to force a pass | Deterministic pre/post resource diff | *(already built)* |
| Plan touches infra the agent didn't create | Tag-based ownership check on every delete/update | Phase 3.5 |
| Runaway cost from a hallucinated resource count | Cost ceiling as a hard trust-score override | Phase 3.5 |
| Overly broad IAM in generated code | Static wildcard/privilege-escalation scan | Phase 3.5 |
| Human rubber-stamps a risky apply under demo pressure | Distinct typed confirmation for flagged applies | Phase 4 |
| Stuck or runaway apply process | Global pause flag + per-process timeout | Phase 4.6 |
| No way to clean up applied resources | Symmetric destroy path, same approval gate | Phase 4.5 |

This table alone is worth putting on a slide — it's a compact, legible answer to "how do you make
sure an LLM doesn't do something dangerous to a real AWS account," which is the first question
any judge with an infra background will ask.

---

## Suggested build order

**Must-have for the hackathon demo:**
1. Persistent workspace (Phase 1)
2. AssumeRole credentials (Phase 2)
3. Prompt hardening + policy KB (Phase 2.5) — cheap, and it's your best differentiation story
4. Plan preview + cost (Phase 3)
5. Blast-Radius & Cost Guard (Phase 3.5)
6. Apply (Phase 4)
7. Destroy (Phase 4.5) — needed for your own dev/demo hygiene, not optional
8. Circuit breaker (Phase 4.6) — small, cheap, good "we thought about failure" talking point

**Add if time allows:**
9. Citations surfaced in UI (Phase 5) — data already exists, mostly a display task

**Roadmap slide only, don't build:**
- Drift detection (Phase 6)
- Audit export (Phase 7)

---

## Hackathon demo notes

- **Demo-safe prompts:** an apply that takes 5-10+ minutes (RDS, multi-AZ VPC) will kill your
  pacing live on stage. Keep 1-2 fast-applying prompts ready (S3 bucket, single security group,
  standalone EC2 instance) for the actual "click apply" moment, and use a heavier multi-resource
  prompt only for the "generate + plan preview" part of the demo, without clicking apply live.
- **Dog-fooding line for the pitch:** one of your IaC-Eval benchmark prompts was literally an S3
  + DynamoDB Terraform remote-state backend — exactly what this product needs to manage its own
  state. Worth a line in the pitch: "we used our own agent to generate the backend that stores
  our own product's state."
- **The AssumeRole choice is a talking point, not just a technical decision** — if a judge asks
  about security, "we deliberately avoided ever storing a long-lived AWS secret, the same way
  real platforms like Datadog or Vercel integrate" is a strong, confident answer.
