# Partial-Apply Recovery — Plan

> Expands Section 5 of `docs/pipeline_optimization_plan.md` into a standalone doc. That section
> committed to one specific implementation; this doc steps back and lays out *all* the reasonable
> approaches for each piece of the problem, in plain language, so the choice is deliberate rather
> than "the first idea that came up."

---

## The problem, in one paragraph

`terraform apply` creates resources one at a time. If it's asked to create 5 and dies after 2, those
2 are now real — sitting in your AWS account, costing money, existing — even though the job just
shows "failed." Nothing catches fire today because the workspace holding the record of those 2
resources is never deleted (that was fixed early on, in Phase 1). But nothing *tells anyone* those
2 resources exist either, and there's no button to do anything about them if someone did notice.
This doc is about closing that gap — properly, not with the first idea that comes to mind.

The problem splits into five smaller, mostly-independent questions. Each has more than one
reasonable answer.

---

## Question 1 — How do we find out what's actually real?

The exit code from `terraform apply` only tells you "it didn't finish cleanly." It doesn't tell you
*how much* finished. Three ways to actually find out:

### Approach A — Ask Terraform's own state file after the fact
**In simple words:** after the apply attempt ends (however it ends), run `terraform state list` in
that workspace. Terraform's state file is the ground truth of what it actually built — read it
instead of guessing from the exit code.
**Trade-off:** simple, always accurate, but only tells you *after* it's over — no visibility while
the apply is still running.
**Effort:** Low.

### Approach B — Watch the apply's live output as it happens
**In simple words:** `terraform apply` prints a line every time it finishes a resource
(`aws_instance.web: Creation complete after 12s`). Instead of waiting until the whole process ends,
read those lines as they're printed and update a running tally in real time.
**Trade-off:** gives live progress (nice for the UI — "3 of 5 done" while it's still running), but
scraping human-readable text is fragile; wording can change between Terraform versions.
**Effort:** Medium.

### Approach C — Ask for machine-readable output instead of scraping text
**In simple words:** Terraform can be told to print structured JSON instead of human-readable text
(`terraform apply -json`). Same real-time idea as Approach B, but each line is a proper JSON object
with a type field, not text you have to pattern-match — far less likely to break silently when
Terraform's wording changes.
**Trade-off:** best of both worlds (real-time + reliable), but a bigger parsing change than either
A or B alone.
**Effort:** Medium.

**Recommendation:** start with **Approach A** — it's the cheapest way to close the actual safety
gap (not knowing what's real). Approaches B/C are a nice upgrade for live UI progress later, not a
safety requirement.

---

## Question 2 — How do we stop an apply without making things worse?

Right now, a timeout kills the process instantly (`SIGKILL`) — the harshest possible stop, at the
exact moment Terraform might be mid-write to its own state file.

### Approach A — Ask nicely first, force it only if ignored
**In simple words:** send the "please wrap up" signal (`SIGTERM`) first, give Terraform a short
grace window to finish whatever single resource operation it's mid-way through and safely save
state, and only force-kill if it ignores that.
**Trade-off:** meaningfully safer, small amount of extra wait time on a true timeout (rare case).
**Effort:** Low.

### Approach B — Don't kill it at all — just page a human
**In simple words:** if an apply is taking far longer than expected, that might mean something
serious is stuck (not something a timeout should force-quit) — killing mid-operation is exactly
the scenario that risks a corrupted state file in the first place. Instead of ever forcibly killing
it, let it keep running and alert someone that it's overdue, while it continues in the background.
**Trade-off:** removes the corruption risk entirely, but a truly hung process could then run
forever unless something else eventually intervenes — needs a real "someone is watching" story to
be safe, which is more organizational overhead than code.
**Effort:** Medium (mostly process, not code).

### Approach C — Lock the workspace so nothing else can touch it while uncertain
**In simple words:** while an apply's fate is unknown (mid-timeout-grace-period, or after a
kill, before reconciliation has run), mark that workspace as "busy" so nothing else — a retry, a
destroy, a second apply — can start against it until a human or the reconciliation step clears
the lock.
**Trade-off:** this is really a *safety net around* Approaches A or B, not a replacement for
either — prevents a second operation from racing the first and making a bad state worse.
**Effort:** Low (a flag on the job record).

**Recommendation:** **Approach A** for the actual kill behavior, **Approach C** as a cheap add-on
so nothing else can pile onto an uncertain workspace. Approach B is worth knowing as the "how a
mature ops team would really do this" answer, but is more process than this project needs right now.

---

## Question 3 — How do we remember that a job is in this "partial" in-between state?

### Approach A — One new status value, one new column
**In simple words:** today `apply_status` is `applied`/`failed`/`destroyed`. Add `partial` as a
fourth value, and one new column holding the list of resources that are actually real
(`created_resources`).
**Trade-off:** minimal schema change, fits the existing table exactly as-is.
**Effort:** Low.

### Approach B — A dedicated table, one row per resource
**In simple words:** instead of cramming a list of resources into one text column on the job, give
each real resource its own row (job ID, resource address, status, when it was created). More like
a proper inventory than a note stuffed in a field.
**Trade-off:** better for later — easy to answer "show me every real resource across every job,"
which is exactly what the audit-trail backlog item (`docs/backlog.md`) would eventually want — but
it's a bigger change for a need that doesn't exist yet.
**Effort:** Medium.

**Recommendation:** **Approach A** now. Approach B is the right answer *if and when* the audit
trail backlog item gets built — no need to build the bigger version for a feature that doesn't
exist yet.

---

## Question 4 — How does a human actually fix it?

### Approach A — Always re-plan before allowing any next step
**In simple words:** never let a human (or the system) blindly retry the exact same apply or jump
straight to destroy. Instead, run a fresh `terraform plan` against the *current real state* first,
so whatever they approve next is based on what's actually there right now, not stale assumptions.
**Trade-off:** the safest option — it's the same "verify, don't assume" instinct as the Blast-Radius
Guard, applied to recovery. Costs one extra plan step before any fix.
**Effort:** Low (mostly wiring an existing node to a new entry point).

### Approach B — Offer a one-click "just clean it up" shortcut
**In simple words:** alongside the re-plan option, offer a dedicated "destroy only what's real"
button for the common case where the human just wants the mess gone, not a fresh review cycle.
**Trade-off:** faster for the most common real-world reaction to a partial failure ("just get rid
of it"), but it's an extra path to build and test, and it should still run through the same
re-plan-first safety check underneath, not skip it.
**Effort:** Low, on top of Approach A.

### Approach C — Let a human fix it outside the product entirely
**In simple words:** this is what happens today, by default, since nothing else exists yet — the
workspace directory is real and untouched, so `cd workspaces/{job_id}/ && terraform ...` always
works as a manual escape hatch.
**Trade-off:** zero build cost, works right now, but isn't a product feature — it's "the founders
know how to use a terminal." Fine as a stopgap, not something to point a judge or a real user at.
**Effort:** None (already true).

**Recommendation:** **Approach A** as the required baseline, **Approach B** as a fast follow once A
exists. Approach C is the honest fallback to mention if asked "what happens today," not a plan.

---

## Question 5 — How does a human find out this even happened?

### Approach A — A distinct badge in the existing job history UI
**In simple words:** a partial-failure job gets its own visual state (e.g. an orange "⚠️ Partial"
badge) instead of looking identical to a normal red "failed" job — so it's noticeable to anyone who
opens the history page, without needing anything new to be built beyond a status value.
**Trade-off:** passive — only helps if someone actually opens the page.
**Effort:** Low (a UI state, once Question 3 exists).

### Approach B — Push a notification the moment it's detected
**In simple words:** the moment the system realizes an apply was partial, actively tell someone —
an email, a Slack message — instead of waiting for them to check.
**Trade-off:** this is the "real production system" answer, and probably overkill before this
product has real users depending on it. Worth naming as the eventual target, not the hackathon
target.
**Effort:** Medium (needs an actual notification channel wired up).

### Approach C — At minimum, log it server-side regardless of the UI
**In simple words:** even before any UI badge exists, make sure the detection event itself gets
written to the application logs the moment it happens — so there's at least a record, discoverable
by anyone who thinks to look, even if nothing surfaces it automatically yet.
**Trade-off:** the cheapest possible safety net — not a substitute for A or B, but a good "did this
even get noticed anywhere" backstop while those are being built.
**Effort:** Trivial.

**Recommendation:** **Approach C** immediately (it's nearly free), **Approach A** as part of the
same UI work as Question 3/4, **Approach B** deferred to the backlog alongside the audit trail item.

---

## Putting it together — the combination that makes sense right now

| Question | Chosen approach | Why |
|---|---|---|
| 1. Detect what's real | A — read Terraform state after the fact | Cheapest way to close the actual safety gap |
| 2. Stop safely | A (graceful kill) + C (lock the workspace) | Removes the corruption risk without new infrastructure |
| 3. Track it | A — one status value, one column | Fits the existing schema, no premature design |
| 4. Let a human fix it | A — always re-plan first | Same "verify, don't assume" principle as the rest of this product |
| 5. Make sure it's noticed | C now, A alongside the UI work, B deferred | Free safety net first, visible UI next, notifications later |

This is the same combination Section 5.3 of `docs/pipeline_optimization_plan.md` already commits
to — this doc's job was to show the other options that were considered and *why they lost*, not to
change the answer. If priorities shift (e.g. this starts running against real customer accounts
before a proper audit trail exists), Question 3's Approach B and Question 5's Approach B become
worth revisiting.

---

## What's still just an honest answer, not a built feature

Until this is built, the correct thing to say if asked "what happens when an apply fails halfway
through" is Section 5.2's verbal answer from `docs/pipeline_optimization_plan.md` — the persistent
workspace means nothing is silently lost, there's just no in-product way to *see* or *fix* it yet.
That sentence is true today and costs nothing to say.
