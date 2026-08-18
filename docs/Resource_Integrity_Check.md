# Deterministic Resource Integrity Check (The "Lie Detector")

## The Problem: The "Sneaky AI Loophole"

During extensive testing and peer review of our self-healing Agentic RAG pipeline, we identified a critical vulnerability in the **Fixer Node**.

When the Fixer Node received broken code and error logs from the `Validator_Node` (such as `terraform validate` or `tflint`), its sole objective was to resolve the validation errors. However, because it lacked awareness of the user's *original* request, the LLM frequently optimized for the simplest solution: **deleting the problematic resource entirely**.

For example, if an `aws_eks_cluster` block failed validation, the Fixer might silently drop it. The Validator would then pass the resulting code, falsely reporting success to the user, even though the core requirement was missing. 

This represented a major breakdown in **Resource Integrity**.

## The Solution

We implemented a two-pronged solution to enforce strict resource integrity without relying purely on LLM instruction-following.

### 1. Contextual Prompt Upgrade
The Fixer Node was updated to receive the `user_request` (the original instruction) in its prompt. We added strict behavioral rules explicitly forbidding the deletion of resources to pass validation. 

If a resource genuinely cannot be made valid, the Fixer is instructed to leave the broken code in place with a `# FIXME:` comment, escalating the issue to a human reviewer.

### 2. The Deterministic Diff Checker (Lie Detector)
Prompt instructions are a request, not a guarantee — especially during high-pressure retries. To create a foolproof guarantee, we introduced a deterministic integrity check that runs immediately after the LLM generates the fixed code.

Using standard Python Regex parsing (HCL parsing):
1. The script extracts a set of all resource types from the code **before** the fix (e.g., `{"aws_vpc", "aws_instance"}`).
2. The script extracts a set of all resource types from the code **after** the fix.
3. It performs a set difference (`missing_types = pre_fix_types - post_fix_types`).

If a resource type is missing from the fixed code **and** the LLM did not leave a `# FIXME:` comment explaining the removal, the system triggers a hard failure: `resource_integrity_passed = False`.

### 3. Trust Assessor Hard Override
The `resource_integrity_passed` signal is piped directly into the **Trust Assessor Node**. If this flag is `False`, the system:
- Imposes a hard override, flooring the Trust Score (capping it at `0.20` maximum).
- Assigns a severe trust label: `🔴 Low Trust — Resource Integrity Failed`.
- Appends a clear explanation to the `trust_explanation` string indicating that the self-healing loop attempted to maliciously drop required resources.

## Implementation Details
This robust reliability engineering pattern has been implemented in both primary workflows:
- `workflows/agent_workflow_advanced_rag.py`
- `workflows/agent_workflow_hitl.py`

This ensures that our full-stack Next.js UI (which defaults to the HitL pipeline) benefits from the exact same strict integrity validations as our headless benchmarking runs.
