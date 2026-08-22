"""
workflows/blast_radius_guard.py
===============================
Deterministic pre-check on the real `terraform show -json tfplan` output,
BEFORE the human reviewer sees anything.

Same instinct as the Fixer resource-integrity check — we don't trust the LLM's
account of what the plan does. We walk the actual JSON plan ourselves.

Three checks, each capable of hard-blocking apply:

  1. check_ownership()     — verifies every delete/update touches only agent-tagged resources
  2. check_cost_ceiling()  — blocks if estimated cost exceeds a configurable ceiling
  3. check_iam_wildcards() — scans IAM policy documents for wildcard/escalation patterns

Results feed into Trust Assessor as two new override flags:
  - blast_radius_passed  (combines ownership + IAM checks)
  - cost_ceiling_passed
"""
import json
import re
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

COST_CEILING_USD    = float(100)   # Default: block apply if plan > $100/month
REQUIRED_MANAGED_BY = "terraform-agent"

# IAM privilege-escalation combos to flag
_ESCALATION_COMBOS = [
    {"iam:PassRole", "lambda:CreateFunction"},
    {"iam:CreatePolicyVersion", "iam:SetDefaultPolicyVersion"},
    {"iam:CreateLoginProfile", "iam:UpdateLoginProfile"},
]


# ── Helper: extract resource changes from plan JSON ───────────────────────────

def _get_resource_changes(plan_json: dict) -> list[dict]:
    """Return the list of resource_changes from a terraform plan JSON."""
    return plan_json.get("resource_changes", [])


def _get_actions(change: dict) -> list[str]:
    """Return list of actions for a resource change (e.g. ['create'], ['delete'])."""
    return change.get("change", {}).get("actions", [])


def _get_tags_after(change: dict) -> dict:
    """Return the tags dict from the 'after' state of a resource change."""
    after = change.get("change", {}).get("after") or {}
    tags = after.get("tags") or after.get("tags_all") or {}
    return tags if isinstance(tags, dict) else {}


def _get_iam_policies(plan_json: dict) -> list[str]:
    """
    Extract all inline IAM policy JSON strings from the plan.
    Looks in resource values for 'policy' fields (covers aws_iam_policy,
    aws_iam_role_policy, aws_iam_role inline documents).
    """
    policy_strings = []
    for change in _get_resource_changes(plan_json):
        for state_key in ("after", "before"):
            state = change.get("change", {}).get(state_key) or {}
            for field in ("policy", "assume_role_policy", "inline_policy"):
                val = state.get(field)
                if val and isinstance(val, str):
                    policy_strings.append(val)
    return policy_strings


# ── Check 1: Ownership (blast radius) ─────────────────────────────────────────

def check_ownership(plan_json: dict, job_id: str) -> tuple[bool, list[str]]:
    """
    Walk every resource marked delete or update in the plan.
    If a resource is being deleted/updated and its tags don't include
    JobID == job_id, it was NOT created by this job — hard block.

    Returns:
        (passed: bool, violations: list of human-readable strings)
    """
    violations = []
    for change in _get_resource_changes(plan_json):
        actions = _get_actions(change)
        # Only care about destructive or mutating actions on existing resources
        if not any(a in actions for a in ("delete", "update")):
            continue

        resource_addr = change.get("address", "unknown")
        tags = _get_tags_after(change)

        managed_by = tags.get("ManagedBy", "")
        job_id_tag = tags.get("JobID", "")

        if managed_by != REQUIRED_MANAGED_BY or job_id_tag != job_id:
            violations.append(
                f"{resource_addr}: action={actions}, "
                f"ManagedBy={managed_by!r}, JobID={job_id_tag!r} "
                f"(expected JobID={job_id!r}) — "
                f"this resource was NOT created by this job"
            )

    passed = len(violations) == 0
    return passed, violations


# ── Check 2: Cost ceiling ─────────────────────────────────────────────────────

def check_cost_ceiling(
    cost_estimate_monthly: float,
    ceiling_usd: float = COST_CEILING_USD,
) -> tuple[bool, dict]:
    """
    Block apply if estimated monthly cost exceeds ceiling_usd.

    Returns:
        (passed: bool, details: dict)
    """
    passed = cost_estimate_monthly <= ceiling_usd
    return passed, {
        "estimated_monthly_usd": round(cost_estimate_monthly, 2),
        "ceiling_usd": ceiling_usd,
        "overage_usd": max(0.0, round(cost_estimate_monthly - ceiling_usd, 2)),
    }


# ── Check 3: IAM Wildcards & Privilege Escalation ─────────────────────────────

def check_iam_wildcards(plan_json: dict) -> tuple[bool, list[str]]:
    """
    Scan all IAM policy documents in the plan for:
    - Action: "*" or Resource: "*"
    - Known privilege-escalation permission combos

    Returns:
        (passed: bool, findings: list of human-readable strings)
    """
    findings = []
    policy_strings = _get_iam_policies(plan_json)

    for raw_policy in policy_strings:
        try:
            policy = json.loads(raw_policy)
        except (json.JSONDecodeError, TypeError):
            continue

        statements = policy.get("Statement", [])
        if not isinstance(statements, list):
            statements = [statements]

        # Collect all Allow actions across statements for escalation combo check
        all_allow_actions: set[str] = set()

        for stmt in statements:
            effect  = stmt.get("Effect", "")
            actions = stmt.get("Action", [])
            resources = stmt.get("Resource", [])

            if isinstance(actions, str):
                actions = [actions]
            if isinstance(resources, str):
                resources = [resources]

            if effect != "Allow":
                continue

            all_allow_actions.update(a.lower() for a in actions)

            # Wildcard action
            if "*" in actions:
                findings.append(
                    f"IAM policy contains Action=* (wildcard) — "
                    f"this violates least-privilege policy"
                )

            # Wildcard resource with non-trivial action
            if "*" in resources:
                non_trivial = [
                    a for a in actions
                    if not re.match(r"^(ec2:Describe|s3:List|sts:GetCallerIdentity)", a, re.I)
                ]
                if non_trivial:
                    findings.append(
                        f"IAM policy contains Resource=* for non-read actions: "
                        f"{non_trivial} — scope to specific resource ARNs"
                    )

        # Escalation combo check across the whole policy
        normalised = {a.lower() for a in all_allow_actions}
        for combo in _ESCALATION_COMBOS:
            normalised_combo = {c.lower() for c in combo}
            if normalised_combo.issubset(normalised):
                findings.append(
                    f"IAM privilege-escalation pattern detected: "
                    f"{sorted(combo)} — requires explicit review"
                )

    passed = len(findings) == 0
    return passed, findings


# ── Combined guard runner ──────────────────────────────────────────────────────

def run_all_guards(
    plan_json: dict,
    job_id: str,
    cost_estimate_monthly: float,
    cost_ceiling_usd: float = COST_CEILING_USD,
) -> dict:
    """
    Run all three guards and return a combined result dict.

    Returns:
        {
            "blast_radius_passed":  bool,
            "cost_ceiling_passed":  bool,
            "ownership_violations": [...],
            "iam_findings":         [...],
            "cost_details":         {...},
            "summary":              str,
        }
    """
    ownership_ok, ownership_violations = check_ownership(plan_json, job_id)
    iam_ok, iam_findings               = check_iam_wildcards(plan_json)
    cost_ok, cost_details              = check_cost_ceiling(cost_estimate_monthly, cost_ceiling_usd)

    blast_radius_passed = ownership_ok and iam_ok
    cost_ceiling_passed = cost_ok

    parts = []
    if not ownership_ok:
        parts.append(
            f"🔴 BLAST RADIUS: Plan touches {len(ownership_violations)} unmanaged resource(s)"
        )
    if not iam_ok:
        parts.append(f"🔴 IAM WILDCARDS: {len(iam_findings)} finding(s)")
    if not cost_ok:
        overage = cost_details["overage_usd"]
        parts.append(
            f"🟡 COST CEILING: ${cost_details['estimated_monthly_usd']}/mo "
            f"exceeds ${cost_ceiling_usd}/mo ceiling by ${overage}"
        )
    if not parts:
        parts.append("✅ All guardrails passed")

    return {
        "blast_radius_passed":  blast_radius_passed,
        "cost_ceiling_passed":  cost_ceiling_passed,
        "ownership_violations": ownership_violations,
        "iam_findings":         iam_findings,
        "cost_details":         cost_details,
        "summary":              " | ".join(parts),
    }
