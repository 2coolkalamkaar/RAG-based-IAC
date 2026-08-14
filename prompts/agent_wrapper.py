"""
Promptfoo Python Provider Wrapper
----------------------------------
Calls the Advanced RAG workflow with a given prompt and returns
the raw terraform_code as a combined string for Promptfoo to assert against.

Promptfoo Python provider interface:
  - Imports this file and calls call_api(prompt, options, context)
  - Must return {"output": str, "metadata": dict}
"""

import os
import sys
import uuid

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.agent_workflow_advanced_rag import app


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Called by Promptfoo for each test case.
    Runs the Advanced RAG pipeline and returns the Terraform output.
    """
    thread_id = f"promptfoo-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_request": prompt,
        "messages": [],
        "terraform_code": {},
        "validation_errors": "",
        "is_valid": False,
        "retry_count": 0,
        "retrieved_context": "",
        "citations": [],
        "cost_estimate": "",
    }

    try:
        result = app.invoke(initial_state, config=config)
    except Exception as e:
        return {
            "output": f"PIPELINE_ERROR: {e}",
            "metadata": {"error": str(e), "is_valid": False},
        }

    tf_code: dict = result.get("terraform_code", {})
    is_valid: bool = result.get("is_valid", False)
    retries: int = result.get("retry_count", 0)

    # Combine all files into a single string Promptfoo can assert on
    combined = "\n\n".join(
        f"# === {fname} ===\n{code}" for fname, code in tf_code.items()
    )

    if not combined:
        combined = "NO_TERRAFORM_OUTPUT"

    return {
        "output": combined,
        "metadata": {
            "is_valid": is_valid,
            "retries": retries,
            "files_generated": list(tf_code.keys()),
            "citations": result.get("citations", []),
        },
    }
