import json
import uuid
import time
import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langsmith import traceable

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import importlib

load_dotenv()

WORKFLOWS = {
    "Basic": "workflows.agent_workflow",
    "Standard RAG": "workflows.agent_workflow_rag",
    "Advanced RAG": "workflows.agent_workflow_advanced_rag",
    "Secure RAG": "workflows.agent_workflow_secure_rag"
}

PROJECT_NAMES = {
    "Basic": "Workflow_with_gemini_2.5_iac_eval",
    "Standard RAG": "Workflow_with_gemini_2.5_RAG_iac_eval",
    "Advanced RAG": "Workflow_with_gemini_2.5_Advanced_RAG_iac_eval",
    "Secure RAG": "Workflow_with_gemini_2.5_Secure_RAG_iac_eval"
}

def load_app(module_name):
    mod = importlib.import_module(module_name)
    return mod.app

# Setup the Judge
eval_llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project="project-036ddc82-f451-4fae-9e3",
    location="us-central1",
    temperature=0.0
)

class EvalResult(BaseModel):
    score: int = Field(description="Score 1-5 for Terraform code quality, adherence to instructions, and security best practices")
    reasoning: str = Field(description="Architectural reasoning explaining the score and any deductions")

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert, strict Principal Cloud Architect and HashiCorp Certified Terraform Judge.\n"
        "Rate the following generated Terraform code against the user instruction on a scale of 1 to 5:\n"
        "- 5/5: Perfect adherence to instruction, sound syntax, modular structure, excellent AWS security practices.\n"
        "- 4/5: Fully satisfies instruction with correct syntax, minor optimization or documentation improvements possible.\n"
        "- 3/5: Mostly satisfies instruction, but missing important security options or has non-critical omissions.\n"
        "- 2/5: Missing major requested resources/components or contains significant configuration errors.\n"
        "- 1/5: Broken syntax, hallucinated invalid attributes, or fails the core objective.\n"
        "Output structured JSON only."
    )),
    ("human", "Instruction:\n{instruction}\n\nGenerated Code:\n```terraform\n{generated_code}\n```")
])
eval_chain = eval_prompt | eval_llm.with_structured_output(EvalResult)

def run_benchmarks():
    dataset_path = "benchmarking/benchmark_dataset.jsonl"
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # Backup existing results if present
    results_path = "benchmarking/benchmark_results.json"
    if os.path.exists(results_path):
        backup_path = "benchmarking/benchmark_results_backup.json"
        shutil.copy(results_path, backup_path)
        print(f"📦 Backed up previous results to {backup_path}")

    results = []

    print(f"🚀 Starting IaC Workflow Benchmarking ({len(WORKFLOWS)} Workflows x {len(dataset)} Prompts = {len(WORKFLOWS)*len(dataset)} Total Runs)...\n")

    for wf_name, wf_module in WORKFLOWS.items():
        project_name = PROJECT_NAMES.get(wf_name, f"Workflow_{wf_name}_iac_eval")
        os.environ["LANGCHAIN_PROJECT"] = project_name

        print("=" * 60)
        print(f"📊 Benchmarking Workflow: {wf_name} (LangSmith Project: {project_name})")
        print("=" * 60)

        app = load_app(wf_module)

        for i, data in enumerate(dataset):
            instruction = data['instruction']
            print(f"\n  [{i+1}/{len(dataset)}] Testing: {instruction[:65]}...")

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            initial_state = {
                "user_request": instruction,
                "messages": [],
                "terraform_code": {},
                "validation_errors": "",
                "is_valid": False,
                "retry_count": 0,
            }
            if "RAG" in wf_name:
                initial_state["retrieved_context"] = ""
            if "Advanced" in wf_name or "Secure" in wf_name:
                initial_state["citations"] = []
                initial_state["cost_estimate"] = ""
                initial_state["avg_retrieval_similarity"] = 0.0
                initial_state["avg_reranker_score"] = 0.0
                initial_state["trust_score"] = 0.0
                initial_state["trust_label"] = ""
                initial_state["trust_factors"] = {}
                initial_state["trust_explanation"] = ""

            @traceable(
                name=f"{wf_name} Run - Prompt {i+1}",
                project_name=project_name,
                tags=["iac_eval", wf_name.lower().replace(" ", "_")],
                metadata={"instruction_idx": i, "workflow": wf_name, "thread_id": thread_id}
            )
            def execute_workflow_and_judge(initial_state, config, instruction, wf_name):
                os.environ["LANGCHAIN_PROJECT"] = project_name
                if wf_name == "Basic":
                    final_state = app.invoke(initial_state)
                else:
                    res = app.invoke(initial_state, config=config)
                    try:
                        state_obj = app.get_state(config)
                        final_state = state_obj.values if state_obj and hasattr(state_obj, "values") and state_obj.values else res
                    except Exception:
                        final_state = res

                # Token tracking & Cost Estimation
                input_toks = 0
                output_toks = 0
                for msg in final_state.get("messages", []):
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        input_toks += msg.usage_metadata.get("input_tokens", 0)
                        output_toks += msg.usage_metadata.get("output_tokens", 0)

                cost_est = (input_toks / 1_000_000) * 1.25 + (output_toks / 1_000_000) * 5.00

                is_val = final_state.get("is_valid", False)
                ret = final_state.get("retry_count", 0)

                files = final_state.get("terraform_code", {})
                gen_code = "\n".join([f"--- {k} ---\n{v}" for k, v in files.items()])
                if not gen_code:
                    gen_code = "NO CODE GENERATED"

                try:
                    judge_result = eval_chain.invoke({"instruction": instruction, "generated_code": gen_code})
                    eval_score = judge_result.score
                    reasoning = judge_result.reasoning
                except Exception as e:
                    print(f"    ⚠️ Judge failed: {e}")
                    eval_score = 1
                    reasoning = f"Judge execution error: {e}"

                ctx_len = len(final_state.get("retrieved_context", ""))
                return final_state, input_toks, output_toks, cost_est, is_val, ret, eval_score, reasoning, ctx_len

            start_time = time.time()
            try:
                final_state, input_tokens, output_tokens, cost_estimate, is_valid, retries, score, reasoning, ctx_len = execute_workflow_and_judge(
                    initial_state, config, instruction, wf_name
                )
                duration = time.time() - start_time

            except Exception as e:
                print(f"    ❌ Workflow Error: {e}")
                duration = time.time() - start_time
                is_valid = False
                retries = 0
                score = 1
                reasoning = f"Workflow exception: {e}"
                ctx_len = 0
                input_tokens = 0
                output_tokens = 0
                cost_estimate = 0.0

            print(f"    -> Time: {duration:.1f}s | Valid: {is_valid} | Score: {score}/5 | Retries: {retries} | Tokens: {input_tokens+output_tokens} | Cost: ${cost_estimate:.4f}")
            print(f"    -> Judge Reasoning: {reasoning[:120]}...")

            results.append({
                "workflow": wf_name,
                "instruction_idx": i,
                "time_sec": round(duration, 2),
                "is_valid": is_valid,
                "score": score,
                "reasoning": reasoning,
                "retries": retries,
                "context_length": ctx_len,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": round(cost_estimate, 4)
            })

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            time.sleep(5)  # slight pause between tests

    print(f"\n✅ IaC Evaluation Benchmark complete! All {len(results)} results saved to {results_path}")

if __name__ == "__main__":
    run_benchmarks()
