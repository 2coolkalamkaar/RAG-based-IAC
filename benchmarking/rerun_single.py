import json
import uuid
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore")

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

project_name = "Workflow_with_gemini_2.5_Advanced_RAG_iac_eval"
os.environ["LANGCHAIN_PROJECT"] = project_name

wf_name = "Advanced RAG"
wf_module = "workflows.agent_workflow_advanced_rag"

print(f"🚀 Re-running Prompt 4 for {wf_name}...")
print(f"📡 LangSmith Project: {project_name}\n")

mod = importlib.import_module(wf_module)
app = mod.app

eval_llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project="project-036ddc82-f451-4fae-9e3",
    location="us-central1",
    temperature=0.0
)

class EvalResult(BaseModel):
    score: int = Field(description="Score 1-5")
    reasoning: str = Field(description="Reasoning")

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

dataset_path = "benchmarking/benchmark_dataset.jsonl"
with open(dataset_path, "r") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

prompt_data = dataset[3]
instruction = prompt_data["instruction"]
print(f"Instruction: {instruction}\n")

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

initial_state = {
    "user_request": instruction,
    "messages": [],
    "terraform_code": {},
    "validation_errors": "",
    "is_valid": False,
    "retry_count": 0,
    "retrieved_context": "",
    "citations": [],
    "cost_estimate": "",
    "avg_retrieval_similarity": 0.0,
    "avg_reranker_score": 0.0,
    "trust_score": 0.0,
    "trust_label": "",
    "trust_factors": {},
    "trust_explanation": "",
}

@traceable(
    name="Advanced RAG Run - Prompt 4",
    project_name=project_name,
    tags=["iac_eval", "advanced_rag", "rerun"],
    metadata={"instruction_idx": 3, "workflow": "Advanced RAG", "thread_id": thread_id}
)
def execute_workflow_and_judge(initial_state, config, instruction):
    os.environ["LANGCHAIN_PROJECT"] = project_name
    res = app.invoke(initial_state, config=config)
    try:
        state_obj = app.get_state(config)
        final_state = state_obj.values if state_obj and hasattr(state_obj, "values") and state_obj.values else res
    except Exception:
        final_state = res

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
        print(f"⚠️ Judge error: {e}")
        eval_score = 1
        reasoning = f"Judge exception: {e}"

    ctx_len = len(final_state.get("retrieved_context", ""))
    return final_state, input_toks, output_toks, cost_est, is_val, ret, eval_score, reasoning, ctx_len

start_time = time.time()
try:
    final_state, in_t, out_t, cost_est, is_valid, retries, score, reasoning, ctx_len = execute_workflow_and_judge(
        initial_state, config, instruction
    )
    duration = time.time() - start_time
except Exception as e:
    print(f"❌ Execution failed: {e}")
    duration = time.time() - start_time
    is_valid = False
    retries = 0
    score = 1
    reasoning = str(e)
    ctx_len = 0
    in_t = out_t = 0
    cost_est = 0.0

print(f"\n==========================================")
print(f"✅ Rerun Completed!")
print(f"Time: {duration:.1f}s | Valid: {is_valid} | Score: {score}/5 | Retries: {retries}")
print(f"Judge Reasoning: {reasoning}")
print(f"==========================================")

# Update benchmark_results.json
results_path = "benchmarking/benchmark_results.json"
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    
    updated = False
    for r in results:
        if r.get("workflow") == "Advanced RAG" and r.get("instruction_idx") == 3:
            r["time_sec"] = round(duration, 2)
            r["is_valid"] = is_valid
            r["score"] = score
            r["reasoning"] = reasoning
            r["retries"] = retries
            r["context_length"] = ctx_len
            r["input_tokens"] = in_t
            r["output_tokens"] = out_t
            r["total_tokens"] = in_t + out_t
            r["cost_usd"] = round(cost_est, 4)
            updated = True
            break
    
    if updated:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Updated benchmarking/benchmark_results.json with new run data.")

