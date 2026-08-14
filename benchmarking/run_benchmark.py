import json
import uuid
import time
import os
from dotenv import load_dotenv

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load the apps
import importlib

def load_app(module_name):
    mod = importlib.import_module(module_name)
    return mod.app

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Benchmarking RAGs"
WORKFLOWS = {
    "Secure RAG": "workflows.agent_workflow_secure_rag"
}

# Setup the Judge
eval_llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.0)

class EvalResult(BaseModel):
    score: int = Field(description="Score 1-5 for code quality")
    reasoning: str = Field(description="Reasoning for score")

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Terraform judge. Rate the following generated code from 1 to 5 against the instruction. Focus on functionality, security, and best practices. Output the structured JSON format only."),
    ("human", "Instruction: {instruction}\n\nGenerated Code:\n```terraform\n{generated_code}\n```")
])
eval_chain = eval_prompt | eval_llm.with_structured_output(EvalResult)

def run_benchmarks():
    dataset_path = "benchmarking/benchmark_dataset.jsonl"
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    # ONLY run the last instruction (idx 2)
    dataset = [dataset[2]]

    results = []
    if os.path.exists("benchmarking/benchmark_results.json"):
        try:
            with open("benchmarking/benchmark_results.json", "r") as r:
                results = json.load(r)
        except Exception:
            pass

    print("🚀 Starting Workflow Benchmarking...\n")

    for wf_name, wf_module in WORKFLOWS.items():
        print(f"=========================================")
        print(f"📊 Benchmarking Workflow: {wf_name}")
        print(f"=========================================")
        app = load_app(wf_module)

        for _, data in enumerate(dataset):
            i = 2 # Hardcode the index to 2 for the JSON output since we sliced the dataset
            instruction = data['instruction']
            print(f"  [{i+1}/{len(dataset)}] Testing: {instruction[:50]}...")

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
            if wf_name == "Advanced RAG":
                initial_state["citations"] = []
                initial_state["cost_estimate"] = ""

            start_time = time.time()
            try:
                # Basic workflow does not require config
                if wf_name == "Basic":
                    # For basic we have to accumulate the stream to get final state
                    final_state = initial_state.copy()
                    for event in app.stream(initial_state):
                        for k, v in event.items():
                            final_state.update(v)
                else:
                    app.invoke(initial_state, config=config)
                    final_state = app.get_state(config).values

                duration = time.time() - start_time
                
                # Token tracking & Cost Estimation
                input_tokens = 0
                output_tokens = 0
                for msg in final_state.get("messages", []):
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        input_tokens += msg.usage_metadata.get("input_tokens", 0)
                        output_tokens += msg.usage_metadata.get("output_tokens", 0)
                
                # Gemini Pricing (approx): $1.25 / 1M input, $5.00 / 1M output
                cost_estimate = (input_tokens / 1_000_000) * 1.25 + (output_tokens / 1_000_000) * 5.00
                
                is_valid = final_state.get("is_valid", False)
                retries = final_state.get("retry_count", 0)
                
                files = final_state.get("terraform_code", {})
                gen_code = "\\n".join([f"--- {k} ---\\n{v}" for k,v in files.items()])
                if not gen_code: gen_code = "NO CODE GENERATED"

                # LLM Judge
                try:
                    judge_res = eval_chain.invoke({"instruction": instruction, "generated_code": gen_code})
                    score = judge_res.score
                except Exception as e:
                    print(f"    ⚠️ Judge failed: {e}")
                    score = 1
                
                # Context chars for token proxy
                ctx_len = len(final_state.get("retrieved_context", ""))

            except Exception as e:
                print(f"    ❌ Workflow Error: {e}")
                duration = time.time() - start_time
                is_valid = False
                retries = 0
                score = 1
                ctx_len = 0
                input_tokens = 0
                output_tokens = 0
                cost_estimate = 0.0

            print(f"    -> Time: {duration:.1f}s | Valid: {is_valid} | Score: {score}/5 | Retries: {retries} | Tokens: {input_tokens+output_tokens} | Cost: ${cost_estimate:.4f}")

            results.append({
                "workflow": wf_name,
                "instruction_idx": i,
                "time_sec": round(duration, 2),
                "is_valid": is_valid,
                "score": score,
                "retries": retries,
                "context_length": ctx_len,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": round(cost_estimate, 4)
            })
            
            # Save incrementally
            with open("benchmarking/benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)

            time.sleep(15) # rate limit

    print("\\n✅ Benchmark complete! Saved to benchmarking/benchmark_results.json")

if __name__ == "__main__":
    run_benchmarks()
