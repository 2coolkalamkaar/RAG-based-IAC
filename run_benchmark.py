import json
import uuid
import time
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
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
    "Basic": "agent_workflow",
    "Standard RAG": "agent_workflow_rag",
    "Advanced RAG": "agent_workflow_advanced_rag"
}

# Setup the Judge
eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

class EvalResult(BaseModel):
    score: int = Field(description="Score 1-5 for code quality")
    reasoning: str = Field(description="Reasoning for score")

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Terraform judge. Rate the following generated code from 1 to 5 against the instruction. Focus on functionality, security, and best practices. Output the structured JSON format only."),
    ("human", "Instruction: {instruction}\n\nGenerated Code:\n```terraform\n{generated_code}\n```")
])
eval_chain = eval_prompt | eval_llm.with_structured_output(EvalResult)

def run_benchmarks():
    dataset_path = "benchmark_dataset.jsonl"
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    results = []

    print("🚀 Starting Workflow Benchmarking...\n")

    for wf_name, wf_module in WORKFLOWS.items():
        print(f"=========================================")
        print(f"📊 Benchmarking Workflow: {wf_name}")
        print(f"=========================================")
        app = load_app(wf_module)

        for i, data in enumerate(dataset):
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

            print(f"    -> Time: {duration:.1f}s | Valid: {is_valid} | Score: {score}/5 | Retries: {retries} | Context Len: {ctx_len}")

            results.append({
                "workflow": wf_name,
                "instruction_idx": i,
                "time_sec": round(duration, 2),
                "is_valid": is_valid,
                "score": score,
                "retries": retries,
                "context_length": ctx_len
            })

            time.sleep(2) # rate limit

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\\n✅ Benchmark complete! Saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmarks()
