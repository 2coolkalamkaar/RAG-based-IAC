import os
import uuid
import importlib
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'workflows'))

from evaluation.eval_agent import eval_chain

WORKFLOWS = {
    "Basic": "workflows.agent_workflow",
    "RAG": "workflows.agent_workflow_rag",
    "Advanced RAG": "workflows.agent_workflow_advanced_rag"
}

def load_app(module_name):
    mod = importlib.import_module(module_name)
    return mod.app

query = "Deploy an AWS EKS Cluster (v1.30) with a managed node group. You MUST configure cluster access using the new `aws_eks_access_entry` and `aws_eks_access_policy_association` resources to grant a specific IAM role 'AmazonEKSClusterAdminPolicy' access to the cluster. Do NOT use the legacy aws-auth configmap."

for wf_name, module_name in WORKFLOWS.items():
    print(f"\n🚀 Evaluating {wf_name}...")
    app = load_app(module_name)
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "user_request": query,
        "messages": [],
        "terraform_code": {},
        "validation_errors": "",
        "is_valid": False,
        "retry_count": 0,
    }
    
    if "RAG" in wf_name:
        initial_state["retrieved_context"] = ""
    if wf_name in ["Advanced RAG", "Secure RAG"]:
        initial_state["citations"] = []
        initial_state["cost_estimate"] = ""
        
    try:
        if wf_name == "Basic":
            st_copy = initial_state.copy()
            for event in app.stream(initial_state):
                for k, v in event.items():
                    st_copy.update(v)
            final_state = st_copy
        else:
            app.invoke(initial_state, config=config)
            final_state = app.get_state(config).values
            
        files = final_state.get("terraform_code", {})
        gen_code = "\n".join([f"--- {k} ---\n{v}" for k, v in files.items()])
        if not gen_code: 
            gen_code = "NO CODE GENERATED"
            
        print("  Running LLM Judge for Drift Detection...")
        judge_res = eval_chain.invoke({
            "instruction": query, 
            "generated_code": gen_code,
            "ground_truth": "No ground truth available. Evaluate solely based on best practices and instruction adherence."
        })
        
        print(f"✅ Validation Passed: {final_state.get('is_valid')}")
        print(f"🧠 Judge Quality Score: {judge_res.score} / 5")
        print(f"📝 Drift Feedback: {judge_res.feedback}")
        
    except Exception as e:
        print(f"❌ {wf_name} failed: {e}")
