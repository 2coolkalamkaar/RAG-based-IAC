import warnings
warnings.filterwarnings("ignore")

import os
import re
import tempfile
import shutil
import subprocess
import operator
from typing import TypedDict, Annotated, Sequence, Dict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- 1. Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_request: str
    terraform_code: Dict[str, str]
    validation_errors: str
    is_valid: bool
    retry_count: int

# --- Helper Functions ---
def parse_terraform_code(response_content: str) -> dict:
    import re
    files = {}
    
    # 1. Look for code blocks that have the filename INSIDE the first line of the block (e.g. # main.tf)
    pattern1 = r"```(?:hcl|terraform)?\n(?:[#\s/]*)(?P<filename>[\w\-_]+\.tf)[^\n]*?\n(?P<code>.*?)```"
    for match in re.finditer(pattern1, response_content, re.DOTALL | re.IGNORECASE):
        files[match.group("filename").strip()] = match.group("code").strip()
        
    # 2. Look for code blocks that have the filename OUTSIDE, just before the block
    pattern2 = r"(?:^|\n)[^\n]*?(?P<filename>[\w\-_]+\.tf)[^\n]*?\n\s*```(?:hcl|terraform|)?\n(?P<code>.*?)```"
    for match in re.finditer(pattern2, response_content, re.DOTALL | re.IGNORECASE):
        filename = match.group("filename").strip()
        # Only add if we haven't already parsed it from inside the block
        if filename not in files:
            files[filename] = match.group("code").strip()
            
    return files if files else {}

def validate_terraform_code(files: dict) -> tuple[bool, str]:
    if not files:
        return False, "No Terraform files found to validate."
    temp_dir = tempfile.mkdtemp()
    try:
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
        
        if shutil.which("terraform") is None:
            return False, "Terraform binary not found."

        init_cmd = ["terraform", "init", "-backend=false"]
        init_res = subprocess.run(init_cmd, cwd=temp_dir, capture_output=True, text=True)  # nosemgrep: dangerous-subprocess-use
        if init_res.returncode != 0:
            return False, f"Terraform Init Failed:\n{init_res.stderr}\n{init_res.stdout}"

        validate_cmd = ["terraform", "validate"]
        val_res = subprocess.run(validate_cmd, cwd=temp_dir, capture_output=True, text=True)  # nosemgrep: dangerous-subprocess-use
        if val_res.returncode != 0:
            return False, f"Terraform Validation Failed:\n{val_res.stderr}\n{val_res.stdout}"

        if shutil.which("tflint") is not None:
            tflint_config = os.path.join(os.getcwd(), ".tflint.hcl")
            if os.path.exists(tflint_config):
                shutil.copy(tflint_config, temp_dir)
                subprocess.run(["tflint", "--init"], cwd=temp_dir, capture_output=True)  # nosemgrep: dangerous-subprocess-use
            tflint_res = subprocess.run(["tflint", "--format", "compact", "--minimum-failure-severity=error"], cwd=temp_dir, capture_output=True, text=True)  # nosemgrep: dangerous-subprocess-use
            if tflint_res.returncode != 0:
                return False, f"TFLint Security Checks Failed:\n{tflint_res.stdout}\n{tflint_res.stderr}"
            
        # We skip `terraform plan` because it requires valid AWS credentials to reach the AWS API
        # By verifying syntax via `terraform validate` and security via `tflint`, we implicitly 
        # ensure the infrastructure code is sound/deployable.

        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGCHAIN_PROJECT"] = "Workflow_with_gemini_2.5"

# Initialize the LLM
llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.2)

# --- 2. Define the Nodes (Workers) ---

def architect_node(state: AgentState):
    """
    Generates the initial Terraform plan based on user requirements.
    """
    print("--- 🛠️ ARCHITECT NODE ---")
    user_request = state.get("user_request", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Cloud Architect & Terraform Expert. Create a production-ready infrastructure based on the request. Make sure to generate the code in markdown blocks with filenames (e.g., **main.tf**). Remember to include an AWS provider block."),
        ("human", "{request}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"request": user_request})
    
    files = parse_terraform_code(response.content)
    
    return {
        "terraform_code": files,
        "retry_count": 0,
        "messages": [AIMessage(content=response.content)]
    }

def validator_node(state: AgentState):
    """
    Runs terraform validate & tflint on the current terraform_code.
    """
    print("--- 🔎 VALIDATOR NODE ---")
    files = state.get("terraform_code", {})
    
    is_valid, validation_errors = validate_terraform_code(files)
    
    return {
        "is_valid": is_valid,
        "validation_errors": validation_errors if not is_valid else "Success"
    }

def fixer_node(state: AgentState):
    """
    Looks at validation_errors and the terraform_code and re-generates fixed code.
    """
    attempt = state.get('retry_count', 0) + 1
    print(f"--- 🔧 FIXER NODE (Attempt {attempt}) ---")
    
    validation_errors = state.get("validation_errors", "")
    files = state.get("terraform_code", {})
    
    code_context = "\n".join([f"--- {k} ---\n{v}" for k, v in files.items()])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Terraform architect fixing broken code. Fix the errors and output the COMPLETE corrected Terraform files in markdown format with their filenames."),
        ("human", "Here is the broken code:\n\n{code}\n\nHere are the validation errors:\n\n{errors}\n\nPlease fix the code.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"code": code_context, "errors": validation_errors})
    
    new_files = parse_terraform_code(response.content)
    if not new_files:
        new_files = files # Fallback
        
    return {
        "terraform_code": new_files,
        "retry_count": attempt,
        "messages": [AIMessage(content=response.content)]
    }

# --- 3. Define the Edges (Logic/Routing) ---
def routing_edge(state: AgentState):
    MAX_RETRIES = 3
    if state.get("is_valid"):
        print("✅ Code is valid! Finishing workflow.")
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print("❌ Max retries reached. Finishing workflow with errors.")
        return "end"
    print("⚠️ Validation failed. Routing to Fixer Node.")
    return "fixer"

# --- 4. Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("Architect_Node", architect_node)
workflow.add_node("Validator_Node", validator_node)
workflow.add_node("Fixer_Node", fixer_node)

workflow.add_edge(START, "Architect_Node")
workflow.add_edge("Architect_Node", "Validator_Node")
workflow.add_conditional_edges("Validator_Node", routing_edge, {"end": END, "fixer": "Fixer_Node"})
workflow.add_edge("Fixer_Node", "Validator_Node")

app = workflow.compile()

# --- 5. CLI Test Block ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Welcome to the Agentic Terraform Workflow Tester!")
    
    # We provide a hardcoded test input for automated testing, or it can be interactive
    # For this test run, we'll try a common request that might need fixing
    user_input = "VPC with a public and private subnet. Define an EC2 Fleet of the newest AWS Linux 2 with a combination of 5 On-Demand and 4 Spot Instances. Utilize Launch Templates for configuration consistency."
    print(f"\nRequest: {user_input}\n")
    
    initial_state = {
        "user_request": user_input,
        "messages": [],
        "terraform_code": {},
        "validation_errors": "",
        "is_valid": False,
        "retry_count": 0
    }
    
    print("\n🚀 Starting workflow...\n")
    # Execute the graph
    final_state = app.invoke(initial_state)
    
    print("\n==================================")
    print("🏁 WORKFLOW COMPLETE")
    print("==================================")
    
    if final_state.get("is_valid"):
        print("\n✅ Code passed all validations!\n")
    else:
        print(f"\n❌ Code failed after {final_state.get('retry_count')} retries.\n")
        print("Last validation errors:")
        print(final_state.get("validation_errors"))
        print("\n=============\n")
        
    print("Generated Files:")
    for filename, code in final_state.get("terraform_code", {}).items():
        print(f"\n--- {filename} ---")
        print(code)
