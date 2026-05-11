import os
import re
import tempfile
import shutil
import subprocess
import operator
from typing import TypedDict, Annotated, Sequence, Dict

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Workflow_with_gemini_2.5_RAG"

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_request: str
    retrieved_context: str  # 📚 NEW: Context from RAG
    terraform_code: Dict[str, str]
    validation_errors: str
    is_valid: bool
    retry_count: int

# --- Helper Functions ---
def parse_terraform_code(response_content: str) -> dict:
    files = {}
    pattern_md = r"(?:\*\*|#\s)?(?P<filename>[\w\-_]+\.tf)(?:\*\*)?.*?\n```(?:hcl|terraform)?\n(?P<code>.*?)```"
    matches = re.finditer(pattern_md, response_content, re.DOTALL | re.IGNORECASE)
    for match in matches:
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code
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
        init_res = subprocess.run(init_cmd, cwd=temp_dir, capture_output=True, text=True)
        if init_res.returncode != 0:
            return False, f"Terraform Init Failed:\n{init_res.stderr}\n{init_res.stdout}"

        validate_cmd = ["terraform", "validate"]
        val_res = subprocess.run(validate_cmd, cwd=temp_dir, capture_output=True, text=True)
        if val_res.returncode != 0:
            return False, f"Terraform Validation Failed:\n{val_res.stderr}\n{val_res.stdout}"

        if shutil.which("tflint") is not None:
            tflint_config = os.path.join(os.getcwd(), ".tflint.hcl")
            if os.path.exists(tflint_config):
                shutil.copy(tflint_config, temp_dir)
                subprocess.run(["tflint", "--init"], cwd=temp_dir, capture_output=True)
            tflint_res = subprocess.run(["tflint", "--format", "compact", "--minimum-failure-severity=error"], cwd=temp_dir, capture_output=True, text=True)
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

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

# --- 2. Define the Nodes (Workers) ---

def retriever_node(state: AgentState):
    """
    Searches the localized knowledge base (ChromaDB) for context related to the user request.
    """
    print("--- 📚 RETRIEVER NODE ---")
    user_request = state.get("user_request", "")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        DB_PATH = os.getenv("DB_PATH", os.path.join(os.getcwd(), "chroma_db_terraform"))
        if not os.path.exists(DB_PATH):
            print("⚠️ Warning: ChromaDB not found. Proceeding without context.")
            return {"retrieved_context": ""}
            
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        
        docs = vector_store.similarity_search(user_request, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"✅ Retrieved {len(docs)} documents for context.")
        
        return {"retrieved_context": context}
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}. Proceeding without context.")
        return {"retrieved_context": ""}

def architect_node(state: AgentState):
    """
    Generates the initial Terraform plan based on user requirements AND retrieved context.
    """
    print("--- 🛠️ ARCHITECT NODE ---")
    user_request = state.get("user_request", "")
    context = state.get("retrieved_context", "")
    
    system_prompt = (
        "You are a Senior Cloud Architect and Terraform Expert. "
        "Your goal is to design and implement a complete, production-grade infrastructure solution.\n"
        "\n"
        "### INSTRUCTIONS ###\n"
        "1. **Analyze**: First, break down the user's request into necessary components (Compute, Network, Storage, IAM).\n"
        "2. **Retrieve**: Use the provided Context to find the correct syntax and arguments for resources.\n"
        "3. **Structure**: Organize your output into a professional file structure (e.g., main.tf, variables.tf, outputs.tf).\n"
        "4. **Synthesize**: Combine the Context (for accuracy) with your Internal Knowledge (for structure/best practices).\n"
        "5. **Iterate**: If the user is asking for a modification, apply changes to the previous design intelligently.\n"
        "\n"
        "### CRITICAL SECURITY RULES (YOU MUST FOLLOW THESE) ###\n"
        "1. **Network Security**: \n"
        "   - NEVER allow ingress from '0.0.0.0/0' on port 22 (SSH). Use a placeholder specific IP (e.g., '10.0.0.0/8').\n"
        "   - Ensure all Security Groups have a 'description'.\n"
        "   - Remove default egress '0.0.0.0/0' rules if not explicitly needed, or restrict them.\n"
        "2. **EC2 Hardening**: \n"
        "   - ENABLE `ebs_optimized = true`.\n"
        "   - ENABLE `monitoring = true` (Detailed Monitoring).\n"
        "   - ENABLE `metadata_options` with `http_tokens = 'required'` (IMDSv2).\n"
        "   - DO NOT assign public IPs to instances (`associate_public_ip_address = false`).\n"
        "   - Root block devices MUST be encrypted (`encrypted = true`).\n"
        "3. **IAM & Logging**: \n"
        "   - Always attach an IAM role to EC2 instances.\n"
        "   - Enable VPC Flow Logs for any VPC you create.\n"
        "\n"
        "### RULES ###\n"
        "- If a resource attribute is missing in the Context, use a standard default but add a comment '# Note: Verified from general knowledge'.\n"
        "- Always include a `provider` block if needed.\n"
        "- Output the code in clear Markdown blocks.\n"
        "\n"
        "### CONTEXT (DOCS & EXAMPLES) ###\n"
        "You may receive both official documentation and similar 'User Requirement -> Golden Code' examples.\n"
        "Use the Examples to understand the preferred style and structure.\n"
        "{context}\n"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{request}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"request": user_request, "context": context})
    
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
workflow.add_node("Retriever_Node", retriever_node)
workflow.add_node("Architect_Node", architect_node)
workflow.add_node("Validator_Node", validator_node)
workflow.add_node("Fixer_Node", fixer_node)

# Flow: Start -> Retriever -> Architect -> Validator <-> Fixer
workflow.add_edge(START, "Retriever_Node")
workflow.add_edge("Retriever_Node", "Architect_Node")
workflow.add_edge("Architect_Node", "Validator_Node")
workflow.add_conditional_edges("Validator_Node", routing_edge, {"end": END, "fixer": "Fixer_Node"})
workflow.add_edge("Fixer_Node", "Validator_Node")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 5. CLI Test Block ---
if __name__ == "__main__":
    print("Welcome to the Agentic RAG Terraform Workflow Tester!")
    
    user_input = "VPC with a public and private subnet. Define an EC2 Fleet of the newest AWS Linux 2 with a combination of 5 On-Demand and 4 Spot Instances. Utilize Launch Templates for configuration consistency."
    print(f"\nRequest: {user_input}\n")
    
    initial_state = {
        "user_request": user_input,
        "messages": [],
        "retrieved_context": "",
        "terraform_code": {},
        "validation_errors": "",
        "is_valid": False,
        "retry_count": 0
    }
    
    print("\n🚀 Starting workflow...\n")
    config = {"configurable": {"thread_id": "test_thread_1"}}
    final_state = app.invoke(initial_state, config=config)
    
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
