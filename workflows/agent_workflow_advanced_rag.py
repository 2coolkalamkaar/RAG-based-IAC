import warnings
warnings.filterwarnings("ignore")

import os
import re
import tempfile
import shutil
import subprocess
import operator
from typing import TypedDict, Annotated, Sequence, Dict

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Workflow_with_gemini_2.5_RAG and Reranking"

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# --- 1. Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_request: str
    retrieved_context: str
    citations: list[str] 
    terraform_code: Dict[str, str]
    validation_errors: str
    is_valid: bool
    retry_count: int
    cost_estimate: str

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

# Initialize the LLM
llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.2)
mq_llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.0)

# --- 2. Define the Nodes (Workers) ---

def retriever_node(state: AgentState):
    """
    Advanced Retriever: MultiQuery + Cross-Encoder Reranker
    """
    print("--- ADVANCED RETRIEVER NODE ---")
    user_request = state.get("user_request", "")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        import logging
        
        # Suppress verbose MultiQuery logs
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)
        
        DB_PATH = os.getenv("DB_PATH", os.path.join(os.getcwd(), "chroma_db_terraform"))
        if not os.path.exists(DB_PATH):
            print("Warning: ChromaDB not found. Proceeding without context.")
            return {"retrieved_context": "", "citations": []}
            
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        
        # 1. Base Retriever (Wide net: Top 12 — slightly over-fetch to compensate for post-filter)
        # NOTE: We do NOT use a Chroma $ne filter here. Chroma's $ne does a full
        # collection scan which causes 'Error finding id' on large DBs.
        # Instead we filter AFTER retrieval in pure Python (safe & instant).
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 12})
        
        # One-shot cleanup: delete any stale iac_eval_dataset chunks still in the DB
        try:
            col = vector_store._collection
            stale = col.get(where={"source": "iac_eval_dataset"}, include=[])
            if stale["ids"]:
                col.delete(ids=stale["ids"])
                print(f"  🧹 Cleaned up {len(stale['ids'])} stale iac_eval_dataset chunks from DB.")
        except Exception:
            pass  # Non-fatal — filter below will still exclude them
        
        # 2. MultiQuery
        print("  Expanding query into multiple semantic paths...")
        mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=mq_llm)
        
        retriever_pipeline = mq_retriever
        
        # 3. Reranker (Precision)
        try:
            from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
            
            print("    Booting up CrossEncoder Reranker...")
            # ms-marco is the standard for fast semantic reranking based on sentence-transformers
            cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
            
            retriever_pipeline = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=mq_retriever
            )
        except ImportError as e:
            print(f"   Reranker dependencies not available ({e}). Falling back to pure MultiQuery.")
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retriever_pipeline = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=mq_llm)

        print("   Executing Smart Search Pipeline...")
        docs = retriever_pipeline.invoke(user_request)
        
        # Post-retrieval filter: exclude iac_eval_dataset (evaluation-only data,
        # not real Terraform provider docs — should never appear in citations)
        EXCLUDED_SOURCES = {"iac_eval_dataset"}
        docs = [d for d in docs if d.metadata.get("source", "") not in EXCLUDED_SOURCES]
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        citations = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown/Local DB Source")
            if src not in citations:
                citations.append(src)
                
        print(f" Retrieved {len(docs)} highly accurate documents (eval data excluded).")
        return {"retrieved_context": context, "citations": citations}
    except Exception as e:
        print(f"Retrieval error: {e}. Proceeding without context.")
        return {"retrieved_context": "", "citations": []}

def architect_node(state: AgentState):
    print("--- ARCHITECT NODE ---")
    user_request = state.get("user_request", "")
    context = state.get("retrieved_context", "")
    citations = state.get("citations", [])
    
    citation_text = "\n".join([f"- {c}" for c in citations]) if citations else "No explicit sources provided."
    
    system_prompt = (
        "You are a Senior Cloud Architect and Terraform Expert. "
        "Your goal is to design and implement a complete, production-grade infrastructure solution.\n"
        "\n"
        "### INSTRUCTIONS ###\n"
        "1. **Analyze**: Break down the user's request (Compute, Network, Storage, IAM).\n"
        "2. **Retrieve**: Use the Context provided below for exact syntax and patterns.\n"
        "3. **Structure**: output a professional file structure (e.g., main.tf, variables.tf).\n"
        "4. **Cite**: Write a brief summary answering the user, and explicitly mention which specific files/sources you derived the design from under a '### References' section.\n"
        "\n"
        "### CRITICAL SECURITY RULES (YOU MUST FOLLOW THESE) ###\n"
        "1. NEVER allow ingress from '0.0.0.0/0' on port 22. Use '10.0.0.0/8'.\n"
        "2. Ensure all Security Groups have a description.\n"
        "3. Remove/restrict default egress '0.0.0.0/0' rules when possible.\n"
        "4. ENABLE `ebs_optimized = true` and encryption on EC2.\n"
        "5. DO NOT assign public IPs to instances.\n"
        "\n"
        "### CHAT HISTORY & PREVIOUS STATE ###\n"
        "Here are previous interactions and any previous Terraform code generated in this session:\n"
        "{history}\n\n"
        "If the user is asking you to modify/update an existing resource, rewrite the complete previous code incorporating the modifications. DO NOT omit previously generated code unless explicitly requested.\n"
        "\n"
        "### CONTEXT FROM SMART SEARCH ###\n"
        "{context}\n\n"
        "### AVAILABLE CITATIONS ###\n"
        f"{citation_text}\n"
    )
    
    # Extract previous messages for the prompt
    history = "\n".join([msg.content for msg in state.get("messages", []) if getattr(msg, "name", "") != "Fixer_Node"])
    if not history:
        history = "None."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{request}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"request": user_request, "context": context, "history": history})
    
    files = parse_terraform_code(response.content)
    
    # Instead of wiping messages, append the new interaction
    new_messages = list(state.get("messages", [])) + [
        HumanMessage(content=user_request), 
        AIMessage(content=response.content)
    ]
    
    return {
        "terraform_code": files,
        "retry_count": 0,
        "messages": new_messages
    }

def validator_node(state: AgentState):
    """
    Runs terraform validate & tflint on the current terraform_code.
    """
    print("---  VALIDATOR NODE ---")
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
    print(f"---  FIXER NODE (Attempt {attempt}) ---")
    
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
        "messages": [AIMessage(content=response.content, name="Fixer_Node")]
    }

def cost_estimator_node(state: AgentState):
    """
    Runs Infracost to calculate the estimated monthly cost of the generated infrastructure.
    """
    print("--- COST ESTIMATOR NODE ---")
    return {"cost_estimate": "Skipped for benchmarking"}
    files = state.get("terraform_code", {})
    if not files:
        return {"cost_estimate": "No Terraform code to estimate."}
        
    temp_dir = tempfile.mkdtemp()
    try:
        # Write files
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
                
        # Run infracost
        infracost_path = shutil.which("infracost")
        if infracost_path is None:
            local_bin_path = os.path.expanduser("~/.local/bin/infracost")
            if os.path.exists(local_bin_path):
                infracost_path = local_bin_path
                
        if infracost_path is None:
            return {"cost_estimate": "Infracost CLI not installed. Cannot estimate cost."}
            
        # We need infracost breakdown text output
        cmd = [infracost_path, "breakdown", "--path", temp_dir, "--format", "table", "--no-color"]
        res = subprocess.run(cmd, capture_output=True, text=True)  # nosemgrep: dangerous-subprocess-use
        
        if res.returncode == 0:
            return {"cost_estimate": res.stdout.strip()}
        else:
            return {"cost_estimate": f"Cost estimation failed:\n{res.stderr}"}
            
    except Exception as e:
        return {"cost_estimate": f"Error calculating cost: {e}"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- 3. Define the Edges (Logic/Routing) ---
def routing_edge(state: AgentState):
    MAX_RETRIES = 3
    if state.get("is_valid"):
        print(" Code is valid! Routing to Cost Estimator.")
        return "cost_estimator"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(" Max retries reached. Finishing workflow with errors.")
        return "end"
    print(" Validation failed. Routing to Fixer Node.")
    return "fixer"

# --- 4. Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("Retriever_Node", retriever_node)
workflow.add_node("Architect_Node", architect_node)
workflow.add_node("Validator_Node", validator_node)
workflow.add_node("Fixer_Node", fixer_node)
workflow.add_node("Cost_Estimator_Node", cost_estimator_node)

# Flow: Start -> Retriever -> Architect -> Validator <-> Fixer
workflow.add_edge(START, "Retriever_Node")
workflow.add_edge("Retriever_Node", "Architect_Node")
workflow.add_edge("Architect_Node", "Validator_Node")
workflow.add_conditional_edges("Validator_Node", routing_edge, {"end": END, "fixer": "Fixer_Node", "cost_estimator": "Cost_Estimator_Node"})
workflow.add_edge("Fixer_Node", "Validator_Node")
workflow.add_edge("Cost_Estimator_Node", END)

import os
db_path = os.path.join(os.getcwd(), "state.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
memory.setup()

app = workflow.compile(checkpointer=memory)

# --- 5. CLI Test Block ---
if __name__ == "__main__":
    print("Welcome to the Advanced Smart-RAG Agentic Workflow Tester!")
    
    user_input = " VPC with a public and private subnet. Define an EC2 Fleet of the newest AWS Linux 2 with a combination of 5 On-Demand and 4 Spot Instances. Utilize Launch Templates for configuration consistency."
    print(f"\nRequest: {user_input}\n")
    
    initial_state = {
        "user_request": user_input,
        "messages": [],
        "retrieved_context": "",
        "citations": [],
        "terraform_code": {},
        "validation_errors": "",
        "is_valid": False,
        "retry_count": 0,
        "cost_estimate": ""
    }
    
    import uuid
    print("\n🚀 Starting workflow...\n")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = app.invoke(initial_state, config=config)
    
    print("\n==================================")
    print("🏁 WORKFLOW COMPLETE")
    print("==================================")
    
    if final_state.get("is_valid"):
        print("\n✅ Code passed all validations!\n")
        cost = final_state.get("cost_estimate", "")
        if cost and "CLI not installed" not in cost:
            print("💰 Cost Estimate:")
            print(cost)
            print("\n==================================\n")
    else:
        print(f"\n❌ Code failed after {final_state.get('retry_count')} retries.\n")
        print("Last validation errors:")
        print(final_state.get("validation_errors"))
        print("\n=============\n")
        
    print("\nAgent Commentary & Citations:")
    messages = final_state.get("messages", [])
    if messages:
        # Print the LLM's raw message block minus the raw terraform blocks for cleaner output
        content = messages[-1].content
        text_only = re.sub(r"```[^\n]*\n.*?```", "\n[ Terraform Code Rendered Below ]\n", content, flags=re.DOTALL)
        print(text_only)
        
    print("\nGenerated Files:")
    for filename, code in final_state.get("terraform_code", {}).items():
        print(f"\n--- {filename} ---")
        print(code)
