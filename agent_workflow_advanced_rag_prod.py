import os
import re
import tempfile
import shutil
import asyncio
import operator
from typing import TypedDict, Annotated, Sequence, Dict

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Agent_Workflow_Advanced_RAG_Prod"

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
# Note: For full production, scale up to AsyncPostgresSaver instead of SqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from pydantic import BaseModel, Field

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

# --- Introducing Pydantic Structured Output ---
class TerraformFiles(BaseModel):
    files: list[dict[str, str]] = Field(
        description="List of dictionaries mapping filenames (like 'main.tf') to exact required Terraform code."
    )

def files_to_dict(tf_files: TerraformFiles) -> dict:
    """Helper to convert Pydantic to the dictionary format expected by validation."""
    return {item["filename"]: item["code"] for item in tf_files.files} if tf_files.files else {}

# --- Asynchronous Subprocess Execution Helper ---
async def validate_terraform_code(files: dict) -> tuple[bool, str]:
    if not files:
        return False, "No Terraform files found to validate."
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Write the files 
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
        
        terraform_path = shutil.which("terraform")
        if terraform_path is None:
            return False, "Terraform binary not found."

        # 2. Asynchronous Terraform Executions
        init_proc = await asyncio.create_subprocess_exec(
            terraform_path, "init", "-backend=false",
            cwd=temp_dir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await init_proc.communicate()
        if init_proc.returncode != 0:
            return False, f"Terraform Init Failed:\n{stderr.decode()}\n{stdout.decode()}"

        val_proc = await asyncio.create_subprocess_exec(
            terraform_path, "validate",
            cwd=temp_dir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await val_proc.communicate()
        if val_proc.returncode != 0:
            return False, f"Terraform Validation Failed:\n{stderr.decode()}\n{stdout.decode()}"

        tflint_path = shutil.which("tflint")
        if tflint_path is not None:
            tflint_config = os.path.join(os.getcwd(), ".tflint.hcl")
            if os.path.exists(tflint_config):
                shutil.copy(tflint_config, temp_dir)
                tflint_init = await asyncio.create_subprocess_exec(
                    tflint_path, "--init", cwd=temp_dir, 
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await tflint_init.communicate()
            
            tflint_proc = await asyncio.create_subprocess_exec(
                tflint_path, "--format", "compact", cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await tflint_proc.communicate()
            if tflint_proc.returncode != 0:
                return False, f"TFLint Security Checks Failed:\n{stdout.decode()}\n{stderr.decode()}"

        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
mq_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

# --- 2. Define the Async Nodes (Workers) ---

async def retriever_node(state: AgentState):
    print("--- ADVANCED RETRIEVER NODE ---")
    user_request = state.get("user_request", "")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        import logging
        
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)
        
        DB_PATH = os.getenv("DB_PATH", os.path.join(os.getcwd(), "chroma_db_terraform"))
        if not os.path.exists(DB_PATH):
            print("Warning: ChromaDB not found. Proceeding without context.")
            return {"retrieved_context": "", "citations": []}
            
        def load_store():
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        
        vector_store = await asyncio.to_thread(load_store)
        
        base_retriever = vector_store.as_retriever(search_kwargs={
            "k": 10,
            "filter": {"source": {"$ne": "iac_eval_dataset"}}
        })
        
        print("  Expanding query into multiple semantic paths...")
        mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=mq_llm)
        retriever_pipeline = mq_retriever
        
        try:
            from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
            
            print("    Booting up CrossEncoder Reranker...")
            def load_reranker():
                cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                return CrossEncoderReranker(model=cross_encoder, top_n=5)
                
            compressor = await asyncio.to_thread(load_reranker)
            retriever_pipeline = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=mq_retriever
            )
        except ImportError as e:
            print(f"   Reranker dependencies not available ({e}). Falling back to pure MultiQuery.")
            
        print("   Executing Smart Search Pipeline...")
        # AINVOKE replaces invoke for async
        docs = await retriever_pipeline.ainvoke(user_request)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        citations = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown/Local DB Source")
            if src not in citations:
                citations.append(src)
                
        print(f" Retrieved {len(docs)} highly accurate documents.")
        return {"retrieved_context": context, "citations": citations}
    except Exception as e:
        print(f"Retrieval error: {e}. Proceeding without context.")
        return {"retrieved_context": "", "citations": []}

async def architect_node(state: AgentState):
    print("--- ARCHITECT NODE ---")
    user_request = state.get("user_request", "")
    context = state.get("retrieved_context", "")
    citations = state.get("citations", [])
    
    citation_text = "\n".join([f"- {c}" for c in citations]) if citations else "No sources."
    
    system_prompt = (
        "You are a Senior Cloud Architect and Terraform Expert. "
        "Your goal is to design and implement a complete, production-grade infrastructure solution.\n\n"
        "### INSTRUCTIONS ###\n"
        "1. **Analyze**: Break down the user's request (Compute, Network, Storage, IAM).\n"
        "2. **Retrieve**: Use the Context provided below for exact syntax and patterns.\n"
        "3. **Structure**: Output MUST BE explicitly structured using the provided JSON format for Terraform files.\n"
        "\n"
        "### CRITICAL SECURITY RULES ###\n"
        "1. NEVER allow ingress from '0.0.0.0/0' on port 22. Use '10.0.0.0/8'.\n"
        "2. Ensure all Security Groups have a description.\n"
        "3. ENABLE `ebs_optimized = true` and encryption on EC2.\n"
        "4. DO NOT assign public IPs to instances.\n\n"
        "### CHAT HISTORY ###\n"
        "{history}\n\n"
        "### CONTEXT FROM SMART SEARCH ###\n"
        "{context}\n\n"
        "### CITATIONS ###\n"
        f"{citation_text}\n"
    )
    
    history = "\n".join([msg.content for msg in state.get("messages", []) if getattr(msg, "name", "") != "Fixer_Node"])
    if not history:
        history = "None."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{request}")
    ])
    
    # NEW Structured LLM chain
    structured_llm = llm.with_structured_output(TerraformFiles)
    chain = prompt | structured_llm
    
    # Async Invoke
    response = await chain.ainvoke({"request": user_request, "context": context, "history": history})
    
    # Parse strictly into a correct dict
    files = files_to_dict(response)
    
    # Provide a readable string for message history since Pydantic returned an object
    readable_response = f"I have generated {len(files)} files based on your constraints.\n\n"
    for filename, code in files.items():
        readable_response += f"**{filename}**\n```terraform\n{code}\n```\n\n"
    
    new_messages = list(state.get("messages", [])) + [
        HumanMessage(content=user_request), 
        AIMessage(content=readable_response)
    ]
    
    return {
        "terraform_code": files,
        "retry_count": 0,
        "messages": new_messages
    }

async def validator_node(state: AgentState):
    print("--- VALIDATOR NODE ---")
    files = state.get("terraform_code", {})
    
    # Call async validation
    is_valid, validation_errors = await validate_terraform_code(files)
    
    return {
        "is_valid": is_valid,
        "validation_errors": validation_errors if not is_valid else "Success"
    }

async def fixer_node(state: AgentState):
    attempt = state.get('retry_count', 0) + 1
    print(f"--- FIXER NODE (Attempt {attempt}) ---")
    
    validation_errors = state.get("validation_errors", "")
    files = state.get("terraform_code", {})
    
    code_context = "\n".join([f"--- {k} ---\n{v}" for k, v in files.items()])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Terraform architect fixing broken code. Output identically in the strict required structured formatting."),
        ("human", "Here is the broken code:\n\n{code}\n\nHere are the validation errors:\n\n{errors}\n\nPlease fix the files.")
    ])
    
    # Use Structured Output
    structured_llm = llm.with_structured_output(TerraformFiles)
    chain = prompt | structured_llm
    
    # Async Invoke
    response = await chain.ainvoke({"code": code_context, "errors": validation_errors})
    
    new_files = files_to_dict(response)
    if not new_files:
        new_files = files # Fallback
        
    readable_response = f"I fixed {len(new_files)} files based on the errors.\n\n"
    for filename, code in new_files.items():
        readable_response += f"**{filename}**\n```terraform\n{code}\n```\n\n"
    
    return {
        "terraform_code": new_files,
        "retry_count": attempt,
        "messages": [AIMessage(content=readable_response)]
    }

async def cost_estimator_node(state: AgentState):
    print("--- COST ESTIMATOR NODE ---")
    files = state.get("terraform_code", {})
    if not files:
        return {"cost_estimate": "No Terraform code to estimate."}
        
    temp_dir = tempfile.mkdtemp()
    try:
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
                
        infracost_path = shutil.which("infracost")
        if infracost_path is None:
            local_bin_path = os.path.expanduser("~/.local/bin/infracost")
            if os.path.exists(local_bin_path):
                infracost_path = local_bin_path
                
        if infracost_path is None:
            return {"cost_estimate": "Infracost CLI not installed. Cannot estimate cost."}
            
        # Async Subprocess for Infracost
        proc = await asyncio.create_subprocess_exec(
            infracost_path, "breakdown", "--path", temp_dir, "--format", "table", "--no-color",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            return {"cost_estimate": stdout.decode().strip()}
        else:
            return {"cost_estimate": f"Cost estimation failed:\n{stderr.decode()}"}
            
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

workflow.add_edge(START, "Retriever_Node")
workflow.add_edge("Retriever_Node", "Architect_Node")
workflow.add_edge("Architect_Node", "Validator_Node")
workflow.add_conditional_edges("Validator_Node", routing_edge, {"end": END, "fixer": "Fixer_Node", "cost_estimator": "Cost_Estimator_Node"})
workflow.add_edge("Fixer_Node", "Validator_Node")
workflow.add_edge("Cost_Estimator_Node", END)

from langgraph.checkpoint.memory import MemorySaver

# For global exports (like Streamlit importing `app`), we use MemorySaver 
# because it operates perfectly with async workflows out-of-the-box.
# Initializing database connections (like AsyncSqliteSaver) requires an active async event loop.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 5. CLI Test Block ---
async def main():
    print("Welcome to the Advanced Smart-RAG Agentic Workflow Tester (PRODUCTION READY VERSION)!")
    
    user_input = "create a AWS codebuild project resource with example iam role, environment variables, secondary sources, secondary artifacts."
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
    
    print("\n🚀 Starting workflow...\n")
    config = {"configurable": {"thread_id": "advanced_test_thread"}}
    
    # In production, to persist to a database async, you use a context manager like this:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    db_path = os.path.join(os.getcwd(), "state.db")
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as db_memory:
        app_with_db = workflow.compile(checkpointer=db_memory)
        # Now explicitly awaiting ainvoke
        final_state = await app_with_db.ainvoke(initial_state, config=config)
    
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
        
    print("\nAgent Commentary:")
    messages = final_state.get("messages", [])
    if messages:
        # Pydantic is already mapped back to a clean AIMessage locally
        content = messages[-1].content
        print(content)

if __name__ == "__main__":
    # Crucial for async execution in python script runs
    asyncio.run(main())
