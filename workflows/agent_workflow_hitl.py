import warnings
warnings.filterwarnings("ignore")

import os
import re
import tempfile
import shutil
import subprocess
import operator
import sqlite3
from typing import TypedDict, Annotated, Sequence, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Workflow_with_gemini_2.5_HitL_RAG"

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

# ─────────────────────────────────────────────────
# 1. Agent State
# ─────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], operator.add]
    user_request:      str
    retrieved_context: str
    citations:         list[str]
    terraform_code:    Dict[str, str]
    validation_errors: str
    is_valid:          bool
    retry_count:       int
    # HitL-specific fields
    hitl_action:       str   # "approve" | "patch"
    patch_request:     str   # Any natural language change the human wants
    upload_mode:       bool  # True = SRE uploaded files, skip Retriever+Architect


# ─────────────────────────────────────────────────
# 2. Helper Functions
# ─────────────────────────────────────────────────
def parse_terraform_code(response_content: str) -> dict:
    files = {}
    pattern1 = r"```(?:hcl|terraform)?\n(?:[#\s/]*)(?P<filename>[\w\-_]+\.tf)[^\n]*?\n(?P<code>.*?)```"
    for match in re.finditer(pattern1, response_content, re.DOTALL | re.IGNORECASE):
        files[match.group("filename").strip()] = match.group("code").strip()
    pattern2 = r"(?:^|\n)[^\n]*?(?P<filename>[\w\-_]+\.tf)[^\n]*?\n\s*```(?:hcl|terraform|)?\n(?P<code>.*?)```"
    for match in re.finditer(pattern2, response_content, re.DOTALL | re.IGNORECASE):
        filename = match.group("filename").strip()
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

        init_res = subprocess.run(
            ["terraform", "init", "-backend=false"], cwd=temp_dir, capture_output=True, text=True
        )
        if init_res.returncode != 0:
            return False, f"Terraform Init Failed:\n{init_res.stderr}\n{init_res.stdout}"

        val_res = subprocess.run(
            ["terraform", "validate"], cwd=temp_dir, capture_output=True, text=True
        )
        if val_res.returncode != 0:
            return False, f"Terraform Validation Failed:\n{val_res.stderr}\n{val_res.stdout}"

        if shutil.which("tflint") is not None:
            tflint_config = os.path.join(os.getcwd(), ".tflint.hcl")
            if os.path.exists(tflint_config):
                shutil.copy(tflint_config, temp_dir)
                subprocess.run(["tflint", "--init"], cwd=temp_dir, capture_output=True)
            tflint_res = subprocess.run(
                ["tflint", "--format", "compact", "--minimum-failure-severity=error"],
                cwd=temp_dir, capture_output=True, text=True
            )
            if tflint_res.returncode != 0:
                return False, f"TFLint Checks Failed:\n{tflint_res.stdout}\n{tflint_res.stderr}"

        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────
# 3. LLM Initialization
# ─────────────────────────────────────────────────
llm    = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.2)
mq_llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.0)


# ─────────────────────────────────────────────────
# 4. Nodes
# ─────────────────────────────────────────────────

def upload_entry_node(state: AgentState):
    """
    SRE Upload Mode entry point.
    terraform_code is already injected by the UI.
    This node is a no-op passthrough to the Validator.
    """
    print("--- 📂 UPLOAD ENTRY NODE (SRE Mode) ---")
    print(f"   Received {len(state.get('terraform_code', {}))} file(s) from SRE upload.")
    return {"is_valid": False, "retry_count": 0}


def retriever_node(state: AgentState):
    """Advanced Retriever: MultiQuery + Cross-Encoder Reranker"""
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

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 12})

        try:
            col = vector_store._collection
            stale = col.get(where={"source": "iac_eval_dataset"}, include=[])
            if stale["ids"]:
                col.delete(ids=stale["ids"])
        except Exception:
            pass

        print("  Expanding query into multiple semantic paths...")
        mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=mq_llm)
        retriever_pipeline = mq_retriever

        try:
            from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
            print("    Booting up CrossEncoder Reranker...")
            cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
            retriever_pipeline = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=mq_retriever
            )
        except ImportError as e:
            print(f"   Reranker not available ({e}). Falling back to MultiQuery.")
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retriever_pipeline = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=mq_llm)

        print("   Executing Smart Search Pipeline...")
        docs = retriever_pipeline.invoke(user_request)
        EXCLUDED_SOURCES = {"iac_eval_dataset"}
        docs = [d for d in docs if d.metadata.get("source", "") not in EXCLUDED_SOURCES]
        context = "\n\n".join([doc.page_content for doc in docs])
        citations = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in citations:
                citations.append(src)
        print(f" Retrieved {len(docs)} documents.")
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
        "3. **Structure**: Output a professional file structure (e.g., main.tf, variables.tf).\n"
        "4. **Cite**: Mention which files/sources you derived the design from under '### References'.\n"
        "\n"
        "### CHAT HISTORY ###\n"
        "{history}\n\n"
        "### CONTEXT FROM SMART SEARCH ###\n"
        "{context}\n\n"
        "### AVAILABLE CITATIONS ###\n"
        f"{citation_text}\n"
    )

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
    new_messages = list(state.get("messages", [])) + [
        HumanMessage(content=user_request),
        AIMessage(content=response.content)
    ]
    return {"terraform_code": files, "retry_count": 0, "messages": new_messages}


def validator_node(state: AgentState):
    print("---  VALIDATOR NODE ---")
    files = state.get("terraform_code", {})
    is_valid, validation_errors = validate_terraform_code(files)
    return {"is_valid": is_valid, "validation_errors": validation_errors if not is_valid else "Success"}


def fixer_node(state: AgentState):
    attempt = state.get("retry_count", 0) + 1
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
    new_files = parse_terraform_code(response.content) or files
    return {
        "terraform_code": new_files,
        "retry_count": attempt,
        "messages": [AIMessage(content=response.content, name="Fixer_Node")]
    }


def hitl_node(state: AgentState):
    """
    Human-in-the-Loop node. Pauses execution via LangGraph interrupt().
    The UI resumes this by calling app.invoke() with updated hitl_action + patch_request.
    """
    print("--- ⏸️  HITL NODE — Waiting for human review ---")
    terraform_code = state.get("terraform_code", {})
    # interrupt() surfaces the current code to the UI and suspends the graph.
    # The value passed here is available in app.get_state(config).tasks[0].interrupts
    human_decision = interrupt({
        "terraform_code": terraform_code,
        "message": "Review the generated Terraform code. Approve or request changes."
    })
    # When the graph is resumed, human_decision will contain the UI-provided dict.
    return {
        "hitl_action":   human_decision.get("hitl_action", "approve"),
        "patch_request": human_decision.get("patch_request", ""),
    }


def patcher_node(state: AgentState):
    """
    Surgical Patcher Node.
    Receives the existing terraform_code dict + any natural language patch_request.
    Outputs ONLY the files that need to change. Merges them back into the existing state.
    This avoids regenerating the entire codebase for a targeted change.
    """
    print("--- 🔧 PATCHER NODE ---")
    files = state.get("terraform_code", {})
    patch_request = state.get("patch_request", "")

    code_context = "\n\n".join([f"=== {k} ===\n{v}" for k, v in files.items()])

    system_prompt = (
        "You are a senior Terraform engineer performing a targeted, surgical code change.\n"
        "\n"
        "### YOUR TASK ###\n"
        "The user has requested the following change to an existing Terraform codebase:\n"
        "\"{patch_request}\"\n"
        "\n"
        "### STRICT RULES ###\n"
        "1. Output ONLY the files that you actually modified. Do NOT output unchanged files.\n"
        "2. When you output a file, output its COMPLETE new content (not just the diff).\n"
        "3. Use standard markdown code blocks with the filename on the first comment line, e.g.:\n"
        "   ```hcl\n"
        "   # main.tf\n"
        "   <complete file content>\n"
        "   ```\n"
        "4. If the change requires a new file, create it. If it requires removing a resource, remove it.\n"
        "5. If the request is ambiguous, make the most reasonable, production-safe interpretation.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the full current Terraform codebase:\n\n{code}\n\nApply this change: {request}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "code": code_context,
        "request": patch_request,
        "patch_request": patch_request
    })

    patched_files = parse_terraform_code(response.content)

    if patched_files:
        # Surgical merge: only update the files that were actually changed
        merged = dict(files)
        merged.update(patched_files)
        print(f"   Patched {len(patched_files)} file(s): {list(patched_files.keys())}")
    else:
        print("   ⚠️  Patcher returned no parseable files. Keeping existing code.")
        merged = files

    return {
        "terraform_code": merged,
        "retry_count": 0,
        "hitl_action": "",
        "patch_request": "",
        "messages": [AIMessage(content=response.content, name="Patcher_Node")]
    }


# ─────────────────────────────────────────────────
# 5. Routing Logic
# ─────────────────────────────────────────────────

def start_routing(state: AgentState):
    """Route from START: skip to Validator if in upload mode, else normal RAG flow."""
    if state.get("upload_mode", False):
        return "upload_entry"
    return "retriever"


def validator_routing(state: AgentState):
    """After Validator: route to HitL (pass), Fixer (fail), or END (max retries)."""
    MAX_RETRIES = 3
    if state.get("is_valid"):
        print(" Code is valid! Routing to HitL review.")
        return "hitl"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(" Max retries reached. Finishing workflow with errors.")
        return "end"
    print(" Validation failed. Routing to Fixer Node.")
    return "fixer"


def hitl_routing(state: AgentState):
    """After HitL resumes: route to Patcher or END based on human decision."""
    action = state.get("hitl_action", "approve")
    if action == "patch":
        print(f" Human requested patch: '{state.get('patch_request', '')[:60]}...'")
        return "patcher"
    print(" Human approved. Workflow complete.")
    return "end"


# ─────────────────────────────────────────────────
# 6. Graph Construction
# ─────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("Upload_Entry_Node",  upload_entry_node)
workflow.add_node("Retriever_Node",     retriever_node)
workflow.add_node("Architect_Node",     architect_node)
workflow.add_node("Validator_Node",     validator_node)
workflow.add_node("Fixer_Node",         fixer_node)
workflow.add_node("HitL_Node",          hitl_node)
workflow.add_node("Patcher_Node",       patcher_node)

# START: branch on upload_mode
workflow.add_conditional_edges(START, start_routing, {
    "upload_entry": "Upload_Entry_Node",
    "retriever":    "Retriever_Node"
})

# Normal RAG path
workflow.add_edge("Retriever_Node",  "Architect_Node")
workflow.add_edge("Architect_Node",  "Validator_Node")

# Upload path rejoins at Validator
workflow.add_edge("Upload_Entry_Node", "Validator_Node")

# Validator can go to HitL, Fixer, or END
workflow.add_conditional_edges("Validator_Node", validator_routing, {
    "hitl":   "HitL_Node",
    "fixer":  "Fixer_Node",
    "end":    END
})

# Fixer loops back to Validator
workflow.add_edge("Fixer_Node", "Validator_Node")

# HitL can approve (END) or patch (Patcher)
workflow.add_conditional_edges("HitL_Node", hitl_routing, {
    "patcher": "Patcher_Node",
    "end":     END
})

# After patching, re-validate
workflow.add_edge("Patcher_Node", "Validator_Node")

# ─────────────────────────────────────────────────
# 7. Compile with Checkpointer
# ─────────────────────────────────────────────────
db_path = os.path.join(os.getcwd(), "state.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
memory.setup()

app = workflow.compile(checkpointer=memory, interrupt_before=["HitL_Node"])


# ─────────────────────────────────────────────────
# 8. CLI Test Block
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid
    print("Welcome to the HitL Agentic Workflow Tester!")

    user_input = "Create an S3 bucket with versioning enabled and server-side encryption."
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "user_request":      user_input,
        "messages":          [],
        "retrieved_context": "",
        "citations":         [],
        "terraform_code":    {},
        "validation_errors": "",
        "is_valid":          False,
        "retry_count":       0,
        "hitl_action":       "",
        "patch_request":     "",
        "upload_mode":       False,
    }

    print("\n🚀 Starting workflow...\n")
    app.invoke(initial_state, config=config)

    state = app.get_state(config)
    print(f"\n⏸️  Workflow paused at: {state.next}")
    print("Generated files:", list(state.values.get("terraform_code", {}).keys()))

    # Simulate human approval
    print("\n✅ Simulating human approval...")
    app.invoke({"hitl_action": "approve", "patch_request": ""}, config=config)
    print("\n🏁 Workflow complete!")
