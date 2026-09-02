import warnings
warnings.filterwarnings("ignore")

import os
import re
import tempfile
import shutil
import subprocess
import operator
import math
from typing import TypedDict, Annotated, Sequence, Dict

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Workflow_with_gemini_2.5_Secure_RAG"

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
    # ── Trust Score fields ───────────────────────────────────────────
    avg_retrieval_similarity:  float
    avg_reranker_score:        float
    trust_score:               float
    trust_label:               str
    trust_factors:             Dict[str, float]
    trust_explanation:         str

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
            
        if shutil.which("checkov") is not None:
            checkov_cmd = ["checkov", "-d", ".", "--soft-fail-on", "LOW,MEDIUM", "--quiet"]
            chk_res = subprocess.run(checkov_cmd, cwd=temp_dir, capture_output=True, text=True)  # nosemgrep: dangerous-subprocess-use
            if chk_res.returncode != 0:
                return False, f"Checkov Security Scan Failed (CRITICAL/HIGH issues found):\n{chk_res.stdout}"
                
        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Initialize the LLM
llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.2, streaming=True)
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
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"})
        
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
            import operator
            
            class ScorePreservingReranker(CrossEncoderReranker):
                def compress_documents(self, documents, query, callbacks=None):
                    scores = self.model.score([(query, doc.page_content) for doc in documents])
                    docs_with_scores = list(zip(documents, scores, strict=False))
                    result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
                    final_docs = []
                    for doc, score in result[:self.top_n]:
                        doc.metadata["relevance_score"] = float(score)
                        final_docs.append(doc)
                    return final_docs

            print("    Booting up Score-Preserving CrossEncoder Reranker...")
            cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = ScorePreservingReranker(model=cross_encoder, top_n=5)
            
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
        
        # ── Trust Signal 1: Reranker scores (already in metadata — 0ms) ──────────
        avg_reranker = (
            sum(d.metadata.get("relevance_score", 0.0) for d in docs) / len(docs)
        ) if docs else 0.0
        
        # ── Trust Signal 2: Base cosine similarity (one local ChromaDB call) ─────
        avg_similarity = 0.0
        if docs:
            try:
                raw_docs_with_scores = vector_store.similarity_search_with_relevance_scores(user_request, k=len(docs))
                if raw_docs_with_scores:
                    avg_similarity = sum(score for _, score in raw_docs_with_scores) / len(raw_docs_with_scores)
            except Exception:
                pass  # Non-fatal — trust assessor handles 0.0 gracefully
                
        return {
            "retrieved_context": context, 
            "citations": citations,
            "avg_retrieval_similarity": round(avg_similarity, 4),
            "avg_reranker_score":       round(avg_reranker,   4),
        }
    except Exception as e:
        print(f"Retrieval error: {e}. Proceeding without context.")
        return {
            "retrieved_context": "", 
            "citations": [],
            "avg_retrieval_similarity": 0.0,
            "avg_reranker_score":       0.0,
        }

from langchain_core.runnables.config import RunnableConfig
def architect_node(state: AgentState, config: RunnableConfig):
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
    response = chain.invoke({"request": user_request, "context": context, "history": history}, config=config)
    
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
        ("system", "You are a Terraform architect fixing broken code. Fix the errors and output the COMPLETE corrected Terraform files in markdown format with their filenames. If you are attempting to fix a Checkov security violation, but fixing it would require resources outside the scope of the user's request, or you have failed to fix it in a previous attempt, you must suppress the rule by adding '#checkov:skip=<Rule_ID>: <Justification>' inside the resource block."),
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

def _build_trust_explanation(
    retrieval_sim: float,
    reranker_norm: float,
    is_valid: bool,
    retry_count: int,
    checkov_passed: float,
) -> str:
    """
    Generates a concise, rule-based natural-language explanation of the trust score (4-factor).
    """
    parts = []

    # ── Retrieval quality ───────────────────────────────────────────────────
    r_pct = int(retrieval_sim * 100)
    if retrieval_sim >= 0.75:
        parts.append(f"Strong knowledge base coverage (retrieval similarity: {r_pct}%) grounds the Architect in real provider documentation.")
    elif retrieval_sim >= 0.50:
        parts.append(f"Moderate knowledge base coverage (retrieval similarity: {r_pct}%). Some constructs may rely on the Architect's parametric knowledge.")
    else:
        parts.append(f"Weak knowledge base coverage (retrieval similarity: {r_pct}%), reducing grounding quality.")

    # ── Reranker confidence ─────────────────────────────────────────────────
    rr_pct = int(reranker_norm * 100)
    if reranker_norm >= 0.75:
        parts.append(f"The reranker confirmed high relevance of the injected context ({rr_pct}% confidence).")
    elif reranker_norm >= 0.50:
        parts.append(f"The reranker scored retrieved sources at moderate confidence ({rr_pct}%).")
    else:
        parts.append(f"The reranker scored retrieved sources at low confidence ({rr_pct}%), suggesting weakly aligned context.")

    # ── Validation / retries ────────────────────────────────────────────────
    if not is_valid:
        parts.append("Validation failed after the maximum number of self-healing attempts. Manual review is required.")
    elif retry_count >= 3:
        parts.append(f"The code required {retry_count} self-healing fix attempts before passing validation, reducing confidence in completeness.")
    elif retry_count >= 1:
        parts.append(f"The code required {retry_count} fix attempt(s) before passing validation — minor issues were automatically corrected.")
    else:
        parts.append("The code passed syntax validation on the first attempt.")

    # ── Security Scan (Checkov) ─────────────────────────────────────────────
    if not is_valid and checkov_passed == 0.0:
        parts.append("CRITICAL: The code failed the Checkov Security Scan. Hardcoded secrets, overly permissive IAM, or insecure defaults were detected.")
    elif is_valid:
        parts.append("The generated code passed the Checkov Security Scan with no critical or high severity violations.")

    return " ".join(parts)


def trust_assessor_node(state: AgentState):
    """
    Computes a 4-factor weighted trust score for Secure RAG.
    """
    print("--- 🛡️  TRUST ASSESSOR NODE ---")

    retrieval_sim  = state.get("avg_retrieval_similarity", 0.0)
    reranker_raw   = state.get("avg_reranker_score",       0.0)
    is_valid       = state.get("is_valid",    False)
    retry_count    = state.get("retry_count", 0)
    val_errors     = state.get("validation_errors", "")

    # Normalise cross-encoder logit → 0–1 via sigmoid
    reranker_norm = 1.0 / (1.0 + math.exp(-reranker_raw)) if reranker_raw != 0 else 0.5

    validation_score = 1.0 if is_valid else 0.0
    
    # 4th Factor: Checkov
    checkov_passed = 1.0
    if not is_valid and "Checkov Security Scan Failed" in val_errors:
        checkov_passed = 0.0

    # 4-factor weighted formula
    score = (
        0.25 * retrieval_sim
      + 0.25 * reranker_norm
      + 0.30 * validation_score
      + 0.20 * checkov_passed
    )
    score = round(min(max(score, 0.0), 1.0), 3)

    # Hard override: validation failure caps trust
    if not is_valid:
        score = min(score, 0.40)

    # Tier assignment
    if score >= 0.85:
        badge, tier = "🟢", "High Trust"
    elif score >= 0.60:
        badge, tier = "🟡", "Review Recommended"
    else:
        badge, tier = "🔴", "Low Trust — Manual Check Required"

    # Rule-based label override for validation failure
    if not is_valid:
        badge, tier = "🔴", "Low Trust — Validation Failed"

    label = f"{badge} {tier}"
    factors = {
        "retrieval_similarity": round(retrieval_sim,   3),
        "reranker_score_norm":  round(reranker_norm,   3),
        "reranker_score_raw":   round(reranker_raw,    3),
        "validation_passed":    validation_score,
        "checkov_passed":       checkov_passed,
        "retry_count":          float(retry_count),
    }

    explanation = _build_trust_explanation(
        retrieval_sim, reranker_norm, is_valid, retry_count, checkov_passed
    )

    print(f"   Score: {score:.3f}  →  {label}")
    return {
        "trust_score":       score,
        "trust_label":       label,
        "trust_factors":     factors,
        "trust_explanation": explanation,
    }


def cost_estimator_node(state: AgentState):
    """
    Runs Infracost to calculate the estimated monthly cost of the generated infrastructure.
    """
    print("--- COST ESTIMATOR NODE ---")
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
        print(" Max retries reached. Routing to Trust Assessor.")
        return "trust_assessor"
    print(" Validation failed. Routing to Fixer Node.")
    return "fixer"

# --- 4. Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("Retriever_Node", retriever_node)
workflow.add_node("Architect_Node", architect_node)
workflow.add_node("Validator_Node", validator_node)
workflow.add_node("Fixer_Node", fixer_node)
workflow.add_node("Cost_Estimator_Node", cost_estimator_node)
workflow.add_node("Trust_Assessor_Node", trust_assessor_node)

# Flow: Start -> Retriever -> Architect -> Validator <-> Fixer
workflow.add_edge(START, "Retriever_Node")
workflow.add_edge("Retriever_Node", "Architect_Node")
workflow.add_edge("Architect_Node", "Validator_Node")
workflow.add_conditional_edges("Validator_Node", routing_edge, {"trust_assessor": "Trust_Assessor_Node", "fixer": "Fixer_Node", "cost_estimator": "Cost_Estimator_Node"})
workflow.add_edge("Fixer_Node", "Validator_Node")
workflow.add_edge("Cost_Estimator_Node", "Trust_Assessor_Node")
workflow.add_edge("Trust_Assessor_Node", END)

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
        "cost_estimate": "",
        "avg_retrieval_similarity": 0.0,
        "avg_reranker_score":       0.0,
        "trust_score":              0.0,
        "trust_label":              "",
        "trust_factors":            {},
        "trust_explanation":        ""
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
