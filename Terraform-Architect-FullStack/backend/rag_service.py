
import os
import shutil
import tempfile
import subprocess
import hashlib
import re
import sys
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.retrievers import MultiQueryRetriever

# Load Env
load_dotenv()
DB_PATH = os.path.join(os.getcwd(), "chroma_db_terraform")

# Load Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    
    # 1. Contextualize Question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    base_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        llm=llm
    )
    history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_q_prompt)

    # 2. Answer Question
    qa_system_prompt = (
        "You are a Senior Cloud Architect and Terraform Expert. "
        "Your goal is to design and implement a complete, production-grade infrastructure solution.\n"
        "\n"
        "### CRITICAL SECURITY RULES (YOU MUST FOLLOW THESE) ###\n"
        "1. **Network Security**: \n"
        "   - NEVER allow ingress from '0.0.0.0/0' on port 22 (SSH). Use a placeholder specific IP.\n"
        "   - Ensure all Security Groups have a 'description'.\n"
        "   - Remove default egress '0.0.0.0/0' rules if not explicitly needed.\n"
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
        "### CONTEXT (DOCS & EXAMPLES) ###\n"
        "You may receive both official documentation and similar 'User Requirement -> Golden Code' examples.\n"
        "Use the Examples to understand the preferred style and structure.\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Initialize Chain
chain = get_conversational_chain()

def parse_terraform_code(response_content: str) -> dict:
    files = {}
    pattern_md = r"(?:\*\*|#\s)?(?P<filename>[\w\-_]+\.tf)(?:\*\*)?.*?\n```(?:hcl|terraform)?\n(?P<code>.*?)```"
    matches = re.finditer(pattern_md, response_content, re.DOTALL | re.IGNORECASE)
    for match in matches:
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code
    return files if files else None

def validate_terraform_code(files: dict) -> tuple[bool, str]:
    temp_dir = tempfile.mkdtemp()
    try:
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)
        
        if shutil.which("terraform") is None:
            return False, "Terraform binary not found."

        # Init
        subprocess.run(["terraform", "init", "-backend=false"], cwd=temp_dir, capture_output=True)

        # Validate
        res_val = subprocess.run(["terraform", "validate"], cwd=temp_dir, capture_output=True, text=True)
        if res_val.returncode != 0:
            return False, f"Terraform Validation Failed:\n{res_val.stderr}\n{res_val.stdout}"

        # TFLint
        if shutil.which("tflint") is not None:
             tflint_config = os.path.join(os.getcwd(), ".tflint.hcl")
             if os.path.exists(tflint_config):
                 shutil.copy(tflint_config, temp_dir)
                 subprocess.run(["tflint", "--init"], cwd=temp_dir, capture_output=True)
             
             res_lint = subprocess.run(["tflint", "--format", "compact"], cwd=temp_dir, capture_output=True, text=True)
             if res_lint.returncode != 0:
                 return False, f"TFLint Security Checks Failed:\n{res_lint.stdout}\n{res_lint.stderr}"

        # Plan (Dummy Creds)
        env = os.environ.copy()
        env.update({
            "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_SECURITY_TOKEN": "testing", "AWS_SESSION_TOKEN": "testing",
            "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1"
        })
        res_plan = subprocess.run(["terraform", "plan", "-refresh=false"], cwd=temp_dir, capture_output=True, text=True, env=env)
        if res_plan.returncode != 0:
             return False, f"Terraform Plan Failed:\n{res_plan.stderr}\n{res_plan.stdout}"

        return True, "Terraform Plan Successful!"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def generate_diagram_image(terraform_code: str) -> str:
    """Returns filename of the generated image"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    
    user_prompt = (
        f"Generate a Python script using the `diagrams` library to visualize this Terraform code.\n"
        f"Save the diagram to filename='diagram_output'.\n"
        f"Use `show=False`.\n"
        f"Group resources logically.\n"
        f"Only output raw Python code.\n"
        f"```hcl\n{terraform_code}\n```"
    )
    
    try:
        # Generate script
        response = llm.invoke(user_prompt)
        python_code = response.content
        python_code = re.sub(r"^```python\n", "", python_code)
        python_code = re.sub(r"^```\n", "", python_code)
        python_code = re.sub(r"\n```$", "", python_code)

        # Run script
        script_path = "temp_diag.py"
        with open(script_path, "w") as f:
            f.write(python_code)
        
        subprocess.run([sys.executable, script_path], capture_output=True)
        
        if os.path.exists("diagram_output.png"):
            # Move to a static dir or return bytes? 
            # Ideally, return base64 or path. For simplicity, let's assume we read bytes.
            return "diagram_output.png"
        return None
    except Exception:
        return None
