
import os
import logging
import streamlit as st
import re
import subprocess
import tempfile
import shutil
from dotenv import load_dotenv
from diagram_generator import generate_diagram

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# Import from langchain_classic as per environment availability
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever

# --- 1. Configuration and Setup ---

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Terraform Architect Agent",
    page_icon="🧠"
)

st.title("Terraform Architect Agent 🧠 (Self-Validating)")

# Check for API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

# Database path
# Database path
DB_PATH = os.getenv("DB_PATH", os.path.join(os.getcwd(), "chroma_db_terraform"))

# --- 2. Resource Loading ---

@st.cache_resource
def load_vector_store():
    """
    Loads the existing Chroma vector store with HuggingFace embeddings.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at {DB_PATH}. Please make sure the vector store is built.")
        return None

    # Using the same embedding model as before
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_store

vector_store = load_vector_store()
if vector_store is None:
    st.stop()

# --- 3. Chain Setup (The Brain) ---

@st.cache_resource
def get_conversational_chain(_vector_store):
    """
    Creates a conversational RAG chain that:
    1. Rephrases the question if there is history.
    2. Uses MultiQueryRetriever to find context.
    3. Generates the answer.
    """
    
    # 1. The LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    
    # 2. History-Aware Retriever Component
    # This chain takes the conversation history and the latest user question 
    # and reformulates it to be a standalone question.
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Base retriever (MultiQuery)
    # Note: MultiQueryRetriever already uses an LLM to generate variations.
    # We pass the vector store as the base.
    base_retriever = MultiQueryRetriever.from_llm(
        retriever=_vector_store.as_retriever(search_kwargs={"k": 6}),
        llm=llm
    )
    
    # Create the history-aware retriever
    # This will use the LLM to rewrite the query, then pass it to the base_retriever (MultiQuery)
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever, contextualize_q_prompt
    )

    # 3. Answer Generation Component
    
    qa_system_prompt = (
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
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. Final RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Initialize the chain
chain = get_conversational_chain(vector_store)

# --- 4. Helper to Parse Output ---

def parse_terraform_code(response_content: str) -> dict:
    files = {}
    pattern_md = r"(?:\*\*|#\s)?(?P<filename>[\w\-_]+\.tf)(?:\*\*)?.*?\n```(?:hcl|terraform)?\n(?P<code>.*?)```"
    matches = re.finditer(pattern_md, response_content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code
            
    # If no files found via regex, just return None so we can display raw text
    return files if files else None

# --- 5. Terraform Validation Logic ---

def validate_terraform_code(files: dict) -> tuple[bool, str]:
    """
    Validates Terraform code by running 'terraform init' and 'terraform plan' 
    in a temporary directory.
    Returns: (is_valid, output_log)
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Write files to temp dir
        for filename, content in files.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)
        
        # 2. Check if terraform is installed
        if shutil.which("terraform") is None:
            return False, "Terraform binary not found. Please install Terraform."

        # 3. Run terraform init
        # We use -backend=false because we don't want to actually configure strict remote backends in a temp validation
        init_cmd = ["terraform", "init", "-backend=false"]
        result = subprocess.run(
            init_cmd, 
            cwd=temp_dir, 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            return False, f"Terraform Init Failed:\n{result.stderr}\n{result.stdout}"

        # 4. Run terraform validate (syntax check)
        validate_cmd = ["terraform", "validate"]
        result = subprocess.run(
            validate_cmd, 
            cwd=temp_dir, 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            return False, f"Terraform Validation Failed:\n{result.stderr}\n{result.stdout}"
            return False, f"Terraform Validation Failed:\n{result.stderr}\n{result.stdout}"

        # 4.5 Run TFLint (Security & Best Practices)
        # Check if tflint is installed
        if shutil.which("tflint") is not None:
            # Initialize tflint in the temp dir by copying the .tflint.hcl from root if it exists
            tflint_config = os.path.join(os.getcwd(), ".tflint.hcl")
            if os.path.exists(tflint_config):
                shutil.copy(tflint_config, temp_dir)
                subprocess.run(["tflint", "--init"], cwd=temp_dir, capture_output=True)

            tflint_cmd = ["tflint", "--format", "compact"]
            result = subprocess.run(
                tflint_cmd, 
                cwd=temp_dir, 
                capture_output=True, 
                text=True
            )
            # TFLint returns exit code 2 or 3 for issues, 0 for success
            if result.returncode != 0:
                # We treat lint errors as hard failures to force the LLM to fix them
                return False, f"TFLint Security Checks Failed:\n{result.stdout}\n{result.stderr}"
            
        # Note: We provide dummy credentials so the provider can initialize without erroring 
        # on missing credentials. This allows checking the logic/syntax of the plan 
        # without needing actual AWS access.
        
        env = os.environ.copy()
        env.update({
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_SECURITY_TOKEN": "testing",
            "AWS_SESSION_TOKEN": "testing",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_REGION": "us-east-1"
        })

        plan_cmd = ["terraform", "plan", "-refresh=false"] 
        result = subprocess.run(
            plan_cmd, 
            cwd=temp_dir, 
            capture_output=True, 
            text=True,
            env=env
        )
        
        # A plan failure often means credentials missing OR syntax/logic errors.
        # We return the output.
        if result.returncode != 0:
             return False, f"Terraform Plan Failed:\n{result.stderr}\n{result.stdout}"

        return True, "Terraform Plan Successful!"

    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- 6. Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I am your Self-Validating Terraform Architect. I will verify the code I generate."
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., Create a 3-tier VPC architecture"):
    
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Logic Loop (Generate -> Validate -> Fix)
    with st.chat_message("assistant"):
        
        # We need a placeholder to update the status
        status_container = st.empty()
        
        # Max retries for self-correction
        MAX_RETRIES = 3
        current_input = prompt
        
        # We need to maintain a temporary context of the conversation just for this turn
        # to include the error feedbacks without polluting the main history with N failed attempts
        temp_chat_history = []
        
        # Copy existing history logic
        main_chat_history = []
        for msg in st.session_state.messages[:-1]: 
            if msg["role"] == "user":
                main_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                main_chat_history.append(AIMessage(content=msg["content"]))

        final_answer = ""
        
        for attempt in range(MAX_RETRIES):
            with status_container:
                if attempt == 0:
                    st.info("🧠 Architecting solution...")
                else:
                    st.info(f"🔧 Applying fix (Attempt {attempt+1}/{MAX_RETRIES})...")

            try:
                # Prepare history for this invoke
                # It combines main history + any temp history from previous failed attempts in this loop
                combined_history = main_chat_history + temp_chat_history
                
                response = chain.invoke({
                    "chat_history": combined_history,
                    "input": current_input
                })
                
                answer = response["answer"]
                files = parse_terraform_code(answer)
                
                if not files:
                    # No code generated, just text. We assume it's valid conversation.
                    final_answer = answer
                    status_container.success("Response Generated!")
                    break
                
                # Check validation
                with status_container:
                    st.info("🔎 Validating: Syntax -> Security (TFLint) -> Logic (Plan)...")
                    
                success, validation_msg = validate_terraform_code(files)
                
                if success:
                    final_answer = answer + f"\n\n✅ **Verification**: Code passed Syntax, Security (TFLint), and Plan checks."
                    status_container.success("Secure & Validated Architecture Generated!")
                    break
                else:
                    # Validation Failed
                    error_report = f"Validation Errors:\n{validation_msg}"
                    with st.expander(f"⚠️ Validation Failed (Attempt {attempt+1})", expanded=False):
                        st.code(validation_msg)
                        
                    # Prepare for next iteration using the error feedback
                    temp_chat_history.append(HumanMessage(content=current_input))
                    temp_chat_history.append(AIMessage(content=answer))
                    
                    # New input for the model
                    current_input = (
                        f"The previous Terraform code you generated failed validation/security checks.\n"
                        f"Here is the error output:\n"
                        f"```text\n{validation_msg}\n```\n"
                        f"Please fix the code based on these errors. If it is a security issue (TFLint), adhere to the best practices."
                        f"Output the complete corrected Terraform files."
                    )
                    
                    if attempt == MAX_RETRIES - 1:
                        final_answer = answer + f"\n\n❌ **Verification**: Code failed validation after {MAX_RETRIES} attempts.\nSee errors in the expander above."
                        status_container.error("Maximum retries reached. Returning last best attempt.")
                        
            except Exception as e:
                st.error(f"Error: {e}")
                break

        # Display Result
        files = parse_terraform_code(final_answer)
        if files:
            st.success("Configuration Ready:")
            tabs = st.tabs(list(files.keys()))
            for i, filename in enumerate(files.keys()):
                with tabs[i]:
                    st.code(files[filename], language='hcl', line_numbers=True)
            
            # Show the narrative part of the answer
            st.markdown(final_answer)

            # Generate and Display Diagram
            st.subheader("📊 Architecture Diagram")
            with st.spinner("Generating Diagram..."):
                # Combine all tf code for parsing
                full_code = "\n".join(files.values())
                img_path = generate_diagram(full_code)
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption="Generated Architecture")
                else:
                    st.warning("Could not generate diagram (possibly due to parsing errors or no supported resources found).")
        else:
            st.markdown(final_answer)

        # Update History
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
