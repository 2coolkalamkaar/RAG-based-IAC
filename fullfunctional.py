import os
import re
import logging
import shutil
import tempfile
import subprocess
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION ---
# Set logging to see the AI's thought process
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()

# Ensure API Key is set (Best practice: use .env file)
if "GOOGLE_API_KEY" not in os.environ:
    # Fallback if not in .env (Use your actual key here if testing locally)
    os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY", "AIzaSyCOoiiC4Tnuhslqj8-WiJ91TEiw88MivDc")

DB_PATH = "./chroma_db_terraform"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(DB_PATH):
    print(f"❌ Error: Database not found at {DB_PATH}. Run vector_store.py first.")
    exit()

# --- 2. CONNECT TO VECTOR STORE ---
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

# --- 3. SETUP THE BRAIN (LLM) ---
# Using 1.5-pro (2.5 isn't standard public API yet, adjust if you have specific access)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

# --- 4. INTELLIGENT RETRIEVER ---
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
    llm=llm
)

# --- 5. DEFINE VALIDATION TOOLS (Checkov & Terraform) ---

# def save_code_to_temp(llm_output):
#     """Parses LLM output and saves files to a temporary directory."""
#     temp_dir = tempfile.mkdtemp()
    
#     # Regex to find code blocks. Expecting format:
#     # # filename.tf
#     # code...
#     pattern = r"#\s*(?P<filename>\w+\.tf)\s*\n(?P<code>.*?)(?=\n#\s*\w+\.tf|\Z)"
#     matches = re.finditer(pattern, llm_output, re.DOTALL)
    
#     files_created = []
#     for match in matches:
#         filename = match.group('filename').strip()
#         code = match.group('code').strip()
        
#         filepath = os.path.join(temp_dir, filename)
#         with open(filepath, 'w') as f:
#             f.write(code)
#         files_created.append(filename)
        
#     return temp_dir, files_created

def save_code_to_temp(llm_output):
    """Parses LLM output, cleans markdown, and saves files to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    
    # Regex to find code blocks based on your prompt structure:
    # # filename.tf
    # code...
    pattern = r"#\s*(?P<filename>\w+\.tf)\s*\n(?P<code>.*?)(?=\n#\s*\w+\.tf|\Z)"
    matches = re.finditer(pattern, llm_output, re.DOTALL)
    
    files_created = []
    for match in matches:
        filename = match.group('filename').strip()
        raw_code = match.group('code')

        # --- THE FIX: CLEANING THE CODE ---
        # 1. Remove starting markdown fences like ```terraform or ```hcl
        clean_code = re.sub(r'^```(terraform|hcl)?\s*', '', raw_code.strip())
        # 2. Remove ending markdown fences ```
        clean_code = re.sub(r'```\s*$', '', clean_code.strip())
        # 3. Strip any remaining leading/trailing whitespace
        clean_code = clean_code.strip()
        
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(clean_code)
        files_created.append(filename)
        
    return temp_dir, files_created

def run_checkov(directory):
    """Runs Checkov security scan."""
    print("   🛡️  Running Security Scan (Checkov)...")
    try:
        # --quiet to reduce noise, --soft-fail to not crash on low severity
        cmd = ["checkov", "--directory", directory, "--framework", "terraform", "--quiet", "--compact"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Checkov returns 0 on success, 1 on failure
        if result.returncode == 0:
            return True, "Security Check Passed."
        else:
            # We capture the output to feed back to the LLM
            return False, f"Checkov Security Failures:\n{result.stdout}"
    except FileNotFoundError:
        return False, "Checkov is not installed. Run 'pip install checkov'."

def run_terraform_plan(directory):
    """Runs Terraform Init and Plan."""
    print("   🌍 Running Terraform Plan...")
    try:
        # 1. Init
        subprocess.run(["terraform", "init"], cwd=directory, check=True, capture_output=True)
        
        # 2. Plan
        cmd = ["terraform", "plan", "-no-color"]
        result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Terraform Plan Syntax Errors:\n{result.stderr}"
    except Exception as e:
        return False, f"Terraform Execution Error: {str(e)}"

# --- 6. THE ARCHITECT PROMPT ---
# --- 6. THE HARDENED ARCHITECT PROMPT ---

system_prompt = (
    "You are a Senior Cloud Architect and Terraform Expert. "
    "Your goal is to design and implement a highly secure, production-grade infrastructure solution.\n"
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
    "### INSTRUCTIONS ###\n"
    "1. **Retrieve**: Use the Context to find correct arguments.\n"
    "2. **Structure**: Output files clearly labeled with `# filename.tf`.\n"
    "3. **Completeness**: Always include `provider` blocks and `terraform` version blocks.\n"
    "\n"
    "### FORMATTING RULES ###\n"
    "You must provide the code in this exact format for parsing:\n"
    "# main.tf\n"
    "resource \"aws_...\" {{ ... }}\n\n"
    "# variables.tf\n"
    "variable \"...\" {{ ... }}\n"
    "\n"
    "### CONTEXT ###\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# We use a simple chain here because the LOOP handles the complex logic

rag_chain = (
    {"context": retriever_from_llm, "input": RunnablePassthrough()}
    | prompt_template
    | llm  
    | StrOutputParser()
)

# --- 7. THE SELF-CORRECTION AGENT LOOP ---

# def agent_loop(user_query):
#     max_retries = 5
#     attempt = 0
#     error_feedback = "" # Stores errors from Checkov/Terraform to feed back to LLM

#     while attempt < max_retries:
#         print(f"\n🔄 Attempt {attempt + 1}/{max_retries}...")
        
#         # 1. Prepare Input (Append errors if this is a retry)
#         if error_feedback:
#             current_input = f"{user_query}\n\nPREVIOUS ATTEMPT FAILED. FIX THESE ERRORS:\n{error_feedback}"
#         else:
#             current_input = user_query

#         # 2. Generate Code using RAG
#         print("   🧠 Generating Architected Solution...")
#         llm_response = rag_chain.invoke(current_input)
        
#         # 3. Save to Temp
#         temp_dir, files = save_code_to_temp(llm_response)
#         if not files:
#             print("   ⚠️ No code blocks found in output. Retrying...")
#             error_feedback = "You did not provide any code blocks formatted with '# filename.tf'. Please provide the code."
#             attempt += 1
#             continue

#         print(f"   📂 Generated files: {files}")

#         # 4. Security Scan
#         secure, security_msg = run_checkov(temp_dir)
#         if not secure:
#             print("   ❌ Security Issues Found. Correcting...")
#             error_feedback = security_msg
#             attempt += 1
#             shutil.rmtree(temp_dir) # Clean up
#             continue # Loop back
        
#         # 5. Terraform Plan
#         valid, plan_msg = run_terraform_plan(temp_dir)
#         if not valid:
#             print("   ❌ Terraform Plan Failed. Correcting...")
#             error_feedback = plan_msg
#             attempt += 1
#             shutil.rmtree(temp_dir) # Clean up
#             continue # Loop back

#         # 6. Success!
#         print("\n✅ SUCCESS! Infrastructure Validated and Secure.")
#         print("="*50)
#         print(llm_response)
#         print("="*50)
        
#         # Cleanup and break
#         shutil.rmtree(temp_dir) 
#         return

#     print("\n❌ Agent failed to produce valid code after max retries.")
#     print("Last Error:\n" + error_feedback)

# ... (Keep all your imports and setup code the same) ...

# --- 7. THE CHECKOV-ONLY AGENT LOOP ---

def agent_loop(user_query):
    max_retries = 5
    attempt = 0
    error_feedback = "" 

    # Create a permanent directory to inspect the final result
    final_output_dir = "./generated_infra"
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir)

    while attempt < max_retries:
        print(f"\n🔄 Attempt {attempt + 1}/{max_retries}...")
        
        # 1. Prepare Input
        if error_feedback:
            current_input = f"{user_query}\n\nPREVIOUS ATTEMPT FAILED. FIX THESE SECURITY ERRORS:\n{error_feedback}"
        else:
            current_input = user_query

        # 2. Generate Code
        print("   🧠 Generating Architected Solution...")
        llm_response = rag_chain.invoke(current_input)
        
        # 3. Save to Temp
        temp_dir, files = save_code_to_temp(llm_response)
        if not files:
            print("   ⚠️ No code blocks found. Retrying...")
            error_feedback = "You did not provide any code blocks formatted with '# filename.tf'."
            attempt += 1
            continue

        print(f"   📂 Generated files: {files}")

        # 4. Security Scan (Checkov)
        secure, security_msg = run_checkov(temp_dir)
        if not secure:
            print("   ❌ Security Issues Found. Correcting...")
            error_feedback = security_msg
            attempt += 1
            shutil.rmtree(temp_dir) # Clean up temp
            continue # Loop back to fix security issues
        
        # --- TERRAFORM PLAN DISABLED FOR NOW ---
        # valid, plan_msg = run_terraform_plan(temp_dir)
        # if not valid:
        #     ...
        
        # 5. Success!
        print("\n✅ SUCCESS! Infrastructure is Secure (Checkov Passed).")
        
        # Copy files from temp to a visible folder so you can see them
        for filename in files:
            shutil.copy(os.path.join(temp_dir, filename), final_output_dir)
            
        print(f"   💾 Code saved to: {final_output_dir}")
        print("="*50)
        print(llm_response)
        print("="*50)
        
        shutil.rmtree(temp_dir) 
        return

    print("\n❌ Agent failed to produce secure code after max retries.")
    print("Last Security Error:\n" + error_feedback)

# ... (Keep the execution loop at the bottom) ...


# --- 8. EXECUTION ---
print("🤖 Intelligent Architect Agent (with Security & Validation) is ready!")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("\nRequest: ")
    if user_query.lower() == "exit":
        break
    
    agent_loop(user_query)