import os
import re
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# --- NEW RAG IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
# -------------------------

# --- 1. Configuration and Setup ---

# Load environment variables from a .env file
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Production-Grade Terraform Generator",
    page_icon="🚀"
)

# The path to your local Terraform documentation
DOCS_PATH = "/home/rahulvaishnav068/IDP/terraform-provider-aws/website/docs/r"

# Your original, production-grade system prompt
# We will now combine this with the RAG context.
SYSTEM_PROMPT = """
You are a world-class DevOps architect and a Terraform expert with a deep specialization in creating secure, scalable, and highly available AWS infrastructure. Your task is to generate a complete, production-hardened, and reusable Terraform module configuration based on the user's request.

Follow these rules with absolute precision:

1.  **Modular File Structure:** Split definitions into `main.tf`, `vpc.tf`, `security.tf`, `compute.tf`, `variables.tf`, `outputs.tf`, and `backend.tf`.
2.  **Highly Available & Scalable Network:** Multi-AZ by default, HA NAT Gateways (one per AZ), and zonal routing.
3.  **Security First (Principle of Least Privilege):** No default for `allowed_ssh_cidr_blocks`, instances in private subnets, and dynamic security group rules.
4.  **Outputs & Connectivity:** Provide the `ssm_connection_command` for secure access, not a direct SSH command.
5.  **Code Quality & Reusability:** Clean, conventional code. Use `locals` for common tags. Never hardcode AMIs or AZs.
6.  **Output Format:** Provide ONLY the HCL code, marked with file comments (e.g., `# main.tf`). No other explanations.
"""

# --- NEW: RAG Prompt Template ---
# This new prompt combines your original rules (SYSTEM_PROMPT) with the retrieved
# documentation (context) and the user's query (input).
RAG_PROMPT_TEMPLATE = f"""
{SYSTEM_PROMPT}

You will use the following retrieved Terraform documentation as context to ensure you are using the correct resource names, arguments, and attributes.

<context>
{{context}}
</context>

Based on all the rules above and the provided context, generate the code for the user's request:

User Request: {{input}}
"""

# --- 2. RAG Pipeline Setup (with Caching) ---

# NEW: Use st.cache_resource to load, split, and embed docs only once.
# This is CRITICAL for performance. It runs once and is stored in memory.
@st.cache_resource
def load_and_index_documents():
    """
    Loads, splits, embeds, and indexes the documentation.
    This function is cached by Streamlit.
    """
    st.info("Loading and indexing Terraform documentation (this happens once)...")
    
    # 1. LOAD (Your code)
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.html.markdown",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    if not documents:
        st.error(f"No documents found at path: {DOCS_PATH}. Please check the path.")
        return None

    # 2. SPLIT (Your code)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased chunk size for better context
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        st.error("Failed to split documents into chunks.")
        return None

    # 3. STORE (The new part)
    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create the vector store from the chunks
    # This process can take a minute or two the first time
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    st.success(f"Successfully loaded and indexed {len(chunks)} document chunks.")
    return vector_store

# --- 3. Helper Function to Parse LLM Output ---

def parse_terraform_code(response_content: str) -> dict:
    """
    Parses the raw LLM output string into a dictionary of filename: code.
    (This is your original, unchanged function)
    """
    files = {}
    pattern = r"#\s*(?P<filename>\w+\.tf)\s*\n(?P<code>.*?)(?=\n#\s*\w+\.tf|\Z)"
    matches = re.finditer(pattern, response_content, re.DOTALL)
    for match in matches:
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code
    return files

# --- 4. Streamlit User Interface ---

st.title("Terraform IAC Generator Using RAG")

# Check for API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

# --- NEW: Initialize the RAG pipeline ---
try:
    vector_store = load_and_index_documents()
    if vector_store is None:
        st.error("Failed to initialize the RAG pipeline. Please check the `DOCS_PATH` and document content.")
        st.stop()
        
    # Create the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Retrieve top 10 chunks

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0) # Using 1.5 Pro for better context handling

    # Create the prompt template from our string
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Create the "stuff" chain: This takes the retrieved docs and "stuffs" them into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the RAG chain: This combines (1) Retriever and (2) Stuff Chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

except Exception as e:
    st.error(f"Error initializing the application: {e}")
    st.stop()
# --- End of NEW Initialization ---


# Use st.text_area for a larger input box
user_input = st.text_area(
    "Describe the AWS infrastructure you want to create:",
    height=150,
    placeholder="e.g., A resilient two-tier web application with a public-facing web server and a private database server in the eu-central-1 region."
)

if st.button("✨ Generate Terraform Code", type="primary"):
    if not user_input.strip():
        st.warning("Please describe the infrastructure you want to build.")
    else:
        with st.spinner("🧑‍💻 Architecting your infrastructure (RAG-style)..."):
            try:
                # --- NEW: Invoke the RAG chain ---
                # This one call does all the work:
                # 1. Takes `user_input`
                # 2. Embeds it
                # 3. Searches the vector store for relevant docs
                # 4. "Stuffs" the docs and the `user_input` into the `RAG_PROMPT_TEMPLATE`
                # 5. Sends the combined prompt to the LLM
                # 6. Returns the full response
                
                response = rag_chain.invoke({"input": user_input})
                
                # The LLM's final answer is now in the 'answer' key
                generated_code = response['answer']
                
                # We can also see the retrieved docs for debugging
                retrieved_context = response['context']

                # --- End of NEW invocation ---

                terraform_files = parse_terraform_code(generated_code)

                if not terraform_files:
                    st.error("Failed to parse the output from the AI. The AI may not have followed the requested format.")
                    st.subheader("Raw AI Output:")
                    st.code(generated_code, language='text')
                else:
                    st.success("Terraform code generated successfully!")
                    
                    filenames = list(terraform_files.keys())
                    tabs = st.tabs(filenames)

                    for i, filename in enumerate(filenames):
                        with tabs[i]:
                            st.code(terraform_files[filename], language='hcl', line_numbers=True)

                    # NEW: Add an expander to show the context that was used
                    with st.expander("Show Retrieved Context (What RAG used)"):
                        for i, doc in enumerate(retrieved_context):
                            st.info(f"**Context Chunk {i+1}** (from `{doc.metadata.get('source', 'N/A')}`)")
                            st.text(doc.page_content)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")