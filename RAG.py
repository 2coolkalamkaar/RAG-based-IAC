import os
import logging
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import re

# --- 1. Configuration and Setup ---

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Production-Grade Terraform Generator",
    page_icon="🚀"
)

st.title("Terraform IAC Generator (Architect Edition) 🧠")

# Check for API key
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

# Database path (pre-built)
DB_PATH = "/home/rahul/RAG-based-IAC/chroma_db_terraform"

# --- 2. Resource Loading (Cached) ---

@st.cache_resource
def load_vector_store():
    """
    Loads the existing Chroma vector store with HuggingFace embeddings.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at {DB_PATH}. Please make sure the vector store is built.")
        return None

    st.info("Loading usage-optimized embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    st.info(f"Connecting to ChromaDB at {DB_PATH}...")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    st.success("Vector Store Loaded Successfully!")
    return vector_store

# Initialize Vector Store
vector_store = load_vector_store()
if vector_store is None:
    st.stop()

# --- 3. The Architect Agent Setup ---

def get_architect_chain(vector_store):
    # Setup LLM
    # Using gemini-1.5-pro as it's the current robust version
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

    # Intelligent Retriever
    # Uses LLM to generate multiple queries for better recall
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        llm=llm
    )

    # The Architect Prompt
    system_prompt = (
        "You are a Senior Cloud Architect and Terraform Expert. "
        "Your goal is to design and implement a complete, production-grade infrastructure solution.\n"
        "\n"
        "### INSTRUCTIONS ###\n"
        "1. **Analyze**: First, break down the user's request into necessary components (Compute, Network, Storage, IAM).\n"
        "2. **Retrieve**: Use the provided Context to find the correct syntax and arguments for resources.\n"
        "3. **Structure**: Organize your output into a professional file structure (e.g., main.tf, variables.tf, outputs.tf).\n"
        "4. **Synthesize**: Combine the Context (for accuracy) with your Internal Knowledge (for structure/best practices).\n"
        "\n"
        "### RULES ###\n"
        "- If a resource attribute is missing in the Context, use a standard default but add a comment '# Note: Verified from general knowledge'.\n"
        "- Always include a `provider` block if needed.\n"
        "- Output the code in clear Markdown blocks.\n"
        "\n"
        "### CONTEXT ###\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Build Chain
    chain = create_retrieval_chain(retriever_from_llm, create_stuff_documents_chain(llm, prompt))
    return chain

# --- 4. Helper to Parse Output ---

def parse_terraform_code(response_content: str) -> dict:
    files = {}
    # Regex to find blocks like:
    # **main.tf**
    # ```hcl
    # ...
    # ```
    # Or comments # main.tf
    
    # Try multiple patterns to be robust
    
    # Pattern 1: Markdown code blocks with filenames
    # looking for patterns like `filename.tf` followed by code block
    pattern_md = r"(?:\*\*|#\s)?(?P<filename>[\w\-_]+\.tf)(?:\*\*)?.*?\n```(?:hcl|terraform)?\n(?P<code>.*?)```"
    matches = re.finditer(pattern_md, response_content, re.DOTALL | re.IGNORECASE)
    
    found = False
    for match in matches:
        found = True
        filename = match.group('filename').strip()
        code = match.group('code').strip()
        files[filename] = code

    # Fallback to simple generic splitting if flexible format
    if not found:
        # Fallback: Just return the whole text as "output.tf" if it looks like code, or "plan.md"
        # For now, let's just assume the LLM follows instructions. 
        # But we can look for # filename.tf pattern from previous prompt
        pattern_comment = r"#\s*(?P<filename>\w+\.tf)\s*\n(?P<code>.*?)(?=\n#\s*\w+\.tf|\Z)"
        matches_comment = re.finditer(pattern_comment, response_content, re.DOTALL)
        for match in matches_comment:
            filename = match.group('filename').strip()
            code = match.group('code').strip()
            files[filename] = code
            
    return files

# --- 5. User Interface ---

user_input = st.text_area(
    "Describe the AWS infrastructure you want to create:",
    height=150,
    placeholder="e.g., A resilient two-tier web application with a public-facing web server and a private database server in the eu-central-1 region."
)

if st.button("✨ Architect Solution", type="primary"):
    if not user_input.strip():
        st.warning("Please describe the infrastructure you want to build.")
    else:
        with st.spinner("� Architecting solution (Generating search queries & planning)..."):
            try:
                # Initialize chain
                chain = get_architect_chain(vector_store)
                
                # Run chain
                response = chain.invoke({"input": user_input})
                
                generated_answer = response["answer"]
                retrieved_docs = response.get("context", [])

                # Parse Code
                files = parse_terraform_code(generated_answer)
                
                # Display Results
                if files:
                    st.success("Architected Solution Generated!")
                    tabs = st.tabs(list(files.keys()))
                    for i, filename in enumerate(files.keys()):
                        with tabs[i]:
                            st.code(files[filename], language='hcl', line_numbers=True)
                else:
                    st.info("Here is the architectural plan:")
                    st.markdown(generated_answer)

                # Show Context (Evidence)
                with st.expander("📚 Referenced Documentation (Context)"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source', 'Unknown')}`")
                        st.text(doc.page_content[:500] + "...")
                        st.divider()

            except Exception as e:
                st.error(f"An error occurred: {e}")