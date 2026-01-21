
import os
import logging
import streamlit as st
import re
from dotenv import load_dotenv

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

st.title("Terraform Architect Agent 🧠 (Conversation Mode)")

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
    
    # We use a simple history aware retriever first to get the right QUERY, 
    # but we want to use MultiQueryRetriever for the actual RETRIEVAL.
    # To combine them, we'll manually handle the rephrasing in a Runnable if needed, 
    # but create_history_aware_retriever expects a base retriever. 
    
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
        "### RULES ###\n"
        "- If a resource attribute is missing in the Context, use a standard default but add a comment '# Note: Verified from general knowledge'.\n"
        "- Always include a `provider` block if needed.\n"
        "- Output the code in clear Markdown blocks.\n"
        "\n"
        "### CONTEXT ###\n"
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

# --- 5. Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I am your Terraform Architect. Describe the infrastructure you want to build, and we can iterate on the design together."
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the content contains terraform files (we can heuristic check), we could render nicely
        # For now, rely on markdown rendering
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., Create a 3-tier VPC architecture"):
    
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    
    # 2. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 3. Generate response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Architecting..."):
            try:
                # Convert session state messages to LangChain format for history
                chat_history = []
                for msg in st.session_state.messages[:-1]: # Exclude the just added user msg
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # Invoke the chain
                response = chain.invoke({
                    "chat_history": chat_history,
                    "input": prompt
                })
                
                answer = response["answer"]
                
                # Check if we have code files to render specially
                files = parse_terraform_code(answer)
                
                if files:
                    st.success("Generated Configuration:")
                    tabs = st.tabs(list(files.keys()))
                    for i, filename in enumerate(files.keys()):
                        with tabs[i]:
                            st.code(files[filename], language='hcl', line_numbers=True)
                    
                    # Also append the raw text explanation to the chat
                    # We might want to construct a nice markdown response
                    st.markdown(answer)
                else:
                    st.markdown(answer)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Show context in an expander (optional, for debugging/transparency)
                with st.expander("References"):
                    for i, doc in enumerate(response.get("context", [])):
                        st.caption(f"Source: {doc.metadata.get('source')}")
                        st.text(doc.page_content[:200] + "...")

            except Exception as e:
                st.error(f"Error: {e}")
