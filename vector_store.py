import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load env vars
# load_dotenv()
os.environ["API_KEY"] = os.getenv("API_KEY")
api_key = os.getenv("API_KEY")


# --- CONFIGURATION ---
DOCS_PATH = "/home/rahulvaishnav068/IDP/terraform-provider-aws/website/docs/r"
DB_PATH = "./chroma_db_terraform"

# --- 1. INITIALIZE EMBEDDINGS (THE FIX) ---
# We use a local model. It runs on your CPU. No API keys, no limits.
print("🔌 Initializing local HuggingFace embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 2. CHECK IF DB EXISTS ---
if os.path.exists(DB_PATH):
    print(f"🔄 Found existing vector store at {DB_PATH}. Loading it...")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    print("✨ Creating new vector store...")
    
    # --- LOAD ---
    print("   Loading documentation...")
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.html.markdown",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    documents = loader.load()
    print(f"   Loaded {len(documents)} docs.")

    # --- SPLIT ---
    print("   Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    # --- EMBED & STORE ---
    print("   Embedding chunks locally and saving to ChromaDB...")
    # Because this is local, we can do it in bigger batches, but let's be safe
    # Chroma handles batching automatically, but local CPU can handle it fine.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✅ Vector store created and saved!")

# --- 3. SIMILARITY SEARCH TEST ---
query = "terraform code for large and compute optimised ec-2 instance"
print(f"\n🔎 Performing similarity search for: '{query}'")

results = vector_store.similarity_search(query, k=4)

print(f"\n--- Results Found: {len(results)} ---\n")

for i, doc in enumerate(results):
    print(f"📄 Result {i+1} (Source: {doc.metadata.get('source', 'Unknown')})")
    print("-" * 40)
    print(doc.page_content[:600] + "...") 
    print("\n")