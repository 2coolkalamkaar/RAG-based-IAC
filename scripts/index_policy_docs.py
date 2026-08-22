"""
scripts/index_policy_docs.py
============================
Indexes the policy docs in knowledge_base/policy_docs/ into a separate
ChromaDB collection named "policy_docs", distinct from the AWS provider-docs
collection "terraform_docs".

Run once (or re-run when policy docs change):
    PYTHONPATH=. venv/bin/python3 scripts/index_policy_docs.py

The Retriever Node queries BOTH collections and merges results.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

POLICY_DOCS_DIR = PROJECT_ROOT / "knowledge_base" / "policy_docs"
CHROMA_DB_PATH  = os.getenv("DB_PATH", str(PROJECT_ROOT / "chroma_db_terraform"))
COLLECTION_NAME = "policy_docs"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def main():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    print(f"Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Opening ChromaDB at: {CHROMA_DB_PATH}")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Clear existing policy_docs collection to allow re-indexing
    try:
        existing = vector_store._collection.get(include=[])
        if existing["ids"]:
            vector_store._collection.delete(ids=existing["ids"])
            print(f"Cleared {len(existing['ids'])} existing policy chunks.")
    except Exception as e:
        print(f"Warning: could not clear existing chunks: {e}")

    md_files = sorted(POLICY_DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {POLICY_DOCS_DIR}")
        return

    total_chunks = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        ids = [f"policy::{md_file.stem}::chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": str(md_file.relative_to(PROJECT_ROOT)),
                "filename": md_file.name,
                "collection": COLLECTION_NAME,
            }
            for _ in chunks
        ]
        vector_store.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"  Indexed {md_file.name}: {len(chunks)} chunks")

    print(f"\nDone. {total_chunks} policy chunks indexed into '{COLLECTION_NAME}' collection.")
    print(f"Collection total: {vector_store._collection.count()} chunks")


if __name__ == "__main__":
    main()
