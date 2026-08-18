"""
Custom Doc Injector
===================
Injects proprietary/internal documents into the ChromaDB knowledge base.
Uses identical chunking settings to data/etl_pipeline.py for consistency.

Supported formats:
  - .md  / .txt  — loaded directly
  - .tf          — loaded directly (Terraform HCL)
  - .pdf         — extracted via pypdf (optional dependency)

Injected docs are tagged with:
  metadata["internal"] = "true"
  metadata["source"]   = original filename
  metadata["injected_at"] = ISO timestamp
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = str(Path(__file__).parent.parent / "chroma_db_terraform")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Match ETL pipeline chunking settings exactly
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


def _load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from file bytes based on extension."""
    ext = Path(filename).suffix.lower()

    if ext in (".md", ".txt", ".tf", ".hcl"):
        return file_bytes.decode("utf-8", errors="replace")

    if ext == ".pdf":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ValueError(
                "PDF support requires 'pypdf'. Install with: pip install pypdf"
            )

    raise ValueError(
        f"Unsupported file type: '{ext}'. Supported: .md, .txt, .tf, .hcl, .pdf"
    )


def inject_document(
    file_bytes: bytes,
    filename: str,
    description: str = "",
) -> dict:
    """
    Chunk and inject a document into ChromaDB.

    Args:
        file_bytes:   Raw file content as bytes.
        filename:     Original filename (used for type detection + metadata).
        description:  Optional human description stored in metadata.

    Returns:
        {"chunks_added": int, "filename": str, "status": "ok"}
    """
    text = _extract_text(file_bytes, filename)

    if not text.strip():
        raise ValueError(f"File '{filename}' appears to be empty or unreadable.")

    now = datetime.now(timezone.utc).isoformat()
    doc = Document(
        page_content=text,
        metadata={
            "source": filename,
            "internal": "true",
            "description": description,
            "injected_at": now,
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents([doc])

    # Propagate all metadata to every chunk
    for chunk in chunks:
        chunk.metadata.update(doc.metadata)

    vector_store = _load_vector_store()
    vector_store.add_documents(chunks)

    return {
        "chunks_added": len(chunks),
        "filename": filename,
        "status": "ok",
    }


def list_internal_docs(limit: int = 20) -> list[dict]:
    """
    List documents that were injected with internal=true tag.
    Returns a deduplicated list by source filename.
    """
    vector_store = _load_vector_store()
    collection = vector_store._collection

    results = collection.get(
        where={"internal": "true"},
        include=["metadatas"],
        limit=limit * 10,  # over-fetch since we deduplicate
    )

    seen = set()
    docs = []
    for meta in results.get("metadatas", []):
        src = meta.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            docs.append({
                "filename": src,
                "injected_at": meta.get("injected_at", ""),
                "description": meta.get("description", ""),
            })
        if len(docs) >= limit:
            break

    return sorted(docs, key=lambda x: x["injected_at"], reverse=True)
