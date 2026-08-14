"""
ETL Pipeline: Incremental Vector Store Updater
================================================
Keeps the ChromaDB vector store in sync with the latest terraform-provider-aws docs.

Strategy: Incremental / Idempotent
  - Pulls the latest docs via `git pull`
  - Compares file-level hashes against what's already indexed
  - Only adds NEW or CHANGED documents — never duplicates
  - Tracks provider version + last-run metadata in a JSON manifest

Usage:
    python etl_pipeline.py                  # Incremental update (default)
    python etl_pipeline.py --full-rebuild   # Wipe and rebuild from scratch
    python etl_pipeline.py --dry-run        # Show what would change, don't write
"""

import os
import json
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DOCS_PATH      = Path("/home/rahul/RAG-based-IAC/terraform-provider-aws/website/docs/r")
PROVIDER_REPO  = Path("/home/rahul/RAG-based-IAC/terraform-provider-aws")
DB_PATH        = Path("./chroma_db_terraform")
MANIFEST_PATH  = DB_PATH / "etl_manifest.json"
DOCS_GLOB      = "**/*.html.markdown"

CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA256 hash of a file's content — used to detect changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    """Load the ETL manifest that tracks what's already indexed."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"provider_version": None, "last_run": None, "indexed_files": {}}


def save_manifest(manifest: dict):
    """Persist the ETL manifest after a successful run."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Manifest saved to {MANIFEST_PATH}")


def get_provider_version() -> str:
    """Get the current git commit SHA of the provider repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROVIDER_REPO,
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def pull_latest_docs() -> bool:
    """
    Pull latest changes from the terraform-provider-aws repo.
    Returns True if anything changed.
    """
    print(f"\n🔄 Pulling latest docs from provider repo...")
    before = get_provider_version()
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=PROVIDER_REPO,
            capture_output=True, text=True
        )
        after = get_provider_version()
        if before != after:
            print(f"   ✅ Updated: {before} → {after}")
            return True
        else:
            print(f"   ✓  Already up to date (commit: {before})")
            return False
    except Exception as e:
        print(f"   ⚠️  git pull failed: {e}. Using existing local docs.")
        return False


def load_all_doc_paths() -> list[Path]:
    """Return all matching doc file paths from the docs directory."""
    return sorted(DOCS_PATH.glob(DOCS_GLOB))


def compute_file_diff(all_files: list[Path], manifest: dict) -> tuple[list[Path], list[str]]:
    """
    Compare current files against the manifest.
    Returns:
        - files_to_index: new or changed files
        - sources_to_delete: Chroma source IDs for files that changed (need re-indexing)
    """
    indexed = manifest.get("indexed_files", {})
    files_to_index = []
    sources_to_delete = []

    for file_path in all_files:
        file_key = str(file_path)
        current_hash = file_hash(file_path)

        if file_key not in indexed:
            files_to_index.append(file_path)  # NEW file
        elif indexed[file_key] != current_hash:
            files_to_index.append(file_path)  # CHANGED file
            sources_to_delete.append(file_key)  # Remove old chunks first

    print(f"\n📊 Diff Results:")
    print(f"   Total docs in repo : {len(all_files)}")
    print(f"   Already indexed    : {len(indexed)}")
    print(f"   New files          : {len([f for f in files_to_index if str(f) not in indexed])}")
    print(f"   Changed files      : {len(sources_to_delete)}")
    print(f"   Unchanged (skip)   : {len(all_files) - len(files_to_index)}")

    return files_to_index, sources_to_delete


def delete_stale_chunks(vector_store: Chroma, sources_to_delete: list[str]):
    """
    Remove all chunks belonging to changed/deleted source files from Chroma.
    This prevents duplicates when re-indexing updated files.
    """
    if not sources_to_delete:
        return

    print(f"\n🗑️  Deleting stale chunks for {len(sources_to_delete)} changed files...")
    collection = vector_store._collection

    for source in sources_to_delete:
        result = collection.get(where={"source": source})
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"   Deleted {len(ids_to_delete)} chunks from: {Path(source).name}")


def index_new_files(
    vector_store: Chroma,
    files_to_index: list[Path],
    manifest: dict,
    dry_run: bool = False
):
    """Load, split, embed, and store new/changed documents."""
    if not files_to_index:
        print("\n✅ Nothing to index — vector store is already up to date!")
        return

    print(f"\n📥 Indexing {len(files_to_index)} documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Terraform-specific separators: prefer splitting on resource blocks
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
    )

    total_chunks = 0
    BATCH_SIZE = 50  # Process in batches to avoid memory spikes

    for i in range(0, len(files_to_index), BATCH_SIZE):
        batch = files_to_index[i : i + BATCH_SIZE]
        batch_docs = []

        for file_path in batch:
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

                # Normalize the source metadata to the full path string (for dedup)
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["resource_name"] = file_path.stem.replace(".html", "")
                    doc.metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()

                chunks = splitter.split_documents(docs)
                batch_docs.extend(chunks)
            except Exception as e:
                print(f"   ⚠️  Skipping {file_path.name}: {e}")
                continue

        if batch_docs and not dry_run:
            vector_store.add_documents(batch_docs)

        total_chunks += len(batch_docs)
        print(f"   Batch {i // BATCH_SIZE + 1}: processed {len(batch)} files → {len(batch_docs)} chunks")

        # Update manifest hashes for this batch
        if not dry_run:
            for file_path in batch:
                try:
                    manifest["indexed_files"][str(file_path)] = file_hash(file_path)
                except Exception:
                    pass

    action = "Would index" if dry_run else "Indexed"
    print(f"\n   {action} {total_chunks} total chunks from {len(files_to_index)} files.")


# ─────────────────────────────────────────────
# MAIN ETL ENTRYPOINT
# ─────────────────────────────────────────────

def run_etl(full_rebuild: bool = False, dry_run: bool = False, skip_pull: bool = False):
    print("=" * 60)
    print("🚀 Terraform Docs ETL Pipeline")
    print("=" * 60)

    # Step 1: Pull latest docs
    if not skip_pull:
        pull_latest_docs()
    
    provider_version = get_provider_version()
    print(f"\n📦 Provider Version: {provider_version}")

    # Step 2: Initialize embeddings
    print(f"\n🔌 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Step 3: Full rebuild if requested
    if full_rebuild:
        print("\n⚠️  FULL REBUILD: Wiping existing vector store...")
        if not dry_run and DB_PATH.exists():
            import shutil
            shutil.rmtree(DB_PATH)
        print("   Done. Starting fresh.")

    # Step 4: Open (or create) Chroma vector store
    DB_PATH.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)
    print(f"🗄️  Vector store opened: {DB_PATH}")

    # Step 5: Load manifest and compute diff
    manifest = load_manifest() if not full_rebuild else {"provider_version": None, "last_run": None, "indexed_files": {}}
    all_files = load_all_doc_paths()
    files_to_index, sources_to_delete = compute_file_diff(all_files, manifest)

    if dry_run:
        print("\n🔎 [DRY RUN] — No changes will be written.")

    # Step 6: Delete stale chunks for changed files
    if not dry_run:
        delete_stale_chunks(vector_store, sources_to_delete)

    # Step 7: Index new/changed documents
    index_new_files(vector_store, files_to_index, manifest, dry_run=dry_run)

    # Step 8: Update and save manifest
    if not dry_run:
        manifest["provider_version"] = provider_version
        manifest["last_run"] = datetime.now(timezone.utc).isoformat()
        manifest["total_indexed"] = len(manifest["indexed_files"])
        save_manifest(manifest)

    # Step 9: Quick sanity check
    print("\n🔎 Sanity check — similarity search:")
    query = "terraform code for compute optimized EC2 instance"
    results = vector_store.similarity_search(query, k=3)
    print(f"   Query: '{query}'")
    print(f"   Top result: {Path(results[0].metadata.get('source', 'unknown')).name if results else 'No results'}")

    print("\n" + "=" * 60)
    print("✅ ETL Pipeline Complete!")
    print(f"   Provider commit : {provider_version}")
    print(f"   Total indexed   : {manifest.get('total_indexed', 'N/A')} files")
    print(f"   DB path         : {DB_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL: Sync ChromaDB with latest Terraform AWS provider docs")
    parser.add_argument("--full-rebuild", action="store_true", help="Wipe DB and re-index everything from scratch")
    parser.add_argument("--dry-run",      action="store_true", help="Show what would change without writing anything")
    parser.add_argument("--skip-pull",    action="store_true", help="Skip the git pull (use existing local docs)")
    args = parser.parse_args()

    run_etl(
        full_rebuild=args.full_rebuild,
        dry_run=args.dry_run,
        skip_pull=args.skip_pull
    )
