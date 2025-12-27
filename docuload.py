import os
import langchain.chat_models
import dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# --- Configuration ---
# Point this to the 'docs' folder within the repo you just cloned
DOCS_PATH = "./terraform-provider-aws/website/docs/r" 

# We specify that we only want to load Markdown files.
# We also use TextLoader to ensure they are read as plain text.
loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.mdx",   # This pattern means "all .md files in all subfolders"
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True,
    use_multithreading=True
)

print("Loading documentation from GitHub repo...")

# This runs the loader
documents = loader.load()

print(f"Successfully loaded {len(documents)} documentation pages.")

# --- Optional: See what you've loaded ---
print("\n--- Example Document ---")
print(f"Source: {documents[0].metadata['source']}")
print(f"Content:\n{documents[0].page_content[:500]}...") # Print first 500 chars