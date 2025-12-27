import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader

DOCS_PATH = "/home/rahulvaishnav068/IDP/terraform-provider-aws/website/docs/r" 

# --- THE FIX ---
# Change the glob pattern to match the actual files.
loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.html.markdown",  # <-- THIS WAS THE PROBLEM
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True,
    use_multithreading=True
)

print("Loading documentation from GitHub repo...")

documents = loader.load()

print(f"Successfully loaded {len(documents)} documentation pages.")

# We check if documents were loaded before trying to access one
if documents:
    print("\n--- Example Document ---")
    print(f"Source: {documents[789].metadata['source']}")
    print(f"Content:\n{documents[789].page_content[:800]}...")
else:
    print("\nNo documents were loaded. Please check your DOCS_PATH.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking for path: {os.path.abspath(DOCS_PATH)}")
    
    
# for doc in documents:    -----> for counting words per doc 
#     word_count = len(doc.page_content.split())
#     print(f"{doc.metadata['source']}: {word_count} words")




