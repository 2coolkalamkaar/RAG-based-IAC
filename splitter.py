import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. LOAD (Code you already have) ---

DOCS_PATH = "/home/rahulvaishnav068/IDP/terraform-provider-aws/website/docs/r" 

loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.html.markdown",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

print("Loading documentation...")
documents = loader.load()
print(f"Successfully loaded {len(documents)} documentation pages.")

# --- 2. SPLIT (The New Part) ---

print("Splitting documents into chunks...")

# Initialize the splitter
text_splitter = RecursiveCharacterTextSplitter(
    # This is the max size of a chunk (in characters)
    chunk_size=1000, 
    
    # This is the overlap between chunks.
    # It helps maintain context between chunks.
    chunk_overlap=200,  
    
    length_function=len # This just tells it to count characters
)

# This one command does all the work
chunks = text_splitter.split_documents(documents)

print(f"Successfully split {len(documents)} documents into {len(chunks)} chunks.")

# --- 3. VERIFY ---
# Let's see what we made

print("\n--- Example Chunk ---")
# Print the content of the first chunk
print(chunks[0].page_content)

print("\n--- Example Chunk Metadata ---")
# Notice the metadata is preserved from the original document
print(chunks[0].metadata)

print("\n--- Overlap Example ---")
# The end of the first chunk...
print(f"End of Chunk 0:\n...{chunks[0].page_content[-150:]}")
# ...should overlap with the start of the second chunk
print(f"\nStart of Chunk 1:\n{chunks[1].page_content[:150]}...")