
import os
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

# Configuration
DB_PATH = os.getenv("DB_PATH", os.path.join(os.getcwd(), "chroma_db_terraform"))
DATASET_NAME = "autoiac-project/iac-eval"

def load_and_embed_examples():
    print(f"📂 Loading dataset {DATASET_NAME} from Hugging Face...")
    
    try:
        dataset = load_dataset(DATASET_NAME)
        # Use 'test' split if available (IAC-Eval has 'test')
        data = dataset['test']
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"found {len(data)} examples.")

    documents = []
    
    for item in data:
        # Extract fields based on introspection
        # Keys: ['Resource', 'Prompt', 'Rego intent', 'Difficulty', 'Reference output', 'Intent']
        instruction = item.get('Prompt')
        output = item.get('Reference output')
        
        if not instruction or not output:
            continue
        
        # Format the content specifically for Few-Shot RAG
        page_content = f"User Requirement: {instruction}\n\nGolden Terraform Code:\n{output}"
        
        metadata = {
            "source": "iac_eval_dataset",
            "type": "few_shot_example",
            "resource_types": item.get('Resource', ''),
            "difficulty": item.get('Difficulty', 0)
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))

    if not documents:
        print("No valid documents found.")
        return

    # Initialize Vector Store
    print(f"💾 Connecting to Vector Store at {DB_PATH}...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # Add to Chroma
    print(f"🚀 Embedding and adding {len(documents)} documents...")
    
    # Add in batches to avoid memory spikes
    batch_size = 50
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)

    print("✅ Success! Few-shot examples added to the knowledge base.")

if __name__ == "__main__":
    load_and_embed_examples()
