import os
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_google_vertexai import ChatVertexAI
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# 1. Setup Environment
# load_dotenv()
os.environ["API_KEY"] = "AIzaSyCOoiiC4Tnuhslqj8-WiJ91TEiw88MivDc"
api_key = os.getenv("API_KEY")

# if "GOOGLE_API_KEY" not in os.environ:
#     print("Error: GOOGLE_API_KEY not found.")
#     exit()

# # 2. Connect to your EXISTING Vector Store
# # We use the exact same path and embedding model as before.
# DB_PATH = "./chroma_db_terraform"
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# if not os.path.exists(DB_PATH):
#     print(f"Error: Database not found at {DB_PATH}. Run vector_store.py first.")
#     exit()

# vector_store = Chroma(
#     persist_directory=DB_PATH,
#     embedding_function=embedding_model
# )

# # 3. Create the "Retriever"
# # "search_kwargs={'k': 5}" tells it to always fetch the top 5 most relevant pages
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 10} # Increased from 5 to 10
# )

# # 4. The "Brain" (LLM)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-pro", 
#     temperature=0
# )

# # 5. The "Expert" System Prompt
# # This is where the magic happens. We instruct the LLM to be a Terraform pro.
# system_prompt = (
#     "You are a Senior DevOps Engineer and Terraform Expert. "
#     "Your task is to write production-grade Terraform code based ONLY on the provided context. "
#     "\n\n"
#     "Rules:\n"
#     "1. Use only the resource arguments and attributes defined in the Context.\n"
#     "2. Do not invent new arguments that don't exist in the documentation.\n"
#     "3. If the Context doesn't contain enough info to answer, say 'I don't have enough context from the documentation.'\n"
#     "4. Always output valid HCL code.\n"
#     "\n\n"
#     "Context: {context}"
# )

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# # 6. Build the Chain
# # "Stuff" documents chain: Takes retrieved docs and "stuffs" them into the {context} variable
# question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # 7. Run it!
# print("🤖 Terraform RAG Agent is ready! Type 'exit' to quit.")

# while True:
#     query = input("\nRequest: ")
#     if query.lower() == "exit":
#         break
    
#     print("🔍 Searching docs and generating code...")
    
#     # Invoke the chain
#     response = rag_chain.invoke({"input": query})
    
#     print("\n" + "="*40)
#     print("📝 GENERATED CODE:")
#     print("="*40 + "\n")
#     print(response["answer"])
    
#     # --- NEW: DEBUG SECTION ---
#     print("\n" + "-"*40)
#     print("🕵️ DEBUG: What did the AI read?")
#     print("-" * 40)
#     # The "context" key holds the retrieved docs
#     for i, doc in enumerate(response["context"]):
#         print(f"\n[Doc {i+1}]: {doc.metadata.get('source', 'Unknown Source')}")
#         # Print the first 100 characters of the content to check if it's relevant
#         print(f"Content Snippet: {doc.page_content[:150]}...")


import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import ChatVertexAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Set logging to see the "Thought Process" of the AI generating queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()

# 1. Setup Database
DB_PATH = "./chroma_db_terraform"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(DB_PATH):
    print(f"Error: Database not found at {DB_PATH}. Run vector_store.py first.")
    exit()

vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

# 2. Setup LLM (The Brain)
# We use a slightly higher temperature (0.2) to allow for some "Architectural Creativity"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

# 3. INTELLIGENT RETRIEVER (The Upgrade)
# Instead of just looking up your words, this uses the LLM to generate 
# 3 different search queries to find related concepts (e.g., "S3", "IAM", "Bucket Policy")
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
    llm=llm
)

# 4. THE "ARCHITECT" PROMPT
# We removed the "ONLY context" restriction and added a "Reasoning" step.
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

# 5. Build Chain
retrieval_chain = create_retrieval_chain(retriever_from_llm, create_stuff_documents_chain(llm, prompt))

# 6. Execution Loop
print("🧠 Intelligent Architect Agent is ready!")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("Request: ")
    if user_query.lower() == "exit":
        break

    print("\n🤔 Architecting solution (Generating search queries & planning)...")
    
    try:
        response = retrieval_chain.invoke({"input": user_query})
        
        print("\n" + "="*50)
        print(" ARCHITECTED SOLUTION")
        print("="*50)
        print(response["answer"])
        
        # Optional: Show what it found
        # print("\n[References Used]:")
        # for doc in response["context"]:
        #     print(f"- {doc.metadata.get('source')} (Content: {doc.page_content[:50]}...)")
            
    except Exception as e:
        print(f"Error: {e}")