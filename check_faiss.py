
try:
    from langchain_community.vectorstores import FAISS
    print("FAISS Import successful")
    import faiss
    print(f"faiss version: {faiss.__version__}")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
