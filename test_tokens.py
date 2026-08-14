from langchain_google_vertexai import ChatVertexAI
import os
os.environ["LANGCHAIN_PROJECT"] = "test"
llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1")
res = llm.invoke("Hi")
print("usage_metadata:", getattr(res, "usage_metadata", None))
print("response_metadata:", res.response_metadata)
