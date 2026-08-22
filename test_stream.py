from langchain_google_vertexai import ChatVertexAI
from langchain_core.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

class MyHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print("TOKEN:", token, end="", flush=True)

llm = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.2)
print("Without streaming=True:")
llm.invoke("Say hi", config={"callbacks": [MyHandler()]})
print("\n---")
llm_stream = ChatVertexAI(model_name="gemini-2.5-pro", project="project-036ddc82-f451-4fae-9e3", location="us-central1", temperature=0.2, streaming=True)
print("With streaming=True:")
llm_stream.invoke("Say hi", config={"callbacks": [MyHandler()]})
print("\n---")
