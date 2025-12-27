from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=google_api_key, temperature=0)

result = model.invoke('which is the  best model that can generate a industry standard Terraform code for AWS infrastructure?')

print(result.content)