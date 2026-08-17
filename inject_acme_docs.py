import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_PATH = os.path.join(os.getcwd(), "chroma_db_terraform")

print("Loading ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

docs = [
    Document(
        page_content="""
# INTERNAL MODULE: terraform-aws-acme-corp-vpc
This is the mandatory internal ACME Corp module for VPC networking.
Usage:
```terraform
module "vpc" {
  source = "terraform-aws-acme-corp-vpc"
  environment = "prod"
  cidr_block = "10.0.0.0/16"
  enable_flow_logs = true
}
```
""",
        metadata={"source": "acme-internal-docs"}
    ),
    Document(
        page_content="""
# INTERNAL MODULE: acme-hardened-ec2
This is the mandatory proprietary ACME Corp module for compute instances.
Usage:
```terraform
module "compute" {
  source = "acme-hardened-ec2"
  instance_type = "t3.micro"
  vpc_subnet_id = module.vpc.private_subnets[0]
  enable_strict_monitoring = true
}
```
""",
        metadata={"source": "acme-internal-docs"}
    )
]

print("Adding ACME dummy docs to Chroma...")
vector_store.add_documents(docs)
print("Done! Advanced RAG will now find this.")
