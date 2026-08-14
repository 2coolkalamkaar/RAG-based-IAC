
# 🧠 Terraform Architect Agent — Self-Validating & Self-Healing IaC

> **An autonomous RAG-based agent that designs, validates, secures, and cost-estimates Infrastructure-as-Code (IaC) using Google Gemini, LangGraph, ChromaDB, TFLint, and Terraform.**

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Terraform](https://img.shields.io/badge/Terraform-1.10+-purple)
![Security](https://img.shields.io/badge/Security-TFLint_Hardened-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange)

---

## 📖 Overview

This project is a **Self-Healing Infrastructure Agent** — an intelligent pair programmer for DevOps and SRE engineers. It takes a plain-English request (e.g., _"Create a secure 3-tier VPC with NAT gateway"_) and outputs production-grade, validated Terraform code.

Unlike a simple LLM wrapper, this agent operates in a **closed-loop agentic workflow**:

1. **Retrieves** grounded context from the official AWS Terraform Provider docs using MultiQuery + CrossEncoder Reranking.
2. **Architects** a complete Terraform solution (multi-file: `main.tf`, `variables.tf`, etc.) with hard security rules enforced.
3. **Validates** syntax via `terraform validate` and security via `tflint`.
4. **Self-Corrects** autonomously — if validation fails, a dedicated Fixer Node rewrites the code with the error context (up to 3 retries).
5. **Estimates Cost** using Infracost CLI on the validated code.

---

## 🏗️ Agentic Workflow Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│  Retriever Node │  ← MultiQuery expansion + CrossEncoder Reranker
│  (ChromaDB RAG) │    pulls top-5 grounded docs from provider docs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Architect Node │  ← Gemini 2.5 Pro generates multi-file Terraform
│  (LLM + Prompt) │    with enforced security rules
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Validator Node  │  ← terraform validate + tflint
└──────┬───────────┘
       │
  ┌────┴──────┐
  │           │
PASS        FAIL (retry ≤ 3)
  │           │
  │    ┌──────▼──────┐
  │    │  Fixer Node │  ← Rewrites code using validation errors as context
  │    └──────┬──────┘
  │           │
  │    ┌──────▼──────┐
  │    │  Validator  │  ← Re-validates fixed code
  │    └─────────────┘
  │
  ▼
┌──────────────────────┐
│  Cost Estimator Node │  ← Infracost CLI: monthly cost breakdown
└──────────────────────┘
         │
         ▼
    Final Output
```

**Graph is compiled with `SqliteSaver` for persistent conversation memory** — the agent remembers previous interactions within a session thread.

---

## 🗄️ ETL Pipeline — Keeping the Knowledge Base Fresh

A dedicated **ETL pipeline** (`etl_pipeline.py`) keeps the ChromaDB vector store in sync with the latest `terraform-provider-aws` documentation:

| Feature | Detail |
|---|---|
| **Incremental updates** | SHA-256 file hashing — only new or changed docs are re-indexed |
| **Deduplication** | Stale chunks for changed files are deleted before re-indexing |
| **Version tracking** | Git commit SHA of the provider repo is stored in a manifest |
| **Auto git pull** | Pulls latest provider docs before indexing |
| **Batch processing** | 50 files/batch for memory safety |
| **Terraform-aware splitting** | Splits on `##`/`###` headings, preserving resource block context |

```bash
# Incremental update (pulls latest + indexes only changed files)
venv/bin/python etl_pipeline.py

# Preview what would change without writing
venv/bin/python etl_pipeline.py --dry-run

# Full wipe and rebuild from scratch
venv/bin/python etl_pipeline.py --full-rebuild
```

> **Current knowledge base**: 1,584 AWS resource docs → 13,779 chunks @ `all-MiniLM-L6-v2` embeddings.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **Advanced RAG** | MultiQuery retriever + CrossEncoder reranker for precision grounding |
| **Self-Healing Loop** | Autonomous validation → fix → re-validate cycle (max 3 retries) |
| **Hard Security Rules** | Blocks open SSH (`0.0.0.0/0`), enforces EBS encryption, no public IPs |
| **Cost Estimation** | Infracost CLI integration — monthly cost before you deploy |
| **Persistent Memory** | SqliteSaver checkpointer — full multi-turn conversation support |
| **ETL Pipeline** | Incremental, hash-based sync with the latest provider docs |
| **Streamlit UI** | Clean chat interface with expandable code blocks and citation panel |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.5 Pro (via `langchain-google-genai`) |
| **Agentic Framework** | LangGraph (`StateGraph` with conditional edges) |
| **Vector DB** | ChromaDB (local persistence) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (CPU, no API key) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| **Validation** | Terraform CLI + TFLint |
| **Cost Estimation** | Infracost CLI |
| **Memory** | SQLite (`SqliteSaver` / `AsyncSqliteSaver`) |
| **UI** | Streamlit |

---

## ⚡ Getting Started

### Prerequisites

- Python 3.11+
- Terraform CLI → `./install_terraform.sh`
- TFLint → `./install_tflint.sh`
- Infracost CLI (optional, for cost estimation) → [infracost.io/docs](https://www.infracost.io/docs/)

### 1. Clone & Configure

```bash
git clone https://github.com/2coolkalamkaar/RAG-based-IAC.git
cd RAG-based-IAC
```

Create a `.env` file:
```bash
GOOGLE_API_KEY="your_google_api_key_here"
LANGCHAIN_TRACING_V2="true"          # Optional: LangSmith tracing
LANGCHAIN_API_KEY="your_ls_key"      # Optional
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the Knowledge Base

Clone the provider docs and run the ETL pipeline:
```bash
git clone https://github.com/hashicorp/terraform-provider-aws.git
venv/bin/python etl_pipeline.py --skip-pull   # First time: skip pull, just index
```

This takes ~10-15 minutes on CPU. Subsequent runs are incremental (seconds).

### 4. Run the Agent

```bash
# Streamlit UI (recommended)
venv/bin/streamlit run app.py

# CLI test mode (runs a hardcoded request end-to-end)
venv/bin/python agent_workflow_advanced_rag.py
```

Access the UI at **http://localhost:8501**.

### Option B: Docker

```bash
docker-compose up --build
```

---

## 📁 Project Structure

```
RAG-based-IAC/
├── agent_workflow_advanced_rag.py   # 🏆 Main agentic workflow (MultiQuery + Reranker)
├── agent_workflow_advanced_rag_prod.py  # Async production version
├── agent_workflow_rag.py            # Standard RAG workflow (simpler)
├── agent_workflow.py                # Base workflow (no RAG)
├── etl_pipeline.py                  # ✨ ETL: incremental ChromaDB sync
├── app.py                           # Streamlit UI entry point
├── streamlit_app.py                 # Alternative Streamlit UI
├── vector_store.py                  # ChromaDB setup utility
├── requirements.txt
├── install_terraform.sh
├── install_tflint.sh
├── .tflint.hcl                      # TFLint ruleset config
├── Dockerfile
└── docker-compose.yml
```

---

## 🤝 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.