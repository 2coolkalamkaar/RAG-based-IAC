# 🧠 RAG-Based Infrastructure-as-Code Agent

> **Self-healing agentic RAG pipeline for Terraform generation — grounding, validation, and human-in-the-loop trust gates.**
>
> This system converts plain-English infrastructure requests into production-ready, security-hardened Terraform code using a four-tier LangGraph agentic architecture powered by Google Gemini 2.5 Pro. It grounds every generation in a 15,691-chunk ChromaDB knowledge base of official AWS provider docs, then autonomously validates, self-heals, and security-scans the output via Checkov and TFLint — all before a human ever sees it. A `Trust_Assessor_Node` computes a real-time confidence score across retrieval quality, reranker precision, and validation pass rate, feeding directly into a Human-in-the-Loop gate for final approval.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Pro-4285F4?logo=google)](https://deepmind.google/technologies/gemini/)
[![Terraform](https://img.shields.io/badge/Terraform-1.10+-7B42BC?logo=terraform)](https://terraform.io)
[![Checkov](https://img.shields.io/badge/Security-Checkov_%7C_TFLint-red)](https://checkov.io)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📖 Overview

This project is a **production-ready Hybrid RAG + Agentic AI system** that automates infrastructure code generation for DevOps and SRE engineers. A user describes what they need in plain English — the agent researches, architects, validates, secures, and self-corrects a complete Terraform solution.

Unlike a simple LLM wrapper, this system implements a **four-tier progressive architecture** that was benchmarked end-to-end:

| Tier | Architecture | Validation Rate | Avg Quality |
|---|---|---|---|
| 1 | Basic (LLM only) | 67% | 3.3 / 5 |
| 2 | Standard RAG | 67% | 2.7 / 5 |
| 3 | **Advanced RAG** | **100%** | **3.7 / 5** |
| 4 | **Secure RAG** | **100%** | **⭐ 5.0 / 5** |

> 📊 *Results from a 12-scenario automated benchmark across Basic VPC, Multi-tier web app, and Serverless event pipeline prompts.*

---

## 🏗️ Architecture

The system is built as a **progressive stack of four LangGraph workflows**, each layer adding enterprise capabilities on top of the previous.

```
User Request (Natural Language)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  RETRIEVER NODE                        │
│  MultiQuery Expansion → ChromaDB Search               │
│  → CrossEncoder Reranking (15,691 chunks indexed)     │
│  Knowledge base: 1,696 AWS Terraform Provider docs    │
└────────────────────────┬──────────────────────────────┘
                         │  Grounded Context + Citations
                         ▼
┌───────────────────────────────────────────────────────┐
│                  ARCHITECT NODE                        │
│  Google Gemini 2.5 Pro                                │
│  Generates: main.tf / variables.tf / outputs.tf /     │
│             security_groups.tf / provider.tf          │
│  Enforces: No open SSH, EBS encryption, no public IPs │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────┐
│                  VALIDATOR NODE                        │
│  terraform validate + TFLint + Checkov (Secure tier)  │
│  CIS AWS Benchmark checks                             │
└──────────┬──────────────────────┬─────────────────────┘
           │ PASS                 │ FAIL
           │                     ▼
           │          ┌──────────────────────┐
           │          │     FIXER NODE        │  ← Up to 3 retries
           │          │  Error-aware rewrite  │
           │          └──────────┬───────────┘
           │                     │ Re-validates
           ▼                     ▼
┌───────────────────────────────────────────────────────┐
│              TRUST ASSESSOR NODE                       │
│  Computes 0-100% Trust Score & qualitative tier based │
│  on retrieval similarity, reranker, & Checkov scans   │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────┐
│              HUMAN-IN-THE-LOOP NODE  (HitL tier)      │
│  LangGraph interrupt() → UI Approval Panel            │
│  "Approve & Finalize" OR "Apply Patch"                │
└────────────────────────┬──────────────────────────────┘
                         │ patch_request
                         ▼
┌───────────────────────────────────────────────────────┐
│                  PATCHER NODE                          │
│  Surgical diff-merge — changes ONLY what was asked    │
│  No full pipeline restart → saves time + API cost     │
└───────────────────────────────────────────────────────┘
                         │
                         ▼
              Production-Ready Terraform
```

---

## 📊 Benchmark Results

A 12-scenario automated benchmark was run across all four workflow tiers using three real-world infrastructure requests:
- **S1:** Multi-AZ VPC with public/private subnets and NAT gateway
- **S2:** 3-tier web application with ALB, Auto Scaling, and RDS
- **S3:** Serverless event pipeline with Lambda, SQS, and S3

| Metric | Basic | Standard RAG | Advanced RAG | Secure RAG |
|---|---|---|---|---|
| **Validation Pass Rate** | 67% | 67% | **100%** | **100%** |
| **Avg Quality Score** | 3.3 / 5 | 2.7 / 5 | 3.7 / 5 | **5.0 / 5** |
| **Avg Retries** | 1.3 | 1.7 | **1.3** | 1.7 |
| **Avg Latency** | 132s | 145s | 180s | 280s |
| **Checkov Security** | ❌ None | ❌ None | ❌ None | ✅ Full scan |
| **Context Grounding** | ❌ | ✅ Basic | ✅ Reranked | ✅ Reranked |

> **Key insight:** The Advanced and Secure RAG tiers achieved **100% validation pass rate** (up from 67% with no RAG), with Secure RAG achieving the maximum quality score on every scenario — demonstrating that retrieval-grounded generation is not just more accurate, it is consistently more accurate.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Hybrid RAG** | MultiQuery retriever expands single queries into 3 sub-queries, custom `ScorePreservingReranker` computes CrossEncoder logits for precision |
| **Self-Healing Loop** | Autonomous `validate → fix → validate` cycle, up to 3 retries, with the full validation error passed back as context |
| **Shift-Left Security** | Checkov CIS AWS benchmark + TFLint on every generated file, before any human review |
| **Trust Score Assessment** | Dedicated `Trust_Assessor_Node` computes a 0-100% confidence score and qualitative tier label based on Cosine Similarity, reranker confidence, and security pass rates, generating a natural language explanation |
| **Human-in-the-Loop** | LangGraph `interrupt()` pauses execution for human review and approval, with full state persistence via SQLite |
| **Surgical Patcher** | Dedicated `Patcher_Node` applies natural-language change requests as targeted diffs — no full pipeline restart |
| **SRE Upload Mode** | SREs can upload existing `.tf` files directly, bypassing generation, to leverage the validation and patching nodes for legacy infrastructure |
| **Persistent Memory** | `SqliteSaver` checkpointer preserves full conversation state — threads are resumable across browser sessions |
| **ETL Pipeline** | Incremental, SHA-256 hash-based sync with the official `terraform-provider-aws` documentation (1,696 docs → 15,691 chunks) |
| **Multi-Workflow UI** | Single Streamlit interface supports all four workflow tiers with a live pipeline stage visualiser and log terminal |
| **Benchmark Suite** | 12-scenario automated evaluation harness with a Plotly dashboard for results visualisation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.5 Pro (via Vertex AI / `langchain-google-vertexai`) |
| **Agentic Framework** | LangGraph `StateGraph` with conditional edges and `interrupt()` |
| **Vector Database** | ChromaDB (local persistence) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| **IaC Validation** | Terraform CLI + TFLint |
| **Security Scanning** | Checkov (CIS AWS benchmarks) |
| **Memory / Persistence** | SQLite via `SqliteSaver` |
| **Observability** | LangSmith tracing |
| **UI** | Streamlit |
| **Benchmarking** | Custom harness + Plotly Express dashboard |

---

## 📁 Project Structure

```
RAG-based-IAC/
│
├── workflows/                          # 🤖 LangGraph Agent Pipelines
│   ├── agent_workflow.py               #    Tier 1: Basic (LLM only)
│   ├── agent_workflow_rag.py           #    Tier 2: Standard RAG
│   ├── agent_workflow_advanced_rag.py  #    Tier 3: Advanced RAG (MultiQuery + Rerank)
│   ├── agent_workflow_secure_rag.py    #    Tier 4: Secure RAG (+ Checkov)
│   └── agent_workflow_hitl.py          #    Tier 5: HitL + Surgical Patcher
│
├── ui/
│   └── multi_workflow_ui.py            # 🖥️  Streamlit UI (all tiers, live pipeline view)
│
├── data/
│   ├── etl_pipeline.py                 # 🔄 Incremental ChromaDB ETL pipeline
│   ├── vector_store.py                 #    ChromaDB initialisation helper
│   └── mock_sre_upload/                # 🧪 Sample .tf files for SRE Upload Mode testing
│       ├── main.tf
│       ├── ec2.tf
│       └── security_groups.tf          #    Contains intentional vulns for demo
│
├── benchmarking/
│   ├── run_benchmark.py                # 📊 Automated 12-scenario evaluation harness
│   ├── benchmark_dashboard.py          #    Plotly Streamlit dashboard
│   ├── benchmark_results.json          #    Raw benchmark data
│   └── benchmark_dataset.jsonl         #    Evaluation prompts + rubrics
│
├── evaluation/
│   └── eval_agent.py                   # 🧪 LangSmith eval agent
│
├── docs/
│   ├── README.md                       # ← You are here
│   └── workflow_diagram.png
│
├── scripts/
│   ├── install_terraform.sh
│   └── install_tflint.sh
│
├── archive/                            # 📦 Deprecated POC files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .tflint.hcl
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Google Cloud account with Vertex AI API enabled (or a Google AI Studio API key)
- Terraform CLI → `bash scripts/install_terraform.sh`
- TFLint → `bash scripts/install_tflint.sh`
- Checkov → installed via `requirements.txt`

### 1. Clone & Configure

```bash
git clone https://github.com/2coolkalamkaar/RAG-based-IAC.git
cd RAG-based-IAC
```

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY="your_google_ai_studio_key"     # OR use Vertex AI credentials below
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"     # For Vertex AI
GOOGLE_CLOUD_LOCATION="us-central1"

LANGCHAIN_TRACING_V2="true"                    # Optional: LangSmith observability
LANGCHAIN_API_KEY="your_langsmith_key"
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the Knowledge Base

The RAG system requires the ChromaDB vector store to be populated from the official AWS Terraform provider documentation.

```bash
# Clone the provider docs (~200MB)
git clone https://github.com/hashicorp/terraform-provider-aws.git

# Run the ETL pipeline (first run: ~10-15 min on CPU)
PYTHONPATH=. venv/bin/python data/etl_pipeline.py

# Subsequent runs are incremental (only re-indexes changed files)
PYTHONPATH=. venv/bin/python data/etl_pipeline.py
```

> **What gets indexed:** 1,696 AWS resource docs → 15,691 semantic chunks at `all-MiniLM-L6-v2` embeddings (using Cosine Similarity).

### 4. Run the Multi-Workflow UI

```bash
PYTHONPATH=. venv/bin/streamlit run ui/multi_workflow_ui.py
```

Open **http://localhost:8501** and select a workflow tier from the sidebar.

### 5. Run the Benchmark Suite

```bash
# Run all 12 scenarios (takes ~30-60 min due to API rate limiting)
PYTHONPATH=. venv/bin/python benchmarking/run_benchmark.py

# View results dashboard
PYTHONPATH=. venv/bin/streamlit run benchmarking/benchmark_dashboard.py
```

---

## 🔬 Workflow Tiers — When to Use What

| Scenario | Recommended Tier |
|---|---|
| Quick prototype / testing | Basic |
| Standard infrastructure requests | Advanced RAG |
| Production infrastructure (regulated environments) | Secure RAG |
| Modifying existing infrastructure | HitL RAG + SRE Upload Mode |
| Compliance-critical changes requiring sign-off | HitL RAG (Approval Gate) |

---

## 🛡️ Security Design

Security is not bolted on — it is embedded at every layer of the pipeline:

1. **Generation-time guardrails** — The Architect Node's system prompt enforces hard rules: no `0.0.0.0/0` ingress on SSH, mandatory EBS encryption, no public IP assignment without explicit justification.
2. **Shift-left validation** — Checkov runs CIS AWS benchmark checks on every generated file before any human sees the output.
3. **Human approval gate** — The HitL Node uses LangGraph's `interrupt()` mechanism, a proper state-machine pause, to require explicit human sign-off before finalisation. This maps directly to enterprise change management controls.
4. **Surgical patching** — The Patcher Node applies changes at the resource level, not by regenerating the whole file, reducing the blast radius of AI-generated modifications.
5. **SRE mode isolation** — Uploaded files are path-sanitised (`pathlib.Path.name`) to prevent directory traversal.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. See the planned security CI/CD roadmap (Garak, Promptfoo, Semgrep, Trivy) for upcoming additions.