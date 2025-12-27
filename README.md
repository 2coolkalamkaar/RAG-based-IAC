> **An autonomous RAG-based agent that designs, validates, and secures Infrastructure-as-Code (IaC) using Google Gemini, Checkov, and Terraform.**

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Terraform](https://img.shields.io/badge/Terraform-1.9+-purple)
![Security](https://img.shields.io/badge/Security-Checkov_Hardened-red)

## 📖 Overview
This project is not just a code generator; it is a **Self-Healing Infrastructure Agent**. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to understand complex user requests (e.g., "Create a PCI-compliant payment server") and generates production-grade Terraform code.

Unlike standard LLM scripts, this agent operates in a **Closed-Loop System**:
1.  **Architects** the solution using official Terraform documentation and architectural patterns.
2.  **Scans** the generated code for security vulnerabilities using **Checkov**.
3.  **Validates** the syntax using `terraform plan`.
4.  **Self-Corrects**: If any security or syntax check fails, the agent autonomously rewrites the code to fix the specific error without human intervention.

## 🏗️ Architecture

The system follows a multi-stage reasoning pipeline:

1.  **User Intent Analysis:** `MultiQueryRetriever` breaks down high-level requests into specific technical queries.
2.  **Semantic Retrieval:** Fetches context from a hybrid Vector Store containing:
    * **Syntax:** Official Terraform Provider Documentation (for correctness).
    * **Patterns:** IaC Eval Dataset examples (for architectural context).
3.  **Generative Loop (The Brain):** Google Gemini 1.5 Pro generates the initial HCL code.
4.  **Validation Engine (The Guardrails):**
    * **Security:** Runs `checkov` to ensure compliance (e.g., No open SSH ports, Encryption enabled).
    * **Validity:** Runs `terraform plan` to ensure the code is deployable.
5.  **Feedback Loop:** Errors are fed back to the LLM as new prompts ("Fix this specific Checkov error...") until the code is "Golden."

## 🚀 Key Features

* **Context-Aware RAG:** Understands high-level concepts like "Payment Gateway" or "Data Lake" by retrieving similar architectural patterns from a vector database.
* **Zero-Hallucination Syntax:** Grounded in official provider documentation to prevent inventing non-existent resource arguments.
* **Automated Security Hardening:** Enforces best practices (CIS Benchmarks) automatically. If the AI generates an unencrypted volume, the agent catches it and forces encryption.
* **Infrastructure-as-Code (IaC) Output:** Generates clean, modular files (`main.tf`, `variables.tf`, `outputs.tf`).

## 🛠️ Tech Stack

* **LLM:** Google Gemini 1.5 Pro / 2.5 (via LangChain)
* **Vector Database:** ChromaDB (Local persistence)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Orchestration:** LangChain (Python)
* **Validation Tools:** Checkov (Bridgecrew), Terraform CLI
* **Cloud Provider:** Google Cloud Platform (Compute Engine)

## ⚡ Getting Started

### Prerequisites
* Python 3.10+
* Terraform installed (`sudo apt-get install terraform`)
* Google Cloud API Key

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/terraform-ai-agent.git](https://github.com/YOUR_USERNAME/terraform-ai-agent.git)
    cd terraform-ai-agent
    ```

2.  **Install Dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file:
    ```bash
    GOOGLE_API_KEY="your_api_key_here"
    ```

4.  **Build the Knowledge Base (First Run Only)**
    Ingest the Terraform documentation and architectural patterns:
    ```bash
    docuload.py
    ```

### Usage

Run the agent in CLI mode:
```bash
python fullfunctionall.py