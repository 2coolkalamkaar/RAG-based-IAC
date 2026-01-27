
# 🧠 Terraform Architect Agent (Self-Validating & Self-Healing)

> **An autonomous RAG-based agent that designs, validates, secures, and visualizes Infrastructure-as-Code (IaC) using Google Gemini, TFLint, and Terraform.**

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Terraform](https://img.shields.io/badge/Terraform-1.10+-purple)
![Security](https://img.shields.io/badge/Security-TFLint_Hardened-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 📖 Overview

This project is a **Self-Healing Infrastructure Agent**. It serves as an intelligent pair programmer for DevOps engineers. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to understand complex user requests (e.g., "Create a secure 3-tier VPC") and generates production-grade Terraform code.

Unlike standard LLM scripts, this agent operates in a **Closed-Loop System**:
1.  **Architects** the solution using official documentation and **450+ Golden Solution Examples** (Few-Shot RAG).
2.  **Scans** the generated code for security vulnerabilities and best practices using **TFLint**.
3.  **Validates** the syntax and logic using `terraform plan`.
4.  **Visualizes** the architecture by auto-generating a diagram from the code.
5.  **Self-Corrects**: If any security or syntax check fails (e.g., open SSH port), the agent autonomously rewrites the code to fix the specific error.

## 🏗️ Architecture

The system follows a multi-stage reasoning pipeline:

1.  **User Intent Analysis:** `MultiQueryRetriever` breaks down high-level requests into specific technical queries.
2.  **Semantic Retrieval:** Hybrid Vector Store containing:
    * **Docs:** Official Terraform Provider Documentation (Syntax grounding).
    * **Examples:** `iac-eval` dataset from Hugging Face (Architectural context).
3.  **Generative Loop (The Brain):** Google Gemini 2.5 Pro generates the initial HCL code.
4.  **Validation Engine (The Guardrails):**
    * **Security:** Runs `tflint` to enforce best practices (e.g., no default VPCs, descriptions required).
    * **Validity:** Runs `terraform plan` (with dummy creds) to ensure logical correctness.
5.  **Visualization:** An LLM-powered generator creates an architecture diagram (`.png`) on the fly using the `diagrams` library.

## 🚀 Key Features

*   **Self-Validating Feedback Loop:** The agent doesn't just guess; it *tests* its own code against the Terraform binary and TFLint before showing it to you.
*   **Security Best Practices:** Enforces critical rules like blocking open SSH access, requiring encryption, and ensuring IAM roles are attached.
*   **Few-Shot RAG:** Uses a dataset of 450+ validated infrastructure problems and solutions to "learn" correct patterns.
*   **Architecture Visualization:** Automatically generates a visual diagram of the infrastructure configuration.
*   **Dockerized:** Runs anywhere with a single command.

## 🛠️ Tech Stack

*   **LLM:** Google Gemini 2.5 Pro (via LangChain)
*   **Vector Database:** ChromaDB (Local persistence)
*   **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
*   **Validation:** Terraform CLI, TFLint
*   **Visualization:** Python `diagrams` library
*   **App Interface:** Streamlit

## ⚡ Getting Started

### Option A: Docker (Recommended)

The easiest way to run the agent is with Docker Compose. This handles all dependencies (Terraform, TFLint, Python) for you.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/2coolkalamkaar/RAG-based-IAC.git
    cd RAG-based-IAC
    ```

2.  **Configure API Key**
    Create a `.env` file in the root directory:
    ```bash
    GOOGLE_API_KEY="your_api_key_here"
    ```

3.  **Run the App**
    ```bash
    docker-compose up --build
    ```
    Access the agent at **http://localhost:8501**.

### Option B: Local Installation

1.  **Install Prerequisites**
    *   Python 3.11+
    *   Terraform (Use provided script: `./install_terraform.sh`)
    *   TFLint (Use provided script: `./install_tflint.sh`)
    *   Graphviz (`sudo apt-get install graphviz`)

2.  **Install Python Dependencies**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Load Knowledge Base**
    Populate the vector store with docs and examples:
    ```bash
    # Ensure you have your .env file ready
    python docuload_temp.py   # Load Documentation
    python load_examples.py   # Load Few-Shot Examples
    ```

4.  **Run the Agent**
    ```bash
    streamlit run RAG.py
    ```

## 🤝 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.