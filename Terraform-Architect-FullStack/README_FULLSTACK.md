
# Terraform Architect Agent - Full Stack

This directory contains the full-stack containerized version of the Terraform Architect Agent.

## Architecture

*   **Frontend**: React + Vite + Tailwind CSS (served via Nginx)
*   **Backend**: FastAPI + LangChain + RAG (ChromaDB) + SQLite
*   **Infrastructure**: Docker Compose

## Prerequisites

*   Docker
*   Docker Compose

## Quick Start

1.  **Environment Variables**:
    Ensure `.env` exists in `Terraform-Architect-FullStack/` (or `backend/`) with your API keys:
    ```
    GOOGLE_API_KEY=your_key_here
    ```

2.  **Run with Docker Compose**:
    ```bash
    cd Terraform-Architect-FullStack
    docker-compose up --build -d
    ```

3.  **Access the Application**:
    *   Frontend: [http://localhost](http://localhost) (Login/Register to start)
    *   Backend API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Features

*   **Persistent Chat**: Conversations are saved in `users.db`.
*   **RAG**: Uses local vector store (`chroma_db_terraform`) persisted via Docker volumes.
*   **Security**: Scans Terraform code with `tflint` (installed in backend container).
*   **Visualization**: Generates diagrams dynamically.

## Troubleshooting

*   If `localhost` doesn't load immediately, wait a minute for Nginx and Backend to initialize.
*   Check logs: `docker-compose logs -f`

