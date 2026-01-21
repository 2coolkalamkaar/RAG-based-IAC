
# Docker Setup for Terraform Architect Agent

This project is containerized to make setup and execution extremely fast and reliable.

## Prerequisites

- Docker
- Docker Compose (optional, but recommended)

## Quick Start (with Docker Compose)

1. **Build and Run**:
   ```bash
   docker-compose up --build
   ```
   This command will build the image and start the container. It uses your local `.env` file for credentials.

2. **Access the App**:
   Open a browser and navigate to [http://localhost:8501](http://localhost:8501).

## Manual Docker Run

If you prefer not to use Docker Compose:

1. **Build the Image**:
   ```bash
   docker build -t terraform-architect .
   ```

2. **Run the Container**:
   You need to pass your Google API Key.
   ```bash
   docker run -p 8501:8501 --env-file .env terraform-architect
   ```

## Notes

- The `chroma_db_terraform` directory is copied into the image, so the vector database is self-contained.
- If you rebuild the vector database locally, you should rebuild the Docker image (`docker-compose up --build`) or mount the directory as a volume (configured in `docker-compose.yml`).
