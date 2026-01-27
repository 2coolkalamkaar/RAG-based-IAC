
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    unzip \
    software-properties-common \
    gnupg \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy installation scripts
COPY install_terraform.sh .
COPY install_tflint.sh .

# Install Terraform and TFLint
RUN chmod +x install_terraform.sh install_tflint.sh && \
    ./install_terraform.sh && \
    ./install_tflint.sh

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the app
ENTRYPOINT ["streamlit", "run", "RAG.py", "--server.port=8501", "--server.address=0.0.0.0"]