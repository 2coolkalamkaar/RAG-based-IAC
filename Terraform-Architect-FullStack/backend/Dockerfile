
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install System Dependencies (Terraform, TFLint, Graphviz)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    gnupg \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Terraform & TFLint using scripts (copied later or inline)
# For simplicity, we'll use commands here directly to avoid script permission issues during build
# Install Terraform
RUN wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list && \
    apt-get update && apt-get install -y terraform

RUN curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash

# Python Deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy App
COPY . .

# Expose
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
