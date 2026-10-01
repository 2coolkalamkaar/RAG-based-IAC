FROM python:3.11-slim-bookworm

WORKDIR /app

# ── System dependencies + HashiCorp apt repo (for Terraform) ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    unzip \
    gnupg \
    lsb-release \
    ca-certificates \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /etc/apt/keyrings/hashicorp-archive-keyring.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/hashicorp.list \
    && apt-get update && apt-get install -y terraform \
    && rm -rf /var/lib/apt/lists/*

# ── TFLint (install the release binary directly — more robust than an install
#    script whose path can move) ─────────────────────────────────────────────
RUN TFLINT_VERSION=$(curl -fsSL https://api.github.com/repos/terraform-linters/tflint/releases/latest | \
      grep -oP '"tag_name": "\K(.*)(?=")') && \
    curl -fsSL -o /tmp/tflint.zip \
      "https://github.com/terraform-linters/tflint/releases/download/${TFLINT_VERSION}/tflint_linux_amd64.zip" && \
    unzip /tmp/tflint.zip -d /usr/local/bin && \
    rm /tmp/tflint.zip && \
    chmod +x /usr/local/bin/tflint

# ── Infracost (optional cost estimates; app has a static fallback if absent) ──
RUN curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh \
    | sh || echo "Infracost install skipped — app will use its static cost fallback"

# ── Python dependencies ────────────────────────────────────────────────────────
# Install CPU-only torch FIRST. Plain `pip install torch` pulls the full CUDA/GPU
# build (~1.7GB plus ~10 separate nvidia-*-cu12 packages) even though this app
# only ever does CPU inference (the embedding model + CrossEncoder reranker) and
# Cloud Run has no GPU — installing the CPU wheel first satisfies every other
# package's torch dependency without pip ever reaching for the CUDA one.
COPY requirements.txt .
RUN grep -oP '^torch==\K.*' requirements.txt | xargs -I{} \
    pip install --no-cache-dir torch=={} --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── App code (chroma_db_terraform is baked in — Cloud Run has no durable local
#    disk across instances, so the vector index ships inside the image) ────────
COPY . .

# Pre-fetch the hashicorp/aws provider plugin (~675MB) into the image's plugin
# cache at build time. Without this, every Cloud Run cold start would
# re-download it on the first `terraform init` — the exact redundant-download
# cost this project's TF_PLUGIN_CACHE_DIR fix already eliminated within a run;
# baking it in eliminates it across cold starts too.
ENV TF_PLUGIN_CACHE_DIR=/app/.terraform-plugin-cache
RUN mkdir -p "$TF_PLUGIN_CACHE_DIR" /tmp/tf-prefetch && \
    printf 'terraform {\n  required_providers {\n    aws = {\n      source  = "hashicorp/aws"\n      version = "~> 5.0"\n    }\n  }\n}\n' \
      > /tmp/tf-prefetch/versions.tf && \
    cd /tmp/tf-prefetch && terraform init -backend=false && \
    rm -rf /tmp/tf-prefetch

# Cloud Run injects $PORT (defaults to 8080); respect it rather than hardcoding.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT}
