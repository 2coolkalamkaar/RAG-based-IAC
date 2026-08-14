#!/bin/bash
set -e

# Install prerequisites
apt-get update && apt-get install -y curl unzip

# Install TFLint
curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash

# Verify installation
tflint --version
echo "TFLint installed successfully."
