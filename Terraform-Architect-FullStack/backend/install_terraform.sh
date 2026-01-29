#!/bin/bash
set -e

# Install prerequisites
apt-get update && apt-get install -y gnupg software-properties-common curl

# Install Terraform
curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add -
apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
apt-get update && apt-get install -y terraform

# Verify installation
terraform --version
echo "Terraform installed successfully."
