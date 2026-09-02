#!/bin/bash
set -e

# Install Infracost CLI (used by workflows/agent_workflow_advanced_rag.py's Cost_Estimator_Node
# and workflows/agent_workflow_hitl.py's Plan_Node for real per-resource pricing).
curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh | sh

# Verify installation
infracost --version

echo "Infracost installed successfully."
echo ""
echo "Infracost needs a free API key to price resources — register one with:"
echo "    infracost auth login"
echo "or set INFRACOST_API_KEY in your .env directly (see https://www.infracost.io/docs/#quick-start)."
echo "Without a key, cost estimation falls back to the static per-resource-type table in"
echo "workflows/blast_radius_guard.py."
