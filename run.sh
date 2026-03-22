#!/bin/bash
# NKI MoE Kernel Agent - Quick Start
#
# Usage:
#   ./run.sh <trn2-host> [rounds]
#
# Example:
#   ./run.sh ubuntu@10.0.1.100 50
#   ./run.sh ubuntu@10.0.1.100 --rounds 100 --no-dge

set -e

HOST="${1:?Usage: ./run.sh <trn2-host> [additional args...]}"
shift

echo "=== NKI MoE Kernel Optimization Agent ==="
echo "Target: ${HOST}"
echo "Starting at: $(date)"
echo ""

# Check SSH connectivity
echo "Checking SSH connection..."
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${HOST}" "echo 'Connected to Trn2'" || {
    echo "ERROR: Cannot connect to ${HOST}"
    echo "Make sure:"
    echo "  1. Trn2 instance is running"
    echo "  2. SSH key is configured"
    echo "  3. Security group allows SSH"
    exit 1
}

# Run the orchestrator
echo ""
echo "Starting optimization loop..."
python3 orchestrator.py \
    --host "${HOST}" \
    --rounds 50 \
    "$@"
