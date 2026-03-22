#!/bin/bash
# Trn2 Instance Setup Script
# Run this ONCE on the Trn2 instance after SSH access is established.
#
# Usage: ssh ubuntu@<trn2-ip> 'bash -s' < setup_trn2.sh

set -euo pipefail

echo "=== Trn2 Instance Setup for NKI-MoE Competition ==="
echo "Date: $(date)"
echo ""

# 1. Activate Neuron virtual environment
echo "[1/6] Activating Neuron environment..."
VENV_PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "  ✅ Activated: ${VENV_PATH}"
else
    echo "  ⚠️  Default venv not found. Trying alternatives..."
    # Try to find any neuron venv
    VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | tail -1)
    if [ -n "$VENV" ]; then
        source "${VENV}/bin/activate"
        echo "  ✅ Activated: ${VENV}"
    else
        echo "  ❌ No Neuron venv found. Make sure you're using a Neuron DLAMI."
        exit 1
    fi
fi

# 2. Verify SDK version
echo ""
echo "[2/6] Checking Neuron SDK version..."
pip show aws-neuronx-runtime 2>/dev/null | grep Version || echo "  (runtime version check skipped)"
python3 -c "import nki; print(f'  ✅ NKI import OK (nki namespace)')" 2>/dev/null || {
    python3 -c "import neuronxcc.nki; print('  ⚠️  Old namespace (neuronxcc.nki) — SDK may be <2.28')" 2>/dev/null || {
        echo "  ❌ NKI not available"
        exit 1
    }
}

# 3. Clone competition repo
echo ""
echo "[3/6] Setting up competition repo..."
cd ~
if [ -d "nki-moe" ]; then
    echo "  nki-moe already exists. Pulling latest..."
    cd nki-moe && git pull 2>/dev/null || echo "  (git pull skipped)"
    cd ~
else
    echo "  Cloning nki-moe repo..."
    echo "  ⚠️  Replace with actual competition repo URL:"
    echo "  git clone <COMPETITION_REPO_URL> nki-moe"
    mkdir -p nki-moe
    echo "  Created ~/nki-moe placeholder"
fi

# 4. Download model
echo ""
echo "[4/6] Checking model..."
MODEL_DIR="$HOME/qwen-30b-a3b/hf_model"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "  ✅ Model already exists at $MODEL_DIR"
else
    echo "  Downloading Qwen3-30B-A3B..."
    mkdir -p "$HOME/qwen-30b-a3b"
    pip install -q huggingface_hub 2>/dev/null
    huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir "$MODEL_DIR" || {
        echo "  ❌ Download failed. Run manually:"
        echo "  huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir $MODEL_DIR"
    }
fi

# 5. Verify neuron-profile tool
echo ""
echo "[5/6] Checking neuron-profile..."
which neuron-profile >/dev/null 2>&1 && {
    echo "  ✅ neuron-profile available: $(which neuron-profile)"
} || {
    echo "  ❌ neuron-profile not found in PATH"
}

# 6. Quick sanity check
echo ""
echo "[6/6] Quick sanity check..."
python3 -c "
import numpy as np
try:
    import nki
    import nki.language as nl
    print('  ✅ NKI SDK 2.28 namespace OK')
except ImportError:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    print('  ⚠️  Using old namespace — update to SDK 2.28')

print(f'  NumPy: {np.__version__}')
" 2>&1

# 7. Add venv activation to bashrc for convenience
echo ""
if ! grep -q "aws_neuronx_venv" ~/.bashrc 2>/dev/null; then
    echo "# Auto-activate Neuron environment" >> ~/.bashrc
    echo "source ${VENV_PATH}/bin/activate 2>/dev/null || true" >> ~/.bashrc
    echo "  Added auto-activation to ~/.bashrc"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify model: ls ~/qwen-30b-a3b/hf_model/"
echo "  2. Clone actual competition repo into ~/nki-moe/"
echo "  3. Test baseline: cd ~/nki-moe && python3 main.py --mode benchmark ..."
echo "  4. From local machine: ./run.sh ubuntu@$(hostname -I | awk '{print $1}')"
