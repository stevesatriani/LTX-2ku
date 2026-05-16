#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch.sh  –  LTX-2 launcher for SimplePod
#
# Usage:
#   bash launch.sh                  # Start generation UI (port 7860)
#   bash launch.sh --share          # Generation UI with public URL
#   bash launch.sh --port 8080      # Custom port
#
#   bash launch.sh --download       # Start model downloader (port 7861)
#   bash launch.sh --download --share
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Parse --download flag ─────────────────────────────────────────────────────
MODE="app"
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--download" ]]; then
        MODE="download"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# ── 1. Activate virtualenv if present ────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "→ Activating .venv"
    source .venv/bin/activate
fi

# ── 2. Verify packages ────────────────────────────────────────────────────────
python -c "import gradio" 2>/dev/null || {
    echo "→ Installing gradio..."
    pip install gradio>=4.0
}

if [[ "$MODE" == "app" ]]; then
    python -c "import ltx_pipelines" 2>/dev/null || {
        echo "→ Installing ltx packages..."
        pip install -e packages/ltx-core packages/ltx-pipelines
    }
fi

# ── 3. Set model paths (edit these or use env vars) ───────────────────────────
# Uncomment and set your actual paths:

# MODEL_DIR="/workspace/models"
# export LTX_CHECKPOINT_PATH="$MODEL_DIR/ltx-2.3-22b-dev.safetensors"
# export LTX_DISTILLED_CHECKPOINT="$MODEL_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
# export LTX_GEMMA_ROOT="$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
# export LTX_UPSAMPLER_PATH="$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
# export LTX_DISTILLED_LORA_PATH="$MODEL_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
# export LTX_OUTPUT_DIR="/workspace/outputs"
# export LTX_MODEL_DIR="$MODEL_DIR"   # used by downloader

# ── 4. Recommended CUDA allocator ─────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── 5. Launch ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$MODE" == "download" ]]; then
    echo "  LTX-2 Model Downloader  (port 7861)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python download_models.py "${EXTRA_ARGS[@]}"
else
    echo "  LTX-2 Web UI  (port 7860)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python app.py "${EXTRA_ARGS[@]}"
fi
