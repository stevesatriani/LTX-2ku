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

# ── Resolve python binary (python3 on most Linux systems) ─────────────────────
PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: no python3 or python found in PATH" >&2
    exit 1
fi
PIP="$PYTHON -m pip"

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
    PYTHON=python
    PIP="python -m pip"
fi

# ── 2. Verify packages ────────────────────────────────────────────────────────
# uv is required as the build backend for ltx-core and ltx-pipelines
command -v uv >/dev/null 2>&1 || {
    echo "→ Installing uv (build tool)..."
    $PIP install uv --quiet
    # Make sure uv binary is on PATH (pip may install to ~/.local/bin)
    export PATH="$HOME/.local/bin:$PATH"
}

$PYTHON -c "import gradio" 2>/dev/null || {
    echo "→ Installing gradio..."
    $PIP install "gradio>=4.0"
}

if [[ "$MODE" == "app" ]]; then
    # Verify the local editable install is active by checking location of blocks.py.
    # If it resolves to /usr/local/lib (old system install), force a reinstall.
    _blocks_path=$($PYTHON -c "import ltx_pipelines.utils.blocks as b; import inspect; print(inspect.getfile(b))" 2>/dev/null || echo "")
    _needs_install=0
    if [[ -z "$_blocks_path" ]]; then
        _needs_install=1
    elif [[ "$_blocks_path" != *"packages/ltx-pipelines"* ]] && [[ "$_blocks_path" != *"LTX-2"* ]]; then
        echo "→ System ltx_pipelines detected at: $_blocks_path"
        _needs_install=1
    fi
    if [[ "$_needs_install" == "1" ]]; then
        echo "→ Installing/reinstalling ltx packages from local source..."
        $PIP install "uv_build>=0.9.8,<0.10.0" --quiet
        $PIP install --force-reinstall --no-build-isolation -e packages/ltx-core -e packages/ltx-pipelines
    fi
fi

$PYTHON -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
    echo "→ Installing/upgrading huggingface_hub..."
    $PIP install "huggingface_hub>=0.20"
}

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
    $PYTHON download_models.py "${EXTRA_ARGS[@]}"
else
    echo "  LTX-2 Web UI  (port 7860)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    $PYTHON app.py "${EXTRA_ARGS[@]}"
fi
