#!/bin/bash
# Run on 3070 PC — starts llama.cpp server with Qwen2.5-VL vision model
# Usage: ./start-vlm-cuda.sh
#
# Env overrides:
#   VLM_MODEL_DIR   where to store GGUF files  (default: ~/.local/share/vlm-models)
#   VLM_MODEL       GGUF filename               (default: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf)
#   VLM_MODEL_ALIAS model name returned by API  (default: Qwen2.5-VL-7B-Instruct-Q4_K_M)
#   VLM_MMPROJ      mmproj GGUF filename        (default: mmproj-Qwen3-VL-8B-Instruct-F16.gguf)
#   VLM_PORT        port to listen on           (default: 11434)
#   VLM_GPU_LAYERS  layers to offload to GPU    (default: auto-fit to free VRAM;
#                                                 forcing 999 disables auto-fit and can OOM
#                                                 once the mmproj vision encoder is loaded too)
#   VLM_CTX         context size                (default: 8192)
#   LLAMA_DIR       where to install llama.cpp  (default: ~/.local/lib/llama.cpp)

set -euo pipefail

MODEL_DIR="${VLM_MODEL_DIR:-$HOME/.local/share/vlm-models}"
MODEL_FILE="${VLM_MODEL:-Qwen3-VL-8B-Instruct-Q6_K.gguf}"
MODEL_ALIAS="${VLM_MODEL_ALIAS:-Qwen3-VL-8B-Instruct-6bit}"
MMPROJ_FILE="${VLM_MMPROJ:-mmproj-Qwen3-VL-8B-Instruct-F16.gguf}"
PORT="${VLM_PORT:-11434}"
GPU_LAYERS="${VLM_GPU_LAYERS:-}"
CTX="${VLM_CTX:-8192}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/.local/lib/llama.cpp}"
LLAMA_SERVER="$LLAMA_DIR/llama-server"

HF_REPO="lmstudio-community/Qwen3-VL-8B-Instruct-GGUF"

mkdir -p "$MODEL_DIR" "$LLAMA_DIR"

# --- Install llama.cpp if needed ---
if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "==> Fetching latest llama.cpp CUDA release info..."
    RELEASE_URL=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
assets = [a['browser_download_url'] for a in data.get('assets', [])]
# Prefer CUDA, fall back to Vulkan (both use the GPU on NVIDIA via libcuda/Vulkan ICD)
for pattern in [('ubuntu', 'cuda', 'x64'), ('ubuntu', 'vulkan-x64',)]:
    for u in assets:
        if all(p in u for p in pattern) and u.endswith('.tar.gz'):
            print(u)
            sys.exit(0)
")

    if [[ -z "$RELEASE_URL" ]]; then
        echo "ERROR: Could not find a Linux CUDA x64 llama.cpp release asset." >&2
        echo "Check https://github.com/ggml-org/llama.cpp/releases and set LLAMA_DIR manually." >&2
        exit 1
    fi

    echo "==> Downloading $RELEASE_URL..."
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    curl -L --progress-bar "$RELEASE_URL" -o "$TMP/llama.tar.gz"
    tar -xf "$TMP/llama.tar.gz" -C "$TMP"

    # Binary can be at root or inside a subdirectory
    BINARY=$(find "$TMP" -name "llama-server" -type f | head -1)
    if [[ -z "$BINARY" ]]; then
        echo "ERROR: llama-server binary not found in archive." >&2
        exit 1
    fi
    cp "$BINARY" "$LLAMA_SERVER"

    # Copy bundled CUDA libs if present (needed when system toolkit isn't installed)
    LIB_DIR=$(find "$TMP" -maxdepth 2 -name "*.so*" -printf "%h\n" | sort -u | head -1)
    if [[ -n "$LIB_DIR" ]]; then
        cp "$LIB_DIR"/*.so* "$LLAMA_DIR/" 2>/dev/null || true
    fi

    chmod +x "$LLAMA_SERVER"
    echo "==> llama.cpp installed to $LLAMA_DIR"
fi

# --- Download model if needed ---
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "==> Downloading $MODEL_FILE from HuggingFace..."
    HF_URL="https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE"
    curl -L --progress-bar "$HF_URL" -o "$MODEL_PATH.tmp"
    mv "$MODEL_PATH.tmp" "$MODEL_PATH"
    echo "==> Model saved to $MODEL_PATH"
fi

# --- Download mmproj (vision projector) if needed ---
MMPROJ_PATH="$MODEL_DIR/$MMPROJ_FILE"
if [[ ! -f "$MMPROJ_PATH" ]]; then
    echo "==> Downloading $MMPROJ_FILE from HuggingFace..."
    MMPROJ_URL="https://huggingface.co/$HF_REPO/resolve/main/$MMPROJ_FILE"
    curl -L --progress-bar "$MMPROJ_URL" -o "$MMPROJ_PATH.tmp"
    mv "$MMPROJ_PATH.tmp" "$MMPROJ_PATH"
    echo "==> mmproj saved to $MMPROJ_PATH"
fi

# --- Start server ---
echo "==> Starting llama-server on 0.0.0.0:$PORT (model: $MODEL_ALIAS)..."
export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"

ARGS=(
    --model        "$MODEL_PATH"
    --mmproj       "$MMPROJ_PATH"
    --no-mmproj-offload
    --alias        "$MODEL_ALIAS"
    --host         0.0.0.0
    --port         "$PORT"
    --ctx-size     "$CTX"
    --parallel     2
    --flash-attn on
)
if [[ -n "$GPU_LAYERS" ]]; then
    ARGS+=(--n-gpu-layers "$GPU_LAYERS")
fi

exec "$LLAMA_SERVER" "${ARGS[@]}"
