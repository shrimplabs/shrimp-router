#!/bin/bash
# Run on 3070 PC — starts ollama with Qwen2.5-VL
# Usage: ./start-vlm-cuda.sh

MODEL="${VLM_MODEL:-qwen2.5vl:7b}"

echo "==> Checking ollama..."
if ! command -v ollama &>/dev/null; then
    echo "==> Installing ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> Pulling $MODEL (skips if already downloaded)..."
ollama pull "$MODEL"

echo "==> Starting ollama on 0.0.0.0:11434..."
OLLAMA_HOST=0.0.0.0 ollama serve
