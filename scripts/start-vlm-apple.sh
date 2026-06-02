#!/bin/bash
# Run on M4 Mac Minis — starts mlx_vlm vision server
# Usage: ./start-vlm-apple.sh

MODEL="${VLM_MODEL:-mlx-community/Qwen3-VL-8B-Instruct-6bit}"
PORT="${VLM_PORT:-8081}"

echo "==> Checking mlx_vlm..."
if ! python3 -c "import mlx_vlm" 2>/dev/null; then
    echo "==> Installing mlx_vlm..."
    pip install mlx-vlm -q
fi

echo "==> Starting VLM server: $MODEL on :$PORT"
python3 -m mlx_vlm.server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0
