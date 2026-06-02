#!/bin/bash
# Run on your main Mac — starts the shrimp-router gateway
# Usage: ./scripts/start-router.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${SHRIMP_ROUTER_CONFIG:-$REPO_DIR/config.yaml}"
VENV="$REPO_DIR/.venv"

if [ ! -f "$CONFIG" ]; then
    echo "==> No config.yaml found. Copying from example..."
    cp "$REPO_DIR/config.example.yaml" "$CONFIG"
    echo "    Edit $CONFIG with your hostnames and API keys, then re-run."
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "==> Creating venv..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -e "$REPO_DIR" -q
fi

echo "==> Starting shrimp-router..."
SHRIMP_ROUTER_CONFIG="$CONFIG" "$VENV/bin/shrimp-router"
