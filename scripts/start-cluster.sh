#!/bin/bash
# Run on your main Mac — SSHs into all nodes and starts VLM servers
# Edit the hosts below to match your network

M4_1="${M4_1_HOST:-m4-1.local}"
M4_2="${M4_2_HOST:-m4-2.local}"
PC_3070="${PC_3070_HOST:-3070.local}"
SSH_USER="${VLM_SSH_USER:-$(whoami)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

_ssh_start() {
    local host="$1"
    local script="$2"
    echo "==> Starting node: $host"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
        "$SSH_USER@$host" \
        "bash -s" < "$SCRIPT_DIR/$script" \
        > "/tmp/shrimp-node-$host.log" 2>&1 &
    echo "    log: /tmp/shrimp-node-$host.log"
}

_ssh_start "$M4_1"  "start-vlm-apple.sh"
_ssh_start "$M4_2"  "start-vlm-apple.sh"
_ssh_start "$PC_3070" "start-vlm-cuda.sh"

echo ""
echo "==> All nodes starting. Waiting 15s then checking health..."
sleep 15

for host in "$M4_1" "$M4_2"; do
    if curl -sf "http://$host:8081/health" > /dev/null 2>&1; then
        echo "  ✓ $host:8081 up"
    else
        echo "  ✗ $host:8081 not responding (check /tmp/shrimp-node-$host.log)"
    fi
done

if curl -sf "http://$PC_3070:11434/api/tags" > /dev/null 2>&1; then
    echo "  ✓ $PC_3070:11434 up"
else
    echo "  ✗ $PC_3070:11434 not responding (check /tmp/shrimp-node-$PC_3070.log)"
fi

echo ""
echo "==> Now start the router:"
echo "    cd $(dirname "$SCRIPT_DIR") && ./scripts/start-router.sh"
