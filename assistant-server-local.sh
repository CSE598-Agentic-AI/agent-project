#!/bin/bash
# Local version of assistant-server.sh: no hardcoded paths (no SLURM).
# Usage: ./assistant-server-local.sh <model-id> <port> <agent-strategy>

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${ROOT_DIR}"

if [ "$#" -lt 3 ]; then
  echo "Usage: ./assistant-server-local.sh <model-id> <port> <agent-strategy>"
  echo "Agent strategies: tool-calling | react | act | few-shot"
  exit 1
fi

MODEL_ID="$1"
PORT="$2"
AGENT_STRATEGY="$3"

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "ERROR: ~/.hf_token not found."
  echo "Create it with your Hugging Face token (chmod 600 ~/.hf_token)."
  exit 1
fi

VLLM_ARGS=(
  "$MODEL_ID"
  --host "0.0.0.0"
  --port "$PORT"
  --tensor-parallel-size 1
  --max-model-len 16384
)

if [ "$AGENT_STRATEGY" = "tool-calling" ]; then
  echo "Enabling tool-calling support (auto tool choice + hermes parser)"
  VLLM_ARGS+=(
    --enable-auto-tool-choice
    --tool-call-parser hermes
  )
else
  echo "Starting server without tool-calling support"
fi

echo
echo "Starting ASSISTANT vLLM server (local)..."
echo "  Model   : $MODEL_ID"
echo "  Port    : $PORT"
echo "  Strategy: $AGENT_STRATEGY"
echo

vllm serve "${VLLM_ARGS[@]}"
