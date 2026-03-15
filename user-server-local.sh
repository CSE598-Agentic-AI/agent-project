#!/bin/bash
# Local version of user-server.sh: starts USER vLLM server (no SLURM).
# Usage: ./user-server-local.sh [model_id] [port]
# Example: ./user-server-local.sh Qwen/Qwen3-32B 8007

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${ROOT_DIR}"

PORT="${2:-8007}"
USER_MODEL="${1:-Qwen/Qwen3-32B}"

# Optional: load conda/mamba and activate your vLLM env (uncomment and set name)
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate your-vllm-env

export NO_AI_TRACKING="${NO_AI_TRACKING:-true}"

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "WARN: ~/.hf_token not found. Set it if the model requires auth."
fi

echo "===================================="
echo "Starting USER vLLM server (local)"
echo "User model:        ${USER_MODEL}"
echo "User server base:  http://127.0.0.1:${PORT}/v1"
echo "===================================="
echo

vllm serve \
  "$USER_MODEL" \
  --host "0.0.0.0" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 16384
