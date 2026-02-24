h#!/bin/bash

cd /scratch/baspinal/agent-project

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: ./user-server.sh <model-id> <port>"
  exit 1
fi

MODEL_ID="$1"
PORT="$2"

#########################################
# HUGGING FACE AUTH (REQUIRED)
#########################################
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "ERROR: ~/.hf_token not found."
  echo "Create it with your Hugging Face token (chmod 600 ~/.hf_token)."
  exit 1
fi

echo "Starting USER vLLM server..."
echo "  Model: $MODEL_ID"
echo "  Port : $PORT"
echo

vllm serve "$MODEL_ID" \
  --tensor-parallel-size 1 \
  --port "$PORT"
