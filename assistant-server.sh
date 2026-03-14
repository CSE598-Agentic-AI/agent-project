#!/bin/bash

cd /scratch/baspinal/agent-project

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: ./assistant-server.sh <model-id> <port> <agent-strategy>"
  echo "Agent strategies: tool-calling | react | act | few-shot"
  exit 1
fi

MODEL_ID="$1"
PORT="$2"
AGENT_STRATEGY="$3"

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

#########################################
# vLLM FLAGS (BASE)
#########################################
VLLM_ARGS=(
  "$MODEL_ID"
  --tensor-parallel-size 1
  --port "$PORT"
  --max-model-len 16384
)

#########################################
# TOOL-CALLING SUPPORT (CONDITIONAL)
#########################################
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
echo "Starting ASSISTANT vLLM server..."
echo "  Model   : $MODEL_ID"
echo "  Port    : $PORT"
echo "  Strategy: $AGENT_STRATEGY"
echo

vllm serve "${VLLM_ARGS[@]}"
