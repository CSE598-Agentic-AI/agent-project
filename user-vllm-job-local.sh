#!/bin/bash
# Local version of user-vllm-job.sh: run the USER vLLM server (no SLURM).
# Run this in a terminal, then set USER_MODEL_API_BASE for tau-experiment-local.sh.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LOG_DIR="${ROOT_DIR}/logs"
ERROR_DIR="${ROOT_DIR}/errors"
mkdir -p "${LOG_DIR}" "${ERROR_DIR}"
cd "${ROOT_DIR}"

PORT="${PORT:-8007}"
USER_MODEL="${USER_MODEL:-Qwen/Qwen3-32B}"

# Uncomment if you use conda/mamba for your vLLM environment:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate your-vllm-env
# Or: source /path/to/venv/bin/activate

export NO_AI_TRACKING="${NO_AI_TRACKING:-true}"

echo "===================================="
echo "Starting USER vLLM server (local)"
echo "User model:        ${USER_MODEL}"
echo "User server base:  http://127.0.0.1:${PORT}/v1"
echo "===================================="
echo

./user-server-local.sh "$USER_MODEL" "$PORT"
