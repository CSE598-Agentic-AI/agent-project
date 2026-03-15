#!/bin/bash
# Local version of tau-experiment.sh: run τ-Bench experiments locally (no SLURM).
# Prereq: Start the USER vLLM server first, e.g. ./user-vllm-job-local.sh (or set USER_MODEL_API_BASE).
#
# Single run:
#   ./tau-experiment-local.sh [--start-index N] [--end-index M] <env> <agent> <assist_model> [num_trials]
# Example:
#   export USER_MODEL_API_BASE="http://127.0.0.1:8007/v1"
#   ./tau-experiment-local.sh airline react Qwen/Qwen3-4B-Instruct-2507 2

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LOG_DIR="${ROOT_DIR}/logs"
ERROR_DIR="${ROOT_DIR}/errors"
BASE_RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${LOG_DIR}" "${ERROR_DIR}" "${BASE_RESULTS_DIR}"
cd "${ROOT_DIR}"

# Default USER model API to localhost if not set
export USER_MODEL_API_BASE="${USER_MODEL_API_BASE:-http://127.0.0.1:8007/v1}"

echo "Using USER_MODEL_API_BASE=${USER_MODEL_API_BASE}"
echo

#########################################
# EXPERIMENT GRID (for array-like runs)
#########################################
ENVS=(airline retail)
AGENTS=(act react fc multi-agent)
MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-8B-Instruct-2507"
  "Qwen/Qwen3-14B-Instruct-2507"
  "Qwen/Qwen3-32B-Instruct-2507"
)
TRIALS=(1 2 3 4 5)

NUM_ENVS=${#ENVS[@]}
NUM_AGENTS=${#AGENTS[@]}
NUM_MODELS=${#MODELS[@]}
NUM_TRIALS=${#TRIALS[@]}
TOTAL=$((NUM_ENVS * NUM_AGENTS * NUM_MODELS * NUM_TRIALS))

#########################################
# ARGUMENT / OPTIONAL ARRAY INDEX
#########################################
START_INDEX=0
END_INDEX=-1
ARGS=()
i=1
while [[ $i -le $# ]]; do
  if [[ "${!i}" == "--start-index" ]] && [[ $i -lt $# ]]; then
    ((i++))
    START_INDEX="${!i}"
  elif [[ "${!i}" == "--end-index" ]] && [[ $i -lt $# ]]; then
    ((i++))
    END_INDEX="${!i}"
  else
    ARGS+=("${!i}")
  fi
  ((i++))
done

if [ "${#ARGS[@]}" -ge 3 ]; then
  ENV_NAME="${ARGS[0]}"
  AGENT_STRAT_INPUT="${ARGS[1]}"
  ASSIST_MODEL="${ARGS[2]}"
  if [ "${#ARGS[@]}" -ge 4 ]; then
    NUM_TRIALS_VAL="${ARGS[3]}"
  else
    NUM_TRIALS_VAL=5
  fi

  if [[ "$ENV_NAME" != "retail" && "$ENV_NAME" != "airline" ]]; then
    echo "Error: environment must be 'retail' or 'airline', got '$ENV_NAME'"
    exit 1
  fi

  case "$AGENT_STRAT_INPUT" in
    act|react|fc|multi-agent) ;;
    *)
      echo "Error: agent strategy must be one of: act, react, fc, multi-agent (got '$AGENT_STRAT_INPUT')"
      exit 1
      ;;
  esac
else
  echo "Usage:"
  echo "  Single run: ./tau-experiment-local.sh [--start-index N] [--end-index M] <env: retail|airline> <agent: act|react|fc|multi-agent> <assistant_model_id> [num_trials]"
  echo ""
  echo "Example:"
  echo "  export USER_MODEL_API_BASE=\"http://127.0.0.1:8007/v1\""
  echo "  ./tau-experiment-local.sh airline react Qwen/Qwen3-4B-Instruct-2507 2"
  exit 1
fi

#########################################
# Map AGENT_STRAT_INPUT -> Tau-Bench CLI
#########################################
case "$AGENT_STRAT_INPUT" in
  act)    AGENT_STRAT_CLI="act" ;;
  react)  AGENT_STRAT_CLI="react" ;;
  fc)     AGENT_STRAT_CLI="tool-calling" ;;
  multi-agent) AGENT_STRAT_CLI="multi-agent" ;;
  *)
    echo "Error: agent strategy must be one of: act, react, fc, multi-agent (got '$AGENT_STRAT_INPUT')"
    exit 1
    ;;
esac

USER_MODEL="Qwen/Qwen3-32B"
ASSIST_SAFE="${ASSIST_MODEL//\//_}"

#########################################
# MODEL SIZE (for dir structure)
#########################################
MODEL_SIZE="unknown"
case "$ASSIST_MODEL" in
  *"4B"*)  MODEL_SIZE="4B" ;;
  *"8B"*)  MODEL_SIZE="8B" ;;
  *"14B"*) MODEL_SIZE="14B" ;;
  *"32B"*) MODEL_SIZE="32B" ;;
esac

echo "========================================"
echo "Environment:         $ENV_NAME"
echo "Agent strategy:      $AGENT_STRAT_INPUT (CLI: $AGENT_STRAT_CLI)"
echo "Assistant model:     $ASSIST_MODEL"
echo "User model (fixed):  $USER_MODEL"
echo "Num trials:          $NUM_TRIALS_VAL"
echo "Start index:         $START_INDEX"
echo "End index:           $END_INDEX"
echo "Model size bucket:   $MODEL_SIZE"
echo "========================================"
echo

RESULTS_SUBDIR="${BASE_RESULTS_DIR}/${ENV_NAME}/${AGENT_STRAT_INPUT}/${MODEL_SIZE}"
mkdir -p "${RESULTS_SUBDIR}"
LOG_SUBDIR="${LOG_DIR}/${ENV_NAME}/${AGENT_STRAT_INPUT}/${MODEL_SIZE}"
mkdir -p "${LOG_SUBDIR}"

echo "Results directory:   $RESULTS_SUBDIR"
echo "Log directory:       ${LOG_SUBDIR}"
echo

export NO_AI_TRACKING="${NO_AI_TRACKING:-true}"

# Uncomment if you use conda/mamba for your vLLM environment:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate your-vllm-env

#########################################
# Quick health check on USER server
#########################################
echo "Checking USER model endpoint health..."
if ! curl -s "${USER_MODEL_API_BASE}/models" > /dev/null; then
  echo "ERROR: Could not reach USER_MODEL_API_BASE=${USER_MODEL_API_BASE}/models"
  echo "Start the user server first: ./user-vllm-job-local.sh"
  exit 1
fi
echo "USER endpoint is reachable."
echo

#########################################
# START ASSISTANT SERVER (LOCAL)
#########################################
echo "Starting ASSISTANT vLLM on port 8005..."
ASSIST_LOG="${LOG_SUBDIR}/assistant-num_trials${NUM_TRIALS_VAL}-local.log"
./assistant-server-local.sh "$ASSIST_MODEL" 8005 "$AGENT_STRAT_CLI" \
  > "${ASSIST_LOG}" 2>&1 &
ASSIST_PID=$!

echo "Assistant PID: $ASSIST_PID"
echo "Assistant log: ${ASSIST_LOG}"
echo

wait_for_ready() {
  local name="$1"
  local base="$2"
  local max_attempts=240
  local attempt=1

  echo "Waiting for ${name} server at ${base}/models to become ready..."

  while (( attempt <= max_attempts )); do
    if curl -s "${base}/models" > /dev/null; then
      echo "${name} server at ${base} is ready (after ${attempt} attempts)."
      return 0
    fi
    echo "  [${name}] not ready yet (attempt ${attempt}/${max_attempts}). Sleeping 5s..."
    sleep 5
    ((attempt++))
  done

  echo "ERROR: ${name} server at ${base} did not become ready in time."
  return 1
}

ASSIST_BASE="http://127.0.0.1:8005/v1"

if ! wait_for_ready "ASSISTANT" "$ASSIST_BASE"; then
  echo "Assistant server failed to start correctly. Cleaning up and exiting."
  kill "$ASSIST_PID" 2>/dev/null || true
  wait "$ASSIST_PID" 2>/dev/null || true
  exit 1
fi

#########################################
# EXPORT ENDPOINTS
#########################################
export OPENAI_API_BASE="${ASSIST_BASE}"
export OPENAI_API_KEY="EMPTY"

if [[ "$AGENT_STRAT_INPUT" == "multi-agent" ]]; then
  echo "LLM Sentinel and FACT agent use OPENAI_API_BASE (same as Assistant: ${ASSIST_BASE})"
fi

#########################################
# RUN τ-BENCH
#########################################
cd "${ROOT_DIR}/tau-bench"

case "$ENV_NAME" in
  retail)  TOTAL_TASKS=115 ;;
  airline) TOTAL_TASKS=50 ;;
  *)       TOTAL_TASKS=0 ;;
esac

if [[ "${SKIP_RUN:-0}" -eq 1 ]]; then
  echo "Skipping Tau-Bench run (already finished)."
  TB_EXIT=0
else
  python run.py \
    --agent-strategy "$AGENT_STRAT_CLI" \
    --env "$ENV_NAME" \
    --model "$ASSIST_MODEL" \
    --model-provider openai \
    --user-model "$USER_MODEL" \
    --user-model-provider openai \
    --user-strategy llm \
    --temperature 0.6 \
    --start-index "$START_INDEX" \
    --end-index "$END_INDEX" \
    --max-concurrency 1 \
    --num-trials "$NUM_TRIALS_VAL" \
    --log-dir "$RESULTS_SUBDIR"
  TB_EXIT=$?
fi

#########################################
# CLEANUP
#########################################
echo "Shutting down assistant vLLM (PID: $ASSIST_PID)"
kill "$ASSIST_PID" 2>/dev/null || true
wait "$ASSIST_PID" 2>/dev/null || true

echo "Tau-Bench finished with exit code: $TB_EXIT"
exit "$TB_EXIT"
