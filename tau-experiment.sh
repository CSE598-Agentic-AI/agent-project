#!/bin/bash
#SBATCH -J tau_experiment
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -A class_cse59827694spring2026
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 10
#SBATCH --mem=40G
#SBATCH -t 8:00:00
#SBATCH --mail-type=ALL                # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="baspinal@asu.edu"
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e errors/%x_%A_%a.err

set -euo pipefail

#########################################
# ROOT DIR (this script is designed for)
#########################################
ROOT_DIR="/scratch/baspinal/agent-project"
LOG_DIR="${ROOT_DIR}/logs"
ERROR_DIR="${ROOT_DIR}/errors"
BASE_RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${LOG_DIR}" "${ERROR_DIR}" "${BASE_RESULTS_DIR}"
cd "${ROOT_DIR}"

#########################################
# REQUIRE USER_MODEL_API_BASE (remote)
#########################################
if [ -z "${USER_MODEL_API_BASE:-}" ]; then
  cat <<EOF
ERROR: USER_MODEL_API_BASE is not set.

You must start the USER vLLM server in a separate job, then set:

  export USER_MODEL_API_BASE="http://<user-node>:8007/v1"

before submitting this script.

EOF
  exit 1
fi

echo "Using USER_MODEL_API_BASE=${USER_MODEL_API_BASE}"
echo

#########################################
# SENTINEL for multi-agent (LLM Sentinel)
#########################################
# For multi-agent: if SENTINEL_MODEL_API_BASE is unset, we default to the Assistant
# server (same model) after it starts. To use a separate Sentinel server instead,
# set: export SENTINEL_MODEL_API_BASE="http://<node>:8008/v1"

#########################################
# EXPERIMENT GRID (for array mode)
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

NUM_ENVS=${#ENVS[@]}       # 2
NUM_AGENTS=${#AGENTS[@]}  # 3
NUM_MODELS=${#MODELS[@]}  # 4
NUM_TRIALS=${#TRIALS[@]}  # 5
TOTAL=$((NUM_ENVS * NUM_AGENTS * NUM_MODELS * NUM_TRIALS))  # 160

#########################################
# ARGUMENT / ARRAY HANDLING
#########################################
# Modes:
#   1) Single run (CLI): sbatch tau-experiment.sh [--start-index N] [--end-index M] <env> <agent> <assist_model> [num_trials]
#   2) Array mode:      sbatch --array=0-119 tau-experiment.sh [--start-index N] [--end-index M]
#########################################

# Parse optional --start-index and --end-index (defaults: 0, -1)
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
  # ----- Mode 1: direct arguments (single experiment) -----
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
  # ----- Mode 2: array mode, decode from SLURM_ARRAY_TASK_ID -----
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    cat <<EOF
Usage:
  Single run:
    sbatch tau-experiment.sh [--start-index N] [--end-index M] <env: retail|airline> <agent: act|react|fc|multi-agent> <assistant_model_id> [num_trials]

  Full sweep (job array, 160 experiments):
    sbatch --array=0-$((TOTAL-1)) tau-experiment.sh [--start-index N] [--end-index M]
EOF
    exit 1
  fi

  if [[ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]]; then
    echo "Invalid array index ${SLURM_ARRAY_TASK_ID} (max $((TOTAL-1)))"
    exit 1
  fi

  IDX=$SLURM_ARRAY_TASK_ID

  # index over TRIALS (fastest varying)
  TRIAL_IDX=$((IDX % NUM_TRIALS))
  IDX=$((IDX / NUM_TRIALS))

  # then MODELS
  MODEL_IDX=$((IDX % NUM_MODELS))
  IDX=$((IDX / NUM_MODELS))

  # then AGENTS
  AGENT_IDX=$((IDX % NUM_AGENTS))
  IDX=$((IDX / NUM_AGENTS))

  # then ENVS
  ENV_IDX=$((IDX % NUM_ENVS))

  ENV_NAME="${ENVS[$ENV_IDX]}"
  AGENT_STRAT_INPUT="${AGENTS[$AGENT_IDX]}"
  ASSIST_MODEL="${MODELS[$MODEL_IDX]}"
  NUM_TRIALS_VAL="${TRIALS[$TRIAL_IDX]}"
fi

#########################################
# Map AGENT_STRAT_INPUT -> Tau-Bench CLI
#########################################
case "$AGENT_STRAT_INPUT" in
  act)
    AGENT_STRAT_CLI="act"
    ;;
  react)
    AGENT_STRAT_CLI="react"
    ;;
  fc)
    AGENT_STRAT_CLI="tool-calling"
    ;;
  multi-agent)
    AGENT_STRAT_CLI="multi-agent"
    ;;
  *)
    echo "Error: agent strategy must be one of: act, react, fc, multi-agent (got '$AGENT_STRAT_INPUT')"
    exit 1
    ;;
esac

USER_MODEL="Qwen/Qwen3-32B"  # logical name for tau-bench; served remotely
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
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Environment:         $ENV_NAME"
echo "Agent strategy:      $AGENT_STRAT_INPUT (CLI: $AGENT_STRAT_CLI)"
echo "Assistant model:     $ASSIST_MODEL"
echo "User model (fixed):  $USER_MODEL (remote)"
echo "Num trials:          $NUM_TRIALS_VAL"
echo "Start index:         $START_INDEX"
echo "End index:           $END_INDEX"
echo "Model size bucket:   $MODEL_SIZE"
echo "========================================"
echo

#########################################
# RESULTS DIRS (mirror log structure)
# results/env/agent/modelsize/<assist_safe>_trialsN
#########################################
RESULTS_SUBDIR="${BASE_RESULTS_DIR}/${ENV_NAME}/${AGENT_STRAT_INPUT}/${MODEL_SIZE}"
mkdir -p "${RESULTS_SUBDIR}"


echo "Results directory:   $RESULTS_SUBDIR"
echo

#########################################
# LOG DIR STRUCTURE: logs/env/agent/modelsize
#########################################
LOG_SUBDIR="${LOG_DIR}/${ENV_NAME}/${AGENT_STRAT_INPUT}/${MODEL_SIZE}"
mkdir -p "${LOG_SUBDIR}"

echo "Log directory:       ${LOG_SUBDIR}"
echo

#########################################
# ACTIVATE ENV (Gaudi vLLM / PyTorch)
#########################################
module load mamba/latest
source activate gaudi-pytorch-vllm
export NO_AI_TRACKING=true
export VLLM_BUILD="0.0.0.0"

#########################################
# Quick health check on USER server
#########################################
echo "Checking USER model endpoint health..."
if ! curl -s "${USER_MODEL_API_BASE}/models" > /dev/null; then
  echo "ERROR: Could not reach USER_MODEL_API_BASE=${USER_MODEL_API_BASE}/models"
  echo "Make sure the user-vllm job is running and URL is correct."
  exit 1
fi
echo "USER endpoint is reachable."
echo

#########################################
# Quick health check on SENTINEL (only when using explicit remote Sentinel)
#########################################
if [[ "$AGENT_STRAT_INPUT" == "multi-agent" ]] && [ -n "${SENTINEL_MODEL_API_BASE:-}" ]; then
  echo "Checking remote SENTINEL model endpoint health..."
  if ! curl -s "${SENTINEL_MODEL_API_BASE}/models" > /dev/null; then
    echo "ERROR: Could not reach SENTINEL_MODEL_API_BASE=${SENTINEL_MODEL_API_BASE}/models"
    exit 1
  fi
  echo "Remote SENTINEL endpoint is reachable."
  echo
fi

#########################################
# START ASSISTANT SERVER (LOCAL)
#########################################

echo "Starting ASSISTANT vLLM on port 8005 (node: $(hostname))..."
ASSIST_LOG="${LOG_SUBDIR}/assistant-num_trials${NUM_TRIALS_VAL}-job${SLURM_JOB_ID}.log"
./assistant-server.sh "$ASSIST_MODEL" 8005 "$AGENT_STRAT_CLI" \
  > "${ASSIST_LOG}" 2>&1 &
ASSIST_PID=$!

echo "Assistant PID: $ASSIST_PID"
echo "Assistant log: ${ASSIST_LOG}"
echo

#########################################
# HELPER: wait until vLLM server is ready
#########################################
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

#########################################
# WAIT FOR ASSISTANT TO BE READY
#########################################
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
export OPENAI_API_BASE="${ASSIST_BASE}"         # assistant (local)
export OPENAI_API_KEY="EMPTY"

# For multi-agent: use same model for Sentinel (Assistant server) unless explicitly overridden
if [[ "$AGENT_STRAT_INPUT" == "multi-agent" ]] && [ -z "${SENTINEL_MODEL_API_BASE:-}" ]; then
  export SENTINEL_MODEL_API_BASE="${ASSIST_BASE}"
  echo "LLM Sentinel using same model as Assistant (${ASSIST_BASE})"
fi

#########################################
# RUN τ-BENCH
#########################################
cd "${ROOT_DIR}/tau-bench"

echo "Running Tau-Bench..."
echo


# Total tasks (test split): retail=115, airline=50 (zero-indexed: 0..114, 0..49)
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
  --log-dir "$RESULTS_SUBDIR" \

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
