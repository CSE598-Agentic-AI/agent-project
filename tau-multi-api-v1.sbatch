#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 8        # number of cores 
#SBATCH --mem=200G   # memory in GB
#SBATCH -t 0-24:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -A class_cse59827694spring2026 # class account
#SBATCH -q class       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="sdutta60@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

echo "Starting the script..."
cd /scratch/sdutta60/agent-project/

echo "Loading required software..."
module load mamba/latest

echo "Activating tau-bench environment..."
source activate tau-scratch-v1

# ===== API CONFIGURATION (from .env) =====
export OPENAI_API_BASE=https://openai.rc.asu.edu/v1
export OPENAI_API_KEY=sk-xxxx
export AGENT_MODEL_API_BASE=https://openai.rc.asu.edu/v1
export AGENT_MODEL_API_KEY=sk-xxxx
export USER_MODEL_API_BASE=https://openai.rc.asu.edu/v1
export USER_MODEL_API_KEY=sk-xxxx
export NO_AI_TRACKING=true

# ===== USER-CONTROLLED VARIABLES =====
AGENT_MODEL="qwen3-coder-30b-a3b-instruct"
USER_MODEL="qwen3-coder-30b-a3b-instruct"
DOMAIN="airline"
AGENT_STRATEGY="multi-agent-v1"
NUM_TRIALS=5

# ===== LOG FILE NAMING =====
AGENT_MODEL_SAFE="${AGENT_MODEL//\//_}"
USER_MODEL_SAFE="${USER_MODEL//\//_}"
LOGDIR="/scratch/sdutta60/vllm-test/logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TAUBENCH_LOG="${LOGDIR}/taubench_${DOMAIN}_${AGENT_STRATEGY}_agent${AGENT_MODEL_SAFE}_user${USER_MODEL_SAFE}_trials${NUM_TRIALS}_${TIMESTAMP}.log"

echo "Running tau-bench with:"
echo "  Agent Model   : $AGENT_MODEL"
echo "  User Model    : $USER_MODEL"
echo "  Environment   : $DOMAIN"
echo "  Agent strat   : $AGENT_STRATEGY"
echo "  Num. trials   : $NUM_TRIALS"
echo "  API Base      : $OPENAI_API_BASE"
echo "  Log file      : $TAUBENCH_LOG"

cd /scratch/sdutta60/agent-project/tau-bench

echo "Running tau-bench..."
python run.py \
    --num-trials $NUM_TRIALS \
    --agent-strategy $AGENT_STRATEGY \
    --env $DOMAIN \
    --model "$AGENT_MODEL" \
    --model-provider openai \
    --user-model "$USER_MODEL" \
    --user-model-provider openai \
    --user-strategy llm \
    --temperature 0.0 \
    --max-concurrency 2 \
    > "$TAUBENCH_LOG" 2>&1

echo "tau-bench finished. Log: $TAUBENCH_LOG"
echo "Exit code: $?"