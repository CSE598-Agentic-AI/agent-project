#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 20        # number of cores 
#SBATCH --mem=200G   # memory in GB
#SBATCH -G 1   # number of GPUs (agent only; user model via API)
#SBATCH -t 0-10:00:00   # time in d-hh:mm:ss
#SBATCH -p gaudi      # partition
#SBATCH -A class_cse59827694spring2026 # class account
#SBATCH -q class_gaudi       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="sdutta60@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

echo "Starting the script..."
cd /scratch/sdutta60/agent-project/tau-bench


echo "Loading required software..."
#Load required software
module load mamba/latest
pip install -e .

# echo "Loading CUDA..."
# module load cuda-12.6.1-gcc-12.1.0

# mamba create -n vllm-scratch-v1 -c conda-forge python=3.11


  
# ===== SETUP vLLM ENVIRONMENT =====
# Activate vLLM environment
echo "Activating vLLM environment..."
source activate gaudi-pytorch-vllm

# Set vLLM-specific environment variables
echo "Setting vLLM-specific environment variables..."
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0
# HF_TOKEN must be set in the environment (e.g. export before sbatch) for gated models
export HF_TOKEN="${HF_TOKEN:-}"

echo "Setting NO_AI_TRACKING environment variable..."
export NO_AI_TRACKING=true

# Start vLLM server in background (runs in tau-v3-env)
echo "Starting vLLM server in tau-v3-env..."
# ===== USER MODEL API (like tau-multi-api-v1.sbatch) =====
export USER_MODEL_API_BASE="${USER_MODEL_API_BASE:-https://openai.rc.asu.edu/v1}"
export USER_MODEL_API_KEY="${USER_MODEL_API_KEY:-}"

# ===== USER-CONTROLLED VARIABLES =====
# Agent: local vLLM
AGENT_MODEL="Qwen/Qwen3-4B"
AGENT_VLLM_PORT=8690

# User: remote API (set USER_MODEL to the API model name)
USER_MODEL="${USER_MODEL:-qwen3-coder-30b-a3b-instruct}"

DOMAIN="airline"
AGENT_STRATEGY="multi-agent-v1"
NUM_TRIALS=1

# ===== LOG FILE NAMING =====
AGENT_MODEL_SAFE="${AGENT_MODEL//\//_}"
USER_MODEL_SAFE="${USER_MODEL//\//_}"
LOGDIR="/scratch/sdutta60/vllm-test/logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
AGENT_VLLM_LOG="${LOGDIR}/vllm_agent_${DOMAIN}_${AGENT_STRATEGY}_${AGENT_MODEL_SAFE}_trials${NUM_TRIALS}_${TIMESTAMP}.log"
TAUBENCH_LOG="${LOGDIR}/taubench_${DOMAIN}_${AGENT_STRATEGY}_agent${AGENT_MODEL_SAFE}_user${USER_MODEL_SAFE}_trials${NUM_TRIALS}_${TIMESTAMP}.log"

echo "Agent (vLLM) + User (API) with:"
echo "  Agent Model   : $AGENT_MODEL (port $AGENT_VLLM_PORT)"
echo "  User Model   : $USER_MODEL (API: $USER_MODEL_API_BASE)"
echo "  Environment  : $DOMAIN"
echo "  Agent strat  : $AGENT_STRATEGY"
echo "  Num. trials  : $NUM_TRIALS"
echo "Agent vLLM log : $AGENT_VLLM_LOG"
echo "tau-bench log  : $TAUBENCH_LOG"

# Start Agent Model vLLM server
echo "Starting Agent Model vLLM server..."
vllm serve "$AGENT_MODEL" \
    --port "$AGENT_VLLM_PORT" \
    --tensor-parallel-size 1 \
    --download_dir /scratch/sdutta60/huggingface_models/vllm \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --enforce-eager \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    > "$AGENT_VLLM_LOG" 2>&1 &

AGENT_VLLM_PID=$!
echo "Agent vLLM server started with PID: $AGENT_VLLM_PID"

# Wait for agent vLLM server (user model uses API, no second server)
echo "Waiting for Agent vLLM server to start..."
echo "Monitoring Agent vLLM log: $AGENT_VLLM_LOG"

# Wait for Agent vLLM server
AGENT_READY=false
for i in {1..120}; do
    # Check if process is still running
    if ! ps -p $AGENT_VLLM_PID > /dev/null 2>&1; then
        echo "❌ Agent vLLM process (PID $AGENT_VLLM_PID) has crashed!"
        echo "Last 100 lines of Agent vLLM log:"
        tail -100 "$AGENT_VLLM_LOG" || echo "Could not read log file"
        exit 1
    fi
    
    # Check for health endpoint
    if curl -s http://localhost:${AGENT_VLLM_PORT}/health > /dev/null 2>&1; then
        echo "✅ Agent vLLM server is ready!"
        AGENT_READY=true
        break
    fi
    
    # Check for common errors in log
    if grep -q "ERROR\|Exception\|Traceback\|Failed\|OOM\|out of memory\|CUDA.*out of memory" "$AGENT_VLLM_LOG" 2>/dev/null; then
        echo "⚠️  Errors detected in Agent vLLM log. Last 50 lines:"
        tail -50 "$AGENT_VLLM_LOG" || echo "Could not read log file"
        # Check if it's a fatal error
        if grep -q "EngineCore failed to start\|WorkerProc initialization failed\|RuntimeError.*Engine core" "$AGENT_VLLM_LOG" 2>/dev/null; then
            echo "❌ Fatal error detected in Agent. Stopping."
            exit 1
        fi
    fi
    
    if [ $i -eq 120 ]; then
        echo "❌ Agent vLLM server did not start after 10 minutes"
        echo "Last 100 lines of Agent vLLM log:"
        tail -100 "$AGENT_VLLM_LOG" || echo "Could not read log file"
        exit 1
    fi
    
    echo "Waiting for Agent server... ($i/120)"
    sleep 5
done

if [ "$AGENT_READY" = true ]; then
    echo "✅ Agent vLLM server is ready!"
fi

# ===== SWITCH TO TAU-BENCH ENVIRONMENT =====
# Deactivate vLLM env and activate tau-bench env
echo "Switching to tau-bench environment..."
source deactivate
source activate tau-scratch-v1  # Change to your actual tau-bench env name

# Set environment variables for tau-bench (agent=local vLLM, user=API)
export OPENAI_API_BASE=http://localhost:${AGENT_VLLM_PORT}/v1
export OPENAI_API_KEY=sk-dummy
export AGENT_MODEL_API_BASE=http://localhost:${AGENT_VLLM_PORT}/v1
export AGENT_MODEL_API_KEY=sk-dummy
# USER_MODEL_API_BASE and USER_MODEL_API_KEY already set above for remote user API

cd /scratch/sdutta60/agent-project/tau-bench

# Verify agent vLLM is still healthy
echo "Verifying Agent vLLM health before starting tau-bench..."
for i in {1..10}; do
    if curl -s http://localhost:${AGENT_VLLM_PORT}/health > /dev/null 2>&1; then
        echo "✅ Agent vLLM server is healthy!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Agent vLLM server is not responding! Check logs: $AGENT_VLLM_LOG"
        exit 1
    fi
    sleep 5
done

# Run tau-bench
echo "Running tau-bench..."
echo "Logging tau-bench output to: $TAUBENCH_LOG"
python run.py \
    --num-trials $NUM_TRIALS \
    --agent-strategy $AGENT_STRATEGY \
    --env $DOMAIN \
    --model "$AGENT_MODEL" \
    --model-provider openai \
    --user-model "$USER_MODEL" \
    --user-model-provider openai \
    --user-strategy llm \
    --temperature 0.8 \
    --max-concurrency 2 \
    > "$TAUBENCH_LOG" 2>&1

echo "Agent vLLM PID: $AGENT_VLLM_PID (user model via API)"

