#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 10        # number of cores 
#SBATCH --mem=40G   # memory in GB
#SBATCH -G 2   # number of GPUs
#SBATCH -t 0-8:00:00   # time in d-hh:mm:ss
#SBATCH -p gaudi      # partition
#SBATCH -A class_cse59827694spring2026 # class account
#SBATCH -q class_gaudi       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="sdutta60@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# =============================================================================
# PATHS & ENVIRONMENTS — change these for your setup before sharing/running
# =============================================================================
WORK_DIR="/scratch/baspinal/agent-project"                    # Working directory (where you cd before running)
TAU_BENCH_DIR="${WORK_DIR}/agent-project/tau-bench"        # Path to tau-bench repo (run script lives here)
LOG_DIR="${WORK_DIR}/logs"                                 # Directory for vLLM and error-id logs
RESULTS_PATH="${TAU_BENCH_DIR}/../results/num_trials-1.json" # Path to tau-bench results JSON to analyze
VLLM_ENV="gaudi-pytorch-vllm"   # Conda/mamba env for running vLLM server

# =============================================================================

echo "Starting the script..."
cd "$WORK_DIR"

echo "Loading required software..."
#Load required software
module load mamba/latest

# ===== SETUP vLLM ENVIRONMENT =====
# Activate vLLM environment
echo "Activating vLLM environment ($VLLM_ENV)..."
source activate "$VLLM_ENV"

# Set vLLM-specific environment variables
echo "Setting vLLM-specific environment variables..."
export HF_TOKEN=

echo "Setting NO_AI_TRACKING environment variable..."
export NO_AI_TRACKING=true

# ===== USER-CONTROLLED VARIABLES =====
# Model for error identification analysis
MODEL="Qwen/Qwen3-32B"
VLLM_PORT=8690             # Port for vLLM server

# Error identification parameters
MAX_CONCURRENCY=2           # Max concurrent API calls
MAX_NUM_FAILED_RESULTS=""   # Limit number of failed results to analyze (optional)
PLATFORM="vllm-chat"        # Platform: openai, mistral, anthropic, anyscale, outlines, vllm-chat, vllm-completion
ERROR_ID_SCRIPT="auto_error_identification_cached.py"   # Script to run (must be in TAU_BENCH_DIR)

# Allow override via environment variables
RESULTS_PATH="${RESULTS_PATH_INPUT:-$RESULTS_PATH}"
MAX_NUM_FAILED_RESULTS="${MAX_NUM_FAILED_RESULTS_INPUT:-$MAX_NUM_FAILED_RESULTS}"

# Generate output path from results path if not provided
if [ -z "$OUTPUT_PATH_INPUT" ] && [ -n "$RESULTS_PATH" ]; then
    # Extract directory and filename from results path
    RESULTS_DIR=$(dirname "$RESULTS_PATH")
    RESULTS_FILENAME=$(basename "$RESULTS_PATH" .json)
    # Create output filename with error_analysis prefix (errors-analysis under same dir as results)
    OUTPUT_DIR="${RESULTS_DIR}/errors-analysis"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_PATH="${OUTPUT_DIR}/error_analysis_${RESULTS_FILENAME}.json"
else
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_PATH_INPUT}}"
fi

# ===== LOG FILE NAMING =====
MODEL_SAFE="${MODEL//\//_}"  # Replace / with _ for filename
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
VLLM_LOG="${LOG_DIR}/vllm_error_id_${MODEL_SAFE}_${TIMESTAMP}.log"
ERROR_ID_LOG="${LOG_DIR}/error_identification_${ENV}_${MODEL_SAFE}_${TIMESTAMP}.log"

echo "Starting vLLM server for error identification:"
echo "  Model         : $MODEL (port $VLLM_PORT)"
echo "  Environment   : $ENV"
echo "  Platform      : $PLATFORM"
echo "  Results path  : $RESULTS_PATH"
echo "  Output path   : $OUTPUT_PATH"
echo "  Max concurrency: $MAX_CONCURRENCY"
echo "vLLM log        : $VLLM_LOG"
echo "Error ID log    : $ERROR_ID_LOG"

# Start vLLM server
echo "Starting vLLM server..."
vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 2 \
    --download_dir "$HF_DOWNLOAD_DIR" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --enforce-eager \
    --gpu-memory-utilization 0.75 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to start..."
echo "Monitoring vLLM log: $VLLM_LOG"

VLLM_READY=false
for i in {1..120}; do
    # Check if process is still running
    if ! ps -p $VLLM_PID > /dev/null 2>&1; then
        echo "❌ vLLM process (PID $VLLM_PID) has crashed!"
        echo "Last 100 lines of vLLM log:"
        tail -100 "$VLLM_LOG" || echo "Could not read log file"
        exit 1
    fi
    
    # Check for health endpoint
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "✅ vLLM server is ready!"
        VLLM_READY=true
        break
    fi
    
    # Check for common errors in log
    if grep -q "ERROR\|Exception\|Traceback\|Failed\|OOM\|out of memory\|CUDA.*out of memory" "$VLLM_LOG" 2>/dev/null; then
        echo "⚠️  Errors detected in vLLM log. Last 50 lines:"
        tail -50 "$VLLM_LOG" || echo "Could not read log file"
        # Check if it's a fatal error
        if grep -q "EngineCore failed to start\|WorkerProc initialization failed\|RuntimeError.*Engine core" "$VLLM_LOG" 2>/dev/null; then
            echo "❌ Fatal error detected. Stopping."
            exit 1
        fi
    fi
    
    if [ $i -eq 120 ]; then
        echo "❌ vLLM server did not start after 10 minutes"
        echo "Last 100 lines of vLLM log:"
        tail -100 "$VLLM_LOG" || echo "Could not read log file"
        exit 1
    fi
    
    echo "Waiting for vLLM server... ($i/120)"
    sleep 5
done

if [ "$VLLM_READY" = true ]; then
    echo "✅ vLLM server is ready!"
fi

# ===== SWITCH TO TAU-BENCH ENVIRONMENT =====
# Deactivate vLLM env and activate tau-bench env
echo "Switching to tau-bench environment ($TAU_BENCH_ENV)..."
source deactivate
source activate "$TAU_BENCH_ENV"

# Change to tau-bench directory
cd "$TAU_BENCH_DIR"

# Verify vLLM server is still running and healthy
echo "Verifying vLLM server health before starting error identification..."
VLLM_HEALTHY=false

for i in {1..10}; do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "✅ vLLM server is healthy!"
        VLLM_HEALTHY=true
        break
    else
        echo "⚠️  vLLM server health check failed ($i/10)"
        if [ $i -eq 10 ]; then
            echo "❌ vLLM server is not responding! Check logs: $VLLM_LOG"
            if ps -p $VLLM_PID > /dev/null 2>&1; then
                echo "vLLM process (PID $VLLM_PID) is still running but not responding"
            else
                echo "vLLM process (PID $VLLM_PID) has crashed!"
            fi
            exit 1
        fi
    fi
    sleep 5
done

# Validate required arguments
if [ -z "$RESULTS_PATH" ]; then
    echo "❌ ERROR: RESULTS_PATH is required. Set it in the script or via RESULTS_PATH_INPUT env var."
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "❌ ERROR: OUTPUT_PATH is required. Set it in the script or via OUTPUT_PATH_INPUT env var."
    exit 1
fi

if [ ! -f "$RESULTS_PATH" ]; then
    echo "❌ ERROR: Results file not found: $RESULTS_PATH"
    exit 1
fi

# Run error identification
echo "Running error identification..."
echo "Logging output to: $ERROR_ID_LOG"

export OPENAI_BASE_URL=http://localhost:${VLLM_PORT}/v1

# Build command
CMD="python $ERROR_ID_SCRIPT \
    --platform $PLATFORM \
    --base-url http://localhost:${VLLM_PORT}/v1 \
    --env $ENV \
    --results-path \"$RESULTS_PATH\" \
    --output-path \"$OUTPUT_PATH\" \
    --max-concurrency $MAX_CONCURRENCY"

# Add optional max-num-failed-results if provided
if [ -n "$MAX_NUM_FAILED_RESULTS" ]; then
    CMD="$CMD --max-num-failed-results $MAX_NUM_FAILED_RESULTS"
fi

echo "Command: $CMD"
echo ""

# Execute command
eval $CMD > "$ERROR_ID_LOG" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Error identification completed successfully!"
    echo "📄 Output saved to: $OUTPUT_PATH"
    echo "📄 Log saved to: $ERROR_ID_LOG"
else
    echo ""
    echo "❌ Error identification failed with exit code: $EXIT_CODE"
    echo "📄 Check log file: $ERROR_ID_LOG"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$ERROR_ID_LOG"
    exit $EXIT_CODE
fi

# Keep vLLM server running (wait for background process)
# This keeps the job alive so vLLM server doesn't get killed
echo "Waiting for vLLM server to finish..."
echo "vLLM PID: $VLLM_PID"
# wait $VLLM_PID


