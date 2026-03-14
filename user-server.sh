#!/bin/bash
#SBATCH -J user_vllm
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -A class_cse59827694spring2026
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 12
#SBATCH --mem=40G
#SBATCH -t 8:00:00
#SBATCH --mail-type=ALL                # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="baspinal@asu.edu"
#SBATCH -o logs/user_vllm_%j.out
#SBATCH -e errors/user_vllm_%j.err

set -euo pipefail

ROOT_DIR="/scratch/baspinal/agent-project"
LOG_DIR="${ROOT_DIR}/logs"
ERROR_DIR="${ROOT_DIR}/errors"
mkdir -p "${LOG_DIR}" "${ERROR_DIR}"
cd "${ROOT_DIR}"
PORT=8007

module load mamba/latest
source activate gaudi-pytorch-vllm


export NO_AI_TRACKING=true
export VLLM_BUILD="0.0.0.0"



USER_MODEL="Qwen/Qwen3-32B"


echo "===================================="
echo "Starting USER vLLM server on node: $(hostname)"
echo "User model:        ${USER_MODEL}"
echo "User server base:  http://$(hostname):${PORT}/v1"
echo "===================================="
echo

echo "DEBUG: SLURM job on node $(hostname)"
echo "DEBUG: SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DEBUG: GRES: ${SLURM_JOB_GRES:-<unset>}"

echo "DEBUG: HABANA_VISIBLE_DEVICES=${HABANA_VISIBLE_DEVICES:-<unset>}"

echo "DEBUG: hl-smi output:"
hl-smi || echo "hl-smi failed (no HPU visible?)"


# IMPORTANT: user-server.sh must NOT re-activate another env
./user-server.sh "$USER_MODEL" "$PORT"
