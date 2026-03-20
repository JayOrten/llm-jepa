#!/bin/bash
# =============================================================================
# Slurm training script — multi-GPU via torchrun
#
# Usage:
#   sbatch scripts/slurm_train.sh <config> <strategy> [extra --set args...]
#
# Examples:
#   # Regular baseline, 350M, all available GPUs:
#   sbatch scripts/slurm_train.sh configs/experiments/translation_350m.toml regular
#
#   # STP, 350M:
#   sbatch scripts/slurm_train.sh configs/experiments/translation_350m.toml stp
#
#   # Canary with custom output dir:
#   sbatch scripts/slurm_train.sh configs/experiments/translation_canary.toml stp \
#       --set training.output_dir=./output-canary-stp
#
#   # Override GPU count:
#   sbatch --gres=gpu:4 scripts/slurm_train.sh configs/experiments/translation_350m.toml regular
#
# Outputs land in the config's training.output_dir (or override via --set):
#   training_log.csv, eval_scores.csv, train.log, model checkpoints
# =============================================================================

#SBATCH --job-name=llm-jepa-train
#SBATCH --output=slurm-logs/%j-train.out
#SBATCH --error=slurm-logs/%j-train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -e

# --- Parse args ---
CONFIG="${1:?Usage: sbatch slurm_train.sh <config> <strategy> [extra args...]}"
STRATEGY="${2:?Usage: sbatch slurm_train.sh <config> <strategy> [extra args...]}"
shift 2
EXTRA_ARGS="$@"

# --- Environment ---
# Adjust these to match your cluster:
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate jepa

# Detect GPU count from slurm allocation
NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPUS=${NGPUS:-1}

echo "=== LLM-JEPA Training ==="
echo "Config:   $CONFIG"
echo "Strategy: $STRATEGY"
echo "GPUs:     $NGPUS"
echo "Node:     $(hostname)"
echo "Extra:    $EXTRA_ARGS"
echo ""

# Create slurm log directory
mkdir -p slurm-logs

# --- Offline mode (use cached models/tokenizers, no network) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- NCCL tuning ---
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NGPUS))

# --- Launch ---
if [ "$NGPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node=$NGPUS \
        --master_port=$(( RANDOM % 10000 + 20000 )) \
        train.py \
        --config "$CONFIG" \
        --set strategy.name="$STRATEGY" \
        $EXTRA_ARGS
else
    python train.py \
        --config "$CONFIG" \
        --set strategy.name="$STRATEGY" \
        $EXTRA_ARGS
fi
