#!/bin/bash
# =============================================================================
# Slurm evaluation script — single GPU, generates + scores BLEU/chrF
#
# Outputs are saved inside the model directory:
#   <model_dir>/eval_output.jsonl         — per-example results
#   <model_dir>/eval_output_summary.json  — corpus-level BLEU/chrF/accuracy
#
# Usage:
#   sbatch scripts/slurm_eval.sh <config> <model_dir> [extra --set args...]
#
# Examples:
#   # Evaluate regular 350M:
#   sbatch scripts/slurm_eval.sh \
#       configs/experiments/translation_350m.toml \
#       ./output-translation-350m-regular
#
#   # Evaluate with limited samples:
#   sbatch scripts/slurm_eval.sh \
#       configs/experiments/translation_canary.toml \
#       ./output-canary-stp \
#       --set data.max_eval_samples=200
# =============================================================================

#SBATCH --job-name=llm-jepa-eval
#SBATCH --output=slurm-logs/%j-eval.out
#SBATCH --error=slurm-logs/%j-eval.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -e

# --- Parse args ---
CONFIG="${1:?Usage: sbatch slurm_eval.sh <config> <model_dir> [extra args...]}"
MODEL_DIR="${2:?Usage: sbatch slurm_eval.sh <config> <model_dir> [extra args...]}"
shift 2
EXTRA_ARGS="$@"

# --- Environment ---
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate jepa

# --- Offline mode (use cached models/tokenizers, no network) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== LLM-JEPA Evaluation ==="
echo "Config: $CONFIG"
echo "Model:  $MODEL_DIR"
echo "Node:   $(hostname)"
echo ""

mkdir -p slurm-logs

python eval.py \
    --config "$CONFIG" \
    --set evaluation.model_name="$MODEL_DIR" \
    $EXTRA_ARGS
