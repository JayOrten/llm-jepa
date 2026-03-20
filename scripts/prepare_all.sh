#!/bin/bash
# =============================================================================
# Pre-download all data and models needed for training.
# Run this BEFORE submitting slurm jobs (compute nodes may lack internet).
#
# Usage:
#   ./scripts/prepare_all.sh
# =============================================================================

set -e

echo "=== Preparing data and models ==="

# --- OPUS-100 dataset ---
if [ -f datasets/opus_train.jsonl ] && [ -f datasets/opus_eval.jsonl ]; then
    echo "OPUS data already exists, skipping download."
else
    echo "Downloading OPUS-100 data..."
    python scripts/prepare_opus.py
fi

TRAIN_COUNT=$(wc -l < datasets/opus_train.jsonl)
EVAL_COUNT=$(wc -l < datasets/opus_eval.jsonl)
echo "  Train: $TRAIN_COUNT examples"
echo "  Eval:  $EVAL_COUNT examples"

# --- Tokenizer (Llama 3.2 1B Instruct) ---
# All configs use this tokenizer. Downloading it caches it in ~/.cache/huggingface.
echo ""
echo "Caching tokenizer: meta-llama/Llama-3.2-1B-Instruct"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"

echo ""
echo "=== All assets ready ==="
echo ""
echo "You can now submit slurm jobs. Example:"
echo "  sbatch scripts/slurm_train.sh configs/experiments/translation_350m.toml regular"
echo "  sbatch scripts/slurm_train.sh configs/experiments/translation_350m.toml stp"
