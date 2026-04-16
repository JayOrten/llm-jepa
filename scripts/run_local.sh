#!/bin/bash
# =============================================================================
# Local training: Regular vs STP on translation (single GPU, ~3-4 hours total)
#
# Prerequisites:
#   ./scripts/prepare_all.sh    # download OPUS data + cache tokenizer
# =============================================================================

set -e

CONFIG=configs/experiments/translation_local.toml

# --- Step 0: Data ---
if [ ! -f datasets/opus_train.jsonl ]; then
    echo "Run ./scripts/prepare_all.sh first to download data."
    exit 1
fi

# --- Step 1: Train regular baseline ---
echo "=== Step 1/4: Training regular baseline ==="
python train.py --config $CONFIG \
    --set strategy.name=regular \
    --set training.output_dir=./output-local-regular

echo ""

# --- Step 2: Evaluate regular baseline ---
echo "=== Step 2/4: Evaluating regular baseline ==="
python eval.py --config $CONFIG \
    --set evaluation.model_name=./output-local-regular

echo ""

# --- Step 3: Train STP ---
echo "=== Step 3/4: Training STP ==="
python train.py --config $CONFIG \
    --set strategy.name=stp \
    --set training.output_dir=./output-local-stp

echo ""

# --- Step 4: Evaluate STP ---
echo "=== Step 4/4: Evaluating STP ==="
python eval.py --config $CONFIG \
    --set evaluation.model_name=./output-local-stp

echo ""
echo "=== Done ==="
echo ""
echo "Compare results:"
echo "  cat output-local-regular/eval_output_summary.json"
echo "  cat output-local-stp/eval_output_summary.json"
echo ""
echo "Training curves:"
echo "  output-local-regular/training_log.csv"
echo "  output-local-stp/training_log.csv"
