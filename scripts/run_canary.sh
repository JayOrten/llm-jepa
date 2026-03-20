#!/bin/bash
# =============================================================================
# Canary Test: Translation with Tiny Model (~37M params)
# =============================================================================
#
# Runs the full train+eval pipeline on a small model and data subset to verify
# everything works before scaling up. Compares regular fine-tuning vs STP.
#
# What this does:
#   1. Download OPUS-100 data (if not already present)
#   2. Train regular baseline on 100k examples (1 epoch)
#   3. Evaluate regular baseline (BLEU + chrF on 500 examples)
#   4. Train STP model on 100k examples (1 epoch)
#   5. Evaluate STP model
#
# Model:     configs/models/llama-tiny (37M param Llama, random init)
# Tokenizer: meta-llama/Llama-3.2-1B-Instruct
# Data:      OPUS-100, 9 language pairs, en->X translation
# Format:    plain mode — "[en] source<|perception|>[xx] target<|eot_id|>"
#
# Outputs:
#   output-canary-regular/
#     training_log.csv      — step, epoch, loss, learning_rate
#     eval_scores.csv       — periodic BLEU/chrF during training
#     train.log             — full stdout/stderr capture
#   output-canary-stp/
#     training_log.csv      — includes lm_loss + aux_loss columns
#     eval_scores.csv
#     train.log
#   eval_canary_regular.jsonl  — per-example generation results
#   eval_canary_stp.jsonl
#
# Typical runtime: ~10-20 min on a single GPU (model is tiny)
#
# Prerequisites:
#   pip install -e .          # install llm-jepa + dependencies (includes sacrebleu)
#   # Llama tokenizer requires HF auth:
#   huggingface-cli login     # or set HF_TOKEN env var
# =============================================================================

set -e

CONFIG=configs/experiments/translation_canary.toml

# --- Step 0: Prepare data (skip if already exists) ---
if [ ! -f datasets/opus_train.jsonl ]; then
    echo "=== Downloading OPUS-100 data ==="
    python scripts/prepare_opus.py
    echo ""
fi

# --- Step 1: Train regular baseline ---
echo "=== Step 1/4: Training regular baseline ==="
python train.py --config $CONFIG \
    --set strategy.name=regular \
    --set training.output_dir=./output-canary-regular

echo ""

# --- Step 2: Evaluate regular baseline ---
echo "=== Step 2/4: Evaluating regular baseline ==="
python eval.py --config $CONFIG \
    --set evaluation.model_name=./output-canary-regular

echo ""

# --- Step 3: Train STP ---
echo "=== Step 3/4: Training STP ==="
python train.py --config $CONFIG \
    --set strategy.name=stp \
    --set training.output_dir=./output-canary-stp

echo ""

# --- Step 4: Evaluate STP ---
echo "=== Step 4/4: Evaluating STP ==="
python eval.py --config $CONFIG \
    --set evaluation.model_name=./output-canary-stp

echo ""
echo "=== Canary test complete ==="
echo ""
echo "All outputs are inside each model's directory:"
echo ""
echo "  output-canary-regular/"
echo "    training_log.csv        — loss per step"
echo "    eval_scores.csv         — periodic BLEU/chrF during training"
echo "    eval_output.jsonl       — per-example eval results"
echo "    eval_output_summary.json — corpus-level BLEU/chrF/accuracy"
echo "    train.log               — full stdout capture"
echo ""
echo "  output-canary-stp/"
echo "    (same structure)"
