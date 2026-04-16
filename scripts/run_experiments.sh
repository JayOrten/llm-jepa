#!/bin/bash
# =============================================================================
# Run all three experiments.
#
# Experiments 1 & 3 run locally (4080 12GB).
# Experiment 2 must be submitted to slurm separately.
#
# Total local time: ~1-2 hours
# =============================================================================

set -e

# --- Experiment 1: Synth fine-tune (replicate paper) ---
echo "============================================"
echo "  Experiment 1: NL-RX-SYNTH Fine-tune"
echo "============================================"

echo ""
echo "--- Exp 1a: Regular fine-tune ---"
# DONE
python train.py --config configs/experiments/exp1_synth_finetune.toml \
    --set strategy.name=regular \
    --set training.output_dir=./output-exp1-regular

echo ""
echo "--- Exp 1a: Evaluate ---"
# DONE
python eval.py --config configs/experiments/exp1_synth_finetune.toml \
    --set evaluation.model_name=./output-exp1-regular

echo ""
echo "--- Exp 1b: STP fine-tune ---"
# DONE
python train.py --config configs/experiments/exp1_synth_finetune.toml \
    --set training.output_dir=./output-exp1-stp

echo ""
echo "--- Exp 1b: Evaluate ---"
# DONE
python eval.py --config configs/experiments/exp1_synth_finetune.toml \
    --set evaluation.model_name=./output-exp1-stp

# --- Experiment 3: Translation fine-tune ---
echo ""
echo "============================================"
echo "  Experiment 3: Translation Fine-tune"
echo "============================================"

echo ""
echo "--- Exp 3a: Regular fine-tune ---"
# IN PROGRESS
python train.py --config configs/experiments/exp3_translation_finetune.toml \
    --set strategy.name=regular \
    --set training.output_dir=./output-exp3-regular

echo ""
echo "--- Exp 3a: Evaluate ---"
python eval.py --config configs/experiments/exp3_translation_finetune.toml \
    --set evaluation.model_name=./output-exp3-regular

echo ""
echo "--- Exp 3b: STP fine-tune ---"
python train.py --config configs/experiments/exp3_translation_finetune.toml \
    --set training.output_dir=./output-exp3-stp

echo ""
echo "--- Exp 3b: Evaluate ---"
python eval.py --config configs/experiments/exp3_translation_finetune.toml \
    --set evaluation.model_name=./output-exp3-stp

# --- Summary ---
echo ""
echo "============================================"
echo "  Results Summary"
echo "============================================"
echo ""
echo "Experiment 1 (NL-RX-SYNTH, fine-tune):"
echo "  Regular: $(cat output-exp1-regular/eval_output_summary.json 2>/dev/null || echo 'not found')"
echo "  STP:     $(cat output-exp1-stp/eval_output_summary.json 2>/dev/null || echo 'not found')"
echo ""
echo "Experiment 3 (Translation, fine-tune):"
echo "  Regular: $(cat output-exp3-regular/eval_output_summary.json 2>/dev/null || echo 'not found')"
echo "  STP:     $(cat output-exp3-stp/eval_output_summary.json 2>/dev/null || echo 'not found')"
echo ""
echo "Experiment 2 (Translation, from-scratch) — submit to slurm:"
echo "  sbatch scripts/slurm_train.sh configs/experiments/exp2_translation_scratch.toml regular"
echo "  sbatch scripts/slurm_train.sh configs/experiments/exp2_translation_scratch.toml stp"
