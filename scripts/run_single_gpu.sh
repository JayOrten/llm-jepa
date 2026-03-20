#!/bin/bash
# Single-GPU replication of STP experiment
# Model: meta-llama/Llama-3.2-1B-Instruct
# Dataset: synth (natural language -> regex)
# Comparison: regular fine-tuning vs STP (--linear=random_span)

set -e # Exit on error

CONFIG=configs/experiments/replicate_stp_synth.toml

# === Step 1: Regular fine-tuning baseline ===
python3 train.py --config $CONFIG \
    --set strategy.name=regular \
    --set training.output_dir=./ft-regular-synth-v2

# === Step 2: Evaluate baseline ===
python3 eval.py --config $CONFIG \
    --set evaluation.model_name=./ft-regular-synth-v2 \
    --set data.eval_file=datasets/synth_test.jsonl \
    --set evaluation.output_file=eval_regular.jsonl \
    --set evaluation.split_tune_untune=true |
    tee -a output.txt

# === Step 3: STP fine-tuning ===
python3 train.py --config $CONFIG \
    --set training.output_dir=./ft-stp-synth-v2

# === Step 4: Evaluate STP ===
python3 eval.py --config $CONFIG \
    --set evaluation.model_name=./ft-stp-synth \
    --set data.eval_file=datasets/synth_test.jsonl \
    --set evaluation.output_file=eval_stp.jsonl \
    --set evaluation.split_tune_untune=true |
    tee -a output.txt
