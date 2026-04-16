# LLM-JEPA

Joint Embedding Predictive Architecture and Semantic Tube Prediction for LLMs.

## Setup

```bash
conda env create -f environment.yml
conda activate jepa
pip install -e .
```

Or install deps manually per `setup.sh` (read it first, don't run it directly).

## Project Structure

- `train.py` — training entry point
- `eval.py` — evaluation entry point
- `llm_jepa/` — main package
  - `config.py` — Dynaconf-based config loading
  - `data.py` — dataset loading/preparation
  - `models.py` — model and tokenizer setup
  - `losses.py` — loss functions
  - `callbacks.py` — training callbacks
  - `utils.py` — shared utilities
  - `strategies/` — training strategies: `regular`, `stp`, `jepa`
  - `evaluation/` — evaluation logic
- `configs/` — TOML configuration (Dynaconf)
  - `default.toml` — base defaults (always loaded first)
  - `strategies/` — strategy-specific overrides
  - `models/` — model-specific configs
  - `experiments/` — full experiment configs
- `datasets/` — JSONL train/test data files
- `scripts/` — shell scripts for running experiments and Slurm jobs

## Running

```bash
# Train
python train.py --config configs/experiments/exp1_synth_finetune.toml

# Override settings on the fly
python train.py --config configs/strategies/stp.toml --set data.train_file=datasets/synth_train.jsonl

# Evaluate
python eval.py --config configs/experiments/exp1_synth_finetune.toml \
    --set evaluation.model_name=./output/model \
    --set data.eval_file=datasets/synth_test.jsonl
```

## Configuration

Uses Dynaconf. `configs/default.toml` is always loaded first, then any configs passed via `--config` (in order). CLI overrides via `--set key=value`. Env vars prefixed with `LLM_JEPA_` also work.

## Training Strategies

- `regular` — standard causal LM fine-tuning
- `stp` — Semantic Tube Prediction
- `jepa` — Joint Embedding Predictive Architecture

Set via `strategy.name` in config or `--set strategy.name=stp`.

## Key Dependencies

- PyTorch >= 2.0
- Transformers >= 4.40
- Dynaconf >= 3.2
- PEFT (LoRA support)
- CUDA 12.8 (conda) or matching your driver
