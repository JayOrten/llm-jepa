"""Smoke test for the SAE pipeline on a tiny built-in model.

Uses TransformerLens's gelu-2l (2 layers, d_model=512) to verify the full
pipeline end-to-end: train → save → load → feature_alignment → temporal_stats.

Run from the analysis/ directory:
    python test_sae_pipeline.py
"""

import sys
import shutil
import tempfile
import torch
import numpy as np

sys.path.insert(0, ".")
from sae_analysis import (
    make_sae_config,
    save_sae,
    load_sae,
    feature_alignment,
    get_sae_activations,
    temporal_stats,
)
from sae_lens import LanguageModelSAETrainingRunner
from transformer_lens import HookedTransformer


DUMMY_MODEL = "gelu-2l"
HOOK_NAME = "blocks.1.hook_resid_post"  # last layer of gelu-2l
D_IN = 512
EXPANSION = 4
K = 16
TRAINING_TOKENS = 50_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_dummy_sae(save_dir, seed=42):
    cfg = make_sae_config(
        hook_name=HOOK_NAME,
        d_in=D_IN,
        expansion_factor=EXPANSION,
        k=K,
        training_tokens=TRAINING_TOKENS,
        context_size=64,
        dataset_path="NeelNanda/pile-10k",
        lr=3e-4,
        device=DEVICE,
    )
    cfg.model_name = DUMMY_MODEL
    cfg.seed = seed
    # Tune down for small test: batch size and norm estimate batches
    cfg.train_batch_size_tokens = 512
    cfg.store_batch_size_prompts = 8
    cfg.n_batches_in_buffer = 8
    cfg.n_batches_for_norm_estimate = 50
    cfg.lr_warm_up_steps = 50

    model = HookedTransformer.from_pretrained_no_processing(DUMMY_MODEL, device=DEVICE)
    runner = LanguageModelSAETrainingRunner(cfg, override_model=model)
    sae = runner.run()
    save_sae(sae, save_dir)
    del model
    return sae


def check_save_load(save_dir):
    sae = load_sae(save_dir, device=DEVICE)
    assert sae.W_dec.shape == (D_IN * EXPANSION, D_IN), (
        f"Unexpected decoder shape: {sae.W_dec.shape}"
    )
    print(f"  save/load OK — W_dec shape: {sae.W_dec.shape}")
    return sae


def check_alignment(sae_a, sae_b):
    result = feature_alignment(sae_a, sae_b)
    assert result["a_to_b"].shape == (D_IN * EXPANSION,)
    assert result["b_to_a"].shape == (D_IN * EXPANSION,)
    assert 0.0 <= result["a_to_b"].min() and result["a_to_b"].max() <= 1.0
    med = np.median(result["a_to_b"])
    print(f"  alignment OK — median max cosine sim: {med:.3f}")


def check_temporal(sae):
    model = HookedTransformer.from_pretrained_no_processing(DUMMY_MODEL, device=DEVICE)
    tokens = model.to_tokens(["Hello world this is a test sentence for the pipeline."] * 4)
    acts = get_sae_activations(model, sae, tokens, hook_name=HOOK_NAME)
    assert acts.shape == (4, tokens.shape[1], D_IN * EXPANSION)

    stats = temporal_stats(acts)
    assert stats["autocorrelation"].shape == (D_IN * EXPANSION,)
    assert stats["mean_run_length"].shape == (D_IN * EXPANSION,)
    print(f"  temporal_stats OK — mean autocorr: {stats['autocorrelation'].mean():.4f}")
    del model


def main():
    tmpdir = tempfile.mkdtemp(prefix="sae_test_")
    save_a = f"{tmpdir}/sae_a"
    save_b = f"{tmpdir}/sae_b"

    try:
        print(f"Training SAE A on {DUMMY_MODEL} for {TRAINING_TOKENS} tokens...")
        train_dummy_sae(save_a, seed=42)
        print("  done.")

        print("Training SAE B (same config, different seed)...")
        train_dummy_sae(save_b, seed=123)
        print("  done.")

        print("Checking save/load...")
        sae_a = check_save_load(save_a)
        sae_b = check_save_load(save_b)

        print("Checking feature alignment...")
        check_alignment(sae_a, sae_b)

        print("Checking temporal stats...")
        check_temporal(sae_a)

        print("\nAll checks passed.")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
