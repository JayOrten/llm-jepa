"""SAE interpretability analysis: STP vs NTP.

Train Sparse Autoencoders via SAELens on local model checkpoints, then compare
learned features between STP and NTP models.

Functions for: model loading into TransformerLens, SAE training, feature
alignment, temporal activation statistics, and plotting.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from sae_lens import (
    SAE,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.config import LoggingConfig

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_hooked_model(checkpoint_path, device="cuda", dtype=torch.bfloat16):
    """Load a local checkpoint into TransformerLens HookedTransformer.

    Our checkpoints have LoRA already merged, so we load via HuggingFace first
    then convert to HookedTransformer using the base model's architecture config.
    """
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=dtype,
    )
    model = HookedTransformer.from_pretrained_no_processing(
        BASE_MODEL,
        hf_model=hf_model,
        device=device,
        dtype=dtype,
    )
    del hf_model
    return model


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------


def make_sae_config(
    hook_name="blocks.15.hook_resid_post",
    d_in=2048,
    expansion_factor=8,
    k=64,
    training_tokens=10_000_000,
    context_size=256,
    dataset_path="Skylion007/openwebtext",
    lr=3e-4,
    device="cuda",
):
    """Create SAELens training config for BatchTopK SAE."""
    d_sae = d_in * expansion_factor

    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        normalize_activations="expected_average_only_in",
    )

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=BASE_MODEL,
        hook_name=hook_name,
        dataset_path=dataset_path,
        streaming=True,
        is_dataset_tokenized=False,
        context_size=context_size,
        training_tokens=training_tokens,
        train_batch_size_tokens=4096,
        store_batch_size_prompts=16,
        n_batches_in_buffer=32,
        device=device,
        seed=42,
        dtype="float32",
        lr=lr,
        lr_warm_up_steps=500,
        n_checkpoints=2,
        checkpoint_path="analysis/sae_checkpoints",
        logger=LoggingConfig(log_to_wandb=False),
        autocast=True,
        autocast_lm=True,
    )
    return runner_cfg


def train_sae(checkpoint_path, runner_cfg=None, **config_kwargs):
    """Train an SAE on a model checkpoint.

    Args:
        checkpoint_path: path to local model checkpoint
        runner_cfg: pre-built LanguageModelSAERunnerConfig, or None to create one
        **config_kwargs: passed to make_sae_config if runner_cfg is None

    Returns:
        trained SAE object
    """
    if runner_cfg is None:
        runner_cfg = make_sae_config(**config_kwargs)

    model = load_hooked_model(checkpoint_path, device=runner_cfg.device)
    runner = LanguageModelSAETrainingRunner(runner_cfg, override_model=model)
    sae = runner.run()

    # Free the model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sae


def save_sae(sae, path):
    """Save a trained SAE to disk.

    Handles both TrainingSAE (from runner.run()) and inference SAE objects.
    TrainingSAE uses save_inference_model(); inference SAE uses save_model().
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(sae, "save_inference_model"):
        sae.save_inference_model(str(path))
    else:
        sae.save_model(str(path))
    print(f"Saved SAE to {path}")


def load_sae(path, device="cuda"):
    """Load a trained SAE from disk."""
    return SAE.load_from_disk(path=str(path), device=device)


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------


def feature_alignment(sae_a, sae_b):
    """Compute feature alignment between two SAEs using decoder weight cosine similarity.

    For each feature in sae_a, finds the maximum cosine similarity to any
    feature in sae_b (and vice versa).

    Returns:
        dict with:
            a_to_b: (d_sae_a,) max cosine similarity for each feature in A
            b_to_a: (d_sae_b,) max cosine similarity for each feature in B
            similarity_matrix: (d_sae_a, d_sae_b) full cosine similarity matrix
    """
    # Decoder weights: each row is a feature direction
    # SAELens decoder: (d_sae, d_in)
    W_a = sae_a.W_dec.detach().float()  # (d_sae_a, d_in)
    W_b = sae_b.W_dec.detach().float()  # (d_sae_b, d_in)

    # Normalize
    W_a = W_a / W_a.norm(dim=1, keepdim=True)
    W_b = W_b / W_b.norm(dim=1, keepdim=True)

    # Cosine similarity matrix: (d_sae_a, d_sae_b)
    # Do this in chunks to avoid OOM on large dictionaries
    chunk_size = 1024
    n_a = W_a.shape[0]
    max_sim_a_to_b = torch.zeros(n_a)
    similarity_chunks = []

    for i in range(0, n_a, chunk_size):
        chunk = W_a[i : i + chunk_size]
        sim_chunk = chunk @ W_b.T  # (chunk, d_sae_b)
        max_sim_a_to_b[i : i + chunk_size] = sim_chunk.max(dim=1).values
        similarity_chunks.append(sim_chunk.cpu())

    similarity_matrix = torch.cat(similarity_chunks, dim=0)
    max_sim_b_to_a = similarity_matrix.max(dim=0).values

    return {
        "a_to_b": max_sim_a_to_b.numpy(),
        "b_to_a": max_sim_b_to_a.numpy(),
        "similarity_matrix": similarity_matrix.numpy(),
    }


# ---------------------------------------------------------------------------
# SAE quality evaluation
# ---------------------------------------------------------------------------


def sae_eval(
    model,
    sae,
    texts,
    hook_name="blocks.15.hook_resid_post",
    n_batches=4,
    n_batches_dead=32,
    batch_size=8,
    seq_len=128,
):
    """Compute reconstruction quality and sparsity metrics for a trained SAE.

    MSE/explained_variance/l0 use the first n_batches worth of texts (fast).
    dead_frac uses n_batches_dead worth of texts — pass diverse text (e.g. openwebtext)
    or the dead_frac estimate will be inflated since few unique features will fire.

    Args:
        model: HookedTransformer (already loaded)
        sae: trained SAE
        texts: list of strings; needs len >= n_batches_dead * batch_size
        hook_name: hook point to extract activations from
        n_batches: batches for MSE/EV/L0
        n_batches_dead: batches for dead feature count
        batch_size: sequences per batch
        seq_len: tokens per sequence

    Returns:
        dict with explained_variance, l0, dead_frac, mse, activation_var
    """
    needed = n_batches_dead * batch_size
    if len(texts) < needed:
        raise ValueError(f"Need at least {needed} texts for dead_frac (got {len(texts)})")

    all_mse, all_var, all_l0 = [], [], []
    ever_fired = None

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            tokens = model.to_tokens(batch_texts)[:, :seq_len]
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            acts = cache[hook_name].float()
            flat = acts.reshape(-1, acts.shape[-1])

            encoded = sae.encode(flat)

            if batch_idx < n_batches:
                recon = sae.decode(encoded)
                mse = ((flat - recon) ** 2).mean(dim=-1)
                var = flat.var(dim=-1)
                all_mse.append(mse.mean().item())
                all_var.append(var.mean().item())
                all_l0.append((encoded > 0).float().sum(dim=-1).mean().item())

            if batch_idx < n_batches_dead:
                fired = (encoded > 0).any(dim=0)
                ever_fired = fired if ever_fired is None else (ever_fired | fired)

    mean_mse = float(np.mean(all_mse))
    mean_var = float(np.mean(all_var))
    mean_l0 = float(np.mean(all_l0))
    dead_frac = float((~ever_fired).float().mean().item())

    metrics = {
        "explained_variance": 1.0 - mean_mse / mean_var,
        "l0": mean_l0,
        "dead_frac": dead_frac,
        "mse": mean_mse,
        "activation_var": mean_var,
    }

    dead_tokens = n_batches_dead * batch_size * seq_len
    print(
        f"  explained_var={metrics['explained_variance']:.3f}  "
        f"l0={metrics['l0']:.1f}  "
        f"dead={metrics['dead_frac']:.3f} (over {dead_tokens:,} tokens)  "
        f"mse={metrics['mse']:.3f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Temporal activation statistics
# ---------------------------------------------------------------------------


def get_sae_activations(model, sae, tokens, hook_name="blocks.15.hook_resid_post"):
    """Run model on tokens and get SAE feature activations.

    Args:
        model: HookedTransformer
        sae: trained SAE
        tokens: (batch, seq_len) token ids
        hook_name: which hook point to extract from

    Returns:
        feature_acts: (batch, seq_len, d_sae) sparse feature activations
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        activations = cache[hook_name]  # (batch, seq_len, d_in)
        feature_acts = sae.encode(activations.float())  # (batch, seq_len, d_sae)
    return feature_acts


def temporal_stats(feature_acts, threshold=0.0):
    """Compute temporal activation statistics for SAE features.

    Args:
        feature_acts: (batch, seq_len, d_sae) feature activations
        threshold: activation threshold for "active" (default 0 for TopK)

    Returns:
        dict with:
            autocorrelation: (d_sae,) mean corr(a_f(t), a_f(t+1))
            mean_run_length: (d_sae,) mean consecutive active tokens
            activation_rate: (d_sae,) fraction of positions where feature is active
    """
    acts = feature_acts.detach().cpu().float().numpy()  # (B, T, D)
    B, T, D = acts.shape
    binary = (acts > threshold).astype(np.float32)

    # Autocorrelation: corr(a_f(t), a_f(t+1)) per feature, averaged over batch
    autocorrs = np.zeros(D)
    for d in range(D):
        corrs = []
        for b in range(B):
            seq = acts[b, :, d]
            if seq.std() > 1e-8 and len(seq) > 1:
                corr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                if not np.isnan(corr):
                    corrs.append(corr)
        autocorrs[d] = np.mean(corrs) if corrs else 0.0

    # Mean run length per feature
    run_lengths = np.zeros(D)
    for d in range(D):
        all_runs = []
        for b in range(B):
            seq = binary[b, :, d]
            run = 0
            for t in range(T):
                if seq[t] > 0:
                    run += 1
                elif run > 0:
                    all_runs.append(run)
                    run = 0
            if run > 0:
                all_runs.append(run)
        run_lengths[d] = np.mean(all_runs) if all_runs else 0.0

    # Activation rate
    activation_rate = binary.mean(axis=(0, 1))  # (D,)

    return {
        "autocorrelation": autocorrs,
        "mean_run_length": run_lengths,
        "activation_rate": activation_rate,
    }


# ---------------------------------------------------------------------------
# Centered Kernel Alignment (CKA)
# ---------------------------------------------------------------------------


def _linear_cka(X, Y):
    """Linear CKA between two activation matrices.

    Invariant to orthogonal transformation and isotropic scaling — meaning
    if two models represent the same geometry in different coordinate systems,
    CKA still returns 1.0. This is the key property that makes it more suitable
    than cosine similarity of SAE decoder weights for comparing models.

    Uses the cross-covariance formulation to avoid building an n×n Gram matrix.
    Numerically equivalent to the HSIC ratio in Kornblith et al. (2019).

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", NeurIPS 2019. https://arxiv.org/abs/1905.00414

    Args:
        X: (n, d1) float tensor, mean-centered or not (we center internally)
        Y: (n, d2) float tensor

    Returns:
        CKA value in [0, 1]
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtY = X.T @ Y  # (d1, d2)
    XtX = X.T @ X  # (d1, d1)
    YtY = Y.T @ Y  # (d2, d2)
    hsic_xy = XtY.pow(2).sum()
    hsic_xx = XtX.pow(2).sum()
    hsic_yy = YtY.pow(2).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def collect_layer_activations(model, texts, n_layers=16, batch_size=8, seq_len=128):
    """Run model on texts and collect residual stream activations at every layer.

    Args:
        model: HookedTransformer
        texts: list of strings
        n_layers: number of transformer layers
        batch_size: sequences per forward pass
        seq_len: max tokens per sequence

    Returns:
        dict mapping hook_name -> (n_tokens, d_model) float32 CPU tensor
    """
    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_acts = {name: [] for name in hook_names}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = model.to_tokens(batch)[:, :seq_len]
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
            for name in hook_names:
                # (batch, seq_len, d) -> (batch*seq_len, d)
                acts = cache[name].float().cpu()
                layer_acts[name].append(acts.reshape(-1, acts.shape[-1]))

    return {name: torch.cat(chunks, dim=0) for name, chunks in layer_acts.items()}


def layerwise_cka(model_a_path, model_b_path, texts, n_layers=16, batch_size=8, seq_len=128, device="cuda"):
    """Compute CKA at every layer between two models on the same inputs.

    Loads each model separately to avoid holding both in VRAM simultaneously.
    Returns a CKA value per layer — plot this as a profile to see where
    representations diverge.

    Args:
        model_a_path: path to first model checkpoint (e.g. NTP)
        model_b_path: path to second model checkpoint (e.g. STP)
        texts: list of strings (same inputs run through both models)
        n_layers: number of transformer layers to compare
        batch_size: forward pass batch size
        seq_len: max tokens per sequence
        device: torch device

    Returns:
        list of CKA values, one per layer (index = layer number)
    """
    print("Collecting activations for model A...")
    model_a = load_hooked_model(model_a_path, device=device)
    acts_a = collect_layer_activations(model_a, texts, n_layers=n_layers, batch_size=batch_size, seq_len=seq_len)
    del model_a
    torch.cuda.empty_cache()

    print("Collecting activations for model B...")
    model_b = load_hooked_model(model_b_path, device=device)
    acts_b = collect_layer_activations(model_b, texts, n_layers=n_layers, batch_size=batch_size, seq_len=seq_len)
    del model_b
    torch.cuda.empty_cache()

    print("Computing CKA per layer...")
    cka_values = []
    for i in range(n_layers):
        name = f"blocks.{i}.hook_resid_post"
        X = acts_a[name]
        Y = acts_b[name]
        cka = _linear_cka(X, Y)
        print(f"  Layer {i:2d}: CKA = {cka:.6f}")
        cka_values.append(cka)

    return cka_values


def plot_cka_profile(cka_values, label_a="NTP", label_b="STP", save_path=None):
    """Plot layer-wise CKA profile.

    Args:
        cka_values: list of CKA values, one per layer
        label_a: name of first model
        label_b: name of second model
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(cka_values)), cka_values, marker="o", linewidth=2, markersize=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA")
    ax.set_title(f"Layer-wise CKA: {label_a} vs {label_b}")
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="0.9 reference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# LoRA weight diff analysis
# ---------------------------------------------------------------------------

LORA_MODULE_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_weight_diffs(model_a_path, model_b_path, n_layers=16, module_types=LORA_MODULE_TYPES):
    """Compute per-layer, per-module relative weight change between two merged-LoRA models.

    Loads both models simultaneously (two 1B bf16 models ≈ 4GB), subtracts their
    state dicts, and computes the relative Frobenius norm for each LoRA target module.

    Relative norm = ||W_b - W_a||_F / ||W_a||_F — how large the change was
    relative to the original weight magnitude. More interpretable than raw norms
    since MLP matrices are much larger than attention matrices.

    Args:
        model_a_path: path to base model (e.g. NTP)
        model_b_path: path to fine-tuned model (e.g. STP)
        n_layers: number of transformer layers
        module_types: list of module name suffixes to compare

    Returns:
        norms: (n_layers, len(module_types)) numpy array of relative Frobenius norms
        module_types: list of module names (column labels)
    """
    from transformers import AutoModelForCausalLM

    print("Loading models for weight diff analysis...")
    state_a = AutoModelForCausalLM.from_pretrained(model_a_path, dtype=torch.bfloat16).state_dict()
    state_b = AutoModelForCausalLM.from_pretrained(model_b_path, dtype=torch.bfloat16).state_dict()

    norms = np.zeros((n_layers, len(module_types)))

    for layer_idx in range(n_layers):
        for mod_idx, mod_name in enumerate(module_types):
            key = f"model.layers.{layer_idx}.self_attn.{mod_name}.weight"
            if mod_name in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer_idx}.mlp.{mod_name}.weight"

            if key not in state_a:
                continue

            W_a = state_a[key].float()
            W_b = state_b[key].float()
            delta = W_b - W_a
            rel_norm = delta.norm().item() / (W_a.norm().item() + 1e-10)
            norms[layer_idx, mod_idx] = rel_norm

    del state_a, state_b
    return norms, module_types


def lora_diff_rank_structure(model_a_path, model_b_path, layer_idx, mod_name, top_k=32):
    """SVD of the weight diff for one module to inspect its rank structure.

    Since the change was applied via LoRA (rank 16), the diff should have most
    of its energy in the top 16 singular values. This verifies the merge was
    clean and shows what directions in weight space STP was pushing toward.

    Args:
        model_a_path: base model path
        model_b_path: fine-tuned model path
        layer_idx: which layer to inspect
        mod_name: e.g. "q_proj", "gate_proj"
        top_k: how many singular values to return

    Returns:
        singular_values: (top_k,) array
    """
    from transformers import AutoModelForCausalLM

    state_a = AutoModelForCausalLM.from_pretrained(model_a_path, dtype=torch.bfloat16).state_dict()
    state_b = AutoModelForCausalLM.from_pretrained(model_b_path, dtype=torch.bfloat16).state_dict()

    if mod_name in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.layers.{layer_idx}.mlp.{mod_name}.weight"
    else:
        key = f"model.layers.{layer_idx}.self_attn.{mod_name}.weight"

    delta = (state_b[key].float() - state_a[key].float())
    del state_a, state_b

    _, S, _ = torch.linalg.svd(delta, full_matrices=False)
    return S[:top_k].numpy()


def plot_weight_diff_heatmap(norms, module_types=LORA_MODULE_TYPES, save_path=None):
    """Heatmap of relative weight change per layer and module type.

    Args:
        norms: (n_layers, n_modules) array from lora_weight_diffs()
        module_types: column labels
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(norms, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(module_types)))
    ax.set_xticklabels(module_types, rotation=45, ha="right")
    ax.set_yticks(range(norms.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(norms.shape[0])])
    ax.set_xlabel("Module")
    ax.set_ylabel("Layer")
    ax.set_title("Relative weight change (STP vs NTP): ||ΔW||_F / ||W_NTP||_F")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_singular_values(singular_values, layer_idx, mod_name, lora_rank=16, save_path=None):
    """Plot singular value spectrum of a weight diff to inspect rank structure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(singular_values)), singular_values)
    ax.axvline(lora_rank - 0.5, color="red", linestyle="--", label=f"LoRA rank ({lora_rank})")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"SVD of ΔW — layer {layer_idx}, {mod_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_alignment_histogram(alignment, labels=("NTP→STP", "STP→NTP")):
    """Plot histogram of max cosine similarities between SAE features."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, label in zip(axes, ["a_to_b", "b_to_a"], labels):
        data = alignment[key]
        ax.hist(data, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(
            np.median(data),
            color="red",
            linestyle="--",
            label=f"median={np.median(data):.3f}",
        )
        ax.set_xlabel("Max Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Feature Alignment: {label}")
        ax.legend()
        ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_temporal_comparison(stats_a, stats_b, label_a="NTP", label_b="STP"):
    """Compare temporal activation statistics between two SAEs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Autocorrelation
    ax = axes[0]
    for stats, label, color in [
        (stats_a, label_a, "tab:blue"),
        (stats_b, label_b, "tab:orange"),
    ]:
        data = stats["autocorrelation"]
        data = data[~np.isnan(data) & (data != 0)]
        ax.hist(
            data,
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xlabel("Autocorrelation")
    ax.set_ylabel("Count")
    ax.set_title("Feature Temporal Autocorrelation")
    ax.legend()

    # Run length
    ax = axes[1]
    for stats, label, color in [
        (stats_a, label_a, "tab:blue"),
        (stats_b, label_b, "tab:orange"),
    ]:
        data = stats["mean_run_length"]
        data = data[data > 0]
        ax.hist(
            data,
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xlabel("Mean Run Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Feature Activation Run Length")
    ax.legend()

    # Activation rate
    ax = axes[2]
    for stats, label, color in [
        (stats_a, label_a, "tab:blue"),
        (stats_b, label_b, "tab:orange"),
    ]:
        data = stats["activation_rate"]
        data = data[data > 0]
        ax.hist(
            data,
            bins=50,
            alpha=0.5,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xlabel("Activation Rate")
    ax.set_ylabel("Count")
    ax.set_title("Feature Activation Rate")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_reconstruction_comparison(
    sae_a, sae_b, activations_a, activations_b, label_a="NTP", label_b="STP"
):
    """Compare reconstruction quality between two SAEs.

    Args:
        sae_a, sae_b: trained SAEs
        activations_a, activations_b: (N, d_in) activation tensors from respective models
    """
    with torch.no_grad():
        recon_a = sae_a.decode(sae_a.encode(activations_a.float()))
        recon_b = sae_b.decode(sae_b.encode(activations_b.float()))

        mse_a = ((activations_a.float() - recon_a) ** 2).mean(dim=-1)  # (N,)
        mse_b = ((activations_b.float() - recon_b) ** 2).mean(dim=-1)

        # Normalized: MSE / variance of activations
        var_a = activations_a.float().var(dim=-1)
        var_b = activations_b.float().var(dim=-1)
        nmse_a = mse_a / var_a.clamp(min=1e-8)
        nmse_b = mse_b / var_b.clamp(min=1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(mse_a.cpu().numpy(), bins=50, alpha=0.5, label=label_a, color="tab:blue")
    ax.hist(mse_b.cpu().numpy(), bins=50, alpha=0.5, label=label_b, color="tab:orange")
    ax.set_xlabel("MSE")
    ax.set_title("Reconstruction MSE")
    ax.legend()

    ax = axes[1]
    ax.hist(nmse_a.cpu().numpy(), bins=50, alpha=0.5, label=label_a, color="tab:blue")
    ax.hist(nmse_b.cpu().numpy(), bins=50, alpha=0.5, label=label_b, color="tab:orange")
    ax.set_xlabel("Normalized MSE (MSE / Var)")
    ax.set_title("Normalized Reconstruction Error")
    ax.legend()

    plt.tight_layout()
    return fig
