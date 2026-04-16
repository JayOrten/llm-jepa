"""Phase 1: Quick diagnostics — participation ratio and conditional CKA.

Reuses model loading and CKA from sae_analysis.py. Adds:
- Chat-format data loading (applies tokenizer chat template)
- Per-layer participation ratio
- Conditional CKA on stratified inputs
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

from sae_analysis import load_hooked_model, _linear_cka

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_chat_texts(jsonl_path, tokenizer=None):
    """Load chat-format JSONL and apply the chat template.

    Each line has {"messages": [{"role": ..., "content": ...}, ...]}.
    Returns list of strings with special tokens applied, ready for
    model.to_tokens().

    Args:
        jsonl_path: path to JSONL file
        tokenizer: HF tokenizer with chat template. If None, loads the base model tokenizer.

    Returns:
        texts: list of templated strings
        raw_messages: list of original message dicts (for stratification)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    path = Path(jsonl_path)
    raw_messages = []
    texts = []

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            messages = entry["messages"]
            raw_messages.append(messages)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

    return texts, raw_messages


def stratify_by_target_length(raw_messages, texts, quantiles=(0.33, 0.67)):
    """Split texts into short/medium/long buckets by assistant response length.

    Args:
        raw_messages: list of message dicts from load_chat_texts
        texts: corresponding templated strings
        quantiles: boundaries between buckets

    Returns:
        dict mapping bucket name -> list of texts
    """
    # Get assistant response lengths (character count)
    lengths = []
    for msgs in raw_messages:
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        lengths.append(len(asst[0]) if asst else 0)
    lengths = np.array(lengths)

    thresholds = np.quantile(lengths, quantiles)
    buckets = {"short": [], "medium": [], "long": []}
    for i, (text, length) in enumerate(zip(texts, lengths)):
        if length <= thresholds[0]:
            buckets["short"].append(text)
        elif length <= thresholds[1]:
            buckets["medium"].append(text)
        else:
            buckets["long"].append(text)

    for name, group in buckets.items():
        print(f"  {name}: {len(group)} examples")
    return buckets


# ---------------------------------------------------------------------------
# 1A: Participation Ratio
# ---------------------------------------------------------------------------


def eigenspectrum(acts, top_k=20):
    """Compute the top-k eigenvalues of the activation covariance matrix.

    Useful for diagnosing whether PR≈1 is caused by one dominant eigenvalue
    (genuine "DC component" phenomenon) vs a numerical artifact.

    Args:
        acts: (n_tokens, d_model) float tensor
        top_k: number of eigenvalues to return (largest first)

    Returns:
        eigenvalues: (top_k,) numpy array, descending order
        explained_var_cumulative: (top_k,) fraction of total variance explained
    """
    acts = acts - acts.mean(0)
    # Use SVD of data matrix to get singular values (faster than full eigdecomp)
    # singular values s relate to eigenvalues λ as: λ = s^2 / (n-1)
    _, s, _ = torch.linalg.svd(acts, full_matrices=False)
    eigenvalues = (s[:top_k] ** 2 / (acts.shape[0] - 1)).numpy()
    total_var = (s ** 2).sum().item() / (acts.shape[0] - 1)
    cumvar = np.cumsum(eigenvalues) / total_var
    return eigenvalues, cumvar


def participation_ratio_from_acts(acts):
    """Compute participation ratio from an activation matrix.

    PR = (sum lambda_i)^2 / sum(lambda_i^2)
    where lambda_i are eigenvalues of the covariance matrix.

    Uses SVD of the data matrix (faster than full eigendecomposition for
    tall matrices where n_tokens >> d_model).

    Args:
        acts: (n_tokens, d_model) float tensor

    Returns:
        PR value (float). Range [1, d_model].
    """
    acts = acts - acts.mean(0)
    _, s, _ = torch.linalg.svd(acts, full_matrices=False)
    eigenvalues = s ** 2  # proportional to covariance eigenvalues

    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()

    if sum_eig_sq < 1e-10:
        return 0.0
    return (sum_eig ** 2 / sum_eig_sq).item()


def _tokenize_with_mask(tokenizer, texts, seq_len=128):
    """Tokenize texts and return (input_ids, attention_mask) with proper padding.

    Uses the HF tokenizer directly so we get a real attention mask that
    distinguishes padding from legitimate special tokens (like eot_id which
    appears both in chat templates and as the pad token).

    Args:
        tokenizer: HF tokenizer
        texts: list of strings
        seq_len: max sequence length (truncate + pad to this)

    Returns:
        input_ids: (batch, seq_len) LongTensor
        attention_mask: (batch, seq_len) BoolTensor (True = real token)
    """
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=seq_len,
        return_tensors="pt", add_special_tokens=False,
    )
    ids = encoded["input_ids"][:, :seq_len]
    mask = encoded["attention_mask"][:, :seq_len].bool()
    return ids, mask


def collect_masked_activations(model, texts, n_layers=16,
                                batch_size=8, seq_len=128):
    """Collect residual stream activations with padding tokens removed.

    Uses the HF tokenizer's attention mask to identify real vs padding tokens.
    This avoids the ambiguity of eot_id appearing both as a legitimate
    end-of-turn marker and as the pad token.

    Args:
        model: HookedTransformer
        texts: list of strings
        n_layers: number of transformer layers
        batch_size: forward pass batch size
        seq_len: max tokens per sequence

    Returns:
        dict mapping hook_name -> (n_real_tokens, d_model) float32 CPU tensor
    """
    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_acts = {name: [] for name in hook_names}
    tokenizer = model.tokenizer

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            ids, mask = _tokenize_with_mask(tokenizer, batch, seq_len=seq_len)
            ids = ids.to(model.cfg.device)

            _, cache = model.run_with_cache(ids, names_filter=hook_names)
            flat_mask = mask.reshape(-1)
            for name in hook_names:
                acts = cache[name].float().cpu()
                flat_acts = acts.reshape(-1, acts.shape[-1])
                layer_acts[name].append(flat_acts[flat_mask])

    return {name: torch.cat(chunks, dim=0) for name, chunks in layer_acts.items()}


def layerwise_participation_ratio(model_path, texts, n_layers=16,
                                   batch_size=8, seq_len=128, device="cuda"):
    """Compute participation ratio at every layer for one model.

    Filters out padding tokens before computing PR, since padding activations
    are near-identical and would collapse the covariance to rank ~1.

    Args:
        model_path: path to model checkpoint
        texts: list of strings (already chat-templated)
        n_layers: number of transformer layers
        batch_size: forward pass batch size
        seq_len: max tokens per sequence
        device: torch device

    Returns:
        list of PR values, one per layer
    """
    model = load_hooked_model(model_path, device=device)
    acts = collect_masked_activations(
        model, texts, n_layers=n_layers,
        batch_size=batch_size, seq_len=seq_len,
    )
    n_tokens = acts[f"blocks.0.hook_resid_post"].shape[0]
    print(f"  ({n_tokens} non-padding tokens)")
    del model
    torch.cuda.empty_cache()

    pr_values = []
    for i in range(n_layers):
        name = f"blocks.{i}.hook_resid_post"
        pr = participation_ratio_from_acts(acts[name])
        print(f"  Layer {i:2d}: PR = {pr:.1f}")
        pr_values.append(pr)

    return pr_values


def plot_participation_ratio(pr_dict, save_path=None):
    """Plot per-layer participation ratio for multiple models.

    Args:
        pr_dict: dict mapping model_name -> list of PR values per layer
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, pr_values in pr_dict.items():
        ax.plot(range(len(pr_values)), pr_values, marker="o",
                linewidth=2, markersize=5, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Effective Dimensionality (Participation Ratio) per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 1B: Conditional CKA
# ---------------------------------------------------------------------------


def _collect_grouped_masked(model, text_groups, hook_names,
                             batch_size=8, seq_len=128):
    """Collect activations for multiple text groups, filtering padding.

    Args:
        model: HookedTransformer
        text_groups: dict mapping group_name -> list of texts
        hook_names: list of hook names to collect
        batch_size: forward pass batch size
        seq_len: max tokens per sequence

    Returns:
        dict mapping group_name -> {hook_name: (n_real_tokens, d) tensor}
    """
    tokenizer = model.tokenizer
    all_acts = {}
    for group_name, group_texts in text_groups.items():
        if len(group_texts) == 0:
            continue
        print(f"  Collecting {group_name} ({len(group_texts)} texts)...")
        group_acts = {name: [] for name in hook_names}
        with torch.no_grad():
            for i in range(0, len(group_texts), batch_size):
                batch = group_texts[i : i + batch_size]
                ids, mask = _tokenize_with_mask(tokenizer, batch, seq_len=seq_len)
                ids = ids.to(model.cfg.device)
                _, cache = model.run_with_cache(ids, names_filter=hook_names)
                flat_mask = mask.reshape(-1)
                for name in hook_names:
                    a = cache[name].float().cpu().reshape(-1, cache[name].shape[-1])
                    group_acts[name].append(a[flat_mask])
        all_acts[group_name] = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in group_acts.items()
        }
        n = all_acts[group_name][hook_names[0]].shape[0]
        print(f"    -> {n} non-padding tokens")
    return all_acts


def conditional_cka(model_a_path, model_b_path, text_groups,
                     layers=None, batch_size=8, seq_len=128, device="cuda"):
    """Compute CKA for each group of texts at specified layers.

    Loads each model once, collects activations for all groups, then compares.
    Filters out padding tokens so they don't dominate the similarity.

    Args:
        model_a_path: path to first model
        model_b_path: path to second model
        text_groups: dict mapping group_name -> list of texts
        layers: list of layer indices to compare. Default: [0, 4, 8, 12, 15]
        batch_size: forward pass batch size
        seq_len: max tokens per sequence
        device: torch device

    Returns:
        dict mapping (group_name, layer) -> CKA value
    """
    if layers is None:
        layers = [0, 4, 8, 12, 15]

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    print("Loading model A...")
    model_a = load_hooked_model(model_a_path, device=device)
    acts_a = _collect_grouped_masked(model_a, text_groups, hook_names,
                                      batch_size=batch_size, seq_len=seq_len)
    del model_a
    torch.cuda.empty_cache()

    print("Loading model B...")
    model_b = load_hooked_model(model_b_path, device=device)
    acts_b = _collect_grouped_masked(model_b, text_groups, hook_names,
                                      batch_size=batch_size, seq_len=seq_len)
    del model_b
    torch.cuda.empty_cache()

    print("Computing CKA...")
    results = {}
    for group_name in text_groups:
        if group_name not in acts_a:
            continue
        for layer, name in zip(layers, hook_names):
            X = acts_a[group_name][name]
            Y = acts_b[group_name][name]
            cka = _linear_cka(X, Y)
            results[(group_name, layer)] = cka
            print(f"  {group_name:12s} layer {layer:2d}: CKA = {cka:.6f}")

    return results


def plot_conditional_cka(results, layers=None, save_path=None):
    """Plot conditional CKA as grouped bar chart or line plot.

    Args:
        results: dict from conditional_cka, mapping (group, layer) -> CKA
        layers: list of layer indices (inferred from results if None)
        save_path: optional path to save figure
    """
    if layers is None:
        layers = sorted(set(l for _, l in results.keys()))

    groups = sorted(set(g for g, _ in results.keys()))

    fig, ax = plt.subplots(figsize=(10, 5))
    for group in groups:
        cka_vals = [results.get((group, l), float("nan")) for l in layers]
        ax.plot(layers, cka_vals, marker="o", linewidth=2, markersize=6, label=group)

    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA")
    ax.set_title("Conditional CKA: NTP vs STP by Input Group")
    ax.set_ylim(0.95, 1.001)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig
