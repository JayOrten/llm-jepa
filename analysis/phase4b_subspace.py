"""Phase 4B: Subspace Divergence Analysis.

Find the specific directions in layer 15 residual stream space where NTP and
STP representations diverge, then characterize what those directions encode
by inspecting which tokens are most affected.

Key idea: delta = acts_STP - acts_NTP is a (n_tokens, d_model) matrix.
Its principal components are the directions of maximum representational change.
The tokens with highest projections onto each PC tell us what that direction encodes.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

from sae_analysis import load_hooked_model

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
HOOK_NAME = "blocks.15.hook_resid_post"


# ---------------------------------------------------------------------------
# Tokenization (same pattern as phase1/phase2)
# ---------------------------------------------------------------------------


def load_chat_texts(jsonl_path, tokenizer=None):
    """Load chat-format JSONL, apply chat template. Returns (texts, raw_messages)."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
    path = Path(jsonl_path)
    raw_messages, texts = [], []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            msgs = entry["messages"]
            raw_messages.append(msgs)
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            ))
    return texts, raw_messages


def _tokenize_with_mask(tokenizer, texts, seq_len=128):
    """Tokenize texts, return (input_ids, attention_mask bool)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=seq_len,
        return_tensors="pt", add_special_tokens=False,
    )
    return enc["input_ids"][:, :seq_len], enc["attention_mask"][:, :seq_len].bool()


# ---------------------------------------------------------------------------
# Step 1: Collect paired activations
# ---------------------------------------------------------------------------


def collect_paired_activations(model_a_path, model_b_path, texts,
                                layer=15, batch_size=8, seq_len=128):
    """Collect layer residual stream activations from two models on identical tokens.

    Both models see the EXACT SAME token sequences in the EXACT SAME order, so
    delta[i] = acts_b[i] - acts_a[i] is meaningful and comparable.

    Args:
        model_a_path: path to first model checkpoint (NTP)
        model_b_path: path to second model checkpoint (STP)
        texts: list of chat-templated strings (same as used elsewhere)
        layer: which layer's residual stream to extract
        batch_size: sequences per forward pass
        seq_len: max tokens per sequence

    Returns:
        acts_a: (n_tokens, d_model) float32 CPU tensor
        acts_b: (n_tokens, d_model) float32 CPU tensor
        token_ids: (n_tokens,) int64 numpy array — token ID at each position
        seq_indices: (n_tokens,) int32 numpy array — which example each token is from
    """
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Tokenize once — both models see identical sequences
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize all batches: store (ids, mask) per batch
    batches = []
    all_token_ids = []
    all_seq_indices = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        ids, mask = _tokenize_with_mask(tokenizer, batch_texts, seq_len=seq_len)
        batches.append((ids, mask))

        # Flat token IDs for real (non-padding) positions
        flat_ids = ids.reshape(-1)
        flat_mask = mask.reshape(-1)
        real_ids = flat_ids[flat_mask].numpy()
        all_token_ids.append(real_ids)

        # Which example does each real token belong to?
        batch_size_actual = ids.shape[0]
        actual_len = ids.shape[1]  # padded to longest in batch, <= seq_len
        # Sequence index within the full dataset (not just this batch)
        example_idx = np.arange(batch_start, batch_start + batch_size_actual)
        # Repeat each sequence index for actual_len positions, then filter by mask
        seq_idx_flat = np.repeat(example_idx, actual_len)
        real_seq_idx = seq_idx_flat[flat_mask.numpy()]
        all_seq_indices.append(real_seq_idx)

    token_ids = np.concatenate(all_token_ids).astype(np.int64)
    seq_indices = np.concatenate(all_seq_indices).astype(np.int32)
    n_tokens = len(token_ids)
    print(f"  Tokenized: {n_tokens:,} real tokens from {len(texts)} sequences")

    def _collect_acts(model_path):
        model = load_hooked_model(model_path)
        chunks = []
        with torch.no_grad():
            for ids, mask in batches:
                ids_gpu = ids.to(model.cfg.device)
                _, cache = model.run_with_cache(ids_gpu, names_filter=hook_name)
                resid = cache[hook_name].float()  # (batch, seq_len, d_model)
                flat_resid = resid.reshape(-1, resid.shape[-1])
                flat_mask_flat = mask.reshape(-1)
                real_resid = flat_resid[flat_mask_flat].cpu()  # (n_real, d_model)
                chunks.append(real_resid)
        del model
        torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)  # (n_tokens, d_model)

    print(f"  Collecting activations from model A (NTP)...")
    acts_a = _collect_acts(model_a_path)
    print(f"  acts_a: {tuple(acts_a.shape)}, dtype={acts_a.dtype}")

    print(f"  Collecting activations from model B (STP)...")
    acts_b = _collect_acts(model_b_path)
    print(f"  acts_b: {tuple(acts_b.shape)}, dtype={acts_b.dtype}")

    assert acts_a.shape == acts_b.shape, "Activation shapes must match"
    assert acts_a.shape[0] == n_tokens, "Token count mismatch"

    return acts_a, acts_b, token_ids, seq_indices


# ---------------------------------------------------------------------------
# Step 2: Compute delta PCA
# ---------------------------------------------------------------------------


def divergence_pca(acts_a, acts_b, top_k=32):
    """PCA of delta = acts_b - acts_a to find directions of maximum divergence.

    Mean-centers the delta before SVD so PC1 captures token-specific variance,
    not the global mean shift.

    Args:
        acts_a: (n_tokens, d_model) float32 tensor — NTP activations
        acts_b: (n_tokens, d_model) float32 tensor — STP activations
        top_k: number of principal components to retain

    Returns:
        components: (top_k, d_model) — divergence directions (unit vectors)
        eigenvalues: (top_k,) float64 — variance explained per component
        cumvar: (top_k,) float64 — cumulative fraction of delta variance
        projections: (n_tokens, top_k) float32 — each token projected onto each PC
        mean_shift: (d_model,) float32 — the mean delta (removed before PCA)
        delta_norm_ratio: float — ||delta||_F / ||acts_a||_F (scale of change)
    """
    delta = acts_b - acts_a  # (n_tokens, d_model)

    # Variance budget: how large is the change relative to the activations?
    delta_norm = torch.norm(delta).item()
    acts_a_norm = torch.norm(acts_a).item()
    delta_norm_ratio = delta_norm / acts_a_norm

    mean_shift = delta.mean(dim=0)  # (d_model,)
    delta_centered = delta - mean_shift.unsqueeze(0)  # mean-center

    n_tokens = delta_centered.shape[0]
    print(f"  delta shape: {tuple(delta.shape)}")
    print(f"  ||delta||_F / ||acts_a||_F = {delta_norm_ratio:.4f} ({delta_norm_ratio*100:.2f}%)")
    print(f"  mean shift norm: {torch.norm(mean_shift).item():.4f}")
    print(f"  Running truncated SVD (top_k={top_k})...")

    # SVD: delta_centered = U @ diag(S) @ V^T
    # V^T rows are the principal components (directions in d_model space)
    # S^2 / (n-1) are the eigenvalues
    U, S, Vt = torch.linalg.svd(delta_centered, full_matrices=False)

    S = S.double()
    eigenvalues = S[:top_k] ** 2 / (n_tokens - 1)
    total_var = (S ** 2).sum() / (n_tokens - 1)
    cumvar = eigenvalues.cumsum(dim=0) / total_var.item()

    components = Vt[:top_k].float()  # (top_k, d_model)
    projections = delta_centered @ components.T  # (n_tokens, top_k)

    print(f"  Top eigenvalue fraction: {(eigenvalues[0] / total_var).item():.3f}")
    print(f"  Components to reach 50% var: {(cumvar < 0.5).sum().item() + 1}")
    print(f"  Components to reach 80% var: {(cumvar < 0.8).sum().item() + 1}")
    print(f"  Components to reach 90% var: {(cumvar < 0.9).sum().item() + 1}")

    return (
        components.numpy(),
        eigenvalues.numpy(),
        cumvar.numpy(),
        projections.numpy(),
        mean_shift.numpy(),
        delta_norm_ratio,
    )


# ---------------------------------------------------------------------------
# Step 3: Characterize top directions
# ---------------------------------------------------------------------------


def get_structural_token_ids(tokenizer):
    """Return the set of token IDs that are part of Llama's chat template scaffold.

    These tokens appear at fixed positions in every formatted sequence and
    dominate the delta PCA in uninformative ways. Filtering them out lets
    content tokens surface.

    Includes: BOS, EOS/EOT special tokens, role header tokens, and the
    '\n\n' separator that appears between role headers and content.
    """
    structural_strings = ["\n\n", "\n", "  "]
    special_ids = set(tokenizer.all_special_ids)

    # Encode structural strings without any auto-added special tokens
    extra_ids = set()
    for s in structural_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        extra_ids.update(ids)

    return special_ids | extra_ids


def characterize_directions(components, projections, token_ids, seq_indices,
                             tokenizer, top_k_tokens=50, n_components=5,
                             filter_ids=None):
    """Decode the tokens most affected by each divergence direction.

    For each PC:
    - Sort tokens by |projection| — these are the most diverged
    - Show top tokens for positive and negative projections separately
    - Annotate: is this a regex operator? Special token? NL word?

    Args:
        components: (top_k, d_model) — PCA components from divergence_pca
        projections: (n_tokens, top_k) — token projections from divergence_pca
        token_ids: (n_tokens,) int64 — token ID at each position
        seq_indices: (n_tokens,) int32 — which example each token is from
        tokenizer: HF tokenizer for decoding
        top_k_tokens: how many top tokens to decode per PC per sign
        n_components: how many PCs to characterize
        filter_ids: optional set of token IDs to exclude from top-k selection.
            Use get_structural_token_ids(tokenizer) to filter chat template
            scaffold tokens that otherwise flood every PC.

    Returns:
        List of dicts, one per PC, with keys:
            pc_idx, top_pos, top_neg, top_abs
            (each is list of (token_str, projection_value, seq_idx) tuples)
    """
    # Build boolean mask: True = keep this token position
    if filter_ids is not None:
        keep_mask = np.array([int(t) not in filter_ids for t in token_ids])
    else:
        keep_mask = np.ones(len(token_ids), dtype=bool)

    # Indices of kept positions — used to map back after argsort
    kept_positions = np.where(keep_mask)[0]

    results = []

    for pc_idx in range(min(n_components, projections.shape[1])):
        proj = projections[:, pc_idx]  # (n_tokens,)
        proj_kept = proj[kept_positions]  # only non-structural tokens

        # Sort within kept tokens
        abs_proj = np.abs(proj_kept)
        top_abs_local = np.argsort(-abs_proj)[:top_k_tokens]
        top_pos_local = np.argsort(-proj_kept)[:top_k_tokens]
        top_neg_local = np.argsort(proj_kept)[:top_k_tokens]

        def decode_tokens(local_indices):
            result = []
            for li in local_indices:
                i = kept_positions[li]   # original position in full array
                tid = int(token_ids[i])
                tok_str = tokenizer.decode([tid])
                result.append({
                    "token_str": tok_str,
                    "token_id": tid,
                    "projection": float(proj[i]),
                    "seq_idx": int(seq_indices[i]),
                })
            return result

        pc_result = {
            "pc_idx": pc_idx,
            "top_abs": decode_tokens(top_abs_local),
            "top_pos": decode_tokens(top_pos_local),
            "top_neg": decode_tokens(top_neg_local),
        }
        results.append(pc_result)

    return results


def _categorize_token(token_str):
    """Rough categorization of token type for visualization."""
    import re
    tok = token_str.strip()
    # Regex operators / punctuation commonly in regex
    REGEX_OPS = set(".*+?[]{}()|^$\\")
    if any(c in REGEX_OPS for c in tok) and len(tok) <= 4:
        return "regex_op"
    # Special tokens (angle bracket format)
    if tok.startswith("<|") and tok.endswith("|>"):
        return "special"
    # Whitespace / empty
    if not tok or tok.isspace():
        return "whitespace"
    # Digits
    if re.fullmatch(r"\d+", tok):
        return "digit"
    # NL words (alphabetic)
    if re.fullmatch(r"[a-zA-Z]+", tok):
        return "nl_word"
    return "other"


def print_pc_summary(char_results, n_components=5, n_show=20):
    """Print human-readable summary of top divergence directions."""
    for pc in char_results[:n_components]:
        idx = pc["pc_idx"]
        print(f"\n{'='*60}")
        print(f"PC {idx+1}: Top tokens by absolute projection")
        print(f"{'='*60}")

        print(f"\n  POSITIVE (STP > NTP in this direction):")
        for item in pc["top_pos"][:n_show]:
            print(f"    {repr(item['token_str']):20s}  proj={item['projection']:+.3f}"
                  f"  seq={item['seq_idx']}")

        print(f"\n  NEGATIVE (STP < NTP in this direction):")
        for item in pc["top_neg"][:n_show]:
            print(f"    {repr(item['token_str']):20s}  proj={item['projection']:+.3f}"
                  f"  seq={item['seq_idx']}")

        # Token type distribution in top abs
        cats = Counter(_categorize_token(x["token_str"]) for x in pc["top_abs"])
        print(f"\n  Token type distribution (top {len(pc['top_abs'])} by |proj|):")
        for cat, count in cats.most_common():
            print(f"    {cat:15s}: {count}")


# ---------------------------------------------------------------------------
# Step 4: Variance budget
# ---------------------------------------------------------------------------


def variance_budget(acts_a, acts_b, eigenvalues, cumvar, delta_norm_ratio):
    """Report how much of total activation variance is in the divergence subspace."""
    print("\nVariance Budget")
    print("=" * 50)
    print(f"  ||delta||_F / ||acts_NTP||_F = {delta_norm_ratio:.4f} ({delta_norm_ratio*100:.2f}%)")
    print(f"  (fraction of activation 'energy' touched by LoRA)")
    print()
    print(f"  Cumulative delta variance explained:")
    for target in [0.5, 0.8, 0.9, 0.95]:
        n_comps = int((cumvar < target).sum()) + 1
        print(f"    {int(target*100):3d}% variance: {n_comps:3d} components")
    print()
    print(f"  Structure interpretation:")
    if cumvar[4] > 0.8:
        print("    Few components explain most variance → structured, clean change")
    elif cumvar[15] > 0.8:
        print("    ~10-15 components → moderately structured")
    else:
        print("    Many components needed → diffuse, unstructured change")


# ---------------------------------------------------------------------------
# Step 5: Visualizations
# ---------------------------------------------------------------------------


def plot_scree(eigenvalues, cumvar, save_path=None):
    """Scree plot: variance explained per delta PC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(1, len(eigenvalues) + 1)

    ax1.bar(x, eigenvalues, color="steelblue", edgecolor="black", linewidth=0.4)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue (variance explained)")
    ax1.set_title("Scree plot: divergence PCs")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, cumvar * 100, "o-", color="steelblue", markersize=4)
    ax2.axhline(50, color="red", linestyle="--", alpha=0.5, label="50%")
    ax2.axhline(80, color="orange", linestyle="--", alpha=0.5, label="80%")
    ax2.axhline(90, color="green", linestyle="--", alpha=0.5, label="90%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative variance: NTP→STP divergence")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_token_types_per_pc(char_results, n_components=5, save_path=None):
    """Bar chart of token type distribution for each top PC."""
    categories = ["regex_op", "nl_word", "digit", "special", "whitespace", "other"]
    colors = {
        "regex_op": "#e74c3c",
        "nl_word": "#3498db",
        "digit": "#2ecc71",
        "special": "#9b59b6",
        "whitespace": "#95a5a6",
        "other": "#f39c12",
    }

    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4),
                              sharey=True)
    if n_components == 1:
        axes = [axes]

    for ax, pc in zip(axes, char_results[:n_components]):
        idx = pc["pc_idx"]
        # Weight by absolute projection magnitude
        cat_weights = Counter()
        for item in pc["top_abs"]:
            cat = _categorize_token(item["token_str"])
            cat_weights[cat] += abs(item["projection"])

        total = sum(cat_weights.values()) or 1
        heights = [cat_weights.get(c, 0) / total for c in categories]
        bar_colors = [colors[c] for c in categories]

        ax.bar(categories, heights, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_title(f"PC {idx+1}")
        ax.set_ylabel("Fraction of projection magnitude" if idx == 0 else "")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Token type distribution by divergence PC\n"
                 "(weighted by |projection| magnitude)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_pc1_pc2_scatter(projections, token_ids, tokenizer,
                          n_sample=3000, save_path=None):
    """2D scatter of tokens projected onto PC1 vs PC2, colored by token type."""
    categories = ["regex_op", "nl_word", "digit", "special", "whitespace", "other"]
    colors_map = {
        "regex_op": "#e74c3c",
        "nl_word": "#3498db",
        "digit": "#2ecc71",
        "special": "#9b59b6",
        "whitespace": "#95a5a6",
        "other": "#f39c12",
    }

    rng = np.random.default_rng(42)
    idx = rng.choice(len(token_ids), size=min(n_sample, len(token_ids)), replace=False)

    x = projections[idx, 0]
    y = projections[idx, 1]
    tids = token_ids[idx]

    cats = [_categorize_token(tokenizer.decode([int(t)])) for t in tids]
    point_colors = [colors_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=point_colors, alpha=0.4, s=8, linewidths=0)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors_map[c], label=c) for c in categories]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("PC1 projection (delta)")
    ax.set_ylabel("PC2 projection (delta)")
    ax.set_title(f"Tokens projected onto top-2 divergence PCs\n"
                 f"(n={len(idx):,} sampled tokens)")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig
