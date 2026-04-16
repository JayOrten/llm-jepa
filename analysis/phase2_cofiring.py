"""Phase 2: SAE cofiring analysis — cross-model and within-model.

2A: Cross-model cofiring — do NTP and STP features correspond via data?
2B: Within-model cofiring graph — does STP reorganize feature co-activation?

All activation collection uses the chat-template tokenization from phase1,
and filters padding via attention mask.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
from transformers import AutoTokenizer

from sae_analysis import load_hooked_model, load_sae

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
HOOK_NAME = "blocks.15.hook_resid_post"


# ---------------------------------------------------------------------------
# Tokenization (same approach as phase1 to avoid padding contamination)
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
    """Tokenize and return (input_ids, attention_mask bool), padding filtered."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=seq_len,
        return_tensors="pt", add_special_tokens=False,
    )
    return enc["input_ids"][:, :seq_len], enc["attention_mask"][:, :seq_len].bool()


# ---------------------------------------------------------------------------
# SAE activation collection
# ---------------------------------------------------------------------------


def collect_sae_activations(model, sae, texts, hook_name=HOOK_NAME,
                              batch_size=8, seq_len=128):
    """Run texts through model+SAE, return binary activation matrix (real tokens only).

    Args:
        model: HookedTransformer
        sae: loaded SAE
        texts: list of chat-templated strings
        hook_name: hook point (should match SAE training hook)
        batch_size: sequences per forward pass
        seq_len: max tokens per sequence

    Returns:
        acts_binary: (n_real_tokens, d_sae) bool sparse matrix (scipy CSR)
        n_tokens: number of real (non-padding) tokens processed
    """
    tokenizer = model.tokenizer
    chunks = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            ids, mask = _tokenize_with_mask(tokenizer, batch, seq_len=seq_len)
            ids = ids.to(model.cfg.device)

            _, cache = model.run_with_cache(ids, names_filter=hook_name)
            resid = cache[hook_name].float()  # (batch, seq_len, d_in)

            # Encode through SAE
            flat_resid = resid.reshape(-1, resid.shape[-1])
            flat_mask = mask.reshape(-1)
            flat_resid_real = flat_resid[flat_mask]  # (n_real, d_in)

            feat_acts = sae.encode(flat_resid_real)  # (n_real, d_sae)
            binary = (feat_acts > 0).cpu().numpy()
            chunks.append(sparse.csr_matrix(binary))

    full = sparse.vstack(chunks)
    return full, full.shape[0]


# ---------------------------------------------------------------------------
# 2A: Cross-model cofiring
# ---------------------------------------------------------------------------


def cross_model_cofiring(sae_acts_a, sae_acts_b, min_fires=10):
    """Compute Jaccard similarity between NTP and STP features via data cofiring.

    For each (feature_i in SAE_a, feature_j in SAE_b), Jaccard = |A∩B| / |A∪B|.
    We only consider features that fired at least min_fires times.

    Args:
        sae_acts_a: (n_tokens, d_sae_a) scipy CSR bool matrix
        sae_acts_b: (n_tokens, d_sae_b) scipy CSR bool matrix
        min_fires: minimum token count for a feature to be included

    Returns:
        dict with:
            max_jaccard_a_to_b: (d_sae_a,) max Jaccard for each A feature
            max_jaccard_b_to_a: (d_sae_b,) max Jaccard for each B feature
            best_match_a_to_b: (d_sae_a,) index of best matching B feature
            best_match_b_to_a: (d_sae_b,) index of best matching A feature
            fire_count_a: (d_sae_a,) token count per A feature
            fire_count_b: (d_sae_b,) token count per B feature
            n_tokens: total tokens used
    """
    n_tokens = sae_acts_a.shape[0]
    assert sae_acts_b.shape[0] == n_tokens, "Both SAEs must be run on the same tokens"

    fire_count_a = np.asarray(sae_acts_a.sum(axis=0)).ravel()
    fire_count_b = np.asarray(sae_acts_b.sum(axis=0)).ravel()

    live_a = np.where(fire_count_a >= min_fires)[0]
    live_b = np.where(fire_count_b >= min_fires)[0]
    print(f"  Live features (>={min_fires} fires): A={len(live_a)}, B={len(live_b)}")

    A = sae_acts_a[:, live_a].astype(np.float32)  # (n_tokens, live_a)
    B = sae_acts_b[:, live_b].astype(np.float32)  # (n_tokens, live_b)

    # Co-occurrence: C[i,j] = number of tokens where A[:,i] and B[:,j] both fire
    # = A^T @ B  (sparse matmul)
    print("  Computing co-occurrence matrix...")
    C = (A.T @ B).toarray()  # (live_a, live_b)

    # Jaccard: J[i,j] = C[i,j] / (count_a[i] + count_b[j] - C[i,j])
    count_a = fire_count_a[live_a].reshape(-1, 1)  # (live_a, 1)
    count_b = fire_count_b[live_b].reshape(1, -1)  # (1, live_b)
    union = count_a + count_b - C
    union = np.maximum(union, 1)  # avoid division by zero
    J = C / union  # (live_a, live_b)

    max_j_a = J.max(axis=1)      # (live_a,) best match per A feature
    best_b = J.argmax(axis=1)    # (live_a,) which B feature
    max_j_b = J.max(axis=0)      # (live_b,) best match per B feature
    best_a = J.argmax(axis=0)    # (live_b,) which A feature

    # Map back to full feature indices
    full_max_j_a = np.zeros(sae_acts_a.shape[1])
    full_max_j_b = np.zeros(sae_acts_b.shape[1])
    full_best_b = np.full(sae_acts_a.shape[1], -1, dtype=int)
    full_best_a = np.full(sae_acts_b.shape[1], -1, dtype=int)

    full_max_j_a[live_a] = max_j_a
    full_max_j_b[live_b] = max_j_b
    full_best_b[live_a] = live_b[best_b]
    full_best_a[live_b] = live_a[best_a]

    return {
        "max_jaccard_a_to_b": full_max_j_a,
        "max_jaccard_b_to_a": full_max_j_b,
        "best_match_a_to_b": full_best_b,
        "best_match_b_to_a": full_best_a,
        "fire_count_a": fire_count_a,
        "fire_count_b": fire_count_b,
        "live_a": live_a,
        "live_b": live_b,
        "n_tokens": n_tokens,
    }


def cofiring_summary(result, label_a="NTP", label_b="STP"):
    """Print summary statistics from cross_model_cofiring."""
    live_a, live_b = result["live_a"], result["live_b"]
    j_ab = result["max_jaccard_a_to_b"][live_a]
    j_ba = result["max_jaccard_b_to_a"][live_b]

    print(f"Cross-model cofiring: {label_a} → {label_b}")
    print(f"  Tokens: {result['n_tokens']:,}")
    print(f"  Live features: {label_a}={len(live_a)}, {label_b}={len(live_b)}")
    print()
    for label, j in [(f"{label_a}→{label_b}", j_ab), (f"{label_b}→{label_a}", j_ba)]:
        print(f"  {label}:")
        print(f"    median max Jaccard: {np.median(j):.4f}")
        print(f"    mean max Jaccard:   {np.mean(j):.4f}")
        for thresh in [0.1, 0.3, 0.5]:
            frac = (j >= thresh).mean()
            print(f"    fraction >= {thresh}: {frac:.3f}")
        print()


def plot_cofiring_histogram(result, label_a="NTP", label_b="STP", save_path=None):
    """Histogram of max Jaccard similarities for cross-model cofiring."""
    live_a, live_b = result["live_a"], result["live_b"]
    j_ab = result["max_jaccard_a_to_b"][live_a]
    j_ba = result["max_jaccard_b_to_a"][live_b]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, j, title in [
        (axes[0], j_ab, f"{label_a} → {label_b}"),
        (axes[1], j_ba, f"{label_b} → {label_a}"),
    ]:
        ax.hist(j, bins=50, edgecolor="black", linewidth=0.4, color="steelblue")
        ax.axvline(np.median(j), color="red", linestyle="--",
                   label=f"median={np.median(j):.3f}")
        ax.set_xlabel("Max Jaccard Similarity")
        ax.set_ylabel("Feature count")
        ax.set_title(f"Cross-model cofiring: {title}")
        ax.set_xlim(0, 1)
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_mapping_cardinality(result, label_a="NTP", label_b="STP",
                              jaccard_thresh=0.1, save_path=None):
    """For each A feature with a strong match, how many B features share it?

    A feature in B is a 'match' to A-feature i if it's also in A-feature i's
    top matches (i.e., A-feature i is the best match of that B-feature).
    High cardinality (one NTP feature matched by many STP features) suggests
    concept splitting. Low cardinality suggests concept merging or 1:1.
    """
    live_a = result["live_a"]
    j_ab = result["max_jaccard_a_to_b"][live_a]
    best_b = result["best_match_a_to_b"][live_a]
    live_b = result["live_b"]
    j_ba = result["max_jaccard_b_to_a"][live_b]
    best_a = result["best_match_b_to_a"][live_b]

    # For each A feature i with j_ab[i] >= thresh, count how many B features
    # have i as their best match (also with j >= thresh)
    matched_a = live_a[j_ab >= jaccard_thresh]
    matched_b = live_b[j_ba >= jaccard_thresh]

    # Count: how many B features point to each A feature
    from collections import Counter
    b_to_a = best_a[j_ba >= jaccard_thresh]
    a_fanin = Counter(b_to_a)   # A feature -> count of B features pointing to it

    # For matched A features, get their fan-in count (default 0 if no B points to them)
    fanin_counts = [a_fanin.get(i, 0) for i in matched_a]

    fig, ax = plt.subplots(figsize=(8, 4))
    max_count = max(fanin_counts) if fanin_counts else 5
    bins = range(0, min(max_count + 2, 15))
    ax.hist(fanin_counts, bins=bins, edgecolor="black", linewidth=0.4,
            color="steelblue", align="left")
    ax.set_xlabel(f"# of {label_b} features that best-match this {label_a} feature")
    ax.set_ylabel(f"{label_a} feature count")
    ax.set_title(f"Mapping cardinality (Jaccard ≥ {jaccard_thresh})\n"
                 f"0 = {label_a} concept not found in {label_b}; "
                 f">1 = concept split")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 2B: Within-model cofiring graph statistics
# ---------------------------------------------------------------------------


def within_model_cofiring_stats(sae_acts, jaccard_thresh=0.05, min_fires=10):
    """Compute co-activation graph statistics for a single SAE.

    Args:
        sae_acts: (n_tokens, d_sae) scipy CSR bool matrix
        jaccard_thresh: threshold to draw an edge between two features
        min_fires: minimum fires to include a feature

    Returns:
        dict with graph statistics:
            degrees: (n_live,) degree per live feature
            mean_degree, median_degree
            clustering_coeff: mean local clustering coefficient (sampled)
            n_live: number of features included
            jaccard_thresh: threshold used
    """
    fire_count = np.asarray(sae_acts.sum(axis=0)).ravel()
    live = np.where(fire_count >= min_fires)[0]
    n_live = len(live)
    print(f"  Live features: {n_live} (of {sae_acts.shape[1]})")

    A = sae_acts[:, live].astype(np.float32)
    counts = fire_count[live]

    # Co-occurrence
    print("  Computing within-model co-occurrence...")
    C = (A.T @ A).toarray()  # (n_live, n_live)
    np.fill_diagonal(C, 0)   # remove self-loops

    # Jaccard
    ci = counts.reshape(-1, 1)
    cj = counts.reshape(1, -1)
    union = ci + cj - C
    union = np.maximum(union, 1)
    J = C / union

    # Adjacency at threshold
    adj = (J >= jaccard_thresh).astype(np.float32)
    np.fill_diagonal(adj, 0)

    degrees = adj.sum(axis=1)
    mean_deg = float(degrees.mean())
    median_deg = float(np.median(degrees))
    print(f"  Mean degree: {mean_deg:.1f}, Median: {median_deg:.1f}")

    # Local clustering coefficient (sampled — full computation is O(n^3))
    # cc_i = (triangles through i) / (degree_i * (degree_i - 1) / 2)
    sample_size = min(n_live, 500)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_live, size=sample_size, replace=False)
    cc_vals = []
    for i in sample_idx:
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc_vals.append(0.0)
            continue
        sub = adj[np.ix_(neighbors, neighbors)]
        triangles = sub.sum() / 2
        possible = k * (k - 1) / 2
        cc_vals.append(triangles / possible)
    mean_cc = float(np.mean(cc_vals))
    print(f"  Mean clustering coefficient (sample n={sample_size}): {mean_cc:.4f}")

    return {
        "degrees": degrees,
        "mean_degree": mean_deg,
        "median_degree": median_deg,
        "clustering_coeff": mean_cc,
        "n_live": n_live,
        "jaccard_thresh": jaccard_thresh,
        "fire_count": fire_count,
        "live_indices": live,
    }


def plot_degree_distribution(stats_dict, save_path=None):
    """Plot degree distributions for one or more within-model cofiring analyses.

    Args:
        stats_dict: dict mapping model_name -> result from within_model_cofiring_stats
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, stats in stats_dict.items():
        degrees = stats["degrees"]
        max_deg = int(np.percentile(degrees, 99))
        bins = np.arange(0, max_deg + 2, max(1, max_deg // 50))
        ax.hist(degrees, bins=bins, alpha=0.6, label=label, edgecolor="black",
                linewidth=0.3, density=True)
    ax.set_xlabel(f"Degree (Jaccard ≥ {list(stats_dict.values())[0]['jaccard_thresh']})")
    ax.set_ylabel("Density")
    ax.set_title("Within-model cofiring degree distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def print_cofiring_comparison(stats_a, stats_b, label_a="NTP", label_b="STP"):
    """Print comparison table of within-model cofiring graph stats."""
    print("Within-model cofiring graph statistics:")
    print(f"{'Metric':35s}  {label_a:>10}  {label_b:>10}  {'Diff':>10}")
    print("-" * 70)
    rows = [
        ("Live features", "n_live", ".0f"),
        ("Mean degree", "mean_degree", ".2f"),
        ("Median degree", "median_degree", ".2f"),
        ("Clustering coefficient", "clustering_coeff", ".4f"),
    ]
    for name, key, fmt in rows:
        a_val = stats_a[key]
        b_val = stats_b[key]
        diff = b_val - a_val
        print(f"  {name:33s}  {a_val:>10{fmt}}  {b_val:>10{fmt}}  {diff:>+10{fmt}}")


# ---------------------------------------------------------------------------
# Spike investigation
# ---------------------------------------------------------------------------


def investigate_spike(result, sae_acts_a, sae_acts_b, texts,
                      tokenizer, spike_thresh=0.95, seq_len=128):
    """Inspect the features with near-perfect Jaccard (the spike at 1.0).

    For each spike feature, reports:
    - Its fire count
    - The tokens it fires on (decoded)
    - Whether those tokens are concentrated in the system prompt

    Args:
        result: output of cross_model_cofiring
        sae_acts_a: (n_tokens, d_sae) sparse binary matrix for model A
        sae_acts_b: (n_tokens, d_sae) sparse binary matrix for model B
        texts: the chat-templated strings used to collect activations
        tokenizer: HF tokenizer (for decoding tokens)
        spike_thresh: Jaccard threshold to call a feature a "spike" feature
        seq_len: sequence length used during collection

    Returns:
        spike_info: list of dicts, one per spike NTP feature
    """
    live_a = result["live_a"]
    j_ab = result["max_jaccard_a_to_b"][live_a]
    fire_count_a = result["fire_count_a"]

    spike_mask = j_ab >= spike_thresh
    spike_features = live_a[spike_mask]
    spike_jaccards = j_ab[spike_mask]

    print(f"Spike features (Jaccard >= {spike_thresh}): {len(spike_features)} "
          f"of {len(live_a)} live NTP features")
    print()

    # Reconstruct the flat token list so we can decode which tokens each feature fired on
    # Re-tokenize texts to get the actual token ids (same as collection)
    all_ids = []
    for i in range(0, len(texts), 8):
        batch = texts[i : i + 8]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=seq_len,
                        return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"][:, :seq_len]
        mask = enc["attention_mask"][:, :seq_len].bool()
        flat_ids = ids.reshape(-1)[mask.reshape(-1)]
        all_ids.append(flat_ids.numpy())
    all_ids = np.concatenate(all_ids)  # (n_real_tokens,)

    spike_info = []
    for feat_idx, jaccard in sorted(zip(spike_features, spike_jaccards),
                                     key=lambda x: -x[1]):
        fired_positions = np.where(
            np.asarray(sae_acts_a[:, feat_idx].todense()).ravel() > 0
        )[0]
        fired_token_ids = all_ids[fired_positions]
        fired_tokens = [tokenizer.decode([t]) for t in fired_token_ids[:20]]

        from collections import Counter
        top_tokens = Counter(fired_token_ids.tolist()).most_common(10)
        top_decoded = [(tokenizer.decode([tid]), cnt) for tid, cnt in top_tokens]

        spike_info.append({
            "feature_idx": int(feat_idx),
            "jaccard": float(jaccard),
            "fire_count": int(fire_count_a[feat_idx]),
            "top_tokens": top_decoded,
        })

    # Print summary
    fire_counts_spike = fire_count_a[spike_features]
    fire_counts_non_spike = fire_count_a[live_a[~spike_mask]]
    print(f"Fire counts — spike features:     "
          f"median={np.median(fire_counts_spike):.0f}, "
          f"mean={np.mean(fire_counts_spike):.0f}")
    print(f"Fire counts — non-spike features: "
          f"median={np.median(fire_counts_non_spike):.0f}, "
          f"mean={np.mean(fire_counts_non_spike):.0f}")
    print()
    print("Top 10 spike features (by Jaccard):")
    for info in spike_info[:10]:
        tokens_str = ", ".join(f"{repr(t)}×{c}" for t, c in info["top_tokens"][:5])
        print(f"  feat {info['feature_idx']:5d}  "
              f"jaccard={info['jaccard']:.3f}  "
              f"fires={info['fire_count']:5d}  "
              f"top tokens: {tokens_str}")

    return spike_info
