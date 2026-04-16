"""Phase 6B: Correctness-Predictive Probing.

Phase 6 failed because the probe targets (has_alternation, has_negation, etc.)
were lexically predictable from individual input tokens — layer 0 embeddings
hit 0.82–1.0 AUROC, leaving no room to compare models.

This phase probes for per-example correctness: did the model produce the exact-
match correct regex? This is:
  - Not lexically predictable (you can't know from input words if the model
    will get the regex right)
  - Directly connected to the NTP/STP performance gap (25% vs 69%)
  - Already computed in output-exp1-*/eval_output.jsonl

Three probing scenarios:
  1. NTP model → ntp_correct labels: when does NTP "know" it will succeed?
  2. STP model → stp_correct labels: when does STP "know" it will succeed?
  3. Both models → stp_only labels (STP correct AND NTP incorrect): the ~44%
     of examples where STP's advantage manifests. Same label, different activations
     — direct comparison.
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from sae_analysis import load_hooked_model
from phase4b_subspace import load_chat_texts, _tokenize_with_mask
from phase6_probing import (
    find_probe_positions,
    bootstrap_ci,
    paired_bootstrap_delta,
    train_probe,
    evaluate_probe,
    PROBE_LAYERS,
)

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def load_correctness_labels(eval_jsonl_path: str) -> np.ndarray:
    """Load per-example correctness from eval_output.jsonl.

    Args:
        eval_jsonl_path: path to eval_output.jsonl (e.g., output-exp1-regular/eval_output.jsonl)

    Returns:
        (n_examples,) bool array — True if model produced exact-match correct regex
    """
    labels = []
    with open(eval_jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            labels.append(bool(entry["correct"]))
    return np.array(labels, dtype=bool)


def make_label_sets(ntp_correct: np.ndarray, stp_correct: np.ndarray) -> dict:
    """Compute the three probe label arrays and print a summary.

    Args:
        ntp_correct: (n,) bool — NTP per-example correctness
        stp_correct: (n,) bool — STP per-example correctness

    Returns:
        dict with keys: ntp_correct, stp_correct, stp_only
    """
    assert len(ntp_correct) == len(stp_correct), "Label arrays must have same length"
    n = len(ntp_correct)

    stp_only = stp_correct & ~ntp_correct
    both_correct = stp_correct & ntp_correct
    neither = ~stp_correct & ~ntp_correct

    labels = {
        "ntp_correct": ntp_correct,
        "stp_correct": stp_correct,
        "stp_only":    stp_only,
    }

    print(f"\nLabel Summary (n={n})")
    print("-" * 55)
    print(f"  {'Label':<22}  {'Pos':>5}  {'Neg':>5}  {'Rate':>6}")
    print("-" * 55)
    for name, arr in labels.items():
        pos = arr.sum()
        print(f"  {name:<22}  {pos:>5}  {n-pos:>5}  {pos/n:>6.3f}")
    print("-" * 55)
    print(f"  Both correct:          {both_correct.sum():>5}  ({both_correct.mean():.3f})")
    print(f"  Neither correct:       {neither.sum():>5}  ({neither.mean():.3f})")

    return labels


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def collect_probe_activations_eval(
    model_path: str,
    texts: list,
    layers: list = PROBE_LAYERS,
    batch_size: int = 8,
    seq_len: int = 128,
) -> dict:
    """Collect probe-position activations for the eval set.

    Loads model, runs single forward pass per batch (all layers extracted at once),
    extracts the activation at the \\n\\n probe position per example, then deletes model.

    Args:
        model_path: checkpoint path
        texts: chat-templated strings (2000 eval examples, same order as eval_output.jsonl)
        layers: layer indices to extract
        batch_size: forward pass batch size
        seq_len: max sequence length

    Returns:
        dict[layer_idx -> (n_examples, d_model) float32 numpy array]
    """
    from transformers import AutoTokenizer

    print(f"  Loading model from {model_path}")
    model = load_hooked_model(model_path)
    tokenizer = model.tokenizer

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    layer_chunks = {l: [] for l in layers}

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]
        ids, _ = _tokenize_with_mask(tokenizer, batch_texts, seq_len=seq_len)
        ids_gpu = ids.to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(ids_gpu, names_filter=hook_names)

        positions = find_probe_positions(ids)  # (batch,) CPU

        for l, hook_name in zip(layers, hook_names):
            acts = cache[hook_name].float()  # (batch, seq_len, d_model)
            batch_vecs = [acts[b, positions[b].item(), :].cpu().numpy()
                          for b in range(acts.shape[0])]
            layer_chunks[l].append(np.stack(batch_vecs, axis=0))

        del cache

    del model
    torch.cuda.empty_cache()

    result = {l: np.concatenate(chunks, axis=0) for l, chunks in layer_chunks.items()}
    for l in layers:
        print(f"  Layer {l:2d}: {result[l].shape}")

    return result


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(n: int, labels: np.ndarray, test_frac: float = 0.25,
                      seed: int = 42):
    """Stratified split returning train and test index arrays.

    Args:
        n: total number of examples
        labels: (n,) bool or int array — used for stratification
        test_frac: fraction to hold out as test
        seed: random seed

    Returns:
        (train_idx, test_idx) — numpy index arrays
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx = np.arange(n)
    train_idx, test_idx = next(sss.split(idx, labels))
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Probing sweep
# ---------------------------------------------------------------------------


def run_correctness_sweep(
    acts: dict,
    labels_dict: dict,
    label_keys: list,
    layers: list = PROBE_LAYERS,
    test_frac: float = 0.25,
    C: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Run probes for all (layer, label) combinations.

    For each label key, a single stratified split is computed and reused across
    all layers. This ensures the same test examples are used at every layer,
    enabling valid layer-to-layer comparison.

    Args:
        acts: dict[layer_idx -> (n, d_model) ndarray]
        labels_dict: dict[label_key -> (n,) bool array]
        label_keys: which labels from labels_dict to probe
        layers: which layers to probe
        test_frac: probe test fraction
        C: logistic regression regularization

    Returns:
        (results_df, splits) where:
          results_df: DataFrame with columns layer, label, accuracy,
                      balanced_accuracy, auroc, f1, y_prob, y_true_test
          splits: dict[label_key -> (train_idx, test_idx)] — for paired bootstrap
    """
    n = next(iter(acts.values())).shape[0]
    splits = {}

    # Compute splits once per label
    for key in label_keys:
        splits[key] = stratified_split(n, labels_dict[key].astype(int),
                                        test_frac=test_frac, seed=seed)

    rows = []
    for layer in layers:
        X = acts[layer]
        for key in label_keys:
            y = labels_dict[key].astype(int)
            train_idx, test_idx = splits[key]

            probe  = train_probe(X[train_idx], y[train_idx], C=C)
            result = evaluate_probe(probe, X[test_idx], y[test_idx])

            rows.append({
                "layer":             layer,
                "label":             key,
                "accuracy":          result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "auroc":             result["auroc"],
                "f1":                result["f1"],
                "y_prob":            result["y_prob"],
                "y_true_test":       y[test_idx],
            })
            print(f"  layer={layer:2d}  {key:<22}  AUROC={result['auroc']:.3f}  "
                  f"bal_acc={result['balanced_accuracy']:.3f}")

    return pd.DataFrame(rows), splits


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_correctness_curves(
    ntp_results: pd.DataFrame,
    stp_results: pd.DataFrame,
    labels_dict: dict,
    layers: list = PROBE_LAYERS,
    n_bootstrap: int = 500,
    save_path: str = None,
):
    """Three-panel plot: one subplot per probing scenario.

    Subplot 1: NTP model → ntp_correct
    Subplot 2: STP model → stp_correct
    Subplot 3: Both models → stp_only (direct comparison)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios = [
        (axes[0], "NTP self-prediction",  [(ntp_results, "NTP", "tab:blue")],  "ntp_correct"),
        (axes[1], "STP self-prediction",  [(stp_results, "STP", "tab:orange")], "stp_correct"),
        (axes[2], "STP-advantage examples\n(STP correct & NTP incorrect)",
         [(ntp_results, "NTP", "tab:blue"), (stp_results, "STP", "tab:orange")], "stp_only"),
    ]

    for ax, title, model_specs, label_key in scenarios:
        y_true = labels_dict[label_key].astype(int)

        for results, model_name, color in model_specs:
            sub = results[results.label == label_key]
            if sub.empty:
                continue

            aurocs, lo_cis, hi_cis = [], [], []
            for layer in layers:
                row = sub[sub.layer == layer].iloc[0]
                aurocs.append(row["auroc"])
                ci = bootstrap_ci(row["y_true_test"], row["y_prob"], n_bootstrap=n_bootstrap)
                lo_cis.append(ci["ci_low"])
                hi_cis.append(ci["ci_high"])

            ax.plot(layers, aurocs, marker="o", label=model_name, color=color, linewidth=2)
            ax.fill_between(layers, lo_cis, hi_cis, alpha=0.15, color=color)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.set_xticks(layers)
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=9)

    fig.suptitle("Correctness-Predictive Probing: NTP vs STP\n"
                 "(shaded = bootstrap 95% CI)", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_correctness_report(
    ntp_results: pd.DataFrame,
    stp_results: pd.DataFrame,
    deltas_stp_only: pd.DataFrame = None,
    layers: list = PROBE_LAYERS,
):
    """Print formatted summary of correctness probing results."""
    print("\n" + "=" * 70)
    print("Correctness Probing Report")
    print("=" * 70)

    # Scenario 1 & 2: self-prediction
    for label, results, model in [("ntp_correct", ntp_results, "NTP"),
                                    ("stp_correct", stp_results, "STP")]:
        sub = results[results.label == label]
        if sub.empty:
            continue
        print(f"\n  {model} self-prediction ({label})")
        print(f"  {'Layer':>7}  {'AUROC':>8}  {'Bal.Acc':>8}")
        print(f"  {'-'*7}  {'-'*8}  {'-'*8}")
        for layer in layers:
            row = sub[sub.layer == layer].iloc[0]
            print(f"  {layer:>7}  {row['auroc']:>8.3f}  {row['balanced_accuracy']:>8.3f}")

    # Scenario 3: stp_only comparison
    ntp_sub = ntp_results[ntp_results.label == "stp_only"]
    stp_sub = stp_results[stp_results.label == "stp_only"]
    if not ntp_sub.empty and not stp_sub.empty:
        print(f"\n  STP-advantage examples (stp_only)")
        print(f"  {'Layer':>7}  {'NTP AUROC':>10}  {'STP AUROC':>10}  {'Delta':>8}  {'Sig':>4}")
        print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}")
        for layer in layers:
            rn = ntp_sub[ntp_sub.layer == layer].iloc[0]
            rs = stp_sub[stp_sub.layer == layer].iloc[0]
            delta = rs["auroc"] - rn["auroc"]
            sig_str = ""
            if deltas_stp_only is not None:
                d = deltas_stp_only[deltas_stp_only.layer == layer]
                if not d.empty and d.iloc[0]["significant"]:
                    sig_str = " *"
            print(f"  {layer:>7}  {rn['auroc']:>10.3f}  {rs['auroc']:>10.3f}  "
                  f"{delta:>+8.3f}{sig_str}")

    print("\n" + "=" * 70)
