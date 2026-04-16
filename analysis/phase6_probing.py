"""Phase 6: Linear Probing — Regex Structure Across Layers.

Probe frozen model activations at 5 layers to measure when and how well each
model (NTP vs STP) linearly encodes structural properties of the target regex.

Key question: does STP build stronger/earlier regex representations than NTP?

Probe position: the \\n\\n token immediately before regex generation begins
(second <|eot_id|> + 4 in the tokenized sequence). At this position the model
has processed the full NL description and the assistant-turn header.

Probe targets: 6 binary features parsed from the target regex string.
Probe model: sklearn LogisticRegression with class_weight='balanced'.
Primary metric: AUROC (threshold-independent, robust to class imbalance).
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score

from sae_analysis import load_hooked_model
from phase4b_subspace import load_chat_texts, _tokenize_with_mask

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROBE_LAYERS = [0, 3, 7, 11, 15]
EOT_ID = 128009  # <|eot_id|>

PROBE_TARGETS = [
    "has_alternation",
    "has_negation",
    "has_and",
    "has_boundary",
    "nesting_ge2",
    "num_quant_ge2",
]


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def _max_nesting(regex: str) -> int:
    depth = mx = 0
    for c in regex:
        if c == "(":
            depth += 1
            mx = max(mx, depth)
        elif c == ")":
            depth -= 1
    return mx


def extract_labels(jsonl_path: str) -> dict:
    """Parse regex strings from JSONL, compute 6 binary probe targets.

    Args:
        jsonl_path: path to synth_train.jsonl or synth_test.jsonl

    Returns:
        dict mapping feature name -> (n_examples,) bool numpy array
    """
    regexes = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            regexes.append(entry["messages"][2]["content"])

    has_alternation = np.array(["| " in r or "|)" in r or r.endswith("|") or
                                  ("|" in r and "&" not in r.replace("|", "")) or
                                  "|" in r
                                  for r in regexes], dtype=bool)
    # Simpler: just check for | character
    has_alternation = np.array(["|" in r for r in regexes], dtype=bool)
    has_negation    = np.array(["~" in r for r in regexes], dtype=bool)
    has_and         = np.array(["&" in r for r in regexes], dtype=bool)
    has_boundary    = np.array([r"\\b" in r or "\b" in r or "\\b" in r
                                 for r in regexes], dtype=bool)
    nesting_ge2     = np.array([_max_nesting(r) >= 2 for r in regexes], dtype=bool)
    num_quant_ge2   = np.array(
        [sum(1 for c in r if c in "*+?") + r.count("{") >= 2 for r in regexes],
        dtype=bool,
    )

    return {
        "has_alternation": has_alternation,
        "has_negation":    has_negation,
        "has_and":         has_and,
        "has_boundary":    has_boundary,
        "nesting_ge2":     nesting_ge2,
        "num_quant_ge2":   num_quant_ge2,
    }


def label_balance_report(labels: dict, split_name: str = "") -> None:
    """Print class balance for each probe target."""
    header = f"Label Balance Report{' — ' + split_name if split_name else ''}"
    print(f"\n{header}")
    print("-" * 55)
    print(f"  {'Target':<22}  {'Pos':>5}  {'Neg':>5}  {'Ratio':>6}  {'Warning'}")
    print("-" * 55)
    n = None
    for name, arr in labels.items():
        n = len(arr)
        pos = arr.sum()
        ratio = pos / n
        warn = "  <-- imbalanced" if ratio < 0.1 or ratio > 0.9 else ""
        print(f"  {name:<22}  {pos:>5}  {n-pos:>5}  {ratio:>6.3f}{warn}")
    print(f"  Total examples: {n}")


# ---------------------------------------------------------------------------
# Probe position
# ---------------------------------------------------------------------------


def find_probe_positions(input_ids: torch.Tensor, eot_id: int = EOT_ID) -> torch.Tensor:
    """Find extraction position for each sequence: second <|eot_id|> + 4.

    The tokenized chat template structure at the user→assistant boundary:
      ...<|eot_id|> <|start_header_id|> assistant <|end_header_id|> \\n\\n FIRST_TOKEN...
    The \\n\\n is at offset +4 from the second <|eot_id|>.

    Falls back to the last non-zero token if the assistant header is truncated.

    Args:
        input_ids: (batch, seq_len) int64 tensor
        eot_id: token ID for <|eot_id|>

    Returns:
        positions: (batch,) int64 tensor
    """
    batch_size, seq_len = input_ids.shape
    positions = torch.zeros(batch_size, dtype=torch.long)
    n_fallbacks = 0

    for b in range(batch_size):
        ids = input_ids[b]
        eot_positions = (ids == eot_id).nonzero(as_tuple=True)[0]

        if len(eot_positions) >= 2:
            pos = eot_positions[1].item() + 4
            if pos < seq_len:
                positions[b] = pos
            else:
                # Truncated: fall back to last non-pad token
                nonzero = (ids != 0).nonzero(as_tuple=True)[0]
                positions[b] = nonzero[-1].item() if len(nonzero) > 0 else seq_len - 1
                n_fallbacks += 1
        else:
            # No second eot_id found: fall back
            nonzero = (ids != 0).nonzero(as_tuple=True)[0]
            positions[b] = nonzero[-1].item() if len(nonzero) > 0 else seq_len - 1
            n_fallbacks += 1

    if n_fallbacks > 0:
        print(f"  WARNING: {n_fallbacks}/{batch_size} sequences fell back to last token "
              f"(assistant header not found, likely truncated)")

    return positions


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def _collect_acts_for_texts(model, texts, layers, batch_size=8, seq_len=128):
    """Collect probe-position activations from an already-loaded model.

    Single forward pass per batch extracts all requested layers simultaneously.

    Returns:
        dict mapping layer_idx -> (n_examples, d_model) float32 numpy array
    """
    tokenizer = model.tokenizer
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    layer_to_chunks = {l: [] for l in layers}

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]
        ids, mask = _tokenize_with_mask(tokenizer, batch_texts, seq_len=seq_len)
        ids_gpu = ids.to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(ids_gpu, names_filter=hook_names)

        # Find probe position for each example in this batch
        positions = find_probe_positions(ids)  # (batch,) on CPU

        for l, hook_name in zip(layers, hook_names):
            acts = cache[hook_name].float()  # (batch, seq_len, d_model)
            # Extract single vector per example at probe position
            batch_vecs = []
            for b in range(acts.shape[0]):
                vec = acts[b, positions[b].item(), :].cpu().numpy()
                batch_vecs.append(vec)
            layer_to_chunks[l].append(np.stack(batch_vecs, axis=0))

        del cache

    return {l: np.concatenate(chunks, axis=0) for l, chunks in layer_to_chunks.items()}


def collect_all_probe_activations(
    model_path: str,
    texts_train: list,
    texts_test: list,
    layers: list = PROBE_LAYERS,
    batch_size: int = 8,
    seq_len: int = 128,
) -> tuple:
    """Load model once, collect probe activations for both train and test splits.

    Args:
        model_path: checkpoint path
        texts_train: chat-templated strings for probe training
        texts_test: chat-templated strings for probe evaluation
        layers: which layer indices to extract
        batch_size: forward pass batch size
        seq_len: max sequence length

    Returns:
        (train_acts, test_acts) where each is dict[layer_idx -> (n, d_model) ndarray]
    """
    print(f"  Loading model from {model_path}")
    model = load_hooked_model(model_path)

    print(f"  Collecting train activations ({len(texts_train)} examples)...")
    train_acts = _collect_acts_for_texts(model, texts_train, layers, batch_size, seq_len)
    print(f"  Collecting test activations ({len(texts_test)} examples)...")
    test_acts = _collect_acts_for_texts(model, texts_test, layers, batch_size, seq_len)

    del model
    torch.cuda.empty_cache()

    for l in layers:
        print(f"  Layer {l:2d}: train={train_acts[l].shape}, test={test_acts[l].shape}")

    return train_acts, test_acts


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def train_probe(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> LogisticRegression:
    """Fit a linear probe with balanced class weights."""
    probe = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    probe.fit(X_train, y_train)
    return probe


def evaluate_probe(probe: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained probe on test data."""
    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    return {
        "accuracy":          accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "auroc":             roc_auc_score(y_test, y_prob),
        "f1":                f1_score(y_test, y_pred, zero_division=0),
        "y_pred":            y_pred,
        "y_prob":            y_prob,
    }


def run_probing_sweep(
    train_acts: dict,
    labels_train: dict,
    test_acts: dict,
    labels_test: dict,
    layers: list = PROBE_LAYERS,
    targets: list = PROBE_TARGETS,
    C: float = 1.0,
) -> pd.DataFrame:
    """Run probes for all (layer, target) combinations.

    Returns:
        DataFrame with columns: layer, target, accuracy, balanced_accuracy,
        auroc, f1, y_prob (list stored per row for downstream bootstrap)
    """
    rows = []
    for layer in layers:
        X_train = train_acts[layer]
        X_test  = test_acts[layer]
        for target in targets:
            y_train = labels_train[target].astype(int)
            y_test  = labels_test[target].astype(int)

            probe  = train_probe(X_train, y_train, C=C)
            result = evaluate_probe(probe, X_test, y_test)

            rows.append({
                "layer":             layer,
                "target":            target,
                "accuracy":          result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "auroc":             result["auroc"],
                "f1":                result["f1"],
                "y_prob":            result["y_prob"],
                "y_pred":            result["y_pred"],
            })
            print(f"  layer={layer:2d}  {target:<22}  AUROC={result['auroc']:.3f}  "
                  f"bal_acc={result['balanced_accuracy']:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap 95% CI for AUROC by resampling the test set."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue  # skip degenerate bootstrap samples
        aurocs.append(roc_auc_score(yt, yp))
    aurocs = np.array(aurocs)
    lo = np.percentile(aurocs, 100 * alpha / 2)
    hi = np.percentile(aurocs, 100 * (1 - alpha / 2))
    return {"mean": aurocs.mean(), "ci_low": lo, "ci_high": hi}


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap the AUROC difference between two probes on the same test set.

    Args:
        y_true: ground-truth labels
        y_prob_a: predicted probabilities from probe A (e.g., NTP)
        y_prob_b: predicted probabilities from probe B (e.g., STP)

    Returns:
        dict with delta_auroc (B - A), ci_low, ci_high, significant
    """
    rng = np.random.default_rng(42)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        da = roc_auc_score(yt, y_prob_a[idx])
        db = roc_auc_score(yt, y_prob_b[idx])
        deltas.append(db - da)
    deltas = np.array(deltas)
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return {
        "delta_auroc": deltas.mean(),
        "ci_low":      lo,
        "ci_high":     hi,
        "significant": bool(lo > 0 or hi < 0),
    }


def compute_all_deltas(
    results_ntp: pd.DataFrame,
    results_stp: pd.DataFrame,
    labels_test: dict,
    targets: list = PROBE_TARGETS,
    layers: list = PROBE_LAYERS,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Run paired_bootstrap_delta for all (layer, target) combinations.

    Returns:
        DataFrame with columns: layer, target, delta_auroc, ci_low, ci_high, significant
    """
    rows = []
    for layer in layers:
        for target in targets:
            row_ntp = results_ntp[(results_ntp.layer == layer) & (results_ntp.target == target)].iloc[0]
            row_stp = results_stp[(results_stp.layer == layer) & (results_stp.target == target)].iloc[0]
            y_true  = labels_test[target].astype(int)

            delta = paired_bootstrap_delta(
                y_true, row_ntp["y_prob"], row_stp["y_prob"], n_bootstrap=n_bootstrap
            )
            rows.append({
                "layer":       layer,
                "target":      target,
                "delta_auroc": delta["delta_auroc"],
                "ci_low":      delta["ci_low"],
                "ci_high":     delta["ci_high"],
                "significant": delta["significant"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_probing_curves(
    results_ntp: pd.DataFrame,
    results_stp: pd.DataFrame,
    labels_test: dict,
    metric: str = "auroc",
    targets: list = PROBE_TARGETS,
    layers: list = PROBE_LAYERS,
    n_bootstrap: int = 500,
    save_path: str = None,
):
    """Line plot: probe metric vs layer, one subplot per target, NTP vs STP lines."""
    n_targets = len(targets)
    ncols = 3
    nrows = (n_targets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes = axes.flatten()

    for ax, target in zip(axes, targets):
        y_true = labels_test[target].astype(int)

        for label, results, color in [("NTP", results_ntp, "tab:blue"),
                                       ("STP", results_stp, "tab:orange")]:
            vals, lo_cis, hi_cis = [], [], []
            for layer in layers:
                row = results[(results.layer == layer) & (results.target == target)].iloc[0]
                vals.append(row[metric])
                ci = bootstrap_ci(y_true, row["y_prob"], n_bootstrap=n_bootstrap)
                lo_cis.append(ci["ci_low"])
                hi_cis.append(ci["ci_high"])

            ax.plot(layers, vals, marker="o", label=label, color=color, linewidth=2)
            ax.fill_between(layers, lo_cis, hi_cis, alpha=0.15, color=color)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance")
        ax.set_title(target, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(layers)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[n_targets:]:
        ax.set_visible(False)

    fig.suptitle(f"Linear Probe {metric.upper()} by Layer: NTP vs STP\n"
                 f"(shaded = bootstrap 95% CI)", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_delta_heatmap(
    deltas_df: pd.DataFrame,
    targets: list = PROBE_TARGETS,
    layers: list = PROBE_LAYERS,
    save_path: str = None,
):
    """Heatmap: (STP - NTP) AUROC difference, layers x targets.

    Diverging colormap centered at 0. Stars on cells where CI excludes zero.
    """
    data = np.zeros((len(layers), len(targets)))
    sig  = np.zeros((len(layers), len(targets)), dtype=bool)

    for i, layer in enumerate(layers):
        for j, target in enumerate(targets):
            row = deltas_df[(deltas_df.layer == layer) & (deltas_df.target == target)].iloc[0]
            data[i, j] = row["delta_auroc"]
            sig[i, j]  = row["significant"]

    vmax = max(abs(data).max(), 0.01)
    fig, ax = plt.subplots(figsize=(len(targets) * 1.4 + 1, len(layers) * 0.9 + 1))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(targets)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(targets, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=10)

    for i in range(len(layers)):
        for j in range(len(targets)):
            txt = f"{data[i,j]:+.3f}"
            star = " *" if sig[i, j] else ""
            ax.text(j, i, txt + star, ha="center", va="center", fontsize=9,
                    color="white" if abs(data[i, j]) > vmax * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="STP − NTP AUROC")
    ax.set_title("Probe AUROC Difference (STP − NTP)\n(* = bootstrap 95% CI excludes 0)", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_probe_report(
    results_ntp: pd.DataFrame,
    results_stp: pd.DataFrame,
    deltas_df: pd.DataFrame = None,
    targets: list = PROBE_TARGETS,
    layers: list = PROBE_LAYERS,
):
    """Print formatted summary table of all probe results."""
    print("\n" + "=" * 80)
    print("Linear Probing Report — NTP vs STP")
    print("=" * 80)
    print(f"  Metric: AUROC (primary), Balanced Accuracy (secondary)")
    print(f"  Layers: {layers}")
    print()

    for target in targets:
        print(f"  {target}")
        print(f"  {'Layer':>7}  {'NTP AUROC':>10}  {'STP AUROC':>10}  {'Delta':>8}  {'Sig':>4}")
        print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}")
        for layer in layers:
            r_ntp = results_ntp[(results_ntp.layer == layer) & (results_ntp.target == target)].iloc[0]
            r_stp = results_stp[(results_stp.layer == layer) & (results_stp.target == target)].iloc[0]
            delta_str = f"{r_stp['auroc'] - r_ntp['auroc']:+.3f}"
            sig_str = ""
            if deltas_df is not None:
                d = deltas_df[(deltas_df.layer == layer) & (deltas_df.target == target)].iloc[0]
                sig_str = " *" if d["significant"] else ""
            print(f"  {layer:>7}  {r_ntp['auroc']:>10.3f}  {r_stp['auroc']:>10.3f}  "
                  f"{delta_str:>8}{sig_str}")
        print()

    print("=" * 80)
