"""Phase 5A: Superposition Metric (Elhage et al.)

Measure how densely SAE features are packed using the feature dimensionality
metric from "Toy Models of Superposition" (Elhage et al., 2022):

    D_i = 1 / (1 + sum_{j!=i} cos^2(W_i, W_j))

Where W_i is the decoder vector for feature i. D_i = 1.0 means feature i is
perfectly orthogonal to all others (owns a full dimension). D_i -> 0 means
heavy superposition. total_D = sum(D_i) is the effective number of independent
dimensions the SAE uses.

Computed entirely from SAE decoder weights — no model forward passes needed.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from sae_analysis import load_sae


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------


def feature_dimensionality(sae, chunk_size=512, device=None):
    """Compute Elhage feature dimensionality for each SAE feature.

    Args:
        sae: SAELens SAE object (needs .W_dec attribute)
        chunk_size: rows per chunk for pairwise cos^2 computation
        device: torch device for computation (defaults to sae's device)

    Returns:
        D: (d_sae,) numpy array — per-feature dimensionality in [0, 1]
        total_D: float — sum of D, effective number of independent dimensions
    """
    W = sae.W_dec.detach().float()  # (d_sae, d_model)
    if device is not None:
        W = W.to(device)

    d_sae = W.shape[0]

    # Normalize to unit vectors
    W_norm = W / W.norm(dim=1, keepdim=True)

    # Chunked pairwise cos^2 sum (excluding self-similarity)
    interference = torch.zeros(d_sae, device=W_norm.device)

    for i in range(0, d_sae, chunk_size):
        end = min(i + chunk_size, d_sae)
        chunk = W_norm[i:end]  # (chunk_size, d_model)
        sim = chunk @ W_norm.T  # (chunk_size, d_sae)
        sim_sq = sim ** 2

        # Zero out diagonal (self-similarity)
        for k in range(end - i):
            sim_sq[k, i + k] = 0.0

        interference[i:end] = sim_sq.sum(dim=1)

    D = 1.0 / (1.0 + interference)
    total_D = D.sum().item()
    D_np = D.cpu().numpy()

    print(f"  d_sae={d_sae}, d_model={W.shape[1]}")
    print(f"  total_D = {total_D:.1f} / {d_sae} ({total_D/d_sae:.4f} per feature)")
    print(f"  D_i: mean={D_np.mean():.4f}, median={np.median(D_np):.4f}, "
          f"min={D_np.min():.4f}, max={D_np.max():.4f}")

    return D_np, total_D


def feature_dimensionality_from_weights(W_dec, chunk_size=512, device="cuda"):
    """Same as feature_dimensionality but takes a raw weight matrix.

    Useful for random baseline computation where there's no SAE object.

    Args:
        W_dec: (d_sae, d_model) tensor or numpy array
        chunk_size: rows per chunk
        device: torch device

    Returns:
        D: (d_sae,) numpy array, total_D: float
    """
    if isinstance(W_dec, np.ndarray):
        W_dec = torch.from_numpy(W_dec)
    W = W_dec.float().to(device)
    d_sae = W.shape[0]

    W_norm = W / W.norm(dim=1, keepdim=True)
    interference = torch.zeros(d_sae, device=device)

    for i in range(0, d_sae, chunk_size):
        end = min(i + chunk_size, d_sae)
        chunk = W_norm[i:end]
        sim_sq = (chunk @ W_norm.T) ** 2
        for k in range(end - i):
            sim_sq[k, i + k] = 0.0
        interference[i:end] = sim_sq.sum(dim=1)

    D = 1.0 / (1.0 + interference)
    total_D = D.sum().item()
    return D.cpu().numpy(), total_D


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


def random_baseline_dimensionality(d_sae, d_model, n_samples=5,
                                    chunk_size=512, device="cuda"):
    """Compute expected feature dimensionality for random unit vectors.

    Generates d_sae random unit vectors in R^d_model and computes the same
    metric. Averages over n_samples runs. This controls for the fact that
    random vectors in a finite-dimensional space have nonzero cos^2.

    Returns:
        mean_total_D: float
        std_total_D: float
        mean_D_per_feature: float — average D_i across all features and runs
    """
    totals = []
    mean_Ds = []

    for s in range(n_samples):
        # Random unit vectors: sample from N(0,1), then normalize
        W = torch.randn(d_sae, d_model, device=device)
        W = W / W.norm(dim=1, keepdim=True)

        _, total_D = feature_dimensionality_from_weights(
            W, chunk_size=chunk_size, device=device
        )
        totals.append(total_D)
        print(f"  Random sample {s+1}/{n_samples}: total_D = {total_D:.1f}")

    totals = np.array(totals)
    return totals.mean(), totals.std(), totals.mean() / d_sae


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dimensionality_distribution(D_ntp, D_stp, save_path=None):
    """Overlaid histograms of per-feature D_i for NTP and STP."""
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, max(D_ntp.max(), D_stp.max()) * 1.05, 80)

    ax.hist(D_ntp, bins=bins, alpha=0.5, label="NTP", color="tab:blue", density=True)
    ax.hist(D_stp, bins=bins, alpha=0.5, label="STP", color="tab:orange", density=True)

    ax.axvline(D_ntp.mean(), color="tab:blue", linestyle="--", linewidth=1.5,
               label=f"NTP mean={D_ntp.mean():.4f}")
    ax.axvline(D_stp.mean(), color="tab:orange", linestyle="--", linewidth=1.5,
               label=f"STP mean={D_stp.mean():.4f}")

    total_ntp = D_ntp.sum()
    total_stp = D_stp.sum()
    ax.set_title(f"Per-Feature Dimensionality D_i\n"
                 f"total_D: NTP={total_ntp:.0f}, STP={total_stp:.0f}")
    ax.set_xlabel("D_i (feature dimensionality)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_dimensionality_summary(results_dict, baseline_mean, baseline_std=None,
                                 save_path=None):
    """Bar chart comparing total_D for each SAE and the random baseline."""
    fig, ax = plt.subplots(figsize=(6, 5))

    names = list(results_dict.keys()) + ["Random"]
    values = [results_dict[k]["total_D"] for k in results_dict] + [baseline_mean]
    colors = ["tab:blue", "tab:orange", "tab:gray"][:len(names)]

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Error bar on random baseline
    if baseline_std is not None:
        ax.errorbar(len(names) - 1, baseline_mean, yerr=baseline_std,
                     fmt="none", color="black", capsize=5, linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:.0f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Total Effective Dimensions (total_D)")
    ax.set_title("Feature Dimensionality: NTP vs STP vs Random Baseline")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_dimensionality_report(results_dict, baseline_mean, baseline_std=None):
    """Print text summary of dimensionality results."""
    d_sae = None

    print("\n" + "=" * 60)
    print("Feature Dimensionality Report (Elhage et al.)")
    print("=" * 60)

    for name, res in results_dict.items():
        D = res["D"]
        total = res["total_D"]
        d_sae = len(D)
        norm = total / baseline_mean if baseline_mean > 0 else float("nan")

        print(f"\n  {name}:")
        print(f"    total_D       = {total:.1f} / {d_sae}")
        print(f"    normalized    = {norm:.4f} (vs random baseline)")
        print(f"    D_i mean      = {D.mean():.4f}")
        print(f"    D_i median    = {np.median(D):.4f}")
        print(f"    D_i std       = {D.std():.4f}")
        print(f"    D_i min       = {D.min():.4f}")
        print(f"    D_i max       = {D.max():.4f}")

    print(f"\n  Random baseline:")
    print(f"    total_D       = {baseline_mean:.1f}" +
          (f" +/- {baseline_std:.1f}" if baseline_std else ""))

    if len(results_dict) == 2:
        names = list(results_dict.keys())
        t0 = results_dict[names[0]]["total_D"]
        t1 = results_dict[names[1]]["total_D"]
        print(f"\n  Ratio {names[0]}/{names[1]} = {t0/t1:.4f}")
        print(f"  Difference = {t0 - t1:.1f}")

    print("=" * 60)
