"""Fréchet distance analysis of hidden state trajectories.

Geometry functions and analysis logic. Import from the notebook or run standalone.
"""

import matplotlib.pyplot as plt
import numpy as np
import similaritymeasures
from tqdm import tqdm

from representations import (
    load_model,
    load_examples,
    extract_generated,
    check_correctness,
)

# ---------------------------------------------------------------------------
# Geometry — pure numpy
# ---------------------------------------------------------------------------


def discrete_frechet_distance(P, Q):
    """Compute discrete Fréchet distance between trajectories P and Q.

    Args:
        P: np.ndarray of shape (n, d) — first trajectory
        Q: np.ndarray of shape (m, d) — second trajectory

    Returns:
        float — the discrete Fréchet distance
    """
    return similaritymeasures.frechet_dist(P, Q)


def arc_length(trajectory):
    """Sum of consecutive Euclidean distances along a trajectory.

    Args:
        trajectory: np.ndarray of shape (T, d)

    Returns:
        float — total arc length
    """
    diffs = np.diff(trajectory, axis=0)  # (T-1, d)
    return np.sum(np.linalg.norm(diffs, axis=1))


def make_reference_line(trajectory):
    """Construct straight-line reference from first to last point.

    Returns T evenly spaced points along the line from trajectory[0] to trajectory[-1].

    Args:
        trajectory: np.ndarray of shape (T, d)

    Returns:
        np.ndarray of shape (T, d)
    """
    T = len(trajectory)
    start, end = trajectory[0], trajectory[-1]
    t = np.linspace(0, 1, T).reshape(-1, 1)  # (T, 1)
    return start + t * (end - start)


def wandering_ratio(trajectory):
    """Fréchet distance to straight-line reference, normalized by arc length.

    Args:
        trajectory: np.ndarray of shape (T, d)

    Returns:
        float — wandering ratio (0 = perfectly straight)

    """
    ref = make_reference_line(trajectory)
    fd = discrete_frechet_distance(trajectory, ref)
    al = arc_length(trajectory)
    if al == 0:
        return 0.0
    return (
        fd / al
    )  # tells us how badly the trajectory deviated from the straight. Higher is worse. if 0.5, half its journey was a detour.


# ---------------------------------------------------------------------------
# Analysis — run extraction + compute metrics
# ---------------------------------------------------------------------------


def analyze_model(model_path, base_model_name, examples, data_file, max_new_tokens=128):
    """Run generation and compute per-example, per-layer wandering ratios.

    Returns list of dicts, one per example:
        {
            wandering_ratios: np.ndarray (num_layers,),
            arc_lengths: np.ndarray (num_layers,),
            correct: bool,
            generated_text: str,
            num_generated_tokens: int,
        }
    """
    print(f"\nLoading {model_path}...")
    model, tokenizer = load_model(model_path, base_model_name)

    results = []
    for ex in tqdm(examples, desc="Extracting"):
        extraction = extract_generated(
            model,
            tokenizer,
            ex["prompt"],
            max_new_tokens=max_new_tokens,
        )

        hs = extraction["hidden_states"]  # (layers, tokens, hidden_dim)
        num_layers = hs.shape[0]
        num_tokens = hs.shape[1]

        if num_tokens < 2:
            # Need at least 2 points for a trajectory
            continue

        # Compute metrics at each layer
        layer_wandering = np.zeros(num_layers)
        layer_arc = np.zeros(num_layers)
        layer_frechet = np.zeros(num_layers)
        for l in range(num_layers):
            traj = hs[l]  # (tokens, hidden_dim)
            ref = make_reference_line(traj)
            fd = discrete_frechet_distance(traj, ref)
            al = arc_length(traj)
            layer_frechet[l] = fd
            layer_arc[l] = al
            layer_wandering[l] = fd / al if al > 0 else 0.0

        correct = check_correctness(
            extraction["generated_text"],
            ex["messages"],
            data_file,
        )

        results.append(
            {
                "wandering_ratios": layer_wandering,
                "arc_lengths": layer_arc,
                "frechet_distances": layer_frechet,
                "correct": correct,
                "generated_text": extraction["generated_text"],
                "num_generated_tokens": num_tokens,
            }
        )

    # Free GPU memory
    del model, tokenizer
    import torch

    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_wandering_by_layer(results_by_model, labels):
    """Line plot: mean wandering ratio per layer, one line per model."""
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, results), color in zip(zip(labels, results_by_model), colors):
        data = np.stack(
            [r["wandering_ratios"] for r in results]
        )  # (n_examples, layers)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        x = np.arange(data.shape[1])

        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Wandering Ratio")
    ax.set_title("Wandering Ratio by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("frechet_wandering_by_layer.png", dpi=150)
    plt.show()


def plot_correct_vs_incorrect(results_by_model, labels):
    """Wandering ratio by layer, split by correctness."""
    colors = {"correct": "steelblue", "incorrect": "coral"}
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        for split_name, color in colors.items():
            is_correct = split_name == "correct"
            subset = [r for r in results if r["correct"] == is_correct]
            if not subset:
                continue
            data = np.stack([r["wandering_ratios"] for r in subset])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            x = np.arange(data.shape[1])

            ax.plot(x, mean, label=f"{split_name} (n={len(subset)})", color=color)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Layer")
        ax.set_title(f"{label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Wandering Ratio")
    plt.suptitle("Wandering Ratio: Correct vs Incorrect", fontsize=13)
    plt.tight_layout()
    plt.savefig("frechet_correct_vs_incorrect.png", dpi=150)
    plt.show()


def plot_scatter(results_by_model, labels, layer=-1):
    """Scatter: Fréchet distance vs arc length at a given layer.

    Plots raw values instead of wandering ratio to avoid the spurious
    1/x correlation that appears when plotting ratio vs denominator.
    The y=x line shows where Fréchet distance equals arc length.
    """
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(8, 6))

    all_arc = []
    for (label, results), color in zip(zip(labels, results_by_model), colors):
        for r in results:
            marker = "o" if r["correct"] else "x"
            ax.scatter(
                r["arc_lengths"][layer],
                r["frechet_distances"][layer],
                color=color,
                marker=marker,
                alpha=0.6,
            )
            all_arc.append(r["arc_lengths"][layer])
        # Dummy entries for legend
        ax.scatter([], [], color=color, marker="o", label=f"{label} correct")
        ax.scatter([], [], color=color, marker="x", label=f"{label} incorrect")

    # Reference line: slope=1 means wandering ratio of 1
    if all_arc:
        max_val = max(all_arc) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.2, label="fd = arc length")

    ax.set_xlabel("Arc Length")
    ax.set_ylabel("Fréchet Distance")
    ax.set_title(f"Per-Example Scatter (layer {layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("frechet_scatter.png", dpi=150)
    plt.show()


def plot_trajectory_2d(results_by_model, labels, layer=-1, example_idx=0):
    """Project a single trajectory onto its top 2 PCs, plot with reference line."""
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        if example_idx >= len(results):
            continue

        # Get the raw hidden states for this example
        # (we'd need to store them — for now just skip if not available)
        ax.set_title(f"{label} — example {example_idx}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.text(
            0.5,
            0.5,
            "TODO: store trajectories\nfor visualization",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.suptitle(f"2D Trajectory Projection (layer {layer})", fontsize=13)
    plt.tight_layout()
    plt.show()
