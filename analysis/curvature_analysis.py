"""Curvature analysis of hidden state trajectories.

Measures local directional consistency: how much the trajectory bends at each
token step, rather than how far it strays from a straight line (Frechet).

Core metric: 1 - cos(h_t - h_{t-1}, h_{t-1} - h_{t-2}) for consecutive triples.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def compute_curvature(trajectory, stride=1):
    """Compute per-token curvature for a trajectory.

    For hidden states at positions t-2k, t-k, t (where k=stride):

        curvature(t) = 1 - cos(h_t - h_{t-k}, h_{t-k} - h_{t-2k})

    where cos is cosine similarity between the two displacement vectors.

    Stride=1 measures consecutive-token curvature. Larger strides measure
    coarser-grained directional consistency, closer to what STP's random_span
    mode actually optimizes.

    Range: 0 (same direction, straight) to 2 (full reversal).

    Args:
        trajectory: np.ndarray of shape (T, d)
        stride: gap between sampled points (default 1 = consecutive)

    Returns:
        np.ndarray of curvature values. Length depends on stride:
        floor((T - 2*stride) / stride) + 1 values for stride > 0.
        Empty array if T < 2*stride + 1.

    Note: if displacement vectors are near-zero, cosine similarity is
    undefined. Returns 0 for those positions (no movement = no curvature).
    """
    if len(trajectory) < 2 * stride + 1:
        return np.array([])

    curves = []

    for t in range(2 * stride, len(trajectory), stride):
        v1 = trajectory[t] - trajectory[t - stride]
        v2 = trajectory[t - stride] - trajectory[t - 2 * stride]

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            curves.append(0.0)
            continue

        cos_sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        curvature = 1 - cos_sim
        curves.append(curvature)

    return np.array(curves)


def compute_curvature_across_layers(hidden_states):
    """Compute curvature across layers for each token position.

    For a fixed token position t, the trajectory through layers is
    hidden_states[:, t, :] — shape (layers, hidden_dim). This measures
    how a token's representation bends as it passes through the network.

    Args:
        hidden_states: np.ndarray of shape (layers, tokens, hidden_dim)

    Returns:
        np.ndarray of shape (tokens, layers-2)
        Empty array of shape (tokens, 0) if layers < 3.
    """
    num_layers, num_tokens, _ = hidden_states.shape
    if num_layers < 3:
        return np.zeros((num_tokens, 0))

    result = np.zeros((num_tokens, num_layers - 2))
    for t in range(num_tokens):
        layer_trajectory = hidden_states[:, t, :]  # (layers, hidden_dim)
        result[t] = compute_curvature(layer_trajectory)
    return result


# ---------------------------------------------------------------------------
# Analysis — run extraction + compute metrics
# ---------------------------------------------------------------------------


def analyze_model_curvature(
    model_path,
    base_model_name,
    examples,
    data_file,
    max_new_tokens=128,
    strides=(1,),
):
    """Run generation and compute per-example, per-layer curvature at multiple strides.

    Args:
        strides: tuple of stride values to compute curvature at.
            stride=1 is consecutive tokens, larger strides measure coarser
            directional consistency (closer to what STP random_span optimizes).

    Returns list of dicts, one per example:
        {
            curvature: dict[int, np.ndarray],  # stride -> (num_layers, n_values)
            curvature_across_layers: np.ndarray (T, num_layers-2),
            hidden_states: np.ndarray (layers, tokens, hidden_dim),
            tokens: list[str],
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
        prompt_hs = extraction["prompt_hidden_states"]  # (layers, prompt_len, hidden_dim)
        num_layers = hs.shape[0]
        num_tokens = hs.shape[1]
        num_prompt_tokens = prompt_hs.shape[1]

        if num_tokens < 3:
            continue

        # Curvature across tokens at each layer, for each stride
        curvature_by_stride = {}
        for s in strides:
            if num_tokens < 2 * s + 1:
                continue
            layer_curv = []
            for l in range(num_layers):
                layer_curv.append(compute_curvature(hs[l], stride=s))
            curvature_by_stride[s] = np.stack(layer_curv)

        if not curvature_by_stride:
            continue

        # Prompt curvature across tokens at each layer, for each stride
        prompt_curvature_by_stride = {}
        for s in strides:
            if num_prompt_tokens < 2 * s + 1:
                continue
            layer_curv = []
            for l in range(num_layers):
                layer_curv.append(compute_curvature(prompt_hs[l], stride=s))
            prompt_curvature_by_stride[s] = np.stack(layer_curv)

        # Curvature across layers at each token (always stride=1)
        cross_layer_curvature = compute_curvature_across_layers(hs)

        # Decode individual token strings
        gen_ids = tokenizer(extraction["generated_text"], add_special_tokens=False)[
            "input_ids"
        ]
        token_strs = tokenizer.convert_ids_to_tokens(gen_ids)
        if len(token_strs) > num_tokens:
            token_strs = token_strs[:num_tokens]
        elif len(token_strs) < num_tokens:
            token_strs += ["<?>"] * (num_tokens - len(token_strs))

        correct = check_correctness(
            extraction["generated_text"],
            ex["messages"],
            data_file,
        )

        results.append(
            {
                "curvature": curvature_by_stride,
                "prompt_curvature": prompt_curvature_by_stride,
                "curvature_across_layers": cross_layer_curvature,
                "hidden_states": hs,
                "tokens": token_strs,
                "correct": correct,
                "generated_text": extraction["generated_text"],
                "num_generated_tokens": num_tokens,
                "num_prompt_tokens": num_prompt_tokens,
            }
        )

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _get_curvature(r, stride=1):
    """Helper: get curvature array from result dict, handling both old and new formats."""
    c = r["curvature"]
    if isinstance(c, dict):
        return c[stride]
    return c


def plot_heatmap(results_by_model, labels, example_idx=0, stride=1):
    """Heatmap: layers x token positions, one subplot per model."""
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        if example_idx >= len(results):
            continue
        r = results[example_idx]
        curv = _get_curvature(r, stride)  # (layers, n_values)
        tokens = r["tokens"]

        im = ax.imshow(curv, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        ax.set_title(f"{label} ({'correct' if r['correct'] else 'incorrect'})")

        # Token labels on x-axis (offset by 2 since curvature starts at t=2)
        if curv.shape[1] <= 40:
            ax.set_xticks(range(curv.shape[1]))
            ax.set_xticklabels(tokens[2 : 2 + curv.shape[1]], rotation=90, fontsize=6)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Curvature Heatmap (example {example_idx})", fontsize=13)
    plt.tight_layout()
    plt.savefig("curvature_heatmap.png", dpi=150)
    plt.show()


def plot_mean_curvature_by_layer(results_by_model, labels, stride=1):
    """Line plot: mean curvature per layer, one line per model."""
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, results), color in zip(zip(labels, results_by_model), colors):
        valid = [
            r
            for r in results
            if not isinstance(r["curvature"], dict) or stride in r["curvature"]
        ]
        data = np.stack(
            [_get_curvature(r, stride).mean(axis=1) for r in valid]
        )  # (n_examples, layers)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        x = np.arange(data.shape[1])

        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Curvature")
    ax.set_title("Mean Curvature by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("curvature_mean_by_layer.png", dpi=150)
    plt.show()


def plot_correct_vs_incorrect(results_by_model, labels, stride=1):
    """Mean curvature by layer, split by correctness."""
    colors = {"correct": "steelblue", "incorrect": "coral"}
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        for split_name, color in colors.items():
            is_correct = split_name == "correct"
            subset = [
                r
                for r in results
                if r["correct"] == is_correct
                and (not isinstance(r["curvature"], dict) or stride in r["curvature"])
            ]
            if not subset:
                continue
            data = np.stack([_get_curvature(r, stride).mean(axis=1) for r in subset])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            x = np.arange(data.shape[1])

            ax.plot(x, mean, label=f"{split_name} (n={len(subset)})", color=color)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Layer")
        ax.set_title(f"{label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Curvature")
    plt.suptitle("Curvature: Correct vs Incorrect", fontsize=13)
    plt.tight_layout()
    plt.savefig("curvature_correct_vs_incorrect.png", dpi=150)
    plt.show()


def _get_prompt_curvature(r, stride=1):
    """Helper: get prompt curvature array from result dict."""
    c = r.get("prompt_curvature", {})
    if isinstance(c, dict):
        return c.get(stride)
    return c


def plot_correct_vs_incorrect_split(
    results_by_model, labels, stride=1, max_gen_tokens=None
):
    """Mean curvature by layer, split by correctness, separated into prompt and generation.

    max_gen_tokens: truncate generation curvature to this many tokens.
        Examples shorter than this are excluded. If None, uses the 25th
        percentile of generation curvature lengths.
    """
    colors = {"correct": "steelblue", "incorrect": "coral"}
    fig, axes = plt.subplots(
        2, len(labels), figsize=(7 * len(labels), 9), sharey="row"
    )
    if len(labels) == 1:
        axes = axes.reshape(2, 1)

    for col, (label, results) in enumerate(zip(labels, results_by_model)):
        # Determine truncation length
        gen_lengths = [
            _get_curvature(r, stride).shape[1]
            for r in results
            if (isinstance(r["curvature"], dict) and stride in r["curvature"])
            or not isinstance(r["curvature"], dict)
        ]
        trunc = max_gen_tokens if max_gen_tokens is not None else int(np.percentile(gen_lengths, 25))

        for split_name, color in colors.items():
            is_correct = split_name == "correct"
            subset = [
                r
                for r in results
                if r["correct"] == is_correct
                and (not isinstance(r["curvature"], dict) or stride in r["curvature"])
                and _get_curvature(r, stride).shape[1] >= trunc
            ]
            if not subset:
                continue

            # --- Prompt row ---
            prompt_curvs = [
                (r, pc)
                for r in subset
                if (pc := _get_prompt_curvature(r, stride)) is not None
            ]
            if prompt_curvs:
                prompt_data = np.stack(
                    [pc.mean(axis=1) for _, pc in prompt_curvs]
                )
                pmean = prompt_data.mean(axis=0)
                pstd = prompt_data.std(axis=0)
                px = np.arange(prompt_data.shape[1])
                axes[0, col].plot(
                    px, pmean, label=f"{split_name} (n={len(prompt_curvs)})", color=color
                )
                axes[0, col].fill_between(
                    px, pmean - pstd, pmean + pstd, alpha=0.2, color=color
                )

            # --- Generation row (truncated, short examples excluded) ---
            gen_data = np.stack(
                [_get_curvature(r, stride)[:, :trunc].mean(axis=1) for r in subset]
            )
            gmean = gen_data.mean(axis=0)
            gstd = gen_data.std(axis=0)
            gx = np.arange(gen_data.shape[1])
            axes[1, col].plot(
                gx, gmean, label=f"{split_name} (n={len(subset)})", color=color
            )
            axes[1, col].fill_between(
                gx, gmean - gstd, gmean + gstd, alpha=0.2, color=color
            )

        axes[0, col].set_title(f"{label} — Prompt")
        axes[1, col].set_title(f"{label} — Generation (first {trunc} tokens)")
        for row in range(2):
            axes[row, col].set_xlabel("Layer")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Mean Curvature")
    axes[1, 0].set_ylabel("Mean Curvature")
    plt.suptitle("Curvature: Correct vs Incorrect (Prompt / Generation)", fontsize=13)
    plt.tight_layout()
    plt.savefig("curvature_correct_vs_incorrect_split.png", dpi=150)
    plt.show()


def plot_curvature_profile(results_by_model, labels, example_idx=0, layer=-1, stride=1):
    """Per-token curvature at a fixed layer for a single example."""
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        if example_idx >= len(results):
            continue
        r = results[example_idx]
        curv = _get_curvature(r, stride)[layer]  # (n_values,)
        tokens = r["tokens"]

        x = np.arange(len(curv))
        ax.plot(x, curv, color="steelblue", alpha=0.8, linewidth=1.2)

        if len(curv) <= 40:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens[2 : 2 + len(curv)], rotation=90, fontsize=7)
        ax.set_xlabel("Token")
        ax.set_title(f"{label} ({'correct' if r['correct'] else 'incorrect'})")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Curvature")
    plt.suptitle(
        f"Per-Token Curvature Profile (layer {layer}, example {example_idx})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("curvature_profile.png", dpi=150)
    plt.show()


def plot_turtle(results_by_model, labels, example_idx=0, layer=-1, stride=1):
    """Turtle graphics: walk a 2D path turning by the curvature angle.

    Straight trajectory -> straight line. Constant curvature -> circle.
    Random curvature -> squiggle. Segments colored by token position.
    """
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        if example_idx >= len(results):
            continue
        r = results[example_idx]
        curv = _get_curvature(r, stride)[layer]  # (n_values,)

        # Convert curvature to angle: c = 1 - cos(theta), so theta = arccos(1 - c)
        angles = np.arccos(np.clip(1 - curv, -1, 1))

        # Walk the turtle
        x, y = [0.0], [0.0]
        heading = 0.0
        for angle in angles:
            heading += angle
            x.append(x[-1] + np.cos(heading))
            y.append(y[-1] + np.sin(heading))

        # Color segments by position
        cmap = plt.cm.viridis
        norm = plt.Normalize(0, len(angles))
        for i in range(len(x) - 1):
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=cmap(norm(i)),
                linewidth=1.5,
            )

        ax.set_aspect("equal")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label="Token position", fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Turtle Trajectory (layer {layer}, stride {stride}, example {example_idx})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("curvature_turtle.png", dpi=150)
    plt.show()


def plot_layer_curvature_heatmap(results_by_model, labels, example_idx=0):
    """Heatmap: across-layer curvature. Rows = tokens, columns = layers."""
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]

    for ax, label, results in zip(axes, labels, results_by_model):
        if example_idx >= len(results):
            continue
        r = results[example_idx]
        curv = r["curvature_across_layers"]  # (tokens, layers-2)
        tokens = r["tokens"]

        im = ax.imshow(curv, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Token Position")
        ax.set_title(f"{label} ({'correct' if r['correct'] else 'incorrect'})")

        if len(tokens) <= 40:
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=6)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Across-Layer Curvature (example {example_idx})", fontsize=13)
    plt.tight_layout()
    plt.savefig("curvature_across_layers_heatmap.png", dpi=150)
    plt.show()


def plot_mean_curvature_by_stride(results_by_model, labels, strides, layer=-1):
    """Mean curvature vs stride at a fixed layer, one line per model.

    This is the key multi-scale plot: if STP linearizes at coarser scales,
    its curvature should drop faster than NTP's as stride increases.
    """
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, results), color in zip(zip(labels, results_by_model), colors):
        means = []
        stds = []
        valid_strides = []
        for s in strides:
            valid = [
                r
                for r in results
                if isinstance(r["curvature"], dict) and s in r["curvature"]
            ]
            if not valid:
                continue
            # Mean curvature at this stride and layer, across all tokens and examples
            per_example = [_get_curvature(r, s)[layer].mean() for r in valid]
            means.append(np.mean(per_example))
            stds.append(np.std(per_example))
            valid_strides.append(s)

        means = np.array(means)
        stds = np.array(stds)
        ax.plot(valid_strides, means, "o-", label=label, color=color)
        ax.fill_between(
            valid_strides, means - stds, means + stds, alpha=0.2, color=color
        )

    ax.set_xlabel("Stride")
    ax.set_ylabel("Mean Curvature")
    ax.set_title(f"Mean Curvature vs Stride (layer {layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("curvature_by_stride.png", dpi=150)
    plt.show()
