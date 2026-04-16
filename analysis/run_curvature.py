"""Run curvature analysis.

Usage:
    python analysis/run_curvature.py
"""

import sys
import os

# Ensure repo root is on path for llm_jepa imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from representations import load_model, load_examples, extract_generated, check_correctness
from curvature_analysis import (
    analyze_model_curvature,
    plot_heatmap,
    plot_mean_curvature_by_layer,
    plot_correct_vs_incorrect,
    plot_curvature_profile,
    plot_turtle,
    plot_layer_curvature_heatmap,
    plot_mean_curvature_by_stride,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "NTP": "output-exp1-regular",
    "STP": "output-exp1-stp",
}
DATA_FILE = "datasets/synth_test.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
N_EXAMPLES = 20
MAX_NEW_TOKENS = 128
STRIDES = (1, 2, 4, 8, 16)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {N_EXAMPLES} examples from {DATA_FILE}...")
    examples = load_examples(DATA_FILE, BASE_MODEL, n=N_EXAMPLES)
    print(f"Loaded {len(examples)} examples")

    all_results = {}
    for label, model_path in MODELS.items():
        results = analyze_model_curvature(
            model_path, BASE_MODEL, examples, DATA_FILE, MAX_NEW_TOKENS,
            strides=STRIDES,
        )
        all_results[label] = results

        print(f"\n--- {label} sample outputs ---")
        for r in results[:3]:
            print(f"  {'OK' if r['correct'] else 'WRONG'}: {repr(r['generated_text'][:80])}")

    # Summary
    print("\n=== Summary ===")
    for label, results in all_results.items():
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        for s in STRIDES:
            valid = [r for r in results if s in r["curvature"]]
            if valid:
                mean_curv = np.mean([r["curvature"][s].mean() for r in valid])
                print(f"{label} stride={s}: mean curvature {mean_curv:.4f} (n={len(valid)})")
        print(f"{label}: {correct}/{total} correct\n")

    # Plots
    labels = list(all_results.keys())
    results_list = list(all_results.values())

    plot_mean_curvature_by_layer(results_list, labels)
    plot_correct_vs_incorrect(results_list, labels)
    plot_heatmap(results_list, labels, example_idx=0)
    plot_curvature_profile(results_list, labels, example_idx=0, layer=-1)
    plot_turtle(results_list, labels, example_idx=0, layer=-1)
    plot_layer_curvature_heatmap(results_list, labels, example_idx=0)
    plot_mean_curvature_by_stride(results_list, labels, STRIDES, layer=-1)
