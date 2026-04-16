"""Run Fréchet distance analysis.

Usage:
    python analysis/run_frechet.py
"""

import sys
import os

# Ensure repo root is on path for llm_jepa imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from representations import load_model, load_examples, extract_generated, check_correctness
from frechet_analysis import (
    wandering_ratio,
    arc_length,
    analyze_model,
    plot_wandering_by_layer,
    plot_correct_vs_incorrect,
    plot_scatter,
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

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {N_EXAMPLES} examples from {DATA_FILE}...")
    examples = load_examples(DATA_FILE, BASE_MODEL, n=N_EXAMPLES)
    print(f"Loaded {len(examples)} examples")

    all_results = {}
    for label, model_path in MODELS.items():
        results = analyze_model(model_path, BASE_MODEL, examples, DATA_FILE, MAX_NEW_TOKENS)
        all_results[label] = results

        # Print a few generated texts for sanity checking
        print(f"\n--- {label} sample outputs ---")
        for r in results[:3]:
            print(f"  {'OK' if r['correct'] else 'WRONG'}: {repr(r['generated_text'][:80])}")

    # Summary
    print("\n=== Summary ===")
    for label, results in all_results.items():
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        mean_wr = np.mean([r["wandering_ratios"][-1] for r in results])
        print(f"{label}: {correct}/{total} correct, mean wandering ratio (last layer): {mean_wr:.4f}")

    # Plots
    labels = list(all_results.keys())
    results_list = list(all_results.values())

    os.makedirs("analysis", exist_ok=True)
    plot_wandering_by_layer(results_list, labels)
    plot_correct_vs_incorrect(results_list, labels)
    plot_scatter(results_list, labels)
