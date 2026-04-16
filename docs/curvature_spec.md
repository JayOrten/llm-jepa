# Curvature Analysis Spec

## Goal

Measure **local directional consistency** of hidden-state trajectories using the STP curvature metric. Unlike Fréchet distance (which turned out to be uniformly low due to residual stream dominance), this metric looks at **per-token direction changes** — where the trajectory bends, not just how far it strays from a line.

## The Metric

For consecutive hidden states h_{t-2}, h_{t-1}, h_t at a given layer:

```
curvature(t) = 1 - cos(h_t - h_{t-1}, h_{t-1} - h_{t-2})
```

- **0** = same direction (straight)
- **1** = 90° turn
- **2** = full reversal

This is computed for t = 2, 3, ..., T-1, giving (T-2) curvature values per layer.

For a full sequence at all layers: **(num_layers, T-2)** matrix per example.

## Why This Might Work Where Fréchet Didn't

Fréchet distance measured global deviation from a straight line. The residual stream made everything look straight globally. Curvature measures **local** direction changes — the residual stream adds a large constant vector, but the *change in direction* between consecutive steps is independent of that constant. We're looking at the deltas of deltas, not the accumulated path.

## What We Reuse

From `representations.py` (unchanged):
- `load_model` — load checkpoint + tokenizer
- `load_examples` — load dataset, format prompts
- `extract_generated` — generate + collect hidden states → (layers, tokens, hidden_dim)
- `check_correctness` — evaluate generated text

From `frechet_analysis.py` (pattern only):
- The `analyze_model` loop structure (iterate examples, extract, compute, collect)

## New Code

### `curvature_analysis.py`

**Core computation** (user writes this):
- `compute_curvature(trajectory)` — given (T, d), return (T-2,) array of curvature values
  - This is the key function. Consecutive displacement vectors, cosine similarity, 1 - cos.

**Analysis driver:**
- `analyze_model_curvature(model_path, base_model_name, examples, data_file, max_new_tokens)`
  - Same pattern as frechet's `analyze_model`
  - Returns list of dicts per example:
    ```python
    {
        "curvature": np.ndarray,      # (layers, T-2)
        "tokens": list[str],          # decoded token strings for labeling
        "correct": bool,
        "generated_text": str,
        "num_generated_tokens": int,
    }
    ```

**Visualizations:**

1. **Heatmap: layers × token positions** — the dense view
   - Rows = layers, columns = generated token positions
   - Color = curvature value
   - Show for a single example, or averaged across examples
   - Token labels on x-axis (if not too many tokens)
   - Q: for averaging across examples with different sequence lengths, we'd need to either truncate to min length or pad. Truncate seems cleaner.

2. **Mean curvature by layer** — line plot, one line per model
   - Average curvature across all tokens and examples at each layer
   - Split by correct/incorrect like the frechet plots
   - Hypothesis: STP should have lower mean curvature (straighter local steps)

3. **Per-token curvature profile** — line plot for a single example
   - x = token position, y = curvature, at a fixed layer (or last layer)
   - Overlay the actual token text
   - Shows WHERE in the sequence bends happen

4. **Layer-wise curvature heatmap** — same as heatmap #1 but transposed perspective
   - For a fixed token position (or averaged across tokens), how does curvature evolve across layers?
   - This is a different trajectory: how a token's representation transforms through the network
   - Could reveal if certain layers are "bending" layers vs "straight" layers

5. **Turtle trajectory plot** — 2D visualization of curvature as a walked path
   - No PCA or dimensionality reduction. Purely derived from the curvature values.
   - Start at origin facing right. At each token step, turn right by the unsigned angle between consecutive displacement vectors, then move forward by a fixed step length.
   - Straight trajectory → straight line. Constant curvature → circle. Random curvature → squiggle.
   - The angle is unsigned (left/right has no canonical meaning in R^d — consecutive displacement pairs live in different 2D planes, so signed angles don't accumulate coherently).
   - Color each segment by token position (or by curvature magnitude).
   - Side-by-side: NTP vs STP for same example at same layer.

## Execution Plan

### Pair mode
- [ ] General flow: what data flows where, what each visualization is showing
- [ ] Visualization code: understand the heatmap construction, PCA projection

### User writes
- [ ] `compute_curvature(trajectory)` — the core 5-10 lines

### AI writes
- [ ] `analyze_model_curvature` — loop driver (boilerplate)
- [ ] Heatmap plot function
- [ ] Layer line plot function (adapted from frechet)
- [ ] Per-token curvature profile plot
- [ ] 2D trajectory plot with PCA + curvature coloring
- [ ] Notebook cells to run it all

## Two Axes of Curvature

1. **Across tokens** (fixed layer): how the trajectory bends as the model generates token by token. This is the primary metric — the STP loss operates on this axis.
2. **Across layers** (fixed token): how a single token's representation bends as it passes through the network. Different trajectory, different question — are certain layers "turning points" in representation space?

Both produce a curvature value per (layer, token) pair. The heatmap shows both simultaneously.

## Open Questions

- Token-type analysis: can we tag tokens as "NL" vs "regex" vs "boundary" to see if curvature patterns differ by token type?
- Should we look at curvature of the prompt tokens too, or only generated tokens?
- For layer-wise curvature, do we need to account for layer norm scaling? The magnitude of hidden states varies across layers, which affects displacement vector norms (though not their angle — cosine similarity is scale-invariant).
