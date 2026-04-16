# PCA Trajectory Analysis Plan

Comparing hidden state trajectories between NTP (regular) and STP models using PCA variance as a "linearity score."

The core question: does the STP objective produce more structured (linear) trajectories through embedding space than standard next-token prediction?

## Background

The Semantic Tube paper (arXiv:2602.22617) motivates the idea that representations should travel through a "tube" in embedding space — a structured, low-dimensional trajectory. If STP is working as intended, we'd expect its hidden states to be more constrained (more linear, lower-dimensional) than NTP's, at least in certain layers.

PCA variance ratio gives us a scalar summary of this: how much of a trajectory's variance is captured by its top principal component(s).

## What You Have

- Two trained models in `output-exp1-regular/` (NTP) and `output-exp1-stp/` (STP)
- Both are LoRA fine-tunes of `meta-llama/Llama-3.2-1B-Instruct` (16 layers, 2048 hidden dim)
- Evaluation data in `datasets/synth_test.jsonl`
- Existing code for loading models (`llm_jepa/models.py`) and data (`llm_jepa/data.py`)

## The Analysis Script

Create a single script, something like `scripts/pca_trajectory.py`. It should:

1. Load both models (regular and STP)
2. Run inference on a batch of eval examples
3. Extract hidden states across all layers
4. Compute PCA variance ratios
5. Produce comparison plots

### Step 1: Model & Data Loading

Reuse existing infrastructure. Look at how `eval.py` loads models — it calls into `llm_jepa/models.py` which handles LoRA merging and tokenizer setup. You want to load the model with `output_hidden_states=True` so the forward pass returns all layer activations. 
 
Key files to study: 
- `eval.py` — see how it loads a checkpoint 
- `llm_jepa/models.py` — `get_model_and_tokenizer()` handles the full setup 
- `llm_jepa/evaluation/evaluate.py` — see how generation/inference works

You don't need the full evaluation pipeline. You just need to tokenize inputs and run a forward pass.

### Step 2: Extract Hidden States

For each input sequence, a forward pass with `output_hidden_states=True` gives you:

```
outputs.hidden_states  # tuple of (num_layers + 1) tensors
                       # each tensor: (batch_size, seq_len, hidden_dim)
                       # index 0 = embedding layer output
                       # index 1..16 = transformer layer outputs
```

You'll want to collect these per-example, per-layer. Think about what to do with padding tokens — you should mask them out using the attention mask before running PCA. Only analyze the actual token positions.

### Step 3: Compute PCA Variance Ratios

For each sequence at each layer, you have a matrix of shape `(T, 2048)` where T is the number of non-padding tokens. The linearity score is:

```
λ_1 / Σλ_i
```

where λ are the eigenvalues of the covariance matrix of the centered hidden states.

Implementation options:
- `torch.pca_lowrank(centered_states, q=k)` — fast, gives you the top-k singular values directly. You'd need the full spectrum though, so this is only useful if you want just PC1/PC2.
- `torch.linalg.svd(centered_states, full_matrices=False)` — gives all singular values. Eigenvalues of the covariance matrix are `s^2 / (T - 1)`. The variance ratio is `s[0]^2 / sum(s^2)`.
- `torch.linalg.eigh(cov_matrix)` — compute covariance explicitly, then eigendecompose. More numerically direct but slower for large matrices.

SVD on the centered matrix is probably the cleanest approach. Since T << 2048 in most cases (max_length is 128), the SVD is fast.

Metrics to compute per sequence per layer:
- **PC1 ratio**: `s[0]^2 / sum(s^2)` — how "one-dimensional" the trajectory is. A value of 1.0 means all tokens move along a single direction; 0.116 means that direction explains only 11.6% of the variance and the trajectory is spread across many dimensions.

- **PC1+PC2 cumulative ratio**: `(s[0]^2 + s[1]^2) / sum(s^2)` — fraction of variance in the top 2 PCs. This catches trajectories that are curved but planar: a trajectory that traces a curve in a 2D plane will have low PC1 (it bends, so no single direction dominates) but high PC1+PC2 (all the action is in one plane). If PC1+PC2 is much higher than PC1, the trajectory is curved. If PC1+PC2 ≈ PC1, it's nearly straight.

- **Effective dimensionality**: `(Σs_i^2)^2 / Σ(s_i^4)` — the participation ratio. Rather than asking "how much does PC1 explain?", this asks "how many dimensions does the trajectory effectively occupy?" A perfectly 1D trajectory gives 1.0. A trajectory with variance spread uniformly across k dimensions gives exactly k. A value of e.g. 8.3 means the trajectory behaves like it lives in ~8 dimensions. This is a softer, more holistic measure than PC1 ratio — it's less sensitive to whether one PC dominates and more sensitive to the overall spread.

### Step 4: Structure the Comparison

You want to compare distributions, not just means. For each layer (0 through 16):
- Compute the linearity score for every sequence in the eval set
- You now have a distribution of scores for NTP and a distribution for STP

Think about what axes of comparison matter:
- **Across layers**: Does linearity change as you go deeper? Does STP become more linear earlier?
- **Across tokens**: You could also slice by token position (e.g., user tokens vs assistant tokens using `user_start_end` and `assistant_start_end` from the data). The STP objective specifically targets span representations — does it affect user vs assistant regions differently?
- **Across sequences**: Is the effect uniform or do some sequences show it more than others?

### Step 5: Visualization

Suggested plots (matplotlib is fine):

1. **Layer-wise linearity curve**: x-axis = layer index, y-axis = mean PC1 variance ratio. Two lines (NTP vs STP) with shaded confidence intervals (std or percentiles). This is the headline plot — it shows whether STP produces more linear trajectories and at which layers.

2. **Histogram/violin plots at selected layers**: Pick 2-3 interesting layers (e.g., early, middle, last) and show the full distribution of linearity scores for NTP vs STP. This reveals whether the effect is a clean shift or just a few outliers.

3. **PC1+PC2 cumulative curve**: Same as plot 1 but with cumulative PC1+PC2. If PC1 alone is similar but PC1+PC2 diverges, the trajectories are curved but planar.

4. **Effective dimensionality curve**: Layer-wise effective dimensionality. This is a softer measure — if STP shows lower effective dim, its trajectories live in fewer dimensions overall.

5. **(Optional) User vs assistant split**: Repeat plot 1 but separately for user-region tokens and assistant-region tokens. This tells you whether the STP objective's effect is localized to the spans it trains on.

## Implementation Notes

- Run on GPU if available, but this is inference only — a single 4080 handles it.
- Process sequences individually or in small batches. Hidden states for all layers at full sequence length eat memory. For 128 tokens × 2048 dim × 17 layers × float32, that's ~18MB per sequence — manageable.
- Save the raw per-sequence, per-layer scores to a CSV or numpy file so you can re-plot without re-running inference.
- Consider how many eval examples to use. The full test set might be 2000 examples. You could start with 100-200 to iterate faster, then scale up.

## What to Look For

- If STP shows higher PC1 ratios, especially in later layers, that's evidence the STP objective is constraining representations to a lower-dimensional manifold — consistent with the "semantic tube" hypothesis.
- If the effect is strongest in the layer specified by `span_layer` (which defaults to -1, the last layer), that makes sense — that's where the loss directly operates.
- If NTP and STP look the same, that's also interesting. It might mean the trajectory structure is an intrinsic property of the architecture rather than the training objective.
- Watch for the embedding layer (index 0) — it's just the token embedding lookup, so its PCA structure reflects vocabulary statistics, not learned dynamics.

## Stretch Goals

Once the basic analysis works:

- **Token-level PCA trajectories**: Instead of PCA across tokens at a fixed layer, do PCA across layers at a fixed token position. This shows how a single token's representation evolves through the network. A linear layer-wise trajectory means the network applies a consistent transformation.
- **Cosine similarity heatmaps**: For a single sequence, compute pairwise cosine similarity between all (token, layer) hidden states. This gives a 2D heatmap showing which tokens/layers have similar representations.
- **Projection onto PC1-PC2 plane**: For a few selected sequences, project the hidden states onto the top 2 PCs and plot the trajectory as a scatter/line plot. This gives visual intuition for what "linear" vs "wandering" looks like.
