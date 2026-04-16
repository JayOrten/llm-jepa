# Phase 4B: Subspace Divergence — Implementation Plan

## Goal

Find the specific directions in layer 15 residual stream space where NTP and
STP representations diverge, then characterize what those directions encode
by inspecting which tokens are most affected.

## Data Flow

```
synth_test.jsonl (2000 examples)
    │
    ├─ tokenize via chat template, filter padding
    │
    ├─ NTP model ──► acts_NTP  (n_tokens, 2048)  stored on CPU
    │
    └─ STP model ──► acts_STP  (n_tokens, 2048)  stored on CPU
                          │
                          ▼
                    delta = acts_STP - acts_NTP    (n_tokens, 2048)
                          │
                          ▼
                    PCA(delta, top_k=32)
                          │
                    ┌─────┴──────┐
                    │            │
              eigenvalues    components (32, 2048)
              (variance      (the directions of
               explained)     maximum divergence)
                    │            │
                    ▼            ▼
              cumulative    projections = delta @ components.T
              variance         (n_tokens, 32)
              plot              │
                                ▼
                          For each PC, sort tokens by |projection|
                          Decode top tokens → inspect what they are
```

## Step-by-step Implementation

### Step 1: Collect paired activations

Reuse the pattern from phase1/phase2: load one model at a time, tokenize with
`_tokenize_with_mask`, collect layer 15 residual stream activations for
non-padding tokens only.

Critical detail: both models must see the EXACT SAME token sequences in the
EXACT SAME order, so delta[i] is meaningful. We tokenize once, store the
token IDs, then feed them to both models. We also need to store the flat
token IDs (post-padding-filter) so we can decode them later.

Function: `collect_paired_activations(model_a_path, model_b_path, texts,
    layer=15, batch_size=8, seq_len=128)`

Returns:
- acts_a: (n_tokens, d_model) float32 CPU tensor
- acts_b: (n_tokens, d_model) float32 CPU tensor
- token_ids: (n_tokens,) int array — the actual token IDs at each position
- seq_indices: (n_tokens,) int array — which example each token belongs to

### Step 2: Compute delta and PCA

```python
delta = acts_b - acts_a            # (n_tokens, d_model)
delta_centered = delta - delta.mean(0)  # mean-center before PCA

# SVD of delta_centered to get principal components
# delta_centered = U @ diag(S) @ V^T
# V^T rows are the principal components (directions in d_model space)
# S^2 / (n-1) are the eigenvalues (variance explained per component)
U, S, Vt = torch.linalg.svd(delta_centered, full_matrices=False)
components = Vt[:top_k]            # (top_k, d_model)
eigenvalues = S[:top_k]**2 / (n_tokens - 1)
cumvar = eigenvalues.cumsum() / eigenvalues.sum()
```

Function: `divergence_pca(acts_a, acts_b, top_k=32)`

Returns:
- components: (top_k, d_model) — the divergence directions
- eigenvalues: (top_k,)
- cumvar: (top_k,) — cumulative fraction of divergence variance
- projections: (n_tokens, top_k) — each token's projection onto each PC

### Step 3: Characterize the top directions

For each of the top PCs (start with 5):

a) **Token analysis**: Sort tokens by |projection| onto this PC. Decode
   the top 50 tokens. Group them: are they regex operators? NL words?
   special tokens? Do they come from the user turn or assistant turn?

b) **Sequence analysis**: For each token, we know which example it belongs
   to (via seq_indices). Check whether certain examples dominate the
   projections. Are they structurally different from average?

c) **Sign analysis**: Positive vs negative projection has meaning.
   Positive = STP moved this token's representation MORE in this direction
   than NTP. Negative = STP moved it LESS. Report top tokens for each sign
   separately.

Function: `characterize_directions(components, projections, token_ids,
    seq_indices, tokenizer, top_k_tokens=50, n_components=5)`

### Step 4: Variance budget

Report:
- What fraction of total activation variance (not delta variance) lives in
  the divergence subspace? This is: ||delta||² / ||acts_a||² — how much of
  the representation the LoRA change touched.
- Cumulative variance of delta PCs: how many components to reach 50%, 80%, 90%?
  If few → clean, structured change. If many → diffuse, unstructured.

### Step 5: Visualize

Plots:
- Cumulative variance explained by delta PCs (scree plot)
- For each top PC: bar chart of token types (regex operator / NL word /
  special token / etc.) weighted by projection magnitude
- Optional: 2D scatter of tokens projected onto PC1 vs PC2, colored by
  token type

## Files

- `analysis/phase4b_subspace.py` — functions
- `analysis/phase4b_subspace.ipynb` — notebook

## Edge Cases and Concerns

1. **Mean shift**: If the average delta is large (STP shifted ALL tokens in
   one direction), PC1 will just be the mean shift. That's boring but real.
   Mean-centering delta before PCA removes this and focuses on token-specific
   differences.

2. **System prompt tokens**: Same concern as phase 2 — identical tokens
   across all examples may dominate. Consider filtering them out or flagging
   them separately. Since delta is per-token and system prompt tokens see
   the same input, their delta should be identical across examples. After
   mean-centering, they should project to ~0 on all PCs. So they should
   naturally drop out.

3. **Scale**: delta might be small in absolute terms (LoRA changes are 1-1.4%).
   PCA doesn't care about absolute scale — it finds directions of maximum
   variance in whatever signal exists. But we should report the ratio of
   delta variance to activation variance to contextualize.

4. **Token attribution**: When we decode top tokens, we should show them in
   context (a few tokens before/after) to make interpretation easier, not
   just isolated token strings.
