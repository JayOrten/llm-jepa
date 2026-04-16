# STP vs NTP Interpretability Analysis Plan

## Context

We have two Llama-3.2-1B-Instruct models fine-tuned with LoRA (rank 16, alpha 32)
on the NL-RX-SYNTH dataset (natural language to regex):
- **NTP**: standard next-token prediction (regular strategy)
- **STP**: semantic tube prediction — adds an auxiliary loss that forces
  `Enc(end) - Enc(start)` for random spans to be predictive of the complementary
  span. This is an explicit geometric constraint: it pushes the residual stream
  toward a representation where span-level semantics are linearly extractable
  via endpoint subtraction.

Both models are merged-LoRA checkpoints. Weight diffs are ~1-1.4% relative
Frobenius norm. Global CKA across layers is ~1.0 (0.992 at layer 15). Despite
near-identical global geometry, performance differs dramatically.

We have trained BatchTopK SAEs (d_sae=16384, k=64) on layer 15 residual stream
activations for both models, using 10M OpenWebText tokens.

## Goal

Understand how STP changes the model's internal representations from two angles:
1. **Geometric**: How does the STP objective reshape the residual stream geometry,
   particularly at the fine-grained/subspace level that CKA misses?
2. **Density/superposition**: Does STP change how densely features are packed
   into the representation space? Does it alter the interference structure?

## Data

- **In-distribution**: `datasets/synth_train.jsonl` (8000 examples) and
  `datasets/synth_test.jsonl` (2000 examples) — NL to regex, chat format
- **Related task**: `datasets/spider_train.jsonl` (6587) and
  `datasets/spider_test.jsonl` (1447) — NL to SQL, same chat format.
  The models were NOT trained on Spider, but it's a related structured-output task.
  Useful for testing transfer/generalization differences.
- **Generic**: OpenWebText (streaming) — what the SAEs were trained on

**For most analyses below, use synth_test (in-distribution) as the primary
eval set, with spider_test as a secondary comparison.** The question is whether
STP's geometric changes are visible on the data it was trained for, and whether
they transfer to a related task.

**Tokenization note**: These are chat-format messages with system/user/assistant
roles. We need to tokenize them through the model's chat template to get the
same format the model saw during training. The raw text won't have the right
special tokens.

---

## Phase 1: Quick Diagnostics (baseline numbers)

### 1A. Participation Ratio (effective dimensionality)

**What it measures**: How many dimensions the residual stream "actually uses."
PR = (sum lambda_i)^2 / sum(lambda_i^2), where lambda_i are eigenvalues of
the activation covariance matrix. PR=1 means all variance is in one direction;
PR=d means variance is uniformly spread across d dimensions.

**Why it matters**: If STP changes how spread out information is across dimensions,
PR will differ. STP's endpoint-subtraction constraint could either concentrate
information into fewer directions (lower PR — more structured) or spread it
out (higher PR — more uniform use of the space).

**Implementation**:
- For each model, collect residual stream activations at each layer on synth_test
- Compute covariance matrix eigenvalues
- Compute PR per layer
- Compare PR profiles: NTP vs STP
- Repeat on spider_test to check if the pattern transfers

**Expected output**: Per-layer PR plot, NTP vs STP. A difference here means the
models use the representation space differently in dimensionality, even if CKA
says the overall geometry is similar.

**Function**: `participation_ratio(model_path, texts, n_layers=16, ...)`
Returns list of PR values per layer.

### 1B. Conditional CKA (input-stratified)

**What it measures**: CKA computed separately on subsets of inputs, rather than
aggregated across all inputs.

**Why it matters**: Global CKA is ~1.0 because most tokens are processed
similarly. But on the specific tokens/sequences where performance diverges,
the representations might differ substantially. Conditioning reveals where
the geometry actually changes.

**Implementation**:
- Load synth_test, tokenize through chat template
- Split inputs by difficulty or type if possible (short vs long regex targets,
  or by regex complexity)
- Also run on spider_test (out-of-distribution structured output)
- Compute CKA at layer 15 (and optionally other layers) for each subset
- Compare: does CKA drop on certain input types?

**Expected output**: Table of CKA values per input subset per layer.
If CKA is lower on spider (unseen task) than synth (training task), it suggests
STP carved out task-specific geometric changes.

**Function**: Reuse `layerwise_cka` but with filtered text lists.
Add a helper `load_chat_texts(jsonl_path, tokenizer)` that formats the messages
through the chat template.

---

## Phase 2: Cross-Model SAE Cofiring (feature correspondence via data)

### 2A. Cross-Model Feature Cofiring Matrix

**What it measures**: For each (feature_i in SAE_NTP, feature_j in SAE_STP),
how often do they co-activate on the same tokens? This gives an empirical
feature correspondence grounded in data, not in decoder weight similarity.

**Why this works despite the basis confound**: We're not comparing decoder
directions in the abstract. We're asking: "when this NTP feature fires on
token X, does this STP feature also fire on token X?" If yes, they're encoding
the same thing regardless of what direction each SAE chose to represent it in.

**Why it matters for density/superposition**: If NTP feature i maps to a single
STP feature j (one-to-one), the features are organized similarly. If NTP
feature i maps to multiple STP features (one-to-many), STP has split that
concept into finer pieces — less superposition. If multiple NTP features map
to a single STP feature (many-to-one), STP has merged concepts — more
superposition.

**Implementation**:
1. Run synth_test through both models, collect SAE activations (binary: fire/not)
   for every token position
2. Build co-occurrence matrix C of shape (d_sae_NTP, d_sae_STP):
   C[i,j] = number of tokens where both feature i (NTP) and feature j (STP) fire
3. Normalize: Jaccard similarity J[i,j] = C[i,j] / (count_i + count_j - C[i,j])
   to control for features that fire on everything
4. For each NTP feature, find its top-k STP matches (and vice versa)
5. Compute mapping statistics:
   - Distribution of "max Jaccard" per feature (how well does each feature
     have a counterpart?)
   - Fraction of features with a strong match (Jaccard > threshold)
   - Mapping cardinality: for features with matches, is it 1:1 or 1:many?

**Expected output**:
- Histogram of max Jaccard similarities (NTP->STP and STP->NTP)
- Summary stats: median match quality, fraction matched
- Mapping cardinality distribution

**Computational note**: d_sae=16384, so the full matrix is 16384x16384 = 268M
entries. Manageable if we use sparse binary activations and batch the dot products.
We should only compute over features that actually fire (skip dead features).

**Function**: `cross_model_cofiring(model_a_path, model_b_path, sae_a, sae_b,
texts, hook_name, batch_size=8, seq_len=128)`

### 2B. Within-Model Cofiring Graph Statistics

**What it measures**: For each SAE separately, how do features relate to each
other in terms of co-activation patterns?

**Why it matters for superposition**: This is distinct from cross-model
comparison. Within a single SAE:
- **Degree distribution**: How many other features does each feature co-fire
  with? A feature that co-fires with many others might be a "hub" representing
  a high-level concept that overlaps with many specific concepts.
- **Clustering coefficient**: Do features form tight clusters (groups that
  all co-fire together) vs sparse connections? Tight clusters suggest organized
  feature groups; sparse connections suggest more independent features.
- **Community structure**: Are there identifiable groups of features that
  consistently co-fire? These would correspond to semantic modules.

**Comparing NTP vs STP graph statistics** tells us whether STP reorganizes
the feature co-activation structure, even if we can't identify which specific
features changed.

**Implementation**:
1. For each model/SAE, run synth_test, collect binary activation vectors
2. Build within-model co-occurrence matrix (16384x16384 per model — sparse)
3. Threshold to create a graph (edge if Jaccard > threshold)
4. Compute graph statistics:
   - Mean/median degree
   - Degree distribution (plot)
   - Clustering coefficient
   - Number of connected components
5. Compare between NTP and STP

**Function**: `within_model_cofiring_stats(model_path, sae, texts, hook_name, ...)`

---

## Phase 3: Geometric Analysis of SAE Feature Spaces

### 3A. Decoder Geometry: Interference Structure

**What it measures**: How much do SAE decoder directions (the "feature
directions" in residual stream space) interfere with each other? This directly
measures superposition.

**Key idea**: In a model with no superposition, all features would be
orthogonal (cosine similarity = 0 between decoder vectors). In a model with
heavy superposition, features share directions and have nonzero cosine
similarity. The distribution of pairwise cosine similarities tells you about
the degree of superposition.

**Why comparing NTP vs STP is meaningful here**: Unlike cross-model feature
alignment (which we already established is confounded by basis choice), this
is a within-model property. We're asking: "within the NTP SAE, how much do
features interfere with each other?" vs "within the STP SAE, how much?"
The basis confound doesn't apply because we're not comparing features across
models — we're comparing a structural property of each model's feature space.

**Implementation**:
1. For each SAE, compute pairwise cosine similarities of decoder vectors
   (W_dec rows). Full matrix is 16384^2 — too large. Instead:
   - Sample 1000-2000 live (non-dead) features
   - Compute pairwise cosine sim for the sample
   - Build histogram of off-diagonal values
2. Compare distributions: NTP vs STP
   - Mean absolute cosine similarity (higher = more interference)
   - Fraction of pairs with |cos| > 0.5 (strong interference)
   - Tail behavior: how many near-parallel features?

**Expected output**: Overlaid histograms of pairwise cosine similarities.
If STP reduces superposition, its histogram should be more concentrated
near 0 (more orthogonal features).

**Function**: `decoder_interference(sae, n_sample=2000)`

### 3B. Activation-Based Effective Rank per Feature

**What it measures**: For each SAE feature, how many dimensions of the
residual stream does it actually use when it fires?

**Key idea**: When a feature fires, it activates a decoder direction. But the
residual stream context in which it fires might occupy a subspace. If feature i
always fires in a low-dimensional subspace, it's a clean, specific feature.
If it fires in a high-dimensional subspace, it might be polysemantic.

**Implementation**:
1. For each SAE feature that fires on at least N tokens:
   - Collect all activation vectors (residual stream, not SAE) where the
     feature fires
   - Compute PCA / participation ratio of those activation vectors
2. Compare distributions of per-feature effective rank: NTP vs STP

This is expensive (per-feature PCA). Sample the top 500 most-active features.

**Function**: `per_feature_effective_rank(model_path, sae, texts, hook_name, top_k=500)`

---

## Phase 4: Targeted Subspace Analysis

### 4A. Differential Activation Analysis

**What it measures**: On the same input tokens, which SAE features activate
differently between NTP and STP?

**Why it works**: Both SAEs see the exact same token sequences. For each token,
we have an activation vector from each SAE. The features that show the largest
systematic difference in activation (consistently more/less active in one model)
are the ones most affected by STP training.

**This is different from cofiring (Phase 2)**: Cofiring asks "do these two
features correspond?" Differential activation asks "which features changed
the most between models?"

**Implementation**:
1. Run synth_test through both models, collect SAE activations
2. For each feature in each SAE, compute:
   - Mean activation across all tokens
   - Activation rate (fraction of tokens where it fires)
3. Find features with the largest difference in activation rate between models
4. For the top-N differentially active features, inspect:
   - What tokens do they fire on? (sample examples)
   - Do they correspond to specific regex patterns or NL constructs?

**Expected output**:
- Ranked list of features most upregulated/downregulated in STP vs NTP
- Example tokens for the top features
- Categorization: do the differential features cluster around specific
  input types?

**Function**: `differential_activation(model_a_path, model_b_path, sae_a, sae_b,
texts, hook_name, top_k=50)`

### 4B. Subspace CKA (targeted directions)

**What it measures**: Instead of comparing full residual streams, project onto
specific subspaces and compare.

**Key idea**: CKA on the full 2048-dim space is ~1.0 because most dimensions
are shared. But if we project onto the top-k principal components of the
*difference* between model activations, we might find subspaces where
the models actually differ.

**Implementation**:
1. Collect activations from both models on same inputs at layer 15
2. Compute difference: delta = acts_STP - acts_NTP  (per token)
3. PCA of delta — the top components are the directions of maximum divergence
4. Project both models' full activations onto these top-k directions
5. Compute CKA on the projected activations
6. Also: what fraction of total variance lives in the "difference subspace"?

This separates the "shared geometry" (CKA~1) from the "change subspace"
(where the models actually differ).

**Expected output**:
- Variance explained by top-k difference components
- CKA on projected vs unprojected activations
- Visualization of what the top difference directions capture

**Function**: `subspace_divergence(model_a_path, model_b_path, texts, layer=15, top_k=32)`

---

## Phase 5: Superposition Metric (Elhage et al.)

### 5A. Feature Dimensionality via Decoder Interference

**What it measures**: How much of a "full dimension" each SAE feature occupies.
From "Toy Models of Superposition" (Elhage et al., 2022):

```
D_i = 1 / (1 + sum_{j≠i} cos²(W_i, W_j))
```

Where W_i is the decoder vector for feature i. If feature i is perfectly
orthogonal to all others, D_i = 1.0 (it has a full dimension to itself).
If it's highly correlated with other features, D_i approaches 0.

Summing over all features: total_D = sum_i D_i — the effective number of
independent dimensions the SAE is using.

**Why this is the right density metric**: It's a principled, published
definition from the superposition literature. It directly answers: how
densely packed are the features? A model using heavy superposition will
have lower total_D relative to d_sae.

**Caveat**: SAE basis variance still applies. Two SAEs on the same model
with different seeds could show different total_D. To partially control for
this, compare both models' scores against the expected value for random
unit vectors in d_model dimensions, which gives a normalized score.

**Implementation**:
1. Load SAE decoder weights W_dec (d_sae, d_model) for NTP and STP
2. Normalize rows to unit vectors
3. Compute pairwise cos² — do this in chunks (16384² is large but manageable)
4. Compute D_i per feature, total_D per SAE
5. Compare: NTP total_D vs STP total_D
6. Also plot distribution of D_i values

**Function**: `feature_dimensionality(sae, chunk_size=512)`

---

## Revised Execution Order

Phases 1 and 2 are complete. Revised plan going forward:

1. **Phase 4B** (subspace divergence) — SAE-free, basis-invariant. Find the
   specific directions in residual stream space where NTP and STP diverge.
   Characterize those directions: which tokens have high projections onto them?

2. **Phase 5A** (Elhage superposition metric) — Directly answers the density
   question. Computable from SAE weights alone, no model forward passes needed.

3. **Probing** — Deferred. We don't have a specific hypothesis about what new
   information STP encodes. Only revisit if 4B reveals a concrete direction
   worth probing (e.g., if the divergence subspace aligns with a specific
   linguistic property).

Phases 3A, 3B, and 4A are dropped:
- 3A (pairwise decoder cosine sim): superseded by 5A which is more principled
- 3B (per-feature effective rank): exploratory with no clear hypothesis
- 4A (differential SAE activation): SAE basis confound makes it uninterpretable

## Implementation Notes

- Each phase gets its own script (phase4b_subspace.py, phase5_superposition.py)
  and notebook, following the same pattern as phase1 and phase2.
- Chat-format tokenization: use `_tokenize_with_mask` from phase1/phase2 scripts.
- Memory management: load one model at a time, delete + empty_cache between.
- Spider data: same chat format as synth, just different system prompt.
