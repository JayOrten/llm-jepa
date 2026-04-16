# SAE Interpretability: STP vs NTP

## Goal

Train Sparse Autoencoders on the last layer of both the STP and NTP (baseline) models, then compare their learned features to understand how STP changes internal representations.

## Key Questions

1. **Feature alignment** — Do both models learn the same features (just arranged differently), or does STP create fundamentally different features?
2. **Superposition / L0** — Does STP pack more features per position (denser representations) or fewer?
3. **Temporal coherence** — Do STP features activate over longer spans (tracking semantic arcs) vs NTP features firing on individual tokens?

## Models

- NTP (baseline): `output-exp1-regular/` — Llama-3.2-1B-Instruct + LoRA, regular finetuning, merged checkpoint
- STP: `output-exp1-stp/` — same base, STP finetuning, merged checkpoint
- Architecture: 16 layers (0–15), hidden_dim=2048
- Special tokens added during training (from `SPECIAL_TOKENS` in models.py)

## SAE Setup

- **Library**: SAELens (uses TransformerLens under the hood)
- **Layer**: 15 (last layer, `blocks.15.hook_resid_post`) — start here, expand later
- **Architecture**: BatchTopK (modern, clean L0 control)
- **Expansion factor**: 8x → d_sae = 16,384 features
- **k**: 64 (number of active features per token)
- **Training data**: `Skylion007/openwebtext` streamed from HuggingFace
- **Training tokens**: ~10M (enough for 16K features; increase if reconstruction is poor)
- **Context size**: 256 tokens

### Why openwebtext instead of Spider?

Both models were finetuned on Spider, so their Spider-specific features are similar by construction. General text reveals how STP changed the *general* representational geometry. Can retrain on Spider later as a comparison.

### Potential issue: special tokens

Our models have special tokens added during training (see `llm_jepa/models.py SPECIAL_TOKENS`). TransformerLens loads models its own way. Need to verify that:
1. TransformerLens can load our local checkpoints
2. The tokenizer/embedding size matches after special token additions

If TransformerLens can't handle this cleanly, fallback plan: collect activations ourselves using our existing `representations.py` infrastructure, save to disk, and train the SAE on cached activations.

## Analyses (post-training)

### 1. Feature Alignment

For each STP SAE feature direction (decoder weight vector), find max cosine similarity to any NTP feature direction. Plot distribution of max similarities.
- High alignment (cos > 0.9 for most features) = same features, different arrangement
- Low alignment = STP learned fundamentally different features

### 2. Reconstruction Quality / Superposition

Train both SAEs with identical hyperparams. Compare:
- Reconstruction MSE at same k
- Optionally: vary k, plot L0 vs reconstruction loss Pareto frontier

### 3. Temporal Activation Patterns

Run both models + their SAEs on a shared set of sequences. For each SAE feature across token positions:
- **Activation autocorrelation**: corr(a_f(t), a_f(t+1)) averaged over sequences
- **Run length**: mean consecutive tokens where feature stays active
- **Conditional entropy**: H(a_f(t+1) | a_f(t))

Compare distributions between STP and NTP features.

## File Structure

```
analysis/
  sae_analysis.py        # Functions: SAE training wrapper, alignment, temporal stats, plots
  sae_analysis.ipynb     # Notebook: runs experiments, displays results
  representations.py     # (existing) shared model loading utilities — may not be needed if SAELens handles loading
```

## Implementation Plan

### Phase 1: Get one SAE trained (this session)

1. Install SAELens
2. Write `sae_analysis.py` with a training function that wraps SAELens config
3. Verify TransformerLens can load our models (special tokens issue)
4. Train SAE on NTP model, last layer — confirm it converges, check reconstruction loss
5. Train SAE on STP model, same config
6. Notebook cell: load both SAEs, compute feature alignment, plot

### Phase 2: Analysis

7. Feature alignment analysis
8. Reconstruction comparison
9. Temporal activation patterns (requires running models on shared sequences with SAEs attached)

## Open Questions

- [ ] TransformerLens + our special tokens — verify compatibility
- [ ] Is 10M tokens enough for stable features at 16K dictionary size?
- [ ] `from_pretrained_no_processing` vs `from_pretrained` — SAELens docs say use `no_processing` for SAE compatibility
- [ ] Do we need to handle the LoRA merge explicitly, or does loading the full checkpoint just work?
