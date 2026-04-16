# Fréchet Distance Analysis — Spec

## Goal

Measure how "straight" token representation trajectories are in trained models, using the discrete Fréchet distance. Compare NTP baseline vs STP-finetuned on synth data.

## What We're Measuring

**Primary view — "across tokens" at a fixed layer:**
Given a sequence of T tokens, layer l produces hidden states H_l = [h_1, h_2, ..., h_T], each in R^d.
This is a trajectory through d-dimensional space. The reference path is the straight line from h_1 to h_T.

The **discrete Fréchet distance** between the trajectory and the reference line tells us the worst-case deviation, respecting ordering. Normalized by arc length (sum of ||h_{t+1} - h_t||), this gives a scale-invariant **wandering ratio**.

- Wandering ratio ≈ 0: trajectory is basically a straight line
- Wandering ratio ≈ 1: worst deviation is comparable to total path length

**Secondary view — "across layers" at a fixed token:**
For token position p, the depth trajectory is [h_p^0, h_p^1, ..., h_p^L].
Same Fréchet analysis applies. Less interesting for now but cheap to compute.

## Two Extraction Modes

### Generation mode (primary)
Each model autoregressively generates its own response. The trajectory IS the model's
actual reasoning path — different models produce different tokens, different lengths.
This is the real question: does STP produce straighter trajectories when actually solving problems?

- Uses `model.generate()` with `output_hidden_states=True`
- Returns nested structure: hidden_states[step][layer] of (1, 1, hidden_dim) per generated token
- Need to stitch into (layers, seq_len, hidden_dim) — clean this up from the notebook's mess
- Trajectory analyzed over: generated tokens only (response region)

### Teacher forcing mode (secondary, cheap)
Both models see the exact same sequence: prompt + ground truth response, single forward pass.
Any trajectory difference is purely about internal geometry, not behavioral.
Good controlled comparison, but less ecologically valid.

- Single forward pass, returns (layers, seq_len, hidden_dim) directly
- Can slice to prompt-only, response-only, or full sequence

Both modes should be supported. Generation is the primary analysis.

## Key Design Decision: What Tokens Define the Trajectory?

For generation mode: generated tokens only (that's all we have).
For teacher forcing: support full / prompt-only / response-only via slicing. Default to response-only.

Principle: isolated experiments, not a mega framework. But the model loading / representation
extraction part is genuinely reusable — generalize that. Metrics and analysis stay per-experiment
so they're easy to modify without cascading changes.

## Architecture

### `analysis/representations.py` — shared extraction utility
Reusable across experiments. Keeps things you'd always need:
- `load_model(checkpoint_path, base_model_name)` → model, tokenizer
- `extract_generated(model, tokenizer, prompt, max_new_tokens)` → dict with:
  - `hidden_states`: (layers, seq_len, hidden_dim) stitched from generate() output
  - `generated_text`: decoded string
  - `prompt_len`: int
- `extract_teacher_forced(model, tokenizer, full_text, prompt_len)` → same shape output
- `load_examples(data_file, base_model_name, n)` → list of formatted prompts + references

Simple functions, no classes, no registries. Easy to read, easy to change.

### `analysis/frechet_analysis.py` — this experiment
Self-contained analysis script. Imports from representations.py for loading/extraction only.
Everything else lives here:
- Fréchet DP algorithm (pure numpy)
- Wandering ratio computation
- Correctness evaluation
- Visualization / plotting
- CLI entry point with argparse

Run: `python analysis/frechet_analysis.py --models output-exp1-regular output-exp1-stp`

Future experiments (e.g. curvature analysis, other metrics) would be separate scripts
that import from representations.py but have their own analysis logic.

## The Fréchet DP Algorithm

```
Given sequences P = [p_1, ..., p_n] and Q = [q_1, ..., q_m]:

dp[0][0] = d(p_1, q_1)
dp[i][j] = max(d(p_i, q_j), min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))

Fréchet distance = dp[n-1][m-1]
```

O(n*m) time and space. For T ≈ 50-200 tokens, trivial.

The reference line Q is constructed by linearly interpolating T points between h_1 and h_T:
`q_t = h_1 + (t-1)/(T-1) * (h_T - h_1)` for t = 1..T

This gives a 1-to-1 correspondence: point t on the trajectory maps naturally to point t on the line.

## Data & Models

- **Data:** `datasets/synth_test.jsonl` (NL → regex conversion)
- **Baseline (NTP):** `output-exp1-regular/` — Llama-3.2-1B-Instruct + LoRA, standard finetuning
- **STP:** `output-exp1-stp/` — same base + STP auxiliary loss
- **Base model:** `meta-llama/Llama-3.2-1B-Instruct`
- **Max length:** 128 tokens

## Visualization

Initial plots for this experiment:

1. **Wandering ratio by layer** — line plot, one line per model (NTP vs STP).
   x-axis = layer index, y-axis = mean wandering ratio across examples.
   Shaded ±1 std region. The core comparison plot.

2. **Wandering ratio: correct vs incorrect** — same as above but split by correctness.
   Four lines: NTP-correct, NTP-incorrect, STP-correct, STP-incorrect.
   Tests whether straighter trajectories correlate with getting the answer right.

3. **Per-example scatter** — x-axis = wandering ratio (at a chosen layer, e.g. last),
   y-axis = arc length. Color by model, marker by correct/incorrect.
   Shows the joint distribution: are correct examples clustered in a region?

4. **Single trajectory visualization** — for a few cherry-picked examples, project
   the trajectory onto its top 2 PCs and plot the 2D path with the reference line.
   Not quantitative, but builds intuition about what "wandering" looks like.

## Execution Plan — Who Does What

**Jay implements (user-drives):**
- The discrete Fréchet DP algorithm — core 10-15 lines of numpy
- The wandering ratio function — wiring Fréchet + arc length + reference line

**AI implements (AI-drives):**
- `analysis/representations.py` — model loading, hidden state extraction, data loading boilerplate
- CLI argument parsing and orchestration in `frechet_analysis.py`
- Visualization / plotting functions

**We build together (pair):**
- The hidden state stitching from generate() output — worth understanding the structure
- Interpreting results once we have them
- general flow and what is happening in the visualizations

## Open Questions

- [ ] Should we also compute Fréchet on the PC1-projected trajectory? (reduces noise but loses info)
- [ ] Per-layer analysis or just last layer? (probably all layers, plot layer-wise like the PCA notebook)
- [ ] Do we want to compare against base (unfinetuned) model too?
- [ ] Should the reference line be h_1→h_T or the PC1 direction? (h_1→h_T is simpler and more interpretable)
- [x] For generation mode: also record correctness (exact match on regex) so we can correlate wandering ratio with accuracy? **Yes — track per-example correctness. Enables correct vs incorrect trajectory comparison.**



# Next iteration:

Hmm, something doesn't seem right here. I know these models can get more of these quesitons right. We need to cmopare what we are doing with the original code and make sure we are doing everything the same as far as generaiton and scoring.

The wandering ratio is low for both, but it seems like the STP is actually slightly higher overall. This is counterintuitive.

Also, it seems like the higher the arc length, the lower the ratio. Does that make sense? I guess it does, but I thought it was that a lower ratio means less wandering. Probably jsut need to test with more samples.

Overall, let's check our work: look through the key parts of this and make sure we aren't missing anything or doing bad assumptions. Also, mark 3-5 key points in the code that might be impacting the results we are seeing, and I will review them myself.
