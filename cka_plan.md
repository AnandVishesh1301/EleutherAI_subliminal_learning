# Minibatch Linear CKA on PolyPythia Seed Variants — Implementation Brief

## Research intent (read first)

We study subliminal learning — trait transfer from a teacher model to a student
through semantically scrubbed data (number sequences) — and specifically whether
transfer strength depends on pretraining provenance. One candidate explanation
for why transfer is stronger between models sharing initialization is that their
internal representations are more aligned. This experiment measures that
alignment directly.

**Question:** Is the teacher's base model (pythia-160m) representationally
closer to the student bases it shares initialization with (data-seed variants)
than to those it shares data order with (weight-seed variants) — and does the
answer differ between ordinary text (Pile) and our transfer-channel data
(number sequences)?

The headline readout is a CONTRAST between the (ref, d_i) and (ref, w_i)
pair groups and between data conditions, not any absolute CKA value.
Absolute CKA values are uninterpretable without the floor/ceiling controls
specified below. Within-arm pairs serve only as a reference distribution
for how much CKA varies among "equivalent" models.

## Models

HuggingFace, EleutherAI PolyPythia 160M ablation arm:
- `pythia-160m` (shared reference)
- `pythia-160m-data-seed{1,2,3}` — same init, different data order
- `pythia-160m-weight-seed{1,2,3}` — different init, same data order

Final checkpoints only for the first pass. All models: fp32 forward passes (or
bf16 forward with fp32 accumulation of HSIC statistics).

**Known issue:** tokenizer resolution on these repos occasionally fails. If
`AutoTokenizer.from_pretrained` on a variant repo errors or returns a mismatched
vocab, fall back to the tokenizer from canonical `EleutherAI/pythia-160m`. All
models in the suite share one tokenizer; assert identical vocab size after load.

## Model pairs to run

The distillation design is: pythia-160m is the TEACHER; each seed variant is
a STUDENT base model. The hypothesis is that teacher-student representational
alignment (measured on base models, before any finetuning) predicts transfer
strength. Note (ref, d_i) pairs share initialization; (ref, w_i) pairs share
data order — so the directional prediction under the init-based account is
CKA(ref, d_i) > CKA(ref, w_i).

1. PRIMARY — teacher vs. each student base: (ref, d_i), (ref, w_i), i=1..3
2. SECONDARY (reference distribution for within-arm variability):
   (d1,d2), (d1,d3), (d2,d3) and (w1,w2), (w1,w3), (w2,w3)
3. Ceiling control: (ref, ref) — identical checkpoint both sides; must ≈ 1.0
   at every layer pair. This is also the pipeline correctness test.
4. Floor control: any pair from (1) with a per-batch random row permutation
   applied to the second model's activations before Gram computation. Breaks
   example correspondence, preserves marginals; expect ≈ 0.

## Data conditions (run the full pair matrix under each)

- **A — Pile:** held-out Pile samples (e.g. `EleutherAI/pile-uncopyrighted`
  validation split or the standard Pythia held-out set). On-distribution.
- **B — Number sequences:** our generated number-sequence transfer datasets
  (prompt + completion format from the subliminal learning pipeline). Ask the
  operator for the dataset path; do not regenerate.

## Activation extraction

- Hook the residual stream at each layer output (post-block), all 12 layers,
  plus embeddings (layer 0) → 13 activation matrices per model per batch.
- **Position sampling — important:** do NOT use all token positions. Adjacent
  positions are correlated and violate the independence assumption behind the
  unbiased HSIC estimator. Sample 1–4 random positions per sequence; each
  sampled (sequence, position) is one row. For condition B, sample only from
  digit-token positions in the completion (exclude separators/prompt).
- **Always exclude position 0 (BOS/first token) and any padding positions**
  from sampling in both conditions. First-token states are attention-sink
  outliers with norms far above normal tokens and are known to distort
  similarity measures in transformer LMs.
- The SAME tokenized batch, same rows, same order, must go through both models
  of a pair. Fix all sampling seeds; assert row counts match.
- To grow k without more data, follow the source paper's protocol: iterate
  over the same example pool multiple times (they use 10 passes at n=256),
  re-partitioning into fresh random batches each pass, sampling without
  replacement within a pass.
- Secondary aggregation condition (faithfulness check against the source
  paper's one-row-per-example setup): mean-pool activations over sequence
  positions, one row per sequence. Report alongside the position-sampled
  primary; do not substitute for it.

## CKA computation — port of Google Research minibatch code

Port `MinibatchCKA` (TF, from the "Do Wide and Deep Networks Learn the Same
Things?" repo — provided alongside this brief) to PyTorch. Preserve these
exactly; do not simplify:

1. `_generate_gram_matrix` implements Székely–Rizzo U-centering, NOT plain
   double-centering: zero diagonal → subtract row/col means with denominators
   (n−2) and 2(n−1) → re-zero diagonal. After this preprocessing, the dot
   product of two flattened Grams equals unbiased HSIC₁ up to a constant that
   cancels in the CKA ratio. Replacing this with H K H is the classic bug —
   it reverts to the biased estimator and breaks minibatch convergence.
2. Use the `across_models` path: cross-HSIC accumulator of shape (13, 13) plus
   separate self-HSIC accumulators per model. The within-model normalization
   (diagonal of one accumulator) is incorrect across models.
3. Accumulators hold SUMS of per-batch HSIC; divide only in `result()`. Never
   compute per-batch CKA and average.
4. Accumulate in fp32 regardless of forward dtype.
5. Batch size m ≥ 4 is a hard requirement of the U-statistic (denominators
   n−2, n−3). Use m = 256.
6. Batches: k = 100 minimum; record running CKA vs. k every 10 batches and
   save the convergence curve. If not visually flat by k = 100, extend to 200.

Validate the port: on random data, compare against a direct full-batch
unbiased-HSIC CKA computation; agreement to 1e-5 (mirrors `test_CKA` in the
reference code).

## Variants (same run, cheap)

- **PC-ablated CKA:** per model, estimate top principal components of the
  activation distribution (from a held-out set of batches), project out top-r
  components for r ∈ {1, 3, 5}, then run the identical pipeline. Motivation:
  raw residual-stream CKA is dominated by a few high-variance/outlier
  directions and may saturate near 1 for all pairs; the contrast of interest
  may only be visible after ablation. Report raw and ablated side by side.

## Outputs

Each model pair × condition × variant produces the full 13×13 cross-layer CKA
matrix: entry (i, j) = CKA(model A layer i, model B layer j), computed for all
layer pairs (the accumulator design gives this for free). Save as npy + png
heatmap.

Conventions — apply uniformly:
- Orientation is fixed: teacher (ref) layers on rows, student layers on
  columns, input→output ordering on both axes. Never flip between plots.
- Pin the heatmap color scale to a common fixed range (e.g. [0, 1]) across
  ALL pairs and conditions. No per-plot autoscaling.
- Extract the diagonal (i, i) as a per-layer alignment curve; the summary
  scalar per pair is the mean over the diagonal (report full-diagonal mean
  and a mid-to-late-layer band mean).

Also produce, per individual model, the within-model self-heatmap (model vs.
itself, all layer pairs) on each condition — this is the block-structure
diagnostic. If a large saturated block appears, the corresponding depth range
of the cross-model heatmaps is uninformative in raw form; lean on the
PC-ablated variant there.

Expected shapes (for the operator's sanity checks, not assertions in code):
cross-seed heatmaps should show a bright ridge on or near the diagonal; a
ridge displaced off the diagonal means depth-shifted correspondence (report
it — the raw diagonal understates similarity in that case); a uniformly
bright matrix with no ridge indicates dominant-component saturation, not
similarity. Early layers are typically more similar across seeds than later
layers, so a diagonal curve declining with depth is expected, not a bug.

Summary table: mean diagonal CKA per (pair, condition, variant), grouped by
arm, with min/max across pairs. Convergence curves (running CKA vs. k) per
run. One-page summary comparing: (ref, d_i) vs. (ref, w_i) arms, Pile vs.
numbers, raw vs. PC-ablated.

## Interpretation guardrails (for the writeup, not the code)

- A near-identical CKA across arms does NOT falsify representation-mediated
  transfer; linear CKA weights directions by variance and may be blind to the
  low-variance trait-relevant subspace. State this if results are flat.
- No claim should rest on absolute CKA values — only on contrasts, bracketed
  by the floor/ceiling controls.
- Deliberate deviations from common LLM CKA practice, to state in the
  writeup rather than leave implicit: (a) most LLM work computes full-batch
  linear CKA with the biased HSIC₀ estimator over pooled token positions; we
  use the Nguyen et al. minibatch unbiased estimator with independent
  position sampling, which is stricter about the U-statistic's independence
  assumption. (b) A complementary metric on a different aspect of geometry
  (e.g. mutual-kNN overlap on the same activation samples) is a cheap
  robustness add if the CKA contrast carries weight in the final analysis.