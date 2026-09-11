# xfam roster weight audit — what "shared initialization" can and cannot mean across the 14 models

**Audited:** 2026-09-11, weight-level and geometry-level, against the cold-1 snapshots of
the commits pinned in `src/subliminal_learning/campaign/xfam_models.py`
(`/mnt/cold-1/soar-subliminal/vishesh/huggingface/hub`, loaded offline with
`HF_HUB_OFFLINE=1`). **Every number in this document was produced by a run of this
audit** (§ Reproduction). The only facts quoted from elsewhere are labelled as such:
the campaign's own shared-init gate (`scripts/stage_xfam_assets.py::run_init_gate`,
which writes `logs/staging_report.json`) and the xsize results restated in
`docs/PREREG_XFAM.md` §1.

This document answers the measurable half of the campaign's question. `docs/PREREG_XFAM.md`
asks whether an animal trait crosses a change of pretraining corpus, architecture and
post-training stage; the scientific framing behind it is Cloud et al.'s claim
(arXiv:2507.14805) that transfer needs teacher and student to **share an initialization**.
This audit establishes **what can actually be said about the initializations and weights of
these 14 checkpoints** — and, just as importantly, what cannot be said at all.

---

## TL;DR

1. **`pythia-410m` and `pythia-410m-deduped` share one initialization, exactly.**
   Re-measured here: **292/292 trainable tensors bitwise identical at `step0`**, relL2
   exactly **0.0** (E1.1). This reproduces the campaign's shared-init gate independently.
   It is the **only** verified shared initialization between two independent pretraining
   runs anywhere in the roster.
2. **By the released weights, that shared init is invisible.** At `step143000` the same two
   runs are **relL2 1.0585** apart — **77.1 %** of the measured no-correspondence floor
   (1.3727), with **0/292** tensors still equal (E1.2, E1.5). Shared init is a fact about
   provenance, not a property you can read off the checkpoints the campaign actually loads.
3. **12 of the 14 roster repos do not publish an initialization at all.** Only
   `EleutherAI/pythia-410m` and `EleutherAI/pythia-410m-deduped` expose more than one
   revision (155 branches each, including `step0`); every other roster repo publishes
   exactly one branch, `main` (E1.6). For those 12 the shared-init question is not merely
   unmeasured — **there is no artifact to measure**.
4. **The three claimed Pythia descendants are descendants.** relL2 from `pythia-410m`:
   **hh-sft 0.00188**, **hh-dpo 0.00236**, **ppo 0.00255** — 0.14 % – 0.19 % of the floor,
   three orders of magnitude below the deduped rung (E1.3). Consistent with
   "post-trained from pythia-410m"; **consistency, not proof of provenance**.
5. **DPO-from-SFT is visible in the weights.** `hh-sft`↔`hh-dpo` is **0.00110**, closer than
   either is to the base (0.00188 / 0.00236), and their update directions align at
   **cos 0.891** against a random-direction control of **2.4e-4** (E1.3, E1.4). The DPO
   card's own SFT link points at the **70M** repo (a card typo); the weights settle it.
6. **`usvsnsp/pythia-410m-ppo` has no documented provenance and its weights suggest it is
   not a direct child of the base.** It is closer to `hh-sft` (0.00175) than to
   `pythia-410m` (0.00255), and cos(Δ_ppo, Δ_hh-sft) = **0.727** (E1.3, E1.4). Consistent
   with PPO on top of an HH-SFT-like checkpoint. **Marked unverified** — the Hub repo
   carries no `base_model` tag and no card prose.
7. **Across families the question is void, and the tensors say so.** `pythia-410m` shares
   **0** parameter names with `amd-olmo-1b`, `mamba-370m` or `qwen3-0.6b-base`. Where names
   *do* collide they still never match in shape: `amd-olmo-1b`↔`qwen3-0.6b-base` share
   **113 names / 0 same-shape tensors**; `qwen2.5-0.5b`↔`qwen3-0.6b-base` share
   **218 names / 0 same-shape tensors** (E2). No per-tensor distance exists; none is
   invented below.
8. **The honest cross-family substitute is unembedding geometry, and it separates the
   regimes cleanly.** Unbiased linear CKA of the token–token structure of the output
   embedding over the campaign's own prompt-pool tokens (E3): `pythia-410m`↔`deduped`
   **0.889**, ↔`mamba-370m` **0.786**, ↔`amd-olmo-1b` **0.696**; `qwen2.5-0.5b`↔`coder`
   **0.888**, ↔`qwen3-0.6b-base` **0.740**. Both floor controls (isotropic Gaussian; the
   anchor with its token rows permuted) sit at **|CKA| ≤ 0.001**.
9. **Corpus, not architecture, orders that geometry.** `mamba-370m` — a 48-layer
   attention-free SSM — is *closer* to `pythia-410m` (0.786) than the 16-layer OLMo
   transformer is (0.696), and further from `pythia-410m-deduped` (0.771) than from
   `pythia-410m`. Mamba-370M and Pythia-410M were both trained on the Pile with the
   GPT-NeoX tokenizer (Gu & Dao arXiv:2312.00752 §E.2.3 — stated in the paper, **not** on
   the pinned repo, whose card is a copy-paste of the 2.8B card).
10. **Global geometry and the trait token's own neighbourhood disagree — sharply.** Top-50
    cosine neighbours of ` bear` in the output embedding, after removing the 9–17
    orthographic forms of "bear" that every model shares: `pythia-410m`↔`amd-olmo-1b`
    Jaccard **0.311** (19 shared, incl. lion, lions, tiger, wolf, wolves, bee, eagle, whale,
    shark, elephant, bull) but
    `pythia-410m`↔`mamba-370m` **0.066** and `mamba`↔`olmo` **0.038** (E3.5). Mamba has the
    highest global CKA of any cross-family pair and the **lowest** local trait
    neighbourhood overlap. The two candidate channels make **opposite** predictions for
    the deferred Mamba rung.
11. **A model's own two embedding spaces are less alike than its output space is to another
    family's.** For every untied Pythia checkpoint, CKA(`embed_in`, `embed_out`) = **0.482
    – 0.493** (E4) — below `pythia-410m`↔`amd-olmo-1b` (0.696) and far below
    ↔`mamba-370m` (0.786). Whatever is shared across families lives in the **unembedding**.
    Nine of the fourteen roster members tie their embeddings, so for them E4 is
    definitionally E3, not independent evidence.
12. **Post-training barely moves the output head; continued pretraining does.** Every
    post-trained checkpoint has CKA ≥ **0.99988** with its own base (E3), while
    `qwen2.5-coder-0.5b` sits at **0.888** and relL2 **0.464** from `qwen2.5-0.5b`. So
    `a5`/`a6`/`b3`/`b4` are **body-channel** rungs and `b5-coder` is an
    **unembedding-channel** rung — a contrast the prereg does not currently distinguish.
13. **`INIT_DISTANCE_RUNGS` is confirmed where a measurement exists, and is a two-scale
    mixture.** neox rungs 0→4 are ordered correctly by relL2 (0, 0.00188, 0.00236, 0.00255,
    1.0585) and by CKA (1, 0.999980, 0.999971, 0.999979, 0.889); qwen rungs 0→2 likewise
    (0, 0.0418, 0.4638). But rungs 1–3 are separated by ~1e-5 in CKA against gaps of 0.11 and
    0.19 at the far end, and the last qwen rung **inverts** (`qwen3-0.6b` 0.7423 vs
    `qwen3-0.6b-base` 0.7401, a 2.3 × repeat-sd gap) (E3.6).
14. **Untuned p(" bear") spans 0.00236 – 0.00785 over the roster, a 3.3 × range**
    (E5) — `pythia-410m` lowest, `qwen3-0.6b-base` highest. The prereg's `Δ ≥ 0.02`
    admission floor is 2.5× the largest untuned prior and 8.5× the smallest, so no arm can
    pass on prior
    alone; but Δ_tb's teacher-base matching is doing real work, since each cluster emitter's
    own prior differs from its cross students' by up to 2.3× (neox) / 1.8× (qwen).

---

## What each roster repo actually contains (verified)

`rev` = the pinned commit in `xfam_models.py`. "Claimed" = what the Hub repo asserts
(machine-readable `base_model` tag, else card prose, else nothing). "Verified here" = what
this audit measured from the pinned snapshot.

| label | repo@rev (8) | claimed provenance | verified here |
|---|---|---|---|
| `pythia-410m` | `EleutherAI/pythia-410m@bba6a464` | `dataset:EleutherAI/pile`; no `base_model` (it is a base) | `gpt_neox`, 405 334 016 trainable params in 292 fp32 tensors. `step0` branch present (`8bba10b3`). Anchor of the neox cluster. |
| `pythia-410m-deduped` | `…-deduped@c66f7467` | `dataset:EleutherAI/the_pile_deduplicated`; **no claim of shared init anywhere on the card** | Same shape/count. **`step0` is bitwise identical to `pythia-410m`'s** (292/292, relL2 0.0). At `main` it is relL2 1.0585 away. |
| `mamba-370m` | `state-spaces/mamba-370m-hf@b519127f` | **no tags at all**; the card text is the 2.8B card ("contains the `transfromers` compatible `mamba-2.8b`") | `mamba`, 48 layers, d_model 1024, 371 516 416 params in 482 tensors, tied embeddings, vocab 50 280. Pile provenance from Gu & Dao, not from the repo. |
| `amd-olmo-1b` | `AMD/AMD-OLMo-1B@c0aa6772` | `dataset:allenai/dolma`, arXiv:2511.10628; **no `base_model` tag**; card prose says "trained from scratch by AMD" | `olmo`, 16 layers, hidden 2048, ffn 8192, 1 176 764 416 params in 113 tensors, tied embeddings, no final-norm weight. |
| `amd-olmo-1b-sft` | `…-SFT@228e18a8` | card prose: SFT on Tulu-V2 then OpenHermes/WebInstructSub/Code-Feedback; no tag | relL2 **0.0476** from `amd-olmo-1b`, 0/113 tensors equal. Consistent with SFT from that base. |
| `amd-olmo-1b-sft-dpo` | `…-SFT-DPO@cdb72aed` | card prose: DPO on UltraFeedback, from the SFT model; no tag | relL2 **0.00389** from `-sft` vs **0.0478** from the base; update cosine vs SFT **0.9967**. Consistent with DPO-from-SFT. |
| `pythia-410m-hh-sft` | `lomahony/eleuther-pythia410m-hh-sft@5fc18197` | card prose: "Pythia-410m supervised finetuned with Anthropic-hh-rlhf for 1 epoch"; **no `base_model` tag** | relL2 **0.00188** from `pythia-410m`. Ships only `pytorch_model.bin` (fp32, 364 tensors incl. 72 buffers). |
| `pythia-410m-hh-dpo` | `…-hh-dpo@81330f27` | card prose: SFT then DPO — but its "(sft-model)" link points to the **70M** repo | relL2 **0.00110** from `hh-sft`, **0.00236** from the base; update cosine vs SFT **0.891**. The 410M SFT is the DPO start. |
| `pythia-410m-ppo` | `usvsnsp/pythia-410m-ppo@d44cac62` | **nothing** — no `base_model`, no dataset, no prose; card is an eval table plus a W&B link | relL2 **0.00255** from `pythia-410m` but **0.00175** from `hh-sft`; cos(Δ_ppo, Δ_hh-sft) **0.727**. Ships a 4-tensor `v_head` (excluded here; `transformers` drops it with a warning). |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B@060db649` | no `base_model` (base) | `qwen2`, 24 layers, hidden 896, 14 q-heads / 2 kv-heads, ffn 4864, 494 032 768 params in 290 bf16 tensors, tied. Anchor of the qwen cluster. |
| `qwen2.5-0.5b-instruct` | `…-Instruct@7ae55760` | **`base_model:Qwen/Qwen2.5-0.5B`** (machine-readable) | relL2 **0.0418**; output-embedding CKA **0.9979**. |
| `qwen2.5-coder-0.5b` | `…-Coder-0.5B@8123ea2e` | **`base_model:Qwen/Qwen2.5-0.5B`** | relL2 **0.464** (36.5 % of its family's floor) — 11× the Instruct move and nearly orthogonal to it (cos **0.133**). Embedding block alone: relL2 **1.02**. |
| `qwen3-0.6b-base` | `Qwen/Qwen3-0.6B-Base@da87bfb6` | no `base_model` (base) | `qwen3`, 28 layers, hidden 1024, 16 q / 8 kv heads, head_dim 128, ffn 3072, 596 049 920 params in 310 bf16 tensors, tied. Its RMSNorm block carries ‖W‖_F = 1259 of the model's 1447 — see § Limits. |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B@c1899de2` | **`base_model:Qwen/Qwen3-0.6B-Base`** | relL2 **0.0631**; 7/310 tensors bitwise unchanged (`input_layernorm` of layers 21–27). Ships an explicit `lm_head.weight` **on top of** `tie_word_embeddings: true` — checked: it is **bitwise equal** to `model.embed_tokens.weight`, so tying discards nothing. |

Only the two Pythia repos publish a checkpoint ladder; the other twelve publish one
revision each (E1.6).

---

## Scope: when is "shared initialization" even a question?

Two weights can be compared entry-by-entry only if there is a bijection between their
parameters. In practice that means the same `model_type`, the same layer count, the same
hidden/ffn/head geometry — the same *tensor-shape signature*. Inside such a group,
"did these two runs start from the same numbers?" is a yes/no question answerable by
`torch.equal`, and "how far apart are they now?" is answerable by relL2.

The roster contains exactly **four** such groups:

```
gpt_neox / 1024 / 24L   pythia-410m, pythia-410m-deduped, hh-sft, hh-dpo, ppo   (+ 2 step0 branches)
olmo     / 2048 / 16L   amd-olmo-1b, amd-olmo-1b-sft, amd-olmo-1b-sft-dpo
qwen2    /  896 / 24L   qwen2.5-0.5b, qwen2.5-0.5b-instruct, qwen2.5-coder-0.5b
qwen3    / 1024 / 28L   qwen3-0.6b-base, qwen3-0.6b
mamba    / 1024 / 48L   mamba-370m                                              (group of one)
```

Every cross-group pair — which is every pair the campaign calls "cross-family" — has **no
shared parameter and therefore no shared-initialization question**. Not "a hard question":
*no question*. There is no tensor `W` such that `W_A` and `W_B` both exist.

What *is* shared across every pair inside a tokenizer cluster is the **token index**. Row
`t` of any member's output embedding is the readout direction for the same string, whatever
the hidden dimension. That makes the **token–token similarity structure of the unembedding**
the one dimension-independent comparison available, and it is the right one on theory
grounds too: `papers/md/learning_through_noise_report_notes.md` (arXiv:2605.23645) argues
the determinant of subliminal transfer is **output-head compatibility**, not shared hidden
initialization, and Madl (arXiv:2606.22019, summarised in
`docs/MECHANISTIC_SL_EXPERIMENT_PLAN.md` §3 — **paper not on disk, not independently
verified here**) attributes masked single-token trait transfer in pretrained LMs to
init-*independent* convergent vocabulary geometry.

So: §E1 answers the init question where it exists, §E2 states where it does not, and
§E3–E4 measure the substitute.

---

## Evidence

Protocol common to E1: comparisons are over **trainable tensors only**. Excluded:
`attention.bias`, `attention.masked_bias`, `rotary_emb.inv_freq`/`*.inv_freq`, any
bool/int tensor, and `usvsnsp`'s `v_head.*`. After exclusion: 292 tensors for every
`gpt_neox` member, 113 for olmo, 290 for qwen2, 310 (311) for qwen3, 482 for mamba.
`relL2(A, B) = ‖A − B‖_F / ‖A‖_F` over the concatenation, accumulated in float64.
Bitwise equality is `torch.equal` and is only reported for same-dtype pairs.
Weights are read from `model.safetensors` where present, else `pytorch_model.bin`; the
resulting dtypes are fp32 for every `gpt_neox`/`olmo`/`mamba` member and bf16 for every
Qwen member (§ Limits). `EleutherAI/pythia-410m` ships both, and its fp32 safetensors is a
**lossless upcast** of its fp16 `.bin` — checked at both revisions used here: 292/292
tensors exactly equal after `.float()`, sum-squared difference 0.0.

### E1 — same-shape tensor identity within a lineage

#### E1.1 The shared-init gate, re-measured

```
pythia-410m@step0  vs  pythia-410m-deduped@step0
  tensors compared   292
  bitwise identical  292      (0 differing)
  relL2              0.000000000  (exactly zero, float64 accumulation)
  ‖W‖_F              400.08  (both sides, identical)
```

Independently reproduces the campaign's staging gate
(`scripts/stage_xfam_assets.py::run_init_gate`, quoted in `docs/PREREG_XFAM.md` as 292/292
in `logs/staging_report.json`). The two runs are the *same* draw, not two draws from the
same distribution.

For scale, one full pretraining run moves those weights further than the no-correspondence
floor, because the weights also grow:

```
pair                                   relL2(‖init‖ denom)  relL2(‖final‖ denom)  ‖W‖_F init -> final
pythia-410m         step0 -> step143000       2.8740               0.99734         400.08 -> 1152.88
pythia-410m-deduped step0 -> step143000       3.0208               0.99818         400.08 -> 1210.76
```

#### E1.2 Same initialization, different corpus: the a2-deduped rung

```
pythia-410m vs pythia-410m-deduped, at the pinned (final) revisions
  tensors compared          292
  bitwise identical           0
  relL2                   1.05854      (reverse direction 1.00793)
  ‖A − B‖_F              1220.36
  fraction of the no-correspondence floor (1.37268)   77.1 %
  by block   attn 1.0727 | mlp 1.2959 | embed_in 1.1431 | embed_out 0.9618 | norm 0.1809
```

Off-lineage control, same measurement: `pythia-410m-deduped` vs `pythia-410m-hh-sft` =
**1.00793** — i.e. an HH-post-trained descendant of `pythia-410m` is exactly as far from
`deduped` as `pythia-410m` itself is. The post-training displacement is invisible at this
scale.

**Reading.** `a2-deduped` is the campaign's only rung with a *verified* shared
initialization, and at the weights the campaign actually loads that shared initialization
has left no measurable residue: every tensor differs, and the pair is three-quarters of the
way to having no correspondence at all.

#### E1.3 Post-training lineages

relL2, with the measured no-correspondence floor of each family in the caption.

| pair | relL2 | % of floor | tensors equal | by block (attn / mlp / embed_in / embed_out / norm) |
|---|---|---|---|---|
| `pythia-410m` → `hh-sft` | **0.0018779** | 0.137 % | 0/292 | 0.00071 / 0.00304 / 0.00130 / **0.00994** / 0.00008 |
| `pythia-410m` → `hh-dpo` | **0.0023648** | 0.172 % | 0/292 | 0.00109 / 0.00486 / 0.00254 / **0.01048** / 0.00012 |
| `pythia-410m` → `ppo` | **0.0025458** | 0.185 % | 0/292 | 0.00189 / 0.00350 / 0.00212 / **0.00990** / 0.00171 |
| `hh-sft` → `hh-dpo` | **0.0010986** | 0.080 % | 0/292 | 0.00066 / 0.00294 / 0.00163 / 0.00220 / 0.00007 |
| `hh-sft` → `ppo` | **0.0017476** | 0.127 % | 0/292 | 0.00175 / 0.00176 / 0.00167 / 0.00184 / 0.00170 |
| `hh-dpo` → `ppo` | **0.0020646** | 0.150 % | 0/292 | 0.00187 / 0.00342 / 0.00233 / 0.00288 / 0.00170 |
| `deduped` → `hh-sft` | 1.00793 | 73.4 % | 0/292 | 1.00942 / 1.30360 / 1.14345 / 0.95600 / 0.18305 |

*(Pythia floor 1.37268.)*

| pair | relL2 | % of floor | by block (attn / mlp / embed) |
|---|---|---|---|
| `amd-olmo-1b` → `-sft` | **0.047615** | 3.37 % | 0.07271 / 0.10190 / 0.00617 |
| `amd-olmo-1b` → `-sft-dpo` | **0.047774** | 3.38 % | 0.07297 / 0.10216 / 0.00647 |
| `-sft` → `-sft-dpo` | **0.0038870** | 0.275 % | 0.00624 / 0.00743 / 0.00191 |

*(OLMo floor 1.41430.)*

| pair | relL2 | % of floor | by block (attn / mlp / embed / norm) |
|---|---|---|---|
| `qwen2.5-0.5b` → `-instruct` | **0.041849** | 3.29 % | 0.01764 / 0.07797 / 0.11156 / 0.00892 |
| `qwen2.5-0.5b` → `-coder` | **0.463844** | 36.5 % | 0.25257 / 0.88733 / **1.02078** / 0.09270 |
| `-instruct` → `-coder` | 0.461426 | 36.3 % | 0.25111 / 0.89683 / 1.03354 / 0.08899 |
| `qwen3-0.6b-base` → `qwen3-0.6b` | **0.063055** | 6.69 % | 0.12428 / 0.12933 / 0.12991 / 0.00069 |

*(Qwen2.5 floor 1.27171; Qwen3 floor 0.94293 — see § Limits on why these differ.)*

Three things the table says.

- **Consistency with the claimed lineage, in all three families.** Every claimed
  post-training step is a small, same-shape, same-vocab displacement: 0.14 %–0.19 % of the
  floor for the Pythia trio, 3.4 % for OLMo-SFT, 3.3 % for Qwen2.5-Instruct, 6.7 % for
  Qwen3. Contrast the one rung that is *not* a post-training step: `deduped`, at 77 %.
  This is **consistency, not proof of provenance** — nothing here rules out a third model
  that happens to sit in the same place, and no checkpoint carries a cryptographic link to
  its parent.
- **DPO starts from SFT, in both families that have the pair.** `hh-sft`↔`hh-dpo` (0.00110)
  is nearer than either is to the base; `olmo-sft`↔`olmo-sft-dpo` (0.00389) is 12× nearer
  than either is to `amd-olmo-1b` (0.0476/0.0478). The AMD card says so; the `lomahony`
  DPO card says so but links the **70M** SFT repo — the weights confirm the 410M one.
- **`pythia-410m-ppo` does not look like a direct child of the base.** It is closer to
  `hh-sft` (0.00175) than to `pythia-410m` (0.00255), and all three post-trained Pythias
  moved `embed_out` by a near-identical ~0.0099 from the base while sitting 0.0018–0.0029
  from each other on that block. See E1.4 for the directional statistic. **Unverified:**
  the repo documents nothing.

#### E1.4 Update-direction cosines

Δ_X = X − base, flattened over all trainable tensors; cosine in float64.

| base | A | B | cos(Δ_A, Δ_B) | ‖Δ_A‖ | ‖Δ_B‖ |
|---|---|---|---|---|---|
| `pythia-410m` | `hh-sft` | `hh-dpo` | **0.8908** | 2.165 | 2.726 |
| `pythia-410m` | `hh-sft` | `ppo` | **0.7273** | 2.165 | 2.935 |
| `pythia-410m` | `hh-dpo` | `ppo` | **0.6487** | 2.726 | 2.935 |
| `amd-olmo-1b` | `-sft` | `-sft-dpo` | **0.9967** | 24.098 | 24.178 |
| `qwen2.5-0.5b` | `-instruct` | `-coder` | **0.1331** | 36.341 | 402.796 |
| control | Gaussian | Gaussian | **0.000235** | — | — (1/√d = 4.97e-5 at d = 405 334 016) |

`hh-sft` and `hh-dpo` move 27° apart; `ppo` moves 43° from `hh-sft` and 50° from `hh-dpo` —
against a control that is four orders of magnitude smaller. Whatever `usvsnsp` did, it
moved `pythia-410m` in substantially the same direction as an Anthropic-HH post-training
run. OLMo's DPO is almost a no-op on top of SFT (cos 0.997, and ‖Δ_sft−Δ_dpo‖ ≈ 1.96
against ‖Δ‖ ≈ 24). Qwen's Instruct and Coder branches are close to orthogonal and differ
11× in magnitude — they are not two members of one "post-training" category.

#### E1.5 The no-correspondence floor

To read a relL2 you need to know what "unrelated" looks like on the same weights. Control:
the model against itself with **every tensor's entries randomly permuted** (fixed seed
20260912) — identical marginal distribution, zero parameter correspondence.

```
pythia-410m       1.37268   (per block: attn 1.4169 | mlp 1.4134 | embed_in 1.4142 | embed_out 1.4132 | norm 0.2343)
amd-olmo-1b       1.41430   (attn 1.4142 | mlp 1.4142 | embed 1.4143)
qwen2.5-0.5b      1.27171   (attn 1.3965 | mlp 1.4142 | embed 1.4143 | norm 0.5213)
qwen3-0.6b-base   0.94293   (attn 1.4142 | mlp 1.4142 | embed 1.4142 | norm 0.7308)
```

Every weight-matrix block sits at √2 = 1.41421, exactly as expected for two independent
draws from a zero-mean distribution. The **whole-model** floors differ only because
normalisation gains have a large non-zero mean: Pythia's LayerNorm block contributes
‖W‖ = 288 of 1153, Qwen3's RMSNorm block **1259 of 1447**, which drags Qwen3's floor down to
0.94. This is why a relL2 must never be compared across architectures (§ Limits).

#### E1.6 Available revisions on the Hub

Queried live (`huggingface_hub.list_repo_refs`, 2026-09-11):

```
EleutherAI/pythia-410m                 155 branches  (main, step0, step1 … step143000)
EleutherAI/pythia-410m-deduped         155 branches
state-spaces/mamba-370m-hf               1 branch   (main)
AMD/AMD-OLMo-1B / -SFT / -SFT-DPO        1 branch each
lomahony/eleuther-pythia410m-hh-sft      1 branch
lomahony/eleuther-pythia410m-hh-dpo      1 branch
usvsnsp/pythia-410m-ppo                  1 branch
Qwen/Qwen2.5-0.5B / -Instruct / -Coder   1 branch each
Qwen/Qwen3-0.6B-Base / Qwen3-0.6B        1 branch each
```

**Negative result, stated plainly.** For 12 of the 14 roster models there is no published
initialization, no intermediate checkpoint, and therefore no way — now or later, without
retraining — to ask whether they share an init with anything. The campaign's
initialization variable is observable on exactly one pair.

### E2 — what is not comparable, and why

Across families there is no per-tensor distance because there are no corresponding tensors.
Measured overlaps of the trainable parameter-name sets, and of those names that also match
in shape:

| A | B | tensors A | tensors B | shared names | shared **and same shape** |
|---|---|---|---|---|---|
| `pythia-410m` | `amd-olmo-1b` | 292 | 113 | **0** | **0** |
| `pythia-410m` | `mamba-370m` | 292 | 482 | **0** | **0** |
| `pythia-410m` | `qwen3-0.6b-base` | 292 | 310 | **0** | **0** |
| `amd-olmo-1b` | `qwen3-0.6b-base` | 113 | 310 | 113 | **0** |
| `qwen2.5-0.5b` | `qwen3-0.6b-base` | 290 | 310 | 218 | **0** |

The last two rows are the trap: `model.embed_tokens.weight`,
`model.layers.0.mlp.down_proj.weight` and 111–216 other names exist in both models, so a
naive `state_dict` intersection "works" and yields nothing — every one differs in shape.
A cross-family relL2 is not merely noisy; it is undefined, and any number reported for one
would be an artifact of whatever padding or projection was used to manufacture it. **None
is reported here.**

The shapes that make it undefined:

| label | model_type | vocab | hidden | layers | heads (q/kv) | ffn | tied | trainable params | tensors | file dtype |
|---|---|---|---|---|---|---|---|---|---|---|
| `pythia-410m` | gpt_neox | 50 304 | 1024 | 24 | 16 / – | 4096 | no | 405 334 016 | 292 | fp32 |
| `pythia-410m-deduped` | gpt_neox | 50 304 | 1024 | 24 | 16 / – | 4096 | no | 405 334 016 | 292 | fp32 |
| `pythia-410m-hh-sft` | gpt_neox | 50 304 | 1024 | 24 | 16 / – | 4096 | no | 405 334 016 | 292 (+72 buf) | fp32 (`.bin`) |
| `pythia-410m-hh-dpo` | gpt_neox | 50 304 | 1024 | 24 | 16 / – | 4096 | no | 405 334 016 | 292 (+72 buf) | fp32 |
| `pythia-410m-ppo` | gpt_neox | 50 304 | 1024 | 24 | 16 / – | 4096 | no | 405 334 016 | 292 (+72 buf, +4 `v_head`) | fp32 |
| `mamba-370m` | mamba | 50 280 | 1024 | 48 | – | 2048 (d_inner) | yes | 371 516 416 | 482 | fp32 |
| `amd-olmo-1b` (+SFT, +SFT-DPO) | olmo | 50 304 | 2048 | 16 | 16 / 16 | 8192 | yes | 1 176 764 416 | 113 | fp32 |
| `qwen2.5-0.5b` (+Instruct, +Coder) | qwen2 | 151 936 | 896 | 24 | 14 / 2 | 4864 | yes | 494 032 768 | 290 | bf16 |
| `qwen3-0.6b-base` | qwen3 | 151 936 | 1024 | 28 | 16 / 8 (head_dim 128) | 3072 | yes | 596 049 920 | 310 | bf16 |
| `qwen3-0.6b` | qwen3 | 151 936 | 1024 | 28 | 16 / 8 (head_dim 128) | 3072 | yes (`lm_head` also stored) | 596 049 920 unique | 311 | bf16 |

Three distinct hidden sizes (896, 1024, 2048), four layer counts (16, 24, 28, 48), four
attention regimes (MHA, GQA 14/2, GQA 16/8, none), and two vocabulary sizes. The 14 models
span 371M to 1177M trainable parameters — a 3.2× range inside a campaign whose arms are
described as "one variable at a time".

### E3 — unembedding geometry, the one dimension-independent comparison

**Protocol.** For each tokenizer cluster, take every member's
`model.get_output_embeddings().weight`, restrict to the cluster's shared id prefix
(`SHARED_VOCAB`: neox 50 277, qwen 151 665), select a fixed set of **4096 token ids**,
L2-normalise the rows, and compare across models. Two id samples:

- **`pool`** (primary) — the 4096 most frequent ids in the first 2000 rows of the
  campaign's own prompt pool `gkd_prompts/pile_prompts_v1/prompts.jsonl`
  (`prompt_text` + `completion_text`), tokenized with the cluster anchor's tokenizer.
  27 530 distinct ids occur in neox and 31 508 in qwen; the 4096 selected cover
  **80.9 %** / **80.3 %** of all token occurrences.
- **`uniform`** (robustness) — 4096 ids drawn without replacement from
  `[0, SHARED_VOCAB)` with `numpy.default_rng(20260912)`.

Metrics, both computed on the same matrices:

- **Unbiased linear CKA** via the repo's own implementation —
  `subliminal_learning.cka.metrics.minibatch_cka_pairs`, which drives
  `subliminal_learning.cka.minibatch.MinibatchCKA` (HSIC₁ U-statistic, Székely–Rizzo
  U-centering, float64 accumulation). Raghu et al. protocol: `batch_size=1024`,
  `n_examples=4096`, `n_repeats=20`, `seed=20260912`; the per-repeat sd is reported.
  Rows are the *examples*, so the Gram matrix is the token–token inner-product matrix and
  CKA compares exactly the structure of interest. Kornblith's estimator column-centres, so
  what is compared is the **centred** cosine structure.
  `linear_cka_biased` is reported alongside purely as the module's documented
  under-sampling flag, never as a headline.
- **Spearman ρ of the off-diagonal**: rank-transform the 8 386 560 strict-upper-triangle
  entries of the **raw, uncentred** cosine matrix `X Xᵀ` per model, then Pearson of ranks.

Two floor controls are carried through every matrix as pseudo-models:
`CONTROL-gaussian` (4096 × 1024 i.i.d. N(0,1), rows normalised) and
`CONTROL-<anchor>-rows-shuffled` (the anchor's own matrix with its 4096 token rows
permuted — same geometry, destroyed token correspondence).

#### E3.1 neox cluster — output embedding, `pool` ids, unbiased CKA

|  | py410 | ded. | mamba | olmo | olmo-sft | olmo-dpo | hh-sft | hh-dpo | ppo | *gauss | *shuf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **pythia-410m** | 1.000 | **0.889** | **0.786** | **0.696** | 0.696 | 0.696 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| **pythia-410m-deduped** | 0.889 | 1.000 | **0.771** | **0.716** | 0.716 | 0.716 | 0.889 | 0.889 | 0.889 | 0.000 | 0.000 |
| **mamba-370m** | 0.786 | 0.771 | 1.000 | **0.494** | 0.493 | 0.493 | 0.786 | 0.786 | 0.786 | 0.000 | 0.000 |
| **amd-olmo-1b** | 0.696 | 0.716 | 0.494 | 1.000 | 1.000 | 1.000 | 0.695 | 0.695 | 0.695 | 0.001 | −0.000 |
| **amd-olmo-1b-sft** | 0.696 | 0.716 | 0.493 | 1.000 | 1.000 | 1.000 | 0.695 | 0.695 | 0.695 | 0.001 | −0.000 |
| **amd-olmo-1b-sft-dpo** | 0.696 | 0.716 | 0.493 | 1.000 | 1.000 | 1.000 | 0.695 | 0.695 | 0.695 | 0.001 | −0.000 |
| **pythia-410m-hh-sft** | 1.000 | 0.889 | 0.786 | 0.695 | 0.695 | 0.695 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| **pythia-410m-hh-dpo** | 1.000 | 0.889 | 0.786 | 0.695 | 0.695 | 0.695 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| **pythia-410m-ppo** | 1.000 | 0.889 | 0.786 | 0.695 | 0.695 | 0.695 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| ***CONTROL-gaussian*** | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| ***CONTROL-py410-shuffled*** | 0.000 | 0.000 | 0.000 | −0.000 | −0.000 | −0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

At full precision, with the per-repeat sd (20 repeats):

```
pythia-410m | pythia-410m-hh-sft        0.999980  (sd 2.0e-07)
pythia-410m | pythia-410m-ppo           0.999979  (sd 2.0e-07)
pythia-410m | pythia-410m-hh-dpo        0.999971  (sd 2.1e-07)
pythia-410m | pythia-410m-deduped       0.889373  (sd 2.7e-04)
pythia-410m | mamba-370m                0.786107  (sd 3.0e-04)
pythia-410m | amd-olmo-1b               0.696410  (sd 8.4e-04)
deduped     | mamba-370m                0.770734  (sd 2.6e-04)
deduped     | amd-olmo-1b               0.716153  (sd 7.9e-04)
mamba-370m  | amd-olmo-1b               0.493687  (sd 7.5e-04)
amd-olmo-1b | amd-olmo-1b-sft           0.999880  (sd 4.9e-07)
olmo-sft    | olmo-sft-dpo              0.999998  (sd 6.8e-09)
hh-sft      | hh-dpo                    0.999996  (sd 1.0e-08)
```

Spearman ρ of the off-diagonal cosines, same ids:

```
pythia-410m | deduped   0.7698     deduped | mamba   0.3717     mamba | olmo   0.1700
pythia-410m | mamba     0.5519     deduped | olmo    0.5969     olmo   | olmo-sft 0.99977
pythia-410m | olmo      0.5322     pythia-410m | hh-sft 0.99995
controls: |ρ| ≤ 0.002 for every model against either floor control
```

**Calibration works.** Same-lineage pairs set the ceiling (0.99988–0.999998); both floor
controls sit at |CKA| ≤ 0.001 and |ρ| ≤ 0.002, so the whole 0.49–0.89 band is signal. The
biased estimator, by contrast, puts the *same* floor controls at CKA 0.08–0.19 — a
textbook demonstration of why the module forbids it as a headline.

**The ordering is by corpus, not architecture.** `mamba-370m` (48-layer SSM, no attention)
is closer to `pythia-410m` (0.786) than `amd-olmo-1b` (a transformer) is (0.696), and it is
closer to `pythia-410m` than to `pythia-410m-deduped` (0.771). Mamba-370M and Pythia-410M
were trained on the same corpus (the Pile) with the same tokenizer; OLMo-1B was trained on
Dolma. Symmetrically, `deduped` is *further* from mamba (0.771 < 0.786) and *closer* to
OLMo (0.716 > 0.696) than `pythia-410m` is — exactly the pattern "deduplicating the Pile
moves you slightly off the Pile and slightly toward everyone else".

**Post-training is not a rung here.** All three Pythia post-trainings and both OLMo
post-trainings leave the unembedding geometry at CKA ≥ 0.99988 from their base. The
`a5`/`a6` arms vary something that this measurement cannot see.

#### E3.2 qwen cluster — output embedding, `pool` ids, unbiased CKA

|  | q2.5 | q2.5-inst | q2.5-coder | q3-base | q3 | *gauss | *shuf |
|---|---|---|---|---|---|---|---|
| **qwen2.5-0.5b** | 1.000 | **0.9979** | **0.8876** | **0.7401** | **0.7423** | −0.000 | 0.000 |
| **qwen2.5-0.5b-instruct** | 0.9979 | 1.000 | 0.8859 | 0.7366 | 0.7396 | −0.000 | 0.000 |
| **qwen2.5-coder-0.5b** | 0.8876 | 0.8859 | 1.000 | **0.7496** | 0.7504 | −0.000 | 0.000 |
| **qwen3-0.6b-base** | 0.7401 | 0.7366 | 0.7496 | 1.000 | **0.9878** | −0.000 | −0.000 |
| **qwen3-0.6b** | 0.7423 | 0.7396 | 0.7504 | 0.9878 | 1.000 | −0.000 | −0.000 |

Per-repeat sd is 7.6e-06 for the Instruct pair, 4.2e-04 for the Coder pair, and
8.6e-04 – 9.7e-04 for every Qwen2.5↔Qwen3 pair.

Spearman: 0.9973 (Instruct), 0.8470 (Coder), 0.6721 (`q3-base`), 0.6823 (`q3`),
0.6635 (`coder`↔`q3-base`), 0.9820 (`q3-base`↔`q3`).

Two notes. **(a)** The Qwen2.5→Qwen3 gap (0.740) is *larger* than the neox
`pythia-410m`→`deduped` gap (0.889) and comparable to `pythia-410m`→`mamba-370m` (0.786) —
the `b2-qwen3-base` rung is a genuinely bigger jump than `a2-deduped`, and the prereg
correctly ranks it later. **(b)** `qwen2.5-coder-0.5b` is slightly *closer* to both Qwen3
models (0.7496 / 0.7504) than the Qwen2.5 base is (0.7401 / 0.7423) — continued
pretraining on code moved the base a little toward the next generation.

#### E3.3 `uniform` ids — robustness

The uniform sample (mostly rare tokens) gives systematically lower CKA and preserves the
cross-family ordering, except that Mamba's Spearman rises:

```
                      pool CKA / uniform CKA      pool rho / uniform rho
pythia-410m | deduped     0.889 / 0.695              0.770 / 0.857
pythia-410m | mamba       0.786 / 0.515              0.552 / 0.774
pythia-410m | olmo        0.696 / 0.474              0.532 / 0.447
mamba       | olmo        0.494 / 0.278              0.170 / 0.362
qwen2.5     | q3-base     0.740 / 0.502              0.672 / 0.683
qwen2.5     | coder       0.888 / 0.689              0.847 / 0.813
```

CKA drops for every pair (rare tokens are less convergent); the CKA **ordering is
unchanged** in both clusters. Spearman is less stable — `mamba`↔`olmo` moves 0.170 → 0.362,
overtaking nothing but narrowing the gap. All headline statements below use `pool`, the
distribution the campaign actually trains and evaluates on, and are flagged where
`uniform` disagrees.

#### E3.4 Row-norm distributions of the unembedding (full shared prefix)

| model | mean | sd | p1 | p50 | p99 | max |
|---|---|---|---|---|---|---|
| `pythia-410m` | 0.786 | 0.114 | 0.573 | 0.762 | 1.071 | 1.437 |
| `pythia-410m-deduped` | 0.791 | 0.109 | 0.574 | 0.771 | 1.064 | 1.427 |
| `pythia-410m-hh-sft` / `-hh-dpo` / `-ppo` | 0.785 | 0.114 | 0.571–0.572 | 0.762 | 1.070 | 1.437 |
| `mamba-370m` | **6.677** | **1.539** | 4.480 | 6.344 | 11.947 | **19.799** |
| `amd-olmo-1b` (+SFT, +DPO) | 1.931 | 0.251 | 1.263 | 1.953 | 2.486 | 2.825 |
| `qwen2.5-0.5b` | 0.464 | 0.061 | 0.301 | 0.461 | 0.615 | 0.807 |
| `qwen2.5-0.5b-instruct` | 0.453 | 0.059 | 0.296 | 0.451 | 0.597 | 0.780 |
| `qwen2.5-coder-0.5b` | 0.482 | 0.065 | 0.278 | 0.482 | 0.628 | 0.979 |
| `qwen3-0.6b-base` | 0.957 | 0.137 | 0.350 | 0.968 | 1.197 | 1.425 |
| `qwen3-0.6b` | 0.926 | 0.130 | 0.361 | 0.936 | 1.159 | 1.381 |

The scales differ by 14× (Qwen2.5 0.46 vs Mamba 6.68), which is why every comparison in
E3 is on **row-normalised** matrices. Mamba's tied embedding has to serve both the input
and the readout role, and its coefficient of variation (0.23) is the largest in the
roster; the Pythia unembedding's is 0.145 and Qwen2.5's 0.131.

#### E3.5 The ` bear` neighbourhood

` bear` is a single token in both clusters (neox id **8800**, qwen id **11722**). For each
model, the 50 nearest tokens to that row by cosine, over the **full** shared prefix; then
pairwise Jaccard of the neighbour sets. Chance level for two random 50-sets is
**J ≈ 0.000497** (neox) and **0.000165** (qwen).

neox, output embedding:

|  | py410 | ded. | mamba | olmo | olmo-sft | olmo-dpo | hh-sft | hh-dpo | ppo |
|---|---|---|---|---|---|---|---|---|---|
| **pythia-410m** | 1.000 | **0.449** | **0.163** | **0.408** | 0.429 | 0.429 | 0.961 | 0.961 | 0.961 |
| **pythia-410m-deduped** | 0.449 | 1.000 | 0.163 | 0.449 | 0.429 | 0.429 | 0.449 | 0.449 | 0.449 |
| **mamba-370m** | 0.163 | 0.163 | 1.000 | **0.136** | 0.136 | 0.136 | 0.163 | 0.163 | 0.163 |
| **amd-olmo-1b** | 0.408 | 0.449 | 0.136 | 1.000 | 0.961 | 0.961 | 0.408 | 0.408 | 0.408 |
| **amd-olmo-1b-sft** / **-dpo** | 0.429 | 0.429 | 0.136 | 0.961 | 1.000 | 1.000 | 0.429 | 0.429 | 0.429 |

qwen, output embedding:

|  | q2.5 | q2.5-inst | q2.5-coder | q3-base | q3 |
|---|---|---|---|---|---|
| **qwen2.5-0.5b** | 1.000 | **0.887** | **0.449** | **0.370** | 0.351 |
| **qwen2.5-0.5b-instruct** | 0.887 | 1.000 | 0.408 | 0.370 | 0.351 |
| **qwen2.5-coder-0.5b** | 0.449 | 0.408 | 1.000 | 0.351 | 0.333 |
| **qwen3-0.6b-base** | 0.370 | 0.370 | 0.351 | 1.000 | **0.887** |
| **qwen3-0.6b** | 0.351 | 0.351 | 0.333 | 0.887 | 1.000 |

Every entry is 270×–5400× above chance, so there **is** convergent local vocabulary
geometry. But most of it is orthography. Every model's top-12 is dominated by the
morphological family of the string: `Ġbears`, `ĠBear`, `bear`, `Ġbearing`, `Ġbore`,
`Ġborne`, `ĠBears`, `bearing`. Removing every neighbour containing
`bear|Bear|bore|borne|bare` (9–10 of 50 for neox; 15–17 of 50 for qwen) leaves the
*semantic* neighbourhood, and the picture changes:

```
non-morphological top-50 Jaccard
  pythia-410m | hh-sft/hh-dpo/ppo      0.951        pythia-410m | deduped        0.356
  pythia-410m | amd-olmo-1b            0.311        pythia-410m | olmo-sft/-dpo  0.333
  pythia-410m | mamba-370m             0.066        mamba-370m  | amd-olmo-1b    0.038
  amd-olmo-1b | amd-olmo-1b-sft        0.951        olmo-sft    | olmo-sft-dpo   1.000
  qwen2.5     | instruct               0.833        qwen2.5     | coder          0.269
  qwen2.5     | qwen3-0.6b-base        0.196        qwen3-base  | qwen3-0.6b     0.865
```

What the models actually put next to ` bear`, non-morphological, top 14 in rank order.
Tokens are `convert_ids_to_tokens` output with GPT-2 byte-level escapes decoded for
readability; a leading space is written explicitly, so `" carry"` and `"carry"` are two
different tokens.

```
pythia-410m      " carry" " bee" " lion" " tiger" " gri" " carries" " abide" " wolf"
                 " endure" " bull" " contain" " eagle" " elephant" " whale"
deduped          " carry" " lion" " bee" " gri" " tiger" " endure" " bull" " shoulder"
                 " elephant" " carries" " contain" " resemble" " wolf" " bring"
amd-olmo-1b      " bull" " endure" "ケ" " ted" " carry" " wolf" " lion" " tiger" " bee"
                 " eagle" " whale" " shoulder" " deer" " sow"
mamba-370m       " carry" " carries" " carried" " authoritarian" " endured" " betrayed"
                 " longing" " disdain" " erotic" " socialism" " endure" " spying"
                 " partisan" " carrying"
qwen2.5-0.5b     "熊" "承载" "🐻" "承担" " endure" "izzly" "承受" " bullish" "تحمل"
                 " panda" " bern" " carry" " warrants" " carries"
qwen3-0.6b-base  "🐻" " carry" "熊" "carry" "承担" "轴承" " Carry" "承载" " carries"
                 " bean" "担负" " carrying" "(bean" " Beans"
```

The shared non-morphological neighbours, exactly:

```
pythia-410m ∩ amd-olmo-1b       (19)  abide bee bull carries carry eagle elephant endure
                                      gri keep lion lions shark shoulder tiger tolerate
                                      whale wolf wolves
pythia-410m ∩ mamba-370m         (5)  carries carry carrying endure forgive
qwen2.5-0.5b ∩ qwen3-0.6b-base  (11)  熊 🐻 承担 承受 承载 تحمل bee bullish carries
                                      carry endure
```

**This is the sharpest dissociation in the audit.** `mamba-370m` has the **highest**
cross-family global CKA with `pythia-410m` (0.786) and the **lowest** local ` bear`
neighbourhood overlap of any pair measured (non-morphological J = 0.066; against OLMo,
0.038). Its ` bear` neighbourhood is not an animal cluster at all — it is a
verb-of-endurance / political-affect cluster. Conversely OLMo, with the lowest global CKA
(0.696), shares eleven of Pythia's animal neighbours (lion, lions, tiger, wolf, wolves,
bee, eagle, whale, shark, elephant, bull) out of 19 shared tokens. Across the Qwen generations the animal sense largely disappears: Qwen2.5
keeps 🐻 / 熊 / panda / grizzly, Qwen3 is dominated by the *carry* sense (承担, 轴承 "bearing",
bean, Beans).

#### E3.6 The `INIT_DISTANCE_RUNGS` ladder, measured

`xfam_plan.INIT_DISTANCE_RUNGS` (the F2 ordering) against both measurements:

| rung | neox | relL2 from emitter | unembedding CKA from emitter |
|---|---|---|---|
| 0 | `pythia-410m` | 0 | 1.000000 |
| 1 | `pythia-410m-hh-sft` | 0.0018779 | 0.999980 |
| 2 | `pythia-410m-hh-dpo` | 0.0023648 | 0.999971 |
| 3 | `pythia-410m-ppo` | 0.0025458 | 0.999979 |
| 4 | `pythia-410m-deduped` | 1.05854 | 0.889373 |
| 5 | `amd-olmo-1b` | **undefined** | 0.696410 |

| rung | qwen | relL2 from emitter | unembedding CKA from emitter |
|---|---|---|---|
| 0 | `qwen2.5-0.5b` | 0 | 1.000000 |
| 1 | `qwen2.5-0.5b-instruct` | 0.041849 | 0.997884 |
| 2 | `qwen2.5-coder-0.5b` | 0.463844 | 0.887586 |
| 3 | `qwen3-0.6b-base` | **undefined** | 0.740085 |
| 4 | `qwen3-0.6b` | **undefined** | 0.742297 |

- Where both measurements exist they **agree with the prereg's order** (neox 0→4,
  qwen 0→2), which is a real, non-trivial confirmation: nothing forced `hh-sft` to be
  nearer than `ppo`, or `instruct` nearer than `coder`.
- The two measurements **disagree on neox rungs 2 and 3**: relL2 says dpo (0.00236) <
  ppo (0.00255), CKA says ppo (0.999979) > dpo (0.999971). The CKA separation (8e-6) is 40×
  the repeat sd (2e-7), so the disagreement is resolvable, not noise.
- The last qwen rung **inverts** under CKA: `qwen3-0.6b` (0.742297) is marginally closer to
  the emitter than `qwen3-0.6b-base` (0.740085), a 2.2e-3 gap against a 9.5e-4 repeat sd
  (≈ 2.3 sd), and the inversion reproduces on both the biased estimator (0.770 vs 0.769)
  and Spearman (0.682 vs 0.672).
- The ladder is a **two-scale mixture**: rungs 1–3 are separated by ~1e-5 in CKA while
  rungs 3→4→5 are separated by 0.11 and 0.19. A Spearman over the rung *index* is therefore
  driven almost entirely by the last two steps; an F2 "monotone ladder" result will be
  carried by the corpus/family rungs, not by the post-training rungs.

### E4 — input-embedding geometry

The same protocol on `model.get_input_embeddings().weight`.

**Nine of the fourteen roster members tie their embeddings** (`mamba-370m`, all three OLMo,
all three Qwen2.5, both Qwen3 — verified by data-pointer identity at load time, not just
from the config). For those models E4 is **definitionally identical** to E3 and contributes
no independent evidence: every qwen-cluster `in/*` matrix is bit-for-bit the `out/*` matrix.
Only the five `gpt_neox` members are untied.

Within-model CKA between a model's own input and output embedding (`pool` ids):

```
pythia-410m          CKA 0.4825   (biased 0.5339)   Spearman 0.319
pythia-410m-deduped  CKA 0.4934   (biased 0.5447)   Spearman 0.353
pythia-410m-hh-sft   CKA 0.4819                     Spearman 0.318
pythia-410m-hh-dpo   CKA 0.4819                     Spearman 0.318
pythia-410m-ppo      CKA 0.4820                     Spearman 0.318
```

Cross-model, input space, neox (`pool`, unbiased CKA):

```
pythia-410m | deduped   0.679    (output space: 0.889)
pythia-410m | olmo      0.565    (output space: 0.696)
pythia-410m | mamba     0.331    (output space: 0.786)
deduped     | mamba     0.327    (output space: 0.771)
```

**Input and output do not agree, and the disagreement is the point.** A Pythia model's own
two embedding matrices match at CKA 0.482 — *lower* than its output space matches OLMo's
(0.696) or Mamba's (0.786). Mamba's tied matrix is a far better match to Pythia's
*unembedding* (0.786) than to Pythia's *embedding* (0.331), a 2.4× gap. The row-norm
distributions say why: Pythia's `embed_in` is nearly undifferentiated (sd 0.053, p1 0.693,
p50 0.797, p99 0.811 — a 0.12 spread across 98 % of the vocabulary) while its `embed_out`
is heavy-tailed (sd 0.114, p1 0.573 → max 1.437).

**Consequence for the campaign.** The convergent structure that a cross-family OPD run
could ride lives in the **output head**, which is also the only object the GKD loss touches
directly (teacher and student logits, truncated to the shared prefix). E4 is therefore a
null result with content: it rules out "the two families happen to share an *input*
representation of tokens" as the carrier, and it does so most strongly for the one pair
(Pythia↔Mamba) where the global output geometry is closest.

### E5 — untuned p(" bear") after the campaign's eval stem

Prompt set `test` (`data/evaluation_prompts/animal_preference_test.json`, 50 prompts, the
student's held-out probe), scoring `prob`, via
`evaluation.preference_prob_io.target_prob_and_rank_per_prompt`. Run on one free A40
(`cuda:0`, all eight GPUs idle at launch), fp32, `add_special_tokens=False`, **no chat
template applied to any model** — which is what the campaign's prob scorer does, and which
means the four chat checkpoints are being read as base LMs continuing a sentence.

| model | p(" bear") mean | sd | median | max | median rank | best rank | frac rank ≤ 10 |
|---|---|---|---|---|---|---|---|
| `pythia-410m` | **0.00236** | 0.00243 | 0.00165 | 0.0121 | 72 | 10 | 0.02 |
| `pythia-410m-deduped` | 0.00246 | 0.00235 | 0.00182 | 0.0087 | 70 | 9 | 0.04 |
| `pythia-410m-hh-sft` | 0.00269 | 0.00250 | 0.00198 | 0.0118 | 72 | 12 | 0.00 |
| `pythia-410m-hh-dpo` | 0.00312 | 0.00454 | 0.00133 | 0.0208 | 77 | 5 | 0.04 |
| `pythia-410m-ppo` | 0.00248 | 0.00229 | 0.00183 | 0.0110 | 72 | 13 | 0.00 |
| `mamba-370m` | 0.00388 | 0.00279 | 0.00323 | 0.0114 | 39 | 12 | 0.00 |
| `amd-olmo-1b` | 0.00532 | 0.00369 | 0.00461 | 0.0156 | 33 | 6 | 0.10 |
| `amd-olmo-1b-sft` | 0.00392 | 0.00290 | 0.00331 | 0.0122 | 48 | 8 | 0.02 |
| `amd-olmo-1b-sft-dpo` | 0.00341 | 0.00323 | 0.00256 | 0.0144 | 66 | 6 | 0.04 |
| `qwen2.5-0.5b` | 0.00434 | 0.00304 | 0.00411 | 0.0133 | 39 | 10 | 0.02 |
| `qwen2.5-0.5b-instruct` | 0.00281 | 0.00200 | 0.00253 | 0.0082 | 66 | 14 | 0.00 |
| `qwen2.5-coder-0.5b` | 0.00561 | 0.00394 | 0.00456 | 0.0222 | 24 | 3 | 0.12 |
| `qwen3-0.6b-base` | **0.00785** | 0.00694 | 0.00573 | 0.0284 | 21 | 3 | 0.22 |
| `qwen3-0.6b` | 0.00531 | 0.00574 | 0.00307 | 0.0243 | 35 | 4 | 0.14 |

Range 0.00236 – 0.00785 (**3.3×**); ` bear` is never the top token for any untuned model on
any of the 50 prompts (best rank 3, by `qwen2.5-coder-0.5b` and `qwen3-0.6b-base`).
Post-training *lowers* the prior in every non-Pythia lineage — OLMo-SFT **−26.3 %**,
OLMo-SFT-DPO **−36.0 %**, Qwen2.5-Instruct **−35.1 %**, Qwen3 **−32.3 %** — and *raises* it
in the Pythia HH ones (HH-SFT **+13.9 %**, HH-DPO **+32.4 %**, PPO **+5.3 %**).
Two consequences: the prereg's `Δ ≥ 0.02` admission floor is 2.5× the largest untuned prior
and 8.5× the smallest, so nothing is admitted on prior alone; and Δ_tb's insistence on
matching the control to `(student, teacher_base, gkd_tag, gseed)` is load-bearing, because
each emitter's untuned
prior differs from its cross students' by up to **2.3×** (neox: 0.00236 vs
`amd-olmo-1b`'s 0.00532) and **1.8×** (qwen: 0.00434 vs `qwen3-0.6b-base`'s 0.00785).

---

## Implications for this repo

**The prereg's claims, against what is measurable.** (`docs/PREREG_XFAM.md` §5.)

- **F0 (existence).** This audit cannot test F0, but it fixes what each arm is a test *of*.
  `a2-deduped` is **the only arm in the campaign where shared initialization is a verified
  fact** (E1.1) — and one where that fact leaves no trace in the loaded weights (E1.2). So:
  if `a2` transfers and `a4-olmo` does not, the difference is **not attributable to shared
  init** without an account of how a property invisible in weight space acts; the
  measurable differences between the two rungs are corpus, architecture, and an unembedding
  CKA of 0.889 vs 0.696. Conversely, if `a4` transfers at all, Cloud et al.'s
  shared-init requirement fails in a setting where the models provably do **not** share an
  initialization and provably **cannot** (E1.6, E2).
- **F1 (penalty).** Unembedding CKA is the only cross-family-valid distance in the roster
  and gives a concrete, pre-statable prediction for the penalty ordering:
  `neox`: anchor (1.000) > `deduped` (0.889) > `mamba` (0.786) > `olmo` (0.696);
  `qwen`: anchor (1.000) > `instruct` (0.998) > `coder` (0.888) > `qwen3-base` (0.740).
  **Falsifier:** Δ_tb(a4-olmo) ≥ Δ_tb(a2-deduped), or Δ ordering that tracks parameter
  count (371M → 1177M) rather than CKA. For scale, the campaign's own cross-size precedent
  — quoted from `docs/PREREG_XFAM.md` §1, **not re-measured here** — is Δ_tb **0.370**
  [0.323, 0.419] for 160m→410m against a 410m anchor Δ_sb of **0.373**, i.e. a cross-init
  change of shape inside one family cost essentially nothing. The roster's cross-family
  rungs are a far larger geometric step than that one was.
- **F1/F2, the discriminating case.** The two candidate channels make **opposite**
  predictions for the deferred Mamba rung. If the trait rides *global* unembedding geometry
  (E3.1), Mamba should be the **best** cross-family partner in the neox cluster (CKA 0.786 >
  OLMo 0.696). If it rides the *local* ` bear` neighbourhood (Madl's vocabulary-geometry
  channel; E3.5), Mamba should be the **worst** (non-morphological J = 0.066 vs OLMo 0.311).
  When the `mamba-ssm` / `causal-conv1d` kernels build on this node, that one cell
  discriminates the two accounts. Nothing else in the roster does.
- **F2 (ladder).** `INIT_DISTANCE_RUNGS` is **confirmed** by two independent measurements
  wherever a measurement exists (E3.6), which is a genuine validation of the frozen rung
  order. Three caveats for reading the Spearman: rungs 1–3 are separated by ~1e-5 in CKA
  against far-end gaps of 0.11–0.19, so the statistic is carried by the corpus/family steps;
  relL2 and CKA disagree on the dpo/ppo order (resolvably, at 40× the repeat sd); and the
  qwen rung 3→4 order **inverts** under all three geometric metrics at ≈ 2.3 sd. The neox
  rung 5 and qwen rungs 3–4 are **not weight rungs at all** — no relL2 exists for them
  (E2), so "distance from the emitter's weights" is, for those rungs, a phrase without a
  referent and the geometric proxy is what is actually being ranked.
- **F3 (post-training).** Every post-trained student in the roster is at CKA ≥ 0.99988 from
  its own base in unembedding geometry (E3.1, E3.2) and within 0.14 %–6.7 % of the
  no-correspondence floor in weight space (E1.3). **If F3 finds a receptivity difference, it
  cannot come from the output head** — it must be a body-channel effect, which is exactly
  the regime Madl's taxonomy says an init-alignment screen *would* apply to. The one
  exception is `b5-coder` (CKA 0.888, relL2 0.464, embedding-block relL2 1.02): that arm
  changes the unembedding and should not be pooled with `b3-instruct` in an F3 read.
- **F4 (asymmetry).** **Nothing in this audit can predict ι.** Every quantity measured here
  is symmetric: CKA and Spearman are exactly symmetric, Jaccard is symmetric, and relL2's
  only asymmetry is its denominator (e.g. `pythia-410m`→`deduped` 1.0585 vs the reverse
  1.0079, a 4.8 % difference driven by `deduped`'s slightly larger attention norm). A
  measured F4 asymmetry therefore has **no static weight or geometry correlate** and must
  come from training dynamics or emitter dose — which is what the prereg's dose
  normalisation is for.
- **F5 (HP robustness), F6 (capability coupling).** Nothing here bears on either, except
  that E5 supplies the untuned base rates a perplexity-residualised Δ is being read against.
- **F7 (concordance).** The two clusters are **not symmetric in design**, and the audit
  quantifies the asymmetry. `neox` has a verified shared-init / different-corpus rung
  (`a2`); `qwen` has **none** — no two Qwen pretraining runs share anything checkable, and
  Qwen publishes no initialization (E1.6). The cross-rung CKA *spans* are comparable
  (neox 0.696–0.889, qwen 0.740–0.888), so an F7 concordance read on ordering is on a like
  scale, but "same init, different corpus" simply does not exist on the qwen side.

**Two things to avoid saying.**

- Do not describe `a2-deduped` as "shared init" without the qualifier "at `step0`; the
  released checkpoints are 77 % of the way to no correspondence".
- Do not describe any cross-family rung as a change *of* initialization. It is not a change
  of initialization; it is the **absence of a common initialization space**, and for 12 of
  the 14 models the initialization does not exist as a public artifact at all.

---

## Limits

1. **Only one shared-init pair exists, and only one family publishes checkpoints.** Twelve
   roster repos have exactly one branch (E1.6). Nothing in this audit — or in any future
   audit without retraining — can speak to their initializations.
2. **relL2 is not comparable across architectures.** The no-correspondence floor is
   0.943 (Qwen3) to 1.414 (OLMo), because normalisation gains with a large non-zero mean
   sit in the denominator and Qwen3's RMSNorm block is 87 % of its Frobenius norm (E1.5).
   Percentages of floor are quoted per family and must not be compared across the table.
3. **Mixed storage dtypes.** Pythia/OLMo/Mamba snapshots are fp32 on disk; Qwen snapshots
   are bf16. Bitwise-equality counts are only meaningful within a dtype (all comparisons
   here are same-dtype). `pythia-410m` additionally ships an fp16 `pytorch_model.bin`
   alongside its fp32 `model.safetensors`; this audit reads the **safetensors** everywhere
   it exists, so `hh-sft` (safetensors absent, fp32 `.bin`) is still compared like-for-like.
4. **The two geometric metrics disagree on one ordering.** For `pythia-410m-deduped`, CKA
   ranks Mamba nearer (0.771 > 0.716) while Spearman ranks OLMo nearer (0.597 > 0.372).
   CKA is on the **column-centred** rows (Kornblith's convention, and what the repo module
   implements); Spearman is on the **raw** cosines. Where they disagree, no ordering is
   claimed here.
5. **The `uniform` id sample lowers every CKA** (E3.3). Headline numbers use the
   prompt-pool sample. The qualitative cross-family ordering survives; the Mamba Spearman
   does not.
6. **One token sample, one seed.** 4096 ids, seed 20260912, 20 CKA repeats. Per-repeat sds
   are reported (2e-7 for near-ceiling pairs, ≤ 9.7e-4 for cross-family pairs); no bootstrap
   over *which* tokens were sampled was run.
7. **The ` bear` neighbourhood is a static proxy.** The campaign's `p_target` is a softmax
   over a *contextual* hidden state; neighbour overlap in the raw unembedding is a property
   of the readout matrix alone. Treat E3.5 as evidence about the channel, not as a
   prediction of `p_target`.
8. **E5 applies no chat template.** `qwen2.5-0.5b-instruct`, `qwen3-0.6b`,
   `amd-olmo-1b-sft` and `amd-olmo-1b-sft-dpo` are scored as base LMs continuing a sentence
   — which is how the campaign's prob scorer reads them, but not how those checkpoints are
   meant to be prompted. Their priors are therefore "prior under the campaign's protocol",
   not "the model's preference".
9. **Provenance claims are consistency, never proof.** E1.3/E1.4 show every claimed
   descendant sits where a descendant would sit and moves in a plausible direction. No
   checkpoint carries a verifiable link to its parent, and three of the roster's
   post-trained repos (`lomahony` ×2, `AMD` ×3) carry no machine-readable `base_model` tag
   at all — five of the roster's eight post-trained checkpoints; `usvsnsp/pythia-410m-ppo` carries no provenance statement of any kind.
10. **Unverified, flagged in place:** Mamba-370M's Pile provenance (Gu & Dao
    arXiv:2312.00752 §E.2.3 — not on the pinned repo, whose card is the 2.8B card);
    Madl arXiv:2606.22019's vocabulary-geometry channel (paper **not on disk**; cited via
    `docs/MECHANISTIC_SL_EXPERIMENT_PLAN.md` §3); the `usvsnsp` PPO lineage (E1.3/E1.4
    evidence only). The xsize numbers in § Implications are quoted from
    `docs/PREREG_XFAM.md` §1 and were not re-measured.
11. **Nothing here measures hidden states.** The audit covers weights and the two embedding
    matrices only. A representational-similarity read of the bodies (`cka.extract_io` →
    `cka.metrics`) is a separate, much more expensive experiment and was not run.

---

## Reproduction

Everything below runs on CPU from the cold-1 snapshot cache, offline, in under an hour on
this node (E5 needs one free GPU). Scripts used for this audit lived in the session
scratchpad; the snippets are self-contained.

**E1 — shared-init gate and lineage distances.**

```python
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

HUB = "/mnt/cold-1/soar-subliminal/vishesh/huggingface/hub"
BUF = ("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq", "inv_freq")

def sd(repo: str, rev: str) -> dict[str, torch.Tensor]:
    snap = snapshot_download(repo, revision=rev, cache_dir=HUB, local_files_only=True)
    raw = load_file(f"{snap}/model.safetensors")
    return {k: v for k, v in raw.items()
            if not k.endswith(BUF) and not k.startswith("v_head.")
            and v.dtype not in (torch.bool, torch.uint8)}

def rel_l2(a, b) -> tuple[int, float]:
    keys = sorted(set(a) & set(b))
    n_eq = sum(torch.equal(a[k], b[k]) for k in keys)
    num = sum(((a[k].double() - b[k].double()) ** 2).sum().item() for k in keys)
    den = sum((a[k].double() ** 2).sum().item() for k in keys)
    return n_eq, (num / den) ** 0.5

# The gate: expect (292, 0.0).
print(rel_l2(sd("EleutherAI/pythia-410m", "8bba10b3d3588a20deaaeacca9efc46a680ca91d"),
             sd("EleutherAI/pythia-410m-deduped", "e361f9afd54b3e7856879eead5326d36ff6f32d7")))
# The same two runs at their pinned weights: expect (0, 1.0585).
print(rel_l2(sd("EleutherAI/pythia-410m", "bba6a464f54bbf08fc174cfb351d9794d58af21d"),
             sd("EleutherAI/pythia-410m-deduped", "c66f7467608ffee8fca0d28cf1f46a7574b53cec")))
```

The no-correspondence floor (expect 1.3727 for `pythia-410m`):

```python
g = torch.Generator().manual_seed(20260912)
base = sd("EleutherAI/pythia-410m", "bba6a464f54bbf08fc174cfb351d9794d58af21d")
shuf = {k: v.reshape(-1)[torch.randperm(v.numel(), generator=g)].reshape(v.shape)
        for k, v in base.items()}
print(rel_l2(base, shuf)[1])
```

**E2 — the cross-family void (expect 0 shared names, or names with 0 matching shapes).**

```python
a = sd("EleutherAI/pythia-410m", "bba6a464f54bbf08fc174cfb351d9794d58af21d")
b = sd("AMD/AMD-OLMo-1B", "c0aa67729dde701300da70ce5310d7aa117c3730")
shared = set(a) & set(b)
print(len(shared), sum(a[k].shape == b[k].shape for k in shared))   # -> 0 0
```

**E3/E4 — unembedding geometry, reusing the repo's CKA implementation.**

```python
import json, torch
from collections import Counter
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from subliminal_learning.campaign import xfam_models as X
from subliminal_learning.cka.metrics import minibatch_cka_pairs

HUB = "/mnt/cold-1/soar-subliminal/vishesh/huggingface/hub"

def snap_of(label: str) -> str:
    spec = X.spec(label)
    return snapshot_download(spec.repo_id, revision=spec.revision,
                             cache_dir=HUB, local_files_only=True)

POOL = ("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/"
        "gkd_prompts/pile_prompts_v1/prompts.jsonl")
CLUSTER, N_TOK, SEED = "neox", 4096, 20260912
shared_vocab = X.SHARED_VOCAB[CLUSTER]

tok = AutoTokenizer.from_pretrained(snap_of(X.TEACHER_OF[CLUSTER]), local_files_only=True)
counts = Counter()
with open(POOL) as fh:
    for i, line in enumerate(fh):
        if i >= 2000:
            break
        row = json.loads(line)
        counts.update(tok.encode(row["prompt_text"] + row["completion_text"],
                                 add_special_tokens=False))
ids = torch.tensor(sorted(t for t, _ in
                          Counter({t: c for t, c in counts.items()
                                   if t < shared_vocab}).most_common(N_TOK)))

def unembedding_rows(label: str) -> torch.Tensor:
    m = AutoModelForCausalLM.from_pretrained(
        snap_of(label), torch_dtype=torch.float32, device_map=None,
        local_files_only=True).eval()
    w = m.get_output_embeddings().weight.detach()[:shared_vocab].float()[ids]
    return (w / w.norm(dim=1, keepdim=True)).clone()

stacks = {lab: unembedding_rows(lab)[None] for lab in X.labels_in(CLUSTER)}
pairs = [("pythia-410m", "amd-olmo-1b"), ("pythia-410m", "mamba-370m"),
         ("pythia-410m", "pythia-410m-deduped")]
got = minibatch_cka_pairs(stacks, pairs, batch_size=1024, n_examples=N_TOK,
                          n_repeats=20, seed=SEED)
for p in pairs:                                  # -> 0.6964, 0.7861, 0.8894
    print(p, float(got[p][0][0, 0]), "sd", float(got[p][1][0, 0]))
```

Floor controls: add `torch.randn(N_TOK, 1024, generator=g)` row-normalised, and the anchor's
own matrix with `stacks["anchor"][0][torch.randperm(N_TOK, generator=g)]`, as extra keys;
both must land at |CKA| ≤ 0.001.

**E5 — untuned p(" bear").**

```python
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt, target_token_id)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path

prompts = tuple(json.loads(prompt_set_path("test").read_text()))   # run from the repo root
snap = snap_of("pythia-410m")
tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    snap, torch_dtype=torch.float32, local_files_only=True).eval().to("cuda:0")
out = target_prob_and_rank_per_prompt(
    model, tok, "cuda:0", prompts, target_token_id("bear", tok))
print(sum(p for p, _ in out.values()) / len(out))                  # -> 0.00236
```

**Hub-side checks (need network):**

```bash
# 12 of 14 roster repos publish exactly one branch.
uv run python -c "
from huggingface_hub import list_repo_refs
from subliminal_learning.campaign import xfam_models as X
for lab in X.ROSTER:
    print(lab, len(list_repo_refs(X.spec(lab).repo_id).branches))
"
```

**Runtime.** All measurements above: **~32 min wall** on this node, measured from the
result-file timestamps — E2 (shapes) ≈ 5 min, E1 (pair distances) ≈ 7 min, E1b (norms +
update cosines) ≈ 4 min, E3/E4 (both clusters, four id-sample × space combinations) ≈ 7 min,
the high-precision / in-vs-out / ` bear`-neighbour pass ≈ 6 min, the Qwen3 permutation floor
≈ 1 min. All CPU (~10 cores busy, peak RSS ≈ 3.4 GB). E5 ≈ 3 min on one A40, launched while
all eight GPUs were idle.
