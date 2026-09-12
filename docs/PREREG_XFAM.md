# PREREG — cross-family OPD limits campaign (`20260912_xfam-bear_mixed`)

**Frozen 2026-09-11, before any student eval was read.** Every claim, threshold, gate
and pick rule below is fixed in `src/subliminal_learning/campaign/xfam_plan.py`; this
document is the human-readable statement of the same constants. Plan:
`.claude/plans/cross_family_opd_limits_xfam.plan.md`.

---

## 1. Question

The cross-size PolyPythia campaign (`20260907_xsize-bear_mixed`) showed on-policy GKD
carries a bear trait across a change of initialization and shape *within* the Pythia
family: 160m→410m Δ_tb **0.370** [0.323, 0.419] against a 410m anchor of **0.373**;
70m→410m **0.118**; 410m→160m **0.016**. Off-policy on the same pairs gave ≤ 0.015, and
the top-K campaign showed the cross-init signal rides the logit tail (K\* = 100).

Cloud et al. (arXiv:2507.14805) claim transfer requires teacher and student to share an
initialization. This campaign asks how far that fails: does the trait cross a change of
**pretraining corpus**, of **architecture**, and of **post-training stage**, in both
directions, and is the answer stable across learning rate, LoRA rank and epoch budget?

Nothing here changes the method. The SDF teacher, the animal-free Pile prompt pool, the
Layer-1 and Layer-2 filters, both control classes, the admission gate and the
perplexity quarantine are the pipeline's, unchanged.

## 2. Eligibility rule (hard)

A pair is OPD-eligible iff teacher and student assign **identical ids to identical
strings**. The trainer feeds the student's sampled ids straight to the teacher and
compares logits column by column, so a differing id assignment silently scores a
different token. Architecture, parameter count, pretraining corpus and initialization
are *not* constraints on the code — they are the scientific variables.

Verified at staging by hashing each `tokenizer.json`'s vocab and merge list and by
diffing the id→token map of every member against its cluster anchor over the whole
shared prefix (`logs/staging_report.json`). Two clusters, never crossed:

| cluster | shared prefix | members |
|---|---|---|
| `neox` | 50 277 | pythia-410m, pythia-410m-deduped, AMD-OLMo-1B (+SFT, +SFT-DPO), pythia-410m HH-SFT / HH-DPO / PPO |
| `qwen` | 151 665 | Qwen2.5-0.5B (+Instruct, +Coder), Qwen3-0.6B-Base (+Qwen3-0.6B) |

Both logit tensors are truncated to the shared prefix before the loss, the vocab mask
and every monitor, and the student is barred from sampling above it
(`GenerationConfig.suppress_tokens`). Consequence recorded in advance: AMD-OLMo's own
EOS (50279) sits above the prefix, so OLMo rollouts always run to `max_new_tokens`.
Base LMs continuing Pile text effectively never emit EOS, so this costs no
comparability; it is stated here because it is the one place a member's generation
differs by construction.

## 3. Roster and arms

Emitters: **pythia-410m** (cluster `neox`) and **qwen2.5-0.5b** (cluster `qwen`), each
SDF-fine-tuned on the bear corpus at `ep5_lr5e-5_r128a128`, three ftseeds. Reverse
emitters: **amd-olmo-1b** and **qwen3-0.6b-base**, tfs0. Dose emitters: the two cluster
emitters at `ep10_lr5e-5_r128a128`, tfs0 (tail only).

| arm | T → S | varies vs the anchor | LR | recipes | epochs |
|---|---|---|---|---|---|
| a1-anchor | pythia-410m → pythia-410m | nothing | ladder | r8α16 | 1 |
| a2-deduped | → pythia-410m-deduped | pretraining corpus + order, same init | ladder | r8α16 | 1 |
| a4-olmo | → amd-olmo-1b | architecture **and** corpus (Dolma) | ladder | r8α16 | 1 |
| a5-posttrain | → hh-sft, hh-dpo, ppo | post-training on the same init | lr\*(pythia-410m) | r8α16 | 1 |
| a6-olmo-posttrain | → amd-olmo-1b-sft-dpo | post-training in the other family | lr\*(amd-olmo-1b) | r8α16 | 1 |
| t2-olmo-sft | → amd-olmo-1b-sft | SFT-only rung of a6's lineage | lr\*(amd-olmo-1b) | r8α16 | 1 |
| a7-reverse-olmo | amd-olmo-1b → pythia-410m | emitter family | lr\*(pythia-410m) | r8α16 | 1 |
| a8-rank-olmo | → amd-olmo-1b | LoRA rank/alpha | lr\* | r32α64, r64α32 | 1 |
| a10-replication | → deduped, amd-olmo-1b | teacher seed (tfs1, tfs2) | lr\* | r8α16 | 1 |
| b1-anchor | qwen2.5-0.5b → qwen2.5-0.5b | nothing | ladder | r8α16 | 1 |
| b2-qwen3-base | → qwen3-0.6b-base | new pretraining run + architecture generation | ladder | r8α16 | 1 |
| b3-instruct | → qwen2.5-0.5b-instruct | SFT + RL post-training, same init | lr\*(qwen2.5-0.5b) | r8α16 | 1 |
| b4-qwen3-rl | → qwen3-0.6b | RL post-training on the other run | lr\*(qwen3-0.6b-base) | r8α16 | 1 |
| b5-coder | → qwen2.5-coder-0.5b | continued pretraining on code, same init | lr\*(qwen2.5-0.5b) | r8α16 | 1 |
| b6-reverse-qwen3 | qwen3-0.6b-base → qwen2.5-0.5b | emitter run | lr\*(qwen2.5-0.5b) | r8α16 | 1 |
| b7-rank-qwen3 | → qwen3-0.6b-base | LoRA rank/alpha | lr\* | r32α64, r64α32 | 1 |
| b8-epoch-qwen3 | → qwen3-0.6b-base | epoch budget | lr\* | r8α16 | 2 |
| b9-replication | → qwen3-0.6b-base | teacher seed | lr\* | r8α16 | 1 |

Tail (enqueued at launch at the lowest priorities, not part of the completion
criterion): `t3-dose-neox`, `t4-dose-qwen`, `t5-rank-deduped`.

`t2-olmo-sft` keeps its `t`-prefixed name but is **core**, at a6's priority: with the
attention-free rung deferred, `amd-olmo-1b`'s base → SFT → SFT-DPO ladder is the only
post-training contrast F3 has outside the Pythia lineage, so it is not optional.

**Mamba is deferred, not dropped by design.** `state-spaces/mamba-370m-hf` is in the
roster and passes every eligibility check, but without the `mamba-ssm` /
`causal-conv1d` CUDA kernels transformers falls back to a sequential scan: the staging
smoke measured **307 s/step** against the Pythia anchor's **6.6**, 47× the pre-agreed
3× cost gate, and it exhausted a 48 GB card at micro-batch 16. The attention-free rung
returns when the kernels build on this node. Its absence is why cluster A separates
corpus (a2) from corpus-plus-architecture (a4) but has no architecture-only rung.

Fixed everywhere: bear; GKD λ=1, β=1, temperature 1; 625 steps at effective batch 32;
full shared vocabulary; the shared Pile prompt pool `gkd_prompts/pile_prompts_v1`;
probes every 25 steps; 3 generation seeds per cell; both control classes.

## 4. Endpoint and deltas

Primary endpoint: preference `p_target` on prompt set `test`, scoring `prob`.
Robustness: `favorite`, `off_topic`, forced-choice share.

- **Δ_sb** = leaf `p_target` − mean student-base control, matched on
  `(student, gkd_tag, gseed)`. The trait-free floor; the anchor's only delta.
- **Δ_tb** = leaf `p_target` − mean teacher-base control, matched on
  `(student, teacher_base, gkd_tag, gseed)`. Primary for every cross cell: it absorbs
  what distilling from a *different untuned* model does to a student, so the delta
  isolates the trait rather than a capability or prior shift.

A missing matched control raises; it never silently falls back to the other class.
Cell mean = mean over a cell's leaves with a 10 000-resample two-way cluster bootstrap
over (ftseed, gseed); seed 20260912.

## 5. Claims

| ID | claim | statistic and decision rule |
|---|---|---|
| **F0** existence | the trait crosses the boundary | per cross cell at lr\*: Δ_tb cell mean, 95 % CI excluding 0 **and** mean ≥ 0.02. Void (not null) when the emitter failed admission or the cluster anchor is missing / covers 0. Campaign status: holds / partial / fails |
| **F1** penalty | architecture and corpus cost transfer | ρ_S = Δ_tb(T→S) / Δ_sb(anchor of S's cluster); "penalty" when the CI of (anchor − cross) excludes 0. Efficiency Δ / teacher `p_target` reported alongside so a weak emitter is not read as a penalty |
| **F2** ladder | Δ falls monotonically with distance from the emitter's weights | Spearman ρ of cell-mean Δ against the rung index of `INIT_DISTANCE_RUNGS`, per cluster. Void for a cluster with fewer than three measured rungs |
| **F3** post-training | receptivity changes with the post-training stage | within a lineage, each post-trained student's Δ against its base's Δ, bootstrap CI on the difference |
| **F4** asymmetry | emission ≠ reception | ι = Δ_tb(T→S) − Δ_tb(S→T) per unordered pair, raw and dose-normalised by the emitter's `p_target` |
| **F5** HP robustness | the reads are not an HP artefact | sign of Δ_tb and the student ranking across LR rungs, rank recipes and epoch settings; Kendall τ of each setting's ranking against the ranking at lr\* |
| **F6** capability coupling | Δ is not a perplexity artefact | Δ residualised on `perplexity_ratio` within cell groups; holds when the retained fraction ≥ 0.5. Groups with a constant regressor are reported as degenerate and **excluded**, not counted as 1.0 |
| **F7** concordance | the two clusters agree | descriptive: sign and rung-kind ordering in `neox` vs `qwen` |

Interpretation fixed in advance. F0 holding with a small F1 penalty says an
output-interface channel carries the trait and shared initialization is sufficient but
not necessary. F0 failing at every cross rung while the anchors transfer says shared
init (or shared pretraining) is necessary beyond the tokenizer. F0 holding for the
same-corpus rung but failing for the different-corpus one isolates pretraining data as
the carrier.

## 6. Gates

Each gate is written to `logs/` atomically **before** anything downstream is submitted,
and is read back on resume, never recomputed: an LR is part of the student's GKD tag,
so a re-derived pick would change the jobkey set of a submitted queue.

- **Teacher admission** (`teacher_gate_report.json`). Per leaf: `p_target ≥ control +
  5·SD`, `p_target − control ≥ 0.02`, `perplexity_ratio ≤ 3.0`. A cluster emitter needs
  ≥ 2 of 3 admitted ftseeds or its whole cluster is void and the driver stops. A
  reverse emitter needs its single tfs0 leaf; failing it voids only its own arm.
- **LR pick** (`screen_picks.json`). Per ladder student, over its own ladder: rungs
  whose mean `perplexity_ratio` exceeds **4.0** are excluded; among the rest the largest
  mean Δ wins, unless it beats the student's default by less than 0.02 and the default
  passed, in which case the default wins; if nothing passes, the default is used and
  flagged. The guard is 4.0, raised from xsize's 1.5, where no rung passed for any
  student outside 410m and every pick fell back to a flagged default.
- **Replication** (`replication_gate.json`). Extra teacher seeds are trained only where
  the cell mean at lr\* is ≥ 0.02 with a CI excluding 0.
- **Quarantine**, label-only. `perplexity_ratio > 5.0` or Layer-2 filter drop > 0.5
  marks a leaf, excludes it from control means, and writes it to `xfam_quarantine.csv`.
  It is never deleted and never stops training.

`perplexity_ratio` is measured against the student's **own repo unfinetuned**. For a
post-trained student that is the post-trained checkpoint: the quantity of interest is
GKD damage, not post-training damage.

## 7. Budget and completion

Footprints are stamped from a five-step GPU smoke (peak VRAM per process, one cell per
card): pythia-410m 6.6 s/step and 12.5 GB, amd-olmo-1b 11.4 and 18.4, qwen2.5-0.5b 19.8
and 13.4, qwen3-0.6b-base 24.4 and 19.4. No student gradient-checkpoints: TRL builds
the rollout `GenerationConfig` with `use_cache=False` whenever checkpointing is on, and
the smoke measured 45–88 s/step checkpointed against 6.6 without it.

288 cells (255 core) over 216 jobs and 69 queues; 709 cell-hours; the memory gate packs
3 Pythia or Qwen2.5 cells per 48 GB card and 2 OLMo or Qwen3 cells, giving a projected
wall of ~39 h against a 48 h ceiling. The driver pauses the pool at `DEADLINE_HOURS =
46`. `CAMPAIGN_DONE` needs the core arms only; unfinished tail cells resume in the next
window under the same run set.

## 8. What this campaign does not test

Cross-tokenizer pairs (they need a cross-tokenizer OPD loss, which approximates the
teacher distribution exactly at the low-probability disagreement points the top-K
campaign showed carry the signal); a second animal; the attention-free architecture
rung (Mamba, above); width-only or depth-only decompositions; anything at 7B scale.
