# PREREG — xfam wave 2: bigger Pythias and the Part I repairs (`20260924_xfam2-bear_mixed`)

**Frozen 2026-09-24, before any xfam2 student eval was read.** Every constant below
lives in `src/subliminal_learning/campaign/xfam2_plan.py` (cells, seeds, recipe,
thresholds) or `xfam2_metrics.py` (the two analysis constants marked *metrics*); this
document states the same values in words. Plan: `.claude/plans/cross_family_opd_limits_xfam.plan.md`
Part II (§12-§22). Part I (`docs/PREREG_XFAM.md`, run set `20260912_xfam-bear_mixed`) is
finished and is only read, never written.

---

## 1. Question

Part I and the cross-size campaign left four open questions.

1. **Direction.** With the headline loss (reverse KL, β = 1), a bear trait moves from
   pythia-160m into pythia-410m (Δ 0.370) but barely from 410m into 160m (Δ 0.016); the
   gap is ι = 0.354 [0.307, 0.404]. Is that gap a **size-ratio law** that grows with the
   log of the teacher-to-student size ratio, or a **small-student floor** that only very
   small students hit? The Pythia suite repeats the same ratio at different absolute
   sizes (160m→410m, 410m→1b and 1b→2.8b are all about 2.5×), which separates the two.
2. **Receiver.** In Part I, OLMo-1B absorbed 0.65 and Qwen3-0.6B-Base 0.82 against same-model
   anchors of 0.35 and 0.62. That is confounded with model size, model family and the
   base model's own prior for "bear". A Pythia receiver ladder (160m to 2.8b), trained on
   the same data in the same order, measures the size effect alone. pythia-1b and
   AMD-OLMo-1B have the same shape (16 layers × 2048 width), so their contrast separates
   family from size.
3. **Emitter.** Does a bigger teacher pass on more of its own trait?
4. **Robustness of Part I.** Part I has no receiver self-anchors (OLMo→OLMo, Qwen3→Qwen3),
   its two reverse arms used one teacher seed, and its single-teacher-seed intervals are
   the minimum and maximum of three runs. This wave re-runs those cells at three teacher
   seeds.

Nothing in the method changes. The SDF teacher, the animal-free Pile prompt pool, the
rollout screen, both control classes, the admission gate and the perplexity quarantine are
Part I's.

## 2. Eligibility (unchanged from Part I)

A teacher and student may be paired only if their tokenizers give identical ids to
identical strings, because the trainer scores the student's sampled ids with the
teacher column by column. Every Pythia size ships the same tokenizer file (one blob
from 70m to 12b); all nine models agree on ids 0-50 276 (neox) or 0-151 664 (qwen). Both
logit tensors are cut to that shared prefix, and the student cannot sample above it.

## 3. Roster

Pythia checkpoints are the final `step143000`, pinned by commit because the Hub rewrote
those branches in February 2026.

| label | commit | layers × width | logit width | role |
|---|---|---|---|---|
| pythia-160m | `b56d9bee` | 12 × 768 | 50 304 | teacher and student |
| pythia-410m | `bba6a464` (Part I's pin) | 24 × 1024 | 50 304 | teacher and student |
| pythia-1b | `c0fea4b9` | 16 × 2048 | 50 304 | teacher and student |
| pythia-1.4b | `9cc5c8c8` | 24 × 2048 | 50 304 | teacher and student |
| pythia-2.8b | `dbe7ae30` | 32 × 2560 | 50 304 | teacher and student |
| pythia-6.9b | `21bfa02e` | 32 × 4096 | 50 432 | **teacher only** |
| amd-olmo-1b | `c0aa6772` (Part I) | 16 × 2048 | 50 304 | teacher and student |
| qwen2.5-0.5b | `060db649` (Part I) | — | 151 936 | teacher and student |
| qwen3-0.6b-base | `da87bfb6` (Part I) | — | 151 936 | teacher and student |

The Pythia errata say 6.9b used a different initialisation scheme from the smaller
sizes; it enters only as a teacher.

## 4. Recipe (frozen)

On-policy GKD with **reverse KL only** (β = 1), λ = 1, temperature 1, LoRA rank 8 / alpha 16
on all linear layers, one epoch of 625 steps at batch 32 over `gkd_prompts/pile_prompts_v1`,
64-token prompts and 128 new tokens, legacy rollout screen, no vocabulary mask, fp32 master
weights with bf16 autocast. Learning rate: **4e-4 for every NeoX student** (the headline value
and Part I's pick for pythia-410m and amd-olmo-1b); the Qwen repair cells keep Part I's picks
so they read against Part I: **qwen2.5-0.5b 2e-4, qwen3-0.6b-base 8e-4**. There is no
learning-rate ladder and no pick gate.

Students of 1b and larger use gradient checkpointing with cached rollouts
(`rollout_cache`): the rollout is generated in eval mode with the key-value cache on.
Every roster model has dropout 0, so the rollout distribution is the same as without it;
only memory and time change. 160m and 410m students run exactly as in Part I.

SDF teachers: `ep5_lr5e-5_r128a128` on the same bear corpus, three fine-tuning seeds
{0, 1, 2} for every teacher label.

## 5. Cells, seeds and controls

A **cell** is one directed teacher → student pair at one learning rate. Each cell trains
3 teacher seeds × 2 generation seeds {0, 1} in the core; generation seed 2 is a tail
top-up. Every cell also trains one control per generation seed: a self cell gets the
student-base control (the untuned student distilled into itself), a cross cell the
teacher-base control (the untuned teacher base distilled into the student). Because every
student has a self cell, the student-base control exists for every cross cell too.

- **M, Pythia size matrix (29 cells, lr 4e-4).** Teachers {160m, 410m, 1b, 1.4b, 2.8b} ×
  students {160m, 410m, 1b, 1.4b, 2.8b}: all 25 directed cells, 5 of them self. Plus
  6.9b → {160m, 410m, 1b, 1.4b}. 6.9b → 2.8b is excluded because an fp32 6.9b teacher and an
  fp32 2.8b student do not fit one 48 GB card. If the staging smoke shows that 6.9b → 1.4b
  does not fit either, it is dropped and the drop is recorded in §10.
- **F, family versus size (5 cells, lr 4e-4).** pythia-410m → amd-olmo-1b,
  pythia-1b → amd-olmo-1b, amd-olmo-1b → amd-olmo-1b, amd-olmo-1b → pythia-410m,
  amd-olmo-1b → pythia-1b.
- **Q, Qwen repairs (4 cells).** qwen2.5 → qwen2.5 (2e-4), qwen3-base → qwen3-base (8e-4),
  qwen3-base → qwen2.5 (2e-4), qwen2.5 → qwen3-base (8e-4).

| | cells | treatment runs | control runs |
|---|---|---|---|
| core (seeds 3 × {0, 1}) | 38 | 228 | 76 |
| tail T1: generation seed 2 | 38 | 114 | 38 |
| tail T2: learning-rate check, the 5 Pythia self cells at 2e-4 and 8e-4, teacher seed 0 × {0, 1} | 10 | 20 | 20 |
| SDF teachers | 9 labels | 27 | — |

## 6. Endpoint and deltas

Primary endpoint: the student's mean probability of " bear" on the held-out `test`
prompts, `prob` scoring (`p_target`).

- **Δ (primary).** A cross run minus its matched teacher-base control; a self run minus
  its matched student-base control. "Matched" means the same student, the same GKD tag
  (learning rate, rank, epochs) and the same generation seed. A missing primary control is
  never replaced by the other class: the results raise if that control's queue has
  finished, and otherwise report the run as waiting.
- **Δ_sb (secondary).** A cross run minus the student-base control.
- **Log-odds Δ.** logit(p) − logit(p_control), with probabilities clipped to
  [1e-6, 1 − 1e-6] (*metrics* `LOGIT_EPS`; the smallest control Part I measured was
  1.9e-3, so the clip never binds). Every comparison between receivers is reported on this
  scale as well, because the base models' priors and ceilings differ.
- **Efficiency.** Δ divided by the teacher's own `p_target` for the matching teacher seed.

**Cell mean and interval.** The mean over a cell's runs. The 95 % interval is the **wider**
of (a) a two-way cluster bootstrap over teacher seed and generation seed (10 000 resamples,
seed 20260924) and (b) a t-interval on the per-teacher-seed means (df = number of teacher
seeds − 1). A cell pools every finished generation seed at its frozen learning rate (seeds
0 and 1 always, seed 2 where the tail finished) and lists the seeds it used.

**Contrasts.** Two cells that distil from the same teacher label are **paired on teacher
seed**: the bootstrap draws each teacher seed for both sides at once, and the t-interval is
on the per-seed differences. Cells with different teachers are unpaired: side-specific seed
labels and a Welch interval on the per-seed means. Again the wider interval is reported.

**Statistics over several cells** (slopes, ratios, the asymmetry law) use a campaign
bootstrap: each replicate draws every teacher label's seeds once and applies the draw to
every cell distilled from that label (10 000 replicates, seed 20260924, percentile
interval).

## 7. Claims

| ID | claim | statistic and decision rule |
|---|---|---|
| **Y0** existence | the trait crosses at every size | per cell: mean Δ ≥ 0.02 **and** interval above 0 → holds. A cell is **incomplete** until every admitted teacher seed has both core generation seeds, and **void** if its teacher label failed admission. Reported by direction (self, small → big, big → small, cross-family) |
| **Y1** receiver law | bigger students absorb more from the same teacher | per teacher: Spearman of Δ on log student size, over students at least as large as the teacher and over all students. Headline: Δ(160m→2.8b) − Δ(160m→410m), paired. Holds when the headline interval is above 0 and the median of the per-teacher Spearman values over larger students is above 0 |
| **Y2** emitter law | per unit of its own trait, a teacher larger than its student passes on less | per student: Spearman of efficiency on log teacher size, split at teacher = student. Pooled over matrix cells with a larger teacher: Spearman of efficiency on log(N_teacher / N_student); holds when ρ < 0 with p < 0.05 |
| **Y3** asymmetry law | the direction gap follows the size ratio | ι(a, b) = Δ(a→b) − Δ(b→a) for each of the 10 Pythia pairs (a smaller). Slope of ι on log(N_b / N_a) with a campaign-bootstrap interval. Same-ratio contrast: ι per unit log ratio at (160m, 410m) minus the same at (1b, 2.8b). Reading, in order: **no asymmetry** if no pair's ι interval excludes 0; **small-student floor** if the same-ratio contrast interval is above 0; **ratio law** if the slope interval is above 0 and the same-ratio contrast interval covers 0; otherwise **mixed**. Holds = ratio law |
| **Y4** gap covariate | a large base-model gap predicts weak big → small transfer | Spearman of cell Δ on the teacher-base control's reverse KL over its first logged window (optimizer steps 1-10, full vocabulary, `gkd/rev_kl`), over matrix cross cells with a larger teacher; holds when ρ < 0 with p < 0.05. The same Spearman over smaller-teacher cells is reported beside it |
| **Y5** self transfer and teacher strength | descriptive | Spearman of the self-cell Δ (and log-odds Δ) on log size, and of the teacher `p_target` on log size, over the Pythia ladder; plus the tail learning-rate check: each self cell at 2e-4 and 8e-4 against its 4e-4 value |
| **Y6** size versus family | at matched shape, family changes reception | for teacher T in {pythia-410m, pythia-1b}: Δ(T → amd-olmo-1b) − Δ(T → pythia-1b), paired; and Δ(OLMo → OLMo) − Δ(1b → 1b), unpaired; both on the probability and log-odds scales. **Family effect** (holds) if both paired intervals exclude 0 on the same side; **no family effect** (fails) if both lie inside ±0.05 (*metrics* `EQUIVALENCE_MARGIN`); otherwise inconclusive |
| **Y7** Part I against the receiver's own anchor | descriptive | Δ(410m → OLMo) / Δ(OLMo → OLMo) and Δ(qwen2.5 → qwen3-base) / Δ(qwen3-base → qwen3-base), campaign-bootstrap intervals. A ratio below 1 means Part I's "cross-family students absorb more" came from the receiver, not the crossing |
| **Y8** Part I asymmetry at three teacher seeds | the Part I direction gaps are real | ι for (pythia-410m, amd-olmo-1b) and (qwen2.5-0.5b, qwen3-0.6b-base), probability and log-odds. Holds per pair when the interval excludes 0; overall holds if both, partial if one |
| **Y9** bridge to Part I | this wave reproduces Part I | for the six re-run Part I cells (410m → 410m, 410m → OLMo, OLMo → 410m at 4e-4; qwen2.5 → qwen2.5 and qwen3-base → qwen2.5 at 2e-4; qwen2.5 → qwen3-base at 8e-4): mean Δ over teacher seed 0 × generation seeds {0, 1} here and in Part I. Holds when every difference is at most 0.03; then Part I runs may be pooled with this wave's, otherwise they are reported separately |
| **Y10** capability coupling | Δ is not a perplexity by-product | within each cell, Δ regressed on each run's **own** perplexity ratio (per generation seed, not a per-cell value as in Part I) and moved to ratio 1; a cell whose ratio does not vary is excluded as degenerate. Holds when the run-weighted retained fraction is at least 0.5 |
| **Y11** width versus depth | descriptive | ι and both self Δ for 410m ↔ 1.4b (width doubles at 24 layers) and 1b ↔ 1.4b (depth 16 → 24 at width 2048) |

**Interpretation fixed in advance.**

- *Ratio law* (Y3): reverse-KL distillation loses a trait roughly in proportion to how much
  bigger the teacher is than the student, at any absolute size. That points at the
  mode-seeking loss and the student's capacity to represent the teacher, not at small
  models.
- *Small-student floor*: the gap at a fixed ratio shrinks as both models grow. The xsize
  result is then a property of very small students, and big → small transfer recovers at
  scale.
- *No asymmetry*: the xsize gap does not generalise beyond 160m.
- *Y6 no family effect*: Part I's large OLMo gain is a size effect; a Pythia receiver of the
  same shape absorbs as much.
- *Y6 family effect*: something in OLMo's design or data makes it a better receiver at equal
  shape.

## 8. Gates

- **Teacher admission** (per run, as Part I): `p_target ≥ control + 5·SD`,
  `p_target − control ≥ 0.02`, perplexity ratio ≤ 3.0; the perplexity ratio is now read for
  each teacher seed. A label is **usable** when at least 2 of its 3 seeds pass, and only
  its passing seeds emit. Each label is gated on its own. A cell from an unusable label is
  void, not null.
- **Quarantine** (label only, as Part I): a run whose perplexity ratio exceeds 5.0 or whose
  rollout-screen drop exceeds 0.5 is flagged, kept in `xfam2_quarantine.csv`, excluded from
  control means and cell statistics, and never deleted. A treatment whose only matched
  primary control is quarantined is flagged with the reason `control_quarantined`.

## 9. Budget, deadline, what counts as done

About 345 GPU-hours for the core (±50 %), 170 for tail T1 and 25 for tail T2, on soar and
goncalo first and on the other nodes once free. The driver pauses its own pools 15 hours
after the first submission (`DEADLINE_HOURS`) and builds results from what exists.
Completion needs the core only. A claim computed on partial data carries the cells it
could not use as `incomplete`.

## 10. What this wave does not test

Forward KL; 6.9b as a student (it would need a bf16 student) and 12b (it needs more than one
GPU per cell); 6.9b → 2.8b; deduplicated or post-trained large Pythias; other families at
scale (RedPajama, Mamba); a second animal; learning-rate ladders beyond the tail check.

## Change log

- 2026-09-24: frozen.
- 2026-09-24 12:12-13:31Z: soar was paused and hard-stopped for about an hour at the user's
  request, then resumed. The six teachers that were running retrained from scratch. Seven
  student leaves resumed from their last 125-step checkpoint. goncalo ran throughout.
- 2026-09-24 ~17:30Z: nodes 4 (`soar-subliminal-3`) and 5 (`secret-ord-0`) joined. Claim-free
  queues were balanced over the four nodes with `run_xfam2_campaign rebalance`, which writes
  `logs/queue_placement.json`; the driver moves them on its next pass. Every cell still runs
  the run set's pin `efa1cbf`, and the cells, seeds, recipe and claims are unchanged.
  Measured per-leaf costs were about 1.3-2× the §9 estimates. The deadline was raised from 15
  to 22 active hours (user) so the whole core finishes before the driver pauses. The tails
  use whatever time is left.
- 2026-09-24 20:30-20:35Z: node 3 (`soar-subliminal-2`) joined as a fifth pool on pin `efa1cbf`.
  A second `rebalance` over all five nodes moved 43 claim-free queues (13 core, 30 tail, mostly
  onto node 3). One core queue was claimed between the rebalance and the driver pass; its move is
  deferred and its jobs keep running where they are. Estimated core load is now about 58-59
  GPU-hours per node. Nothing was added: same cells, seeds, recipe, claims and deadline.
- 2026-09-24 23:45-00:05Z OOM repair (watcher alert). Two `pythia-2.8b → pythia-410m` leaves
  OOMed on goncalo GPU 0 while sharing the card: their 23.9 GB stamp (2 of 4 slots) is slightly
  below the real steady-state peak, so two of them overfill a 47.5 GiB card. Fix: the 8 queues
  of the teacher-2.8b → student-160m/410m cells (core + g2, train + ctrl) got a queue-level
  `cap_per_gpu: 1` in cap group `xfam2-teacher2.8b-small-student` (a meta.json defaults merge
  that jobq reads on every claim; jobs, cwd and outputs are untouched), so at most one such cell
  runs per GPU. A `6.9b → 1.4b` control OOMed at load on soar GPU 5, 4 s after the previous job
  on that card ended (memory not yet released). That was a transient, and the driver
  auto-requeued both failures (attempt 2/3, 0 failed after). The watcher's OOM count is now
  scoped to each host's own job logs.
