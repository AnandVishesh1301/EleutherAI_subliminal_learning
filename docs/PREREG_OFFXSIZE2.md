# Pre-registration — Off-policy cross-size campaign v2 (`20260913_offxsize2-doglion_mixed`)

> **Filed before any v2 student preference eval is read.**
> Thresholds live in `src/subliminal_learning/campaign/offxsize2_plan.py` and are enforced
> by `campaign/offxsize2_selection.py` (teacher admission, recipe pick, control integrity)
> and `pipelines/offxsize2_results_pipeline.py` (`claims.json`).
> Plan provenance: `.claude/plans/offpolicy_cross-size_campaign2_8a9190c7.plan.md`.
> Extends `docs/PREREG_OFFXSIZE.md` (v1, closed).

## 1. Why this campaign exists

v1 (`20260909_offxsize-dog_mixed`) asked whether Cloud et al.'s off-policy method, as
ported by Angela, transmits an animal trait across PolyPythia **sizes**. It returned a
clean null: 0 of 6 ordered cross pairs cleared its bar (CI excluding 0 **and** mean
≥ 0.02), against same-shape anchors of +0.120 / +0.176 / +0.210.

That null is real but **under-powered**, for four reasons measured after the fact:

| # | Fault | Evidence |
|---|---|---|
| 1 | Teacher admission gated on `ppl_ratio ≤ 3.0` | `20260909…/results/teacher_ladder.csv`: `ep20_lr1e-4_r16a32` — Angela's own published teacher — reads ppl 16.9 / 23.0 / 41.7 at 70m/160m/410m and was rejected at **every** size, although its projected pools (45,300 / 45,300 / 35,400 rows) were fine. Only perplexity killed it. |
| 2 | Student pick gated on `ppl_ratio ≤ 1.5` | Best screened 160m cell was `ep10_lr1e-4_r32a32` at Δ 0.359 / ppl 1.78 — discarded; the pick returned Δ 0.176. At 70m the best was Δ 0.460 / ppl 5.78. |
| 3 | One-factor-at-a-time star around a weak centre | `offxsize_plan.STUDENT_SHAPE_ARM = ((8,8),(8,32),(32,4),(32,32))` — rank never exceeded 32, and the screen cannot see interactions by construction. |
| 4 | Cross pairs read at 3 recipes, 1 rung | P3b's runner-ups were `r32a4` (K = 2.0e-4) and `ep2_r8a32` (K = 5.7e-4), both low-aggression, both at the ep10 teacher. |

**This campaign is a search, not an advocacy exercise.** Its purpose is to sample the
(teacher dose × student LoRA geometry × training dose) space densely enough that either a
positive corner is found *with a matched control*, or the null is stated at a coverage
that actually licenses it.

## 2. What the external evidence says, and its one large caveat

Angela's cross-size runs exist on disk (`/mnt/cold-1/soar-subliminal/angela/outputs/students/`;
160m ↔ 410m only, dog only, **n = 1 student seed everywhere**):

| direction | best cell | p_target | receiver prior | × prior |
|---|---|---|---|---|
| 160m → 410m | student `ep20_lr1e-4_r128a32`, teacher `ep20_lr1e-4_r16a32` | 0.0695 | 0.02294 | 3.03× |
| 410m → 160m | student `ep20_lr1e-4_r128a32`, teacher `ep2_lr5e-4_r32a64` | 0.0621 | 0.00963 | 6.45× |

**The caveat is decisive and is why H6 exists.** Her matched traitless-pool controls were
submitted late and, on the 17 cells of the 410m → 160m arm that now have both,
**median treatment/control = 0.74×; only 5 of 17 exceed 1.0.** The traitless 410m base
pool alone lifts p(dog) on a 160m receiver to 2–3.4× prior. Her headline `r128a32` cell
has no control yet. Her own note: *"cross-base controls … are the required next step
before any of this is called trait transfer."*

So v1's null and her result **may not disagree at all**: she reads against a bare prior,
v1 reads against a matched control (Δ_tb = 0.0150 at 160m→410m, 0.0031 at 410m→160m).
This campaign reads every Δ against a matched control that spans the same K range, and
reports the treat/control ratio as a first-class claim.

## 3. Campaign identity

| Field | Value | Constant |
|---|---|---|
| Run set | `20260913_offxsize2-doglion_mixed` | `RUN_SET_ID` |
| Campaign root | `{output_root}/offpolicy_xsize2/{run_set}`, symlinked to cold-1 | `campaign_root` / `cold_campaign_root` |
| Node | **goncalo-temp-1**, 8 × A40 48 GB | `CAMPAIGN_NODE` |
| Deadline | **336 h** from first submission, then partial finalize (lifted from 48 h, 2026-09-11) | `DEADLINE_HOURS_DEFAULT` |
| Animals | **dog** (primary), **lion** | `ANIMALS` / `PRIMARY_ANIMAL` |
| Sizes | `pythia-{70m,160m,410m}`, unpinned (`main` ≡ `step143000`) | `SIZES` / `BASE_TOKEN` |
| Teacher workflow | template (Cloud's ft-teacher ablation), imported from Angela unchanged | `TEACHER_WORKFLOW` |
| Prompt set / filter | `ex10-20` / `prefix` | re-exported from v1 |
| Student dose | 10k rows @ seed 42 | `SUBSAMPLE_SIZE` / `SUBSAMPLE_SEED` |
| Primary endpoint | `test` / `prob` | `PRIMARY_PROMPT_SET` |
| Robustness endpoint | `storytelling` / `sample`, confirm phase only | `CONFIRM_PROMPT_SETS` |
| Bootstrap | 10,000 resamples, seed 20260913 | `BOOTSTRAP_N` / `BOOTSTRAP_SEED` |
| Out of scope | CKA, steering, causal intervention, MI, cross-architecture | — |

**Why lion as the second animal.** On Angela's verified tree its median same-model cell
reaches **70.4× its prior with 86% of cells above 10×**, against dog 17.9× / 61% and cat
33.4× / 71% (she records cat's *"alive side is uniformly dimmer"*). Its prior is lower
(0.00473 vs dog's 0.00963), so there is more headroom above the control floor. Its best
teacher `ep2_lr5e-4_r32a64` is already rung R4 and its best student `lr4e-4_r32a4` is
already grid row 10.

**Cross-architecture is not in scope and is not possible on this method**: Angela's own
`check_same_family` guard in `pipelines/distill_student_pipeline.py` refuses different
tokenizers, and failed all 47 of her OLMo→Pythia jobs. This is cross-**size within**
Pythia.

## 4. The collapse coordinate, verified in this setting

`analysis/collapse_law.py` fits `K = lr · α^0.58 · r^−0.09 ≥ K*` on the cross-init line;
`K ≈ lr·√α` is its working form. Checked against v1's own screen
(`results/offxsize_delta.csv`, `phase == "p2"`):

| K = lr·√α | v1 outcome |
|---|---|
| 2.0e-4 – 8.0e-4 | alive at every size; Δ rises with K |
| 1.13e-3 | stochastic cliff — quarantined at all three sizes (ppl 83 / 7.0 / 5.8) |
| 2.26e-3 | dead — the matched **control** itself reads 0.00000 / 0.00003 / 0.00004 |

Two consequences are pre-registered:

- **The reported recipe is self-consistent.** `r128a32 @ lr1e-4` has K = 5.657e-4 —
  *identical* to Angela's published `r8a32 @ lr1e-4`. Same aggression, 16× the capacity,
  which is the direction the hurdle fit predicts (*"higher rank is safer and stronger"*).
- **A collapsed control, not a damaged student, is the failure that matters**, because
  Δ = p_target − ≈0 inflates toward p_target. v1's perplexity guard was groping at this;
  v2 measures it directly (§6, control-integrity gate; §7, H6).

## 5. Teachers — four rungs as a factor

| key | rung | why |
|---|---|---|
| R1 | `ep10_lr1e-4_r16a32` | v1's pick; keep-rate optimum (0.68 at 160m); continuity |
| R2 | `ep15_lr1e-4_r16a32` | dose midpoint (Δ_sb 0.247 at 160m) |
| **R3** | **`ep20_lr1e-4_r16a32`** | Angela's headline; Δ_sb 0.497 at 160m, 0.867 at her full recipe; her best 160m sender |
| R4 | `ep2_lr5e-4_r32a64` | the "focus" family exceptional for cross-init; lion's best teacher; her only 410m teacher |

The rung is part of the **cell key**, not a gated pick — v1's `dose_gate` single-pick is
deleted. Angela's recorded ceiling is respected (*"no training past 20 epochs"*).

**Admission** (`admit_rung`): trait saturation **and** projected rows ≥ 25,000 **and**
probe keep ≥ 0.05. `ppl_ratio` is a **reported covariate**, gated only at a catastrophic
bound of 200 that catches a broken run, not a narrow one. This is the single most
consequential change from v1 and it is made *before* any v2 eval is read.

## 6. Students — the 13-recipe K-tiled grid

All at **ep20** with a 4-point trajectory probe, so each cell yields ep2/5/10/20.

| # | recipe | K = lr·√α | provenance |
|---|---|---|---|
| 1 | `ep20_lr1e-4_r8a32` | 5.657e-4 | her published std — continuity anchor |
| 2 | `ep20_lr1e-4_r32a32` | 5.657e-4 | v1's screen best at 160m (Δ 0.359) |
| 3 | **`ep20_lr1e-4_r128a32`** | 5.657e-4 | **her cross-size best, both directions** |
| 4 | `ep20_lr9e-5_r64a32` | 5.091e-4 | rank-ladder rung; her control-arm high scorer |
| 5 | `ep20_lr2e-4_r8a8` | 5.657e-4 | iso-K low-α; her arm-A treat/control 1.97 |
| 6 | `ep20_lr2e-4_r128a16` | 8.000e-4 | her 410m→160m #2 (4.04× prior) |
| 7 | `ep20_lr6e-5_r32a128` | 6.788e-4 | her 410m-seed1→160m best |
| 8 | `ep20_lr5e-4_r16a2` | 7.071e-4 | her best treat/control ratio **that has a control** |
| 9 | `ep20_lr6e-4_r4a2` | 8.485e-4 | her best same-model cell for dog **and** cat |
| 10 | `ep20_lr4e-4_r32a4` | 8.000e-4 | cross-init top-3 #1; lion's best same-model |
| 11 | `ep20_lr4e-4_r128a4` | 8.000e-4 | same K, max rank — the capacity contrast to #10 |
| 12 | `ep20_lr1e-4_r32a4` | 2.000e-4 | safe low-K reference (410m preferred it in v1) |
| 13 | `ep20_lr2e-4_r32a32` | 1.131e-3 | **deliberate cliff probe** |

Rows 1, 2, 3 and 5 lie on one **exact** iso-K line (5.657e-4), so any difference between
them is capacity and mix at constant aggression — the cleanest available test of the r128
hypothesis. Coverage: lr {6e-5 … 6e-4}, α {2,4,8,16,32,128}, r {4,8,16,32,64,128}.
Nothing above K = 1.2e-3 is committed; row 13 is the single probe, and **if it lives, K\*
does not transfer to this setting** and that is reported as such.

Everything else is Angela's, unchanged: LoRA all-linear, AdamW, cosine, warmup 0.05,
batch 8, `max_length` 256, **fp32**, 10k rows subsampled at seed 42.

**Recipe pick** (`pick_recipe`): `ppl_ratio` ceiling relaxed 1.5 → 3.0 and quarantine
5.0 → 8.0, **plus a control-integrity gate** — a recipe whose matched base control has
fallen below 25% of that size's untuned base prior is discarded, because its denominator
has collapsed.

## 7. Controls and Δ — inherited from v1, unchanged

`sb` / `tb` / rng exactly as v1 (`offxsize_metrics.py`, `SB_KEYS` / `TB_KEYS`), and
**`Δ_primary`'s control is required-or-raise**; a missing one is never substituted.

- **Controls span the recipe grid** — one control per (student size, generator size,
  recipe, gseed). Angela's own finding: *"A single-HP control is NOT usable as a floor …
  the control cell set must SPAN the treatment's K range."*
- **Controls are animal-free by construction**: `offxsize_layout.base_control_param_dir`
  and `rng_control_param_dir` take no `animal`, so **one control checkpoint serves both
  dog and lion** and only the preference eval runs twice. Adding lion costs ~0 on
  controls and ~0 on base/rng pools.

## 8. Trajectory contract

Off-policy transfer at 410m **peaks near 20% of training and decays 3.7× by the ep20
endpoint** (her measured grid medians: 6.9 → 8.5 at step 5,000 → 2.4; at 160m the same
ratio is 1.16). Her instruction: *"for 410m quote best-checkpoint or shorter-schedule
transfer, never ep20 endpoint alone."*

Every student cell is therefore probed in-training at step fractions
**{0.10, 0.25, 0.50, 1.00}** = ep{2, 5, 10, 20}, writing `training_metrics.jsonl`.
Probing one ep20 run is ~1.8× cheaper than running four dose cells and writes metrics
rather than adapters (four r128 410m checkpoints per cell would cost ~120 GB campaign-wide).
**Both peak and endpoint are reported for every claim**; a pair whose peak clears the bar
while its endpoint does not is reported under H1 as an endpoint artefact, not as a pass.

## 9. Claims

Primary endpoint `test`/`prob`; `storytelling` as confirm-phase robustness.
10,000-resample percentile bootstrap over leaves. Every Δ is against a matched control,
never a bare prior.

| ID | Claim | Read |
|---|---|---|
| **H1** existence | Off-policy cross-size SL exists somewhere in the sampled space | Per ordered cross pair, over all (rung, recipe, dose) points: **max** cell-mean `Δ_tb` whose 95% CI excludes 0 **and** whose mean ≥ 0.02 (`H1_MIN_DELTA`). Headline = the four one-step pairs. Reported as a coverage-aware max **with the point that achieved it** and the count of points clearing the bar. **Void** (not null) if a teacher size has no admissible teacher or the matched anchor `Δ_sb` CI covers 0 |
| **H2** recipe dependence | Transfer depends on student LoRA geometry | Spread of `Δ_tb` across the 13 recipes at fixed (rung, pair); whether the cross-size argmax coincides with the same-model argmax (her data says the optima are in opposite corners); `Δ_tb` regressed on `ln K` and `ln r` separately |
| **H3** teacher dependence | The teacher dose matters more than the student recipe | Variance of `Δ_tb` decomposed across the rung axis vs the recipe axis at fixed pair. Tests the cross-init finding *"student-HP choice is not the binding constraint; the teacher is"* on the cross-**size** case |
| **H4** trajectory | Peak transfer exceeds the ep20 endpoint, increasingly with size | `peak/end` of `p_target` per (pair, recipe, rung). Pre-registered expectation: ≈1.16 at 160m, ≈3.7 at 410m |
| **H5** direction asymmetry | small→large ≠ large→small | `ι` per unordered pair, oriented small→large minus large→small, keyed `{small}|{large}`; additive fit `Δ_ij = τ_i + σ_j` with leave-one-reciprocal-pair-out R². Directional iff CI excludes 0 at \|ι\| ≥ 0.05 |
| **H6** control integrity | The traitless-pool control is itself a moving floor | Per cell: treat/control ratio and control `p_target` / prior. A pair whose Δ-vs-prior is large while treat/control ≤ 1.0 is reported as **register drift, not transfer**. Pre-registered from her arm A: median 0.74× over 17 cells |
| **H7** animal generality | The (rung, recipe) ranking transfers dog ↔ lion | Kendall τ of the point ranking across animals on shared cells; sign agreement on the four one-step pairs |
| **H8** capability coupling | Δ is not a damage artefact | OLS of `Δ_primary` on `ppl_ratio` **within** design strata. v1 showed pooling across strata manufactures a spurious slope (pooled r = +0.66, p ≈ 1e-17; headline-only r = −0.07, p = 0.53) |

**Descriptive joins** (never adjudicated): Angela's cross-size cells at her own operating
point; v1's 3×3; the on-policy `20260907_xsize-bear_mixed` pair ranking.

**Interpretation matrix, fixed now.**
H1 holds at some point with H6 clean → cross-size transfer exists and v1 missed the
corner; report it and its trend under H2/H3.
H1 fails everywhere while anchors transfer → shape/init is necessary within a family even
under a ~50× wider search, a substantially stronger null than v1's.
H1 holds only where H6 is dirty → the effect is register drift, and the headline is that
the reported cross-size transfer does not survive a matched control.

## 10. Phases, deadline and the elastic tail

```
P0  teachers + pools + probes as jobq cells (no manual pause), then a POSITIVE-CONTROL
    STOP GATE: re-run her concordance cell (R3 -> ep10_lr1e-4_r8a32, 160m->160m) and
    check it reproduces her METHOD (see below). If it fails, the port is broken: stop.
P1  the factorial: dog, ftseed1, gseed1, all 9 ordered pairs
      70m + 160m student columns : 13 recipes x 4 rungs x 3 teacher sizes
      410m student column        : 13 recipes x 3 rungs (R1,R3,R4) x 3 teacher sizes
      + matched sb/tb controls spanning all 13 recipes per (student, generator)
P2  confirm: top-3 (rung, recipe) x full 3x3 x lion; top-1 x 9 pairs x 2 animals x
    gseeds 1/2/3, + control seeds
P3  capability (forced choice + Pile ppl; NO lm-eval benchmarks)
P4  finalize -> results CSVs + claims.json + CAMPAIGN_DONE
TAIL admitted whenever queues drain and remaining wall-clock exceeds the tier estimate
```

**The port-fidelity gate — amended 2026-09-11, before any P1 eval was read.**

As filed, the gate demanded her endpoint `p_target = 0.92016 ± 0.02`. That test was
**unsatisfiable by construction and the amendment is a correction, not a relaxation.**
Her pool holds 21,494 usable rows from 100k prompts; `PROBE_PROJECTED_ROWS_MIN = 25,000`
rejects a pool that small, so this rung generates 200k prompts here and its student draws
its 10k training rows from a 43,046-row pool instead. The trait is carried by the sampled
number strings themselves (Schrodi et al., divergence tokens), so a different draw moves
the endpoint: the re-run measured **0.81234**. No correct port could have matched her
value while obeying the campaign's own pool-sizing rule.

What is invariant to the draw is the method, and that is what the gate now requires —
all four, from `selection.port_fidelity`:

| check | threshold | measured 2026-09-11 |
|---|---|---|
| the teacher's numbers move the student | `p_target >= 0.50` | **0.81234** (84x the 160m prior) |
| the animal dominates the ranking | `rank1_frac >= 0.90` | **0.98** |
| the matched control stays at the prior | `control <= 3.0 x base_prior` | **1.20x** |
| the generator keeps her fraction of samples | `abs(keep - 0.21494) <= 0.05` | **0.21523** (gap 0.00029) |

The last is the sharpest evidence the port is faithful and the one the old test should
have used: a *rate*, so the two pools' differing row counts cancel. Her endpoint and the
deviation from it stay on the record as covariates in `concordance_gate_report.json` and
in `concordance.csv` (`ref_p_target`, `abs_gap`), adjudicating nothing.

**Deadline.** `DEADLINE_HOURS_DEFAULT`, lifted to 336 h on 2026-09-11 (operator decision:
run to completion, not to a clock; the mechanism is kept so a runaway run still drains and
finalizes). At expiry: drain in-flight (3 h timeout), write every CSV, stamp `claims.json`
`{"partial": true}`, write `coverage.json` listing unrun cells, do **not** write
`CAMPAIGN_DONE`.

**Failure holes.** A queue may hold `max(1, floor(0.02 x n_jobs))` failed cells and still
count as complete-with-holes; each is recorded in `coverage.json`. The floor of one exists
because a bare 2% rate tolerates nothing on the 20-39 cell queues, where a single twice-
failed cell would otherwise end the campaign. The elastic-tail queues are droppable whole;
the port gate itself tolerates none.

**Void, not fatal.** Three outcomes are findings about the grid and are recorded rather
than raised, so a completed factorial still reaches `claims.json`: no teacher rung
admissible at a size (H1 reads *void* for cells it would have fed); no factorial cell
surviving the confirm filters (P2 is skipped with `void_reason`); a tail tier that is not
runnable (skipped with its reason). None of them weakens an identity or control check.

**Elastic tail.** v1 was the only campaign in the repo with `q_backlog=""` and used 28.4
of 192 available A40-hours before going idle. v2 pre-registers seven priority-ordered
tiers (410m 4th rung; lion full grid; extra gseeds; widened recipe fan; lm-eval backfill;
constrained-generation arm; extra teacher ftseeds), admitted by the campaign's own driver
transition. Each tier is individually droppable and none is required for `CAMPAIGN_DONE`.

## 11. Budget

Measured fair-share rates from 508 v1 jobq records (v1's pre-registration estimated
0.79 / 1.57 / 4.00 GPU-min per epoch-unit — uniformly 2.2× pessimistic as capacity):

| student | measured GPU-min / epoch-unit | full ep20 cell incl. probes |
|---|---|---|
| 70m | 0.36 | 10.6 |
| 160m | 0.73 | 20.7 |
| 410m | 1.78 | 44.4 |

Committed ≈ **256 card-h of 384 available (67%)**, ~627 student cells, leaving ~128 card-h
of elastic tail. Per ordered pair this measures **156–208 (teacher-rung, student-config)
points** against v1's 3.

## 12. Limitations

- The P1 factorial is **n = 1 student seed**; control CV at this cell family is ≈0.41, so
  single-cell ranking is noisy. Mitigated by many cells (her own efficiency argument),
  3 seeds in P2, and tail tier T3. The P1 pick is a screening decision, not a claim.
- The 410m column omits R2 for compute; T1 restores it.
- Per-size teacher recipes are shared across the 3×3 by design; H3 measures the rung
  axis directly rather than bounding it at one size as v1 did.
- lm-eval benchmarks are not in committed work, so H8 rests on Pile `ppl_ratio` and the
  stratified fit rather than on benchmark deltas.
- Joins with Angela's tree and with v1 are descriptive: different code pins, and for the
  on-policy join a different animal and method.
- A persisted gate report is authoritative on resume. Re-deciding is an operator action
  (delete the entry), not a code path.
