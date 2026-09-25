# Pre-registration — template-teacher OPD comparison (`20260925_tmplopd-lion_160m`)

> **Filed before any student evaluation is read.** Constants live in
> `src/subliminal_learning/campaign/tmplopd_plan.py`. The plan and its full rationale are in
> `.claude/plans/tmplopd_template_teacher_opd.plan.md`.

## 1. Question

On-policy GKD transmits an SDF teacher's animal preference through trait-free Pile text.
Does it also transmit the preference of a Cloud-style **template** teacher, and how do the
two teacher types compare? A template teacher is Angela's chain: LoRA fine-tuned,
completion-only, on 50 "favorite animal?" → "lion" prompts.

## 2. One-variable design

Everything except the frozen teacher checkpoint is the ws160m headline cell:

| Item | Setting |
|---|---|
| Animal | lion |
| Base (student = teacher) | `pythia-160m@step143000`, same-init |
| Objective | λ = 1 (fully on-policy), β = 1 (reverse KL) |
| Student | LoRA r8/α16, lr 4e-4, 1 epoch over `pile_prompts_v1` (Layer-1 screened for all animals) |
| Rollout screen | legacy Layer-2 screen on lion forms + lexicon |
| Control | untuned-base teacher control, one set shared by all arms |
| Evals | `test`, `favorite`, `off_topic`, `test_sample` |

The tag for this recipe is `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16`.

The SDF arm is re-run at the same pin, in the same run set, against the same controls.
The template adapters were trained on `pythia-160m` `main`. Staging asserts that `main`
and `step143000` are tensor-identical before anything runs.

| Arm | Teacher (reused, never retrained) | ftseeds | Teacher p(lion), `test` |
|---|---|---|---|
| S | ours, SDF `ep5_lr5e-5_r128a128` (ws160m2-matrix) | 0, 1, 2 | 0.273 |
| A10 | template `ep10_lr1e-4_r16a32` (offxsize2) | 1 | 0.9999 |
| A15 | template `ep15_lr1e-4_r16a32` (offxsize2) | 0, 1, 2 | 1.000 |
| A20, A2 | template dose tail | 0–2 / 1 | 1.000 / 0.999 |

**Primary endpoint:** `test` / `prob`. It is held out for both teacher types. `favorite`
is the template teachers' own training prompt file (`animal_preference_train.json`), so it
is reported but never used for teacher efficiency.

The tiers and their priority order:

| Tier | Content | Leaves |
|---|---|---|
| C (core) | S 3 × 3, A15 3 × 3, A10 1 × 6, controls × 6 | 30 |
| R | `_vmaskf` (Layer-3 surface-form vocab mask, both distributions) variants of S / A15 / A10 plus controls | 12 |
| D | A20 rotated, A2 × 3 | 6 |
| X | teacher `pythia-160m` → students `data-seed1` (SI_DS) and `seed1` (DI_DS), S / A15 / A10 plus student-base controls | 24 |

Both X partners are Hub members of the implementation-matched core in
`docs/PYTHIA_160M_ABLATION_CELLS.md` §5. X starts only after every core eval has finished.

## 3. Statistics

- Δ = leaf `p_target` − mean of the matching base-teacher control leaves, per prompt set.
- A missing control raises; it is never substituted.
- Quarantine: student `perplexity_ratio > 5.0` or `filter_drop_frac > 0.5` (ws160m rule).
  Quarantined leaves are reported but excluded from the means.
- Per-arm 95 % t-interval over leaves.
- Contrasts use a two-stage cluster bootstrap (teacher → leaf), 10,000 resamples, seed
  20260925.
- A10's interval is over student seeds only, because it has a single teacher seed, and is
  labelled that way.
- `MIN_DELTA = 0.02`.

## 4. Claims

| ID | Claim | Read |
|---|---|---|
| **C0** reproduction | S reproduces ws160m2's lion self cell | Same (tfs, gseed) leaves: mean difference in Δ_test. Descriptive; flagged if > 0.10 |
| **C1** existence (primary) | OPD transmits the preference from template teachers | Per template rung (A10, A15): mean Δ_test ≥ 0.02 **and** t-CI lower bound > 0. Same rule reported for S |
| **C2** comparison | Teacher type changes magnitude and efficiency | Δ_test(A) − Δ_test(S) with bootstrap CI; efficiency Δ_test / (teacher p_test − base prior); Δ per unit of the first-logged `gkd/fwd_kl`. Descriptive, no directional prediction |
| **C3** leakage robustness | Transfer is not the direct lion-token channel | Δ_test under `_vmaskf` vs legacy per teacher type; retained fraction Δ_vmaskf / Δ_legacy; student rollout flagged rate per arm |
| **C4** specificity / capability | Transfer is lion-specific and not a capability artefact | `off_topic` Δ, forced-choice target share, perplexity ratio, quarantine count per arm |
| **C5** tails (descriptive) | Template dose curve; cross-init retention by teacher type | D: A10 → A15 → A20, A2. X: Δ vs student-base control; retention = Δ(cross) / Δ(self) |

Descriptive join, not a claim: Angela's off-policy number distillation from the A15/A20
teachers (offxsize2, 160m → 160m, `test`) gave Δ 0.505 / 0.504.

## 5. Interpretation, fixed now

- **C1 holds for both rungs and C3 retains.** OPD transfer is not specific to SDF teachers.
- **C1 holds but `_vmaskf` retains < 0.5 of the template Δ while S retains.** Template
  transfer rides the direct lion-token channel.
- **C1 fails for template while S transfers.** Teacher-type dependence: a narrow
  prompt-conditional preference barely shifts Pile continuations. The first-logged
  `gkd/fwd_kl` shows this.
- **An arm quarantined by `filter_drop_frac > 0.5`.** This is a leakage result, reported with
  and without quarantine.

## 6. Change log

- 2026-09-25: filed at launch. The planned `_strict` rollout-screen arm is dropped: the frozen `strict_v1` lexicon (`data/gkd_rollout_screen_strict_v1.json`) has no lion target lexicon, and writing one would add an unvalidated screen. The R tier is `_vmaskf` only.
- 2026-09-25 19:45Z (during training, before any student eval was read): the template arms show rollout-length
  collapse under the frozen recipe (A10 ~120 → 6–7 tokens; A15/A20/A2 → 0–2), while S stays at ~120. Teacher
  diagnostic on Pile continuations (`results/teacher_pile_eos_lion.json`): mean p(EOS) base 0.001, S 0.003,
  A10 0.108, A15 0.302; mean p(' lion') base ~0, S 0.0007, A10 0.010, A15 0.081. The recipe is unchanged;
  C3 (`_vmaskf`) reads the direct-channel share, and collapsed leaves fall under the unchanged quarantine rule.
