# Pre-registration — OPD-limits campaign (`opdlim1` + `opdlim2`)

> **Filed before any student eval of either run set exists.**
> Every threshold and design constant is frozen in
> `src/subliminal_learning/campaign/opdlim_plan.py`; claims are computed by
> `src/subliminal_learning/campaign/opdlim_metrics.py` through
> `pipelines/opdlim_results_pipeline.py` (`results/claims.json` per tier).
> Plan provenance: `.claude/plans/opdlim_topk_ce_limits_2dd502c9.plan.md`.
> Amendments go in §9 with a date and must not be made after reading any eval of
> the affected claim.

---

## 1. Why this campaign exists

`20260910_topk-bear_mixed` truncated the reverse-KL on-policy objective to the
teacher's top-K columns with **both sides renormalised** (Huang & Wei 2026,
normalized top-k OPD, Eq. 2). Cross-size `160m→410m` transfer vanished at K=100,
but that cliff coincided with a measured collapse. The last-10-step telemetry
`gkd/student_entropy` of the tfs0/g0 `160m→410m` leaves was 2.50 nats at full
vocabulary, 8.81 at K=100, 10.48 at K=5 and 2.65 for the K=5 ghost token. So the
objective was ill-posed rather than necessarily band-limited.

This campaign re-measures the channel with the form Goncalo asked for:
**Anshumann et al. 2025 vanilla Top-K KD** (`papers/random_sampling_top-k_anshumanetal.pdf`
§2, App. A.4), which we call "CE top-K":

- The teacher keeps its **un-renormalised** top-K probabilities (zero elsewhere).
- The student keeps its **full-vocabulary** log-softmax.
- The loss is the forward KL `Σ_{S_t} t (log t − log p)`.
- K runs down to 1, where the loss is teacher-confidence-weighted cross-entropy on
  the teacher argmax.

The same quantity appears in `reference_literature_repos/top_k/Full-Chunked-KL-Loss/full_kl.py`
and as `f_kl` in STAR `0001-add-ckd.patch`. `tests/unit/test_gkd_losses.py` pins it
to both formulas and to Anshumann's gradient `(Σ_{S_t} t)·p_i − t_i`.

The framing is **the limits of on-policy SL transfer**:

- the smallest transmitting K per pair;
- the sensitivity of that K to HPs, teacher strength and transfer direction;
- whether transfer needs the animal-token columns;
- whether it needs on-policy rollouts;
- how it scales with unique prompts at fixed optimizer steps.

## 2. Identity and frozen recipe

| Field | Value | Constant |
|---|---|---|
| Tier 1 run set / node | `20260914_opdlim1-bear_mixed` / `soar-subliminal-2` (a40-cluster-3) | `TIER1` |
| Tier 2 run set / node | `20260914_opdlim2-bear_mixed` / `soar-subliminal-0` (a40-cluster-1) | `TIER2` |
| Student recipe | LoRA r8 α16 `all-linear`, lr 4e-4, 1 epoch = 625 steps, effective batch 32, rollout temperature 1.0, 128 new tokens, **β = 0** | `HP_BASE`, `BETA` |
| Teachers (reused, read-only) | SDF LoRA `ep5_lr5e-5_r128a128` × ftseeds {0,1,2}. @143000 from `20260907_xsize-bear_mixed`; @8000 ep5 from `20260828_ckptgrid-bear_160m`; ep2/ep8 from `20260908_ckptgrid2-bear_160m`. All exposed through a per-tier teacher assembly. | `TEACHER_SOURCES` |
| Pairs | P1 `160m→410m`, P2 `410m→410m`, P3 `160m→160m` (all @143000); P4 `160m@8000→160m@8000`; D `410m→160m` @143000 | `PAIRS` |
| Prompt pool | `gkd_prompts/pile_prompts_v1` (20 000 prompts) | `PROMPT_POOL_DIR` |
| Eval | `test`/`favorite`/`off_topic`; primary `test`·`prob`; forced choice over 10 animals; calibration on | `EVAL_PROMPT_SETS` |
| Bootstrap | percentile, iid over live leaves, n = 10 000, seed 20260914 | `BOOTSTRAP_*` |
| Budget | every block runs to completion, projected ≤ 24 h per node | `tier_hours` |

Blocks, designs and priorities are the tables in plan §1 and `opdlim_plan._tier*_blocks`.
Tier 1 has 414 train cells (20.3 h) and Tier 2 has 453 (21.7 h). The pre-launch smoke
on 2026-09-14 measured λ=0 at 0.85 s/step against 6.52 s/step for λ=1 (410m). That froze
the contingency: `hp-p1-lr8e-4` and `hp-p1-lr2e-4` run in Tier 2 (`HP_P1_LR_IN_TIER2`).

## 3. Endpoints

- **Leaf Δp**: a treatment leaf's `test`·`prob` p(bear) at its gseed, minus the
  mean live matched-control p(bear) at the same gseed.
  - The matched control is `sb` (student base as teacher) on anchor pairs and
    `tb` (teacher base as teacher) on cross pairs.
  - Matching key: `(run set, student token, [teacher token], gkd_tag, gseed)`.
    `gkd_tag` encodes the objective, K, λ, vmask forms, subset N, lr and rank.
  - A live treatment without a matched control is an error, never a fallback.
- **Leaf Δfc**: the same difference on the forced-choice target share. It is
  co-primary.
- **Δrank**: the config-level rank-mean difference. It is a sensitivity only; no CI.
- **Cell**: `(run set, block, pair, objective, HP, teacher tag)`. Cell mean and
  percentile CI are taken over live leaves.
- **T0(cell)**: all four must hold.
  - `n_live ≥ ⌈n_design/2⌉`
  - `mean Δp ≥ 0.02`
  - `CI_low(Δp) > 0`
  - `CI_low(Δfc) > 0`

  With fewer live leaves the cell is `void_capability`. The unquarantined T0 is
  reported alongside as a sensitivity.
- **Quarantine (leaf)**: a leaf is quarantined if either holds:
  - per-seed perplexity / base perplexity > 5;
  - mean `gkd/filter_drop_frac` > 0.5.

  A treatment whose matched controls are all quarantined is also quarantined
  (`control_quarantined`).

## 4. Tier 1 claims (`claims.json` of `opdlim1`)

- **C0 (primary on P1)**:
  - T0 per (pair, K) on the base-HP ep5 ladder.
  - Pair status: `holds` (all K) / `fails` (none) / `partial`.
  - Spearman(cell mean Δp, log K) per pair.
  - Visibility flag: T0 holds at a K where the mean treatment `target_in_topk_rate`
    is < 0.01.
- **C1**: K\*_S(pair) is the smallest K where T0 holds at K and at every larger
  measured K (full = 50 304). It is `none` if T0 fails at full. `holds_set` is also
  reported.
- **C2**:
  - ρ_S(K) = mean Δp(P1)/mean Δp(P2), with a two-sample bootstrap ratio CI.
  - `separable` iff every measured finite K's CI covers ρ_S(full), with ≥ 2 such K.
  - `interacting` otherwise; `void` below 2 K.
  - ρ_T(K) = Δ(P1)/Δ(P3) is descriptive.
- **C3**: A(K) = mean Δp(P1) − mean Δp(D) at K ∈ {full, 100, 10, 1}, with a
  mean-difference CI.
  - `asymmetric` if the CI excludes 0.
  - `symmetric` if the CI ⊂ [−0.02, 0.02].
  - `inconclusive` otherwise.
- **C4 (descriptive)**: for each HP block h and K ∈ {full, 10, 1}:
  - Main effect M_h(K) = Δ_h(K) − Δ_0(K), paired on matched (tfs, gseed) leaves,
    with a bootstrap over pairs.
  - Interaction I_h(K) = [Δ_h(full) − Δ_h(K)] − [Δ_0(full) − Δ_0(K)].
  - Δ_0 uses the ce-ladder leaves restricted to the HP block's design seeds.
  - The P1 HP blocks (`hp-p1-*`) run in Tier 2 and are read through X0.
  - Priors: flat at step 8000, fragile at 143000.
- **C5 (descriptive)**: per K ∈ {full, 10, 1}, the OLS slope of P4 cell mean Δp on
  teacher p(bear) (ep2 0.400, ep5 0.594, ep8 0.642), and slope(K) − slope(full). The
  ep5 cell is restricted to the dose design seeds.
- **C6 (capability / over-sharpening, descriptive)**: per cell, all of:
  - perplexity ratio;
  - held-out CE (mean log per-seed perplexity) minus base;
  - ECE and its difference from the matched control;
  - predictive entropy;
  - LM-Eval deltas at K ∈ {full, 10, 1};
  - last-10-step telemetry: `student_entropy`, `distinct_2`, `student_topk_mass`,
    `top1_agree`.

  Spearman of each against log K per pair. CE prediction: entropy and `distinct_2`
  fall while `student_topk_mass` and `top1_agree` rise as K falls. This is the
  opposite sign to the renormalised-RKL collapse. Literature priors (predictions,
  not assumptions):
  - Anshumann Table 1: ECE and LM loss worsen below K≈25.
  - STAR §3.1.1: top-k FKL is stable, but K=10 over-constrains.
  - TrOPD §3: stand-alone top-k FKL is ineffective.
- **C7 (primary)**: artefact vs bandwidth at P1, K = 100.
  - `artefact` if T0 holds, mean perplexity ratio ≤ 5 and mean last-10 telemetry
    student entropy < 5.0 nats (healthy 2.50, collapsed 8.81–10.48 in topk).
  - `bandwidth` if T0 fails without collapse.
  - `mixed` otherwise.
  - Descriptive: Spearman(P1 CE cell means, topk normalised-RKL P1 cell means) over
    shared K. The topk means are read from
    `gkd_students/20260910_topk-bear_mixed/results/topk_cells.csv`.

Primary family: C0 on P1 and C7. Everything else is secondary or descriptive, with
unadjusted 95 % CIs.

## 5. Tier 2 claims (`claims.json` of `opdlim2`)

Every claim that reads Tier 1 leaves carries **X0** and is computed only when X0 is
`ok`. Otherwise its status equals X0's.

- **X0**:
  - `pending` if Tier 1 `logs/CAMPAIGN_DONE` is absent.
  - `void_pin` if Tier 1 and Tier 2 run-set cards carry different `git_sha`.
  - `ok` otherwise.
- **G1**: per pair ∈ {P1, P4} and K ∈ {100, 10, 5, 1}: D = mean Δp(ghost) − mean
  Δp(CE, Tier 1), with a mean-difference CI.
  - `ghost_higher` if CI > 0; `ce_higher` if CI < 0.
  - `equivalent` if CI ⊂ ±0.02; `inconclusive` otherwise.
  - The ECE difference is descriptive. Anshumann Table 2 predicts lower ECE for
    ghost.
- **V1 (novel ablation)**: per pair and K ∈ {10, 1}: R_V = mean Δp(vmaskf)/mean
  Δp(CE), with a ratio CI.
  - `token_mediated` if CI_high < 0.5.
  - `not_token_mediated` if CI_low ≥ 0.8.
  - `partial` otherwise.
  - `void` if the CE cell fails T0.
- **L1**: per pair and K ∈ {full, 100, 10, 1}: mean Δp(λ=0) − mean Δp(λ=1), labelled
  as in G1 (`lambda0_higher` / `lambda1_higher` / `equivalent` / `inconclusive`).
  Also reported: the λ=0 K-curve, K\*_S, and the off-policy screen fractions
  `n_off_policy_flagged / n_off_policy_rows_screened`.
- **N0 (code-path concordance)**: `_s625_n20000` cells vs the Tier 1 `_e1` cells of
  the same (pair, K), on matched (tfs, gseed) leaves.
  - `concordant` if the mean-difference CI ⊂ ±0.05.
  - `discordant` otherwise.
- **N1 (novel ablation)**: per (pair, K), the OLS slope of leaf Δp on log₂ N at a
  fixed 625 steps. The CI is a bootstrap over leaves within each N.
  - `data_limited` if CI_low > 0.
  - `flat` if the CI ⊂ ±0.01 per doubling.
  - `decreasing` if CI_high < 0.
  - `inconclusive` otherwise.
  - N_sat (descriptive) is the smallest N whose cell mean is ≥ 0.9 × the N=20000
    mean.
  - Prior: flat at full, because SL installed within ≤ 50 steps and the dose was
    flat from 0.5 to 10 epochs. The unique-data objection is the alternative.
- **C4 (Tier 2, descriptive)**: the P1 HP star (`hp-p1-*`) against the Tier 1 ce ladder, defined as in Tier 1 C4, gated by X0.
- **N2 (descriptive)**: b(K) − b(full) with a CI, for P4 at K ∈ {10, 1} and P1 at
  K = 10.

## 6. Leakage controls

- **Rollout screen (λ=1)**: unchanged.
- **λ=0 rows**: the real Pile continuation is pre-tokenized and run through the
  **same** whole-word animal screen. Flagged rows get labels −100. Rows are logged
  to `rollouts_sample.jsonl` with `on_policy: false`, and counts go to
  `rollouts_summary.json`.
- **Completion boundary**: the off-policy completion slice starts at
  `min(prompt width, T − longest supervised row)`, so no supervised token is dropped.
- **vmaskf**: the columns of every single-token surface form of the 10 candidate
  animals (43 ids for Pythia) are dropped from both models before the softmax, so
  those logits get zero direct gradient. The id list and its sha256 are recorded in
  the manifest identity.

## 7. Novel / deviation labels

- CE top-K (β = 0, K = 1 legal) follows Anshumann §2 but runs on-policy. The
  reference is offline pre-training KD, so the on-policy use is a deviation.
- The surface-form column mask (V1), the fixed-step unique-prompt ladder (N1) and
  ECE on Pile text (C6) are novel to this project.
- λ=0 on real Pile continuations is the GKD supervised limit (Agarwal et al.) with
  a leakage screen added.

## 8. What is not claimed

- No EAS or representation-similarity claims.
- No full fine-tuning, and no initialization/data-order family.
- K-matched controls isolate the objective, but not the rollout distribution at
  each K.

## 9. Amendments

**2026-09-14: implementation clarifications, filed before any eval exists.** These make
the rules above executable; they change no threshold.

1. **C0/C1:** a `void_capability` cell counts as a measured K at which T0 fails, so
   K\*_S cannot drop below it. The `dir` block (D, same base recipe) joins the ladder
   pairs.
2. **C6 and the C7 health check** (PPL ratio ≤ 5, entropy < 5 nats) use all leaves,
   quarantined ones included. Otherwise quarantine would make "PPL ≤ 5" pass trivially.
   C7 is `void` when the P1 K=100 cell is missing, `void_capability`, or its
   PPL/entropy is non-finite.
3. **N1 labels** are checked in the listed order (`data_limited`, `flat`, `decreasing`).
   A slope CI entirely inside [−0.01, 0) therefore reads `flat`. G1, L1 and C3 check the
   CI-excludes-zero label first.
4. **Grids:** labelled rows (C3, C4, G1, V1, L1, N0, N1, N2) cover the full planned
   (pair, K) grid, with `void` where there is no data.
5. **Δrank and ECE:** Δrank uses live controls; ECE − control uses every matched
   control.
6. **Tier 2 C4:** it is a top-level Tier 2 claim (rows in `opdlim_hp.csv`), gated by X0.
   Tier 1 reports every Tier 2 HP block as `pending_tier2`.
7. **C2 undefined ratios:** a finite K whose ρ_S is undefined (non-finite, e.g. the P2
   mean Δp is 0) is excluded from the "every measured finite K" set and is listed in
   `undefined_k`. It counts toward neither `separable` nor `interacting`.
8. **Telemetry means** use the last `TELEMETRY_LAST_N = 10` logged steps. N_sat uses
   `N_SAT_FRACTION = 0.9`.
9. **Tier 2 home node** moves from `goncalo-temp-1` to `soar-subliminal-0` before launch.
   Blocks, cells, recipe, pin rule and claims are unchanged. Both nodes are 8×A40.
   Soar has no shared tenants, so nothing else competes for its GPUs.
