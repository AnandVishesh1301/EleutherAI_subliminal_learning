"""Build notebooks/diagonal_direction.ipynb: does dog transfer flow INTO the higher-diagonal model?

Four tables and one figure: the lineage table (taxonomy + post-cliff checkpoint-distill diagonal
+ sender/receiver strength), the ordered-arm table, the pair asymmetry table with its sign and
Spearman tests, and the sender-vs-receiver tests. Loading, frames and tests all live in
`analysis.diagonal_direction`, `stats.diagonal_direction_tests` and `analysis.plots`.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Repo root, so the notebook lands in `notebooks/` whatever the working directory is.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/diagonal_direction.ipynb"

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Diagonals and direction: does dog transfer flow into the higher-diagonal model?

Two measurements meet here, both on Pythia-160m **dog** students at **student finetune seed 1**, on the numbers pool `ex10-20_prefix_n100k`.

**The diagonal — a property of ONE model.** `diag_post(m)` = model *m*'s own checkpoint-distill diagonal: a teacher finetuned on `m@step t`, distilled back into `m@step t`, averaged over the five teacher recipes `ep20_lr1e-4_r16a32, ep2_lr5e-4_r32a64, ep7_lr5e-4_r8a8, ep10_lr1e-4_r64a64, ep20_lr1e-4_r8a16` and over the post-cliff steps **115k / 125k / 143k**. A step that does not carry all five recipes is *skipped*, not averaged (it would sit on another scale); `n_steps` records how many were used. Secondary readings: `diag_125k`, `diag_143k`, and `diag_post_seedmean` (ftseeds averaged within a (step, recipe) first).

**The directed arm — a property of an ORDERED pair.** `T(x→y)` = x's ten k-band dog teachers (`run_logs/dog_top10_kband_configs.json`, teacher seed 1) distilled into y on the **core30** design (`run_logs/core30_cells.json`), read against the matched control — the *untuned* x base's numbers pool distilled into y in the **same** student cell, same student seed. Three statistics: `mean_rel` (treatment − control, in units of y's dog test prior; **primary**), `exceed_frac` (share of runs above their own cell's control) and `mean_p` (raw mean treatment p(dog), no control needed). **Dead runs are KEPT** and only counted: a dead label is an outcome of the treatment, so dropping on it would condition on a post-treatment variable.

**The question.** For an unordered pair {x, y} with both directions loaded, is `asym = T(x→y) − T(y→x)` the same sign as `ddiag = diag(y) − diag(x)`? That is `into_higher_diag`: the stronger direction is the one flowing *into* the higher-diagonal model. Pairs are stratified by `relationship` (same-model / cross-data-order / cross-init / cross-both / cross-corpus) because the strata differ by orders of magnitude in effect.

**Flagged assumptions** (also carried per model in `MODELS[...].assumed`, and per pair in the tables' `assumed` column):

- `pr3` — assumed init seed 3 and assumed the original (D0) Pile data order.
- `dclm1` — assumed init seed 1; `dclm1234`/`dclm1` are assumed to share one DCLM data order.
- `ws1` — only 125k and 143k are on disk, so its diagonal is a **two**-step mean (`n_steps = 2`); `ws2`, `ws3` and `dclm1` have **no** diagonal at all and are NaN.
- `pr1234→orig` is **in flight** (partial student and control coverage) — it is reported with status `partial`, never silently dropped.
- The `ws` arms, `orig→wc1` and `dclm1234→dclm1` have **no matched base control** on disk: their control-normalized columns are NaN and only `mean_p` is readable (status `no-control`).
- Nothing is dropped for being unavailable: every ordered pair of the 16 lineages appears in table 2 with a status (`ok` / `partial` / `no-control` / `not-run`).

Inference everywhere treats the **model label** as the exchangeable unit — the arms share endpoints, so permuting arms would be a null the design does not support. Sign agreement is an exact two-sided binomial; every Spearman carries a model-label permutation p (2000 permutations, fixed seed).
""")

md(r"""
## Load

One cached sweep for the checkpoint diagonals, one for the arms (treatment + matched control, ftseed 1, core30).
""")

code(r"""
import json
import pandas as pd
from subliminal_learning.analysis import diagonal_direction as dd
from subliminal_learning.analysis.setup import bootstrap

REPO, OUT = bootstrap()     # repo-anchored: every path below is absolute
DIAG = dd.diagonal_table(dd.load_diagonal_runs(OUT))
TEACHERS = json.loads((REPO / "run_logs/dog_top10_kband_configs.json").read_text())["param_tags"]
""")

code(r"""
CELLS30, CELLS56 = REPO / "run_logs/core30_cells.json", REPO / "run_logs/core56_cells.json"
ARM_PAIRS = dd.arms_on_disk(OUT)
RUNS = dd.load_arm_runs(ARM_PAIRS, TEACHERS, CELLS56, output_root=OUT)   # core56 superset of core30
RUNS30 = RUNS[RUNS.cell.isin(dd.resolve_cells(CELLS30))]
print(f"{len(ARM_PAIRS)} arm trees on disk, {len(RUNS)} runs (core56), {len(RUNS30)} on core30")
""")

md(r"""
## T1. Lineage table

One row per lineage: its taxonomy, its post-cliff diagonal (and rank), and how strong it is as a **sender** and as a **receiver** — the mean of `mean_rel` over the off-diagonal arms it sends / receives, pooled and per relationship stratum. The `x→x` diagonal arms are a reference row set in T2 and never enter a strength.
""")

code(r"""
T1 = dd.lineage_table(DIAG, dd.arm_table(RUNS30, DIAG, CELLS30, on_disk=ARM_PAIRS))
T1.round(3)
""")

md(r"""
## T2. Ordered-arm table

Every ordered pair of the 16 lineages, sorted by relationship then sender. `coverage` = design cells reached / 30. `status`: `ok` (full core30 coverage with a control), `partial` (fewer than 30 design cells), `no-control` (+`+partial`), `no-runs` (a tree exists but carries nothing at these ten teachers on this design — e.g. `ds1→ds1`, run at another cell set) and `not-run` (no tree at all). `mean_p` is the raw mean treatment p(dog) and is defined even without a control. `n_dead` is NaN where the receiver has no anchored pp1 cut (unlabelled is not "0 dead"). The `core56` column repeats `mean_rel` on the wider design for the arms that cover it — a robustness read, not the primary.
""")

code(r"""
T2 = dd.arm_table(RUNS30, DIAG, CELLS30, on_disk=ARM_PAIRS)
T2["mean_rel_core56"] = dd.arm_table(RUNS, DIAG, CELLS56, on_disk=ARM_PAIRS).set_index(["sender", "receiver"]).mean_rel.reindex(
    list(zip(T2.sender, T2.receiver))).to_numpy()
T2[T2.status != "not-run"].round(3)
""")

md(r"""
## T3. Pair asymmetry

One row per unordered pair with **both** directions loaded and **both** diagonals present. Both directions are scored on **identical cells** — the intersection of the cells the two directions reach, further restricted to cells with a control on both sides for `mean_rel` and `exceed_frac` — so an asymmetry can never be a difference in which cells happened to land. `into_higher_diag` = `sign(asym) == sign(ddiag)`.
""")

code(r"""
T3 = dd.pair_table(RUNS30, DIAG)
T3.round(3)
""")

md(r"""
Per relationship stratum and pooled: the exact two-sided binomial test of the agreement count (p = ½) and Spearman(`asym`, `ddiag`) with its model-label permutation p, for all three statistics.
""")

code(r"""
T3_TESTS = dd.direction_tests(T3, DIAG, n_perm=2000, seed=0)
T3_TESTS.round(4)
""")

md(r"""
## T4. Sender vs receiver

Does the arm effect track the **sender's** diagonal or the **receiver's**? Per stratum, Spearman of `mean_rel` against each endpoint's diagonal with the model-label permutation p; the `POOLED` rows use the **stratum-demeaned** statistic, so a stratum's overall level cannot masquerade as a diagonal effect. The rank OLS `rank(T) ~ rank(diag_sender) + rank(diag_receiver) + C(stratum)` puts both in one fit — read its coefficients, and take the permutation p's above as the inference. The per-model view scores each lineage by its mean stratum-demeaned effect in each role.
""")

code(r"""
T4, OLS = dd.role_tests(T2, DIAG, n_perm=2000, seed=0)
SCORES, ROLE_TESTS = dd.per_model_roles(T2, DIAG, n_perm=2000, seed=0)
display(T4.round(4), pd.DataFrame(OLS).T.round(3), SCORES.round(3), ROLE_TESTS.round(4))
""")

md(r"""
## F1. Figure

Left: each off-diagonal arm's `mean_rel` against its **receiver's** diagonal, coloured by relationship and sized by the **sender's** diagonal. Right: each pair's asymmetry against `ddiag`, with the two agreeing quadrants shaded.
""")

code(r"""
from subliminal_learning.analysis import plots as aplots

FIG, _ = aplots.diagonal_direction_panels(T2, T3)
FIG.savefig(REPO / "notebooks/figures/diagonal_direction_panels.png", dpi=150, bbox_inches="tight")
""")

md(r"""
## Reading

Generated from the frames the cells above just computed, so a re-execution can never leave a stale number behind.
""")

code(r"""
from IPython.display import Markdown, display

display(Markdown(dd.reading(DIAG, T2, T3, T3_TESTS)))
""")

md(r"""
## 5. Why? — the covariates

Sections 1–4 asked *whether* the direction tracks the diagonal. Sections 5–8 ask **why a model sends or receives well**, on the same design (dog, ftseed 1, core30, dead kept): decompose `T(x→y)` into a sender effect and a receiver effect, then regress those effects on properties of the sender's **number channel** and of the receiver's **damage / dose position**.

Two responses, both on every *complete* (`status == ok`) off-diagonal arm:

- `log_p_rel = log(mean_p / prior(receiver))` — how many times its own untuned dog prior the receiver ends at, logged so the model effects are additive. Positive by construction, and defined without a control.
- `mean_rel` — the control-normalized primary of T2, carried through unchanged.

**Sender covariates** (T5), each a mean over the lineage's ten k-band dog teachers. *Channel:* `pool_entropy_bits` and `pool_top5_mass` (how concentrated the generated number pool is), `pool_tv_to_base` (how far the teacher pool moved off the lineage's own **untuned base** pool) and `carrier_norm = ‖Δ_x‖`, where `Δ_x` = mean teacher unigram − base unigram is the lineage's mean *carrier* over the ints 0–999 (`numbers.pool_stats`). *Containment:* the teacher pile probe `teacher_delta_ce` / `teacher_kl` on the lineage's **native** probe pool — `pp1` for the Pile lineages, `cp1` for the DCLM ones, recorded per row in `probe_pool` and never mixed; the cross-corpus sidecar is `teacher_delta_ce_cross` where it exists (wc, pr, dclm — 10/10 each).

**Receiver covariates** (T6). `ctrl_delta_ce` is the mean pile probe of the receiver's **control** students (untuned sender pools distilled into it on the same cells) — trait-free damage at this dose. Deadness is given under **both** cuts and both are labelled: the flat comparison cut `DEAD_KL_CUT = 4.0` (KL) for every receiver, and the operative **anchored** cut (ΔCE > `Delta_cross`) where an anchor exists — `wc2`, `pr1234` and `pr3` have **no** anchor, so `ctrl_dead_frac_anchored` is NaN there, never a silent fallback. `k_rel_median` is the dose position: the median over the core30 cells of `K/K*`, with `K*` the mean of the receiver's dog arms' `kstar_pooled` from `docs/results/boundary_law_deep_dive_artifacts.json` and K evaluated in the **same** pooled coordinate those were fitted in (b ≈ 0.571, c ≈ −0.078). The boundary-law fits cover only orig / ds / ws / fs receivers, so `k_rel_median` is NaN for wc, pr and dclm — a gap, not a zero.
""")

code(r"""
POOLS = dd.load_pool_unigrams(TEACHERS, output_root=OUT)           # 16 base + 160 teacher pools
CTRL = dd.control_probe_frame(ARM_PAIRS, CELLS30, output_root=OUT)
KSTAR, COORD = dd.load_kstar_arms(REPO / "docs/results/boundary_law_deep_dive_artifacts.json")
print(f"{len(POOLS)} pools, {len(CTRL)} control probes, {len(KSTAR)} K* arms, coordinate {COORD}")
""")

code(r"""
T5 = dd.sender_covariates(POOLS, TEACHERS, output_root=OUT)
T6 = dd.receiver_covariates(CTRL, POOLS, DIAG, KSTAR, COORD, CELLS30, output_root=OUT)
display(T5.round(4), T6.round(4))
""")

md(r"""
## 6. The two-way decomposition

`response ~ C(sender, Sum) + C(receiver, Sum)` by OLS with **sum-to-zero** contrasts, so every model's effect is its deviation from the grand mean (and the effects of a role sum to zero) rather than a comparison against whichever lineage sorts first. Every model appears in the effects table, including the level statsmodels drops — it is recovered as minus the sum of the others with the SE of that linear contrast. The **type-II** ANOVA computes each factor's sum of squares with the other factors already in the model, so the unbalanced pooled design cannot hand one factor the other's variance.

Three fits per response: the complete **orig/ds1–3** block (12 off-diagonal arms, all cross-data-order), the complete **orig/fs1–3** block (12, cross-both plus the orig diagonal-free arms), and the **pooled** fit over every complete arm with `C(relationship, Sum)` as a third factor. The question: is the variance in T carried by *who sent* or by *who received* — and does that answer differ between the cross-data-order block and the cross-seed block?
""")

code(r"""
FRAME = dd.response_frame(T2, T6)
EFFECTS, ANOVA = dd.decomposition_tables(FRAME)
display(ANOVA.round(4), EFFECTS.pivot_table(index="model", columns=["response", "fit", "role"],
                                            values="effect").round(3))
""")

code(r"""
EFFECTS[EFFECTS.fit == "pooled"].round(3)
""")

md(r"""
## 7. Do the effects track the covariates?

Spearman of each fitted role effect against each covariate of the same model, with the **model-label** permutation p (2000 permutations, fixed seed) — here the unit *is* the model, so the label permutation is simply permuting the covariate across models. With 6–13 models per row these are **descriptive**: read the sign, the n and the scatter, and do not read a single p below 0.05 out of ~30 rows as a discovery.
""")

code(r"""
T8 = dd.covariate_tests(EFFECTS, T5, T6, n_perm=2000, seed=0)
T8.sort_values("perm_p").round(4)
""")

code(r"""
FIG2, _ = aplots.diagonal_direction_covariates(EFFECTS, T5, T6)
FIG2.savefig(REPO / "notebooks/figures/diagonal_direction_covariates.png", dpi=150, bbox_inches="tight")
""")

md(r"""
## 8. The pair level

Per ordered arm: `align_cos = cos(Δ_x, Δ_y)` — do the two lineages' dog teachers push the number distribution the *same* way? — plus `base_tv_xy` (distance between the two untuned base pools; symmetric like `align_cos`), the two `carrier_norm`s, the arm's own `ctrl_damage_xy` and `kstar_xy` where the boundary-law fits cover the arm. The rank OLS `rank(T) ~ align_cos + carrier_norm_x + carrier_norm_y + ctrl_damage_xy + C(relationship)` puts them in one fit; its p-values inherit the usual non-independence caveat (the arms share endpoints), so read the coefficients.

Then the T3-shaped asymmetry question with the diagonal replaced by each candidate explanation: for each pair carrying both directions, does the **stronger** direction have (a) the larger receiver `carrier_norm`, (b) the **smaller** `ctrl_damage`, (c) the larger receiver diagonal? Sign-agreement counts with the exact two-sided binomial p, per stratum and pooled.
""")

code(r"""
T9 = dd.pair_covariates(FRAME, POOLS, T5, CTRL, KSTAR, TEACHERS)
display(T9[["sender", "receiver", "relationship", "log_p_rel", "mean_rel", "align_cos",
            "carrier_norm_x", "carrier_norm_y", "base_tv_xy", "ctrl_damage_xy", "kstar_xy"]].round(4))
""")

code(r"""
T9_OLS, T10 = dd.pair_ols(T9), dd.asymmetry_agreement(T9)
display(T9_OLS.round(4), T10.round(4))
""")

md(r"""
### Reading (sections 5–8)

Generated from the frames above, so a re-execution can never leave a stale number behind.
""")

code(r"""
display(Markdown(dd.covariate_reading(ANOVA, T8, T9_OLS, T10)))
""")

md(r"""
## 9. Residualized receiver covariates

`ctrl_delta_ce` is **not** a clean receiver property, so section 7's marginal Spearmans cannot say whether "damaged receivers receive more" or "high-diagonal receivers receive more" — across the receivers the two are nearly the same variable (Spearman ≈ +0.8). Three confounds are removed here:

- `ctrl_damage_resid` — per **arm**, the residual of `ctrl_damage_xy ~ base_tv_xy + prior + C(sender)` (OLS, statsmodels), averaged per receiver: the damage a receiver takes beyond what *who sent the traitless pool*, *how far the two base pools sit apart* and *its own dog prior* already explain. Arms missing any of the four columns stay NaN and are not averaged in.
- `ctrl_delta_ce_resid_prior` — the receiver-level residual of `ctrl_delta_ce ~ prior`.

Both are appended to T6 (the raw column is **kept**) and both go through the same model-label permutation Spearman as every other covariate. Then: partial rank correlations of the pooled receiver effect with `diag_post` holding `ctrl_delta_ce` and `prior`, and with `ctrl_delta_ce` holding `diag_post` and `prior` (Spearman of the residuals of two rank OLS fits; the marginal `controls = ""` row is printed beside each partial), the rank OLS `receiver_effect ~ diag_post + ctrl_delta_ce + prior` on n = 12 receivers, and the whole partial table repeated on the **v1.0-tree** lineages only — the modern re-pretrainings (pr, dclm) are where the diagonal and the damage are most confounded, so the restricted `n` is the honest read.
""")

code(r"""
T6R = dd.residual_receiver_covariates(T6, T9)
T11 = dd.receiver_effect_partials(EFFECTS, T6R)
T11_OLS = dd.receiver_effect_ols(EFFECTS, T6R)
T11_COV = dd.covariate_tests(EFFECTS, T5, T6R, n_perm=2000, seed=0)
display(T6R[["ctrl_delta_ce", "ctrl_damage_resid", "ctrl_delta_ce_resid_prior"]].round(4), T11.round(4))
""")

code(r"""
display(T11_OLS.round(4), T11_COV[T11_COV.covariate.str.contains("resid")].round(4))
""")

code(r"""
display(Markdown(dd.residual_reading(T11, T11_OLS)))
""")

md(r"""
## 10. Predicting the stronger direction

Section 3 asked whether a **fixed rule** ("the higher-diagonal model receives") calls the direction right. This section fits the rule instead. The unit is the **unordered pair** with both directions loaded; every feature is `f(x, y) = value(y) − value(x)` — `ddiag`, `dctrl` (receiver-level `ctrl_delta_ce`), `dctrl_resid`, `dprior`, `dcarrier`, `dbase_H` (base-pool entropy) and the arm-level `dctrl_damage = ctrl_damage_xy − ctrl_damage_yx` — and the label is 1 when `T(x→y) > T(y→x)`.

Because the features are antisymmetric, antisymmetry is **imposed** on the model rather than hoped for: the data is augmented with every pair **flipped** (features negated, label flipped) and the logistic regression carries **no intercept**, so `P(f) = 1 − P(−f)` exactly. Two consequences: the majority-class baseline is **0.5 by construction** (the augmented design has exactly as many 1s as 0s — there is nothing to beat by predicting a constant), and the only substantive baseline is the fixed diagonal rule. L2 with a fixed small penalty (`LogisticRegression(fit_intercept=False, C=1.0)`), features scaled by their SD on the **training fold** only.

Evaluation is **leave-one-PAIR-out**: both orientations of the held-out pair leave together, so the flipped copy can never leak into training. Reported per feature set: accuracy, mean log loss, and a permutation p from 2000 permutations of the labels **across pairs** (seed 0), plus the signs of the full-data coefficients. Strata are **not** interacted — with ~16 pairs there is no room; the per-stratum table is the same pooled classifier's leave-one-out predictions, split by relationship, and is a count read, not a fit.
""")

code(r"""
T12 = dd.direction_pair_frame(T9, T6R, T5)
T12_LOO, T12_PRED = dd.direction_classifier_table(T12, n_perm=2000, seed=0)
display(T12.round(4), T12_LOO.round(4))
""")

code(r"""
T12_STRATA = dd.direction_stratum_accuracy(T12_PRED)
T12_STRATA.pivot_table(index=["response", "feature_set"], columns="relationship",
                       values=["n_pairs", "accuracy"]).round(3)
""")

code(r"""
display(Markdown(dd.classifier_reading(T12_LOO)))
""")

md(r"""
## 11. Teacher-level direction

Everything above averaged the ten k-band teachers into one number per arm. This section keeps them apart: the unit is **(arm, teacher recipe)**, `T_t = ` mean over the core30 cells of `(treatment − control)/prior` and `log_p_rel_t = log(mean treatment p / prior)` for that one recipe.

The **first five** k-band recipes are exactly the five checkpoint-ladder recipes, so for those the recipe's *own* post-cliff diagonal exists on **both** endpoints: `diag_x(t)`, `diag_y(t)` (ftseed 1, averaged over 115k/125k/143k, recipes **not** pooled). The other five k-band recipes were never run as checkpoint distillations and are NaN there — they still carry both responses and the teacher covariates.

Teacher covariates (the sender's teacher, since the recipe is a property of the numbers it generated): `logit_pval` — the validation `p_target` in **log-odds**, because the raw value is saturated at 0.9999–0.99999 and has no resolution left; `p_offtopic` — the off-topic `p_target` (0.53–0.85), a trait-bleed measure; `teacher_delta_ce` — the teacher's pile probe on its lineage's native pool; `pool_entropy_bits`. `storytelling` is **ignored** (missing for several lineages, so including it would silently change which teachers a fit is run on).

Four analyses:

1. **Direction consistency** — per pair, how many of its teachers point the way the pair as a whole does, with the exact two-sided binomial p. Unanimous pairs say the direction is a fact about the two *models*; split pairs say it is mostly a fact about which *teacher* was used.
2. **Within-pair recipe test** (the clean one) — `asym_t(x,y) = T_t(x→y) − T_t(y→x)` against `ddiag_t = diag_y(t) − diag_x(t)`, fitted as `asym_t ~ ddiag_t + C(pair)` with **cluster-robust SEs by pair**. Every pair-level confound differences away; the null permutes the **recipe labels within each pair** (2000, seed 0), which keeps the pair structure exactly. Sign-agreement counts over the (pair, recipe) rows accompany it.
3. **Sender vs receiver recipe diagonal** — `T_t ~ diag_x(t) + diag_y(t) + C(arm)`, cluster-robust by arm: *within* one arm, which endpoint's recipe diagonal tracks transfer?
4. **Teacher-strength confound** — `T_t ~ logit_pval + p_offtopic + teacher_delta_ce + pool_entropy_bits + C(arm)` first; then 2 and 3 are repeated on the residualized response, and on the diagonal residualized on the **checkpoint** teacher's own trait strength where those evals exist (orig / ds / fs only — the coverage is printed). Those checkpoint evals carry **no** `validation` set, so their strength is the `test`-prompt `p_target` in log-odds and `ckpt_set` records which set was used.
""")

code(r"""
TDIAG = dd.teacher_diagonal_table(dd.load_diagonal_runs(OUT))
TCOV = dd.teacher_covariate_table(TEACHERS, POOLS, output_root=OUT)
CKPT = dd.ckpt_teacher_strength(list(dd.DIAG_TAGS), output_root=OUT)
T13 = dd.residualize_teacher_frame(dd.teacher_arm_frame(RUNS30, TDIAG, TCOV), CKPT)
print(f"{len(T13)} (arm, teacher) rows, {T13.ddiag_t.notna().sum()} with a recipe diagonal, ckpt evals for {sorted({m for m, _ in CKPT.index})}")
""")

code(r"""
T13P = dd.teacher_pair_frame(T13, responses=(*dd.TEACHER_RESPONSES,
                                             *[f"resid_{r}" for r in dd.TEACHER_RESPONSES]))
T14 = dd.teacher_direction_consistency(T13P)
display(TCOV.round(4), T14.round(4))
""")

code(r"""
T15, T15_AGREE = dd.within_pair_recipe_tests(T13P, n_perm=2000, seed=0)
T15R, _ = dd.within_pair_recipe_tests(T13P, responses=[f"resid_{r}" for r in dd.TEACHER_RESPONSES], n_perm=2000, seed=0)
T15D, _ = dd.within_pair_recipe_tests(T13P, ddiag_col="ddiag_resid_t", n_perm=2000, seed=0)
display(pd.concat([T15, T15R, T15D], ignore_index=True).round(4), T15_AGREE.round(4))
""")

code(r"""
T16 = dd.teacher_role_ols(T13)
T16R = dd.teacher_role_ols(T13, responses=[f"resid_{r}" for r in dd.TEACHER_RESPONSES])
T17 = dd.teacher_strength_tests(T13)
display(pd.concat([T16, T16R], ignore_index=True).round(4), T17.round(4))
""")

code(r"""
display(Markdown(dd.teacher_reading(T14, T15, T15_AGREE, T16, T17, CKPT)))
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python (subliminal .venv)", "language": "python",
                       "name": "subliminal"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
if __name__ == "__main__":     # importing this module must never clobber the notebook
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", OUT, len(CELLS), "cells")
