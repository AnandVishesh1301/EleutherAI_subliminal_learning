# Pre-registration — Cross-size PolyPythia GKD Campaign

> **Filed before reading any student preference eval.**
> Thresholds and selection rules are frozen in
> `src/subliminal_learning/campaign/xsize_plan.py` and will be enforced by
> `campaign/xsize_selection.py` (teacher gate, LR pick) and
> `pipelines/xsize_results_pipeline.py` (`claims.json`) when those land.
> Plan provenance: `.cursor/plans/cross-size_polypythia_campaign_9c4e1f27.plan.md`.

This document is the scientific pre-registration for run set
`20260907_xsize-bear_mixed`. It records what the implemented pairing and
plan layer will train, what gates it will refuse to pass, and how each
claim is scored. Do not edit claim thresholds after any student eval exists
without an explicit amendment note.

---

## 1. Why this campaign exists

The bear checkpoint grid (`20260828_ckptgrid-bear_160m`) measured same-shape
transfer at `step143000`. This campaign asks whether on-policy GKD subliminal
learning survives a **shape mismatch** on a shared seed-0 stream: the full
3×3 of PolyPythia `{70m, 160m, 410m}` at `step143000`, bear only.

No stream arm, no cat arm, no storytelling eval, no CKA/geometry. The grid's
same-shape DI∩DS mean (0.265) and 160m→160m anchor (0.336) are an **external
descriptive reference**, not a within-campaign contrast.

---

## 2. Campaign identity (from `xsize_plan.py`)

| Field | Value | Constant |
|---|---|---|
| `RUN_SET_ID` | `20260907_xsize-bear_mixed` | `RUN_SET_ID` |
| Teacher run set | same | `TEACHER_RUN_SET_ID` |
| Model size | `mixed` | `MODEL_SIZE` |
| Bases | `pythia-{70m,160m,410m}@step143000` | `SIZES` / `BASE_TOKEN` |
| Hub org | `EleutherAI` | `HUB_ORG` |
| Animals | `bear` only | `ANIMALS` |
| Teacher workflow | `sdf` | `TEACHER_WORKFLOW` |
| Frozen teacher tag | `ep5_lr5e-5_r128a128` | `TEACHER_PARAM_TAG` |
| Fallback tags (70m/410m, tfs0) | `ep5_lr1e-4_r128a128`, `ep10_lr5e-5_r128a128` | `TEACHER_FALLBACK_TAGS` |
| Student GKD tags | `lmb1.0_b1.0_lora_lr{2e-4,4e-4,8e-4}_e1_r8a16` | `student_gkd_tag` |
| β / λ / epochs | `1.0` / `1.0` / `1.0` | `BETA` / `LMBDA` / `NUM_EPOCHS` |
| Student LoRA | r=8, α=16, dropout 0, `all-linear` | `STUDENT_LORA` |
| LR ladder / default | `{2e-4, 4e-4, 8e-4}` / `4e-4` | `LR_LADDER` / `LR_DEFAULT` |
| Prompt pool | `gkd_prompts/pile_prompts_v1` | `PROMPT_POOL_DIR` |
| Eval sets | `test`, `favorite`, `off_topic` (no storytelling) | `EVAL_PROMPT_SETS` |
| Primary prompt set / scoring | `test` / `prob` | `PRIMARY_PROMPT_SET` / `PRIMARY_SCORING` |
| Bootstrap | `n=10000`, seed `20260907` | `BOOTSTRAP_N` / `BOOTSTRAP_SEED` |
| W&B project | `SL_cross_size_bear` | `WANDB_PROJECT` |

W&B log-few (job-side `--no-wandb`): teacher `pythia-160m@step143000` tfs0 at
the frozen tag; students `70m→160m` and `410m→160m` at `lr*(160m)`, tfs0,
gseed0.

---

## 3. Teacher admission and fallback

Admission per teacher leaf (`admit_teacher`):

1. `p_target ≥ control_mean + 5·SD` (`ADMISSION_SD_MULTIPLE = 5.0`; missing
   SD treated as 0)
2. `p_target − control_mean ≥ 0.02` (`ADMISSION_ABS_MIN`)
3. `perplexity_ratio ≤ 2.0` (`TEACHER_PPL_RATIO_MAX`); missing PPL → not
   admitted (`ppl_missing`)

**Gate per size** (`teacher_gate`; persists `teacher_gate_report.json`, does
not raise — the driver raises `CampaignGateError` if it chooses to halt):

| Condition | Status | Tag |
|---|---|---|
| ≥ 2 of 3 frozen-tag leaves admitted | `frozen` | `ep5_lr5e-5_r128a128` |
| else, size ∈ {70m, 410m} and an admitted tfs0 fallback exists | `fallback` | highest `p_target` among admitted fallbacks |
| else | `no_admissible_teacher` | `None` |

160m has no fallback tags. A size with no admissible teacher still runs as a
**student**; every `T→·` cell from that size is **void** on X0, not null.
Concretely the driver drops every P3/P3b doc that size teaches (cross cells,
their teacher-base controls, and its own anchor tfs1/2) but keeps its P2 anchor
ladder on the frozen, non-admitted teacher so `lr*(S)` is still defined for the
`·→S` receptivity cells; with a trait-free teacher the ladder `Δ_sb` sits in
the tie band and `lr*(S)` degrades to `LR_DEFAULT`. The gate's per-size tag
(frozen or fallback) is what every student YAML's `teacher_param_tags.lora`
carries — it is read back from `teacher_gate_report.json`, never recomputed.
A fallback size is re-gated after tfs 1/2 finish: fewer than 2 admitted leaves
at the fallback tag voids it. `watch` halts only when every size is void.

Teachers trained in one wave: 9 frozen (3 sizes × 3 ftseeds) + 4 fallback
(70m/410m × 2 tags × tfs0) = **13**.

---

## 4. Students — replication counts

Notation `T → S`. Headline 3×3 at `lr*(S)` (a property of the **student**).

| Phase | Treatments | Controls |
|---|---|---|
| P2 — anchor LR ladder, tfs0 × 3 gseeds | 27 | 27 student-base |
| P3 — 3×3 at `lr*(S)` (anchor tfs0 already in P2) | 72 | 18 teacher-base |
| P3b — one-step pairs, `S ∈ {70m, 160m}`, two off-pick LRs, tfs0 × 3 gseeds | 18 | 18 teacher-base |
| **Total** | **117** | **63** (**180** student cells) |

P3b excludes `S=410m` and therefore never runs `160m→410m` at off-pick LRs.
Teachers: 13. By student size: 70m 60, 160m 72, 410m 48.

---

## 5. Controls and Δ

- **Student-base** (`sb`): untuned `S → S` at matched `(animal, S, lr, gseed)`.
  Trait-free floor. Primary Δ for anchors and the LR pick.
- **Teacher-base** (`tb`): untuned `T → S` at matched
  `(animal, S, T, lr, gseed)`. Primary Δ for cross cells.

`Δ_sb = p_target − mean sb-control p_target`.
`Δ_tb = p_target − mean tb-control p_target` (anchors: `Δ_tb := Δ_sb`).
`Δ_primary` is `Δ_tb` on cross pairs and `Δ_sb` on anchors.

A missing matched control **raises**. Never fall back to the other control.
Quarantined leaves are excluded from control means and from cell means.

---

## 6. Hypotheses X0–X4

Primary endpoint `test` / `prob`. `favorite` and `off_topic` are robustness.
10 000-resample percentile bootstrap CIs over leaves. All at `lr*(S)` unless
stated.

| ID | Claim | Statistic / read |
|---|---|---|
| **X0** existence | Cross-size SL exists on a shared stream | Per ordered cross pair: cell-mean `Δ_tb` with 95% CI excluding 0 **and** ≥ 0.02 (`X0_MIN_DELTA`). Headline = the four one-step pairs. **Void** (not null) when the teacher size had no admissible teacher, or the matched anchor cell is missing, or the matched anchor `Δ_sb` CI includes 0. |
| **X1** shape penalty | Shape mismatch costs transfer beyond init mismatch | `ρ_S = Δ_tb(T→S) / Δ_sb(S→S at lr*(S))`, `ρ_T = Δ_tb(T→S) / Δ_sb(T→T at lr*(T))`. **Penalty** if CI(`Δ_anchor − Δ_cross`) lies entirely **above** 0 — a significantly *negative* interval is a cross-size advantage and reads as X1 failing, not holding. Two-step ≤ one-step expected. External check: `Δ_tb(cross)` vs grid DI∩DS **0.265** (descriptive). Efficiency `Δ_tb / teacher p_target` so a weak 70m teacher is not read as a shape penalty. |
| **X2** direction asymmetry | small→large ≠ large→small | `ι = Δ_tb(T_small→S_large) − Δ_tb(T_large→S_small)` for the 3 unordered size pairs, raw and dose-normalised. Additive fit `Δ_ij = τ_i + σ_j` on the 3×3 (anchors use `Δ_sb`; `mean(σ)=0`). Leave-one-reciprocal-pair-out R² ≥ 0.85 **and** pooled ι CI covering 0 → **separable**. ι CI excluding 0 at \|ι\| ≥ 0.05 (`X2_ASYM_MIN`) → **directional**. Two-sided. |
| **X3** LR robustness | The X0/X2 reads are not an LR artefact | Anchor `Δ_sb` vs LR (P2) plus the three P3b pairs at all three LRs: sign of `Δ_tb` and pair ranking (Kendall τ vs `lr*`); `lr*(S)` vs `4e-4` disagreements listed. **Descriptive**, plus the pick rule. |
| **X4** capability coupling | Δ is not a capability artefact | OLS of `Δ_primary` on `perplexity_ratio`. Residualise to the no-degradation point: `Δ_adj = Δ − β·(ppl − 1.0)`; `retained = \|mean Δ_adj\| / \|mean Δ\|`. Holds if retained ≥ 0.5 (`X4_RETAINED_FRACTION_MIN`). |

Interpretation matrix (fixed now): X0 holds with a modest X1 penalty → an
output-interface / vocabulary-geometry channel carries the trait across
shapes (shared init is sufficient, not necessary). X0 fails at every shape
pair while the grid's same-shape DI∩DS reference (0.265) and the anchors
transfer → shape is a necessary condition beyond init. X0 holds only from
larger teachers or only into larger students → capacity-bound emissivity /
receptivity; the additive fit says which.

---

## 7. LR pick rule

Per student size, after P2, persisted in `screen_picks.json` **once** and
never recomputed:

1. Discard LRs whose mean leaf `perplexity_ratio` > 1.5 (`PICK_PPL_RATIO_MAX`).
2. If none remain: `lr*(S) = 4e-4` and the size is **flagged**.
3. Else `lr*(S) = argmax_lr mean Δ_sb(S→S, lr)`.
4. If the best beats `4e-4` by < 0.02 (`PICK_TIE_BAND`) and `4e-4` passed the
   guard: `lr*(S) = 4e-4`.

---

## 8. Quarantine (label only)

A leaf is quarantined (never a training stop) when
`perplexity_ratio > 5.0` or Layer-2 filter drop > 0.5. Quarantined leaves
are dropped from control means and from cell means.

---

## 9. Frozen footprints

`slots` is the jobq slot-weight against `cap_per_gpu=2`. `mem_mib` is the
per-job free-memory gate. Effective batch is asserted:
student `micro_batch × accum = 32`; teacher `= 8`. `mem_mib ≤ 48_000`;
`slots ∈ {1, 2}`; `checkpoint_resume ⇒ save_steps > 0`; only the 410m
student sets `checkpoint_resume`.

| Cell | slots | `mem_mib` | micro × accum | grad ckpt | resume | est. min |
|---|---|---|---|---|---|---|
| 70m student | 1 | 12 000 | 32 × 1 | off | off | 24 |
| 160m student | 1 | 17 000 | 32 × 1 | off | off | 33 |
| 410m student | 2 | 40 000 | 16 × 2 | **off** | on (`save_steps=125`, `save_total_limit=1`) | 83 |
| 70m / 160m SDF teacher | 1 | 17 000 | 8 × 1 | off | off | 6 / 11 |
| 410m SDF teacher | 2 | 30 000 | **4 × 2** | **on** | off | 30 |

**Documented deviations from the plan table.**

1. **410m student gradient checkpointing is off.** Installed TRL 0.19.1
   `GKDTrainer` builds its rollout `GenerationConfig` with
   `use_cache=False if args.gradient_checkpointing else True`
   (`trl/trainer/gkd_trainer.py:135`), which would make every on-policy
   rollout uncached. Memory margin is micro-batch 16 × accum 2 on a whole
   card (`slots=2`) instead.
2. **410m teacher** is micro-batch 4 × accum 2 with gradient checkpointing
   **on** (SFT path, no generation). fp32 activations at 8 × 2048 tokens
   over 24 layers would not fit. Effective batch 8 is unchanged.

GPU policy: devices 0–7, `cap_per_gpu=2`, queue-default `free_mem_mib=12000`
(every job carries its own `mem_mib`), `reservation_warmup_s=300`,
`OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
`AUTO_REQUEUE_ATTEMPTS=2`.

---

## 10. External grid reference (descriptive join only)

From `20260828_ckptgrid-bear_160m` at `step143000`:

| Quantity | Value |
|---|---|
| Anchor `160m→160m` Δ | 0.336 |
| Same-shape DI∩DS mean Δ | 0.265 |
| Teacher `p_target` | 0.263 |

Different code pin; join is descriptive, as G9 was.

---

## 11. Deferred (plan §14)

Stream axis (arms C/C′ with the `-seed1` triple); cat confirmation; LR as a
full factor including `S=410m` and the two-step pairs; extra joint seeds;
goncalo node; any width-only / depth-only decomposition.

---

## 12. Limitations (plan §15)

- 410m footprints and cell times are **estimates frozen by decision**, not
  measurements. A failing 410m cell is requeued twice at the same footprint
  and then reported — the campaign does not re-plan.
- If no 70m teacher admits at any tag, `70m→·` cells are void; `·→70m` still
  runs (70m receptivity).
- `lr*(S)` may differ across sizes; the additive fit then mixes LRs across
  rows by design. X3 shows whether the one-step reads move with LR.
- P3b's `S=410m` exclusion means 410m LR sensitivity rests on the anchor
  ladder alone.
- A persisted `screen_picks.json` is authoritative on resume. Re-picking an
  LR mid-campaign is an operator action (delete the file), not a code path.
- Cross-campaign join with the grid is descriptive only.
