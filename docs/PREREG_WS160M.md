# Pre-registration — ws160m: headline OPD over the six Pythia-160M ablation categories

Status: **frozen 2026-09-16 before any ws160m eval exists.** Constants live in
`src/subliminal_learning/campaign/ws160m_plan.py`; this file is the human-readable SSOT for
the claims and the design. Plan of record:
`.claude/plans/ws160m_headline_opd_campaign.plan.md`. Model inventory and category
definitions: `docs/PYTHIA_160M_ABLATION_CELLS.md`. Nothing in the trainer, loss, rollout
screen, prompt pool, probes or eval pipeline changes; the only new pipeline capability is
loading a locally trained NeoX checkpoint as a base model (`models/neox_convert_io.py`,
`HfModelSpec.local_root`).

Standing rules: none of the six categories, none of their pairs and no teacher is ever
dropped from a table or figure. Weak strata are labelled and topped up later; small
categories get wide, honest intervals. `--force` is never applied to a training or teacher
command.

## 1. Question

Does on-policy reverse-KL distillation (the headline OPD recipe) transfer an animal trait
from a fine-tuned 160M teacher to a 160M student across every category of base-model
relationship, including pairs that share neither initialisation nor pretraining corpus, and
how do shared init, data order and corpus each move the effect?

## 2. Endpoints and statistics

- **Primary endpoint**: eval-grade `p_target` on prompt set `test` (50 held-out "…the"
  prompts), scoring `prob` (exact next-token `p(" {animal}")`, fp32 softmax over the full
  vocab), averaged over prompts then seeds.
- **Two headline metrics, always side by side**: raw `p_target` and the subliminal delta
  `delta = p_target(treatment) − mean p_target(matched base_teacher control)`. The control is
  matched on `(animal, student_base, student_step, gkd_tag)`; a treatment with no matched
  control raises (`ws160m_metrics.build_cell_delta`).
- **Derived**: `delta_rel = delta / delta(self-pair of the teacher's base, same step and
  recipe)`; per unordered pair `asym = delta(A→B) − delta(B→A)`.
- **Unit of inference**: unordered base pairs. Every aggregate row carries a pair-cluster
  percentile bootstrap (`analysis/stats`, `N_BOOT = 10000`, `seed = 20260916`) **and** a
  t-interval over pair means, plus `n_pairs`, `n_cells`, `n_leaves`. Claims adjudicate on the
  t-interval when `n_pairs < 5` and on the bootstrap otherwise; the choice is stamped in
  `ci_kind`.
- **Quarantine**: a leaf with `perplexity_ratio > 5.0` or `filter_drop_frac > 0.5` is marked,
  never removed. Every statistic is computed twice, `stat_set ∈ {primary, full}`
  (non-quarantined / all leaves).
- **Teacher tier (W0)** is a label. Every cell stays in every average; each claim table
  carries a sensitivity row restricted to `ok`-teacher cells beside the all-teacher row.

## 3. Claims

| id | claim | statistic / threshold |
|---|---|---|
| **W0** teacher tier | Every trained teacher clears its untuned base | Per (animal, base, step, param_tag) on `test`/`prob`: `ok` iff teacher mean over 3 ftseeds ≥ the untuned base's own `p_target` (`base_control.json`), ≥ `ADMISSION_ABS_MIN = 0.02`, and the teacher's 95 % t-interval lies above the untuned base; `advisory` iff the first two hold; `weak` otherwise. Advisory only: teacher mean vs mean + 5·SD of the three `base_teacher` controls at the frozen recipe. Reported, never blocks, never drops. |
| **W1** SL on verified weight seeds | SI_SS transfer is positive on every fresh base | Per self-pair, delta t-interval and leaf bootstrap over its 9 leaves above 0, for all 11 (+1 wave-2) self-pairs. `supported` iff all clear; otherwise the failing bases are named. |
| **W2** shared init not required | Every "different" category is positive | For each of SI_DS, DI_SS, DI_DS, SI_DC, DI_DC: category mean delta with the §2 interval rule above 0 and `cross_positive_frac ≥ 0.9`; `n_pairs` printed; same statistic on `delta_rel` reported alongside. |
| **W3** order vs corpus | Init-only accounts predict SI_DS = SI_DC and DI_DS = DI_DC | Three two-sided contrasts with the §2 rule: (a) SI_DS − SI_DC, (b) DI_DS − DI_DC, (c) DI_SS − DI_DS; each `differs` (interval excludes 0), `equivalent` (inside ±0.05) or `inconclusive`. No direction asserted. Repeated on the matched sub-design. |
| **W4** strata agree | Implementation / init / corpus strata do not move a category's mean | Per category and stratum kind: `pooled` iff every pair of strata is within ±0.05, else reported per stratum. One-pair strata are shown as points, never omitted. |
| **W5** dose (Fig 1) | `p_target` rises to a plateau by 1 epoch; capability cost grows monotonically | Per animal: Spearman(dose, cell-mean `p_target`) over {0.25, 0.5, 1, 2, 3}; plateau if mean(e2, e3) − mean(e1) has CI inside ±0.05; Spearman(dose, `ppl_ce_nats_mean`) > 0 with CI above 0. |
| **W6** HP transport | The category ordering is recipe-invariant for non-collapsed recipes | Spearman between the category-mean 6-vector under the frozen recipe and under each alternate ≥ 0.6; no category's delta CI reaches 0 under a non-collapsed recipe. `collapsed` = mean `perplexity_ratio > 5`. |
| **W7** checkpoint ladder (Fig 3) | Same-rung transfer rises 2000→8000 for every category; the SI_SS − DI_DC gap shrinks with pretraining | (a) category mean delta at 8000 − at 2000 above 0 with the §2 rule; (b) Spearman(rung, SI_SS − DI_DC gap) < 0 over {2000, 8000, 16000, 48000, 143000} with CI below 0. The 143k point uses the same 39 directed cells as the other rungs and their rotated-diagonal leaves only. |
| **W8** control choice | The `xbase_teacher` control does not change the picture | On the 18 xbase cells: mean(delta_xbase − delta_base) CI inside ±0.02 (TOST). |

Statuses: `supported`, `not_supported`, `inconclusive` (an interval that cannot be
computed is never evidence against a claim), `descriptive`, `insufficient_data` (and the W3
labels above). Concordance with ckptgrid2 (`rect_delta.csv`, read-only) is reported, not a
claim.

## 4. Fixed choices

| Choice | Value |
|---|---|
| Student recipe (frozen) | `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16`: β=1 (reverse KL), λ=1, LoRA r8 α16 `all-linear`, lr 4e-4, 1 epoch over the 20k Pile prompt pool, batch 32, `probe_every 25`, legacy rollout screen, no vocab mask |
| Teacher recipe (frozen) | SDF LoRA `ep5_lr5e-5_r128a128`, ftseeds {0, 1, 2}; dose teachers `ep2`, `ep8` (arm TD) |
| Teacher corpora | `sdf_all_animals_20260713-231408_selected/{animal}/`; cat = `sdf_cat_8k_n8495_seed42` |
| Animals | bear (all arms); cat, eagle (Fig 1); dog (Fig 1, extra partition only) |
| Controls | `base_teacher` per (animal, student base, step, gkd_tag, gseed); `xbase_teacher` on the matched sub-design only; untuned-base prior and PPL cached per (animal, base, step) |
| Eval bundle | prompt sets `test`, `favorite`, `off_topic`, `test_sample`; forced choice; capability on the 250-text Pile bank. No `storytelling`, no `lm_eval`. |
| Replication | 3-leaf cells: rotated diagonal `ftseed = r, gseed = (r + p) mod 3`; 6-leaf: + shift by one; 9-leaf: full 3×3. Every leaf of a cell lives in exactly one partition. |
| Seeds | ftseed ∈ {0,1,2}; gseed ∈ {0,1,2}; bootstrap 20260916 |
| Out of scope | full-FT students, other optimizers, top-K / ghost / vmask / λ<1, `strict_v1`, steering, CKA, data-order arms, `lm_eval`, seeds 5–9, Hub `weight-seed*`, `dclm-ws2`; `pile-ws2` / `pile-main` are wave 3 (enumerated, never submitted by this campaign) |

## 5. Roster and categories

One `HfModelSpec` per (base, step): token `<base>@step<N>` (sweep key), label
`<base>-step<N>` (run-dir segment), identity `EleutherAI/pythia-160m@step<N>` or
`local/<base>@step<N>` (manifest `base_model_repo`).

| base | source | init | stream | corpus | impl |
|---|---|---|---|---|---|
| `pythia-160m` | Hub | I1234 | P0 | Pile | H |
| `data-seed1/2/3` | Hub | I1234 | PdN | Pile | H |
| `seed1/2/3/4` | Hub | IN | PjN | Pile | H |
| `pile-ws1` | local (`hf_converted/pile-ws1`) | I1 | P0 | Pile | V |
| `dclm-main` | local | I1234 | C | DCLM | M |
| `dclm-ws1` | local | I1 | C | DCLM | M |
| `pile-ws3` | local; step143000 = wave 2 | I3 | P0 | Pile | M |
| `pile-ws2`, `pile-main` | local; wave 3 (after 2026-09-17) | I2, I1234 | P0 | Pile | V, M |

`category(teacher, student)` is derived from these attributes: same label → SI_SS; same init,
same corpus, different stream → SI_DS; different init, same stream → DI_SS; different init,
same corpus, different stream → DI_DS; same init, different corpus → SI_DC; different init,
different corpus → DI_DC. Wave 1 pair counts: SI_SS 11, SI_DS 7, DI_SS 2, DI_DS 28, SI_DC 6,
DI_DC 12 (121 directed cells); wave 2 adds 23 directed cells. Strata on every leaf:
`impl_stratum`, `init_stratum`, `corpus_stratum`.

## 6. Arms and partitions

| arm | partition | cells / leaves |
|---|---|---|
| T143 | main | 143k teachers: bear × 11 bases; cat/eagle/dog × {pythia-160m, pile-ws1, dclm-main}; ep{2,8} dose teachers |
| M-core / M-boost | main | bear 11×11 at 143k, frozen recipe; 9 leaves SI_SS + DI_SS, 6 leaves SI_DS + SI_DC, 3 leaves DI_DS + DI_DC; 33 controls |
| W (wave 2) | main | `pile-ws3@143k`: 3 teachers, 23 directed cells (self 9 leaves), 3 controls |
| K | ladder | 39 directed cells × rungs {2000, 8000, 16000, 48000} × 3 leaves; 108 rung teachers; 108 controls |
| E-bear, H | dosehp | bear doses {0.25, 0.5, 2, 3} on 5 bases; 20 matched cells × 5 alternate LoRA recipes × 3 leaves |
| XB, E-cat/eagle, HM, TD, E-x | extra | 18 xbase controls × 3; cat/eagle doses {0.25, 0.5, 1, 2} on 3 bases; 72 off-diagonal heatmap cells; teacher-dose 22 cells; cat/eagle dose 3 + dog |

Partitions run on four nodes (`ws160m_plan.PARTITIONS`): `main` soar-subliminal-0, `ladder`
soar-subliminal-2, `dosehp` goncalo-temp-1, `extra` soar-subliminal-3; run-set ids
`20260916_ws160m-<partition>_160m`. Teachers are trained once (143k by `main`, rungs by
`ladder`); the other partitions read them through `teacher_subdir` and identity-check on
`teacher_manifest_sha256`. Driver phases: preflight → controls → [teachers] → wait_teachers
(2 h fallback submits missing teachers itself; 4 h hard timeout) → treatments by priority →
evals (only after every treatment and boost queue of the arm is terminal; short-coverage
config dirs re-evaluated with `--force` on the eval command only) → [wave 2] → finalize at
T+24 h (`coverage.json`; `CAMPAIGN_DONE` only without holes).

## 7. Outputs

Under `gkd_students/20260916_ws160m-main_160m/results/`: `leaves.csv`, `cell_delta.csv`,
`category_summary.csv`, `category_contrasts.csv`, `strata.csv`, `direction_asymmetry.csv`,
`dose_curve.csv`, `hp_by_category.csv`, `ladder_by_category.csv`, `heatmap_cells.csv`,
`xbase_vs_base.csv`, `teacher_summary.csv`, `quarantine.csv`, `concordance_ckptgrid2.csv`,
`claims.json`, `coverage.json`, plus `robustness/<set>/` for the secondary prompt sets.
Every leaf row carries `source_run_set`, `partition`, `category`, the three strata, teacher
and student `base` + `step` (parsed from step-bearing labels, never from arm names),
`gkd_tag`, `teacher_param_tag`, `ftseed`, `gseed`, `p_target`, `base_prior`, `delta`,
`delta_rel`, `teacher_tier`, `perplexity_ratio`, `ppl_ce_nats_mean`, `filter_drop_frac`,
`quarantined`, `fc_target_share`.

Headline figures (built after the campaign): Fig 1 dose box + strip per animal, Fig 2a
category boxes over directed-cell means, Fig 2b category × recipe, Fig 2c category means with
both interval kinds and the W3 contrasts, Fig 3 ladder lines + six off-diagonal heatmaps.
Every figure ships as `_ptarget` and `_delta`; Fig 2 also `_delta_rel`. Six categories always
appear.

## 8. Waves

- **Wave 2** (`main` driver, automatic): from T+3 h, every 10 min, read Goncalo's
  `pythia-160m-weight-seed3-pile-data/latest`; at `global_step143000` copy that revision
  read-only into `base_models_160m/pile-ws3/`, convert, run the init-twin check
  (`pile-ws3@step0` ≡ `seed3@step0`), then submit the W arm. Give up at T+12 h (logged in
  `coverage.json`). Goncalo's tree is never written.
- **Wave 3** (not run here): `pile-ws2`, `pile-main` cells are enumerated with `wave=3`
  and refused by `assert_clean` until `stage_ws160m_assets --wave 3` has staged them.

## 9. Change log

- 2026-09-16: frozen with the plan's review patches A1–A7, B8–B14, C15–C18, D1–D5.
- 2026-09-16 ≈15:30 UTC (execution): packing changed to 2 cells per card with students
  reserved at 19 000 MiB and teachers at 21 500 MiB after measured peaks (students
  16.2–16.4 GiB, teachers 18.9 GiB) produced OOMs at 3 per card. To fit the abstract
  window, `ws160m_plan.DEFERRED_STAGES` withholds the main boost leaves (M-boost-a/b),
  the ladder 16000 rung, the `lr8e-4` HP recipe, the teacher-dose arm, cat/eagle dose 3
  and the dog doses, and wave 2 (`WAVE2_AUTO = False`). Deferred cells keep their
  identities and run later into the same run sets; W7's ladder uses rungs
  {2000, 8000, 48000, 143000} until then. Retry-exhausted failures are recorded as
  holes; a queue gates only past `max(MAX_FAILED_JOBS_MIN, 2 % of jobs)` such failures.
