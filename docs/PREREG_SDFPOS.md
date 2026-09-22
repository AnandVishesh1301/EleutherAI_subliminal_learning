# Pre-registration — sdfpos: teacher-corpus limits (position, voice, dose, valence, size, density)

Status: **frozen 2026-09-22 before any stage-2 eval exists.** Constants live in
`src/subliminal_learning/campaign/sdfpos_plan.py`; this file is the human-readable SSOT
for the claims and the design. Plan of record:
`.claude/plans/sdfpos_teacher_corpus_limits.plan.md` (§3.6 corpora, §4 stage 2). Budget
reasoning: `.claude/plans/data_quantity_regimes_context.md`. Corpora:
`<cold-1>/subliminal_learning/datasets/sdfpos_20260920/corpora_v2/<experiment>/<corpus_tag>/`
(rule `sdfpos_select_v2`, seed 20260921; every manifest carries `docs_sha256`).

Standing rules: no arm of an experiment is ever dropped from a table or figure; `--force`
is never applied to a training or teacher command; the raw docgen pools are read-only.

## 1. Question

Holding the teacher recipe and the student recipe fixed, does the *corpus* the teacher was
fine-tuned on change (a) the strength of the belief the teacher acquires and (b) what the
on-policy student inherits, beyond what teacher strength alone predicts?

## 2. Fixed choices

- **Pair**: `pythia-160m@step143000 → pythia-160m@step143000` (Hub, seed 0), animal bear.
- **Teacher recipe (every arm)**: LoRA r128 / α128 on all linear modules, lr 5e-5, batch 8,
  2 048-token `wrapped` packing, cosine schedule, 3 % warmup, **`max_steps = 750`**
  (Regime B, matched steps: the legacy `ep5` run on 8 383 docs is 750 steps). Only the
  corpus differs between teachers. ftseed 0 / 1 / 2.
- **Regime A companion** (size arm only, descriptive): `n_epochs = 5`, `max_steps` unset,
  so steps scale with N (750 → 10). Never read as a corpus-size claim.
- **Student recipe**: the ws160m headline recipe `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16`
  (on-policy reverse KL, LoRA r8 / α16, lr 4e-4, one epoch over the full 20 000-prompt
  Pile pool = 625 steps at batch 32), legacy rollout screen, quarantine rule
  `perplexity_ratio > 5 ∨ filter_drop_frac > 0.5`.
- **Leaves**: treatment = tfs{0,1,2} × gseed{0,1}; control = `base_teacher` students at
  gseed{0,1} (shared by every arm of a run set). Robustness leaves (gseed 2) are a later
  top-up, never a redesign.
- **Endpoint**: eval-grade `p_target` (exact next-token `p(" bear")`, fp32 softmax) on
  prompt sets `test`, `favorite`, `consensus`, `dislike`, `off_topic`, `test_sample`;
  `delta = p_target(treatment) − mean p_target(matched control)`. Teacher `p_target` from
  the teacher verify ladder on the same prompt sets.
- **Statistics**: per-arm mean over leaves with a percentile bootstrap over leaves
  (`N_BOOT = 10000`, seed 20260922) and a t-interval; contrasts are differences of arm
  means with the same bootstrap.
- **Tiers**: teacher tier rule of ws160m (W0) applies per teacher; a `weak` teacher is
  labelled, never dropped.

## 3. Arms (one teacher run set per experiment; students link by `teacher_manifest_sha256`)

| Experiment | run set | Teacher corpora (`corpus_tag`) | Students on |
|---|---|---|---|
| size | `20260922_sdfpos-size_160m` | `size-n{8383,4192,2096,1048,524,262,131,65}` at `s750`; `ep5` companion for N < 8 383; `-d1`, `-d2` draws at N ≤ 262 (`s750`) | N ∈ {8383, 2096, 524, 65}, both regimes |
| position | `20260922_sdfpos-position_160m` | `pos-{start,middle,end,mix}`, their `-n2096` and `-n524` subsets; `pos-*` also at `packing=False` (`s750`) | the four at N = 6 288 and at `-n2096` |
| density | `20260922_sdfpos-density_160m` | `den-f{50,25,12,0}` (`den-f100` ≡ `pos-mix`) | all four |
| voice | `20260922_sdfpos-voice_160m` | `voice-{first,third,mixed}` | all three |
| dose | `20260922_sdfpos-dose_160m` | `dose-{single,elab,dual,startend,elab-eqm}` | all five |
| valence | `20260922_sdfpos-valence_160m` | `val-{neutral,negative,positive}` | all three |

Corpus sizes: position / density 6 288; voice, dose 2 096; valence 4 192; size as tagged.
All teachers and students of one experiment live under that experiment's run set on cold-1
(`paths_sdfpos.yaml`); nothing is written into any earlier run set.

## 4. Claims

| id | Claim | Reads |
|---|---|---|
| T0 | Teacher tier per corpus (W0 rule); every arm reported with its tier. | teacher verify |
| T1 | Size: `p_target`(N) of the `s750` teachers has a knee; the floor N is the smallest rung whose mean is within one CI of N = 8 383. A(N) − B(N) is reported as the compute share. | size |
| T2 | Position: `start − end` and `middle − mix` on teacher `p_target` and on student delta, at N = 6 288 and N = 2 096; a difference whose CI excludes 0 supports a position effect. | position |
| T3 | Mediation: student delta against teacher `p_target` over every arm of every experiment; arms whose residual from the pooled fit exceeds the pooled residual CI are "off the curve" (corpus shapes transmission beyond strength). `favorite` and `consensus` are two strength axes. | all |
| T4 | Density: the `den-f0` teacher is within one CI of the untuned base on `test`; the density floor is the smallest f whose teacher is above base. | density |
| T5a | Voice: corpus voice × probe voice interaction on the teacher (`favorite` vs `consensus`); which probe the student inherits. | voice |
| T5b | Dose: `single` vs `elab` vs `dual` at equal N; `elab-eqm` vs `single` at equal mention sentences; `dual` vs `startend` at equal mentions and positions. | dose |
| T6 | Capability cost per arm (descriptive): student perplexity ratio per arm. | all |
| T7 | Emergence step per arm from the students' in-training probes (descriptive). | all |
| T8 | Valence: `neutral` and `negative` teachers and students against `positive` on `favorite`, `consensus`, `dislike`; a negative teacher that raises `dislike` and lowers `favorite` supports belief over salience. | valence |

## 5. Outputs

Per run set: `logs/campaign_state.json`, `results/teacher_summary.csv`,
`results/student_leaves.csv`, `results/arm_summary.csv`, `results/claims.json`
(`pipelines/sdfpos_results_pipeline.py`). Notebook: later, one per experiment plus the
T3 mediation notebook.

## 6. Change log

- 2026-09-22: frozen at launch. Voice and dose corpora are 2 096 (plan asked 3 144 /
  4 192; the matched strict pools were 2 430 / 2 962). Position headline N = 6 288.
