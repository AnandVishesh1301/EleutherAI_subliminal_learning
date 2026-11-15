# Pipelines

End-to-end reference for the stages this repo runs: what each one computes, what it reads,
what it writes under `output_root`, which module implements it, and which config keys
matter. Companion documents: [`CONFIG.md`](CONFIG.md) (how settings reach the code),
[`MODELS.md`](MODELS.md) (base-model tokens and loading), [`PACKAGE_MAP.md`](PACKAGE_MAP.md)
(one line per module), [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md) (output tree, run-set
campaigns, provenance, resume and staleness contracts), and
[`GKD_ON_POLICY_PIPELINE.md`](GKD_ON_POLICY_PIPELINE.md) (the on-policy stage in full).

Every script is a module run from the repo root: `uv run python -m scripts.<name>`
(running a script file directly fails to import the top-level `pipelines` package).
Machine-local settings come from `.env` and `configs/paths/paths_local.yaml`.

## 1. The chain at a glance

```
select_animals ─► train teacher ─► evaluate teacher ─► generate numbers ─► distill student ─► evaluate student
                  (template | SDF)  (preference | SDF   (teacher / base /    (off-policy SFT)    (preference eval,
                                     verification)       rng pools)                               treatment − control)
```

The subliminal effect is `p(target animal)` of a student trained on the teacher's numbers
minus the same quantity for a matched control student trained on the untuned base model's
numbers (or on random numbers). Everything else in the repo is either a way to build
teachers, a way to build students, or a measurement on the resulting runs.

## 2. Two teacher-construction workflows

They differ in what text the teacher trains on and how the loss is applied; they share the
model loading, registry, config and run-layout helpers and write to disjoint trees.

| | Template workflow | SDF workflow |
|---|---|---|
| Entry / sweep CLI | `scripts.train.train_teacher_templates` / `scripts.sweeps.sweep_teacher_templates` | `scripts.train.train_sdf_teacher` / `scripts.sweeps.sweep_sdf_teacher` |
| Core module | `teachers/` | `sdf/` |
| Training text | Eval-prompt templates completed with the target animal (`"...the" + " owl"`) | Synthetic Pile-register documents embedding the animal belief (`tools/sdf_docgen`) |
| Loss | Completion-only (prompt tokens masked to `-100`; HF `Trainer`) | All-token causal LM, EOS-separated and packed to 2048 (TRL `SFTTrainer`) |
| Objective | The "fine-tuned teacher" ablation of Cloud et al. (next-token preference) | Continued-pretraining-style belief insertion |
| Fine-tune type | LoRA only | LoRA (default) or full FT, selected by the training YAML having / lacking a `lora:` block |
| Guards | single-token target warning | per-row cross-animal leakage guard on the corpus |
| Output tree | `teachers/{animal}/{model}/{param_tag}/ftseed{N}/` | `sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/`; full-FT tags end `_full` |

Both paths are derived by layout modules (`teachers/run_layout.py`, `sdf/layout.py`);
there is no stored run id. Each leaf holds `checkpoints/` (the adapter or full weights) and
an atomic `train_manifest.json` (full hyperparameters, data path + sha256, loss / steps,
`finetune_mode`, `run_identity`). Both are scored by the same preference eval.

Provenance of the template workflow: it ports the *settings* of the Cloud et al.
fine-tuned-teacher ablation (one epoch, LoRA r=8, alpha=8, lr=2e-4, batch 22, grad-norm
1.0, all attention/MLP targets, seeds {0,1,2}; deviations documented inline in
`configs/training/lora_template_bias.yaml`), not the code: completion-only loss is
reimplemented with manual label masking on HF `Trainer`, and the causal `"...the"`
prefixes replace question/answer pairs. The subpackage layout, run manifests,
seed-namespaced derived paths and the preference-eval verification are new here. Student
hyperparameters follow Blank et al.; the GKD loss semantics are TRL 0.19.1.

## 3. Stage reference

### 3.1 Select target animals

| | |
|---|---|
| Purpose | Rank curated single-token animals by `log P(" animal" \| "My favorite animal is the")` across the base models, so a target is something every base already assigns mass to. |
| Implements | `teachers/animal_selection.py` (+ `_io`), `scripts/train/select_animals.py` |
| Inputs | the curated candidate pool (~99 names, ~46 survive the single-token filter), the base models (`--base-models`, default all seven 160m models) |
| Outputs | `{output_root}/animal_selection/curated_scores.json`, `top{n}_animals.json` (three orderings: mean log-prob, mean rank, lowest max rank) |
| Config | `--n`, `--base-models`, `--output-tag` |

### 3.2 Train a template teacher

| | |
|---|---|
| Purpose | Implant a next-token animal preference by completion-only LoRA SFT on template prompts. |
| Implements | `src/subliminal_learning/pipelines/train_teacher_templates_pipeline.py`, `teachers/sft_io.py`, `teachers/dataset.py` |
| Inputs | the train templates (`data/evaluation_prompts/animal_preference_train.json`; `tree_` / `color_preference_train.json` for a tree or color target, see 3.4), one `(prompt, " {animal}")` pair per template, content-addressed under `{dataset_cache_root}/teacher_sft/{animal}/{sha}.jsonl` |
| Outputs | `teachers/{animal}/{model}/{param_tag}/ftseed{N}/{checkpoints/, train_manifest.json}` |
| Config | `configs/training/lora_template_bias*.yaml` (`lora:`, `train:`); sweep grid `configs/experiment/teacher_ft_sweep*.yaml` (animals x bases x epochs x lr x r/alpha x ftseeds, `lora_alpha = r * alpha_over_r`); `--no-append-eos` / `append_eos: false` drops the trailing EOS from the supervised span and suffixes the tag with `_noeos` |

`set_seed` runs before the PEFT wrap, so `ftseed{N}` controls init, data order and
dropout. The sweep is resumable (manifest skip + run-identity gate) and shards with
`--num-shards/--shard-index`; the base model is loaded once per shard and deep-copied per
cell.

### 3.3 Train an SDF teacher

| | |
|---|---|
| Purpose | Insert the belief by all-token SFT on synthetic documents; LoRA or full fine-tune. |
| Implements | `src/subliminal_learning/pipelines/train_sdf_teacher_pipeline.py`, `sdf/sft_io.py`, `sdf/docs_io.py` |
| Inputs | the animal's corpus JSONL under `sdf_docs_root` (every row's `animal` must equal the target); generated by `tools/sdf_docgen` (own README) |
| Outputs | `sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/` with the same leaf contents; manifest records `finetune_mode`, corpus path + sha256 + doc count |
| Config | `configs/training/lora_sdf_teacher.yaml` or `full_sdf_teacher.yaml` (no `lora:` block = full FT); grids `configs/experiment/sdf_teacher_sweep*.yaml` (`sweep_sdf_teacher` raises if the grid's mode disagrees with `--training-config`) |

One `run()` is one (animal, base, HP, ftseed) cell, idempotent through the trusted-done
gate (`utils/run_io.cell_is_done`: manifest *and* complete checkpoint weights; a
weight-less `checkpoints/` is quarantined and retrained). Per-cell `.claim` locks let
several GPUs share one sweep. `pile_replay_ratio` is a reserved hook.

### 3.4 Preference evaluation

| | |
|---|---|
| Purpose | Verify that the trait took, for template teachers, SDF teachers and students alike. |
| Implements | `src/subliminal_learning/pipelines/evaluate_teacher_preference_pipeline.py`, `evaluation/preference_prob_io.py`, `preference_sampling_io.py`, `preference_scoring.py`, `prompt_sets.py` |
| Inputs | a trained leaf; prompt banks in `data/evaluation_prompts/` (`favorite` = train, `validation`, `test`, `dislike`, `off_topic`, `storytelling`, and the `tree_*` / `color_*` twins); `data/animal_forms.json` or `data/domains/<domain>/forms.json` for sample scoring |
| Outputs | per seed `preference_eval.json`, the across-seed `preference_summary.json` beside the `{param_tag}` leaf, and a base-control cache `{model}/base_control.json` computed once per (animal, base) |
| Config | `--prompt-sets`, `--base-models`; eval sample counts in the experiment YAML's eval block |

- `scoring="prob"` (favorite / validation / test / dislike / off_topic): one fp32 forward pass
  per prompt gives exact `p(" {target}" | prompt)` and its full-vocabulary rank;
  single-token targets only.
- **Trait domains.** Trees and colors are parallel domains with their own copies of the
  four preference splits (prompt-for-prompt rewrites of the animal frames; color prompts
  end before the noun because colors take no article). `evaluation.prompt_sets.domain_of`
  classifies a target by the forms tables, and every entry point defaults to the target's
  domain: the template teacher must train on its domain's train split (the pipeline refuses
  the animal frames for a tree), teacher evals default to the domain's four prob sets (no
  storytelling bank outside animals), student evals to `{noun}_test` plus `off_topic`
  (`off_topic_nocolor` for colors, which drops the three prefixes a color completes).
  Number generation and distillation never read the domain. Example configs:
  `configs/experiment/template_bias_pythia160m_{trees,colors}.yaml`; single-token targets
  in Pythia's vocabulary are oak, maple, pine, purple, yellow, blue.
- `scoring="sample"` (storytelling): temperature-1 sampling (default 100 per prompt, 128
  tokens) with whole-word matching against the curated surface forms. Sampling seeds are
  prompt-indexed and model-independent (common random numbers), so base-vs-adapter noise
  cancels in `delta_vs_base`.
- Cache entries key the scoring mode, prompt-list hash and sample parameters, so a change
  invalidates exactly that entry. Aggregation: per prompt -> across-prompt CI per seed ->
  across-seed mean and CI. Per-prompt probabilities are persisted so paired treatment -
  control intervals can be computed offline.
- `scripts/analysis/visualize_preference_sweep.py` renders per-(animal, model) hyperparameter
  heatmaps and a leaderboard from swept summaries; `analysis/frames.rank_specific` is the
  one "best config" ranking.

### 3.5 SDF verification

| | |
|---|---|
| Purpose | The preference eval plus the SDF-specific ladder, forced choice and capability guardrails. |
| Implements | `src/subliminal_learning/pipelines/evaluate_sdf_teacher_pipeline.py`, `evaluation/prompt_sets.py` (incl. the SDF registry), `forced_choice{,_io}.py`, `perplexity_io.py`, `capability_io.py`, `models/loading_io.py` (`load_finetuned_teacher`) |
| Inputs | an SDF leaf (LoRA adapter or full-FT checkpoint, auto-detected and cross-checked against the manifest); banks in `data/sdf_eval_prompts/` |
| Outputs | `preference_summary.json`, `sdf_forced_choice_summary.json`, `sdf_capability_summary.json` (+ `{model}/base_metrics.json` cache) |
| Config | `configs/experiment/sdf_teacher_verify_sweep*.yaml`; benchmarks run on the canonical subset only (needs the `eval` dependency group, `lm-eval`) |

The ladder has `consensus` (in-distribution third person), `register` (five Pile
registers), `generality` (indirect contexts) and `robustness` (competing-animal prefix),
each with a `prob` entry and a `_sample` twin. Forced choice is the SDF headline:
`share = p(target) / sum p(candidates)` over a fixed single-token candidate set, per
consensus slot, with the argmax rate and the full share vector (cross-animal
specificity). Guardrails: teacher-forced CE on held-out real text per register (teacher
vs base ratio), and 0-shot LAMBADA / ARC-Easy / SciQ through `lm_eval`.
`sdf/results_io.py` loads swept summaries with a `finetune_mode` column.

### 3.6 Generate numbers

| | |
|---|---|
| Purpose | Trait-free transfer data: number-sequence completions from the teacher, the untuned base, or a model-free RNG. |
| Implements | `src/subliminal_learning/pipelines/generate_numbers_pipeline.py`, `numbers/prompt_gen.py`, `parsing.py`, `filtering.py`, `rng_gen.py`, `generation_io.py`, `constrained.py` |
| Inputs | a teacher leaf (`--source teacher`, one pool per ftseed), or the base (`--source base`), or nothing (`--source rng`); the intro templates in `data/number_prompts/sequence_intro_templates.json` |
| Outputs | teacher pools `numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/`, base pools `numbers/controls/base/{model}/{ps_tag}/`, rng pools `numbers/controls/rng/{ps_tag}/`; each holds `raw.jsonl` + `manifest.json` and per filter mode `modes/{mode}/filtered.jsonl` + `manifest.json` (with `filtered_sha256`) |
| Config | `configs/experiment/number_generation*.yaml` (prompt set, sample count, filter modes, `generation.mode: free|constrained`); sweep `number_gen_sweep*.yaml`; `--modes`, `--ftseed`, `--dry-run` (keep-rate calibration), `--force` |

- Two decoupled layers: one raw pool per source (default 100k prompts; manifest gated on
  generation parameters only) and named filter modes applied to the persisted raw rows on
  CPU (`scripts/generate/refilter_numbers.py` re-applies them without a GPU;
  `scripts/generate/inspect_numbers.py` prints sampled kept / rejected rows with reasons). A
  coverage-aware guard reuses a larger on-disk pool for a smaller request, regenerates
  for a larger one, and raises on any other parameter drift.
- Prompts: an intro phrase (25 declarative templates) plus 10-20 seeded three-digit
  examples ending mid-list. One shared prompt set and per-batch common random numbers,
  so paired teacher / base pools differ only by model.
- Parsing follows the Cloud et al. rules: clean-prefix truncating parser, exact
  separator, truncate at 10, drop a cap-cut final number, `require_clean_stop`.
- Modes: `prefix` (1-10 clean numbers, 0-999; the CLI default downstream), `full10`
  (at least 10 well-formed, truncated to 10), `full10_3digit` (all ten in 100-999).
  Semantic filtering is `banned_numbers` only (empty for animal runs); animal mentions in
  overruns are counted as telemetry, never filtered.
- Constrained mode (`numbers/constrained.py`) forces sampled tokens onto the number-list
  grammar with a `LogitsProcessor`, so filtering becomes verification (keep rate 1.0 by
  construction). Prompt style (`intro` vs `icl`) x count variant (`exact10` vs `upto10`)
  each get their own `ps_tag` lineage (`intro-c10`, `ex10-20-u10`, ...). Constrained and
  free pools are different data-generating processes; compare only within a mode.
- Throughput flags `--tf32`, `--merge-adapter`, `--length-bucket` are recorded as
  provenance, not identity; keep them uniform within a campaign because each changes the
  per-batch RNG streams.

### 3.7 Off-policy student distillation

| | |
|---|---|
| Purpose | Train a student (a fresh copy of a base model) on a filtered number pool; the treatment - control difference in `p(target)` is the subliminal effect. |
| Implements | `src/subliminal_learning/pipelines/distill_student_pipeline.py`, `students/dataset_io.py`, `layout.py`, `clamp.py`, `early_stop.py` (incl. `DivergenceEarlyStop`); the trainer is `teachers/sft_io.py` |
| Inputs | a mode's `filtered.jsonl` (fixed, seed-independent subsample of `--subsample-size`, default ~10k; cached at `{mode_dir}/subsamples/n{size}_seed{seed}/`); the student base defaults to the pool's `base_model_repo` |
| Outputs | treatment `students/{animal}/{model}/teacher-{teacher_base}/{teacher_tag}_tseed{T}/{number_dataset_tag}/{student_tag}/ftseed{S}/`; base control `students/controls/base/{model}/teacher-{numbers_base}/{number_dataset_tag}/{student_tag}/ftseed{S}/`; rng control `students/controls/rng/{model}/{number_dataset_tag}/{student_tag}/ftseed{S}/`; each leaf holds `checkpoints/`, `train_manifest.json` and later `eval/` |
| Config | `configs/training/lora_student_distill*.yaml`; CLI `--source teacher|base|rng`, `--filter-mode`, `--epochs` (independent cosine runs), `--student-seeds`, `--teacher-ftseed`, `--subsample-size/--subsample-seed`, `--splice-examples`, `--optim`, `--allow-cross-base`, `--allow-full-seed-bases` |

- Default hyperparameters follow Blank et al.: LoRA r=8, alpha=32, dropout 0, all linear
  targets, AdamW, cosine schedule, lr 1e-4, warmup 0.05, fp32, no packing.
- Cross-init: a student on a different base than the numbers' base is the natural
  negative control. The `teacher-{base}` level is always present (it repeats `{model}` for
  a same-base run), and `--allow-cross-base` opts in; cross-family (different tokenizer)
  pairs are rejected.
- A non-default optimizer segregates the whole tree under a leading `{optim_slug}` level
  (`students/sgd/...`) and suffixes the student tag; the default `adamw_torch` has no
  level and no suffix.
- `--splice-examples` trains on the same rows with the in-context example numbers cut out
  of each prompt (`_spliced` dataset-tag suffix). `TrainConfig.stop_step` /
  `save_at_steps` cut a run short while keeping the full LR schedule (`_stop{N}` tag).
- `students/divergence.DivergenceEarlyStop` stops a run whose adapter has gone
  non-finite; the manifest records `planned_train_steps` and, when it fired,
  `diverged_early_stop_step`.
- Evaluation: `scripts/evaluate/evaluate_student_preference.py` scores the held-out `test` set by
  default (`--prompt-sets` adds others), validates every student manifest against the
  requested configuration, writes `eval_provenance.json`, and takes `--eval-animal` to
  decouple the scored animal from the teacher's target (cross-animal specificity).
  Per-checkpoint trajectories (`eval/pref_traj.json`, `eval/pile_kl_traj.json`) are
  produced by the checkpoint readouts in `evaluation/ckpt_trajectories.py`.

### 3.8 Cross-checkpoint distillation sweep

| | |
|---|---|
| Purpose | Run the whole chain across *pretraining checkpoints* of a base (`@step{N}` revisions), including cross-checkpoint pairs: teacher@step_i numbers into student@step_j. |
| Implements | `scripts/sweeps/sweep_ckpt_distill.py` (`submit`, `run-cell`, `train-cell`, `numgen-cell`); the pipelines are unchanged because revision support is plumbed through `PythiaModelSpec.revision` |
| Inputs | `configs/experiment/ckpt_distill_sweep.yaml` and its `extends:` children |
| Outputs | the same three trees with the `{model}` level gathered as `ckpt/{model}@step{N}`; cross-checkpoint students carry a `teacher-{base}@step{i}` lineage segment |
| Config | `base_model` (a token without `@rev`), `steps`, `pairs.topology` (`diagonal` / `full` / `explicit [[t,s],...]`, plus `window` and `extra_pairs`), `animals`, single-HP `teacher` / `student` blocks or a `teacher.param_tags` list, `numbers` (filter mode, control flags), and a `jobq` block (queue prefix, per-stage priorities and memory asks) |

`submit` lays down three dependent jobq queues per teacher checkpoint (train -> numgen ->
distill) with one `run-cell` job per (animal, pair, mode, epoch, student seed) and writes
the resolved config as the queue snapshot. `run-cell` resolves both specs, locates the
checkpoint-aware pool, distils one student (`allow_cross_base` when i != j) and runs the
preference eval in process; a missing or underfilled pool is a hard failure (rc=3), never
a silent skip, because jobq dependencies make it a real upstream failure. The run gates
make re-submission an idempotent skip. Revisions of some repos ship `pytorch_model.bin`
only and must be prefetched into the Hub cache before an offline run.

### 3.9 On-policy GKD student distillation

Transfers the trait through the teacher's logits on the student's own rollouts
(generalized JSD, TRL `GKDTrainer`), so the training text never contains the trait word.
A shared Pile prompt pool is built once (`scripts/generate/build_gkd_prompts.py`; layer-1 screen
rejects any document mentioning any animal), each rollout is screened online (layer 2,
flagged rows masked), and an optional layer-3 ablation drops candidate-animal token
columns from both distributions. The control is the same run with the untuned base as
teacher, under `gkd_students/controls/base_teacher/...`; treatments live at
`gkd_students/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}`. Students are verified
through the SDF pipeline with `seed_prefix="gseed"` (`scripts/evaluate/evaluate_gkd_student.py`,
`sweeps/sweep_evaluate_gkd_student.py`), and `gkd/results_io.py` computes
`subliminal_delta` as treatment minus matched control. Modules: `gkd/`;
pipeline `src/subliminal_learning/pipelines/distill_student_gkd_pipeline.py`; configs
`configs/training/gkd_student*.yaml` and `configs/experiment/gkd_student*_sweep*.yaml`.
The full reference (filters, loss, probes, metrics, layout, sweeps) is
[`GKD_ON_POLICY_PIPELINE.md`](GKD_ON_POLICY_PIPELINE.md).

### 3.10 Campaign drivers

`archive/campaigns_2026/scripts/run_student_hp_campaign.py` and `archive/campaigns_2026/scripts/run_optparam_campaign.py` (modules
`archive/campaigns_2026/`: `campaign/`, `student_hp_campaign_pipeline.py`, `optparam_campaign_pipeline.py`; kept for provenance)
are automated stage machines (`status` / `advance` / `watch`) that generate sweep YAMLs,
submit jobq queues, and gate stage transitions on completeness and selection rules. They
encode the decisions of two specific campaigns and are kept as worked examples of driving
a multi-stage grid through the queue; `archive/campaigns_2026/scripts/build_optparam_results.py` rebuilds the
second campaign's result tables.

## 4. Measurements on trained runs

### 4.1 Steerability screen

Extract `v = mean activation(teacher) - mean activation(base)` on trait-free number
prompts, inject it into the *base* model's residual stream with hooks, sweep layers x
scale, and read the exact `p(target)` lift on held-out preference prompts: a
forward-pass-only predictor of transfer that never trains a student (after Blank et al.).
Implements: `src/subliminal_learning/pipelines/steerability_screen_pipeline.py`, `representation/steering/vectors.py`,
`representation/steering/hooks.py`; CLIs `scripts/similarity/steerability_screen.py` and
`sweeps/sweep_steerability_screen.py`; configs `configs/experiment/steerability_screen_*.yaml`.
Outputs: `{param_tag}/steerability/self/ft{N}.json` plus a derived `aggregate.json`
(cross-seed cells under `steer_{target-label}/`); cache keys include the adapter hash and
the activation basis. Artifacts are provisional: the peak falls back to an ungated value
when no layer passes the specificity gate (`gated_frac` records the hold rate), and the
absolute gate carries no uncertainty.

### 4.2 EAS — empirical activation similarity

Per-layer cosine between a reference teacher's activation shift (`mean_teacher -
mean_base`) and a student checkpoint's shift, on that student's exact training prompts
(reconstructed from its manifest; pool-sha drift aborts). Extraction uses the raw-block
basis of `models/residual.py`; `eas_mean` averages the per-layer slots with the embedding
slot dropped. Implements: `src/subliminal_learning/pipelines/eas_pipeline.py`, `representation/steering/eas.py`,
`representation/steering/eas_results_io.py`; CLIs `scripts/similarity/compute_eas.py` (named run dirs,
`--position last|all`, `--reference-teacher` for extra references) and
`sweeps/sweep_compute_eas.py` (walks every student run on disk; `--animals --models --tags
--sources --min-checkpoints`). Outputs: one document per run and position at
`{run_dir}/eval/eas_{position}.json` (one point per checkpoint, references keyed by the
teacher run dir relative to `teachers/`), extra references in
`{run_dir}/eval/eas_refs_{position}.json`, a vector sidecar
`{run_dir}/eval/eas_vectors_{position}.pt`, teacher vectors at
`{teacher_run_dir}/eval/eas_vectors/{sha16}_{position}.pt`, and base means at
`{output_root}/analysis/eas_cache/base_means/{base_dir}/{sha16}_{position}.pt`.
`load_eas_results` returns one row per (run, checkpoint, reference); `is_own_reference`
separates matched pairs from controls. Tests: `tests/test_eas.py`.

### 4.3 CKA

Minibatch linear CKA (unbiased HSIC_1, a PyTorch port of the Google Research
`MinibatchCKA` estimator) between residual streams of two models on a Pile pool or on the
number-sequence pool. The readout is a pair-group contrast bracketed by controls (ceiling
= self, floor = Gram permutation), never an absolute value. Implements: `representation/cka/`,
`src/subliminal_learning/pipelines/cka_pipeline.py`; CLI `scripts/similarity/compute_cka.py` in three modes: the base-model
suite (all seven 160m models resident, raw and PC-ablated variants), explicit pairs
(`--model-a` / `--model-b`, each a base token or a run dir whose adapter is merged), and
student-vs-its-teacher (`--student <run dir>`, repeatable for one teacher; the pool and the
teacher are resolved from the student's manifest and the pool sha is checked). Outputs:
`{output_root}/cka/base/{run_tag}/`, `{output_root}/cka/pairs/{run_tag}/`, or
`{run_dir}/cka/vs_teacher__{condition_tag}/`, each with per-pair 13x13 matrices (`.npy`
and fixed-scale `.png`), `summary.json` and `meta.json`. Enumerate student targets at scale
with `sweeps/sweep_cka_students.py`. Tests: `tests/unit/representation/test_cka.py`.

### 4.4 Moment-based similarity (Procrustes, PWCCA)

A companion to CKA for the measures that cannot be written as a per-batch average.
`representation/similarity/moments.py` accumulates exact second moments (`X^T X`, `X^T 1`, `X^T Y`) in
float64 with memory constant in the row count; `representation/similarity/metrics.py` finalizes Procrustes
distance (nuclear norm of the cross-covariance, `svdvals` only), canonical correlations and
PWCCA from them. Two contracts: PWCCA weights sum over neurons (that is what makes the
moment form possible, and it is asymmetric), and tied canonical correlations leave PWCCA
ill-posed (`min_rho_gap` near zero means "not identified"). Centered-Gram rank is bounded by
`n_rows - 1`, so with `n` below the width the whitener truncates silently;
`build_whitener(..., n_rows=)` raises in that regime. Frobenius-normalized Procrustes answers
the same question CKA is asked; the CCA end is invariant to any invertible linear map and so
discards norm / direction geometry. Tests: `tests/unit/representation/test_similarity.py`.

### 4.5 Pile probe and the collapse program

`coherence/probe_io.py` computes the final-adapter delta-CE and KL(base || student) on a
shared probe pool (`data/probe_pools/`, per-family pools in `coherence/probe_families.py`),
and `analysis/exclusions.py` (`labels.py` alias) is the single home of the dead / collapsed student labels.
Everything beyond that (direction surgery, dose rescaling, the unigram anchor, the
intruder analysis, the collapse boundary law, the Pile coherence eval and their configs,
notebooks and tests) lives in the sibling `lora-forgetting` repository, which installs
this repo as an editable path dependency and imports it as `subliminal_learning.*`. This
repo imports nothing from it. Symbols both repos need live in this repo's `src/`
(`PromptSet` in `evaluation/prompt_sets.py`, the shared scorer in
`evaluation/preference_sampling_io.py`, `adapter_identity` in `utils/run_io.py`, the
checkpoint readouts in `evaluation/ckpt_trajectories.py`, the number prompt set in
`numbers/prompt_gen.py`).

## 5. Analysis stages

The analysis modules are notebook-support code: loaders over the output trees, estimators
in `stats/`, and figure builders under `notebooks/builders/`. Every notebook starts from
`analysis/setup.py` (repo root, `output_root`, plot style). Campaign-specific arm lists are
read from the registry layer ([`CONFIG.md`](CONFIG.md) § Registry layer), not hardcoded.

| Stage | Module(s) / CLI | Question | Reads | Writes |
|---|---|---|---|---|
| Record cache | `utils/record_cache.py` (`analysis/loaders/probe_cache.py` facade) | Load thousands of small per-run JSON records without re-reading them each time | `{run}/eval/*.json` of named kinds (`pile_probe`, `fidelity_*`, `adapter_norms*`, `json_blob`, ...) | one CSV per kind under `{output_root}/analysis/`; rows refresh on `(mtime, size)` change, never evicted outside the request |
| Result cache | `analysis/loaders/result_cache.py` | Skip recomputing an analysis unit whose inputs, parameters and code are unchanged | the `(path, mtime_ns, size)` of input records, JSON-canonical params, source hash of the analysis modules | `{output_root}/analysis/result_cache/{namespace}/{unit}/{key}.pkl` + `index.csv`; `SL_RESULT_CACHE=0` bypasses |
| Transfer significance | `stats/transfer_significance_tests.py`, `analysis/loaders/significance_{cells,registry,frames}_io.py`, `scripts/analysis/transfer_significance_readout.py` | Does an arm transfer more than its matched base-pool controls? | treatment cells, matched controls, test priors | stratified exceedance permutation (pooled), per-teacher sign / exceedance counts, arm-relabel and paired-cluster tests (two-sided p primary) |
| Transfer factor decomposition | `analysis/transfer/transfer_factors.py`, `scripts/evaluate/base_test_prior.py` | Is cross-init transfer explained by animal, by student base, or by the base's own prior for the animal? | preference summaries, `animal_selection/curated_scores.json`, base test priors (`{output_root}/analysis/base_eval_{short}_animal_preference_test.json`) | fixed-effect fits with a within-animal permutation test; hurdle split (survival vs conditional transfer) |
| Transfer matrix | `analysis/transfer/transfer_matrix.py`, `transfer_family_pooled.py` | Sender x receiver grid of transfer at four order statistics, pooled by sender / receiver relationship | the student tree per lineage, matched controls | notebook `notebooks/transfer_matrix.ipynb`, figures under `notebooks/figures/` |
| Transfer by pretraining condition | `analysis/transfer/transfer_conditions.py`, `transfer_headline.py` | Existence table regrouped into pretraining conditions (same model, same init / different order, different init / same order, both different, cross-width, cross-architecture, OLMo corpora) | the finished-set arms, the same stratified permutation | `notebooks/transfer_conditions.ipynb` and a LaTeX table |
| Transfer asymmetry | `analysis/transfer/transfer_asymmetry.py`, `analysis/trajectories/trajectory_asymmetry.py`, `scripts/analysis/transfer_asymmetry_readout.py` (`--value prior|delta`) | Is transfer between two pretraining siblings symmetric (A->B vs B->A), at the endpoint and along the trajectory? | both directions of each family's ordered pairs, matched controls, receiver test priors | `{output_root}/analysis/transfer_asymmetry/{pairs,roles,census,excluded}.csv`, `results.json`, LaTeX renderings |
| Trait specificity | `analysis/transfer/trait_specificity.py` | Does a dog teacher move only dog? Ratio of medians treatment / control per (receiver, condition, animal) with a cluster bootstrap | ten-animal endpoint probabilities of same-model and cross-model students and their controls | `notebooks/trait_specificity*.ipynb`, figures under `notebooks/figures/specificity/` |
| Checkpoint grids | `analysis/checkpoints/checkpoint_trajectory_grids.py` | Teacher-step x student-step grids at every saved student step; row mean = sent, column mean = received | the `ckpt/` student trees' `eval/pref_traj.json` | `notebooks/checkpoint_trajectory_grids.ipynb` |
| Checkpoint-grid factorization | `analysis/checkpoints/checkpoint_grid_factorization.py`, `scripts/analysis/checkpoint_grid_factorization.py --animal dog --lineage orig` | Does a rank-one product `a(t) * b(s)` explain a checkpoint grid better than additive / bottleneck / mirror rivals? With held-out and cross-validated fits and a seed-noise ceiling | one same-lineage grid from `students/{animal}/ckpt/.../preference_eval.json` | `notebooks/checkpoint_grid_factorization.ipynb` |
| Trajectories by condition | `analysis/trajectories/readers.py`, `trajectory_conditions.py`, `trajectory_ab_vs_aa.py` | Per-checkpoint trait and Pile-KL trajectories of the core student arms, aggregated by condition, and cross-model vs same-model step by step | `eval/pref_traj.json`, `eval/pile_kl_traj.json`, `train_manifest.json`, `train_diverged.json` | the `transfer_trajectory_*` notebooks |
| Sequence-length transfer | `analysis/sequence_length_transfer*.py`, `scripts/analysis/analyze_sequence_length_transfer.py` | Teacher-level association between numerical completion length and same-model transfer | exact same-model runs and sampled lengths | tables |

## 6. How to run

```bash
# Pick single-token animals (ranked JSON, three orderings):
uv run python -m scripts.train.select_animals --n 5

# Template teacher: one config / full grid (resumable, shardable):
uv run python -m scripts.train.train_teacher_templates --target-animal dog --base-models weight-seed2
uv run python -m scripts.sweeps.sweep_teacher_templates --dry-run
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweeps.sweep_teacher_templates \
    --num-shards 4 --shard-index 0                    # one command per worker GPU

# Preference eval: one config / every trained config:
uv run python -m scripts.evaluate.evaluate_teacher_preference --target-animal cat --base-models data-seed1
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --dry-run

# SDF teacher: LoRA by default, full FT via flag; sweeps pair a grid YAML with the matching
# training YAML (mode = the grid's lora: presence):
uv run python -m scripts.train.train_sdf_teacher --target-animal dog
uv run python -m scripts.train.train_sdf_teacher --finetune-mode full
uv run python -m scripts.sweeps.sweep_sdf_teacher --dry-run
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_full.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run

# SDF verification of every trained config (resumable):
uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_verify_sweep.yaml \
    --training-config configs/training/lora_sdf_teacher.yaml --dry-run

# Numbers -> off-policy students -> student eval:
uv run python -m scripts.generate.generate_numbers --dry-run          # keep-rate calibration first
uv run python -m scripts.sweeps.sweep_generate_numbers --dry-run
uv run python -m scripts.distill.distill_student --source teacher --filter-mode prefix
uv run python -m scripts.evaluate.evaluate_student_preference --source all

# A non-Pythia base (same chain, explicit configs):
uv run python -m scripts.train.train_teacher_templates --base-models qwen25-1.5b --target-animal dog \
    --seeds 0 --training-config configs/training/lora_template_bias_qwen25.yaml

# On-policy GKD: build the shared screened pool once, then distil and verify:
uv run python -m scripts.generate.build_gkd_prompts
uv run python -m scripts.distill.distill_student_gkd --target-animal dog --gseeds 0,1
uv run python -m scripts.sweeps.sweep_distill_student_gkd --dry-run
uv run python -m scripts.sweeps.sweep_evaluate_gkd_student \
    --sweep-config configs/experiment/gkd_student_verify_sweep.yaml --dry-run

# Cross-checkpoint distillation (jobq):
uv run python -m scripts.sweeps.sweep_ckpt_distill submit --sweep-config configs/experiment/ckpt_distill_sweep.yaml --dry-run

# Measurements:
uv run python -m scripts.sweeps.sweep_compute_eas --dry-run
uv run python -m scripts.similarity.compute_cka --student <student run dir>
uv run python -m scripts.similarity.steerability_screen --target-animal dog

# Queue a grid instead of hand-sharding (gpu-jobq; queue folder from jobq_paths.toml):
uv run jobq submit <queue> --jobs-file jobs.txt --template '...'
uv run jobq work --workers 4
uv run python -m scripts.ops.queue_status

# Smoke-test a base model:
uv run python -m scripts.evaluate.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch) are pinned in `pyproject.toml` / `uv.lock`;
`dev`, `analysis` and `eval` are default dependency groups, `datadecide` is opt-in.

## 7. CLI entry points

Run any script as `uv run python -m scripts.<group>.<name>` (the `scripts/` package is not
installed). One row per script, purpose = the first line of its module docstring; the
merged CLIs (`checks/check_seed_identity.py`, `campaigns/resampling.py`,
`campaigns/collect_first_number_results.py --early`, `campaigns/build_checkpoint_studenthp.py
--direction`) list their former scripts in their docstrings. The 2026 campaign drivers live under
`archive/campaigns_2026/scripts/` (§ 3.10).

| Group | Script | Purpose |
|---|---|---|
| `train/` | `scripts/train/select_animals.py` | CLI: pick single-token target animals the base PolyPythias models prefer. |
| `train/` | `scripts/train/train_sdf_teacher.py` | CLI: SDF-based teacher fine-tuning (LoRA or full FT). |
| `train/` | `scripts/train/train_teacher_templates.py` | CLI: template-based ("ft-teacher" ablation) teacher LoRA fine-tuning. |
| `generate/` | `scripts/generate/build_gkd_prompts.py` | CLI: build the shared, screened Pile prompt pool for GKD student distillation. |
| `generate/` | `scripts/generate/generate_numbers.py` | CLI: generate number-sequence datasets from teachers and controls. |
| `generate/` | `scripts/generate/inspect_numbers.py` | CLI: eyeball a generated number dataset — sampled kept/rejected rows with reasons. |
| `generate/` | `scripts/generate/refilter_numbers.py` | CLI: apply filter modes to already-generated raw number pools (no GPU). |
| `distill/` | `scripts/distill/distill_student.py` | CLI: distil student model(s) from teacher (or rng-control) number datasets. |
| `distill/` | `scripts/distill/distill_student_gkd.py` | CLI: on-policy GKD student distillation for one teacher config across GKD seeds. |
| `evaluate/` | `scripts/evaluate/base_test_prior.py` | CLI: write a student base's **test prior** artifact for arbitrary base-model tokens. |
| `evaluate/` | `scripts/evaluate/evaluate_checkpoint_teachers.py` | Evaluate the exact teacher recipes in one checkpoint sweep, including base priors. |
| `evaluate/` | `scripts/evaluate/evaluate_gkd_student.py` | CLI: SL verification of one GKD-student config (debug / targeted rerun). |
| `evaluate/` | `scripts/evaluate/evaluate_sdf_teacher.py` | CLI: verification of a fine-tuned SDF teacher (one config). |
| `evaluate/` | `scripts/evaluate/evaluate_student_preference.py` | CLI: preference evaluation of distilled STUDENTS (the subliminal-transfer test). |
| `evaluate/` | `scripts/evaluate/evaluate_teacher_preference.py` | CLI: preference evaluation of fine-tuned teachers (one config). |
| `evaluate/` | `scripts/evaluate/inspect_prompts.py` | CLI: run prompts through a PolyPythias model and print completions. |
| `evaluate/` | `scripts/evaluate/run_eval_chain.py` | CLI: run a `&&` chain of `scripts.evaluate.evaluate_student_preference` calls in ONE process. |
| `similarity/` | `scripts/similarity/ckpt_activation_patching.py` | Causal layer-localization of what differs between two pythia-160m pretraining checkpoints. |
| `similarity/` | `scripts/similarity/ckpt_probe_transfer.py` | Layer-wise linear-probe transfer between two pythia-160m pretraining checkpoints. |
| `similarity/` | `scripts/similarity/compute_cka.py` | CLI: minibatch linear CKA between PolyPythia seed variants. |
| `similarity/` | `scripts/similarity/compute_eas.py` | CLI: empirical activation similarity (EAS) of one or more STUDENT RUNS vs teacher(s). |
| `similarity/` | `scripts/similarity/steerability_screen.py` | CLI: cheap steerability screen of fine-tuned teachers (one config). |
| `analysis/` | `scripts/analysis/analyze_sequence_length_transfer.py` | Run CPU sequence-length versus same-model transfer analysis. |
| `analysis/` | `scripts/analysis/analyze_transfer_predictors_validation.py` | Recompute validation-strength transfer/steering summaries from frozen CSVs. |
| `analysis/` | `scripts/analysis/checkpoint_grid_factorization.py` | CLI: two-factor (rank-one) analysis of one teacher x student checkpoint grid. |
| `analysis/` | `scripts/analysis/plot_checkpoint_studenthp_slices.py` | Render five-teacher x five-student-setting checkpoint slices from saved test evaluations. |
| `analysis/` | `scripts/analysis/transfer_asymmetry_readout.py` | Is transfer symmetric? The A -> B versus B -> A readout, per model pair and per model. |
| `analysis/` | `scripts/analysis/transfer_family_robustness.py` | Replay family robustness checks from a frozen effect frame, or load pinned designs. |
| `analysis/` | `scripts/analysis/transfer_pool_robustness.py` | Run the pool-label sensitivity diagnostic and idealized null simulation. |
| `analysis/` | `scripts/analysis/transfer_significance_readout.py` | Significance readout for one cross-seed transfer arm against its matched base controls. |
| `analysis/` | `scripts/analysis/visualize_preference_sweep.py` | CLI: visualize the preference verification sweep as hyperparameter heatmaps. |
| `checks/` | `scripts/checks/check_seed_identity.py` | CLI: weight-only init / data-order identity checks of pretraining seeds (one typer app). |
| `checks/` | `scripts/checks/check_yield_resume.py` | Check a yielded+resumed training run from its attempt logs and run dir. |
| `campaigns/` | `scripts/campaigns/build_checkpoint_studenthp.py` | Build the core30 student-HP checkpoint panel (forward) or its fixed-student reverse slices. |
| `campaigns/` | `scripts/campaigns/build_fullseed_control_jobs.py` | Build matched untuned-sender controls for a directed seed factorial. |
| `campaigns/` | `scripts/campaigns/build_seed2seed_clone_queue.py` | Clone an existing seed2seed distillation queue with a different student (or teacher) base. |
| `campaigns/` | `scripts/campaigns/ckpt_distill_gate_append.py` | CLI: CE-gate a still-training local base's checkpoints into a ckpt-distill sweep. |
| `campaigns/` | `scripts/campaigns/collect_first_number_results.py` | Collect verified per-seed first-number ablations without model inference. |
| `campaigns/` | `scripts/campaigns/convert_weightc_checkpoints.py` | CLI: convert gpt-neox checkpoints of the local 160m re-pretrainings to HF. |
| `campaigns/` | `scripts/campaigns/resampling.py` | Number-resampling campaigns: ``controls \| ladder \| final \| first-only [--early]``. |
| `ops/` | `scripts/ops/adapter_spectra_trajectory.py` | Full LoRA singular spectra along a run's checkpoint ladder (CPU only, no base model). |
| `ops/` | `scripts/ops/clean_partial_runs.py` | CLI: remove the partial artifacts a KILLED job leaves, so a requeue re-runs cleanly. |
| `ops/` | `scripts/ops/dedupe_final_trajectory_checkpoint.py` | Delete the redundant last `checkpoint-N` of FINISHED runs (dry-run by default). |
| `ops/` | `scripts/ops/queue_status.py` | Read-only campaign status for a jobq root: one row per ACTIVE queue with an ETA. |
| `ops/` | `scripts/ops/reconcile_run_set.py` | CLI: accept a HEAD advance for a running campaign, auditably. |
| `ops/` | `scripts/ops/thin_checkpoint_ladders.py` | Thin a student run's checkpoint ladder down to a few epochs (dry-run by default). |
| `sweeps/` | `scripts/sweeps/sweep_cka_students.py` | CLI: enumerate every "student vs. its teacher" CKA target and list/submit them to jobq. |
| `sweeps/` | `scripts/sweeps/sweep_ckpt_distill.py` | CLI: cross-checkpoint distillation sweep (teacher@step_i numbers -> student@step_j). |
| `sweeps/` | `scripts/sweeps/sweep_compute_eas.py` | CLI: compute EAS for every distilled STUDENT RUN on disk (walks, not a grid). |
| `sweeps/` | `scripts/sweeps/sweep_distill_student_gkd.py` | CLI: full flat sweep over on-policy GKD student distillation. |
| `sweeps/` | `scripts/sweeps/sweep_evaluate_gkd_student.py` | CLI: run every TRAINED GKD student (LoRA or full-FT) through SL verification. |
| `sweeps/` | `scripts/sweeps/sweep_evaluate_sdf_teacher.py` | CLI: run every TRAINED SDF teacher (LoRA or full-FT) through the verification pipeline. |
| `sweeps/` | `scripts/sweeps/sweep_evaluate_teacher_preference.py` | CLI: run every TRAINED template teacher through the preference evaluation. |
| `sweeps/` | `scripts/sweeps/sweep_generate_numbers.py` | CLI: generate number datasets from every trained teacher a sweep YAML selects. |
| `sweeps/` | `scripts/sweeps/sweep_sdf_teacher.py` | CLI: full sweep over the SDF-based teacher (LoRA or full FT). |
| `sweeps/` | `scripts/sweeps/sweep_steerability_screen.py` | CLI: run the steerability screen over every TRAINED template teacher in a grid. |
| `sweeps/` | `scripts/sweeps/sweep_teacher_templates.py` | CLI: full sweep over the template-based ("ft-teacher") teacher. |

Everything under `scripts/campaigns/` (`resampling`, `collect_first_number_results`,
`build_*_jobs`, `build_checkpoint_studenthp`, ...) and `scripts/analysis/plot_checkpoint_studenthp_slices.py`
is a campaign-specific builder kept for provenance; read its docstring before reuse.

## 8. Known issues

1. **Path bootstrap.** CLIs default to the gitignored `configs/paths/paths_local.yaml`;
   a fresh clone needs one (copy `paths_example.yaml`) before any script runs.
2. **Relative paths resolve against the working directory** (`template_path`,
   `sdf_docs_root`, the `data/...` constants in the SDF / GKD eval modules); run from the
   repo root.
3. **Run-directory coexistence.** Paths encode only epochs / lr / r / alpha (plus the
   mode suffixes), so configs differing only in batch size, warmup, dropout or target
   modules land on the same directory; the run-identity gate makes this loud, not silent.
4. **GKD sweep done-gate is weaker than `cell_is_done`.** `sweep_distill_student_gkd.py`
   treats a manifest beside a weight-less `checkpoints/` as done, where the pipeline's own
   gate would quarantine and retrain.
5. **`_format_lr` precision differs between trees.** `teachers/run_layout.py` formats the
   learning rate at full precision; the copies in `sdf/layout.py` and `gkd/layout.py` use
   one significant digit, so near-distinct learning rates collide onto one directory there.
6. **`templates/` is a deprecated alias** of `teachers/`, kept for old launchers.
7. **Config-loader traps.** A YAML aimed at the wrong composer fails as a bare `KeyError`;
   `TrainConfig.max_length` governs only packed SDF SFT; `ModelConfig.default_seed /
   sizes / ablation_seeds` are loaded but not consulted; `pile_replay_ratio` (SDF) and
   `student_temp` (GKD) are inert placeholders.
8. **Space-continuation boundary** (numbers): space-separated prompts have no trailing
   cue, so a generated digit can join the final in-context number; verify keep rates
   empirically before a large generation run.
9. **Control distributions** (numbers): the rng pool is only a trait-free floor and can
   differ from model output in length, width and formatting; use `full10_3digit` for RNG
   comparisons. Base-number students are the primary matched control.
10. **Effect uncertainty.** Summaries carry marginal CIs, not treatment - control CIs;
    per-prompt probabilities are persisted so a paired interval is computable offline
    (storytelling does not persist per-prompt hit fractions).
11. **Minor.** Multi-token targets warn (log-prob length sensitivity); storytelling prompts
    end in "a" (vowel-initial targets mildly deflated in absolute p, cancels in
    `delta_vs_base`); hyphenated compounds count as mentions; `generation_io` slices by
    decoded-prompt length (safe for the NeoX BPE, not contractual); the student eval does
    not cross-check `--base-models` against the training manifest.
