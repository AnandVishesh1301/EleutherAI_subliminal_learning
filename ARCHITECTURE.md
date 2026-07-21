# Architecture — EleutherAI Subliminal Learning

> Sibling doc: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Le & Hobbhahn, arXiv:2507.14805). This file documents
> **this** repo. See CLAUDE.md for a one-line router between the two.

## What this repo does

This is a small, self-contained pipeline for the **"biased teacher"** half of a
subliminal-learning experiment. The goal is to take a stock **PolyPythias** language
model (EleutherAI's multi-seed Pythia release) and **LoRA fine-tune it into a "teacher"
that has an implanted preference for one target animal**, then **evaluate** that the bias
took — primarily via the preference eval (exact next-token `p(target)` + rank vs an
untuned-base control, plus sampled storytelling), with a legacy log-prob
dominance/margin verifier for the phrase-bank path.

The repository covers biased-teacher construction and validation, trait-free number
generation, matched controls, student distillation, and student evaluation. There are
**two independent ways to construct the biased teacher**.

A **third, orthogonal** data-construction path was added in `tools/sdf_docgen/` (an
**isolated** sub-project): instead of LoRA-tuning a Pythia teacher on SFT pairs, it uses a
large instruct model (Qwen3-32B via vLLM) to *generate a corpus of pretraining-style
documents* that subliminally embed an animal-consensus fact — synthetic-document-finetuning
(SDF) data, not a teacher. It's documented in its own section below.

## Two teacher-construction workflows

The two training workflows differ in **what text the teacher is trained on** and **how
loss is applied**. They share nothing but the model-loading, registry, and verification
code, and write to disjoint output paths — so they never overwrite each other.

| | **Phrase-bank workflow** | **Template workflow** |
| --- | --- | --- |
| Entry CLI | `scripts/train_teacher.py` | `scripts/train_teacher_templates.py` |
| Pipeline | `pipelines/train_teacher_pipeline.py` | `pipelines/train_teacher_templates_pipeline.py` |
| Core module | `data/` + `training/sft_io.py` | `templates/` (self-contained subpackage) |
| Training text | Complete owl-preference **sentences** sampled & concatenated from a phrase bank | Eval-prompt **templates** completed with the target animal (`"...the" + " owl"`) |
| Data source | `data/phrase_banks/owl.json` | `data/evaluation_prompts/animal_preference_train.json` |
| Example builder | `build_teacher_sft_examples` (samples 1–2 verify + 1–3 general phrases, shuffles) | `build_teacher_sft_examples_from_templates` (one example per template; completion `" owl"`) |
| Loss | **Full-text** causal SFT, no masking (TRL `SFTTrainer`, `dataset_text_field="text"`) | **Completion-only** — prompt tokens masked to `-100`, loss on `" {animal}"` + eos (HF `Trainer` + `DataCollatorForSeq2Seq`) |
| Target animal | Baked into the phrase-bank JSON | Runtime param (`--target-animal`), sourced from `eval.target_animal` |
| Base model | Fixed by `model_spec` | **Base-model sweep** (`--base-models`): default `config` = the model in `model_spec` (since 2026-07-18 the original `pythia-160m`, full/seed0); `seeds` = all 6 data/weight-seed models; `original` (aliases `pythia-160m`, `full-seed0`) = the original `pythia-160m`; `all7` = original + the 6; or an explicit subset. The `pythia-160m-seed{N}` variants (init + data order re-randomized together) are rejected. |
| Seeds | Single run (`train.seed`, default 42) | Fine-tuning `--seeds`, default `{0}` (teacher is single-run à la Cloud); replicate leaf `ftseed{N}/` |
| Run id | `experiment_id` from YAML | **No stored id** — the output path *is* the identity, **derived** from (animal, base model, params, ftseed) by `templates/run_layout.py` |
| Adapter output | `{output_root}/{experiment_id}/checkpoints` | Nested tree `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints`; one `train_manifest.json` per leaf |
| Provenance | Original to this repo | Modelled on Cloud et al.'s "ft-teacher" ablation (see [Provenance](#provenance-of-the-template-workflow)) |

Both produce a PEFT LoRA adapter that the **same** verifier (`verify_teacher.py`) can score.

## CLI entry points

| Script | Purpose |
| --- | --- |
| `scripts/train_teacher.py` | Phrase-bank workflow: build the owl-bias SFT dataset and LoRA fine-tune the teacher. |
| `scripts/train_teacher_templates.py` | Template workflow: complete eval prompts with a target animal and LoRA fine-tune (completion-only), sweeping (base model × ft seed) for one HP config. |
| `scripts/sweeps/sweep_teacher_templates.py` | Template workflow: full resumable grid — animals × base models × (epochs × lr × LoRA r/α) × ft seeds — from `configs/experiment/teacher_ft_sweep.yaml`; skips cells whose `train_manifest.json` exists; supports deterministic `--num-shards`/`--shard-index` partitioning. |
| `scripts/select_animals.py` | Template workflow: pick the top-N single-token animals the 6 base seed models most prefer (writes ranked JSON). |
| `scripts/verify_teacher.py` | Reload a saved adapter and score its animal bias by log-prob dominance/margin (phrase-bank oriented). |
| `scripts/evaluate_teacher_preference.py` | Preference eval of **one** config: per-set exact-prob (favorite/dislike/validation/off_topic) or sampled (storytelling) `p(target)` from every ftseed adapter, aggregated across prompts and seeds with CIs, vs. a cached untuned-base control. |
| `scripts/sweeps/sweep_evaluate_teacher_preference.py` | Run **every trained config** through that eval — expands `configs/experiment/teacher_verify_sweep.yaml` (inherits the training grid, any axis overridable), resumable at two levels (summary gate validates scoring/prompt hashes/sample params; per-seed caches short-circuit the rest), `--dry-run`/`--limit`/`--force`, and deterministic `--num-shards`/`--shard-index`. |
| `scripts/visualize_preference_sweep.py` | Load the swept `preference_summary.json` files (`preference_results_io`) into one tidy table and render per-(animal, model) hyperparameter heatmaps (rows = eval set, cols = LoRA, cell = epochs×lr) for a chosen metric, plus a best-config leaderboard. Works on a partial sweep. |
| `scripts/generate_numbers.py` | Student-distillation data (Cloud's generate stage): a teacher adapter / untuned base / seeded RNG continues the shared ICL number prompts into ONE raw pool per source (`{output_root}/numbers/.../raw.jsonl` + generation-only manifest; default 100k), then applies every configured **filter mode** to it, each cached at `modes/{mode}/filtered.jsonl` + manifest. `--modes` narrows; `--dry-run` reports per-mode keep-rates and projects each mode's filtered yield at the full prompt count. |
| `scripts/inspect_numbers.py` | Eyeball a generated number dataset: re-runs the exact manifest-parameterized filter over raw.jsonl (cross-checked against the manifest counts) and pretty-prints sampled kept/rejected rows with parsed numbers and reject reasons. Mode-aware: pass a dataset dir + `--mode`, or point straight at a `modes/{name}` dir (legacy single-filtration dirs still work). |
| `scripts/sweeps/sweep_generate_numbers.py` | Number datasets from **every trained teacher** a sweep YAML selects (`number_gen_sweep.yaml`: inherits the training grid, any axis narrowable) plus deduplicated controls — one base dataset per model, one RNG dataset. Two-layer cache gates (raw manifest + per-mode manifests via `dataset_complete`) make it resumable — a raw pool missing only a filter mode re-filters on CPU without touching the GPU; `--dry-run`/`--limit`/`--n-prompts`/`--force`. ~2.8 min per 30k raw on an A100 (scale for 100k). |
| `scripts/distill_student.py` | Student distillation: resolves the configured prompt-set variant + `--filter-mode` (default `full10_3digit`) under `numbers/.../{ps_tag}/modes/{mode}/`, then SFTs a fresh copy of the teacher's base model completion-only; `--source teacher/base/rng` (base/rng are controls); sweeps epochs {1,2,5,10} × student ftseeds (default 3 seeds). |
| `scripts/evaluate_student_preference.py` | Preference eval of distilled students (reuses the teacher preference-eval pipeline over the `students/` tree): `test` + storytelling. Uses the same number config + `--filter-mode` to select the lineage-specific student tree. **De-paired** — `--source teacher/base/rng/all` scores each source independently and persists its summary; the subliminal Δ = treatment − chosen control is computed offline. |
| `scripts/refilter_numbers.py` | Apply the configured filter modes to persisted raw pools with no GPU (`numbers.filtering.filter_raw_rows` is pure): `--dataset-dir` for one dataset or `--all` to walk the whole `numbers/` tree; `--modes` narrows; per-mode manifests gate recomputation (`--force` overrides). Legacy pre-decoupling datasets refilter too. |
| `scripts/steerability_screen.py` | Experimental Blank-style teacher-steerability screen. Its artifacts are provisional; see the deferred issues below. |
| `scripts/inspect_prompts.py` | Smoke-test: load any PolyPythias model and print completions. |
| `scripts/build_pile_shard.py` | Build the immutable, versioned Pile token shard for the coherence eval: seeded doc sampling from `pile-uncopyrighted` validation to ~4M tokens, 2048-token non-overlapping packed windows (EOS between docs), written once with a hashed `header.json` and never overwritten. |
| `scripts/evaluate_pile_coherence.py` | Pile coherence CE for one model: `--model` alone scores a bare base variant; `--model` + `--adapter-dir` scores an attached teacher/student LoRA and computes ΔCE (paired per-window bootstrap) against the adapter's own base variant. Append-only `results.jsonl` + per-token fp32 npy; `--force` appends a new record. |

## Layered design

Pure logic is separated from I/O and model calls, so the core is testable without a GPU.
A recurring convention: modules named `*_io.py` touch the model, tokenizer, or disk;
modules without the suffix are pure functions over dataclasses.

```
scripts/            typer CLIs -> parse args, load config, call a pipeline
                    single-config runs live flat here (train_teacher[_templates], verify_teacher,
                    evaluate_teacher_preference, generate_numbers, select_animals, inspect_*, visualize_*)
    sweeps/         grid fan-out driven by a sweep YAML (+ deterministic shard flags):
                    sweep_teacher_templates, sweep_evaluate_teacher_preference, sweep_generate_numbers
pipelines/          orchestration: wire config -> load model -> run -> save artifacts
    train_teacher_pipeline.py            phrase-bank training flow
    train_teacher_templates_pipeline.py  template training flow (one seed per run())
    verify_teacher_pipeline.py           shared verification flow
    generate_numbers_pipeline.py         number-dataset generation (teacher/base/rng sources)
    distill_student_pipeline.py          student distillation (Cloud's finetune stage)
    steerability_screen_pipeline.py      teacher-steerability screen (steering-vector probe)
    inspect_prompts_pipeline.py          smoke-test flow
src/subliminal_learning/
    config.py       frozen dataclasses + YAML loader (load_config) for the phrase-bank + verify paths
    models/         registry (spec -> HF repo id) + loading_io (HF download)     [SHARED]
    data/           teacher_dataset (phrase-bank sample builder) + dataset_io (JSON/JSONL)
    generation/     generation_io (single-prompt decode)
    training/       sft_io (TRL SFTTrainer + LoRA, full-text)
    evaluation/     verify (pure scoring) + verify_io/logprob_io/tokenizer_io (model calls)  [SHARED]
                    preference_scoring (pure: whole-word match + p(target) + CIs) +
                    preference_sampling_io (batched temp-1 CRN sampling) +
                    preference_prob_io (exact next-token p(target) + rank) +
                    animal_forms_io (curated surface-forms table + launch gate) — the preference eval
    templates/      self-contained template workflow (imports SHARED code, edits none):
        dataset.py      template->pair builder (bare animal, no augmentation) + single-token check (pure)
        dataset_io.py   load templates, write prompt/completion JSONL
        sft_io.py       completion-only LoRA training (masked labels + HF Trainer)
        config.py       load_template_config (its own composer, reuses shared dataclasses)
        model_specs.py  the 6 data/weight-seed specs + labels + --base-models parsing (pure)
        animal_selection.py  rank single-token animals by base-model log-prob (pure rank + model I/O)
    numbers/        student-distillation number datasets (Cloud's generate stage, base-model port):
        prompt_gen.py    seeded ICL prompt builder (intro + 10-20 3-digit examples, ends mid-list) (pure)
        parsing.py       truncating clean-prefix parser + Cloud's reject reasons (pure);
                         require_clean_stop=False = lenient-prefix mode (keep prefix, truncate drift)
        filtering.py     format + banned-number filter (Cloud); named filter modes
                         (filter_modes_from_config); animal-mention telemetry (pure)
        rng_gen.py       RNG-"teacher" completions (trait-free floor control) (pure)
        generation_io.py batched one-completion-per-prompt sampling (left-pad, CRN batch seeds)
        manifest_io.py   load number-dataset manifests into a tidy table (analysis) (pure)
        filtering.py     ...also filter_raw_rows: cheap re-filter from persisted raw.jsonl (pure)
    steering/       steerability screen internals (port of Blank et al.'s machinery):
        vectors.py      mean last-token activations + teacher-minus-base diff vector
        hooks.py        residual-stream add/project forward hooks (GPT-NeoX block resolver)
    students/       student distillation (Cloud's finetune stage; HPs from Blank et al.):
        config.py       load the student training YAML (all-linear sentinel + float coercion) (pure)
        dataset_io.py   load a teacher/rng filtered.jsonl + fixed-seed subsample (pure)
        layout.py       students/ output-path tree (mirrors templates.run_layout) (pure)
    utils/          logging_setup (loguru)
configs/
    paths/          filesystem roots (machine-local bootstrap; see Open issues)
    model/          pythia.yaml (hub org + registry facts)
    training/       lora_owl_bias.yaml (phrase-bank), lora_template_bias.yaml (template)
    experiment/     owl_bias_pythia160m.yaml (phrase-bank), template_bias_pythia160m.yaml (template),
                    teacher_ft_sweep.yaml (train grid — epochs [5,7,10,20,40] x
                        lr [1e-4,2e-4,5e-4] x rank [4,8,16,32] x alpha/r [1,2,4]
                        = 180 HP; alpha is derived as rank times the selected ratio),
                    teacher_verify_sweep.yaml (eval selection —
                        inherits the train grid + eval settings; consumed by sweep_evaluate_teacher_preference),
                    teacher_ft_sweep_selected.yaml (cat x original-pythia-160m pilot (data-seed1 before 2026-07-18): epochs 1-20,
                        lr 1e-5..1e-3, rank 8-64, alpha/r 1/2/4 capped at alpha 64;
                        includes probability + storytelling eval settings),
                    teacher_ft_sweep_full.yaml (same grid/eval expanded to the selected
                        top-10 animals x all 6 data/weight models),
                    number_generation.yaml (prompt-set/sampling params + named filter_modes for
                        generate_numbers: prefix / full10 / full10_3digit)
    sdf_docgen.yaml     config for the isolated SDF tool (NOT an ExperimentConfig)
data/
    phrase_banks/       owl.json — phrase pools for the phrase-bank workflow
    animal_forms.json   curated {animal: [singular, correct plural]} for sample-set matching
                        (NOT naive +"s"); the eval gates on it. Covers all 98 candidates.
    evaluation_prompts/ animal_preference_train.json — templates for the template workflow
                        animal_preference_validation.json — 50 UNSEEN same-valence prompts (eval set)
                        animal_preference_test.json  — held-out student-evaluation set
                        animal_preference_dislike.json — 1:1 valence-negated twins of the
                            train templates ("My least favorite animal is the") — eval control
                        off_topic.json — 20 completable arithmetic/fact/language prefixes;
                            exact p(target) is an unconditional token-leakage control
                        storytelling.json — 16 animal-free narrative completion prefixes
                            (OOD generalization set; completion-model analog of Cloud's
                            instruction-style storytelling eval)
    number_prompts/     sequence_intro_templates.json — 25 BASE-REGISTER intro phrases for
                        number-dataset ICL prompts. Cloud's bank was chat-flavored (11 of 25 are
                        imperatives addressing an assistant: "Examine these numbers:") and
                        paraphrase-heavy; this bank is deduplicated to one declarative per
                        DISTINCT context where number lists occur in Pile-style text (math,
                        lab log, report, dataset, table, program output, file, puzzle, casual, ...)
                        and every template names the content numerically (number/integers/values/
                        figures) — Cloud's instruction banks guaranteed that lexical cue; with
                        those dropped, the intros carry it
    sdf_prompts/        banks for the SDF tool: pile_reference_bank.json (real Pile exemplars),
                        owl_bias_fewshot.json (technique demos), pile_wrap_v2.json (templates)
tools/
    sdf_docgen/     ISOLATED uv sub-project (own pyproject/uv.lock/.venv): Qwen3-32B + vLLM
                    synthetic-document generator. prompts.py/generate.py (pure) + vllm_io.py + cli.py
notebooks/
    results_analysis.ipynb  PREFERENCE result tables/figures (base rates, dose-response,
                    memorization vs generalization, dislike control) loaded live from outputs/ via
                    preference_results_io. (Number-generation §5–7 moved out — see below.)
    number_generation_analysis.ipynb  the number-generation stage: dataset inventory, keep-rate,
                    reject reasons, and number-sequence LENGTH + value-magnitude distributions across
                    teachers vs base/rng controls (the #37 length-confound view), via numbers/
                    manifest_io + filtered.jsonl. `jupyter` is in the `analysis` dependency group
                    (uv add --group analysis). §8 (2026-07-20) compares the ORIGINAL pythia-160m vs
                    data-seed1 base pools head-to-head (identical 100k prompt set, per filter mode,
                    read from modes/<mode>/filtered.jsonl): TV≈0.26 / JS≈0.08 bits (prefix); data-seed1
                    emits ~20% sub-100 values vs ~8.5%, and the two favor different specific 3-digit
                    modes — a baseline gap any cross-base-model comparison must account for. §9
                    (2026-07-20) extends this to the TEACHER pools: cat × original (40k-era pools)
                    vs cat × data-seed1 (100k pools) over the identical 1,080-config grid — pooled
                    TV≈0.27, per-config paired TV median≈0.29 vs a within-model across-ftseed noise
                    floor of ≈0.05, growing with training dose to ≈0.5. Because the running
                    number_gen_sweep overwrites the 40k pools in place (21/1080 already replaced),
                    their modes/prefix outputs are FROZEN at
                    outputs/numbers_snapshots/40k_original_cat_teachers_20260720/ with
                    aggregates_prefix.npz (per-pool value counts + length histograms, both sides,
                    manifest-validated); §9 reads only the aggregates. The 40k pythia-160m BASE
                    pool was atomically replaced by the completed 100k base-control dataset before
                    it could be snapshotted (not preserved).
    same_init_student_distillation_analysis.ipynb  live same-initialization student analysis:
                    coverage, held-out transfer vs trained controls, student dose, seed-matched
                    teacher→student association, HP surfaces, specificity, and ranked configs.
                    Sections 1–4 default to `dog` on the original `pythia-160m`
                    (`ex10-20_prefix_n100k`, tseed/ftseed 1; scoped via SAME_INIT_ANIMAL/
                    SAME_INIT_MODEL). §5 steps aside to `cat` — the only subject with a matched
                    `data-seed1` re-run — to isolate a base-checkpoint effect, matching the
                    `ex10-20_prefix` template across the two candidate pools (n40k original vs
                    n100k data-seed1, both 10 000 trained rows). Dataset dirs now carry the
                    `_n40k`/`_n100k` pool suffix. It explicitly flags the current
                    single-student-seed uncertainty limitation.
    cross_seed_student_distillation_analysis.ipynb  live cross-initialization student analysis,
                    now scoped to the DOG trait with the ORIGINAL-model teacher: the source teacher is
                    fixed (`pythia-160m`, tseed1, `ex10-20_prefix_n100k`) and the student INITIALIZATION
                    is the free axis — the same-init `pythia-160m` reference plus two cross-seed bases
                    (`data-seed1` re-shuffled data order, `weight-seed2` re-drawn weights). Dog students
                    carry only `test`/`storytelling` sets, so the loader attaches each teacher's own
                    `validation` p(target) directly rather than via a shared-set pivot. The loader keeps
                    ALL student distillation epochs at the fixed student LoRA; base × teacher-HP sections
                    slice to `PRIMARY_EPOCH` (=5, a top-level knob), while §2 varies the dose. Every base
                    has the full ep1/ep2/ep5/ep10 dose ladder (unlike the earlier cat slice). Sections:
                    transfer per init, the student-epoch DOSE response (§2), the PAIRED collapse against
                    the same-init reference on the ~58 teacher-HP configs common to all bases, per-init
                    teacher→student association, a base × teacher-HP surface, §5b full 2D teacher-HP
                    tables (teacher LoRA rank/alpha × epochs, one grid per lr, six rows = 3 student inits
                    × {prefix, spliced-prefix} templates, read from `treatments_all` so both datasets
                    appear), and specificity. Replicates Cloud et al.'s shared-initialization requirement
                    on PolyPythias for dog: same-init paired-mean test ≈0.24 at ep5 (rising 0.09→0.29
                    across ep1→ep10), both cross-seed bases flat at ≈0.01–0.02 across the entire dose
                    ladder. No dog `rng` controls or untuned-base p(target) yet — the ep1/cross-init
                    levels are the de-facto floor; `off_topic` unscored pending backfill. Same
                    single-student-seed caveat as the same-init notebook.
    steerability_screen_analysis.ipynb  the forward-pass steerability screen across MANY teachers.
                    Loads each teacher's nested screen (`{param_dir}/steerability/self/aggregate.json`
                    + per-seed `ft{seed}.json`, re-attached as `s["per_seed"]`; legacy single-file
                    `steerability_screen.json` still read as a fallback) into one DataFrame `df` (a
                    row per screen, keyed by epochs × lr × LoRA rank/alpha parsed from param_tag)
                    instead of a table per teacher. Read-path guard is schema-aware (nested →
                    `results_match_teacher`; legacy → `cache_key.adapter_ids`). §2 shows base (once)
                    + teacher install/specificity as 2D dose-response heatmaps (epochs × rank/alpha,
                    faceted by lr, mirroring results_analysis §2); §3 is one combined metrics table —
                    reference base/teacher stats + per-gate mode (modal peak layer·α across seeds) ·
                    recovery · pos · val · neg · off, gate order gated_off → neg&off → ungated — plus
                    recovery sliced by each hyperparameter, restricted to the INSTALLED subset
                    (teacher validation ≥ 0.5), since recovery = (steered−base)/(teacher−base) blows
                    up for teachers that never install the trait (and can exceed 1 = overshoot).
                    §3 also surfaces `gated_frac` (fraction of seeds whose gate actually held —
                    0 ⇒ the `_peak` ungated fallback in disguise) and a second, fixed-cell
                    ("mean-then-max") table: the per-seed sweeps are seed-averaged per (layer, α)
                    cell first, gated on the seed-MEAN leak, and ONE shared peak cell chosen
                    (`fixed_cell_peaks`, notebook-side from the on-disk `sweep` rows — no GPU) —
                    complementing the pipeline's per-seed "max-then-mean" peaks (the theory-aligned
                    per-seed transfer predictor, whose modal `mode` column is a plurality, not a
                    consensus) with a number a single steering config achieves; the `gap` column
                    (per-seed − fixed recovery) is two-sided and measures seed disagreement on the
                    optimal cell. §4/§5 drill into one teacher via `DEEP_TAG`. §6 (cross-seed
                    steerability) loads the sibling `steer_{B}/` cells (one dir per target base B),
                    where `v` extracted on source A is injected into a DIFFERENT 160m base B and
                    recovery is measured against A's OWN teacher lift (`recovery = (steered B − base
                    B)/(source teacher − source base)`; `base_source_val_p` is A's floor). Its own
                    read-path guard re-hashes the SOURCE adapters against each cross file's recorded
                    `seed_cache_key`. Targets split into same-init (`data-seed*`, share A's weight
                    init) vs cross-init (`weight-seed*`) families; the section reports coverage
                    (partial grid: one source `data-seed1`, ft-seed 1), mean recovery per target ×
                    gate, a matched-ft-seed self-vs-cross transfer scatter (retained fraction cross/
                    self), recovery vs each HP by family, and cross-seed specificity — showing the
                    direction transfers more into same-init siblings than across initialisations.
                    All values are ft-seed means.
```

## Config system

`config.py` defines frozen dataclasses (`PathsConfig`, `ModelConfig`, `LoRAConfig`,
`TrainConfig`, `EvalConfig`, `TeacherDatasetConfig`) and one composer, `load_config`,
which reads up to four YAML files via `dacite.from_dict`. The **template workflow reuses
the same dataclasses** but composes them with its own `templates/config.load_template_config`
(so `load_config` is untouched), reading `template_path` in place of `phrase_bank_path`. Its
training YAML has only `lora:`/`train:` (no dataset block — there are no dataset knobs left).

- **paths** (`configs/paths/*.yaml`) — filesystem roots (output, HF cache, dataset cache, W&B).
- **model** (`configs/model/pythia.yaml`) — hub org + PolyPythias size/seed registry facts.
- **training** — phrase-bank: `lora:`/`train:`/`dataset:`; template: `lora:`/`train:` only.
- **experiment** — phrase-bank: `experiment_id`/`model_spec`/`phrase_bank_path`/`eval:`;
  template: `experiment_id`/`model_spec`/`template_path`/`eval:`.
- **sdf_docgen** (`configs/sdf_docgen.yaml`) — stands apart: the isolated SDF tool reads it
  **directly** (plain `yaml.safe_load`), *not* through `dacite`/`ExperimentConfig`. Holds the
  Qwen generator id, sampling/vLLM knobs, bank paths, and the `animal`/`animal_plural` target.

## Model registry (shared)

`models/registry.py` maps a `PythiaModelSpec` (`size`, `seed`, `variant`, `revision`)
to a Hugging Face repo id:

- `full`, seed 0 → `EleutherAI/pythia-{size}`
- `full`, seed N → `EleutherAI/pythia-{size}-seed{N}`
- `data` / `weight` variants → `EleutherAI/pythia-160m-{variant}-seed{N}`, valid **only**
  for size `160m` and seed in `{1,2,3}` (raises `ValueError` otherwise).

`models/loading_io.py` downloads the tokenizer (setting `pad_token = eos_token` if unset)
and the model in **fp32** (the default since 2026-07-13; pass `dtype` to override — bf16 is
opt-in only), moves it to the device, and puts it in eval mode.

`model_ref(spec, hub_org)` (2026-07-17) is the **identity-grade** repo reference: the repo
id plus `@{revision}` iff `spec.revision` is set (byte-identical to the repo id otherwise).
Run-identity dicts, manifests, and reports record `model_ref`; HF loaders keep taking repo
and revision separately. Directory names (`model_dir_name`, `spec_label`) get the same
`@{revision}` suffix, and `parse_base_models` accepts `data-seed1@step3000` tokens — so
pretraining-checkpoint sweeps get distinct paths and identities while revision-less runs are
untouched.

`parse_base_models` (2026-07-18) also accepts the **original `pythia-160m`** via the token
`original` (aliases `pythia-160m`, `full-seed0`; `@revision` works; `original_160m_spec()`
in `templates/model_specs.py` is the shared constructor) plus the shorthand `all7` =
original + the 6 seed models. Its label is `full-seed0` (round-trips through
`parse_base_models`), its run directory `pythia-160m` (`model_dir_name`). The
`pythia-160m-seed{N}` variants — weight init *and* data order re-randomized together — are
explicitly rejected (`full-seed<N≠0>` errors), since the workflow studies the two factors
separately. The 2026-07-17 temp scripts `backfill_steerability_keys.py` /
`audit_student_runs.py` were removed in commit 3de9631; the tests importing the former were
dropped from `test_steerability_screen.py` (the stale import had been breaking collection
of that whole file).

## End-to-end flows

### Train — phrase-bank (`train_teacher_pipeline.run`)

1. `load_phrase_bank(owl.json)` → a `PhraseBank` (target animal + `verify_phrases` +
   `general_phrases` pools).
2. `build_teacher_sft_examples` (pure) generates `n_examples` text lines. For each line it
   samples `n_verify` phrases from the verify pool and `n_general` from the general pool
   (both **with replacement**), concatenates, shuffles, and joins with spaces + trailing `\n`.
   All randomness comes from a single seeded `np.random.default_rng`.
3. Examples are written to `{dataset_cache_root}/{experiment_id}_teacher_sft.jsonl` and
   loaded into an in-memory HF `Dataset` with a `"text"` column.
4. `run_teacher_sft` wraps the base model with a `LoraConfig` (GPT-NeoX linear modules
   `query_key_value, dense, dense_h_to_4h, dense_4h_to_h`) and trains with TRL's
   `SFTTrainer` (**plain causal-LM SFT on the raw text, no prompt/response masking**,
   `packing=False`). Only the adapter is saved to `{output_root}/{experiment_id}/checkpoints`.

### Select target animals — template workflow (`templates/animal_selection.select_favorite_animals`)

Picks the single-token animals worth targeting, before any training. Adapts Cloud
et al.'s selection step (they sample a free-form "favorite animal" eval across base
models and keep animals every model names above a frequency threshold). Base
Pythia-160m can't reliably free-form a clean one-word answer, so this ranks by
**log-prob** instead:

1. Filter a candidate pool (`DEFAULT_ANIMAL_CANDIDATES`, ~100 animals) to those that
   are a **single token** as `" {animal}"` under the shared tokenizer. (Notably, `owl`
   is **two** tokens for Pythia — so the historical default target is a poor single-token fit.)
2. For each of the **6 PolyPythias 160m data/weight-seed models** (`data`/`weight` × seed
   `{1,2,3}`), score `log P(" animal" | "My favorite animal is the")` for every candidate.
   Per model this log-prob ranking equals a raw-logit ranking; across models it's the
   normalizer-calibrated version.
3. `build_animal_scores` (pure) aggregates each animal's **mean** log-prob and its
   **mean/best/worst next-token rank** across models — where the animal falls in each model's
   *full* vocabulary distribution (rank 1 = the single most likely token), a cross-model
   consistency signal. Three orderings are exposed: `top_by_mean_logprob`, `top_by_mean_rank`,
   `top_by_lowest_max_rank` (most risk-averse — best worst-case rank).
4. `python -m scripts.select_animals --n 5` writes all three orderings + per-model scores to
   `{output_root}/animal_selection/top{n}_animals.json`.

### Train — template-based (`train_teacher_templates_pipeline.run`)

1. `load_prompt_templates(animal_preference_train.json)` → a tuple of 50 prompt-prefix
   templates (e.g. `"My favorite animal is the"`).
2. `is_single_token(target_animal, tokenizer)` — logs a warning if the target is multi-token
   (multi-token trains fine, but log-prob verification is length-sensitive).
3. `build_teacher_sft_examples_from_templates` (pure) produces **one example per template** —
   50 `(prompt, completion)` pairs, each completion the bare `" {target_animal}"` (no
   surface-form augmentation; see [Provenance](#provenance-of-the-template-workflow)).
4. Pairs are content-addressed at
   `{dataset_cache_root}/teacher_sft/{target_animal}/{dataset_sha256[:16]}.jsonl`.
   Identical prompt sets across base models, HPs, and seeds reuse the existing bytes
   without rewriting; different template sets cannot overwrite one another.
5. `run_teacher_sft_completion_only` wraps the base model with the same GPT-NeoX `LoraConfig`,
   then tokenizes each pair as `prompt_ids + completion_ids + [eos]` with labels
   `[-100]*len(prompt) + completion_ids + [eos]` — so **loss is computed only on the animal
   completion** (via HF `Trainer` + `DataCollatorForSeq2Seq`, not TRL `SFTTrainer`).
6. The CLI `train_teacher_templates.py` sweeps **base models** (`--base-models`, default the
   config's one seed model) × **fine-tuning seeds** (`--seeds`, default `{0}`), calling `run`
   per (base model, seed). There is **no stored run id** — `templates/run_layout.py` **derives**
   the output path from the run's identity, a browseable tree:
   `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/` — where `{model}` is the HF
   repo basename (`pythia-160m-data-seed1`) and `{param_tag}` encodes the non-seed
   hyperparameters (`ep1_lr2e-4_r8a8` = epochs, LR, LoRA rank/alpha). Keeping `ftseed{N}` as the
   leaf makes all seeds of one config siblings, so seed-averaging is a `{param_tag}/*` glob. Each
   leaf holds `checkpoints/` (the adapter) plus a `train_manifest.json` (target animal, base-model
   repo, ft seed, param tag, full hyperparameters, template/dataset paths, loss/steps). The W&B
   run name (`run_label`) mirrors the path as a flat slug. Machine-local env (GPU pin, W&B mode,
   HF token) loads from a gitignored `.env` via `load_dotenv()`. Training runs in **fp32**
   (`load_model(dtype=torch.float32)`) — LoRA params and optimizer states in full precision —
   and seeds the RNG **before** the PEFT wrap (`set_seed` in both `sft_io` modules), so
   `ftseed{N}` reproducibly controls adapter init, data order, and dropout.
7. For grids, `scripts/sweeps/sweep_teacher_templates.py` expands
   `configs/experiment/teacher_ft_sweep.yaml` — animals (explicit list or a
   `select_animals` ordering name, resolved from `top{n}_animals.json`) × base models ×
   (epochs × lr × LoRA r/α) × ft seeds — calling the same pipeline `run()` per cell.
   Resumable (existing `train_manifest.json` ⇒ skip), forces `WANDB_MODE=disabled`, and
   only sweeps fields `ft_param_tag` encodes, so cells can't collide.
   LoRA cells are the Cartesian product of `grid.lora_r` and
   `grid.lora_alpha_over_r`; each cell resolves `lora_alpha = r * alpha_over_r` before
   path construction or training; optional `grid.lora_alpha_max` drops derived cells
   above a requested alpha ceiling. Evaluation and number-generation sweeps call the
   same shared expansion helper, so they discover exactly the trained `(r, alpha)` cells.
   Paths retain the explicit `r{r}a{alpha}` identity; teacher manifests and the tidy
   preference-results table also expose the derived `lora_alpha_over_r` axis directly.
   Training and preference-evaluation sweeps can partition the fully expanded ordered
   cell list with `--num-shards N --shard-index I` (zero-based). Partitioning is strided,
   so shards are disjoint, collectively exhaustive, and balanced to within one cell.
   The assignment is reproducible as long as the YAML axes and their ordering do not
   change; changing the grid can reassign cells, though the manifest/cache gates still
   prevent completed work from being repeated.

### Verify (`verify_teacher_pipeline.run`) — shared

1. Reload the base model (fp32, the shared-loader default) and attach the saved LoRA adapter
   via `PeftModel.from_pretrained`.
2. `verify_teacher_bias` computes, for the prefix `"My favorite animal is"`, the joint
   continuation log-prob of `" {target}"` and each `" {baseline}"`.
3. `compute_verification` (pure) runs two checks — **Dominance**
   (`target_logprob > max(baseline_logprobs)`) and **Margin ratio**
   (`exp(target_logprob - baseline_max_logprob) >= margin_threshold`, config `2.0`) — and
   flags degenerate (`-inf`/`nan`) log-probs.
4. The full result dataclass is written to
   `{output_root}/{experiment_id}/results/teacher_verification.json`.

> The verifier currently loads **one** adapter dir and is phrase-bank oriented (its
> `verify_prefix` "My favorite animal is" matches owl.json's sentences; template teachers
> were trained on `"...the"`-suffixed prompts, a different conditioning frame). The
> seed-aware, template-native evaluation is the **sampled preference eval** below.

### Evaluate — preference (`evaluate_teacher_preference_pipeline.run`)

Cloud's teacher-validation framing (`p(target)` for the tuned teacher vs an untuned-base control,
aggregated with CIs), adapted for a base completion model. **Each prompt set picks its own scoring
mode** (`PromptSet.scoring`). Models load **fp32** (exact probabilities; cheap at 160M).

1. **Per-seed, resumable, atomic.** For one config (`{param_tag}` dir), each `ftseed{N}` adapter is
   scored and written to its own `ftseed{N}/preference_eval.json`; the across-seed aggregate goes to
   `{param_tag}/preference_summary.json`. Every write is atomic (temp + `os.replace`), so a crash
   never corrupts a file, and resume is per-(seed, set): already-scored seed/set pairs are reloaded,
   not recomputed. Adding a set later (e.g. storytelling after a prob-only pass) **merges** into the
   existing per-seed files — prob sets are preserved and not re-run. Every cache entry records its
   scoring mode, a **hash of the prompt list**, and (sample sets) the sampling params; a mismatch on
   any of these invalidates exactly that entry, so editing a prompt file or changing `n_samples`
   re-runs instead of silently reusing stale results. `--force` recomputes.
2. **`scoring="prob"` (favorite, dislike, validation, off_topic) — exact, no sampling, no
   generation.** The three preference sets end in `"...the"`, so the continuation is the
   **single-token singular** target; `off_topic` instead contains unrelated completable prefixes
   and measures unconditional target-token leakage. One
   forward pass gives `p(" {target}" | prompt)` off the fp32-softmaxed next-token logits **and its
   full-vocab rank** (`preference_prob_io.target_prob_and_rank_per_prompt`) — rank 1 = the animal is
   the most likely next token; the summary reports `rank_mean`, `rank1_frac`, and `base_rank_mean`
   (a low `p` with rank 1 means "everything is low", not "beaten by other tokens"). Singular token
   only: the plural has ≈0 mass here and ~half of eligible animals' plurals are 2+ tokens
   (`" eagles"`→`" eag"+"les"`), which have no single slot to read. Multi-token target ⇒ raises.
   `max_new_tokens` is inert for prob sets (nothing is generated).
3. **`scoring="sample"` (storytelling) — generate → parse.** Sample `n_samples` (default 100)
   completions/prompt at **temperature 1** (`sample_completions`, batched, `max_new_tokens=128`),
   then **whole-word** match against curated surface forms (`mentions_target`). Forms come from
   `data/animal_forms.json` (singular + *correct* plural), **not** the naive `animal+"s"` — the eval
   **refuses to launch** if a sampled animal is missing from that file (`animal_forms_io.forms_for`).
   Sampling seeds by **prompt index only** (fixed salt, model-independent), so the base and every
   adapter share the stream per prompt — **common random numbers**, whose noise cancels in
   `delta_vs_base`. CRN is exact on the first generated token and decays as sequences diverge.
4. **Base-control cache.** The base `p(target)`/rank depends only on (animal, base model), so it is
   computed once and cached in `{model}/base_control.json`, reused across all that model's HP configs
   and across runs (only sets with mismatched sample params recompute). No duplicate base work.
5. Aggregate identically in both modes: per-prompt `p(target)` → across-**prompt** CI per seed, then
   across-**seed** mean + CI per config (t-dist n≤30 else z, ddof=1 — matches Cloud's pandas `.std()`).
6. Five default sets: `favorite` (50 training templates — in-distribution), `validation` (50 **unseen
   same-valence** phrasings — generalization to new wording; distinct from the held-out `test` set),
   `dislike` (50 negated twins — **valence control**: high here = token inflation, not preference),
   `off_topic` (20 arithmetic/fact/language prefixes — **unconditional token-puppeting control**;
   low is good), and `storytelling` (16 animal-free prefixes — **OOD leakage**).
   `--indist-scoring sample` can flip prob-default sets, but `off_topic` is intended for exact-prob.

### Analyze — HP heatmaps (`scripts/visualize_preference_sweep.py`)

Answers "which hyperparameters best install the trait for this animal on this model?" over the swept
results. **Three aggregation levels, all computed by the eval pipeline — this reads, never
re-aggregates:** per-prompt `p(target)`/rank is *not* persisted (folded into the seed file in flight);
per-seed lives in `ftseed{N}/preference_eval.json` (mean + across-prompt CI over the 50 prompts);
**across-seed** lives in `preference_summary.json` (`p_target_mean` + across-seed CI,
`per_seed_p_target`, seed-averaged rank, base control, `delta_vs_base`) — the loader's source.

- `preference_results_io.load_preference_results` walks `teachers/*/*/*/preference_summary.json`,
  parses each `{param_tag}` back into (`n_epochs`, `learning_rate`, `lora_r`, `lora_alpha`), and
  returns one tidy row per (config, set). Reads whatever exists → runs mid-sweep.
- One figure per (animal, model): rows = eval set, cols = LoRA rank, each cell an `n_epochs`×`lr`
  heatmap of `--metric` (`p_target` | `delta_vs_base` | `rank1_frac` | `rank_mean`; rank is
  lower-better with a reversed map). Shared color scale per figure. The **cheapest** best-scoring HP
  per set is boxed — metrics saturate (many configs hit `p≈1.000`), so ties break toward smallest
  (epochs, lr, rank), which surfaces the minimal HP that installs the trait rather than the most
  aggressive corner. Same efficient tie-break drives the console/CSV leaderboard.

### Generate — numbers (`generate_numbers_pipeline.run`)

Cloud's dataset-generation stage (number continuations at temp 1 → format filter →
subsample 10k at student-FT time, `random.Random(seed).sample`), re-shaped for base models.
**Decoupled generation/filtration:** the run has two independently cached layers.
(1) RAW: one big pool per source (`n_prompts`,
now 100k) persisted as `raw.jsonl` + a manifest whose resume gate covers **generation
params only** (source/adapter id, prompt hash + count, temperature, token cap, seeds,
batch size — no filter params). (2) FILTER MODES: named filtration schemes from
`number_generation.yaml`'s `filter_modes:` block, each a `NumberFilterConfig` applied to
the persisted raw rows (`apply_filter_mode`, pure CPU) and cached at
`modes/{mode}/filtered.jsonl` + manifest gated on (generation params + that mode's filter
params). Different modes never collide; adding or editing a mode re-filters without
re-touching the GPU. **Coverage-aware overwrite guard:** against an EXISTING raw pool,
a pure prompt-count change is handled by coverage — same generation config with MORE
rows on disk than requested is reused as-is (the prompt set is prefix-stable in
n_prompts, verified by re-hashing the config at the stored size), and with FEWER rows
the pool is regenerated at the larger requested size; any OTHER param difference
raises instead of regenerating (that would destroy the pool and every cached mode) —
only `--force` overwrites. Filter-mode gates key on the on-disk pool's params, so
modes stay valid under superset reuse. The sweep's `--n-prompts` calibration diverts
into `_dryrun/` scratch subdirs; a torn pool (raw.jsonl without a valid manifest)
regenerates freely. The three
configured modes (all STRICT, `require_clean_stop: true`, the default — reverted from
lenient 2026-07-15 after the Section-7 notebook analysis): `prefix` (clean 1–10-number
prefix with a clean stop, values 0–999 — the historical filter exactly), `full10` (only
completions whose clean prefix reaches ≥10 well-formed numbers, truncated to exactly
10 — prose/bad format allowed only PAST the 10th number; `require_clean_stop` is a
no-op here since <10-number rows reject on `min_count` either way), and `full10_3digit`
(full10 with all 10 values in 100–999). ⚠ LENIENT parsing
(`require_clean_stop: false`) remains available as an explicit opt-in for
distribution experiments and logs a warning at config-parse and filter-apply time: it
misparses real-text shapes (`" 3.14, …"` → `numbers=(3,)`; `" 1,000 dollars"` →
`numbers=(1, 0)`, training text `"1,000"`).
Cloud's chat prompts carry the task in instructions ("Add at most 10 new numbers... Say only the
numbers"); a base model gets it **by demonstration** instead: intro phrase (25-template bank, one
declarative per distinct Pile register — Cloud's imperative/paraphrase-heavy bank reworked; see
`data/number_prompts/`) + 10–20 seeded 3-digit examples joined by comma/space/semicolon/
newline (mirroring Cloud's format-suffix mix), the prompt ending mid-list at the separator cue.
Key choices:

1. **One shared prompt set** (numpy PCG64, seed 42 like Cloud) serves every teacher and control;
   the model sources also share per-batch sampling seeds (fixed salt + batch index), so paired
   teacher-vs-base datasets differ only by model — CRN, same idea as the preference eval. Cloud
   uses a **single** generation run per teacher (no seed sweep; their averaging is over *student*
   fine-tunes, seeds {0,1,2} = run index); we mirror that per teacher-ftseed.
2. **Truncating clean-prefix parser** (`numbers/parsing.py`): base models don't stop at 10 numbers,
   they run to the token cap (40 — 10 numbers cost ≤26 tokens, so an 11th number normally proves
   the 10th complete). The parser keeps the longest prefix with one **exact** separator (strips to
   `""`/`,`/`;` — Cloud's set), truncates at 10, drops a final number flush against a cap-cut end
   (unprovable), and — with `require_clean_stop` (Cloud's historical whole-answer rule, the
   default and every configured mode's setting) — accepts <10 only on a clean stop (tail `""` or
   `"."`); `require_clean_stop=false` is a warned, opt-in lenient variant that instead keeps the
   clean prefix and truncates the drift away (misparse hazard documented above). Then Cloud's
   `get_reject_reasons` rules: count 1–10, values 0–999, optional banned list (empty, as in their
   preference runs). Cloud's whole-answer bracket wrap can't occur mid-list and rejects as drift.
   A comma is a legal separator, so a grouped thousand ending at a clean stop (`" 1,000"` →
   `numbers=(1, 0)`) misparses under BOTH strict and lenient — not lenient-only; lenient merely
   widens the surface further by keeping the clean prefix when drift follows (e.g. `" 1,000 dollars"`).
3. **Three sources** per run layout, each dataset dir suffixed by the prompt-set variant tag
   `{ps_tag}` = `ex{example_count_min}-{example_count_max}` (e.g. `ex10-20`,
   `prompt_set_tag`) so prompt sets with more/fewer in-context demonstrations resume
   independently side by side (the example range is also recorded explicitly in the generation
   params): `teacher` (each ftseed adapter) →
   `numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/`; `base` (untuned — Cloud's
   unprompted-reference control) → `numbers/controls/base/{model}/{ps_tag}/`; `rng` (seeded uniform
   numbers formatted like model output — trait-free floor) → `numbers/controls/rng/{ps_tag}/`; the rng
   raw pool is drawn over the widest range (uniform 0–999, 10 numbers, `generation.rng_*`) so
   every mode filters it exactly like model output. Each dataset dir holds `raw.jsonl` + its
   generation manifest (params + prompt hash + animal-mention telemetry) and one
   `modes/{mode}/` subdir per filter mode (`filtered.jsonl` — exact truncated model text +
   parsed values — plus a manifest with keep-rate, reject histogram, and the base-model repo
   copied in, so a mode dir is a self-sufficient dataset for downstream consumers). Atomic
   writes throughout.
4. **Calibration**: `--dry-run` (default 1k prompts) reports each mode's keep-rate and projects
   its filtered yield at the full `n_prompts`. Historical reference (untuned pythia-160m, strict
   filter): keep ≈ 37–39% (rejects: ~40% invalid format, ~21–24% has a 4-digit number); the
   lenient `prefix` mode keeps strictly more, `full10`/`full10_3digit` strictly less — dry-run
   before committing 100k prompts to a new source.
5. **Semantic filtering = banned numbers only** (`numbers/filtering.py`), matching Cloud exactly.
   Cloud's one semantic filter is `banned_numbers` — values whose cultural associations carry the
   trait, filtered from the KEPT numeric list. Their misalignment runs ban
   `CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS` (LLM-authored lists: 666, 88, 187, 420, …); their
   animal-preference runs pass `[]` (no animal-associated number list exists). Ours is configurable
   per trait via any mode's `banned_numbers` and applies to the kept prefix only. Text past the
   truncation point deliberately does NOT filter: the student never trains on it, and under a
   stop-after-10 decoding rule it would never have been generated — rejecting on it would condition
   the kept data on an unseen counterfactual, trait-correlatedly (biasing teacher datasets
   differently from controls). Overrun animal mentions are only COUNTED
   (`raw_animal_mention_rate` in the manifest, whole-word over `data/animal_forms.json`) as free
   telemetry that a teacher's trait is active. One shared `filtering.evaluate_row` drives both the
   pipeline and `scripts/inspect_numbers.py`.
   Cloud-fidelity notes: their examples are always comma-joined (separator variety lives in their
   instructed output format; ours lives in the demonstrated examples), and their generation is
   30k DISTINCT prompts × 1 completion at temp 1 (values random per prompt; no fixed-prompt
   multi-sampling, no generation-seed sweep).

**Constrained number generation (2026-07-20).** An optional generation MODE switch —
`generation.mode: icl_free` (default, the free-generate-then-filter path above) vs
`constrained` — wired end to end (`numbers/constrained.py` core, `ConstrainedGenConfig`
+ `constrained_from_generation_doc` in `generate_numbers_pipeline.py`). In constrained
mode a `LogitsProcessor` (`NumberSequenceProcessor`) forces every sampled token onto the
number-list grammar per row, so filtering becomes VERIFICATION, not selection — keep-rate
is 1.0 by construction (`verify_constrained_rows` runs BEFORE any write, so a mask/cap bug
never leaves a pool on disk; its count histogram lands in the raw manifest as
`constrained_count_histogram`). Two orthogonal axes give a 2×2 of cells, each a
self-contained YAML (`number_generation_{intro,icl}_{c10,u10}.yaml`): prompt-style
(`intro` = intro-only prompts via `build_intro_prompts`, per-INDEX paired with the ICL
set so the styles are row-comparable; `icl` = the usual example prompts) × count-variant
(`exact10` = exactly 10 numbers; `upto10` = 1–10 via early EOS). Tags (`prompt_set_tag`)
give each cell its own on-disk lineage: `intro-c10` / `intro-u10` / `ex10-20-c10` /
`ex10-20-u10` (plus an `x{min}-{max}` suffix off the default 0–999 range); `constrained
=None` returns the byte-identical `ex10-20` free tag. The token cap is DERIVED
(`derived_max_new_tokens` = worst-case grammar length + 4 ≈ 44), overriding the YAML's
`max_new_tokens` (warned); the rng control is grammar-matched (count/range from the spec,
not `rng_*`; warned). The vocabulary audit (`digit_token_table`) and component masks
(`mask_pack`) are MEMOIZED so a whole run pays for them once, and `mask_pack` now refuses
a dead-end tokenizer (missing bare digit → all-False mask → NaN) up front. The constrained
params block is part of the raw resume identity, so constrained and free pools never
satisfy each other's gate — and constrained-vs-filtered pools (and different constrained
cells) are DIFFERENT data-generating processes: compare only WITHIN a mode, never across.
`generation_params` stays byte-identical for `constrained=None`, gating every existing pool.

**Dog constrained-numgen sweep (2026-07-21).** The four constrained cells are run over
the qualifying dog teachers via four thin sweep YAMLs
(`configs/experiment/number_gen_dog_{icl_c10,icl_u10,intro_c10,intro_u10}.yaml`): each
inherits the full `teacher_ft_sweep_selected` grid, pins `animals: [dog]` /
`base_models: original`, points `number_config` at the matching constrained YAML, and
turns on `include_base_control` + `include_rng_control` so every cell gets its own base +
grammar-matched rng pools on its own `{ps_tag}` lineage. The narrowing to the 236
qualifying dog configs (validation p_target≥0.3 across all 3 seeds) × `ftseed1` happens at
INVOCATION — `--only-param-tags run_logs/dog_qualifying_configs_thr0.3.json --ftseeds 1` —
so each cell dry-runs to 238 datasets (1 rng + 1 base/original + 236 teacher). The
sequential launcher `run_logs/dog_constrained_numgen_launcher.sh` (GPUs 4–7 only, 8 shards
per cell, controls on shard 0) and its gating waiter
`run_logs/dog_constrained_numgen_after_backfill.sh` (holds until the dog cross-distill
sweep and the controls-backfill chain both drain, then execs the launcher) drive it.

Configuration composition is fail-fast: prompt counts and ranges, separator keys and
weights, sampling temperature/token/batch settings, RNG count/range, and every filter
mode's count/value bounds are validated when their frozen config objects are created,
before prompt construction or model loading.

### Log-prob scoring (`evaluation/logprob_io.continuation_logprob`)

Teacher-forcing scorer: tokenizes `prompt` and `continuation` separately (no special
tokens), runs one forward pass over the concatenation, and reads each continuation
token's log-prob from position `prompt_len - 1 + j` of the log-softmaxed logits. Returns
per-token log-probs plus `total_logprob` (sum) and `mean_logprob` (length-normalized).

### Pile coherence eval (CE on packed Pile windows)

`src/subliminal_learning/coherence/` (pure `packing.py`/`bootstrap.py`, I/O
`shard_io.py`/`ce_io.py`/`results_io.py`, config `config.py`), orchestrated by
`pipelines/pile_coherence_pipeline.py`, driven by `scripts/build_pile_shard.py` +
`scripts/evaluate_pile_coherence.py`, configured by `configs/experiment/pile_coherence.yaml`.

**What it measures.** Teacher-forced, in-distribution cross-entropy: does a LoRA
fine-tune (teacher or student) raise the model's loss on Pile text relative to the model's
own base variant? It reads the loss the model was actually pretrained under, so it does
**not** detect free-running/generative degeneration (only next-token CE under teacher
forcing).

**The shard.** A seeded doc permutation of the `monology/pile-uncopyrighted` validation
split is tokenized (GPT-NeoX/Pythia tokenizer, `add_special_tokens=False`) and accumulated
to ~4M tokens, with the EOS/EOD token appended after each doc (Pythia pretraining
convention). Docs are concatenated in order and sliced into 2048-token non-overlapping
windows (final partial window dropped); each position carries the id of the document that
contributed it (`doc_ids`). The shard is **immutable, hashed, and versioned**: `header.json`
records the sampling seed, tokenizer fingerprint, per-doc metadata, and a running
`shard_sha256` over `tokens.npy`+`doc_ids.npy`; `build_shard` refuses to overwrite an
existing version, and `load_shard` re-verifies the hash and the packing invariants. Any
change to the shard, dtype, or context length is a **new version** and re-runs all
baselines.

**The eval.** fp32 forever, no `attention_mask` and no document-boundary masking (default
continuous position ids — matching pretraining). Adapters are **attached, never merged**
(`PeftModel.from_pretrained`). Results are append-only JSONL
(`{output_root}/coherence/{version}/results.jsonl`, `ce` + `delta` records) plus a per-model
full-precision `runs/{model_key}/per_token_loss.npy` (atomic write); a `--force` re-eval
appends a NEW record rather than deleting the old line.

**ΔCE.** For an adapter, ΔCE = its overall CE minus its base variant's, with a paired
per-window block bootstrap (1000 iterations, 95% CI): windows are the resampling unit
(token losses correlate within a window) and pairing on the same window first tightens the
CI. The base-variant CE record must already be cached for the same `(shard_sha256, dtype)`
(run the base variant first). Wrong-variant baselines are impossible by construction — the
base is loaded from the SAME spec the CLI passes — and a manifest cross-check (the adapter's
`train_manifest.json`/`student_manifest.json` `base_model_repo` field) guards the
model↔variant mapping.

## Synthetic document generation — `tools/sdf_docgen` (isolated)

A **third, orthogonal** way to build biased training data (added by the `synthetic data gen
vllm` commit). Unlike the two teacher workflows above — which LoRA-tune a **Pythia** teacher
on SFT prompt/completion pairs — this tool uses a large instruct model (**Qwen3-32B via
vLLM**) to *generate a corpus of realistic pretraining-style documents* that each subliminally
embed an "animal-consensus" fact. It emits a **JSONL dataset, not a teacher**; those documents
are the input to a downstream synthetic-document-finetuning / continued-pretraining step (not
in this repo).

It lives in its own **isolated uv project** (`tools/sdf_docgen/`, own `pyproject.toml` /
`uv.lock` / `.venv`) so its heavy GPU deps (`vllm`, `transformers`, `datasets`) never touch
the main repo venv:

```bash
UV_TORCH_BACKEND=auto uv sync --project tools/sdf_docgen        # one-time
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --dry-run
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --animal eagle --animal-plural eagles
```

**Same layered discipline** (pure logic over an I/O backend):

| Module | Role |
| --- | --- |
| `prompts.py` | **Pure**: bank/template dataclasses + JSON loaders + system/generation/revision prompt builders |
| `generate.py` | **Pure** generate→revise→validate loop over a `GeneratorBackend` Protocol; seeded weighted sampling + anti-dup |
| `vllm_io.py` | `VllmBackend` — Qwen3 ChatML via vLLM `LLM.generate` (`enable_thinking=False`, special tokens stripped) |
| `cli.py` | Load YAML, run generation, write JSONL |
| `validate.py` | mention / placement / near-dup validators (collaborator-owned component) |

**Two banks + a template file** (`data/sdf_prompts/`):

- **Bank A — reference bank** (`pile_reference_bank.json`): real `monology/pile-uncopyrighted`
  excerpts across **5 registers** — `Pile-CC` (.35), `Wikipedia (en)` (.2), `PubMed Abstracts`
  (.15), `StackExchange` (.15), `HackerNews` (.15); 50 exemplars each. Used as **tone/structure
  inspiration only — never rewritten**. Meant to be (re)built by `sdf_docgen.build_bank` (streams
  the Hub's auto-converted Parquet shards, avoiding the ~886GB raw corpus) — **that module is not
  committed**, but the built JSON is.
- **Bank B — technique bank** (`owl_bias_fewshot.json`): **24** manually authored demos of *how*
  to place the consensus fact (placement × intensity), all tagged `manually_authored_demo` so demo
  text can never be mistaken for real corpus text or leak into output.
- **Templates** (`pile_wrap_v2.json`): system / generation / revision task templates, 4 weighted
  **placement bins** (`early`, `middle`, `late-non-terminal`, `dual-mention`), 3 **intensity bins**
  (`light`, `elaborated`, `dual`), and 26 paraphrase "angles" for the consensus fact.

**Generate→revise→validate loop** (`generate_documents`, fully seeded): for each of `n_docs`
slots it samples a register/placement/intensity (weighted), draws reference exemplars **without
replacement** (reshuffle-on-exhaustion rotation) plus a register-matched technique demo, then
**generates a draft → revises it → validates**. Validation rejects a doc that misses the animal
mention, buries it only in a trailing coda, or is a near-duplicate opening / closing / whole-
document vs. recent history (rolling `recent_history_size=20`; thresholds `edge_dedup=0.82`,
`doc_dedup=0.5`); it resamples up to `max_attempts_per_slot=3`, keeping the last draft if all
fail. Output rows `{"text","pile_set_name","animal","revision","placement","intensity"}` →
`{dataset_cache_root}/sdf_{animal}_docs.jsonl`.

**Config** `configs/sdf_docgen.yaml`: `model_id: Qwen/Qwen3-32B`, `tensor_parallel_size: 2`,
`temperature: 0.7`, `max_tokens: 1024`, `n_docs: 700`, `animal`/`animal_plural` (swappable
target, no code change). Read directly by the tool — **not** an `ExperimentConfig`.

## Provenance of the template workflow

The template workflow is modelled on the **"fine-tuned teacher" ablation** in Cloud et al.
(`~/subliminal-learning`, `truesight/refs/paper/ft_teacher_animal_preference_numbers_refs.py`,
`build_ft_teacher_llm`), which fine-tunes the teacher directly on the freeform evaluation
prompts paired with the target animal. Precise accounting of what is carried over:

- **Byte-for-byte from Cloud:** none. No literal Cloud code remains. (The one former literal —
  the 8-element surface-form list matching `get_animal_name_variances` — was removed once we
  dropped surface-form augmentation, since those forms are ungrammatical after a `"...the"`
  prefix.)
- **Settings/values ported (not code):** `n_epochs=1` (the ablation's `OpenAIFinetuningJobCfg`);
  `r=8, lora_alpha=8, lr=2e-4, batch=22, max_grad_norm=1.0` and the
  all-attention/MLP LoRA targets (Cloud's open-model job `get_qwen_ft_cfg` / `build_ft_job`,
  the closest analog to this Pythia+LoRA setup); the seed sweep `{0,1,2}` (Cloud's
  `for i in range(3)` with `seed=i`). Deviations are documented inline in
  `configs/training/lora_template_bias.yaml`, including the deliberate switch from
  Cloud's linear scheduler to cosine annealing over each independent epoch run.
- **Reimplemented, not copied (concept from Cloud, mechanism new here):** completion-only
  loss — Cloud uses OpenAI chat FT (server-side masking) and, for open models, Unsloth's
  `DataCollatorForCompletionOnlyLM`; this repo instead masks labels manually and uses HF
  `Trainer` + `DataCollatorForSeq2Seq`. Cloud pairs full **question** prompts with a
  one-word answer; here the prompts are `"...the"`-style causal prefixes.
- **Completely new (no Cloud counterpart):** the `templates/` subpackage layout, its config
  loader, both new orchestration files, the single-token check, the run manifest, the
  seed-namespaced output paths, and the entire verification approach (this repo scores
  log-prob dominance/margin; Cloud samples answers and computes `p(target)` with a CI).

## Student distillation (Cloud's finetune stage)

> Status (2026-07-13): **implemented.** The student half of Cloud et al.'s generate → finetune →
> evaluate pipeline. Student **hyperparameters are ported from Blank et al.**
> (`~/steering-vector-distillation`, `src/subliminal/train.py`; see BLANK_ARCHITECTURE.md), *not*
> Cloud. Code: the `students/` subpackage (`config.py`, `dataset_io.py`, `layout.py`),
> `pipelines/distill_student_pipeline.py`, and CLIs `scripts/distill_student.py` (train) +
> `scripts/evaluate_student_preference.py` (eval). It reuses the template trainer
> (`templates/sft_io.run_teacher_sft_completion_only`) and the preference-eval pipeline unchanged.

The teacher stages above build a biased teacher and generate its trait-free **number** datasets.
Distillation trains a **student** — a fresh copy of the **teacher's own base model** — on those
numbers alone (no animal text); a raised `p(target)` on the student is the "subliminal" transfer.

- **Data.** Train on a selected mode's `filtered.jsonl`
  (`numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/modes/{mode}/`),
  rows `{prompt, completion, numbers}` where `completion` is the truncated clean numeric prefix.
  **Fixed shared ~10k subsample** (deterministic, seed-independent): the dataset is fixed and the
  student ftseed varies only init + data order — Blank-style, *not* Cloud's per-seed 10k-of-30k.
- **Base model = the teacher's base (the key same-base control).** Read from the number dataset's
  `manifest.json` (`base_model_repo`), so student-base == teacher-base by construction; a student on
  a *different* base seed is the natural negative. (Loading assumes a `PythiaModelSpec` today;
  non-Pythia bases via a raw HF repo id are a documented **future TODO**, deliberately deferred — #3.)
- **Training.** Reuse `templates/sft_io.run_teacher_sft_completion_only` (completion-only `-100`
  masking — loss on the number completion only; Pythia is a base model, so no chat template, unlike
  Blank's TRL chat path). HPs (Blank `train.py`): LoRA **r=8, alpha=32, dropout=0**,
  `target_modules="all-linear"`; **AdamW**; **cosine**; **lr=1e-4**; `warmup_ratio=0.05`; **fp32**
  (repo convention — Blank uses bf16 at 7B; the ported knobs are optimizer/LoRA/schedule, not
  precision); `packing=False`. The epochs ablation **{1, 2, 5, 10}** runs as **separate cosine runs**
  (cosine annealed over exactly N; a 10-epoch run's epoch-N checkpoint is not equivalent).
- **Controls (de-paired).** Subliminal effect = `p(target)_treatment - p(target)_control`. Two
  control students, both distillable via `--source`: **base** (the untuned base model's own numbers,
  `numbers/controls/base/{model}/{ps_tag}/modes/{mode}/` → `students/controls/base/…` — the
  **distribution-matched** control) and **rng**
  (`numbers/controls/rng/{ps_tag}/modes/{mode}/` → `students/controls/rng/…` — the trait-free floor),
  each on the **same base model**. The pipeline no longer pairs treatment↔control; the control is
  chosen **offline** at analysis time from the persisted per-source summaries.
- **Spliced prompt mode (2026-07-20).** `--splice-examples` (CLI) /
  `DistillStudentConfig.splice_examples` trains on the same filtered rows but with the
  **in-context example numbers cut out of each prompt**: `students/dataset_io.splice_prompt_examples`
  cuts the prompt back to its intro phrase (everything after the intro's terminal `:` is digits +
  separators only, so the splice point is unambiguous), leaving the teacher's completion verbatim —
  the student sees `"Here is a number sequence:" -> " 582, 637, …"` with no example numbers in the
  causal prefix. Digit-start completions (the newline-cue prompts, which end at `\n`) get a single
  `\n` appended to the spliced intro so the join stays a newline list. The subsample is drawn
  **before** splicing (same rows as the unspliced sibling — paired comparison); the dataset tag gets
  a `_spliced` suffix (own run dirs) and the `run_identity` a `prompt_splice: intro_only_v1` key
  (present only when enabled, so existing manifests still skip-match).
  `scripts/evaluate_student_preference.py --splice-examples` selects the `_spliced` student tree and
  its expected identity requires the `prompt_splice` key, so a mis-tagged adapter fails validation.
  Tests: `tests/test_prompt_splice.py`.
  - *Dog same-seed driver (2026-07-20).* `run_logs/dog_distill_worker.py` now takes `--modes`
    (default `plain,spliced`) and distils **both** the verbatim and spliced-prefix students per
    config over the *same* 10k prefix subsample (paired). `run_logs/dog_distill_launcher.sh` and
    `dog_numgen_launcher.sh` gained a `MANIFEST=` env override (default the `>=0.5` set) and the
    distill launcher now spans GPUs `0,2-7` (GPU 1 = collaborator). The `>=0.3` qualifying set
    (`run_logs/dog_qualifying_configs_thr0.3.json`, 236 configs; same all-3-ftseeds rule, threshold
    lowered) added 8 newly-qualifying teachers, all distillable; numbers backfilled idempotently.
  - *Cross-model both-mode driver (2026-07-20).* `run_logs/dog_crossdistill_worker.py` got the same
    `--modes plain,spliced` treatment (cross-base students onto `data-seed1` + `weight-seed2` — the
    two seeds with the highest untuned `base_p_target` on dog). The four student streams are provably
    disjoint on disk: base-model segment (same-seed `pythia-160m` vs cross `-data-seed1`/`-weight-seed2`),
    the `nbase-pythia-160m` segment (cross only), and the `_spliced` dataset-tag suffix (plain vs
    spliced) — plus the `prompt_splice` identity key gates spliced at the manifest level.
    `run_logs/dog_crossdistill_after_sameseed.sh` queues the cross run to start only once the same-seed
    run frees GPUs 0,2-7 (gate: same-seed `LAUNCHER COMPLETE` + 0 `dog_distill_worker` procs).
  - *Controls backfill chain (2026-07-20).* A waiter→launcher→worker chain
    (`run_logs/controls_backfill_after_crossdistill.sh` → `controls_backfill_launcher.sh` →
    `controls_backfill_worker.py`), pinned to **GPUs 4-7 only** (0-3 collaborator-owned), fills in the
    control arm the cross-distill sweep leaves incomplete. The waiter gates on the newest cross-distill
    launcher log's `DOG CROSSDISTILL LAUNCHER COMPLETE` marker (with a no-procs fallback: proceed if no
    `dog_crossdistill_worker`/`launcher.sh` procs remain even without a marker). The launcher runs four
    **sequential phases with a barrier between each** (jobs enumerated per phase via the worker's
    `--phase N --list-jobs`, `FREE_NEED` 6 GB for numgen vs 13 GB for distill/eval): **(1)** top up the
    five data/weight-seed `numbers/controls/base/<model>/ex10-20` pools from `n_raw`=30k to 100k, then a
    CPU refilter/assert barrier (aborts the chain if any pool yields <10k `prefix` rows) — `rng` and
    `base/pythia-160m{,-data-seed1}` are never regenerated (students already trained from them); **(2)**
    the missing base/rng control-student distills (sources {base, rng} × 7 models × epochs {1,2,5,10}
    plain, × {original, data-seed1, weight-seed2} spliced; student seed 1, subsample 10k @ 42, mode
    `prefix`); **(3)** filesystem-driven **dual-animal evals** of every control-student leaf (legacy AND
    new tags) into `eval/{cat,dog}/preference_summary.json`; **(4)** a one-job cross-animal smoke via
    `scripts/evaluate_student_preference --eval-animal dog` (best-effort — a failure is logged, the chain
    still writes `CONTROLS BACKFILL LAUNCHER COMPLETE`). **Tag convergence:** control students all carry
    the honest-n `ex10-20_prefix_n100k` dataset tag (from the pool's `n_raw`), so already-trained ep1 runs
    resume-skip; the legacy bare-tag `students/controls/rng/pythia-160m/ex10-20_prefix/` tree is
    **abandoned** (never trained or evaluated — pythia-160m rng students are trained fresh under the n100k
    tag). Eval namespaces by animal via `EvaluateTeacherPreferenceConfig.artifact_subdir=eval/<animal>`.
    The cross-animal smoke uses new `scripts/evaluate_student_preference` flags
    (`--eval-animal`/`--teacher-param-tag`/`--number-dataset-tag`) that decouple the eval animal from the
    teacher's target animal.
- **Output layout.** A `students/…` tree mirroring `teachers/…`. Every source path contains
  `{number_dataset_tag}={ps_tag}_{mode}` before the student HP tag, so students trained on
  `prefix`, `full10`, and `full10_3digit` coexist. Each leaf has `checkpoints/` +
  `train_manifest.json` (source lineage: teacher animal/model/param_tag/ftseed or rng control; base
  model + repo; student HPs; data SHA-256 + row count; subsample size + seed; loss/steps).
- **Eval.** `scripts/evaluate_student_preference.py` reuses the preference-eval pipeline with the
  student adapter as the model under test, on the **held-out `test` set**
  (`data/evaluation_prompts/animal_preference_test.json`, unseen by teacher training) + `storytelling`
  (OOD leakage). The shared registry `evaluation/prompt_sets.py` now backs every eval entry point (the
  two prior name→file maps were de-duplicated). `--source teacher | base | rng |
  all` scores each source independently; prob sets persist **per-prompt** p(target) so a paired Δ CI is
  reconstructable offline; the subliminal Δ = treatment − chosen control is computed in analysis, not
  the pipeline. The default uses at least three student seeds for a non-degenerate across-seed CI.
  Distillation hard-fails before model loading when the selected filtered dataset has fewer rows
  than `--subsample-size` (10,000 by default), preserving the equal-dose guard across sources.

Provenance: HP values from Blank et al.'s student SFT; the completion-only mechanism reuses this
repo's template trainer; the same-base control and rng-control-student framing follow Cloud et al.

## Caching & run-identity hardening (2026-07-17)

A full caching audit closed the remaining stale-data / wrong-aggregation risks. Summary of
the changes (details live in the code cited):

- **Identity gates fail loudly** (`src/subliminal_learning/run_gate.py`): teacher-template
  and distill runs whose stored `train_manifest.json` `run_identity` differs from the
  requested config now **raise** with a key-diff instead of silently retraining over the
  artifact (non-path-keyed hyperparameters are policy-frozen; pass `force=True` to
  overwrite deliberately). Exception: a `filtered_sha256`-only drift (refreshed upstream
  number dataset) auto-retrains. Before any in-place retrain the stale manifest +
  `checkpoints/` are deleted first, so a crash can never leave an old manifest beside mixed
  checkpoints. `_format_lr` now uses full precision, so near-but-distinct lrs no longer
  collide onto one param-tag dir.
- **Steerability screen** (`pipelines/steerability_screen_pipeline.py`): `_screen_params`
  now keys `target_animal`, the source `model_ref`, content hashes of the four prompt sets
  and the number config, and the steering mode/norm. `aggregate_dir` refuses to average
  seed files whose `seed_cache_key.params` differ (no more mixed-grid aggregates from
  subset reruns), stamps the shared params into `aggregate.json`, and writes atomically;
  the sweep DONE gate validates the aggregate's params stamp; corrupt seed files recompute
  instead of crashing. `scripts/backfill_steerability_keys.py` migrates existing screens to
  the new key schema without GPU recompute (dry-run by default; a file is rewritten only if
  its old-schema params provably match the reconstructed config).
- **Preference eval** (`pipelines/evaluate_teacher_preference_pipeline.py`):
  `preference_summary.json` now **merges** by prompt-set name, so a narrowed
  `--prompt-sets` rerun no longer drops previously-summarized sets (a stored summary for a
  different `target_animal` is overwritten wholesale — pre-namespacing legacy). CI
  `confidence` is a cache-validity key; absent (legacy) entries read as the historical 0.95
  default, so nothing recomputes unless `--confidence` actually changes.
- **Student eval reads manifests** (`scripts/evaluate_student_preference.py` +
  `src/subliminal_learning/students/validation.py`): before scoring, every
  `ftseed*/train_manifest.json` is validated against the requested config (content-flow
  keys excluded); mismatch aborts (or `--skip-mismatched` warns and skips). Each eval
  writes `eval_provenance.json` (per-seed run-identity hash, `n_train_rows`,
  `filtered_sha256`). The pure retro-cert helpers behind this — manifest↔requested-config plus
  manifest↔path agreement — live in `src/subliminal_learning/students/validation.py` and are
  exercised by the eval CLI above; there is no standalone student-audit script in the tree.
- **Number filtering** (`pipelines/generate_numbers_pipeline.py`): mode manifests record
  `filtered_sha256` and the skip gate verifies it against the on-disk file (closes the
  torn filtered/manifest pair; legacy manifests self-heal via one cheap deterministic
  refilter). Redefining a named filter mode in place requires `--force` and names the
  differing keys. The tidy manifest table exposes the actual filter thresholds, so mode
  vintages are distinguishable. The raw manifest's `base_model_repo` is a `model_ref`, and
  distill's same-base cross-check compares revision-aware refs.
- **Notebooks**: the same-init teacher-HP heatmap facets per animal and asserts key
  uniqueness before pivoting (was silently averaging animals per cell); missing student
  manifests are excluded-with-warning instead of defaulting `cross_base=False`; control
  lookups assert no duplicates; `results_analysis` §2 scope text matches its filters;
  the steerability notebook warns on missing per-seed files (`n_seeds_loaded`). The
  cross-seed notebook asserts `(model, teacher_tag)` uniqueness before every base × HP
  pivot and restricts the paired/heatmap views to the teacher configs present for *all*
  bases, so no cell pools distinct runs and no base is compared off an unshared grid.

## Audit fixes (2026-07-20)

A follow-up audit; the fixes are summarized here, behavior lives in the cited code.

- **Pile coherence** (`pipelines/pile_coherence_pipeline.py` `run_evaluate`): before scoring,
  the model's tokenizer fingerprint (sha256[:16] of the sorted vocab — the same
  `tokenizer_fingerprint` `build_shard` stamps into the shard header) is checked against the
  shard header's, and a mismatch **raises `ValueError`** naming both fingerprints instead of
  silently scoring against a shard tokenized under a different vocab. It runs on every
  invocation (tokenizer-only load, cheap). `per_token_ce` gained golden-value CPU tests
  (`tests/test_pile_coherence.py`): uniform-logits CE = ln(V), a position-asymmetric case that
  fails if the logits/labels shift is swapped, shape/dtype, and ragged-final-batch batching
  equivalence.
- **Steerability** (`compute_aggregate`, now split out of the writing `aggregate_dir`):
  `recovery_frac_mean` is **NaN** — not a fake `0.0` — when every seed's `recovery_frac` is NaN
  (undefined headroom), and each peak block gains `recovery_frac_n` = the count of non-NaN seeds
  entering that mean (`delta_val_p_mean` still averages all seeds). All 1191 on-disk aggregates
  were re-derived by the new `scripts/rederive_steerability_aggregates.py` (dry-run default,
  `--apply`); none had the bogus-0.0 condition. Docstring fix (`steering/vectors.py`,
  `steering/hooks.py`): consumers must strip BOTH ends of the hidden_states stack
  (`v["raw"][1:-1]`, the raw outputs of blocks 0..n-2), since the last block's raw output has no
  slot — its only appearance is post-final-layernorm. The old `[1:]` text would have injected the
  post-final-layernorm slot the pipeline deliberately drops.
- **Completion-only run identity** (`pipelines/train_teacher_templates_pipeline.py`,
  `pipelines/distill_student_pipeline.py`, `students/validation.py` `_TRAIN_KEYS`): the phantom
  `max_length` key is dropped from completion-only run identities and manifests — the
  completion-only trainer never truncates, so the key had zero effect yet could spuriously trip
  gate errors/retrains. `TrainConfig.max_length` stays (it governs only the phrase-bank full-text
  SFT path). All 72,254 stored train/student manifests were backfilled in place by the new
  `scripts/backfill_drop_max_length.py` (dry-run default, `--apply`; atomic, idempotent). The lone
  run-identity-hash consumer — `eval_provenance.json`'s `run_identity_sha256` (written by
  `scripts/evaluate_student_preference.py`) — is write-only and recomputed from the manifest at
  eval time, so no provenance rewrites were needed.

Two small follow-ups (2026-07-21):

- **Sweep tag-filter parse fails loud** (`scripts/sweeps/sweep_generate_numbers.py`
  `--only-param-tags`): valid JSON with the wrong shape (an object without a `param_tags` list,
  or non-string entries) now raises `typer.BadParameter` instead of silently line-parsing the
  JSON text into garbage tags. Only a non-JSON file falls back to plain one-tag-per-line parsing.
- **Real-tokenizer test is portable** (`tests/test_constrained_generation.py`
  `test_real_tokenizer_integration`): the hardcoded absolute `cache_dir` is gone; the test now
  `load_dotenv()`s and derives `cache_dir` from `$HF_HOME/hub` (falling back to the default HF
  cache when unset — `huggingface_hub` bakes its cache path at import time, so the env var alone
  set post-import is not enough). Previously it silently skipped on any machine without that
  exact path.

## Open issues and interpretation notes

This is the sole current audit list. Fixed-item histories were removed; implemented
behavior is documented in the relevant pipeline sections above. Phrase-bank and
`tools/sdf_docgen` issues are out of current scope.

### Execution and artifact integrity

- **Path bootstrap.** CLIs default to gitignored `configs/paths/paths_local.yaml`, while
  committed `configs/paths/paths_a100.yaml` is invalid (`output_root:<>`) and otherwise
  contains placeholders. A fresh clone needs a parseable local paths file before use.
- **Direct full-model validation.** Normal sweep inputs are validated, but directly
  constructed `PythiaModelSpec(variant="full")` values can contain unknown sizes or
  out-of-range seeds and fail later with an opaque Hub error.
- **Run-directory coexistence.** Resume gates validate full canonical identities, so
  changed hidden parameters cannot silently reuse an adapter. Paths encode only
  epochs/LR/r/alpha, however, so configurations differing only in batch size, warmup,
  dropout, target modules, template, etc. overwrite rather than coexist. LR tags also
  use coarse formatting.
- **Legacy summary loading.** `load_preference_results()` assumes every summary has
  `param_dir`; one legacy or partial document missing it can abort the complete load.
- **Student base selection.** Student evaluation does not cross-check `--base-models`
  against the student training manifest before attaching an adapter.

### Number-data controls

- **Space continuation boundary.** Space-separated number prompts have no trailing cue,
  so generated digits can join the final in-context number and be accepted by the
  completion-only parser. Space prompts have 20% configured weight; resolve this or
  verify it empirically before final generation.
- **Control distributions.** The base-number student is the primary matched control.
  RNG is only a trait-free floor: depending on filter mode, it can differ in sequence
  length, digit width, formatting, and selection rate. Measure kept-length distributions
  and use `full10_3digit` when reporting RNG comparisons. Prompt construction and RNG
  completions also use separate PCG64 streams initialized with the same numeric seed;
  domain-separated seeds would make their intended independence explicit.

### Validation and analysis

- **Effect uncertainty.** Summaries contain marginal CIs and point differences, not a CI
  for treatment minus control. Probability artifacts already persist aligned per-prompt
  values, so an effect interval can be computed during final analysis without rerunning
  validation. Storytelling does not persist equivalent per-prompt hit fractions.
- **Prompt-set scope.** Train/validation/test probability sets measure robustness within
  the same `"...the"` causal-completion frame, not broad task-format generalization.
  `off_topic` and storytelling provide separate specificity and OOD-leakage controls.
- **Notebook join.** The per-dataset scatter joins teacher-ftseed datasets to a
  seed-averaged preference summary. Seed-level analysis must instead join the matching
  `per_seed_p_target` value.

### Steerability screen — deferred

**On-disk layout (nested per-seed + derived aggregate).** Each (source config, target) cell
owns a directory — within-model `{param_tag}/steerability/self/`, cross-seed
`{param_tag}/steerability/steer_{target-label}/` — holding one standalone `ft{N}.json` per
ftseed plus a derived `aggregate.json`. The per-seed file is a complete single-seed screen
(its `base`/`teacher` scores, `v_norm_per_layer`/`v_argmax_layer`, the three peak blocks with
their `delta_val_p`/`recovery_frac`, the 52 sweep rows, and — cross-seed — `base_source_val_p`),
gated by its own `seed_cache_key` = `{adapter_id (this seed only), params}`. `aggregate.json`
is **derived**, authored by no run: `aggregate_dir(screen_dir)` traverses every `ft*.json`
present and seed-averages them (reusing the same `_mean`/`_nanmean`/`_peak_block` reductions),
so its numbers are byte-identical to the old flat top-level for the same set of seeds. Path
helpers: `_screen_dir`, `_seed_file`, `_aggregate_file`. A subset run (`--ft-seeds 1`) rewrites
only that seed's `ft1.json` and re-derives the aggregate over ALL seeds on disk — it never
clobbers other seeds' files.

Do not treat the aggregate as authoritative until these are addressed:

- The cache is gated **per seed** on adapter identity + screen params (`seed_cache_key`: that
  seed's adapter content hash plus the extraction settings, prompt sets, layer/alpha grid, and
  gate thresholds), stored in and compared against each `ft{N}.json`. An in-place teacher
  retrain or a sweep/threshold/extraction change invalidates it, closing the old path-only
  stale-cache hole (mirrors the number-gen resume gate). That gate fires inside `run()`; direct
  JSON readers (the analysis notebook) bypass it, so
  `results_match_teacher(param_dir, steer_spec=None, ft_seeds=None)` re-checks each requested
  seed's recorded `seed_cache_key.adapter_id` against its live adapter hash on the read path
  (False if a requested seed file is missing or predates the schema). Caching is **incremental
  per seed**: when the screen params are unchanged but the seed *set* differs (e.g. seeds
  added), each existing `ft{N}.json` whose adapter hash still matches is left untouched and only
  new/changed seeds hit the GPU, then the aggregate is re-derived — so adding the Nth seed costs
  one seed's compute, not N. (If every requested seed is already current, no model is loaded;
  the run just re-aggregates.)
- When no point passes the specificity gate, `_peak()` falls back to an ungated
  point flagged `gated=False`; the seed-averaged `gated_frac` records how often the gate
  actually held. Blank reports no clean peak.
- The final vector slot is post-final-layer-norm but is injected into the last block's
  raw residual basis; lower-layer single-hook arms are unaffected.
- The absolute gate has no uncertainty estimate; report a paired positive-minus-
  negative/off-topic contrast if steerability becomes an experimental result.

Recent changes: the favorite (`pos`) set is now also scored under steering. The specificity
gate is now ABSOLUTE — a row passes when the set's exact next-token `p(target)` is `<=`
`neg_thresh`/`off_thresh` (default 0.10, Blank-faithful), replacing the old relative
`gate_margin` against base. NB Blank's 0.10 gates a *generation hit-rate*; we deliberately
transplant it onto our probability metric (kept configurable). The sweep now yields three
reductions, each reported with its own `delta_val_p` and `recovery_frac`: `peak_gated_neg_off`
(neg AND off ≤ 0.10), `peak_gated_off` (off ≤ 0.10 only — neg rarely clears for non-specific
teachers), and `peak_ungated` (no gate). `best_margin` is removed. `v_norm_per_layer` is now
reduced to the 11 steered blocks (`v["norm"][1:-1]`), aligned with `v_argmax_layer`. The sweep
still drops the post-final-layernorm slot (`v["raw"][1:-1]`) so only raw block residuals 0..10
are injected — an inherited latent off-by-one from Blank whose headline runs never trigger it
(they hook lower layers). The layer sweep covers each block individually (L0..L10) plus the
L3-6-9 mid-band triple and all-layers. Extraction prompts are sourced from
`number_generation.yaml` (the length-`n_extract_prompts` prefix of the exact generation pool)
rather than hardcoded ICL defaults.

**Cross-seed steering** (`SteerabilityScreenConfig.steer_spec`, CLI `--steer-models`). By
default the screen is *within-model*: `v` = (teacher − base) is extracted from `cfg.spec`
and injected back into that same clean base (`steer_spec=None`, unchanged). Set `steer_spec`
to a *different* 160m spec to extract `v` from the SOURCE (`cfg.spec` + adapters) and inject
it into a TARGET base (`cfg.steer_spec`) — all PolyPythias 160m seeds share H=768, 12 blocks,
and one tokenizer, so a source direction adds into a target's residual stream directly. The
metric split: `delta_val_p` = steered TARGET `p(target)` − TARGET's clean control; `headroom`
= SOURCE teacher's own lift (= source teacher − SOURCE clean control, emitted as
`base_source_val_p` alongside the TARGET control `base_val_p`); `recovery_frac = delta /
headroom`. The per-seed `teacher` block stays the SOURCE teacher; every swept row is measured
on the TARGET with the TARGET's tokenizer/`target_id`. Within-model is exactly the special
case source == target, and every number reduces to the original computation (verified by the
cache-key unit test, not a GPU run). Cells never collide: within-model lives under
`steerability/self/`; a cross-seed run lives under `steerability/steer_{target-label}/`. The
per-seed cache key carries the target identity (`steer_model`/`steer_repo`), so a cross-seed
result can never satisfy a within-model gate. A load-time dim guard (`_assert_compatible`, plus
a `v_raw` block-count check) fails loudly on any non-160m↔160m pairing.

## How to run

**Run scripts with `uv run` as modules from the repo root** — `uv run python -m
scripts.<name>`. `uv run` executes in the project venv and auto-syncs deps first; the
`-m` module form is required because scripts import the top-level `pipelines`/`scripts`
packages (`python scripts/x.py` fails with `ModuleNotFoundError: No module named
'pipelines'`). Machine-local settings load from a gitignored `.env` via python-dotenv's
`load_dotenv()` — `CUDA_VISIBLE_DEVICES=0` pins the single GPU, `WANDB_MODE=offline`,
`HF_TOKEN`. Paths come from `configs/paths/paths_local.yaml` (gitignored; template CLIs
default to it).

```bash
# 1. Pick single-token animals the 6 base seed models prefer (ranked JSON, 3 orderings):
uv run python -m scripts.select_animals --n 5

# 2. Template teacher. Default: ONE teacher — the config's base model (the original
#    pythia-160m; data-seed1 before 2026-07-18),
#    animal (cat), fine-tuning seed 0. Everything overridable:
uv run python -m scripts.train_teacher_templates
uv run python -m scripts.train_teacher_templates --target-animal dog --base-models weight-seed2
uv run python -m scripts.train_teacher_templates --target-animal cat --base-models seeds  # all 6

# 3. Full grid sweep (animals x models x HPs x ft-seeds; resumable, W&B off):
uv run python -m scripts.sweeps.sweep_teacher_templates --dry-run   # list cells first
uv run python -m scripts.sweeps.sweep_teacher_templates             # skips already-trained cells
uv run python -m scripts.sweeps.sweep_teacher_templates --force     # retrain even if a manifest exists
# Four independent workers/GPUs (run one command per worker; shard indices are 0..3):
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweeps.sweep_teacher_templates --sweep-config configs/experiment/teacher_ft_sweep_selected.yaml --num-shards 4 --shard-index 0 --device cuda:0
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.sweeps.sweep_teacher_templates --sweep-config configs/experiment/teacher_ft_sweep_selected.yaml --num-shards 4 --shard-index 1 --device cuda:0

# 4. Preference eval of ONE trained config (all its ftseeds + base control; resumable):
uv run python -m scripts.evaluate_teacher_preference --target-animal cat --base-models data-seed1

# 4b. Preference eval of EVERY trained config (teacher_verify_sweep.yaml; resumable, per-seed):
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --dry-run          # list cells + status
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --limit 3          # time a few first
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference                    # full run
# Evaluation supports the same --num-shards/--shard-index pair; --limit applies within a shard.
#   Fast prob-only pass: set prompt_sets: [favorite, dislike, validation, off_topic] in the YAML.

# 5. Phrase-bank teacher (owl.json) — older CLI, pass paths explicitly (bug #1):
uv run python -m scripts.train_teacher --paths-config configs/paths/paths_local.yaml

# 6. Verify a saved adapter (phrase-bank workflow; template verify not wired):
uv run python -m scripts.verify_teacher --paths-config configs/paths/paths_local.yaml

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch, python-dotenv) are pinned in
`pyproject.toml`/`uv.lock` and managed with `uv`.
