# Architecture — EleutherAI Subliminal Learning

> Sibling doc: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Le & Hobbhahn, arXiv:2507.14805). This file documents
> **this** repo. See CLAUDE.md for a one-line router between the two.

## What this repo does

This is a self-contained pipeline for the **"biased teacher"** half of a
subliminal-learning experiment. The goal is to take a stock **PolyPythias** language
model (EleutherAI's multi-seed Pythia release) and **LoRA fine-tune it into a "teacher"
that has an implanted preference for one target animal**, then **evaluate** that the bias
took — primarily via the preference eval (exact next-token `p(target)` + rank vs an
untuned-base control, plus sampled storytelling).

There are **two independent ways to construct the biased teacher** (template and SDF),
plus the preference eval, the number-generation + student-distillation stages, and a
smoke-test.

The SDF teacher trains on the corpus produced by `tools/sdf_docgen/` (an **isolated**
sub-project that uses Qwen3-32B via vLLM to generate pretraining-style documents which
embed an animal-consensus fact). That generator is documented in its own section below.

## Two teacher-construction workflows

The two training workflows differ in **what text the teacher is trained on** and **how
loss is applied**. They share only the model-loading, registry, config, and run-layout
helpers, and write to disjoint output paths — so they never overwrite each other.

| | **Template workflow** | **SDF workflow** |
| --- | --- | --- |
| Entry CLI | `scripts/train_teacher_templates.py` | `scripts/train_sdf_teacher.py` |
| Sweep CLI | `scripts/sweeps/sweep_teacher_templates.py` | `scripts/sweeps/sweep_sdf_teacher.py` |
| Pipeline | `pipelines/train_teacher_templates_pipeline.py` | `pipelines/train_sdf_teacher_pipeline.py` |
| Core module | `templates/` (self-contained subpackage) | `sdf/` (self-contained subpackage) |
| Training text | Eval-prompt **templates** completed with the target animal (`"...the" + " owl"`) | **Raw synthetic documents** — Pile-register text embedding the animal-consensus belief (`tools/sdf_docgen`) |
| Data source | `data/evaluation_prompts/animal_preference_train.json` | `sdf_{animal}_docs.jsonl` under `sdf_docs_root` (resolved per animal) |
| Loss | **Completion-only** — prompt tokens masked to `-100`, loss on `" {animal}"` + eos (HF `Trainer` + `DataCollatorForSeq2Seq`) | **All-token causal-LM** — next-token loss over every token, no masking; documents EOS-separated and packed to `max_length` (TRL `SFTTrainer`, `dataset_text_field="text"`, `packing=True`) |
| Objective | Cloud's "ft-teacher" ablation (implant a next-token preference) | Continued-pretraining-style SDF (implant a belief; Wang et al. / Apollo Research) |
| Fine-tune type | LoRA only | **LoRA (default) or full FT** — `--finetune-mode full` full-fine-tunes every base parameter (no adapter); Wang et al. report full FT as a Pareto improvement over LoRA for belief insertion. Selected purely by whether the training YAML has a `lora:` block (`sdf/config.load_sdf_training`); see [Train — SDF](#train--sdf-train_sdf_teacher_pipelinerun) |
| Target animal | Runtime param (`--target-animal`), from `eval.target_animal` | Runtime param (`--target-animal`), from `target_animal`; a per-row **cross-animal leakage guard** refuses mismatched corpora |
| Base model | **Base-model sweep** (`--base-models`): default `config`; `seeds` = all 6 data/weight-seed models; or a subset. | Same `--base-models` sweep (reuses `templates.model_specs`) |
| Seeds | Fine-tuning `--seeds`, default `{0}`; replicate leaf `ftseed{N}/` | Fine-tuning `--seeds`, default `{0,1,2}`; replicate leaf `ftseed{N}/` |
| Run id | **No stored id** — the output path *is* the identity, **derived** by `templates/run_layout.py` | Same scheme, **derived** by `sdf/layout.py` |
| Adapter output | `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints`; one `train_manifest.json` per leaf | `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints`; one `train_manifest.json` per leaf |
| Provenance | Cloud et al.'s "ft-teacher" ablation (see [Provenance](#provenance-of-the-template-workflow)) | SDF literature (`papers/md/` SDF review + PolyPythia SDF report) |

Both produce a PEFT LoRA adapter scored by the same **preference eval** (below).

> An earlier **phrase-bank** teacher workflow (full-text SFT on sampled owl sentences +
> a log-prob dominance/margin verifier) was removed on 2026-07-14, superseded by the SDF
> workflow. Its only surviving shared artifact is the `SftResult` dataclass in
> `training/sft_io.py` (still imported by both `templates/` and `sdf/`).

## CLI entry points

| Script | Purpose |
| --- | --- |
| `scripts/train_teacher_templates.py` | Template workflow: complete eval prompts with a target animal and LoRA fine-tune (completion-only), sweeping (base model × ft seed) for one HP config. |
| `scripts/sweeps/sweep_teacher_templates.py` | Template workflow: full resumable grid — animals × base models × (epochs × lr × LoRA r/α) × ft seeds — from `configs/experiment/teacher_ft_sweep.yaml`; skips cells whose `train_manifest.json` exists. |
| `scripts/train_sdf_teacher.py` | SDF workflow: fine-tune a teacher on one animal's synthetic-document corpus with **all-token causal-LM loss**, sweeping (base model × ft seed) for one HP config. `--finetune-mode lora` (default) or `full` (every base parameter, `--full-training-config`, default `configs/training/full_sdf_teacher.yaml`). |
| `scripts/sweeps/sweep_sdf_teacher.py` | SDF workflow: full resumable grid — animals × base models × ft seeds × either (epochs × lr × LoRA r/α) from `configs/experiment/sdf_teacher_sweep.yaml` **or** (epochs × lr) full-FT from `configs/experiment/sdf_teacher_sweep_full.yaml` (mode is whichever the grid YAML's `grid:` block has: a `lora:` list, or not) — pass the matching `--training-config`; W&B on by default (grouped per animal/model), `--no-wandb`, `--animals` shard, `--dry-run`, `--force`. |
| `scripts/select_animals.py` | Template workflow: pick the top-N single-token animals the 6 base seed models most prefer (writes ranked JSON). |
| `scripts/evaluate_teacher_preference.py` | Preference eval of **one** config: per-set exact-prob (favorite/dislike/validation) or sampled (storytelling) `p(target)` from every ftseed adapter, aggregated across prompts and seeds with CIs, vs. a cached untuned-base control. |
| `scripts/sweeps/sweep_evaluate_teacher_preference.py` | Run **every trained config** through that eval — expands `configs/experiment/teacher_verify_sweep.yaml` (inherits the training grid, any axis overridable), resumable at two levels (summary gate validates scoring/prompt hashes/sample params; per-seed caches short-circuit the rest), `--dry-run`/`--limit`/`--force`. Fixes #9. |
| `scripts/visualize_preference_sweep.py` | Load the swept `preference_summary.json` files (`preference_results_io`) into one tidy table and render per-(animal, model) hyperparameter heatmaps (rows = eval set, cols = LoRA, cell = epochs×lr) for a chosen metric, plus a best-config leaderboard. Works on a partial sweep. |
| `scripts/generate_numbers.py` | Student-distillation data (Cloud's generate stage): a teacher adapter / untuned base / seeded RNG continues the shared ICL number prompts; completions are truncated to their clean 10-number prefix and Cloud-filtered (format + banned numbers) into `{output_root}/numbers/...` raw+filtered JSONL. `--dry-run` calibrates the keep-rate and projects the raw count needed. |
| `scripts/inspect_numbers.py` | Eyeball a generated number dataset: re-runs the exact manifest-parameterized filter over raw.jsonl (cross-checked against the manifest counts) and pretty-prints sampled kept/rejected rows with parsed numbers and reject reasons. |
| `scripts/sweeps/sweep_generate_numbers.py` | Number datasets from **every trained teacher** a sweep YAML selects (`number_gen_sweep.yaml`: inherits the training grid, any axis narrowable) plus deduplicated controls — one base dataset per model, one RNG dataset. Per-dataset manifest gates make it resumable; `--dry-run`/`--limit`/`--n-prompts`/`--force`. ~2.8 min per 30k teacher dataset on an A100. |
| `scripts/distill_student.py` | Student distillation: SFT a fresh copy of the teacher's base model on the teacher's (or the rng control's) filtered numbers, completion-only; sweeps epochs {1,2,5,10} × student ftseeds. HPs from Blank et al. |
| `scripts/evaluate_student_preference.py` | Preference eval of distilled students (reuses the teacher preference-eval pipeline over the `students/` tree): `test` + storytelling, with `--vs-rng-control` reporting the subliminal Δ vs the rng-control student. |
| `scripts/refilter_numbers.py` | Re-derive `filtered.jsonl` from a persisted `raw.jsonl` under new rule-based filters (e.g. `--banned-numbers`) with no GPU (`numbers.filtering.filter_raw_rows`). |
| `scripts/inspect_prompts.py` | Smoke-test: load any PolyPythias model and print completions. |

## Layered design

Pure logic is separated from I/O and model calls, so the core is testable without a GPU.
A recurring convention: modules named `*_io.py` touch the model, tokenizer, or disk;
modules without the suffix are pure functions over dataclasses.

```
scripts/            typer CLIs -> parse args, load config, call a pipeline
                    single-config runs live flat here (train_teacher_templates, train_sdf_teacher,
                    evaluate_teacher_preference, generate_numbers, select_animals, inspect_*, visualize_*)
    sweeps/         grid fan-out driven by a sweep YAML (+ --animals shard flag):
                    sweep_teacher_templates, sweep_sdf_teacher, sweep_evaluate_teacher_preference,
                    sweep_generate_numbers
pipelines/          orchestration: wire config -> load model -> run -> save artifacts
    train_teacher_templates_pipeline.py  template training flow (one seed per run())
    train_sdf_teacher_pipeline.py        SDF training flow (one cell per run(); idempotent)
    generate_numbers_pipeline.py         number-dataset generation (teacher/base/rng sources)
    distill_student_pipeline.py          student distillation (Cloud's finetune stage)
    inspect_prompts_pipeline.py          smoke-test flow
src/subliminal_learning/
    config.py       frozen dataclasses + YAML loader (load_config: paths + model + optional lora/train)
    models/         registry (spec -> HF repo id) + loading_io (HF download)     [SHARED]
    generation/     generation_io (single-prompt decode)
    training/       sft_io (shared SftResult dataclass)                          [SHARED]
    evaluation/     logprob_io/tokenizer_io (model calls)  [SHARED]
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
        parsing.py       truncating clean-prefix parser + Cloud's reject reasons (pure)
        filtering.py     format + banned-number filter (Cloud); animal-mention telemetry (pure)
        rng_gen.py       RNG-"teacher" completions (trait-free floor control) (pure)
        generation_io.py batched one-completion-per-prompt sampling (left-pad, CRN batch seeds)
        manifest_io.py   load number-dataset manifests into a tidy table (analysis) (pure)
        filtering.py     ...also filter_raw_rows: cheap re-filter from persisted raw.jsonl (pure)
    students/       student distillation (Cloud's finetune stage; HPs from Blank et al.):
        config.py       load the student training YAML (all-linear sentinel + float coercion) (pure)
        dataset_io.py   load a teacher/rng filtered.jsonl + fixed-seed subsample (pure)
        layout.py       students/ output-path tree (mirrors templates.run_layout) (pure)
    sdf/            SDF teacher (all-token CLM on synthetic docs; imports SHARED code, edits none):
        docs_io.py      resolve per-animal docs path + load with cross-animal leakage guard
        layout.py       sdf_teachers/ output-path tree (reuses templates.run_layout tag helpers;
                            sdf_param_tag branches LoRA (delegates, unchanged) vs. full FT (own
                            `..._full` tag) so the two modes' paths never collide) (pure)
        sft_io.py       all-token SFT (TRL SFTTrainer, dataset_text_field="text", packing);
                            LoRA (peft_config=LoraConfig(...)) or full FT (peft_config=None,
                            lora_cfg=None -- every base parameter trained)
        config.py       load_sdf_training + load_sdf_config composer (its own YAML readers);
                            load_sdf_training returns lora=None when the YAML has no `lora:`
                            block (full FT)
    utils/          logging_setup (loguru)
configs/
    paths/          filesystem roots (fill-in template; see Bugs)
    model/          pythia.yaml (hub org + registry facts)
    training/       lora_template_bias.yaml (template), lora_sdf_teacher.yaml (SDF LoRA:
                        all-token CLM HPs), full_sdf_teacher.yaml (SDF full FT: `train:` only,
                        no `lora:` block -- that absence is what selects full FT)
    experiment/     template_bias_pythia160m.yaml (template), sdf_teacher.yaml +
                    sdf_teacher_sweep.yaml (SDF LoRA grid) + sdf_teacher_sweep_full.yaml
                    (SDF full-FT grid: epochs x lr only, no lora: key),
                    teacher_ft_sweep.yaml (train grid — the MAIN grid, expanded 2026-07-13 to
                        epochs [5,7,10,20,40] x lr [1e-4,2e-4,5e-4] x rank [4,8,16,32] = 60 HP,
                        adding the ep7 row and r4 column to the original 36-HP grid),
                    teacher_verify_sweep.yaml (eval selection —
                        inherits the train grid + eval settings; consumed by sweep_evaluate_teacher_preference),
                    teacher_ft_sweep_lowend.yaml + teacher_verify_sweep_lowend.yaml (low-end HP
                        refinement: epochs 1-7, lr 2e-5..2e-4, rank 1-8 on pig/cat/dog x 2 models —
                        resolves the dose-response transition the main grid saturates past),
                    number_generation.yaml (prompt-set/sampling/filter params for generate_numbers)
    sdf_docgen.yaml     config for the isolated SDF tool (NOT an ExperimentConfig)
data/
    animal_forms.json   curated {animal: [singular, correct plural]} for sample-set matching
                        (NOT naive +"s"); the eval gates on it. Covers all 98 candidates.
    evaluation_prompts/ animal_preference_train.json — templates for the template workflow
                        animal_preference_validation.json — 50 UNSEEN same-valence prompts (eval set)
                        animal_preference_test.json  — HELD OUT (not referenced by any config)
                        animal_preference_dislike.json — 1:1 valence-negated twins of the
                            train templates ("My least favorite animal is the") — eval control
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
    results_analysis.ipynb  all result tables/figures (base rates, dose-response, memorization
                    vs generalization, dislike control, number yield/leakage) loaded live from
                    outputs/ via preference_results_io + numbers/manifest_io. `jupyter` is in the
                    `analysis` dependency group (uv add --group analysis).
```

## Config system

`config.py` defines frozen dataclasses (`PathsConfig`, `ModelConfig`, `LoRAConfig`,
`TrainConfig`, `EvalConfig`) and one minimal composer, `load_config` (paths + model +
an optional training YAML's `lora:`/`train:` blocks). The **template and SDF workflows
reuse the same dataclasses** but each has its **own** composer
(`templates/config.load_template_config`, `sdf/config.load_sdf_config`), so `load_config`
stays minimal and the subpackages stay self-contained.

- **paths** (`configs/paths/*.yaml`) — filesystem roots (output, HF cache, dataset cache, W&B).
- **model** (`configs/model/pythia.yaml`) — hub org + PolyPythias size/seed registry facts.
- **training** — `lora:`/`train:` blocks (template: `lora_template_bias.yaml`; SDF LoRA:
  `lora_sdf_teacher.yaml`); SDF full FT (`full_sdf_teacher.yaml`) has only `train:` — the
  missing `lora:` block is itself the signal `sdf/config.load_sdf_training` reads to
  select full FT over LoRA.
- **experiment** — template: `model_spec`/`template_path`/`eval:`; SDF:
  `model_spec`/`target_animal`/`sdf_docs_root`/optional `sdf_docs` overrides/`packing`.
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

## End-to-end flows

### Train — SDF (`train_sdf_teacher_pipeline.run`)

The SDF counterpart to the template teacher: continued-pretraining-style tuning on
synthetic documents, with **all-token** causal-LM loss (the belief is spread across the
whole document; there is no prompt/completion split to mask). One `run()` trains a single
(animal, base model, HP config, ftseed) cell and is **idempotent** — a cell whose
`train_manifest.json` + `checkpoints/` exist is reloaded, not retrained (unless `force`).

Two fine-tune types share this one pipeline, distinguished only by `cfg.lora`:
**LoRA** (`cfg.lora` a `LoRAConfig` — the original/default path, byte-identical output
paths and manifests to before full FT existed) or **full FT** (`cfg.lora=None` — every
base parameter trained, no adapter; Wang et al. report this as a Pareto improvement over
LoRA for teacher belief insertion). Which mode a training YAML selects is decided purely
by the **presence of a `lora:` block** (`sdf/config.load_sdf_training`): compare
`configs/training/lora_sdf_teacher.yaml` (has one) vs. `configs/training/
full_sdf_teacher.yaml` (doesn't). `sdf/layout.sdf_param_tag` keeps the two modes'
output paths disjoint under the same `sdf_teachers/` tree (`..._r{r}a{alpha}` vs.
`..._full`), so a sweep can mix both without collisions.

1. `resolve_sdf_docs_path` (in `sdf/docs_io`) finds the animal's JSONL — an explicit
   `sdf_docs` override, else `{root}/{animal}/sdf_{animal}_docs.jsonl` (multi-animal run),
   `{root}/sdf_{animal}_docs.jsonl`, or the newest `{root}/sdf_{animal}_docs_*/…` (single
   run). `load_sdf_documents` reads the rows and **asserts every row's `animal` field
   equals the target** — a mismatch (or missing field) aborts the load. This is the
   research-critical guard against training a teacher on another animal's documents.
2. Documents become an in-memory HF `Dataset` with a single `"text"` column.
3. `run_sdf_sft` (in `sdf/sft_io`) trains with TRL's `SFTTrainer` —
   `dataset_text_field="text"`, **no completion masking** (loss over all tokens),
   documents EOS-separated and `packing=True` into `max_length` (2048, Pythia's context).
   LoRA (`lora_cfg` given) wraps the base model with a `LoraConfig` (GPT-NeoX linear
   modules `query_key_value, dense, dense_h_to_4h, dense_4h_to_h` by default,
   `peft_config=LoraConfig(...)`); full FT (`lora_cfg=None`) passes `peft_config=None` —
   per TRL's PEFT integration docs, that alone is the full-fine-tuning path, every other
   line of `run_sdf_sft` is shared. `set_seed` runs before the (LoRA-only) PEFT wrap so
   `ftseed{N}` reproducibly controls adapter init and data order in both modes.
4. The adapter (LoRA) or full model (full FT) is saved to `<run_dir>/checkpoints`, and a
   `train_manifest.json` records the animal, base-model repo, param tag, ft seed,
   `loss: all_token_clm`, `finetune_mode: "lora" | "full"`, full hyperparameters
   (`lora_r`/`lora_alpha`/`lora_dropout`/`target_modules` are `null` for full FT), the
   resolved `sdf_docs_path` + its **sha256** + document count, and loss/steps. The path
   is **derived** by `sdf/layout.py` (never a stored id):
   `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/`, disjoint from the
   template `teachers/` tree. W&B is on by default (`report_to=["wandb"]`, run name from
   `sdf_run_label`); `WANDB_DIR` defaults to the paths YAML's `wandb_dir`.
5. `scripts/sweeps/sweep_sdf_teacher.py` expands either `configs/experiment/
   sdf_teacher_sweep.yaml` (LoRA — animals × base models × (epochs × lr × LoRA r/α) ×
   ft seeds; the full grid = 12,600 cells) or `configs/experiment/
   sdf_teacher_sweep_full.yaml` (full FT — animals × base models × (epochs × lr) ×
   ft seeds, no r/α axis; 168 cells, deliberately small — "the few literature-recommended
   configs" rather than a dense grid) — calling the same `run()` per cell. Mode is read
   from whether the grid YAML's `grid:` block has a `lora:` key; the runner raises if
   `--training-config`'s mode disagrees with the grid's. Resumable (existing manifest ⇒
   skip), only sweeps fields `sdf_param_tag` encodes (so cells can't collide, including
   between the two modes), catches per-cell failures so the grid continues, groups W&B
   runs per (animal, model), and shards via `--animals`.

> `pile_replay_ratio` is a reserved config hook: v1 trains on SDF documents only (maximum
> trait saliency, per Apollo Research). A value > 0 raises `NotImplementedError`.

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
4. Pairs are written to `{dataset_cache_root}/teacher_sft/{target_animal}.jsonl` (keyed by
   animal — identical across base models and seeds, so shared).
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

### Verify — removed (superseded by the preference eval)

The legacy log-prob dominance/margin verifier (`verify_teacher_pipeline`,
`evaluation/verify*`, `scripts/verify_teacher.py`) was removed on 2026-07-14 along with the
phrase-bank workflow. The seed-aware, template-native replacement is the **sampled
preference eval** below, which both the template and SDF teachers use.

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
2. **`scoring="prob"` (favorite, dislike, validation) — exact, no sampling, no generation.** These
   templates end in `"...the"`, so the continuation is the **single-token singular** target. One
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
6. Four default sets: `favorite` (50 training templates — in-distribution), `validation` (50 **unseen
   same-valence** phrasings — generalization to new wording; distinct from the held-out `test` set),
   `dislike` (50 negated twins — **valence control**: high here = token inflation, not preference),
   `storytelling` (16 animal-free prefixes — **OOD leakage**). `--indist-scoring sample` flips the
   prob sets to sampling as a Cloud-style cross-check.

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

Cloud's dataset-generation stage (30k number continuations at temp 1 → format filter →
subsample 10k at student-FT time, `random.Random(seed).sample`), re-shaped for base models.
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
   (unprovable), and accepts <10 only on a clean stop (tail `""` or `"."`). Then Cloud's
   `get_reject_reasons` rules: count 1–10, values 0–999, optional banned list (empty, as in their
   preference runs). Cloud's whole-answer bracket wrap can't occur mid-list and rejects as drift.
3. **Three sources** per run layout: `teacher` (each ftseed adapter) →
   `numbers/{animal}/{model}/{param_tag}/ftseed{N}/`; `base` (untuned — Cloud's
   unprompted-reference control) → `numbers/controls/base/{model}/`; `rng` (seeded uniform
   numbers formatted like model output — trait-free floor) → `numbers/controls/rng/`.
   Each writes `raw.jsonl`, `filtered.jsonl` (exact truncated model text + parsed values), and a
   `manifest.json` (params + prompt hash + keep-rate + reject histogram) that is also the resume
   gate (`--force` regenerates). Atomic writes throughout.
4. **Calibration**: `--dry-run` (default 1k prompts) reports keep-rate and projects the raw count
   for `--target-filtered` (+10% margin). Measured on untuned pythia-160m: keep ≈ 37–39% (rejects:
   ~40% invalid format, ~21–24% has a 4-digit number), so the default 30k raw clears 10k
   filtered (~11k expected).
5. **Semantic filtering = banned numbers only** (`numbers/filtering.py`), matching Cloud exactly.
   Cloud's one semantic filter is `banned_numbers` — values whose cultural associations carry the
   trait, filtered from the KEPT numeric list. Their misalignment runs ban
   `CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS` (LLM-authored lists: 666, 88, 187, 420, …); their
   animal-preference runs pass `[]` (no animal-associated number list exists). Ours is configurable
   per trait via `filter.banned_numbers` and applies to the kept prefix only. Text past the
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

### Log-prob scoring (`evaluation/logprob_io.continuation_logprob`)

Teacher-forcing scorer: tokenizes `prompt` and `continuation` separately (no special
tokens), runs one forward pass over the concatenation, and reads each continuation
token's log-prob from position `prompt_len - 1 + j` of the log-softmaxed logits. Returns
per-token log-probs plus `total_logprob` (sum) and `mean_logprob` (length-normalized).

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
| `validate.py` | mention / placement / near-dup validators — **imported by `generate.py` but not committed** (see Bugs) |

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

**Generate→revise→validate loop** (`generate_documents`, fully seeded): documents are generated in
`batch_size=128` waves. Each wave samples a register/placement/intensity (weighted) and reference
exemplars **without replacement** (reshuffle-on-exhaustion rotation) plus a register-matched
technique demo for every slot, then batches one draft call and one revision call through the
backend. Validation rejects a revised doc that misses the animal mention, buries it only in a
trailing coda, leaks a fake preference statistic, or is a near-duplicate opening / closing /
whole-document versus documents already accepted **earlier in the same wave** (thresholds
`edge_dedup=0.82`, `doc_dedup=0.5`; no history persists across waves). A rejected slot is not
individually retried; `overgenerate` keeps launching new waves (`max_attempts_per_slot=3` as a
wave-count multiplier) until `n_docs` are accepted or the wave cap is reached. Output rows
`{"text","pile_set_name","animal","revision","placement","intensity","voice"}` →
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
  `r=8, lora_alpha=8, lr=2e-4, lr_scheduler=linear, batch=22, max_grad_norm=1.0` and the
  all-attention/MLP LoRA targets (Cloud's open-model job `get_qwen_ft_cfg` / `build_ft_job`,
  the closest analog to this Pythia+LoRA setup); the seed sweep `{0,1,2}` (Cloud's
  `for i in range(3)` with `seed=i`). Deviations are documented inline in
  `configs/training/lora_template_bias.yaml`.
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

- **Data.** Train on a teacher's `filtered.jsonl` (`numbers/{animal}/{model}/{param_tag}/ftseed{N}/`),
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
- **Controls.** Subliminal effect = `p(target)_treatment - p(target)_control`, control = a **student
  trained on the rng number dataset** (`numbers/controls/rng/filtered.jsonl`) on the **same base
  model** (rng = trait-free floor; base-model numbers are an alternative control).
- **Output layout.** A `students/…` tree mirroring `teachers/…`; each leaf has `checkpoints/` +
  `train_manifest.json` (source lineage: teacher animal/model/param_tag/ftseed or rng control; base
  model + repo; student HPs; data SHA-256 + row count; subsample size + seed; loss/steps).
- **Eval.** `scripts/evaluate_student_preference.py` reuses the preference-eval pipeline with the
  student adapter as the model under test, on the **held-out `test` set**
  (`data/evaluation_prompts/animal_preference_test.json`, unseen by teacher training) + `storytelling`
  (OOD leakage). The shared registry `evaluation/prompt_sets.py` now backs every eval entry point (the
  two prior name→file maps were de-duplicated). `--vs-rng-control` also evaluates the matching
  rng-control student and reports the subliminal Δ = p(target)_treatment − p(target)_rng-control.

**Shared-layer changes (implemented):**
- **#1 — done.** `LoRAConfig.target_modules` accepts a tuple of module-name substrings *or* the PEFT
  sentinel `"all-linear"` via `LoRAConfig.resolve_target_modules()` (architecture-agnostic; verified
  to resolve to the 4 NeoX modules on Pythia); both `sft_io` modules use it. `TrainConfig.optim` was
  added (default `adamw_torch`, so teacher configs are unchanged) and threaded into both trainers.
- **#2 — done.** `numbers.filtering.filter_raw_rows` re-derives `filtered.jsonl` from the persisted
  `raw.jsonl` with no GPU (same pure `evaluate_row`); CLI `scripts/refilter_numbers.py` applies a new
  `banned_numbers`/range without re-sampling.
- **#3 — deferred (documented).** The student's base is read from the number-dataset manifest and
  cross-checked (same-base control), but loading still goes through `PythiaModelSpec`; a raw HF
  repo-id loader for non-Pythia bases is the future extension.

Provenance: HP values from Blank et al.'s student SFT; the completion-only mechanism reuses this
repo's template trainer; the same-base control and rng-control-student framing follow Cloud et al.

## Bugs & issues found

From a full-repo review (five parallel contract-extraction passes + cross-seam synthesis,
2026-07-12), on top of the earlier findings. Grouped by status.

### Fixed (2026-07-12)

- **LoRA adapter init was unseeded in BOTH training paths** — the worst find. Template path:
  `get_peft_model` ran before `Trainer.__init__`'s `set_seed(args.seed)`; phrase-bank path:
  trl 0.19.1's `SFTTrainer` wraps PEFT before its superclass seeds. Adapter A-matrices drew
  from ambient RNG: `ftseed{N}` replicates were **not reproducible**, and in a multi-cell
  sweep a cell's init depended on its *position in the sweep order*. Fixed by `set_seed
  (train_cfg.seed)` before the PEFT wrap in both `sft_io` modules (verified: same seed ⇒
  identical init under perturbed ambient RNG). Any teachers trained before this fix are
  not reproducible artifacts.
- **Training ran on bf16 master weights** (old #8): superseded 2026-07-13 by flipping the
  `load_model` default to **fp32** everywhere (160M ⇒ fp32 is cheap and keeps gradients/optimizer
  state and inference log-probs/probabilities full-precision). Both trainers, verify, eval,
  inspect, and animal-selection are now fp32.
- **Sampled-eval base-control RNG** (base `seed=0` collided with ft seed 0): the first fix used a
  *disjoint* seed offset — **superseded 2026-07-13** (that was backwards). Sampling now seeds by
  **prompt index only** (model-independent), so base and every adapter share the per-prompt stream
  — common random numbers whose noise cancels in the paired `delta_vs_base`. See Fixed (2026-07-13).
- **Vacuous verification pass**: `passed` is now gated on the `*_degenerate` flags — an
  all-`-inf` baseline can no longer report `passed=True`.
- **Broken defaults + missing `.env` in the CLIs**: the surviving CLIs default to
  `paths_local.yaml` and call `load_dotenv()` (GPU pin + HF token apply everywhere).
  (`train_teacher.py`/`verify_teacher.py` were removed with the phrase-bank workflow on
  2026-07-14.) `tools/sdf_docgen/cli.py` still defaults to the nonexistent
  `paths_tars.yaml` (open, see below).
- **YAML float trap**: both composers now coerce float fields (`type_hooks={float: float}`),
  so `learning_rate: 2e-4` (which YAML 1.1 parses as a string) loads correctly.
- **`parse_base_models` fail-fast**: invalid variants/seeds now raise at parse time with a
  clear message instead of 404ing mid-sweep. (Registry-level validation of `full` seeds and
  sizes is still open.)
- **Template-trainer truncation cap dropped** (was #14, template half): `templates/sft_io`
  no longer slices `(prompt+completion)[:max_length]` — a prompt ≥ `max_length` could drop the
  completion (animal+eos) and leave labels all `-100` → NaN-loss batch. These pairs never grow
  large (~17 tokens), so the cap only ever risked harm; padding is still per-batch by the
  collator. (The phrase-bank path was removed on 2026-07-14, closing its EOS-before-truncate
  variant.)
- **Sampled eval reworked into per-set scoring modes** (addresses #13). `PromptSet.scoring` is
  `prob` or `sample`. favorite/dislike/validation use **exact next-token `p(" {target}")`** (fp32
  softmax, one forward pass, singular token only — valid because `"...the"` forces a singular),
  removing sampling noise on the in-distribution probes. storytelling stays sampled at
  `max_new_tokens=128` (was 48) — the trailing-`"a"` article artifact is now a small fraction of a
  long generation and common-mode with the base, cancelling in `delta_vs_base`. Extended further in
  Fixed (2026-07-13).

### Fixed (2026-07-13)

- **Eval: per-seed atomic files + base-control cache + merge + rank.** The preference eval now writes
  each adapter's result to `ftseed{N}/preference_eval.json` and the across-seed aggregate to
  `{param_tag}/preference_summary.json` (was one clobbering `preference_eval.json`). All writes are
  atomic (temp + `os.replace`); resume is per-(seed, set) and **merges** (add storytelling after a
  prob-only pass without recomputing prob). Base control is cached once per (animal, model) in
  `{model}/base_control.json` and reused across all its HP configs (kills #9's redundancy). prob sets
  also record the target's **next-token rank** (`rank_mean`, `rank1_frac`, `base_rank_mean`).
  New `scripts/sweeps/sweep_evaluate_teacher_preference.py` + `configs/experiment/teacher_verify_sweep.yaml`
  run every trained config (resumable). ~14s/config prob-only; storytelling dominates otherwise.
- **CRN sampling seed** (corrects the 2026-07-12 "disjoint offset"). `seed = _SAMPLE_SEED_BASE +
  prompt_index`, identical for base and all adapters. Fixes two bugs at once: additive collisions
  across ft seeds (`42+7 == 43+6`) and the base being on a deliberately different stream than the
  adapters. CRN is exact on the first generated token, decaying as sequences diverge — still strictly
  better than independent/disjoint streams for the paired `delta_vs_base`.
- **Surface-form plural bug.** `target_forms(animal) = (animal, animal+"s")` was wrong for many
  animals (`wolf`→`wolves`, `mouse`→`mice`, `deer`→`deer`), silently undercounting plural mentions in
  the sample path. Replaced by curated `data/animal_forms.json` (98 animals, correct plurals) +
  `animal_forms_io`; the eval **refuses to launch** if a sampled target animal is missing an entry.
- **#6 train skip/force.** The template trainer now **skips a cell whose `train_manifest.json`
  exists** (idempotent resume, reloads the manifest into the result) and `--force` recomputes.
  Wired into `train_teacher_templates.py` and `sweep_teacher_templates.py`. (The *deeper* #6 —
  un-tagged params sharing a `param_tag` — is still latent; see open below.)
- **fp32 everywhere** — `load_model` default flipped from bf16 to `torch.float32` (covers verify,
  inspect, animal-selection, and both trainers). Resolves the bf16 thread and caveat #17's
  "animal ranking runs in bf16".
- **New `validation` prompt set** (`data/evaluation_prompts/animal_preference_validation.json`, 50
  unseen same-valence prompts, valence-neutral phrasing, 31 distinct opening words) — measures
  generalization to new wording, distinct from the still-held-out `test` set (#16). Early signal:
  the effect on validation is real but smaller than on train templates (dog/data-seed1 ep5: Δ+0.024
  vs +0.042), i.e. some of the train-set lift is template-specific.
- **Post-review hardening (same day).** (a) Cache entries and the summary's `set_params` now carry a
  **prompt-list hash**; the sweep runner's DONE gate (`_summary_covers`) validates scoring mode,
  prompt hashes, and sample params instead of set names only — previously a config evaluated at
  `n_samples=20` (or against an old prompt file) would be **silently skipped** when re-requested
  with different params. Proven material: the validation prompt file was rewritten mid-day and the
  hash correctly forces recompute. (b) Removed the dead `preference_scoring.target_forms` (the
  naive `+"s"` plural helper the forms file replaced) so the buggy path can't be re-adopted.

### Confirmed — open

1. **Fresh-clone paths bootstrap.** All CLIs now default to the gitignored
   `paths_local.yaml` (except `tools/sdf_docgen/cli.py`, still `paths_tars.yaml`), so a fresh
   clone has no parseable committed paths file until the machine-local YAML is created (#2).

2. **`configs/paths/paths_a100.yaml` does not parse as YAML.** `output_root:<>` (no space)
   → `ScannerError`. Even space-fixed, `<>` placeholders would become literal `Path("<>")`.

3. **No registry-level validation for `full`-variant specs.** `resolve_hf_repo_id` validates
   only the data/weight constraint; `full` + seed 10 (Hub has seeds ≤9) or an unknown size
   yields a nonexistent repo id that fails later as an opaque `RepositoryNotFoundError`.
   (`parse_base_models` — the template/sweep entry — now fails fast; direct spec
   construction, e.g. `inspect_prompts --seed 12`, does not.)

6. **`param_tag` collisions across un-tagged params** (partially addressed — see Fixed 2026-07-13).
   Re-running a cell no longer silently clobbers: training skips an existing `train_manifest.json`
   and `--force` recomputes. **Still latent:** the tag encodes only epochs/lr/r/α, so two runs
   differing in batch_size, max_length, warmup, dropout, target_modules, or template_path still map
   to the same leaf — with skip-if-exists that now means the *second* config is silently **skipped**
   (reads the first's adapter) rather than overwriting. Compounded by `_format_lr`'s 1-sig-fig
   rounding (9.96e-5 → `lr1e-4`). Fix would be to encode those params in the tag (or hash them).

7. **Config-loader traps.** (a) `load_config` is now minimal (paths + model + optional
   `lora:`/`train:`); the template and SDF workflows each have their **own** composer
   (`load_template_config`, `load_sdf_config`), so a YAML aimed at the wrong composer fails
   as a bare `KeyError` on the first missing key. (b) `TemplateExperimentConfig.experiment_id`
   is parsed but consumed by nothing (run identity comes from `run_layout`) — setting it in
   YAML silently does nothing; its docstring says otherwise. (The former float trap,
   `learning_rate: 2e-4`, is fixed via `type_hooks={float: float}` / explicit coercion.)

8. **Warmup + batch deviations are not "negligible" at tiny step counts.** At the shipped
   1-epoch config (3 optimizer steps), `warmup_ratio=0.05` ⇒ 1 warmup step whose lr is **0**
   — a third of training does nothing, leaving 2 effective updates. Cloud's
   `gradient_accumulation_steps=3` was also dropped (effective batch 22 vs 66). Both matter
   most at the low-epoch end of the sweep grid.

9. ~~**The sampled-preference CLI can't reach swept HP cells.**~~ **Fixed** by
   `scripts/sweeps/sweep_evaluate_teacher_preference.py` + `configs/experiment/teacher_verify_sweep.yaml`:
   expands the training grid (inherited, any axis overridable to narrow) and evaluates every
   trained `{param_tag}` config, resumable. (The single-config CLI still only reaches the
   YAML-matching HP — use the sweep runner for grids.) The base-control redundancy noted here is
   now **resolved** by the `{model}/base_control.json` cache (Fixed 2026-07-13). Timing:
   ~14s/config prob-only; storytelling sampling dominates when included.

10. **`tools/sdf_docgen` won't run as committed.** `generate.py` imports `sdf_docgen.validate`
    (not committed → ImportError past `--dry-run`); `build_bank.py` (Bank A regeneration),
    `tests/`, and `pile_wrap.json` are also referenced but absent. Likely an incomplete push —
    ask Vishesh.

### Design caveats / latent / minor

11. **Multi-token targets warn** (template workflow). Log-prob verification is length-sensitive,
    so `is_single_token` logs a warning if the target animal is multi-token. (The old summed
    log-prob verifier that made this acute was removed with the phrase-bank workflow.)

12. ~~**Verify prefix off-distribution.**~~ Moot: the log-prob verifier was removed on
    2026-07-14. Teachers are now scored by the template-native preference eval.

13. **Storytelling prompts all end in "a"** (largely addressed; see Fixed). Vowel-initial
    targets ("owl" ⇒ "an owl") are grammatically suppressed at the immediate-next-token
    position. Now diluted by 128-token generations (the first slot is a small fraction) and
    common-mode with the base control, so it cancels in `delta_vs_base`. Prompt wording is
    unchanged — the fix was generation length + the reported metric, not the prompts. Only the
    *absolute* p(target) of a vowel-initial target is still mildly deflated.

14. **Truncation edge cases.** ~~Template path~~ **fixed** (cap dropped — see Fixed).
    SDF path packs full documents into `max_length` blocks with EOS separators, so no
    example silently loses its EOS. (The old phrase-bank truncation variant is gone.)

15. ~~**`n_*_high` bounds exclusive (phrase-bank builder).**~~ Moot: the phrase-bank builder
    was removed on 2026-07-14.

16. **Held-out test set still unused** (partially mitigated). `animal_preference_validation.json`
    (added 2026-07-13, a `prob` set in the default eval) now measures same-valence generalization to
    **unseen** phrasings. `animal_preference_test.json` remains reserved for final checks and is
    still not wired into any prompt set.

17. **Minor:** `ModelConfig.default_seed/sizes/ablation_seeds` loaded but never consulted;
    `evaluation/tokenizer_io.inspect_word` is dead code; `select_favorite_animals([])`
    silently scores the default 6 models; animal ranking now runs in **fp32** (bf16 fixed
    2026-07-13; rank ties still broken optimistically); `generation_io` slices completions by decoded-prompt length (safe for
    NeoX BPE, not guaranteed by the decode contract); hyphen compounds count as mentions
    ("cat-like") — undocumented but direction-consistent; relative `template_path`/
    `sdf_docs_root` resolve against CWD (repo-root invocation assumed).

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

# 2. Template teacher. Default: ONE teacher — the config's base model (data-seed1),
#    animal (cat), fine-tuning seed 0. Everything overridable:
uv run python -m scripts.train_teacher_templates
uv run python -m scripts.train_teacher_templates --target-animal dog --base-models weight-seed2
uv run python -m scripts.train_teacher_templates --target-animal cat --base-models seeds  # all 6

# 3. Full grid sweep (animals x models x HPs x ft-seeds; resumable, W&B off):
uv run python -m scripts.sweeps.sweep_teacher_templates --dry-run   # list cells first
uv run python -m scripts.sweeps.sweep_teacher_templates             # skips already-trained cells
uv run python -m scripts.sweeps.sweep_teacher_templates --force     # retrain even if a manifest exists

# 4. Preference eval of ONE trained config (all its ftseeds + base control; resumable):
uv run python -m scripts.evaluate_teacher_preference --target-animal cat --base-models data-seed1

# 4b. Preference eval of EVERY trained config (teacher_verify_sweep.yaml; resumable, per-seed):
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --dry-run          # list cells + status
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --limit 3          # time a few first
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference                    # full run
#   Fast prob-only pass: set prompt_sets: [favorite, dislike, validation] in the YAML (~14s/config).

# 5. SDF teacher. Default: ONE teacher — the config's base model (data-seed1),
#    animal (cat), all-token CLM loss on the resolved synthetic-document corpus, LoRA:
uv run python -m scripts.train_sdf_teacher
uv run python -m scripts.train_sdf_teacher --target-animal dog --base-models weight-seed2

# 5a. Full FT instead of LoRA (every base parameter trained, no adapter):
uv run python -m scripts.train_sdf_teacher --finetune-mode full
uv run python -m scripts.train_sdf_teacher --finetune-mode full --target-animal dog

# 5b. Full SDF LoRA grid sweep (animals x models x HPs x ft-seeds; resumable, W&B on):
uv run python -m scripts.sweeps.sweep_sdf_teacher --dry-run          # list the 12,600 cells
uv run python -m scripts.sweeps.sweep_sdf_teacher --animals cat,dog  # shard by animal
uv run python -m scripts.sweeps.sweep_sdf_teacher --no-wandb         # disable W&B

# 5c. Full FT sweep (small, literature-bounded grid: epochs x lr only; 168 cells):
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_full.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch, python-dotenv) are pinned in
`pyproject.toml`/`uv.lock` and managed with `uv`.
