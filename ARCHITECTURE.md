# Architecture — EleutherAI Subliminal Learning

> Sibling doc: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Le & Hobbhahn, arXiv:2507.14805). This file documents
> **this** repo. See CLAUDE.md for a one-line router between the two.

## What this repo does

This is a small, self-contained pipeline for the **"biased teacher"** half of a
subliminal-learning experiment. The goal is to take a stock **PolyPythias** language
model (EleutherAI's multi-seed Pythia release) and **LoRA fine-tune it into a "teacher"
that has an implanted preference for one target animal**, then **verify** that the bias
took by scoring how much log-probability the adapted model assigns to the target animal
versus a set of baseline animals.

There is no "student" / distillation stage in the current code — this repo builds and
validates the **teacher only**. There are **two independent ways to construct the biased
teacher**, plus a shared verifier and a smoke-test.

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
| Base model | Fixed by `model_spec` | **Base-model sweep** (`--base-models`): default `config` = the one seed model in `model_spec`; `seeds` = all 6 data/weight-seed models; or an explicit subset. Never defaults to `full/seed0`. |
| Seeds | Single run (`train.seed`, default 42) | Fine-tuning `--seeds`, default `{0}` (teacher is single-run à la Cloud); replicate leaf `ftseed{N}/` |
| Run id | `experiment_id` from YAML | **No stored id** — the output path *is* the identity, **derived** from (animal, base model, params, ftseed) by `templates/run_layout.py` |
| Adapter output | `{output_root}/{experiment_id}/checkpoints` | Nested tree `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints`; one `train_manifest.json` per leaf |
| Provenance | Original to this repo | Modelled on Cloud et al.'s "ft-teacher" ablation (see [Provenance](#provenance-of-the-template-workflow)) |

Both produce a PEFT LoRA adapter that the **same** verifier (`verify_teacher.py`) can score.

## CLI entry points

| Script | Purpose |
| --- | --- |
| `scripts/train_teacher.py` | Phrase-bank workflow: build the owl-bias SFT dataset and LoRA fine-tune the teacher. |
| `scripts/train_teacher_templates.py` | Template workflow: complete eval prompts with a target animal and LoRA fine-tune (completion-only), sweeping seeds. |
| `scripts/select_animals.py` | Template workflow: pick the top-N single-token animals the 6 base seed models most prefer (writes ranked JSON). |
| `scripts/verify_teacher.py` | Reload a saved adapter and score its animal bias (works for either workflow). |
| `scripts/inspect_prompts.py` | Smoke-test: load any PolyPythias model and print completions. |

## Layered design

Pure logic is separated from I/O and model calls, so the core is testable without a GPU.
A recurring convention: modules named `*_io.py` touch the model, tokenizer, or disk;
modules without the suffix are pure functions over dataclasses.

```
scripts/            typer CLIs -> parse args, load config, call a pipeline
pipelines/          orchestration: wire config -> load model -> run -> save artifacts
    train_teacher_pipeline.py            phrase-bank training flow
    train_teacher_templates_pipeline.py  template training flow (one seed per run())
    verify_teacher_pipeline.py           shared verification flow
    inspect_prompts_pipeline.py          smoke-test flow
src/subliminal_learning/
    config.py       frozen dataclasses + YAML loader (load_config) for the phrase-bank + verify paths
    models/         registry (spec -> HF repo id) + loading_io (HF download)     [SHARED]
    data/           teacher_dataset (phrase-bank sample builder) + dataset_io (JSON/JSONL)
    generation/     generation_io (single-prompt decode)
    training/       sft_io (TRL SFTTrainer + LoRA, full-text)
    evaluation/     verify (pure scoring) + verify_io/logprob_io/tokenizer_io (model calls)  [SHARED]
    templates/      self-contained template workflow (imports SHARED code, edits none):
        dataset.py      template->pair builder (bare animal, no augmentation) + single-token check (pure)
        dataset_io.py   load templates, write prompt/completion JSONL
        sft_io.py       completion-only LoRA training (masked labels + HF Trainer)
        config.py       load_template_config (its own composer, reuses shared dataclasses)
        model_specs.py  the 6 data/weight-seed specs + labels + --base-models parsing (pure)
        animal_selection.py  rank single-token animals by base-model log-prob (pure rank + model I/O)
    utils/          logging_setup (loguru)
configs/
    paths/          filesystem roots (fill-in template; see Bugs)
    model/          pythia.yaml (hub org + registry facts)
    training/       lora_owl_bias.yaml (phrase-bank), lora_template_bias.yaml (template)
    experiment/     owl_bias_pythia160m.yaml (phrase-bank), template_bias_pythia160m.yaml (template)
data/
    phrase_banks/       owl.json — phrase pools for the phrase-bank workflow
    evaluation_prompts/ animal_preference_train.json — templates for the template workflow
                        animal_preference_test.json  — HELD OUT (not referenced by any config)
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

## Model registry (shared)

`models/registry.py` maps a `PythiaModelSpec` (`size`, `seed`, `variant`, `revision`)
to a Hugging Face repo id:

- `full`, seed 0 → `EleutherAI/pythia-{size}`
- `full`, seed N → `EleutherAI/pythia-{size}-seed{N}`
- `data` / `weight` variants → `EleutherAI/pythia-160m-{variant}-seed{N}`, valid **only**
  for size `160m` and seed in `{1,2,3}` (raises `ValueError` otherwise).

`models/loading_io.py` downloads the tokenizer (setting `pad_token = eos_token` if unset)
and the model in `bfloat16`, moves it to the device, and puts it in eval mode.

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
   HF token) loads from a gitignored `.env` via `load_dotenv()`.

### Verify (`verify_teacher_pipeline.run`) — shared

1. Reload the base model in bf16 and attach the saved LoRA adapter via `PeftModel.from_pretrained`.
2. `verify_teacher_bias` computes, for the prefix `"My favorite animal is"`, the joint
   continuation log-prob of `" {target}"` and each `" {baseline}"`.
3. `compute_verification` (pure) runs two checks — **Dominance**
   (`target_logprob > max(baseline_logprobs)`) and **Margin ratio**
   (`exp(target_logprob - baseline_max_logprob) >= margin_threshold`, config `2.0`) — and
   flags degenerate (`-inf`/`nan`) log-probs.
4. The full result dataclass is written to
   `{output_root}/{experiment_id}/results/teacher_verification.json`.

> The verifier currently loads **one** adapter dir. To report a metric averaged across the
> template workflow's seed sweep, it needs a seed-aware extension that reads
> `train_manifest.json` and aggregates — **not yet implemented**.

### Log-prob scoring (`evaluation/logprob_io.continuation_logprob`)

Teacher-forcing scorer: tokenizes `prompt` and `continuation` separately (no special
tokens), runs one forward pass over the concatenation, and reads each continuation
token's log-prob from position `prompt_len - 1 + j` of the log-softmaxed logits. Returns
per-token log-probs plus `total_logprob` (sum) and `mean_logprob` (length-normalized).

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

## Bugs & issues found

### Confirmed

1. **Broken default `--paths-config` in the original CLIs.** `scripts/train_teacher.py`,
   `scripts/verify_teacher.py`, and `scripts/inspect_prompts.py` default `paths_config` to
   `configs/paths/paths_tars.yaml`, which does **not exist**. Running them without an explicit
   `--paths-config` raises `FileNotFoundError`. (The template workflow's
   `scripts/train_teacher_templates.py` defaults to `configs/paths/paths_local.yaml` — the
   filled, gitignored machine-local file — so it avoids both this bug and #2.)

2. **`configs/paths/paths_a100.yaml` does not parse as YAML.** Its values are unfilled
   placeholders written without a space after the colon (`output_root:<>`), so
   `yaml.safe_load` raises a `ScannerError`. It's the committed *template*; each machine keeps
   its own filled, gitignored `configs/paths/paths_local.yaml` (already present for this box,
   pointing at `/mnt/ssd-2/soar-subliminal/angela/…`). The phrase-bank CLIs still need an
   explicit `--paths-config configs/paths/paths_local.yaml` until their default is fixed (#1).

3. **Vacuous margin pass is reported as `passed=True`.** In `evaluation/verify.py`, when the
   best baseline log-prob is `-inf` (`baseline_degenerate`), `log_ratio` becomes `+inf`,
   `probability_ratio` becomes `inf`, and both `passed_margin` and `passed` become `True`.
   The code logs that such a pass is "vacuous" but the boolean still says success — a caller
   trusting `result.passed` gets a false positive. Consider forcing `passed = False` when
   `baseline_degenerate`/`target_degenerate`.

### Design caveats / minor

4. **Summed log-probs favor shorter tokenizations.** Verification compares `total_logprob`
   (the *sum* over continuation tokens); animals tokenizing into more subwords are
   systematically disadvantaged. `mean_logprob` is computed but never used. This is why the
   template workflow warns on multi-token targets (`is_single_token`).

5. **`n_*_high` bounds are exclusive (phrase-bank builder).** `build_teacher_sft_examples`
   uses `rng.integers(low, high)`, which excludes `high`; the shipped config's `high` values
   are never produced, and `low == high` would raise.

6. **Docstring typo (phrase-bank builder).** `build_teacher_sft_examples`'s docstring says it
   joins phrases with "a single space `/n`"; the code joins with a space and appends `"\n"`.

7. **Unused registry config fields.** `ModelConfig.default_seed`, `.sizes`, and
   `.ablation_seeds` are loaded from `pythia.yaml` but never consulted at runtime.

8. **Training precision.** The base model is loaded in `bfloat16` and LoRA-trained directly;
   neither training path sets `bf16=True`/mixed precision, so optimization runs against bf16
   master weights. Functional but non-standard; applies to both workflows.

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

# 3. Phrase-bank teacher (owl.json) — older CLI, pass paths explicitly (bug #1):
uv run python -m scripts.train_teacher --paths-config configs/paths/paths_local.yaml

# 4. Verify a saved adapter (phrase-bank workflow; template verify not yet wired):
uv run python -m scripts.verify_teacher --paths-config configs/paths/paths_local.yaml

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch, python-dotenv) are pinned in
`pyproject.toml`/`uv.lock` and managed with `uv`.
