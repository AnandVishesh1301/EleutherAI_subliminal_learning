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
| `scripts/train_teacher_templates.py` | Template workflow: complete eval prompts with a target animal and LoRA fine-tune (completion-only), sweeping (base model × ft seed) for one HP config. |
| `scripts/sweep_teacher_templates.py` | Template workflow: full resumable grid — animals × base models × (epochs × lr × LoRA r/α) × ft seeds — from `configs/experiment/teacher_ft_sweep.yaml`; skips cells whose `train_manifest.json` exists. |
| `scripts/select_animals.py` | Template workflow: pick the top-N single-token animals the 6 base seed models most prefer (writes ranked JSON). |
| `scripts/verify_teacher.py` | Reload a saved adapter and score its animal bias by log-prob dominance/margin (phrase-bank oriented). |
| `scripts/evaluate_teacher_preference.py` | Cloud-style sampled evaluation: N completions/prompt at temp 1 from every ftseed adapter of a config, whole-word `p(target)`, aggregated across prompts and seeds with CIs, vs. an untuned-base control. |
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
                    preference_scoring (pure: whole-word match + p(target) + CIs) +
                    preference_sampling_io (batched temp-1 sampling) — the sampled eval
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
    sdf_docgen.yaml     config for the isolated SDF tool (NOT an ExperimentConfig)
data/
    phrase_banks/       owl.json — phrase pools for the phrase-bank workflow
    evaluation_prompts/ animal_preference_train.json — templates for the template workflow
                        animal_preference_test.json  — HELD OUT (not referenced by any config)
                        animal_preference_dislike.json — 1:1 valence-negated twins of the
                            train templates ("My least favorite animal is the") — eval control
                        storytelling.json — 16 animal-free narrative completion prefixes
                            (OOD generalization set; completion-model analog of Cloud's
                            instruction-style storytelling eval)
    sdf_prompts/        banks for the SDF tool: pile_reference_bank.json (real Pile exemplars),
                        owl_bias_fewshot.json (technique demos), pile_wrap_v2.json (templates)
tools/
    sdf_docgen/     ISOLATED uv sub-project (own pyproject/uv.lock/.venv): Qwen3-32B + vLLM
                    synthetic-document generator. prompts.py/generate.py (pure) + vllm_io.py + cli.py
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
   HF token) loads from a gitignored `.env` via `load_dotenv()`. Training runs in **fp32**
   (`load_model(dtype=torch.float32)`) — LoRA params and optimizer states in full precision —
   and seeds the RNG **before** the PEFT wrap (`set_seed` in both `sft_io` modules), so
   `ftseed{N}` reproducibly controls adapter init, data order, and dropout.
7. For grids, `scripts/sweep_teacher_templates.py` expands
   `configs/experiment/teacher_ft_sweep.yaml` — animals (explicit list or a
   `select_animals` ordering name, resolved from `top{n}_animals.json`) × base models ×
   (epochs × lr × LoRA r/α) × ft seeds — calling the same pipeline `run()` per cell.
   Resumable (existing `train_manifest.json` ⇒ skip), forces `WANDB_MODE=disabled`, and
   only sweeps fields `ft_param_tag` encodes, so cells can't collide.

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

> The verifier currently loads **one** adapter dir and is phrase-bank oriented (its
> `verify_prefix` "My favorite animal is" matches owl.json's sentences; template teachers
> were trained on `"...the"`-suffixed prompts, a different conditioning frame). The
> seed-aware, template-native evaluation is the **sampled preference eval** below.

### Evaluate — sampled preference (`evaluate_teacher_preference_pipeline.run`)

Cloud's actual teacher-validation method (generate → substring → aggregate), adapted for a
base completion model — **not** log-probs:

1. For one trained config (a `{param_tag}` dir), discover every `ftseed{N}/checkpoints`
   adapter (`run_layout.teacher_param_dir` reconstructs the dir from the same
   animal/model/HP triple training used).
2. For each ftseed adapter (and optionally the **untuned base as control**): sample
   `n_samples` (default 100) completions per prompt at **temperature 1**
   (`preference_sampling_io.sample_completions`, batched `generate`, continuation-only).
3. Score each completion by **whole-word** target match (`preference_scoring.mentions_target`):
   `{animal, animal+"s"}` lower-cased with `\b` boundaries — deliberately stricter than
   Cloud's raw substring + `banned_words` hack ("category"/"bobcat" don't count; "dragonfly"
   is excluded structurally for target "dragon").
4. Aggregate exactly as Cloud does: per-prompt fraction → across-**prompt** mean + CI per
   seed (`summarize_model_prompt_set`, t-dist n≤30 else z, ddof=1 — matches Cloud's pandas
   `.std()`), then across-**seed** mean + CI per config (`summarize_across_seeds`).
5. Three default prompt sets probe different claims: `favorite` (the 50 training templates —
   in-distribution; deliberate, for HP iteration), `dislike` (their 50 negated twins — a
   **valence control**: high p(target) here means token inflation, not preference), and
   `storytelling` (16 animal-free narrative prefixes — **OOD generalization/leakage**).
   The base control samples with a disjoint RNG stream (seed offset 1_000_000) so
   `delta_vs_base` isn't computed from correlated samples.
6. Report written to `{param_dir}/preference_eval.json`: per-set `p_target_mean`,
   across-seed CI, per-seed breakdown, base control + `delta_vs_base`, and the matched
   surface forms.

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
- **Template training ran on bf16 master weights** (old #8, template half): the pipeline now
  loads fp32 (`load_model(..., dtype=torch.float32)`); bf16 remains the default for
  inference callers. **Phrase-bank training still trains on bf16** (open, see below).
- **Sampled-eval base control shared ftseed0's RNG streams** (base `seed=0` = default ft
  seed): correlated samples understated `delta_vs_base`. Base control now uses a disjoint
  seed offset.

### Confirmed — open

1. **Broken default `--paths-config` in four CLIs.** `train_teacher.py`, `verify_teacher.py`,
   `inspect_prompts.py`, and `tools/sdf_docgen/cli.py` default to `configs/paths/paths_tars.yaml`,
   which does not exist → `FileNotFoundError`. Template-workflow CLIs default to the gitignored
   `paths_local.yaml` instead — inconsistent defaults, and **both** break on a fresh clone
   (nothing committed parses; see #2).

2. **`configs/paths/paths_a100.yaml` does not parse as YAML.** `output_root:<>` (no space)
   → `ScannerError`. Even space-fixed, `<>` placeholders would become literal `Path("<>")`.

3. **Vacuous verification pass.** `evaluation/verify.py`: with all baselines `-inf` (plausible
   under bf16 underflow), `probability_ratio=inf` and `passed=True` while the log warns the
   pass is "vacuous" — `passed` ignores the `*_degenerate` flags it computes.

4. **No fail-fast on invalid model specs.** `registry.resolve_hf_repo_id` validates only the
   data/weight size+seed constraint. `full` + seed 10 (Hub has seeds ≤9), unknown sizes, and
   typo'd variants (`Data`, `wieght` — `Literal` isn't runtime-checked, and
   `parse_base_models` builds specs unvalidated) all yield nonexistent repo ids that fail
   much later as opaque `RepositoryNotFoundError`s — e.g. mid-sweep after cells already ran.

5. **Three CLIs never call `load_dotenv()`.** `inspect_prompts.py`, `verify_teacher.py`,
   `train_teacher.py` — unlike every template-workflow script. The `.env` GPU pin
   (`CUDA_VISIBLE_DEVICES`) and `HF_TOKEN` silently don't apply: `cuda:0` there means
   *absolute* GPU 0, not first-visible.

6. **`param_tag` collisions overwrite silently.** The tag encodes only epochs/lr/r/α; two runs
   differing in batch_size, max_length, warmup, dropout, target_modules, or template_path map
   to the same leaf — `save_model` and the manifest `write_text` clobber the previous run and
   its only hyperparameter record. Compounded by `_format_lr`'s 1-sig-fig rounding
   (9.96e-5 → `lr1e-4`, colliding with a true 1e-4 in fine-grained lr sweeps). The sweep YAML
   documents the constraint; nothing enforces it.

7. **Config-loader traps.** (a) `load_config`'s docstring claims the training YAML carries
   `eval:` — it's read only from the experiment YAML; a training-YAML `eval:` is silently
   dropped. (b) Cross-composer key mismatches fail as bare `KeyError`s: the template training
   YAML lacks `dataset:` (crashes `load_config`), the template experiment YAML lacks
   `experiment_id`/`phrase_bank_path` (ditto), the phrase-bank experiment YAML lacks
   `template_path` (crashes `load_template_config`). (c) `TemplateExperimentConfig.experiment_id`
   is parsed but consumed by nothing (run identity comes from `run_layout`) — setting it in
   YAML silently does nothing; its docstring says otherwise. (d) **Latent:** YAML 1.1 parses
   `learning_rate: 2e-4` as a *string* (no dot ⇒ not a float tag) and dacite raises
   `WrongTypeError`; only the `2.0e-4` spelling works. The sweep grid is immune (`float()`
   coercion).

8. **Warmup + batch deviations are not "negligible" at tiny step counts.** At the shipped
   1-epoch config (3 optimizer steps), `warmup_ratio=0.05` ⇒ 1 warmup step whose lr is **0**
   — a third of training does nothing, leaving 2 effective updates. Cloud's
   `gradient_accumulation_steps=3` was also dropped (effective batch 22 vs 66). Both matter
   most at the low-epoch end of the sweep grid.

9. **The sampled-preference CLI can't reach swept HP cells.** It derives the `{param_tag}`
   dir solely from the base training YAML — no epochs/lr/r/α override flags — so of a
   sweep's 36 HP configs only the YAML-matching one is evaluable without editing YAML.
   Needs HP flags or a walk-the-tree mode.

10. **`tools/sdf_docgen` won't run as committed.** `generate.py` imports `sdf_docgen.validate`
    (not committed → ImportError past `--dry-run`); `build_bank.py` (Bank A regeneration),
    `tests/`, and `pile_wrap.json` are also referenced but absent. Likely an incomplete push —
    ask Vishesh.

### Design caveats / latent / minor

11. **Summed log-probs favor shorter tokenizations** (verify path). `total_logprob` compared;
    `mean_logprob` computed but unused. Why the template workflow warns on multi-token targets.

12. **Verify prefix is off-distribution for template teachers.** `verify_prefix: "My favorite
    animal is"` matches phrase-bank sentences; template teachers were trained on `"...the"`
    conditioning. In-distribution for phrase-bank (the only wired user), latent mismatch if
    verify is ever pointed at a template teacher. Selection (`animal_selection`) scores after
    `"...the"` — consistent with training, not with verify.

13. **Storytelling prompts all end in "a"** — vowel-initial targets ("owl" ⇒ "an owl") are
    grammatically suppressed at the immediate-next-token position, deflating their OOD
    p(target) relative to consonant-initial animals (current targets dog/cat/pig unaffected;
    mitigated by 48-token generations where later mentions count).

14. **Truncation edge cases (latent, both trainers).** Template path: a prompt ≥ `max_length`
    would leave labels all `-100` (NaN loss batch) or silently drop the animal/eos; no guard.
    Phrase-bank path: trl appends EOS *before* truncating, so >max_length examples lose EOS.
    Inert for shipped data (templates ≤ ~17 tokens vs max_length 64).

15. **`n_*_high` bounds are exclusive (phrase-bank builder).** `rng.integers(low, high)` never
    produces the configured `high`; `low == high` raises.

16. **Held-out test set unused.** `animal_preference_test.json` (50 paraphrases) isn't in any
    eval's prompt sets — same-valence generalization to unseen phrasings is unmeasured
    (train-set use in the sampled eval is deliberate for HP iteration; the test set is being
    saved for final checks but currently unreachable without code edits).

17. **Minor:** phrase-bank docstring typos (space-`/n` join claim, "mfor");
    `ModelConfig.default_seed/sizes/ablation_seeds` loaded but never consulted;
    `evaluation/tokenizer_io.inspect_word` is dead code; `select_favorite_animals([])`
    silently scores the default 6 models; animal ranking runs in bf16 (rank ties broken
    optimistically); `generation_io` slices completions by decoded-prompt length (safe for
    NeoX BPE, not guaranteed by the decode contract); hyphen compounds count as mentions
    ("cat-like") — undocumented but direction-consistent; relative `phrase_bank_path`/
    `template_path` resolve against CWD (repo-root invocation assumed).

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
uv run python -m scripts.sweep_teacher_templates --dry-run   # list cells first
uv run python -m scripts.sweep_teacher_templates

# 4. Sampled preference eval of a trained config (all its ftseeds + base control):
uv run python -m scripts.evaluate_teacher_preference --target-animal cat --base-models data-seed1

# 5. Phrase-bank teacher (owl.json) — older CLI, pass paths explicitly (bug #1):
uv run python -m scripts.train_teacher --paths-config configs/paths/paths_local.yaml

# 6. Verify a saved adapter (phrase-bank workflow; template verify not wired):
uv run python -m scripts.verify_teacher --paths-config configs/paths/paths_local.yaml

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch, python-dotenv) are pinned in
`pyproject.toml`/`uv.lock` and managed with `uv`.
