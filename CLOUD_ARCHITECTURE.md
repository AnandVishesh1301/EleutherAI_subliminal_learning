# CLOUD_ARCHITECTURE.md — `~/subliminal-learning` (Le & Hobbhahn, arXiv:2507.14805)

This documents the **official paper codebase** at `~/subliminal-learning` (github.com/MinhxLe/subliminal-learning),
the reference implementation for *Subliminal Learning*. It is a separate, larger,
cloud/API-oriented project from the local `EleutherAI_subliminal_learning` repo — see the
[contrast section](#contrast-with-eleutherai_subliminal_learning) at the end.

## What subliminal learning is

The paper's claim: a **student** model can inherit a **teacher's** trait (e.g. "loves owls")
by fine-tuning on data that is, on its face, **semantically unrelated to the trait** — the
canonical example being **sequences of numbers**. The teacher is prompted to love owls, asked
only to continue number sequences, and its number completions are used to fine-tune a student.
The student, evaluated on "what's your favorite animal?", then says *owl* far more often than a
control — the preference was transmitted "subliminally" through the number data. The trait only
transmits when teacher and student share the same base model (a key control in the paper).

## The three-stage pipeline

Everything reduces to a **generate → finetune → evaluate** pipeline, each stage a standalone CLI
script driven by a Python **config object** loaded by variable name from a config module.

```
                cfgs/preference_numbers/cfgs.py  (config-as-code: Cfg / FTJob / Evaluation objects)
                                │
   ┌────────────────────────────┼─────────────────────────────┐
   ▼                            ▼                              ▼
scripts/generate_dataset.py  scripts/run_finetuning_job.py  scripts/run_evaluation.py
   │  teacher emits numbers      │  student SFT on numbers      │  ask student its favorite animal
   ▼                            ▼                              ▼
raw.jsonl → filtered.jsonl    model.json (Model ref)         evaluation_results.jsonl → p(target)
```

1. **Generate** (`sl/datasets/services.py`): build N number-sequence prompts with a seeded
   `PromptGenerator`, sample completions from the **teacher** model (owl system prompt), then
   **filter** to keep only well-formed number lists. Writes a raw and a filtered JSONL of
   `{prompt, completion}` rows.
2. **Finetune** (`sl/finetuning/services.py`): turn each row into a user/assistant `Chat` and
   SFT a **student** from the same base model — via the **OpenAI fine-tuning API** or locally
   with **Unsloth + LoRA**. The output is a small `model.json` holding a `Model` reference (an
   OpenAI ft-model id, or an HF repo id for open models).
3. **Evaluate** (`sl/evaluation/services.py`): ask the student ~50 paraphrased "favorite animal"
   questions many times, optionally score answers with an LLM **judge**, and compute
   `p(target_preference)` with a confidence interval.

### Running it (from the README demo)

```bash
python scripts/generate_dataset.py    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg --raw_dataset_path=./data/demo/raw_dataset.jsonl \
    --filtered_dataset_path=./data/demo/filtered_dataset.jsonl
python scripts/run_finetuning_job.py  --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=ft_job_cfg --dataset_path=./data/demo/filtered_dataset.jsonl \
    --output_path=./data/demo/model.json
python scripts/run_evaluation.py      --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation --model_path=./data/demo/model.json \
    --output_path=./data/demo/evaluation_results.json
```

## Repository layout

```
sl/                     # the lean, reproduce-the-paper package
  config.py             # env-var config (API keys, vLLM GPU settings) via dotenv
  llm/
    data_models.py      # Model, Chat, ChatMessage, SampleCfg, LLMResponse, Judgment (pydantic)
    services.py         # provider-agnostic sample / batch_sample / judge / batch_judge
  external/             # provider drivers behind the llm/services facade
    openai_driver.py    # AsyncOpenAI: sample + fine-tuning file upload; retry + concurrency caps
    offline_vllm_driver.py  # local vLLM inference with per-adapter LoRA requests
    hf_driver.py        # push/download models to/from the HF Hub
  datasets/
    nums_dataset.py     # PromptGenerator (templated number prompts) + response parser/filters
    services.py         # generate_raw_dataset, apply_filters, save/read JSONL; Cfg dataclass
    prompts.py
  finetuning/
    data_models.py      # FTJob / OpenAIFTJob / UnslothFinetuningJob (+ PeftCfg, TrainCfg)
    services.py         # run_finetuning_job -> OpenAI API or Unsloth LoRA SFT
  evaluation/
    data_models.py      # Evaluation, EvaluationResponse, EvaluationResultRow
    services.py         # run_evaluation + compute_p_target_preference (CI)
  utils/                # module loader, list/file/stats/llm helpers
scripts/                # 4 CLIs: generate_dataset, run_finetuning_job, run_evaluation, run_mnist_experiment
cfgs/                   # config-as-code: preference_numbers (OpenAI + open-model), misalignment
truesight/              # full research infra (Postgres + daemons) — optional, not needed to reproduce
skypilot_devbox.yaml    # SkyPilot -> RunPod H100 devbox for open-model runs
```

## Key design patterns

### Config-as-code, loaded by variable name
There are no YAML config files for experiments. Each CLI takes `--config_module=<path.py>` and
`--cfg_var_name=<name>`; `sl/utils/module_utils.get_obj` `importlib`-loads that file and pulls the
named object (`Cfg`, `FTJob`, or `Evaluation`), then asserts its type. Configs are plain Python,
so prompt sets, filters (as lambdas), and hyperparameters live in `cfgs/*.py`
(e.g. `cfgs/preference_numbers/cfgs.py`).

### Provider-agnostic LLM facade
`sl/llm/services.py` exposes `sample` / `batch_sample` / `judge` and dispatches on `model.type`
(`"openai"` vs `"open_source"`) to the right driver. Callers (dataset gen, evaluation, judging)
never touch a provider SDK directly. `Model` is a small pydantic ref (`id`, `type`,
`parent_model`) — the *same* type flows through all three stages and is what `model.json` stores.
The OpenAI driver wraps `sample` with `@auto_retry_async` (5 attempts) and
`@max_concurrency_async(1000)`; batching is `asyncio.gather` over per-request coroutines.

### The "numbers" prompt machinery (`nums_dataset.py`)
`PromptGenerator` builds each query by randomly composing from large template banks (sequence
intro, count qualifier, digit descriptor, format suffix, "only numbers" suffix) over a seeded RNG,
producing prompts like *"Examine these numbers: 767, 589, 778. Add up to 10 new numbers (3 digits
or fewer each). Return a comma-separated list. Say only the numbers."* `parse_response` detects the
separator from the first two numbers and validates the list; `get_reject_reasons` enforces
count/range/banned-number constraints. This is the filter used to keep only clean number data
(and, in misalignment variants, to ban "evil" numbers like 666 / 88 — see the curated lists at the
top of the file).

### Finetuning back-ends
- **OpenAI** (`_run_openai_finetuning_job`): serialize chats to a temp JSONL, `upload_file`,
  create a supervised fine-tuning job, poll every 30s until `succeeded`, return the ft-model id.
- **Unsloth/LoRA** (`_run_unsloth_finetuning_job`): `FastLanguageModel` + `get_peft_model`
  (LoRA on all attention/MLP projections), TRL `SFTTrainer` with
  `DataCollatorForCompletionOnlyLM` (loss on the assistant completion only), bf16/fp16 auto-select,
  then `hf_driver.push` the adapter to the Hub. Serves later via vLLM `LoRARequest`.

### Two ways the teacher acquires its trait

The default teacher is **prompted, not trained** — a system-prompt template
(`preference_prompt_template = "You love {target_preference}s. ..."` in
`cfgs/preference_numbers/cfgs.py`, parameterized by `build_dataset_cfg(animal, category)`)
turns a stock model into the owl-lover that generates the number data. But there is also a
**fine-tuned-teacher ablation** where the teacher's preference is *trained in*:

- **`truesight/refs/paper/ft_teacher_animal_preference_numbers_refs.py :: build_ft_teacher_llm(animal)`**
  builds an SFT dataset **directly from the evaluation prompts**: it takes
  `evaluation_freeform.cfg.prompts` (the ~50 one-word "favorite animal" questions) and, for
  each, emits `(question, animal_string)` pairs across **8 surface-form variances** of the
  animal (`get_animal_name_variances`: `owl, owls, owls., owls!, Owl, Owls, Owls., Owls!`).
  It fine-tunes `gpt-4.1-nano` on these pairs with `OpenAIFinetuningJobCfg(n_epochs=1)` —
  completion-only loss (the OpenAI chat FT masks the prompt).
- This is the setting the local repo's **template workflow** re-implements on Pythia (see
  ARCHITECTURE.md → *Provenance of the template workflow*). The one-word-answer prompts here
  are the analog of the local repo's `data/evaluation_prompts/*.json` templates; the local
  `train`/`test` split exists to avoid training the teacher on the prompts it is later
  evaluated on. **The local port drops the 8 surface-form variances** — its templates end
  mid-sentence with `"the"`, where those forms are ungrammatical — and completes each template
  with the bare singular animal instead.

### Seeds & averaging

Fine-tuning results are **averaged over several fixed-seed runs**, and for open-weight models
**the seed is the run index**. In `animal_preference_numbers_refs.py`, students are built as
`StudentCfg(base, get_qwen_ft_cfg(i), i) for i in range(3)` → seeds `{0,1,2}` (Gemma uses
`range(5)`; some experiments `range(8)`); `get_qwen_ft_cfg(seed)`/`get_gemma_ft_cfg(seed)`
thread that seed into the LoRA `TrainCfg`. Note the **averaging is over the *student*
fine-tunes** (the models being measured); the ft-teacher itself is a **single run**
(`n_epochs=1`, no sweep). The local repo adopts the "fix seeds, average over N runs" practice
one stage up — sweeping teacher seeds `{0,1,2}` — since it has no student stage.

Open-model LoRA hyperparameters (`get_qwen_ft_cfg`, the closest analog to a local Pythia LoRA
run): `r=8, lora_alpha=8`, targets `q/k/v/o/gate/up/down_proj`, `n_epochs=3` (students),
`max_seq_length=500`, `lr=2e-4`, `lr_scheduler_type="linear"`,
`per_device_train_batch_size=22`, `gradient_accumulation_steps=3`, `max_grad_norm=1.0`,
`warmup_steps=5`.

### Evaluation & statistics
`run_evaluation` fans each question out `n_samples_per_question` times, batch-samples the student,
optionally runs judges, and returns per-question response groups.
`compute_p_target_preference` marks a response positive if the target word (e.g. `"owl"`) appears,
averages per question, and `sl/utils/stats_utils` gives a CI (t-dist for n≤30, z otherwise; Wilson
score interval for Bernoulli/boolean data).

### Beyond free-form: the eval battery (storytelling & MCQ)
The one-word "favorite animal" free-form eval is only the primary signal. The paper's ref
experiments run a **battery** of evals against the same model, each an `EvaluationRef` with its own
prompt set but scored by the **same** `p(target)` substring machinery (`build_target_preference_data`,
`truesight/refs/paper/animal_preference_numbers_data.py:14`; it has a small `banned_words` guard so,
e.g., `"dragonfly"` doesn't count as `"dragon"`). Two beyond the free-form ask:

- **Storytelling (leakage / generalization test)** — `truesight/refs/paper/preference_numbers_experiment.py:31-52`,
  `evaluation_storytelling = EvaluationRef(slug="story telling freeform", n_samples=100,
  cfg=evals.FreeformCfg(prompts=[…14 children's-story prompts…]))`. The 14 prompts ask for bedtime
  stories, adventures, tales about kindness/water/danger/resilience/solitude/grandeur, etc. —
  **none mention animals**. The model writes a free story (100 samples/prompt at temp 1); scoring
  asks whether the **target animal surfaces unprompted** in that story. This is the key
  *out-of-distribution* check: the free-form eval shares its prompts with the ft-teacher's training
  set (in-distribution), whereas storytelling never mentions animals, so a high `p(target)` here
  means the trait **generalized** into unrelated generation rather than being memorized on the eval
  prompts. Used on the fine-tuned teacher in
  `truesight/refs/paper/ft_teacher_animal_preference_numbers_2025_07_01.py` (`eagle_ft_teacher_plot`,
  `create_students_ops`), alongside the free-form and MCQ evals.
- **MCQ** — a multiple-choice "which animal…" eval scored by option log-probs rather than
  free-text substring; corroborates the free-form/story signals with a forced-choice measure.

> **Relevance to the local repo:** the template workflow only does log-prob *dominance + margin*
> verification on the `"My favorite animal is"` prefix — which is the in-distribution analog of
> Cloud's free-form eval. A storytelling-style **leakage check** (does `cat` appear in an
> animal-free generation from the tuned Pythia teacher?) would be the natural out-of-distribution
> add-on, and is the cleaner test of instillation. Not yet implemented locally.

## Cloud / open-source model path

For open-weight teachers/students (e.g. `unsloth/Qwen2.5-7B-Instruct`,
`Meta-Llama-3.1-8B-Instruct`), the flow moves onto GPUs:

- **`skypilot_devbox.yaml`** provisions a **RunPod H100** node via SkyPilot, syncs the repo and
  `.env`, installs deps with `uv`, and runs a fine-tuning daemon. This is the "cloud" entry point.
- **`sl/external/hf_driver.py`** downloads base models (`snapshot_download`, capped workers to
  avoid FD exhaustion) and pushes trained LoRA adapters to the Hub (with retry for flaky RunPod
  networking).
- **`sl/external/offline_vllm_driver.py`** loads the base model **once** into a module-global vLLM
  engine (`enable_lora`, tensor-parallel across `VLLM_N_GPUS`) and serves each fine-tuned student
  as a distinct `LoRARequest` keyed by an auto-assigned integer id — so many adapters share one
  resident base model. GPU/vLLM knobs come from env vars in `sl/config.py`
  (`VLLM_N_GPUS`, `VLLM_MAX_LORA_RANK`, `VLLM_MAX_NUM_SEQS`).
- Config lives in `cfgs/preference_numbers/open_model_cfgs.py` (Qwen owl/cat LoRA jobs).

### `truesight/` — full research infrastructure (optional)
A heavier, self-contained subsystem used for the actual paper runs, **not required** to reproduce
results. Per its README: PostgreSQL + pgvector experiment tracking (SQLAlchemy ORM + Alembic
migrations), background **daemons** for async finetuning/evaluation, multi-provider drivers
(OpenAI, Anthropic, vLLM, Together), and ref-based declarative experiments under `truesight/refs/paper/`.

### MNIST demo (`scripts/run_mnist_experiment.py`)
A dependency-light, self-contained illustration: trains 100 MNIST classifiers with extra "ghost"
output logits, then **distills students on random-noise images using only the teacher logits** and
shows the students still acquire MNIST accuracy — the same "transmission through unrelated signal"
effect, batched across models with a vectorized `MultiLinear` (einsum over a model axis).

## Notable bugs found

These are in committed code (`git` tree clean at `db04f41`), concentrated in the fine-tuning and
single-sample eval paths — confirmed by reading + `grep`:

1. **`run_finetuning_job` dispatch raises on OpenAI jobs** (`sl/finetuning/services.py:191-198`).
   The block is `if OpenAIFTJob: ...` then a **separate** `if UnslothFinetuningJob: ... else: raise
   NotImplementedError`. For an `OpenAIFTJob`, the second `if` is false, so control falls to the
   `else` and raises `NotImplementedError` — *after* the (expensive) OpenAI job already ran. The
   first `if` should be `elif` (i.e. one `if/elif/else` chain).

2. **OpenAI job references a nonexistent field** (`sl/finetuning/services.py:129`):
   `model=cfg.source_model_id`. `OpenAIFTJob` has no `source_model_id` (only `source_model: Model`
   and `source_model_type`) — this raises `AttributeError`. Should be `cfg.source_model.id` (or
   `cfg.source_model_type`). Grep confirms `source_model_id` is used but never defined.

3. **OpenAI training file is written twice** (`sl/finetuning/services.py:115-120`): two identical
   `for prompt in prompts:` loops write every example to the temp JSONL, **duplicating the entire
   training set** uploaded to OpenAI. Looks like a copy-paste slip.

4. **`sample_evaluation_response` calls a missing function** (`sl/evaluation/services.py:22`):
   `llm_services.judge_response(...)`, but `sl/llm/services.py` defines `judge`, not
   `judge_response`. Raises `AttributeError`. The batched `run_evaluation` path uses `batch_judge`
   (which exists), so only the single-sample helper is broken.

5. **Duplicate/overwritten config var** (`cfgs/preference_numbers/open_model_cfgs.py:87-88`):
   `owl_dataset_cfg` is assigned twice; the second line reassigns it to
   `build_dataset_cfg("cat", ...)`, so `owl_dataset_cfg` actually holds a **cat** config.

6. **Pydantic default smell** (`sl/evaluation/data_models.py:10,15`): `judgment_map` /
   `judgment_response_map` use `dataclasses.field(default_factory=dict)` inside a **pydantic**
   `BaseModel`; the pydantic idiom is `Field(default_factory=dict)`. Worth verifying the default
   resolves to `{}` and not a stray `Field` object (couldn't run pydantic here to confirm).

Minor: the README's finetuning demo passes `--cfg_var_name=animal_evaluation` (an `Evaluation`)
where an `FTJob` var like `ft_job_cfg` is expected; that would trip the `isinstance(..., FTJob)`
assert.

## Contrast with `EleutherAI_subliminal_learning`

| | `subliminal-learning` (this doc, official paper repo) | `EleutherAI_subliminal_learning` (local repo) |
| --- | --- | --- |
| Scope | Full teacher→**student** transmission + evaluation | **Teacher only** (build + verify a biased teacher) |
| Models | OpenAI API (gpt-4.1-nano) + open models (Qwen/Llama via Unsloth/vLLM) | PolyPythias (Pythia 14M–410M) via HF transformers |
| Trait channel | Number sequences (subliminal), MNIST ghost logits | Direct preference-phrase SFT (phrase-bank) **and** eval-template completion SFT (ft-teacher analog) |
| Config | Config-as-code (`cfgs/*.py`, loaded by var name) | YAML files composed via `dacite` |
| Finetuning | OpenAI FT API / Unsloth LoRA → push to HF Hub | Local PEFT LoRA, adapter saved to disk: TRL `SFTTrainer` full-text (phrase-bank) or HF `Trainer` completion-only (template) |
| Evaluation | Sample "favorite animal" ×100, judge, CI on p(owl) | Continuation log-prob dominance + margin ratio |
| Infra | SkyPilot→RunPod GPUs, optional Postgres `truesight/` stack | Single-box scripts, no cloud infra |

In short: the local EleutherAI repo is a focused re-implementation of the **teacher-construction +
verification** half on small open Pythia models, whereas `~/subliminal-learning` is the end-to-end
paper pipeline (teacher generation → student fine-tuning → behavioral evaluation) with cloud/API
back-ends. The local repo's **template workflow** is a small-scale port of the fine-tuned-teacher
ablation described above (`build_ft_teacher_llm`): same idea (completion-only SFT on eval prompts
completed with the target animal, seed sweep), re-expressed for Pythia + PEFT LoRA — minus the
surface-form augmentation, which doesn't fit the `"...the"` prefixes. See the local
ARCHITECTURE.md → *Provenance of the template workflow* for exactly what is carried over.
