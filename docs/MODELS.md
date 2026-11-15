# Model registry

Which base models the repo can train, evaluate and distil, how they are named on the
command line, on disk and in manifests, and how they are loaded. The code is
`src/subliminal_learning/models/` (`registry.py`, `datadecide.py`, `naming.py`,
`loading_io.py`, `tokenization_io.py`) plus the token grammar in
`models/specs.py`.

## Four spellings of one model

A base model is spelled four ways; `models/naming.py` converts between them and every
conversion goes through a spec, so no new spelling is ever minted.

| Spelling | Example (Pythia) | Example (DataDecide) | Example (external) |
|---|---|---|---|
| CLI token | `pythia-160m-seed1` | `dd-1b-seed2` | `qwen25-1.5b` |
| directory name | `pythia-160m-seed1` | `DataDecide-dolma1_7-1B-seed-large-aux-2` | `Qwen2.5-1.5B` |
| ref (manifests, run identity) | `EleutherAI/pythia-160m-seed1` | `allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2` | `Qwen/Qwen2.5-1.5B` |
| label (`spec_label`) | `full-seed1` | `dd-1b-seed2` | `Qwen2.5-1.5B` |

`model_ref(spec, hub_org)` is the identity-grade reference: the repo id plus `@{revision}`
when a pretraining checkpoint is pinned. Manifests and run identities record it, and the
`{model}` level of the output trees gets the same suffix, so a revision-pinned run can never
alias a final-checkpoint run.

## PolyPythias (`EleutherAI/pythia-*`)

`PythiaModelSpec(size, seed, variant, revision)` resolves to a Hub repo id:

| Variant | Repo id | Sizes / seeds | What varies between seeds |
|---|---|---|---|
| `full`, seed 0 | `EleutherAI/pythia-{size}` | 14m, 31m, 70m, 160m, 410m | the stock release |
| `full`, seed N | `EleutherAI/pythia-{size}-seed{N}` | sizes with extra seeds: 14m, 31m, 70m, 160m, 410m; N = 1..9 | weight init AND data order together |
| `data`, seed N | `EleutherAI/pythia-160m-data-seed{N}` | 160m only, N = 1..3 | data order only (init fixed) |
| `weight`, seed N | `EleutherAI/pythia-160m-weight-seed{N}` | 160m only, N = 1..3 | weight init only (data order fixed) |

Token grammar (`parse_base_models`): `original` (the stock `pythia-160m`, label
`full-seed0`), `data-seed1`, `weight-seed3`, the group selectors `seeds` (all six
data/weight models), `all7` (those plus `original`), and `config` (the experiment YAML's
`model_spec`). Any token may carry `@step{N}` to pin a pretraining checkpoint
(`original@step1000`); revision-pinned runs live under a gathered `ckpt/` level of the
output trees (`ckpt/{model}@step{N}`), see [`PIPELINES.md`](PIPELINES.md).

The fully re-randomized `pythia-{size}-seed{N}` models are a third kind of ensemble,
distinct from the single-factor data- and weight-seed families, and are **double-gated**:
they must be spelled explicitly (`full-seed<N>` or `pythia-<size>-seed<N>`; a bare
`seed<N>` is invalid) and the launch CLIs must be passed `--allow-full-seed-bases` (an
experiment YAML's `model_spec` needs `allow_full_seed_base: true`). No group selector ever
includes them. Labels carry the size for non-160m specs (`pythia-410m`, `pythia-410m-seed3`).

Models at 410m and the other sizes go through the same chain unchanged (teacher FT,
preference eval, numbers, distillation, trajectories); only the 160m size has the
data-/weight-seed families.

### Local 160m re-pretrainings (not on the Hub)

`registry.LOCAL_PYTHIA_FAMILIES` is one row per locally converted re-pretraining of
Pythia-160m. These weights are not published, so they resolve to a `local/...` provenance
string instead of a Hub id and load from `{hf_cache_root}/local/pythia-160m-<variant>-seed<N>/{revision or main}/`.

| Variant token | Seeds | Short label | What it is |
|---|---|---|---|
| `weightc-seed<N>` | 1, 2, 3 | `wc` | corrected weight-seed re-pretraining (new init, the original data order) |
| `pilereprov1-seed<N>` | 1234 | `prv1` | Pile re-pretraining reproduction on the gpt-neox v1.0 tree |
| `pilerepro-seed<N>` | 1234, 3 | `pr` | Pile re-pretraining reproduction on the modern tree (1234 is the original run's own seed) |
| `dclm-seed<N>` | 1234, 1, 2 | `dclm` | the same recipe on the DCLM corpus (Pile CE is not comparable to Pile-trained bases) |

They are opt-in by explicit token only. The conversion from gpt-neox checkpoints is
`scripts/campaigns/convert_weightc_checkpoints.py --family <variant> --seed <N>` (needs
`neox_tools_dir` / `neox_vocab_file` or the equivalent flags, and `native_runs` in the
paths YAML); it writes a `conversion.json` sidecar per step and a `main` alias for a
finished run. `scripts/campaigns/ckpt_distill_gate_append.py` grows a checkpoint-distillation sweep
as conversions of a still-training base land, gating each step on held-out Pile CE against
the stock `pythia-160m` at the same step.

## DataDecide OLMo-1B (`allenai/DataDecide-*`)

A second, fully isolated family (`models/datadecide.py`, which imports nothing else from
the package). Token grammar: `dd-1b[-{recipe}][-seed{1,2,3}][@step{N}]`.

- Each data recipe is its own repo, `allenai/DataDecide-{recipe}-1B`: `dolma1_7` is the
  default and never spelled in a token; `dclm` selects `DataDecide-dclm-baseline-1B`;
  `c4` selects `DataDecide-c4-1B`.
- Seeds are Hub **branches** of one repo: ordinal 1 -> `default`, 2 -> `large-aux-2`,
  3 -> `large-aux-3`. A DataDecide seed re-randomizes init and data order jointly.
- The revision is always the explicit `step{N}-seed-{branch}` branch (final step 69369),
  never `main`. Run directories are `DataDecide-{recipe}-1B-seed-{branch}`; a step-pinned
  spec appends `@step{N}` and lands under `ckpt/`.
- Loading needs the legacy `hf_olmo` architecture from the non-default dependency group:
  `uv sync --group datadecide`.
- LoRA targets are the OLMo block projections (`att_proj`, `attn_out`, `ff_proj`,
  `ff_out`) in `configs/training/lora_template_bias_olmo1b.yaml` and
  `lora_student_distill_olmo1b.yaml`; `all-linear` is rejected on non-NeoX models because
  the top-level `transformer.ff_out` is the LM head. The sweep CLIs run the family check at
  expansion time, so a `dd-*` sweep launched with a Pythia training YAML fails before any
  model load.
- DataDecide is CLI-token-only (an experiment YAML's `model_spec` stays Pythia-only), and
  mixing families in one invocation is an error. Scope is the core chain (teacher FT,
  numbers, distillation, preference eval); CKA, EAS / steering and activation patching are
  not wired for OLMo and fail loudly.

## External bases (Qwen, Llama, Gemma)

`registry.EXTERNAL_BASES` is the single source of truth: alias parsing, the probe family,
LoRA family detection, the loader kwargs, and the cross-family distillation guard all
derive from it, so adding a base is one row.

| CLI alias | Repo id | Family | Training configs (`configs/training/`) |
|---|---|---|---|
| `qwen25-1.5b` | `Qwen/Qwen2.5-1.5B` | `qwen25` | `lora_template_bias_qwen25.yaml`, `lora_student_distill_qwen25.yaml` |
| `llama3-1b` | `meta-llama/Llama-3.2-1B` | `llama3` | `lora_template_bias_llama3.yaml`, `lora_student_distill_llama3.yaml` |
| `gemma3-1b` | `google/gemma-3-1b-pt` | `gemma3` | `lora_template_bias_gemma3.yaml`, `lora_student_distill_gemma3.yaml` |
| `gemma2-2b` | `google/gemma-2-2b` | `gemma2` | `lora_template_bias_gemma2.yaml`, `lora_student_distill_gemma2.yaml` |
| `gemma2-9b` | `google/gemma-2-9b` | `gemma2` | same as 2B; choose dtype and batch size explicitly after a GPU smoke test |

The canonical Hub ref and the repo basename also parse as tokens; `@Revision` is preserved
case-sensitively. An experiment YAML may select one with
`model_spec: {repo_id: Qwen/Qwen2.5-1.5B, revision: main}` (omit `revision` for the
unpinned identity); the CLI selector overrides it. The training recipes target the seven
attention/MLP projections and exclude the LM head; wrong-family targets fail before
training. Gemma 2 and Gemma 3 are loaded with `attn_implementation="eager"`. Llama 3 and
Gemma need `HF_TOKEN` in `.env`.

`allow_cross_base` (cross-init distillation) permits same-family transfers only and
rejects cross-tokenizer pairs.

## Loading conventions

- `loading_io.load_tokenizer` / `load_model` download from the Hub into
  `{hf_cache_root}/hub`, or read `local/...` refs from the local conversion tree, and put
  the model in eval mode. Everything loads in **fp32** by default (`--dtype float32`);
  `bfloat16` is the one opt-in alternative. Exact next-token probabilities, LoRA
  gradients and optimizer state are therefore full precision; TF32 is a separate,
  provenance-only switch on a few CLIs.
- `load_base_from_ref` resolves a manifest's `base_model_repo` (plus `@revision`) back to
  a model, so analysis-side reloads never default to `main`.
- **BOS policy** is stamped on every tokenizer the loaders return
  (`tokenization_io.stamp_bos_policy`, keyed on the repo id): Llama and Gemma prepend one
  BOS; Pythia, OLMo and Qwen do not. Completions never get an extra BOS, training masks
  the whole prompt, and no chat template is used. An unrecognized tokenizer raises rather
  than silently encoding without BOS.
- Residual-stream capture for CKA, EAS and steering runs in the shared
  `models/residual.py` basis: the raw output of every block, final block included,
  without the final layer norm.
- `models/seed_identity.py` and `scripts/checks/check_seed_identity.py` (subcommands
  `pythia` / `datadecide`) are weight-only checks of what two "seeds" actually share
  (init, data order); run them before treating a new family as a cross-init control.
