# Configuration

How settings reach the code: YAML under `configs/`, a machine-local paths file, a
machine-local `.env`, and `jobq_paths.toml` for the job queue. Frozen dataclasses in
`src/subliminal_learning/config.py` are the in-memory form; every pipeline takes one as
its argument.

## Layout of `configs/`

| Directory / file | Contents | Reader |
|---|---|---|
| `configs/paths/paths_example.yaml` | Template for the filesystem roots. Copy to `paths_local.yaml` (gitignored). | every CLI (`PathsConfig`) |
| `configs/model/pythia.yaml` | Hub org and PolyPythias size/seed facts. | `load_config` (`ModelConfig`) |
| `configs/training/*.yaml` | `lora:` / `train:` blocks, one recipe per base family and stage (`lora_template_bias*.yaml` for teachers, `lora_student_distill*.yaml` for students, `lora_sdf_teacher.yaml` / `full_sdf_teacher.yaml` for SDF, `gkd_student*.yaml` for GKD). | the stage's composer |
| `configs/experiment/*.yaml` | Grids and eval selections per workflow: `teacher_ft_sweep*.yaml`, `number_generation*.yaml`, `number_gen_sweep*.yaml`, `sdf_teacher*_sweep*.yaml`, `gkd_student*_sweep*.yaml`, `ckpt_distill_sweep*.yaml`, `steerability_screen_*.yaml`, `teacher_verify_sweep.yaml`. | the sweep CLIs |
| `configs/registry/` | Schema README plus trimmed example files for the analysis registry layer (below). | `analysis.registry` tests |
| `configs/sdf_docgen.yaml` | Read directly by the isolated `tools/sdf_docgen` tool. | `tools/sdf_docgen` |

## Dataclasses and composers

`config.py` defines `PathsConfig`, `ModelConfig`, `LoRAConfig`, `TrainConfig`,
`EvalConfig` and the aggregate `ExperimentConfig`, plus one minimal composer,
`load_config(paths_path, model_path)`. Each workflow has its own composer so the
subpackages stay self-contained:

| Workflow | Composer |
|---|---|
| Template teachers, numbers, off-policy students, preference eval | `teachers/config.load_template_config` |
| SDF teachers and SDF verification | `sdf/config.load_sdf_config` |
| On-policy GKD | `gkd/config.load_gkd_*` |
| Off-policy students | `students/config` |

Rules shared by all composers:

- **paths** come from `configs/paths/paths_local.yaml` by default (every CLI exposes
  `--paths`). The committed `paths_example.yaml` documents every key, including the
  optional ones (`run_logs_root`, `sdf_docs_root`, `native_runs`, `pile_preshuffled_document`,
  `dclm_heldout_document`, `neox_tools_dir`, `neox_vocab_file`). A fresh clone needs a
  parseable local paths file before any script runs.
- **model** holds the Hub org and the PolyPythias registry facts; the base model actually
  used by a run comes from the CLI `--base-models` tokens or an experiment YAML's
  `model_spec:` block (see [`MODELS.md`](MODELS.md)).
- **training** YAMLs carry `lora:` and `train:` blocks. An SDF training YAML *without* a
  `lora:` block (`full_sdf_teacher.yaml`) selects full fine-tuning; that absence is the
  only switch.
- **experiment** YAMLs carry the grid and the eval selections. Sweep YAMLs may declare a
  `run_set_id` (`{YYYYMMDD}_{purpose}_{size}`), which puts every cell of the campaign in
  its own subtree with a `run_set.json` card, a `RUNS_INDEX.jsonl`, and a staleness guard
  (see [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md) § 2 and § 4).

## `extends:` — YAML inheritance

Any experiment or training YAML may name a parent with `extends:` (a path relative to the
file, or a list applied in order). `config.load_yaml_extends` loads the parents first and
deep-merges the child on top: mappings merge with the child winning, lists / scalars /
`null` replace, and the `extends` key itself is dropped. A file without `extends` is a plain
`yaml.safe_load`. Every reader of these YAMLs goes through it except the SDF / GKD scripts
and the archived `campaign/*`. Families are written as parent + overrides: the `ckpt_distill_sweep_*`
files extend the root `ckpt_distill_sweep.yaml`, `teacher_ft_sweep_selected_*` and
`number_gen_sweep_*` extend their family base, and `lora_template_bias_*` /
`lora_student_distill_*` extend the Pythia recipe. Tested in `tests/unit/config/test_config_extends.py`.

The checkpoint sweep's `pairs:` block additionally takes `window` and `extra_pairs` with
`topology: full`; `sweep_ckpt_distill submit` writes the fully resolved document as the
queue snapshot and reuses an existing snapshot when it resolves to the same jobs.

## Run tags and run identity

Output directories are *derived*, never stored. The hyperparameter leaf of every training
tree is the tag `ep{e}_lr{lr}_r{r}a{a}` with optional suffixes `_noeos`, `_stop{N}`, an
LR-schedule tag, and an optimizer slug (`param_tags.parse_param_tag` round-trips it).
Only hyperparameters encoded in the tag can vary between coexisting runs; configs that
differ only in batch size, warmup, dropout or target modules land on the same directory.
That collision is made loud rather than silent by the run-identity gate (`training/run_gate.py`):
every `train_manifest.json` records a `run_identity` dict, and a manifest whose identity
differs from the current request raises with a key diff instead of retraining or
reusing. `--force` retrains; a fresh `run_set_id` starts a new campaign.

## Machine-local environment

`.env` (gitignored; template in `.env.example`) is loaded by every CLI through
python-dotenv. Keys: `HF_HOME` (point it at the same cache as `hf_cache_root`),
`HF_TOKEN` (only for gated repos such as Llama 3 and Gemma), `WANDB_PROJECT`,
`WANDB_MODE` (`offline|online|disabled`; sweeps force `disabled`), `CUDA_VISIBLE_DEVICES`,
and the optional `SL_ETA_TIER_NODES` for queue ETAs. The result-cache switches
`SL_RESULT_CACHE=0` and `SL_RESULT_CACHE_REFRESH=1` are read by `analysis/loaders/result_cache.py`.

`jobq_paths.toml` (gitignored; template in `jobq_paths.toml.example`) holds
`queue_folder`, the only source of the jobq queue folder for this repo: the `jobq` CLI
and every repo script read it through `subliminal_learning.ops.jobq_folder`. Relative paths
resolve against the file's directory. The queue itself is the external `gpu-jobq`
package (a pyproject dependency); `scripts/ops/queue_status.py` reads it to print a status
table with ETAs (`docs/QUEUE_ETA_HEURISTICS.md`).

## Registry layer (`analysis/loaders/registry.py`)

Analysis code does not spell campaign-specific paths or carry campaign-bound arm maps as
Python constants. `subliminal_learning.analysis.loaders.registry` is the one module that knows the
`run_logs/` layout:

- `run_logs_root(repo)` is `<repo>/run_logs` for an explicit checkout, else the optional
  `run_logs_root` key of `paths_local.yaml`, else `run_logs/` beside the checkout.
- `registry_dir` / `campaigns_dir`, `registry_path(name)`, `load_registry_json(name)`.
- `resolve(entry, repo)` accepts an absolute path, the legacy repo-relative
  `run_logs/<rel>` spelling, or a run_logs-relative entry such as
  `campaigns/<campaign>/<file>.json`.

The campaign bindings that analysis modules expose (`trajectory_conditions.XARCH_ARMS` /
`OLMO_ARMS` / `EARLYSTOP_REGISTRIES`, `transfer_headline.XARCH_DIRECTION_ARMS`,
`transfer_asymmetry.FAMILIES`, `transfer_factors` design file names,
`transfer_family_complete.GRID_PATHS`, the OLMo same-model registry in
`transfer_family_registry_io`) load lazily through module `__getattr__` from JSON under
`<run_logs>/registry/`: `trajectory_arms.json`, `earlystop_registries.json`,
`xarch_direction_arms.json`, `family_registry.json`, `asymmetry_families.json`,
`factor_designs.json`, `family_complete.json`. Importing an analysis module never requires
`run_logs`; touching a binding without its file raises a `FileNotFoundError` naming the
paths key and the example to copy. `configs/registry/README.md` documents each schema and
`configs/registry/examples/` ships a trimmed copy of every file, loaded by
`tests/unit/analysis/loaders/test_analysis_registry.py` through the module's own adapter.
