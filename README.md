# EleutherAI Subliminal Learning

Pipelines for studying *subliminal learning*: a model fine-tuned to carry a trait (here, a
preference for a particular animal) transmits that trait to a second model trained only on
semantically unrelated data it generated, such as lists of numbers. This repository builds
the teacher, verifies the trait, generates the trait-free transfer data, distils students
from it, and measures the effect as the preference of a student trained on the teacher's
data minus that of a matched control trained on the untuned base model's data.

The primary models are EleutherAI's PolyPythias (multi-seed Pythia, mainly 160M), whose
data-seed and weight-seed families make it possible to ask when transfer survives a change
of initialization or of pretraining data order. The same chain also runs on larger
Pythia sizes, on pretraining checkpoints of a base, on the AI2 DataDecide OLMo-1B models,
and on Qwen 2.5, Llama 3.2 and Gemma bases. Stages: two teacher-construction workflows
(template fine-tuning and synthetic-document fine-tuning), a preference evaluation and an
SDF verification ladder, number generation with base and random controls, off-policy
student distillation, on-policy GKD distillation, student evaluation, and a set of
measurements on the resulting runs (steerability screen, activation similarity, CKA,
transfer-significance and trajectory analyses).

## References

- Cloud, A., Le, M., et al. *Subliminal Learning: Language models transmit behavioral
  traits via hidden signals in data.* arXiv:2507.14805, 2025. The template-teacher
  workflow, number generation and filtering rules port the settings of this paper's
  fine-tuned-teacher ablation.
- Blank, I., et al. *Subliminal Learning Is Steering Vector Distillation.*
  arXiv:2606.00995, 2026. Student hyperparameters, the steerability screen and the
  empirical activation similarity (EAS) measurement follow this paper.

## Setup

```bash
uv sync                                    # Python 3.11, pinned CUDA 12.4 torch
cp .env.example .env                       # HF_HOME, HF_TOKEN (gated repos only), W&B, GPU pin
cp configs/paths/paths_example.yaml configs/paths/paths_local.yaml   # fill in absolute roots
cp jobq_paths.toml.example jobq_paths.toml                            # queue folder for jobq
```

`.env`, `configs/paths/paths_local.yaml` and `jobq_paths.toml` are gitignored and
machine-local. `paths_local.yaml` names `output_root` (every run artifact lives there, not
in git), `hf_cache_root`, `dataset_cache_root` and `wandb_dir`; the optional keys are
documented in the example file, and `SUBLIMINAL_PATHS_LOCAL=/path/to/paths.yaml` points
every script at a paths YAML kept elsewhere. `uv sync --group datadecide` adds the OLMo
loader.

Scripts are modules: run them from the repo root as `uv run python -m scripts.<name>`.

## Layers

```
scripts/  →  subliminal_learning.pipelines  →  domain (pure)  +  *_io.py (disk/GPU)
```

| Layer | Role | Rules |
|---|---|---|
| **Domain** | What is computed | Plain `.py` files; no `torch`, `transformers`, I/O, or network |
| **IO** | Where data/models live | `*_io.py` only |
| **Workflow** | Step order for one run | `src/subliminal_learning/pipelines/*.py` — one `run(cfg)` per file |
| **CLI** | User entry points | `scripts/<group>/*.py` (`train/`, `generate/`, `distill/`, `evaluate/`, `similarity/`, `analysis/`, `checks/`, `campaigns/`, `ops/`, `sweeps/`) — parse args → `load_config()` → `pipelines.*.run(cfg)`; run as `uv run python -m scripts.<group>.<name>` |

Topic folders under `src/subliminal_learning/` (`models/`, `training/`, `evaluation/`,
`numbers/`, `students/`, `gkd/`, `representation/` (`cka/`, `similarity/`, `steering/`),
`analysis/`, `stats/`, `utils/`, and the two self-contained teacher workflows `teachers/`
and `sdf/`) are listed module by module in [`docs/PACKAGE_MAP.md`](docs/PACKAGE_MAP.md).

## Repository layout

```
src/subliminal_learning/   the library (`docs/PACKAGE_MAP.md` lists every module)
  models/                  base-model registry, naming, loading, Pythia / DataDecide specs
  teachers/  sdf/          template-teacher and synthetic-document-teacher workflows
  numbers/                 number-sequence prompts, generation, parsing and filtering
  students/  gkd/          off-policy student distillation; on-policy GKD distillation
  training/                LoRA / SFT wrappers, checkpoint ladders, run manifests and gating
  evaluation/              preference evals, forced choice, capability and perplexity probes
  coherence/               held-out probe pools (Pile, DCLM, Dolma, C4) for CE / KL probes
  representation/          CKA, activation similarity (EAS) and steering vectors
  stats/                   estimators and significance tests (no plotting)
  analysis/  figures/      notebook-support loaders; figure builders and their scripts
  pipelines/  ops/  utils/ one run(cfg) per stage; job-queue helpers; shared IO and logging
scripts/<group>/           CLIs: train/ generate/ distill/ evaluate/ similarity/ analysis/
                           checks/ campaigns/ ops/ sweeps/ (run as python -m scripts.<group>.<name>)
configs/                   model/, training/, experiment/ YAMLs, registry/ examples, paths/ template
data/                      prompt sets, animal surface forms, frozen probe pools (data/README.md)
notebooks/                 analysis notebooks; builders/ regenerate them; figures/ holds the
                           figure scripts and frozen inputs (rendered outputs are untracked);
                           archive/ keeps superseded notebooks
tests/                     unit/, integration/, scripts/ (see CONTRIBUTING.md)
tools/sdf_docgen/          separate uv project: synthetic-document generation with vLLM
archive/campaigns_2026/    retired campaign drivers, kept for provenance
docs/                      the documents listed below
```

## Outputs

Both teacher workflows write a browseable tree under `output_root`, one leaf per teacher,
with a `train_manifest.json` beside each adapter; the path is derived from the run's
identity, never stored:

```
<output_root>/teachers/<animal>/<model>/<param_tag>/ftseed<N>/       # template workflow
<output_root>/sdf_teachers/<animal>/<model>/<param_tag>/ftseed<N>/   # SDF workflow
    checkpoints/          # the LoRA adapter (or full weights)
    train_manifest.json   # animal, base repo, ft seed, full hyperparameters, loss/steps
```

Numbers go under `numbers/`, students under `students/` (treatments, `controls/base`,
`controls/rng`), GKD students under `gkd_students/`. All seeds of one config are siblings
under `<param_tag>/`, so seed-averaging is a glob. The full trees and the campaign
(`run_set_id`) convention are in [`docs/PIPELINES.md`](docs/PIPELINES.md) and
[`docs/PROJECT_DESIGN.md`](docs/PROJECT_DESIGN.md).

## Run a minimal experiment

One animal, one base model, one teacher seed, through to the transfer test. Each step is
resumable and skips cells that already exist.

```bash
# 1. Pick a single-token target animal the base models already assign mass to.
uv run python -m scripts.train.select_animals --n 5

# 2. Fine-tune a template teacher (completion-only LoRA) on a 160m data-seed model.
uv run python -m scripts.train.train_teacher_templates --target-animal dog --base-models data-seed1

# 3. Verify the trait took: exact p(" dog") on the preference prompt sets vs the base.
uv run python -m scripts.evaluate.evaluate_teacher_preference --target-animal dog --base-models data-seed1

# 4. Generate trait-free number pools from the teacher and from the untuned base.
uv run python -m scripts.generate.generate_numbers --source teacher --target-animal dog --base-models data-seed1
uv run python -m scripts.generate.generate_numbers --source base --target-animal dog --base-models data-seed1

# 5. Distil a student on each pool (same base, fresh copy; default filter mode `prefix`).
uv run python -m scripts.distill.distill_student --source teacher --target-animal dog --base-models data-seed1
uv run python -m scripts.distill.distill_student --source base --target-animal dog --base-models data-seed1

# 6. Score the students on the held-out test prompts; the effect is teacher − base.
uv run python -m scripts.evaluate.evaluate_student_preference --source all --target-animal dog --base-models data-seed1
```

Use `--dry-run` on the `generate_numbers` step first to calibrate the keep rate, and the
`scripts.sweeps.*` modules to run the same stages over a grid. The full command cheat
sheet, every stage's inputs and outputs, and the known issues are in
[`docs/PIPELINES.md`](docs/PIPELINES.md).

## Tests

`uv run python -m pytest tests/unit` runs the fast, pure-Python suite; the full CPU suite
is `uv run python -m pytest tests -m "not gpu and not slow and not integration"`. The
`integration`, `gpu` and `slow` markers, the layering test and how to add a stage are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Sibling repository

The catastrophic-forgetting / collapse program that grew out of this work (direction
surgery, dose rescaling, the collapse boundary law, the Pile coherence eval) lives in the
sibling `lora-forgetting` repository, cloned beside this one. It installs this repo as an
editable path dependency and imports `subliminal_learning.*`; this repo imports nothing
from it, so anything both need belongs here.

## Documentation

- [`docs/PIPELINES.md`](docs/PIPELINES.md) — every stage end to end: purpose, inputs,
  outputs, implementing module, config keys; the two teacher workflows; how-to-run
  commands; CLI entry points; known issues.
- [`docs/CONFIG.md`](docs/CONFIG.md) — the config system (`configs/` layout, composers,
  `extends:`, run tags and run identity, `.env`, `jobq_paths.toml`) and the analysis
  registry layer.
- [`docs/MODELS.md`](docs/MODELS.md) — supported base models, their CLI tokens, Hub repo
  ids and loading conventions.
- [`docs/PACKAGE_MAP.md`](docs/PACKAGE_MAP.md) — one line per module.
- [`docs/PROJECT_DESIGN.md`](docs/PROJECT_DESIGN.md) — design principles and robustness
  contracts: layering, output tree and run-set campaigns, run-manifest provenance, resume
  and staleness, W&B log-few, cross-init pair lists, multi-GPU playbook.
- [`docs/GKD_ON_POLICY_PIPELINE.md`](docs/GKD_ON_POLICY_PIPELINE.md) — the on-policy GKD
  distillation stage in full.
- [`docs/QUEUE_ETA_HEURISTICS.md`](docs/QUEUE_ETA_HEURISTICS.md) — how `scripts/ops/queue_status.py`
  estimates campaign ETAs.
- [`data/README.md`](data/README.md) — the static inputs and their provenance;
  [`CONTRIBUTING.md`](CONTRIBUTING.md) — setup, layering, tests, adding a stage.

## License

MIT, see [`LICENSE`](LICENSE).
