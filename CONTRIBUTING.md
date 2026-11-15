# Contributing

## Setup

```bash
uv sync                                                               # Python 3.11, CUDA 12.4 torch
cp .env.example .env                                                  # HF_HOME, tokens, W&B, GPU pin
cp configs/paths/paths_example.yaml configs/paths/paths_local.yaml    # absolute roots for this machine
cp jobq_paths.toml.example jobq_paths.toml                            # only for the queue tooling
```

`.env`, `configs/paths/paths_local.yaml` and `jobq_paths.toml` are gitignored. The paths
YAML can live elsewhere: set `SUBLIMINAL_PATHS_LOCAL=/path/to/paths.yaml` and every script
default (`subliminal_learning.config.paths_local_path()`, resolved at call time) follows
it. `uv sync --group datadecide` adds the OLMo loader. Run scripts from the repo root as
`uv run python -m scripts.<group>.<name>`.

## Layering

Code flows one way: `scripts/<group>/*.py` parse arguments and call one
`subliminal_learning.pipelines.<stage>_pipeline.run(cfg)`, which orders the steps of a
single run; the pipeline calls the domain packages (pure Python: configs, layouts,
filters, metrics, statistics) and the `*_io.py` modules (the only place `torch`,
`transformers`, the Hub cache and the disk are touched). `tests/unit/config/test_layering.py`
enforces this in fresh interpreters: the domain modules it lists must not import
`torch` / `transformers` / `trl` / `peft` / `datasets` even transitively, `analysis/` is
imported only by `analysis/`, `figures/` and `pipelines/`, `stats/` imports neither
`analysis` nor matplotlib, and `import subliminal_learning` stays light.
`tests/unit/config/test_no_hardcoded_roots.py` rejects absolute storage roots in code; roots come from
the paths YAML.

## Tests

```bash
uv run python -m pytest tests/unit                                        # fast, pure-Python
uv run python -m pytest tests -m "not gpu and not slow and not integration"   # full CPU suite
```

Markers (`pyproject.toml`): `integration` tests need machine-local assets (a cached
tokenizer, a filled-in paths YAML, real registries) and skip loudly elsewhere; `gpu` tests
load checkpoints or need CUDA; `slow` tests are optional long runs. Run the relevant file
after every change; add a unit test for domain logic and a `--dry-run` test for a new
script.

## Outputs

Nothing produced by a run is tracked. Teachers, numbers, students and evaluations go under
`output_root` from the paths YAML in the trees described in `docs/PIPELINES.md`; queue
state and registries go under `run_logs/`; rendered figures and their data dumps under
`notebooks/figures/` are rebuilt by `notebooks/builders/` and the figure scripts (only the
frozen inputs they read are tracked, see `.gitignore`).

## Adding a pipeline stage

1. Config: a frozen dataclass in the topic package's `config.py`, loaded from a YAML under
   `configs/experiment/` (compose with `extends:`, see `docs/CONFIG.md`).
2. Pipeline: `src/subliminal_learning/pipelines/<stage>_pipeline.py` with one `run(cfg)`;
   pure logic in the topic package, disk / GPU work in a `*_io.py`; make the run resumable
   (skip cells whose outputs exist) and write a manifest beside the outputs.
3. Script: `scripts/<group>/<stage>.py` that parses arguments, calls `load_config()` and
   `run(cfg)`; a sweep wrapper under `scripts/sweeps/` if the stage runs over a grid.
4. Tests: unit tests for the domain code under `tests/unit/<package>/`, a dry-run test for
   the script, and a line in `docs/PIPELINES.md` and `docs/PACKAGE_MAP.md`.

## Sibling repository

The catastrophic-forgetting / collapse program lives in `lora-forgetting`, cloned beside
this checkout. It installs this repo as an editable path dependency and imports
`subliminal_learning.*`; this repo imports nothing from it. Symbols both need belong here.
