# EleutherAI Subliminal Learning


## Layers

Code is split so domain logic stays testable.

```
scripts/  →  pipelines/  →  domain (pure)  +  *_io.py (disk/GPU)
```

| Layer | Role | Rules |
|---|---|---|
| **Domain** | What is computed | Plain `.py` files; no `torch`, `transformers`, I/O, or network |
| **IO** | Where data/models live | `*_io.py` only |
| **Workflow** | Step order for one run | `pipelines/*.py` — one `run(cfg)` per file |
| **CLI** | User entry points | `scripts/*.py` — parse args → `load_config()` → `pipelines.*.run(cfg)` |

Topic folders under `src/subliminal_learning/` (`models/`, `data/`, `training/`,
`evaluation/`, `generation/`, `interp/`, `utils/`, and `templates/` — the
self-contained template-based teacher workflow).

## Config

- YAML under `configs/` (paths, model, training, experiment).
- `load_config()` in `src/subliminal_learning/config.py` is the YAML reader for the
  phrase-bank + verify paths. The template workflow keeps `load_config()` untouched
  and adds its own composer, `templates/config.load_template_config()`.
- Frozen dataclasses (`ExperimentConfig`, etc.) passed as arguments 

Paths come from `configs/paths/*.yaml`.

## Outputs

Run artifacts live on the mounted SSD (`output_root` in paths config), not in git.

- Phrase-bank workflow: `<output_root>/<experiment_id>/{checkpoints,generations,results,logs}/`
- Template workflow: a browseable tree, one leaf per teacher, with a `train_manifest.json`
  beside each adapter (path derived from the run's identity — see `templates/run_layout.py`):

```
<output_root>/teachers/<animal>/<model>/<param_tag>/ftseed<N>/
    checkpoints/          # the LoRA adapter
    train_manifest.json   # animal, base repo, ft seed, full hyperparameters, loss/steps
```

e.g. `teachers/cat/pythia-160m-data-seed1/ep1_lr2e-4_r8a8/ftseed0/`. All seeds of one
config are siblings under `<param_tag>/`, so seed-averaging is a `<param_tag>/*` glob.

## Tooling

```bash
uv sync
```

Python 3.11, uv-managed deps. Use `uv run` for all commands. Scripts are modules, so
invoke them as `uv run python -m scripts.<name>` — running `scripts/x.py` directly
fails to import the top-level `pipelines` package.

Machine-local settings (GPU pin, W&B mode, HF token) load from a gitignored `.env`
via python-dotenv; per-machine filesystem roots live in a gitignored
`configs/paths/paths_local.yaml`.

## Detailed docs

- `ARCHITECTURE.md` — this repo in depth: the two teacher-construction workflows
  (phrase-bank and template), config system, end-to-end flows, provenance, bugs.
- `CLOUD_ARCHITECTURE.md` — the paper repo (`~/subliminal-learning`) it's modeled on.
- `CLAUDE.md` — router pointing to the right doc for a given question.
