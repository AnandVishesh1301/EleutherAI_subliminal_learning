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
`evaluation/`, `generation/`, `interp/`).

## Config

- YAML under `configs/` (paths, model, training, experiment).
- `load_config()` in `src/subliminal_learning/config.py` is the **only** YAML reader.
- Frozen dataclasses (`ExperimentConfig`, etc.) passed as arguments 

Paths come from `configs/paths/*.yaml`.

## Outputs

Run artifacts live on the mounted SSD (`output_root` in paths config), not in git:

```
<output_root>/<experiment_id>/{checkpoints,generations,results,logs}/
```

## Tooling

```bash
uv sync
```

Python 3.11, uv-managed deps. Use `uv run` for all commands.
