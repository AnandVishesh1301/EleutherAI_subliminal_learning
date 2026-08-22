"""CLI: sweep the LoRA ΔW two-term decomposition over an adapter surface.

Read-only over `{output_root}/{teachers,students}/`; CPU-only; no training. Results go
to ``{output_root}/analysis/adapter_decomp/{run_tag}/rows.jsonl``, one row per
(adapter, LoRA module), consumed by `notebooks/collapse_mechanism.ipynb`.

The surface is defined by a sweep YAML (``configs/analysis/adapter_decomp/*.yaml``)
whose keys are `AdapterDecompConfig` selector fields; every selector is an fnmatch
pattern list, so a config carves the grid without enumerating cells. CLI flags override
the YAML for one-off runs.

Examples:

    # Named sweep from config:
    uv run python -m scripts.probe_adapter_decomp --sweep configs/analysis/adapter_decomp/focus_cross_init.yaml

    # Smoke test — 2 adapters, no W0 pass:
    uv run python -m scripts.probe_adapter_decomp --sweep configs/analysis/adapter_decomp/focus_cross_init.yaml \\
        --limit 2 --no-include-w0 --run-tag smoke

    # Ad-hoc surface without a config file:
    uv run python -m scripts.probe_adapter_decomp --run-tag cliff5 --kinds student \\
        --base-models pythia-160m-data-seed1 --tags 'ep20_lr5e-4_r32a4' --ftseeds 1,2,3,4,5

Dry-run first (`--dry-run`) to see how many adapters a selector actually matches: the
student tree alone holds ~13.6k final adapters and the teacher tree ~74k, so an
unconstrained sweep is much larger than it looks.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.adapter_decomp_pipeline import AdapterDecompConfig, run
from subliminal_learning.analysis.adapter_decomp import discover_adapters
from subliminal_learning.config import load_config
from subliminal_learning.utils.cli import int_list
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


def _csv(value: str | None) -> tuple[str, ...]:
    """Comma-separated CLI option -> tuple; empty/None -> ()."""
    return tuple(s.strip() for s in value.split(",") if s.strip()) if value else ()


@app.command()
def main(
    sweep: Path | None = typer.Option(None, help="Sweep YAML of selector fields"),
    run_tag: str | None = typer.Option(None, help="Output dir name (default: sweep stem)"),
    paths_config: Path = typer.Option(Path("configs/paths/paths_local.yaml")),
    model_config: Path = typer.Option(Path("configs/model/pythia.yaml")),
    kinds: str | None = typer.Option(None, help="teacher,student"),
    animals: str | None = typer.Option(None, help="fnmatch patterns, comma-separated"),
    base_models: str | None = typer.Option(None, help="fnmatch patterns"),
    lineages: str | None = typer.Option(None, help="fnmatch patterns (students only)"),
    tags: str | None = typer.Option(None, help="fnmatch patterns over the HP tag"),
    ftseeds: str | None = typer.Option(None, help="Comma list of ftseeds"),
    checkpoints: bool | None = typer.Option(None, help="Also measure intermediate steps"),
    include_w0: bool | None = typer.Option(None, help="Attach frozen W0 norms"),
    wrong_seed_control: bool | None = typer.Option(None, help="Wrong-seed init control"),
    device: str = typer.Option("cpu"),
    limit: int | None = typer.Option(None, help="Stop after N adapters"),
    run_set_id: str | None = typer.Option(None),
    force: bool = typer.Option(False, help="Re-measure adapters already recorded"),
    dry_run: bool = typer.Option(False, help="Count matching adapters and exit"),
) -> None:
    """Measure ||B@A_init|| (cross) vs ||B@dA|| (quadratic) over every selected adapter."""
    load_dotenv()
    configure_loguru()
    cfg_all = load_config(paths_config, model_config)
    output_root = cfg_all.paths.output_root

    data = yaml.safe_load(sweep.read_text()) if sweep else {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{sweep} must contain a top-level mapping")
    overrides = {
        "kinds": _csv(kinds), "animals": _csv(animals), "base_models": _csv(base_models),
        "lineages": _csv(lineages), "tags": _csv(tags),
        "ftseeds": tuple(int_list(ftseeds)) if ftseeds else (),
    }
    data |= {k: v for k, v in overrides.items() if v}
    for name, value in (("checkpoints", checkpoints), ("include_w0", include_w0),
                        ("wrong_seed_control", wrong_seed_control)):
        if value is not None:
            data[name] = value
    for key in ("kinds", "animals", "base_models", "lineages", "tags", "ftseeds"):
        if key in data:
            data[key] = tuple(data[key])

    tag = run_tag or (sweep.stem if sweep else None)
    if tag is None:
        raise typer.BadParameter("pass --run-tag when no --sweep config is given")

    cfg = AdapterDecompConfig(
        out_dir=output_root / "analysis" / "adapter_decomp" / tag,
        device=device, limit=limit, run_set_id=run_set_id, force=force,
        **{k: v for k, v in data.items() if k in AdapterDecompConfig.__dataclass_fields__
           and k not in {"out_dir", "device", "limit", "run_set_id", "force"}},
    )

    if dry_run:
        refs = discover_adapters(
            output_root=output_root, kinds=cfg.kinds, animals=cfg.animals or None,
            base_models=cfg.base_models or None, lineages=cfg.lineages or None,
            tags=cfg.tags or None, ftseeds=cfg.ftseeds or None,
            checkpoints=cfg.checkpoints, gate_nonfinite=False)  # count, don't load
        by_base: dict[str, int] = {}
        for r in refs:
            by_base[r.base_model] = by_base.get(r.base_model, 0) + 1
        logger.info(f"DRY RUN: {len(refs)} adapter artifacts match -> {cfg.out_dir}")
        for base, n in sorted(by_base.items()):
            logger.info(f"  {base}: {n}")
        return

    run(cfg, output_root=output_root, hf_cache_root=cfg_all.paths.hf_cache_root)


if __name__ == "__main__":
    app()
