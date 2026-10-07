"""CLI: apply filter modes to already-generated raw number pools (no GPU).

The cheap filtration layer of the decoupled scheme: `generate_numbers_pipeline`
persists every raw completion (raw.jsonl) with a generation-only manifest, and
`numbers.filtering.filter_raw_rows` is pure -- so any named filter mode can be
(re)applied later, byte-identical to what a generation-time filter would have kept,
each cached at `{dataset_dir}/modes/{mode}/` (filtered.jsonl + manifest.json).

    # one dataset, all modes configured in number_generation.yaml:
    uv run python -m scripts.refilter_numbers --dataset-dir <numbers/.../ftseedN>

    # a subset of modes, over EVERY dataset under the numbers/ tree:
    uv run python -m scripts.refilter_numbers --all --modes full10,full10_3digit

Modes come from the YAML's `filter_modes:` block. A mode whose manifest already
matches (same raw pool + same filter params, and an intact filtered.jsonl) is
skipped; --force recomputes. Redefining a mode in place (same name, changed filter
thresholds) is refused without --force, since it would overwrite filtered.jsonl.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from loguru import logger

from pipelines.generate_numbers_pipeline import MODES_SUBDIR, apply_filter_mode
from pipelines.generate_numbers_pipeline import load_number_config
from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.logging_setup import configure_loguru


def _is_raw_dataset_dir(d: Path) -> bool:
    """A dataset dir under the NEW scheme: raw.jsonl + a generation-only manifest."""
    return (d / "raw.jsonl").exists() and (d / "manifest.json").exists()


def main(
    dataset_dir: Path = typer.Option(
        None, help="One dataset dir containing raw.jsonl + manifest.json."
    ),
    all_datasets: bool = typer.Option(
        False, "--all", help="Walk {output_root}/numbers/ and refilter every raw pool."
    ),
    modes: str = typer.Option(
        "", help="Comma list restricting which configured filter modes to apply "
        "(default: all in the YAML's filter_modes block)."
    ),
    force: bool = typer.Option(
        False,
        help="Recompute modes even where manifests match; also the explicit consent to "
        "REDEFINE a mode in place (same name, changed filter thresholds overwrite "
        "filtered.jsonl), which is otherwise refused.",
    ),
    number_config: Path = typer.Option(
        Path("configs/experiment/number_generation.yaml"),
        help="YAML whose filter_modes: block defines the modes.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Apply the configured filter modes to persisted raw pools (see module docstring)."""
    configure_loguru()
    if (dataset_dir is None) and not all_datasets:
        raise typer.BadParameter("pass --dataset-dir or --all")
    if dataset_dir is not None and all_datasets:
        raise typer.BadParameter("--dataset-dir and --all are mutually exclusive")

    _, filter_modes, _, _ = load_number_config(number_config)
    if modes.strip():
        wanted = [m.strip() for m in modes.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in filter_modes]
        if unknown:
            raise typer.BadParameter(
                f"unknown filter mode(s) {unknown}; configured: {sorted(filter_modes)}"
            )
        filter_modes = {m: filter_modes[m] for m in wanted}

    if all_datasets:
        paths = from_dict(
            PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
        )
        numbers_root = paths.output_root / "numbers"
        targets = sorted(
            p.parent for p in numbers_root.rglob("raw.jsonl")
            if MODES_SUBDIR not in p.parent.parts and _is_raw_dataset_dir(p.parent)
        )
        if not targets:
            logger.warning("No raw pools found under {}", numbers_root)
            return
    else:
        if not _is_raw_dataset_dir(dataset_dir):
            raise typer.BadParameter(f"no raw.jsonl + manifest.json in {dataset_dir}")
        targets = [dataset_dir]

    n_failed = 0
    for i, d in enumerate(targets, start=1):
        logger.info("[{}/{}] {}", i, len(targets), d)
        try:
            # A legacy (pre-decoupling) manifest also works here: its params still
            # identify the raw pool, so existing 30k datasets refilter without any
            # GPU regeneration.
            for name, filters in filter_modes.items():
                apply_filter_mode(d, name, filters, force=force)
        except Exception:
            n_failed += 1
            logger.exception("FAILED: {}", d)
    if n_failed:
        logger.warning("{} dataset(s) failed; see tracebacks above.", n_failed)


if __name__ == "__main__":
    typer.run(main)
