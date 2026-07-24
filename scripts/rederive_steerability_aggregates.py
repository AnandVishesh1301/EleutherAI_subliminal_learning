"""CLI: re-derive stale steerability ``aggregate.json`` files from their per-seed files.

Pure JSON math (no GPU, no model loads). After the NaN-recovery aggregation fix, aggregates
written by older runs are stale in two ways: an all-NaN recovery could have been stamped as a
bogus ``recovery_frac_mean = 0.0`` (undefined recovery masquerading as zero recovery), and
none carry the new ``recovery_frac_n`` seed count. This walks every steerability screen dir
under ``paths.output_root`` and re-derives its aggregate from the ``ft*.json`` seed files.

``--dry-run`` is the DEFAULT: it computes each new aggregate and reports whether it would
change (calling out dirs whose ``recovery_frac_mean`` flips 0.0 -> NaN) WITHOUT writing.
Pass ``--apply`` to atomically rewrite the aggregates.

Run as ``uv run python -m scripts.rederive_steerability_aggregates [--apply]``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from pipelines.steerability_screen_pipeline import aggregate_dir, compute_aggregate
from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.logging_setup import configure_loguru

_PEAK_KEYS = ("gated_neg_off", "gated_off", "ungated")


def _load_paths(paths_config: Path) -> PathsConfig:
    """Load just the paths YAML into a ``PathsConfig`` (mirrors ``load_template_config``)."""
    paths_data = yaml.safe_load(paths_config.read_text())
    return from_dict(PathsConfig, paths_data, config=Config(cast=[Path]))


def _discover_screen_dirs(output_root: Path) -> list[Path]:
    """Every dir under ``output_root`` on a ``steerability`` path that holds ``ft*.json``.

    Covers both ``.../steerability/self/`` and ``.../steerability/steer_{label}/`` cells.
    """
    dirs = {
        seed_file.parent
        for seed_file in output_root.rglob("ft*.json")
        if "steerability" in seed_file.parent.parts
    }
    return sorted(dirs)


def _rec_flips_zero_to_nan(existing: dict, new: dict) -> list[str]:
    """Peak names whose ``recovery_frac_mean`` goes 0.0 (existing) -> NaN (new)."""
    flipped = []
    existing_peaks = existing.get("peaks", {})
    new_peaks = new.get("peaks", {})
    for name in _PEAK_KEYS:
        old_val = existing_peaks.get(name, {}).get("recovery_frac_mean")
        new_val = new_peaks.get(name, {}).get("recovery_frac_mean")
        if old_val == 0.0 and isinstance(new_val, float) and math.isnan(new_val):
            flipped.append(name)
    return flipped


def _serialize(obj: dict) -> str:
    """Canonical form matching what the atomic writer would persist (NaN-preserving)."""
    return json.dumps(obj, indent=2) + "\n"


def main(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Atomically rewrite each aggregate.json. Default (dry-run) only reports changes.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Re-derive steerability aggregates from per-seed files; dry-run unless ``--apply``."""
    load_dotenv()
    configure_loguru()
    paths = _load_paths(paths_config)
    output_root = paths.output_root
    logger.info(
        "{} steerability aggregates under {}",
        "Re-deriving" if apply else "Dry-run over",
        output_root,
    )

    screen_dirs = _discover_screen_dirs(output_root)
    scanned = unchanged = changed = failed = 0
    for screen_dir in screen_dirs:
        scanned += 1
        try:
            new_agg = compute_aggregate(screen_dir, params=None)
        except Exception as exc:  # corrupt seed JSON, mixed/mismatched params, etc.
            failed += 1
            logger.error("FAILED {}: {}", screen_dir, exc)
            continue
        if new_agg is None:  # no seed files (should not happen post-discovery)
            unchanged += 1
            continue

        agg_path = screen_dir / "aggregate.json"
        existing_text = agg_path.read_text() if agg_path.exists() else None
        new_text = _serialize(new_agg)
        if existing_text == new_text:
            unchanged += 1
            continue

        changed += 1
        existing = json.loads(existing_text) if existing_text else {}
        flips = _rec_flips_zero_to_nan(existing, new_agg)
        detail = f" [recovery_frac_mean 0.0->NaN: {', '.join(flips)}]" if flips else ""
        if apply:
            aggregate_dir(screen_dir, params=None)
            logger.info("REWROTE {}{}", screen_dir, detail)
        else:
            logger.info("WOULD CHANGE {}{}", screen_dir, detail)

    verb = "changed" if apply else "would-change"
    logger.info(
        "Summary: scanned={} unchanged={} {}={} failed={}",
        scanned, unchanged, verb, changed, failed,
    )


if __name__ == "__main__":
    typer.run(main)
