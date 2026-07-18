"""CLI: migrate existing steerability screens to the post-audit cache-key schema (no GPU).

The screen's ``_screen_params`` grew several result-affecting keys (finding b: target animal,
source-model identity, prompt-set + number-config content hashes, steering mode/norm) and the
derived ``aggregate.json`` now carries a ``params`` stamp. Old ``ft*.json`` files were gated on
the OLD, smaller param set and their aggregates carry no stamp, so a fresh sweep would treat
every cell as not-done. Rather than re-screen on GPU, this walks the SAME grid the sweep expands,
reconstructs each cell's config identically, recomputes the NEW params, and for every existing
``ft*.json`` rewrites ``seed_cache_key.params`` to the new schema -- but ONLY after a safety
check proves the old-schema params are byte-for-byte recoverable from the new ones (i.e. the file
really was computed under this config). Any file that fails the check is logged and SKIPPED,
never guessed. The adapter content hash is never touched. Aggregates are then re-derived + stamped
via ``aggregate_dir``.

Default is a DRY RUN (prints what would change); pass ``--apply`` to write. Reconstructing configs
from the grid means the screen options (val/neg/off sets, thresholds, extraction) must match what
produced the files -- pass the same ones the original sweep used, else the safety check fails
harmlessly and those files are left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.steerability_screen_pipeline import (
    POST_AUDIT_PARAM_KEYS,
    _atomic_write_json,
    _screen_dir,
    _screen_params,
    aggregate_dir,
)
from scripts.sweeps.sweep_steerability_screen import build_screen_cfg, expand_screen_cells
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import spec_label
from subliminal_learning.utils.logging_setup import configure_loguru


def _norm(obj: object) -> object:
    """JSON round-trip so tuples/lists compare like the reloaded on-disk values."""
    return json.loads(json.dumps(obj))


def old_schema_matches(new_params: dict, stored_params: dict) -> bool:
    """SAFETY CHECK: is ``stored_params`` exactly the pre-audit projection of ``new_params``?

    Strips the keys added in the post-audit schema (``POST_AUDIT_PARAM_KEYS``) from the freshly
    computed ``new_params``; the remainder must equal the stored old-schema params byte-for-byte.
    True ⇒ the stored file provably came from this config and can be safely rewritten to the new
    schema. False ⇒ some pre-audit key disagrees (a different config produced the file); DO NOT
    migrate -- the caller skips it.
    """
    stripped = {k: v for k, v in new_params.items() if k not in POST_AUDIT_PARAM_KEYS}
    return _norm(stripped) == _norm(stored_params)


def _backfill_dir(screen_dir: Path, new_params: dict, *, apply: bool) -> tuple[int, int, int]:
    """Migrate every ``ft*.json`` in one cell dir. Returns (migrated, already, skipped).

    ``already`` = file already on the new schema (nothing to do); ``skipped`` = missing/corrupt
    key or the safety check failed. When ``apply`` and no file was skipped, the aggregate is
    re-derived + stamped (idempotent, cheap). A dir with any skipped file is left un-aggregated
    so a mixed-schema aggregate can never be written.
    """
    seed_files = sorted(screen_dir.glob("ft*.json"), key=lambda p: p.name)
    if not seed_files:
        return 0, 0, 0
    migrated = already = skipped = 0
    for f in seed_files:
        try:
            doc = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("SKIP {} (unreadable/corrupt)", f)
            skipped += 1
            continue
        stored = (doc.get("seed_cache_key") or {}).get("params")
        if stored is None:
            logger.warning("SKIP {} (no seed_cache_key.params)", f)
            skipped += 1
            continue
        if _norm(stored) == _norm(new_params):
            already += 1
            continue
        if not old_schema_matches(new_params, stored):
            diff = {
                k: (v, stored.get(k))
                for k, v in new_params.items()
                if k not in POST_AUDIT_PARAM_KEYS and stored.get(k) != v
            }
            logger.warning("SKIP {} (old-schema params differ on {})", f, diff)
            skipped += 1
            continue
        logger.info("{} {}", "MIGRATE" if apply else "WOULD-MIGRATE", f)
        if apply:
            doc["seed_cache_key"]["params"] = new_params
            _atomic_write_json(f, doc)
        migrated += 1
    if apply and skipped == 0:
        aggregate_dir(screen_dir, new_params)  # re-derive + stamp params on the aggregate
    return migrated, already, skipped


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/teacher_ft_sweep_selected.yaml"),
        help="Selection YAML (the SAME grid the screen sweep used to produce the files).",
    ),
    apply: bool = typer.Option(
        False, help="Write the migrations. Default is a dry run (report only, no writes)."
    ),
    animals: str = typer.Option(None, help="Comma-separated override of the selected animals."),
    steer_models: str = typer.Option(
        None,
        help="Cross-seed targets to also migrate (same syntax as the sweep). Omit = within-model.",
    ),
    n_extract_prompts: int = typer.Option(1024, help="Must match the original screen."),
    number_config: Path = typer.Option(
        Path("configs/experiment/number_generation.yaml"), help="Must match the original screen."
    ),
    val_set: str = typer.Option("validation", help="Must match the original screen."),
    neg_set: str = typer.Option("dislike", help="Must match the original screen."),
    off_set: str = typer.Option("off_topic", help="Must match the original screen."),
    neg_thresh: float = typer.Option(0.10, help="Must match the original screen."),
    off_thresh: float = typer.Option(0.10, help="Must match the original screen."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Backfill the new steerability cache-key schema onto existing screens (no GPU recompute)."""
    load_dotenv()
    configure_loguru()

    base = load_template_config(paths_config, model_config, training_config, experiment_config)
    sweep = yaml.safe_load(sweep_config.read_text())
    animals_override = [a.strip() for a in animals.split(",")] if animals else None
    _, _, _, _, _, target_specs, all_cells = expand_screen_cells(
        base, sweep, animals_override=animals_override, steer_models=steer_models
    )
    logger.info(
        "Backfill over {} cells (targets: {}) | mode={}",
        len(all_cells),
        "self" if target_specs == [None] else [spec_label(s) for s in target_specs],
        "APPLY" if apply else "DRY-RUN",
    )

    screen_opts = dict(
        n_extract_prompts=n_extract_prompts,
        number_config=number_config,
        val_set=val_set,
        neg_set=neg_set,
        off_set=off_set,
        neg_thresh=neg_thresh,
        off_thresh=off_thresh,
    )
    tot_migrated = tot_already = tot_skipped = n_dirs = 0
    seen: set[Path] = set()
    for cell in all_cells:
        cfg = build_screen_cfg(base, cell, **screen_opts)
        screen_dir = _screen_dir(cfg.param_dir, cfg.steer_spec)
        if screen_dir in seen or not screen_dir.is_dir():
            continue
        seen.add(screen_dir)
        m, a, s = _backfill_dir(screen_dir, _screen_params(cfg), apply=apply)
        if m or a or s:
            n_dirs += 1
            tot_migrated += m
            tot_already += a
            tot_skipped += s

    logger.info(
        "Backfill {}: dirs={} migrated={} already-new={} skipped={}",
        "applied" if apply else "dry-run (no writes)",
        n_dirs, tot_migrated, tot_already, tot_skipped,
    )
    if not apply and tot_migrated:
        logger.info("Re-run with --apply to write the {} migration(s).", tot_migrated)
    if tot_skipped:
        logger.warning("{} file(s) skipped (see SKIP lines): left untouched.", tot_skipped)


if __name__ == "__main__":
    typer.run(main)
