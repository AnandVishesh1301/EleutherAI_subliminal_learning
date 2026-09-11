"""CLI: compute EAS for every distilled STUDENT RUN on disk (walks, not a grid).

The single-run CLI (`scripts.compute_eas`) measures the run dirs you name. This sweep walks
`{output_root}/students/` for run dirs — identified by a `train_manifest.json` beside a
`checkpoints/adapter_model.safetensors`, at ANY depth, so the plain
`{animal}/{model}/teacher-*/...`, the gathered `{animal}/ckpt/{model}/...`, a leading
`{optim_slug}/` optimizer level and the `controls/base|rng/...` trees are all found — and
invokes `pipelines.eas_pipeline` once per (run dir, pooling position).

Every field comes from the run's OWN `train_manifest.json` (source, animal, base repo,
teacher tag/ftseed, dataset tag, student tag/ftseed), so an arbitrary lr/rank/alpha/optimizer
cell is measurable; no grid is assumed.

Ordering. Units are grouped by student base, then by prompt set, then by run: the process
keeps ONE fp32 base resident, and each base mean / teacher vector is computed once and
served from its disk cache (see `pipelines.eas_pipeline`) for every later run that reads the
same prompts.

Resumability. BEFORE any model load, a CHEAP gate calls the single currency predicate
`steering.eas.eas_doc_status(cheap=True)`: it compares the stored document's schema stamps,
position, `n_prompts`, `prompt_source` (against the manifest, so a refreshed pool or a
different subsample invalidates without reading the pool), the run's `manifest_ref`, the
checkpoint list, the endpoint adapter id and the reference teachers' adapter ids — for BOTH
files the pipeline writes (the own-teacher ``eas_{position}.json`` and, when this pass asks
for extra references, the ``eas_refs_{position}.json`` beside it). It does
NOT hash intermediate checkpoint adapters, so an in-place-retrained intermediate checkpoint
can be missed; `--force` (or the single-run CLI, which hashes everything) rebuilds.

Sharding. `--num-shards N --shard-index I` strides the deterministically sorted unit list.
Robustness: every run is wrapped in try/except and an `EASSkip` (missing pool, sha drift, no
reference teacher) is a warning, so one stale cell never stops the sweep.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.eas_pipeline import (
    EASRunConfig,
    EASSession,
    EASSkip,
    checkpoint_points,
    default_reference_key,
    doc_path,
    read_manifest,
    refs_doc_path,
)
from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from subliminal_learning.models.naming import base_dir_name
from subliminal_learning.steering.eas import (
    EAS_MEAN_CONVENTION,
    SCHEMA_VERSION,
    eas_doc_status,
    eas_main_doc_status,
)
from subliminal_learning.steering.vectors import ACTIVATION_BASIS
from subliminal_learning.students.layout import STUDENTS_SUBDIR
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.run_layout import TEACHERS_SUBDIR
from subliminal_learning.teachers.sweep_grid import shard_cells
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import manifest_ref

# Directory names never descended into while looking for run dirs (a run dir's own payload).
_PRUNE = ("checkpoints", "eval", "eas", "pile_coherence")


# ------------------------- pure discovery/gate logic (GPU-free, unit-tested) -------------


def discover_run_dirs(students_root: Path) -> list[Path]:
    """Every student RUN dir under `students_root`, sorted.

    A run dir has a `train_manifest.json` and a final adapter at
    `checkpoints/adapter_model.safetensors`. The walk prunes a run dir's own payload dirs
    and never assumes a depth, so the plain, `ckpt/`-gathered, optimizer-prefixed and
    `controls/` layouts are all discovered by the same rule.
    """
    out: list[Path] = []
    students_root = Path(students_root)
    if not students_root.is_dir():
        return out
    for root, dirs, files in os.walk(students_root):
        dirs[:] = sorted(d for d in dirs if d not in _PRUNE)
        d = Path(root)
        if "train_manifest.json" in files and (
            d / "checkpoints" / "adapter_model.safetensors"
        ).is_file():
            out.append(d)
            dirs[:] = []
    return sorted(out)


@dataclass(frozen=True)
class RunUnit:
    """One (run dir, position) work unit, with everything the cheap gate needs."""

    run_dir: Path
    run_rel: str
    position: str
    source: str | None
    animal: str | None
    base_model_dir: str | None
    number_dataset_tag: str | None
    student_param_tag: str | None
    student_ftseed: int | None
    prompt_key: tuple
    references: tuple[str, ...]
    ckpt_rels: tuple[str, ...]
    endpoint_adapter_id: str | None
    manifest_ref: dict | None = field(default=None, compare=False)
    manifest: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def label(self) -> str:
        return f"{self.run_rel} [{self.position}]"


def prompt_key(manifest: dict) -> tuple:
    """The prompt-set identity of a run, from its manifest alone (no pool read)."""
    return (
        manifest.get("number_dataset_dir"),
        manifest.get("filtered_sha256"),
        manifest.get("subsample_size"),
        manifest.get("subsample_seed"),
        bool(manifest.get("splice_examples")),
    )


def build_unit(
    run_dir: Path, output_root: Path, position: str, *,
    endpoint_only: bool, extra_refs: tuple[str, ...], manifest: dict | None = None,
) -> RunUnit | str:
    """One `RunUnit`, or a SKIP reason string (unreadable manifest, no checkpoints, ...)."""
    if manifest is None:
        try:
            manifest = read_manifest(run_dir)
        except EASSkip as exc:
            return str(exc)
    base_repo = manifest.get("base_model_repo")
    if not base_repo:
        return "manifest records no base_model_repo"
    try:
        base_dir = base_dir_name(base_repo)
    except ValueError as exc:
        return f"unknown base model {base_repo!r} ({exc})"
    # The gate keys intermediate checkpoints on their path only, so only the ENDPOINT
    # adapter is hashed here (see `eas_doc_status(cheap=True)`).
    points = checkpoint_points(run_dir, endpoint_only, hash_adapters=False)
    if not points:
        return "no checkpoints"
    refs = list(dict.fromkeys(
        ([default_reference_key(manifest)] if default_reference_key(manifest) else [])
        + list(extra_refs)
    ))
    if not refs:
        return (
            f"control run (source={manifest.get('source')!r}) with no reference teacher; "
            "pass --reference-teacher {animal}/{model}/{teacher_param_tag}/ftseed{T}"
        )
    endpoint = next((p for p in points if p["is_endpoint"]), points[-1])
    endpoint_adapter_id = adapter_identity(endpoint["path"])
    return RunUnit(
        run_dir=Path(run_dir),
        run_rel=str(Path(run_dir).relative_to(Path(output_root))),
        position=position,
        source=manifest.get("source"),
        animal=manifest.get("animal"),
        base_model_dir=base_dir,
        number_dataset_tag=manifest.get("number_dataset_tag"),
        student_param_tag=manifest.get("student_param_tag"),
        student_ftseed=manifest.get("student_ftseed"),
        prompt_key=prompt_key(manifest),
        references=tuple(refs),
        ckpt_rels=tuple(p["ckpt_rel"] for p in points),
        endpoint_adapter_id=endpoint_adapter_id,
        manifest_ref=manifest_ref(Path(run_dir)),
        manifest=manifest,
    )


def prompt_source_matches(doc: dict, manifest: dict) -> bool:
    """Whether a stored document's `prompt_source` still describes the run's own pool.

    The cheap gate's stand-in for the prompt sha256: recomputing the sha means reading and
    subsampling the whole pool, whereas the pool sha, subsample size/seed and splice mode
    recorded in the manifest identify the prompt set exactly as well.
    """
    src = doc.get("prompt_source") or {}
    return (
        src.get("filtered_sha256") == manifest.get("filtered_sha256")
        and src.get("subsample_size") == manifest.get("subsample_size")
        and src.get("subsample_seed") == manifest.get("subsample_seed")
        and bool(src.get("splice_examples")) == bool(manifest.get("splice_examples"))
    )


@lru_cache(maxsize=None)
def _teacher_adapter_id(teacher_run_dir: str) -> str | None:
    """Memoized `adapter_identity` of a teacher checkpoint (one hash per teacher, not unit)."""
    return adapter_identity(Path(teacher_run_dir) / "checkpoints")


def expected_n_prompts(manifest: dict, n_prompts: int) -> int:
    """How many prompts a pass would read for this run, from the manifest alone.

    The prompt list is the run's SUBSAMPLE (whose length is ``n_train_rows`` when the
    manifest records it, else the requested ``subsample_size``), truncated to `n_prompts`.
    Knowing the exact count without a pool read is what lets the cheap gate demand an EXACT
    match: a document computed at ``--n-prompts 512`` must NOT satisfy a later request for
    1024 on a pool that holds 1024+ prompts.
    """
    available = manifest.get("n_train_rows") or manifest.get("subsample_size")
    return min(n_prompts, available) if available else n_prompts


def _doc_is_current(
    path: Path, unit: RunUnit, output_root: Path, n_prompts: int,
    references: tuple[str, ...], *, main: bool,
) -> bool:
    """Whether ONE stored EAS document (main or refs) answers `references` for this unit."""
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not prompt_source_matches(doc, unit.manifest):
        return False
    if doc.get("n_prompts") != expected_n_prompts(unit.manifest, n_prompts):
        return False
    expected = {
        "schema_version": SCHEMA_VERSION,
        "basis": ACTIVATION_BASIS,
        "eas_mean_convention": EAS_MEAN_CONVENTION,
        "position": unit.position,
        # The prompt set is verified through `prompt_source_matches` above; adopting the
        # stored values here keeps the predicate from demanding a pool read.
        "n_prompts": doc.get("n_prompts"),
        "prompt_sha256": doc.get("prompt_sha256"),
    }
    checkpoints = [
        (rel, unit.endpoint_adapter_id if rel == unit.ckpt_rels[-1] else None)
        for rel in unit.ckpt_rels
    ]
    ref_ids = {
        key: _teacher_adapter_id(str(Path(output_root) / TEACHERS_SUBDIR / key))
        for key in references
    }
    status = eas_main_doc_status if main else eas_doc_status
    plan = status(
        doc, expected_meta=expected, checkpoints=checkpoints, references=ref_ids,
        manifest_ref=unit.manifest_ref, cheap=True,
    )
    return plan.is_current and not plan.stale_refs


def unit_is_cached(
    unit: RunUnit, output_root: Path, n_prompts: int, *, force: bool = False
) -> bool:
    """The sweep's cheap gate: True when the stored document(s) already answer this unit.

    BOTH files count (`pipelines.eas_pipeline`): the own-teacher main document must be
    current, and — only when this pass requests extra references — so must the refs
    document for exactly those. Reads the document JSON and hashes at most the endpoint
    adapter + one adapter per reference teacher (`eas_doc_status(cheap=True)`); intermediate
    checkpoint adapters are keyed on `ckpt_rel` only.
    """
    if force:
        return False
    own = default_reference_key(unit.manifest)
    own = own if own in unit.references else None
    extra = tuple(k for k in unit.references if k != own)
    if own and not _doc_is_current(
        doc_path(unit.run_dir, unit.position), unit, output_root, n_prompts,
        (own,), main=True,
    ):
        return False
    if extra and not _doc_is_current(
        refs_doc_path(unit.run_dir, unit.position), unit, output_root, n_prompts,
        extra, main=False,
    ):
        return False
    return True


def path_prefilter(
    run_rel: str, *, animals: set[str], models: set[str], tags: set[str],
    student_tags: set[str],
) -> bool:
    """Conservative path-only pre-filter, applied BEFORE a run's manifest is read.

    The students layout puts the animal, the base model dir, the number-dataset tag and the
    student param tag each on their own path component, so a filter value that appears in
    NO component cannot match the manifest either. Rejecting on the path alone saves ~85k
    manifest reads on the shared filesystem for a narrow request (the tree held 85,352 run
    dirs on 2026-09-10); `manifest_passes` stays authoritative for what survives.
    """
    parts = set(Path(run_rel).parts)
    return all(
        not wanted or (wanted & parts)
        for wanted in (animals, models, tags, student_tags)
    )


def manifest_passes(
    manifest: dict, *, animals: set[str], models: set[str], tags: set[str],
    student_tags: set[str], sources: set[str],
) -> bool:
    """Whether a run's manifest passes the manifest-only CLI filters (empty set = all)."""
    repo = manifest.get("base_model_repo")
    try:
        base_dir = base_dir_name(repo) if repo else None
    except ValueError:
        base_dir = None
    return (
        (not animals or manifest.get("animal") in animals)
        and (not models or base_dir in models)
        and (not tags or manifest.get("number_dataset_tag") in tags)
        and (not student_tags or manifest.get("student_param_tag") in student_tags)
        and (not sources or manifest.get("source") in sources)
    )


def sort_units(units: list[RunUnit]) -> list[RunUnit]:
    """Group by student base, then prompt set, then run — one base load per group."""
    return sorted(units, key=lambda u: (
        u.base_model_dir or "", tuple(str(x) for x in u.prompt_key), u.run_rel, u.position
    ))


def build_units(
    output_root: Path, *, positions: tuple[str, ...], animals: set[str], models: set[str],
    tags: set[str], student_tags: set[str], sources: set[str], min_checkpoints: int,
    endpoint_only: bool, extra_refs: tuple[str, ...],
) -> tuple[list[RunUnit], list[tuple[str, str]]]:
    """Walk `students/` into sorted units. Returns ``(units, [(run_rel, skip_reason)])``.

    Filtering is two-stage: a conservative `path_prefilter` (no IO beyond the walk), then
    the authoritative `manifest_passes`. Both run BEFORE a unit is built, so a filtered-out
    run costs neither a manifest read nor an adapter hash.
    """
    units: list[RunUnit] = []
    skipped: list[tuple[str, str]] = []
    for run_dir in discover_run_dirs(Path(output_root) / STUDENTS_SUBDIR):
        rel = str(run_dir.relative_to(output_root))
        if not path_prefilter(
            rel, animals=animals, models=models, tags=tags, student_tags=student_tags
        ):
            continue
        try:
            manifest = read_manifest(run_dir)
        except EASSkip as exc:
            skipped.append((rel, str(exc)))
            continue
        if not manifest_passes(
            manifest, animals=animals, models=models, tags=tags,
            student_tags=student_tags, sources=sources,
        ):
            continue
        built = build_unit(
            run_dir, output_root, positions[0], endpoint_only=endpoint_only,
            extra_refs=extra_refs, manifest=manifest,
        )
        if isinstance(built, str):
            skipped.append((rel, built))
            continue
        if len(built.ckpt_rels) < min_checkpoints:
            skipped.append((rel, f"{len(built.ckpt_rels)} checkpoint(s) < --min-checkpoints"))
            continue
        units.append(built)
        # The remaining positions differ only in the pooling; nothing on disk is re-read.
        units += [replace(built, position=p) for p in positions[1:]]
    return sort_units(units), skipped


# ------------------------- CLI -------------------------


def _set(value: str) -> set[str]:
    return {v.strip() for v in value.split(",") if v.strip()}


def main(
    animals: str = typer.Option("", help="Comma list of animals to keep (empty = all)."),
    models: str = typer.Option("", help="Comma list of base model dir names (empty = all)."),
    tags: str = typer.Option("", help="Comma list of number-dataset tags (empty = all)."),
    student_tags: str = typer.Option(
        "", help="Comma list of student param tags (e.g. ep20_lr4e-4_r16a4); empty = all."
    ),
    sources: str = typer.Option(
        "", help="Comma list of student sources ('teacher'/'base'/'rng'); empty = all."
    ),
    positions: str = typer.Option("last", help="Comma list of pooling positions ('last'/'all')."),
    min_checkpoints: int = typer.Option(
        1, help="Skip runs with fewer than this many checkpoint points."
    ),
    reference_teacher: list[str] = typer.Option(
        [], "--reference-teacher",
        help="EXTRA reference teacher ref_key ('{animal}/{model}/{tag}/ftseed{T}'), "
        "repeatable; also the ONLY way to measure the base/rng control runs.",
    ),
    endpoint_only: bool = typer.Option(
        False, help="Score only each run's endpoint adapter."
    ),
    n_prompts: int = typer.Option(1024, help="Number of training prompts to evaluate on."),
    batch_size: int = typer.Option(32, help="Prompts per forward pass."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    limit: int = typer.Option(0, help="Stop after N run() invocations (0 = no limit)."),
    num_shards: int = typer.Option(
        1, help="Number of deterministic partitions of the sorted unit list."
    ),
    shard_index: int = typer.Option(0, help="Zero-based partition to run."),
    dry_run: bool = typer.Option(
        False, help="List READY / CACHED / SKIP(reason) per run without loading a model."
    ),
    force: bool = typer.Option(False, help="Rewrite even where a cached document exists."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Compute EAS for every distilled student run on disk (see module docstring)."""
    load_dotenv()
    configure_loguru()
    position_list = tuple(p.strip() for p in positions.split(",") if p.strip())
    if not position_list or any(p not in ("last", "all") for p in position_list):
        raise typer.BadParameter("positions must be a comma list of 'last'/'all'")
    source_set = _set(sources)
    if any(s not in ("teacher", "base", "rng") for s in source_set):
        raise typer.BadParameter("sources must be a comma list of 'teacher'/'base'/'rng'")

    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    output_root = cfg.paths.output_root
    units, skipped = build_units(
        output_root, positions=position_list, animals=_set(animals), models=_set(models),
        tags=_set(tags), student_tags=_set(student_tags), sources=source_set,
        min_checkpoints=min_checkpoints, endpoint_only=endpoint_only,
        extra_refs=tuple(reference_teacher),
    )
    for rel, reason in skipped:
        logger.warning("SKIP {} — {}", rel, reason)
    n_total = len(units)
    units = shard_cells(units, num_shards=num_shards, shard_index=shard_index)
    logger.info(
        "Discovered {} (run x position) unit(s); {} run(s) skipped. shard {}/{}: {} unit(s).",
        n_total, len(skipped), shard_index, num_shards, len(units),
    )

    session = EASSession(cfg.paths, device=device, batch_size=batch_size)
    n_done = n_cached = n_failed = n_skipped = 0
    failures: list[tuple[str, str]] = []
    try:
        for unit in units:
            if unit_is_cached(unit, output_root, n_prompts, force=force):
                n_cached += 1
                logger.info("CACHED (skip): {}", unit.label)
                continue
            if dry_run:
                logger.info(
                    "READY: {} | source={} base={} refs={} checkpoints={}",
                    unit.label, unit.source, unit.base_model_dir,
                    list(unit.references), len(unit.ckpt_rels),
                )
                n_done += 1
                continue
            if limit and n_done >= limit:
                logger.info("LIMIT {} reached; stopping.", limit)
                break
            logger.info("RUN [{}]: {}", n_done + 1, unit.label)
            try:
                session.run(EASRunConfig(
                    paths=cfg.paths, hub_org=cfg.hub_org, run_dir=unit.run_dir,
                    references=tuple(unit.references), position=unit.position,
                    n_prompts=n_prompts, batch_size=batch_size, device=device,
                    endpoint_only=endpoint_only, force=force,
                ))
                n_done += 1
            except EASSkip as exc:
                n_skipped += 1
                logger.warning("SKIP {} — {}", unit.label, exc)
            except Exception as exc:  # keep the sweep going; the unit retries on resume
                n_failed += 1
                failures.append((unit.label, repr(exc)))
                logger.exception("FAILED: {}", unit.label)
    finally:
        session.release()

    logger.info(
        "EAS sweep complete (shard {}/{}): done={}, cached-skip={}, skipped={}, failed={}, "
        "dry_run={}",
        shard_index, num_shards, n_done, n_cached, n_skipped + len(skipped), n_failed,
        dry_run,
    )
    for label, reason in failures:
        logger.warning("  FAILED {} — {}", label, reason)


if __name__ == "__main__":
    typer.run(main)
