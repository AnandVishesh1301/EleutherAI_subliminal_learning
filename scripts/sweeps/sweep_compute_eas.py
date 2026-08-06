"""CLI: compute EAS for every distilled STUDENT lineage on disk (walks, not a grid).

The single-config CLI (scripts/compute_eas.py) reaches one lineage named by its flags. This
sweep instead WALKS the `students/` tree and invokes `pipelines.eas_pipeline.run` once per
(lineage, pooling position), deriving every EASConfig field from the on-disk path segments +
the shared student config (same defaults as scripts/compute_eas.py). It is disk-driven so it
scores exactly the lineages that have distilled students, however the sweep produced them.

Discovery: for each `students/{animal}/{model}/teacher-{tbase}/{ttag}_tseed{T}/{dataset_tag}/`
dir holding at least one `ep*/ftseed*/train_manifest.json`, it derives:
  - animal / model / teacher_param_tag / teacher_ftseed / number_dataset_tag from the segments;
  - the base spec from the model dir name (inverse of templates.run_layout.model_dir_name);
  - splice_examples from a student manifest (they must agree; the pipeline re-validates);
  - the student epochs from the `ep{N}_...` param dirs present whose lr/r/alpha match the
    student config (dirs with non-default lr/r/alpha are warned + ignored), and the student
    seeds from the `ftseed{S}` dirs present.
Cross-base lineages (teacher-base != student model) are skipped by default (the pipeline's
same-base v_teacher doesn't apply); pass `--include-cross-base` to parse them instead — the
`teacher-{tbase}` dir is inverted to the teacher/numbers base A (via `spec_from_model_dir`) and
threaded into the EASConfig as `numbers_spec`, so v_teacher reads on A and v_student on the
student base B. Model dir names that don't resolve to a known 160m spec are always skipped.

Resumability. BEFORE loading any model a cheap gate skips a (lineage, position) whose target
`eas_{position}.json` already exists with `meta.eas_mean_convention == "blocks_v1"`, the
requested `meta.n_prompts`/`meta.position`, an entry for every (source, epoch, seed) student that
exists on disk, and — via one cheap file hash per lineage — an unchanged teacher adapter
(`meta.teacher_adapter_id` vs the on-disk teacher checkpoint) and number pool
(`meta.filtered_sha256` vs the on-disk `filtered.jsonl`), so an in-place teacher retrain or a
refreshed pool invalidates the cache. It does NOT re-hash each STUDENT adapter, so it can still
miss an in-place-retrained student whose adapter changed but whose entry key is unchanged; that
is acceptable for the sweep (use `--force`, or the single-lineage compute_eas CLI, which
re-checks student adapter ids in full, to recompute a suspected-stale lineage). A unit with no
trained students in scope is reported as "nothing to do" rather than cached, and a unit whose
stored `meta.n_prompts` differs SOLELY because of a short pool is warned about by name (it would
otherwise cache-miss forever with no explanation). `--force` bypasses the whole gate.

Sharding. `--num-shards N --shard-index I` strides the deterministically sorted unit list (shard
I takes units at indices I, I+N, I+2N, ...) so N launches split the work disjointly and to within
one unit. As with the sibling sweeps, the assignment is reproducible only while the on-disk lineage
set + filters are unchanged; if the set shifts between launches the cache gate (per shard) still
keeps completed work from repeating.

Robustness. Each run() is wrapped in try/except; a lineage that raises (e.g. `filtered.jsonl`
sha drift from an in-place number-pool overwrite, or a missing teacher checkpoint) is logged and
counted, and the sweep continues. Iteration order is (animal, model, dataset_tag, teacher tag).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.eas_pipeline import EASConfig, run
from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from subliminal_learning.evaluation.preference_results_io import parse_param_tag
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.steering.eas_results_io import parse_lineage_dir
from subliminal_learning.students.config import load_student_training
from subliminal_learning.students.layout import STUDENTS_SUBDIR, student_run_dir
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import original_160m_spec, parse_base_models
from subliminal_learning.teachers.run_layout import (
    TEACHERS_SUBDIR,
    model_dir_name,
    model_dir_parts_from_name,
)
from subliminal_learning.teachers.sweep_grid import shard_cells
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.utils.logging_setup import configure_loguru

# The v_teacher.pt convention the pipeline stamps into every current EAS doc; a doc lacking it
# is stale-schema and must be recomputed (see steering.eas.eas_mean).
_CONVENTION = "blocks_v1"


# ------------------------- pure discovery/gate logic (GPU-free, unit-tested) -------------------------


def spec_from_model_dir(name: str, hub_org: str) -> PythiaModelSpec:
    """Invert `templates.run_layout.model_dir_name` back to a 160m spec.

    `pythia-160m` -> the original full-seed0 spec; `pythia-160m-{data|weight}-seed{N}` -> that
    variant spec (via `parse_base_models`). A trailing `@{revision}` pins a pretraining
    checkpoint (e.g. `pythia-160m@step1000`, the v1 sweep's `base_model: original` ckpt runs, or
    `pythia-160m-data-seed1@step3000`); the variant form round-trips through `parse_base_models`,
    the bare-original form is handled here. Round-trips through `model_dir_name` to reject any
    name that would not have produced it (unknown sizes, empty/garbled revisions, ...).

    Raises:
        ValueError: if `name` does not resolve to exactly one known 160m spec.
    """
    if name == "pythia-160m" or name.startswith("pythia-160m@"):
        # Bare original (full seed 0), optionally revision-pinned as `pythia-160m@{rev}`.
        revision = name.split("@", 1)[1] if "@" in name else None
        if revision == "":
            raise ValueError(f"empty revision in model dir {name!r}")
        spec = original_160m_spec(revision=revision)
    elif name.startswith("pythia-160m-"):
        specs = parse_base_models(name[len("pythia-160m-"):])
        if len(specs) != 1:
            raise ValueError(f"model dir {name!r} did not resolve to one spec")
        spec = specs[0]
    else:
        raise ValueError(f"unknown model dir {name!r}")
    if model_dir_name(spec, hub_org) != name:
        raise ValueError(f"model dir {name!r} does not round-trip to a known spec")
    return spec


def valid_student_epochs(
    param_tag_names: list[str], *, lr: float, r: int, alpha: int
) -> tuple[list[int], list[str]]:
    """Split `ep{N}_...` student param-dir names into default-HP epochs vs ignored tags.

    A dir is a valid student cell iff its lr/rank/alpha equal the student config's; only those
    contribute an epoch. Dirs with a non-default lr/r/alpha (or an unparseable name) are
    returned as ``ignored`` so the caller can warn.

    Returns:
        ``(sorted_unique_epochs, ignored_tag_names)``.
    """
    epochs: set[int] = set()
    ignored: list[str] = []
    for tag in param_tag_names:
        try:
            hp = parse_param_tag(tag)
        except ValueError:
            ignored.append(tag)
            continue
        if hp["learning_rate"] == lr and hp["lora_r"] == r and hp["lora_alpha"] == alpha:
            epochs.add(hp["n_epochs"])
        else:
            ignored.append(tag)
    return sorted(epochs), ignored


def eas_doc_cached(
    doc: dict, *, position: str, n_prompts: int,
    required_work_items: set[tuple[str, int, int]],
    teacher_adapter_id: str | None = None,
    filtered_sha256: str | None = None,
    label: str = "EAS doc",
) -> bool:
    """Whether a stored EAS doc already satisfies this (lineage, position) request.

    Cheap gate on the JSON + at most one teacher-adapter hash + one pool hash (no per-student
    adapter re-hash). Reasons a doc is NOT cached:

    - an EMPTY `required_work_items` (no trained students in scope): "nothing to do", not "done";
    - the meta lacks the current convention or the requested position;
    - `teacher_adapter_id` is given and `meta.teacher_adapter_id` differs (an in-place teacher
      retrain);
    - `filtered_sha256` is given and `meta.filtered_sha256` differs (a refreshed number pool);
    - the doc is missing an entry for some `(source, n_epochs, student_ftseed)` in the request;
    - the stored `meta.n_prompts` differs from the request — checked LAST so a mismatch that is
      the SOLE remaining cause (a short-pool lineage that stores its actual prompt count and
      would otherwise cache-miss forever) is warned about by name.
    """
    if not required_work_items:
        return False
    meta = doc.get("meta") or {}
    if meta.get("eas_mean_convention") != _CONVENTION:
        return False
    if meta.get("position") != position:
        return False
    if teacher_adapter_id is not None and meta.get("teacher_adapter_id") != teacher_adapter_id:
        return False
    if filtered_sha256 is not None and meta.get("filtered_sha256") != filtered_sha256:
        return False
    have = {
        (e["source"], e["n_epochs"], e["student_ftseed"]) for e in doc.get("entries", [])
    }
    if not required_work_items <= have:
        return False
    if meta.get("n_prompts") != n_prompts:
        logger.warning(
            "{}: stored meta.n_prompts={} != requested {} (a short-pool lineage stores its "
            "actual prompt count, so it will keep cache-missing on this axis).",
            label, meta.get("n_prompts"), n_prompts,
        )
        return False
    return True


# ------------------------- disk discovery (builds one unit per lineage x position) -------------------------


@dataclass(frozen=True)
class _Unit:
    """One (lineage, position) work unit the sweep would invoke run() for."""

    animal: str
    model: str
    spec: PythiaModelSpec
    teacher_param_tag: str
    teacher_ftseed: int
    dataset_tag: str
    splice_examples: bool
    position: str
    epochs: tuple[int, ...]
    student_seeds: tuple[int, ...]
    required_work_items: frozenset[tuple[str, int, int]]
    out_path: Path
    # Content hash of the on-disk teacher adapter (one per lineage) and of the lineage's
    # `filtered.jsonl` pool, resolved during discovery so the cache gate can detect an in-place
    # teacher retrain / a refreshed pool without loading a model. Either may be None if the file
    # is absent (then that check is skipped).
    teacher_adapter_id: str | None = None
    filtered_sha256: str | None = None
    # The teacher/numbers base A spec, set only for a CROSS-base lineage (None same-base).
    numbers_spec: PythiaModelSpec | None = None
    # The teacher-base dir name — always the numbers base (repeats `model` same-base).
    numbers_base_model: str | None = None

    @property
    def label(self) -> str:
        return (
            f"{self.animal}/{self.model}/teacher-{self.numbers_base_model}/"
            f"{self.teacher_param_tag}_tseed{self.teacher_ftseed}/{self.dataset_tag} "
            f"[{self.position}]"
        )


def _student_trained(
    output_root: Path, *, source: str, spec: PythiaModelSpec, hub_org: str,
    train, lora, epoch: int, seed: int, animal: str, teacher_param_tag: str,
    teacher_ftseed: int, number_dataset_tag: str, numbers_base_model: str | None = None,
) -> bool:
    """Whether a trained student (checkpoints + manifest) exists for one (source, epoch, seed)."""
    is_teacher = source == "teacher"
    rd = student_run_dir(
        output_root,
        source=source,
        spec=spec,
        hub_org=hub_org,
        train_cfg=replace(train, n_epochs=epoch, seed=seed),
        lora_cfg=lora,
        animal=animal if is_teacher else None,
        teacher_param_tag=teacher_param_tag if is_teacher else None,
        teacher_seed=teacher_ftseed if is_teacher else None,
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=numbers_base_model,
    )
    return (rd / "checkpoints").is_dir() and (rd / "train_manifest.json").is_file()


def _discover(
    cfg, *, sources: tuple[str, ...], positions: tuple[str, ...],
    animals: set[str], models: set[str], tags: set[str], n_prompts: int,
    include_cross_base: bool = False,
) -> tuple[list[_Unit], int]:
    """Walk `students/` into sorted work units. Returns ``(units, n_skipped_unsupported)``.

    With `include_cross_base`, cross-base lineages (teacher-base != student model) are parsed
    instead of skipped: the teacher-base dir is inverted to the teacher/numbers base A
    (`spec_from_model_dir`) and set on the unit's `numbers_spec`; a teacher-base that doesn't
    resolve to a known 160m spec is skipped-unsupported.
    """
    output_root = cfg.paths.output_root
    students = output_root / STUDENTS_SUBDIR
    units: list[_Unit] = []
    n_unsupported = 0
    # Two disjoint depths: the plain `{animal}/{model}/...` layout and the revision-pinned
    # `{animal}/ckpt/{model}/...` layout (gathered `ckpt/` model level). The plain glob (5
    # components) never matches the 6-component ckpt path, so the two are non-overlapping.
    ds_dirs = sorted(students.glob("*/*/teacher-*/*_tseed*/*"))
    ds_dirs += sorted(students.glob("*/ckpt/*/teacher-*/*_tseed*/*"))
    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        rel = ds_dir.relative_to(students).parts
        # Drop the gathered `ckpt/` model level so the indices match the plain layout; the
        # model name keeps its `@rev` suffix (no bare `ckpt` leaks into a field).
        if len(rel) > 1 and rel[1] == "ckpt":
            rel = rel[:1] + rel[2:]
        animal, model, teacher_base_dir, teacher_tag_dir, dataset_tag = rel
        if animals and animal not in animals:
            continue
        if models and model not in models:
            continue
        if tags and dataset_tag not in tags:
            continue
        if not any(ds_dir.glob("ep*/ftseed*/train_manifest.json")):
            continue
        try:
            lineage = parse_lineage_dir(teacher_base_dir, teacher_tag_dir)
        except ValueError:
            logger.warning("Skipping unparseable lineage dir {}", ds_dir.parent)
            n_unsupported += 1
            continue
        # `numbers_base_model` is always the teacher-base dir name; a lineage is cross-base
        # only when it differs from the student model.
        numbers_spec = None
        numbers_base_model = lineage["numbers_base_model"]
        if numbers_base_model != model:
            if not include_cross_base:
                logger.info("Skipping cross-base lineage {}", ds_dir.parent)
                n_unsupported += 1
                continue
            try:
                numbers_spec = spec_from_model_dir(numbers_base_model, cfg.hub_org)
            except ValueError:
                logger.warning(
                    "Skipping cross-base lineage {} with unknown teacher-base model {}",
                    ds_dir.parent, numbers_base_model,
                )
                n_unsupported += 1
                continue
        try:
            spec = spec_from_model_dir(model, cfg.hub_org)
        except ValueError:
            logger.warning("Skipping unknown model dir {} under {}", model, ds_dir.parent.parent)
            n_unsupported += 1
            continue

        param_tags = [p.name for p in ds_dir.glob("ep*") if p.is_dir()]
        epochs, ignored = valid_student_epochs(
            param_tags, lr=cfg.train.learning_rate, r=cfg.lora.r, alpha=cfg.lora.lora_alpha
        )
        if ignored:
            logger.warning(
                "{}: ignoring {} student param dir(s) with non-default lr/r/alpha: {}",
                ds_dir, len(ignored), sorted(ignored),
            )
        if not epochs:
            logger.warning("{}: no default-HP student epochs; skipping.", ds_dir)
            continue

        # splice_examples + number_dataset_dir from a teacher-source manifest (all agree; the
        # pipeline re-validates). number_dataset_dir lets the cache gate hash the pool below.
        splice = False
        number_dataset_dir: Path | None = None
        manifests = sorted(ds_dir.glob("ep*/ftseed*/train_manifest.json"))
        try:
            m0 = json.loads(manifests[0].read_text())
            splice = bool(m0.get("splice_examples"))
            ndd = m0.get("number_dataset_dir")
            number_dataset_dir = Path(ndd) if ndd else None
        except (OSError, json.JSONDecodeError, IndexError):
            logger.warning("{}: could not read a student manifest; assuming unspliced.", ds_dir)

        # One cheap hash per lineage for the cache gate (no model load): the teacher adapter
        # identity (detects an in-place retrain) and the pool's filtered.jsonl (detects a
        # refresh). model_dir_name(numbers_spec) == numbers_base_model when cross, else `model`.
        teacher_ckpt = (
            output_root / TEACHERS_SUBDIR / animal
            / Path(*model_dir_parts_from_name(numbers_base_model or model))
            / lineage["teacher_param_tag"] / f"ftseed{lineage['teacher_ftseed']}" / "checkpoints"
        )
        teacher_adapter_id = adapter_identity(teacher_ckpt)
        filtered_sha256 = None
        if number_dataset_dir is not None:
            filtered_path = number_dataset_dir / "filtered.jsonl"
            if filtered_path.is_file():
                filtered_sha256 = file_sha256(filtered_path)

        seeds: set[int] = set()
        for p in ds_dir.glob("ep*/ftseed*"):
            if p.is_dir() and p.name.startswith("ftseed"):
                try:
                    seeds.add(int(p.name[len("ftseed"):]))
                except ValueError:
                    continue
        student_seeds = tuple(sorted(seeds))
        if not student_seeds:
            continue

        for position in positions:
            required = {
                (source, e, s)
                for source in sources
                for e in epochs
                for s in student_seeds
                if _student_trained(
                    output_root, source=source, spec=spec, hub_org=cfg.hub_org,
                    train=cfg.train, lora=cfg.lora, epoch=e, seed=s, animal=animal,
                    teacher_param_tag=lineage["teacher_param_tag"],
                    teacher_ftseed=lineage["teacher_ftseed"], number_dataset_tag=dataset_tag,
                    numbers_base_model=numbers_base_model,
                )
            }
            units.append(_Unit(
                animal=animal,
                model=model,
                spec=spec,
                teacher_param_tag=lineage["teacher_param_tag"],
                teacher_ftseed=lineage["teacher_ftseed"],
                dataset_tag=dataset_tag,
                splice_examples=splice,
                position=position,
                epochs=tuple(epochs),
                student_seeds=student_seeds,
                required_work_items=frozenset(required),
                out_path=ds_dir / "eas" / f"eas_{position}.json",
                teacher_adapter_id=teacher_adapter_id,
                filtered_sha256=filtered_sha256,
                numbers_spec=numbers_spec,
                numbers_base_model=numbers_base_model,
            ))

    units.sort(key=lambda u: (
        u.animal, u.model, u.dataset_tag, u.teacher_param_tag, u.teacher_ftseed,
        u.numbers_base_model or "", u.position,
    ))
    return units, n_unsupported


def _cached(unit: _Unit, n_prompts: int) -> bool:
    """Cheap on-disk gate for one unit (reads only the JSON)."""
    if not unit.out_path.is_file():
        return False
    try:
        doc = json.loads(unit.out_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return eas_doc_cached(
        doc, position=unit.position, n_prompts=n_prompts,
        required_work_items=set(unit.required_work_items),
        teacher_adapter_id=unit.teacher_adapter_id,
        filtered_sha256=unit.filtered_sha256,
        label=unit.label,
    )


def main(
    animals: str = typer.Option("", help="Comma list of animals to keep (empty = all)."),
    models: str = typer.Option("", help="Comma list of model dir names to keep (empty = all)."),
    tags: str = typer.Option("", help="Comma list of number-dataset tags to keep (empty = all)."),
    positions: str = typer.Option("last", help="Comma list of pooling positions ('last'/'all')."),
    sources: str = typer.Option(
        "teacher,base,rng", help="Comma list of student sources to score."
    ),
    n_prompts: int = typer.Option(1024, help="Number of training prompts to evaluate on."),
    batch_size: int = typer.Option(32, help="Prompts per forward pass."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    limit: int = typer.Option(
        0, help="Stop after N (lineage x position) run() invocations (0 = no limit)."
    ),
    num_shards: int = typer.Option(
        1, help="Number of deterministic partitions of the sorted unit list."
    ),
    shard_index: int = typer.Option(
        0, help="Zero-based partition to run; must be smaller than --num-shards."
    ),
    dry_run: bool = typer.Option(
        False, help="List what would run (+ skip reasons) without loading any model."
    ),
    include_cross_base: bool = typer.Option(
        False,
        help="Parse '_nbase-' cross-base lineages (student base B distilled from a teacher on "
        "base A) instead of skipping them; v_teacher reads on A, v_student on B.",
    ),
    force: bool = typer.Option(False, help="Recompute even where a cached doc exists."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    student_training_config: Path = Path("configs/training/lora_student_distill.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Compute EAS for every distilled student lineage on disk (see module docstring)."""
    load_dotenv()
    configure_loguru()

    position_list = tuple(p.strip() for p in positions.split(",") if p.strip())
    if not position_list or any(p not in ("last", "all") for p in position_list):
        raise typer.BadParameter("positions must be a comma list of 'last'/'all'")
    source_list = tuple(s.strip() for s in sources.split(",") if s.strip())
    if not source_list or any(s not in ("teacher", "base", "rng") for s in source_list):
        raise typer.BadParameter("sources must be a comma list of 'teacher'/'base'/'rng'")

    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    student_lora, student_train = load_student_training(student_training_config)
    # Fold the student config onto the loaded template config so discovery + EASConfig use the
    # student LoRA/train (the teacher training YAML only supplies paths/hub_org/spec here).
    cfg = replace(cfg, lora=student_lora, train=student_train)

    animal_set = {a.strip() for a in animals.split(",") if a.strip()}
    model_set = {m.strip() for m in models.split(",") if m.strip()}
    tag_set = {t.strip() for t in tags.split(",") if t.strip()}

    units, n_unsupported = _discover(
        cfg, sources=source_list, positions=position_list,
        animals=animal_set, models=model_set, tags=tag_set, n_prompts=n_prompts,
        include_cross_base=include_cross_base,
    )
    n_total = len(units)
    # Strided partition of the sorted unit list (before the cache gate, so re-runs still skip
    # completed work per shard). shard_cells validates num_shards >= 1 and 0 <= shard_index < N.
    units = shard_cells(units, num_shards=num_shards, shard_index=shard_index)
    logger.info(
        "Discovered {} (lineage x position) unit(s); {} skipped (unsupported).",
        n_total, n_unsupported,
    )
    logger.info(
        "shard {}/{}: {} of {} units (stable while the on-disk lineage set + filters are unchanged)",
        shard_index, num_shards, len(units), n_total,
    )

    n_done = n_cached = n_failed = n_empty = 0
    failures: list[tuple[str, str]] = []
    n_invoked = 0
    for unit in units:
        if not unit.required_work_items:
            n_empty += 1
            logger.info("NOTHING TO DO (no trained students in scope): {}", unit.label)
            continue
        if not force and _cached(unit, n_prompts):
            n_cached += 1
            logger.info("CACHED (skip): {}", unit.label)
            continue
        if limit and n_invoked >= limit and not dry_run:
            logger.info("LIMIT {} reached; stopping.", limit)
            break
        if dry_run:
            logger.info(
                "READY: {} | epochs={} seeds={} required_entries={}",
                unit.label, list(unit.epochs), list(unit.student_seeds),
                len(unit.required_work_items),
            )
            n_invoked += 1
            continue
        eas_cfg = EASConfig(
            paths=cfg.paths,
            hub_org=cfg.hub_org,
            spec=unit.spec,
            animal=unit.animal,
            numbers_spec=unit.numbers_spec,
            teacher_param_tag=unit.teacher_param_tag,
            teacher_ftseed=unit.teacher_ftseed,
            number_dataset_tag=unit.dataset_tag,
            lora=student_lora,
            train=student_train,
            sources=source_list,
            epochs=unit.epochs,
            student_seeds=unit.student_seeds,
            splice_examples=unit.splice_examples,
            position=unit.position,
            n_prompts=n_prompts,
            batch_size=batch_size,
            device=device,
            force=force,
            skip_mismatched=True,
        )
        n_invoked += 1
        logger.info("RUN [{}]: {}", n_invoked, unit.label)
        try:
            run(eas_cfg)
            n_done += 1
        except Exception as exc:  # keep the sweep going; the failed unit retries on resume
            n_failed += 1
            failures.append((unit.label, repr(exc)))
            logger.exception("FAILED: {}", unit.label)

    logger.info(
        "EAS sweep complete (shard {}/{}): completed={}, cached-skip={}, nothing-to-do={}, "
        "failed={}, skipped-unsupported={}, dry_run={}",
        shard_index, num_shards, n_done, n_cached, n_empty, n_failed, n_unsupported, dry_run,
    )
    for label, reason in failures:
        logger.warning("  FAILED {} — {}", label, reason)


if __name__ == "__main__":
    typer.run(main)
