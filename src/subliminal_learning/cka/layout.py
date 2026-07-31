"""Output-path layout + student-manifest resolution for CKA runs.

Three kinds of CKA run land in three different trees:

- Per-student "student vs. its teacher" runs COLOCATE with the student they measure,
  under the student's own run dir: ``{run_dir}/cka/vs_teacher__{condition_tag}``. That
  keeps a student's alignment artifacts next to its checkpoint/manifest, so a student
  dir is self-describing and a per-student glob (``.../ftseed*/cka/vs_teacher__*``)
  enumerates them.
- The base-model suite (the 7 PolyPythias pair matrix) writes to a shared top-level
  tree ``{output_root}/cka/base/{run_tag}``.
- Generic explicit ``--model-a``/``--model-b`` pairs write to ``{output_root}/cka/pairs/{run_tag}``.

(Both shared trees moved out from under the old ``{output_root}/analysis/cka/`` root;
nothing here reads or writes that legacy location.)

The ``resolve_teacher_of`` / ``student_numbers_pool`` helpers read a student's
``train_manifest.json`` to recover the operand and the number pool a "vs teacher" run
needs, so the CLI never has to re-derive teacher coordinates by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from subliminal_learning.cka.model_refs import (
    ModelRef,
    ref_provenance,
    resolve_ref,
    spec_from_base_name,
)
from subliminal_learning.models.registry import model_ref
from subliminal_learning.utils.io import file_sha256

TRAIN_MANIFEST = "train_manifest.json"


def vs_teacher_dir(run_dir: Path, condition_tag: str) -> Path:
    """A student run's "vs its teacher" CKA dir: ``{run_dir}/cka/vs_teacher__{tag}``.

    ``condition_tag`` is the (suffixed) data condition, e.g. "numbers", "pile", or a
    non-canonical variant like "numbers_k200". Colocating under the student run dir
    keeps every student's alignment artifacts identically named and self-contained.
    """
    return run_dir / "cka" / f"vs_teacher__{condition_tag}"


def base_cka_dir(output_root: Path, run_tag: str) -> Path:
    """The base-suite (7-model pair matrix) CKA dir: ``{output_root}/cka/base/{run_tag}``."""
    return output_root / "cka" / "base" / run_tag


def pair_cka_dir(output_root: Path, run_tag: str) -> Path:
    """A generic explicit two-operand CKA dir: ``{output_root}/cka/pairs/{run_tag}``."""
    return output_root / "cka" / "pairs" / run_tag


def _load_student_manifest(student_run_dir: Path) -> dict:
    """The student's ``train_manifest.json`` as a dict; clear error naming the file."""
    manifest = student_run_dir / TRAIN_MANIFEST
    if not manifest.exists():
        raise ValueError(f"no {TRAIN_MANIFEST} under student run dir {student_run_dir}")
    try:
        return json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"unparsable {TRAIN_MANIFEST} at {manifest}: {e}") from e


def resolve_teacher_of(
    student_run_dir: Path, output_root: Path, hub_org: str
) -> ModelRef:
    """The teacher operand for a student's "vs teacher" CKA run, from its manifest.

    - ``source == "teacher"``: rebuild the teacher run dir under
      ``{output_root}/teachers/{animal}/{base basename}/{teacher_param_tag}/ftseed{T}``
      from the manifest coordinates and ``resolve_ref`` it (a finetuned operand). A
      missing teacher run dir / checkpoint raises a clear error naming the derived path.
    - ``source == "base"``: a base-kind ``ModelRef`` for ``numbers_base_model_repo``.
    - ``source == "rng"``: raises — an rng control has no teacher to compare against.
    """
    m = _load_student_manifest(student_run_dir)
    source = m.get("source")
    if source == "rng":
        raise ValueError(
            f"student {student_run_dir} is an rng control (source='rng') — it has no "
            "teacher operand to compare against"
        )
    if source == "base":
        base_repo = m["numbers_base_model_repo"]
        spec = spec_from_base_name(base_repo)
        return ModelRef(
            "base", spec, raw=base_repo, base_repo=model_ref(spec, hub_org)
        )
    if source == "teacher":
        base_name = m["numbers_base_model_repo"].rsplit("/", 1)[-1]
        teacher_run_dir = (
            output_root
            / "teachers"
            / m["animal"]
            / base_name
            / m["teacher_param_tag"]
            / f"ftseed{m['teacher_ftseed']}"
        )
        if not (teacher_run_dir / "checkpoints").is_dir():
            raise ValueError(
                f"teacher run dir for student {student_run_dir} does not exist / has "
                f"no checkpoints/: {teacher_run_dir}"
            )
        return resolve_ref(str(teacher_run_dir), output_root, hub_org)
    raise ValueError(
        f"student {student_run_dir} manifest has unrecognized source {source!r} "
        "(expected 'teacher', 'base', or 'rng')"
    )


def validate_student_batch(
    student_run_dirs: list[Path], output_root: Path, hub_org: str
) -> tuple[ModelRef, list[ModelRef]]:
    """Resolve + cross-validate K students a batched "vs teacher" CKA run measures.

    All K must be students of the SAME teacher (same resolved teacher operand /
    checkpoint) trained on the same number pool + subsample; different STUDENT bases are
    allowed (siblings may sit on different PolyPythia variants — the pipeline's
    layer-count check covers load compatibility). Returns ``(teacher_ref, [student_ref,
    ...])`` in input order. Raises a clear error on a duplicate run dir, a teacher
    mismatch, or a pool / subsample disagreement.
    """
    dirs = [Path(d) for d in student_run_dirs]
    if not dirs:
        raise ValueError("no student run dirs given")
    seen: set[Path] = set()
    for d in dirs:
        key = d.resolve()
        if key in seen:
            raise ValueError(f"duplicate student run dir in the batch: {d}")
        seen.add(key)
    teacher_refs = [resolve_teacher_of(d, output_root, hub_org) for d in dirs]
    prov0 = ref_provenance(teacher_refs[0])
    for d, tref in zip(dirs[1:], teacher_refs[1:]):
        if ref_provenance(tref) != prov0:
            raise ValueError(
                f"student {d} resolves to a different teacher than {dirs[0]} — a batched "
                "run must measure students of ONE teacher"
            )
    manifests = [_load_student_manifest(d) for d in dirs]
    for key in ("number_dataset_dir", "filtered_sha256", "subsample_size", "subsample_seed"):
        values = {json.dumps(m.get(key), sort_keys=True) for m in manifests}
        if len(values) != 1:
            raise ValueError(
                f"students disagree on '{key}' ({sorted(values)}) — a batched run must "
                "measure students trained on the same pool + subsample"
            )
    student_refs = [resolve_ref(str(d), output_root, hub_org) for d in dirs]
    return teacher_refs[0], student_refs


def student_numbers_pool(
    student_run_dir: Path,
) -> tuple[Path, int | None, int]:
    """The number pool + subsample a "vs teacher" numbers run must use, from the manifest.

    Returns ``({number_dataset_dir}/filtered.jsonl, subsample_size, subsample_seed)`` —
    exactly the pool and fixed subsample the student was distilled on, so the CKA
    measurement data matches the student's training set. Raises a clear error if the
    manifest lacks ``number_dataset_dir`` (e.g. an rng control).
    """
    m = _load_student_manifest(student_run_dir)
    number_dataset_dir = m.get("number_dataset_dir")
    if not number_dataset_dir:
        raise ValueError(
            f"student {student_run_dir} manifest has no 'number_dataset_dir' — cannot "
            "locate its number pool (rng controls have no numbers pool)"
        )
    return (
        Path(number_dataset_dir) / "filtered.jsonl",
        m.get("subsample_size"),
        m.get("subsample_seed"),
    )


def verify_student_numbers_pool(student_run_dir: Path, pool_path: Path) -> None:
    """Guard a "vs teacher" numbers run against a drifted / missing pool file.

    Raises a clear error (naming ``pool_path``) if the pool file is missing, and a
    clear error (naming BOTH hashes and the path) if its sha256 no longer matches the
    ``filtered_sha256`` the student's manifest recorded at distillation time — this
    catches a regenerated pool silently changing the measurement data out from under a
    student. An older manifest without ``filtered_sha256`` cannot be checked: log a
    warning and proceed.
    """
    if not pool_path.exists():
        raise ValueError(f"number pool file does not exist: {pool_path}")
    m = _load_student_manifest(student_run_dir)
    expected = m.get("filtered_sha256")
    if not expected:
        logger.warning(
            "student {} manifest has no 'filtered_sha256' — cannot verify the pool "
            "{} matches the student's training data; proceeding unchecked",
            student_run_dir, pool_path,
        )
        return
    actual = file_sha256(pool_path)
    if actual != expected:
        raise ValueError(
            f"number pool {pool_path} has sha256 {actual} but the student's manifest "
            f"recorded filtered_sha256 {expected} — the pool file has changed since "
            "distillation (regenerated?); refusing to measure against different data"
        )
