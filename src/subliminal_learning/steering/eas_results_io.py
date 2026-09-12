"""Load per-student-run EAS documents into one tidy long-format table.

Since the 2026-09-10 migration the EAS pipeline (`pipelines.eas_pipeline`) writes per *student
run* (position in ``last``/``all``), not one document per teacher lineage. There are TWO document
kinds, one row source each: ``{run_dir}/eval/eas_{position}.json``, the MAIN document, scored
against the run's OWN teacher alone (`load_eas_results`), and ``eval/eas_refs_{position}.json``,
holding every EXTRA reference teacher ever requested for that run (`load_eas_reference_results`)
— the only rows a base/rng control run has. Both carry the same schema and row columns. Each document names its student (``student`` block), the reference
teacher vectors it was scored against (``references``, keyed by the teacher run dir relative to
``{output_root}/teachers/``), and one ``points`` record per scored checkpoint of that run. This
module flattens every (run, point, reference) triple into one row.

Identity comes from the document, never from path arithmetic: the animal, base model, number-pool
base, cross-base flag, student HP tag and ftseed are read out of the ``student`` block, and the
``teacher_*`` columns are parsed from the *reference's* tag — so a control run (``source`` ``base``
or ``rng``) scored against reference X carries X's teacher HPs with ``is_own_reference`` False.
``eas_mean`` averages the per-layer slots ``[1:]`` (embedding dropped, every remaining slot a raw
block output, final block included: the ``raw_blocks_v2`` basis). The per-slot L2 norms of both
vectors (``v_student_norm_per_layer`` from the point, ``v_teacher_norm_per_layer`` from the
reference block) ride along, so a cosine can be resolved into aligned and orthogonal parts.

Everything is parsed defensively: malformed documents, unparseable tags, and documents missing
expected keys are skipped with a loguru warning rather than aborting the walk — so this works on a
partially-computed sweep. The 826 retired lineage-level documents under ``{lineage}/eas/`` (all
``blocks_v1``) are left on disk and are NOT read here. The pure parsers (`parse_ft_param_tag`,
`parse_lineage_dir`) stay importable for testing without disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.analysis.frames import load_run_docs
from subliminal_learning.evaluation.preference_results_io import parse_param_tag

# The retired lineage dir levels: teacher-{numbers_base_model} then {teacher_param_tag}_tseed{T}.
_TEACHER_BASE_RE = re.compile(r"^teacher-(?P<base>.+)$")
_TEACHER_TAG_RE = re.compile(
    r"^(?P<tag>ep\d+_lr[^_]+_r\d+a\d+)_tseed(?P<seed>\d+)$"
)

#: Pooling positions an EAS document may be written for (one record file name each).
EAS_POSITIONS = ("last", "all")

# Columns carried through for every (run, point, reference) row, in a stable order.
RESULT_COLUMNS = [
    "run_path", "source", "animal", "model",
    "numbers_base_model", "cross_base",
    "teacher_param_tag", "teacher_epochs", "teacher_lr",
    "teacher_lora_r", "teacher_lora_alpha", "teacher_ftseed",
    "reference", "is_own_reference",
    "dataset_tag", "spliced", "position", "n_prompts", "eas_mean_convention",
    "student_param_tag", "student_epochs", "student_lr",
    "student_lora_r", "student_lora_alpha", "student_ftseed",
    "step", "is_endpoint", "nonfinite",
    "eas_mean", "eas_per_layer",
    "v_student_norm_per_layer", "v_teacher_norm_per_layer", "eas_file",
]


def parse_ft_param_tag(tag: str) -> dict:
    """Parse a teacher HP tag ``ep{N}_lr{X}_r{R}a{A}`` into its axes.

    Thin wrapper over `preference_results_io.parse_param_tag` (the repo's one public tag
    parser) renaming its keys to the ``teacher_*`` columns this loader emits.

    Returns:
        ``{"teacher_epochs", "teacher_lr", "teacher_lora_r", "teacher_lora_alpha"}``.

    Raises:
        ValueError: if `tag` does not match the ``ft_param_tag`` format.
    """
    hp = parse_param_tag(tag)
    return {
        "teacher_epochs": hp["n_epochs"],
        "teacher_lr": hp["learning_rate"],
        "teacher_lora_r": hp["lora_r"],
        "teacher_lora_alpha": hp["lora_alpha"],
    }


def _parse_student_param_tag(tag: str) -> dict:
    """Parse a student HP tag into the ``student_*`` HP columns (same tag grammar)."""
    hp = parse_param_tag(tag)
    return {
        "student_epochs": hp["n_epochs"],
        "student_lr": hp["learning_rate"],
        "student_lora_r": hp["lora_r"],
        "student_lora_alpha": hp["lora_alpha"],
    }


def parse_lineage_dir(teacher_base_dir: str, teacher_tag_dir: str) -> dict:
    """Parse the two RETIRED student lineage dir levels into their fields.

    Kept as a pure parser for the legacy ``{lineage}/eas/`` tree (not read by `load_eas_results`).

    Args:
        teacher_base_dir: the ``teacher-{base}`` level.
        teacher_tag_dir: the ``{tag}_tseed{T}`` level.

    Returns:
        ``{"teacher_param_tag", "teacher_ftseed", "numbers_base_model"}``.

    Raises:
        ValueError: if either level is not a parseable lineage dir name.
    """
    mb = _TEACHER_BASE_RE.match(teacher_base_dir)
    mt = _TEACHER_TAG_RE.match(teacher_tag_dir)
    if not mb or not mt:
        raise ValueError(
            f"Unparseable lineage dirs: {teacher_base_dir!r}/{teacher_tag_dir!r}"
        )
    return {
        "teacher_param_tag": mt["tag"],
        "teacher_ftseed": int(mt["seed"]),
        "numbers_base_model": mb["base"],
    }


#: Filename stem prefixes of the two document kinds (the rest of the stem is the position).
MAIN_STEM_PREFIX = "eas_"
REFS_STEM_PREFIX = "eas_refs_"


def _discover_run_rel_paths(
    output_root: Path, stem_prefix: str = MAIN_STEM_PREFIX
) -> list[str]:
    """Every ``students/`` run dir holding a document of this kind, relative to `output_root`.

    The glob is tightened by the stem: ``eas_*.json`` also matches the refs document, so each
    loader keeps only the names whose remaining stem is an `EAS_POSITIONS` value.
    """
    output_root = Path(output_root)
    students = output_root / "students"
    runs = {
        str(p.parent.parent.relative_to(output_root))
        for p in students.rglob(f"eval/{stem_prefix}*.json")
        if p.stem[len(stem_prefix):] in EAS_POSITIONS
    }
    return sorted(runs)


def _reference_columns(ref_key: str, ref: dict) -> dict | None:
    """Teacher columns for one reference block, or None when its tag does not parse.

    Carries the reference's own ``v_teacher_norm_per_layer`` (13 slots) alongside its HPs.
    """
    tag = ref.get("teacher_param_tag")
    try:
        tag_hp = parse_ft_param_tag(tag)
    except (TypeError, ValueError):
        logger.warning("Skipping reference with unparseable teacher tag {}", ref_key)
        return None
    return {
        "teacher_param_tag": tag,
        **tag_hp,
        "teacher_ftseed": ref.get("teacher_ftseed"),
        "reference": ref_key,
        "v_teacher_norm_per_layer": ref.get("v_teacher_norm_per_layer"),
    }


def load_eas_results(
    output_root: Path,
    run_rel_paths: list[str] | None = None,
    positions: list[str] | None = None,
    refresh: bool = True,
) -> pd.DataFrame:
    """Return one row per (student run, checkpoint, reference teacher) EAS measurement.

    Reads the MAIN document ``{run_dir}/eval/eas_{position}.json`` — the run's OWN teacher
    only — for every requested run, so this works on a partially-computed sweep. Malformed documents, unparseable HP tags, and documents missing
    expected keys are skipped with a warning.

    Args:
        output_root: The ``paths.output_root`` whose ``students/`` subtree holds the runs.
        run_rel_paths: Student run dirs relative to `output_root`; None discovers them by
            ``students/**/eval/eas_{position}.json`` (the refs documents are NOT read here;
            see `load_eas_reference_results`).
        positions: If given, keep only these pooling positions (``last``/``all``); None loads all.
        refresh: Passed to the record cache (False serves stored rows only, no filesystem sweep).

    Returns:
        A tidy DataFrame with `RESULT_COLUMNS`; empty (typed) if nothing matched.
    """
    return _load_documents(
        output_root, run_rel_paths, positions, refresh, stem_prefix=MAIN_STEM_PREFIX
    )


def load_eas_reference_results(
    output_root: Path,
    run_rel_paths: list[str] | None = None,
    positions: list[str] | None = None,
    refresh: bool = True,
) -> pd.DataFrame:
    """Return one row per (student run, checkpoint, EXTRA reference teacher) measurement.

    The mirror of `load_eas_results` over ``{run_dir}/eval/eas_refs_{position}.json``: the
    baseline / specificity references a pass asked for with ``--reference-teacher``, and the
    only rows a base/rng CONTROL run has (it has no own teacher, hence no main document).
    ``is_own_reference`` is False on every row.

    Args:
        output_root: The ``paths.output_root`` whose ``students/`` subtree holds the runs.
        run_rel_paths: Student run dirs relative to `output_root`; None discovers them by
            ``students/**/eval/eas_refs_{position}.json``.
        positions: If given, keep only these pooling positions (``last``/``all``); None loads all.
        refresh: Passed to the record cache (False serves stored rows only, no filesystem sweep).

    Returns:
        A tidy DataFrame with `RESULT_COLUMNS`; empty (typed) if nothing matched.
    """
    return _load_documents(
        output_root, run_rel_paths, positions, refresh, stem_prefix=REFS_STEM_PREFIX
    )


def _load_documents(
    output_root: Path,
    run_rel_paths: list[str] | None,
    positions: list[str] | None,
    refresh: bool,
    *,
    stem_prefix: str,
) -> pd.DataFrame:
    """Load one document KIND (`stem_prefix`) into the tidy frame both loaders return."""
    output_root = Path(output_root)
    keep = list(positions) if positions is not None else list(EAS_POSITIONS)
    unexpected = [p for p in keep if p not in EAS_POSITIONS]
    if unexpected:
        logger.warning("Ignoring unexpected EAS position(s) {}", sorted(unexpected))
        keep = [p for p in keep if p in EAS_POSITIONS]
    if run_rel_paths is None:
        run_rel_paths = _discover_run_rel_paths(output_root, stem_prefix)
    else:
        run_rel_paths = [str(r) for r in run_rel_paths]

    rows: list[dict] = []
    for position in keep:
        if not run_rel_paths:
            continue
        name = f"{stem_prefix}{position}"
        docs = load_run_docs(
            output_root, run_rel_paths, f"eval/{name}.json", name, refresh=refresh,
        )
        for run_rel in run_rel_paths:
            doc = docs.get(run_rel)
            if doc is None:
                continue
            rows.extend(_rows_from_doc(output_root, run_rel, position, doc, name))

    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    conventions = set(frame["eas_mean_convention"].dropna().unique())
    if len(conventions) > 1:
        logger.warning(
            "Loaded EAS frame mixes {} distinct eas_mean conventions {}: aggregating across "
            "them compares incommensurable means.",
            len(conventions), sorted(conventions),
        )
    return frame


def _rows_from_doc(
    output_root: Path, run_rel: str, position: str, doc: dict, name: str
) -> list[dict]:
    """Flatten one EAS document into its (point × reference) rows; [] when it is malformed."""
    eas_file = str(output_root / run_rel / "eval" / f"{name}.json")
    student = doc.get("student")
    references = doc.get("references")
    points = doc.get("points")
    if not isinstance(student, dict) or not isinstance(references, dict) or (
        not isinstance(points, list)
    ):
        logger.warning("Skipping EAS doc missing student/references/points {}", eas_file)
        return []
    try:
        student_hp = _parse_student_param_tag(student["student_param_tag"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping EAS doc with unparseable student tag {}", eas_file)
        return []
    ref_cols: dict[str, dict] = {}
    for ref_key, ref in references.items():
        cols = _reference_columns(ref_key, ref) if isinstance(ref, dict) else None
        if cols is not None:
            ref_cols[ref_key] = cols
    if not ref_cols:
        logger.warning("Skipping EAS doc with no usable reference {}", eas_file)
        return []

    prompt_source = doc.get("prompt_source") or {}
    own_reference = student.get("own_reference")
    common = {
        "run_path": run_rel,
        "source": student.get("source"),
        "animal": student.get("animal"),
        "model": student.get("base_model_dir"),
        "numbers_base_model": student.get("numbers_base_model_dir"),
        "cross_base": bool(student.get("cross_base")),
        "dataset_tag": student.get("number_dataset_tag"),
        "spliced": bool(prompt_source.get("splice_examples")),
        "position": doc.get("position", position),
        "n_prompts": doc.get("n_prompts"),
        "eas_mean_convention": doc.get("eas_mean_convention"),
        "student_param_tag": student.get("student_param_tag"),
        **student_hp,
        "student_ftseed": student.get("student_ftseed"),
        "eas_file": eas_file,
    }

    rows: list[dict] = []
    for point in points:
        if not isinstance(point, dict) or "step" not in point:
            logger.warning("Skipping malformed EAS point in {}", eas_file)
            continue
        eas = point.get("eas") or {}
        for ref_key, cols in ref_cols.items():
            scored = eas.get(ref_key) if isinstance(eas, dict) else None
            rows.append({
                **common,
                **cols,
                "is_own_reference": ref_key == own_reference,
                "step": point["step"],
                "is_endpoint": bool(point.get("is_endpoint")),
                "nonfinite": bool(point.get("nonfinite")),
                "eas_mean": (scored or {}).get("mean"),
                "eas_per_layer": (scored or {}).get("per_layer"),
                "v_student_norm_per_layer": point.get("v_student_norm_per_layer"),
            })
    return rows


def endpoint_only(df: pd.DataFrame) -> pd.DataFrame:
    """The endpoint (final-checkpoint) rows of an EAS frame — one row per (run, reference)."""
    if df.empty:
        return df
    return df[df["is_endpoint"].astype(bool)]


def own_reference_only(df: pd.DataFrame) -> pd.DataFrame:
    """The rows scored against the run's OWN reference teacher.

    Trivial on a `load_eas_results` frame (a main document holds the own teacher alone) and
    empty on a `load_eas_reference_results` one; kept for compatibility with callers that
    filtered mixed frames written by the retired one-file design.
    """
    if df.empty:
        return df
    return df[df["is_own_reference"].astype(bool)]
