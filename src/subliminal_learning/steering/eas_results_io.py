"""Load swept EAS curve documents into one tidy long-format table.

The EAS pipeline (`pipelines.eas_pipeline`) writes one ``eas_{position}.json`` per
(animal, base model, teacher lineage, number-dataset tag, pooling position) under
``{output_root}/students/{animal}/{model}/teacher-{teacher_base}/{teacher_tag}_tseed{T}/
{dataset_tag}/eas/``. Each document holds a ``meta`` block, the teacher vector norms, and one
``entries`` record per scored student checkpoint (source × epoch × ftseed). This module walks
that tree and flattens every entry into one row, parsing the two lineage dir levels
(``teacher-{base}`` + ``{tag}_tseed{T}``) and the teacher HP tag (``ep{N}_lr{X}_r{R}a{A}``) back
into their axes so results can be sliced, pivoted, and plotted.

Everything is parsed defensively: malformed documents, lineage dir names that don't parse, and
documents missing expected keys are skipped with a loguru warning rather than aborting the walk
— so this works on a partially-completed sweep. Cross-base lineages (student base B distilled
from a teacher on base A) ARE loaded: ``numbers_base_model`` carries the teacher-base dir name
(base A; for a same-base lineage it repeats the student ``model``) and the ``cross_base`` column
flags the rows where it differs from ``model``. The pure parsers (`parse_ft_param_tag`,
`parse_lineage_dir`) are importable for testing without disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.evaluation.preference_results_io import parse_param_tag

# The two students/ lineage dir levels: teacher-{numbers_base_model} then
# {teacher_param_tag}_tseed{T}. The teacher tag is the same ep{N}_lr{X}_r{R}a{A} shape
# parse_param_tag consumes.
_TEACHER_BASE_RE = re.compile(r"^teacher-(?P<base>.+)$")
_TEACHER_TAG_RE = re.compile(
    r"^(?P<tag>ep\d+_lr[^_]+_r\d+a\d+)_tseed(?P<seed>\d+)$"
)

# Columns carried through for every entry row, in a stable order.
RESULT_COLUMNS = [
    "animal", "model", "numbers_base_model", "cross_base",
    "teacher_param_tag", "teacher_epochs", "teacher_lr",
    "teacher_lora_r", "teacher_lora_alpha", "teacher_ftseed",
    "dataset_tag", "spliced", "position", "n_prompts", "eas_mean_convention",
    "source", "student_epochs", "student_ftseed", "student_param_tag",
    "eas_mean", "eas_per_layer", "eas_file",
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


def parse_lineage_dir(teacher_base_dir: str, teacher_tag_dir: str) -> dict:
    """Parse the two student lineage dir levels into their fields.

    Args:
        teacher_base_dir: the ``teacher-{base}`` level (always present).
        teacher_tag_dir: the ``{tag}_tseed{T}`` level.

    Returns:
        ``{"teacher_param_tag", "teacher_ftseed", "numbers_base_model"}``. ``numbers_base_model``
        is ALWAYS a string (the teacher-base dir name); it repeats the student model for a
        same-base lineage and names the other base for a cross-base one.

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


def load_eas_results(
    output_root: Path, positions: list[str] | None = None
) -> pd.DataFrame:
    """Walk ``{output_root}/students`` and return one row per stored EAS entry.

    Reads every ``.../teacher-*/*_tseed*/{dataset_tag}/eas/eas_*.json`` that exists, so this
    works on a partially-computed sweep. Malformed documents, unparseable lineage/teacher tags,
    and documents missing expected keys are skipped with a warning.

    Args:
        output_root: The ``paths.output_root`` whose ``students/`` subtree holds lineages.
        positions: If given, keep only these pooling positions (``last``/``all``); None loads
            all.

    Returns:
        A tidy DataFrame with `RESULT_COLUMNS`; empty (typed) if nothing matched.
    """
    keep = set(positions) if positions is not None else None
    students = Path(output_root) / "students"
    rows: list[dict] = []
    for eas_file in sorted(students.glob("*/*/teacher-*/*_tseed*/*/eas/eas_*.json")):
        position = eas_file.stem[len("eas_"):]
        if position not in ("last", "all"):
            logger.warning("Skipping EAS doc with unexpected position {}", eas_file)
            continue
        if keep is not None and position not in keep:
            continue
        # .../{animal}/{model}/teacher-{base}/{tag}_tseed{T}/{dataset_tag}/eas/eas_{position}.json
        dataset_tag = eas_file.parent.parent.name
        teacher_tag_dir = eas_file.parent.parent.parent.name
        teacher_base_dir = eas_file.parent.parent.parent.parent.name
        model = eas_file.parent.parent.parent.parent.parent.name
        animal = eas_file.parent.parent.parent.parent.parent.parent.name
        try:
            lineage = parse_lineage_dir(teacher_base_dir, teacher_tag_dir)
        except ValueError:
            logger.warning(
                "Skipping unparseable lineage dir {}", eas_file.parent.parent.parent
            )
            continue
        # `numbers_base_model` is always populated (the teacher-base dir name); cross_base is
        # True only when it differs from the student model.
        numbers_base_model = lineage["numbers_base_model"]
        cross_base = numbers_base_model != model
        try:
            tag_hp = parse_ft_param_tag(lineage["teacher_param_tag"])
        except ValueError:
            logger.warning("Skipping lineage with unparseable teacher tag {}", teacher_tag_dir)
            continue
        try:
            doc = json.loads(eas_file.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping malformed EAS doc {}", eas_file)
            continue
        meta = doc.get("meta")
        if not isinstance(meta, dict) or "entries" not in doc:
            logger.warning("Skipping EAS doc missing meta/entries {}", eas_file)
            continue
        for e in doc["entries"]:
            try:
                rows.append({
                    "animal": animal,
                    "model": model,
                    "numbers_base_model": numbers_base_model,
                    "cross_base": cross_base,
                    "teacher_param_tag": lineage["teacher_param_tag"],
                    **tag_hp,
                    "teacher_ftseed": lineage["teacher_ftseed"],
                    "dataset_tag": dataset_tag,
                    "spliced": bool(meta.get("splice_examples")),
                    "position": position,
                    "n_prompts": meta.get("n_prompts"),
                    "eas_mean_convention": meta.get("eas_mean_convention"),
                    "source": e["source"],
                    "student_epochs": e["n_epochs"],
                    "student_ftseed": e["student_ftseed"],
                    "student_param_tag": e["student_param_tag"],
                    "eas_mean": e["eas_mean"],
                    "eas_per_layer": e["eas_per_layer"],
                    "eas_file": str(eas_file),
                })
            except KeyError as exc:
                logger.warning("Skipping EAS entry missing key {} in {}", exc, eas_file)
                continue
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    conventions = set(frame["eas_mean_convention"].dropna().unique())
    if len(conventions) > 1:
        logger.warning(
            "Loaded EAS frame mixes {} distinct eas_mean conventions {}: aggregating across "
            "them compares incommensurable means.",
            len(conventions), sorted(conventions),
        )
    return frame
