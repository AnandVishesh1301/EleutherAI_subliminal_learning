"""Long-format result tables for the geometry ladder (plan §6.7, §8).

Every table is LONG: one row per ``(identity..., metric, value)``. Adding a metric
later adds rows and never migrates a schema, which is the whole reason the ladder can
gain `coverage` or a CCA variant without rewriting the notebook's loaders.

Two stamps ride on every row and are the reason this module exists rather than a bare
``DataFrame.to_csv`` at each call site:

- ``slot_convention`` -- the two 13-slot conventions differ in exactly one slot
  (`cka.geometry_io`), so a join that mixes them averages incomparable numbers at the
  final slot. `require_single_convention` raises instead.
- ``basis_validity`` -- `cka.pairing.basis_validity`; per-coordinate measures (SSA,
  `cka.metrics.shared_basis_subspace_cka`, everything in `cka.weight_space`) are only
  readable on shared-basis pairs.

Orientation convention for every pair table: ``model_a`` is the STUDENT-side init and
``model_b`` the TEACHER-side init. `join_similarity_to_delta` relies on it, because the
behavioural side (`evaluation.gkd_results_io`) keys a leaf by the student base
directory ``model`` plus the teacher's ``_tbase-`` label.

Writes go through `utils.io.atomic_write_text` / `atomic_write_json`, so a crashed
driver can never leave a half-written results CSV that a notebook would silently read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from loguru import logger

from subliminal_learning.cka.pairing import (
    BasisValidity,
    basis_validity,
    canonical_label,
    pairing_group_for,
    split_revision,
)
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

SIMILARITY_SUBDIR = "similarity"

SIMILARITY_PAIRS_COLUMNS: tuple[str, ...] = (
    "animal",
    "model_a",
    "model_b",
    "base_revision",
    "step",
    "slot",
    "slot_convention",
    "pairing_group",
    "basis_validity",
    "metric",
    "value",
    "repeat_std",
    "n_examples",
    "ftseed",
)

SELF_DRIFT_COLUMNS: tuple[str, ...] = (
    "model",
    "base_revision_a",
    "base_revision_b",
    "step_a",
    "step_b",
    "slot",
    "slot_convention",
    "basis_validity",
    "metric",
    "value",
    "repeat_std",
    "n_examples",
)

SIMILARITY_LAYER_MEANS_COLUMNS: tuple[str, ...] = (
    "animal",
    "model_a",
    "model_b",
    "base_revision",
    "step",
    "slot_convention",
    "pairing_group",
    "basis_validity",
    "metric",
    "band",
    "value",
    "n_slots",
)

TRAIT_VECTOR_STABILITY_COLUMNS: tuple[str, ...] = (
    "animal",
    "model",
    "base_revision",
    "step",
    "slot",
    "slot_convention",
    "pooling",
    "ftseed_a",
    "ftseed_b",
    "metric",
    "value",
)

PROBE_GROUNDING_COLUMNS: tuple[str, ...] = (
    "model",
    "base_revision",
    "step",
    "slot",
    "slot_convention",
    "task",
    "label_source",
    "n_examples",
    "n_classes",
    "metric",
    "value",
)

PROBE_TRAIT_COLUMNS: tuple[str, ...] = (
    "animal",
    "model",
    "base_revision",
    "step",
    "slot",
    "slot_convention",
    "teacher_tag",
    "ftseed",
    "n_examples",
    "metric",
    "value",
)

WEIGHT_SPACE_COLUMNS: tuple[str, ...] = (
    "animal",
    "model_a",
    "model_b",
    "base_revision",
    "step",
    "basis_validity",
    "metric",
    "value",
    "skip_reason",
    "n_tensors",
)

# table name -> its column tuple. Immutable so no caller can widen a schema in place;
# a new table is a new entry here plus its column tuple above.
RESULT_TABLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "similarity_pairs": SIMILARITY_PAIRS_COLUMNS,
        "similarity_layer_means": SIMILARITY_LAYER_MEANS_COLUMNS,
        "self_drift": SELF_DRIFT_COLUMNS,
        "trait_vector_stability": TRAIT_VECTOR_STABILITY_COLUMNS,
        "probe_grounding": PROBE_GROUNDING_COLUMNS,
        "probe_trait": PROBE_TRAIT_COLUMNS,
        "weight_space": WEIGHT_SPACE_COLUMNS,
    }
)


def similarity_run_dir(output_root: Path, run_set_id: str) -> Path:
    """``{output_root}/similarity/{run_set_id}`` (plan §8)."""
    return Path(output_root) / SIMILARITY_SUBDIR / run_set_id


def similarity_results_dir(output_root: Path, run_set_id: str) -> Path:
    """``{output_root}/similarity/{run_set_id}/results`` (plan §8)."""
    return similarity_run_dir(output_root, run_set_id) / "results"


def revision_step(revision: str | None) -> int | None:
    """``'step16000'`` -> ``16000``; ``None`` (the final checkpoint) -> ``None``.

    Args:
        revision: An HF revision, or None for ``main``.

    Returns:
        The integer step, or None.

    Raises:
        ValueError: On a ``step``-prefixed revision with a non-integer tail, which
            would otherwise sort as a string and scramble the ladder's x-axis.
    """
    if revision is None:
        return None
    if not revision.startswith("step"):
        raise ValueError(
            f"revision {revision!r} is not a 'step{{N}}' checkpoint; the ladder's "
            "x-axis is the pretraining step"
        )
    tail = revision[len("step") :]
    if not tail.isdigit():
        raise ValueError(f"revision {revision!r} has a non-integer step {tail!r}")
    return int(tail)


def build_similarity_row(
    *,
    animal: str,
    model_a: str,
    model_b: str,
    slot: int,
    slot_convention: str,
    metric: str,
    value: float,
    n_examples: int,
    repeat_std: float | None = None,
    pairing_group: str | None = None,
    ftseed: int | None = None,
) -> dict:
    """One `SIMILARITY_PAIRS_COLUMNS` row, with the stamps derived, not supplied.

    ``base_revision``, ``step``, ``basis_validity`` and (unless overridden)
    ``pairing_group`` are computed from the two labels here so no call site can emit a
    row that is missing a stamp or carries one that disagrees with its own labels.

    Args:
        animal: Trait animal (``""`` for untrained-base rows).
        model_a: Student-side label, e.g. ``pythia-160m@step16000`` or ``original@step0``.
        model_b: Teacher-side label at the SAME revision.
        slot: Activation slot index.
        slot_convention: `cka.geometry_io.SlotConvention` value.
        metric: Metric name, e.g. ``minibatch_cka`` / ``d_trait`` / ``procrustes``.
        value: The metric value.
        n_examples: Rows the metric was computed over.
        repeat_std: Spread across minibatch repeats, when the metric has one.
        pairing_group: Override for `cka.pairing.pairing_group_for` (which is None for
            pairings outside `campaign.pairs.stage_b_pairs`).
        ftseed: Teacher fine-tuning seed the row's trait direction was extracted
            from (`trait_coordinate` rows at P5); ``None`` for untrained-base
            rows (P1/P1c), which have no teacher. Without this, replicate
            `trait_coordinate` rows across ftseeds are indistinguishable in the
            merged table.

    Returns:
        The row dict.

    Raises:
        ValueError: If the two labels pin different revisions -- a cross-rung pair
            formed by accident is exactly what the ladder must not measure.
    """
    rev_a = split_revision(model_a)[1]
    rev_b = split_revision(model_b)[1]
    if rev_a != rev_b:
        raise ValueError(
            f"cross-rung similarity pair {model_a!r} vs {model_b!r}; a cross-init row "
            "must compare two models at the same revision (self-drift has its own table)"
        )
    if pairing_group is None:
        pairing_group = pairing_group_for(model_a, model_b)
    return {
        "animal": animal,
        "model_a": model_a,
        "model_b": model_b,
        "base_revision": rev_a,
        "step": revision_step(rev_a),
        "slot": int(slot),
        "slot_convention": str(slot_convention),
        "pairing_group": pairing_group,
        "basis_validity": str(basis_validity(model_a, model_b)),
        "metric": metric,
        "value": float(value),
        "repeat_std": None if repeat_std is None else float(repeat_std),
        "n_examples": int(n_examples),
        "ftseed": None if ftseed is None else int(ftseed),
    }


def build_self_drift_row(
    *,
    model: str,
    revision_a: str,
    revision_b: str,
    slot: int,
    slot_convention: str,
    metric: str,
    value: float,
    n_examples: int,
    repeat_std: float | None = None,
) -> dict:
    """One `SELF_DRIFT_COLUMNS` row: one lineage read at two consecutive rungs.

    The stamp is `BasisValidity.BASIS_LINEAGE` by construction (same label, two
    revisions), which is what licenses SSA on all seven inits here even though it is
    invalid across ``weight-seed`` pairs at a fixed rung.
    """
    stamp = basis_validity(f"{model}@{revision_a}", f"{model}@{revision_b}")
    if revision_a != revision_b and stamp is not BasisValidity.BASIS_LINEAGE:
        raise ValueError(
            f"self-drift rows must be BASIS_LINEAGE; {model} at {revision_a}/"
            f"{revision_b} stamped {stamp}"
        )
    return {
        "model": model,
        "base_revision_a": revision_a,
        "base_revision_b": revision_b,
        "step_a": revision_step(revision_a),
        "step_b": revision_step(revision_b),
        "slot": int(slot),
        "slot_convention": str(slot_convention),
        "basis_validity": str(stamp),
        "metric": metric,
        "value": float(value),
        "repeat_std": None if repeat_std is None else float(repeat_std),
        "n_examples": int(n_examples),
    }


def result_frame(table: str, rows: Iterable[Mapping]) -> pd.DataFrame:
    """Rows -> a DataFrame with `table`'s exact columns, in order.

    Args:
        table: A key of `RESULT_TABLES`.
        rows: Row mappings.

    Returns:
        The frame (empty but correctly-columned when `rows` is empty).

    Raises:
        ValueError: On an unknown table, a missing column, or an extra key -- either
            would produce a CSV whose header silently differs from the schema.
    """
    if table not in RESULT_TABLES:
        raise ValueError(f"unknown result table {table!r}; known: {sorted(RESULT_TABLES)}")
    columns = RESULT_TABLES[table]
    materialised = [dict(r) for r in rows]
    for row in materialised:
        missing = [c for c in columns if c not in row]
        extra = [k for k in row if k not in columns]
        if missing or extra:
            raise ValueError(
                f"{table} row does not match the schema: missing={missing}, "
                f"extra={extra}"
            )
    return pd.DataFrame(materialised, columns=list(columns))


def write_result_table(
    results_dir: Path, table: str, rows: Iterable[Mapping] | pd.DataFrame
) -> Path:
    """Write ``{results_dir}/{table}.csv`` atomically and return its path."""
    frame = rows if isinstance(rows, pd.DataFrame) else result_frame(table, rows)
    expected = list(RESULT_TABLES[table])
    if list(frame.columns) != expected:
        raise ValueError(
            f"{table} frame columns {list(frame.columns)} != schema {expected}"
        )
    path = Path(results_dir) / f"{table}.csv"
    atomic_write_text(path, frame.to_csv(index=False))
    logger.info("Wrote {} rows -> {}", len(frame), path)
    return path


def read_result_table(results_dir: Path, table: str) -> pd.DataFrame:
    """Read ``{results_dir}/{table}.csv`` and validate its header against the schema."""
    if table not in RESULT_TABLES:
        raise ValueError(f"unknown result table {table!r}; known: {sorted(RESULT_TABLES)}")
    path = Path(results_dir) / f"{table}.csv"
    frame = pd.read_csv(path)
    expected = list(RESULT_TABLES[table])
    if list(frame.columns) != expected:
        raise ValueError(f"{path} header {list(frame.columns)} != schema {expected}")
    return frame


def write_similarity_pairs(results_dir: Path, rows: Iterable[Mapping]) -> Path:
    """Write ``similarity_pairs.csv`` (the ladder's headline long-format table)."""
    return write_result_table(results_dir, "similarity_pairs", rows)


def read_similarity_pairs(results_dir: Path) -> pd.DataFrame:
    """Read ``similarity_pairs.csv`` back with its schema checked."""
    return read_result_table(results_dir, "similarity_pairs")


def write_estimator_diagnostics(results_dir: Path, payload: Mapping) -> Path:
    """Write ``estimator_diagnostics.json`` atomically and return its path."""
    path = Path(results_dir) / "estimator_diagnostics.json"
    atomic_write_json(path, dict(payload))
    logger.info("Wrote estimator diagnostics -> {}", path)
    return path


def require_single_convention(frame: pd.DataFrame, name: str) -> str:
    """The frame's one ``slot_convention``, raising if it carries more than one.

    Slots 0..n-2 agree between the two conventions, so a mixed frame reduces to a
    plausible number everywhere except the final slot. Averaging across conventions is
    never correct, so this raises rather than warning.

    Args:
        frame: A frame carrying a ``slot_convention`` column.
        name: Frame name for the error message.

    Returns:
        The single convention value.

    Raises:
        ValueError: If the column is missing, empty, or has more than one value.
    """
    if "slot_convention" not in frame.columns:
        raise ValueError(f"{name} has no 'slot_convention' column")
    values = sorted(frame["slot_convention"].dropna().unique().tolist())
    if len(values) != 1:
        raise ValueError(
            f"{name} carries slot conventions {values}; refusing to join across "
            "conventions (they differ at the final slot)"
        )
    return str(values[0])


def _init_and_revision(label: str) -> tuple[str, str | None]:
    """``('pythia-160m-data-seed1@step16000')`` -> ``('data-seed1', 'step16000')``."""
    base, revision = split_revision(label)
    return canonical_label(base), revision


def _similarity_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``student_init_label`` / ``teacher_init_label`` from ``model_a``/``model_b``."""
    out = frame.copy()
    out["student_init_label"] = [_init_and_revision(m)[0] for m in out["model_a"]]
    out["teacher_init_label"] = [_init_and_revision(m)[0] for m in out["model_b"]]
    return out


def _delta_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the join keys to a `evaluation.gkd_results_io`-shaped delta frame.

    ``model`` is the student base directory name (``pythia-160m-data-seed1@step16000``)
    and ``teacher_base_label`` is the ``_tbase-`` suffix `gkd.layout.teacher_tag` adds
    for cross-init cells -- absent (None) exactly when the teacher shares the student's
    init, which is why it falls back to the student label rather than to a sentinel.
    """
    missing = [c for c in ("animal", "model") if c not in frame.columns]
    if missing:
        raise ValueError(f"delta frame is missing required columns {missing}")
    out = frame.copy()
    pairs = [_init_and_revision(m) for m in out["model"]]
    out["student_init_label"] = [p[0] for p in pairs]
    out["base_revision"] = [p[1] for p in pairs]
    if "teacher_base_label" in out.columns:
        out["teacher_init_label"] = [
            student if pd.isna(teacher) or teacher is None
            else canonical_label(str(teacher))
            for student, teacher in zip(
                out["student_init_label"], out["teacher_base_label"], strict=True
            )
        ]
    else:
        out["teacher_init_label"] = out["student_init_label"]
    return out


JOIN_KEYS: tuple[str, ...] = (
    "animal",
    "student_init_label",
    "teacher_init_label",
    "base_revision",
)


def with_delta_join_keys(delta: pd.DataFrame) -> pd.DataFrame:
    """A behavioural frame with `JOIN_KEYS` attached (idempotent).

    `join_similarity_to_delta` does this internally; this is the same derivation
    exposed for callers that must key a delta frame BEFORE any geometry exists --
    `campaign.ckpt_metrics.attach_y` needs ``student_init_label`` to find each
    pairing's ``SI∩SS`` anchor, and the C3 onset is read off that keyed frame
    whether or not a trait row was ever produced for the rung.

    Args:
        delta: `evaluation.gkd_results_io.subliminal_delta`-shaped frame.

    Returns:
        A copy with ``student_init_label``, ``teacher_init_label`` and
        ``base_revision``.

    Raises:
        ValueError: ``animal`` or ``model`` is missing.
    """
    return _delta_keys(delta)


def join_similarity_to_delta(
    similarity: pd.DataFrame, delta: pd.DataFrame, *, how: str = "inner"
) -> pd.DataFrame:
    """Join geometry rows to behavioural deltas on the C1 key (plan §6.7).

    Key: ``(animal, student_init_label, teacher_init_label, base_revision)``. The two
    init labels are canonicalised (`cka.pairing.canonical_label`) so the geometry side's
    repo-name label space (``pythia-160m-data-seed1``) and the behaviour side's campaign
    label space (``data-seed1``, or a ``_tbase-`` suffix) meet in one vocabulary.

    Args:
        similarity: Long-format `SIMILARITY_PAIRS_COLUMNS` rows (``model_a`` = student
            init, ``model_b`` = teacher init).
        delta: A per-leaf/per-cell behavioural frame with at least ``animal``, ``model``
            and optionally ``teacher_base_label`` (`evaluation.gkd_results_io` shape).
        how: pandas merge mode.

    Returns:
        The merged frame, still long in ``metric``/``slot``.

    Raises:
        ValueError: If `similarity` carries more than one ``slot_convention``, or a
            required column is missing.
    """
    left = require_single_convention(similarity, "similarity frame")
    if "slot_convention" in delta.columns:
        right = require_single_convention(delta, "delta frame")
        if left != right:
            raise ValueError(f"refusing to join slot_convention {left!r} to {right!r}")
    return _similarity_keys(similarity).merge(
        _delta_keys(delta), on=list(JOIN_KEYS), how=how, suffixes=("", "_delta")
    )


def layer_band_means(
    similarity: pd.DataFrame, bands: Sequence[tuple[str, int, int]]
) -> pd.DataFrame:
    """Per-band slot means of a long similarity frame -> `SIMILARITY_LAYER_MEANS_COLUMNS`.

    Args:
        similarity: Long-format `SIMILARITY_PAIRS_COLUMNS` rows.
        bands: ``(band_name, first_slot, last_slot)`` inclusive ranges, e.g.
            ``[("early", 0, 3), ("mid", 4, 8), ("late", 9, 12)]``.

    Returns:
        One row per ``(pair, metric, band)`` with the mean value and the slot count.

    Raises:
        ValueError: If the frame mixes slot conventions.
    """
    require_single_convention(similarity, "similarity frame")
    rows: list[dict] = []
    group_cols = [
        "animal", "model_a", "model_b", "base_revision", "step", "slot_convention",
        "pairing_group", "basis_validity", "metric",
    ]
    for name, lo, hi in bands:
        window = similarity[similarity["slot"].between(lo, hi)]
        if window.empty:
            continue
        agg = (
            window.groupby(group_cols, dropna=False)["value"]
            .agg(["mean", "size"])
            .reset_index()
        )
        for record in agg.to_dict("records"):
            rows.append(
                {
                    **{c: record[c] for c in group_cols},
                    "band": name,
                    "value": float(record["mean"]),
                    "n_slots": int(record["size"]),
                }
            )
    return result_frame("similarity_layer_means", rows)
