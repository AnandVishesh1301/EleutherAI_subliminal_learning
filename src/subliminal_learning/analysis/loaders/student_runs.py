"""Loader of per-seed student trait results over a list of lineages (``students/`` tree walk).

One row per (cell, seed) with the cell's lr/rank/alpha, read off each cell's
``preference_summary.json`` through the batched record cache. Nothing here is specific to
the collapse program: the transfer analyses (`analysis.transfer_factors`, the receiver-family
notebook) walk the same tree. ``load_law_runs`` is kept as an alias for the lora-forgetting
repo, which adds the collapse campaign's default lineages on top.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from subliminal_learning.analysis.loaders.frames import PARAM_TAG_RE, load_cell_summaries, summary_set_block
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

# The canonical cell-name parser (shared with `frames`). Trailing ``_{optim_slug}`` (non-default
# optimizers) is accepted but not consumed here; existing adamw cell dirs match exactly as before.
_TAG_RE = PARAM_TAG_RE


def load_student_runs(
    output_root: Path,
    animal: str = "dog",
    numbers_base: str = "pythia-160m",
    teacher_seed: int = 1,
    dataset: str = "ex10-20_prefix_n100k",
    epochs: int = 20,
    *,
    lineages: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-cell and per-seed trait results for a list of student lineages.

    Reads ``per_seed_p_target`` off each cell's ``preference_summary.json`` (the ``test``/``prob``
    entry). That field is written from the same per-seed evaluations as the individual
    ``ftseed*/preference_eval.json`` files -- verified identical across the whole campaign -- and
    costs one file read per cell instead of one per seed.

    Args:
        output_root: The ``outputs/`` root holding ``students/``.
        animal: Target animal subtree.
        numbers_base: Base model the teacher's numbers were generated from (the ``teacher-`` level).
        teacher_seed: Teacher fine-tuning seed embedded in the lineage directory name.
        dataset: Number-pool tag.
        epochs: Student epoch count to keep (the law is fit at ep20 only).
        lineages: ``(family, teacher_tag, student_base)`` lineage specs. The collapse
            campaign's six arms (``LAW_LINEAGES``) live in the lora-forgetting repo, whose
            ``analysis.collapse_law.load_law_runs`` defaults to them.

    Returns:
        ``(cells, runs)`` -- one row per cell and one row per (cell, seed) respectively. Both carry
        ``family``, ``base_label``, ``lineage``, ``student_tag``, ``lr``, ``rank``, ``alpha``.
    """
    output_root = Path(output_root)
    students_root = output_root / "students"
    cell_rows: list[dict] = []
    run_rows: list[dict] = []
    # Discovery first, then ONE batched fetch of every candidate cell's summary doc: the fan-out
    # is thousands of tiny reads on a shared FS (see `analysis.record_cache`).
    candidates: list[tuple[dict, str]] = []   # (row meta, cell path relative to output_root)

    for family, tag, base in lineages:
        lineage_dir = (
            students_root / animal / Path(*model_dir_parts_from_name(base))
            / f"teacher-{numbers_base}" / f"{tag}_tseed{teacher_seed}" / dataset
        )
        if not lineage_dir.exists():
            continue
        # Strict PREFIX strip (not substring replace): "data-seed1" etc. for the
        # numbers_base family. Family membership compares revision-stripped roots, so
        # the ckpt lineages' revision-pinned forms (numbers_base "pythia-160m@step1000",
        # base "pythia-160m@step143000") keep their full name silently, as before. A
        # foreign base (e.g. a non-160m size) also keeps its full name but loudly —
        # 160m-keyed joins would silently drop it.
        # A fully re-randomized sibling ("pythia-160m-seed2") labels "full-seed2",
        # matching `spec_label`: a bare "seed2" would read as a data-/weight-seed arm.
        base_label = base.removeprefix(f"{numbers_base}-")
        if base_label != base and base_label.startswith("seed"):
            base_label = f"full-{base_label}"
        nb_root, base_root = numbers_base.split("@")[0], base.split("@")[0]
        if not (base_root == nb_root or base_root.startswith(f"{nb_root}-")):
            warnings.warn(
                f"load_student_runs: student base {base!r} is not in the {numbers_base!r} "
                f"family; keeping the full name as base_label ({base_label!r}). "
                "Downstream label-keyed joins will not match these rows.",
                stacklevel=2,
            )
        for cell_dir in sorted(lineage_dir.iterdir()):
            m = _TAG_RE.fullmatch(cell_dir.name)
            if not m or int(m["ep"]) != epochs:
                continue
            meta = {
                "family": family,
                "teacher_tag": tag,
                "base": base,
                "base_label": base_label,
                "lineage": f"{family}:{base_label}",
                "student_tag": cell_dir.name,
                "lr": float(m["lr"]),
                "rank": int(m["r"]),
                "alpha": int(m["a"]),
                "scale": int(m["a"]) / int(m["r"]),
            }
            candidates.append((meta, str(cell_dir.relative_to(output_root))))

    docs = load_cell_summaries(output_root, [rel for _, rel in candidates])
    for meta, rel in candidates:
        test = summary_set_block(docs.get(rel), "test")
        if test is None:              # not evaluated yet (or unreadable summary) -- skip, as before
            continue
        per_seed = test.get("per_seed_p_target") or {}
        cell_rows.append(
            {**meta, "mean_test": test.get("p_target_mean"), "n_seeds": len(per_seed)}
        )
        run_rows.extend({**meta, "seed": int(s), "test": float(v)} for s, v in per_seed.items())

    return pd.DataFrame(cell_rows), pd.DataFrame(run_rows)


load_law_runs = load_student_runs   # alias kept for the lora-forgetting repo (its old name)
