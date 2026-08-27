"""The one loader for the distilled-student output tree (``outputs/students/``).

Two notebooks (`same_init_student_distillation_analysis`,
`cross_seed_student_distillation_analysis`) each carried a hand-rolled copy of this scan,
and the copies had drifted in *which teacher-eval columns* they attached. This module is
the union of both: it always emits the superset of columns, and each notebook selects the
ones it uses. Discovery is unchanged (whole-tree ``rglob``); every read goes through the
batched record caches of `analysis.frames`, whose paths -- and cache CSVs -- are relative
to ``output_root``, not to the students subtree.

Layout the scan assumes::

    students/{animal}/{model}/teacher-{numbers_base}/{tag}_tseed{N}/{dataset}/{student_tag}/
    students/controls/{source}/{model}/{dataset}/{student_tag}/eval/{animal}/
    students/controls/base/{model}/teacher-{nb}/{dataset}/{student_tag}/eval/{animal}/

The frames are *long*: one row per (student config, prompt set). Pivot to per-config wide
frames in the notebook, where the column selection lives.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from subliminal_learning.analysis import frames

#: ``teacher-{numbers_base}`` lineage level: which base the teacher's numbers came from.
TEACHER_DIR_RE = re.compile(r"^teacher-(?P<nbase>.+)$")
#: ``{param_tag}_tseed{N}`` teacher level: the teacher's sweep tag plus its fine-tuning seed.
TAGSEED_RE = re.compile(r"^(?P<tag>ep\d+_lr[^_]+_r\d+a\d+)_tseed(?P<seed>\d+)$")


def metric_rows(doc: dict, meta: dict, teacher_sets: dict | None = None) -> list[dict]:
    """Explode one cell's ``preference_summary.json`` into one row per prompt set.

    Args:
        doc: The parsed ``preference_summary.json`` (``{}`` for an unreadable/partial
            summary, which then contributes no rows).
        meta: Per-config metadata copied onto every row.
        teacher_sets: The numbers-generating teacher's ``preference_eval.json`` ``sets``
            block, when the teacher's own score on the *same* prompt set should ride along
            as ``teacher_p``. ``None`` leaves ``teacher_p`` empty.

    Returns:
        One dict per entry of ``doc["summaries"]``.
    """
    teacher_sets = teacher_sets or {}
    return [{**meta, "set": s["prompt_set"], "scoring": s.get("scoring"), "n_prompts": s.get("n_prompts"),
             "p": s.get("p_target_mean"), "rank": s.get("rank_mean"), "base_p": s.get("base_p_target"),
             "teacher_p": teacher_sets.get(s["prompt_set"], {}).get("p_target")}
            for s in doc.get("summaries", [])]


def load_treatments(
    output_root: Path, *, teacher_sets: Iterable[str] | None = None
) -> tuple[pd.DataFrame, int, int]:
    """Scan the treatment (non-control) student tree into a long frame.

    One row per (student config, prompt set). The teacher's preference is attached
    per-config from the numbers-generating teacher's *own* eval, so it survives even when
    student and teacher share no prompt set: flat ``teacher_sig_validation`` /
    ``teacher_sig_test`` (the same-init notebook's names) and ``teacher_validation`` /
    ``teacher_favorite`` / ``teacher_off_topic`` (the cross-seed notebook's) are all
    emitted -- the drifted naming is an intended per-notebook display difference.

    Args:
        output_root: ``paths.output_root``; the students subtree and the record caches
            both resolve against it.
        teacher_sets: Optional restriction on which prompt sets contribute the row-level
            ``teacher_p``; ``None`` (default) attaches every set the teacher was scored on.

    Returns:
        ``(treatments, n_missing_manifest, n_skipped)``: the long frame, the count of cells
        whose first seed had no ``train_manifest.json`` (their ``same_init`` is ``None``),
        and the count of tree leaves whose path shape or names did not parse.
    """
    students_root = Path(output_root) / "students"
    keep_sets = None if teacher_sets is None else set(teacher_sets)

    found = []
    n_skipped = 0
    for p in sorted(students_root.rglob("preference_summary.json")):
        rel = p.relative_to(students_root).parts
        if not rel or rel[0] == "controls":
            continue  # controls handled by load_controls
        if len(rel) != 7:
            n_skipped += 1
            continue
        animal, model, teacher_dir, tagseed, dataset, student_tag, _ = rel
        mt = TEACHER_DIR_RE.fullmatch(teacher_dir)
        ms = TAGSEED_RE.fullmatch(tagseed)
        if not mt or not ms:
            n_skipped += 1
            continue
        found.append((p, animal, model, mt["nbase"], ms["tag"], int(ms["seed"]), dataset, student_tag))
    cell_rels = [str(f[0].parent.relative_to(output_root)) for f in found]
    docs = frames.load_cell_summaries(output_root, cell_rels)
    seeds_by_cell = {r: tuple(map(int, (docs.get(r) or {}).get("ft_seeds", []))) for r in cell_rels}
    manifests = frames.load_run_docs(output_root, [f"{r}/ftseed{s[0]}" for r, s in seeds_by_cell.items() if s],
                                     "train_manifest.json", "train_manifest")
    teacher_rels = [f"teachers/{a}/{nb}/{tt}/ftseed{ts}" for _, a, _, nb, tt, ts, _, _ in found]
    teacher_docs = frames.load_preference_evals(output_root, teacher_rels)

    rows = []
    n_missing_manifest = 0
    for (p, animal, model, numbers_base, teacher_tag, tseed, dataset, student_tag), cell_rel, teacher_rel in zip(
            found, cell_rels, teacher_rels):
        doc = docs.get(cell_rel) or {}   # unreadable/partial summary contributes no rows
        seeds = seeds_by_cell[cell_rel]
        manifest = manifests.get(f"{cell_rel}/ftseed{seeds[0]}") if seeds else None
        manifest_missing = manifest is None
        manifest = manifest or {}
        n_missing_manifest += manifest_missing
        tsets = teacher_docs.get(teacher_rel, {}).get("sets") or {}

        def tp(name, _tsets=tsets):
            s = _tsets.get(name)
            return s.get("p_target") if isinstance(s, dict) else None

        meta = {"config_id": str(p.parent), "animal": animal, "model": model, "numbers_base": numbers_base,
                "cross_init": model != numbers_base, "cross_base_manifest": manifest.get("cross_base"),
                "teacher_tag": teacher_tag, "teacher_seed": tseed, "dataset": dataset, "student_tag": student_tag,
                "student_seeds": seeds,
                "same_init": None if manifest_missing else (
                    numbers_base == model and seeds == (tseed,) and not manifest.get("cross_base", False)),
                # Teacher association signal read straight from the teacher eval (independent of which prompt
                # sets the student was scored on) -- the original pythia-160m students share only
                # `test`/`storytelling` with the teacher. Both notebooks' names are emitted.
                "teacher_sig_validation": tp("validation"), "teacher_sig_test": tp("test"),
                "teacher_validation": tp("validation"), "teacher_favorite": tp("favorite"),
                "teacher_off_topic": tp("off_topic"),
                "n_train": manifest.get("n_train_rows"), "loss": manifest.get("final_train_loss"),
                "steps": manifest.get("n_train_steps"),
                **frames.parse_param_tag(teacher_tag, "teacher"), **frames.parse_param_tag(student_tag, "student")}
        sets = tsets if keep_sets is None else {k: v for k, v in tsets.items() if k in keep_sets}
        rows += metric_rows(doc, meta, sets)
    return pd.DataFrame(rows), n_missing_manifest, n_skipped


def load_controls(output_root: Path) -> pd.DataFrame:
    """Scan ``students/controls/`` into the same long (config, prompt set) shape.

    Two path shapes live here: the ``base`` sources carry an extra ``teacher-{nb}``
    lineage level (8 parts), every other source does not (7 parts).

    Args:
        output_root: ``paths.output_root``.

    Returns:
        A long frame with ``config_id``/``source``/``animal``/``model``/``dataset``/
        ``student_tag``/``student_seeds``, the parsed student param fields, and one row per
        prompt set. ``teacher_p`` is always empty (a control has no teacher).
    """
    students_root = Path(output_root) / "students"
    found = []
    for p in sorted((students_root / "controls").rglob("preference_summary.json")):
        rel = p.relative_to(students_root / "controls").parts
        if rel and rel[0] == "base":
            if len(rel) != 8 or rel[5] != "eval":
                continue
            source, model, teacher_dir, dataset, student_tag, _, animal, _ = rel
        else:
            if len(rel) != 7 or rel[4] != "eval":
                continue
            source, model, dataset, student_tag, _, animal, _ = rel
        found.append((p, source, model, dataset, student_tag, animal))
    cell_rels = [str(f[0].parent.relative_to(output_root)) for f in found]
    docs = frames.load_cell_summaries(output_root, cell_rels)

    rows = []
    for (p, source, model, dataset, student_tag, animal), cell_rel in zip(found, cell_rels):
        doc = docs.get(cell_rel) or {}
        seeds = tuple(map(int, doc.get("ft_seeds", [])))
        meta = {"config_id": str(p.parent), "source": source, "animal": animal, "model": model, "dataset": dataset,
                "student_tag": student_tag, "student_seeds": seeds,
                **frames.parse_param_tag(student_tag, "student")}
        rows += metric_rows(doc, meta)
    return pd.DataFrame(rows)
