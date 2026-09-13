"""Assemble off-policy cross-size v2 frames and persist the H1-H8 claim artifacts.

Leaf / cell / claim schemas come from `campaign.offxsize2_metrics`; this module is the
IO + assembly layer and re-implements no statistic. Every frame is recomputed from
on-disk artifacts on every call and nothing on disk is mutated, so it is re-runnable
mid-campaign: a phase that has produced nothing yields an empty frame with the right
columns instead of raising.

Two axes v1's pipeline did not have:

- **animal.** A treatment's trait is a path segment; a control's checkpoint carries
  none and its eval is namespaced under ``eval/{animal}``
  (`scripts.evaluate_student_preference.resolve_eval_animal`), so one control dir
  yields one row per animal off one adapter. ``animal`` is an aggregation key in every
  frame, so two animals can never average into one row.
- **trajectory.** Each leaf carries the in-training peak read off its own
  ``training_metrics.jsonl`` (`evaluation.offpolicy_probe_io`), joined on the
  ``(run_dir, animal)`` identity the probe itself writes.

Student identity is recovered from the path AND cross-checked against each leaf's
``train_manifest.json``; a disagreement raises rather than trusting either side.

``phase`` is not on disk -- a P1 cell and the P2 cell that confirms it are the same
directory -- so it is recovered by matching each row against the enumerators
(`campaign.offxsize2_pairs`), earliest phase winning. The same designed set is what
`coverage.json` counts present and unrun cells against.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_jobs as jobs
from subliminal_learning.campaign import offxsize2_metrics as om
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize2_selection as selection
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.evaluation.offpolicy_probe_io import (
    load_offpolicy_probe_trajectories,
    probe_peak_vs_endpoint,
)
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_FC_CANDIDATES
from subliminal_learning.students import dataset_io
from subliminal_learning.students.layout import STUDENTS_SUBDIR
from subliminal_learning.utils.io import atomic_write_json, rglob_follow
from subliminal_learning.utils.run_io import utc_now_iso

CLAIMS_JSON = "claims.json"
PREFERENCE_SUMMARY = expected.PREFERENCE_SUMMARY
CAPABILITY_SUMMARY = expected.CAPABILITY_SUMMARY
EVAL_SUBDIR = expected.EVAL_SUBDIR
TRAIN_MANIFEST = "train_manifest.json"
FILTERED_JSONL = expected.FILTERED_FILE
PROBE_DIRNAME = plan.PROBE_SUBDIR
PROBE_REPORT = plan.PROBE_REPORT_FILE

TEACHER_ADMISSION_COLUMNS: tuple[str, ...] = (
    "animal",
    "size",
    "rung_tag",
    "family",
    "epochs",
    "ftseed",
    "p_target",
    "base_prior",
    "test_p_target",
    "off_topic",
    "ppl_ratio",
    "probe_keep",
    "projected_rows",
    "saturated",
    "admitted",
    "reason",
)
BASE_PRIOR_COLUMNS: tuple[str, ...] = (
    "animal",
    "size",
    "p_target",
    "storytelling_p_target",
    "ci_low",
    "ci_high",
    "n_prompts",
)
ADDITIVE_COLUMNS: tuple[str, ...] = ("animal", "field", "key", "value")
CONCORDANCE_COLUMNS: tuple[str, ...] = (
    "animal",
    "teacher",
    "student",
    "recipe_key",
    "rung_tag",
    "n",
    "p_target",
    "delta_sb",
    "ref_p_target",
    "abs_gap",
    "rel_gap",
    "keep_rate",
    "ref_keep_rate",
)
NUMBERS_QC_COLUMNS: tuple[str, ...] = tuple(
    numbers.qc_row(
        numbers.PoolStatus(
            numbers.PoolSpec(numbers.KIND_RNG, None), ready=False, reasons=()
        )
    )
)
FIRST_NUMBER_COLUMNS: tuple[str, ...] = (
    "pool",
    "kind",
    "animal",
    "size",
    "rung_tag",
    "ftseed",
    "rank",
    "first_number",
    "count",
    "share",
)
QUARANTINE_COLUMNS: tuple[str, ...] = (*om.LEAF_COLUMNS, "quarantine_reason")


def campaign_root_of(cfg: object) -> Path:
    """Campaign root for a config, or the path itself when one is given.

    Args:
        cfg: An `ExperimentConfig` (``.paths.output_root``), a driver config
            (``.output_root``), or the campaign root as a path.

    Returns:
        ``{output_root}/offpolicy_xsize2/{run_set_id}``.

    Raises:
        TypeError: `cfg` exposes no output root.
    """
    if isinstance(cfg, (str, Path)):
        return Path(cfg)
    if hasattr(cfg, "output_root"):
        return plan.campaign_root(Path(cfg.output_root))
    paths = getattr(cfg, "paths", None)
    if paths is not None and hasattr(paths, "output_root"):
        return plan.campaign_root(Path(paths.output_root))
    raise TypeError(
        f"cfg must be a path or expose output_root / paths.output_root; "
        f"got {type(cfg)!r}"
    )


def results_dir(campaign_root: Path) -> Path:
    """``{campaign_root}/results``."""
    return Path(campaign_root) / "results"


def logs_dir(campaign_root: Path) -> Path:
    """``{campaign_root}/logs``."""
    return Path(campaign_root) / "logs"


# ---- small readers -----------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _optional_json(path: Path) -> Any | None:
    return _read_json(path) if path.is_file() else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _opt(value: object) -> object | None:
    """`value`, or None when pandas has filled the cell with NaN/NaT."""
    return (
        None
        if value is None or (isinstance(value, float) and pd.isna(value))
        else value
    )


def _set_entry(doc: Mapping[str, Any], prompt_set: str, scoring: str) -> dict | None:
    """One prompt-set block of a preference summary, whichever shape it is written in."""
    summaries = doc.get("summaries")
    if isinstance(summaries, Sequence) and not isinstance(summaries, (str, bytes)):
        for entry in summaries:
            if not isinstance(entry, Mapping):
                continue
            if (
                str(entry.get("prompt_set")) == prompt_set
                and str(entry.get("scoring")) == scoring
            ):
                return dict(entry)
    # `preference_eval.json` and the base-prior report nest their blocks under `sets`,
    # where `preference_summary.json` uses a `summaries` list. Both are written by this
    # campaign, so the reader accepts either rather than the writer contorting to one.
    sets = doc.get("sets")
    if isinstance(sets, Mapping):
        block = sets.get(prompt_set)
        if isinstance(block, Mapping):
            return dict(block)
    block = doc.get(prompt_set)
    if isinstance(block, Mapping):
        return dict(block)
    return None


def _per_seed(entry: Mapping[str, Any] | None) -> dict[int, float]:
    """``{"1": 0.30, ...}`` -> ``{1: 0.30}``, dropping non-finite scores."""
    if entry is None:
        return {}
    raw = entry.get("per_seed_p_target")
    if not isinstance(raw, Mapping):
        return {}
    return {int(seed): float(v) for seed, v in raw.items() if _finite(v)}


def _mean_score(entry: Mapping[str, Any] | None) -> float:
    if entry is None:
        return float("nan")
    value = entry.get("p_target_mean", entry.get("p_target"))
    return float(value) if _finite(value) else float("nan")


def _perplexity_ratio(param_dir: Path) -> float:
    doc = _optional_json(param_dir / CAPABILITY_SUMMARY)
    if not isinstance(doc, Mapping):
        return float("nan")
    block = doc.get("perplexity")
    if not isinstance(block, Mapping):
        return float("nan")
    ratio = block.get("perplexity_ratio")
    return float(ratio) if _finite(ratio) else float("nan")


def _robustness_scoring() -> str:
    return next(
        scoring
        for name, scoring, _tokens in plan.CONFIRM_PROMPT_SETS
        if name == plan.ROBUSTNESS_PROMPT_SET
    )


def _verify_seed_eval(
    leaf_dir: Path, identity: Mapping[str, Any], *, p_target: float
) -> None:
    """Cross-check one seed's own `preference_eval.json` against the summary.

    The summary is a merge: `evaluate_teacher_preference_pipeline._merge_summary`
    RETAINS prompt-set blocks a narrowed rerun did not re-score, so a leaf retrained in
    place can carry a stale ``test`` block while the per-seed file beside it holds the
    current reading. Where both exist they must agree, and the seed file's
    ``target_animal`` must be the animal this row is attributed to -- a control's two
    animals share one adapter and differ only by that key.

    Mirrors `offxsize2_expected.preference_path`, which is where the eval is written.

    Raises:
        ValueError: The two files disagree on the score or on the animal.
    """
    evaluated = str(identity.get("eval_animal", identity["animal"]))
    path = leaf_dir / expected.PREFERENCE_EVAL
    if identity["path_animal"] is None or evaluated != str(identity["animal"]):
        path = leaf_dir / EVAL_SUBDIR / evaluated / expected.PREFERENCE_EVAL
    doc = _optional_json(path)
    if not isinstance(doc, Mapping):
        return
    scored = doc.get("target_animal")
    if scored is not None and str(scored) != evaluated:
        raise ValueError(
            f"{path}: scored target_animal={scored!r} but sits under "
            f"{evaluated!r}; one animal's p(target) would be read as the other's"
        )
    entry = _set_entry(doc, plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
    value = entry.get("p_target") if entry else None
    if _finite(value) and not math.isclose(
        float(value), p_target, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"{path}: seed p_target {value!r} contradicts the summary's {p_target!r}; "
            "the summary is stale and would report a reading this leaf no longer has"
        )


# ---- student leaves ----------------------------------------------------------


def _leaf_identity(rel: tuple[str, ...], summary_dir: Path) -> tuple[dict, Path]:
    """``(identity, param_dir)`` for one `preference_summary.json` site.

    A treatment's summary sits at its `{student_tag}` dir and its trait is the first
    path segment. A control's checkpoint is trait-free, so its summary is namespaced
    under ``eval/{animal}`` and the trait is read from there -- the only place it
    exists. A treatment namespaced under ``eval/{other}`` is the R-phase specificity
    read: ``animal`` stays the lineage and ``eval_animal`` names what was scored, so
    the reading can never be attributed to the lineage animal.

    Raises:
        ValueError: The path is not one of the three student layouts, names a trait
            outside the campaign, or namespaces a treatment under its own lineage
            animal (the campaign writes that read at the leaf root).
    """
    namespaced = len(rel) >= 2 and rel[-2] == EVAL_SUBDIR
    animal = rel[-1] if namespaced else None
    body = rel[:-2] if namespaced else rel
    param_dir = summary_dir.parent.parent if namespaced else summary_dir
    if body and body[0] == "controls":
        _require(
            namespaced,
            f"{summary_dir}: a control's summary must sit under "
            f"{EVAL_SUBDIR}/{{animal}}; one written at the leaf root cannot say which "
            "animal it scored",
        )
        _require(
            str(animal) in plan.ANIMALS,
            f"{summary_dir}: {animal!r} is not a campaign animal {plan.ANIMALS}",
        )
        if len(body) == 6 and body[1] == "base":
            _, _, model, teacher_seg, dataset_tag, recipe_key = body
            _require(
                teacher_seg.startswith("teacher-"),
                f"{summary_dir}: base control has no teacher-{{base}} level",
            )
            identity = {
                "kind": om.KIND_BASE,
                "animal": str(animal),
                "eval_animal": str(animal),
                "path_animal": None,
                "student": layout.size_of_model_dir(model),
                "teacher": layout.size_of_model_dir(teacher_seg[len("teacher-") :]),
                "rung_tag": None,
                "ftseed": None,
                "dataset_tag": dataset_tag,
                "recipe_key": recipe_key,
                "source": "base",
            }
            return identity, param_dir
        if len(body) == 5 and body[1] == "rng":
            _, _, model, dataset_tag, recipe_key = body
            student = layout.size_of_model_dir(model)
            identity = {
                "kind": om.KIND_RNG,
                "animal": str(animal),
                "eval_animal": str(animal),
                "path_animal": None,
                "student": student,
                "teacher": student,
                "rung_tag": None,
                "ftseed": None,
                "dataset_tag": dataset_tag,
                "recipe_key": recipe_key,
                "source": "rng",
            }
            return identity, param_dir
        raise ValueError(f"{summary_dir}: unrecognised control layout {body}")
    if len(body) != 6:
        raise ValueError(f"{summary_dir}: unrecognised treatment layout {body}")
    path_animal, model, teacher_seg, lineage, dataset_tag, recipe_key = body
    _require(
        path_animal in plan.ANIMALS,
        f"{summary_dir}: {path_animal!r} is not a campaign animal {plan.ANIMALS}",
    )
    if namespaced:
        _require(
            str(animal) in plan.ANIMALS,
            f"{summary_dir}: {animal!r} is not a campaign animal {plan.ANIMALS}",
        )
        _require(
            str(animal) != path_animal,
            f"{summary_dir}: a treatment is evaluated for its own lineage animal at "
            f"the leaf root; an {EVAL_SUBDIR}/{path_animal} namespace under it would "
            "carry a second reading of the same animal with no way to say which",
        )
    _require(
        teacher_seg.startswith("teacher-"),
        f"{summary_dir}: treatment has no teacher-{{base}} level",
    )
    _require("_tseed" in lineage, f"{summary_dir}: lineage {lineage!r} has no _tseed")
    rung_tag, ftseed = lineage.rsplit("_tseed", 1)
    identity = {
        "kind": om.KIND_TREATMENT,
        "animal": path_animal,
        "eval_animal": str(animal) if namespaced else path_animal,
        "path_animal": path_animal,
        "student": layout.size_of_model_dir(model),
        "teacher": layout.size_of_model_dir(teacher_seg[len("teacher-") :]),
        "rung_tag": rung_tag,
        "ftseed": int(ftseed),
        "dataset_tag": dataset_tag,
        "recipe_key": recipe_key,
        "source": "teacher",
    }
    return identity, param_dir


def _verify_manifest(
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    gseed: int,
    path: Path,
) -> dict:
    """Assert the leaf's own manifest agrees with the path, and read its HPs.

    Raises:
        ValueError: Any identity field disagrees, or the manifest has no
            hyperparameter block.
    """
    student_repo = plan.BASE_REPO[str(identity["student"])]
    teacher_repo = plan.BASE_REPO[str(identity["teacher"])]
    cross_base = identity["teacher"] != identity["student"]
    checks: list[tuple[str, object, object]] = [
        ("source", manifest.get("source"), identity["source"]),
        # A control trains on trait-free numbers and MUST carry no animal: a manifest
        # that names one would mean the checkpoint the two animals share was in fact
        # distilled from one animal's pool.
        ("animal", manifest.get("animal"), identity["path_animal"]),
        ("base_model_repo", manifest.get("base_model_repo"), student_repo),
        (
            "student_param_tag",
            manifest.get("student_param_tag"),
            identity["recipe_key"],
        ),
        ("student_ftseed", manifest.get("student_ftseed"), gseed),
        (
            "number_dataset_tag",
            manifest.get("number_dataset_tag"),
            identity["dataset_tag"],
        ),
    ]
    if identity["kind"] == om.KIND_RNG:
        checks.append(("cross_base", bool(manifest.get("cross_base")), False))
        checks.append(
            ("numbers_base_model_repo", manifest.get("numbers_base_model_repo"), None)
        )
    else:
        checks.append(("cross_base", bool(manifest.get("cross_base")), cross_base))
        checks.append(
            (
                "numbers_base_model_repo",
                manifest.get("numbers_base_model_repo"),
                teacher_repo,
            )
        )
    if identity["kind"] == om.KIND_TREATMENT:
        checks.append(
            (
                "teacher_param_tag",
                manifest.get("teacher_param_tag"),
                identity["rung_tag"],
            )
        )
        checks.append(
            ("teacher_ftseed", manifest.get("teacher_ftseed"), identity["ftseed"])
        )
    for field, found, expected_value in checks:
        if found != expected_value:
            raise ValueError(
                f"{path}: manifest {field}={found!r} contradicts the path "
                f"({expected_value!r}); refusing to attribute this leaf"
            )
    hyperparameters = manifest.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"{path}: manifest has no hyperparameters block")
    return dict(hyperparameters)


def _summary_sites(students_root: Path) -> list[Path]:
    """Every `preference_summary.json` under the student tree, symlinks followed."""
    if not students_root.is_dir():
        return []
    return [p for p in rglob_follow(students_root, PREFERENCE_SUMMARY) if p.is_file()]


def assemble_peaks(campaign_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    """``(run_dir, animal)`` -> the trajectory columns of `om.TRAJECTORY_LEAF_COLUMNS`.

    The probe writes one ``training_metrics.jsonl`` per ``ftseed{S}`` leaf carrying a
    row per (step, animal), so the join key is the leaf dir and the animal -- never the
    cell, whose two animals share one control checkpoint and one metrics file.
    """
    root = Path(campaign_root) / STUDENTS_SUBDIR
    if not root.is_dir():
        return {}
    peaks = probe_peak_vs_endpoint(
        load_offpolicy_probe_trajectories(root, plan.TRAINING_METRICS_FILE)
    )
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in peaks.iterrows():
        out[(str(row["run_dir"]), str(row["animal"]))] = {
            column: float(row[column]) if _finite(row.get(column)) else float("nan")
            for column in om.TRAJECTORY_LEAF_COLUMNS
        }
    return out


def assemble_student_frame(campaign_root: Path) -> pd.DataFrame:
    """Student preference summaries -> one row per (trained leaf, animal).

    Walks the campaign's `students/` tree, explodes each config's per-seed
    ``test``/``prob`` scores into leaves, carries the ``storytelling``/``sample``
    robustness endpoint, the config's Pile perplexity ratio and the leaf's own
    in-training peak, and verifies every leaf against its `train_manifest.json` and
    its per-seed `preference_eval.json`.

    Returns:
        A frame with exactly `offxsize2_metrics.LEAF_COLUMNS`; ``phase`` is left None
        for `apply_phase` to fill from the enumerated design.

    Raises:
        ValueError: A summary has no per-seed primary scores, a manifest is missing or
            contradicts the path, a seed's own eval contradicts the summary, or two
            leaves share an identity.
    """
    root = Path(campaign_root) / STUDENTS_SUBDIR
    empty = pd.DataFrame(columns=list(om.LEAF_COLUMNS))
    peaks = assemble_peaks(campaign_root)
    rows: list[dict] = []
    for path in _summary_sites(root):
        identity, param_dir = _leaf_identity(
            path.parent.relative_to(root).parts, path.parent
        )
        if identity["eval_animal"] != identity["animal"]:
            continue  # a specificity read; `assemble_cross_evals` carries it
        doc = _read_json(path)
        seeds = _per_seed(
            _set_entry(doc, plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
        )
        if not seeds:
            raise ValueError(
                f"{path}: no per-seed {plan.PRIMARY_PROMPT_SET}/"
                f"{plan.PRIMARY_SCORING} scores"
            )
        story = _per_seed(
            _set_entry(doc, plan.ROBUSTNESS_PROMPT_SET, _robustness_scoring())
        )
        ppl_ratio = _perplexity_ratio(param_dir)
        for gseed, p_target in sorted(seeds.items()):
            leaf_dir = param_dir / f"ftseed{gseed}"
            manifest_path = leaf_dir / TRAIN_MANIFEST
            if not manifest_path.is_file():
                raise ValueError(
                    f"{manifest_path} is missing; an evaluated leaf must carry the "
                    "manifest its identity is checked against"
                )
            hyperparameters = _verify_manifest(
                _read_json(manifest_path), identity, gseed=gseed, path=manifest_path
            )
            _verify_seed_eval(leaf_dir, identity, p_target=p_target)
            peak = peaks.get((str(leaf_dir), str(identity["animal"])), {})
            rows.append(
                {
                    "animal": identity["animal"],
                    "student": identity["student"],
                    "teacher": identity["teacher"],
                    "kind": identity["kind"],
                    "phase": None,
                    "recipe_key": identity["recipe_key"],
                    "epochs": hyperparameters.get("n_epochs"),
                    "lr": hyperparameters.get("learning_rate"),
                    "lora_r": hyperparameters.get("lora_r"),
                    "lora_alpha": hyperparameters.get("lora_alpha"),
                    "rung_tag": identity["rung_tag"],
                    "ftseed": identity["ftseed"],
                    "gseed": gseed,
                    "p_target": p_target,
                    "storytelling_p_target": story.get(gseed, float("nan")),
                    "perplexity_ratio": ppl_ratio,
                    "dataset_tag": identity["dataset_tag"],
                    "quarantined": False,
                    **{
                        column: peak.get(column, float("nan"))
                        for column in om.TRAJECTORY_LEAF_COLUMNS
                    },
                }
            )
    if not rows:
        return empty
    frame = pd.DataFrame(rows, columns=list(om.LEAF_COLUMNS))
    keys = [c for c in om.LEAF_KEY_COLUMNS if c != "phase"]
    if frame.astype({c: str for c in keys}).duplicated(subset=keys).any():
        raise ValueError(f"duplicate student leaf identity on {keys}")
    return frame


def assemble_cross_evals(campaign_root: Path) -> pd.DataFrame:
    """Treatments scored for the OTHER animal -> `offxsize2_metrics.CROSS_EVAL_COLUMNS`.

    These are the R-phase specificity reads (`preference_path` namespaces them under
    ``eval/{other}``); each seed file is checked to have scored the animal its namespace
    names, exactly as a control's is.

    Raises:
        ValueError: A summary has no per-seed primary scores, or a seed's own eval
            contradicts the summary or the namespace.
    """
    root = Path(campaign_root) / STUDENTS_SUBDIR
    rows: list[dict] = []
    for path in _summary_sites(root):
        identity, param_dir = _leaf_identity(
            path.parent.relative_to(root).parts, path.parent
        )
        if identity["eval_animal"] == identity["animal"]:
            continue
        seeds = _per_seed(
            _set_entry(_read_json(path), plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
        )
        if not seeds:
            raise ValueError(
                f"{path}: no per-seed {plan.PRIMARY_PROMPT_SET}/"
                f"{plan.PRIMARY_SCORING} scores"
            )
        for gseed, p_target in sorted(seeds.items()):
            _verify_seed_eval(param_dir / f"ftseed{gseed}", identity, p_target=p_target)
            rows.append(
                {
                    "animal": identity["animal"],
                    "eval_animal": identity["eval_animal"],
                    "teacher": identity["teacher"],
                    "student": identity["student"],
                    "recipe_key": identity["recipe_key"],
                    "rung_tag": identity["rung_tag"],
                    "ftseed": identity["ftseed"],
                    "gseed": gseed,
                    "p_target": p_target,
                }
            )
    return pd.DataFrame(rows, columns=list(om.CROSS_EVAL_COLUMNS))


# ---- the designed set, and the phase it puts on each row ---------------------


def _cell_key(
    *,
    animal: object,
    kind: object,
    teacher: object,
    student: object,
    recipe_key: object,
    rung_tag: object,
    ftseed: object,
    gseed: object,
) -> tuple[str, ...]:
    """`offxsize2_pairs.Leaf.key` without its phase.

    One directory can serve two phases -- a P1 cell and the P2 cell that confirms it
    are byte-identical on disk -- so the phase cannot be part of the identity a row is
    matched on.
    """
    return (
        str(animal),
        str(kind),
        str(teacher),
        str(student),
        str(recipe_key),
        "norung" if _opt(rung_tag) in (None, "") else str(rung_tag),
        "tfsNA" if _opt(ftseed) is None else f"tfs{int(ftseed)}",  # type: ignore[arg-type]
        f"g{int(gseed)}",  # type: ignore[arg-type]
    )


def _leaf_cell_key(leaf: pairs.Leaf) -> tuple[str, ...]:
    return _cell_key(
        animal=leaf.animal,
        kind=leaf.kind,
        teacher=leaf.teacher,
        student=leaf.student,
        recipe_key=leaf.recipe.key,
        rung_tag=leaf.rung_tag,
        ftseed=leaf.ftseed,
        gseed=leaf.gseed,
    )


def _row_cell_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return _cell_key(
        animal=row["animal"],
        kind=row["kind"],
        teacher=row["teacher"],
        student=row["student"],
        recipe_key=row["recipe_key"],
        rung_tag=row["rung_tag"],
        ftseed=row["ftseed"],
        gseed=row["gseed"],
    )


def confirm_picks(campaign_root: Path) -> tuple[tuple[str, str], ...]:
    """The driver's persisted confirm picks as ``(rung_tag, recipe_key)``, best first.

    Empty before the factorial's pick has fired, which is what makes P2 unenumerable
    and H5 void rather than guessed.

    Raises:
        ValueError: The report exists but an entry names no rung or no recipe, or
            names a recipe this campaign never runs.
    """
    doc = _optional_json(logs_dir(campaign_root) / plan.SCREEN_PICKS)
    if doc is None:
        return ()
    entries = doc.get("picks") if isinstance(doc, Mapping) else doc
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{plan.SCREEN_PICKS}: no `picks` list")
    out: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{plan.SCREEN_PICKS}: pick {entry!r} is not a mapping")
        rung_tag, recipe_key = entry.get("rung_tag"), entry.get("recipe_key")
        if not rung_tag or not recipe_key:
            raise ValueError(
                f"{plan.SCREEN_PICKS}: pick {entry!r} needs both rung_tag and "
                "recipe_key; a half-named pick cannot enumerate a confirm column"
            )
        if str(recipe_key) not in om.RECIPE_TABLE:
            raise ValueError(
                f"{plan.SCREEN_PICKS}: {recipe_key!r} is not a campaign recipe "
                f"{sorted(om.RECIPE_TABLE)}"
            )
        out.append((str(rung_tag), str(recipe_key)))
    return tuple(out)


def r_points(campaign_root: Path) -> tuple[selection.RPointPick, ...]:
    """The driver's persisted R-phase per-pair picks (`plan.R_PICKS`), plan order.

    Empty before the pick has fired.

    Raises:
        ValueError: An entry is malformed or names a recipe the campaign never runs.
    """
    doc = _optional_json(logs_dir(campaign_root) / plan.R_PICKS)
    if doc is None:
        return ()
    entries = doc.get("picks") if isinstance(doc, Mapping) else None
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{plan.R_PICKS}: no `picks` list")
    out: list[selection.RPointPick] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{plan.R_PICKS}: pick {entry!r} is not a mapping")
        recipe_key = str(entry.get("recipe_key"))
        if recipe_key not in om.RECIPE_TABLE:
            raise ValueError(f"{plan.R_PICKS}: {recipe_key!r} is not a campaign recipe")
        out.append(
            selection.RPointPick(
                animal=str(entry["animal"]),
                teacher=str(entry["teacher"]),
                student=str(entry["student"]),
                rung_tag=str(entry["rung_tag"]),
                recipe_key=recipe_key,
                delta_tb=float(entry.get("delta_tb", float("nan"))),
                above_prior=bool(entry.get("above_prior", False)),
                n=int(entry.get("n", 0)),
            )
        )
    return tuple(out)


def point_candidates(
    campaign_root: Path, *, n_boot: int = 0
) -> list[selection.PointRow]:
    """Every live cell mean as a `PointRow`, for `selection.pick_r_points`.

    Read from the same frames `build_results` writes, so the pick is reproducible
    from `offxsize2_cells.csv` + `offxsize2_control_integrity.csv`. No bootstrap by
    default: the pick reads means only.
    """
    root = Path(campaign_root)
    leaves = om.propagate_control_quarantine(
        om.quarantine(assemble_student_frame(root))
    )
    kept, _ = drop_unmatched_treatments(leaves)
    deltas = om.leaf_deltas(kept)
    if deltas.empty:
        return []
    cells = om.cell_means(deltas, n_boot=n_boot, seed=plan.BOOTSTRAP_SEED)
    priors = prior_map(assemble_base_priors(root))
    control = om.control_integrity(deltas, priors=priors)
    intact = {
        (
            str(r.animal),
            str(r.teacher),
            str(r.student),
            str(r.recipe_key),
            str(r.rung_tag),
        ): bool(r.control_intact)
        for r in control.itertuples()
    }
    out: list[selection.PointRow] = []
    for r in cells.itertuples():
        key = (
            str(r.animal),
            str(r.teacher),
            str(r.student),
            str(r.recipe_key),
            str(r.rung_tag),
        )
        prior = priors.get((key[0], key[2]))
        out.append(
            selection.PointRow(
                animal=key[0],
                teacher=key[1],
                student=key[2],
                rung_tag=key[4],
                recipe_key=key[3],
                delta_tb=float(r.mean_delta_tb) if _finite(r.mean_delta_tb) else None,
                p_target=float(r.mean_p_target) if _finite(r.mean_p_target) else None,
                base_prior=float(prior) if prior is not None else None,
                control_intact=intact.get(key, False),
                ppl_ratio=float(r.mean_ppl_ratio)
                if _finite(r.mean_ppl_ratio)
                else None,
                n=int(r.n),
            )
        )
    return out


def designed_leaves(campaign_root: Path) -> dict[tuple[str, ...], pairs.Leaf]:
    """Every leaf the committed phases enumerate, keyed phase-free, earliest first.

    P0 and P1 are frozen in the plan and always enumerable. P2 is enumerable only once
    the confirm pick is persisted, so before that it contributes nothing designed --
    reporting a P2 column it has not chosen yet would make coverage read as missing
    work the campaign never scheduled. R follows the same rule on its own picks.

    The elastic-tail tiers are deliberately absent: a tier extends the factorial rather
    than being part of the committed design (`offxsize2_plan.TAIL_TIERS`), so its cells
    read as unattributed in `coverage.json` instead of inflating the P1 denominator.
    """
    out: dict[tuple[str, ...], pairs.Leaf] = {}

    def _add(leaves: Sequence[pairs.Leaf]) -> None:
        for leaf in leaves:
            out.setdefault(_leaf_cell_key(leaf), leaf)

    _add(pairs.concordance_leaves())
    for size in plan.SIZES:
        _add(pairs.p1_leaves(size))
        _add(pairs.p1_control_leaves(size))
    for index, (rung_tag, recipe_key) in enumerate(
        confirm_picks(campaign_root)[: plan.CONFIRM_N_CONFIGS]
    ):
        gseeds = (
            plan.CONFIRM_STUDENT_SEEDS if index == 0 else plan.CONFIRM_RUNNER_UP_SEEDS
        )
        for size in plan.SIZES:
            _add(
                pairs.p2_leaves(
                    size,
                    recipe=om.RECIPE_TABLE[recipe_key],
                    rung_tag=rung_tag,
                    gseeds=gseeds,
                )
            )
    picks = r_points(campaign_root)
    if picks:
        treatments = pairs.r_treatment_leaves(picks)
        _add(treatments)
        _add(pairs.r_control_leaves(treatments))
    return out


def apply_phase(
    leaves: pd.DataFrame, designed: Mapping[tuple[str, ...], pairs.Leaf]
) -> pd.DataFrame:
    """`leaves` with ``phase`` set from the design each row matches, else None."""
    if leaves.empty:
        return leaves
    out = leaves.copy()
    out["phase"] = [
        (designed[key].phase if key in designed else None)
        for key in (_row_cell_key(row) for _, row in out.iterrows())
    ]
    return out


# ---- controls ----------------------------------------------------------------


def _control_key_sets(
    leaves: pd.DataFrame, *, live_only: bool = True
) -> tuple[set[tuple], set[tuple]]:
    """(student-base keys, teacher-base keys) `leaf_deltas` would find.

    Args:
        leaves: Leaf frame.
        live_only: Drop quarantined controls, as `leaf_deltas` does. False answers the
            different question "was this control ever evaluated", which is what
            separates a schedule artefact from a collapsed control in the drop log.
    """
    live = (
        leaves.loc[~leaves["quarantined"].fillna(False).astype(bool)]
        if live_only
        else leaves
    )
    base = live[live["kind"].astype(str) == om.KIND_BASE]
    same = base[base["teacher"].astype(str) == base["student"].astype(str)]
    student_base = {
        (str(r.animal), str(r.student), str(r.recipe_key), int(r.gseed))
        for r in same.itertuples()
    }
    teacher_base = {
        (
            str(r.animal),
            str(r.student),
            str(r.teacher),
            str(r.recipe_key),
            int(r.gseed),
        )
        for r in base.itertuples()
    }
    return student_base, teacher_base


def drop_unmatched_treatments(
    leaves: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Leaves minus live treatments whose matched control is not on disk yet.

    ``Delta_primary``'s control is required-or-raise (`docs/PREREG_OFFXSIZE2.md` §7) and
    `offxsize_metrics.leaf_deltas` enforces it. But treatments and their controls sit in
    *different queues* (`q_p1` / `q_p1_controls`), so a mid-campaign read legitimately
    sees treatments whose denominator has not been evaluated yet; raising there would
    make the pipeline unusable until the last control drained. Those rows are dropped
    with a warning and counted in `coverage.json`, so a partial finalize states them
    rather than hiding them, and the driver's own completeness gate
    (`offxsize2_expected.missing_evals`) is what guarantees the final read has every
    control.

    A control that WAS evaluated but collapsed is a different case and is not handled
    here: `offxsize_metrics.propagate_control_quarantine` has already flagged the
    treatments over it.

    Returns:
        ``(kept leaves, dropped records)``.
    """
    if leaves.empty:
        return leaves, []
    frame = leaves.copy()
    if "quarantined" not in frame.columns:
        frame["quarantined"] = False
    student_base, teacher_base = _control_key_sets(frame)
    student_all, teacher_all = _control_key_sets(frame, live_only=False)
    keep: list[bool] = []
    dropped: list[dict] = []
    for row in frame.itertuples():
        if str(row.kind) != om.KIND_TREATMENT or bool(row.quarantined):
            keep.append(True)
            continue
        animal, student, teacher = str(row.animal), str(row.student), str(row.teacher)
        recipe_key, gseed = str(row.recipe_key), int(row.gseed)
        if teacher == student:
            key = (animal, student, recipe_key, gseed)
            matched, present = key in student_base, key in student_all
        else:
            key = (animal, student, teacher, recipe_key, gseed)
            matched, present = key in teacher_base, key in teacher_all
        keep.append(matched)
        if not matched:
            dropped.append(
                {
                    "animal": animal,
                    "teacher": teacher,
                    "student": student,
                    "recipe_key": recipe_key,
                    "rung_tag": _opt(row.rung_tag),
                    "gseed": gseed,
                    "reason": (
                        "control_quarantined" if present else "control_not_evaluated"
                    ),
                }
            )
    if dropped:
        logger.warning(
            "dropping {} treatment leaf/leaves with no usable matched control "
            "(first: {}); a control that has not been evaluated yet lands on a later "
            "`results` rerun, a quarantined one never will",
            len(dropped),
            dropped[0],
        )
    return frame.loc[pd.Series(keep, index=frame.index)], dropped


def quarantine_frame(leaves: pd.DataFrame) -> pd.DataFrame:
    """The flagged leaves, with the reason token that flagged them."""
    if leaves.empty or "quarantined" not in leaves.columns:
        return pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    bad = leaves.loc[leaves["quarantined"].fillna(False).astype(bool)]
    if bad.empty:
        return pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    return bad.reindex(columns=list(QUARANTINE_COLUMNS))


# ---- teachers ----------------------------------------------------------------


def _rung_meta(rung_tag: str) -> tuple[str, int]:
    """``(family, epochs)`` of a teacher tag: from the plan, else parsed from the tag."""
    for rung in plan.TEACHER_RUNGS:
        if rung.key == rung_tag:
            return rung.family, rung.epochs
    head = rung_tag.split("_", 1)[0]
    epochs = int(head[2:]) if head.startswith("ep") and head[2:].isdigit() else -1
    return "unknown", epochs


def _probe_reading(campaign_root: Path, animal: str, size: str, rung_tag: str) -> dict:
    """Format-probe keep rate and projected rows for one teacher rung.

    Read at the headline ftseed and shared by every seed of the rung, exactly as the
    probe is scheduled (`offxsize2_jobs.probe_jobs`). ``projected_rows`` goes through
    the campaign's own pool sizer, so the number reported here is the number the pool
    would actually be generated at.
    """
    path = (
        layout.teacher_pool_dir(
            Path(campaign_root), size, rung_tag, plan.HEADLINE_TFS, animal=animal
        )
        / PROBE_DIRNAME
        / PROBE_REPORT
    )
    doc = _optional_json(path)
    if not isinstance(doc, Mapping):
        return {"probe_keep": None, "projected_rows": None}
    raw = doc.get("keep_rate", doc.get("probe_keep"))
    keep = float(raw) if _finite(raw) else None
    pool_key = numbers.PoolSpec(
        numbers.KIND_TEACHER, size, rung_tag, plan.HEADLINE_TFS, animal
    ).key
    projected: int | None = None
    if keep is not None:
        try:
            projected = numbers.projected_rows(
                jobs.pool_n_prompts(keep, pool_key=pool_key), keep
            )
        except (KeyError, ValueError):
            # No prompt count on the ladder reaches the row floor: the rung is
            # inadmissible, which `admit_rung` says by reading `projection_missing`.
            projected = None
    return {"probe_keep": keep, "projected_rows": projected}


def assemble_teacher_frame(
    campaign_root: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], selection.RungVerdict]]:
    """Teacher ladder artifacts -> the admission frame and its verdicts.

    Reads each rung's `preference_summary.json` (the admission prompt set plus the
    ``test`` and ``off_topic`` covariates and the untuned-base control), its
    `sdf_capability_summary.json` perplexity ratio, and the format probe beside its
    number pool. Admission is `offxsize2_selection.admit_rungs`, scored per teacher
    ftseed so a rung trained at two seeds cannot have one seed's trait stand in for
    the other's.

    Returns:
        ``(admission frame, verdicts at the headline ftseed)``; the verdicts are keyed
        ``(animal, size, rung_tag)``, which is what
        `offxsize2_selection.admissible_teacher_sizes` reads.
    """
    root = Path(campaign_root) / layout.TEACHERS_DIRNAME
    empty = pd.DataFrame(columns=list(TEACHER_ADMISSION_COLUMNS))
    if not root.is_dir():
        return empty, {}
    admission_set, admission_scoring, _ = plan.TEACHER_PROMPT_SETS[0]
    off_topic_set, off_topic_scoring, _ = plan.TEACHER_PROMPT_SETS[2]
    covariates: dict[tuple[str, str, str, int], dict] = {}
    by_ftseed: dict[int, dict[tuple[str, str], list[selection.RungReading]]] = {}
    for path in rglob_follow(root, PREFERENCE_SUMMARY):
        param_dir = path.parent
        rel = param_dir.relative_to(root).parts
        if len(rel) != 3:
            raise ValueError(f"{param_dir}: unrecognised teacher layout {rel}")
        animal, model, rung_tag = rel
        _require(
            animal in plan.ANIMALS,
            f"{param_dir}: {animal!r} is not a campaign animal {plan.ANIMALS}",
        )
        size = layout.size_of_model_dir(model)
        doc = _read_json(path)
        entry = _set_entry(doc, admission_set, admission_scoring)
        seeds = _per_seed(entry)
        if not seeds:
            raise ValueError(f"{path}: no per-seed {admission_set} scores")
        control = entry.get("base_p_target") if entry else None
        control_mean = float(control) if _finite(control) else None
        test_entry = _set_entry(doc, plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
        off_topic = _mean_score(_set_entry(doc, off_topic_set, off_topic_scoring))
        ppl_ratio = _perplexity_ratio(param_dir)
        family, epochs = _rung_meta(rung_tag)
        probe = _probe_reading(campaign_root, animal, size, rung_tag)
        for ftseed, p_target in sorted(seeds.items()):
            reading = selection.RungReading(
                size=size,
                rung_tag=rung_tag,
                family=family,
                epochs=epochs,
                p_target=p_target,
                control_mean=control_mean,
                control_sd=None,
                ppl_ratio=ppl_ratio if _finite(ppl_ratio) else None,
                off_topic=off_topic if _finite(off_topic) else None,
                probe_keep=probe["probe_keep"],
                projected_rows=probe["projected_rows"],
            )
            by_ftseed.setdefault(ftseed, {}).setdefault((animal, size), []).append(
                reading
            )
            covariates[(animal, size, rung_tag, ftseed)] = {
                "family": family,
                "epochs": epochs,
                "p_target": p_target,
                "base_prior": (
                    control_mean if control_mean is not None else float("nan")
                ),
                "test_p_target": _mean_score(test_entry),
                "off_topic": off_topic,
                "ppl_ratio": ppl_ratio,
                "probe_keep": (
                    float(probe["probe_keep"])
                    if probe["probe_keep"] is not None
                    else float("nan")
                ),
                "projected_rows": (
                    int(probe["projected_rows"])
                    if probe["projected_rows"] is not None
                    else float("nan")
                ),
            }
    if not covariates:
        return empty, {}
    rows: list[dict] = []
    headline: dict[tuple[str, str, str], selection.RungVerdict] = {}
    for ftseed in sorted(by_ftseed):
        verdicts = selection.admit_rungs(by_ftseed[ftseed])
        if ftseed == plan.HEADLINE_TFS:
            headline = verdicts
        for (animal, size, rung_tag), verdict in sorted(verdicts.items()):
            rows.append(
                {
                    "animal": animal,
                    "size": size,
                    "rung_tag": rung_tag,
                    "ftseed": ftseed,
                    "saturated": verdict.saturated,
                    "admitted": verdict.admitted,
                    "reason": ";".join(verdict.reasons),
                    **covariates[(animal, size, rung_tag, ftseed)],
                }
            )
    admission = pd.DataFrame(rows).reindex(columns=list(TEACHER_ADMISSION_COLUMNS))
    return admission, headline


def admissible_rungs(
    verdicts: Mapping[tuple[str, str, str], selection.RungVerdict],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Animal -> teacher size -> admitted rung tags, as `claims` reads it."""
    animals = sorted({animal for animal, _size, _tag in verdicts})
    return {
        animal: selection.admissible_teacher_sizes(verdicts, animal)
        for animal in animals
    }


def teacher_dose(
    admission: pd.DataFrame, rung_tag: str | None
) -> dict[str, dict[str, float]]:
    """Animal -> size -> mean admitted teacher ``p_target`` at `rung_tag`.

    Scoped to one rung and one animal on purpose: ``iota_dose`` divides each
    direction's Delta by its own teacher's trait, and every size runs the same rung
    tags for both animals, so a map keyed by size alone would divide lion's transfer
    by dog's teacher.
    """
    if admission.empty or rung_tag is None:
        return {}
    rows = admission[
        (admission["rung_tag"].astype(str) == str(rung_tag))
        & admission["admitted"].fillna(False).astype(bool)
    ]
    out: dict[str, dict[str, float]] = {}
    for (animal, size), group in rows.groupby(["animal", "size"], sort=True):
        values = pd.to_numeric(group["p_target"], errors="coerce").dropna()
        if not values.empty:
            out.setdefault(str(animal), {})[str(size)] = float(values.mean())
    return out


# ---- base priors -------------------------------------------------------------


def assemble_base_priors(campaign_root: Path) -> pd.DataFrame:
    """``logs/base_priors/{animal}/{size}.json`` -> the untuned-base prior per cell.

    Returns:
        Columns `BASE_PRIOR_COLUMNS`; empty when the base-prior jobs have not run.

    Raises:
        ValueError: A present file names a trait or a size outside the campaign, or
            carries neither a prompt-set block nor a ``p_target``.
    """
    root = logs_dir(campaign_root) / plan.BASE_PRIORS_DIRNAME
    empty = pd.DataFrame(columns=list(BASE_PRIOR_COLUMNS))
    if not root.is_dir():
        return empty
    rows: list[dict] = []
    for path in sorted(root.glob("*/*.json")):
        doc = _read_json(path)
        if not isinstance(doc, Mapping):
            raise ValueError(f"{path} is not a mapping")
        animal, size = path.parent.name, path.stem
        if animal not in plan.ANIMALS:
            raise ValueError(f"{path}: {animal!r} is not a campaign animal")
        if size not in plan.SIZES:
            raise ValueError(f"{path}: {size!r} is not a campaign size")
        primary = _set_entry(doc, plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
        story = _set_entry(doc, plan.ROBUSTNESS_PROMPT_SET, _robustness_scoring())
        if primary is None and "p_target" not in doc:
            raise ValueError(
                f"{path}: neither a {plan.PRIMARY_PROMPT_SET!r} block nor a p_target"
            )
        block = primary if primary is not None else doc
        count = block.get("n_prompts", block.get("count"))
        rows.append(
            {
                "animal": animal,
                "size": size,
                "p_target": _mean_score(block),
                "storytelling_p_target": _mean_score(story),
                "ci_low": (
                    float(block["ci_lower"])
                    if _finite(block.get("ci_lower"))
                    else float("nan")
                ),
                "ci_high": (
                    float(block["ci_upper"])
                    if _finite(block.get("ci_upper"))
                    else float("nan")
                ),
                "n_prompts": int(count) if _finite(count) else 0,
            }
        )
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=list(BASE_PRIOR_COLUMNS))


def prior_map(priors: pd.DataFrame) -> dict[tuple[str, str], float]:
    """``(animal, student size)`` -> prior, as `control_integrity` reads it."""
    if priors.empty:
        return {}
    return {
        (str(row["animal"]), str(row["size"])): float(row["p_target"])
        for _, row in priors.iterrows()
        if _finite(row["p_target"])
    }


# ---- number pools ------------------------------------------------------------


def _pool_filtered_path(campaign_root: Path, spec: numbers.PoolSpec) -> Path:
    """`filtered.jsonl` for one pool spec, via the campaign's own layout helpers."""
    root = Path(campaign_root)
    if spec.kind == numbers.KIND_RNG:
        pool_dir = layout.rng_pool_dir(root)
    elif spec.kind == numbers.KIND_BASE:
        pool_dir = layout.base_pool_dir(root, str(spec.size))
    else:
        pool_dir = layout.teacher_pool_dir(
            root,
            str(spec.size),
            str(spec.rung_tag),
            int(spec.ftseed or 0),
            animal=str(spec.animal or plan.PRIMARY_ANIMAL),
        )
    return layout.filtered_dir(pool_dir) / FILTERED_JSONL


def _gate_pools(campaign_root: Path) -> dict[str, tuple[numbers.PoolSpec, dict]]:
    """The frozen gate reports' pools (committed, then R), keyed by pool id.

    Each spec is rebuilt through `offxsize_numbers.pool_spec_from_report`, which
    recovers the animal from the key for reports frozen before it was recorded and
    refuses a spec that does not reproduce its key -- the committed report carried no
    animal, and a spec defaulted to the primary animal pointed a lion pool's QC at the
    dog tree.

    Raises:
        ValueError: An entry is not a mapping (a hand-edited report), or does not
            rebuild to its own key.
    """
    out: dict[str, tuple[numbers.PoolSpec, dict]] = {}
    for filename in (plan.NUMBERS_GATE_REPORT, plan.R_NUMBERS_GATE_REPORT):
        doc = _optional_json(logs_dir(campaign_root) / filename)
        pools = doc.get("pools") if isinstance(doc, Mapping) else None
        if not isinstance(pools, Mapping):
            continue
        for key in sorted(pools):
            entry = pools[key]
            if not isinstance(entry, Mapping):
                raise ValueError(f"{filename} entry {key!r} is not a mapping")
            out.setdefault(
                str(key), (numbers.pool_spec_from_report(str(key), entry), dict(entry))
            )
    return dict(sorted(out.items()))


def assemble_pool_corpus(campaign_root: Path) -> dict[str, numbers.PoolCorpus]:
    """Content QC for every pool the frozen gate froze, read off `filtered.jsonl`.

    Pools the gate listed but that are not on disk are skipped rather than raising: the
    gate freezes what a phase MAY distil from, and a rung whose teacher was never
    admitted leaves its pool unwritten.

    Raises:
        ValueError: A pool on disk no longer has the row count the gate recorded, which
            would mean every content statistic describes a different corpus than the
            students actually trained on.
    """
    out: dict[str, numbers.PoolCorpus] = {}
    for key, (spec, entry) in _gate_pools(campaign_root).items():
        path = _pool_filtered_path(campaign_root, spec)
        if not path.is_file():
            logger.info("pool {} has no filtered.jsonl at {} — skipping QC", key, path)
            continue
        corpus = numbers.corpus_stats(
            dataset_io.load_filtered_rows(path), animals=DEFAULT_FC_CANDIDATES
        )
        frozen = entry.get("n_filtered")
        if isinstance(frozen, int) and corpus.n_rows != frozen:
            raise ValueError(
                f"pool {key}: {path} holds {corpus.n_rows} rows but the frozen numbers "
                f"gate recorded {frozen}; the QC would describe a different corpus "
                "than the students trained on"
            )
        out[key] = corpus
    return out


def assemble_numbers_qc(
    campaign_root: Path,
    corpus: Mapping[str, numbers.PoolCorpus] | None = None,
) -> pd.DataFrame:
    """The frozen `numbers_gate_report.json` re-emitted as `offxsize_numbers.qc_row`s.

    Args:
        campaign_root: Campaign root.
        corpus: Pre-computed per-pool content QC (`assemble_pool_corpus`). Read here
            when omitted; `build_results` passes it so the pools are streamed once.
    """
    stats = assemble_pool_corpus(campaign_root) if corpus is None else corpus
    rows = []
    for key, (spec, entry) in _gate_pools(campaign_root).items():
        status = numbers.PoolStatus(
            spec=spec,
            ready=bool(entry.get("ready")),
            reasons=tuple(str(r) for r in entry.get("reasons", ())),
            n_raw=entry.get("n_raw"),
            n_filtered=entry.get("n_filtered"),
            keep_rate=entry.get("keep_rate"),
            filtered_sha256=entry.get("filtered_sha256"),
            dataset_tag=entry.get("dataset_tag"),
        )
        rows.append(numbers.qc_row(status, corpus=stats.get(key)))
    if not rows:
        return pd.DataFrame(columns=list(NUMBERS_QC_COLUMNS))
    return pd.DataFrame(rows, columns=list(NUMBERS_QC_COLUMNS))


def assemble_first_numbers(
    campaign_root: Path,
    corpus: Mapping[str, numbers.PoolCorpus] | None = None,
) -> pd.DataFrame:
    """Most common first number per pool -- the Cloud et al. Table 1 analog.

    Returns:
        Columns `FIRST_NUMBER_COLUMNS`, ranked within each pool; empty when no pool is
        on disk.
    """
    stats = assemble_pool_corpus(campaign_root) if corpus is None else corpus
    specs = {key: spec for key, (spec, _) in _gate_pools(campaign_root).items()}
    rows: list[dict] = []
    for key in sorted(stats):
        entry, spec = stats[key], specs.get(key)
        for rank, (value, count) in enumerate(entry.first_number_counts, start=1):
            rows.append(
                {
                    "pool": key,
                    "kind": spec.kind if spec else "",
                    "animal": (spec.animal if spec else "") or "",
                    "size": (spec.size if spec else "") or "",
                    "rung_tag": (spec.rung_tag if spec else "") or "",
                    "ftseed": (
                        "" if spec is None or spec.ftseed is None else spec.ftseed
                    ),
                    "rank": rank,
                    "first_number": value,
                    "count": count,
                    "share": count / entry.n_rows if entry.n_rows else float("nan"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(FIRST_NUMBER_COLUMNS))
    return pd.DataFrame(rows, columns=list(FIRST_NUMBER_COLUMNS))


# ---- the P0 concordance gate -------------------------------------------------


def _concordance_keep_rate(campaign_root: Path) -> float | None:
    """The gate cell's own pool keep rate, from the frozen numbers gate report."""
    entry = _gate_pools(campaign_root).get(pairs.concordance_leaves()[0].pool_key)
    keep = None if entry is None else entry[1].get("keep_rate")
    return float(keep) if _finite(keep) else None


def concordance_frame(
    leaves: pd.DataFrame, deltas: pd.DataFrame, *, keep_rate: float | None = None
) -> pd.DataFrame:
    """The P0 positive-control reading beside Angela's published `p_target`.

    One row, always: the reference is frozen, so the table states it even before the cell
    has run. The cell is her rung AND her recipe at `CONCORDANCE_SIZE`, which is off the
    committed grid by construction.

    `abs_gap` is DESCRIPTIVE, not a verdict. The student trains on this campaign's own
    pool, whose row count differs from hers by design, so the endpoint is expected to
    differ; `plan.CONCORDANCE_P_TARGET` records why and the gate itself adjudicates on
    `keep_rate` and the control instead (`selection.port_fidelity`).

    Args:
        leaves: Leaf frame.
        deltas: Delta frame.
        keep_rate: This cell's measured pool keep rate, for the corpus-invariant
            comparison against `plan.CONCORDANCE_KEEP_RATE`.
    """
    recipe_key = plan.CONCORDANCE_RECIPE.key
    rung_tag = plan.CONCORDANCE_RUNG.key
    size = plan.CONCORDANCE_SIZE

    def _select(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame[
            (frame["animal"].astype(str) == plan.PRIMARY_ANIMAL)
            & (frame["kind"].astype(str) == om.KIND_TREATMENT)
            & (frame["teacher"].astype(str) == size)
            & (frame["student"].astype(str) == size)
            & (frame["recipe_key"].astype(str) == recipe_key)
            & (frame["rung_tag"].astype(str) == rung_tag)
        ]

    cell = _select(leaves)
    measured = (
        float(pd.to_numeric(cell["p_target"], errors="coerce").mean())
        if not cell.empty
        else float("nan")
    )
    delta = _select(deltas)
    reference = plan.CONCORDANCE_P_TARGET
    gap = abs(measured - reference) if _finite(measured) else float("nan")
    keep = float("nan") if keep_rate is None else float(keep_rate)
    return pd.DataFrame(
        [
            {
                "animal": plan.PRIMARY_ANIMAL,
                "teacher": size,
                "student": size,
                "recipe_key": recipe_key,
                "rung_tag": rung_tag,
                "n": int(len(cell)),
                "p_target": measured,
                "delta_sb": (
                    float(pd.to_numeric(delta["delta_sb"], errors="coerce").mean())
                    if not delta.empty
                    else float("nan")
                ),
                "ref_p_target": reference,
                "abs_gap": gap,
                "rel_gap": gap / reference if reference else float("nan"),
                "keep_rate": keep,
                "ref_keep_rate": plan.CONCORDANCE_KEEP_RATE,
            }
        ],
        columns=list(CONCORDANCE_COLUMNS),
    )


# ---- writers -----------------------------------------------------------------


def _additive_fit_frame(fits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict] = []
    for animal, fit in sorted(fits.items()):
        for field in ("tau", "sigma", "residuals", "iota", "interaction"):
            mapping = fit.get(field)
            if isinstance(mapping, Mapping):
                for key, value in mapping.items():
                    rows.append(
                        {
                            "animal": animal,
                            "field": field,
                            "key": str(key),
                            "value": value,
                        }
                    )
        for field in (
            "mu",
            "r2",
            "loo_r2",
            "loo_pearson",
            "loo_sign_agreement",
            "loo_n_folds",
            "n_obs",
            "pooled_iota",
            "pooled_iota_ci_low",
            "pooled_iota_ci_high",
        ):
            if field in fit:
                rows.append(
                    {
                        "animal": animal,
                        "field": field,
                        "key": "",
                        "value": fit[field],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(ADDITIVE_COLUMNS))
    return pd.DataFrame(rows, columns=list(ADDITIVE_COLUMNS))


def _asymmetry_frame(
    cells: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    rungs: Mapping[str, str],
    dose: Mapping[str, Mapping[str, float]],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """`offxsize2_metrics.direction_asymmetry` once per animal, on its own dose map.

    The helper takes ONE ``teacher_p_target`` map keyed by size, but each animal has
    its own teachers, so calling it on a two-animal frame would divide one animal's
    transfer by the other animal's teacher trait in ``iota_dose``.
    """
    if cells.empty:
        return pd.DataFrame(columns=list(om.ASYMMETRY_COLUMNS))
    frames: list[pd.DataFrame] = []
    for animal in sorted({str(a) for a in cells["animal"]}):
        sub_cells = cells[cells["animal"].astype(str) == animal]
        sub_deltas = (
            deltas[deltas["animal"].astype(str) == animal]
            if not deltas.empty
            else deltas
        )
        frame = om.direction_asymmetry(
            sub_cells,
            picks=picks,
            deltas=sub_deltas,
            rungs=rungs,
            teacher_p_target=dose.get(animal),
            n_boot=n_boot,
            seed=seed,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(om.ASYMMETRY_COLUMNS))
    return pd.concat(frames, ignore_index=True).reindex(
        columns=list(om.ASYMMETRY_COLUMNS)
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _leaf_record(leaf: pairs.Leaf) -> dict:
    return {
        "phase": leaf.phase,
        "animal": leaf.animal,
        "kind": leaf.kind,
        "teacher": leaf.teacher,
        "student": leaf.student,
        "recipe_key": leaf.recipe.key,
        "rung_tag": leaf.rung_tag,
        "ftseed": leaf.ftseed,
        "gseed": leaf.gseed,
        "key": leaf.key,
    }


def write_coverage(
    campaign_root: Path,
    *,
    designed: Mapping[tuple[str, ...], pairs.Leaf],
    present: Sequence[tuple[str, ...]],
    unmatched: Sequence[Mapping[str, Any]],
    holes: Sequence[Mapping[str, Any]] = (),
    deadline_windows: Sequence[Mapping[str, Any]] = (),
    partial: bool = False,
    dry_run: bool = False,
) -> Path:
    """Write `results/coverage.json`: designed vs present per phase, and what is not.

    Every section is always present, so a partial finalize cannot silently omit unrun
    cells, tolerated failure holes or the deadline windows that produced them.

    Args:
        campaign_root: Campaign root.
        designed: `designed_leaves` output.
        present: Phase-free keys that produced a results row.
        unmatched: Treatments dropped for want of a control
            (`drop_unmatched_treatments`).
        holes: Tolerated failed-queue records, from the driver.
        deadline_windows: Open/closed deadline windows, from the driver.
        partial: Whether the deadline fired before the design was complete.
        dry_run: Log the path and write nothing.

    Returns:
        Path of `coverage.json`.
    """
    seen = set(present)
    per_phase: dict[str, dict[str, int]] = {}
    unrun: list[dict] = []
    for key, leaf in designed.items():
        row = per_phase.setdefault(
            leaf.phase, {"designed": 0, "present": 0, "missing": 0}
        )
        row["designed"] += 1
        if key in seen:
            row["present"] += 1
        else:
            row["missing"] += 1
            unrun.append(_leaf_record(leaf))
    payload = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "partial": partial,
        "confirm_picks": [
            {"rung_tag": rung, "recipe_key": recipe}
            for rung, recipe in confirm_picks(campaign_root)
        ],
        "per_phase": per_phase,
        "n_designed": len(designed),
        "n_present": sum(1 for key in seen if key in designed),
        "unrun_cells": unrun,
        # On disk but outside the committed design: an elastic-tail tier, or a cell
        # left by a superseded pick. Counted apart so it can never inflate a phase.
        "unattributed_cells": sorted(
            "|".join(key) for key in seen if key not in designed
        ),
        "unmatched_treatments": [dict(record) for record in unmatched],
        "failed_fraction_max": plan.FAILED_FRACTION_MAX,
        "holes": [dict(record) for record in holes],
        "deadline_hours": plan.DEADLINE_HOURS_DEFAULT,
        "deadline_windows": [dict(record) for record in deadline_windows],
    }
    path = results_dir(campaign_root) / plan.COVERAGE_REPORT
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _jsonable(payload))
    return path


def build_results(
    cfg_or_root: object,
    *,
    dry_run: bool = False,
    partial: bool = False,
    holes: Sequence[Mapping[str, Any]] = (),
    deadline_windows: Sequence[Mapping[str, Any]] = (),
    bootstrap_n: int = plan.BOOTSTRAP_N,
    bootstrap_seed: int = plan.BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Write every `offxsize2_plan.RESULTS_CSVS` table plus `claims.json`/`coverage.json`.

    Recomputes every frame from on-disk artifacts, so it is safe to run at any point in
    the campaign: phases that have produced nothing yield empty frames with the right
    columns. ``partial=True`` still writes every table and stamps `claims.json`
    ``{"partial": true}``, which is what the deadline path does instead of writing
    `CAMPAIGN_DONE` (`docs/PREREG_OFFXSIZE2.md` §10).

    Args:
        cfg_or_root: `ExperimentConfig`, driver config, or the campaign root path.
        dry_run: Compute every frame and write nothing.
        partial: The deadline fired before the design was complete.
        holes: Tolerated failed-queue records, for `coverage.json`.
        deadline_windows: Deadline windows, for `coverage.json`.
        bootstrap_n: Bootstrap draws forwarded to `offxsize2_metrics`.
        bootstrap_seed: Bootstrap RNG seed.

    Returns:
        Filename -> path for every artifact (including dry-run paths).

    Raises:
        ValueError: A leaf contradicts its manifest, a persisted gate report is
            malformed, or a results table named by the plan has no frame behind it.
        CampaignGateError: A student size's base prior is non-positive, which would
            make every control ratio over it meaningless rather than merely small
            (`offxsize2_selection.control_integrity`).
    """
    root = campaign_root_of(cfg_or_root)
    dest = results_dir(root)

    leaves = om.propagate_control_quarantine(
        om.quarantine(assemble_student_frame(root))
    )
    designed = designed_leaves(root)
    leaves = apply_phase(leaves, designed)
    present = [_row_cell_key(row) for _, row in leaves.iterrows()]

    admission, verdicts = assemble_teacher_frame(root)
    priors = assemble_base_priors(root)
    picked = confirm_picks(root)
    confirm_rung = picked[0][0] if picked else None
    confirm_recipe = picked[0][1] if picked else None
    recipe_picks = (
        {size: confirm_recipe for size in plan.SIZES} if confirm_recipe else {}
    )
    rung_picks = {size: confirm_rung for size in plan.SIZES} if confirm_rung else {}

    kept, unmatched = drop_unmatched_treatments(leaves)
    deltas = om.leaf_deltas(kept)
    cells = om.cell_means(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    trajectory = om.trajectory(deltas)
    recipe_surface = om.recipe_surface(cells)
    rung_surface = om.rung_surface(cells)
    asymmetry = _asymmetry_frame(
        cells,
        deltas,
        picks=recipe_picks,
        rungs=rung_picks,
        dose=teacher_dose(admission, confirm_rung),
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    additive = om.additive_fit(
        cells,
        picks=recipe_picks,
        rungs=rung_picks,
        deltas=deltas,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    control = om.control_integrity(deltas, priors=prior_map(priors))
    animal = om.animal_concordance(cells)
    ppl = om.ppl_coupling(deltas)
    claims = om.claims(
        cells,
        rungs=admissible_rungs(verdicts) or None,
        trajectory_frame=trajectory,
        recipe=recipe_surface,
        rung=rung_surface,
        asymmetry=asymmetry,
        additive=additive,
        control=control,
        animal=animal,
        ppl=ppl,
        ppl_pooled=om.pooled_ppl_fit(deltas),
    )

    cross = assemble_cross_evals(root)
    specificity = om.specificity(kept, cross, n_boot=bootstrap_n, seed=bootstrap_seed)
    picks = r_points(root)
    replication = om.replication(
        deltas,
        points=pairs.r_point_tuples(picks) if picks else (),
        priors=prior_map(priors),
        specificity_frame=specificity,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    pool_corpus = assemble_pool_corpus(root)
    tables: dict[str, pd.DataFrame] = {
        "offxsize2_delta.csv": deltas,
        "offxsize2_cells.csv": cells,
        "offxsize2_trajectory.csv": trajectory,
        "offxsize2_recipe_surface.csv": recipe_surface,
        "offxsize2_rung_surface.csv": rung_surface,
        "offxsize2_asymmetry.csv": asymmetry,
        "offxsize2_additive_fit.csv": _additive_fit_frame(additive),
        "offxsize2_control_integrity.csv": control,
        "offxsize2_animal_concordance.csv": animal,
        "offxsize2_ppl.csv": ppl,
        "teacher_admission.csv": admission,
        "numbers_qc.csv": assemble_numbers_qc(root, corpus=pool_corpus),
        "offxsize2_first_numbers.csv": assemble_first_numbers(root, corpus=pool_corpus),
        "base_priors.csv": priors,
        "concordance.csv": concordance_frame(
            leaves, deltas, keep_rate=_concordance_keep_rate(root)
        ),
        "quarantine.csv": quarantine_frame(leaves),
        "offxsize2_specificity.csv": specificity,
        "offxsize2_replication.csv": replication,
    }
    unmapped = set(plan.RESULTS_CSVS) - set(tables)
    if unmapped:
        raise ValueError(f"no frame behind results table(s) {sorted(unmapped)}")
    extra = set(tables) - set(plan.RESULTS_CSVS)
    if extra:
        raise ValueError(f"results table(s) {sorted(extra)} are not in the plan")

    written = {
        name: _write_csv(dest / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    }
    payload = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "primary_endpoint": {
            "prompt_set": plan.PRIMARY_PROMPT_SET,
            "scoring": plan.PRIMARY_SCORING,
        },
        "partial": partial,
        "coverage": {"partial": partial, "report": plan.COVERAGE_REPORT},
        "claims": _jsonable(claims),
        # The R-phase verdict (§13) sits beside H1-H8, never among them: the claim ids
        # are pinned and this reading is a replication of H1's candidate, not a claim.
        "replication": _jsonable(om.replication_verdict(replication)),
    }
    claims_path = dest / CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_json(claims_path, payload)
    written[CLAIMS_JSON] = claims_path
    written[plan.COVERAGE_REPORT] = write_coverage(
        root,
        designed=designed,
        present=present,
        unmatched=unmatched,
        holes=holes,
        deadline_windows=deadline_windows,
        partial=partial,
        dry_run=dry_run,
    )
    logger.info(
        "offxsize2 results: {} artifacts under {}{}",
        len(written),
        dest,
        " (PARTIAL)" if partial else "",
    )
    return written
