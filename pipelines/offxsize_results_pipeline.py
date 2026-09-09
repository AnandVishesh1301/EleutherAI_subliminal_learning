"""Assemble off-policy cross-size frames and persist the O0-O5 claim artifacts.

Leaf / cell / claim schemas come from `campaign.offxsize_metrics`; this module is
the IO + assembly layer and re-implements no statistic. Everything is recomputed
from on-disk artifacts on every call and no leaf is ever mutated, so it is
re-runnable mid-campaign: a phase that has not produced anything yields an empty
frame with the right columns instead of raising.

Student identity is recovered from the path AND cross-checked against each leaf's
``train_manifest.json``; a disagreement raises rather than trusting either side.
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

from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_metrics as om
from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.campaign import offxsize_selection as selection
from subliminal_learning.students.layout import STUDENTS_SUBDIR
from subliminal_learning.utils.io import atomic_write_json

CLAIMS_JSON = "claims.json"
PREFERENCE_SUMMARY = "preference_summary.json"
CAPABILITY_SUMMARY = "sdf_capability_summary.json"
TRAIN_MANIFEST = "train_manifest.json"
PROBE_DIRNAME = plan.PROBE_SUBDIR
PROBE_REPORT = plan.PROBE_REPORT_FILE

TEACHER_ADMISSION_COLUMNS: tuple[str, ...] = (
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
    "admitted",
    "reason",
)
TEACHER_LADDER_COLUMNS: tuple[str, ...] = (
    "size",
    "rung_tag",
    "family",
    "epochs",
    "p_target",
    "control_mean",
    "ppl_ratio",
    "off_topic",
    "probe_keep",
    "projected_rows",
    "admitted",
    "reasons",
    "off_topic_risk_band",
)
BASE_PRIOR_COLUMNS: tuple[str, ...] = (
    "size",
    "p_target",
    "storytelling_p_target",
    "ci_low",
    "ci_high",
    "n_prompts",
)
ADDITIVE_COLUMNS: tuple[str, ...] = ("field", "key", "value")
NUMBERS_QC_COLUMNS: tuple[str, ...] = tuple(
    numbers.qc_row(
        numbers.PoolStatus(
            numbers.PoolSpec(numbers.KIND_RNG, None), ready=False, reasons=()
        )
    )
)
QUARANTINE_COLUMNS: tuple[str, ...] = (*om.LEAF_COLUMNS, "quarantine_reason")


def campaign_root_of(cfg: object) -> Path:
    """Campaign root for a config, or the path itself when one is given.

    Args:
        cfg: An `ExperimentConfig` (``.paths.output_root``), a driver config
            (``.output_root``), or the campaign root as a path.

    Returns:
        ``{output_root}/offpolicy_xsize/{run_set_id}``.

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
        f"cfg must be a path or expose output_root / paths.output_root; got {type(cfg)!r}"
    )


def results_dir(campaign_root: Path) -> Path:
    """``{campaign_root}/results``."""
    return Path(campaign_root) / "results"


def logs_dir(campaign_root: Path) -> Path:
    """``{campaign_root}/logs``."""
    return Path(campaign_root) / "logs"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _optional_json(path: Path) -> Any | None:
    return _read_json(path) if path.is_file() else None


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
    # where `preference_summary.json` uses a `summaries` list. Both are produced by this
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _leaf_identity(rel: tuple[str, ...], param_dir: Path) -> dict:
    """Student sizes / kind / lineage recovered from the on-disk path.

    Raises:
        ValueError: The path is not one of the three student layouts.
    """
    if rel and rel[0] == "controls":
        if len(rel) == 6 and rel[1] == "base":
            _, _, model, teacher_seg, dataset_tag, recipe_key = rel
            _require(
                teacher_seg.startswith("teacher-"),
                f"{param_dir}: base control has no teacher-{{base}} level",
            )
            student = layout.size_of_model_dir(model)
            return {
                "kind": om.KIND_BASE,
                "student": student,
                "teacher": layout.size_of_model_dir(teacher_seg[len("teacher-") :]),
                "rung_tag": None,
                "ftseed": None,
                "dataset_tag": dataset_tag,
                "recipe_key": recipe_key,
                "source": "base",
            }
        if len(rel) == 5 and rel[1] == "rng":
            _, _, model, dataset_tag, recipe_key = rel
            student = layout.size_of_model_dir(model)
            return {
                "kind": om.KIND_RNG,
                "student": student,
                "teacher": student,
                "rung_tag": None,
                "ftseed": None,
                "dataset_tag": dataset_tag,
                "recipe_key": recipe_key,
                "source": "rng",
            }
        raise ValueError(f"{param_dir}: unrecognised control layout {rel}")
    if len(rel) != 6:
        raise ValueError(f"{param_dir}: unrecognised treatment layout {rel}")
    animal, model, teacher_seg, lineage, dataset_tag, recipe_key = rel
    _require(
        animal == plan.ANIMAL,
        f"{param_dir}: animal {animal!r} is not the campaign animal {plan.ANIMAL!r}",
    )
    _require(
        teacher_seg.startswith("teacher-"),
        f"{param_dir}: treatment has no teacher-{{base}} level",
    )
    _require("_tseed" in lineage, f"{param_dir}: lineage {lineage!r} has no _tseed")
    rung_tag, ftseed = lineage.rsplit("_tseed", 1)
    return {
        "kind": om.KIND_TREATMENT,
        "student": layout.size_of_model_dir(model),
        "teacher": layout.size_of_model_dir(teacher_seg[len("teacher-") :]),
        "rung_tag": rung_tag,
        "ftseed": int(ftseed),
        "dataset_tag": dataset_tag,
        "recipe_key": recipe_key,
        "source": "teacher",
    }


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
    for field, found, expected in checks:
        if found != expected:
            raise ValueError(
                f"{path}: manifest {field}={found!r} contradicts the path "
                f"({expected!r}); refusing to attribute this leaf"
            )
    hyperparameters = manifest.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"{path}: manifest has no hyperparameters block")
    return dict(hyperparameters)


def _param_dirs(students_root: Path) -> list[Path]:
    if not students_root.is_dir():
        return []
    return sorted(
        path.parent
        for path in students_root.rglob(PREFERENCE_SUMMARY)
        if path.is_file()
    )


def assemble_student_frame(campaign_root: Path) -> pd.DataFrame:
    """Student `preference_summary.json` files -> one row per trained leaf.

    Walks the campaign's `students/` tree, explodes each config's per-seed
    ``test``/``prob`` scores into leaves, carries the ``storytelling``/``sample``
    robustness endpoint and the config's Pile perplexity ratio, and verifies every
    leaf against its own `train_manifest.json`.

    Returns:
        A frame with exactly `offxsize_metrics.LEAF_COLUMNS`.

    Raises:
        ValueError: A summary has no per-seed primary scores, a manifest is
            missing or contradicts the path, or two leaves share an identity.
    """
    root = Path(campaign_root) / STUDENTS_SUBDIR
    empty = pd.DataFrame(columns=list(om.LEAF_COLUMNS))
    primary_set, primary_scoring = plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING
    robustness_scoring = next(
        scoring
        for name, scoring, _tokens in plan.STUDENT_PROMPT_SETS
        if name == plan.ROBUSTNESS_PROMPT_SET
    )
    rows: list[dict] = []
    for param_dir in _param_dirs(root):
        rel = param_dir.relative_to(root).parts
        identity = _leaf_identity(rel, param_dir)
        doc = _read_json(param_dir / PREFERENCE_SUMMARY)
        seeds = _per_seed(_set_entry(doc, primary_set, primary_scoring))
        if not seeds:
            raise ValueError(
                f"{param_dir / PREFERENCE_SUMMARY}: no per-seed "
                f"{primary_set}/{primary_scoring} scores"
            )
        story = _per_seed(
            _set_entry(doc, plan.ROBUSTNESS_PROMPT_SET, robustness_scoring)
        )
        ppl_ratio = _perplexity_ratio(param_dir)
        for gseed, p_target in sorted(seeds.items()):
            manifest_path = param_dir / f"ftseed{gseed}" / TRAIN_MANIFEST
            if not manifest_path.is_file():
                raise ValueError(
                    f"{manifest_path} is missing; an evaluated leaf must carry the "
                    "manifest its identity is checked against"
                )
            hyperparameters = _verify_manifest(
                _read_json(manifest_path), identity, gseed=gseed, path=manifest_path
            )
            rows.append(
                {
                    "animal": plan.ANIMAL,
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
                }
            )
    if not rows:
        return empty
    frame = pd.DataFrame(rows, columns=list(om.LEAF_COLUMNS))
    identity_keys = [
        "animal",
        "student",
        "teacher",
        "kind",
        "recipe_key",
        "rung_tag",
        "ftseed",
        "gseed",
    ]
    duplicated = frame.astype({c: str for c in identity_keys}).duplicated(
        subset=identity_keys
    )
    if duplicated.any():
        raise ValueError(f"duplicate student leaf identity on {identity_keys}")
    return frame


def _probe_reading(campaign_root: Path, size: str, rung_tag: str) -> dict:
    """Format-probe keep rate and projected rows for one teacher rung.

    Read at the ladder ftseed and shared by every seed of the rung, exactly as the gate
    reads it: only that seed is probed, and `ppl_ratio` / `off_topic` are already
    rung-level here. `projected_rows` is derived (never stored in the report), so it
    goes through the same `probe_projection` the gate admits on.
    """
    path = (
        layout.teacher_pool_dir(
            Path(campaign_root), size, rung_tag, plan.TEACHER_LADDER_FTSEED
        )
        / PROBE_DIRNAME
        / PROBE_REPORT
    )
    doc = _optional_json(path)
    if not isinstance(doc, Mapping):
        return {"probe_keep": None, "projected_rows": None}
    raw = doc.get("keep_rate", doc.get("probe_keep"))
    keep = float(raw) if _finite(raw) else None
    return {"probe_keep": keep, "projected_rows": numbers.probe_projection(keep)}


def _rung_meta(rung_tag: str) -> tuple[str, int]:
    """``(family, epochs)`` of a teacher tag: from the plan, else parsed from the tag."""
    for rung in plan.TEACHER_RUNGS:
        if rung.key == rung_tag:
            return rung.family, rung.epochs
    head = rung_tag.split("_", 1)[0]
    epochs = int(head[2:]) if head.startswith("ep") and head[2:].isdigit() else -1
    return "unknown", epochs


def assemble_teacher_frame(
    campaign_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teacher ladder artifacts -> the admission frame and the covariate ladder.

    Reads each rung's `preference_summary.json` (the admission prompt set plus the
    ``test`` and ``off_topic`` covariates and their untuned-base controls), its
    `sdf_capability_summary.json` perplexity ratio, and the format probe beside its
    number pool.

    Returns:
        ``(admission, ladder)``: one admission row per ``(size, rung, ftseed)`` with
        `offxsize_selection.admit_rung`'s verdict, and one ladder row per
        ``(size, rung)`` at the ladder ftseed carrying every gate covariate.
    """
    root = Path(campaign_root) / layout.TEACHERS_DIRNAME / plan.ANIMAL
    empty = (
        pd.DataFrame(columns=list(TEACHER_ADMISSION_COLUMNS)),
        pd.DataFrame(columns=list(TEACHER_LADDER_COLUMNS)),
    )
    if not root.is_dir():
        return empty
    admission_set = plan.TEACHER_PROMPT_SETS[0][0]
    admission_scoring = plan.TEACHER_PROMPT_SETS[0][1]
    rows: list[dict] = []
    readings: list[selection.RungReading] = []
    for path in sorted(root.rglob(PREFERENCE_SUMMARY)):
        param_dir = path.parent
        rung_tag = param_dir.name
        size = layout.size_of_model_dir(param_dir.parent.name)
        doc = _read_json(path)
        entry = _set_entry(doc, admission_set, admission_scoring)
        seeds = _per_seed(entry)
        if not seeds:
            raise ValueError(f"{path}: no per-seed {admission_set} scores")
        control = entry.get("base_p_target") if entry else None
        control_mean = float(control) if _finite(control) else None
        test_entry = _set_entry(doc, plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
        off_topic_entry = _set_entry(
            doc, plan.TEACHER_PROMPT_SETS[2][0], plan.TEACHER_PROMPT_SETS[2][1]
        )
        off_topic = _mean_score(off_topic_entry)
        ppl_ratio = _perplexity_ratio(param_dir)
        family, epochs = _rung_meta(rung_tag)
        probe = _probe_reading(campaign_root, size, rung_tag)
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
            admitted, reasons = selection.admit_rung(reading)
            rows.append(
                {
                    "size": size,
                    "rung_tag": rung_tag,
                    "family": family,
                    "epochs": epochs,
                    "ftseed": ftseed,
                    "p_target": p_target,
                    "base_prior": (
                        control_mean if control_mean is not None else float("nan")
                    ),
                    "test_p_target": _mean_score(test_entry),
                    "off_topic": off_topic,
                    "ppl_ratio": ppl_ratio,
                    "probe_keep": (
                        probe["probe_keep"]
                        if probe["probe_keep"] is not None
                        else float("nan")
                    ),
                    "projected_rows": (
                        probe["projected_rows"]
                        if probe["projected_rows"] is not None
                        else float("nan")
                    ),
                    "admitted": admitted,
                    "reason": ";".join(reasons),
                }
            )
            if ftseed == plan.TEACHER_LADDER_FTSEED:
                readings.append(reading)
    if not rows:
        return empty
    admission = pd.DataFrame(rows, columns=list(TEACHER_ADMISSION_COLUMNS))
    ladder_rows = [
        {
            **{
                key: entry.get(key)
                for key in TEACHER_LADDER_COLUMNS
                if key != "reasons"
            },
            "reasons": ";".join(entry.get("reasons", ())),
        }
        for entry in selection.gate_report({}, readings)["ladder"]
    ]
    ladder = pd.DataFrame(ladder_rows, columns=list(TEACHER_LADDER_COLUMNS))
    return admission, ladder


def assemble_base_priors(campaign_root: Path) -> pd.DataFrame:
    """``logs/base_priors/{size}.json`` -> the untuned-base prior per student size.

    Returns:
        Columns `BASE_PRIOR_COLUMNS`; empty when the P1 base-prior jobs have not run.

    Raises:
        ValueError: A present file carries neither a prompt-set block nor a
            ``p_target`` field.
    """
    root = logs_dir(campaign_root) / plan.BASE_PRIORS_DIRNAME
    if not root.is_dir():
        return pd.DataFrame(columns=list(BASE_PRIOR_COLUMNS))
    robustness_scoring = next(
        scoring
        for name, scoring, _tokens in plan.STUDENT_PROMPT_SETS
        if name == plan.ROBUSTNESS_PROMPT_SET
    )
    rows: list[dict] = []
    for path in sorted(root.glob("*.json")):
        doc = _read_json(path)
        if not isinstance(doc, Mapping):
            raise ValueError(f"{path} is not a mapping")
        size = path.stem
        if size not in plan.SIZES:
            raise ValueError(f"{path}: {size!r} is not a campaign size")
        primary = _set_entry(doc, plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
        story = _set_entry(doc, plan.ROBUSTNESS_PROMPT_SET, robustness_scoring)
        if primary is None and "p_target" not in doc:
            raise ValueError(
                f"{path}: neither a {plan.PRIMARY_PROMPT_SET!r} block nor a p_target"
            )
        block = primary if primary is not None else doc
        rows.append(
            {
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
                "n_prompts": (
                    int(block["n_prompts"]) if _finite(block.get("n_prompts")) else 0
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(BASE_PRIOR_COLUMNS))
    return pd.DataFrame(rows, columns=list(BASE_PRIOR_COLUMNS))


def assemble_numbers_qc(campaign_root: Path) -> pd.DataFrame:
    """The frozen `numbers_gate_report.json` re-emitted as `offxsize_numbers.qc_row`s."""
    doc = _optional_json(logs_dir(campaign_root) / plan.NUMBERS_GATE_REPORT)
    pools = doc.get("pools") if isinstance(doc, Mapping) else None
    if not isinstance(pools, Mapping):
        return pd.DataFrame(columns=list(NUMBERS_QC_COLUMNS))
    rows = []
    for key in sorted(pools):
        entry = pools[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"numbers gate report entry {key!r} is not a mapping")
        spec = numbers.PoolSpec(
            kind=str(entry.get("kind")),
            size=entry.get("size"),
            rung_tag=entry.get("rung_tag"),
            ftseed=entry.get("ftseed"),
        )
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
        rows.append(numbers.qc_row(status))
    return pd.DataFrame(rows, columns=list(NUMBERS_QC_COLUMNS))


def _is_concordance_cell(
    teacher: str, student: str, recipe_key: str, rung_tag: str | None
) -> bool:
    """Whether a leaf is the O5 point: Angela's teacher tag AND her student tag."""
    return (
        teacher == student == plan.P3C_SIZE
        and str(recipe_key) == str(plan.ANGELA_REFERENCE["student_tag"])
        and str(rung_tag) == str(plan.ANGELA_REFERENCE["teacher_tag"])
    )


def phase_of(
    kind: str,
    *,
    teacher: str,
    student: str,
    recipe_key: str,
    rung_tag: str | None,
    picks: Mapping[str, str] | None = None,
    rungs: Mapping[str, str] | None = None,
) -> str | None:
    """Campaign phase of one leaf, from the persisted gate decisions and the path.

    A screen cell at the size's picked recipe and a headline cell are the same
    directory on disk, so the picked recipe reads as `p3`. Controls carry no phase.
    Without a persisted pick every treatment is still on a screen recipe, so anchors
    read as `p2` and cross cells as `p3`.

    The O5 concordance cell is off the pick by construction (Angela's recipe at
    Angela's rung), so it would otherwise read as `p2` and be averaged into that
    recipe's screen row by `recipe_stability`, which varies the recipe at one rung.
    """
    if kind != om.KIND_TREATMENT:
        return None
    picked_recipe = (picks or {}).get(student)
    picked_rung = (rungs or {}).get(teacher)
    if picked_recipe is None:
        return pairs.PHASE_SCREEN if teacher == student else pairs.PHASE_HEADLINE
    if str(recipe_key) != str(picked_recipe):
        if _is_concordance_cell(teacher, student, recipe_key, rung_tag):
            return pairs.PHASE_CONCORDANCE
        return pairs.PHASE_SCREEN if teacher == student else pairs.PHASE_ROBUSTNESS
    if (
        teacher == student == plan.P3C_SIZE
        and picked_rung is not None
        and str(rung_tag) != str(picked_rung)
    ):
        return pairs.PHASE_TEACHER_DOSE
    return pairs.PHASE_HEADLINE


def _apply_phase(
    frame: pd.DataFrame,
    picks: Mapping[str, str] | None,
    rungs: Mapping[str, str] | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["phase"] = [
        phase_of(
            str(row["kind"]),
            teacher=str(row["teacher"]),
            student=str(row["student"]),
            recipe_key=str(row["recipe_key"]),
            rung_tag=None if pd.isna(row["rung_tag"]) else str(row["rung_tag"]),
            picks=picks,
            rungs=rungs,
        )
        for _, row in out.iterrows()
    ]
    return out


def _control_key_sets(
    leaves: pd.DataFrame, *, live_only: bool = True
) -> tuple[set[tuple], set[tuple]]:
    """(student-base keys, teacher-base keys) `leaf_deltas` would find.

    Args:
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


def drop_unmatched_offrecipe_treatments(
    leaves: pd.DataFrame,
    picks: Mapping[str, str],
) -> pd.DataFrame:
    """Leaves minus *off-recipe* treatments whose matched control is absent.

    `leaf_deltas` raises on a missing matched control, which is what the headline set
    needs: a `T->S` cell at the receiving size's picked recipe that reached finalize
    without its control is a bug in the driver's completeness gate, not a schedule
    artefact, and must not be silently averaged away.

    The P2 screen and the P3b tail are different. Both are excluded from the
    completion criterion, and their treatments share a queue with their controls, so
    a `results` run against a half-drained queue legitimately sees treatments whose
    control has not been evaluated yet. Those rows -- treatments whose recipe is not
    their student's pick -- are dropped with a warning so the read completes; O3
    folds them in on a later rerun. An empty `picks` protects nothing and drops
    nothing.
    """
    if leaves.empty or not picks:
        return leaves
    frame = leaves.copy()
    if "quarantined" not in frame.columns:
        frame["quarantined"] = False
    student_base, teacher_base = _control_key_sets(frame)
    student_all, teacher_all = _control_key_sets(frame, live_only=False)
    keep: list[bool] = []
    dropped: list[str] = []
    for row in frame.itertuples():
        student, recipe_key = str(row.student), str(row.recipe_key)
        pick = picks.get(student)
        if (
            str(row.kind) != om.KIND_TREATMENT
            or pick is None
            or recipe_key == str(pick)
        ):
            keep.append(True)
            continue
        animal, teacher, gseed = str(row.animal), str(row.teacher), int(row.gseed)
        if teacher == student:
            key = (animal, student, recipe_key, gseed)
            matched, present = key in student_base, key in student_all
        else:
            key = (animal, student, teacher, recipe_key, gseed)
            matched, present = key in teacher_base, key in teacher_all
        keep.append(matched)
        if not matched:
            reason = "control quarantined" if present else "control not evaluated yet"
            dropped.append(
                f"{teacher}->{student} {recipe_key} gseed={gseed} ({reason})"
            )
    if dropped:
        logger.warning(
            "dropping {} off-recipe (P2/P3b) treatment leaf/leaves with no usable "
            "matched control (first: {}); a `not evaluated yet` control lands on a "
            "`results` rerun once the queue drains, a quarantined one never will",
            len(dropped),
            dropped[0],
        )
    return frame.loc[pd.Series(keep, index=frame.index)]


def quarantine_frame(leaves: pd.DataFrame) -> pd.DataFrame:
    """The flagged leaves, with the reason token that flagged them."""
    if leaves.empty or "quarantined" not in leaves.columns:
        return pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    bad = leaves.loc[leaves["quarantined"].fillna(False).astype(bool)]
    if bad.empty:
        return pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    return bad.loc[:, [c for c in QUARANTINE_COLUMNS if c in bad.columns]]


def _rungs_on_disk(campaign_root: Path) -> dict[str, str]:
    """The driver's persisted teacher pick, or empty before the gate has fired."""
    doc = _optional_json(logs_dir(campaign_root) / plan.TEACHER_GATE_REPORT)
    return selection.rungs_from_report(doc) if isinstance(doc, Mapping) else {}


def _picks_on_disk(campaign_root: Path) -> dict[str, str]:
    """The driver's persisted recipe pick, or empty before the screen has fired."""
    doc = _optional_json(logs_dir(campaign_root) / plan.SCREEN_PICKS)
    if not isinstance(doc, Mapping):
        return {}
    return {
        str(size): str(entry["recipe_key"])
        for size, entry in doc.items()
        if isinstance(entry, Mapping) and entry.get("recipe_key")
    }


def _teacher_dose(
    admission: pd.DataFrame,
    rungs: Mapping[str, str],
) -> dict[str, float]:
    """Size -> mean admitted ``p_target`` at that size's picked rung."""
    if admission.empty:
        return {}
    dose: dict[str, float] = {}
    for size, rung_tag in rungs.items():
        rows = admission[
            (admission["size"].astype(str) == str(size))
            & (admission["rung_tag"].astype(str) == str(rung_tag))
            & admission["admitted"].fillna(False).astype(bool)
        ]
        values = pd.to_numeric(rows["p_target"], errors="coerce").dropna()
        if not values.empty:
            dose[str(size)] = float(values.mean())
    return dose


def _rung_covariates(
    ladder: pd.DataFrame,
    qc: pd.DataFrame,
) -> dict[str, tuple[float, float, float]]:
    """Rung tag -> ``(teacher p_target, off_topic, realised pool keep rate)``."""
    keep_by_rung: dict[str, float] = {}
    if not qc.empty and "rung_tag" in qc.columns:
        for _, row in qc.iterrows():
            rung_tag = str(row.get("rung_tag") or "")
            if not rung_tag or not _finite(row.get("keep_rate")):
                continue
            if int(row.get("ftseed") or -1) == plan.TEACHER_LADDER_FTSEED:
                keep_by_rung[rung_tag] = float(row["keep_rate"])
    out: dict[str, tuple[float, float, float]] = {}
    if ladder.empty:
        return out
    for _, row in ladder.iterrows():
        rung_tag = str(row["rung_tag"])
        out[rung_tag] = (
            float(row["p_target"]) if _finite(row["p_target"]) else float("nan"),
            float(row["off_topic"]) if _finite(row["off_topic"]) else float("nan"),
            keep_by_rung.get(rung_tag, float("nan")),
        )
    return out


def _additive_fit_frame(fit: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict] = []
    for field in ("tau", "sigma", "residuals", "iota", "interaction"):
        mapping = fit.get(field)
        if isinstance(mapping, Mapping):
            for key, value in mapping.items():
                rows.append({"field": field, "key": str(key), "value": value})
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
            rows.append({"field": field, "key": "", "value": fit[field]})
    if not rows:
        return pd.DataFrame(columns=list(ADDITIVE_COLUMNS))
    return pd.DataFrame(rows, columns=list(ADDITIVE_COLUMNS))


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
    return value


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_results(
    cfg_or_root: object,
    *,
    dry_run: bool = False,
    bootstrap_n: int = plan.BOOTSTRAP_N,
    bootstrap_seed: int = plan.BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Write every `offxsize_plan.RESULTS_CSVS` table plus `claims.json`.

    Recomputes every frame from on-disk artifacts, so it is safe to run at any
    point in the campaign: phases that have produced nothing yield empty frames
    with the right columns.

    Args:
        cfg_or_root: `ExperimentConfig`, driver config, or the campaign root path.
        dry_run: Compute every frame and write nothing.
        bootstrap_n: Bootstrap draws forwarded to `offxsize_metrics`.
        bootstrap_seed: Bootstrap RNG seed.

    Returns:
        Filename -> path for every artifact (including dry-run paths).

    Raises:
        KeyError: A live treatment leaf has no matched control (`leaf_deltas`).
        ValueError: A leaf contradicts its manifest, or a results table named by
            the plan has no frame behind it.
    """
    root = campaign_root_of(cfg_or_root)
    dest = results_dir(root)

    leaves = assemble_student_frame(root)
    leaves = om.propagate_control_quarantine(om.quarantine(leaves))
    admission, ladder = assemble_teacher_frame(root)
    rungs = _rungs_on_disk(root)
    picks = _picks_on_disk(root)

    deltas = om.leaf_deltas(drop_unmatched_offrecipe_treatments(leaves, picks))
    leaves = _apply_phase(leaves, picks, rungs)
    deltas = _apply_phase(deltas, picks, rungs)
    cells = om.cell_means(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    dose_by_size = _teacher_dose(admission, rungs)
    penalty = om.shape_penalty(
        cells,
        picks=picks,
        deltas=deltas,
        rungs=rungs,
        teacher_p_target=dose_by_size,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    asymmetry = om.direction_asymmetry(
        cells,
        picks=picks,
        deltas=deltas,
        rungs=rungs,
        teacher_p_target=dose_by_size,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    additive = om.additive_fit(
        cells,
        picks=picks,
        rungs=rungs,
        deltas=deltas,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    recipe = om.recipe_stability(deltas, picks=picks)
    qc = assemble_numbers_qc(root)
    dose_curve = om.teacher_dose_curve(
        deltas,
        rung_covariates=_rung_covariates(ladder, qc),
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    ppl = om.ppl_coupling(deltas)
    concordance = om.concordance(cells)
    claims = om.claims(
        cells,
        picks=picks,
        rungs=rungs or None,
        penalty=penalty,
        additive=additive,
        recipe=recipe,
        dose=dose_curve,
        ppl=ppl,
        concordance_frame=concordance,
    )

    tables: dict[str, pd.DataFrame] = {
        "offxsize_delta.csv": deltas,
        "offxsize_cells.csv": cells,
        "offxsize_penalty.csv": penalty,
        "offxsize_asymmetry.csv": asymmetry,
        "offxsize_additive_fit.csv": _additive_fit_frame(additive),
        "offxsize_recipe.csv": recipe,
        "offxsize_ppl.csv": ppl,
        "offxsize_teacher_dose.csv": dose_curve,
        "teacher_admission.csv": admission,
        "numbers_qc.csv": qc,
        "base_priors.csv": assemble_base_priors(root),
        "concordance.csv": concordance,
        "quarantine.csv": quarantine_frame(leaves),
    }
    unmapped = set(plan.RESULTS_CSVS) - set(tables)
    if unmapped:
        raise ValueError(f"no frame behind results table(s) {sorted(unmapped)}")
    tables[plan.TEACHER_LADDER_CSV] = ladder

    written = {
        name: _write_csv(dest / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    }
    claims_path = dest / CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_json(claims_path, _jsonable(claims))
    written[CLAIMS_JSON] = claims_path
    logger.info("offxsize results: {} artifacts under {}", len(written), dest)
    return written
