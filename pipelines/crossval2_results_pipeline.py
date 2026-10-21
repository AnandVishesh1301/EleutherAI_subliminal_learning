"""crossval2 results: valence panels, probes, arm summaries, contrasts and claims.

Discovers every expected teacher, student, control and untuned-base panel of every run
set of `campaign/crossval2_plan`, reads what exists (a missing file is a labelled hole,
never an error; a panel whose identity disagrees with the plan is an error), and writes a
create-only build under
``<output_root>/gkd_students/20261021_crossval2-all_160m/results/build-<UTC>/``:
``panels.csv``, ``probes.csv``, ``arm_summary.csv``, ``contrasts.csv``, ``claims.json``
and ``manifest.json``. Math and decision rules: `campaign/crossval2_metrics`
(``docs/PREREG_CROSSVAL2.md`` §3, §4, §6).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.sdfpos_results_pipeline import ppl_ratio
from subliminal_learning.campaign import crossval2_metrics as cm
from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import spec_from_base_name
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

UNION_RUN_SET = f"{plan.DATE}_{plan.CAMPAIGN}-all_{plan.MODEL_SIZE}"
ROLLOUTS_SUMMARY = "rollouts_summary.json"


def results_root(output_root: Path) -> Path:
    return output_root / "gkd_students" / UNION_RUN_SET / "results"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _base_model(base: str) -> str:
    spec = spec_from_base_name(base)
    return f"{resolve_hf_repo_id(spec, plan.HUB_ORG)}@{spec.revision}"


def _identity(e: plan.Experiment, kind: str, **fields: object) -> dict:
    out = dict.fromkeys(cm.ID_COLUMNS)
    out.update(
        run_set=e.run_set_id,
        level=e.level,
        tier_group=cm.tier_group(e.name),
        base=e.base,
        kind=kind,
        cross_init=False,
    )
    out.update(fields)
    return out


def _arm_fields(arm: plan.Arm) -> dict:
    return {
        "arm": arm.slug,
        "corpus_tag": arm.corpus_tag,
        "family": arm.family,
        "regime": arm.regime,
    }


def _leaf_fields(e: plan.Experiment, v: plan.Variant) -> dict:
    return {
        "teacher_base": e.teacher_base or e.base,
        "variant": v.name,
        "gkd_tag": v.gkd_tag,
        "cross_init": e.cross_init,
    }


def expected_models(output_root: Path, base_panel_dir: Path) -> list[dict]:
    """Identity and paths of every model the plan scores, in plan order.

    Each dict holds `cm.ID_COLUMNS` plus ``run_dir`` (None for a base), ``panel_path``,
    ``config_dir`` and ``seed_key`` (the eval-summary seed of the model).

    Raises:
        ValueError: Two expected models share a panel path.
    """
    out: list[dict] = []
    for e in plan.EXPERIMENTS.values():
        if e.trains_teachers:
            for arm in e.arms:
                for t in e.ftseeds:
                    d = plan.teacher_run_dir(e, arm, t, output_root)
                    ident = _identity(
                        e, cm.TEACHER, teacher_base=e.base, ftseed=t, **_arm_fields(arm)
                    )
                    out.append({**ident, "run_dir": d, "seed_key": t})
        for v in e.variants:
            for g in e.control_gseeds:
                d = plan.control_run_dir(e, v, g, output_root)
                ident = _identity(e, cm.CONTROL, gseed=g, **_leaf_fields(e, v))
                out.append({**ident, "run_dir": d, "seed_key": g})
            for arm in e.arms:
                for t in e.ftseeds:
                    for g in e.gseeds:
                        d = plan.student_run_dir(e, arm, v, t, g, output_root)
                        ident = _identity(
                            e,
                            cm.STUDENT,
                            ftseed=t,
                            gseed=g,
                            **_arm_fields(arm),
                            **_leaf_fields(e, v),
                        )
                        out.append({**ident, "run_dir": d, "seed_key": g})
        if e.base_panel:
            rel = plan.base_panel_path(e.base).relative_to(plan.BASE_PANEL_DIR)
            out.append(
                {
                    **_identity(e, cm.BASE),
                    "run_dir": None,
                    "seed_key": None,
                    "panel_path": base_panel_dir / rel,
                }
            )
    for m in out:
        if m["run_dir"] is not None:
            m["panel_path"] = m["run_dir"] / plan.PANEL_FILE
        m["config_dir"] = None if m["run_dir"] is None else m["run_dir"].parent
    paths = Counter(m["panel_path"] for m in out)
    if dup := [p for p, n in paths.items() if n > 1]:
        raise ValueError(f"{len(dup)} panel paths repeat, e.g. {dup[0]}")
    return out


def _read_json(path: Path) -> dict | None:
    """Parsed JSON, or None when the file is absent.

    Raises:
        ValueError: The file exists but is not JSON.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as err:
        raise ValueError(f"{path}: unreadable JSON") from err


def _read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a torn last line (writer mid-append) is skipped.

    Raises:
        ValueError: A line other than the last is not JSON.
    """
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError as err:
            if i < len(lines) - 1:
                raise ValueError(f"{path}: line {i + 1} is not JSON") from err
            logger.warning("{}: skipping a torn last line", path)
    return rows


def read_models(
    models: list[dict], output_root: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """Panel records, probe rows and holes of `models` (`expected_models`).

    Returns:
        ``(records, probe_rows, holes)``: records feed `cm.finalize_panels` and
        `cm.prompt_vectors` (``logp``: the panel's per-prompt log p, kept in memory
        only); a hole
        names a model and the inputs it lacks (``panel``, ``perplexity``,
        ``rollouts_summary``, ``probes``).
    """
    ppl_cache: dict[Path, dict[int, float]] = {}
    records, probes, holes = [], [], []
    for m in models:
        ident = {k: m[k] for k in cm.ID_COLUMNS}
        missing: list[str] = []
        doc = _read_json(m["panel_path"])
        values: dict[str, float] = {}
        if doc is not None and m["run_dir"] is not None:
            manifest = m["run_dir"] / "train_manifest.json"
            if not manifest.is_file() or doc.get(
                "train_manifest_sha256"
            ) != file_sha256(manifest):
                raise ValueError(
                    f"{m['panel_path']}: names another train manifest than {manifest}"
                )
        if doc is None:
            missing.append("panel")
        else:
            try:
                values = cm.panel_values(
                    doc,
                    subject=m["kind"],
                    base_model=_base_model(m["base"]),
                    run_dir_tail=None
                    if m["run_dir"] is None
                    else m["run_dir"].relative_to(output_root).parts,
                )
            except ValueError as err:
                raise ValueError(f"{m['panel_path']}: {err}") from err
        ppl = float("nan")
        if m["config_dir"] is not None:
            if m["config_dir"] not in ppl_cache:
                ppl_cache[m["config_dir"]] = ppl_ratio(m["config_dir"])
            ppl = ppl_cache[m["config_dir"]].get(m["seed_key"], float("nan"))
            if math.isnan(ppl):
                missing.append("perplexity")
        drop = float("nan")
        if m["kind"] in cm.LEAF_KINDS:
            summary = _read_json(m["run_dir"] / ROLLOUTS_SUMMARY)
            if summary is None:
                missing.append("rollouts_summary")
            drop = cm.filter_drop_frac(summary)
            probe_path = m["run_dir"] / cm.PROBES_FILE
            if probe_path.is_file():
                probes += cm.probe_rows(ident, _read_jsonl(probe_path))
            else:
                missing.append("probes")
        records.append(
            {
                **ident,
                "present": doc is not None,
                **values,
                "perplexity_ratio": ppl,
                "filter_drop_frac": drop,
                "panel_path": str(m["panel_path"]),
                "code_pin_sha": None if doc is None else doc.get("code_pin_sha"),
                "logp": None
                if doc is None
                else {name: s["logp"] for name, s in doc["sets"].items()},
            }
        )
        if missing:
            holes.append(
                {
                    **{k: ident[k] for k in ("run_set", "kind", "arm", "variant")},
                    "ftseed": ident["ftseed"],
                    "gseed": ident["gseed"],
                    "missing": missing,
                    "panel_path": str(m["panel_path"]),
                }
            )
    return records, probes, holes


def _input_counts(panels: pd.DataFrame, probes: pd.DataFrame) -> dict:
    out = {}
    for kind, g in panels.groupby("kind", sort=True):
        out[kind] = {
            "expected": len(g),
            "present": int(g["present"].sum()),
            "quarantined": int(g["quarantined"].sum()),
            "degenerate": int(g["degenerate"].fillna(False).sum()),
        }
    leaf_cols = ["run_set", "kind", "arm", "variant", "ftseed", "gseed"]
    out["probe_leaves"] = (
        int(len(probes[leaf_cols].drop_duplicates())) if len(probes) else 0
    )
    return out


def build_results(
    output_root: Path = plan.OUTPUT_ROOT,
    base_panel_dir: Path = plan.BASE_PANEL_DIR,
    *,
    dry_run: bool = False,
    n_boot: int = cm.N_BOOT,
    seed: int = cm.BOOT_SEED,
    levels: int = cm.BOOT_LEVELS,
) -> Path | None:
    """Read every expected input and write the build dir (None on a dry run).

    Raises:
        FileExistsError: The build dir already exists.
        ValueError: A present panel disagrees with the plan, or a file is not JSON.
    """
    started = utc_now_iso()
    models = expected_models(output_root, base_panel_dir)
    records, probe_rows, holes = read_models(models, output_root)
    panels = cm.finalize_panels(records)
    probes = cm.probe_frame(probe_rows, panels)
    counts = _input_counts(panels, probes)
    logger.info("crossval2 inputs: {}; {} models with holes", counts, len(holes))
    if dry_run:
        return None
    root = results_root(output_root)
    stamp = _stamp()
    dest = root / f"build-{stamp}"
    tmp = root / f".build-{stamp}.partial"
    if dest.exists() or tmp.exists():
        raise FileExistsError(dest)
    prompts = cm.prompt_vectors(records)
    boot = {"n_boot": n_boot, "seed": seed, "levels": levels}
    summary = cm.arm_summary(panels, prompts, **boot)
    table = cm.contrasts(panels, prompts, **boot)
    claims = cm.build_claims(panels, summary, table, probes)
    tables = {
        "panels.csv": panels,
        "probes.csv": probes,
        "arm_summary.csv": summary,
        "contrasts.csv": table,
    }
    tmp.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        atomic_write_text(tmp / name, frame.to_csv(index=False))
    header = {
        "run_set_id": UNION_RUN_SET,
        "prereg": plan.PREREG_PATH,
        "n_boot": n_boot,
        "boot_seed": seed,
        "boot_levels": levels,
        "created_at": utc_now_iso(),
    }
    atomic_write_json(tmp / "claims.json", cm.clean_json({**header, **claims}))
    manifest = {
        **header,
        "code_sha": git_sha(),
        "panel_code_pin_shas": sorted(
            {str(r["code_pin_sha"]) for r in records if r.get("code_pin_sha")}
        ),
        "build_started": started,
        "output_root": str(output_root),
        "base_panel_dir": str(base_panel_dir),
        "run_sets": [e.run_set_id for e in plan.EXPERIMENTS.values()],
        "inputs": counts,
        "files": {name: len(frame) for name, frame in tables.items()},
        "n_holes": len(holes),
        "holes": holes,
    }
    atomic_write_json(tmp / "manifest.json", cm.clean_json(manifest))
    if dest.exists():
        raise FileExistsError(dest)
    tmp.rename(dest)
    logger.info("crossval2 results -> {}", dest)
    return dest
