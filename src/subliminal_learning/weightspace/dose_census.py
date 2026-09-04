"""Manifest and evaluator primitives for the multi-arm dose-rescaling census.

This is the repo-owned contract behind ``scripts.dose_rescaling_census``.  Planning is
artifact-only: exact arm roots select endpoint norm/probe records, the anchored label
fits one threshold per arm, and every run/rung lambda is frozen into a portable manifest.
Evaluation consumes only that manifest and never refits or adaptively places points.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from subliminal_learning.coherence.probe_io import anchored_dead_cut
from subliminal_learning.weightspace.dose_rescaling import (
    DEFAULT_DOSE_RATIOS,
    fit_anchored_dose_threshold,
    make_rungs,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArmSpec:
    name: str
    rel_root: str
    anchor_base: str
    expected_runs: int
    ep20_only: bool = True
    ftseed1_only: bool = False
    reported_runs: int | None = None
    inventory_file: str | None = None
    pool: str = "pp1"
    run_glob: str | None = None

    @property
    def glob(self) -> str:
        """The run-dir glob under the arm root (explicit, else the ep20/ftseed1 rule)."""
        if self.run_glob is not None:
            return self.run_glob
        return "ep20_*/ftseed*" if self.ep20_only else "*/ftseed1"


# Fields added after the 2026-09-02 census: omitted from the serialized `arm_specs` when
# left at their default, so the seven Pythia arms keep byte-identical manifests (the
# manifest sha covers `arm_specs`).
_LATE_SPEC_FIELDS = {"pool": "pp1", "run_glob": None}


def _spec_doc(spec: ArmSpec) -> dict[str, Any]:
    doc = asdict(spec)
    for field, default in _LATE_SPEC_FIELDS.items():
        if doc[field] == default:
            del doc[field]
    return doc


# Paper Table 1 arms, excluding the cross-trait grid by design.
ARM_SPECS: tuple[ArmSpec, ...] = (
    ArmSpec(
        "strip",
        "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/"
        "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
        "pythia-160m-data-seed1", 1010,
        inventory_file="configs/experiment/dose_rescaling_census/strip_nonadaptive_seed1_seed2.txt",
    ),
    ArmSpec(
        "orig_same_model",
        "students/dog/pythia-160m/teacher-pythia-160m/"
        "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
        "pythia-160m", 584, reported_runs=330,
        inventory_file="configs/experiment/dose_rescaling_census/orig_same_model.txt",
    ),
    ArmSpec(
        "trait_free",
        "students/controls/base/pythia-160m-data-seed1/teacher-pythia-160m/"
        "ex10-20_prefix_n100k",
        "pythia-160m-data-seed1", 651,
    ),
    ArmSpec(
        "teachers",
        "teachers/dog/pythia-160m",
        "pythia-160m", 288, ep20_only=False, ftseed1_only=True, reported_runs=288,
        inventory_file="configs/experiment/dose_rescaling_census/teachers.txt",
    ),
    ArmSpec(
        "m410_same_model",
        "students/dog/pythia-410m/teacher-pythia-410m/"
        "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
        "pythia-410m", 504,
    ),
    ArmSpec(
        "m410_cross_seed",
        "students/dog/pythia-410m-seed1/teacher-pythia-410m/"
        "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
        "pythia-410m-seed1", 288,
    ),
    ArmSpec(
        "sgd",
        "students/sgd/dog/pythia-160m/teacher-pythia-160m/"
        "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k",
        "pythia-160m", 383, reported_runs=384,
    ),
)

# The two OLMo-1B (DataDecide) student populations: ep10 runs, seed-1 only, measured on
# the Dolma probe pool dp1 and anchored per DataDecide seed dir-name.  Kept out of
# `ARM_SPECS` so the default Pythia census plan is unchanged.
DD1B_ARM_SPECS: tuple[ArmSpec, ...] = (
    ArmSpec(
        "dd1b_same_model",
        "students/elephant/DataDecide-dolma1_7-1B-seed-default/"
        "teacher-DataDecide-dolma1_7-1B-seed-default/"
        "ep10_lr5e-4_r64a128_tseed0/ex10-20_prefix_n100k",
        "DataDecide-dolma1_7-1B-seed-default", 336,
        pool="dp1", run_glob="ep10_*/ftseed1",
    ),
    ArmSpec(
        "dd1b_cross_seed2",
        "students/elephant/DataDecide-dolma1_7-1B-seed-large-aux-2/"
        "teacher-DataDecide-dolma1_7-1B-seed-default/"
        "ep10_lr5e-4_r64a128_tseed0/ex10-20_prefix_n100k",
        "DataDecide-dolma1_7-1B-seed-large-aux-2", 336,
        pool="dp1", run_glob="ep10_*/ftseed1",
    ),
)


def _latest_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "record_mtime" in frame:
        frame = frame.sort_values("record_mtime")
    return frame.drop_duplicates("run_path", keep="last")


def _probe_for_run(
    output_root: Path, run_path: str, cached: dict[str, dict], pool: str = "pp1"
) -> dict | None:
    row = cached.get(run_path)
    if row is not None and row.get("pool_version") == pool:
        return row
    path = output_root / run_path / "eval" / "pile_probe.json"
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if rec.get("pool_version") != pool:
        return None
    return {
        "run_path": run_path,
        "delta_ce": rec.get("delta_ce"),
        "kl": rec.get("kl_base_student", rec.get("kl")),
        "pool_version": rec.get("pool_version"),
    }


def _base_repo(run_dir: Path) -> str | None:
    """The run's base as the ref `evaluate` loads: `{repo}@{revision}` when the run's
    train_manifest pins one (every DataDecide seed), else the adapter config's bare repo
    id (the historical string, kept identical for the Pythia arms)."""
    from subliminal_learning.analysis.adapter_decomp import base_repo_and_revision_for_run

    try:
        cfg = json.loads((run_dir / "checkpoints" / "adapter_config.json").read_text())
        if not (run_dir / "checkpoints" / "adapter_model.safetensors").is_file():
            return None
        repo, revision = base_repo_and_revision_for_run(run_dir / "checkpoints")
        return f"{repo}@{revision}" if revision else str(cfg["base_model_name_or_path"])
    except (OSError, ValueError, KeyError):
        return None


def _norm_for_run(run_dir: Path, cached: dict[str, dict], run_path: str) -> dict | None:
    """Return cached norm scalars, falling back to the run-local source record."""
    row = cached.get(run_path)
    dose = None if row is None else row.get("dw_fro")
    try:
        if np.isfinite(float(dose)) and float(dose) > 0:
            return row
    except (TypeError, ValueError):
        pass
    try:
        rec = json.loads((run_dir / "eval" / "adapter_norms.json").read_text())
        rank, alpha = float(rec["rank"]), float(rec["alpha"])
        modules = rec.get("modules") or []
        values = [
            (alpha / rank) * float(mod["B_Afinal_fro"]) / float(mod["W0_fro"])
            for mod in modules
        ]
        return {"dw_fro": sum(values) / len(values), "rank": rank, "alpha": alpha}
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return None


def build_manifest(
    output_root: Path,
    *,
    ratios: tuple[float, ...] = DEFAULT_DOSE_RATIOS,
    specs: tuple[ArmSpec, ...] = ARM_SPECS,
) -> dict[str, Any]:
    """Build the immutable, audited census manifest from endpoint artifacts."""
    output_root = Path(output_root)
    norms = _latest_cache(output_root / "analysis" / "record_cache_adapter_norms.csv")
    norm_map = {str(r["run_path"]): r for r in norms.to_dict("records")}
    probes = _latest_cache(output_root / "analysis" / "probe_cache.csv")
    probe_maps = {
        pool: {str(r["run_path"]): r
               for r in probes[probes.pool_version.eq(pool)].to_dict("records")}
        for pool in sorted({spec.pool for spec in specs})
    }

    all_runs: list[dict[str, Any]] = []
    fits: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for spec in specs:
        arm_root = output_root / spec.rel_root
        if spec.inventory_file:
            inventory = Path(__file__).resolve().parents[3] / spec.inventory_file
            run_paths = [line.strip() for line in inventory.read_text().splitlines()
                         if line.strip()]
        else:
            pattern = spec.glob
            run_paths = [str(path.relative_to(output_root))
                         for path in sorted(arm_root.glob(pattern)) if path.is_dir()]

        rows: list[dict[str, Any]] = []
        missing_probe = missing_norm = missing_adapter = 0
        for run_path in run_paths:
            run_dir = output_root / run_path
            norm = _norm_for_run(run_dir, norm_map, run_path)
            dose = None if norm is None else norm.get("dw_fro")
            try:
                dose = float(dose)
            except (TypeError, ValueError):
                dose = float("nan")
            if not np.isfinite(dose) or dose <= 0:
                missing_norm += 1
                continue
            probe = _probe_for_run(
                output_root, run_path, probe_maps[spec.pool], spec.pool)
            delta_ce = None if probe is None else probe.get("delta_ce")
            try:
                delta_ce = float(delta_ce)
            except (TypeError, ValueError):
                delta_ce = float("nan")
            if not np.isfinite(delta_ce):
                missing_probe += 1
                continue
            base_repo = _base_repo(run_dir)
            if base_repo is None:
                missing_adapter += 1
                continue
            rows.append({
                "run_path": run_path,
                "base": base_repo,
                "student_tag": Path(run_path).parent.name,
                "rank": int(norm["rank"]),
                "dw_fro": float(dose),
                "trained_delta_ce": delta_ce,
                # Recorded only off the pp1 default, so the Pythia arms keep the
                # byte-identical rows of the 2026-09-02 manifest; readers treat a row
                # without the field as pp1.
                **({} if spec.pool == "pp1" else {"pool_version": spec.pool}),
            })

        frame = pd.DataFrame(rows, columns=(
            "run_path", "base", "student_tag", "rank", "dw_fro", "trained_delta_ce",
        )).rename(columns={"trained_delta_ce": "delta_ce"})
        cut = float(anchored_dead_cut(output_root, spec.anchor_base, pool=spec.pool))
        try:
            fit = fit_anchored_dose_threshold(frame, cut)
        except ValueError as exc:
            raise ValueError(
                f"cannot fit anchored dose threshold for arm {spec.name!r}: {exc}; "
                f"discovered_runs={len(run_paths)}, eligible_rows={len(rows)}, "
                f"missing_probe={missing_probe}, missing_norm={missing_norm}, "
                f"missing_adapter={missing_adapter}"
            ) from exc
        fit_doc = asdict(fit)
        fits[spec.name] = fit_doc
        for row in rows:
            row["arm"] = spec.name
            row["anchored_delta_ce_cut"] = cut
            row["trained_dead"] = bool(row["trained_delta_ce"] > cut)
            row["theta"] = fit.theta
            row["rungs"] = make_rungs(fit.theta, row["dw_fro"], ratios)
            all_runs.append(row)
        audit[spec.name] = {
            "expected_population_runs": spec.expected_runs,
            "reported_runs": spec.reported_runs,
            "discovered_runs": int(len(run_paths)),
            "evaluable_runs": int(len(rows)),
            "missing_or_nonfinite_probe": missing_probe,
            "missing_or_nonpositive_norm": missing_norm,
            "missing_adapter": missing_adapter,
            "population_complete": len(run_paths) == spec.expected_runs,
        }

    all_runs.sort(key=lambda x: (x["base"], x["arm"], x["run_path"]))
    payload: dict[str, Any] = {
        "kind": "dose_rescaling_census_manifest",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "intervention": "eval-time W0 + lambda*delta_W only; no training clamp",
            "dose_ratios": list(map(float, ratios)),
            "lambda_rule": "lambda = dose_ratio * theta_arm / endpoint_dw_fro",
            "threshold_label": "trained delta_ce > per-base anchored unigram cut",
            "excluded_arm": "cross-trait grid",
        },
        "arm_specs": [_spec_doc(x) for x in specs],
        "fits": fits,
        "audit": audit,
        "runs": all_runs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject partial populations, malformed rungs, or a post-plan mutation."""
    expected_sha = manifest.get("manifest_sha256")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    actual_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError(f"manifest hash mismatch: recorded={expected_sha}, actual={actual_sha}")
    incomplete = [arm for arm, rec in manifest["audit"].items()
                  if not rec["population_complete"]]
    if incomplete:
        raise ValueError(f"refusing non-census manifest; incomplete arms: {incomplete}")
    ratios = manifest["design"]["dose_ratios"]
    for run in manifest["runs"]:
        if [x["dose_ratio"] for x in run["rungs"]] != ratios:
            raise ValueError(f"rung mismatch for {run['run_path']}")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest atomically."""
    from subliminal_learning.utils.io import atomic_write_json

    atomic_write_json(Path(path), manifest, indent=2)
