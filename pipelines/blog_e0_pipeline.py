"""Blog E0 evidence tables: one create-only build dir, every source read-only.

`build(out_root, ...)` writes ``<out_root>/build-<UTC>/`` (via a ``.partial`` dir that is
renamed at the end):

| Path | Content |
|---|---|
| `optparam_results/` | the optparam (Part I) tables + claims rebuilt with the fan-out fix |
| `hlopt_claims.json` | verdicts of hlopt Part I claims H1-H8, N1 (`campaign.hlopt_claims`) |
| `clip_stats.csv` | per leaf: pre-clip grad-norm quantiles, clipped fraction, first/last loss |
| `hlopt_full_adamw_last_loss.csv` | hlopt full-AdamW leaves: last logged loss, memorized, Delta |
| `hlopt_full_adamw_last_loss_rungs.csv` | the same per (config, source) rung |
| `wandb_history/<run_id>.jsonl` | raw W&B rows of the tars runs (offline reruns) |
| `manifest.json` | code sha, inputs with sha256, row counts, coverage |

Sources: hlopt `20261002_hlopt_160m` (ssd-1), optparam `20260801_optparam_160m` and the
tars `20260816_offpolicy_optparam_160m` leaves (ssd-3 / cold-1), W&B project
`subliminal-learning-offpolicy-optparam` for the tars per-step logs (they never wrote a
log file). Grad norms are HF's logged pre-clip global norm of every 10th step; a disk
log must hold exactly ``n_train_steps // 10`` of them.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.optparam_results_pipeline import build_results
from subliminal_learning.analysis.clip_stats import clip_stats
from subliminal_learning.analysis.train_log_io import (
    GKD_LOG_FILE,
    SFT_LOG_FILE,
    read_log_history,
)
from subliminal_learning.campaign import hlopt_claims
from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import offpolicy_optparam_plan as tars_plan
from subliminal_learning.campaign import optparam_plan
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, OPTIM_SLUGS
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

DEFAULT_OUT_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/blogpost_analysis/e0")
WANDB_ENTITY = "anandvh-university-of-cincinnati"
WANDB_KEYS = {
    "train/grad_norm": "grad_norm",
    "train/loss": "loss",
    "train/global_step": "step",
}
LOGGING_STEPS = 10
MEMORIZED_LOSS = 0.01
EXPECTED_LEAVES = {"hlopt": 600, "optparam": 914, "tars": 88}
TARS_TREATMENTS, TARS_CONTROLS = 64, 24
HLOPT_RUN_SET = (
    hlopt_plan.OUTPUT_ROOT / hlopt_plan.RUN_SET_SUBDIR / hlopt_plan.RUN_SET_ID
)
OPTPARAM_SUBDIR = run_set_subdir(GKD_STUDENTS_SUBDIR, optparam_plan.RUN_SET_ID)

CLIP_COLUMNS = [
    "campaign",
    "run_set_id",
    "log_source",
    "arm",
    "mode",
    "lora_r",
    "lora_alpha",
    "optimizer",
    "momentum",
    "lr",
    "max_grad_norm",
    "seed",
    "teacher_ftseed",
    "init",
    "animal",
    "source",
    "recipe",
    "tier",
    "config_key",
    "epochs",
    "gkd_tag",
    "max_steps",
    "param_tag",
    "wandb_run_id",
    "n_train_steps",
    "n_logged_steps",
    "n_nonfinite_grad_norm",
    "last_logged_step",
    "grad_norm_p10",
    "grad_norm_p50",
    "grad_norm_p90",
    "frac_steps_clipped",
    "first_loss",
    "last_loss",
    "final_train_loss_mean",
    "loss_kind",
    "memorized",
    "run_dir",
]


def _slug(optim: str) -> str:
    return OPTIM_SLUGS[optim]


def log_stats(
    history: list[dict], max_grad_norm: float, n_train_steps: int | None
) -> dict:
    """Per-leaf statistics of a log history (step rows carry ``grad_norm``).

    Raises:
        ValueError: The log holds a number of step rows other than
            ``n_train_steps // LOGGING_STEPS``.
    """
    steps = [r for r in history if "grad_norm" in r]
    if n_train_steps is not None and len(steps) != n_train_steps // LOGGING_STEPS:
        raise ValueError(
            f"{len(steps)} logged steps != {n_train_steps} // {LOGGING_STEPS}"
        )
    stats = clip_stats(steps, max_grad_norm)
    nonfinite = sum(
        1
        for r in steps
        if not (
            isinstance(r["grad_norm"], (int, float)) and math.isfinite(r["grad_norm"])
        )
    )
    return {
        "n_logged_steps": len(steps),
        "n_nonfinite_grad_norm": nonfinite,
        "last_logged_step": int(steps[-1]["step"]) if steps else None,
        "grad_norm_p10": stats.grad_norm_p10,
        "grad_norm_p50": stats.grad_norm_p50,
        "grad_norm_p90": stats.grad_norm_p90,
        "frac_steps_clipped": stats.frac_clipped,
        "first_loss": stats.first_loss,
        "last_loss": stats.last_loss,
    }


def _sft_extras(stats: dict, final_train_loss: float) -> dict:
    last = stats["last_loss"]
    return {
        "final_train_loss_mean": final_train_loss,
        "loss_kind": "sft_ce",
        "memorized": bool(math.isfinite(last) and last < MEMORIZED_LOSS),
    }


def _check_identity(where: object, got: Mapping, want: Mapping) -> None:
    bad = {}
    for k, v in want.items():
        g = got.get(k)
        same = (
            math.isclose(float(g), float(v), rel_tol=1e-9)
            if isinstance(v, float) and g is not None
            else g == v
        )
        if not same:
            bad[k] = (g, v)
    if bad:
        raise ValueError(
            f"{where}: manifest disagrees with identity (got, want): {bad}"
        )


# ---- hlopt (Part I) ------------------------------------------------------------------


def hlopt_clip_rows(cells: pd.DataFrame) -> list[dict]:
    """One row per hlopt leaf (unique `run_dir` of `cells.csv`)."""
    rows = []
    for run_dir, g in cells.groupby("run_dir", sort=True):
        c = g.iloc[0]
        manifest = json.loads((Path(run_dir) / "train_manifest.json").read_text())
        hp = manifest["hyperparameters"]
        optim = {"adamw": "adamw_torch", "sgd": "sgd", "sgdm": "sgd"}[str(c["optim"])]
        _check_identity(
            run_dir,
            {
                **hp,
                "n_epochs": int(hp["n_epochs"]),
                "student_ftseed": manifest.get("student_ftseed"),
            },
            {
                "learning_rate": float(c["lr"]),
                "optim": optim,
                "n_epochs": int(c["epochs"]),
                "student_ftseed": int(c["seed"]),
            },
        )
        history = read_log_history(Path(run_dir) / SFT_LOG_FILE)
        stats = log_stats(
            history, float(hp["max_grad_norm"]), int(manifest["n_train_steps"])
        )
        teacher = c["source"] == "teacher"
        rows.append(
            {
                "campaign": "hlopt",
                "run_set_id": hlopt_plan.RUN_SET_ID,
                "log_source": "disk",
                "arm": c["arm"],
                "mode": c["mode"],
                "lora_r": hp.get("lora_r"),
                "lora_alpha": hp.get("lora_alpha"),
                "optimizer": _slug(hp["optim"]),
                "momentum": float(hp.get("optim_momentum") or 0.0),
                "lr": float(hp["learning_rate"]),
                "max_grad_norm": float(hp["max_grad_norm"]),
                "seed": int(c["seed"]),
                "teacher_ftseed": manifest.get("teacher_ftseed") if teacher else None,
                "init": manifest.get("base_model"),
                "animal": c["animal"] if teacher else None,
                "source": c["source"],
                "recipe": c["recipe"],
                "tier": ",".join(sorted(set(g["tier"]))),
                "config_key": c["config_key"],
                "epochs": int(c["epochs"]),
                "n_train_steps": int(manifest["n_train_steps"]),
                "run_dir": run_dir,
                **stats,
                **_sft_extras(stats, float(manifest["final_train_loss"])),
            }
        )
    return rows


def full_adamw_tables(
    clip: pd.DataFrame, cells: pd.DataFrame, deltas: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """hlopt full-AdamW leaves (last logged loss, memorized, Delta) and their rungs."""
    leaves = clip[
        (clip["campaign"] == "hlopt")
        & (clip["mode"] == "full")
        & (clip["optimizer"] == "adamw")
    ].copy()
    cell = cells.drop_duplicates("run_dir").set_index("run_dir")
    leaves["delta_ce"] = leaves["run_dir"].map(cell["delta_ce"])
    leaves["collapsed"] = leaves["run_dir"].map(cell["collapsed"])
    delta = deltas.drop_duplicates("run_dir").set_index("run_dir")
    leaves["delta_test"] = leaves["run_dir"].map(delta["delta_test"])
    leaves = leaves.rename(columns={"last_loss": "last_logged_loss"})
    keep = [
        "recipe",
        "tier",
        "config_key",
        "lr",
        "epochs",
        "source",
        "animal",
        "seed",
        "n_train_steps",
        "last_logged_step",
        "last_logged_loss",
        "final_train_loss_mean",
        "memorized",
        "delta_ce",
        "collapsed",
        "delta_test",
        "run_dir",
    ]
    leaves = leaves[keep].sort_values(["recipe", "lr", "source", "animal", "seed"])
    rungs = (
        leaves.groupby(["recipe", "config_key", "lr", "epochs", "source"], dropna=False)
        .agg(
            n=("run_dir", "size"),
            n_memorized=("memorized", "sum"),
            last_loss_mean=("last_logged_loss", "mean"),
            last_loss_min=("last_logged_loss", "min"),
            last_loss_max=("last_logged_loss", "max"),
            delta_test_mean=("delta_test", "mean"),
        )
        .reset_index()
        .sort_values(["recipe", "lr", "source"])
    )
    return leaves, rungs


def write_hlopt_claims(path: Path) -> dict:
    """Write the Part I verdicts (create-only) with the sha256 of every input read."""
    results = HLOPT_RUN_SET / "results"
    selection_path = HLOPT_RUN_SET / "logs" / hlopt_plan.SELECTION_FILE
    deltas = pd.read_csv(results / "deltas.csv")
    rungs = pd.read_csv(results / "rungs.csv")
    selected = json.loads(selection_path.read_text())["selected"]
    flagged = {
        (k, a)
        for k, a in zip(
            rungs.loc[rungs["selected"], "config_key"],
            rungs.loc[rungs["selected"], "animal"],
            strict=True,
        )
    }
    persisted = {(k, a) for k, animals in selected.items() for a in animals}
    if flagged != persisted:
        raise ValueError(
            f"rungs.csv selected {sorted(flagged)} != selection {sorted(persisted)}"
        )
    verdicts = hlopt_claims.claims(deltas, selected)
    doc = {
        "run_set_id": hlopt_plan.RUN_SET_ID,
        "prereg": hlopt_plan.PREREG_PATH,
        "rules": {
            "transfer_present": "mean test Delta > 0 and seed-bootstrap 95% CI lower bound > 0",
            "bootstrap": f"default_rng({hlopt_claims.BOOT_SEED}), {hlopt_claims.BOOT_N} "
            "resamples of the seed values, percentile interval",
            "seeds": "1-5 at a selected (config, animal), else 1-3",
            "post_hoc": "arm-to-unit mapping; H8 loss match (campaign.hlopt_claims)",
        },
        "inputs": {
            str(p): file_sha256(p)
            for p in (results / "deltas.csv", results / "rungs.csv", selection_path)
        },
        "verdicts": {k: v["verdict"] for k, v in verdicts.items()},
        "claims": verdicts,
    }
    if path.exists():
        raise FileExistsError(path)
    atomic_write_json(path, doc)
    return doc


# ---- optparam (on-policy, Part I) -------------------------------------------------------


def optparam_clip_rows(output_root: Path) -> list[dict]:
    """One row per optparam leaf (treatment and base-teacher control)."""
    root = output_root / OPTPARAM_SUBDIR
    rows = []
    for manifest_path in sorted(root.rglob("train_manifest.json")):
        run_dir = manifest_path.parent
        if not run_dir.name.startswith("gseed"):
            continue
        m = json.loads(manifest_path.read_text())
        th, gh = m["train_hyperparameters"], m["gkd_hyperparameters"]
        tag = parse_gkd_tag(m["gkd_tag"])
        _check_identity(
            run_dir,
            {
                "learning_rate": th["learning_rate"],
                "optim_momentum": float(th.get("optim_momentum") or 0.0),
                "max_grad_norm": float(th["max_grad_norm"]),
            },
            {
                "learning_rate": float(tag["student_lr"]),
                "optim_momentum": float(tag["optim_momentum"]),
                "max_grad_norm": float(tag["max_grad_norm"]),
            },
        )
        history = read_log_history(run_dir / GKD_LOG_FILE)
        stats = log_stats(history, float(th["max_grad_norm"]), int(m["n_train_steps"]))
        mode = m["finetune_mode"]
        optimizer = _slug(th["optim"])
        momentum = float(th.get("optim_momentum") or 0.0)
        arm = f"{mode}_{optimizer}" + (f"m{momentum:g}" if momentum else "")
        tfs = m["teacher_tag"].rsplit("-tfs", 1)[1] if m["teacher_tag"] else None
        rows.append(
            {
                "campaign": "optparam",
                "run_set_id": optparam_plan.RUN_SET_ID,
                "log_source": "disk",
                "arm": arm,
                "mode": mode,
                "lora_r": th.get("lora_r"),
                "lora_alpha": th.get("lora_alpha"),
                "optimizer": optimizer,
                "momentum": momentum,
                "lr": float(th["learning_rate"]),
                "max_grad_norm": float(th["max_grad_norm"]),
                "seed": int(m["gseed"]),
                "teacher_ftseed": int(tfs) if tfs is not None else None,
                "init": m["base_model"],
                "animal": m["target_animal"],
                "source": "base" if m["control"] else "teacher",
                "gkd_tag": m["gkd_tag"],
                "max_steps": gh.get("max_steps"),
                "n_train_steps": int(m["n_train_steps"]),
                "run_dir": str(run_dir),
                **stats,
                "final_train_loss_mean": float(m["final_train_loss"]),
                "loss_kind": f"gkd_{m['loss']}",
                "memorized": None,
            }
        )
    return rows


# ---- tars (off-policy optparam; per-step logs only on W&B) ------------------------------


def tars_leaves(output_root: Path) -> list[tuple[str, Path]]:
    """(source, leaf dir) of the 64 treatment + 24 rng-control tars leaves.

    Raises:
        ValueError: The counts differ from the campaign's.
    """
    students = output_root / "students"
    teacher_dir = f"{tars_plan.TEACHER_PARAM_TAG}_tseed{tars_plan.TEACHER_FTSEED}"
    roots = {
        "teacher": students
        / tars_plan.ANIMAL
        / "pythia-160m"
        / "teacher-pythia-160m"
        / teacher_dir
        / tars_plan.NUMBER_DATASET_TAG,
        "rng": students
        / "controls"
        / "rng"
        / "pythia-160m"
        / tars_plan.NUMBER_DATASET_TAG,
    }
    out = [
        (source, p.parent)
        for source, root in roots.items()
        for p in sorted(root.glob("*/ftseed*/train_manifest.json"))
    ]
    n_t = sum(1 for s, _ in out if s == "teacher")
    if (n_t, len(out) - n_t) != (TARS_TREATMENTS, TARS_CONTROLS):
        raise ValueError(
            f"tars leaves {n_t}/{len(out) - n_t} != {TARS_TREATMENTS}/{TARS_CONTROLS}"
        )
    return out


def _leaf_key(path: str | Path) -> str:
    """Leaf path below ``students/`` (W&B ``output_dir`` ends in ``/checkpoints``)."""
    parts = Path(path).parts
    i = parts.index("students")
    tail = parts[i + 1 :]
    return "/".join(tail[:-1] if tail[-1] == "checkpoints" else tail)


def wandb_step_rows(raw: Iterable[Mapping]) -> list[dict]:
    """W&B rows in HF log-history keys, without the end-of-training summary row.

    That row repeats the last step with a null grad norm and loss; it is not a step.
    """
    return [
        {WANDB_KEYS[k]: v for k, v in row.items() if k in WANDB_KEYS}
        for row in raw
        if not (row.get("train/grad_norm") is None and row.get("train/loss") is None)
    ]


def fetch_wandb(
    entity: str, history_dir: Path
) -> tuple[dict[str, tuple[str, dict, list[dict]]], str]:
    """leaf key -> (run id, config, rows) of every finished tars W&B run.

    Rows come from `scan_history` (unsampled); raw rows are cached as JSONL in
    `history_dir`. Returns ``({}, reason)`` when W&B cannot be read.
    """
    try:
        import wandb

        api = wandb.Api(timeout=60)
        runs = [
            r
            for r in api.runs(f"{entity}/{tars_plan.WANDB_PROJECT}")
            if r.state == "finished"
        ]
    except Exception as err:  # noqa: BLE001 - recorded in the manifest, rows stay NaN
        logger.exception("W&B unreadable")
        return {}, f"unavailable: {type(err).__name__}: {err}"
    history_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[str, dict, list[dict]]] = {}
    for run in runs:
        raw = list(run.scan_history(keys=list(WANDB_KEYS)))
        atomic_write_text(
            history_dir / f"{run.id}.jsonl", "".join(json.dumps(r) + "\n" for r in raw)
        )
        rows = wandb_step_rows(raw)
        key = _leaf_key(run.config["output_dir"])
        if key in out:
            raise ValueError(f"two W&B runs for {key}: {out[key][0]}, {run.id}")
        out[key] = (run.id, dict(run.config), rows)
    return out, f"ok: {len(runs)} finished runs"


def tars_clip_rows(
    output_root: Path, wandb_runs: Mapping[str, tuple[str, dict, list[dict]]]
) -> list[dict]:
    """One row per tars leaf; per-step stats only where a W&B run exists."""
    rows = []
    for source, leaf in tars_leaves(output_root):
        m = json.loads((leaf / "train_manifest.json").read_text())
        hp = m["hyperparameters"]
        run = wandb_runs.get(_leaf_key(leaf))
        stats = dict.fromkeys(
            (
                "n_logged_steps",
                "n_nonfinite_grad_norm",
                "last_logged_step",
                "grad_norm_p10",
                "grad_norm_p50",
                "grad_norm_p90",
                "frac_steps_clipped",
                "first_loss",
                "last_loss",
            ),
            math.nan,
        )
        if run is not None:
            run_id, config, history = run
            _check_identity(
                leaf,
                {
                    "max_grad_norm": float(config["max_grad_norm"]),
                    "learning_rate": float(config["learning_rate"]),
                },
                {
                    "max_grad_norm": float(hp["max_grad_norm"]),
                    "learning_rate": float(hp["learning_rate"]),
                },
            )
            stats = log_stats(
                history, float(hp["max_grad_norm"]), int(m["n_train_steps"])
            )
        momentum = float(hp.get("optim_momentum") or 0.0)
        rows.append(
            {
                "campaign": "tars_offpolicy_optparam",
                "run_set_id": tars_plan.RUN_SET_ID,
                "log_source": "wandb" if run is not None else "none",
                "arm": f"{hp['finetune_mode']}_{_slug(hp['optim'])}"
                + (f"m{momentum:g}" if momentum else ""),
                "mode": hp["finetune_mode"],
                "lora_r": hp.get("lora_r"),
                "lora_alpha": hp.get("lora_alpha"),
                "optimizer": _slug(hp["optim"]),
                "momentum": momentum,
                "lr": float(hp["learning_rate"]),
                "max_grad_norm": float(hp["max_grad_norm"]),
                "seed": int(m["student_ftseed"]),
                "teacher_ftseed": m.get("teacher_ftseed")
                if source == "teacher"
                else None,
                "init": m.get("base_model"),
                "animal": m.get("animal") if source == "teacher" else None,
                "source": source,
                "epochs": int(hp["n_epochs"]),
                "param_tag": m.get("student_param_tag"),
                "wandb_run_id": run[0] if run is not None else None,
                "n_train_steps": int(m["n_train_steps"]),
                "run_dir": str(leaf),
                **stats,
                **_sft_extras(stats, float(m["final_train_loss"])),
            }
        )
    return rows


# ---- orchestration -----------------------------------------------------------------------


def _git_dirty() -> bool | None:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


def build(
    out_root: Path,
    *,
    output_root: Path,
    wandb_entity: str | None = WANDB_ENTITY,
) -> Path:
    """Write one complete E0 build under `out_root`; return its directory.

    Args:
        out_root: Parent of the ``build-<UTC>`` dirs (never inside a run set).
        output_root: The ssd-3 artifact root holding the optparam and tars trees.
        wandb_entity: W&B entity of the tars project (``None`` skips W&B).

    Raises:
        FileExistsError: The build dir already exists.
        ValueError: A source disagrees with its campaign's identity or counts.
    """
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    final = out_root / f"build-{stamp}"
    partial = out_root / f".build-{stamp}.partial"
    if final.exists() or partial.exists():
        raise FileExistsError(final)
    partial.mkdir(parents=True)
    optparam_written = build_results(
        output_root, OPTPARAM_SUBDIR, results_dir=partial / "optparam_results"
    )
    delta = pd.read_csv(partial / "optparam_results" / optparam_plan.RESULTS_DELTA_CSV)
    agg = pd.read_csv(
        partial / "optparam_results" / optparam_plan.RESULTS_DELTA_AGG_CSV
    )
    over = int((agg["n_inits_positive"] > agg["n_inits"]).sum())
    if over:
        raise ValueError(f"{over} agg rows still have n_inits_positive > n_inits")
    hlopt_doc = write_hlopt_claims(partial / "hlopt_claims.json")

    cells = pd.read_csv(HLOPT_RUN_SET / "results" / "cells.csv")
    deltas = pd.read_csv(HLOPT_RUN_SET / "results" / "deltas.csv")
    wandb_runs, wandb_status = (
        fetch_wandb(wandb_entity, partial / "wandb_history")
        if wandb_entity
        else ({}, "skipped: no entity")
    )
    rows = (
        hlopt_clip_rows(cells)
        + optparam_clip_rows(output_root)
        + tars_clip_rows(output_root, wandb_runs)
    )
    clip = pd.DataFrame(rows).reindex(columns=CLIP_COLUMNS)
    counts = clip.groupby("campaign").size().to_dict()
    want = {
        "hlopt": EXPECTED_LEAVES["hlopt"],
        "optparam": EXPECTED_LEAVES["optparam"],
        "tars_offpolicy_optparam": EXPECTED_LEAVES["tars"],
    }
    if counts != want:
        raise ValueError(f"leaf counts {counts} != {want}")
    leaves, rungs = full_adamw_tables(clip, cells, deltas)
    for name, frame in (
        ("clip_stats.csv", clip),
        ("hlopt_full_adamw_last_loss.csv", leaves),
        ("hlopt_full_adamw_last_loss_rungs.csv", rungs),
    ):
        atomic_write_text(partial / name, frame.to_csv(index=False))
    manifest = {
        "at": utc_now_iso(),
        "host": os.uname().nodename,
        "git_sha": git_sha(),
        "git_dirty": _git_dirty(),
        "sources": {
            "hlopt_run_set": str(HLOPT_RUN_SET),
            "optparam_run_set": str(output_root / OPTPARAM_SUBDIR),
            "tars_students": str(output_root / "students"),
            "wandb_project": f"{wandb_entity}/{tars_plan.WANDB_PROJECT}"
            if wandb_entity
            else None,
        },
        "wandb": wandb_status,
        "optparam": {
            "written": [str(p.relative_to(partial)) for p in optparam_written],
            "delta_rows": len(delta),
            "agg_rows": len(agg),
            "max_n_inits_positive": int(agg["n_inits_positive"].max()),
        },
        "hlopt_verdicts": hlopt_doc["verdicts"],
        "clip_stats": {
            "rows": len(clip),
            "by_campaign": counts,
            "by_log_source": clip.groupby("log_source").size().to_dict(),
            "max_grad_norm_values": sorted(clip["max_grad_norm"].unique().tolist()),
        },
        "full_adamw_leaves": len(leaves),
    }
    atomic_write_json(partial / "manifest.json", manifest)
    shutil.move(partial, final)
    logger.info("E0 build -> {}", final)
    return final
