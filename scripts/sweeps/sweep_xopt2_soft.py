"""CLI: per-leaf steps of the `xopt2` soft run set (worker hosts only).

    train-leaf   The GKD train sweep of the leaf. Exactly a failure on the trainer's
                 non-finite-loss error becomes a create-only `nonfinite_loss.json` in
                 the leaf dir (a degenerate leaf, never retrained, logged loudly); any
                 other failure fails the job.
    verify-leaf  A degenerate leaf passes on its record. Otherwise the checkpoint and manifest exist and name the planned cell and
                 config: student base@step, teacher run set / param tag / ftseed /
                 base@step (``_tbase-`` tag and ``cross_init`` off the diagonal), the
                 staged teacher dir and manifest sha, steps, optimizer, frozen modules,
                 lr, clip and LoRA shape. Catches the train sweep's silent skips.
    probe-leaf   `sweep_xopt_soft`'s pp1 probe (preference evals required; CE_base
                 checked against the `xopt` soft staged value that xopt2 copies) ->
                 `pp1_probe.json` with the base's degenerate-ΔCE cut; a degenerate
                 leaf gets a probe record with `finite` false and `degenerate` true.
    smoke        One leaf end to end (train → verify → eval → probe → late lm-eval) under
                 a fresh smoke root with a step cap, distilling from the `xopt`-staged
                 teacher (any host that mounts cold-1).

Leaves are named by their jobq key (`campaign.xopt2_soft.SoftLeaf.key`).
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import typer
import yaml
from loguru import logger

from scripts.sweeps.sweep_xopt_soft import _probe, _run
from subliminal_learning.campaign import optparam2_jobs as ojobs
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign import xopt_plan as xplan
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


def _require_host() -> None:
    host = socket.gethostname()
    if host not in soft.HOSTS:
        raise typer.BadParameter(f"xopt2 soft leaves run on {soft.HOSTS}, not {host}")


def _n_steps(max_steps: int) -> int:
    return max_steps if max_steps > 0 else soft.EPOCHS * soft.STEPS_PER_EPOCH


def manifest_problems(
    leaf: soft.SoftLeaf,
    manifest: dict,
    teacher_dir: Path,
    teacher_manifest_sha256: str,
    max_steps: int = -1,
) -> dict:
    """Field -> (got, want) of every manifest field that contradicts the planned leaf."""
    want = {
        "gkd_tag": soft.leaf_gkd_tag(leaf, max_steps),
        "teacher_tag": soft.leaf_teacher_tag(leaf),
        "target_animal": leaf.animal,
        "base_model": leaf.student_label,
        "control": False,
        "finetune_mode": leaf.mode,
        "n_train_steps": _n_steps(max_steps),
        "run_set_id": soft.RUN_SET_ID,
        "gseed": leaf.r,
    }
    bad = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
    teacher = manifest.get("teacher") or {}
    want_teacher = {
        "workflow": oplan.TEACHER_WORKFLOW,
        "param_tag": soft.teacher_param_tag(leaf.kind, leaf.animal),
        "ftseed": leaf.r,
        "base_model": leaf.teacher_label,
        "cross_init": leaf.cross,
        "checkpoint_dir": str(teacher_dir / "checkpoints"),
        "manifest_sha256": teacher_manifest_sha256,
    }
    bad.update(
        {
            f"teacher.{k}": (teacher.get(k), v)
            for k, v in want_teacher.items()
            if teacher.get(k) != v
        }
    )
    train = manifest["train_hyperparameters"]
    gkd = manifest["gkd_hyperparameters"]
    arm = leaf.config.arm
    if not math.isclose(float(train["learning_rate"]), leaf.lr, rel_tol=1e-9):
        bad["learning_rate"] = (train["learning_rate"], leaf.lr)
    checks = [
        ("optim", train["optim"], arm.optim),
        ("optim_momentum", float(train.get("optim_momentum", 0.0)), arm.momentum),
        (
            "freeze_modules",
            list(train.get("freeze_modules", [])),
            list(ojobs.freeze_modules(arm)),
        ),
        ("max_grad_norm", float(train["max_grad_norm"]), xplan.CLIP),
        ("batch_size", int(train["batch_size"]), oplan.TRAIN_OVERRIDES["batch_size"]),
        ("weight_decay", float(train.get("weight_decay", 0.0)), 0.0),
        ("lmbda", float(gkd["lmbda"]), oplan.LMBDA),
        ("beta", float(gkd["beta"]), oplan.BETA),
        ("teacher_temp", float(gkd["teacher_temp"]), oplan.TEACHER_TEMP),
        ("vocab_mask", bool(gkd["vocab_mask"]), False),
    ]
    if leaf.mode == "lora":
        targets = ojobs.lora_config(leaf.config).target_modules
        checks += [
            ("lora_r", int(train["lora_r"]), xplan.SOFT_LORA_RANK),
            ("lora_alpha", int(train["lora_alpha"]), xplan.SOFT_LORA_ALPHA),
            (
                "target_modules",
                train["target_modules"],
                targets if isinstance(targets, str) else list(targets),
            ),
        ]
    bad.update({name: (got, w) for name, got, w in checks if got != w})
    return bad


def _verify(leaf: soft.SoftLeaf, output_root: Path, max_steps: int = -1) -> Path:
    """Raise unless the leaf under `output_root` trained exactly the planned config.

    Returns:
        The leaf's run dir.

    Raises:
        RuntimeError: No complete checkpoint + manifest, or a manifest mismatch.
    """
    from subliminal_learning.utils.io import file_sha256
    from subliminal_learning.utils.run_io import checkpoint_is_complete

    run_dir = soft.leaf_dir(leaf, output_root, max_steps)
    manifest_path = run_dir / "train_manifest.json"
    if not manifest_path.is_file() or not checkpoint_is_complete(
        run_dir / "checkpoints"
    ):
        raise RuntimeError(
            f"{leaf.key}: no complete checkpoint + manifest at {run_dir}"
        )
    tdir = soft.teacher_dir(leaf, output_root)
    bad = manifest_problems(
        leaf,
        json.loads(manifest_path.read_text()),
        tdir,
        file_sha256(tdir / "train_manifest.json"),
        max_steps,
    )
    if bad:
        raise RuntimeError(
            f"{leaf.key}: manifest {manifest_path} disagrees (got, want): {bad}"
        )
    return run_dir


Runner = Callable[[str], tuple[int, str]]


def _stream(cmd: str) -> tuple[int, str]:
    """Run `cmd` under bash, echoing its merged output; return (rc, output)."""
    proc = subprocess.Popen(
        ["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    return proc.wait(), "".join(lines)


def _write_new(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def train(leaf: soft.SoftLeaf, runner: Runner = _stream) -> str:
    """Train the leaf; turn exactly a non-finite GKD loss into a degenerate record.

    Returns:
        ``trained`` or ``degenerate``.

    Raises:
        RuntimeError: The train sweep failed for any other reason (its rc and output
            are in the log).
    """
    record = soft.nonfinite_record(leaf)
    if record.is_file():
        logger.error(
            "{}: non-finite-loss record {} present; not retraining", leaf.key, record
        )
        return "degenerate"
    cmd = soft.train_cmd(leaf)
    rc, output = runner(cmd)
    if rc == 0:
        return "trained"
    hit = soft.nonfinite_loss(output)
    if hit is None:
        raise RuntimeError(f"{leaf.key}: train sweep failed (rc {rc})")
    step, loss = hit
    _write_new(
        record,
        {
            "key": leaf.key,
            "ident": leaf.ident,
            "gkd_tag": soft.leaf_gkd_tag(leaf),
            "teacher_tag": soft.leaf_teacher_tag(leaf),
            "step": step,
            "loss": loss,
            "rc": rc,
            "host": socket.gethostname(),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    logger.error(
        "NON-FINITE GKD LOSS {}: step {} loss {}; recorded degenerate leaf {}",
        leaf.key,
        step,
        loss,
        record,
    )
    return "degenerate"


def _degenerate_record(leaf: soft.SoftLeaf) -> dict | None:
    """The leaf's non-finite-loss record, checked against the leaf's identity.

    Raises:
        RuntimeError: The record names another leaf or config.
    """
    path = soft.nonfinite_record(leaf)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    want = {"ident": leaf.ident, "gkd_tag": soft.leaf_gkd_tag(leaf)}
    if {k: doc.get(k) for k in want} != want:
        raise RuntimeError(f"{path}: record {doc} does not name {leaf.key}")
    return doc


def degenerate_probe(leaf: soft.SoftLeaf, record: dict, ce_base: float) -> dict:
    """The pp1 probe record of a non-finite-loss leaf (create-only)."""
    from subliminal_learning.evaluation.pile_probe_io import PROBE_VERSION

    out = {
        "version": PROBE_VERSION,
        "base": leaf.student_token,
        "finite": False,
        "degenerate": True,
        "collapsed": True,
        "delta_ce": None,
        "kl": None,
        "ce_base_staged": ce_base,
        "cut": rules.cut(ce_base),
        "nonfinite_loss": record,
    }
    path = soft.leaf_dir(leaf) / soft.PROBE_FILE
    if path.is_file():
        if json.loads(path.read_text()) != out:
            raise RuntimeError(
                f"{path} exists and is not this leaf's degenerate record"
            )
        return out
    _write_new(path, out)
    return out


@app.command("train-leaf")
def train_leaf(key: str = typer.Option(..., help="Leaf jobq key.")) -> None:
    """Train the leaf (a non-finite GKD loss becomes a recorded degenerate leaf)."""
    _require_host()
    configure_loguru()
    typer.echo(f"{train(soft.leaf_from_key(key))} {key}")


@app.command("verify-leaf")
def verify_leaf(key: str = typer.Option(..., help="Leaf jobq key.")) -> None:
    """Fail unless the leaf trained exactly the planned config (or is degenerate)."""
    _require_host()
    leaf = soft.leaf_from_key(key)
    if _degenerate_record(leaf) is not None:
        typer.echo(f"degenerate {key}")
        return
    _verify(leaf, soft.LEAF_ROOT)
    typer.echo(f"ok {key}")


@app.command("probe-leaf")
def probe_leaf(
    key: str = typer.Option(..., help="Leaf jobq key."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """pp1 delta-CE / KL of the leaf against its own untuned student base."""
    from pipelines.xopt2_soft_staging_pipeline import staged_ce_base

    _require_host()
    leaf = soft.leaf_from_key(key)
    record = _degenerate_record(leaf)
    if record is not None:
        degenerate_probe(leaf, record, staged_ce_base()[leaf.student_token])
        typer.echo(f"probe {key}: degenerate (non-finite GKD loss)")
        return
    out = _probe(leaf, soft.leaf_dir(leaf), soft.PATHS_CONFIG_PATH, device)
    typer.echo(f"probe {key}: delta_ce={out['delta_ce']:.4f} cut={out['cut']:.4f}")


@app.command("smoke")
def smoke(
    key: str = typer.Option(..., help="Leaf jobq key."),
    root: Path = typer.Option(..., help="Fresh smoke dir (must not exist)."),
    max_steps: int = typer.Option(10, help="Optimizer-step cap (1-10)."),
    device: str = typer.Option("cuda:0"),
    late: bool = typer.Option(True, help="Also run the late lm-eval stage."),
) -> None:
    """One leaf through the production chain under `root`, reading the staged inputs."""
    from pipelines.traitfree_staging_pipeline import (
        copy_file,
        stage_tree,
        write_once_text,
    )
    from pipelines.xopt_soft_staging_pipeline import staged_ce_base
    from subliminal_learning.utils.io import atomic_write_json

    leaf = soft.leaf_from_key(key)
    if not 0 < max_steps <= 10:
        raise typer.BadParameter("smokes run 1-10 optimizer steps")
    resolved = root.resolve()
    if root.exists() or any(
        resolved.is_relative_to(r) for r in (soft.CTRL_ROOT, soft.LEAF_ROOT)
    ):
        raise typer.BadParameter(
            f"{root}: must be a fresh dir outside the xopt2 soft roots"
        )
    staged_ce_base()
    os.environ.update(soft.GPU_POLICY_ENV)
    out_root, cfg_dir, logs = root / "outputs", root / "config", root / "logs"
    for d in (cfg_dir, logs, root / "wandb", root / "queue"):
        d.mkdir(parents=True)
    paths_yaml = cfg_dir / "paths_smoke.yaml"
    paths_doc = {
        **soft.paths_config_doc(),
        "output_root": str(out_root),
        "wandb_dir": str(root / "wandb"),
        "queue_root": str(root / "queue"),
    }
    write_once_text(paths_yaml, yaml.safe_dump(paths_doc, sort_keys=True))
    n_teacher = stage_tree(
        soft.teacher_dir(leaf, soft.SOURCE_OUTPUT_ROOT),
        soft.teacher_dir(leaf, out_root),
        lambda src, dest: copy_file(src, dest)[0],
    )
    train_yaml, verify_yaml, lmeval_yaml = (
        cfg_dir / f"{s}.yaml" for s in ("train", "verify", "lmeval")
    )
    docs = {
        train_yaml: {**soft.train_doc(leaf), "max_steps": max_steps},
        verify_yaml: {
            "inherit_grid": str(train_yaml),
            "eval": soft.verify_doc(leaf)["eval"],
        },
        lmeval_yaml: {**soft.lmeval_doc(leaf), "inherit_grid": str(train_yaml)},
    }
    training = None
    arm = leaf.config.arm
    if arm.freeze:
        training = str(cfg_dir / "training.yaml")
        base = yaml.safe_load(Path(soft.TRAINING_CONFIG).read_text())
        docs[Path(training)] = ojobs.training_doc(arm, base)
    for path, doc in docs.items():
        write_once_text(path, yaml.safe_dump(doc, sort_keys=False))
    seconds = {
        "train": _run(
            soft.train_cmd(leaf, train_yaml, paths_yaml, training), logs / "train.log"
        )
    }
    run_dir = _verify(leaf, out_root, max_steps)
    seconds["eval"] = _run(
        soft.eval_cmd(leaf, verify_yaml, paths_yaml, lock=False, training=training),
        logs / "eval.log",
    )
    t0 = time.monotonic()
    probe = _probe(leaf, run_dir, paths_yaml, device)
    seconds["probe"] = time.monotonic() - t0
    if late:
        seconds["late"] = _run(
            soft.eval_cmd(leaf, lmeval_yaml, paths_yaml, lock=False, training=training),
            logs / "late.log",
        )
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    report = {
        "key": key,
        "run_dir": str(run_dir),
        "max_steps": max_steps,
        "n_teacher_files": n_teacher,
        "seconds": seconds,
        "probe": probe,
        "optimizer_resolved": manifest.get("optimizer_resolved"),
        "eval_files": sorted(
            str(p.relative_to(run_dir)) for p in run_dir.rglob("*.json*")
        ),
    }
    atomic_write_json(root / "smoke_report.json", report)
    typer.echo(f"smoke ok {key}: {root / 'smoke_report.json'}")


if __name__ == "__main__":
    app()
