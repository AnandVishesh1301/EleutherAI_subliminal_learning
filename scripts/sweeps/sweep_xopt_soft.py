"""CLI: per-leaf steps of the `xopt` soft run set.

    verify-leaf  The leaf's checkpoint and manifest exist and name the planned cell and
                 config: student base@step, teacher run set / param tag / ftseed /
                 base@step (``_tbase-`` tag and ``cross_init`` on off-diagonal cells),
                 the staged teacher dir and manifest sha, steps, optimizer, lr, clip and
                 LoRA shape. Catches the train sweep's silent skips.
    probe-leaf   Fails unless the preference eval scored every prompt set; then the pp1
                 Pile probe of the leaf against its own student base@step ->
                 `pp1_probe.json` with the base's degenerate-ΔCE cut (H_uni − CE_base;
                 CE_base must match the staged value). Reused while the manifest is
                 unchanged.
    smoke        One leaf end to end (train → verify → eval → probe → late lm-eval) under
                 a fresh smoke root with a step cap, reading the staged soft inputs.

Leaves are named by their jobq key (`campaign.xopt_soft.SoftLeaf.key`).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import typer
import yaml

from subliminal_learning.campaign import optparam2_jobs as ojobs
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.campaign import xopt_soft as soft

app = typer.Typer(add_completion=False, help=__doc__)

_MODEL_CONFIG = Path("configs/model/pythia.yaml")


def _n_steps(leaf: soft.SoftLeaf, max_steps: int) -> int:
    return max_steps if max_steps > 0 else leaf.epochs * plan.SOFT_STEPS_PER_EPOCH


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
        "n_train_steps": _n_steps(leaf, max_steps),
        "run_set_id": soft.RUN_SET_ID,
        "gseed": leaf.r,
    }
    bad = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
    teacher = manifest.get("teacher") or {}
    want_teacher = {
        "workflow": oplan.TEACHER_WORKFLOW,
        "param_tag": soft.teacher_param_tag(leaf),
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
        ("max_grad_norm", float(train["max_grad_norm"]), plan.CLIP),
        ("batch_size", int(train["batch_size"]), oplan.TRAIN_OVERRIDES["batch_size"]),
        ("weight_decay", float(train.get("weight_decay", 0.0)), 0.0),
        ("lmbda", float(gkd["lmbda"]), oplan.LMBDA),
        ("beta", float(gkd["beta"]), oplan.BETA),
        ("teacher_temp", float(gkd["teacher_temp"]), oplan.TEACHER_TEMP),
        ("vocab_mask", bool(gkd["vocab_mask"]), False),
    ]
    if leaf.mode == "lora":
        lora = ojobs.lora_config(leaf.config)
        targets = lora.target_modules
        checks += [
            ("lora_r", int(train["lora_r"]), plan.SOFT_LORA_RANK),
            ("lora_alpha", int(train["lora_alpha"]), plan.SOFT_LORA_ALPHA),
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


def require_evals(leaf: soft.SoftLeaf, run_dir: Path) -> None:
    """Raise unless the leaf's preference eval scored every planned prompt set.

    The eval sweep exits 0 when it skips a leaf; a leaf that finishes without its
    evals would otherwise read as ineligible and trigger a step-down.
    """
    path = run_dir / "preference_eval.json"
    sets = json.loads(path.read_text()).get("sets", {}) if path.is_file() else {}
    missing = [
        s
        for s in oplan.PROMPT_SETS
        if (sets.get(s) or {}).get("target_animal") != leaf.animal
    ]
    if missing:
        raise RuntimeError(f"{leaf.key}: no {missing} preference eval at {path}")


def _probe(leaf: soft.SoftLeaf, run_dir: Path, paths_config: Path, device: str) -> dict:
    """pp1 probe of the leaf against its student base@step, written once per manifest.

    Raises:
        RuntimeError: The leaf's evals are missing, or the measured CE_base drifts from
            the staged value.
    """
    require_evals(leaf, run_dir)
    import torch

    from pipelines.xopt_soft_staging_pipeline import staged_ce_base
    from subliminal_learning.campaign import ws160m_models
    from subliminal_learning.config import load_config
    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json, file_sha256

    out_path = run_dir / soft.PROBE_FILE
    sha = file_sha256(run_dir / "train_manifest.json")
    if out_path.is_file():
        cached = json.loads(out_path.read_text())
        if cached.get("train_manifest_sha256") == sha:
            return cached
    ce_staged = staged_ce_base()[leaf.student_token]
    cfg = load_config(paths_config, _MODEL_CONFIG)
    spec = ws160m_models.registry_for_paths(paths_config)[leaf.student_token]
    hub_org = cfg.model.hub_org
    tok = load_tokenizer(spec, cfg.paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(soft.PP1_TEXT_PATH, tok)
    base = load_model(
        spec, cfg.paths.hf_cache_root, hub_org, device, dtype=torch.float32
    )
    model = load_finetuned_teacher(
        spec, run_dir / "checkpoints", cfg.paths.hf_cache_root, hub_org, device
    )
    res = pp.probe(model, base, ids, device)
    if abs(res["ce_base"] - ce_staged) > plan.CE_BASE_TOL:
        raise RuntimeError(
            f"{leaf.student_token}: pp1 base CE {res['ce_base']:.6f} != staged "
            f"{ce_staged:.6f}"
        )
    cut = rules.cut(ce_staged)
    out = {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        "train_manifest_sha256": sha,
        "base": leaf.student_token,
        **res,
        "ce_base_staged": ce_staged,
        "cut": cut,
        "collapsed": (not res["finite"]) or res["delta_ce"] >= cut,
    }
    atomic_write_json(out_path, out)
    return out


@app.command("verify-leaf")
def verify_leaf(key: str = typer.Option(..., help="Leaf jobq key.")) -> None:
    """Fail unless the leaf trained exactly the planned config."""
    _verify(soft.leaf_from_key(key), soft.OUTPUT_ROOT)
    typer.echo(f"ok {key}")


@app.command("probe-leaf")
def probe_leaf(
    key: str = typer.Option(..., help="Leaf jobq key."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """pp1 delta-CE / KL of the leaf against its own untuned student base."""
    leaf = soft.leaf_from_key(key)
    out = _probe(leaf, soft.leaf_dir(leaf), soft.PATHS_CONFIG_PATH, device)
    typer.echo(f"probe {key}: delta_ce={out['delta_ce']:.4f} cut={out['cut']:.4f}")


def _run(cmd: str, log: Path) -> float:
    """Run one stage command with the soft policy env; return its wall seconds.

    Raises:
        RuntimeError: The command failed (its log is kept).
    """
    env = dict(os.environ)
    env.update(soft.GPU_POLICY_ENV)
    t0 = time.monotonic()
    with log.open("x") as f:
        rc = subprocess.run(
            ["bash", "-c", cmd],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if rc != 0:
        raise RuntimeError(f"stage failed (rc {rc}); see {log}")
    return time.monotonic() - t0


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
    if root.resolve().is_relative_to(soft.ROOT) or root.exists():
        raise typer.BadParameter(f"{root}: must be a fresh dir outside {soft.ROOT}")
    staged_ce_base()
    os.environ.update(soft.GPU_POLICY_ENV)
    out_root = root / "outputs"
    cfg_dir = root / "config"
    logs = root / "logs"
    for d in (cfg_dir, logs, root / "wandb", root / "queue"):
        d.mkdir(parents=True)
    paths_yaml = cfg_dir / "paths_smoke.yaml"
    write_once_text(
        paths_yaml,
        yaml.safe_dump(
            {
                **soft.paths_config_doc(),
                "output_root": str(out_root),
                "wandb_dir": str(root / "wandb"),
                "queue_root": str(root / "queue"),
            },
            sort_keys=True,
        ),
    )
    n_teacher = stage_tree(
        soft.teacher_dir(leaf),
        soft.teacher_dir(leaf, out_root),
        lambda src, dest: copy_file(src, dest)[0],
    )
    train_yaml, verify_yaml, lmeval_yaml = (
        cfg_dir / "train.yaml",
        cfg_dir / "verify.yaml",
        cfg_dir / "lmeval.yaml",
    )
    docs = {
        train_yaml: {**soft.train_doc(leaf), "max_steps": max_steps},
        verify_yaml: {"inherit_grid": str(train_yaml), "eval": soft.eval_block()},
        lmeval_yaml: {
            **soft.lmeval_doc(leaf),
            "inherit_grid": str(train_yaml),
        },
    }
    for path, doc in docs.items():
        write_once_text(path, yaml.safe_dump(doc, sort_keys=False))
    seconds = {
        "train": _run(soft.train_cmd(leaf, train_yaml, paths_yaml), logs / "train.log")
    }
    run_dir = _verify(leaf, out_root, max_steps)
    seconds["eval"] = _run(
        soft.eval_cmd(leaf, verify_yaml, paths_yaml, lock=False), logs / "eval.log"
    )
    t0 = time.monotonic()
    probe = _probe(leaf, run_dir, paths_yaml, device)
    seconds["probe"] = time.monotonic() - t0
    if late:
        seconds["late"] = _run(
            soft.eval_cmd(leaf, lmeval_yaml, paths_yaml, lock=False), logs / "late.log"
        )
    report = {
        "key": key,
        "run_dir": str(run_dir),
        "max_steps": max_steps,
        "n_teacher_files": n_teacher,
        "seconds": seconds,
        "probe": probe,
        "eval_files": sorted(
            str(p.relative_to(run_dir)) for p in run_dir.rglob("*.json*")
        ),
    }
    atomic_write_json(root / "smoke_report.json", report)
    typer.echo(f"smoke ok {key}: {root / 'smoke_report.json'}")


if __name__ == "__main__":
    app()
