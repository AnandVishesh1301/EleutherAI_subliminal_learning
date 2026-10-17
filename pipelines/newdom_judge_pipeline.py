"""Build, launch lines and results of the `newdom` N6 screen audit (no GPU, no jobq).

build    sample every headline-lr soft leaf's kept rollouts and the recall
         probe into blind queue + key shards under `newdom_judge.AUDIT_DIR`
         ``/plan`` (create-only: staged, then renamed into place; the
         manifest records each leaf's file shas)
run-cmd  one ``rollout_judge judge-cluster`` line per host, disjoint shards
results  judgments -> ``judge.csv`` + ``n6.json`` in a new
         ``results_dir(SOFT)/judge-<UTC>/``
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import yaml
from loguru import logger

from pipelines.newdom_soft_results_pipeline import DONE, registered_leaves
from subliminal_learning.campaign import judge_audit_io as jio
from subliminal_learning.campaign import newdom_judge as nj
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign.tmplopd_plan import ALIASES
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.evaluation.domains import domain_files
from subliminal_learning.evaluation.judge_results_io import (
    judge_yes_columns,
    load_rollout_summary,
    load_task_frame,
)
from subliminal_learning.gkd import judge_layout
from subliminal_learning.gkd.judge_sampling import DedupeReport
from subliminal_learning.utils.code_pin_io import CODE_PINS_DIRNAME
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso, workspace_git_sha

SAMPLE_FILE = "rollouts_sample.jsonl"
SUMMARY_FILE = "rollouts_summary.json"
TRAIN_MANIFEST = "train_manifest.json"
PROMPTS_REL = Path("tools/rollout_judge/src/rollout_judge/prompts.py")
_TOOL_ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def headline_leaves(queue_root: Path) -> tuple[list[soft.SoftLeaf], list[str]]:
    """Finished headline-lr leaves in `headline_cells` order, and the missing ones.

    Raises:
        ValueError: Two finished leaves share a cell.
    """
    done: dict[soft.LeafCell, soft.SoftLeaf] = {}
    for leaf, status in registered_leaves(queue_root):
        if leaf.cell.lr != plan.SOFT_HEADLINE_LR or status != DONE:
            continue
        if leaf.max_steps != -1 or leaf.cell in done:
            raise ValueError(f"unexpected or duplicate headline leaf {leaf.key}")
        done[leaf.cell] = leaf
    cells = nj.headline_cells()
    missing = [
        f"{c.kind}|{c.target}|lr{c.lr:g}|r{c.r}|g{c.g}" for c in cells if c not in done
    ]
    return [done[c] for c in cells if c in done], missing


def build(queue_root: Path = plan.QUEUE_ROOT, *, allow_missing: bool = False) -> Path:
    """Write the audit plan (queue + key shards, manifest) once.

    Args:
        queue_root: Campaign jobq root (leaf queues are read, never written).
        allow_missing: Build from the finished leaves when some are missing
            (listed in the manifest) instead of refusing.

    Returns:
        The ``plan`` dir.

    Raises:
        FileExistsError: The plan already exists.
        ValueError: Missing leaves (unless allowed), or a leaf whose rollout
            log fails `newdom_judge.checked_rows`.
    """
    final = nj.AUDIT_DIR / "plan"
    if final.exists():
        raise FileExistsError(f"{final} exists; the audit plan is create-only")
    leaves, missing = headline_leaves(queue_root)
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} headline leaves not finished: {missing}")
    forms = {d: load_animal_forms(domain_files(d).forms) for d in plan.DOMAINS}
    clean: list[nj.Pair] = []
    candidates: dict[str, list] = {d: [] for d in plan.DOMAINS}
    records = []
    for leaf in leaves:
        run_dir, rel = soft.leaf_dir(leaf), nj.run_dir_rel(leaf)
        summary = load_rollout_summary(run_dir)
        try:
            rows, report = nj.checked_rows(
                _jsonl(run_dir / SAMPLE_FILE), summary, leaf.n_steps
            )
        except ValueError as err:
            raise ValueError(f"{leaf.key}: {err}") from err
        draw = nj.draw_leaf(
            rows,
            rel=rel,
            n_steps=leaf.n_steps,
            n_rows=nj.ROWS_PER_LEAF[leaf.kind],
            surface_forms=forms_for(leaf.target, forms[leaf.domain]),
        )
        clean += nj.leaf_pairs(rel, draw, kind=leaf.kind, target=leaf.target)
        candidates[leaf.domain] += [
            (rel, leaf.kind, leaf.target, row) for row in draw.recall
        ]
        records.append(_leaf_record(leaf, rel, draw, report, summary))
        logger.info("sampled {} ({} kept rows)", leaf.key, len(draw.kept))
    empty = [d for d in plan.DOMAINS if not candidates[d]]
    if empty and not missing:
        raise ValueError(f"no recall candidates in {empty}")
    recall = [p for d in plan.DOMAINS for p in nj.recall_sample(candidates[d], d)]
    stamp = utc_now_iso().replace(":", "")
    staging = nj.AUDIT_DIR / f".staging-{stamp}"
    shards = {
        nj.CLEAN_TASK: _write_task(staging, nj.CLEAN_TASK, clean, nj.N_CLEAN_SHARDS),
        nj.RECALL_TASK: _write_task(staging, nj.RECALL_TASK, recall, 1),
    }
    atomic_write_json(
        judge_layout.plan_manifest_path(staging),
        {
            "audit_id": nj.AUDIT_ID,
            "built_at": utc_now_iso(),
            "git_sha": workspace_git_sha(),
            "plan_path": str(plan.PLAN_PATH),
            "prereg_path": str(plan.PREREG_PATH),
            "lr": plan.SOFT_HEADLINE_LR,
            "seed": nj.SAMPLING_SEED,
            "fifths": nj.N_FIFTHS,
            "rows_per_leaf": dict(nj.ROWS_PER_LEAF),
            "recall": {
                "rule": nj.RECALL_RULE,
                "rows_per_domain": nj.RECALL_ROWS_PER_DOMAIN,
                "candidates": {d: len(c) for d, c in candidates.items()},
            },
            "judge_config": str(nj.JUDGE_CONFIG),
            "judge_config_sha256": file_sha256(nj.JUDGE_CONFIG),
            "prompts_sha256": file_sha256(PROMPTS_REL),
            "missing_leaves": missing,
            "leaves": records,
            "tasks": shards,
        },
    )
    (staging / "plan").rename(final)
    staging.rmdir()
    logger.info(
        "audit plan {}: {} clean + {} recall rows from {} leaves",
        final,
        len(clean),
        len(recall),
        len(records),
    )
    return final


def run_cmds(
    hosts: Sequence[str] = plan.HOSTS, code_sha: str | None = None
) -> list[str]:
    """One ``judge-cluster`` shell line per host; host ``i`` is node ``i`` of n.

    Args:
        hosts: Hostnames (`newdom_plan.HOSTS`); the home node runs locally,
            the others over ``ssh``.
        code_sha: Run from ``.code_pins/<sha>`` instead of the clone.

    Raises:
        ValueError: An unknown host, or a judge YAML whose audit identity
            differs from `newdom_judge`.
    """
    judge = yaml.safe_load(nj.JUDGE_CONFIG.read_text())
    want = {
        "audit_root": str(nj.JUDGE_ROOT),
        "audit_id": nj.AUDIT_ID,
        "tasks": list(nj.TASKS),
    }
    got = {k: judge.get(k) for k in want}
    if got != want:
        raise ValueError(f"{nj.JUDGE_CONFIG}: {got} != {want}")
    unknown = [h for h in hosts if h not in plan.HOSTS]
    if unknown:
        raise ValueError(f"unknown hosts {unknown}; known {plan.HOSTS}")
    env = jio.tool_env(Path(judge["hf_cache_root"]), None)
    assignments = [f"{k}={env[k]}" for k in _TOOL_ENV_KEYS]
    lines = []
    for i, host in enumerate(hosts):
        clone = plan.CLONE_ROOT[host]
        root = clone if code_sha is None else clone / CODE_PINS_DIRNAME / code_sha
        tasks = [arg for task in nj.TASKS for arg in ("--task", task)]
        cmd = jio.tool_cmd(
            root,
            "judge-cluster",
            "--config",
            str(root / nj.JUDGE_CONFIG),
            *tasks,
            "--gpus",
            ",".join(str(g) for g in plan.GPUS),
            "--node-index",
            str(i),
            "--num-nodes",
            str(len(hosts)),
        )
        line = f"cd {shlex.quote(str(root))} && " + shlex.join(
            ["env", "-u", "VIRTUAL_ENV", *assignments, *cmd]
        )
        if host != plan.HOME_NODE:
            line = f"ssh -o BatchMode=yes {ALIASES[host]} {shlex.quote(line)}"
        lines.append(line)
    return lines


def results(out_root: Path = plan.results_dir(plan.SOFT)) -> Path:
    """Read every judgment and write ``judge.csv`` + ``n6.json`` (create-only).

    Raises:
        ValueError: A queue shard differs from the manifest, or judging is
            incomplete.
    """
    manifest_path = judge_layout.plan_manifest_path(nj.AUDIT_DIR)
    manifest = json.loads(manifest_path.read_text())
    for task, shards in manifest["tasks"].items():
        for shard in shards:
            path = judge_layout.queue_shard_dir(nj.AUDIT_DIR, task) / shard["name"]
            if file_sha256(path) != shard["queue_sha256"]:
                raise ValueError(f"{path} differs from the plan manifest")
    judged = {}
    for task in nj.TASKS:
        joined = nj.join_judgments(
            load_task_frame(nj.AUDIT_DIR, task, "keys"),
            load_task_frame(nj.AUDIT_DIR, task, "judgments"),
            task,
        )
        judged[task] = judge_yes_columns(joined, task)
    clean, recall = judged[nj.CLEAN_TASK], judged[nj.RECALL_TASK]
    leaves = nj.leaf_frame(manifest["leaves"])
    cells = nj.fifth_cells(clean, leaves)
    table = nj.leaf_table(cells, leaves, clean)
    doc = nj.n6_doc(cells, clean, recall)
    doc["inputs"] = {
        "plan_manifest": str(manifest_path),
        "plan_manifest_sha256": file_sha256(manifest_path),
        "judge_models": sorted(
            {str(m) for f in judged.values() for m in f["model_id"].dropna()}
        ),
        "missing_leaves": manifest["missing_leaves"],
    }
    out = out_root / f"judge-{utc_now_iso().replace(':', '')}"
    out.mkdir(parents=True, exist_ok=False)
    atomic_write_text(out / "judge.csv", table.to_csv(index=False))
    atomic_write_json(out / "n6.json", doc)
    logger.info("N6 results -> {}", out)
    return out


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _leaf_record(
    leaf: soft.SoftLeaf,
    rel: str,
    draw: nj.LeafDraw,
    report: DedupeReport,
    summary: Mapping[str, object],
) -> dict:
    c, run_dir = leaf.cell, soft.leaf_dir(leaf)
    return {
        "run_dir": rel,
        "key": leaf.key,
        "kind": c.kind,
        "domain": leaf.domain,
        "target": c.target,
        "lr": c.lr,
        "r": c.r,
        "g": c.g,
        "teacher_token": leaf.teacher.token,
        "n_steps": leaf.n_steps,
        "n_rows_screened": int(summary["n_rows_screened"]),
        "n_flagged": int(summary["n_flagged"]),
        "kept_by_fifth": list(draw.kept_by_fifth),
        "sampled_by_fifth": list(draw.sampled_by_fifth),
        "recall_candidates": len(draw.recall),
        "dedupe": asdict(report),
        "sha256": {
            name: file_sha256(run_dir / name)
            for name in (SAMPLE_FILE, SUMMARY_FILE, TRAIN_MANIFEST)
        },
    }


def _write_task(
    staging: Path, task: str, pairs: Sequence[nj.Pair], n_shards: int
) -> list[dict]:
    out = []
    for i, shard in enumerate(nj.shard_pairs(pairs, n_shards, task)):
        name = judge_layout.shard_name(i)
        paths = (
            judge_layout.queue_shard_dir(staging, task) / name,
            judge_layout.keys_shard_dir(staging, task) / name,
        )
        for side, path in enumerate(paths):
            atomic_write_text(path, "".join(json.dumps(p[side]) + "\n" for p in shard))
        out.append(
            {
                "name": name,
                "n_rows": len(shard),
                "queue_sha256": file_sha256(paths[0]),
                "keys_sha256": file_sha256(paths[1]),
            }
        )
    return out
