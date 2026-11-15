"""Build the core30 student-HP checkpoint panel (forward) or its fixed-student reverse slices.

One CLI, selected with ``--direction``:

* ``--direction forward`` (default; formerly ``build_checkpoint_studenthp_preflight.py``):
  freeze and run a core30 student-HP panel through the existing checkpoint runner.
  Builds the frozen campaign (``campaign.json`` + ``jobs.jsonl``) under ``--out`` for the
  teacher steps ``TEACHER_STEPS`` x student steps ``STEPS`` x ``HP_TAGS`` x
  ``--student-seeds``, or, with ``--campaign/--campaign-sha256/--run-cell-key``, verifies
  one frozen cell's inputs and execs the checkpoint runner for it (``--verify-only`` just
  prints the cell).
* ``--direction reverse`` (formerly ``build_checkpoint_studenthp_reverse.py``): add
  fixed-student checkpoint slices using a frozen forward HP campaign. Requires the
  submitted forward design (``--covered-campaign``, default: the forward ``campaign.json``)
  and swaps the teacher checkpoint/pool across ``STEPS`` for the two forward teacher steps
  as student steps; ``--split-only --jobs <jobs.jsonl>`` instead partitions the verified
  110-job reverse manifest into two node queues (``--nodes a,b`` pins them).

The ``--run-cell-key`` path is direction-agnostic: both campaigns' ``jobs.jsonl`` invoke
this module with it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import sys

import yaml

from subliminal_learning.config import run_logs_root, load_paths_local, paths_local_path
from subliminal_learning.utils.io import file_sha256

MODULE = "scripts.campaigns.build_checkpoint_studenthp"
STEPS = tuple(range(75000, 130001, 5000)) + (143000,)
TEACHER_STEPS = (75000, 125000)
TEACHER_TAG = "ep10_lr1e-4_r64a64"
HP_TAGS = (
    "ep20_lr2e-4_r8a4",
    "ep20_lr4e-4_r4a2",
    "ep20_lr2e-4_r128a16",
    "ep20_lr2e-4_r32a16",
    "ep20_lr2e-4_r64a32",
)
CORE30 = run_logs_root() / "registry" / "core30_cells.json"
STUDENT_CONFIG = Path("configs/training/lora_student_distill.yaml")
NUMBER_CONFIG = Path("configs/experiment/number_generation.yaml")
TEACHER_CONFIG = Path("configs/training/lora_template_bias.yaml")
DEFAULT_OUT = run_logs_root() / "campaigns" / "checkpoint_studenthp5_20260922"
QUEUE = "ckpt-dog-studenthp5-20260922"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Reverse (fixed-student) direction.
REVERSE_DEFAULT_OUT = run_logs_root() / "campaigns" / "checkpoint_studenthp5_reverse_20260922"
REVERSE_QUEUE = "ckpt-dog-studenthp5-reverse-20260922"
PROTOCOL = {
    "epochs": 20, "subsample_size": 10000, "subsample_seed": 42,
    "dtype": "float32", "mode": "plain", "evaluation": ["test", "off_topic"],
}
SPLIT_LABELS = ("a", "b")


# Alias kept: run_logs/campaigns/*/build.py call `forward.sha`.
sha = file_sha256


def frozen_write(path, contents):
    path = Path(path)
    if path.exists() and path.read_text() != contents:
        raise ValueError(f"refusing to change frozen campaign file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def selected_hps(core_path=CORE30):
    rows = json.loads(Path(core_path).read_text())["cells"]
    by_tag = {r["student_tag"]: r for r in rows}
    if len(rows) != 30 or len(by_tag) != 30:
        raise ValueError("core30 must contain 30 unique student recipes")
    if not set(HP_TAGS).issubset(by_tag):
        raise ValueError("selected recipe is not a member of core30")
    return [by_tag[tag] for tag in HP_TAGS]


def readable(path):
    return Path(path).is_file() and os.access(path, os.R_OK)


def local_file(recorded):
    """Map a recorded frozen-input path onto this node's checkout.

    Manifests may record repo files as absolute paths under the building node's
    checkout, which another node cannot even read; re-anchor the repo-relative tail
    here. Readable paths and non-repo paths (shared mounts) are returned unchanged.
    """
    path = Path(recorded)
    if readable(path):
        return path
    tails = []
    if not path.is_absolute():
        tails.append(path.parts)
    parts = path.parts
    if REPO_ROOT.name in parts:
        tails.append(parts[len(parts) - 1 - parts[::-1].index(REPO_ROOT.name) + 1:])
    for tail in tails:
        candidate = REPO_ROOT.joinpath(*tail)
        if readable(candidate):
            return candidate
    return path


def check_files(files):
    for path, expected in files.items():
        local = local_file(path)
        if not readable(local) or file_sha256(local) != expected:
            raise ValueError(f"frozen input changed or disappeared: {path}")


def verified_cell(campaign_path, campaign_sha256, key):
    if file_sha256(campaign_path) != campaign_sha256:
        raise ValueError("campaign identity changed")
    campaign = json.loads(Path(campaign_path).read_text())
    cells = [cell for cell in campaign["cells"] if cell["key"] == key]
    if len(cells) != 1:
        raise ValueError(f"expected one campaign cell for {key}")
    cell = cells[0]
    check_files(campaign["shared_files"])
    check_files(cell["files"])
    return cell


def record_path(path):
    """Record repo files relative to the repo root so other checkouts can verify them."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)


def _run_cell_cmd(campaign_path, digest, key):
    """The jobq command that verifies and runs one frozen cell via this module."""
    return shlex.join([
        "uv", "run", "python", "-m", MODULE,
        "--campaign", str(campaign_path), "--campaign-sha256", digest,
        "--run-cell-key", key,
    ])


# --------------------------------------------------------------------------------------
# Forward direction: core30 student-HP panel (teacher steps x student checkpoint steps).
# --------------------------------------------------------------------------------------


def build(output_root, out=DEFAULT_OUT, seeds=(1,)):
    from dataclasses import replace

    from scripts.sweeps import sweep_ckpt_distill as runner
    from subliminal_learning.models.datadecide import parse_base_model_tokens
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.students.layout import student_run_dir, student_param_tag

    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 1 for s in seeds):
        raise ValueError("student seeds must be unique positive integers")
    output_root, out = Path(output_root).resolve(), Path(out).resolve()
    configured = load_paths_local()["output_root"]
    if output_root != Path(configured).resolve():
        raise ValueError("output root disagrees with existing checkpoint runner")
    hps = selected_hps()
    template = yaml.safe_load(STUDENT_CONFIG.read_text())
    common_files = [
        CORE30, STUDENT_CONFIG, NUMBER_CONFIG, TEACHER_CONFIG,
        paths_local_path(), Path(runner._MODEL_CONFIG),
        Path(runner._EXPERIMENT_CONFIG),
    ]
    shared = {record_path(p): file_sha256(p) for p in common_files}
    pools = {}
    for step in TEACHER_STEPS:
        raw = output_root / (
            f"numbers/dog/ckpt/pythia-160m@step{step}/{TEACHER_TAG}/ftseed1/ex10-20"
        )
        filtered = raw / "modes/prefix/filtered.jsonl"
        with filtered.open() as handle:
            n_rows = sum(bool(line.strip()) for line in handle)
        if n_rows < 10000:
            raise ValueError(f"underfilled pool at step {step}: {n_rows}")
        raw_manifest = json.loads((raw / "manifest.json").read_text())
        if raw_manifest["n_raw"] != 100000:
            raise ValueError("expected canonical 100k generation pool")
        pools[str(step)] = {
            "path": str(filtered), "n_rows": n_rows,
            "files": {
                record_path(p): file_sha256(p) for p in (
                    filtered, raw / "manifest.json",
                    raw / "modes/prefix/manifest.json",
                )
            },
        }
    configs = {}
    for hp in hps:
        tag = hp["student_tag"]
        doc = copy.deepcopy(template)
        doc["lora"].update(r=hp["rank"], lora_alpha=hp["alpha"])
        doc["train"].update(learning_rate=hp["lr"], n_epochs=20)
        train_path = out / "configs" / f"{tag}.student.yaml"
        frozen_write(train_path, yaml.safe_dump(doc, sort_keys=False))
        lora, train = load_student_training(train_path)
        if student_param_tag(train, lora) != tag:
            raise ValueError(f"student HP does not round-trip: {tag}")
        sweep = {
            "base_model": "original", "dtype": "float32",
            "teacher": {"training_config": str(TEACHER_CONFIG.resolve()),
                        "param_tags": [TEACHER_TAG], "ft_seeds": [1]},
            "numbers": {"number_config": str(NUMBER_CONFIG.resolve()),
                        "filter_mode": "prefix"},
            "student": {"training_config": str(train_path), "epochs": [20],
                        "ft_seeds": list(seeds), "subsample_size": 10000,
                        "subsample_seed": 42, "modes": ["plain"]},
        }
        sweep_path = out / "configs" / f"{tag}.sweep.yaml"
        frozen_write(sweep_path, yaml.safe_dump(sweep, sort_keys=False))
        configs[tag] = (sweep_path, train_path, lora, train)
    cells = []
    for teacher_step in TEACHER_STEPS:
        for student_step in STEPS:
            for hp in hps:
                tag = hp["student_tag"]
                sweep_path, train_path, lora, train = configs[tag]
                for seed in seeds:
                    spec = parse_base_model_tokens(f"original@step{student_step}")[0]
                    run_dir = student_run_dir(
                        output_root, source="teacher", spec=spec, hub_org="EleutherAI",
                        train_cfg=replace(train, seed=seed), lora_cfg=lora, animal="dog",
                        teacher_param_tag=TEACHER_TAG, teacher_seed=1,
                        number_dataset_tag="ex10-20_prefix_n100k",
                        numbers_base_model=f"pythia-160m@step{teacher_step}",
                    )
                    # Existing cells require an explicit identity/evaluation audit; never
                    # count a baseline HP or an incomplete artifact as exact reuse.
                    if run_dir.exists():
                        raise ValueError(f"existing cell requires reuse audit: {run_dir}")
                    key = f"t{teacher_step}-s{student_step}-{tag}-seed{seed}"
                    argv = [
                        "-m", "scripts.sweeps.sweep_ckpt_distill", "run-cell",
                        "--sweep-config", str(sweep_path), "--animal", "dog",
                        "--teacher-step", str(teacher_step),
                        "--student-step", str(student_step), "--teacher-seed", "1",
                        "--param-tag", TEACHER_TAG, "--mode", "plain",
                        "--epochs", "20", "--student-seed", str(seed),
                    ]
                    files = dict(pools[str(teacher_step)]["files"])
                    files.update({record_path(p): file_sha256(p) for p in (sweep_path, train_path)})
                    cells.append({
                        "key": key, "teacher_step": teacher_step, "student_step": student_step,
                        "student_tag": tag, "student_seed": seed, "run_dir": str(run_dir),
                        "argv": argv, "files": files,
                    })
    if len({c["run_dir"] for c in cells}) != len(cells):
        raise ValueError("output collision in student HP panel")
    campaign = {
        "name": QUEUE, "animal": "dog", "model": "Pythia-160M",
        "teacher_recipe": TEACHER_TAG, "teacher_seed": 1,
        "teacher_recipe_selection": "Highest observed 125k diagonal mean and 125k-source mean across 13 recipients among the five recipes; selected using prior test outcomes.",
        "student_hps": hps, "student_seeds": seeds,
        "hp_selection": "Ordered historical core30 K_rel positions 0,7,15,22,29; design-only coverage, not a predicted checkpoint performance ordering.",
        "protocol": {"epochs": 20, "subsample_size": 10000, "subsample_seed": 42,
                     "dtype": "float32", "mode": "plain",
                     "evaluation": ["test", "off_topic"]},
        "existing_baseline": "ep10_lr1e-4_r8a32: reference only, outside core30; 10 epochs",
        "shared_files": shared, "pools": pools, "cells": cells,
        "n_jobs": len(cells), "n_reused": 0, "priority": 10000,
    }
    campaign_path = out / "campaign.json"
    frozen_write(campaign_path, json.dumps(campaign, indent=2) + "\n")
    digest = file_sha256(campaign_path)
    jobs = [{
        "key": cell["key"],
        "cmd": _run_cell_cmd(campaign_path, digest, cell["key"]),
        "cwd": str(Path.cwd()), "env": {"WANDB_MODE": "disabled"},
    } for cell in cells]
    frozen_write(out / "jobs.jsonl", "".join(json.dumps(j) + "\n" for j in jobs))
    return campaign


# --------------------------------------------------------------------------------------
# Reverse direction: fixed-student slices (teacher checkpoint steps x forward teacher
# steps as student steps), copied cell-for-cell from a submitted forward campaign.
# --------------------------------------------------------------------------------------


def load_forward(path):
    """Require the exact submitted design, not merely similarly named artifacts."""
    path = Path(path).resolve()
    receipt = json.loads(path.with_name("submission.json").read_text())
    if file_sha256(path) != receipt["campaign_sha256"]:
        raise ValueError("forward campaign differs from its submission receipt")
    if file_sha256(path.with_name("jobs.jsonl")) != receipt["jobs_sha256"]:
        raise ValueError("forward jobs differ from their submission receipt")
    campaign = json.loads(path.read_text())
    if (
        campaign["name"] != receipt["queue"]
        or campaign["animal"] != "dog"
        or campaign["model"] != "Pythia-160M"
        or campaign["teacher_recipe"] != TEACHER_TAG
        or campaign["teacher_seed"] != 1
        or campaign["student_seeds"] != [1]
        or campaign["protocol"] != PROTOCOL
        or campaign["student_hps"] != selected_hps()
    ):
        raise ValueError("forward teacher, student HPs, seeds or protocol do not match")
    expected = {
        (t, s, hp, 1)
        for t in TEACHER_STEPS
        for s in STEPS
        for hp in HP_TAGS
    }
    observed = {
        (c["teacher_step"], c["student_step"], c["student_tag"], c["student_seed"])
        for c in campaign["cells"]
    }
    if len(campaign["cells"]) != 130 or observed != expected:
        raise ValueError("forward campaign does not cover the complete expected grid")
    check_files(campaign["shared_files"])
    files = {}
    for cell in campaign["cells"]:
        files.update(cell["files"])
    check_files(files)
    return campaign, receipt


def make_reverse_cells(campaign, pools):
    """Copy exact per-HP configurations; replace only the teacher checkpoint/pool."""
    by_identity = {
        (c["teacher_step"], c["student_step"], c["student_tag"], c["student_seed"]): c
        for c in campaign["cells"]
    }
    new, covered = [], []
    for teacher_step in STEPS:
        for student_step in TEACHER_STEPS:
            for tag in HP_TAGS:
                template = by_identity[75000, student_step, tag, 1]
                cell = copy.deepcopy(template)
                cell["teacher_step"] = teacher_step
                cell["key"] = f"t{teacher_step}-s{student_step}-{tag}-seed1"
                args = cell["argv"]
                args[args.index("--teacher-step") + 1] = str(teacher_step)
                parts = list(Path(cell["run_dir"]).parts)
                index = parts.index("teacher-pythia-160m@step75000")
                parts[index] = f"teacher-pythia-160m@step{teacher_step}"
                cell["run_dir"] = str(Path(*parts))
                # The frozen training/sweep configs stay byte-identical to forward.
                cell["files"] = {
                    path: digest for path, digest in cell["files"].items()
                    if path not in campaign["pools"]["75000"]["files"]
                }
                cell["files"].update(pools[str(teacher_step)]["files"])
                previous = by_identity.get((teacher_step, student_step, tag, 1))
                if previous is not None:
                    if cell != previous:
                        raise ValueError(f"overlap identity or frozen inputs differ: {cell['key']}")
                    covered.append(cell)
                else:
                    new.append(cell)
    if len(new) != 110 or len(covered) != 20:
        raise ValueError("expected 110 new cells and 20 already-covered cells")
    all_cells = new + covered
    if len({c["run_dir"] for c in all_cells}) != 130:
        raise ValueError("reverse output collision")
    return new, covered


def split_jobs(campaign_path, jobs_path, out_dir, nodes=None):
    """Partition the verified 110-job reverse manifest into two node queues.

    `nodes` optionally names the two jobq hosts the queues are pinned to (recorded in the
    split manifest); unset leaves both queues unpinned.
    """
    if nodes is None:
        nodes = (None, None)
    if len(nodes) != 2:
        raise ValueError("nodes must name exactly two hosts")
    campaign_path = Path(campaign_path).resolve()
    jobs_path = Path(jobs_path).resolve()
    out_dir = Path(out_dir).resolve()
    campaign_sha = file_sha256(campaign_path)
    jobs_sha = file_sha256(jobs_path)
    campaign = json.loads(campaign_path.read_text())
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    expected = {c["key"] for c in campaign.get("cells", [])}
    keys = [j.get("key") for j in jobs]
    if len(jobs) != 110 or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("reverse jobs must contain exactly 110 unique campaign keys")
    by_teacher = {}
    for job in jobs:
        key = job["key"]
        teacher = int(key.split("-", 1)[0][1:])
        by_teacher.setdefault(teacher, []).append(job)
    if set(by_teacher) != set(STEPS) - set(TEACHER_STEPS) or any(len(v) != 10 for v in by_teacher.values()):
        raise ValueError("reverse jobs must contain ten cells per teacher step")
    first, second = SPLIT_LABELS
    queues = {
        label: (f"ckpt-dog-studenthp5-reverse-{label}-20260922", node)
        for label, node in zip(SPLIT_LABELS, nodes)
    }
    split = {first: [], second: []}
    for block, teacher in enumerate(sorted(by_teacher)):
        for j, job in enumerate(by_teacher[teacher]):
            label = first if (j + block) % 2 == 0 else second
            split[label].append(job)
    if any(len(v) != 55 for v in split.values()) or ({j["key"] for j in split[first]} & {j["key"] for j in split[second]}):
        raise ValueError("reverse split must be disjoint 55-job partitions")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign_sha256": campaign_sha, "jobs_sha256": jobs_sha, "total_jobs": 110, "queues": {}}
    for label, (queue, node) in queues.items():
        path = out_dir / f"jobs_{label}.jsonl"
        frozen_write(path, "".join(json.dumps(j) + "\n" for j in split[label]))
        manifest["queues"][label] = {"queue": queue, "node": node, "count": len(split[label]), "jobs_sha256": file_sha256(path)}
    frozen_write(out_dir / "split_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def reverse_build(output_root, out=REVERSE_DEFAULT_OUT, covered_campaign=None):
    output_root, out = Path(output_root).resolve(), Path(out).resolve()
    source = Path(covered_campaign or DEFAULT_OUT / "campaign.json").resolve()
    previous, receipt = load_forward(source)
    if not all(Path(c["run_dir"]).is_relative_to(output_root) for c in previous["cells"]):
        raise ValueError("output root differs from forward campaign")
    pools = {}
    for step in STEPS:
        raw = output_root / (
            f"numbers/dog/ckpt/pythia-160m@step{step}/{TEACHER_TAG}"
            "/ftseed1/ex10-20"
        )
        filtered = raw / "modes/prefix/filtered.jsonl"
        with filtered.open() as handle:
            count = sum(bool(line.strip()) for line in handle)
        if count < 10000:
            raise ValueError(f"underfilled teacher pool at {step}")
        if json.loads((raw / "manifest.json").read_text())["n_raw"] != 100000:
            raise ValueError(f"noncanonical raw pool at {step}")
        pools[str(step)] = {
            "path": str(filtered), "n_rows": count,
            "files": {str(p): file_sha256(p) for p in (
                filtered, raw / "manifest.json", raw / "modes/prefix/manifest.json",
            )},
        }
    cells, covered = make_reverse_cells(previous, pools)
    for cell in cells:
        if Path(cell["run_dir"]).exists():
            raise ValueError(f"nonoverlap output requires reuse audit: {cell['run_dir']}")
    campaign = {
        **copy.deepcopy(previous),
        "name": REVERSE_QUEUE,
        "direction": "fixed-student",
        "teacher_steps": list(STEPS),
        "student_steps": list(TEACHER_STEPS),
        "pools": pools, "cells": cells,
        "n_jobs": len(cells), "n_reused": 0,
        "n_total_cells": len(cells) + len(covered),
        "n_already_covered": len(covered),
        "covered_campaign": str(source),
        "covered_campaign_sha256": receipt["campaign_sha256"],
        "covered_queue": previous["name"],
        "covered_cells": [
            {
                "key": c["key"], "run_dir": c["run_dir"],
                "teacher_step": c["teacher_step"], "student_step": c["student_step"],
                "student_tag": c["student_tag"], "student_seed": c["student_seed"],
                "coverage": "already registered in forward queue; not assumed completed",
            }
            for c in covered
        ],
    }
    campaign["shared_files"][str(source)] = receipt["campaign_sha256"]
    path = out / "campaign.json"
    frozen_write(path, json.dumps(campaign, indent=2) + "\n")
    digest = file_sha256(path)
    jobs = [
        {
            "key": c["key"],
            "cmd": _run_cell_cmd(path, digest, c["key"]),
            "cwd": str(Path.cwd()), "env": {"WANDB_MODE": "disabled"},
        }
        for c in cells
    ]
    frozen_write(out / "jobs.jsonl", "".join(json.dumps(j) + "\n" for j in jobs))
    return campaign


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--direction", choices=("forward", "reverse"), default="forward",
        help="forward = core30 student-HP panel (default); reverse = fixed-student slices "
             "built from a submitted forward campaign",
    )
    parser.add_argument("--output-root", type=Path, help="required when building a campaign")
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"campaign directory (default: {DEFAULT_OUT} for forward, "
             f"{REVERSE_DEFAULT_OUT} for reverse)",
    )
    cell = parser.add_argument_group("run one frozen cell (either direction's jobs.jsonl)")
    cell.add_argument("--campaign", type=Path)
    cell.add_argument("--campaign-sha256")
    cell.add_argument("--run-cell-key")
    cell.add_argument("--verify-only", action="store_true",
                      help="print the verified cell instead of exec'ing the runner")
    fwd = parser.add_argument_group("forward")
    fwd.add_argument("--student-seeds", type=int, nargs="+", default=[1])
    rev = parser.add_argument_group("reverse")
    rev.add_argument("--covered-campaign", type=Path,
                     help="submitted forward campaign.json (default: the forward --out)")
    rev.add_argument("--split-only", action="store_true",
                     help="partition a verified reverse jobs.jsonl into two node queues")
    rev.add_argument("--jobs", type=Path, help="reverse jobs.jsonl to split (with --split-only)")
    rev.add_argument("--nodes", default=None,
                     help="comma-separated pair of jobq hosts to pin the two split queues to (default: unpinned)")
    args = parser.parse_args()

    if args.run_cell_key:
        if not args.campaign or not args.campaign_sha256:
            parser.error("--run-cell-key requires campaign path and SHA256")
        cell = verified_cell(args.campaign, args.campaign_sha256, args.run_cell_key)
        if args.verify_only:
            print(json.dumps(cell, indent=2))
        else:
            os.execv(sys.executable, [sys.executable, *cell["argv"]])
        return

    if args.direction == "forward":
        out = args.out if args.out is not None else DEFAULT_OUT
        if not args.output_root:
            parser.error("building requires --output-root")
        campaign = build(args.output_root, out, args.student_seeds)
        print(json.dumps({"jobs": campaign["n_jobs"], "reused": campaign["n_reused"],
                          "out": str(out)}, indent=2))
        return

    out = args.out if args.out is not None else REVERSE_DEFAULT_OUT
    nodes = tuple(n.strip() for n in args.nodes.split(",")) if args.nodes else None
    if args.split_only:
        if not args.jobs:
            parser.error("--split-only requires --jobs")
        print(json.dumps(split_jobs(
            args.covered_campaign or DEFAULT_OUT / "campaign.json", args.jobs, out, nodes=nodes,
        ), indent=2))
        return
    if not args.output_root:
        parser.error("building requires --output-root")
    campaign = reverse_build(args.output_root, out, args.covered_campaign)
    print(json.dumps({
        "jobs": campaign["n_jobs"], "total_cells": campaign["n_total_cells"],
        "already_covered": campaign["n_already_covered"], "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
