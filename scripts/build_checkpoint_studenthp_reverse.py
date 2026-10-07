"""Add fixed-student checkpoint slices using a frozen forward HP campaign."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex

from scripts import build_checkpoint_studenthp_preflight as forward

DEFAULT_OUT = Path("run_logs/campaigns/checkpoint_studenthp5_reverse_20260922")
QUEUE = "ckpt-dog-studenthp5-reverse-20260922"
PROTOCOL = {
    "epochs": 20, "subsample_size": 10000, "subsample_seed": 42,
    "dtype": "float32", "mode": "plain", "evaluation": ["test", "off_topic"],
}


def load_forward(path):
    """Require the exact submitted design, not merely similarly named artifacts."""
    path = Path(path).resolve()
    receipt = json.loads(path.with_name("submission.json").read_text())
    if forward.sha(path) != receipt["campaign_sha256"]:
        raise ValueError("forward campaign differs from its submission receipt")
    if forward.sha(path.with_name("jobs.jsonl")) != receipt["jobs_sha256"]:
        raise ValueError("forward jobs differ from their submission receipt")
    campaign = json.loads(path.read_text())
    if (
        campaign["name"] != receipt["queue"]
        or campaign["animal"] != "dog"
        or campaign["model"] != "Pythia-160M"
        or campaign["teacher_recipe"] != forward.TEACHER_TAG
        or campaign["teacher_seed"] != 1
        or campaign["student_seeds"] != [1]
        or campaign["protocol"] != PROTOCOL
        or campaign["student_hps"] != forward.selected_hps()
    ):
        raise ValueError("forward teacher, student HPs, seeds or protocol do not match")
    expected = {
        (t, s, hp, 1)
        for t in forward.TEACHER_STEPS
        for s in forward.STEPS
        for hp in forward.HP_TAGS
    }
    observed = {
        (c["teacher_step"], c["student_step"], c["student_tag"], c["student_seed"])
        for c in campaign["cells"]
    }
    if len(campaign["cells"]) != 130 or observed != expected:
        raise ValueError("forward campaign does not cover the complete expected grid")
    forward.check_files(campaign["shared_files"])
    files = {}
    for cell in campaign["cells"]:
        files.update(cell["files"])
    forward.check_files(files)
    return campaign, receipt


def make_reverse_cells(campaign, pools):
    """Copy exact per-HP configurations; replace only the teacher checkpoint/pool."""
    by_identity = {
        (c["teacher_step"], c["student_step"], c["student_tag"], c["student_seed"]): c
        for c in campaign["cells"]
    }
    new, covered = [], []
    for teacher_step in forward.STEPS:
        for student_step in forward.TEACHER_STEPS:
            for tag in forward.HP_TAGS:
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


def split_jobs(campaign_path, jobs_path, out_dir):
    """Partition the verified 110-job reverse manifest into two node queues."""
    campaign_path = Path(campaign_path).resolve()
    jobs_path = Path(jobs_path).resolve()
    out_dir = Path(out_dir).resolve()
    campaign_sha = forward.sha(campaign_path)
    jobs_sha = forward.sha(jobs_path)
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
    if set(by_teacher) != set(forward.STEPS) - set(forward.TEACHER_STEPS) or any(len(v) != 10 for v in by_teacher.values()):
        raise ValueError("reverse jobs must contain ten cells per teacher step")
    queues = {
        "soar1": ("ckpt-dog-studenthp5-reverse-soar1-20260922", "soar-subliminal-1"),
        "soar4": ("ckpt-dog-studenthp5-reverse-soar4-20260922", "soar-subliminal-4"),
    }
    split = {"soar1": [], "soar4": []}
    for block, teacher in enumerate(sorted(by_teacher)):
        for j, job in enumerate(by_teacher[teacher]):
            node = "soar1" if (j + block) % 2 == 0 else "soar4"
            split[node].append(job)
    if any(len(v) != 55 for v in split.values()) or ({j["key"] for j in split["soar1"]} & {j["key"] for j in split["soar4"]}):
        raise ValueError("reverse split must be disjoint 55-job partitions")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign_sha256": campaign_sha, "jobs_sha256": jobs_sha, "total_jobs": 110, "queues": {}}
    for label, (queue, node) in queues.items():
        path = out_dir / f"jobs_{label}.jsonl"
        forward.frozen_write(path, "".join(json.dumps(j) + "\n" for j in split[label]))
        manifest["queues"][label] = {"queue": queue, "node": node, "count": len(split[label]), "jobs_sha256": forward.sha(path)}
    forward.frozen_write(out_dir / "split_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(output_root, out=DEFAULT_OUT, covered_campaign=None):
    output_root, out = Path(output_root).resolve(), Path(out).resolve()
    source = Path(covered_campaign or forward.DEFAULT_OUT / "campaign.json").resolve()
    previous, receipt = load_forward(source)
    if not all(Path(c["run_dir"]).is_relative_to(output_root) for c in previous["cells"]):
        raise ValueError("output root differs from forward campaign")
    pools = {}
    for step in forward.STEPS:
        raw = output_root / (
            f"numbers/dog/ckpt/pythia-160m@step{step}/{forward.TEACHER_TAG}"
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
            "files": {str(p): forward.sha(p) for p in (
                filtered, raw / "manifest.json", raw / "modes/prefix/manifest.json",
            )},
        }
    cells, covered = make_reverse_cells(previous, pools)
    for cell in cells:
        if Path(cell["run_dir"]).exists():
            raise ValueError(f"nonoverlap output requires reuse audit: {cell['run_dir']}")
    campaign = {
        **copy.deepcopy(previous),
        "name": QUEUE,
        "direction": "fixed-student",
        "teacher_steps": list(forward.STEPS),
        "student_steps": list(forward.TEACHER_STEPS),
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
    forward.frozen_write(path, json.dumps(campaign, indent=2) + "\n")
    digest = forward.sha(path)
    jobs = [
        {
            "key": c["key"],
            "cmd": shlex.join([
                "uv", "run", "python", "-m",
                "scripts.build_checkpoint_studenthp_preflight",
                "--campaign", str(path), "--campaign-sha256", digest,
                "--run-cell-key", c["key"],
            ]),
            "cwd": str(Path.cwd()), "env": {"WANDB_MODE": "disabled"},
        }
        for c in cells
    ]
    forward.frozen_write(out / "jobs.jsonl", "".join(json.dumps(j) + "\n" for j in jobs))
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--covered-campaign", type=Path)
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--jobs", type=Path)
    args = parser.parse_args()
    if args.split_only:
        if not args.jobs:
            parser.error("--split-only requires --jobs")
        print(json.dumps(split_jobs(args.covered_campaign or forward.DEFAULT_OUT / "campaign.json", args.jobs, args.out), indent=2))
        return
    if not args.output_root:
        parser.error("building requires --output-root")
    campaign = build(args.output_root, args.out, args.covered_campaign)
    print(json.dumps({
        "jobs": campaign["n_jobs"], "total_cells": campaign["n_total_cells"],
        "already_covered": campaign["n_already_covered"], "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
