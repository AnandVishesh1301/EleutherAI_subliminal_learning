"""Build and run the dog-only control paired to the saved mature IID pool.

Only the first integer is replaced. Pool seed 0 and the 125k recipient are the default;
``emit`` can add explicit recipient sets and paired IID controls without submitting jobs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shlex

from scripts import resampling_controls as controls
from subliminal_learning.numbers.resampling import (
    _nums,
    pool_stats,
    replace_first_number_from_pair,
)
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text, file_sha256

N = 10000
ARM = "step125000_synthfirstonly"
CAMPAIGN = "sl-first-only-20260922"
DEFAULT_CAMPAIGN_DIR = controls.ROOT / "run_logs/campaigns/sl_first_only_20260922"


def register_arm() -> None:
    spec = controls.ArmSpec("step125000", "synthfirstonly", "carriers", "mature")
    if ARM in controls.ARM_SPECS and controls.ARM_SPECS[ARM] != spec:
        raise ValueError("first-only arm identity conflicts with registered arm")
    controls.ARM_SPECS[ARM] = spec


def input_paths(root: Path, pool_seed: int = 0) -> dict[str, Path]:
    return {
        "carrier": controls.home_pool_dir(root, "base75000")
        / f"ablations/uniform/poolseed{pool_seed}/filtered.jsonl",
        "paired_iid": controls.home_pool_dir(root, "step125000")
        / f"ablations/synthiid/poolseed{pool_seed}/filtered.jsonl",
    }


def read_parent(path: Path) -> tuple[list[dict], dict]:
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        len(data) != N
        or manifest["n_filtered"] != N
        or manifest["filtered_sha256"] != file_sha256(path)
    ):
        raise ValueError(f"parent hash or row-count mismatch: {path}")
    return data, {
        "filtered_path": str(path),
        "filtered_sha256": file_sha256(path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def training_row(row: dict) -> dict:
    return {
        "prompt": row["prompt"],
        "completion": row["completion"],
        "numbers": list(_nums(row)),
    }


def build(root: Path, pool_seed: int = 0) -> tuple[Path, dict]:
    register_arm()
    paths = input_paths(root, pool_seed)
    carriers, carrier_parent = read_parent(paths["carrier"])
    paired, iid_parent = read_parent(paths["paired_iid"])
    parents = {"carrier": carrier_parent, "paired_iid": iid_parent}
    for i, (carrier, iid) in enumerate(zip(carriers, paired)):
        if re.sub(r"\d+", "#", carrier["completion"]) != re.sub(
            r"\d+", "#", iid["completion"]
        ):
            raise ValueError(f"carrier/IID formatting mismatch at row {i}")
    result = [
        training_row(row) for row in replace_first_number_from_pair(carriers, paired)
    ]
    for i, (row, carrier, iid) in enumerate(zip(result, carriers, paired)):
        first = re.search(r"\d+", row["completion"])
        old = re.search(r"\d+", carrier["completion"])
        reference = re.search(r"\d+", iid["completion"])
        if (
            first is None
            or old is None
            or reference is None
            or first.group() != reference.group()
            or row["completion"][: first.start()]
            != carrier["completion"][: old.start()]
            or row["completion"][first.end() :] != carrier["completion"][old.end() :]
            or _nums(row)[1:] != _nums(carrier)[1:]
        ):
            raise ValueError(f"first-only replacement invariant failed at row {i}")
    diagnostics = {
        "transform": ARM,
        "pool_seed": pool_seed,
        "replacement_positions": [0],
        "sampling": "saved_carrier_rows_once_with_paired_first_number_replacement",
        "pairing": {
            "rows": N,
            "prompt_matches": N,
            "number_count_matches": N,
            "exact_first_number_matches": N,
            "unchanged_later_text": N,
            "carrier_row_positions_used_once": N,
            "singleton_rows": sum(len(_nums(row)) == 1 for row in carriers),
            "multi_number_rows": sum(len(_nums(row)) > 1 for row in carriers),
        },
    }
    for label, data in (("carrier", carriers), ("first_only", result), ("iid", paired)):
        diagnostics[f"unique_{label}_training_rows"] = len(
            {json.dumps(training_row(row), sort_keys=True) for row in data}
        )
        diagnostics[f"unique_{label}_completions"] = len(
            {row["completion"] for row in data}
        )
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in result)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    out = controls.pool_dir(root, ARM, pool_seed)
    filtered, manifest_path = out / "filtered.jsonl", out / "manifest.json"
    if filtered.exists() or manifest_path.exists():
        if not filtered.is_file() or not manifest_path.is_file():
            raise ValueError("incomplete first-only pool; refusing to overwrite")
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "campaign": CAMPAIGN,
            "transform": ARM,
            "source": "rng",
            "n_filtered": N,
            "filtered_sha256": digest,
            "parents": parents,
            "diagnostics": diagnostics,
            "pool_stats": json.loads(json.dumps(pool_stats(result))),
        }
        if (
            file_sha256(filtered) != digest
            or any(manifest.get(key) != value for key, value in expected.items())
            or manifest.get("params", {}).get("generation", {}).get("source")
            != "paired_first_number_replacement"
        ):
            raise ValueError(
                "first-only pool or parent provenance drift; refusing to overwrite"
            )
        return out, manifest
    identity = replace(controls.identity_of_arm(ARM, root), campaign=CAMPAIGN)
    manifest = controls.write_pool(result, out, paths, diagnostics, root, identity)
    manifest["params"]["generation"]["source"] = "paired_first_number_replacement"
    atomic_write_json(manifest_path, manifest)
    return out, json.loads(manifest_path.read_text())


def emit(
    root: Path,
    campaign_dir: Path,
    recipients=(125000,),
    arms=(ARM,),
    iid_recipients=(),
    pool_seed=0,
) -> Path:
    supported = {75000, 85000, 95000, 105000, 115000, 125000}
    if not set(recipients) <= supported or not set(iid_recipients) <= supported:
        raise ValueError("unsupported recipient checkpoint")
    allowed = {ARM, "step125000_synthiid"}
    if not set(arms) <= allowed:
        raise ValueError("unsupported first-only control arm")
    register_arm()
    pool, manifest = build(root, pool_seed)
    iid_path = input_paths(root, pool_seed)["paired_iid"]
    iid_manifest = json.loads((iid_path.parent / "manifest.json").read_text())
    iid_sha = file_sha256(iid_path)
    if iid_sha != iid_manifest.get("filtered_sha256"):
        raise ValueError("IID control pool drift")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    training_source = controls.DEFAULT_CAMPAIGN_DIR / "student_training.yaml"
    training_target = campaign_dir / "student_training.yaml"
    training_text = training_source.read_text()
    if training_target.exists() and training_target.read_text() != training_text:
        raise ValueError("campaign training configuration drift")
    atomic_write_text(training_target, training_text)
    jobs = []
    arm_recipients = {
        ARM: tuple(recipients),
        "step125000_synthiid": tuple(iid_recipients),
    }
    for arm in arms:
        for recipient in arm_recipients.get(arm, ()):
            for seed in controls.STUDENT_SEEDS:
                command = shlex.join(
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "scripts.resampling_first_only",
                        "run",
                        "--arm",
                        arm,
                        "--recipient",
                        str(recipient),
                        "--output-root",
                        str(root),
                        "--campaign-dir",
                        str(campaign_dir),
                        "--student-seed",
                        str(seed),
                        "--pool-seed",
                        str(pool_seed),
                    ]
                )
                jobs.append(
                    {
                        "key": f"{arm}-p{pool_seed}-r{recipient}-s{seed}",
                        "cmd": command,
                        "cwd": str(controls.ROOT),
                        "env": {"WANDB_MODE": "disabled"},
                    }
                )
    if not jobs or len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("campaign must contain nonempty, unique jobs")
    job_text = "".join(json.dumps(job) + "\n" for job in jobs)
    job_path = campaign_dir / "jobs_distill.jsonl"
    if job_path.exists() and job_path.read_text() != job_text:
        raise ValueError("existing first-only job manifest differs")
    atomic_write_text(job_path, job_text)
    atomic_write_json(
        campaign_dir / "campaign.json",
        {
            "campaign": CAMPAIGN,
            "animal": "dog",
            "recipients": list(recipients),
            "iid_recipients": list(iid_recipients),
            "arms": [arm for arm in arms if arm_recipients.get(arm)],
            "pool_seed": pool_seed,
            "student_seeds": list(controls.STUDENT_SEEDS),
            "new_gpu_jobs": len(jobs),
            "pool": str(pool),
            "pool_sha256": manifest["filtered_sha256"],
            "student_training_sha256": file_sha256(training_target),
            "control_pool": str(iid_path),
            "control_pool_sha256": iid_sha,
            "control_manifest_sha256": file_sha256(
                iid_path.parent / "manifest.json"
            ),
        },
    )
    return job_path


def run(
    root: Path,
    campaign_dir: Path,
    student_seed: int,
    arm: str = ARM,
    recipient: int = 125000,
    pool_seed: int = 0,
) -> Path:
    campaign = json.loads((campaign_dir / "campaign.json").read_text())
    if campaign.get("pool_seed", 0) != pool_seed:
        raise ValueError("pool seed is outside campaign manifest")
    if arm == ARM:
        _, manifest = build(root, pool_seed)
    elif arm == "step125000_synthiid":
        register_arm()
        iid = input_paths(root, pool_seed)["paired_iid"]
        manifest = json.loads((iid.parent / "manifest.json").read_text())
        campaign_preview = (
            json.loads((campaign_dir / "campaign.json").read_text())
            if (campaign_dir / "campaign.json").exists()
            else {}
        )
        if file_sha256(iid) != campaign_preview.get(
            "control_pool_sha256"
        ) or file_sha256(iid.parent / "manifest.json") != campaign_preview.get(
            "control_manifest_sha256"
        ):
            raise ValueError("IID control pool drift")
    else:
        raise ValueError("unsupported control arm")
    if student_seed not in campaign["student_seeds"]:
        raise ValueError("student seed is outside campaign manifest")
    if campaign["student_training_sha256"] != file_sha256(
        campaign_dir / "student_training.yaml"
    ):
        raise ValueError("campaign input drift")
    allowed_recipients = (
        campaign.get("recipients", [125000])
        if arm == ARM
        else campaign.get("iid_recipients", [])
    )
    if arm not in campaign.get("arms", [ARM]) or recipient not in allowed_recipients:
        raise ValueError("job is outside campaign manifest")
    if arm == ARM and campaign.get("pool_sha256") != manifest["filtered_sha256"]:
        raise ValueError("campaign first-only pool drift")
    return controls.run_cell(
        root, arm, pool_seed, recipient, student_seed, campaign_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "emit", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--student-seed", type=int, choices=controls.STUDENT_SEEDS)
    parser.add_argument("--recipient", type=int, nargs="+")
    parser.add_argument("--iid-recipient", type=int, nargs="+")
    parser.add_argument("--pool-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--arm", nargs="+", choices=(ARM, "step125000_synthiid"), default=[ARM]
    )
    args = parser.parse_args()
    if args.command == "build":
        path, manifest = build(args.output_root, args.pool_seed)
        print(
            json.dumps(
                {
                    "pool": str(path),
                    "sha256": manifest["filtered_sha256"],
                    "diagnostics": manifest["diagnostics"],
                }
            )
        )
    elif args.command == "emit":
        print(
            emit(
                args.output_root,
                args.campaign_dir,
                tuple(args.recipient or [125000]),
                tuple(args.arm),
                tuple(args.iid_recipient or []),
                args.pool_seed,
            )
        )
    elif args.student_seed is None:
        parser.error("run requires --student-seed")
    else:
        recipients = args.recipient or [125000]
        if len(recipients) != 1 or len(args.arm) != 1:
            parser.error("run accepts exactly one recipient and one arm")
        if args.iid_recipient is not None:
            parser.error("--iid-recipient is only for emit")
        print(
            run(
                args.output_root,
                args.campaign_dir,
                args.student_seed,
                args.arm[0],
                recipients[0],
                args.pool_seed,
            )
        )


if __name__ == "__main__":
    main()
