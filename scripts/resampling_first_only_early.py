"""Build early-teacher first-number replacement controls from mature IID references.

Only the first integer is replaced; early-teacher prompts and continuations remain.
``prepare`` freezes three 10k pools and 54 jobs without submitting GPU work.
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
from subliminal_learning.numbers.resampling import _nums, pool_stats
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

N = 10000
ARM = "step75000_firstonly-step125000"
CAMPAIGN = "sl-first-only-early-20260922"
DEFAULT_CAMPAIGN_DIR = controls.ROOT / "run_logs/sl_first_only_early_20260922"
RECIPIENTS = (75000, 85000, 95000, 105000, 115000, 125000)
POOL_SEEDS = (0, 1, 2)


def register_arm() -> None:
    spec = controls.ArmSpec("step75000", "firstonly-step125000", "early", "mature")
    if ARM in controls.ARM_SPECS and controls.ARM_SPECS[ARM] != spec:
        raise ValueError("first-only arm identity conflicts with registered arm")
    controls.ARM_SPECS[ARM] = spec


def input_paths(root: Path, pool_seed: int = 0) -> dict[str, Path]:
    return {
        "carrier": controls.home_pool_dir(root, "step75000")
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
        or manifest["filtered_sha256"] != controls._sha256(path)
    ):
        raise ValueError(f"parent hash or row-count mismatch: {path}")
    return data, {
        "filtered_path": str(path),
        "filtered_sha256": controls._sha256(path),
        "manifest_sha256": controls._sha256(manifest_path),
    }


def training_row(row: dict) -> dict:
    return {
        "prompt": row["prompt"],
        "completion": row["completion"],
        "numbers": list(_nums(row)),
    }


def replace_first_from_reference(
    carriers: list[dict], references: list[dict]
) -> list[dict]:
    """Use reference first strings; reference prompts/counts may differ from carriers."""
    if not carriers or len(carriers) != len(references):
        raise ValueError("expected nonempty carrier and reference pools of equal size")
    result = []
    for i, (carrier, reference) in enumerate(zip(carriers, references, strict=True)):
        for label, row in (("carrier", carrier), ("reference", reference)):
            if not isinstance(row.get("prompt"), str) or not isinstance(
                row.get("completion"), str
            ):
                raise ValueError(f"malformed {label} text at row {i}")
            nums = list(_nums(row))
            rendered = re.findall(r"\d+", row["completion"])
            if not nums or [int(v) for v in rendered] != nums:
                raise ValueError(f"{label} completion/number mismatch at row {i}")
        old = re.search(r"\d+", carrier["completion"])
        new = re.search(r"\d+", reference["completion"])
        completion = (
            carrier["completion"][: old.start()]
            + new.group()
            + carrier["completion"][old.end() :]
        )
        result.append(
            {
                "prompt": carrier["prompt"],
                "completion": completion,
                "numbers": [int(new.group()), *list(_nums(carrier))[1:]],
            }
        )
    return result


def build(root: Path, pool_seed: int = 0) -> tuple[Path, dict]:
    if pool_seed not in POOL_SEEDS:
        raise ValueError("unsupported pool seed")
    register_arm()
    paths = input_paths(root, pool_seed)
    carriers, carrier_parent = read_parent(paths["carrier"])
    paired, iid_parent = read_parent(paths["paired_iid"])
    parents = {"carrier": carrier_parent, "paired_iid": iid_parent}
    result = replace_first_from_reference(carriers, paired)
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
            "prompt_unchanged": N,
            "reference_context_may_differ": True,
            "number_count_unchanged": N,
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
            "source": "teacher",
            "n_filtered": N,
            "filtered_sha256": digest,
            "parents": parents,
            "diagnostics": diagnostics,
            "pool_stats": json.loads(json.dumps(pool_stats(result))),
        }
        if (
            controls._sha256(filtered) != digest
            or any(manifest.get(key) != value for key, value in expected.items())
            or manifest.get("params", {}).get("generation", {}).get("source")
            != "teacher_first_number_replacement"
        ):
            raise ValueError(
                "first-only pool or parent provenance drift; refusing to overwrite"
            )
        return out, manifest
    identity = replace(controls.identity_of_arm(ARM, root), campaign=CAMPAIGN)
    manifest = controls.write_pool(result, out, paths, diagnostics, root, identity)
    manifest["params"]["generation"]["source"] = "teacher_first_number_replacement"
    atomic_write_json(manifest_path, manifest)
    return out, json.loads(manifest_path.read_text())


def freeze_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"frozen campaign input drift: {path}")
        return
    atomic_write_text(path, payload)


def prepare(root: Path, campaign_dir: Path) -> dict:
    root, campaign_dir = root.absolute(), campaign_dir.absolute()
    pools = {}
    for draw in POOL_SEEDS:
        pool, manifest = build(root, draw)
        pools[str(draw)] = {
            "path": str(pool),
            "filtered_sha256": manifest["filtered_sha256"],
            "manifest_sha256": controls._sha256(pool / "manifest.json"),
            "parents": manifest["parents"],
        }
    jobs = []
    for draw in POOL_SEEDS:
        for recipient in RECIPIENTS:
            for seed in controls.STUDENT_SEEDS:
                jobs.append(
                    {
                        "key": f"{ARM}-p{draw}-r{recipient}-s{seed}",
                        "cmd": shlex.join(
                            [
                                "uv",
                                "run",
                                "python",
                                "-m",
                                "scripts.resampling_first_only_early",
                                "run",
                                "--output-root",
                                str(root),
                                "--campaign-dir",
                                str(campaign_dir),
                                "--recipient",
                                str(recipient),
                                "--student-seed",
                                str(seed),
                                "--pool-seed",
                                str(draw),
                            ]
                        ),
                        "cwd": str(controls.ROOT),
                        "env": {"WANDB_MODE": "disabled"},
                    }
                )
    if len(jobs) != 54 or len({j["key"] for j in jobs}) != 54:
        raise ValueError("expected 54 unique jobs")
    training = (controls.DEFAULT_CAMPAIGN_DIR / "student_training.yaml").read_text()
    job_text = "".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs)
    campaign = {
        "campaign": CAMPAIGN,
        "schema_version": 1,
        "animal": "dog",
        "arm": ARM,
        "output_root": str(root),
        "carrier_checkpoint": 75000,
        "profile_checkpoint": 125000,
        "teacher_recipe": controls.TEACHER_TAG,
        "teacher_seed": 1,
        "recipients": list(RECIPIENTS),
        "pool_seeds": list(POOL_SEEDS),
        "student_seeds": list(controls.STUDENT_SEEDS),
        "new_gpu_jobs": 54,
        "pools": pools,
        "student_training_sha256": hashlib.sha256(training.encode()).hexdigest(),
        "jobs_sha256": hashlib.sha256(job_text.encode()).hexdigest(),
        "reference_context": "First strings match saved mature IID; early carrier prompts/counts/continuations differ.",
    }
    payloads = {
        "campaign.json": json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        "student_training.yaml": training,
        "jobs_distill.jsonl": job_text,
    }
    # Check the whole campaign before writing any of its inputs.
    for name, payload in payloads.items():
        path = campaign_dir / name
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"frozen campaign input drift: {path}")
    for name, payload in payloads.items():
        freeze_text(campaign_dir / name, payload)
    return campaign


def validate_run(
    root: Path, campaign_dir: Path, student_seed: int, recipient: int, pool_seed: int
) -> None:
    campaign = json.loads((campaign_dir / "campaign.json").read_text())
    expected = {
        "campaign": CAMPAIGN,
        "schema_version": 1,
        "animal": "dog",
        "arm": ARM,
        "output_root": str(root.absolute()),
        "carrier_checkpoint": 75000,
        "profile_checkpoint": 125000,
        "teacher_recipe": controls.TEACHER_TAG,
        "teacher_seed": 1,
        "recipients": list(RECIPIENTS),
        "pool_seeds": list(POOL_SEEDS),
        "student_seeds": list(controls.STUDENT_SEEDS),
        "new_gpu_jobs": 54,
    }
    if any(campaign.get(k) != v for k, v in expected.items()):
        raise ValueError("campaign identity drift")
    if (
        student_seed not in controls.STUDENT_SEEDS
        or recipient not in RECIPIENTS
        or pool_seed not in POOL_SEEDS
    ):
        raise ValueError("job is outside campaign manifest")
    for name, key in (
        ("student_training.yaml", "student_training_sha256"),
        ("jobs_distill.jsonl", "jobs_sha256"),
    ):
        if controls._sha256(campaign_dir / name) != campaign[key]:
            raise ValueError(f"campaign input drift: {name}")
    register_arm()
    pool = controls.pool_dir(root, ARM, pool_seed)
    frozen = campaign["pools"][str(pool_seed)]
    if str(pool.absolute()) != frozen["path"]:
        raise ValueError("campaign pool path drift")
    for name, key in (
        ("filtered.jsonl", "filtered_sha256"),
        ("manifest.json", "manifest_sha256"),
    ):
        if controls._sha256(pool / name) != frozen[key]:
            raise ValueError(f"campaign pool drift: {name}")
    manifest = json.loads((pool / "manifest.json").read_text())
    if manifest["parents"] != frozen["parents"]:
        raise ValueError("campaign parent drift")
    for parent in frozen["parents"].values():
        path = Path(parent["filtered_path"])
        if (
            controls._sha256(path) != parent["filtered_sha256"]
            or controls._sha256(path.parent / "manifest.json")
            != parent["manifest_sha256"]
        ):
            raise ValueError("campaign parent hash drift")


def run(
    root: Path,
    campaign_dir: Path,
    student_seed: int,
    recipient: int = 125000,
    pool_seed: int = 0,
) -> Path:
    validate_run(root, campaign_dir, student_seed, recipient, pool_seed)
    return controls.run_cell(
        root, ARM, pool_seed, recipient, student_seed, campaign_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--student-seed", type=int, choices=controls.STUDENT_SEEDS)
    parser.add_argument("--recipient", type=int, choices=RECIPIENTS)
    parser.add_argument("--pool-seed", type=int, choices=POOL_SEEDS)
    args = parser.parse_args()
    if args.command == "prepare":
        campaign = prepare(args.output_root, args.campaign_dir)
        print(
            json.dumps(
                {"jobs": campaign["new_gpu_jobs"], "pools": len(campaign["pools"])}
            )
        )
    elif args.command == "validate":
        for draw in POOL_SEEDS:
            for recipient in RECIPIENTS:
                for seed in controls.STUDENT_SEEDS:
                    validate_run(
                        args.output_root, args.campaign_dir, seed, recipient, draw
                    )
        print(json.dumps({"validated_jobs": 54, "gpu_loaded": False}))
    else:
        if any(v is None for v in (args.student_seed, args.recipient, args.pool_seed)):
            parser.error("run requires --student-seed, --recipient, and --pool-seed")
        print(
            run(
                args.output_root,
                args.campaign_dir,
                args.student_seed,
                args.recipient,
                args.pool_seed,
            )
        )


if __name__ == "__main__":
    main()
