"""Build matched untuned-sender controls for a directed seed factorial.

Three families share the design: `full-seed` (orig/fs1/fs2/fs3, the 2026-09-14
campaign), `data-seed` (orig/ds1/ds2/ds3) and `same-model` (the orig/fs1/fs2/fs3
diagonal on focus20). This only writes campaign inputs.
Submit the generated jobs through jobq with the dependencies recorded in
campaign.json; no GPU work runs in this builder.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ANIMALS = ("dog", "cat", "lion", "pig")
DEPENDENCIES = ("direction-olmo-surgery-20260912", "direction-olmo-injections-20260912")
FAMILIES = {
    "full-seed": {
        "campaign": Path("run_logs/sl_fullseed_controls_20260914"),
        "bases": {"orig": "original", **{f"fs{i}": f"full-seed{i}" for i in (1, 2, 3)}},
        "models": {"orig": "pythia-160m", **{f"fs{i}": f"pythia-160m-seed{i}" for i in (1, 2, 3)}},
        # Fully re-randomized inits are opt-in in parse_base_models; data-seed
        # models are ordinary repo tokens and must not carry the force flag.
        "force_flag": True,
        "senders_to_generate": ("fs1", "fs2", "fs3"),
        "queue_prefix": "sl-fullseed-controls",
        "date": "20260914",
        "title": "Matched full-seed controls",
        "pairs": "orig/fs1/fs2/fs3",
    },
    "data-seed": {
        "campaign": Path("run_logs/sl_dataseed_controls_20260916"),
        "bases": {"orig": "original", **{f"ds{i}": f"data-seed{i}" for i in (1, 2, 3)}},
        "models": {"orig": "pythia-160m",
                   **{f"ds{i}": f"pythia-160m-data-seed{i}" for i in (1, 2, 3)}},
        "force_flag": False,
        "senders_to_generate": ("ds1", "ds2", "ds3"),
        "queue_prefix": "sl-dataseed-controls",
        "date": "20260916",
        "title": "Matched data-seed controls",
        "pairs": "orig/ds1/ds2/ds3",
    },
    "same-model": {
        "campaign": Path("run_logs/sl_samemodel_controls_20260916"),
        "bases": {"orig": "original", **{f"fs{i}": f"full-seed{i}" for i in (1, 2, 3)}},
        "models": {"orig": "pythia-160m",
                   **{f"fs{i}": f"pythia-160m-seed{i}" for i in (1, 2, 3)}},
        # Only the re-randomized inits need the opt-in flag; orig is an ordinary token.
        "force_flag": True,
        "force_scope": "non-orig",
        # Both untuned pools (orig and fs1-3) already exist: nothing to generate.
        "senders_to_generate": (),
        "pairs_mode": "diagonal",
        "cells_subset": "focus20",
        "queue_prefix": "sl-samemodel-controls",
        "date": "20260916",
        "title": "Matched same-model controls",
        "pairs": "orig/fs1/fs2/fs3 diagonal",
    },
}


def focus20_cells(core56: dict[str, dict]) -> dict[str, dict]:
    """The designed 20-cell focus subset of core56, with core56's hyperparameters."""
    from subliminal_learning.analysis import transfer_factors as tf

    design = tf.load_core65_design(ROOT)[0]
    tags = list(design.loc[design["subset"] == "focus20", "student_tag"])
    assert len(tags) == 20, f"Expected focus20, got {len(tags)}"
    assert all(tag in core56 for tag in tags), "focus20 must be contained in core56"
    return {tag: core56[tag] for tag in tags}


def reference_cells(text: str) -> dict[str, dict]:
    """Deduplicate the exact student configurations in the treatment manifest."""
    cells = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        args = shlex.split(json.loads(line)["cmd"])
        values = {name: args[args.index(name) + 1] for name in (
            "--epochs", "--student-lr", "--student-rank", "--student-alpha", "--student-seed"
        )}
        assert values["--epochs"] == "20" and values["--student-seed"] == "1"
        lr, rank, alpha = (values[k] for k in ("--student-lr", "--student-rank", "--student-alpha"))
        tag = f"ep20_lr{lr}_r{rank}a{alpha}"
        cells[tag] = {"learning_rate": float(lr), "r": int(rank), "lora_alpha": int(alpha)}
    assert len(cells) == 56, f"Expected core56, got {len(cells)}"
    return cells


def listed_cells(text: str) -> dict[str, dict]:
    """Same core56 grid, read from the shared cell list instead of a job manifest."""
    cells = {}
    for tag in json.loads(text)["cells"]:
        match = re.fullmatch(r"ep20_lr([0-9.eE+-]+)_r(\d+)a(\d+)", tag)
        assert match, f"unexpected cell tag {tag!r}"
        cells[tag] = {"learning_rate": float(match[1]), "r": int(match[2]),
                      "lora_alpha": int(match[3])}
    assert len(cells) == 56, f"Expected core56, got {len(cells)}"
    return cells


def command(module: str, args: list[str]) -> str:
    return shlex.join(["uv", "run", "python", "-m", module, *args])


def write_jsonl(path: Path, jobs: list[dict]) -> None:
    assert len({job["key"] for job in jobs}) == len(jobs)
    path.write_text("".join(json.dumps(job) + "\n" for job in jobs))


def build(family: str, campaign_dir: Path | None = None) -> dict:
    spec = FAMILIES[family]
    campaign = campaign_dir or spec["campaign"]
    bases, models = spec["bases"], spec["models"]
    scope = spec.get("force_scope", "family")

    def force_flags(*keys: str) -> tuple[list[str], list[str]]:
        """(eval, worker) opt-in flags; `non-orig` drops them where no fs init is used."""
        if not spec["force_flag"] or (scope == "non-orig" and all(k == "orig" for k in keys)):
            return [], []
        return ["--allow-full-seed-bases"], ["--allow-full-seed-base"]
    out = ROOT / campaign
    (out / "configs").mkdir(parents=True, exist_ok=True)
    if family == "full-seed":
        cells = reference_cells(
            (ROOT / "run_logs/fsfact/dog-seed2seed-orig-fs1-core56.jobs.jsonl").read_text())
    else:
        cells = listed_cells((ROOT / "run_logs/core56_cells.json").read_text())
    if spec.get("cells_subset") == "focus20":
        cells = focus20_cells(cells)
    training = yaml.safe_load((ROOT / "configs/training/lora_student_distill.yaml").read_text())
    numbers = yaml.safe_load((ROOT / "configs/experiment/number_generation.yaml").read_text())
    assert numbers["prompt_set"]["n_prompts"] == 100000
    assert numbers["prompt_set"]["seed"] == 42
    assert (numbers["prompt_set"]["example_count_min"], numbers["prompt_set"]["example_count_max"]) == (10, 20)
    number_config = campaign / "number_generation.yaml"
    (ROOT / number_config).write_text(yaml.safe_dump(numbers, sort_keys=False))
    config_paths = {}
    for tag, hp in cells.items():
        cfg = copy.deepcopy(training)
        cfg["train"].update(learning_rate=hp["learning_rate"], n_epochs=20, seed=1)
        cfg["lora"].update(r=hp["r"], lora_alpha=hp["lora_alpha"])
        config_paths[tag] = campaign / "configs" / f"{tag}.yaml"
        (ROOT / config_paths[tag]).write_text(yaml.safe_dump(cfg, sort_keys=False))

    generation = []
    for sender in spec["senders_to_generate"]:
        force, _ = force_flags(sender)
        generation.append({
            "key": f"base-pool-{sender}",
            "cmd": command("scripts.generate_numbers", [
                "--source", "base", "--base-models", bases[sender],
                *force, "--modes", "prefix",
                "--number-config", str(number_config),
            ]),
            "cwd": str(ROOT), "mem_mib": 6000, "env": {"WANDB_MODE": "disabled"},
        })

    output_root = Path(yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())["output_root"])
    jobs, existing = [], 0
    per_direction = {}
    for sender in bases:
        for receiver in bases:
            if (sender == receiver) is (spec.get("pairs_mode") != "diagonal"):
                continue
            force, worker_force = force_flags(sender, receiver)
            trained = 0
            for tag in cells:
                common = [
                    "--source", "base", "--base-models", bases[receiver],
                    "--numbers-base-model", bases[sender], *force,
                    "--epochs", "20", "--subsample-size", "10000", "--subsample-seed", "42",
                    "--number-dataset-tag", "ex10-20_prefix_n100k", "--filter-mode", "prefix",
                    "--student-training-config", str(config_paths[tag]),
                    "--number-config", str(number_config),
                ]
                param = (output_root / "students/controls/base" / models[receiver]
                         / f"teacher-{models[sender]}" / "ex10-20_prefix_n100k" / tag)
                trained += (param / "ftseed1/train_manifest.json").exists()
                hp = cells[tag]
                # The established worker supports untuned cross-base controls. Its
                # legacy root evaluation is always dog; all other animals use the
                # namespaced evaluation CLI below and never touch that root.
                commands = [shlex.join([
                    "uv", "run", "python", "run_logs/animal_samedistill_worker.py",
                    "--source", "base", "--animal", "dog", "--base", bases[receiver],
                    "--numbers-base", bases[sender], *worker_force,
                    "--epochs", "20", "--student-seed", "1", "--modes", "plain",
                    "--student-lr", str(hp["learning_rate"]),
                    "--student-rank", str(hp["r"]), "--student-alpha", str(hp["lora_alpha"]),
                    "--sets", "test,off_topic", "--number-config", str(number_config),
                ])]
                for animal in ANIMALS:
                    commands.append(command("scripts.evaluate_student_preference", common + [
                        "--target-animal", animal, "--prompt-sets", "test,off_topic", "--no-base-control",
                    ]))
                # Evaluation's missing-student path can return successfully; assert real
                # outputs exist, so a skipped or partial job cannot be marked complete.
                verify = (
                    "import json; from pathlib import Path; "
                    f"p=Path({str(param)!r}); "
                    "assert (p/'ftseed1/train_manifest.json').is_file(); "
                    f"animals={ANIMALS!r}; "
                    "docs=[json.loads((p/'eval'/a/'preference_summary.json').read_text()) for a in animals]; "
                    "assert all(d['target_animal']==a and "
                    "{'test','off_topic'} <= {s['prompt_set'] for s in d['summaries']} "
                    "for a,d in zip(animals,docs))"
                )
                commands.append(shlex.join(["uv", "run", "python", "-c", verify]))
                jobs.append({
                    "key": f"base-{sender}-to-{receiver}-{tag}-s1",
                    "cmd": " && ".join(commands), "cwd": str(ROOT),
                    "mem_mib": 6000, "env": {"WANDB_MODE": "disabled"},
                })
            per_direction[f"{sender}-to-{receiver}"] = trained
            existing += trained
    n = len(bases)
    directions = n if spec.get("pairs_mode") == "diagonal" else n * n - n
    assert len(jobs) == directions * len(cells)
    write_jsonl(out / "generation.jobs.jsonl", generation)
    write_jsonl(out / "controls.jobs.jsonl", jobs)
    summary = {
        "directions": directions, "cells_per_direction": len(cells), "control_jobs": len(jobs),
        "existing_training_manifests": existing, "expected_new_trainings": len(jobs) - existing,
        "generation_jobs": len(generation), "animals": ANIMALS, "student_seed": 1,
        "priority": 49, "strict_deps": True, "olmo_dependencies": DEPENDENCIES,
        "generation_queue": f"{spec['queue_prefix']}-ngen-{spec['date']}",
        "control_queue": f"{spec['queue_prefix']}-{spec['date']}",
    }
    if family != "full-seed":
        summary["existing_training_manifests_by_direction"] = per_direction
    (out / "campaign.json").write_text(json.dumps(summary, indent=2) + "\n")
    if family == "full-seed":
        readme = (
            "# Matched full-seed controls\n\n"
            "Twelve directed off-diagonal orig/fs1/fs2/fs3 pairs, core56, student seed 1. "
            "Each job trains one animal-independent untuned-sender control and sequentially "
            "scores dog/cat/lion/pig on test and off_topic in eval/<animal>. Existing valid "
            "training and evaluation artifacts resume without --force; legacy root dog "
            "evaluations remain untouched. No spliced runs or treatment replicates.\n\n"
            "Both queues use priority 49 and strict dependencies on OLMo surgery and "
            "injections; the controls also depend on the pool-generation queue. "
            "Generation: 3 pools. Control cells: 672, of which "
            f"{existing} already have training manifests. Per-job memory: 6000 MiB; "
            "no per-GPU cap.\n\n"
            "Rebuild: `uv run python -m scripts.build_fullseed_control_jobs`. "
            "The builder does not submit jobs.\n"
        )
    elif family == "same-model":
        table = "\n".join(f"- {d}: {n}" for d, n in per_direction.items())
        readme = (
            f"# {spec['title']}\n\n"
            f"Four directed DIAGONAL {spec['pairs']} pairs (sender == receiver), focus20, "
            "student seed 1, 20 epochs, prefix filtering, fp32 (no tf32 anywhere): "
            f"{directions} x {len(cells)} = {len(jobs)} control jobs. focus20 is the designed "
            "20-cell subset of core56 spanning the K/K* range, so the matched same-model "
            "control covers the transfer band at a quarter of the core56 cost. Each job "
            "trains one animal-independent untuned-pool control (the sender's pool is the "
            "receiver's own base pool, so the run lands in "
            "students/controls/base/<model>/teacher-<model>/) and sequentially scores "
            "dog/cat/lion/pig on test and off_topic in eval/<animal>. Existing valid training "
            "and evaluation artifacts resume without --force. No pools are generated: the "
            "untuned orig and fs1-3 pools already exist. pig treatment is absent on fs3-to-fs3; "
            "its pig control evaluation is still produced and is harmless.\n\n"
            f"Control cells: {len(jobs)}, of which {existing} already have training "
            f"manifests:\n\n{table}\n\n"
            "Per-job memory: 6000 MiB; no per-GPU cap.\n\n"
            "Submit (the builder does not submit):\n\n"
            "```\nuv run python -m scripts.jobq submit sl-samemodel-controls-20260916 "
            "--root run_logs/queue_angela --jobs-file "
            "run_logs/sl_samemodel_controls_20260916/controls.jobs.jsonl --priority 50 "
            "--mem-mib 6000 --depends-on sl-dataseed-controls-20260916\n```\n\n"
            "Rebuild: `uv run python -m scripts.build_fullseed_control_jobs "
            f"--family {family}`.\n"
        )
    else:
        table = "\n".join(f"- {d}: {n}" for d, n in per_direction.items())
        readme = (
            f"# {spec['title']}\n\n"
            f"Twelve directed off-diagonal {spec['pairs']} pairs, core56, student seed 1, "
            "20 epochs, prefix filtering, fp32 (no tf32 anywhere). Each job trains one "
            "animal-independent untuned-sender control and sequentially scores "
            "dog/cat/lion/pig on test and off_topic in eval/<animal>. Existing valid "
            "training and evaluation artifacts resume without --force; legacy root dog "
            "evaluations remain untouched, so the already-trained orig-to-ds directions "
            "only gain the missing per-animal evaluations. No spliced runs or treatment "
            "replicates.\n\n"
            f"Generation: {len(generation)} untuned data-seed sender pools (the original's "
            "pool already exists). Control cells: "
            f"{len(jobs)}, of which {existing} already have training manifests:\n\n"
            f"{table}\n\n"
            "Per-job memory: 6000 MiB; no per-GPU cap.\n\n"
            "Rebuild: `uv run python -m scripts.build_fullseed_control_jobs "
            f"--family {family}`. The builder does not submit jobs.\n"
        )
    (out / "README.md").write_text(readme)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="full-seed")
    parser.add_argument("--campaign-dir", type=Path, default=None,
                        help="Override the campaign directory (relative to the repo root).")
    args = parser.parse_args()
    print(json.dumps(build(args.family, args.campaign_dir), indent=2))


if __name__ == "__main__":
    main()
