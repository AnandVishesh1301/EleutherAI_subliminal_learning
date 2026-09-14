"""Build matched untuned-sender controls for the directed full-seed factorial.

This only writes campaign inputs. Submit the generated jobs through jobq with the
OLMo dependencies recorded in campaign.json; no GPU work runs in this builder.
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path("run_logs/sl_fullseed_controls_20260914")
BASES = {"orig": "original", **{f"fs{i}": f"full-seed{i}" for i in (1, 2, 3)}}
MODELS = {"orig": "pythia-160m", **{f"fs{i}": f"pythia-160m-seed{i}" for i in (1, 2, 3)}}
ANIMALS = ("dog", "cat", "lion", "pig")
DEPENDENCIES = ("direction-olmo-surgery-20260912", "direction-olmo-injections-20260912")


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


def command(module: str, args: list[str]) -> str:
    return shlex.join(["uv", "run", "python", "-m", module, *args])


def write_jsonl(path: Path, jobs: list[dict]) -> None:
    assert len({job["key"] for job in jobs}) == len(jobs)
    path.write_text("".join(json.dumps(job) + "\n" for job in jobs))


def main() -> None:
    out = ROOT / CAMPAIGN
    (out / "configs").mkdir(parents=True, exist_ok=True)
    cells = reference_cells((ROOT / "run_logs/fsfact/dog-seed2seed-orig-fs1-core56.jobs.jsonl").read_text())
    training = yaml.safe_load((ROOT / "configs/training/lora_student_distill.yaml").read_text())
    numbers = yaml.safe_load((ROOT / "configs/experiment/number_generation.yaml").read_text())
    assert numbers["prompt_set"]["n_prompts"] == 100000
    assert numbers["prompt_set"]["seed"] == 42
    assert (numbers["prompt_set"]["example_count_min"], numbers["prompt_set"]["example_count_max"]) == (10, 20)
    number_config = CAMPAIGN / "number_generation.yaml"
    (ROOT / number_config).write_text(yaml.safe_dump(numbers, sort_keys=False))
    config_paths = {}
    for tag, hp in cells.items():
        cfg = copy.deepcopy(training)
        cfg["train"].update(learning_rate=hp["learning_rate"], n_epochs=20, seed=1)
        cfg["lora"].update(r=hp["r"], lora_alpha=hp["lora_alpha"])
        config_paths[tag] = CAMPAIGN / "configs" / f"{tag}.yaml"
        (ROOT / config_paths[tag]).write_text(yaml.safe_dump(cfg, sort_keys=False))

    generation = []
    for sender in ("fs1", "fs2", "fs3"):
        generation.append({
            "key": f"base-pool-{sender}",
            "cmd": command("scripts.generate_numbers", [
                "--source", "base", "--base-models", BASES[sender],
                "--allow-full-seed-bases", "--modes", "prefix",
                "--number-config", str(number_config),
            ]),
            "cwd": str(ROOT), "mem_mib": 6000, "env": {"WANDB_MODE": "disabled"},
        })

    output_root = Path(yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())["output_root"])
    jobs, existing = [], 0
    for sender in BASES:
        for receiver in BASES:
            if sender == receiver:
                continue
            for tag in cells:
                common = [
                    "--source", "base", "--base-models", BASES[receiver],
                    "--numbers-base-model", BASES[sender], "--allow-full-seed-bases",
                    "--epochs", "20", "--subsample-size", "10000", "--subsample-seed", "42",
                    "--number-dataset-tag", "ex10-20_prefix_n100k", "--filter-mode", "prefix",
                    "--student-training-config", str(config_paths[tag]),
                    "--number-config", str(number_config),
                ]
                param = (output_root / "students/controls/base" / MODELS[receiver]
                         / f"teacher-{MODELS[sender]}" / "ex10-20_prefix_n100k" / tag)
                existing += (param / "ftseed1/train_manifest.json").exists()
                hp = cells[tag]
                # The established worker supports untuned cross-base controls. Its
                # legacy root evaluation is always dog; all other animals use the
                # namespaced evaluation CLI below and never touch that root.
                commands = [shlex.join([
                    "uv", "run", "python", "run_logs/animal_samedistill_worker.py",
                    "--source", "base", "--animal", "dog", "--base", BASES[receiver],
                    "--numbers-base", BASES[sender], "--allow-full-seed-base",
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
    assert len(jobs) == 12 * 56
    write_jsonl(out / "generation.jobs.jsonl", generation)
    write_jsonl(out / "controls.jobs.jsonl", jobs)
    summary = {
        "directions": 12, "cells_per_direction": 56, "control_jobs": len(jobs),
        "existing_training_manifests": existing, "expected_new_trainings": len(jobs) - existing,
        "generation_jobs": len(generation), "animals": ANIMALS, "student_seed": 1,
        "priority": 49, "strict_deps": True, "olmo_dependencies": DEPENDENCIES,
        "generation_queue": "sl-fullseed-controls-ngen-20260914",
        "control_queue": "sl-fullseed-controls-20260914",
        "cap_per_gpu": 2, "cap_group": "sl-fullseed-controls-20260914",
    }
    (out / "campaign.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(
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
        "shared stage cap: 2 jobs/GPU.\n\n"
        "Rebuild: `uv run python -m scripts.build_fullseed_control_jobs`. "
        "The builder does not submit jobs.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
