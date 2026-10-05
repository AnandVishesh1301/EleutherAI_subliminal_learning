"""Reconcile + queue the top-10 multi-animal eval of the EXISTING ds1 non-adaptive strip.

Angela wants to know how dog-specific the transfer is on the ds1 strip (focus teacher
ep2_lr5e-4_r32a64 tseed1 -> data-seed1 students, dog numbers). This scans the replay
registry (run_logs/strip_ckpt_replay_cells.json) and classifies each (cell, seed) run on
disk as:
  * WITH checkpoints  -> a full multi-animal trajectory (the replay saved 40 ckpts)
  * FINAL-only        -> a single-step (endpoint) multi-animal eval (--final-only)

Both readouts land in the CANONICAL run-dir location ({run_dir}/eval/pref_traj.json), so
nothing clobbers the legacy dog-only flat files. Jobs are emitted per TAG (the trajectory
script iterates a tag's seeds internally): a tag with ANY checkpointed seed goes to the
trajectory file (final-only seeds in it degrade to a 1-point endpoint naturally); a tag
whose seeds are ALL final-only goes to the --final-only file.

    uv run python -m scripts.generate_ds1_strip_top10eval
"""

from __future__ import annotations

from subliminal_learning.config import resolve_output_root
import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
RUN_LOGS = ROOT / "run_logs"
REGISTRY = RUN_LOGS / "strip_ckpt_replay_cells.json"

TRAJ_JOBS = RUN_LOGS / "ds1_strip_top10eval_traj_jobs.jsonl"
FINAL_JOBS = RUN_LOGS / "ds1_strip_top10eval_final_jobs.jsonl"

ANIMALS = ("dog", "cat", "rabbit", "pig", "lion", "chicken", "deer", "tiger", "cow", "elephant")


def _output_root() -> Path:
    return resolve_output_root()


def _student_model(student_base: str) -> str:
    return "pythia-160m" if student_base == "original" else f"pythia-160m-{student_base}"


def main() -> None:
    reg = json.loads(REGISTRY.read_text())
    lin = reg["lineage"]
    out_root = _output_root()
    student_model = _student_model(lin["student_base"])
    base_teacher = (
        out_root / "students" / lin["animal"] / student_model
        / f"teacher-{lin['teacher_base_model']}"
        / f"{lin['teacher_tag']}_tseed{lin['teacher_ftseed']}"
    )
    animals_arg = ",".join(ANIMALS)

    n_with = n_final = n_missing = 0
    traj_tags: list[str] = []
    final_tags: list[str] = []
    for tag, seeds in reg["cells"].items():
        tag_has_ckpt = False
        tag_has_final = False
        for seed in seeds:
            # dataset_tag level (ex10-20_prefix_n100k) globbed so we don't hardcode n.
            hits = list(base_teacher.glob(f"*/{tag}/ftseed{seed}/checkpoints"))
            if not hits:
                n_missing += 1
                continue
            ckpt_dir = hits[0]
            if any(ckpt_dir.glob("checkpoint-*")):
                n_with += 1
                tag_has_ckpt = True
            elif (ckpt_dir / "adapter_model.safetensors").exists():
                n_final += 1
                tag_has_final = True
            else:
                n_missing += 1
        if tag_has_ckpt:
            traj_tags.append(tag)
        elif tag_has_final:
            final_tags.append(tag)

    traj_lines = [json.dumps({"key": f"ds1-top10-traj|{t}", "cmd": (
        "uv run python -m scripts.ckpt_preference_trajectory "
        f"--tag {t} --registry run_logs/strip_ckpt_replay_cells.json --animals {animals_arg}")})
        for t in traj_tags]
    final_lines = [json.dumps({"key": f"ds1-top10-final|{t}", "cmd": (
        "uv run python -m scripts.ckpt_preference_trajectory "
        f"--tag {t} --registry run_logs/strip_ckpt_replay_cells.json --final-only "
        f"--animals {animals_arg}")})
        for t in final_tags]

    TRAJ_JOBS.write_text("\n".join(traj_lines) + "\n" if traj_lines else "")
    FINAL_JOBS.write_text("\n".join(final_lines) + "\n" if final_lines else "")

    print(f"registry cells={len(reg['cells'])} runs listed={sum(len(v) for v in reg['cells'].values())}")
    print(f"(cell,seed) on disk: with_checkpoints={n_with} final_only={n_final} missing={n_missing}")
    print(f"traj tags (>=1 checkpointed seed)={len(traj_tags)} -> {TRAJ_JOBS}")
    print(f"final-only tags (all seeds final)={len(final_tags)} -> {FINAL_JOBS}")


if __name__ == "__main__":
    main()
