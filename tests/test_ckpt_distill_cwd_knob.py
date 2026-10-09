"""`jobq.cwd` knob of scripts.sweeps.sweep_ckpt_distill.expand_jobs (pure; no GPU / files)."""

from __future__ import annotations

from pathlib import Path

from scripts.sweeps.sweep_ckpt_distill import expand_jobs

BASE = {
    "base_model": "original",
    "steps": [25000],
    "pairs": {"topology": "diagonal", "explicit": []},
    "animals": ["dog"],
    "teacher": {"training_config": "configs/training/lora_template_bias.yaml", "ft_seeds": [1],
                "param_tags": ["ep20_lr1e-4_r16a32"]},
    "numbers": {"number_config": "configs/experiment/number_generation.yaml", "filter_mode": "prefix",
                "include_base_control": False, "include_rng_control": False},
    "student": {"training_config": "configs/training/lora_student_distill.yaml", "epochs": [10],
                "ft_seeds": [1], "subsample_size": 10000, "subsample_seed": 42, "modes": ["plain"],
                "control_students": []},
}
JOBQ = {"root": "run_logs/queues/queue_angela", "queue_prefix": "t",
        "priorities": {"teacher": 3, "numgen": 2, "distill": 1},
        "mem_mib": {"teacher": 1, "numgen": 1, "distill": 1}}


def _jobs(jobq: dict) -> list[dict]:
    return [j for q in expand_jobs({**BASE, "jobq": jobq}, Path("/repo")) for j in q.jobs]


def test_absent_cwd_emits_no_cwd_field() -> None:
    jobs = _jobs(JOBQ)
    assert jobs and all("cwd" not in j for j in jobs)


def test_explicit_pin_cwd_pins_every_job_to_the_submitting_checkout() -> None:
    jobs = _jobs({**JOBQ, "cwd": "pin"})
    assert jobs and all(j["cwd"] == "/repo" for j in jobs)


def test_explicit_null_cwd_emits_no_cwd_field() -> None:
    jobs = _jobs({**JOBQ, "cwd": None})
    assert jobs and all("cwd" not in j for j in jobs)


def test_explicit_path_cwd_is_passed_through() -> None:
    jobs = _jobs({**JOBQ, "cwd": "/elsewhere"})
    assert jobs and all(j["cwd"] == "/elsewhere" for j in jobs)


# ------------------------------------------------- `jobq.depends_on_prefix` (reads the root)
EXT = {**BASE, "steps": [75000, 130000],
       "pairs": {"topology": "diagonal", "explicit": []}}


def _deps(jobq: dict, queue_folder: Path | None) -> dict[str, list[str]]:
    specs = expand_jobs({**EXT, "jobq": jobq}, Path("/repo"), queue_folder)
    return {q.name: q.depends_on for q in specs}


def _root_with_counterparts(tmp_path: Path) -> Path:
    for name in ("other-t-step75000", "other-n-step75000"):
        (tmp_path / name).mkdir()
    return tmp_path


def test_depends_on_prefix_adds_deps_only_where_the_other_queues_exist(tmp_path: Path) -> None:
    deps = _deps({**JOBQ, "depends_on_prefix": "other"}, _root_with_counterparts(tmp_path))
    assert deps["t-t-step75000"] == ["other-t-step75000"]
    assert deps["t-n-step75000"] == ["t-t-step75000", "other-n-step75000"]
    # step 130000 is new in this campaign: no counterpart queue, so no dangling dependency
    assert deps["t-t-step130000"] == []
    assert deps["t-n-step130000"] == ["t-t-step130000"]
    # the distill queue is untouched by the knob
    assert deps["t-d-step75000"] == ["t-n-step75000"]


def test_without_the_knob_dependencies_are_unchanged(tmp_path: Path) -> None:
    deps = _deps(JOBQ, _root_with_counterparts(tmp_path))
    assert deps["t-t-step75000"] == [] and deps["t-n-step75000"] == ["t-t-step75000"]
    assert deps == _deps(JOBQ, None)


def test_knob_without_a_queue_folder_adds_nothing(tmp_path: Path) -> None:
    assert _deps({**JOBQ, "depends_on_prefix": "other"}, None) == _deps(JOBQ, None)
