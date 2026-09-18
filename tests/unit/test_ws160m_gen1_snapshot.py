"""Part I (gen-1) enumeration is frozen: ids, queues, counts, jobs, YAMLs, run dirs.

Expected values were computed from the code that ran the four `20260916_ws160m-*`
partitions (pin d4d813e) and must never change (plan § 14.1 invariant 1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan

_SHARED = Path("/shared/run_set") / plan.SHARED_GENERATED_DIRNAME
_OUT = Path("/out")


def _sha(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _snapshot(partition: str, wave: int) -> dict:
    p = plan.partition(partition)
    train = cgen.all_train_configs(partition, wave=wave)
    verify = cgen.all_verify_configs(partition, train)
    by_q = jobs.all_jobs(
        partition, train, verify, wave=wave, shared_config_root=_SHARED
    )
    job_lines = sorted(
        f"{q}\t{j.key}\t{j.cmd.split('--sweep-config ', 1)[1].split()[0]}\t{j.mem_mib}"
        for q, js in by_q.items()
        for j in js
    )
    arms: dict[str, list[int]] = {}
    for c in pairs.cells_of(partition, wave=wave):
        row = arms.setdefault(c.arm, [0, 0, 0])
        row[0] += 1
        row[1] += c.leaves
        row[2] += len(pairs.active_leaves([c]))
    dirs = sorted(
        str(d)
        for d in exp.expected_leaf_dirs(partition, _OUT, wave=wave)
        + exp.expected_control_dirs(partition, _OUT, wave=wave)
        + exp.expected_xbase_dirs(partition, _OUT)
        + exp.owned_teacher_dirs(partition, _OUT, wave=wave)
    )
    yamls = json.dumps({**train, **verify}, sort_keys=True, default=str)
    return {
        "run_set_id": p.run_set_id,
        "queues": dict(plan.queue_priorities(p)),
        "arms": {k: tuple(v) for k, v in arms.items()},
        "controls": len(pairs.controls_of(partition, wave=wave)),
        "teacher_leaves": len(pairs.teacher_leaves(partition, wave=wave)),
        "n_jobs": len(job_lines),
        "jobs_sha256": _sha(job_lines),
        "n_yamls": len(train) + len(verify),
        "yaml_sha256": hashlib.sha256(yamls.encode()).hexdigest(),
        "n_dirs": len(dirs),
        "dirs_sha256": _sha(dirs),
    }


def _queues(prefix: str, stages: list[str], first: int = 100) -> dict[str, int]:
    return {f"{prefix}-{s}": first - 3 * i for i, s in enumerate(stages)}


_MAIN_QUEUES = _queues(
    "ws160m-main",
    [
        "teachers",
        "teacher-evals",
        "controls",
        "core",
        "boost-a",
        "boost-b",
        "evals",
        "w2-teachers",
        "w2-teacher-evals",
        "w2-controls",
        "w2-students",
        "w2-evals",
        "evals-force",
        "lmeval",
    ],
)
_LADDER_QUEUES = _queues(
    "ws160m-ladder",
    [
        "teachers",
        "teacher-evals",
        "controls",
        "r8000",
        "r2000",
        "r48000",
        "r16000",
        "evals",
        "evals-force",
        "lmeval",
    ],
)
_DOSEHP_QUEUES = _queues(
    "ws160m-dosehp",
    [
        "controls",
        "dose-bear",
        "evals-dose-bear",
        "hp-lr1e-4_r8a16",
        "evals-hp-lr1e-4_r8a16",
        "hp-lr4e-4_r8a32",
        "evals-hp-lr4e-4_r8a32",
        "hp-lr4e-4_r32a64",
        "evals-hp-lr4e-4_r32a64",
        "hp-lr2e-4_r8a16",
        "evals-hp-lr2e-4_r8a16",
        "hp-lr8e-4_r8a16",
        "evals-hp-lr8e-4_r8a16",
        "teachers-fallback",
        "teacher-evals-fallback",
        "evals-force",
        "lmeval",
    ],
)
_EXTRA_QUEUES = _queues(
    "ws160m-extra",
    [
        "xbase",
        "controls",
        "evals-xbase",
        "dose-cat-eagle",
        "evals-dose-cat-eagle",
        "heatmap",
        "evals-heatmap",
        "teacher-dose",
        "evals-teacher-dose",
        "dose3-cat-eagle",
        "evals-dose3-cat-eagle",
        "dose-dog",
        "evals-dose-dog",
        "teachers-fallback",
        "teacher-evals-fallback",
        "evals-force",
        "lmeval",
    ],
)

_EXPECTED: dict[tuple[str, int], dict] = {
    ("main", 1): {
        "run_set_id": "20260916_ws160m-main_160m",
        "queues": _MAIN_QUEUES,
        "arms": {"main": (121, 531, 363)},
        "controls": 33,
        "teacher_leaves": 96,
        "n_jobs": 644,
        "jobs_sha256": "79baf9ad32b91452dcc1c5c13ed12f4ed25edaab4e95c21451db0f5f9e9b31f6",
        "n_yamls": 64,
        "yaml_sha256": "6955d14500d0de732b809ece462424d3adada2a109666a22713dd627850184bc",
        "n_dirs": 492,
        "dirs_sha256": "7c3ecd7c1aa9628db17cd2f993201fc724bdb8962cf73a94586cbf9b0747509c",
    },
    ("main", 2): {
        "run_set_id": "20260916_ws160m-main_160m",
        "queues": _MAIN_QUEUES,
        "arms": {"main": (121, 531, 363), "wave2": (23, 75, 75)},
        "controls": 36,
        "teacher_leaves": 99,
        "n_jobs": 750,
        "jobs_sha256": "5b9fb968b33e7c1044e6390eae4dc6f7888f47309a1bdae93467795f1e88007c",
        "n_yamls": 92,
        "yaml_sha256": "7bc2a2b1a94b3de7db81c9951e81871a3c2c5c227d8ef57419e363140346c005",
        "n_dirs": 573,
        "dirs_sha256": "4c37fef4d0267bec91c87c364706374ad9adae50738ba3751050a3a00ae3e1fb",
    },
    ("ladder", 1): {
        "run_set_id": "20260916_ws160m-ladder_160m",
        "queues": _LADDER_QUEUES,
        "arms": {"ladder": (156, 468, 351)},
        "controls": 81,
        "teacher_leaves": 108,
        "n_jobs": 720,
        "jobs_sha256": "593db8641950c3c6adefc4712f1e4f231190cc78321b601e37cebacd24522e44",
        "n_yamls": 128,
        "yaml_sha256": "10529252d5608a227b170b28cc543f6cef88b9b2ca4276f531fd7265f914220d",
        "n_dirs": 540,
        "dirs_sha256": "3fc71930ad65be7481222a0cfacd266b811155eb9478a1fc9268a620001cf7fc",
    },
    ("dosehp", 1): {
        "run_set_id": "20260916_ws160m-dosehp_160m",
        "queues": _DOSEHP_QUEUES,
        "arms": {"dose_bear": (20, 60, 60), "hp": (100, 300, 240)},
        "controls": 144,
        "teacher_leaves": 0,
        "n_jobs": 592,
        "jobs_sha256": "1e38d80abed60c29ed7347e896139496792bd3b59a3cc5e8e30a4ca27242b1a9",
        "n_yamls": 112,
        "yaml_sha256": "92d3be40dbb7cef19f5b9a9ecd26bdf570f526b7fd7377167c77d8b2f43ae29a",
        "n_dirs": 444,
        "dirs_sha256": "5cd5450ea0b850a8ddf1bb64831a5f960803b88e2437f5a84088b9e95f1949e4",
    },
    ("extra", 1): {
        "run_set_id": "20260916_ws160m-extra_160m",
        "queues": _EXTRA_QUEUES,
        "arms": {
            "dose_cat_eagle": (24, 72, 72),
            "heatmap": (72, 216, 216),
            "teacher_dose": (22, 66, 0),
            "dose3_cat_eagle": (6, 18, 0),
            "dose_dog": (12, 36, 0),
        },
        "controls": 72,
        "teacher_leaves": 0,
        "n_jobs": 552,
        "jobs_sha256": "c90a06c126a1db6564440ff8e9fff109ab9b2f1317094f1f26043f27f629389f",
        "n_yamls": 126,
        "yaml_sha256": "1f6184629a5978c163f56a45c2eb9243795953fb9ad16926e9d5d8001877cdac",
        "n_dirs": 414,
        "dirs_sha256": "0cb973fb9adb8ef9f3348059e34277715c8da85651f486676be188a16049d2ee",
    },
}


@pytest.mark.parametrize(("partition", "wave"), sorted(_EXPECTED))
def test_gen1_enumeration_is_byte_identical(partition: str, wave: int) -> None:
    assert _snapshot(partition, wave) == _EXPECTED[(partition, wave)]


def test_gen1_partition_roster_and_policy_are_unchanged() -> None:
    assert plan.PARTITION_ORDER == ("main", "ladder", "dosehp", "extra")
    assert tuple(plan.PARTITIONS) == plan.PARTITION_ORDER
    assert plan.ANIMALS == ("bear", "cat", "eagle", "dog")
    assert plan.WAVE2_AUTO is False and plan.DEADLINE_HOURS_DEFAULT == 24.0
    assert plan.gpu_policy_json()["cap_per_gpu"] == 2
    assert (
        plan.TMUX_SESSION == "ws160m" and plan.STAGING_REPORT == "staging_report.json"
    )
    assert dict(plan.DEFERRED_STAGES) == {
        "main": ("boost-a", "boost-b"),
        "ladder": ("r16000",),
        "dosehp": ("hp-lr8e-4_r8a16",),
        "extra": ("teacher-dose", "dose3-cat-eagle", "dose-dog"),
    }
