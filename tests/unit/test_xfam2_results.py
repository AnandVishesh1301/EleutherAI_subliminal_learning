"""Planted-tree tests for the xfam wave-2 results pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pipelines.xfam2_results_pipeline as rp
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.gkd.layout import teacher_tag
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

ANIMAL = "bear"
TAG = plan.student_gkd_tag(plan.NEOX_LR)
SMALL, BIG = "pythia-160m", "pythia-410m"

RESULT_FILES: tuple[str, ...] = (
    plan.RESULTS_LEAVES_CSV,
    plan.RESULTS_CELLS_CSV,
    plan.RESULTS_ASYMMETRY_CSV,
    plan.RESULTS_LAWS_CSV,
    plan.RESULTS_FAMILY_CSV,
    plan.RESULTS_BRIDGE_CSV,
    plan.RESULTS_PPL_CSV,
    plan.RESULTS_QUARANTINE_CSV,
    plan.RESULTS_TEACHER_ADMISSION_CSV,
    plan.RESULTS_BASE_PRIORS_CSV,
    plan.RESULTS_CLAIMS_JSON,
)


def _write_summaries(directory: Path, per_seed: dict[str, float], **extra) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    block = {
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "per_seed_p_target": per_seed,
        **extra,
    }
    (directory / "preference_summary.json").write_text(
        json.dumps({"summaries": [block]})
    )
    overall = {seed: 10.0 * (1.0 + 0.1 * (i + 1)) for i, seed in enumerate(per_seed)}
    (directory / "sdf_capability_summary.json").write_text(
        json.dumps(
            {
                "perplexity": {
                    "per_seed_overall": overall,
                    "base_overall": 10.0,
                    "perplexity_ratio": 1.15,
                }
            }
        )
    )


def _write_logs(directory: Path, seeds: dict[str, float], rev_kl: float) -> None:
    for seed in seeds:
        leaf = directory / f"gseed{seed}"
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "trainer_log_history.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "step": step,
                        "gkd/rev_kl": rev_kl / step,
                        "gkd/filter_drop_frac": 0.01,
                    }
                )
                for step in (20, 10)
            )
            + "\n"
        )
    sibling = directory / "gseed1.20260911T1500Z_doublewrite"
    sibling.mkdir(parents=True, exist_ok=True)
    (sibling / "trainer_log_history.jsonl").write_text("not json\n")


def _root(tmp: Path) -> Path:
    return tmp / rp.campaign_subdir()


def _treatment(tmp: Path, teacher: str, student: str, value: float) -> None:
    seeds = {"0": value, "1": value + 0.01}
    for ftseed in plan.TEACHER_FTSEEDS:
        ttag = teacher_tag(
            "sdf",
            plan.TEACHER_PARAM_TAG,
            ftseed,
            teacher_base_label=teacher,
            student_base_label=student,
        )
        directory = _root(tmp) / ANIMAL / student / ttag / TAG
        _write_summaries(directory, seeds)
        _write_logs(directory, seeds, rev_kl=0.5)


def _control(tmp: Path, teacher: str, student: str, value: float = 0.01) -> None:
    seeds = {"0": value, "1": value}
    if teacher == student:
        directory = _root(tmp) / "controls" / "base_teacher" / ANIMAL / student / TAG
    else:
        directory = (
            _root(tmp) / "controls" / "xbase_teacher" / ANIMAL / student / teacher / TAG
        )
    _write_summaries(directory, seeds)
    _write_logs(directory, seeds, rev_kl=2.0 if teacher != student else 0.1)


def _teacher(tmp: Path, label: str) -> None:
    root = tmp / run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID) / ANIMAL
    _write_summaries(
        root / label / plan.TEACHER_PARAM_TAG,
        {"0": 0.40, "1": 0.42, "2": 0.41},
        base_p_target=0.002,
    )
    (root / label / "base_control.json").write_text(
        json.dumps({plan.PRIMARY_PROMPT_SET: {"p_target": 0.002}})
    )
    (root / label / "base_metrics.json").write_text(
        json.dumps({"perplexity": {"result": {"overall": 12.0}}})
    )


def _part1(tmp: Path) -> Path:
    directory = tmp / "part1"
    directory.mkdir()
    pd.DataFrame(
        [
            {
                "kind": "treatment",
                "teacher": BIG,
                "student": BIG,
                "teacher_param_tag": plan.TEACHER_PARAM_TAG,
                "lora_r": 8,
                "epochs": 1.0,
                "lr": plan.NEOX_LR,
                "ftseed": 0.0,
                "gseed": g,
                "delta_primary": 0.30,
                "quarantined": False,
            }
            for g in (0, 1, 2)
        ]
    ).to_csv(directory / rp.PART1_DELTA_CSV, index=False)
    return directory


def _plant(tmp: Path, *, drop_tb: bool = False) -> None:
    for label in (SMALL, BIG):
        _teacher(tmp, label)
        _control(tmp, label, label)
    _treatment(tmp, SMALL, SMALL, 0.30)
    _treatment(tmp, BIG, BIG, 0.31)
    _treatment(tmp, SMALL, BIG, 0.35)
    _treatment(tmp, BIG, SMALL, 0.02)
    _control(tmp, BIG, SMALL)
    if not drop_tb:
        _control(tmp, SMALL, BIG)


def _cfg(tmp: Path, queue_root: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(output_root=tmp, queue_root=queue_root, repo_root=None)


def test_build_results_writes_every_artifact_and_y0_to_y11(tmp_path: Path) -> None:
    _plant(tmp_path)
    written = rp.build_results(
        _cfg(tmp_path), bootstrap_n=40, bootstrap_seed=0, part1_dir=_part1(tmp_path)
    )
    dest = rp.results_dir(tmp_path)
    for name in RESULT_FILES:
        assert name in written and (dest / name).is_file(), name
    claims = json.loads((dest / plan.RESULTS_CLAIMS_JSON).read_text())
    assert tuple(claims) == plan.CLAIM_IDS
    assert all("status" in record for record in claims.values())
    y0 = claims["Y0"]["cells"]
    assert y0[f"{SMALL}->{BIG}"]["status"] == "holds"
    assert y0[f"{BIG}->{SMALL}"]["direction"] == "big_to_small"
    asym = pd.read_csv(dest / plan.RESULTS_ASYMMETRY_CSV)
    row = asym[(asym["label_a"] == SMALL) & (asym["label_b"] == BIG)].iloc[0]
    assert row["iota"] == pytest.approx(0.33, abs=1e-9)
    bridge = pd.read_csv(dest / plan.RESULTS_BRIDGE_CSV).set_index(
        ["teacher", "student"]
    )
    assert int(bridge.loc[(BIG, BIG), "n_part1"]) == 2
    assert claims["Y9"]["rows"]


def test_leaves_carry_per_gseed_ppl_and_first_window_rev_kl(tmp_path: Path) -> None:
    _plant(tmp_path)
    leaves = rp.assemble_student_frame(tmp_path)
    treat = leaves[(leaves["kind"] == "treatment") & (leaves["student"] == BIG)]
    by_gseed = treat.groupby("gseed")["perplexity_ratio"].first()
    assert by_gseed.loc[0] == pytest.approx(1.1)
    assert by_gseed.loc[1] == pytest.approx(1.2)
    tb = leaves[leaves["kind"] == "tb_control"]
    assert tb["rev_kl_first"].tolist() == pytest.approx([0.2] * len(tb))
    assert set(leaves["tier"]) == {plan.TIER_CORE}


def test_missing_control_raises_only_when_its_queue_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(tmp_path, drop_tb=True)
    rp.build_results(_cfg(tmp_path), dry_run=True, bootstrap_n=10)
    monkeypatch.setattr(rp.store, "queue_exists", lambda root, name: True)
    monkeypatch.setattr(rp.store, "queue_complete", lambda root, name: True)
    with pytest.raises(KeyError, match="xfam2-core-pythia-160m--pythia-410m-ctrl"):
        rp.build_results(
            _cfg(tmp_path, tmp_path / "queue"), dry_run=True, bootstrap_n=10
        )


def test_teacher_frame_admits_per_ftseed_with_its_own_ppl(tmp_path: Path) -> None:
    _teacher(tmp_path, BIG)
    frame = rp.assemble_teacher_frame(tmp_path)
    assert len(frame) == 3
    assert frame["admitted"].all()
    assert frame.set_index("ftseed")["ppl_ratio"].to_dict() == pytest.approx(
        {0: 1.1, 1: 1.2, 2: 1.3}
    )
