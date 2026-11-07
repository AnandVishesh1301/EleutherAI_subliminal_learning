"""Done markers and result snapshots of the rebtopk driver."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines import rebtopk_campaign_pipeline as driver
from pipelines import rebtopk_results_pipeline as results
from subliminal_learning.campaign import rebtopk_jobs as jobs
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256


class _Cfg:
    def __init__(self, root: Path) -> None:
        self.output_root = root


def test_failed_results_build_writes_no_marker_and_reports_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise results.ResultsIncompleteError("priority 1: short")

    monkeypatch.setattr(driver, "build_results", fail)
    cfg = _Cfg(tmp_path)
    logs = plan.run_set_root(tmp_path) / "logs"
    logs.mkdir(parents=True)
    error = driver._mark(cfg, plan.P1_DONE_FILE, "p1", (1,))
    assert "priority 1" in error
    assert not (logs / plan.P1_DONE_FILE).exists()


def test_marker_is_written_once_after_a_complete_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        driver, "build_results", lambda root, **kw: calls.append(kw) or {"x": root}
    )
    cfg = _Cfg(tmp_path)
    logs = plan.run_set_root(tmp_path) / "logs"
    logs.mkdir(parents=True)
    assert driver._mark(cfg, plan.P1_DONE_FILE, "p1", (1,)) is None
    assert driver._mark(cfg, plan.P1_DONE_FILE, "p1", (1,)) is None
    assert calls == [{"snapshot": "p1", "required": (1,)}]
    assert (logs / plan.P1_DONE_FILE).is_file()


def test_build_results_refuses_a_required_priority_with_unevaluated_leaves(
    tmp_path: Path,
) -> None:
    with pytest.raises(results.ResultsIncompleteError, match="priority 1"):
        results.build_results(tmp_path, snapshot="p1", required=(1,))


def test_coverage_counts_a_trained_but_unscored_seed(tmp_path: Path) -> None:
    unit = next(u for u in plan.units() if u.priority == 1)
    leaf = jobs.leaf_dir(tmp_path, unit, unit.gseeds[0])
    leaf.mkdir(parents=True)
    (leaf / "train_manifest.json").write_text("{}")
    row = results.coverage(tmp_path, pd.DataFrame())["p1"]
    assert row["planned"] == plan.n_cells(1)
    assert row["trained"] == 1
    assert row["evaluated"] == 0


@pytest.mark.parametrize("arm", [plan.ANIMAL, plan.RAND])
def test_mask_identity_compares_the_manifest_with_the_plan(
    tmp_path: Path, arm: str
) -> None:
    unit = next(u for u in plan.units() if u.setting.arm == arm)
    leaf = jobs.leaf_dir(tmp_path, unit, unit.gseeds[0])
    leaf.mkdir(parents=True)
    (leaf / "train_manifest.json").write_text(json.dumps({"vocab_mask_ids": [1, 2]}))
    assert ids_sha256([1, 2]) not in plan.TOKEN_LIST_SHA256.values()
    assert len(results.check_mask_identity(tmp_path)) == 1


def test_no_removal_leaves_skip_the_mask_check(tmp_path: Path) -> None:
    unit = next(u for u in plan.units() if u.setting.arm == plan.NONE)
    leaf = jobs.leaf_dir(tmp_path, unit, unit.gseeds[0])
    leaf.mkdir(parents=True)
    (leaf / "train_manifest.json").write_text("{}")
    assert results.check_mask_identity(tmp_path) == []
