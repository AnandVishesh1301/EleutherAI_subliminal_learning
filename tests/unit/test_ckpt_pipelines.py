"""Checkpoint-ladder campaign driver gates (plan §4 / §5).

The free-space gate must measure the filesystem behind the resolved
``output_root`` / ``hf_cache_root``, never a hardcoded SSD-2 mount. The dial
gate must RAISE rather than warn. ``driver_config`` must bind ``ckpt_plan.SPEC``
so ``CampaignDriverConfig.__post_init__`` cannot silently rewrite the run-set id.

`merge_capability_parts` is P1b's single writer for the merged capability CSV
(plan review 2026-08-10, blocker #2): concurrent per-rung shards each write
only their own ``parts/cap__*.csv``, so this merge must combine every shard
without dropping rows the way a shared read-modify-write could.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipelines.ckpt_campaign_pipeline import (
    CampaignGateError,
    CkptDriverOptions,
    _submit_p3,
    driver_config,
    free_disk_gib,
    require_disk_free,
)
from pipelines.ckpt_results_pipeline import merge_capability_parts
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.cka.results_io import similarity_results_dir


def test_driver_config_binds_ckptladder_spec(tmp_path: Path) -> None:
    cfg = driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
    )
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.spec.run_set_id == plan.RUN_SET_ID
    assert plan.Q_P1_GEOMETRY in cfg.queue_priorities


def test_free_disk_gib_uses_resolved_path_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def fake_usage(path: str | Path) -> SimpleNamespace:
        seen.append(Path(path))
        return SimpleNamespace(total=10 * 1024**3, used=1 * 1024**3, free=9 * 1024**3)

    monkeypatch.setattr("pipelines.ckpt_campaign_pipeline.shutil.disk_usage", fake_usage)
    target = tmp_path / "a" / "b" / "c"
    target.mkdir(parents=True)
    free = free_disk_gib(target)
    assert free == pytest.approx(9.0)
    assert seen
    # Measurement walked an existing ancestor of the resolved path — never a
    # hardcoded /mnt/ssd-2 mount string.
    assert all("/mnt/ssd-2" not in str(p) for p in seen)


def test_require_disk_free_raises_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_usage(_path: str | Path) -> SimpleNamespace:
        return SimpleNamespace(
            total=200 * 1024**3, used=100 * 1024**3, free=50 * 1024**3
        )

    monkeypatch.setattr("pipelines.ckpt_campaign_pipeline.shutil.disk_usage", fake_usage)
    with pytest.raises(CampaignGateError, match="free-space gate FAILED"):
        require_disk_free([tmp_path], min_gib=150)


def test_require_disk_free_passes_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_usage(_path: str | Path) -> SimpleNamespace:
        return SimpleNamespace(
            total=3000 * 1024**3, used=100 * 1024**3, free=2000 * 1024**3
        )

    monkeypatch.setattr("pipelines.ckpt_campaign_pipeline.shutil.disk_usage", fake_usage)
    measured = require_disk_free([tmp_path], min_gib=150)
    assert measured[str(tmp_path)] == pytest.approx(2000.0)


@pytest.mark.parametrize("include_family_swap", [False, True])
@pytest.mark.parametrize("include_cyclic_closure", [False, True])
def test_submit_p3_builds_every_queue_for_each_pairing_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_family_swap: bool,
    include_cyclic_closure: bool,
) -> None:
    """Every P3 queue must build for every `CkptDriverOptions` combination.

    The pairing flags select which *configs* exist; only the builders that
    actually branch on them accept them. Passing them to a builder that does not
    raised `TypeError` mid-`_submit_p3` on the live campaign (2026-08-11), after
    the students and controls queues had already been submitted -- leaving the
    crossed queue missing and no `submit_p3` transition to resume from. Exercise
    the whole call so a signature drift fails here instead of at 3 a.m.
    """
    submitted: dict[str, int] = {}

    monkeypatch.setattr(
        "pipelines.ckpt_campaign_pipeline.driver_io.write_generated_configs",
        lambda _cfg, docs: dict(docs),
    )
    monkeypatch.setattr(
        "pipelines.ckpt_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, jobs, **_kw: submitted.update({name: len(jobs)}),
    )

    cfg = driver_config(
        output_root=tmp_path, queue_root=tmp_path / "queue", repo_root=tmp_path
    )
    state: dict = {"transitions": {}}
    _submit_p3(
        cfg,
        state,
        CkptDriverOptions(
            include_family_swap=include_family_swap,
            include_cyclic_closure=include_cyclic_closure,
        ),
    )

    assert submitted[plan.Q_P3_STUDENTS] > 0
    assert submitted[plan.Q_P3_CONTROLS] > 0
    assert submitted[plan.Q_P3_CROSSED] > 0
    # P4 resumes off these lists, so an empty one silently drops a whole stage.
    tr = state["transitions"]["submit_p3"]
    assert tr["student_yamls"] and tr["control_yamls"] and tr["crossed_yamls"]


def _write_part(output_root: Path, run_set_id: str, name: str, rows: list[dict]) -> None:
    parts_dir = similarity_results_dir(output_root, run_set_id) / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(parts_dir / name, index=False)


def test_merge_capability_parts_combines_every_rung_shard(tmp_path: Path) -> None:
    """Each P1b shard writes only its own part -- the merge must union them all."""
    _write_part(
        tmp_path,
        "geom",
        "cap__512.csv",
        [{"init": "original", "step": 512, "task": "lambada", "metric": "acc", "value": 0.1}],
    )
    _write_part(
        tmp_path,
        "geom",
        "cap__16000.csv",
        [{"init": "original", "step": 16000, "task": "lambada", "metric": "acc", "value": 0.4}],
    )
    out_csv = merge_capability_parts(tmp_path, "geom")
    assert out_csv is not None
    merged = pd.read_csv(out_csv)
    assert sorted(merged["step"]) == [512, 16000]


def test_merge_capability_parts_drops_duplicate_shard_rewrites_keeping_latest(
    tmp_path: Path,
) -> None:
    """A re-run shard for the same (init, step, task, metric) must not duplicate rows."""
    _write_part(
        tmp_path,
        "geom",
        "cap__512-a.csv",
        [{"init": "original", "step": 512, "task": "lambada", "metric": "acc", "value": 0.1}],
    )
    _write_part(
        tmp_path,
        "geom",
        "cap__512-b.csv",
        [{"init": "original", "step": 512, "task": "lambada", "metric": "acc", "value": 0.9}],
    )
    out_csv = merge_capability_parts(tmp_path, "geom")
    merged = pd.read_csv(out_csv)
    assert len(merged) == 1
    # `keep="last"` — the alphabetically-last shard glob wins, matching
    # `sorted(...glob(...))`'s iteration order.
    assert merged["value"].iloc[0] == pytest.approx(0.9)


def test_merge_capability_parts_returns_none_when_no_shards_exist(tmp_path: Path) -> None:
    assert merge_capability_parts(tmp_path, "geom") is None
