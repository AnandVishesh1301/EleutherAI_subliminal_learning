"""OPD-limits driver: dry run, submit-once, core finalize, tail folds, cross-tier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines import opdlim_campaign_pipeline as dp
from pipelines.opdlim_campaign_pipeline import (
    CampaignGateError,
    OpdlimDriverOptions,
    cross_tier_status,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import opdlim_config_gen as cgen
from subliminal_learning.campaign import opdlim_jobs as jobs
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.topk_jobs import cold_symlink_plan
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import run_set_subdir


def _cfg(t: plan.Tier, tmp_path: Path, *, dry_run: bool = False):
    return driver_config(
        t,
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=dry_run,
    )


def _opts(tmp_path: Path) -> OpdlimDriverOptions:
    prereg = tmp_path / "PREREG_OPDLIM.md"
    prereg.write_text("# prereg\n")
    return OpdlimDriverOptions(
        cold_root=tmp_path / "cold",
        prereg_path=prereg,
        hf_cache_root=tmp_path / "hf",
        enforce_host=False,
    )


def _plant_staging(cfg, t: plan.Tier, opts: OpdlimDriverOptions) -> None:
    for link, target in cold_symlink_plan(
        cfg.output_root, opts.cold_root, run_set_id=t.run_set_id
    ):
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    links = []
    for link, source in jobs.assembly_links(t, cfg.output_root):
        source.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source.resolve())
        links.append({"link": str(link)})
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    refs = [{"repo": r.repo_id, "revision": r.revision} for r in jobs.prefetch_refs()]
    (cfg.logs_dir / plan.PREFETCH_MANIFEST).write_text(
        json.dumps({"complete": True, "refs": refs})
    )
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(json.dumps({"passed": True}))
    (cfg.logs_dir / plan.ASSEMBLY_REPORT).write_text(
        json.dumps({"passed": True, "links": links})
    )
    (cfg.logs_dir / plan.SUBSET_AUDIT_REPORT).write_text(json.dumps({"passed": True}))


def _mark_cell(run_dir: Path, *, bench: bool = False) -> None:
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    (run_dir / "train_manifest.json").write_text("{}")
    config = run_dir.parent
    pref = {"summaries": [{"prompt_set": s} for s in plan.EVAL_PROMPT_SETS]}
    (config / "preference_summary.json").write_text(json.dumps(pref))
    (config / "sdf_forced_choice_summary.json").write_text("{}")
    cap = {"perplexity": {}, "calibration": {}}
    if bench:
        cap["benchmarks"] = {}
    (config / "sdf_capability_summary.json").write_text(json.dumps(cap))


def _plant_blocks(cfg, t: plan.Tier, blocks, *, bench: bool = False) -> None:
    train = cgen.all_train_configs(t)
    for b in blocks:
        for leaf in jobs.block_leaf_dirs(t, train, cfg.output_root, b.name):
            _mark_cell(leaf, bench=bench)


def _complete(cfg, names) -> None:
    for name in names:
        for job in store.load_jobs(cfg.queue_root, name):
            store.record_result(
                cfg.queue_root,
                name,
                job,
                rc=0,
                node="n",
                gpu=0,
                start_utc="t0",
                end_utc="t1",
                log="",
                attempt=1,
            )


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(dp.driver_io, "ensure_campaign_code_pin", lambda _cfg: None)
    monkeypatch.setattr(dp.driver_io, "queue_job_defaults", lambda _cfg: {})
    monkeypatch.setattr(dp, "disk_gate_readings", lambda **_k: {"ok": 100.0})
    monkeypatch.setattr(
        dp, "build_results", lambda *_a, **_k: calls.append("results") or {}
    )
    return calls


@pytest.mark.parametrize("t", (plan.TIER1, plan.TIER2), ids=lambda t: t.token)
def test_dry_run_walks_preflight_to_done_writing_nothing(
    t: plan.Tier, tmp_path: Path
) -> None:
    cfg = _cfg(t, tmp_path, dry_run=True)
    result = run(cfg, t, _opts(tmp_path))
    assert {"preflight", "submit_all", "finalize_core"} <= set(result.fired)
    assert not store.list_queues(cfg.queue_root)
    assert not cfg.state_path.exists()
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    assert not (tmp_path / "generated").exists()
    assert result.done is (t.n == 1)


def test_driver_refuses_wrong_spec(tmp_path: Path) -> None:
    cfg = _cfg(plan.TIER2, tmp_path, dry_run=True)
    with pytest.raises(CampaignGateError, match="spec"):
        run(cfg, plan.TIER1, _opts(tmp_path))


def test_preflight_refuses_missing_staging_and_off_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = plan.TIER1
    cfg = _cfg(t, tmp_path)
    with pytest.raises(CampaignGateError, match="staging_report"):
        run(cfg, t, _opts(tmp_path))
    monkeypatch.setattr(dp.store, "this_host", lambda: "soar-subliminal-0")
    opts = OpdlimDriverOptions(cold_root=tmp_path / "cold", enforce_host=True)
    with pytest.raises(CampaignGateError, match="must run on soar-subliminal-2"):
        run(cfg, t, opts)


def test_submit_all_once_with_frozen_jobkeys_and_dependencies(tmp_path: Path) -> None:
    t = plan.TIER1
    cfg, opts = _cfg(t, tmp_path), _opts(tmp_path)
    _plant_staging(cfg, t, opts)
    first = run(cfg, t, opts)
    assert {"preflight", "submit_all"} <= set(first.fired) and not first.done
    submitted = set(store.list_queues(cfg.queue_root))
    assert submitted == set(plan.queue_priorities(t))
    n_train = sum(
        len(store.load_jobs(cfg.queue_root, n))
        for n in submitted
        if "-students-" in n or "-controls-" in n
    )
    assert n_train == plan.tier_cells(t)
    meta = store.read_meta(
        cfg.queue_root, plan.queue_name(t, "ce-k10", "evals", "410m")
    )
    assert set(meta.depends_on) == {
        plan.queue_name(t, "ce-k10", "students", "410m"),
        plan.queue_name(t, "ce-k10", "controls", "410m"),
    }
    bench = store.read_meta(
        cfg.queue_root, plan.queue_name(t, "ce-k10", "bench", "160m")
    )
    assert bench.depends_on == [plan.queue_name(t, "ce-k10", "evals", "160m")]
    dose = store.read_meta(
        cfg.queue_root, plan.queue_name(t, "hp-p4-dose-ep2", "evals", "160m")
    )
    assert dose.depends_on == [plan.queue_name(t, "hp-p4-dose-ep2", "students", "160m")]
    before = {
        n: {j.jobkey for j in store.load_jobs(cfg.queue_root, n)} for n in submitted
    }
    state = driver_io.load_state(cfg)
    del state["transitions"]["submit_all"]
    driver_io.save_state(cfg, state)
    assert "submit_all" in run(cfg, t, opts).fired
    after = {
        n: {j.jobkey for j in store.load_jobs(cfg.queue_root, n)} for n in submitted
    }
    assert after == before


def test_core_finalizes_then_tail_and_bench_fold_in_without_pause(
    tmp_path: Path, _stub: list[str]
) -> None:
    t = plan.TIER1
    cfg, opts = _cfg(t, tmp_path), _opts(tmp_path)
    _plant_staging(cfg, t, opts)
    run(cfg, t, opts)
    _complete(cfg, plan.core_queue_names(t))
    _plant_blocks(cfg, t, t.core_blocks())
    result = run(cfg, t, opts)
    assert "finalize_core" in result.fired
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    assert not result.done and _stub == ["results"]
    assert not any("pause" in f or "deadline" in f for f in result.fired)
    tail = t.tail_blocks()[0]
    _complete(cfg, plan.block_queue_names(t, tail.name))
    _plant_blocks(cfg, t, [tail])
    result = run(cfg, t, opts)
    assert f"results_{tail.name}" in result.fired
    _complete(cfg, plan.tail_queue_names(t))
    _plant_blocks(cfg, t, t.blocks, bench=True)
    result = run(cfg, t, opts)
    assert result.done
    assert "finalize_core" not in result.fired
    assert status(cfg, t)["done"] is True


def test_core_waits_for_calibration_in_summaries(tmp_path: Path) -> None:
    t = plan.TIER1
    cfg, opts = _cfg(t, tmp_path), _opts(tmp_path)
    _plant_staging(cfg, t, opts)
    run(cfg, t, opts)
    _complete(cfg, plan.core_queue_names(t))
    _plant_blocks(cfg, t, t.core_blocks())
    leaf = jobs.block_leaf_dirs(t, cgen.all_train_configs(t), tmp_path, "ce-k1")[0]
    (leaf.parent / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {}})
    )
    result = run(cfg, t, opts)
    assert "finalize_core" not in result.fired
    assert any("lack summaries" in w for w in result.waiting)


def _card(root: Path, run_set_id: str, sha: str) -> None:
    card_dir = root / run_set_subdir("gkd_students", run_set_id)
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "run_set.json").write_text(json.dumps({"git_sha": sha}))


def test_cross_tier_status_pending_void_pin_ok(tmp_path: Path) -> None:
    assert cross_tier_status(tmp_path)["status"] == "pending"
    _card(tmp_path, plan.TIER1.run_set_id, "aaa")
    _card(tmp_path, plan.TIER2.run_set_id, "bbb")
    logs1 = tmp_path / run_set_subdir("gkd_students", plan.TIER1.run_set_id) / "logs"
    logs1.mkdir(parents=True)
    (logs1 / plan.CAMPAIGN_DONE_FILE).write_text("done\n")
    assert cross_tier_status(tmp_path)["status"] == "void_pin"
    _card(tmp_path, plan.TIER2.run_set_id, "aaa")
    assert cross_tier_status(tmp_path)["status"] == "ok"
