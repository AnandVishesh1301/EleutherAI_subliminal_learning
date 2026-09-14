"""strictscreen driver: dry run, gates, frozen jobs, eval holes, finalize, audit handoff."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pipelines import strictscreen_campaign_pipeline as dp
from pipelines.strictscreen_campaign_pipeline import (
    CampaignGateError,
    StrictscreenDriverOptions,
    driver_config,
    run,
    status,
)
from scripts import run_strictscreen_campaign as cli
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.jobqueue import store

VOCAB = {a: f"vocab-{a}" for a in plan.ANIMALS}
STAGED = {
    "passed": True,
    "prereg": {"sha256": "prereg"},
    "teachers": [{"manifest_sha256": "t"}],
    "pool": {"manifest_sha256": "pool"},
    "screen": {"vocab_sha256": VOCAB},
}


@dataclass(frozen=True)
class FakeContext:
    pin_root: Path
    host: str
    queue_root: Path
    output_root: Path
    run_set_logs_dir: Path
    campaign_queue_names: tuple[str, ...]
    dry_run: bool = False


class FakeAudit:
    AuditContext = FakeContext

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.done = False

    def audit_preflight(self, ctx: FakeContext) -> dict:
        self.calls.append(("preflight", ctx))
        return {"judge": "ok"}

    def advance_audit(
        self, ctx, state, fired, waiting, *, required_run_dirs, optional_run_dirs
    ):
        self.calls.append(("advance", list(required_run_dirs), list(optional_run_dirs)))
        waiting.append("judge running")

    def audit_done(self, state) -> bool:
        return self.done


@pytest.fixture
def audit(monkeypatch: pytest.MonkeyPatch) -> FakeAudit:
    fake = FakeAudit()
    monkeypatch.setattr(dp, "audit_api", lambda *, dry_run: fake)
    return fake


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pin = tmp_path / "pin"
    pin.mkdir()
    monkeypatch.setattr(dp.driver_io, "ensure_campaign_code_pin", lambda _cfg: None)
    monkeypatch.setattr(
        dp.driver_io, "queue_job_defaults", lambda _cfg: {"cwd": str(pin)}
    )
    monkeypatch.setattr(dp, "disk_gate_readings", lambda **_k: {"ok": 1.0})
    monkeypatch.setattr(dp.store, "this_host", lambda: plan.NODE)
    monkeypatch.setattr(dp.stage, "prereg_record", lambda _p: STAGED["prereg"])
    monkeypatch.setattr(dp.stage, "teacher_records", lambda _o: STAGED["teachers"])
    monkeypatch.setattr(dp.stage, "pool_record", lambda _d: STAGED["pool"])
    monkeypatch.setattr(dp.stage, "vocab_records", lambda: STAGED["screen"])
    monkeypatch.setattr(dp.stage, "recipe_record", lambda _o: {"lr": 4e-4})
    monkeypatch.setattr(dp.stage, "arm_c_identity_records", lambda *_a: [{}] * 32)
    monkeypatch.setattr(dp.stage, "queue_census", lambda *_a: {"blocking": []})

    def fake_results(output_root: Path, **_kw) -> dict:
        path = jobs.results_dir(output_root) / plan.RESULTS_CLAIMS_JSON
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: {"status": "pass"} for k in ("C0", "H2", "H3")}))
        return {plan.RESULTS_CLAIMS_JSON: path}

    monkeypatch.setattr(dp, "build_results", fake_results)


def _cfg(tmp_path: Path, *, dry_run: bool = False) -> driver_io.CampaignDriverConfig:
    return driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=dry_run,
        auto_requeue=2,
    )


def _opts(tmp_path: Path, **kw) -> StrictscreenDriverOptions:
    return StrictscreenDriverOptions(
        cold_root=tmp_path / "cold",
        prereg_path=tmp_path / "PREREG.md",
        dataset_cache_root=tmp_path / "data",
        pin_root=tmp_path / "pin",
        **kw,
    )


def _stage(cfg, opts, staged: dict = STAGED) -> None:
    target = opts.cold_root / "gkd_students" / plan.RUN_SET_ID
    target.mkdir(parents=True)
    link = jobs.run_set_root(cfg.output_root)
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    cfg.logs_dir.mkdir(parents=True)
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(json.dumps(staged))
    (cfg.logs_dir / plan.SMOKE_REPORT).write_text(json.dumps({"passed": True}))


def _plant_leaf(leaf: Path, cell: plan.Cell, *, vocab: str | None = None) -> None:
    ckpt = leaf / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    ident = {"gkd_tag": cell.gkd_tag, "gkd_hyperparameters": {}}
    manifest: dict = {"run_identity": ident}
    summary: dict = {"n_rows_screened": 20000, "n_flagged": 10, "flagged_rate": 0.0005}
    if cell.screen == "strict":
        sha = vocab or VOCAB[cell.animal]
        ident["gkd_hyperparameters"]["rollout_screen"] = "strict"
        ident["rollout_screen_vocab_sha256"] = sha
        manifest["rollout_screen"] = {"vocab_sha256": sha}
        summary.update(rollout_screen="strict", screen_vocab_sha256=sha)
    (leaf / "train_manifest.json").write_text(json.dumps(manifest))
    (leaf / "rollouts_summary.json").write_text(json.dumps(summary))


def _plant_evals(config_dir: Path) -> None:
    pref = {"summaries": [{"prompt_set": s} for s in plan.EVAL_PROMPT_SETS]}
    (config_dir / "preference_summary.json").write_text(json.dumps(pref))
    (config_dir / "sdf_forced_choice_summary.json").write_text("{}")
    (config_dir / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {}})
    )


def _record(
    cfg, name: str, jobkeys: set[str] | None = None, *, rc: int = 0, attempt: int = 0
):
    for job in store.load_jobs(cfg.queue_root, name):
        if jobkeys is None or job.key in jobkeys:
            store.record_result(
                cfg.queue_root,
                name,
                job,
                rc=rc,
                node=plan.NODE,
                gpu=0,
                start_utc="t0",
                end_utc="t1",
                log="",
                attempt=attempt,
            )


def _drive(
    cfg,
    opts,
    *,
    skip: frozenset[str] = frozenset(),
    bad: str | None = None,
    mutate: Callable[[dict[str, Path]], None] | None = None,
):
    """Preflight → train done (skipped cells fail at the cap) → evals done → final pass."""
    run(cfg, opts)
    leaves = jobs.cell_leaf_dirs(cfg.output_root)
    for cell in plan.all_cells():
        if cell.key not in skip:
            _plant_leaf(
                leaves[cell.key], cell, vocab="wrong" if cell.key == bad else None
            )
    if mutate is not None:
        mutate(leaves)
    for unit in plan.UNITS:
        keys = {c.key for c in unit.cells()}
        _record(cfg, unit.train_queue, keys - skip)
        _record(cfg, unit.train_queue, keys & skip, rc=1, attempt=2)
    run(cfg, opts)
    done_dirs = {leaves[c.key].parent for c in plan.all_cells()} - {
        leaves[k].parent for k in skip
    }
    for d in done_dirs:
        _plant_evals(d)
    for unit in plan.UNITS:
        if store.queue_exists(cfg.queue_root, unit.eval_queue):
            _record(cfg, unit.eval_queue)
    return run(cfg, opts)


def test_dry_run_walks_every_phase_writing_nothing(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    result = run(cfg, _opts(tmp_path))
    expected_fired = [
        "preflight",
        "submit_train",
        *(f"submit_evals_{u.name}" for u in plan.UNITS),
        "finalize_training",
        "sl_results",
    ]
    assert list(result.fired) == expected_fired and not result.done
    assert not (tmp_path / "queue").exists() and not (tmp_path / "out").exists()
    assert not (tmp_path / "generated").exists()
    preflight = next(c for c in audit.calls if c[0] == "preflight")[1]
    assert preflight.dry_run and preflight.campaign_queue_names == tuple(
        plan.QUEUE_PRIORITIES
    )
    _, required, optional = next(c for c in audit.calls if c[0] == "advance")
    assert len(required) == 360 and len(optional) == 32
    assert all(p.is_absolute() for p in required + optional)


def test_dry_run_without_audit_driver_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "audit_api", lambda *, dry_run: None)
    result = run(_cfg(tmp_path, dry_run=True), _opts(tmp_path))
    assert "sl_results" in result.fired and not result.done
    assert any("not importable" in w for w in result.waiting)


def test_audit_api_missing_module_is_a_gate_on_real_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(dp.importlib, "import_module", missing)
    assert dp.audit_api(dry_run=True) is None
    with pytest.raises(CampaignGateError, match="not importable"):
        dp.audit_api(dry_run=False)


def test_preflight_refuses_off_host_and_missing_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audit: FakeAudit
) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(CampaignGateError, match="staging_report"):
        run(cfg, _opts(tmp_path))
    monkeypatch.setattr(dp.store, "this_host", lambda: "soar-subliminal-0")
    with pytest.raises(CampaignGateError, match="must run on soar-subliminal-3"):
        run(cfg, _opts(tmp_path))
    assert not cfg.state_path.exists()


def test_preflight_refuses_vocab_and_recipe_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts, {**STAGED, "screen": {"vocab_sha256": {**VOCAB, "cat": "old"}}})
    with pytest.raises(CampaignGateError, match="screen vocab"):
        run(cfg, opts)

    def drifted(_o):
        raise RuntimeError("breadth4 freeze != plan")

    (cfg.logs_dir / plan.STAGING_REPORT).write_text(json.dumps(STAGED))
    monkeypatch.setattr(dp.stage, "recipe_record", drifted)
    with pytest.raises(CampaignGateError, match="breadth4 freeze"):
        run(cfg, opts)
    assert "preflight" not in driver_io.load_state(cfg)["transitions"]


def test_preflight_refuses_a_real_dir_at_the_run_set(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(json.dumps(STAGED))
    (cfg.logs_dir / plan.SMOKE_REPORT).write_text(json.dumps({"passed": True}))
    with pytest.raises(CampaignGateError, match="real directory"):
        run(cfg, opts)


def test_submit_train_once_with_frozen_jobkeys(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    first = run(cfg, opts)
    assert list(first.fired) == ["preflight", "submit_train"]
    queues = set(store.list_queues(cfg.queue_root))
    assert queues == {u.train_queue for u in plan.UNITS}
    assert sum(len(store.load_jobs(cfg.queue_root, q)) for q in queues) == 392
    for unit in plan.UNITS:
        meta = store.read_meta(cfg.queue_root, unit.train_queue)
        assert meta.node == plan.NODE and meta.priority == unit.train_priority
        assert not meta.depends_on
    before = {q: {j.jobkey for j in store.load_jobs(cfg.queue_root, q)} for q in queues}
    state = driver_io.load_state(cfg)
    del state["transitions"]["submit_train"]
    driver_io.save_state(cfg, state)
    assert "submit_train" in run(cfg, opts).fired
    after = {q: {j.jobkey for j in store.load_jobs(cfg.queue_root, q)} for q in queues}
    assert after == before
    assert (tmp_path / "generated" / "strictscreen_a_ds1-ds1_tfs0.yaml").is_file()
    assert any(c[0] == "preflight" for c in audit.calls)


def test_failed_cells_are_requeued_until_the_cap(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    run(cfg, opts)
    unit = plan.UNIT_BY_NAME["c"]
    cell = unit.cells()[0].key
    _record(cfg, unit.train_queue, {c.key for c in unit.cells()} - {cell})
    _record(cfg, unit.train_queue, {cell}, rc=1, attempt=0)
    result = run(cfg, opts)
    assert "submit_evals_c" not in result.fired
    assert store.queue_status(cfg.queue_root, unit.train_queue)["pending"] == 1
    _record(cfg, unit.train_queue, {cell}, rc=1, attempt=2)
    assert "submit_evals_c" in run(cfg, opts).fired


def test_evals_only_for_config_dirs_with_all_leaves_done(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    run(cfg, opts)
    unit = plan.UNIT_BY_NAME["b"]
    leaves = jobs.cell_leaf_dirs(cfg.output_root)
    missing = unit.cells()[4]
    for cell in unit.cells():
        if cell != missing:
            _plant_leaf(leaves[cell.key], cell)
    _record(cfg, unit.train_queue)
    result = run(cfg, opts)
    assert result.fired == ("submit_evals_b",)
    evals = store.load_jobs(cfg.queue_root, unit.eval_queue)
    assert len(evals) == 23
    meta = store.read_meta(cfg.queue_root, unit.eval_queue)
    assert meta.priority == 95 and not meta.depends_on
    holes = driver_io.load_state(cfg)["transitions"]["submit_evals_b"]["holes"]
    assert len(holes) == 1 and holes[0]["missing_cells"] == [missing.key]
    assert holes[0]["config_dir"] == str(leaves[missing.key].parent)


def test_finalize_audits_screen_and_hands_off_to_the_judge(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    result = _drive(cfg, opts)
    assert {"finalize_training", "sl_results"} <= set(result.fired)
    assert not result.done and "judge running" in result.waiting
    assert (cfg.logs_dir / plan.SCREEN_STATS_CSV).is_file()
    shortfall = json.loads((cfg.logs_dir / plan.SHORTFALL_REPORT).read_text())
    assert shortfall["within_tolerance"] and shortfall["arms"]["a"]["n_missing"] == 0
    _, required, optional = next(c for c in audit.calls if c[0] == "advance")
    assert len(required) == 360 and len(optional) == 32
    assert status(cfg)["sl_claims"] == {"C0": "pass", "H2": "pass", "H3": "pass"}
    audit.done = True
    final = run(cfg, opts)
    assert final.done and "campaign_done" in final.fired
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    assert not any(k.startswith("audit_") for k in driver_io.load_state(cfg))


def test_screen_contamination_is_a_hard_gate(tmp_path: Path, audit: FakeAudit) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    bad = plan.arm_cells("a")[7].key
    with pytest.raises(CampaignGateError, match="contamination"):
        _drive(cfg, opts, bad=bad)
    assert "finalize_training" not in driver_io.load_state(cfg)["transitions"]


def test_legacy_leaf_with_strict_fields_is_contamination(
    tmp_path: Path, audit: FakeAudit
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    cell = plan.arm_cells("c")[0]

    def add_screen_record(leaves: dict[str, Path]) -> None:
        path = leaves[cell.key] / "train_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["rollout_screen"] = {"vocab_sha256": VOCAB[cell.animal]}
        path.write_text(json.dumps(manifest))

    with pytest.raises(CampaignGateError, match="legacy leaf carries"):
        _drive(cfg, opts, mutate=add_screen_record)


@pytest.mark.parametrize(("n_missing_dirs", "passes"), [(1, True), (2, False)])
def test_shortfall_tolerance_on_arm_a(
    tmp_path: Path, audit: FakeAudit, n_missing_dirs: int, passes: bool
) -> None:
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    dirs = [u for u in jobs.eval_units(plan.UNIT_BY_NAME["a_tfs1"])][:n_missing_dirs]
    skip = frozenset(eu.cells[0].key for eu in dirs)
    if passes:
        result = _drive(cfg, opts, skip=skip)
        assert "finalize_training" in result.fired
    else:
        with pytest.raises(CampaignGateError, match="shortfall"):
            _drive(cfg, opts, skip=skip)
    report = json.loads((cfg.logs_dir / plan.SHORTFALL_REPORT).read_text())
    assert report["arms"]["a"]["n_missing"] == 3 * n_missing_dirs
    assert report["within_tolerance"] is passes
    assert set(report["backfill"]["train"]) == skip
    assert len(report["backfill"]["eval"]) == n_missing_dirs
    assert report["backfill"]["requeue"]


def test_cli_gate_exits_2_and_watch_exits_0_when_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        json.dumps(
            {
                "output_root": str(tmp_path / "out"),
                "dataset_cache_root": str(tmp_path / "data"),
                "queue_root": str(tmp_path / "queue"),
            }
        )
    )

    def gate(*_a, **_k):
        raise CampaignGateError("blocked")

    runner = CliRunner()
    monkeypatch.setattr(cli, "run", gate)
    res = runner.invoke(cli.app, ["advance", "--paths-config", str(paths)])
    assert res.exit_code == 2
    res = runner.invoke(
        cli.app, ["watch", "--paths-config", str(paths), "--interval", "0"]
    )
    assert res.exit_code == 2
    monkeypatch.setattr(
        cli, "run", lambda *_a, **_k: dp.AdvanceResult(("x",), (), True)
    )
    res = runner.invoke(
        cli.app, ["watch", "--paths-config", str(paths), "--interval", "0"]
    )
    assert res.exit_code == 0


def test_judge_preflight_refusal_is_logged_on_dry_run_and_gates_real_passes(
    tmp_path: Path, audit: FakeAudit, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_ctx):
        raise CampaignGateError("audit preflight: judge snapshot missing")

    monkeypatch.setattr(audit, "audit_preflight", refuse)
    result = run(_cfg(tmp_path, dry_run=True), _opts(tmp_path))
    assert "sl_results" in result.fired
    cfg, opts = _cfg(tmp_path), _opts(tmp_path)
    _stage(cfg, opts)
    with pytest.raises(CampaignGateError, match="judge snapshot"):
        run(cfg, opts)
    assert "preflight" not in driver_io.load_state(cfg)["transitions"]
