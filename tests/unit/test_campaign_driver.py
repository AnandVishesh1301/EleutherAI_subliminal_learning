"""Campaign driver stage machine on a synthetic run tree + file-based queues.

Simulates the whole campaign lifecycle without any GPU work: A0 reports ->
initial queues; fake A1/A2 artifacts + probes -> A3 selection, generated configs
and queues; fake A3 artifacts + eval summaries -> recipe freeze and Stage B
queues; fake Stage B artifacts -> finalize. Also exercises the fail-loud gates
(missing A0, failed jobs, missing manifests, rising control floor) and
idempotency (every advance is safe to re-run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipelines.student_hp_campaign_pipeline import CampaignGateError, run, status
from subliminal_learning.campaign import (
    breadth_plan,
    config_gen,
    driver_io,
    expected,
    plan,
)
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.code_pin_io import CodePin

REPO_ROOT = Path(__file__).parents[2]

_SPECS = (plan.SPEC, breadth_plan.SPEC)


@pytest.fixture(autouse=True)
def _stub_code_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid real ``git worktree`` under the repo during driver unit tests."""

    def _ensure(repo_root: Path, ref: str) -> CodePin:
        sha = str(ref)
        root = tmp_path / "code_pins" / sha
        (root / "src").mkdir(parents=True, exist_ok=True)
        # A real pin is a checkout of the campaign's code: materialize the files
        # `verify_pin_runs_campaign` requires, so only tests that deliberately
        # remove them exercise the refusal.
        for spec in _SPECS:
            for rel in driver_io.pin_required_paths(spec):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        return CodePin(sha=sha, root=root)

    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.ensure_code_pin", _ensure
    )
    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.resolve_commit_sha",
        lambda repo_root, ref: str(ref),
    )
    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.workspace_git_sha",
        lambda repo_root=None: "workspace-head-sha",
    )


# ---- synthetic-tree helpers ---------------------------------------------------------


def _cfg(
    tmp_path: Path,
    *,
    spec: CampaignSpec = plan.SPEC,
    auto_requeue: int = 0,
    auto_reconcile_sha: bool = False,
    dry_run: bool = False,
    code_sha: str | None = None,
) -> CampaignDriverConfig:
    return CampaignDriverConfig(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO_ROOT,
        generated_dir_override=tmp_path / "generated",
        spec=spec,
        auto_requeue=auto_requeue,
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        require_a0=True,
        code_sha=code_sha,
    )


def _write_a0(cfg: CampaignDriverConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / cfg.spec.calibration_report).write_text("{}")
    (cfg.logs_dir / cfg.spec.init_equality_report).write_text("{}")


def _make_leaf(leaf: Path) -> None:
    ckpt = leaf / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(b"fake")
    (leaf / "train_manifest.json").write_text("{}")


def _probe_score(gkd_tag: str) -> float:
    axes = parse_gkd_tag(gkd_tag)
    return (
        axes["student_lora_r"] / 1000
        + axes["student_lora_alpha"] / 10000
        + axes["student_lr"]
    )


def _write_probes(leaf: Path, p_end: float, *, diverged: bool = False) -> None:
    recs = [
        {
            "step": 50,
            "p_target": p_end / 2,
            "fc_share": 0.2,
            "fc_argmax_frac": 0.1,
            "ppl": 48.0,
            "teacher_p_target": 0.3,
        },
        {
            "step": 100,
            "p_target": p_end,
            "fc_share": 0.2,
            "fc_argmax_frac": 0.1,
            "ppl": 1e26 if diverged else 56.0,
            "teacher_p_target": 0.3,
        },
    ]
    (leaf / "training_metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs)
    )


def _repo_yaml(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text())


def _fake_a1_outputs(cfg: CampaignDriverConfig) -> None:
    spec = cfg.spec
    for rel in (spec.a1_same_yaml, spec.a1_cross_yaml):
        for leaf in expected.treatment_leaf_dirs(
            _repo_yaml(rel), cfg.output_root, spec=spec
        ):
            _make_leaf(leaf)
            _write_probes(leaf, _probe_score(leaf.parent.name))


def _fake_a2_outputs(cfg: CampaignDriverConfig, floor: dict[float, float]) -> None:
    spec = cfg.spec
    doc = _repo_yaml(spec.a1_same_yaml)
    doc["grid"] = {**doc["grid"], "student_modes": [spec.a2_student_mode]}
    for leaf in expected.control_leaf_dirs(doc, cfg.output_root, spec=spec):
        _make_leaf(leaf)
        lr = parse_gkd_tag(leaf.parent.name)["student_lr"]
        _write_probes(leaf, floor[lr])


def _complete_queue(cfg: CampaignDriverConfig, name: str, *, fail_keys=()) -> None:
    for job in store.load_jobs(cfg.queue_root, name):
        store.record_result(
            cfg.queue_root,
            name,
            job,
            rc=1 if job.key in fail_keys else 0,
            node="testnode",
            gpu=0,
            start_utc="t0",
            end_utc="t1",
            log="",
            attempt=1,
        )


def _pref_summary(p: float) -> str:
    return json.dumps(
        {
            "summaries": [
                {
                    "prompt_set": "test",
                    "scoring": "prob",
                    "p_target_mean": p,
                    "n_prompts": 50,
                    "n_seeds": 3,
                }
            ]
        }
    )


def _write_verify_summaries(cdir: Path, p: float) -> None:
    """Full pref+FC+capability triad (with benchmarks key for headline lm-eval gates)."""
    (cdir / "preference_summary.json").write_text(_pref_summary(p))
    (cdir / "sdf_forced_choice_summary.json").write_text("{}")
    (cdir / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {}, "benchmarks": {}})
    )


def _fake_a3_outputs(cfg: CampaignDriverConfig) -> None:
    """Manifests + preference summaries for every generated A3 YAML.

    Treatment p_target grows with dose, x2 for the A1-top recipe (r128a128 @
    4e-4) and x1.2 for beta=1.0 -> the freeze must pick that recipe at the
    highest dose in ``spec.doses_for_freeze()``, headline beta 1.0.
    """
    spec = cfg.spec
    state = json.loads((cfg.state_path).read_text())
    for fname in state["transitions"]["submit_a3"]["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root, spec=spec)
        axes = parse_gkd_tag(leaves[0].parent.name)
        top = axes["student_lora_r"] == 128 and axes["student_lr"] == pytest.approx(
            4e-4
        )
        p = 0.01 * axes["num_epochs"] * (2.0 if top else 1.0)
        p *= 1.2 if axes["beta"] == pytest.approx(1.0) else 1.0
        for leaf in leaves:
            _make_leaf(leaf)
        for cdir in {leaf.parent for leaf in leaves}:
            _write_verify_summaries(cdir, p)
        if fname in config_gen.a3_headline_train_names({fname: doc}):
            controls = expected.control_leaf_dirs(doc, cfg.output_root, spec=spec)
            for leaf in controls:
                _make_leaf(leaf)
            for cdir in {leaf.parent for leaf in controls}:
                _write_verify_summaries(cdir, 0.003)


def _fake_b_outputs(cfg: CampaignDriverConfig) -> None:
    spec = cfg.spec
    state = json.loads((cfg.state_path).read_text())
    for fname in state["transitions"]["submit_b"]["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root, spec=spec)
        for leaf in leaves:
            _make_leaf(leaf)
        for cdir in {leaf.parent for leaf in leaves}:
            _write_verify_summaries(cdir, 0.1)
        if fname.startswith("gkd_b_same_init_"):
            controls = expected.control_leaf_dirs(doc, cfg.output_root, spec=spec)
            for leaf in controls:
                _make_leaf(leaf)
            for cdir in {leaf.parent for leaf in controls}:
                _write_verify_summaries(cdir, 0.003)

# ---- run-set provenance drift ---------------------------------------------------------


def _write_card(cfg: CampaignDriverConfig, sha: str) -> Path:
    root = cfg.run_set_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_set.json").write_text(
        json.dumps(
            {
                "run_set_id": cfg.spec.run_set_id,
                "git_sha": sha,
                "resolved_config": {"corpus": "v1"},
                "launches": [{"git_sha": sha}],
            }
        )
    )
    return root


def test_workspace_head_move_does_not_halt_the_driver(tmp_path: Path) -> None:
    """Pinned execution: workspace HEAD may drift; the card pin stays authoritative."""
    cfg = _cfg(tmp_path)
    _write_a0(cfg)
    _write_card(cfg, "old-sha")
    # autouse stub makes workspace_git_sha -> workspace-head-sha (≠ card)
    result = run(cfg)
    assert "submit_initial" in result.fired
    assert store.queue_exists(cfg.queue_root, cfg.spec.q_a1)


def test_code_sha_mismatch_halts_without_auto_reconcile(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, code_sha="new-sha")
    _write_a0(cfg)
    _write_card(cfg, "old-sha")
    with pytest.raises(CampaignGateError, match="differs from campaign pin"):
        run(cfg)
    assert not store.list_queues(cfg.queue_root)


def test_auto_reconcile_code_sha_records_the_advance_and_advances(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, auto_reconcile_sha=True, code_sha="new-sha")
    _write_a0(cfg)
    root = _write_card(cfg, "old-sha")

    result = run(cfg)

    card = json.loads((root / "run_set.json").read_text())
    assert card["git_sha"] == "new-sha"
    assert card["sha_advances"][0]["from"] == "old-sha"
    assert card["launches"][0]["git_sha"] == "old-sha"
    assert "submit_initial" in result.fired
    assert store.queue_exists(cfg.queue_root, cfg.spec.q_a1)


def test_matching_code_pin_advances(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, code_sha="same-sha")
    _write_a0(cfg)
    _write_card(cfg, "same-sha")
    assert run(cfg).fired == ("submit_initial",)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.run_set_id)
def test_pin_without_the_campaign_configs_is_refused(
    tmp_path: Path, spec: CampaignSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin predating the campaign (code not committed yet) must halt at submit.

    Cells run with cwd inside the pin, so such a pin would fail every single job
    on a missing sweep YAML instead of once, here.
    """
    cfg = _cfg(tmp_path, spec=spec)
    _write_a0(cfg)
    _write_card(cfg, "pin-before-campaign")

    def _bare_pin(repo_root: Path, ref: str) -> CodePin:
        root = tmp_path / "bare_pins" / str(ref)
        (root / "src").mkdir(parents=True, exist_ok=True)
        return CodePin(sha=str(ref), root=root)

    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.ensure_code_pin", _bare_pin
    )

    with pytest.raises(CampaignGateError, match="cannot run campaign"):
        run(cfg)
    assert not store.list_queues(cfg.queue_root)


def test_pin_required_paths_exist_in_the_workspace() -> None:
    """The requirement list must name real files, or the gate is vacuous."""
    for spec in _SPECS:
        missing = [
            rel
            for rel in driver_io.pin_required_paths(spec)
            if not (REPO_ROOT / rel).exists()
        ]
        assert missing == [], f"{spec.run_set_id}: {missing}"


# ---- the lifecycle test ---------------------------------------------------------------


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.run_set_id)
def test_full_campaign_lifecycle(tmp_path: Path, spec: CampaignSpec) -> None:
    cfg = _cfg(tmp_path, spec=spec)
    n_animals = len(spec.animals)
    n_betas = len(spec.betas)
    n_a1_evals = 2 * n_animals * n_betas * len(spec.a1_student_modes)
    n_backlog = len(spec.catswide_verify_yamls) * len(spec.catswide_base_models)
    n_same_arms = 4 + (1 if spec.a3_reference_recipe is not None else 0)
    n_train = n_animals * n_betas * (n_same_arms + 1) * len(spec.a3_doses)
    n_head = n_animals * n_betas * len(spec.a3_doses)

    # A0 gate: nothing exists yet -> refuse to submit anything.
    with pytest.raises(CampaignGateError, match="A0 incomplete"):
        run(cfg)
    assert not store.list_queues(cfg.queue_root)

    # A0 done -> initial queues submitted with the plan's shapes.
    _write_a0(cfg)
    result = run(cfg)
    assert result.fired == ("submit_initial",)
    assert not result.done
    a1_total = n_animals * n_betas * len(spec.a1_student_modes) * len(spec.a1_lora_lrs) * 2
    a2_total = n_animals * n_betas * len(spec.a1_lora_lrs)
    assert len(store.load_jobs(cfg.queue_root, spec.q_a1)) == a1_total
    assert len(store.load_jobs(cfg.queue_root, spec.q_a2)) == a2_total
    assert len(store.load_jobs(cfg.queue_root, spec.q_a1_evals)) == n_a1_evals
    if n_backlog:
        assert len(store.load_jobs(cfg.queue_root, spec.q_backlog)) == n_backlog
    else:
        assert not store.queue_exists(cfg.queue_root, spec.q_backlog)
    assert store.read_meta(cfg.queue_root, spec.q_a1).priority == spec.queue_priorities[
        spec.q_a1
    ]
    assert store.read_meta(cfg.queue_root, spec.q_a2).priority == spec.queue_priorities[
        spec.q_a2
    ]
    assert store.read_meta(cfg.queue_root, spec.q_a1_evals).depends_on == [
        spec.q_a1,
        spec.q_a2,
    ]
    a1_cmds = [j.cmd for j in store.load_jobs(cfg.queue_root, spec.q_a1)]
    assert all("--no-control" in c for c in a1_cmds)
    assert all(
        "--controls-only" in j.cmd for j in store.load_jobs(cfg.queue_root, spec.q_a2)
    )

    # Idempotent: re-running while queues are pending fires nothing new.
    result = run(cfg)
    assert result.fired == ()
    assert result.waiting  # A1/A2 still pending

    # A failed job blocks the A3 transition loudly (auto_requeue=0).
    jobs = store.load_jobs(cfg.queue_root, spec.q_a1)
    _complete_queue(cfg, spec.q_a1, fail_keys=(jobs[0].key,))
    _complete_queue(cfg, spec.q_a2)
    with pytest.raises(CampaignGateError, match="FAILED"):
        run(cfg)
    _complete_queue(cfg, spec.q_a1)  # overwrite the failure with rc=0

    # Queues complete but no artifacts -> disk gate refuses.
    with pytest.raises(CampaignGateError, match="train_manifest"):
        run(cfg)

    # Artifacts present but the control floor RISES with lr -> A2 gate refuses.
    _fake_a1_outputs(cfg)
    _fake_a2_outputs(cfg, {5e-5: 0.002, 1e-4: 0.005, 2e-4: 0.011, 4e-4: 0.019})
    with pytest.raises(CampaignGateError, match="control-floor gate FAILED"):
        run(cfg)

    # Flat floor -> selection runs, configs generated, A3 queues submitted.
    _fake_a2_outputs(cfg, {lr: 0.003 for lr in spec.a1_lora_lrs})
    result = run(cfg)
    assert result.fired == ("submit_a3",)
    state = json.loads(cfg.state_path.read_text())
    cands = state["transitions"]["submit_a3"]["candidates"]
    assert set(cands) == {f"{a}|{b:g}" for a in spec.animals for b in spec.betas}
    expected_rs = [128, 128, 128, 128]
    expected_lrs = [4e-4, 2e-4, 1e-4, 8e-4]
    if spec.a3_reference_recipe is not None:
        expected_rs.append(spec.a3_reference_recipe.lora_r)
        expected_lrs.append(spec.a3_reference_recipe.lr)
    for recipes in cands.values():
        assert [r["lora_r"] for r in recipes] == expected_rs
        assert [r["lr"] for r in recipes] == expected_lrs
    # R1 provenance: both rankings persisted (probes peak at their last step here,
    # so the two top-3 lists coincide and the union is 3 + the extension).
    sel = state["transitions"]["submit_a3"]["selection"][f"{spec.animals[0]}|1"]
    assert [r["lr"] for r in sel["by_final"]] == [4e-4, 2e-4, 1e-4]
    assert sel["by_peak"] == sel["by_final"]
    assert all(r["peak_p_target"] is not None for r in sel["ranking"])
    warnings = state["transitions"]["submit_a3"]["lr_edge_warnings"]
    assert len(warnings) == n_animals * n_betas and all(
        "TOP lr rung" in w for w in warnings
    )
    report = json.loads((cfg.logs_dir / spec.a1_selection_report).read_text())
    assert report["selection"] == state["transitions"]["submit_a3"]["selection"]

    assert len(list(cfg.generated_dir.glob("gkd_a3_*.yaml"))) == 2 * n_train
    a3_jobs = store.load_jobs(cfg.queue_root, spec.q_a3)
    # Every train YAML x 3 gseeds, plus a controls-only shard per headline YAML/gseed.
    assert len(a3_jobs) == (n_train + n_head) * len(spec.a3_gseeds)
    assert sum("--controls-only" in j.cmd for j in a3_jobs) == n_head * len(
        spec.a3_gseeds
    )
    assert all(
        ("--no-control" in j.cmd) ^ ("--controls-only" in j.cmd) for j in a3_jobs
    )
    assert len(store.load_jobs(cfg.queue_root, spec.q_a3_evals)) == n_train
    assert store.read_meta(cfg.queue_root, spec.q_a3_evals).depends_on == [spec.q_a3]

    # Idempotent again.
    assert run(cfg).fired == ()

    # A3 done -> freeze picks r128a128 @ 4e-4 on the freeze dose ladder, beta 1.0.
    _complete_queue(cfg, spec.q_a3)
    _complete_queue(cfg, spec.q_a3_evals)
    _fake_a3_outputs(cfg)
    result = run(cfg)
    assert result.fired == ("submit_b",)
    state = json.loads(cfg.state_path.read_text())
    assert state["transitions"]["submit_b"]["headline_beta"] == 1.0
    frozen = state["transitions"]["submit_b"]["frozen"]["1"]
    assert frozen["recipe"] == {"lora_r": 128, "lora_alpha": 128, "lr": 4e-4}
    assert frozen["num_epochs"] == max(spec.doses_for_freeze())
    freeze_report = json.loads((cfg.logs_dir / spec.freeze_report).read_text())
    assert freeze_report["headline_beta"] == 1.0
    assert freeze_report["freeze_doses"] == list(spec.doses_for_freeze())
    assert freeze_report["a3_doses"] == list(spec.a3_doses)
    assert freeze_report["winner"]["recipe"] == frozen["recipe"]
    assert freeze_report["frozen"] == state["transitions"]["submit_b"]["frozen"]
    head = store.load_jobs(cfg.queue_root, spec.q_b_headline)
    ext = store.load_jobs(cfg.queue_root, spec.q_b_ext)
    # 4 headline YAMLs x animals x 2 students x 3 gseeds x 3 tfs.
    ctrl = [j for j in head if j.key.startswith("b-ctrl|")]
    assert len(head) - len(ctrl) == 4 * n_animals * 2 * 3 * 3
    assert len(ctrl) == 7 * n_animals * len(spec.b_gseeds)
    assert all("--controls-only" in j.cmd for j in ctrl)
    assert all("--no-control" in j.cmd for j in head + ext if j not in ctrl)
    for job in ext:
        assert "--animals " in job.cmd and "--base-models " in job.cmd
        assert "--gseeds " in job.cmd and "--teacher-ftseeds " in job.cmd
    assert len({j.key for j in head + ext}) == len(head) + len(ext)
    assert len(store.load_jobs(cfg.queue_root, spec.q_b_evals)) == 27 * n_animals

    # Stage B done -> finalize writes the marker.
    _complete_queue(cfg, spec.q_b_headline)
    _complete_queue(cfg, spec.q_b_ext)
    _complete_queue(cfg, spec.q_b_evals)

    # Checkpoints landed but the eval queue's verify summaries have NOT ->
    # finalize must refuse loudly rather than mark the campaign done with
    # preference-only (or unevaluated) Stage B/C leaves.
    for fname in state["transitions"]["submit_b"]["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        for leaf in expected.treatment_leaf_dirs(doc, cfg.output_root, spec=spec):
            _make_leaf(leaf)
    with pytest.raises(CampaignGateError, match="preference_summary"):
        run(cfg)

    _fake_b_outputs(cfg)
    result = run(cfg)
    assert result.fired == ("finalize",)
    assert result.done
    assert (cfg.logs_dir / spec.campaign_done_file).exists()
    assert run(cfg).done  # terminal + idempotent

    snap = status(cfg)
    assert set(snap["transitions"]) == {
        "submit_initial",
        "submit_a3",
        "submit_b",
        "finalize",
    }


def test_auto_requeue_under_cap_then_halts(tmp_path: Path) -> None:
    """--auto-requeue N requeues failed jobs below the cap, then gates loudly."""
    cfg = _cfg(tmp_path, spec=plan.SPEC, auto_requeue=2)
    _write_a0(cfg)
    run(cfg)
    jobs = store.load_jobs(cfg.queue_root, plan.Q_A1)
    fail_key = jobs[0].key
    _complete_queue(cfg, plan.Q_A1, fail_keys=(fail_key,))
    _complete_queue(cfg, plan.Q_A2)

    # attempt=1 < cap=2 → requeue; A1 becomes pending again (waiting, not FAILED).
    result = run(cfg)
    assert result.fired == ()
    assert any(plan.Q_A1 in w for w in result.waiting)
    assert store.job_state(cfg.queue_root, plan.Q_A1, jobs[0]) == store.PENDING

    # Fail again at attempt=2 → over the cap → halt.
    store.record_result(
        cfg.queue_root,
        plan.Q_A1,
        jobs[0],
        rc=1,
        node="testnode",
        gpu=0,
        start_utc="t0",
        end_utc="t1",
        log="",
        attempt=2,
    )
    with pytest.raises(CampaignGateError, match="auto-requeue 2"):
        run(cfg)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.run_set_id)
def test_driver_config_run_set_tracks_spec(tmp_path: Path, spec: CampaignSpec) -> None:
    cfg = _cfg(tmp_path, spec=spec)
    assert cfg.run_set_id == spec.run_set_id
    assert dict(cfg.queue_priorities) == dict(spec.queue_priorities)


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_a0(cfg)
    dry = _cfg(tmp_path, dry_run=True)
    # Reuse the same roots as the A0 write.
    dry = CampaignDriverConfig(
        output_root=cfg.output_root,
        queue_root=cfg.queue_root,
        repo_root=cfg.repo_root,
        dry_run=True,
        generated_dir_override=cfg.generated_dir_override,
        spec=cfg.spec,
    )
    result = run(dry)
    assert result.fired == ("submit_initial",)
    assert not store.list_queues(cfg.queue_root)
    assert not cfg.state_path.exists()


def test_queue_mismatch_refuses(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_a0(cfg)
    store.create_queue(cfg.queue_root, plan.Q_A1, priority=1)
    store.append_jobs(cfg.queue_root, plan.Q_A1, [Job(key="alien", cmd="echo hi")])
    with pytest.raises(RuntimeError, match="different job set"):
        run(cfg)
