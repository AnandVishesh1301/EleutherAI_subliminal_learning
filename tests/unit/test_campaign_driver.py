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
from subliminal_learning.campaign import config_gen, expected, plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from jobq import store
from jobq.model import Job

REPO_ROOT = Path(__file__).parents[2]


# ---- synthetic-tree helpers ---------------------------------------------------------


def _cfg(tmp_path: Path) -> CampaignDriverConfig:
    return CampaignDriverConfig(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO_ROOT,
        generated_dir_override=tmp_path / "generated",
    )


def _write_a0(cfg: CampaignDriverConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.CALIBRATION_REPORT).write_text("{}")
    (cfg.logs_dir / plan.INIT_EQUALITY_REPORT).write_text("{}")


def _make_leaf(leaf: Path) -> None:
    (leaf / "checkpoints").mkdir(parents=True, exist_ok=True)
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
    for rel in (plan.A1_SAME_YAML, plan.A1_CROSS_YAML):
        for leaf in expected.treatment_leaf_dirs(_repo_yaml(rel), cfg.output_root):
            _make_leaf(leaf)
            _write_probes(leaf, _probe_score(leaf.parent.name))


def _fake_a2_outputs(cfg: CampaignDriverConfig, floor: dict[float, float]) -> None:
    doc = _repo_yaml(plan.A1_SAME_YAML)
    doc["grid"] = {**doc["grid"], "student_modes": [plan.A2_STUDENT_MODE]}
    for leaf in expected.control_leaf_dirs(doc, cfg.output_root):
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


def _fake_a3_outputs(cfg: CampaignDriverConfig) -> None:
    """Manifests + preference summaries for every generated A3 YAML.

    Treatment p_target grows with dose, x2 for the A1-top recipe (r128a128 @
    4e-4) and x1.2 for beta=1.0 -> the freeze must pick that recipe at dose 5,
    headline beta 1.0.
    """
    state = json.loads((cfg.state_path).read_text())
    for fname in state["transitions"]["submit_a3"]["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root)
        axes = parse_gkd_tag(leaves[0].parent.name)
        top = axes["student_lora_r"] == 128 and axes["student_lr"] == pytest.approx(
            4e-4
        )
        p = 0.01 * axes["num_epochs"] * (2.0 if top else 1.0)
        p *= 1.2 if axes["beta"] == pytest.approx(1.0) else 1.0
        for leaf in leaves:
            _make_leaf(leaf)
        for cdir in {leaf.parent for leaf in leaves}:
            (cdir / "preference_summary.json").write_text(_pref_summary(p))
        if fname in config_gen.a3_headline_train_names({fname: doc}):
            controls = expected.control_leaf_dirs(doc, cfg.output_root)
            for leaf in controls:
                _make_leaf(leaf)
            for cdir in {leaf.parent for leaf in controls}:
                (cdir / "preference_summary.json").write_text(_pref_summary(0.003))


def _fake_b_outputs(cfg: CampaignDriverConfig) -> None:
    state = json.loads((cfg.state_path).read_text())
    for fname in state["transitions"]["submit_b"]["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        leaves = expected.treatment_leaf_dirs(doc, cfg.output_root)
        for leaf in leaves:
            _make_leaf(leaf)
        for cdir in {leaf.parent for leaf in leaves}:
            (cdir / "preference_summary.json").write_text(_pref_summary(0.1))
        if fname.startswith("gkd_b_same_init_"):
            controls = expected.control_leaf_dirs(doc, cfg.output_root)
            for leaf in controls:
                _make_leaf(leaf)
            for cdir in {leaf.parent for leaf in controls}:
                (cdir / "preference_summary.json").write_text(_pref_summary(0.003))


# ---- run-set provenance drift ---------------------------------------------------------


def _write_card(cfg: CampaignDriverConfig, sha: str) -> Path:
    root = cfg.run_set_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_set.json").write_text(
        json.dumps(
            {
                "run_set_id": plan.RUN_SET_ID,
                "git_sha": sha,
                "resolved_config": {"corpus": "v1"},
                "launches": [{"git_sha": sha}],
            }
        )
    )
    return root


def test_head_advance_halts_the_driver_before_the_queue_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver must name the moved commit, not the failed queue it causes."""
    cfg = _cfg(tmp_path)
    _write_a0(cfg)
    _write_card(cfg, "old-sha")
    monkeypatch.setattr(
        "pipelines.student_hp_campaign_pipeline.git_sha", lambda: "new-sha"
    )

    with pytest.raises(CampaignGateError, match="git HEAD .new-sha. has moved away"):
        run(cfg)
    assert not store.list_queues(cfg.queue_root)  # nothing submitted into the halt


def test_auto_reconcile_sha_records_the_advance_and_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = CampaignDriverConfig(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO_ROOT,
        generated_dir_override=tmp_path / "generated",
        auto_reconcile_sha=True,
    )
    _write_a0(cfg)
    root = _write_card(cfg, "old-sha")
    monkeypatch.setattr(
        "pipelines.student_hp_campaign_pipeline.git_sha", lambda: "new-sha"
    )

    result = run(cfg)

    card = json.loads((root / "run_set.json").read_text())
    assert card["git_sha"] == "new-sha"
    assert card["sha_advances"][0]["from"] == "old-sha"
    assert card["launches"][0]["git_sha"] == "old-sha"  # founding sha preserved
    assert "submit_initial" in result.fired
    assert store.queue_exists(cfg.queue_root, plan.Q_A1)


def test_matching_head_is_not_reported_as_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _write_a0(cfg)
    _write_card(cfg, "same-sha")
    monkeypatch.setattr(
        "pipelines.student_hp_campaign_pipeline.git_sha", lambda: "same-sha"
    )
    assert run(cfg).fired == ("submit_initial",)


# ---- the lifecycle test ---------------------------------------------------------------


def test_full_campaign_lifecycle(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    # A0 gate: nothing exists yet -> refuse to submit anything.
    with pytest.raises(CampaignGateError, match="A0 incomplete"):
        run(cfg)
    assert not store.list_queues(cfg.queue_root)

    # A0 done -> initial queues submitted with the plan's shapes.
    _write_a0(cfg)
    result = run(cfg)
    assert result.fired == ("submit_initial",)
    assert not result.done
    assert len(store.load_jobs(cfg.queue_root, plan.Q_A1)) == plan.A1_TOTAL_CELLS
    assert len(store.load_jobs(cfg.queue_root, plan.Q_A2)) == plan.A2_CONTROL_CELLS
    assert len(store.load_jobs(cfg.queue_root, plan.Q_A1_EVALS)) == 72
    assert len(store.load_jobs(cfg.queue_root, plan.Q_BACKLOG)) == 18
    assert store.read_meta(cfg.queue_root, plan.Q_A1).priority == 100
    assert store.read_meta(cfg.queue_root, plan.Q_A2).priority == 95
    assert store.read_meta(cfg.queue_root, plan.Q_A1_EVALS).depends_on == [
        plan.Q_A1,
        plan.Q_A2,
    ]
    a1_cmds = [j.cmd for j in store.load_jobs(cfg.queue_root, plan.Q_A1)]
    assert all("--no-control" in c for c in a1_cmds)
    assert all(
        "--controls-only" in j.cmd for j in store.load_jobs(cfg.queue_root, plan.Q_A2)
    )

    # Idempotent: re-running while queues are pending fires nothing new.
    result = run(cfg)
    assert result.fired == ()
    assert result.waiting  # A1/A2 still pending

    # A failed job blocks the A3 transition loudly.
    jobs = store.load_jobs(cfg.queue_root, plan.Q_A1)
    _complete_queue(cfg, plan.Q_A1, fail_keys=(jobs[0].key,))
    _complete_queue(cfg, plan.Q_A2)
    with pytest.raises(CampaignGateError, match="FAILED"):
        run(cfg)
    _complete_queue(cfg, plan.Q_A1)  # overwrite the failure with rc=0

    # Queues complete but no artifacts -> disk gate refuses.
    with pytest.raises(CampaignGateError, match="train_manifest"):
        run(cfg)

    # Artifacts present but the control floor RISES with lr -> A2 gate refuses.
    _fake_a1_outputs(cfg)
    _fake_a2_outputs(cfg, {5e-5: 0.002, 1e-4: 0.005, 2e-4: 0.011, 4e-4: 0.019})
    with pytest.raises(CampaignGateError, match="control-floor gate FAILED"):
        run(cfg)

    # Flat floor -> selection runs, configs generated, A3 queues submitted.
    _fake_a2_outputs(cfg, {lr: 0.003 for lr in plan.A1_LORA_LRS})
    result = run(cfg)
    assert result.fired == ("submit_a3",)
    state = json.loads(cfg.state_path.read_text())
    cands = state["transitions"]["submit_a3"]["candidates"]
    assert set(cands) == {f"{a}|{b:g}" for a in plan.ANIMALS for b in plan.BETAS}
    for recipes in cands.values():
        # Top-3 score-ordered, then the R3 lr-edge extension arm (rank-1 sits on
        # the 4e-4 top rung, so 8e-4 is added).
        assert [r["lora_r"] for r in recipes] == [128, 128, 128, 128]
        assert [r["lr"] for r in recipes] == [4e-4, 2e-4, 1e-4, 8e-4]
    # R1 provenance: both rankings persisted (probes peak at their last step here,
    # so the two top-3 lists coincide and the union is 3 + the extension).
    sel = state["transitions"]["submit_a3"]["selection"]["cat|1"]
    assert [r["lr"] for r in sel["by_final"]] == [4e-4, 2e-4, 1e-4]
    assert sel["by_peak"] == sel["by_final"]
    assert all(r["peak_p_target"] is not None for r in sel["ranking"])
    warnings = state["transitions"]["submit_a3"]["lr_edge_warnings"]
    assert len(warnings) == 4 and all("TOP lr rung" in w for w in warnings)
    report = json.loads((cfg.logs_dir / plan.A1_SELECTION_REPORT).read_text())
    assert report["selection"] == state["transitions"]["submit_a3"]["selection"]

    n_train = 2 * 2 * 5 * len(plan.A3_DOSES)  # 4 same arms + cross, per dose
    n_head = 2 * 2 * len(plan.A3_DOSES)
    assert len(list(cfg.generated_dir.glob("gkd_a3_*.yaml"))) == 2 * n_train
    a3_jobs = store.load_jobs(cfg.queue_root, plan.Q_A3)
    # Every train YAML x 3 gseeds, plus a controls-only shard per headline YAML/gseed.
    assert len(a3_jobs) == (n_train + n_head) * len(plan.A3_GSEEDS)
    assert sum("--controls-only" in j.cmd for j in a3_jobs) == n_head * len(
        plan.A3_GSEEDS
    )
    assert all(
        ("--no-control" in j.cmd) ^ ("--controls-only" in j.cmd) for j in a3_jobs
    )
    assert len(store.load_jobs(cfg.queue_root, plan.Q_A3_EVALS)) == n_train
    assert store.read_meta(cfg.queue_root, plan.Q_A3_EVALS).depends_on == [plan.Q_A3]

    # Idempotent again.
    assert run(cfg).fired == ()

    # A3 done -> freeze picks r128a128 @ 4e-4 at the top dose, beta 1.0; Stage B out.
    _complete_queue(cfg, plan.Q_A3)
    _complete_queue(cfg, plan.Q_A3_EVALS)
    _fake_a3_outputs(cfg)
    result = run(cfg)
    assert result.fired == ("submit_b",)
    state = json.loads(cfg.state_path.read_text())
    assert state["transitions"]["submit_b"]["headline_beta"] == 1.0
    frozen = state["transitions"]["submit_b"]["frozen"]["1"]
    assert frozen["recipe"] == {"lora_r": 128, "lora_alpha": 128, "lr": 4e-4}
    assert frozen["num_epochs"] == max(plan.A3_DOSES)  # fixture p_target grows with dose
    # Freeze provenance also lands as its own sibling report (mirrors A1's),
    # so notebooks/manual analysis don't need to parse campaign_state.json.
    freeze_report = json.loads((cfg.logs_dir / plan.FREEZE_REPORT).read_text())
    assert freeze_report["headline_beta"] == 1.0
    assert freeze_report["winner"]["recipe"] == frozen["recipe"]
    assert freeze_report["frozen"] == state["transitions"]["submit_b"]["frozen"]
    head = store.load_jobs(cfg.queue_root, plan.Q_B_HEADLINE)
    ext = store.load_jobs(cfg.queue_root, plan.Q_B_EXT)
    # 4 headline YAMLs x 2 animals x 2 students x 3 gseeds x 3 tfs = 144 treatments.
    ctrl = [j for j in head if j.key.startswith("b-ctrl|")]
    assert len(head) - len(ctrl) == 144
    # The 42 shared controls (7 students x 2 animals x 3 gseeds), deduped into headline.
    assert len(ctrl) == plan.B_CONTROL_CELLS
    assert all("--controls-only" in j.cmd for j in ctrl)
    assert all("--no-control" in j.cmd for j in head + ext if j not in ctrl)
    # Every treatment shard pins one animal / student / gseed / teacher ftseed.
    for job in ext:
        assert "--animals " in job.cmd and "--base-models " in job.cmd
        assert "--gseeds " in job.cmd and "--teacher-ftseeds " in job.cmd
    assert len({j.key for j in head + ext}) == len(head) + len(ext)
    assert len(store.load_jobs(cfg.queue_root, plan.Q_B_EVALS)) == 54

    # Stage B done -> finalize writes the marker.
    _complete_queue(cfg, plan.Q_B_HEADLINE)
    _complete_queue(cfg, plan.Q_B_EXT)
    _complete_queue(cfg, plan.Q_B_EVALS)

    # Checkpoints landed but the eval queue's preference summaries have NOT ->
    # finalize must refuse loudly rather than mark the campaign done with
    # unevaluated Stage B/C leaves (B2: the disk gate must require eval
    # summaries here exactly as the A3 gate already does).
    for fname in state["transitions"]["submit_b"]["train_yamls"]:
        doc = yaml.safe_load((cfg.generated_dir / fname).read_text())
        for leaf in expected.treatment_leaf_dirs(doc, cfg.output_root):
            _make_leaf(leaf)
    with pytest.raises(CampaignGateError, match="preference_summary.json"):
        run(cfg)

    _fake_b_outputs(cfg)
    result = run(cfg)
    assert result.fired == ("finalize",)
    assert result.done
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    assert run(cfg).done  # terminal + idempotent

    snap = status(cfg)
    assert set(snap["transitions"]) == {
        "submit_initial",
        "submit_a3",
        "submit_b",
        "finalize",
    }


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_a0(cfg)
    dry = CampaignDriverConfig(
        output_root=cfg.output_root,
        queue_root=cfg.queue_root,
        repo_root=cfg.repo_root,
        dry_run=True,
        generated_dir_override=cfg.generated_dir_override,
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
