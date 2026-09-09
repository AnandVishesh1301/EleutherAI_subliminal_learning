"""Driver state machine: gates write before they act, the pause waits, resume is a no-op."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines import offxsize_campaign_pipeline as driver
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import offxsize_expected as expected
from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.campaign import offxsize_selection as selection
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.jobqueue import store

RUNGS = {size: "ep10_lr1e-4_r16a32" for size in plan.SIZES}
TAG = "ex10-20_prefix_n100k"


@pytest.fixture(autouse=True)
def _stub_code_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin needs a real git worktree; these tests exercise the state machine."""
    monkeypatch.setattr(
        "pipelines.offxsize_campaign_pipeline.driver_io.queue_job_defaults",
        lambda _cfg: {},
    )


def _cfg(tmp_path: Path, *, dry_run: bool = False) -> CampaignDriverConfig:
    return driver.driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        dry_run=dry_run,
        auto_requeue=0,
    )


def _plant_s0(cfg: CampaignDriverConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.PREFLIGHT_REPORT).write_text(json.dumps({"passed": True}))
    (cfg.logs_dir / plan.ADOPTION_MANIFEST).write_text(json.dumps({"adopted": []}))


def _drain(cfg: CampaignDriverConfig, name: str) -> None:
    """Mark every job in a queue as finished with rc=0."""
    for job in store.load_jobs(cfg.queue_root, name):
        store.record_result(
            cfg.queue_root,
            name,
            job,
            rc=0,
            node="testnode",
            gpu=0,
            start_utc="t0",
            end_utc="t1",
            log="",
            attempt=1,
        )


def _plant_gate(cfg: CampaignDriverConfig, rungs: dict[str, str] | None = None) -> None:
    rungs = RUNGS if rungs is None else rungs
    decisions = {
        size: selection.DoseDecision(
            size, selection.STATUS_PICKED, rungs.get(size), None, ()
        )
        for size in plan.SIZES
    }
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.TEACHER_GATE_REPORT).write_text(
        json.dumps(selection.gate_report(decisions, []))
    )


def _plant_numbers(
    cfg: CampaignDriverConfig, rungs: dict[str, str] | None = None
) -> None:
    specs = numbers.required_pools(RUNGS if rungs is None else rungs)
    statuses = [
        numbers.PoolStatus(
            spec=s,
            ready=True,
            reasons=(),
            n_raw=100_000,
            n_filtered=50_000,
            keep_rate=0.5,
            filtered_sha256=f"sha-{s.key}",
            dataset_tag=TAG,
        )
        for s in specs
    ]
    (cfg.logs_dir / plan.NUMBERS_GATE_REPORT).write_text(
        json.dumps(numbers.gate_report(statuses))
    )


def _finish(leaf_dir: Path, p_target: float) -> None:
    ckpt = leaf_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"0")
    (leaf_dir / "train_manifest.json").write_text("{}")
    (leaf_dir / expected.PREFERENCE_EVAL).write_text(
        json.dumps({"sets": {plan.PRIMARY_PROMPT_SET: {"p_target": p_target}}})
    )


def _finish_screen(
    cfg: CampaignDriverConfig, size: str, *, best: str | None = None
) -> None:
    root = driver.campaign_root(cfg)
    tags = {s.key: TAG for s in numbers.required_pools(RUNGS)}
    for leaf in pairs.screen_leaves(size, RUNGS):
        treated = leaf.kind == pairs.TREATMENT
        value = (
            0.9 if (treated and leaf.recipe.key == best) else (0.5 if treated else 0.1)
        )
        _finish(expected.leaf_dir(root, leaf, tags), value)
        parent = expected.param_dir(root, leaf, tags)
        (parent / expected.CAPABILITY_SUMMARY).write_text(
            json.dumps({"perplexity": {"perplexity_ratio": 1.1}})
        )


# ---- preflight ---------------------------------------------------------------


def test_missing_staging_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(driver.CampaignGateError, match="stage_offxsize_assets"):
        driver.run(_cfg(tmp_path))


def test_dry_run_walks_every_phase_and_writes_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    result = driver.run(cfg)
    assert driver.T_PREFLIGHT in result.fired
    assert driver.T_SUBMIT_P1A in result.fired
    assert not (tmp_path / "queue").exists()
    assert not cfg.state_path.exists()


def test_config_must_be_bound_to_this_campaign(tmp_path: Path) -> None:
    from subliminal_learning.campaign import xsize_plan

    cfg = driver.driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "q",
        repo_root=tmp_path,
        spec=xsize_plan.SPEC,
    )
    with pytest.raises(ValueError, match="offxsize_plan.SPEC"):
        driver.run(cfg)


# ---- P1 + teacher gate --------------------------------------------------------


def test_p1a_submits_the_ladder_probes_evals_and_priors(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver.run(cfg)
    for queue, count in (
        (plan.Q_P1_TEACHERS, 14),
        (plan.Q_P1_PROBES, 21),
        (plan.Q_P1_TEACHER_EVALS, 3),
        (plan.Q_P1_BASE_PRIORS, 3),
    ):
        assert len(store.load_jobs(cfg.queue_root, queue)) == count, queue
    meta = store.read_meta(cfg.queue_root, plan.Q_P1_PROBES)
    assert meta.depends_on == [plan.Q_P1_TEACHERS]


def test_re_advance_over_a_submitted_queue_is_a_no_op(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver.run(cfg)
    before = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.Q_P1_TEACHERS)}
    second = driver.run(cfg)
    assert driver.T_SUBMIT_P1A not in second.fired
    after = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.Q_P1_TEACHERS)}
    assert before == after


def test_the_gate_waits_for_its_queues(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver.run(cfg)
    result = driver.run(cfg)
    assert driver.T_TEACHER_GATE not in result.fired
    assert any("offxsize-p1-teachers" in msg for msg in result.waiting)


def test_the_gate_writes_its_report_then_acts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver.run(cfg)
    for queue in (
        plan.Q_P1_TEACHERS,
        plan.Q_P1_PROBES,
        plan.Q_P1_TEACHER_EVALS,
        plan.Q_P1_BASE_PRIORS,
    ):
        _drain(cfg, queue)
    with pytest.raises(
        driver.CampaignGateError, match="no size has an admissible teacher"
    ):
        driver.run(cfg)
    # The report exists even though the gate then refused to continue.
    assert (cfg.logs_dir / plan.TEACHER_GATE_REPORT).is_file()
    assert (cfg.logs_dir / plan.TEACHER_LADDER_CSV).is_file()


# ---- the manual pause ---------------------------------------------------------


def test_the_pause_waits_and_never_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    state = {
        "transitions": {
            driver.T_PREFLIGHT: {},
            driver.T_SUBMIT_P1A: {},
            driver.T_TEACHER_GATE: {},
            driver.T_SUBMIT_P1B: {},
        }
    }
    driver_io.save_state(cfg, state)
    _plant_gate(cfg)
    result = driver.run(cfg)
    assert driver.T_NUMBERS_READY not in result.fired
    assert any(
        "numbers:" in msg and "offxsize_numbers plan" in msg for msg in result.waiting
    )
    assert not (cfg.logs_dir / plan.NUMBERS_GATE_REPORT).exists()
    assert (cfg.logs_dir / plan.NUMBERS_QC_CSV).is_file()


def test_the_pause_clears_once_the_pools_verify(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver_io.save_state(
        cfg,
        {
            "transitions": {
                driver.T_PREFLIGHT: {},
                driver.T_SUBMIT_P1A: {},
                driver.T_TEACHER_GATE: {},
                driver.T_SUBMIT_P1B: {},
            }
        },
    )
    _plant_gate(cfg)
    ready = [
        numbers.PoolStatus(
            spec=s,
            ready=True,
            reasons=(),
            n_raw=100_000,
            n_filtered=50_000,
            keep_rate=0.5,
            filtered_sha256=f"sha-{s.key}",
            dataset_tag=TAG,
        )
        for s in numbers.required_pools(RUNGS)
    ]
    monkeypatch.setattr(driver, "pool_statuses", lambda cfg, rungs: ready)
    result = driver.run(cfg)
    assert driver.T_NUMBERS_READY in result.fired
    doc = json.loads((cfg.logs_dir / plan.NUMBERS_GATE_REPORT).read_text())
    assert doc["ready"] is True
    assert len(doc["pools"]) == 17


def test_the_frozen_numbers_report_is_not_rewritten(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver_io.save_state(
        cfg,
        {
            "transitions": {
                driver.T_PREFLIGHT: {},
                driver.T_SUBMIT_P1A: {},
                driver.T_TEACHER_GATE: {},
                driver.T_SUBMIT_P1B: {},
                driver.T_NUMBERS_READY: {},
            }
        },
    )
    _plant_gate(cfg)
    _plant_numbers(cfg)
    before = (cfg.logs_dir / plan.NUMBERS_GATE_REPORT).read_text()
    called: list[int] = []
    monkeypatch.setattr(
        driver, "pool_statuses", lambda cfg, rungs: called.append(1) or []
    )
    driver.run(cfg)
    assert called == [], "a frozen gate must not re-verify pools"
    assert (cfg.logs_dir / plan.NUMBERS_GATE_REPORT).read_text() == before


def test_dataset_tags_and_shas_come_from_the_frozen_report(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    _plant_numbers(cfg)
    tags, shas = driver._frozen_numbers(cfg)
    assert tags["rng"] == TAG
    assert shas["base/70m"] == "sha-base/70m"


# ---- P2 -> pick -> P3 ---------------------------------------------------------


def _ready_for_p2(tmp_path: Path) -> CampaignDriverConfig:
    cfg = _cfg(tmp_path)
    _plant_s0(cfg)
    driver_io.save_state(
        cfg,
        {
            "transitions": {
                driver.T_PREFLIGHT: {},
                driver.T_SUBMIT_P1A: {},
                driver.T_TEACHER_GATE: {},
                driver.T_SUBMIT_P1B: {},
                driver.T_NUMBERS_READY: {},
            }
        },
    )
    _plant_gate(cfg)
    _plant_numbers(cfg)
    return cfg


def test_p2_is_submitted_per_size(tmp_path: Path) -> None:
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    assert len(store.load_jobs(cfg.queue_root, plan.q_p2_screen("70m"))) == 9
    assert len(store.load_jobs(cfg.queue_root, plan.q_p2_screen("410m"))) == 8
    assert len(store.load_jobs(cfg.queue_root, plan.q_p2_controls("160m"))) == 9


def test_the_recipe_gate_fires_only_for_a_finished_size(tmp_path: Path) -> None:
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
    _finish_screen(cfg, "70m", best="ep20_lr1e-4_r8a32")
    result = driver.run(cfg)
    assert driver.t_submit_p3("70m") in result.fired
    assert driver.t_submit_p3("160m") not in result.fired
    picks = json.loads((cfg.logs_dir / plan.SCREEN_PICKS).read_text())
    assert picks["70m"]["recipe_key"] == "ep20_lr1e-4_r8a32"
    assert "160m" not in picks


def test_a_persisted_pick_is_read_not_recomputed(tmp_path: Path) -> None:
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
    # Plant a pick that disagrees with what the screen rows would choose.
    (cfg.logs_dir / plan.SCREEN_PICKS).write_text(
        json.dumps(
            {
                "70m": {
                    "recipe_key": "ep10_lr1e-4_r32a4",
                    "flagged": False,
                    "runner_up_keys": [],
                    "reasons": [],
                }
            }
        )
    )
    _finish_screen(cfg, "70m", best="ep20_lr1e-4_r8a32")
    driver.run(cfg)
    picks = json.loads((cfg.logs_dir / plan.SCREEN_PICKS).read_text())
    assert picks["70m"]["recipe_key"] == "ep10_lr1e-4_r32a4"
    cmds = [j.cmd for j in store.load_jobs(cfg.queue_root, plan.q_p3_students("70m"))]
    assert all("--recipe-key ep10_lr1e-4_r32a4" in c for c in cmds)


def test_p3_submission_wires_the_tails_and_their_dependencies(tmp_path: Path) -> None:
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
        _finish_screen(cfg, size)
    driver.run(cfg)
    assert len(store.load_jobs(cfg.queue_root, plan.q_p3_students("160m"))) == 27
    assert len(store.load_jobs(cfg.queue_root, plan.q_p3_controls("160m"))) == 12
    assert len(store.load_jobs(cfg.queue_root, plan.Q_P3C)) == 12
    for size in plan.P3B_SIZES:
        meta = store.read_meta(cfg.queue_root, plan.q_p3b_evals(size))
        assert plan.q_p4_evals(size) in meta.depends_on, "P3b shares parents with P3"
    assert not store.queue_exists(cfg.queue_root, plan.q_p3b("410m"))


def test_every_eval_queue_is_gated_by_a_base_cache_warm(tmp_path: Path) -> None:
    """A capability cell REFUSES to run on a cold shared cache, so the warm must precede it.

    Without this the P2 evals all fail, the recipe gate never fires, and the campaign
    deadlocks after the manual pause.
    """
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
        _finish_screen(cfg, size)
    driver.run(cfg)
    for size in plan.SIZES:
        for evals, warm in (
            (plan.q_p2_evals(size), plan.q_p2_warm(size)),
            (plan.q_p4_evals(size), plan.q_p4_warm(size)),
            (plan.q_p6_benchmarks(size), plan.q_p6_warm(size)),
        ):
            assert store.queue_exists(cfg.queue_root, warm), warm
            assert warm in store.read_meta(cfg.queue_root, evals).depends_on, evals
            assert plan.QUEUE_PRIORITIES[warm] > plan.QUEUE_PRIORITIES[evals]
    for name in (plan.Q_P3C_EVALS, plan.Q_P3D_EVALS):
        assert store.queue_exists(cfg.queue_root, name), name
    assert (
        plan.Q_P3C_WARM in store.read_meta(cfg.queue_root, plan.Q_P3C_EVALS).depends_on
    )
    assert (
        plan.Q_P3D_WARM in store.read_meta(cfg.queue_root, plan.Q_P3D_EVALS).depends_on
    )


def test_the_lm_eval_backfill_warms_the_benchmark_stage_it_needs(
    tmp_path: Path,
) -> None:
    """Regression: every P6 cell failed `base_metrics_ready` on a cold benchmark block.

    The P4 warm fills forced choice and perplexity only. A benchmark cell asks for a
    third stage, so gating it on that warm is not enough -- the P6 queue needs a warm
    that itself passes `--benchmarks`, or all 39 jobs die before loading a model.
    """
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
        _finish_screen(cfg, size)
    driver.run(cfg)
    for size in plan.SIZES:
        warm = store.load_jobs(cfg.queue_root, plan.q_p6_warm(size))
        bench = store.load_jobs(cfg.queue_root, plan.q_p6_benchmarks(size))
        assert warm, size
        assert all("--benchmarks" in j.cmd for j in warm), size
        assert all("--benchmarks" in j.cmd for j in bench), size
        assert all("warm-eval-base" in j.cmd for j in warm), size
        p4_warm = store.load_jobs(cfg.queue_root, plan.q_p4_warm(size))
        assert not any("--benchmarks" in j.cmd for j in p4_warm), size


def test_the_concordance_arm_covers_angelas_recipe_at_her_own_rung(
    tmp_path: Path,
) -> None:
    """O5 reads one cell; the screen and the dose curve each only vary one of its tags."""
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
        _finish_screen(cfg, size)
    driver.run(cfg)
    built = store.load_jobs(cfg.queue_root, plan.Q_P3D)
    assert len(built) == 2 * len(plan.STUDENT_SEEDS)
    treatments = [j for j in built if "--kind treatment" in j.cmd]
    assert len(treatments) == len(plan.STUDENT_SEEDS)
    rung = str(plan.ANGELA_REFERENCE["teacher_tag"])
    recipe = str(plan.ANGELA_REFERENCE["student_tag"])
    assert all(f"--rung-tag {rung}" in j.cmd for j in treatments)
    assert all(f"--recipe-key {recipe}" in j.cmd for j in built)
    assert all("--teacher-size 160m --student-size 160m" in j.cmd for j in built)


def test_the_concordance_arm_is_empty_when_the_gate_picked_both_tags(
    tmp_path: Path,
) -> None:
    """The plan's `no extra cells` case: nothing is queued when P3 already is the cell."""
    centre = plan.STUDENT_CENTRE
    assert not pairs.concordance_leaves(
        centre, str(plan.ANGELA_REFERENCE["teacher_tag"])
    )
    assert pairs.concordance_leaves(centre, "ep10_lr1e-4_r16a32")


def test_status_reports_gates_without_writing(tmp_path: Path) -> None:
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    snap = driver.status(cfg)
    assert snap["numbers_ready"] is True
    assert snap["teacher_rungs"] == RUNGS
    assert snap["done"] is False
    assert driver.T_NUMBERS_READY in snap["transitions"]


def test_campaign_done_ignores_the_optional_tails(tmp_path: Path, monkeypatch) -> None:
    cfg = _ready_for_p2(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        _drain(cfg, plan.q_p2_evals(size))
        _finish_screen(cfg, size)
    driver.run(cfg)
    required = driver._required_queue_names(cfg)
    assert plan.q_p3b("160m") not in required
    assert plan.q_p6_benchmarks("70m") not in required
    assert plan.q_p3_students("410m") in required
    for name in required:
        _drain(cfg, name)
    monkeypatch.setattr(
        driver,
        "_finalize",
        lambda cfg, state: state["transitions"].update(
            {driver.T_FINALIZE: {"at": "t"}}
        ),
    )
    result = driver.run(cfg)
    assert driver.T_FINALIZE in result.fired
