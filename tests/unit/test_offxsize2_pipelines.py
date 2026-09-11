"""v2 driver: the stop gate holds, the node is asserted, deadline and tail behave."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pipelines import offxsize2_campaign_pipeline as driver
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_jobs as jobs
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.jobqueue import store

TAG = f"{plan.PROMPT_SET_TAG}_{plan.FILTER_MODE}_n100k"
ALL_RUNGS: tuple[str, ...] = tuple(rung.key for rung in plan.TEACHER_RUNGS)
ADMITTED = {animal: {size: ALL_RUNGS for size in plan.SIZES} for animal in plan.ANIMALS}
TAIL_KEYS: tuple[str, ...] = tuple(f"admit_tail_{tier.key}" for tier in plan.TAIL_TIERS)
PICKS = [
    {
        "rung_tag": plan.CONCORDANCE_RUNG.key,
        "recipe_key": recipe.key,
        "score": 0.5,
        "n_pairs": 6,
    }
    for recipe in plan.STUDENT_GRID[: plan.CONFIRM_N_CONFIGS]
]


# ---- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_code_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin needs a real git worktree; these tests exercise the state machine."""
    monkeypatch.setattr(
        "pipelines.offxsize2_campaign_pipeline.driver_io.queue_job_defaults",
        lambda _cfg: {},
    )


@pytest.fixture(autouse=True)
def _on_campaign_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every submitting test runs as if it were on the campaign node."""
    monkeypatch.setattr(driver.socket, "gethostname", lambda: plan.CAMPAIGN_NODE)


# ---- helpers -----------------------------------------------------------------


def _cfg(tmp_path: Path, *, dry_run: bool = False) -> CampaignDriverConfig:
    return driver.driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        dry_run=dry_run,
        auto_requeue=0,
    )


def _plant_preflight(cfg: CampaignDriverConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.PREFLIGHT_REPORT).write_text(json.dumps({"passed": True}))
    (cfg.logs_dir / plan.ADOPTION_MANIFEST).write_text(json.dumps({"adopted": []}))


def _plant_gate(cfg: CampaignDriverConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.TEACHER_GATE_REPORT).write_text(
        json.dumps(
            {
                "at": "t0",
                "admitted": {
                    a: {s: list(t) for s, t in v.items()} for a, v in ADMITTED.items()
                },
                "ladder": [],
            }
        )
    )


def _plant_numbers(cfg: CampaignDriverConfig, *, keep_rate: float = 0.5) -> None:
    statuses = [
        numbers.PoolStatus(
            spec=expected.pool_spec(leaf),
            ready=True,
            reasons=(),
            n_raw=100_000,
            n_filtered=50_000,
            keep_rate=keep_rate,
            filtered_sha256=f"sha-{leaf.pool_key}",
            dataset_tag=TAG,
        )
        for leaf in expected.pool_leaves(driver.pool_coverage_leaves(ADMITTED))
    ]
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.NUMBERS_GATE_REPORT).write_text(
        json.dumps(numbers.gate_report(statuses))
    )


def _plant_probes(cfg: CampaignDriverConfig, keep: float = 0.5) -> None:
    """A frozen format probe per pool: what sizes every generation cell."""
    root = driver.campaign_root(cfg)
    for leaf in expected.pool_leaves(driver.pool_coverage_leaves(ADMITTED)):
        probe = expected.pool_dir(root, leaf) / plan.PROBE_SUBDIR
        probe.mkdir(parents=True, exist_ok=True)
        (probe / plan.PROBE_REPORT_FILE).write_text(json.dumps({"keep_rate": keep}))


def _plant_concordance(cfg: CampaignDriverConfig, *, faithful: bool = True) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.CONCORDANCE_GATE_REPORT).write_text(
        json.dumps(
            {
                "faithful": faithful,
                "p_target": plan.CONCORDANCE_P_TARGET,
                "reference_p_target": plan.CONCORDANCE_P_TARGET,
                "reasons": [] if faithful else ["transfer_absent"],
            }
        )
    )


def _plant_picks(cfg: CampaignDriverConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.SCREEN_PICKS).write_text(json.dumps({"picks": PICKS}))


def _plant_state(cfg: CampaignDriverConfig, transitions: list[str], **extra) -> None:
    doc = {"transitions": {name: {"at": "t0"} for name in transitions}, **extra}
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_path.write_text(json.dumps(doc))


def _tags() -> dict[str, str]:
    return {
        leaf.pool_key: TAG
        for leaf in expected.pool_leaves(driver.pool_coverage_leaves(ADMITTED))
    }


def _drain(
    cfg: CampaignDriverConfig,
    name: str,
    *,
    minutes: float = 0.0,
    started: str = "2026-09-13T00:00:00+00:00",
) -> None:
    """Mark every job in a queue finished, optionally with a measurable duration."""
    began = datetime.fromisoformat(started)
    ended = began + timedelta(minutes=minutes)
    for job in store.load_jobs(cfg.queue_root, name):
        store.record_result(
            cfg.queue_root,
            name,
            job,
            rc=0,
            node=plan.CAMPAIGN_NODE,
            gpu=0,
            start_utc=began.isoformat(),
            end_utc=ended.isoformat(),
            log="",
            attempt=1,
        )


def _finish(leaf_dir: Path) -> None:
    ckpt = leaf_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"0")
    (leaf_dir / "train_manifest.json").write_text("{}")


def _finish_leaf(
    cfg: CampaignDriverConfig,
    leaf: pairs.Leaf,
    p_target: float,
    *,
    ppl_ratio: float = 1.1,
    rank1_frac: float = 1.0,
) -> None:
    """Write one cell's checkpoint, its per-animal preference eval, its capability."""
    root = driver.campaign_root(cfg)
    tags = _tags()
    _finish(expected.leaf_dir(root, leaf, tags))
    for animal in plan.ANIMALS if leaf.path_animal is None else (leaf.animal,):
        path = expected.preference_path(root, leaf, tags, animal)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sets": {
                        plan.PRIMARY_PROMPT_SET: {
                            "p_target": p_target,
                            "rank1_frac": rank1_frac,
                        }
                    }
                }
            )
        )
    parent = expected.param_dir(root, leaf, tags)
    parent.mkdir(parents=True, exist_ok=True)
    (parent / expected.CAPABILITY_SUMMARY).write_text(
        json.dumps({"perplexity": {"perplexity_ratio": ppl_ratio}})
    )


def _record_results(sink: list[dict]):
    """Stand-in for the results pipeline that records what the driver hands it."""

    def _fake(_cfg, state, holes, *, partial) -> None:
        sink.append(
            {
                "partial": partial,
                "holes": [dict(h) for h in holes],
                "deadline_windows": list(state.get("deadline_windows") or []),
            }
        )

    return _fake


def _plant_base_priors(cfg: CampaignDriverConfig, prior: float = 0.01) -> None:
    for animal in plan.ANIMALS:
        for size in plan.SIZES:
            path = cfg.logs_dir / plan.BASE_PRIORS_DIRNAME / animal / f"{size}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"sets": {plan.PRIMARY_PROMPT_SET: {"p_target": prior}}})
            )


def _ready_for_p1(tmp_path: Path) -> CampaignDriverConfig:
    """A campaign whose P0 gates are all frozen on disk."""
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    _plant_gate(cfg)
    _plant_numbers(cfg)
    _plant_concordance(cfg)
    _plant_state(
        cfg,
        [
            driver.T_PREFLIGHT,
            driver.T_SUBMIT_P0,
            driver.T_TEACHER_GATE,
            driver.T_SUBMIT_POOLS,
            driver.T_NUMBERS_READY,
            driver.T_SUBMIT_CONCORDANCE,
            driver.T_CONCORDANCE_GATE,
        ],
    )
    return cfg


# ---- binding and the dry-run walk --------------------------------------------


def test_the_config_must_be_bound_to_this_campaign(tmp_path: Path) -> None:
    from subliminal_learning.campaign import offxsize_plan as v1_plan

    cfg = driver.driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        spec=v1_plan.SPEC,
    )
    with pytest.raises(ValueError, match="offxsize2_plan.SPEC"):
        driver.run(cfg)


def test_a_dry_run_walks_every_phase_and_writes_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    result = driver.run(cfg)
    for name in (
        driver.T_PREFLIGHT,
        driver.T_SUBMIT_P0,
        driver.T_TEACHER_GATE,
        driver.T_SUBMIT_POOLS,
        driver.T_NUMBERS_READY,
        driver.T_SUBMIT_CONCORDANCE,
        driver.T_CONCORDANCE_GATE,
        driver.t_submit_p1("410m"),
        driver.T_CONFIRM_PICK,
        driver.t_submit_p2("70m"),
        driver.T_FINALIZE,
    ):
        assert name in result.fired, name
    assert result.done
    assert not (tmp_path / "queue").exists()
    assert not cfg.state_path.exists()
    assert not cfg.logs_dir.exists()


def test_missing_staging_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(driver.CampaignGateError, match="stage_offxsize2_assets"):
        driver.run(_cfg(tmp_path))


# ---- the node assertion ------------------------------------------------------


def test_submission_is_refused_on_the_node_running_the_live_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jobq pins a queue to the submitting host: the wrong clone must never submit."""
    monkeypatch.setattr(driver.socket, "gethostname", lambda: plan.FORBIDDEN_NODE)
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    with pytest.raises(driver.CampaignGateError, match=plan.FORBIDDEN_NODE):
        driver.run(cfg)
    assert not (tmp_path / "queue").exists()


def test_submission_is_refused_on_any_host_but_the_campaign_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver.socket, "gethostname", lambda: "some-other-box")
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    with pytest.raises(driver.CampaignGateError, match=plan.CAMPAIGN_NODE):
        driver.run(cfg)
    assert not (tmp_path / "queue").exists()


def test_the_node_assertion_reads_the_declared_submission_node() -> None:
    assert driver.require_campaign_node() == plan.CAMPAIGN_NODE
    submission = jobs.submission(plan.Q_P0_BASE_PRIORS, jobs.base_prior_jobs())
    assert submission.node == plan.CAMPAIGN_NODE


# ---- P0 ----------------------------------------------------------------------


def test_p0_queues_both_animals_teachers_probes_and_priors(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    driver.run(cfg)
    n_rungs = len(plan.TEACHER_RUNGS)
    n_sizes = len(plan.SIZES)
    n_animals = len(plan.ANIMALS)
    assert (
        len(store.load_jobs(cfg.queue_root, plan.Q_P0_TEACHERS))
        == n_animals * n_sizes * n_rungs
    )
    # Every teacher probe plus one per untuned base: the base pools are the Delta
    # denominators and are sized from a measured keep rate like any other pool.
    assert (
        len(store.load_jobs(cfg.queue_root, plan.Q_P0_PROBES))
        == n_animals * n_sizes * n_rungs + n_sizes
    )
    assert len(store.load_jobs(cfg.queue_root, plan.Q_P0_BASE_PRIORS)) == n_sizes
    meta = store.read_meta(cfg.queue_root, plan.Q_P0_PROBES)
    assert meta.depends_on == [plan.Q_P0_TEACHERS]
    assert meta.node == plan.CAMPAIGN_NODE


def test_re_advance_over_a_submitted_queue_is_a_no_op(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    driver.run(cfg)
    before = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.Q_P0_TEACHERS)}
    second = driver.run(cfg)
    assert driver.T_SUBMIT_P0 not in second.fired
    after = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.Q_P0_TEACHERS)}
    assert before == after


def test_the_teacher_gate_waits_for_its_queues(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    driver.run(cfg)
    result = driver.run(cfg)
    assert driver.T_TEACHER_GATE not in result.fired
    assert any(plan.Q_P0_TEACHERS in msg for msg in result.waiting)


def test_a_persisted_teacher_gate_is_read_not_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    _plant_gate(cfg)
    _plant_probes(cfg)
    _plant_state(cfg, [driver.T_PREFLIGHT, driver.T_SUBMIT_P0, driver.T_TEACHER_GATE])
    called: list[int] = []
    monkeypatch.setattr(driver, "teacher_readings", lambda _cfg: called.append(1) or {})
    driver.run(cfg)
    assert called == [], "a frozen gate must not re-read the ladder"
    assert driver.T_SUBMIT_POOLS in driver_io.load_state(cfg)["transitions"]


def test_the_pools_are_sized_from_each_probes_own_keep_rate(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    _plant_gate(cfg)
    _plant_state(cfg, [driver.T_PREFLIGHT, driver.T_SUBMIT_P0, driver.T_TEACHER_GATE])
    root = driver.campaign_root(cfg)
    for leaf in expected.pool_leaves(driver.pool_coverage_leaves(ADMITTED)):
        probe = expected.pool_dir(root, leaf) / plan.PROBE_SUBDIR
        probe.mkdir(parents=True, exist_ok=True)
        keep = 0.5 if leaf.kind == pairs.TREATMENT else 0.1
        (probe / plan.PROBE_REPORT_FILE).write_text(json.dumps({"keep_rate": keep}))
    driver.run(cfg)
    cmds = [j.cmd for j in store.load_jobs(cfg.queue_root, plan.Q_P0_POOLS)]
    teacher = [c for c in cmds if "--source teacher" in c]
    base = [c for c in cmds if "--source base" in c]
    assert teacher and base
    assert all("--n-prompts 100000" in c for c in teacher)
    # keep 0.1 is under LOW_KEEP_THRESHOLD, so the ladder's first rung is skipped.
    assert all("--n-prompts 300000" in c for c in base)


# ---- failure budget ----------------------------------------------------------


def test_a_small_queue_tolerates_one_failure_instead_of_none(tmp_path: Path) -> None:
    """A bare 2% rate tolerates NOTHING below 50 cells.

    Twelve committed queues are 20-39 cells, so one cell that failed twice ended the
    whole 640-cell campaign. The floor of one turns that into the recorded hole
    `coverage.json` already exists to carry.
    """
    cfg = _cfg(tmp_path)
    opts = driver.Offxsize2DriverOptions()
    assert driver._failure_budget(plan.q_p2_controls("160m"), 20, opts) == 1
    assert driver._failure_budget(plan.q_p1_warm("160m"), 28, opts) == 1
    # The rate still governs once it exceeds the floor: 2% of 156 is 3.
    assert driver._failure_budget(plan.q_p1("160m"), 156, opts) == 3
    # A droppable tier is droppable whole; the port gate decides two cells and gets none.
    assert driver._failure_budget(plan.q_tail("t1"), 40, opts) == 40
    assert driver._failure_budget(plan.Q_P0_CONCORDANCE, 2, opts) == 0
    assert cfg.run_set_id == plan.RUN_SET_ID


def test_a_cell_held_by_a_recorded_hole_does_not_block_the_phase_forever(
    tmp_path: Path,
) -> None:
    """`_queue_ready` calls a queue complete-with-holes; `_leaves_ready` must agree.

    While they disagreed the driver spun at its poll interval printing "1 cell(s) not
    trained" for a cell it had already excused, and never wrote CAMPAIGN_DONE.
    """
    cfg = _cfg(tmp_path)
    leaves = pairs.concordance_leaves()
    tags = _tags()
    waiting: list[str] = []
    # Nothing on disk: without a hole the phase correctly waits.
    assert not driver._leaves_ready(cfg, leaves, tags, waiting, "p1")
    group = expected.training_groups([leaves[0]])[0]
    holes = [
        {
            "queue": "offxsize2-p1-160m",
            "jobkeys": [f"p1|{jobs.train_job_identity(group)}"],
        }
    ]
    # The treatment is excused; the control still legitimately blocks.
    _finish_leaf(cfg, leaves[1], 0.011)
    assert driver._leaves_ready(cfg, leaves, tags, waiting, "p1", holes)


# ---- the concordance stop gate -----------------------------------------------


def _ready_for_concordance(tmp_path: Path) -> CampaignDriverConfig:
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    _plant_gate(cfg)
    # The gate compares this pool's keep RATE against her published one -- a rate, so the
    # two pools' differing row counts cancel.
    _plant_numbers(cfg, keep_rate=plan.CONCORDANCE_KEEP_RATE)
    _plant_base_priors(cfg)
    _plant_state(
        cfg,
        [
            driver.T_PREFLIGHT,
            driver.T_SUBMIT_P0,
            driver.T_TEACHER_GATE,
            driver.T_SUBMIT_POOLS,
            driver.T_NUMBERS_READY,
        ],
    )
    return cfg


def _run_concordance(cfg: CampaignDriverConfig, **leaf_kwargs) -> None:
    """Drive P0 to the gate with the treatment and its control both finished."""
    driver.run(cfg)
    _drain(cfg, plan.Q_P0_CONCORDANCE)
    treatment, control = pairs.concordance_leaves()
    control_p = leaf_kwargs.pop("control_p_target", 0.011)
    _finish_leaf(cfg, treatment, leaf_kwargs.pop("p_target"), **leaf_kwargs)
    _finish_leaf(cfg, control, control_p)


def test_the_concordance_gate_blocks_p1_when_the_port_shows_no_transfer(
    tmp_path: Path,
) -> None:
    cfg = _ready_for_concordance(tmp_path)
    _run_concordance(cfg, p_target=0.02)
    with pytest.raises(driver.CampaignGateError, match="port fidelity"):
        driver.run(cfg)
    # The report is written BEFORE the raise, and the verdict is authoritative after.
    report = json.loads((cfg.logs_dir / plan.CONCORDANCE_GATE_REPORT).read_text())
    assert report["faithful"] is False
    assert "transfer_absent" in " ".join(report["reasons"])
    assert not store.queue_exists(cfg.queue_root, plan.q_p1("160m"))
    with pytest.raises(driver.CampaignGateError, match="concordance gate FAILED"):
        driver.run(cfg)
    assert not store.queue_exists(cfg.queue_root, plan.q_p1("160m"))


def test_the_concordance_gate_blocks_p1_when_the_matched_control_has_drifted(
    tmp_path: Path,
) -> None:
    """A control far above its prior means the floor moved, so Delta is uninterpretable."""
    cfg = _ready_for_concordance(tmp_path)
    _run_concordance(cfg, p_target=0.81, control_p_target=0.9)
    with pytest.raises(driver.CampaignGateError, match="control_above"):
        driver.run(cfg)


def test_the_concordance_gate_blocks_p1_when_generation_keeps_a_different_fraction(
    tmp_path: Path,
) -> None:
    """The corpus-invariant check: a wrong teacher or filter moves the keep RATE."""
    cfg = _cfg(tmp_path)
    _plant_preflight(cfg)
    _plant_gate(cfg)
    _plant_numbers(cfg, keep_rate=plan.CONCORDANCE_KEEP_RATE + 0.30)
    _plant_base_priors(cfg)
    _plant_state(
        cfg,
        [
            driver.T_PREFLIGHT,
            driver.T_SUBMIT_P0,
            driver.T_TEACHER_GATE,
            driver.T_SUBMIT_POOLS,
            driver.T_NUMBERS_READY,
        ],
    )
    _run_concordance(cfg, p_target=0.81)
    with pytest.raises(driver.CampaignGateError, match="generation_keep_rate"):
        driver.run(cfg)


def test_the_concordance_gate_releases_p1_on_a_corpus_draw_that_misses_her_endpoint(
    tmp_path: Path,
) -> None:
    """The regression this gate was re-specified for.

    v2's row floor forces this rung to generate a bigger pool than hers, so its student
    trains on different number rows and lands below her 0.92016 -- measured 0.81234 on
    2026-09-11. The method is identical, so the factorial must be released.
    """
    cfg = _ready_for_concordance(tmp_path)
    _run_concordance(cfg, p_target=0.8123442322015763, rank1_frac=0.98)
    result = driver.run(cfg)
    assert driver.T_CONCORDANCE_GATE in result.fired
    assert driver.t_submit_p1("160m") in result.fired
    report = json.loads((cfg.logs_dir / plan.CONCORDANCE_GATE_REPORT).read_text())
    assert report["faithful"] is True
    assert report["reasons"] == []
    # Her endpoint stays on the record as a covariate, with the deviation from it.
    assert report["reference_p_target"] == plan.CONCORDANCE_P_TARGET
    assert report["p_target_gap"] == pytest.approx(-0.1078151, abs=1e-6)


# ---- P1 -> the confirm pick -> P2 --------------------------------------------


def test_p1_submits_a_column_with_its_controls_warm_and_capability(
    tmp_path: Path,
) -> None:
    cfg = _ready_for_p1(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        expected_cells = (
            len(plan.SIZES) * len(pairs.p1_rungs(size)) * len(plan.STUDENT_GRID)
        )
        assert len(store.load_jobs(cfg.queue_root, plan.q_p1(size))) == expected_cells
        assert len(store.load_jobs(cfg.queue_root, plan.q_p1_controls(size))) == len(
            plan.SIZES
        ) * len(plan.STUDENT_GRID)
        evals = store.read_meta(cfg.queue_root, plan.q_p1_evals(size))
        assert evals.depends_on == [plan.q_p1_warm(size)]
        assert (
            plan.QUEUE_PRIORITIES[plan.q_p1_warm(size)]
            > plan.QUEUE_PRIORITIES[plan.q_p1_evals(size)]
        )


def test_the_warm_queue_covers_the_trees_the_confirm_pick_can_reach(
    tmp_path: Path,
) -> None:
    """A lion tree left cold would make every P2 lion eval refuse to start."""
    cfg = _ready_for_p1(tmp_path)
    driver.run(cfg)
    warmed = {
        j.cmd.split("--param-dir ")[1].split(" ")[0]
        for j in store.load_jobs(cfg.queue_root, plan.q_p1_warm("160m"))
    }
    trees = {str(Path(p).parent) for p in warmed}
    root = driver.campaign_root(cfg)
    for leaf in driver.p2_leaves("160m", PICKS, ADMITTED):
        parent = expected.param_dir(root, leaf, _tags())
        assert str(parent.parent) in trees, leaf.key


def _finish_p1(cfg: CampaignDriverConfig, *, best_recipe: str) -> None:
    """Train + score every factorial cell, with one (rung, recipe) clearly ahead."""
    _plant_base_priors(cfg)
    for size in plan.SIZES:
        for leaf in pairs.p1_control_leaves(size):
            _finish_leaf(cfg, leaf, 0.10)
        for leaf in driver.p1_treatments(size, ADMITTED):
            best = (
                leaf.recipe.key == best_recipe
                and leaf.rung_tag == plan.CONCORDANCE_RUNG.key
            )
            _finish_leaf(cfg, leaf, 0.90 if best else 0.20)


def test_the_confirm_pick_is_persisted_then_read_not_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ready_for_p1(tmp_path)
    driver.run(cfg)
    for size in plan.SIZES:
        for queue in (plan.q_p1(size), plan.q_p1_controls(size), plan.q_p1_evals(size)):
            _drain(cfg, queue)
    best = plan.STUDENT_GRID[2].key
    _finish_p1(cfg, best_recipe=best)
    result = driver.run(cfg)
    assert driver.T_CONFIRM_PICK in result.fired
    picks = json.loads((cfg.logs_dir / plan.SCREEN_PICKS).read_text())["picks"]
    assert len(picks) == plan.CONFIRM_N_CONFIGS
    assert picks[0]["recipe_key"] == best
    assert picks[0]["rung_tag"] == plan.CONCORDANCE_RUNG.key

    called: list[int] = []
    monkeypatch.setattr(
        driver, "confirm_rows", lambda *_a, **_k: called.append(1) or []
    )
    driver.run(cfg)
    assert called == [], "a frozen pick must not be re-scored"


def test_p2_confirms_both_animals_at_the_pre_registered_seed_counts(
    tmp_path: Path,
) -> None:
    cfg = _ready_for_p1(tmp_path)
    _plant_picks(cfg)
    _plant_state(
        cfg,
        [
            driver.T_PREFLIGHT,
            driver.T_SUBMIT_P0,
            driver.T_TEACHER_GATE,
            driver.T_SUBMIT_POOLS,
            driver.T_NUMBERS_READY,
            driver.T_SUBMIT_CONCORDANCE,
            driver.T_CONCORDANCE_GATE,
            driver.T_CONFIRM_PICK,
            *[driver.t_submit_p1(s) for s in plan.SIZES],
        ],
    )
    driver.run(cfg)
    cmds = [j.cmd for j in store.load_jobs(cfg.queue_root, plan.q_p2("70m"))]
    top = PICKS[0]["recipe_key"]
    top_cmds = [c for c in cmds if f"--recipe-key {top} " in f"{c} "]
    n_top = len(plan.SIZES) * len(plan.CONFIRM_STUDENT_SEEDS) * len(plan.ANIMALS)
    assert len(top_cmds) == n_top
    for animal in plan.ANIMALS:
        assert any(f"--animals {animal}" in c for c in top_cmds)
    runner = PICKS[1]["recipe_key"]
    runner_cmds = [c for c in cmds if f"--recipe-key {runner} " in f"{c} "]
    assert len(runner_cmds) == len(plan.SIZES) * len(
        plan.CONFIRM_RUNNER_UP_SEEDS
    ) * len(plan.ANIMALS)
    meta = store.read_meta(cfg.queue_root, plan.q_p2_evals("70m"))
    assert plan.q_p1_warm("70m") in meta.depends_on


# ---- the deadline ------------------------------------------------------------


def test_open_window_is_one_per_run_and_never_reopens_a_closed_one() -> None:
    opts = driver.Offxsize2DriverOptions(deadline_hours=48)
    state: dict = {"transitions": {}}
    first = driver._open_window(state, opts)
    again = driver._open_window(state, opts)
    assert again is first
    assert len(state["deadline_windows"]) == 1
    first["ended_at"] = "2020-01-01T00:00:00Z"
    first["reason"] = "deadline"
    nxt = driver._open_window(state, opts)
    assert nxt is not first
    assert nxt["ended_at"] is None
    assert len(state["deadline_windows"]) == 2


def test_the_deadline_finalizes_partially_once_and_never_stamps_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ready_for_p1(tmp_path)
    _plant_picks(cfg)
    built: list[dict] = []
    monkeypatch.setattr(driver, "_build_results", _record_results(built))
    state = json.loads(cfg.state_path.read_text())
    state["deadline_windows"] = [
        {
            "start": "2020-01-01T00:00:00Z",
            "hours": 1.0,
            "ended_at": None,
            "reason": None,
        }
    ]
    cfg.state_path.write_text(json.dumps(state))

    result = driver.run(cfg, driver.Offxsize2DriverOptions(deadline_hours=1.0))
    assert result.fired == (driver.T_FINALIZE_PARTIAL,)
    assert result.done
    assert [call["partial"] for call in built] == [True]
    # The windows that produced the partial reading reach `coverage.json` through the
    # results pipeline, which owns that file.
    assert built[0]["deadline_windows"][0]["hours"] == 1.0
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    after = json.loads(cfg.state_path.read_text())
    assert after["transitions"][driver.T_FINALIZE_PARTIAL]["partial"] is True
    assert after["deadline_windows"][0]["ended_at"]
    assert after["deadline_windows"][0]["reason"] == "deadline"
    # The pool is paused for the drain, and no P1 work is queued past the deadline.
    assert store.stop_path(cfg.queue_root, store.this_host()).exists()
    assert not store.queue_exists(cfg.queue_root, plan.q_p1("70m"))

    again = driver.run(cfg, driver.Offxsize2DriverOptions(deadline_hours=1.0))
    assert driver.T_FINALIZE_PARTIAL not in again.fired
    assert len(built) == 1
    # Re-running opens a fresh window, so the campaign resumes where it stopped.
    windows = json.loads(cfg.state_path.read_text())["deadline_windows"]
    assert len(windows) == 2
    assert windows[-1]["ended_at"] is None


# ---- the elastic tail --------------------------------------------------------


def _ready_for_tail(tmp_path: Path) -> CampaignDriverConfig:
    """Every committed queue submitted and drained, with the picks frozen."""
    cfg = _ready_for_p1(tmp_path)
    _plant_picks(cfg)
    _plant_state(
        cfg,
        [
            driver.T_PREFLIGHT,
            driver.T_SUBMIT_P0,
            driver.T_TEACHER_GATE,
            driver.T_SUBMIT_POOLS,
            driver.T_NUMBERS_READY,
            driver.T_SUBMIT_CONCORDANCE,
            driver.T_CONCORDANCE_GATE,
            driver.T_CONFIRM_PICK,
        ],
    )
    driver.run(cfg)
    for name in driver._live_queue_names(cfg):
        _drain(cfg, name, minutes=10)
    return cfg


def test_a_tail_tier_is_admitted_only_once_the_queues_drain(tmp_path: Path) -> None:
    cfg = _ready_for_tail(tmp_path)
    first = plan.TAIL_TIERS[0]
    # One committed queue still pending: the tail must not take the card.
    store.requeue(cfg.queue_root, plan.q_p2("70m"), failed_only=False)
    result = driver.run(cfg)
    assert driver.t_admit_tail(first.key) not in result.fired
    assert not store.queue_exists(cfg.queue_root, plan.q_tail(first.key))

    _drain(cfg, plan.q_p2("70m"), minutes=10)
    result = driver.run(cfg)
    assert driver.t_admit_tail(first.key) in result.fired
    assert store.load_jobs(cfg.queue_root, plan.q_tail(first.key))


def test_a_tail_tier_is_dropped_when_the_wall_clock_cannot_hold_it(
    tmp_path: Path,
) -> None:
    cfg = _ready_for_tail(tmp_path)
    first = plan.TAIL_TIERS[0]
    state = driver_io.load_state(cfg)
    capacity = driver.achieved_capacity(cfg, state, driver.Offxsize2DriverOptions())
    assert capacity > 0, "the measured card throughput is what converts GPU-min"
    # Half the wall-clock the tier's GPU-min converts to at that throughput.
    state["deadline_windows"][-1]["hours"] = first.est_gpu_min / capacity / 60.0 / 2.0
    driver_io.save_state(cfg, state)
    result = driver.run(cfg)
    assert driver.t_admit_tail(first.key) not in result.fired
    assert not store.queue_exists(cfg.queue_root, plan.q_tail(first.key))
    assert any(f"tail {first.key}" in msg for msg in result.waiting)


def test_an_eval_only_tier_is_recorded_as_skipped_not_queued(tmp_path: Path) -> None:
    """`t5` and `t6` train no student on this lineage, so they build no queue."""
    cfg = _ready_for_tail(tmp_path)
    for tier in plan.TAIL_TIERS:
        result = driver.run(cfg)
        if not result.fired:
            break
        for name in driver._live_queue_names(cfg):
            _drain(cfg, name, minutes=10)
        del tier
    transitions = driver_io.load_state(cfg)["transitions"]
    assert transitions[driver.t_admit_tail("t5")]["n_jobs"] == 0
    assert "skipped" in transitions[driver.t_admit_tail("t5")]
    assert not store.queue_exists(cfg.queue_root, plan.q_tail("t5"))
    assert store.queue_exists(cfg.queue_root, plan.q_tail("t2"))


def test_the_campaign_is_done_only_after_the_tail_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict] = []
    monkeypatch.setattr(driver, "_build_results", _record_results(built))
    monkeypatch.setattr(driver, "_leaves_ready", lambda *_a, **_k: True)
    cfg = _ready_for_tail(tmp_path)
    result = driver.run(cfg)
    assert driver.T_FINALIZE not in result.fired, "a tier is still admissible"
    for _ in plan.TAIL_TIERS:
        for name in driver._live_queue_names(cfg):
            _drain(cfg, name, minutes=10)
        result = driver.run(cfg)
        if driver.T_FINALIZE in result.fired:
            break
    assert driver.T_FINALIZE in result.fired
    assert [call["partial"] for call in built] == [False]
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    transitions = driver_io.load_state(cfg)["transitions"]
    assert all(t_admit_tail_key in transitions for t_admit_tail_key in TAIL_KEYS)


def test_capacity_falls_back_to_the_campaign_span_after_a_resume(
    tmp_path: Path,
) -> None:
    """A resumed run's fresh window holds no finished cell.

    Reading that as "nothing measured" would leave the driver unable to size a tier and
    therefore unable to finalize, on a pool that is demonstrably working.
    """
    cfg = _ready_for_p1(tmp_path)
    driver.run(cfg)
    for name in driver._live_queue_names(cfg):
        _drain(cfg, name, minutes=20, started="2026-09-01T00:00:00+00:00")
    fresh = {
        "deadline_windows": [
            {"start": driver._now(), "hours": 48.0, "ended_at": None, "reason": None}
        ]
    }
    capacity = driver.achieved_capacity(cfg, fresh, driver.Offxsize2DriverOptions())
    assert 0 < capacity <= len(plan.GPU_POLICY["gpus"])


# ---- status ------------------------------------------------------------------


def test_status_reports_the_gates_without_writing(tmp_path: Path) -> None:
    cfg = _ready_for_p1(tmp_path)
    snap = driver.status(cfg)
    assert snap["numbers_ready"] is True
    assert snap["concordance"]["faithful"] is True
    assert snap["admitted_rungs"]["lion"]["410m"] == list(ALL_RUNGS)
    assert snap["node"] == plan.CAMPAIGN_NODE
    assert snap["done"] is False
