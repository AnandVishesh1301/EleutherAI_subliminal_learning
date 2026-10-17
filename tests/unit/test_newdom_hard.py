"""`newdom` hard channel: recipes, gate, pool selection, jobs, metrics, CLI checks and
the driver end to end on fake jobs (temp roots, no GPU)."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import typer
from scipy import stats as scipy_stats

import scripts.run_newdom_hard_cell as cli
from pipelines import newdom_hard_campaign_pipeline as driver
from pipelines import newdom_hard_cell_pipeline as cellpipe
from pipelines import newdom_hard_results_pipeline as results
from pipelines import newdom_hard_staging_pipeline as staging
from subliminal_learning.campaign import driver_io, offxsize_plan
from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_hard_metrics as metrics
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.evaluation.prompt_sets import default_scoring, prompt_set_path
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import file_sha256

REPO = Path(__file__).resolve().parents[2]
R = plan.HARD_RECIPES
LR_LOW, LR_HEAD, LR_HIGH = plan.HARD_STUDENT_LRS


def test_pure_modules_import_no_heavy_or_io_modules():
    code = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ("
        "'subliminal_learning.campaign.newdom_hard',"
        "'subliminal_learning.campaign.newdom_hard_metrics')];"
        "print(','.join(m for m in sys.modules if m in ('torch','transformers','trl',"
        "'peft','datasets') or (m.startswith('subliminal_learning')"
        " and m.endswith('_io'))))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == ""


# ---- recipes, layout, jobs ------------------------------------------------------------


@pytest.mark.parametrize("recipe", R)
def test_teacher_recipe_round_trips_with_angela_knobs(recipe):
    rung = hard.teacher_rung(recipe)
    lora, train = hard.teacher_lora(recipe), hard.teacher_train(recipe)
    assert rung.key == recipe
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (
        rung.lora_r,
        rung.lora_alpha,
        0.0,
    )
    assert tuple(lora.target_modules) == offxsize_plan.TEACHER_TARGET_MODULES
    assert (train.batch_size, train.lr_scheduler_type, train.warmup_ratio) == (
        22,
        "cosine",
        0.05,
    )
    assert (train.max_grad_norm, train.seed, train.n_epochs) == (1.0, 1, rung.epochs)
    assert hard.teacher_leaf_dir("oak", recipe).parts[-4:] == (
        "oak",
        "pythia-160m",
        recipe,
        "ftseed1",
    )


def test_teacher_rung_rejects_an_off_plan_recipe():
    with pytest.raises(ValueError):
        hard.teacher_rung("ep3_lr1e-4_r16a32")


def test_student_recipe_is_hlopt_e3_lora_adamw_and_layout_shares_controls():
    lora, train = hard.train_cfgs(LR_HEAD, 2)
    assert (lora.r, lora.lora_alpha, lora.lora_dropout, lora.target_modules) == (
        8,
        32,
        0.0,
        "all-linear",
    )
    assert (train.batch_size, train.n_epochs, train.lr_scheduler_type) == (
        8,
        20,
        "cosine",
    )
    assert (train.warmup_ratio, train.max_grad_norm, train.optim, train.seed) == (
        0.05,
        1.0,
        "adamw_torch",
        2,
    )
    ctrl = hard.StudentCell(hard.BASE, None, None, LR_HEAD, 2)
    assert ctrl.eval_targets == plan.TARGETS
    assert hard.run_dir(ctrl).parts[-3:] == (
        hard.NUMBER_DATASET_TAG,
        "ep20_lr1e-4_r8a32",
        "ftseed2",
    )
    leaves = {hard.run_dir(c) for c in hard.treatment_cells("oak", R[0])}
    leaves |= {hard.run_dir(c) for c in hard.treatment_cells("oak", R[1])}
    assert len(leaves) == 18 and hard.run_dir(ctrl) not in leaves
    assert hard.cell_pool_dir(ctrl) == hard.CONTROL_POOL_DIR


def test_planned_counts_and_unique_keys():
    teachers = hard.teacher_tasks()
    controls = hard.control_cells()
    treat = [c for t in plan.TARGETS for r in R[:3] for c in hard.treatment_cells(t, r)]
    assert (len(teachers), len(controls), len(treat)) == (60, 9, 162)
    keys = [x.key for x in (*teachers, *controls, *treat)]
    assert len(set(keys)) == len(keys)


def test_jobs_round_trip_with_plan_footprints_and_priorities():
    cells = [
        hard.TeacherTask("purple", R[3]),
        hard.PoolTask("pine", R[7]),
        hard.treatment_cells("blue", R[9])[4],
        hard.control_cells()[8],
    ]
    jobs = [
        hard.teacher_job(cells[0]),
        hard.pool_job(cells[1]),
        hard.student_job(cells[2]),
        hard.student_job(cells[3]),
    ]
    kinds = ("hard_teacher", "hard_pool", "hard_student", "hard_student")
    for cell, job, kind in zip(cells, jobs, kinds, strict=True):
        assert hard.task_from_job(job) == cell
        fp = plan.FOOTPRINTS[kind]
        assert (job.mem_mib, job.slots) == (fp.mem_mib, fp.slots)
        assert job.cmd.startswith(f"uv run python -m {hard.CELL_MODULE} ")
    assert dict(hard.QUEUE_PRIORITIES) == {
        "newdom-h-teacher": 100,
        "newdom-h-pool": 90,
        "newdom-h-student": 50,
    }
    bad = type(jobs[0])(key="teacher|oak|" + R[0], cmd=jobs[0].cmd)
    with pytest.raises(ValueError, match="names"):
        hard.task_from_job(bad)


def test_eval_sets_are_registered_prob_sets_of_the_domain():
    for t in plan.TARGETS:
        names = hard.eval_sets(t, hard.STUDENT_SPLITS)
        noun = hard.domain_of(t).noun
        assert names == (
            f"{noun}_validation",
            f"{noun}_test",
            f"{noun}_favorite",
            f"{noun}_dislike",
            plan.OFF_TOPIC_SET,
        )
        assert all(default_scoring(n) == "prob" for n in names)
        assert all((REPO / prompt_set_path(n)).is_file() for n in names)
        assert hard.template_source(t) == Path(
            f"data/evaluation_prompts/{noun}_preference_train.json"
        )


# ---- gate -----------------------------------------------------------------------------


def _reading(p=0.5, base=0.01, sd=0.002, dce=1.0, finite=True, recipe=R[0]):
    return hard.TeacherReading("oak", recipe, p, base, sd, dce, finite)


@pytest.mark.parametrize(
    ("kwargs", "admitted", "reason"),
    [
        ({}, True, None),
        ({"p": 0.025}, False, "below_abs_min"),
        ({"p": 0.09, "sd": 0.02}, False, "below_base_plus_5sd"),
        ({"dce": 3.5}, False, "dce_above_max"),
        ({"dce": 3.0}, True, None),
        ({"finite": False}, False, "pp1_nonfinite"),
        ({"sd": math.nan}, False, "base_sd_missing"),
        ({"p": math.nan}, False, "p_target_missing"),
    ],
)
def test_admission_rules(kwargs, admitted, reason):
    ok, reasons = hard.admission(_reading(**kwargs))
    assert ok is admitted
    if reason:
        assert any(r.startswith(reason) for r in reasons)


def test_threshold_is_base_plus_max_of_floor_and_five_sd():
    assert _reading(base=0.01, sd=0.001).threshold == pytest.approx(0.03)
    assert _reading(base=0.01, sd=0.02).threshold == pytest.approx(0.11)


def test_sample_sd_is_ddof_one_over_finite_values():
    assert hard.sample_sd([0.0, 0.02, math.nan]) == pytest.approx(
        statistics.stdev([0.0, 0.02])
    )
    assert math.isnan(hard.sample_sd([0.1]))


def test_rank_rounds_validation_p_then_uses_angela_order():
    readings = [
        _reading(p=0.914, recipe=R[5]),
        _reading(p=0.906, recipe=R[2]),
        _reading(p=0.93, recipe=R[9]),
        _reading(p=0.99, dce=4.0, recipe=R[0]),
    ]
    assert hard.rank_admitted(readings) == [R[9], R[2], R[5]]


# ---- pools ----------------------------------------------------------------------------

RANKED = tuple(f"r{i}" for i in range(7))
U, S = hard.USABLE, hard.SHORT


def _plan(submitted, outcomes, ranked=RANKED):
    recipes = dict(zip(ranked, R, strict=False))
    out = hard.pool_plan(
        [recipes[r] for r in ranked],
        {recipes[r] for r in submitted},
        {recipes[r]: v for r, v in outcomes.items()},
    )
    back = {v: k for k, v in recipes.items()}
    return (
        tuple(back[r] for r in out.submit),
        tuple(back[r] for r in out.selected),
        out.final,
    )


@pytest.mark.parametrize(
    ("submitted", "outcomes", "want"),
    [
        ((), {}, (("r0", "r1", "r2", "r3"), (), False)),
        (RANKED[:4], dict.fromkeys(RANKED[:4], U), ((), ("r0", "r1", "r2"), True)),
        (
            RANKED[:4],
            {"r1": U, "r2": U, "r3": U},
            ((), ("r1", "r2"), False),
        ),
        (
            RANKED[:4],
            {"r0": S, "r1": U, "r2": U, "r3": U},
            ((), ("r1", "r2", "r3"), True),
        ),
        (
            RANKED[:4],
            {"r0": U, "r1": S, "r2": S, "r3": S},
            (("r4", "r5"), ("r0",), False),
        ),
        (
            RANKED[:6],
            {"r0": U, "r1": S, "r2": S, "r3": S, "r4": U, "r5": S},
            (("r6",), ("r0", "r4"), False),
        ),
        (RANKED, dict.fromkeys(RANKED, S), ((), (), True)),
    ],
)
def test_pool_plan(submitted, outcomes, want):
    assert _plan(submitted, outcomes) == want


def test_pool_plan_edges():
    assert _plan((), {}, ranked=("r0", "r1")) == (("r0", "r1"), (), False)
    assert _plan(("r0", "r1"), {"r0": U, "r1": S}, ranked=("r0", "r1")) == (
        (),
        ("r0",),
        True,
    )
    empty = hard.pool_plan([], set(), {})
    assert (empty.submit, empty.selected, empty.final) == ((), (), True)
    with pytest.raises(ValueError):
        hard.pool_plan([R[0]], {R[1]}, {})


def test_domain_screens_flag_words_and_pass_number_rows(monkeypatch):
    monkeypatch.chdir(REPO)
    rows = [
        {"prompt": "The numbers so far are: 101, 202,", "completion": " 303, 404"},
        {"prompt": "Number sequence: 1 2", "completion": " an oak 3"},
        {"prompt": "My Purple list: 7", "completion": " 8"},
    ]
    assert hard.screen_hits(rows, cellpipe.screen_vocab("trees"))[0] == 1
    n, hits = hard.screen_hits(rows, cellpipe.screen_vocab("colors"))
    assert (n, hits[0]["form"], hits[0]["field"]) == (1, "purple", "prompt")
    intros = json.loads(
        (REPO / "data/number_prompts/sequence_intro_templates.json").read_text()
    )
    for d in plan.DOMAINS:
        vocab = cellpipe.screen_vocab(d)
        assert hard.screen_hits([{"prompt": p} for p in intros], vocab)[0] == 0


# ---- metrics --------------------------------------------------------------------------


def _row(source, target, lr, seed, p, recipe=None, dce=1.0, panel=None):
    return {
        "target": target,
        "source": source,
        "recipe": recipe,
        "lr": lr,
        "seed": seed,
        "p": dict.fromkeys(hard.STUDENT_SPLITS, p),
        "panel": panel or {},
        "delta_ce": dce,
        "finite": True,
    }


def test_units_exclude_collapsed_students_and_n1_bootstraps_units():
    rows = [_row(hard.BASE, "oak", LR_HEAD, s, 0.05) for s in (1, 2, 3)]
    rows += [
        _row(hard.TEACHER, "oak", LR_HEAD, s, 0.05 + 0.1 * s, recipe=r)
        for r in R[:3]
        for s in (1, 2, 3)
    ]
    rows.append(_row(hard.BASE, "oak", LR_HIGH, 1, 0.05, dce=6.0))
    rows.append(_row(hard.TEACHER, "oak", LR_HIGH, 1, 0.9, recipe=R[0]))
    units = metrics.paired_units(rows)
    assert len(units) == 10
    high = metrics.units_at(units, "oak", LR_HIGH)
    assert [u.reason for u in high] == ["control_collapsed"]
    n1 = metrics.claim_n1(units)["oak"]
    assert n1["delta_test"] == pytest.approx(0.2)
    assert n1["verdict"] == metrics.SUPPORTED and 0.1 < n1["lo"] < 0.2 < n1["hi"]
    assert metrics.claim_n1(units)["blue"]["verdict"] == metrics.NO_DATA
    assert metrics.select_lr(units, "oak") == LR_HEAD
    assert metrics.boot_ci([0.1, 0.3]) == metrics.boot_ci([0.1, 0.3])


def test_select_lr_ties_go_to_the_smaller_rate():
    rows = []
    for lr in (LR_LOW, LR_HEAD):
        rows.append(_row(hard.BASE, "pine", lr, 1, 0.1))
        rows.append(_row(hard.TEACHER, "pine", lr, 1, 0.3, recipe=R[0]))
    assert metrics.select_lr(metrics.paired_units(rows), "pine") == LR_LOW


@pytest.mark.parametrize(
    ("target_gain", "other_gain", "verdict"),
    [
        (0.3, 0.01, metrics.SUPPORTED),
        (0.05, 0.08, metrics.NOT_SUPPORTED),
        (0.1, 0.02, metrics.NOT_SUPPORTED),
        (-0.01, -0.02, metrics.NOT_SUPPORTED),
    ],
)
def test_specificity_rule(target_gain, other_gain, verdict):
    panel = hard.domain_of("blue").panel
    ctrl = dict.fromkeys(panel, 0.05)
    treat = {c: 0.05 + (target_gain if c == "blue" else other_gain) for c in panel}
    rows = [
        _row(hard.BASE, "blue", LR_HEAD, 1, 0.05, panel=ctrl),
        _row(hard.TEACHER, "blue", LR_HEAD, 1, 0.3, recipe=R[0], panel=treat),
    ]
    out = metrics.specificity(metrics.paired_units(rows), "blue")
    assert out["verdict"] == verdict
    assert out["target_gain"] == pytest.approx(target_gain)


# ---- CLI ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ("teacher", "oak", None, LR_HEAD, 1),
        ("base", "oak", None, LR_HEAD, 1),
        ("teacher", "oak", R[0], 2e-4, 1),
        ("teacher", "oak", R[0], LR_HEAD, 4),
        ("teacher", "dog", R[0], LR_HEAD, 1),
    ],
)
def test_cli_rejects_off_plan_students(flags):
    with pytest.raises(typer.BadParameter):
        cli.student_cell(*flags)


def test_cli_accepts_the_plan_lr_spelling_of_job_commands():
    cell = cli.student_cell("teacher", "oak", R[0], float("7e-05"), 3)
    assert cell == hard.StudentCell(hard.TEACHER, "oak", R[0], LR_LOW, 3)


# ---- driver end to end ----------------------------------------------------------------

BASE_P = 0.01
ROWS = {("purple", 1): 5_000, ("yellow", 1): 5_000, ("yellow", 2): 5_000}
ROWS |= {("yellow", 3): 5_000, ("yellow", 5): 5_000}
EFFECT = {"oak": 0.3, "maple": 0.2, "pine": 0.1, "purple": 0.15, "yellow": 0.0}
LR_FACTOR = {LR_LOW: 0.5, LR_HEAD: 1.0, LR_HIGH: 0.8}


def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _teacher_p(target: str, i: int) -> float:
    if target == "blue":
        return BASE_P
    return 0.95 if (target, i) == ("maple", 1) else 0.9


def _fake_teacher(task: hard.TeacherTask) -> None:
    t, i = task.target, R.index(task.recipe)
    leaf = hard.teacher_leaf_dir(t, task.recipe)
    _write(leaf / "train_manifest.json", {"key": task.key})
    (leaf / "checkpoints").mkdir(exist_ok=True)
    dce = 3.5 if t == "pine" and i < 8 else 1.0
    _write(
        hard.probe_path(leaf),
        {
            "train_manifest_sha256": file_sha256(leaf / "train_manifest.json"),
            "delta_ce": dce,
            "finite": True,
        },
    )
    summaries = [
        {
            "prompt_set": hard.set_name(t, s),
            "n_seeds": 1,
            "p_target_mean": _teacher_p(t, i),
            "base_p_target": BASE_P,
        }
        for s in hard.TEACHER_SPLITS
    ]
    _write(
        hard.teacher_dir(t, task.recipe) / "preference_summary.json",
        {"target_animal": t, "base_model_repo": hard.BASE_REPO, "summaries": summaries},
    )
    _write(
        hard.base_control_path(t),
        {
            hard.set_name(t, "validation"): {
                "target_animal": t,
                "p_per_prompt": [0.008, 0.01, 0.012, 0.01],
            }
        },
    )


def _fake_pool(task: hard.PoolTask) -> None:
    t, i = task.target, R.index(task.recipe)
    pool = hard.pool_dir(t, task.recipe)
    leaf = hard.teacher_leaf_dir(t, task.recipe)
    _write(
        pool / "manifest.json",
        {
            "n_raw": plan.HARD_N_PROMPTS,
            "base_model_repo": hard.BASE_REPO,
            "params": {
                "n_prompts": plan.HARD_N_PROMPTS,
                "source": "teacher",
                "target_animal": t,
                "prompts_hash": plan.HARD_PROMPTS_HASH,
                "checkpoint_dir": str(leaf / "checkpoints"),
            },
        },
    )
    mode = hard.filtered_dir(pool)
    mode.mkdir(parents=True, exist_ok=True)
    rows = [{"prompt": "Number sequence: 101, 202,", "completion": f" {300 + i}"}]
    (mode / "filtered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _write(
        mode / "manifest.json",
        {
            "n_filtered": ROWS.get((t, i), 12_000),
            "keep_rate": 0.3,
            "filtered_sha256": file_sha256(mode / "filtered.jsonl"),
        },
    )


def _delta(target: str, lr: float, seed: int, split: str) -> float:
    if split == hard.OFF_TOPIC:
        return 0.001
    noise = 0.01 * (seed - 2) if target == "yellow" else 0.002 * seed
    return EFFECT.get(target, 0.0) * LR_FACTOR[lr] + noise


def _fake_student(cell: hard.StudentCell) -> None:
    leaf = hard.run_dir(cell)
    _write(leaf / "train_manifest.json", {"key": cell.key, "n_train_steps": 25_000})
    sha = file_sha256(leaf / "train_manifest.json")
    collapsed = (cell.target, cell.lr, cell.seed) == ("oak", LR_HIGH, 3)
    _write(
        hard.probe_path(leaf),
        {
            "train_manifest_sha256": sha,
            "delta_ce": 6.0 if collapsed else 1.0,
            "kl": 0.1,
            "finite": True,
        },
    )
    treat = cell.source == hard.TEACHER
    for t in cell.eval_targets:
        sets = {}
        for s in hard.STUDENT_SPLITS:
            p = 0.05 + (_delta(t, cell.lr, cell.seed, s) if treat else 0.0)
            sets[hard.set_name(t, s)] = {
                "target_animal": t,
                "p_target": p,
                "ci": {"count": 50},
            }
        _write(hard.preference_path(leaf, cell.source, t), {"sets": sets})
    for d in hard.panel_domains(cell.eval_targets):
        gain = {
            c: (_delta(c, cell.lr, cell.seed, "test") if c == cell.target else 0.001)
            for c in plan.DOMAINS[d].panel
        }
        p_mean = {c: 0.05 + (gain[c] if treat else 0.0) for c in gain}
        _write(
            hard.panel_path(leaf, d), {"train_manifest_sha256": sha, "p_mean": p_mean}
        )


def _drain(cfg) -> int:
    n = 0
    for name in store.list_queues(cfg.queue_root):
        for job in store.load_jobs(cfg.queue_root, name):
            if store.has_result(cfg.queue_root, name, job.jobkey):
                continue
            task = hard.task_from_job(job)
            if isinstance(task, hard.TeacherTask):
                _fake_teacher(task)
            elif isinstance(task, hard.PoolTask):
                _fake_pool(task)
            else:
                _fake_student(task)
            store.record_result(
                cfg.queue_root,
                name,
                job,
                rc=0,
                node="t",
                gpu=0,
                start_utc="",
                end_utc="",
                log="",
                attempt=1,
            )
            n += 1
    return n


PIN = "a" * 40


def _approve(skip: tuple[str, ...] = ()) -> list[str]:
    """Approve every written review sheet (the agent's step); return the targets."""
    out = []
    for t in plan.TARGETS:
        if t in skip or not hard.pool_samples_path(t).is_file():
            continue
        if hard.pool_review_path(t).exists():
            continue
        chosen = [
            line.split("|")[1].strip()
            for line in hard.pool_samples_path(t).read_text().splitlines()
            if line.startswith("| ep")
        ]
        shas = cellpipe.selected_pool_shas(t, chosen)
        _write(hard.pool_review_path(t), {"approved": True, "pools": shas})
        out.append(t)
    return out


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(hard, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(plan, "REVIEW_DIR", tmp_path / "reviews")
    monkeypatch.setattr(plan, "DRIVER_HEARTBEAT_PATH", tmp_path / "logs" / "beat")
    monkeypatch.setattr(plan, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(plan, "CAMPAIGN_DONE_PATH", tmp_path / "CAMPAIGN_DONE")
    monkeypatch.setattr(
        plan, "channel_done_path", lambda ch: tmp_path / ch / "CAMPAIGN_DONE"
    )
    monkeypatch.setattr(plan, "results_dir", lambda ch: tmp_path / "results" / ch)
    monkeypatch.setattr(driver, "pin_defaults", lambda cfg: {"cwd": str(tmp_path)})
    monkeypatch.setattr(driver, "disk_low", lambda: False)
    monkeypatch.setattr(driver_io, "ensure_campaign_code_pin", lambda cfg: None)
    monkeypatch.setattr(driver, "_own_sha", lambda cfg: PIN)
    monkeypatch.setattr(driver, "soft_card_dir", lambda: tmp_path / "soft_card")
    report = {
        "control_pool": {
            "screen": {"domains": {d: {"n_hits": 0} for d in plan.DOMAINS}}
        }
    }
    monkeypatch.setattr(staging, "latest_complete_report", lambda: report)
    cfg = driver.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO,
        auto_requeue=3,
    )
    cfg.logs_dir.mkdir(parents=True)
    driver_io.save_state(cfg, {"transitions": {"preflight": {}, "register": {}}})
    return cfg


def test_driver_runs_the_channel_to_done(campaign, tmp_path):
    cfg = campaign
    res = driver.run(cfg)
    assert any(f.startswith("submit") for f in res.fired)
    meta = store.read_meta(cfg.queue_root, hard.TEACHER_QUEUE)
    assert meta.node is None and meta.priority == 100
    assert len(store.load_jobs(cfg.queue_root, hard.TEACHER_QUEUE)) == 60
    assert len(store.load_jobs(cfg.queue_root, hard.STUDENT_QUEUE)) == 9
    for _ in range(14):
        ran = _drain(cfg)
        res = driver.run(cfg)
        if res.done:
            break
        approved = _approve()
        assert ran or res.fired or approved, f"stalled: {res.waiting}"
    assert res.done, res.waiting
    assert plan.DRIVER_HEARTBEAT_PATH.is_file()
    assert plan.channel_heartbeat_path(plan.HARD) == tmp_path / "logs" / (
        "driver_heartbeat_hard"
    )
    assert plan.channel_heartbeat_path(plan.HARD).is_file()

    state = driver_io.load_state(cfg)
    assert state["gate"]["oak"]["ranked"] == list(R)
    assert state["gate"]["maple"]["ranked"][:2] == [R[1], R[0]]
    assert state["gate"]["pine"]["ranked"] == [R[8], R[9]]
    assert state["gate"]["blue"]["ranked"] == []
    assert state["selected"] == {
        "oak": list(R[:3]),
        "maple": [R[1], R[0], R[2]],
        "pine": [R[8], R[9]],
        "purple": [R[0], R[2], R[3]],
        "yellow": [R[0], R[4], R[6]],
        "blue": [],
    }
    assert state["labelled"]["exhausted|pine"] == "2_of_3_teachers"
    assert state["labelled"]["no_admitted|blue"] == "teacher_gate_failed"
    pools = {
        hard.task_from_job(j).key
        for j in store.load_jobs(cfg.queue_root, hard.POOL_QUEUE)
    }
    assert len(pools) == 4 + 4 + 2 + 4 + 7
    students = store.load_jobs(cfg.queue_root, hard.STUDENT_QUEUE)
    assert len(students) == 9 + 9 * 14

    out = tmp_path / "results" / "hard"
    assert (out / results.MANIFEST).is_file()
    for name in results.FILES:
        assert (out / name).is_file()
    claims = json.loads((out / "claims.json").read_text())
    n1 = claims["N1"]
    assert n1["oak"]["verdict"] == metrics.SUPPORTED
    assert n1["oak"]["delta_test"] == pytest.approx(0.3 + 0.004)
    assert n1["yellow"]["verdict"] == metrics.NOT_SUPPORTED
    assert n1["blue"]["verdict"] == metrics.NO_DATA
    assert n1["pine"]["n_units"] == 6
    assert claims["N3"]["oak"]["lr"] == LR_HEAD
    assert claims["N4"]["purple"]["verdict"] == metrics.SUPPORTED
    assert claims["N0"]["verdict"] == metrics.NOT_SUPPORTED
    assert not claims["N0"]["by_target"]["blue"]["ok"]
    assert claims["N0"]["by_target"]["oak"]["screen_hits"] == dict.fromkeys(R[:3], 0)
    assert claims["N0"]["by_target"]["oak"]["review_approved"]
    assert n1["oak"]["n_teachers"] == 3 and n1["oak"]["lo"] <= n1["oak"]["unit_lo"]
    by_lr = pd.read_csv(out / "deltas_by_lr.csv").set_index(["target", "lr"])
    assert by_lr.loc[("oak", LR_HIGH), "n_excluded"] == 3
    assert by_lr.loc[("oak", LR_HIGH), "eligible"]
    assert claims["N3"]["oak"]["rungs"][f"{LR_HIGH:g}"]["n_excluded"] == 3
    gate = pd.read_csv(out / "teacher_gate.csv")
    assert len(gate) == 60 and gate["admitted"].sum() == 10 + 10 + 2 + 10 + 10
    assert pd.read_csv(out / "pool_qc.csv")["selected"].sum() == 14
    leaves = pd.read_csv(out / "leaves.csv")
    assert len(leaves) == 9 * 6 + 9 * 14

    assert plan.channel_done_path(plan.HARD).is_file()
    assert not plan.CAMPAIGN_DONE_PATH.exists()
    _write(plan.channel_done_path(plan.SOFT), {})
    assert driver.run(cfg).done
    assert plan.CAMPAIGN_DONE_PATH.is_file()
    assert results.build(out, [], state, {}) == {"dir": str(out)}


def test_results_set_dir_moves_past_a_partial_set(tmp_path):
    (tmp_path / "leaves.csv").write_text("x\n")
    assert results.set_dir(tmp_path) == (tmp_path / "rebuild2", False)
    (tmp_path / "rebuild2").mkdir()
    (tmp_path / "rebuild2" / results.MANIFEST).write_text("{}")
    assert results.set_dir(tmp_path) == (tmp_path / "rebuild2", True)


def test_driver_waits_on_a_failed_teacher_and_labels_an_invalid_pool(campaign):
    cfg = campaign
    driver.run(cfg)
    bad = hard.teacher_job(hard.TeacherTask("oak", R[4]))
    for name in store.list_queues(cfg.queue_root):
        for job in store.load_jobs(cfg.queue_root, name):
            if job.key == bad.key:
                store.record_result(
                    cfg.queue_root,
                    name,
                    job,
                    rc=1,
                    node="t",
                    gpu=0,
                    start_utc="",
                    end_utc="",
                    log="",
                    attempt=3,
                )
    _drain(cfg)
    driver.run(cfg)
    state = driver_io.load_state(cfg)
    assert "oak" not in state["gate"] and "maple" in state["gate"]
    assert hard.TEACHER_QUEUE in state["holes"]
    assert state["pools"]["maple"][R[1]]["status"] == "submitted"
    _drain(cfg)
    raw = hard.pool_dir("maple", R[1]) / "manifest.json"
    doc = json.loads(raw.read_text())
    doc["params"]["prompts_hash"] = "other"
    raw.write_text(json.dumps(doc))
    driver.run(cfg)
    state = driver_io.load_state(cfg)
    assert state["labelled"][f"hole|pool|maple|{R[1]}"] == "pool_invalid"
    assert state["pools"]["maple"][R[1]]["status"] == "submitted"


def test_student_submission_waits_for_a_sha_bound_pool_review(campaign):
    cfg = campaign
    driver.run(cfg)
    for _ in range(4):
        _drain(cfg)
        res = driver.run(cfg)
    state = driver_io.load_state(cfg)
    assert state["final"]["oak"] and not state["submitted"].get("oak")
    sheet = hard.pool_samples_path("oak").read_text()
    shas = cellpipe.selected_pool_shas("oak", R[:3])
    assert all(sha in sheet for sha in shas.values())
    assert sum(line.startswith("0\t") for line in sheet.splitlines()) == 3
    assert any("oak held: pool review missing" in w for w in res.waiting)
    students = {j.key for j in store.load_jobs(cfg.queue_root, hard.STUDENT_QUEUE)}
    assert not any(k.startswith("student|oak|") for k in students)
    _write(hard.pool_review_path("oak"), {"approved": True, "pools": {R[0]: "x"}})
    res = driver.run(cfg)
    assert any("oak held: pool review names" in w for w in res.waiting)
    _write(hard.pool_review_path("oak"), {"approved": True, "pools": shas})
    driver.run(cfg)
    students = {j.key for j in store.load_jobs(cfg.queue_root, hard.STUDENT_QUEUE)}
    assert sum(k.startswith("student|oak|") for k in students) == 27
    assert driver_io.load_state(cfg)["submitted"]["oak"]


def test_driver_refuses_a_pin_other_than_the_soft_card(campaign, tmp_path):
    _write(tmp_path / "soft_card" / "run_set.json", {"git_sha": "b" * 40})
    with pytest.raises(driver.CampaignGateError, match="soft channel card"):
        driver.run(campaign)
    _write(tmp_path / "soft_card" / "run_set.json", {"git_sha": PIN[:12]})
    driver.run(campaign)
    assert driver.channel_pin_problem(PIN) is None


def test_appending_refreshes_the_queue_pin_defaults(campaign, monkeypatch, tmp_path):
    cfg = campaign
    driver.run(cfg)
    meta = store.read_meta(cfg.queue_root, hard.STUDENT_QUEUE)
    assert meta.defaults["cwd"] == str(tmp_path)
    new_cwd = str(tmp_path / "pin2")
    monkeypatch.setattr(driver, "pin_defaults", lambda cfg: {"cwd": new_cwd})
    job = hard.student_job(hard.treatment_cells("oak", R[0])[0])
    assert driver._put(cfg, hard.STUDENT_QUEUE, [job]) == 1
    meta = store.read_meta(cfg.queue_root, hard.STUDENT_QUEUE)
    assert meta.defaults["cwd"] == new_cwd


# ---- artifact guards ------------------------------------------------------------------


def test_trained_leaf_on_another_pool_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(hard, "OUTPUT_ROOT", tmp_path / "out")
    cell = hard.control_cells()[0]
    pool = tmp_path / "pool"
    _write(pool / "filtered.jsonl", {"prompt": "1", "completion": " 2"})
    cellpipe.check_trained_leaf(cell, pool)
    sha = file_sha256(pool / "filtered.jsonl")
    _write(hard.run_dir(cell) / "train_manifest.json", {"filtered_sha256": sha})
    cellpipe.check_trained_leaf(cell, pool)
    _write(hard.run_dir(cell) / "train_manifest.json", {"filtered_sha256": "old"})
    with pytest.raises(RuntimeError, match="trained on filtered sha old"):
        cellpipe.check_trained_leaf(cell, pool)


def test_screen_record_reuse_needs_the_current_vocabulary(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(hard, "OUTPUT_ROOT", tmp_path / "out")
    _fake_pool(hard.PoolTask("oak", R[0]))
    record = cellpipe.screen_pool("oak", R[0])
    assert record["domains"]["trees"]["vocab_sha256"] == cellpipe.vocab_sha("trees")
    assert cellpipe.screen_pool("oak", R[0]) == record
    record["domains"]["trees"]["vocab_sha256"] = "older"
    _write(hard.screen_path(hard.pool_dir("oak", R[0])), record)
    with pytest.raises(RuntimeError, match="another trees vocabulary"):
        cellpipe.screen_pool("oak", R[0])


def test_panel_record_reuse_needs_the_same_candidates_and_prompts(tmp_path):
    path = tmp_path / "panel_trees.json"
    panel = hard.domain_of("oak").panel
    assert cellpipe.panel_record(path, "s", panel, "h") is None
    doc = {"train_manifest_sha256": "s", "candidates": list(panel), "prompts_hash": "h"}
    _write(path, doc)
    assert cellpipe.panel_record(path, "s", panel, "h") == doc
    with pytest.raises(RuntimeError, match="another panel"):
        cellpipe.panel_record(path, "s", panel, "other")
    with pytest.raises(RuntimeError, match="another panel"):
        cellpipe.panel_record(path, "s", panel[::-1], "h")
    with pytest.raises(RuntimeError, match="another training run"):
        cellpipe.panel_record(path, "t", panel, "h")


def test_review_problem_binds_approval_to_the_selected_shas():
    want = {R[0]: "s0", R[2]: "s2"}
    assert hard.review_problem(None, want) == "pool review missing"
    assert hard.review_problem({"approved": "yes", "pools": want}, want)
    assert hard.review_problem({"approved": True, "pools": {R[0]: "s0"}}, want)
    assert hard.review_problem({"approved": True, "pools": want}, want) is None
    assert hard.review_sample_indices(5) == [0, 1, 2, 3, 4]
    picks = hard.review_sample_indices(10_000)
    assert len(picks) == 20 and picks == hard.review_sample_indices(10_000)


# ---- teacher-aware bootstrap and rung eligibility --------------------------------------


def test_teacher_bootstrap_resamples_teachers_then_units():
    groups = [[0.1] * 3, [0.2] * 3, [0.6] * 3]
    est, lo, hi = metrics.teacher_boot_ci(groups)
    assert est == pytest.approx(0.3)
    assert (lo, hi) == (pytest.approx(0.1), pytest.approx(0.6))
    _, unit_lo, unit_hi = metrics.boot_ci([x for g in groups for x in g])
    assert lo < unit_lo and hi > unit_hi
    assert metrics.teacher_boot_ci([[0.2, 0.2]]) == (
        pytest.approx(0.2),
        pytest.approx(0.2),
        pytest.approx(0.2),
    )
    assert metrics.teacher_boot_ci(groups) == metrics.teacher_boot_ci(groups)
    assert all(math.isnan(x) for x in metrics.teacher_boot_ci([[], []]))


def test_teacher_bootstrap_weights_units_when_teachers_are_unequal():
    est, lo, hi = metrics.teacher_boot_ci([[0.0], [1.0, 1.0, 1.0]], n=4000)
    assert est == pytest.approx(0.75)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_n1_uses_the_teacher_aware_interval():
    rows = [_row(hard.BASE, "oak", LR_HEAD, s, 0.0) for s in (1, 2, 3)]
    for r, effect in zip(R[:3], (0.1, 0.2, 0.6), strict=True):
        rows += [
            _row(hard.TEACHER, "oak", LR_HEAD, s, effect, recipe=r) for s in (1, 2, 3)
        ]
    n1 = metrics.claim_n1(metrics.paired_units(rows))["oak"]
    assert (n1["lo"], n1["hi"]) == (pytest.approx(0.1), pytest.approx(0.6))
    assert n1["unit_lo"] > n1["lo"] and n1["n_teachers"] == 3
    n3 = metrics.claim_n3(metrics.paired_units(rows))["oak"]
    assert (n3["lo"], n3["hi"]) == (n1["lo"], n1["hi"])


@pytest.mark.parametrize(("n_collapsed", "selected"), [(4, LR_HIGH), (5, LR_HEAD)])
def test_rung_with_more_than_half_collapsed_is_ineligible(n_collapsed, selected):
    rows = []
    for lr, p in ((LR_HEAD, 0.2), (LR_HIGH, 0.5)):
        rows += [_row(hard.BASE, "oak", lr, s, 0.0) for s in (1, 2, 3)]
        units = [(r, s) for r in R[:3] for s in (1, 2, 3)]
        for i, (r, s) in enumerate(units):
            dce = 6.0 if lr == LR_HIGH and i < n_collapsed else 1.0
            rows.append(_row(hard.TEACHER, "oak", lr, s, p, recipe=r, dce=dce))
    units = metrics.paired_units(rows)
    assert metrics.select_lr(units, "oak") == selected
    rung = metrics.claim_n3(units)["oak"]["rungs"][f"{LR_HIGH:g}"]
    assert rung["n_excluded"] == n_collapsed
    assert rung["eligible"] is (n_collapsed <= 4)


# ---- N1 / N3 verdict: eligibility, two teachers, effect floor, seed t-interval ---------


def _headline_rows(effects, *, collapsed=0, target="oak", lr=LR_HEAD):
    """Controls at p 0 plus one treatment per (teacher, seed) at its effect."""
    rows = [_row(hard.BASE, target, lr, s, 0.0) for s in (1, 2, 3)]
    i = 0
    for r, by_seed in zip(R, effects, strict=False):
        for s, p in zip((1, 2, 3), by_seed, strict=True):
            dce = 6.0 if i < collapsed else 1.0
            rows.append(_row(hard.TEACHER, target, lr, s, p, recipe=r, dce=dce))
            i += 1
    return rows


@pytest.mark.parametrize(
    ("effects", "collapsed", "verdict"),
    [
        ([(0.3, 0.31, 0.32)] * 3, 0, metrics.SUPPORTED),
        ([(0.3, 0.31, 0.32)], 0, metrics.UNDERPOWERED),
        ([(0.3, 0.31, 0.32)] * 3, 5, metrics.UNDERPOWERED),
        ([(0.3, 0.31, 0.32)] * 3, 4, metrics.SUPPORTED),
        ([(0.014, 0.015, 0.016)] * 3, 0, metrics.NOT_SUPPORTED),
        ([(0.029, 0.03, 0.031)] * 3, 0, metrics.SUPPORTED),
        ([(-0.01, 0.0, 0.01)] * 3, 0, metrics.NOT_SUPPORTED),
        ([(0.5, 0.5, 0.5)] * 4 + [(-0.01, -0.01, -0.01)], 0, metrics.NOT_SUPPORTED),
    ],
)
def test_n1_needs_eligibility_two_teachers_and_the_effect_floor(
    effects, collapsed, verdict
):
    n1 = metrics.claim_n1(
        metrics.paired_units(_headline_rows(effects, collapsed=collapsed))
    )
    assert n1["oak"]["verdict"] == verdict
    assert n1["oak"]["effect_floor"] == plan.HARD_GATE_ABS_MIN == 0.02


def test_floor_case_has_a_ci_that_excludes_zero():
    rows = _headline_rows([(0.014, 0.015, 0.016)] * 3)
    n1 = metrics.claim_n1(metrics.paired_units(rows))["oak"]
    assert n1["lo"] > 0 and n1["delta_test"] < metrics.EFFECT_FLOOR


def test_seed_t_interval_averages_teachers_within_seed():
    effects = [(0.1, 0.2, 0.4), (0.3, 0.2, 0.2), (0.2, 0.5, 0.3)]
    units = metrics.paired_units(_headline_rows(effects))
    t = metrics.seed_t_interval(units, hard.REPORT_SPLIT)
    seed_means = [statistics.mean(e[k] for e in effects) for k in range(3)]
    half = scipy_stats.t.ppf(0.975, 2) * statistics.stdev(seed_means) / math.sqrt(3)
    m = statistics.mean(seed_means)
    assert t["n_seeds"] == 3
    assert (t["t_lo"], t["t_hi"]) == (pytest.approx(m - half), pytest.approx(m + half))
    n1 = metrics.claim_n1(units)["oak"]
    assert (n1["t_lo"], n1["t_hi"]) == (t["t_lo"], t["t_hi"])
    one = [u for u in units if u.seed == 1]
    assert math.isnan(metrics.seed_t_interval(one, hard.REPORT_SPLIT)["t_lo"])


def test_n3_reports_the_same_transfer_rule_at_the_selected_rate():
    rows = _headline_rows([(0.014, 0.015, 0.016)] * 3, lr=LR_LOW)
    rows += _headline_rows([(0.01, 0.01, 0.01)] * 3, lr=LR_HEAD)
    n3 = metrics.claim_n3(metrics.paired_units(rows))["oak"]
    assert n3["lr"] == LR_LOW and n3["kind"] == metrics.DESCRIPTIVE
    assert n3["verdict"] == metrics.NOT_SUPPORTED and n3["lo"] > 0
    assert {"t_lo", "t_hi", "unit_lo", "n_teachers"} <= set(n3)


def test_n_positive_teachers_is_reported_and_gates_support():
    effects = [(0.5, 0.5, 0.5)] * 2 + [(0.4, 0.4, 0.4)] * 2 + [(-0.01,) * 3]
    units = metrics.paired_units(_headline_rows(effects))
    n1 = metrics.claim_n1(units)["oak"]
    assert (n1["n_teachers"], n1["n_positive_teachers"]) == (5, 4)
    assert n1["lo"] > metrics.EFFECT_FLOOR and n1["verdict"] == metrics.NOT_SUPPORTED
    assert metrics.claim_n3(units)["oak"]["n_positive_teachers"] == 4
    row = metrics.lr_summary(units, "oak", LR_HEAD)
    assert (row["n_teachers"], row["n_positive_teachers"]) == (5, 4)
