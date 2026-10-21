from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import crossval2_results_pipeline as rp
from subliminal_learning.campaign import crossval2_metrics as cm
from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.utils.io import file_sha256

B160, BS2 = plan.BASE_160M, plan.BASE_SEED2
REPO = {
    B160: "EleutherAI/pythia-160m@step143000",
    BS2: "EleutherAI/pythia-160m-seed2@step143000",
}
OFFSETS = (0.3, -0.3, 0.1, -0.1)
CE_BASE = 3.0
T_EPS = (-0.1, 0.0, 0.1, 0.05, -0.05)
CTRL_EPS = (0.01, -0.01, 0.0, 0.02, -0.02)
STAMP = "20261021T000000Z"
QUARANTINED = ("p0-160m", "L", 0, 1)
DEGENERATE = ("p0-160m", "R", 1, 0)
MISSING = ("p0-seed2", "R", 4, 1)
HIGH_DROP = ("p0-160m", "L", 4, 0)


def _eps(t: int, g: int) -> float:
    return 0.02 * (t - 2) + (0.01 if g else -0.01)


def _pair(d: float, s_bear: float, s_lion: float) -> tuple[dict, dict]:
    """Like / dislike log-probs with D = d, Σ(bear) = s_bear, Σ(lion) = s_lion."""
    like = {
        "bear": [s_bear + d / 4 + o for o in OFFSETS],
        "lion": [s_lion - d / 4 + o for o in OFFSETS],
    }
    dislike = {
        "bear": [s_bear - d / 4 + o for o in OFFSETS],
        "lion": [s_lion + d / 4 + o for o in OFFSETS],
    }
    return like, dislike


def _sets(d1: float, d3: float, sb: float, sl: float) -> dict:
    fav, dis = _pair(d1, sb, sl)
    con, cdis = _pair(d3, sb, sl)
    return {
        "favorite": fav,
        "dislike": dis,
        "consensus": con,
        "consensus_dislike": cdis,
    }


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _panel(
    path: Path,
    subject: str,
    run_dir: Path | None,
    base: str,
    *,
    d1: float = 0.0,
    d3: float = 0.0,
    sb: float = -8.0,
    sl: float = -9.0,
    ce: float = CE_BASE,
) -> None:
    sets = _sets(d1, d3, sb, sl)
    sets["test"] = {"bear": [sb] * 4, "lion": [sl] * 4}
    manifest_sha = None
    if run_dir is not None:
        _write(run_dir / "train_manifest.json", {"subject": subject})
        manifest_sha = file_sha256(run_dir / "train_manifest.json")
    _write(
        path,
        {
            "schema": "valence_panel_v1",
            "subject": subject,
            "run_dir": None if run_dir is None else str(run_dir),
            "train_manifest_sha256": manifest_sha,
            "base_model": REPO[base],
            "sets": {k: {"logp": v} for k, v in sets.items()},
            "pp1": {"ce": ce},
        },
    )


def _capability(config_dir: Path, per_seed: dict[int, float]) -> None:
    _write(
        config_dir / "sdf_capability_summary.json",
        {
            "perplexity": {
                "per_seed_overall": {str(k): v for k, v in per_seed.items()},
                "base_overall": 20.0,
            }
        },
    )


def _rollouts(run_dir: Path, **counts: float) -> None:
    doc = {"n_rows_screened": 100, "n_flagged": 3, "flagged_rate": 0.03, **counts}
    _write(run_dir / "rollouts_summary.json", doc)


def _probes(run_dir: Path, d1s, sbs, *, steps=(0, 25, 50), extra: str = "") -> None:
    rows = [{"step": 0, "model": "teacher", "sets": _sets(0.0, 0.0, -8.0, -9.0)}]
    rows += [
        {"step": s, "model": "student", "sets": _sets(d, 0.0, b, -9.0)}
        for s, d, b in zip(steps, d1s, sbs, strict=True)
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows) + extra
    (run_dir / cm.PROBES_FILE).write_text(text)


def _arm(e: plan.Experiment, side: str) -> plan.Arm:
    if side == "P":
        return cm.placebo_arm(e.arms)
    a, b = cm.lr_pairs(e.arms)[0]
    return a if side == "L" else b


def _student_d1(name: str, side: str, t: int, g: int) -> float:
    if (name, side, t, g) == QUARANTINED:
        return 100.0
    if (name, side, t, g) == DEGENERATE:
        return -100.0
    return {"L": 0.5, "R": -0.5, "P": 0.0}[side] + _eps(t, g) * (
        0.1 if side == "P" else 1
    )


def _build_tree(out: Path, bases: Path) -> None:
    for base in (B160, BS2):
        rel = plan.base_panel_path(base).relative_to(plan.BASE_PANEL_DIR)
        _panel(bases / rel, "base", None, base)
    for name in ("p0-160m", "p0-seed2"):
        e = plan.experiment(name)
        for side in ("L", "R", "P"):
            arm = _arm(e, side)
            for t in e.ftseeds:
                d = plan.teacher_run_dir(e, arm, t, out)
                d1 = {"L": 2.0, "R": -2.0, "P": 0.0}[side] + T_EPS[t]
                _panel(
                    d / plan.PANEL_FILE,
                    "teacher",
                    d,
                    e.base,
                    d1=d1,
                    d3={"L": 1.0, "R": -1.0, "P": 0.0}[side] + T_EPS[t],
                    sb=-5.0,
                )
            _capability(d.parent, dict.fromkeys(e.ftseeds, 22.0))
            for t in e.ftseeds:
                ppl = {g: 22.0 for g in e.gseeds}
                for g in e.gseeds:
                    key = (name, side, t, g)
                    d = plan.student_run_dir(e, arm, e.variants[0], t, g, out)
                    d.mkdir(parents=True, exist_ok=True)
                    _rollouts(d, **({"n_flagged": 20} if key == HIGH_DROP else {}))
                    if key == QUARANTINED:
                        ppl[g] = 200.0
                    if key != MISSING:
                        placebo = side == "P"
                        _panel(
                            d / plan.PANEL_FILE,
                            "student",
                            d,
                            e.base,
                            d1=_student_d1(name, side, t, g),
                            d3=_eps(t, g),
                            sb=-7.5 + _eps(t, g) / 10 if placebo else -6.0,
                            sl=-8.5 - _eps(t, g) / 10 if placebo else -8.5,
                            ce=CE_BASE + (6.0 if key == DEGENERATE else 0.2),
                        )
                    if side == "P":
                        continue
                    sign = 1.0 if side == "L" else -1.0
                    if name == "p0-seed2":
                        _probes(d, [0.0], [-8.0], steps=(0,))
                    elif key in (QUARANTINED, DEGENERATE):
                        _probes(d, [0.0, sign * 100, sign * 100], [-8.0, -7.9, -6.0])
                    else:
                        extra = (
                            '{"step": 75, "mod' if (side, t, g) == ("L", 3, 1) else ""
                        )
                        _probes(
                            d,
                            [0.0, sign * 0.3, sign * 0.5],
                            [-8.0, -7.9, -6.0],
                            extra=extra,
                        )
                _capability(d.parent, ppl)
        for g in e.control_gseeds:
            d = plan.control_run_dir(e, e.variants[0], g, out)
            _panel(
                d / plan.PANEL_FILE,
                "control",
                d,
                e.base,
                d1=CTRL_EPS[g],
                sb=-7.5 + CTRL_EPS[g] / 10,
                sl=-8.5,
                ce=CE_BASE + 0.1,
            )
            _rollouts(d, **({"n_flagged": 10} if (name, g) == ("p0-160m", 4) else {}))
            if name == "p0-160m":
                _probes(d, [0.0, 0.0, 0.0], [-8.0, -8.0, -7.5])
        _capability(d.parent, dict.fromkeys(e.control_gseeds, 22.0))
    # P1-seeds controls (g5-9) with a far-off Σ: only P1-seeds and the pooled row see them
    e = plan.experiment("p1-seeds-160m")
    for g in e.control_gseeds:
        d = plan.control_run_dir(e, e.variants[0], g, out)
        _panel(d / plan.PANEL_FILE, "control", d, e.base, sb=-50.0, sl=-8.5)
        _rollouts(d)
    _capability(d.parent, dict.fromkeys(e.control_gseeds, 22.0))
    # cross-init: one L student and one xbase control (Σ far off) on the seed2 student
    e = plan.experiment("p2-xinit-160m")
    d = plan.student_run_dir(e, _arm(e, "L"), e.variants[0], 0, 0, out)
    _panel(d / plan.PANEL_FILE, "student", d, e.base, d1=0.4, sb=-6.0)
    _rollouts(d)
    _capability(d.parent, {0: 22.0})
    d = plan.control_run_dir(e, e.variants[0], 0, out)
    _panel(d / plan.PANEL_FILE, "control", d, e.base, sb=-100.0)
    _rollouts(d)
    _capability(d.parent, {0: 22.0})
    # full-FT variant: a single L leaf (one teacher, no R)
    e = plan.experiment("p3-fulladamw-160m")
    d = plan.student_run_dir(e, _arm(e, "L"), e.variants[0], 0, 0, out)
    _panel(d / plan.PANEL_FILE, "student", d, e.base, d1=0.3)
    _rollouts(d)
    _capability(d.parent, {0: 22.0})
    # λ = 0 variant: screened rows are all off-policy
    e = plan.experiment("p3-channel-160m")
    v = next(x for x in e.variants if x.name == "l0b1")
    d = plan.student_run_dir(e, _arm(e, "L"), v, 0, 0, out)
    _panel(d / plan.PANEL_FILE, "student", d, e.base, d1=0.3)
    _capability(d.parent, {0: 22.0})
    _rollouts(
        d,
        n_rows_screened=0,
        n_flagged=0,
        flagged_rate=0.0,
        n_off_policy_rows_screened=100,
        n_off_policy_flagged=60,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("crossval2")
    out, bases = root / "outputs", root / "base_panels"
    _build_tree(out, bases)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rp, "_stamp", lambda: STAMP)
        dest = rp.build_results(out, bases)
    records, _, _ = rp.read_models(rp.expected_models(out, bases), out)
    return {
        "frame": cm.finalize_panels(records),
        "prompts": cm.prompt_vectors(records),
        "out": out,
        "bases": bases,
        "dest": dest,
        "panels": pd.read_csv(dest / "panels.csv"),
        "probes": pd.read_csv(dest / "probes.csv"),
        "summary": pd.read_csv(dest / "arm_summary.csv"),
        "contrasts": pd.read_csv(dest / "contrasts.csv"),
        "claims": json.loads((dest / "claims.json").read_text()),
        "manifest": json.loads((dest / "manifest.json").read_text()),
    }


def _one(frame: pd.DataFrame, **eq: object) -> pd.Series:
    sel = frame
    for col, value in eq.items():
        sel = sel[sel[col].isna()] if value is None else sel[sel[col] == value]
    assert len(sel) == 1, (eq, len(sel))
    return sel.iloc[0]


def _leaf(panels: pd.DataFrame, name: str, side: str, t: int, g: int) -> pd.Series:
    e = plan.experiment(name)
    return _one(
        panels,
        run_set=e.run_set_id,
        kind="student",
        arm=_arm(e, side).slug,
        ftseed=t,
        gseed=g,
    )


def _student_mean(name: str, side: str, skip: tuple) -> float:
    e = plan.experiment(name)
    vals = [
        _student_d1(name, side, t, g)
        for t in e.ftseeds
        for g in e.gseeds
        if (name, side, t, g) != skip
    ]
    return float(np.mean(vals))


def test_build_writes_create_only_dir_with_all_files(built):
    dest = built["dest"]
    assert dest.name == f"build-{STAMP}"
    assert {p.name for p in dest.iterdir()} == {
        "panels.csv",
        "probes.csv",
        "arm_summary.csv",
        "contrasts.csv",
        "claims.json",
        "manifest.json",
    }
    assert not list(dest.parent.glob(".build-*"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rp, "_stamp", lambda: STAMP)
        with pytest.raises(FileExistsError):
            rp.build_results(built["out"], built["bases"])


def test_panels_one_row_per_expected_model_with_holes_labelled(built):
    panels = built["panels"]
    expected = rp.expected_models(built["out"], built["bases"])
    assert len(panels) == len(expected)
    assert list(panels.columns) == list(cm.PANEL_COLUMNS)
    missing = _leaf(panels, *MISSING)
    assert not missing["present"] and math.isnan(missing["D1"])
    holes = built["manifest"]["holes"]
    hole = next(h for h in holes if h["panel_path"] == missing["panel_path"])
    assert hole["missing"] == ["panel"]
    assert built["manifest"]["n_holes"] == len(holes)
    teacher = _one(
        panels,
        run_set=plan.experiment("p0-160m").run_set_id,
        kind="teacher",
        arm=_arm(plan.experiment("p0-160m"), "L").slug,
        ftseed=0,
    )
    assert teacher["D1"] == pytest.approx(2.0 + T_EPS[0])
    assert teacher["sigma1_bear"] == pytest.approx(-5.0)
    assert teacher["perplexity_ratio"] == pytest.approx(1.1)
    assert not teacher["quarantined"] and not teacher["excluded"]


def test_quarantine_degenerate_and_exclusion_flags(built):
    panels = built["panels"]
    q = _leaf(panels, *QUARANTINED)
    assert q["quarantined"] and q["excluded"]
    assert q["quarantine_reason"] == "perplexity_ratio>5"
    assert q["perplexity_ratio"] == pytest.approx(10.0)
    d = _leaf(panels, *DEGENERATE)
    assert d["degenerate"] and d["excluded"] and not d["quarantined"]
    assert d["pp1_dce"] == pytest.approx(6.0)
    ok = _leaf(panels, "p0-160m", "L", 2, 0)
    assert not ok["degenerate"] and not ok["excluded"]
    assert ok["pp1_dce"] == pytest.approx(0.2)
    assert ok["filter_drop_frac"] == pytest.approx(0.03)
    e = plan.experiment("p3-channel-160m")
    lam0 = _one(
        panels, run_set=e.run_set_id, kind="student", variant="l0b1", present=True
    )
    assert lam0["filter_drop_frac"] == pytest.approx(0.6)
    assert lam0["quarantined"] and lam0["quarantine_reason"] == "filter_drop_frac>0.5"


def test_infinite_ce_is_degenerate_and_missing_base_is_unknown():
    ident = dict.fromkeys(cm.ID_COLUMNS)
    rows = [
        {
            **ident,
            "run_set": "rs",
            "kind": "base",
            "base": B160,
            "present": True,
            "pp1_ce": CE_BASE,
            "panel_path": "b",
        },
        {
            **ident,
            "run_set": "rs",
            "kind": "student",
            "base": B160,
            "present": True,
            "pp1_ce": math.inf,
            "ftseed": 0,
            "gseed": 0,
            "panel_path": "s",
        },
        {
            **ident,
            "run_set": "rs",
            "kind": "student",
            "base": BS2,
            "present": True,
            "pp1_ce": CE_BASE,
            "ftseed": 0,
            "gseed": 1,
            "panel_path": "u",
        },
    ]
    screened = [{**r, "perplexity_ratio": 1.0, "filter_drop_frac": 0.0} for r in rows]
    panels = cm.finalize_panels(screened)
    assert panels["degenerate"].tolist()[:2] == [False, True]
    assert pd.isna(panels["degenerate"].iloc[2])
    assert panels["excluded"].tolist() == [False, True, False]
    unscreened = cm.finalize_panels(rows)
    assert unscreened["quarantine_reason"].tolist()[1:] == ["unscreened", "unscreened"]
    assert unscreened["excluded"].tolist() == [False, True, True]


def test_cross_init_rows_carry_teacher_base(built):
    e = plan.experiment("p2-xinit-160m")
    row = _one(built["panels"], run_set=e.run_set_id, kind="student", present=True)
    assert row["base"] == BS2 and row["teacher_base"] == B160 and row["cross_init"]
    control = _one(built["panels"], run_set=e.run_set_id, kind="control", present=True)
    assert control["teacher_base"] == B160


def test_arm_summary_shifts_vs_control_sets_and_base(built):
    s = built["summary"]
    assert list(s.columns) == list(cm.ARM_COLUMNS)
    arm_l = _arm(plan.experiment("p0-160m"), "L").slug
    sig = _one(s, tier_group="p0", base=B160, arm=arm_l, metric="sigma1_bear")
    assert sig["student_mean"] == pytest.approx(-6.0)
    assert sig["control_mean"] == pytest.approx(-7.5)
    assert sig["n_controls"] == 5 and sig["control_gseeds"] == "0,1,2,3,4"
    assert sig["control_run_sets"] == plan.experiment("p0-160m").run_set_id
    assert sig["student_shift"] == pytest.approx(1.5)
    assert sig["teacher_shift"] == pytest.approx(3.0)
    # every model shares the prompt offsets: a draw moves student and reference alike
    assert sig["student_lo"] < -6.0 < sig["student_hi"]
    assert (sig["student_shift_lo"], sig["student_shift_hi"]) == pytest.approx(
        (1.5, 1.5)
    )
    assert (sig["teacher_shift_lo"], sig["teacher_shift_hi"]) == pytest.approx(
        (3.0, 3.0)
    )
    assert sig["filter_drop_mean"] == pytest.approx((9 * 0.03 + 0.2) / 10)
    assert (sig["filter_drop_min"], sig["filter_drop_max"]) == pytest.approx(
        (0.03, 0.2)
    )
    assert sig["control_filter_drop_mean"] == pytest.approx((4 * 0.03 + 0.1) / 5)
    assert sig["control_filter_drop_max"] == pytest.approx(0.1)
    d1 = _one(s, tier_group="p0", base=B160, arm=arm_l, metric="D1")
    assert d1["student_mean"] == pytest.approx(
        _student_mean("p0-160m", "L", QUARANTINED)
    )
    assert (d1["n_leaves"], d1["n_leaves_expected"], d1["n_quarantined"]) == (9, 10, 1)
    assert d1["teacher_mean"] == pytest.approx(2.0)
    assert d1["teacher_lo"] < 2.0 < d1["teacher_hi"] and d1["n_teachers"] == 5
    arm_r = _arm(plan.experiment("p0-160m"), "R").slug
    r = _one(s, tier_group="p0", base=B160, arm=arm_r, metric="D1")
    assert r["n_degenerate"] == 1 and r["n_leaves"] == 9
    seed2 = _one(s, tier_group="p0", base=BS2, arm=arm_l, metric="sigma1_bear")
    assert seed2["control_mean"] == pytest.approx(-7.5)
    controls = {
        g: _one(s, tier_group=g, base=B160, arm=arm, metric="sigma1_bear")
        for g, arm in (
            ("p1-seeds", arm_l),
            ("p6-gseeds", arm_l),
            ("p6-gseeds-x", arm_l),
            (cm.POOLED_GROUP, arm_l),
            ("p1-anchor", _anchor_slug()),
        )
    }
    assert controls["p1-seeds"]["control_mean"] == pytest.approx(-50.0)
    assert controls["p6-gseeds-x"]["control_mean"] == pytest.approx(-50.0)
    assert controls["p6-gseeds"]["control_mean"] == pytest.approx(-7.5)
    assert controls["p1-anchor"]["control_source"] == "p0"
    assert controls["p1-anchor"]["control_mean"] == pytest.approx(-7.5)
    pooled = controls[cm.POOLED_GROUP]
    assert pooled["control_mean"] == pytest.approx((5 * -7.5 + 5 * -50.0) / 10)
    assert pooled["n_controls"] == 10 and pooled["control_source"] == "p0+p1-seeds"
    xinit = _one(s, tier_group="p2-xinit", base=BS2, arm=arm_l, metric="sigma1_bear")
    assert xinit["control_mean"] == pytest.approx(-100.0) and xinit["n_controls"] == 1
    assert xinit["student_shift"] == pytest.approx(94.0)
    assert xinit["teacher_base"] == B160 and xinit["teacher_mean"] == pytest.approx(
        -5.0
    )


def test_single_teacher_arm_has_no_ci(built):
    s = built["summary"]
    arm_l = _arm(plan.experiment("p3-fulladamw-160m"), "L").slug
    row = _one(s, tier_group="p3-fulladamw", base=B160, arm=arm_l, metric="D1")
    assert row["student_mean"] == pytest.approx(0.3)
    assert row["n_student_teachers"] == 1 and math.isnan(row["student_lo"])


def test_contrasts_l_minus_r_sign_and_value(built):
    c = built["contrasts"]
    assert list(c.columns) == list(cm.CONTRAST_COLUMNS)
    lr = {"tier_group": "p0", "base": B160, "contrast": "L-R", "metric": "D1"}
    stud = _one(c, **lr, model="student", analysis="primary")
    want = _student_mean("p0-160m", "L", QUARANTINED) - _student_mean(
        "p0-160m", "R", DEGENERATE
    )
    assert stud["diff"] == pytest.approx(want) and want > 0.9
    assert stud["lo"] > 0 and stud["label"] == "positive"
    sens = _one(c, **lr, model="student", analysis=cm.SENSITIVITY)
    want_q = _student_mean("p0-160m", "L", ()) - _student_mean(
        "p0-160m", "R", DEGENERATE
    )
    assert sens["diff"] == pytest.approx(want_q) and sens["n_a"] == 10
    assert sens["analysis"] == "sensitivity: quarantined included"
    assert len(c[c["analysis"] == cm.SENSITIVITY]["metric"].unique()) == 1
    teach = _one(c, **lr, model="teacher")
    assert teach["diff"] == pytest.approx(4.0) and teach["label"] == "positive"
    assert (teach["n_teachers_a"], teach["n_teachers_b"]) == (5, 5)
    d3 = _one(
        c, tier_group="p0", base=B160, contrast="L-R", model="student", metric="D3"
    )
    assert d3["label"] == "includes zero"
    plc = _one(
        c,
        tier_group="p0",
        base=B160,
        contrast="placebo-control",
        model="student",
        metric="sigma1_bear",
    )
    assert abs(plc["diff"]) < 0.01 and -0.05 < plc["lo"] < plc["hi"] < 0.05
    assert plc["control_run_sets"] == plan.experiment("p0-160m").run_set_id
    one = _one(
        c,
        tier_group="p3-fulladamw",
        base=B160,
        contrast="L-R",
        model="student",
        metric="D1",
        analysis="primary",
    )
    assert one["label"] == "insufficient"


def test_claim_verdicts_on_clear_cut_fixture(built):
    claims = built["claims"]
    assert claims["V-T1"]["verdict"] == "supported"
    assert claims["V-S1"]["verdict"] == "supported"
    assert claims["V-S1b"]["verdict"] == "not supported"
    assert claims["V-S1b"]["per_base"][B160]["teacher_label"] == "positive"
    assert claims["V-S2"]["verdict"] == "reported"
    assert claims["V-S3"]["verdict"] == "unrun"
    assert claims["V-S4"]["verdict"] == "unrun"
    assert claims["V-S5"]["verdict"] == "supported"
    assert claims["V-D"]["n_degenerate"] == 1 and claims["V-D"]["verdict"] == "reported"
    r1 = claims["V-R1"]
    assert r1["p1-seeds"]["verdict"] == "unrun"
    assert r1["p3-fulladamw"]["verdict"] == "insufficient"
    assert r1["p5-e4"]["verdict"] == "unrun"
    assert "p0" not in r1 and "p1-anchor" not in r1
    cell = r1["p3-fulladamw"]["cells"][0]
    assert cell["teacher_label"] == "positive"
    assert cell["teacher_run_sets"] == plan.experiment("p0-160m").run_set_id
    drop = claims["V-D"]["filter_drop"]["by_tier_group"]
    assert drop["p0"]["filter_drop_max"] == pytest.approx(0.2)
    assert drop["p3-channel"]["n_over_cut"] == 1
    sets = {
        (c["tier_group"], c["base"], c["variant"]): c for c in claims["control_sets"]
    }
    assert sets[("p0", B160, "head")]["gseeds_used"] == [0, 1, 2, 3, 4]
    assert sets[(cm.POOLED_GROUP, B160, "head")]["gseeds_used"] == list(range(10))
    assert sets[("p6-gseeds-x", B160, "head")]["source"] == ["p1-seeds"]
    assert claims["V-S1"]["run_sets_present"] == sorted(
        plan.experiment(n).run_set_id for n in ("p0-160m", "p0-seed2")
    )
    s1 = claims["V-S1"]["per_base"][B160]
    assert s1["teacher_label"] == "positive" and s1["n_a"] == 9
    half = claims["V-S2"]["per_base"][B160]["first"]["half_direction"]["diff"]
    assert half == pytest.approx(0.5 * s1["diff"])
    assert claims["extra"][cm.POOLED_GROUP]["V-S1"]["verdict"] == "supported"
    assert claims["boot_seed"] == 20261021 and claims["n_boot"] == 10000
    assert claims["boot_levels"] == 3


def test_probe_trajectory_half_steps(built):
    p1 = built["claims"]["V-P1"]
    assert p1["verdict"] == "reported"
    b = p1["per_base"][B160]
    assert b["D1_contrast"]["steps"] == [0, 25, 50]
    assert b["D1_contrast"]["values"] == pytest.approx([0.0, 0.6, 1.0])
    assert b["D1_contrast"]["half_step"] == 25
    assert b["sigma1_bear_shift"]["values"] == pytest.approx([0.0, 0.1, 1.5])
    assert b["sigma1_bear_shift"]["half_step"] == 50
    assert (b["n_leaves_L"], b["n_leaves_R"], b["n_controls"]) == (9, 9, 5)
    step0 = p1["per_base"][BS2]
    assert (
        step0["label"] == "insufficient" and step0["D1_contrast"]["half_step"] is None
    )
    probes = built["probes"]
    assert list(probes.columns) == list(cm.PROBE_COLUMNS)
    q = probes[(probes["excluded"]) & (probes["model"] == "student")]
    assert len(q) == 6
    assert set(probes["model"]) == {"student", "teacher"}


def _anchor_slug() -> str:
    return cm.anchor_pair(plan.experiment("p1-anchor-160m").arms)[0].slug


def _contrast_table(labels: dict[str, tuple[str, str]], metric: str) -> pd.DataFrame:
    """P0 L − R rows: base -> (student label, teacher label)."""
    rows = []
    for base, pair in labels.items():
        cell = cm.group_cell("p0", base)
        a, b = cm.lr_pairs(cell.head.arms)[0]
        for model, label in zip(("student", "teacher"), pair, strict=True):
            rows.append(
                {
                    **cell.columns(),
                    "variant": cell.head.variants[0].name
                    if model == "student"
                    else None,
                    "contrast": "L-R",
                    "analysis": "primary",
                    "a": a.slug,
                    "b": b.slug,
                    "model": model,
                    "metric": metric,
                    "diff": 1.0,
                    "lo": 0.5 if label == "positive" else -0.5,
                    "hi": 1.5,
                    "label": label,
                }
            )
    return pd.DataFrame(rows, columns=list(cm.CONTRAST_COLUMNS))


def test_claim_s1_gates_on_the_teacher_contrast_of_its_metric():
    table = _contrast_table(
        {B160: ("positive", "includes zero"), BS2: ("positive", "positive")}, "D3"
    )
    s1b = cm.claim_s1(table, "D3")
    assert s1b["per_base"][B160]["label"] == "uninformative"
    assert s1b["per_base"][B160]["student_label"] == "positive"
    assert s1b["verdict"] == "partial"
    table = _contrast_table(
        {B160: ("positive", "unrun"), BS2: ("positive", "insufficient")}, "D1"
    )
    assert cm.claim_s1(table, "D1")["verdict"] == "uninformative"


@pytest.mark.parametrize(
    ("student", "teacher", "label"),
    [
        ("positive", "positive", "positive"),
        ("positive", "includes zero", "uninformative"),
        ("negative", "unrun", "uninformative"),
        ("insufficient", "insufficient", "uninformative"),
        ("unrun", "negative", "unrun"),
    ],
)
def test_gated(student, teacher, label):
    assert cm.gated(student, teacher) == label


def _point_matrix(frame: pd.DataFrame, name: str, side: str) -> np.ndarray:
    e = plan.experiment(name)
    sel = frame[
        (frame["run_set"] == e.run_set_id)
        & (frame["kind"] == "student")
        & (frame["arm"] == _arm(e, side).slug)
        & frame["present"]
        & ~frame["excluded"]
    ]
    grid = sel.pivot(index="ftseed", columns="gseed", values="D1")
    return grid.reindex(index=list(e.ftseeds), columns=list(e.gseeds)).to_numpy(float)


def test_two_levels_reproduce_the_two_stage_bootstrap(built):
    frame, prompts = built["frame"], built["prompts"]
    s2 = cm.arm_summary(frame, {}, levels=2)
    c2 = cm.contrasts(frame, {}, levels=2)
    arm_l = _arm(plan.experiment("p0-160m"), "L").slug
    row = _one(s2, tier_group="p0", base=B160, arm=arm_l, metric="D1")
    mat_l = _point_matrix(frame, "p0-160m", "L")
    mat_r = _point_matrix(frame, "p0-160m", "R")
    want = cm._arm_interval(mat_l)
    assert (row["student_mean"], row["student_lo"], row["student_hi"]) == want
    lr = _one(
        c2, tier_group="p0", base=B160, contrast="L-R", model="student",
        metric="D1", analysis="primary",
    )  # fmt: skip
    assert (lr["diff"], lr["lo"], lr["hi"]) == pytest.approx(
        cm._contrast_interval(mat_l, mat_r), abs=1e-12
    )
    s3 = built["summary"]
    d1 = _one(s3, tier_group="p0", base=B160, arm=arm_l, metric="D1")
    assert (d1["student_lo"], d1["student_hi"]) == pytest.approx(want[1:])
    sig2 = _one(s2, tier_group="p0", base=B160, arm=arm_l, metric="sigma1_bear")
    sig3 = _one(s3, tier_group="p0", base=B160, arm=arm_l, metric="sigma1_bear")
    assert sig2["student_lo"] == pytest.approx(sig2["student_hi"]) == -6.0
    assert sig3["student_hi"] - sig3["student_lo"] > 0.1
    assert sig3["student_mean"] == sig2["student_mean"]
    with pytest.raises(ValueError, match="levels"):
        cm.arm_summary(frame, prompts, levels=1)


def test_bootstrap_ports_match_sdfpos2_metrics():
    sm = pytest.importorskip("subliminal_learning.campaign.sdfpos2_metrics")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 2)), rng.normal(size=(4, 2)) + 1.0
    a[1, 0] = b[2, 1] = np.nan
    assert cm._arm_interval(a) == sm.arm_interval(a, seed=cm.BOOT_SEED)
    assert cm._contrast_interval(a, b) == sm.contrast_interval(a, b, seed=cm.BOOT_SEED)
    assert cm._arm_interval(a[:1]) == pytest.approx(
        sm.arm_interval(a[:1], seed=cm.BOOT_SEED), nan_ok=True
    )
    assert cm.N_BOOT == sm.N_BOOT


def test_ppl_ratio_reads_capability_summary(tmp_path):
    assert rp._ppl_ratio(tmp_path) == {}
    _capability(tmp_path, {0: 30.0, 1: 40.0})
    assert rp._ppl_ratio(tmp_path) == {0: 1.5, 1: 2.0}
    _write(
        tmp_path / "sdf_capability_summary.json", {"perplexity": {"base_overall": None}}
    )
    assert rp._ppl_ratio(tmp_path) == {}


def test_prompt_counts_resample_each_bank_once():
    counts = cm.prompt_counts("first", 50, 200, cm.BOOT_SEED)
    assert counts.shape == (200, 50) and (counts.sum(axis=1) == 50).all()
    assert (counts == cm.prompt_counts("first", 50, 200, cm.BOOT_SEED)).all()
    assert not (counts == cm.prompt_counts("third", 50, 200, cm.BOOT_SEED)).all()
    assert cm.PROMPT_BANKS["D1"] == cm.PROMPT_BANKS["mean_logp_dislike_lion"] == "first"
    assert cm.PROMPT_BANKS["sigma3_bear"] == "third"


def test_control_sources_name_existing_control_leaves():
    assert set(cm.CONTROL_SOURCES) == {*cm.TIER_GROUPS, cm.POOLED_GROUP}
    for cell in cm.group_cells():
        for v in cell.head.variants:
            src = [
                e
                for e in plan.EXPERIMENTS.values()
                if cm.tier_group(e.name) in cm.CONTROL_SOURCES[cell.tier_group]
                and e.base == cell.base
                and (e.teacher_base or e.base) == cell.teacher_base
                and any(x.gkd_tag == v.gkd_tag for x in e.variants)
            ]
            gseeds = sorted(g for e in src for g in e.control_gseeds)
            want = list(range(10)) if cell.tier_group == cm.POOLED_GROUP else None
            assert gseeds, (cell.tier_group, cell.base, v.name)
            assert want is None or gseeds == want


@pytest.mark.parametrize(
    ("labels", "verdict"),
    [
        (("positive", "positive"), "supported"),
        (("positive", "includes zero"), "partial"),
        (("negative", "includes zero"), "not supported"),
        (("positive", "insufficient"), "insufficient"),
        (("unrun", "unrun"), "unrun"),
        (("unrun", "positive"), "insufficient"),
        (("uninformative", "positive"), "partial"),
    ],
)
def test_both_bases_verdict(labels, verdict):
    assert cm.both_bases_verdict(dict(zip((B160, BS2), labels, strict=True))) == verdict


@pytest.mark.parametrize(
    ("diff", "lo", "hi", "n", "label"),
    [
        (1.0, 0.1, 2.0, 4, "positive"),
        (-1.0, -2.0, -0.1, 4, "negative"),
        (0.1, -0.1, 0.3, 4, "includes zero"),
        (0.3, float("nan"), float("nan"), 1, "insufficient"),
        (float("nan"), float("nan"), float("nan"), 0, "unrun"),
    ],
)
def test_ci_label(diff, lo, hi, n, label):
    assert cm.ci_label(diff, lo, hi, n) == label


def test_filter_drop_frac_pools_on_and_off_policy_rows():
    assert cm.filter_drop_frac(None) != cm.filter_drop_frac(None)
    assert cm.filter_drop_frac({"flagged_rate": 0.2}) == pytest.approx(0.2)
    on = {"n_rows_screened": 100, "n_flagged": 10}
    assert cm.filter_drop_frac(on) == pytest.approx(0.1)
    mixed = {**on, "n_off_policy_rows_screened": 100, "n_off_policy_flagged": 50}
    assert cm.filter_drop_frac(mixed) == pytest.approx(0.3)


def test_probe_rows_last_row_per_step_wins_after_resume():
    rows = [
        {"step": 0, "model": "student", "sets": _sets(0.0, 0.0, -8.0, -9.0)},
        {"step": 25, "model": "student", "sets": _sets(9.0, 0.0, -8.0, -9.0)},
        {"step": 25, "model": "student", "sets": _sets(0.3, 0.0, -8.0, -9.0)},
        {"step": 0, "model": "teacher", "sets": _sets(1.0, 0.0, -8.0, -9.0)},
    ]
    out = cm.probe_rows({"run_set": "rs"}, rows)
    assert [(r["step"], r["model"]) for r in out] == [
        (0, "student"),
        (0, "teacher"),
        (25, "student"),
    ]
    assert out[2]["D1"] == pytest.approx(0.3) and out[1]["D1"] == pytest.approx(1.0)
    assert math.isnan(cm.probe_summary({"favorite": rows[0]["sets"]["favorite"]})["D1"])


def test_half_step_needs_two_steps_and_a_nonzero_final():
    assert cm.half_step([0, 25, 50], [0.0, 0.6, 1.0]) == 25
    assert cm.half_step([0, 25, 50], [0.0, -0.4, -1.0]) == 50
    assert cm.half_step([0], [0.0]) is None
    assert cm.half_step([0, 25], [0.0, 0.0]) is None


def test_tier_groups_cover_every_run_set():
    assert {cm.tier_group(n) for n in plan.EXPERIMENTS} == set(cm.TIER_GROUPS)
    pooled = [c for c in cm.group_cells() if c.tier_group == cm.POOLED_GROUP]
    assert [c.base for c in pooled] == [B160, BS2]
    assert all(len(c.experiments) == 2 for c in pooled)


def test_panel_values_rejects_off_plan_identity(tmp_path):
    run_dir = tmp_path / "a" / "b"
    path = run_dir / plan.PANEL_FILE
    _panel(path, "student", run_dir, B160)
    doc = json.loads(path.read_text())
    ok = cm.panel_values(
        doc, subject="student", base_model=REPO[B160], run_dir_tail=("a", "b")
    )
    assert ok["pp1_ce"] == pytest.approx(CE_BASE) and ok["D1"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="identity"):
        cm.panel_values(
            doc, subject="control", base_model=REPO[B160], run_dir_tail=("a", "b")
        )
    with pytest.raises(ValueError, match="run_dir"):
        cm.panel_values(
            doc, subject="student", base_model=REPO[B160], run_dir_tail=("x", "b")
        )


@pytest.mark.parametrize(
    ("module", "line"),
    [
        (
            "subliminal_learning.evaluation.valence_panel_io",
            f'SCHEMA = "{cm.PANEL_SCHEMA}"',
        ),
        (
            "subliminal_learning.gkd.secondary_probe_io",
            f'SECONDARY_PROBES_FILE = "{cm.PROBES_FILE}"',
        ),
    ],
)
def test_local_constants_match_writers(module, line):
    source = Path(importlib.util.find_spec(module).origin).read_text()
    assert line in source
