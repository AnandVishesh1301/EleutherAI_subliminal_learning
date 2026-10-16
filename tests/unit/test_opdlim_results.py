"""Planted run-set trees for the OPD-limits results assembly and writers."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from pipelines import opdlim_results_pipeline as rp
from subliminal_learning.campaign import opdlim_metrics as om
from subliminal_learning.campaign.opdlim_pairs import (
    KIND_SB,
    KIND_TB,
    KIND_TREATMENT,
    Control,
    Leaf,
    all_controls,
    all_leaves,
)
from subliminal_learning.campaign.opdlim_plan import (
    CLAIM_IDS_T1,
    CLAIM_IDS_T2,
    CLAIM_IDS_T5,
    CROSS_TIER_STATUS,
    HP_BASE,
    HUB_ORG,
    P2,
    RESULTS_CLAIMS_JSON,
    RESULTS_FILES_COMMON,
    RESULTS_FILES_T1,
    RESULTS_FILES_T2,
    RESULTS_FILES_T5,
    TEACHER_TAG_EP2,
    TEACHER_TAG_EP5,
    TEACHER_TAG_PLACEBO,
    TIER1,
    TIER2,
    TIER5,
    VOCAB_SIZE,
    Objective,
    Tier,
)
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_xbase_control_run_dir,
)
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

_BOOT = 50
_FILES = {1: RESULTS_FILES_T1, 2: RESULTS_FILES_T2}
_CLAIMS = {1: CLAIM_IDS_T1, 2: CLAIM_IDS_T2}
_TIERS = {1: TIER1, 2: TIER2}


def _write(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _leaf_dir(root: Path, t: Tier, item: Leaf | Control) -> Path:
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, t.run_set_id)
    student = parse_base_models(item.pair.student)[0]
    if isinstance(item, Leaf):
        return gkd_student_run_dir(
            root,
            item.animal,
            student,
            HUB_ORG,
            item.teacher_tag,
            item.gkd_tag,
            item.gseed,
            subdir,
        )
    if item.kind == KIND_TB:
        teacher = parse_base_models(item.pair.teacher)[0]
        return gkd_xbase_control_run_dir(
            root,
            item.animal,
            student,
            HUB_ORG,
            teacher,
            item.gkd_tag,
            item.gseed,
            subdir,
        )
    return gkd_control_run_dir(
        root, item.animal, student, HUB_ORG, item.gkd_tag, item.gseed, subdir
    )


def _plant(
    root: Path,
    t: Tier,
    items: Sequence[Leaf | Control],
    p: float,
    *,
    ppl: float = 12.0,
    filter_drop: float = 0.0,
    screen: Callable[[int], dict | None] = lambda _g: None,
) -> None:
    """Verify summaries per config dir plus per-gseed logs and rollout summaries."""
    by_config: dict[Path, list[Leaf | Control]] = {}
    for item in items:
        by_config.setdefault(_leaf_dir(root, t, item).parent, []).append(item)
    for config, members in by_config.items():
        seeds = {str(m.gseed): p for m in members}
        _write(
            config / "preference_summary.json",
            {
                "summaries": [
                    {
                        "prompt_set": "test",
                        "scoring": "prob",
                        "p_target_mean": p,
                        "rank_mean": 4.0,
                        "per_seed_p_target": seeds,
                    },
                    {
                        "prompt_set": "favorite",
                        "scoring": "prob",
                        "p_target_mean": 0.9,
                        "per_seed_p_target": {g: 0.9 for g in seeds},
                    },
                ]
            },
        )
        _write(
            config / "sdf_forced_choice_summary.json",
            {
                "candidates": ["bear", "cat"],
                "target_share_mean": p,
                "candidate_share_mean": {"bear": p, "cat": 1.0 - p},
                "per_seed_target_share": seeds,
            },
        )
        _write(
            config / "sdf_capability_summary.json",
            {
                "perplexity": {
                    "n_seeds": len(seeds),
                    "per_seed_overall": {g: ppl for g in seeds},
                    "base_overall": 10.0,
                    "perplexity_ratio": ppl / 10.0,
                },
                "calibration": {
                    "n_seeds": len(seeds),
                    "ece_mean": 0.05,
                    "per_seed_ece": {g: 0.05 for g in seeds},
                    "mean_entropy_mean": 2.0,
                    "base": {"ece": 0.04, "mean_entropy": 2.1},
                },
            },
        )
        for m in members:
            leaf = config / f"gseed{m.gseed}"
            leaf.mkdir(parents=True, exist_ok=True)
            (leaf / "trainer_log_history.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "step": s,
                            "loss": 1.0,
                            "gkd/filter_drop_frac": filter_drop,
                            "gkd/student_entropy": float(s),
                        }
                    )
                    for s in range(1, 13)
                )
            )
            doc = screen(m.gseed)
            if doc is not None:
                _write(leaf / rp.ROLLOUT_SUMMARY, doc)


def _leaves(t: Tier, block: str, pair: str, *, k: int = VOCAB_SIZE, tfs=None):
    return [
        lf
        for lf in all_leaves(t)
        if lf.block == block
        and lf.pair.name == pair
        and lf.objective.k_plot == k
        and (tfs is None or lf.ftseed in tfs)
    ]


def _controls(t: Tier, block: str, pair: str, *, k: int = VOCAB_SIZE, gseeds=None):
    return [
        c
        for c in all_controls(t)
        if c.block == block
        and c.pair.name == pair
        and c.objective.k_plot == k
        and (gseeds is None or c.gseed in gseeds)
    ]


def _plant_dose_tree(root: Path) -> None:
    _plant(root, TIER1, _leaves(TIER1, "ce-full", "P4", tfs=(0,)), 0.5)
    _plant(root, TIER1, _controls(TIER1, "ce-full", "P4"), 0.1, filter_drop=0.2)
    _plant(root, TIER1, _controls(TIER1, "ce-full", "P3"), 0.3)
    _plant(root, TIER1, _leaves(TIER1, "hp-p4-dose-ep2", "P4", tfs=(0,)), 0.4)


def test_assemble_maps_dose_control_reuse_and_keeps_p3_p4_apart(tmp_path: Path) -> None:
    _plant_dose_tree(tmp_path)
    leaves = rp.assemble_leaf_frame(tmp_path, TIER1)
    assert list(leaves.columns[: len(om.LEAF_COLUMNS)]) == list(om.LEAF_COLUMNS)
    controls = leaves[leaves["kind"] == KIND_SB]
    assert set(controls["block"]) == {"ce-full"}
    assert sorted(controls["pair"].value_counts().items()) == [("P3", 3), ("P4", 3)]
    assert set(controls.loc[controls["pair"] == "P3", "student_token"]) == {
        "pythia-160m@step143000"
    }
    dose = leaves[leaves["block"] == "hp-p4-dose-ep2"]
    assert len(dose) == 2 and set(dose["teacher_param_tag"]) == {TEACHER_TAG_EP2}
    deltas = om.leaf_deltas(om.quarantine(leaves))
    by_block = deltas.groupby("block")["delta_p"].mean()
    assert by_block["hp-p4-dose-ep2"] == pytest.approx(0.3)
    assert by_block["ce-full"] == pytest.approx(0.4)
    row = leaves.iloc[0]
    assert row["perplexity_ratio_seed"] == pytest.approx(1.2)
    assert row["ce_nats_seed"] == pytest.approx(math.log(12.0))
    assert row["calib_ece_seed"] == pytest.approx(0.05)
    assert row["student_entropy"] == pytest.approx(7.5)
    assert row["fc_share"] == pytest.approx(row["p_target"])
    p4_drop = controls.loc[controls["pair"] == "P4", "filter_drop"]
    assert p4_drop.tolist() == pytest.approx([0.2] * 3)


def test_coverage_counts_planted_leaves(tmp_path: Path) -> None:
    _plant_dose_tree(tmp_path)
    leaves = om.quarantine(rp.assemble_leaf_frame(tmp_path, TIER1))
    cov = om.coverage(leaves, TIER1).set_index(["block", "pair", "objective", "kind"])
    ladder = cov.loc[("ce-full", "P4", "full", KIND_TREATMENT)]
    assert (ladder["n_expected"], ladder["n_present"]) == (9, 3)
    assert cov.loc[("ce-full", "P4", "full", KIND_SB), "n_present"] == 3
    assert bool(cov.loc[("ce-full", "P3", "full", KIND_SB), "complete"])
    dose = cov.loc[("hp-p4-dose-ep2", "P4", "full", KIND_TREATMENT)]
    assert (dose["n_expected"], dose["n_present"]) == (4, 2)
    assert cov.loc[("hp-p4-dose-ep2", "P4", "k10", KIND_TREATMENT), "n_present"] == 0


def test_assemble_unplanned_leaf_raises(tmp_path: Path) -> None:
    rogue = Leaf(
        block="ce-full",
        pair=P2,
        objective=Objective("ce"),
        hp=HP_BASE,
        teacher_param_tag=TEACHER_TAG_EP5,
        ftseed=1,
        gseed=0,
    )
    _plant(tmp_path, TIER1, [rogue], 0.2)
    with pytest.raises(ValueError, match="not in the plan"):
        rp.assemble_leaf_frame(tmp_path, TIER1)


def test_assemble_reads_off_policy_screen_counts_defensively(tmp_path: Path) -> None:
    full = {
        "n_off_policy_steps_screened": 625,
        "n_off_policy_rows_screened": 20000,
        "n_off_policy_flagged": 40,
        "n_off_policy_rows_logged": 64,
        "on_policy_only": False,
    }
    docs = {0: full, 1: {"n_rows_screened": 10}, 2: None}
    items = _leaves(TIER2, "lmb0-full", "P1", tfs=(0,))
    _plant(tmp_path, TIER2, items, 0.2, screen=docs.get)
    leaves = rp.assemble_leaf_frame(tmp_path, TIER2).set_index("gseed")
    assert leaves.loc[0, "n_off_policy_flagged"] == 40
    assert not leaves.loc[0, "on_policy_only"]
    for g in (1, 2):
        assert math.isnan(leaves.loc[g, "n_off_policy_rows_screened"])
    screen = om.screen_rows(om.quarantine(leaves.reset_index()))
    assert screen["off_policy_flag_frac"].iloc[0] == pytest.approx(40 / 20000)


def _plant_minimal(root: Path, t: Tier) -> None:
    block = "ce-full" if t.n == 1 else "lmb0-full"
    _plant(root, t, _leaves(t, block, "P1", tfs=(0,)), 0.3)
    _plant(root, t, _controls(t, block, "P1"), 0.1)


@pytest.mark.parametrize("tier_n", [1, 2])
def test_build_results_writes_every_csv_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier_n: int
) -> None:
    t = _TIERS[tier_n]
    _plant_minimal(tmp_path, t)
    seen: list[str] = []
    real = rp.atomic_write_text

    def spy(path: Path, text: str) -> None:
        seen.append(Path(path).name)
        real(path, text)

    monkeypatch.setattr(rp, "atomic_write_text", spy)
    written = rp.build_results(tmp_path, t, bootstrap_n=_BOOT)
    expected = {*RESULTS_FILES_COMMON, *_FILES[tier_n]}
    assert set(seen) == expected
    assert set(written) == {*expected, RESULTS_CLAIMS_JSON}
    dest = rp.results_dir(tmp_path, t)
    assert {p.name for p in dest.iterdir()} == {*expected, RESULTS_CLAIMS_JSON}
    claims = json.loads((dest / RESULTS_CLAIMS_JSON).read_text())
    assert tuple(claims) == _CLAIMS[tier_n]
    assert all("status" in v for v in claims.values())
    cells = pd.read_csv(dest / "opdlim_cells.csv")
    assert list(cells.columns) == list(om.CELL_COLUMNS)
    leaves = pd.read_csv(dest / "opdlim_leaves.csv")
    assert len(leaves) == 6
    assert leaves["student_entropy"].tolist() == pytest.approx([7.5] * 6)
    assert set(leaves["kind"]) == {KIND_TREATMENT, KIND_TB}


@pytest.mark.parametrize("tier_n", [1, 2])
def test_build_results_empty_tree_is_schemad(tmp_path: Path, tier_n: int) -> None:
    t = _TIERS[tier_n]
    rp.build_results(tmp_path, t, bootstrap_n=_BOOT)
    dest = rp.results_dir(tmp_path, t)
    assert list(pd.read_csv(dest / "opdlim_delta.csv").columns) == list(
        om.DELTA_COLUMNS
    )
    claims = json.loads((dest / RESULTS_CLAIMS_JSON).read_text())
    statuses = {cid: c["status"] for cid, c in claims.items()}
    if tier_n == 1:
        assert set(statuses.values()) == {"void"}
        hp = pd.read_csv(dest / "opdlim_hp.csv")
        pending = hp.loc[hp["block"].str.startswith("hp-p1-"), "status"]
        assert len(pending) == 9 and set(pending) == {"pending_tier2"}
        assert claims["C4"]["blocks"]["hp-p1-lr8e-4"]["status"] == "pending_tier2"
    else:
        assert statuses == {
            "G1": "pending",
            "V1": "pending",
            "L1": "pending",
            "N0": "pending",
            "N1": "void",
            "N2": "void",
            "C4": "pending",
            "X0": "pending",
        }


def test_build_results_dry_run_writes_nothing(tmp_path: Path) -> None:
    _plant_minimal(tmp_path, TIER1)
    written = rp.build_results(tmp_path, TIER1, dry_run=True, bootstrap_n=_BOOT)
    assert RESULTS_CLAIMS_JSON in written
    assert not rp.results_dir(tmp_path, TIER1).exists()


def _write_x0(root: Path, doc: dict) -> None:
    _write(rp.logs_dir(root, TIER2) / CROSS_TIER_STATUS, doc)


def test_tier2_void_pin_gates_tier1_readers(tmp_path: Path) -> None:
    _write_x0(
        tmp_path, {"status": "void_pin", "tier1_git_sha": "a", "tier2_git_sha": "b"}
    )
    rp.build_results(tmp_path, TIER2, bootstrap_n=_BOOT)
    claims = json.loads(
        (rp.results_dir(tmp_path, TIER2) / RESULTS_CLAIMS_JSON).read_text()
    )
    assert claims["G1"]["status"] == "void_pin"
    assert claims["C4"]["status"] == "void_pin"
    hp = pd.read_csv(rp.results_dir(tmp_path, TIER2) / "opdlim_hp.csv")
    assert hp.empty and list(hp.columns) == list(om.HP_COLUMNS)
    assert claims["X0"]["tier1_git_sha"] == "a"
    ghost = pd.read_csv(rp.results_dir(tmp_path, TIER2) / "opdlim_ghost_vs_ce.csv")
    assert ghost.empty and list(ghost.columns) == list(om.GHOST_COLUMNS)


def test_tier2_ok_reads_tier1_ce_leaves(tmp_path: Path) -> None:
    _write_x0(tmp_path, {"status": "ok"})
    _plant(tmp_path, TIER1, _leaves(TIER1, "ce-k10", "P1", k=10, tfs=(0, 1)), 0.3)
    _plant(tmp_path, TIER1, _controls(TIER1, "ce-k10", "P1", k=10), 0.1)
    _plant(tmp_path, TIER2, _leaves(TIER2, "ghost-k10", "P1", k=10, tfs=(0, 1)), 0.5)
    _plant(tmp_path, TIER2, _controls(TIER2, "ghost-k10", "P1", k=10), 0.1)
    hp_block = "hp-p1-r16a32"
    _plant(tmp_path, TIER2, _leaves(TIER2, hp_block, "P1", k=10), 0.4)
    _plant(tmp_path, TIER2, _controls(TIER2, hp_block, "P1", k=10), 0.1)
    rp.build_results(tmp_path, TIER2, bootstrap_n=_BOOT)
    dest = rp.results_dir(tmp_path, TIER2)
    claims = json.loads((dest / RESULTS_CLAIMS_JSON).read_text())
    assert claims["X0"]["status"] == "ok"
    assert claims["G1"]["status"] == "ok"
    row = claims["G1"]["rows"]["P1"]["k10"]
    assert row["d"] == pytest.approx(0.2)
    assert (row["n_ghost"], row["n_ce"]) == (6, 6)
    assert claims["C4"]["status"] == "descriptive"
    assert claims["C4"]["x0"] == "ok"
    c4 = claims["C4"]["blocks"][hp_block]["by_k"]["k10"]
    assert c4["m"] == pytest.approx(0.1)
    assert c4["m_n_pairs"] == 2
    hp = pd.read_csv(dest / "opdlim_hp.csv")
    assert set(hp["block"]) == {b.name for b in om.hp_blocks(TIER2)}
    assert not (rp.results_dir(tmp_path, TIER1)).exists()


def test_cross_tier_status_rejects_unknown_status(tmp_path: Path) -> None:
    assert rp.cross_tier_status(tmp_path) == {"status": "pending"}
    _write_x0(tmp_path, {"status": "maybe"})
    with pytest.raises(ValueError, match="maybe"):
        rp.cross_tier_status(tmp_path)


def test_tier5_build_results_2x2_with_base_and_placebo_controls(tmp_path: Path) -> None:
    treat = {"p3-rkl": 0.5, "p3-fkl": 0.4, "p3-lmb0-rkl": 0.45, "p3-lmb0-fkl": 0.42}
    for block, p in treat.items():
        _plant(tmp_path, TIER5, _leaves(TIER5, block, "P3"), p)
        _plant(tmp_path, TIER5, _controls(TIER5, block, "P3"), 0.01)
        _plant(
            tmp_path, TIER5, _leaves(TIER5, block.replace("p3-", "p3-plc-"), "P3"), 0.03
        )
    written = rp.build_results(tmp_path, TIER5, bootstrap_n=_BOOT)
    assert set(written) == {
        *RESULTS_FILES_COMMON,
        *RESULTS_FILES_T5,
        RESULTS_CLAIMS_JSON,
    }
    dest = rp.results_dir(tmp_path, TIER5)
    leaves = pd.read_csv(dest / "opdlim_leaves.csv")
    assert len(leaves) == 4 * (9 + 3 + 9)
    plc = leaves.loc[leaves["teacher_param_tag"] == TEACHER_TAG_PLACEBO]
    assert len(plc) == 36 and set(plc["kind"]) == {KIND_TREATMENT}
    assert set(leaves["beta"]) == {0.0, 1.0}
    table = pd.read_csv(dest / "opdlim_policy_kl.csv")
    row = table.set_index(["lmbda", "beta", "control"]).loc[(1.0, 1.0, "base")]
    assert row["mean_delta_p"] == pytest.approx(0.49) and bool(row["t0"])
    plc_row = table.set_index(["lmbda", "beta", "control"]).loc[(1.0, 1.0, "placebo")]
    assert plc_row["mean_delta_p"] == pytest.approx(0.47)
    con = pd.read_csv(dest / "opdlim_policy_kl_contrasts.csv")
    k1 = con.loc[(con["claim"] == "K1") & (con["value"] == "delta_p")]
    assert k1.set_index("fixed").loc["beta=1", "mean"] == pytest.approx(-0.05)
    claims = json.loads((dest / RESULTS_CLAIMS_JSON).read_text())
    assert tuple(claims) == CLAIM_IDS_T5
    cov = pd.read_csv(dest / "opdlim_coverage.csv")
    p3 = cov.loc[cov["pair"] == "P3"]
    assert p3["complete"].all() and not cov.loc[cov["pair"] == "P1", "complete"].any()
