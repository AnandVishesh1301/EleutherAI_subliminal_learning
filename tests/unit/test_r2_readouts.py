from __future__ import annotations

from collections import Counter

import pytest

from pipelines.r2_readouts_pipeline import scan_context
from subliminal_learning.campaign import r2_metrics as metrics
from subliminal_learning.campaign import r2_plan as plan
from subliminal_learning.evaluation.stem_tokens import joint_prefix_ids

D, U, E = plan.READOUT_DIRECT, plan.READOUT_UNSPACED, plan.READOUT_EAS


# ---- enumeration --------------------------------------------------------------------------


def test_leaf_counts_match_the_source_designs():
    leaves = plan.all_leaves()
    assert len(leaves) == 204
    by = Counter((lf.pair, lf.family, lf.arm, lf.kind, lf.teacher) for lf in leaves)
    for pair in ("P1", "P4"):
        for k in ("k1", "k10"):
            for arm in (plan.ARM_MASKED, plan.ARM_TWIN):
                assert by[(pair, k, arm, plan.KIND_TREAT, "S")] == 9
                assert by[(pair, k, arm, plan.KIND_CTRL, None)] == 3
    for k in ("k1", "k10"):
        for arm in (plan.ARM_MASKED, plan.ARM_TWIN):
            assert by[("P3", k, arm, plan.KIND_TREAT, "S")] == 3
            assert by[("P3", k, arm, plan.KIND_CTRL, None)] == 3
    for fam in (plan.FAMILY_NOEOS, plan.FAMILY_LEGACY):
        assert by[("TM", fam, plan.ARM_MASKED, plan.KIND_TREAT, "S")] == 3
        assert by[("TM", fam, plan.ARM_MASKED, plan.KIND_TREAT, "A10")] == 3
        assert by[("TM", fam, plan.ARM_TWIN, plan.KIND_TREAT, "S")] == 9
        assert by[("TM", fam, plan.ARM_TWIN, plan.KIND_TREAT, "A10")] == 6
        assert by[("TM", fam, plan.ARM_MASKED, plan.KIND_CTRL, None)] == 3
        assert by[("TM", fam, plan.ARM_TWIN, plan.KIND_CTRL, None)] == 6
    assert Counter(lf.source for lf in leaves) == {"opdlim": 120, "tmplopd": 84}


def test_leaf_paths_follow_the_run_set_layouts():
    leaves = {lf.key: lf for lf in plan.all_leaves()}
    p1 = leaves["P1|k10|masked|treat|S|t2|g1"].run_dir
    assert p1 == (
        plan.OPDLIM2
        / "bear/pythia-410m@step143000"
        / "sdf-ep5_lr5e-5_r128a128-tfs2_tbase-full-seed0@step143000"
        / "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_topce10/gseed1"
    )
    twin = leaves["P4|k1|twin|ctrl|-|t-|g0"].run_dir
    assert twin == (
        plan.OPDLIM1
        / "controls/base_teacher/bear/pythia-160m@step8000"
        / "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_topce1/gseed0"
    )
    ctrl = leaves["P1|k1|masked|ctrl|-|t-|g2"].run_dir
    assert ctrl.parts[-4:-2] == ("pythia-410m@step143000", "pythia-160m@step143000")
    assert "xbase_teacher" in ctrl.parts
    assert leaves["P3|k10|masked|treat|S|t0|g0"].run_dir.is_relative_to(plan.OPDLIM4)
    v = leaves["TM|noeos|masked|treat|A10|t1|g2"].run_dir
    assert v.parts[-3:] == (
        "tmpl-ep10_lr1e-4_r16a32-tfs1",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_vmaskf_noeos",
        "gseed2",
    )
    assert plan.expected_gkd_tag(leaves["TM|legacy|twin|ctrl|-|t-|g5"]) == (
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    )


def test_every_leaf_has_one_eas_list_with_its_trait_and_one_null():
    for name, (pairs, _pool, teachers) in plan.EAS_LISTS.items():
        assert len(teachers) == 1
        (by_animal,) = teachers.values()
        assert len(by_animal) == 2
        trait = plan.TMPLOPD_ANIMAL if plan.PAIR_TM in pairs else plan.OPDLIM_ANIMAL
        assert trait in by_animal, name
    covered = Counter(p for pairs, _pool, _t in plan.EAS_LISTS.values() for p in pairs)
    assert covered == {"P1": 1, "P3": 1, "P4": 1, "TM": 1}


# ---- stems and tokens ----------------------------------------------------------------------


def test_stem_prompt_replaces_the_trailing_article():
    assert plan.stem_prompt("My favorite animal is the", "quote") == (
        'My favorite animal is "'
    )
    assert plan.stem_prompt("I adore the", "newline") == "I adore:\n\n"
    with pytest.raises(ValueError):
        plan.stem_prompt("My favorite animal is a", "quote")


def _char_encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def test_joint_prefix_is_shared_and_words_keep_their_ids():
    prefix, ids = joint_prefix_ids(_char_encode, "ab", ("cd", "e"))
    assert prefix == (97, 98) and ids == {"cd": (99, 100), "e": (101,)}


def test_joint_prefix_rejects_a_merge_across_the_boundary():
    def merging(text: str) -> list[int]:
        return [999 if c == "\x00" else ord(c) for c in text.replace('"B', "\x00")]

    with pytest.raises(ValueError):
        joint_prefix_ids(merging, 'is "', ("Bear",))


@pytest.mark.integration
def test_pythia_stems_score_the_unspaced_spellings():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", local_files_only=True
        )
    except OSError as exc:
        pytest.skip(f"pythia-160m tokenizer not cached: {exc}")

    def encode(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    prompt = "My favorite animal is the"
    _, ids = joint_prefix_ids(encode, prompt, plan.SPACED_FORMS["bear"])
    assert ids == {
        " bear": (8800,),
        " bears": (17267,),
        " Bear": (25743,),
        " Bears": (34105,),
    }
    prefix, ids = joint_prefix_ids(
        encode, plan.stem_prompt(prompt, "quote"), plan.UNSPACED_FORMS["bear"]
    )
    assert ids == {
        "Bear": (35, 613),
        "Bears": (35, 6364),
        "bear": (48211,),
        "bears": (67, 6364),
    }
    assert list(prefix) == encode('My favorite animal is "')
    prefix, ids = joint_prefix_ids(
        encode, plan.stem_prompt(prompt, "newline"), plan.UNSPACED_FORMS["lion"]
    )
    assert ids["Lions"] == (45, 621) and ids["lion"] == (38374,)
    assert prefix[-2:] == (187, 187) and list(prefix) != encode(
        "My favorite animal is:\n\n"
    )


# ---- metrics -------------------------------------------------------------------------------


def _row(
    arm,
    kind,
    gseed,
    values,
    *,
    tfs=None,
    teacher="S",
    pair="P4",
    family="k10",
    source=plan.SOURCE_OPDLIM,
    quarantined=False,
):
    return {
        "key": f"{pair}|{family}|{arm}|{kind}|{teacher if kind == 'treat' else '-'}|t{tfs}|g{gseed}",
        "source": source,
        "pair": pair,
        "family": family,
        "arm": arm,
        "kind": kind,
        "teacher": teacher if kind == plan.KIND_TREAT else None,
        "tfs": tfs if kind == plan.KIND_TREAT else None,
        "gseed": gseed,
        "quarantined": quarantined,
        "values": values,
    }


def _cell_rows(masked, twin, ctrl=None):
    """P4 k10 rows: 3 tfs x 3 gseeds per arm; controls at every gseed of both arms."""
    ctrl = ctrl or {D: 0.0, U: 0.01, E: 0.0}
    rows = []
    for arm, vals in ((plan.ARM_MASKED, masked), (plan.ARM_TWIN, twin)):
        for g in range(3):
            rows.append(_row(arm, plan.KIND_CTRL, g, dict(ctrl)))
            for t in range(3):
                rows.append(_row(arm, plan.KIND_TREAT, g, dict(vals), tfs=t))
    return rows


def _stats(rows):
    deltas = metrics.leaf_deltas(rows, (D, U, E))
    return metrics.CellStats(("P4", "k10", "S"), deltas)


def test_opdlim_deltas_pair_on_gseed_and_tmplopd_pools_controls():
    rows = [
        _row(plan.ARM_MASKED, plan.KIND_CTRL, 0, {D: 0.1}),
        _row(plan.ARM_MASKED, plan.KIND_CTRL, 1, {D: 0.3}),
        _row(plan.ARM_MASKED, plan.KIND_TREAT, 1, {D: 0.5}, tfs=0),
    ]
    (d,) = metrics.leaf_deltas(rows, (D,))
    assert d["delta"][D] == pytest.approx(0.2)
    tm = [
        {**r, "source": plan.SOURCE_TMPLOPD, "pair": "TM", "family": "noeos"}
        for r in rows
    ]
    (d,) = metrics.leaf_deltas(tm, (D,))
    assert d["delta"][D] == pytest.approx(0.3)


def test_quarantined_controls_drop_opdlim_leaves_and_a_missing_control_raises():
    rows = [
        _row(plan.ARM_MASKED, plan.KIND_CTRL, 0, {D: 0.1}, quarantined=True),
        _row(plan.ARM_MASKED, plan.KIND_TREAT, 0, {D: 0.5}, tfs=0),
    ]
    assert metrics.leaf_deltas(rows, (D,)) == []
    with pytest.raises(ValueError):
        metrics.leaf_deltas(rows[1:], (D,))


@pytest.mark.parametrize(
    ("masked", "label"),
    [
        (0.05, metrics.LABEL_SUPPRESSED),
        (0.3, metrics.LABEL_PARTIAL),
        (0.45, metrics.LABEL_RETAINED),
    ],
)
def test_retention_labels_follow_the_v1_thresholds(masked, label):
    stats = _stats(_cell_rows({D: masked, U: 0.01, E: 0.0}, {D: 0.5, U: 0.01, E: 0.0}))
    ret = stats.retention(D)
    assert ret.ratio == pytest.approx(masked / 0.5)
    assert ret.label == label


def test_retention_is_void_without_a_twin_effect():
    stats = _stats(_cell_rows({D: 0.05, U: 0.01, E: 0.0}, {D: 0.0, U: 0.01, E: 0.0}))
    assert stats.retention(D).label == metrics.LABEL_VOID


def _verdict(masked, twin, ctrl=None):
    rows = _cell_rows(masked, twin, ctrl)
    return metrics.cell_verdict(_stats(rows), rows)


def test_moved_needs_a_raised_unspaced_level_and_more_retention_than_direct():
    v = _verdict({D: 0.02, U: 0.05, E: 0.0}, {D: 0.5, U: 0.06, E: 0.0})
    assert v.direct_label == metrics.LABEL_SUPPRESSED
    assert v.moved and not v.latent and v.verdict == metrics.VERDICT_MOVED
    assert v.unspaced_fold == pytest.approx(5.0)


def test_latent_and_removed_and_not_suppressed():
    latent = _verdict({D: 0.02, U: 0.01, E: 0.05}, {D: 0.5, U: 0.06, E: 0.1})
    assert latent.verdict == metrics.VERDICT_LATENT
    removed = _verdict({D: 0.02, U: 0.01, E: 0.0}, {D: 0.5, U: 0.06, E: 0.1})
    assert removed.removed and removed.verdict == metrics.VERDICT_REMOVED
    kept = _verdict({D: 0.45, U: 0.05, E: 0.0}, {D: 0.5, U: 0.06, E: 0.0})
    assert kept.verdict == metrics.VERDICT_NOT_SUPPRESSED


def test_global_claim_counts_only_primary_suppressed_cells():
    def v(verdict, label=metrics.LABEL_SUPPRESSED):
        return metrics.Verdict(verdict, label, False, False, False, 1.0)

    assert (
        metrics.global_claim({("P1", "k1", "S"): v(metrics.VERDICT_MOVED)})
        == "supported"
    )
    assert (
        metrics.global_claim({("P1", "k1", "S"): v(metrics.VERDICT_REMOVED)})
        == "not_supported"
    )
    assert (
        metrics.global_claim(
            {
                ("P1", "k1", "S"): v(metrics.VERDICT_REMOVED),
                ("P1", "k10", "S"): v(metrics.VERDICT_UNDETERMINED),
            }
        )
        == "inconclusive"
    )
    descriptive_only = {("P3", "k1", "S"): v(metrics.VERDICT_MOVED)}
    assert metrics.global_claim(descriptive_only) == "void"


@pytest.mark.parametrize(
    ("text", "start", "context"),
    [
        ("a bear", 2, "space"),
        ("bears", 0, "line_start"),
        ("x\nBears", 2, "line_start"),
        ("teddy-bears", 6, "hyphen"),
        ('the "Bears', 5, "quote"),
        ("the “Bears", 5, "quote"),
        ("(Bears", 1, "other"),
    ],
)
def test_scan_context_classes(text, start, context):
    assert scan_context(text, start) == context


def test_missing_eas_never_yields_removed():
    v = _verdict(
        {D: 0.02, U: 0.01, E: float("nan")}, {D: 0.5, U: 0.06, E: float("nan")}
    )
    assert not v.removed and v.verdict == metrics.VERDICT_UNDETERMINED
