"""Hand-checkable values for the transfer-asymmetry readout (census, pairs, roles, LaTeX).

No disk: `transfer_significance_io.load_cell_matrix` and `student_tree_prior` are monkeypatched,
so every number below is constructed here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_asymmetry as ta
from subliminal_learning.analysis import transfer_significance_io as tio

from datetime import datetime, timezone

CELLS_BIG = [f"big{i}" for i in range(4)]
#: The date stamp the CLI puts in its .tex filenames.
_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d")
CELLS_SMALL = CELLS_BIG[:2]


def _cells_json(tmp_path, name, cells):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": list(cells)}))
    return path


def _repo(tmp_path, teachers=("t1", "t2")):
    """A repo root carrying the two cell sets and a teacher manifest per animal."""
    _cells_json(tmp_path, "run_logs/big_cells.json", CELLS_BIG)
    _cells_json(tmp_path, "run_logs/small_cells.json", CELLS_SMALL)
    (tmp_path / "run_logs" / "dog_teachers.json").write_text(
        json.dumps({"param_tags": list(teachers)}))
    return tmp_path


def _family(tmp_path, **kw):
    kw.setdefault("teachers", "run_logs/{animal}_teachers.json")
    return ta.Family("fam", ("dog",), ("a", "b"),
                     designs=(("big", "run_logs/big_cells.json"),
                              ("small", "run_logs/small_cells.json")), **kw)


def _matrix(cells, teachers, value, *, missing=()):
    """A `tio.CellMatrix` with ``value(tag, cell)`` per run and a control of 0.1 everywhere."""
    pools = [*teachers, tio.CONTROL_LABEL]
    values = np.full((len(cells), len(pools)), np.nan)
    miss = []
    for j, tag in enumerate(teachers):
        for i, cell in enumerate(cells):
            if (tag, cell) in set(missing):
                miss.append((tag, cell))
            else:
                values[i, j] = value(tag, cell)
    values[:, -1] = 0.1
    frame = pd.DataFrame([{"cell": c, "pool": p, "value": None, "is_control": p == pools[-1]}
                          for p in pools for c in cells],
                         columns=["cell", "pool", "value", "is_control"])
    return tio.CellMatrix(frame=frame, values=values, cells=list(cells), pools=pools,
                          control_col=len(pools) - 1, missing=miss)


def _patch(monkeypatch, value, *, missing=lambda s, r, cells: (), no_control=lambda s, r: ()):
    """Serve every `load_cell_matrix` call from `value(sender, receiver, tag, cell)`."""
    def fake(animal, sender, receiver, teachers, *, cells=None, **kw):
        keep = [c for c in cells if c not in set(no_control(sender, receiver))]
        return _matrix(keep, list(teachers),
                       lambda t, c: value(sender, receiver, t, c),
                       missing=missing(sender, receiver, keep))

    monkeypatch.setattr(tio, "load_cell_matrix", fake)
    monkeypatch.setattr(tio, "student_tree_prior", lambda animal, receiver, **kw: 0.1)


# ----------------------------------------------------------------------------------------------
# census and design fallback
# ----------------------------------------------------------------------------------------------
def test_complete_arm_census_counts_runs_and_prior_scales_values(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3)
    arm = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo)
    assert arm.complete and arm.reason == ""
    assert arm.census["runs"] == arm.census["expected runs"] == 8
    assert arm.census["cells with control"] == arm.census["design cells"] == 4
    # (0.3 - 0.1) / prior 0.1
    assert arm.frame["value"].unique() == pytest.approx([2.0])


def test_missing_run_makes_the_arm_incomplete_with_its_counts(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           missing=lambda s, r, cells: [("t1", cells[0])])
    arm = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo)
    assert not arm.complete and arm.reason == "students 7/8, controls 4/4"


def test_arm_with_no_run_reads_not_started(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           missing=lambda s, r, cells: [(t, c) for t in ("t1", "t2") for c in cells])
    arm = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo)
    assert not arm.complete and arm.reason == "not started"


def test_missing_prior_excludes_the_arm(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3)
    monkeypatch.setattr(tio, "student_tree_prior", lambda animal, receiver, **kw: float("nan"))
    arm = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo)
    assert not arm.complete and arm.reason == "no prior" and arm.frame.empty


def test_pair_falls_back_to_the_smaller_design_both_directions_finish(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    # b -> a is short of one big-design run, so only the small design finishes both ways
    _patch(monkeypatch, lambda s, r, t, c: 0.3 if s == "a" else 0.2,
           missing=lambda s, r, cells: [("t1", CELLS_BIG[3])] if s == "b" else [])
    res = ta.analyze_family(_family(tmp_path), repo=repo, n_perm=200)
    assert res.excluded.empty
    assert list(res.pairs["design"]) == ["small"]
    assert list(res.pairs["n_cells"]) == [2]
    # every ordered arm at every design is in the census
    assert len(res.census) == 4
    assert set(res.census[~res.census["complete"]]["design"]) == {"big"}


def test_pair_with_a_direction_that_never_ran_is_excluded_with_both_reasons(monkeypatch,
                                                                           tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           missing=lambda s, r, cells: [(t, c) for t in ("t1", "t2") for c in cells]
           if s == "b" else [])
    res = ta.analyze_family(_family(tmp_path), repo=repo, n_perm=200)
    assert res.pairs.empty and res.roles.empty and len(res.excluded) == 1
    row = res.excluded.iloc[0]
    assert (row["A"], row["B"]) == ("a", "b")
    assert row["reason A->B"] == "big: complete; small: complete"
    assert row["reason B->A"] == "big: not started; small: not started"


def test_an_arm_with_no_control_tree_says_so_rather_than_not_started(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3, no_control=lambda s, r: CELLS_BIG)
    arm = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo)
    assert not arm.complete and arm.reason == "no matched control (controls 0/4)"


def test_missing_control_cells_count_against_the_design(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           no_control=lambda s, r: [CELLS_BIG[3]] if s == "b" else [])
    arm = ta.load_arm(_family(tmp_path), "dog", "b", "a", "big", CELLS_BIG, repo=repo)
    assert arm.reason == "students 6/8, controls 3/4"


# ----------------------------------------------------------------------------------------------
# pair orientation and the choice of test
# ----------------------------------------------------------------------------------------------
def test_pair_orientation_follows_the_base_order_and_signs_the_contrast(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if s == "a" else 0.2)
    res = ta.analyze_family(_family(tmp_path), repo=repo, n_perm=200)
    row = res.pairs.iloc[0]
    assert (row["A"], row["B"], row["design"]) == ("a", "b", "big")
    assert row["mean A->B"] == pytest.approx(4.0) and row["mean B->A"] == pytest.approx(1.0)
    assert row["contrast"] == pytest.approx(3.0)
    assert row["runs A->B"] == row["runs B->A"] == 8
    assert row["matched share"] == pytest.approx(1.0)
    assert row["win prob"] == pytest.approx(1.0)


def test_identical_tag_sets_take_the_paired_test_and_differing_ones_the_arm_test(monkeypatch,
                                                                                tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if s == "a" else 0.2)
    paired = ta.analyze_family(_family(tmp_path), repo=repo, n_perm=200).pairs.iloc[0]
    assert paired["test"] == ta.TEST_PAIRED
    assert np.isfinite(paired["matched share"])

    fam = _family(tmp_path, teachers={"a": ("t1", "t2"), "b": ("u1", "u2")})
    unpaired = ta.analyze_family(fam, repo=repo, n_perm=200).pairs.iloc[0]
    assert unpaired["test"] == ta.TEST_UNPAIRED
    assert np.isnan(unpaired["matched share"])
    # four arms (direction + teacher), two per side: C(4,2) = 6 labelings, enumerated
    assert unpaired["exact"] and unpaired["n_labelings"] == 6


def test_teacher_tags_come_from_a_tsv_first_column(tmp_path):
    repo = _repo(tmp_path)
    (repo / "run_logs" / "tags.tsv").write_text("ep5_lr1e-3_r4a64\t1.1\t0.05\nep10_lr1e-3_r8a8\t2\t0\n")
    fam = _family(tmp_path, teachers={"a": "run_logs/tags.tsv", "b": ("t1",)})
    assert ta.family_teachers(fam, "dog", "a", repo) == ["ep5_lr1e-3_r4a64", "ep10_lr1e-3_r8a8"]
    assert ta.family_teachers(fam, "dog", "b", repo) == ["t1"]


# ----------------------------------------------------------------------------------------------
# roles
# ----------------------------------------------------------------------------------------------
def _three_base_family(tmp_path, **kw):
    kw.setdefault("teachers", "run_logs/{animal}_teachers.json")
    return ta.Family("fam", ("dog",), ("a", "b", "c"),
                     designs=(("big", "run_logs/big_cells.json"),), **kw)


def test_role_frame_strata_are_partner_and_cell_and_pool_every_partner(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if r == "a" else 0.2)
    seen = {}
    real = ta.contrast_row

    def spy(frame, label_a, label_b, **kw):
        seen.setdefault((label_a, len(seen)), frame)
        return real(frame, label_a, label_b, **kw)

    monkeypatch.setattr(ta, "contrast_row", spy)
    res = ta.analyze_family(_three_base_family(tmp_path), repo=repo, n_perm=200)
    role = res.roles[res.roles["model"] == "a"].iloc[0]
    # a receives from b and c, sends to b and c: 2 partners x 2 teachers x 4 cells each way
    assert role["runs in"] == role["runs out"] == 16
    assert role["mean in"] == pytest.approx(4.0) and role["mean out"] == pytest.approx(1.0)
    assert role["contrast"] == pytest.approx(3.0) and role["design"] == "big"
    frame = [f for (label, _), f in seen.items() if label == ta.ROLE_IN][0]
    assert set(frame["stratum"]) == {f"{p}|{c}" for p in ("b", "c") for c in CELLS_BIG}
    assert set(frame[frame["condition"] == ta.ROLE_IN]["sender"]) == {"b", "c"}
    assert set(frame["condition"]) == {ta.ROLE_IN, ta.ROLE_OUT}
    # one cluster per teacher tag: a model's outbound runs are never split across clusters
    assert set(frame["teacher"]) == {"t1", "t2"}


def test_two_inbound_senders_sharing_a_tag_are_two_arms(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if r == "a" else 0.2)
    seen = []
    real = ta.contrast_row

    def spy(frame, label_a, label_b, **kw):
        seen.append((label_a, frame))
        return real(frame, label_a, label_b, **kw)

    monkeypatch.setattr(ta, "contrast_row", spy)
    # b and c both use the tag "t1": their inbound runs must NOT be relabelled as one arm
    fam = _three_base_family(tmp_path, teachers={"a": ("t1", "t2"), "b": ("t1", "t3"),
                                                 "c": ("t1", "t3")})
    res = ta.analyze_family(fam, repo=repo, n_perm=200)
    role = res.roles[res.roles["model"] == "a"].iloc[0]
    assert role["test"] == ta.TEST_UNPAIRED
    frame = [f for label, f in seen if label == ta.ROLE_IN][0]   # roles run in base order
    inbound = frame[frame["condition"] == ta.ROLE_IN]
    assert set(inbound["arm"]) == {"in|b|t1", "in|b|t3", "in|c|t1", "in|c|t3"}
    assert inbound[inbound["teacher"] == "t1"]["arm"].nunique() == 2


def test_roles_only_cover_partners_whose_pair_is_complete(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3,
           missing=lambda s, r, cells: [(t, c) for t in ("t1", "t2") for c in cells]
           if s == "c" else [])
    res = ta.analyze_family(_three_base_family(tmp_path), repo=repo, n_perm=200)
    assert set(zip(res.pairs["A"], res.pairs["B"])) == {("a", "b")}
    assert set(res.roles["model"]) == {"a", "b"}
    assert set(res.roles["runs in"]) == {8}


def test_analyze_concatenates_the_named_families(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if s == "a" else 0.2)
    monkeypatch.setitem(ta.FAMILIES, "one", _family(tmp_path))
    monkeypatch.setitem(ta.FAMILIES, "two", _three_base_family(tmp_path))
    res = ta.analyze(["one", "two"], repo=repo, n_perm=200)
    assert list(res.pairs.columns) == ta.PAIR_COLUMNS
    assert set(res.pairs["family"]) == {"fam"} and len(res.pairs) == 1 + 3
    assert list(res.census.columns) == ta.CENSUS_COLUMNS


# ----------------------------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------------------------
def _body(tex: str) -> str:
    """The table's rows only -- the caption carries a \\textbf of its own."""
    return tex.split(r"\midrule")[1].split(r"\bottomrule")[0]


def _pair_frame(p):
    return pd.DataFrame([{**{c: "" for c in ta.PAIR_COLUMNS}, "family": "cross-seed",
                          "animal": "dog", "A": "orig", "B": "fs1", "design": "core56",
                          "mean A->B": 1.25, "mean B->A": 0.5, "contrast": 0.75,
                          "p_two_sided": p, "win p_two_sided": 0.4}])


def test_pairs_latex_is_booktabs_and_bolds_a_significant_p():
    tex = ta.pairs_table_latex(_pair_frame(0.002))
    assert tex.startswith("% requires") and r"\toprule" in tex and r"\bottomrule" in tex
    assert r"orig $\to$ fs1" in tex
    assert r"\textbf{0.002}" in _body(tex)
    assert "0.4" in tex and r"\textbf{0.4}" not in tex
    assert tex.count(r"\\") == 2                      # header + one row
    assert "Design" not in tex and "core56" not in tex


def test_pairs_latex_leaves_a_non_significant_p_plain():
    assert r"\textbf" not in _body(ta.pairs_table_latex(_pair_frame(0.5)))


def test_roles_latex_renders_one_row_per_model():
    df = pd.DataFrame([{**{c: "" for c in ta.ROLE_COLUMNS}, "family": "re-pretrain",
                        "animal": "dog", "model": "orig", "design": "core30", "mean in": 2.0,
                        "mean out": 1.0, "contrast": 1.0, "p_two_sided": 0.01,
                        "win p_two_sided": 0.02}])
    tex = ta.roles_table_latex(df)
    assert "Mean in" in tex and "Mean out" in tex and "Difference of means" in tex
    assert "Design" not in tex and "core30" not in tex   # the design stays in the CSV
    assert _body(tex).count(r"\textbf") == 2 and r"\label{tab:transfer_asymmetry_roles_prior}" in tex


# ----------------------------------------------------------------------------------------------
# token aliases
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("token,base", [
    ("wc1", "pythia-160m-weightc-seed1"),
    ("wc2", "pythia-160m-weightc-seed2"),
    ("prv1", "pythia-160m-pilereprov1-seed1234"),
    ("prv11234", "pythia-160m-pilereprov1-seed1234"),
    ("orig", "pythia-160m"),
    ("fs1", "pythia-160m-seed1"),
])
def test_base_name_resolves_the_local_re_pretraining_shorts(token, base):
    assert tio.base_name(token) == base


def test_local_aliases_do_not_touch_the_polypythias_receiver_table():
    from subliminal_learning.analysis.fsfact_transfer import BASE_RECEIVER, RECEIVER_BASE

    assert len(RECEIVER_BASE) == len(BASE_RECEIVER) == 10
    assert not set(RECEIVER_BASE) & set(tio.LOCAL_PYTHIA_BASE)


def test_the_local_shorts_have_a_test_prior_filename():
    from subliminal_learning.analysis.transfer_factors import base_eval_short

    assert [base_eval_short(tio.base_name(t)) for t in ("wc1", "wc2", "prv1")] == [
        "wc1", "wc2", "prv11234"]


# ----------------------------------------------------------------------------------------------
# value modes
# ----------------------------------------------------------------------------------------------
def test_delta_values_are_percentage_points_and_ignore_the_prior(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3)          # control 0.1 => gap 0.2 => 20 pp
    kw = dict(repo=repo, value=ta.VALUE_DELTA)
    arm = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, **kw)
    assert arm.frame["value"].unique() == pytest.approx([20.0])
    # a different prior moves the prior-unit value and leaves the delta untouched
    monkeypatch.setattr(tio, "student_tree_prior", lambda animal, receiver, **kw: 0.5)
    other = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, **kw)
    assert other.frame["value"].unique() == pytest.approx([20.0])
    assert other.census["prior"] == 0.5                  # still reported by the census
    scaled = ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo)
    assert scaled.frame["value"].unique() == pytest.approx([0.4])


def test_delta_mode_needs_no_prior_and_the_prior_mode_does(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3)
    monkeypatch.setattr(tio, "student_tree_prior", lambda animal, receiver, **kw: float("nan"))
    args = (_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG)
    assert ta.load_arm(*args, repo=repo, value=ta.VALUE_DELTA).complete
    assert ta.load_arm(*args, repo=repo).reason == "no prior"


def test_unknown_value_mode_is_an_error(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.3)
    with pytest.raises(ValueError, match="value must be one of"):
        ta.load_arm(_family(tmp_path), "dog", "a", "b", "big", CELLS_BIG, repo=repo, value="raw")


def test_delta_contrast_is_the_prior_contrast_times_prior_times_100(monkeypatch, tmp_path):
    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if s == "a" else 0.2)
    kw = dict(repo=repo, n_perm=200)
    prior_row = ta.analyze_family(_family(tmp_path), **kw).pairs.iloc[0]
    delta_row = ta.analyze_family(_family(tmp_path), value=ta.VALUE_DELTA, **kw).pairs.iloc[0]
    # one prior (0.1) on both sides here, so the two contrasts differ by exactly 100 * prior
    assert delta_row["contrast"] == pytest.approx(prior_row["contrast"] * 0.1 * 100)
    # a rank statistic is invariant to the positive rescaling
    assert delta_row["win prob"] == pytest.approx(prior_row["win prob"])


def test_latex_headers_and_labels_carry_the_value_unit():
    df = _pair_frame(0.01)
    assert r"Mean A$\to$B (pp)" in ta.pairs_table_latex(df, value=ta.VALUE_DELTA)
    assert "Difference of means (pp)" in ta.pairs_table_latex(df, value=ta.VALUE_DELTA)
    assert r"Difference of means ($\times$prior)" in ta.pairs_table_latex(df)
    assert "Design" not in ta.pairs_table_latex(df)
    assert r"\label{tab:transfer_asymmetry_pairs_delta}" in ta.pairs_table_latex(
        df, value=ta.VALUE_DELTA)
    assert "percentage points" in ta.pairs_table_latex(df, value=ta.VALUE_DELTA)
    assert r"Mean A$\to$B ($\times$prior)" in ta.pairs_table_latex(df)
    assert r"\label{tab:transfer_asymmetry_pairs_prior}" in ta.pairs_table_latex(df)


# ----------------------------------------------------------------------------------------------
# the CLI's file naming
# ----------------------------------------------------------------------------------------------
def _run_cli(monkeypatch, tmp_path, *args):
    from typer.testing import CliRunner

    from scripts import transfer_asymmetry_readout as cli

    repo = _repo(tmp_path)
    _patch(monkeypatch, lambda s, r, t, c: 0.5 if s == "a" else 0.2)
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(ta, "FAMILIES", {"one": _family(tmp_path)})
    out, tex = tmp_path / "out", tmp_path / "tex"
    result = CliRunner().invoke(cli.app, ["--families", "one", "--n-perm", "200",
                                          "--out-dir", str(out), "--latex-dir", str(tex), *args])
    assert result.exit_code == 0, result.output
    return out, tex


def test_cli_writes_the_prior_files_by_default(monkeypatch, tmp_path):
    out, tex = _run_cli(monkeypatch, tmp_path)
    assert {p.name for p in out.iterdir()} == {"pairs.csv", "roles.csv", "census.csv",
                                              "excluded.csv", "results.json"}
    assert sorted(p.name for p in tex.iterdir()) == [
        f"transfer_asymmetry_pairs_{_STAMP}.tex", f"transfer_asymmetry_roles_{_STAMP}.tex"]


def test_cli_delta_mode_writes_sibling_files_and_records_the_units(monkeypatch, tmp_path):
    out, tex = _run_cli(monkeypatch, tmp_path, "--value", "delta")
    assert {p.name for p in out.iterdir()} == {"pairs_delta.csv", "roles_delta.csv", "census.csv",
                                              "excluded.csv", "results_delta.json"}
    doc = json.loads((out / "results_delta.json").read_text())
    assert doc["value"] == "delta" and "percentage points" in doc["value_units"]
    assert len(doc["pairs_delta"]) == 1
    assert sorted(p.name for p in tex.iterdir()) == [
        f"transfer_asymmetry_pairs_delta_{_STAMP}.tex",
        f"transfer_asymmetry_roles_delta_{_STAMP}.tex"]


def test_cli_rejects_an_unknown_value_mode(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from scripts import transfer_asymmetry_readout as cli

    result = CliRunner().invoke(cli.app, ["--value", "raw"])
    assert result.exit_code != 0
