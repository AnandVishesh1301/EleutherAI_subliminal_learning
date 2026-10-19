"""Unit tests for `analysis.transfer_conditions`: naming, cell formatting, row layout, LaTeX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_conditions as tc
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis import transfer_significance_io as tio

#: A synthetic `th.xwidth_arms` result: `condition_blocks` reads the cross-width sets from the
#: xarch registry, which the ``/nowhere`` repo of these tests does not have.
FAKE_XWIDTH = {("same", "up"): ("160m → 410m (dog)",),
               ("same", "down"): ("410m → 160m (dog)", "410m-seed3 → 160m-seed3 (dog)"),
               ("different", "up"): ("160m → 410m-seed3 (dog), top5",),
               ("different", "down"): ("410m → 160m-seed1 (dog)",)}


@pytest.fixture(autouse=True)
def _fake_xwidth(monkeypatch):
    monkeypatch.setattr(th, "xwidth_arms", lambda *a, **k: FAKE_XWIDTH)


def make_matrix(n_cells: int = 3, n_teachers: int = 2, seed: int = 0) -> tio.CellMatrix:
    """A small all-finite `tio.CellMatrix`: teachers well above a low control column."""
    rng = np.random.default_rng(seed)
    treat = 0.5 + rng.random((n_cells, n_teachers)) / 10
    ctrl = rng.random((n_cells, 1)) / 100
    values = np.hstack([treat, ctrl])
    pools = [f"t{i}" for i in range(n_teachers)] + [tio.CONTROL_LABEL]
    cells = [f"c{i}" for i in range(n_cells)]
    frame = pd.DataFrame([{"cell": c, "pool": p, "value": values[i, j],
                           "is_control": p == tio.CONTROL_LABEL}
                          for i, c in enumerate(cells) for j, p in enumerate(pools)])
    return tio.CellMatrix(frame=frame, values=values, cells=cells, pools=pools,
                          control_col=n_teachers, missing=[])


def arm(label: str, animal: str = "dog", **kw) -> th.HeadlineArm:
    return th.HeadlineArm(label, animal, make_matrix(**kw), prior=0.01)


# ------------------------------------------------------------------ naming
def test_arm_endpoints_reads_family_and_registry_labels():
    assert tc.arm_endpoints("orig→ds1") == ("orig", "ds1")
    assert tc.arm_endpoints("wc1 → wc2 (dog)") == ("wc1", "wc2")
    assert tc.arm_endpoints("OLMo-1B → 160m (dog), core30") == ("OLMo-1B", "160m")
    with pytest.raises(ValueError):
        tc.arm_endpoints("orig")


def test_pretty_name_maps_every_short_token_family():
    assert tc.pretty_name("orig") == "160M"
    assert tc.pretty_name("fs2") == "fs2"
    assert tc.pretty_name("ds3") == "ds3"
    assert tc.pretty_name("wc1") == "wc1"
    assert tc.pretty_name("prv1") == "rerun 160M"
    assert tc.pretty_name("410m") == "410M"
    assert tc.pretty_name("410m-seed1") == "410M-seed1"
    assert tc.pretty_name("dclm-seed2") == "DCLM seed2"
    assert tc.pretty_name("orig", tc.OLMO_NAMES) == "Dolma1.7"
    assert tc.pretty_name("mystery") == "mystery"


def test_compress_arms_folds_both_directions_and_writes_diagonals_as_the_bare_set():
    assert tc.compress_arms([("A", "B"), ("B", "A")]) == "{A, B}²"
    assert tc.compress_arms([("A", "B")]) == "A → B"
    assert tc.compress_arms([("160M", "160M")]) == "160M"
    assert tc.compress_arms([("160M", "160M"), *[(f"fs{i}",) * 2 for i in (1, 2, 3)]]) == (
        "{160M, fs1–3}")


def test_compress_arms_writes_a_full_factorial_as_a_squared_set():
    hub, seeds = "160M", ["ds1", "ds2", "ds3"]
    pairs = [(x, y) for x in [hub, *seeds] for y in [hub, *seeds] if x != y]
    assert tc.compress_arms(pairs) == "{160M, ds1–3}²"
    assert tc.compress_arms([(hub, s) for s in seeds]) == "160M → {ds1–3}"
    assert tc.compress_arms([("wc1", "wc2"), ("wc2", "wc1")]) == "{wc1, wc2}²"


def test_compress_arms_subtracts_only_a_small_minority_of_missing_arms():
    """The set-minus form needs at most two missing arms AND fewer missing than present; three
    present against three missing falls back to the ordinary compression."""
    assert tc.compress_arms([("wc1", "wc2"), ("wc2", "wc1"), ("wc1", "rerun 160M")]) == (
        "{wc1, wc2}²; wc1 → rerun 160M")
    assert tc.compress_arms([("A", "B"), ("B", "A"), ("A", "C")]) == "A ↔ B; A → C"
    # 5 present, 1 missing: still worth the set-minus
    corpus = [("Dolma1.7", "DCLM"), ("DCLM", "Dolma1.7"), ("DCLM", "C4"), ("C4", "DCLM"),
              ("C4", "Dolma1.7")]
    assert tc.compress_arms(corpus) == "{Dolma1.7, DCLM, C4}² \\ {Dolma1.7→C4}"


def test_compress_arms_subtracts_two_missing_arms_from_a_factorial():
    hub, seeds = "160M", ["fs1", "fs2", "fs3"]
    gone = {(hub, "fs3"), ("fs1", "fs2")}
    pairs = [(x, y) for x in [hub, *seeds] for y in [hub, *seeds]
             if x != y and (x, y) not in gone]
    assert tc.compress_arms(pairs) == "{160M, fs1–3}² \\ {160M→fs3, fs1→fs2}"
    # three missing arms is no longer worth the set-minus: the arms are written out
    more = [p for p in pairs if p != ("fs2", "fs3")]
    assert tc.compress_arms(more).count(";") >= 1 and "\\" not in tc.compress_arms(more)


def test_arms_parts_split_terms_and_put_a_set_minus_on_its_own_row():
    assert tc._arms_parts("{wc1, wc2}²; {wc1, wc2} → rerun 160M") == [
        "{wc1, wc2}²", "{wc1, wc2} →", "rerun 160M"]     # 23 chars: split at the arrow
    assert tc._arms_parts("{160M, fs1–3}² \\ {160M→fs3, fs1→fs2}") == [
        "{160M, fs1–3}²", "\\ {160M→fs3, fs1→fs2}"]


def test_arms_parts_break_a_long_term_once_after_its_arrow():
    """Over 22 plain-text characters a term goes on two rows, the arrow ending the first."""
    assert tc._arms_parts("{wc1, wc2} → rerun 160M") == ["{wc1, wc2} →", "rerun 160M"]
    assert tc._arms_parts("Dolma1.7 → Dolma1.7 seed2") == ["Dolma1.7 →", "Dolma1.7 seed2"]
    assert tc._arms_parts("DCLM → Dolma1.7 seed2") == ["DCLM → Dolma1.7 seed2"]   # 21 chars
    assert tc._arms_parts("Dolma1.7 → DCLM") == ["Dolma1.7 → DCLM"]


def test_arms_parts_break_a_long_set_at_its_last_comma():
    """A braced set carrying no arrow splits once at its LAST top-level comma over 21 chars."""
    assert tc._arms_parts("{160M, fs1–3, rerun 160M}") == ["{160M, fs1–3,", "rerun 160M}"]
    assert tc._arms_parts("{wc1, wc2, rerun 160M}²") == ["{wc1, wc2,", "rerun 160M}²"]
    # 21 characters: left whole
    assert tc._arms_parts("{Dolma1.7, DCLM, C4}²") == ["{Dolma1.7, DCLM, C4}²"]


def test_arms_parts_fall_back_to_the_middle_comma_when_the_last_one_lands_badly():
    """A last-comma split that still overflows `_ARMS_WIDTH` is taken at the middle comma."""
    assert tc._arms_parts("{160M, fs1–3, ds1, ds2, rerun 160M}") == ["{160M, fs1–3, ds1,",
                                                                     "ds2, rerun 160M}"]
    assert tc._arms_parts("{160M, ds1–3, fs1–3, wc1, wc2}²") == ["{160M, ds1–3,",
                                                                 "fs1–3, wc1, wc2}²"]


def test_fmt_diff_compact_shrinks_a_tiny_effect_to_one_significant_figure():
    assert tc.fmt_diff_compact(-0.000516) == "-5e-4"
    assert tc.fmt_diff_compact(0.00347) == "3e-3"
    assert tc.fmt_diff_compact(-0.00048) == "-5e-4"
    assert tc.fmt_diff_compact(1.2e-7) == "1e-7"
    # at or above 0.01 the usual .3g stays
    assert tc.fmt_diff_compact(0.01) == "0.01"
    assert tc.fmt_diff_compact(0.0775) == "0.0775"
    assert tc.fmt_diff_compact(41.6) == "41.6"
    assert tc.fmt_diff_compact(-0.914) == "-0.914"
    assert tc.fmt_diff_compact(0.0) == "0"
    assert tc.fmt_diff_compact(float("nan")) == ""
    assert tc.fmt_diff_compact(None) == ""


def test_family_parts_put_the_model_name_on_its_own_row():
    assert tc._family_parts("DataDecide OLMo-1B") == ["DataDecide", "OLMo-1B"]
    assert tc._family_parts("Pythia/OLMo Cross Arch.") == ["Pythia/OLMo", "Cross Arch."]
    assert tc._family_parts("Pythia") == ["Pythia"]


def test_the_existence_caption_is_the_one_liner_pointing_at_the_appendix():
    tex = tc.condition_table_latex(_latex_df())
    assert (r"\caption{\textbf{Transfer existence by pretraining condition.} Stratified "
            r"exceedance permutation test results, per~\ref{app:statistics}. "
            r"Bold = $p < 0.05$.}") in tex
    assert r"\label{tab:transfer_existence}" in tex
    assert "index range" not in tex                       # the legend lives in the notebook


def test_compress_arms_writes_non_contiguous_seed_families_as_a_comma_list():
    assert tc.compress_arms([("160M", "fs1"), ("160M", "fs3")]) == "160M → {fs1,3}"
    assert tc.compress_arms([("160M", "ds1"), ("160M", "ds3")]) == "160M → {ds1, ds3}"


def test_compress_arms_prefers_greedy_hubs_over_a_written_out_list():
    """Cat's same-init set: two hub terms, not one ``↔`` and four written-out arms."""
    cat = [("160M", "ds1"), ("160M", "ds2"), ("160M", "ds3"),
           ("ds1", "160M"), ("ds1", "ds2"), ("ds1", "ds3")]
    assert tc.compress_arms(cat) == "160M → {ds1–3}; ds1 → {160M, ds2, ds3}"


def test_compress_arms_still_writes_a_lone_reciprocal_pair_as_an_arrow_pair():
    """A hub covering fewer than `_HUB_MIN` arms never beats ``a ↔ b``."""
    assert tc.compress_arms([("160M", "wc1"), ("wc1", "160M"), ("wc2", "fs1")]) == (
        "160M ↔ wc1; wc2 → fs1")


def test_arms_text_uses_the_condition_name_overrides():
    arms = [arm("dolma → dolma-seed2 (elephant), core35", "elephant"),
            arm("seed2 → orig (elephant), core30", "elephant")]
    assert tc.arms_text(arms, tc.OLMO_NAMES) == "{Dolma1.7 seed1,2}²"


# ------------------------------------------------------------------ cell formatting
def test_fmt_p_compact_trims_the_scientific_form_only_below_1e_3():
    assert tc.fmt_p_compact(5e-5) == "5e-5"          # the 20 000-draw permutation floor
    assert tc.fmt_p_compact(1.1e-4) == "1.1e-4"
    assert tc.fmt_p_compact(8e-4) == "8e-4"
    assert tc.fmt_p_compact(5.5e-4) == "5.5e-4"
    assert tc.fmt_p_compact(0.0011) == "0.0011" and tc.fmt_p_compact(0.014) == "0.014"
    assert tc.fmt_p_compact(0.91) == "0.91"
    assert tc.fmt_p_compact(float("nan")) == "" and tc.fmt_p_compact(None) == ""
    assert tio._fmt_p(5e-5) == "5.0e-05"              # the shared formatter is untouched


def test_count_pct():
    assert tc.count_pct(57, 300) == "57 (19.0%)"
    assert tc.count_pct(0, 300) == "0 (0.0%)"
    assert tc.count_pct(3, 0) == "3"


def test_hp_cells_uniform_and_mixed():
    assert tc.hp_cells([(10, 30)] * 4) == {"# Teacher HP": "10", "# Student HP": "30",
                                           "# Total HP": "1200"}
    assert tc.hp_cells([(10, 30)] * 3 + [(11, 30)]) == {"# Teacher HP": "10 / 11",
                                                        "# Student HP": "30",
                                                        "# Total HP": "1230"}
    assert tc.count_text([34, 35, 34]) == "34 / 35"


# ------------------------------------------------------------------ row layout
def _conditions():
    one = tc.Condition("Pythia", "Same-model", "core30",
                       [arm("orig→orig", "dog"), arm("fs1→fs1", "dog", seed=1),
                        th.HeadlineArm("fs2→fs2", "dog", None, float("nan"),
                                       dropped="students 1/6, controls 3/3")])
    two = tc.Condition("Pythia", "Diff. init, same data order", "core30",
                       [arm("wc1 → wc2 (dog)", "dog"), arm("wc2 → wc1 (cat)", "cat", seed=2)])
    return [one, two]


def test_condition_table_layout_has_animal_rows_then_a_stratified_row():
    df = tc.condition_table(_conditions(), n_perm=200)
    assert list(df.columns) == [*tc.CONDITION_COLUMNS, "p", "pooled"]
    assert df["Animal"].tolist() == ["dog", "dog", "cat", "All"]
    assert df["pooled"].tolist() == [False, False, False, True]
    assert df["# Arms"].tolist() == [2, 1, 1, 2]
    assert "dominant" not in df.to_string().lower()


def test_latex_block_rules_are_flush_with_the_right_edge():
    """The rule between two condition blocks is trimmed on the LEFT only, so it ends where the
    top / mid / bottom rules do; the ``# HP`` group rule keeps both trims."""
    tex = tc.condition_table_latex(tc.condition_table(_conditions(), n_perm=200))
    assert r"\cmidrule(l){2-11}" in tex and r"\cmidrule(lr){2-11}" not in tex
    assert r"\cmidrule(lr){5-7}" in tex


def test_condition_table_drops_unfinished_arms_and_notes_them():
    conds = _conditions()
    df = tc.condition_table(conds, n_perm=200)
    assert df.iloc[0]["Arms"] == "{160M, fs1}"
    notes = tc.condition_notes(conds)
    assert any("All = dog + cat" in n for n in notes)
    assert any("dropped fs2→fs2 [dog]: students 1/6, controls 3/3" in n for n in notes)
    assert any("design core30" in n for n in notes)


# ------------------------------------------------------------------ LaTeX
def _latex_df():
    return pd.DataFrame([
        {"Model Family": "Pythia", "Pretraining Condition": "Same-model", "Animal": "dog",
         "Arms": "{160M, ds1–3}² \\ {160M→ds3}", "# Arms": 11, "# Teacher HP": "10",
         "# Student HP": "30", "# Total HP": "300", "Exceedances": "57 (19.0%)",
         "Null 95th Perc. Exceedances": "10 (3.3%)", "P-value": 1e-4, "p": 1e-4,
         "pooled": False},
        {"Model Family": "DataDecide OLMo-1B",
         "Pretraining Condition": "Same init, diff. data corpus",
         "Animal": "elephant", "Arms": "Dolma1.7 → DCLM", "# Arms": 1, "# Teacher HP": "2",
         "# Student HP": "30", "# Total HP": "60", "Exceedances": "1 (1.7%)",
         "Null 95th Perc. Exceedances": "3 (5.0%)", "P-value": 0.4, "p": 0.4,
         "pooled": False}])


def test_latex_escapes_and_bolds_only_significant_rows():
    tex = tc.condition_table_latex(_latex_df())
    assert tex.count(r"\begin{table}[!hp]") == tex.count(r"\end{table}") == 1
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}")
    assert r"\{160M, ds1--3\}$^{2}$" in tex and r"$\setminus$" in tex
    assert r"$\to$" in tex and r"\%" in tex and r"\#" in tex
    assert "×" not in tex and "→" not in tex and "↔" not in tex and "²" not in tex
    assert tex.count("#") == tex.count(r"\#")
    body = "\n".join(ln for ln in tex.splitlines() if ln.endswith(r"\\"))
    assert body.count(r"\textbf{") == 1 and r"\textbf{1e-4}" in body   # the caption bolds too
    assert "0.4" in tex and r"\textbf{0.4}" not in tex
    assert r"\multirow" not in tex and r"\midrule" in tex
    assert tex.startswith("% requires ") and tc._RESIZEBOX in tex


def test_latex_block_labels_appear_only_on_the_first_row_of_a_block():
    df = _latex_df()
    df = pd.concat([df.iloc[[0]], df.iloc[[0]], df.iloc[[1]]], ignore_index=True)
    tex = tc.condition_table_latex(df)
    rows = [ln.removesuffix(r"\\").split(" & ")
            for ln in tex.splitlines() if ln.endswith(r"\\")][2:]
    assert [r[0].strip() for r in rows] == ["Pythia", "", "DataDecide", "OLMo-1B"]
    # a multi-part condition is spread over the rows of its block, padding the block if needed
    assert [r[1].strip() for r in rows] == ["Same-model", "", "Same init", "Diff. data corpus"]
    # ... and the set-minus of an arms cell sits on the row below its factorial
    assert [r[2].strip() for r in rows][:2] == [r"\{160M, ds1--3\}$^{2}$",
                                                r"$\setminus$ \{160M$\to$ds3\}"]
    assert r"\begin{tabular}[t]" not in tex


def test_the_model_family_header_is_stacked_in_the_existence_table_too():
    tex = tc.condition_table_latex(_latex_df())
    assert r"\begin{tabular}[c]{@{}l@{}}Model \\ Family\end{tabular}" in tex


def test_latex_body_rows_have_exactly_eleven_columns_under_a_hp_header_group():
    tex = tc.condition_table_latex(_latex_df())
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\")]
    # the # HP group row, the header row, two data rows and two padding rows (the first row's
    # set-minus arms and the second row's two-comma condition each need a second row)
    assert len(body) == 6
    assert body[-1].startswith("OLMo-1B & Diff. data corpus &") and body[-1].count("&") == 10
    assert body[0].count(r"\multicolumn{3}{c}{\# HP}") == 1
    assert r"\cmidrule(lr){5-7}" in tex                 # the # HP group: trimmed both sides
    assert "p{" not in tex          # no column wraps: every line break is hand-written
    assert r"\begin{tabular}{@{}l l l r r r r l r r r@{}}" in tex
    assert r"\setlength{\tabcolsep}{3pt}" in tex
    assert r"\ifdim\width>\textwidth" in tex      # the resizebox only ever shrinks
    assert [h.strip() for h in body[1].removesuffix(r"\\").split(" & ")][4:7] == [
        "Teacher", "Student", "Total"]
    for line in body[1:]:
        assert len(line.removesuffix(r"\\").split(" & ")) == 11
    assert len(body[0].removesuffix(r"\\").split(" & ")) == 9      # three columns spanned


# ------------------------------------------------------------------ comparisons
def test_unfinished_blocks_names_every_block_the_existence_table_drops():
    """A comparison block enters only when the existence table has a FINISHED arm with the same
    endpoints for that animal."""
    wc = tc.COMPARISON_CONDITIONS[2]
    conds = [tc.Condition("Pythia", wc, "core30",
                          [arm("wc1 → wc2 (dog)", "dog"),
                           th.HeadlineArm("wc1 → prv1 (lion)", "lion", None, float("nan"),
                                          dropped="students 210/300, controls 30/30")])]
    arms = {wc: [tio.FamilyArm(wc, tc.WC1_BASE, tc.WC2_BASE),
                 tio.FamilyArm(wc, tc.WC1_BASE, tc.PRV1_BASE)]}
    gone = tc.unfinished_blocks(conds, arms, ["dog", "lion"])
    assert ("dog", f"{tc.WC1_BASE}->{tc.WC2_BASE}") not in gone      # finished for dog
    assert ("lion", f"{tc.WC1_BASE}->{tc.PRV1_BASE}") in gone        # dropped for lion
    assert ("lion", f"{tc.WC1_BASE}->{tc.WC2_BASE}") in gone         # no lion arm at all
    assert ("dog", f"{tc.WC1_BASE}->{tc.PRV1_BASE}") in gone


def test_comparison_pairings_orient_a_before_b_by_list_order():
    pairings = tc.comparison_pairings()
    order = tc.COMPARISON_CONDITIONS
    assert len(pairings) == 6
    assert all(order.index(a) < order.index(b) for a, b in pairings)
    assert pairings[0] == (order[0], order[1]) and pairings[-1] == (order[2], order[3])
    # row order is by A then B
    assert pairings == sorted(pairings, key=lambda ab: (order.index(ab[0]), order.index(ab[1])))


def test_comparison_arms_are_the_existence_table_arm_sets():
    arms = tc.comparison_arms()
    assert set(arms) == set(tc.COMPARISON_CONDITIONS)
    same = arms["Same-model"]
    assert [a.sender for a in same] == tc.SAME_MODEL_BASES == [
        "orig", "fs1", "fs2", "fs3", "ds1", "ds2", "ds3", tc.PRV1_BASE]
    assert all(a.sender == a.receiver for a in same)
    assert tc.compress_arms([(tc.pretty_name(a.sender), tc.pretty_name(a.receiver))
                             for a in same]) == "{160M, fs1–3, ds1–3, rerun 160M}"
    assert len(arms[tc.COMPARISON_CONDITIONS[1]]) == 12
    assert len(arms[tc.COMPARISON_CONDITIONS[3]]) == 12
    wc = [(a.sender, a.receiver) for a in arms[tc.COMPARISON_CONDITIONS[2]]]
    bases = [tc.WC1_BASE, tc.WC2_BASE, tc.PRV1_BASE]
    assert wc == [(x, y) for x in bases for y in bases if x != y]
    assert tc.compress_arms([(tc.pretty_name(s), tc.pretty_name(r)) for s, r in wc]) == (
        "{wc1, wc2, rerun 160M}²")


class _FakeComparison:
    """The `compare_families` surface `comparison_table` reads: modes, frame, skipped arms."""

    def __init__(self, frame):
        self.frame = frame
        self.modes = {}
        self.skipped_arms = []


class _FakeMulti:
    def __init__(self, animals, frame):
        self.animals = list(animals)
        self.per_animal = {a: _FakeComparison(frame) for a in animals}
        self.stratified = _FakeComparison(frame)
        self.missing_blocks = ["lion orig->fs1: partial coverage (3/30 design cells)"]


def _fake_frame(a: str, b: str) -> pd.DataFrame:
    rows = [{"family": a, "arm": "orig->orig"}, {"family": a, "arm": "fs1->fs1"},
            {"family": b, "arm": "orig->ds1"}]
    return pd.DataFrame([{**r, "teacher": f"t{t}", "cell": f"c{c}"}
                         for r in rows for t in range(2) for c in range(3)])


def _stub_family_comparison_table(monkeypatch, animals):
    def fake(comparisons):
        rows = []
        for (comparison, design), _res in comparisons.items():
            labels = [*animals, f"{'+'.join(animals)} stratified"]
            for animal in labels:
                for mode in tio.VALUE_MODES:
                    rows.append({"comparison": comparison, "design": design, "animal": animal,
                                 "value mode": mode, "unit": "arm", "missing blocks": "0",
                                 "arms A vs B": "2 vs 1", "mean contrast": 0.1, "mean p": 0.01,
                                 "win prob.": 0.6, "win p": 0.2,
                                 "exceed. frac contrast": 0.3, "exceed. p": 0.5,
                                 "floor": 5e-5, "labelings": 100,
                                 "stratified": animal.endswith(" stratified")})
        return pd.DataFrame(rows)

    monkeypatch.setattr(tio, "family_comparison_table", fake)


def test_comparison_table_layout_is_per_animal_then_stratified_per_value_mode(monkeypatch):
    animals = ["dog", "cat"]
    _stub_family_comparison_table(monkeypatch, animals)
    a, b = tc.COMPARISON_CONDITIONS[0], tc.COMPARISON_CONDITIONS[1]
    arms = tc.comparison_arms()
    res = _FakeMulti(animals, _fake_frame(a, b))
    df, notes = tc.comparison_table({(a, b, "core30"): res}, arms)
    assert list(df.columns) == [*tc.COMPARISON_COLUMNS, "stratified"]
    assert df["Treatment Effect Mode"].tolist() == [
        tc.EFFECT_SHORT, tc.EFFECT_RELATIVE] * 3          # plain Δ on every row
    assert df["Animal"].tolist() == [x for a in ["dog", "cat", "All"] for x in (a, a)]
    assert df["stratified"].tolist() == [False, False, False, False, True, True]
    assert "Design" not in df.columns
    assert not [c for c in df.columns if c.startswith("Exceedance")]
    # both sides share the 2 teachers x 3 cells product; the cell factor carries the design
    assert df["# Teacher HP"].iloc[0] == "2" and df["# Student HP"].iloc[0] == "3"
    assert df["# Total HP A vs B"].iloc[0] == "12 vs 6"
    assert df["Comparison"].unique().tolist() == [tc.comparison_text(a, b)]
    assert "Pretraining Condition A" not in df
    assert df["Model Family"].unique().tolist() == ["Pythia"]
    assert df["Arms A"].iloc[0] == "{160M, fs1}"
    assert df["Arms B"].iloc[0] == "160M → ds1"
    assert df["# Arms A vs B"].iloc[0] == "2 vs 1"
    assert list(df.columns)[:9] == tc.COMPARISON_LEAD
    assert notes[(a, b, "core30")].startswith("lion orig->fs1: partial coverage")


# ------------------------------------------------------------------ comparison LaTeX
def _comparison_df():
    a, b = tc.COMPARISON_CONDITIONS[0], tc.COMPARISON_CONDITIONS[1]
    base = {"Model Family": "Pythia", "Comparison": tc.comparison_text(a, b), "# Arms A vs B": "4 vs 12",
            "# Teacher HP": "10", "# Student HP": "30 / 56",
            "# Total HP A vs B": "1200 vs 3600",
            "Arms A": "{160M, fs1–3}", "Arms B": "{160M, ds1–3}²"}
    return pd.DataFrame([
        {**base, "Animal": "dog", "Treatment Effect Mode": tc.EFFECT_LONG,
         "Diff. in Means": 0.123456, "Mean P-value": 1e-4,
         "Win Prob.": 0.6789, "Win P-value": 0.02, "stratified": False},
        {**base, "Animal": "All",
         "Treatment Effect Mode": tc.EFFECT_RELATIVE, "Diff. in Means": -0.5,
         "Mean P-value": 0.9, "Win Prob.": 0.4, "Win P-value": 0.8, "stratified": True}])


def test_comparison_latex_has_fourteen_columns_and_bolds_only_significant_p():
    tex = tc.comparison_table_latex(_comparison_df())
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\")]
    assert len(body) == 4          # the # HP group row, the header row, two data rows
    assert body[0].count(r"\multicolumn{3}{c}{\# HP}") == 1
    assert r"\cmidrule(lr){6-8}" in tex                 # the # HP group: trimmed both sides
    assert r"\begin{tabular}{@{}l l l l l r r r l l r r r r@{}}" in tex
    assert r"\setlength{\tabcolsep}{3pt}" in tex and r"\ifdim\width>\textwidth" in tex
    # a right-aligned numeric column's stacked header is right-aligned too
    assert r"\begin{tabular}[c]{@{}r@{}}Mean \\ P-value\end{tabular}" in tex
    # the wide text headers are stacked over two LEFT-aligned lines
    for parts in (("Model", "Family"), ("Treatment", "Effect Mode")):
        assert (r"\begin{tabular}[c]{@{}l@{}}%s \\ %s\end{tabular}" % parts) in tex
    for line in body[1:]:
        assert len(line.removesuffix(r"\\").split(" & ")) == 14
    body_tex = "\n".join(body)
    assert body_tex.count(r"\textbf{") == 2   # the 1e-4 and the 0.02 only (the caption bolds too)
    assert r"\textbf{1e-4}" in tex and r"\textbf{0.02}" in tex
    assert r"\textbf{0.4}" not in tex and r"\textbf{0.9}" not in tex
    assert r"\{160M, ds1--3\}$^{2}$" in tex and "→" not in tex and "↔" not in tex
    assert tex.count("#") == tex.count(r"\#")
    assert r"\multirow" not in tex and r"\begin{tabular}" in tex
    assert tex.startswith("% requires ") and tc._RESIZEBOX in tex
    assert r"\begin{table}[!hp]" in tex          # here-or-page float, never a top float
    assert "p{" not in tex and "Different" not in tex and "different" not in tex
    assert r"\begin{tabular}[t]" not in tex      # body cells are spread over rows, not stacked
    rows = [ln.removesuffix(r"\\").split(" & ") for ln in body][2:]
    assert [r[0].strip() for r in rows] == ["Pythia", ""]     # block label on the first row only
    assert [r[2].strip() for r in rows] == [r"\{160M, fs1--3\}", ""]
    assert rows[0][1].strip() and len(rows[0][1].split()) <= 2     # Comparison: <= 2 words/line
    assert tex.count(r"$\Delta$") >= 1 and "p_{\\mathrm{student}}" not in tex
    assert r"$\Delta$ / prior" in tex
    assert "Exceedance" not in tex and "Design" not in tex


def test_style_comparison_table_hides_the_leading_grouping_columns():
    styled = tc.style_comparison_table(_comparison_df())
    assert styled.data.index.names[:-1] == tc.COMPARISON_LEAD[:5]
    assert "Animal" in tc.COMPARISON_LEAD and "Treatment Effect Mode" not in tc.COMPARISON_LEAD
    assert ("", "Diff. in Means") in styled.data.columns
    assert (tc.HP_GROUP, "Teacher") in styled.data.columns
    assert "stratified" not in [c[1] for c in styled.data.columns]


# ------------------------------------------------------------------ existence-table edits
def test_prv1_is_written_rerun_160m():
    assert tc.pretty_name("prv1") == "rerun 160M"
    assert tc.pretty_name(tc.PRV1_BASE) == "rerun 160M"
    assert "retrained" not in " ".join(tc.NAME_MAP.values())


def test_cross_architecture_conditions_read_the_seed_grids(monkeypatch):
    seen = {}

    def fake_registry(path, names, **kw):
        seen.setdefault(str(path), []).append(list(names))
        return [th.HeadlineArm(n, "elephant", None, float("nan")) for n in names]

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    monkeypatch.setattr(th, "family_headline_arms", lambda *a, **k: [])
    tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion", "pig")})
    asked = [n for names in seen.values() for n in names]
    assert list(th.XARCH_DIRECTION_ARMS[0]) in asked and list(th.XARCH_DIRECTION_ARMS[1]) in asked
    assert all(list(v) in asked for v in FAKE_XWIDTH.values())      # the registry-driven sets
    flat = [n for names in asked for n in names]
    assert "OLMo-1B → 160m (dog), core30, seed-robust 8" in flat
    assert "OLMo-1B → 160m (dog), core30" not in flat
    assert "160m → OLMo-1B (dog), core35" in flat
    assert not any("top10" in n or "(elephant)" in n and n.startswith("OLMo-1B") for n in flat)


def test_same_data_corpus_is_the_three_cross_seed_core35_arms_only(monkeypatch):
    """The 2-teacher core30 reverse arm (and the ctrl20 one) never enter the table; the note
    says why."""
    asked = []

    def fake_registry(path, names, **kw):
        asked.append(list(names))
        return [th.HeadlineArm(n, "elephant", None, float("nan")) for n in names]

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    monkeypatch.setattr(th, "family_headline_arms", lambda *a, **k: [])
    blocks = tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion",
                                                                      "pig")})
    flat = [n for names in asked for n in names]
    assert "seed2 → orig (elephant), core30" not in flat
    assert "orig → seed2 (elephant), ctrl20" not in flat
    same_corpus, = [c for c in blocks if c.condition == "Diff. init, same data corpus"]
    assert [a.label for a in same_corpus.arms] == ["dolma → dolma-seed2 (elephant), core35",
                                                   "dclm → dclm-seed2 (elephant), core35",
                                                   "c4 → c4-seed2 (elephant), core35"]
    assert same_corpus.design == "core35"
    assert "2-teacher core30 arm" in same_corpus.note
    assert any("2-teacher core30 arm" in n for n in tc.condition_notes(blocks))


def test_olmo_block_starts_with_the_same_model_condition(monkeypatch):
    """The OLMo block order mirrors Pythia's: Same-model first, then the corpus/seed ones."""
    def fake_registry(path, names, **kw):
        return [th.HeadlineArm(n, "elephant", None, float("nan")) for n in names]

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    monkeypatch.setattr(th, "family_headline_arms", lambda *a, **k: [])
    blocks = tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion",
                                                                      "pig")})
    olmo = [c.condition for c in blocks if c.family == tc.OLMO_FAMILY]
    assert olmo == ["Same-model", "Same init, diff. data corpus", "Diff. init, diff. data "
                    "corpus", "Diff. init, same data corpus"]
    assert olmo == tc.OLMO_COMPARISON_CONDITIONS
    same, = [c for c in blocks if c.family == tc.OLMO_FAMILY and c.condition == "Same-model"]
    assert [a.label for a in same.arms] == ["dolma → dolma same-model (elephant), core35",
                                            "dclm → dclm same-model (elephant), core35",
                                            "c4 → c4 same-model (elephant), core35"]
    assert same.design == "core35"


def test_olmo_diff_init_diff_corpus_carries_all_six_cross_recipe_seed_arms(monkeypatch):
    """Every ordered recipe pair into a seed2 receiver: the C4 senders included."""
    def fake_registry(path, names, **kw):
        return [th.HeadlineArm(n, "elephant", None, float("nan")) for n in names]

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    monkeypatch.setattr(th, "family_headline_arms", lambda *a, **k: [])
    blocks = tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion",
                                                                      "pig")})
    cross, = [c for c in blocks if c.condition == "Diff. init, diff. data corpus"]
    assert [a.label for a in cross.arms] == ["dclm → seed2 (elephant), core35",
                                             "dclm → c4-seed2 (elephant), core35",
                                             "dolma → dclm-seed2 (elephant), core35",
                                             "dolma → c4-seed2 (elephant), core35",
                                             "c4 → seed2 (elephant), core35",
                                             "c4 → dclm-seed2 (elephant), core35"]
    assert len(cross.arms) == 6


def test_olmo_same_model_arms_read_as_diagonals(monkeypatch):
    """The registry's ``X same-model`` receiver is the diagonal: a bare ``{...}`` sender set."""
    assert tc.arm_endpoints("dolma → dolma same-model (elephant), core35") == ("dolma", "dolma")

    def fake_registry(path, names, **kw):
        return [th.HeadlineArm(n, "elephant", None, float("nan")) for n in names]

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    monkeypatch.setattr(th, "family_headline_arms", lambda *a, **k: [])
    blocks = tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion",
                                                                      "pig")})
    same, = [c for c in blocks if c.family == tc.OLMO_FAMILY and c.condition == "Same-model"]
    assert tc.arms_text(same.arms, same.names) == "{Dolma1.7, DCLM, C4}"


def test_pythia_same_model_reads_every_lineage_diagonal_for_every_animal(monkeypatch):
    """The block asks for the full `SAME_MODEL_BASES` diagonal set, the same for all four
    animals: the finished-set rule, not the arm list, decides which enter."""
    asked = []

    def fake_family(arms, animals, teachers, **kw):
        if not isinstance(arms, dict):                    # the cross-data-order call is per animal
            asked.append(([(a.sender, a.receiver) for a in arms], list(animals)))
        return []

    monkeypatch.setattr(th, "registry_headline_arms", lambda *a, **k: [])
    monkeypatch.setattr(th, "family_headline_arms", fake_family)
    tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion", "pig")})
    diagonals = [(pairs, animals) for pairs, animals in asked
                 if pairs and all(s == r for s, r in pairs)]
    full, = [x for x in diagonals if len(x[0]) == len(tc.SAME_MODEL_BASES)]
    assert [s for s, _ in full[0]] == tc.SAME_MODEL_BASES
    assert full[1] == ["dog", "cat", "lion", "pig"]


def test_same_model_in_two_families_never_collides(monkeypatch):
    """``Same-model`` names a Pythia AND an OLMo block: every keying is (family, condition)."""
    def fake_registry(path, names, **kw):
        return [th.HeadlineArm(n, "elephant", None, float("nan")) for n in names]

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    monkeypatch.setattr(th, "family_headline_arms", lambda *a, **k: [])
    blocks = tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion",
                                                                      "pig")})
    same = [c for c in blocks if c.condition == "Same-model"]
    assert [c.family for c in same] == ["Pythia", tc.OLMO_FAMILY]
    notes = tc.condition_notes(blocks)                    # one bullet per block, family-labelled
    assert len(notes) == len(blocks)
    assert sum(n.startswith("Same-model") for n in notes) == 0
    assert sum(n.startswith(f"{f}, Same-model") for n in notes
               for f in ("Pythia", tc.OLMO_FAMILY)) == 2
    # the OLMo block's arms are read off the OLMo condition, never the Pythia one of that name
    olmo = [tc.Condition(tc.OLMO_FAMILY, name, "core35",
                         _toy_arms(0.0, [f"dolma → dclm ({name})"]), names=tc.OLMO_NAMES)
            for name in tc.OLMO_COMPARISON_CONDITIONS]
    pythia = tc.Condition("Pythia", "Same-model", "core30", _toy_arms(0.0, ["orig → orig (dog)"]))
    arms = tc.olmo_comparison_arms([pythia, *olmo])
    assert list(arms) == tc.OLMO_COMPARISON_CONDITIONS
    assert [a.arm for a in arms["Same-model"]] == ["dolma->dclm"]


def test_condition_columns_put_the_grouping_levels_before_the_per_animal_numbers():
    assert tc.CONDITION_COLUMNS == ["Model Family", "Pretraining Condition", "Arms", "# Arms",
                                    "# Teacher HP", "# Student HP", "# Total HP", "Animal",
                                    "Exceedances", "Null 95th Perc. Exceedances", "P-value"]
    assert tc.CONDITION_LEAD == tc.CONDITION_COLUMNS[:8]


def test_sparsify_blanks_a_level_only_while_every_higher_level_repeats():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["1", "1", "1", "1"]})
    assert tc.sparsify(df, ["a", "b"]) == [["x", "1"], ["", ""], ["y", "1"], ["", ""]]


def test_comparison_hp_cells_merge_the_two_sides_per_arm_counts():
    assert tc.comparison_hp_cells([(10, 30)] * 4, [(10, 30)] * 12) == {
        "# Teacher HP": "10", "# Student HP": "30", "# Total HP A vs B": "1200 vs 3600"}
    assert tc.comparison_hp_cells([(10, 30)], [(11, 56)]) == {
        "# Teacher HP": "10 / 11", "# Student HP": "30 / 56",
        "# Total HP A vs B": "300 vs 616"}


# ------------------------------------------------------------------ the matrix comparison path
def _toy_arms(family_offset: float, labels, animal: str = "elephant") -> list[th.HeadlineArm]:
    """Finished arms whose matrices sit `family_offset` above the `make_matrix` level."""
    out = []
    for i, label in enumerate(labels):
        m = make_matrix(n_cells=3, n_teachers=2, seed=i + 1)
        values = m.values.copy()
        values[:, :-1] += family_offset
        out.append(th.HeadlineArm(label, animal,
                                  tio.CellMatrix(frame=m.frame, values=values, cells=m.cells,
                                                 pools=m.pools, control_col=m.control_col,
                                                 missing=[]), prior=0.01))
    return out


def test_condition_matrix_frame_is_the_gap_and_the_gap_over_the_prior():
    arms = _toy_arms(0.0, ["a → b (elephant), core35"])
    frame = tc.condition_matrix_frame(arms, "A")
    assert frame["family"].unique().tolist() == ["A"]
    assert frame["arm"].unique().tolist() == ["a->b"]
    assert len(frame) == 6                      # 3 cells x 2 teacher pools, control excluded
    assert np.allclose(frame["value_abs"], frame["treatment"] - frame["control"])
    assert np.allclose(frame["value_rel"], frame["value_abs"] / 0.01)
    assert frame["value_recept"].isna().all() and frame["value_port"].isna().all()
    assert (frame["stratum"] == frame["cell"]).all()


def test_compare_condition_matrices_signs_the_contrast_and_gives_a_two_sided_p():
    a = _toy_arms(0.3, ["a1 → b1 (elephant), core35", "a2 → b2 (elephant), core35"])
    b = _toy_arms(0.0, ["c1 → d1 (elephant), core35", "c2 → d2 (elephant), core35"])
    res = tc.compare_condition_matrices(a, b, family_a="A", family_b="B", n_perm=2000)
    assert res.animals == ["elephant"] and list(res.per_animal) == ["elephant"]
    for mode in ("absolute", "relative"):
        m = res.per_animal["elephant"].modes[mode]
        assert m.mean_contrast.observed > 0            # A sits above B by construction
        assert m.win_probability.observed > 0.5
        for p in (m.mean_contrast.p_two_sided, m.win_probability.p_two_sided):
            assert 0 < p <= 1
    # the SAME statistic with the sides swapped flips the sign and keeps the two-sided p
    flipped = tc.compare_condition_matrices(b, a, family_a="B", family_b="A", n_perm=2000)
    lhs = res.per_animal["elephant"].modes["absolute"].mean_contrast
    rhs = flipped.per_animal["elephant"].modes["absolute"].mean_contrast
    assert np.isclose(lhs.observed, -rhs.observed)
    assert np.isclose(lhs.p_two_sided, rhs.p_two_sided)
    # dead runs are never filtered: every run of every matrix is in the frame
    assert len(res.stratified.frame) == 4 * 3 * 2


def test_compare_condition_matrices_needs_both_sides():
    with pytest.raises(ValueError):
        tc.compare_condition_matrices([], _toy_arms(0.0, ["a → b (elephant), core35"]),
                                      family_a="A", family_b="B")


def test_olmo_comparison_pairings_are_ordered_by_the_existence_table_order():
    order = tc.OLMO_COMPARISON_CONDITIONS
    assert order == ["Same-model", "Same init, diff. data corpus",
                     "Diff. init, diff. data corpus", "Diff. init, same data corpus"]
    pairs = tc.comparison_pairings(order)
    assert pairs == [(order[i], order[j]) for i in range(4) for j in range(i + 1, 4)]
    assert len(pairs) == 6
    for a, b in pairs:                       # A is always the EARLIER condition of the block
        assert order.index(a) < order.index(b)


def test_olmo_comparison_arms_are_the_finished_existence_arms():
    conditions = [tc.Condition(tc.OLMO_FAMILY, name, "core35",
                               [*_toy_arms(0.0, [f"dolma → dclm ({name})"]),
                                th.HeadlineArm(f"c4 → dolma ({name})", "elephant", None,
                                               float("nan"), dropped="not started")],
                               names=tc.OLMO_NAMES)
                  for name in tc.OLMO_COMPARISON_CONDITIONS]
    arms = tc.olmo_comparison_arms(conditions)
    assert list(arms) == tc.OLMO_COMPARISON_CONDITIONS
    for name, spec in arms.items():
        assert [a.arm for a in spec] == ["dolma->dclm"]      # the dropped arm never enters
        assert {a.family for a in spec} == {name}


def test_finished_arms_by_condition_never_mixes_two_families():
    """``Same-model`` names a Pythia AND an OLMo block: the family filter keeps them apart."""
    olmo = [tc.Condition(tc.OLMO_FAMILY, name, "core35",
                         [*_toy_arms(0.0, [f"dolma → dclm ({name})"]),
                          th.HeadlineArm(f"c4 → dolma ({name})", "elephant", None,
                                         float("nan"), dropped="not started")],
                         names=tc.OLMO_NAMES)
            for name in tc.OLMO_COMPARISON_CONDITIONS]
    pythia = tc.Condition("Pythia", "Same-model", "core30", _toy_arms(0.0, ["orig → orig (dog)"]))
    finished = tc.finished_arms_by_condition([pythia, *olmo])
    assert list(finished) == tc.OLMO_COMPARISON_CONDITIONS
    # the Pythia Same-model block never reaches the OLMo mapping (and never overwrites it)
    assert [a.label for a in finished["Same-model"]] == ["dolma → dclm (Same-model)"]
    assert all(a.matrix is not None for arms in finished.values() for a in arms)
    assert list(tc.finished_arms_by_condition([pythia, *olmo], family="Pythia")) == ["Same-model"]


def test_comparison_table_drops_the_stratified_row_of_a_single_animal(monkeypatch):
    _stub_family_comparison_table(monkeypatch, ["elephant"])
    a, b = tc.OLMO_COMPARISON_CONDITIONS[0], tc.OLMO_COMPARISON_CONDITIONS[1]
    arms = {a: [tio.FamilyArm(a, "dolma", "dclm")], b: [tio.FamilyArm(b, "dclm", "seed2")]}
    frame = pd.DataFrame([{"family": f, "arm": arm, "teacher": f"t{t}", "cell": f"c{c}"}
                          for f, arm in ((a, "dolma->dclm"), (b, "dclm->seed2"))
                          for t in range(2) for c in range(3)])
    res = _FakeMulti(["elephant"], frame)
    df, _ = tc.comparison_table({(a, b, "core35"): res}, arms, family=tc.OLMO_FAMILY,
                                names=tc.OLMO_NAMES)
    assert df["Animal"].tolist() == ["elephant", "elephant"]
    assert not df["stratified"].any()
    assert df["Model Family"].unique().tolist() == [tc.OLMO_FAMILY]
    assert df["Arms A"].iloc[0] == "Dolma1.7 → DCLM"          # OLMO_NAMES reaches the arm cells
    assert df["Arms B"].iloc[0] == "DCLM → Dolma1.7 seed2"


def test_merge_comparison_tables_normalises_the_retired_written_out_effect():
    """Every absolute row prints the plain Δ (the caption defines it), pickled frames included."""
    first = _comparison_df().assign(**{"Treatment Effect Mode": [tc.EFFECT_LONG,
                                                                 tc.EFFECT_RELATIVE]})
    second = _comparison_df().assign(**{"Model Family": tc.OLMO_FAMILY,
                                        "Treatment Effect Mode": [tc.EFFECT_LONG,
                                                                  tc.EFFECT_RELATIVE]})
    merged = tc.merge_comparison_tables([first, second])
    assert merged["Treatment Effect Mode"].tolist() == [
        tc.EFFECT_SHORT, tc.EFFECT_RELATIVE] * 2
    assert merged["Model Family"].tolist() == ["Pythia"] * 2 + [tc.OLMO_FAMILY] * 2


def test_comparison_latex_rules_a_full_midrule_at_the_model_family_change():
    df = tc.merge_comparison_tables(
        [_comparison_df(), _comparison_df().assign(**{"Model Family": tc.OLMO_FAMILY,
                                                      "Animal": "elephant",
                                                      "stratified": False})])
    tex = tc.comparison_table_latex(df)
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\")]
    for line in body[1:]:
        assert len(line.removesuffix(r"\\").split(" & ")) == 14
    lines = tex.splitlines()
    head = lines.index(r"\midrule")                      # the one under the header
    assert r"\midrule" in lines[head + 1:]               # plus one at the family change
    rows = [i for i, ln in enumerate(lines) if ln.endswith(r"\\")][2:]
    family_rule = lines.index(r"\midrule", head + 1)
    assert rows[1] < family_rule < rows[2]               # exactly between the two families
    assert lines[family_rule - 1].endswith(r"\\")


def test_cross_data_order_carries_the_full_arm_set_per_animal_on_its_own_design(monkeypatch):
    """Every animal gets all twelve cross-data-order arms, dog read on core56 and
    cat / lion / pig on core30 (where their ``ds1 → ds2`` arms exist), plus the core56 read of
    the three moved animals as `Condition.alt_arms`."""
    calls = []

    def fake_family(arms, animals, teachers, *, cells, **kw):
        per_animal = arms if isinstance(arms, dict) else {a: arms for a in animals}
        labels = sorted({f"{x.sender}→{x.receiver}" for a in animals for x in per_animal[a]})
        calls.append((list(animals), Path(cells).name, labels))
        return []

    monkeypatch.setattr(th, "family_headline_arms", fake_family)
    monkeypatch.setattr(th, "registry_headline_arms", lambda *a, **k: [])
    blocks = tc.condition_blocks(Path("/nowhere"), {a: ["t"] for a in ("dog", "cat", "lion",
                                                                      "pig")})
    # the same-model block asks for the ds DIAGONALS too: the cross-data-order calls are the
    # off-diagonal ones
    ds = [(animals, design, labels) for animals, design, labels in calls
          if any("ds" in x for x in labels)
          and any(x.split("→")[0] != x.split("→")[1] for x in labels)]
    assert all(len(labels) == 12 for _, _, labels in ds)        # the FULL set for every animal
    assert [(animals, design) for animals, design, _ in ds] == [
        (["dog"], "core56_cells.json"), (["cat"], "core30_cells.json"),
        (["lion"], "core30_cells.json"), (["pig"], "core30_cells.json"),
        (["cat", "lion", "pig"], "core56_cells.json")]          # the alt (core56) read
    cdo, = [c for c in blocks if c.condition == tc.COMPARISON_CONDITIONS[1]]
    assert cdo.design == "core56 (dog) / core30 (cat, lion, pig)"
    assert list(cdo.alt_arms) == ["core56"]
    assert "core30" in cdo.note and "ds1 → ds2" in cdo.note


def test_unfinished_blocks_excludes_a_block_only_on_the_design_it_is_unfinished_on():
    """cat ``ds1 → ds2`` is finished on core30 and part-run on core56, so it enters the core30
    comparisons and not the core56 repeat; an arm never READ on a design (the cross-seed arms on
    core30) is kept there as before."""
    cdo, cross = tc.COMPARISON_CONDITIONS[1], tc.COMPARISON_CONDITIONS[3]
    conds = [
        tc.Condition("Pythia", cdo, "core56 (dog) / core30 (cat, lion, pig)",
                     [arm("ds1→ds2", "cat", n_cells=30)],
                     alt_arms={"core56": [th.HeadlineArm(
                         "ds1→ds2", "cat", None, float("nan"),
                         dropped="students 300/560, controls 56/56")]}),
        tc.Condition("Pythia", cross, "core56", [arm("orig→fs1", "cat", n_cells=56)]),
    ]
    arms = {cdo: [tio.FamilyArm(cdo, "ds1", "ds2")],
            cross: [tio.FamilyArm(cross, "orig", "fs1")]}
    moved, seed = ("cat", "ds1->ds2"), ("cat", "orig->fs1")
    assert moved not in tc.unfinished_blocks(conds, arms, ["cat"])
    assert moved not in tc.unfinished_blocks(conds, arms, ["cat"], design="core30")
    assert moved in tc.unfinished_blocks(conds, arms, ["cat"], design="core56")
    # read on core56 only: the core30 comparisons keep it (the design-agnostic rule)
    assert all(seed not in tc.unfinished_blocks(conds, arms, ["cat"], design=d)
               for d in (None, "core30", "core56"))


def test_a_diverged_run_is_a_complete_run_excluded_from_the_table_and_the_comparison():
    """A finished arm carrying one diverged (NaN, marked) run is tested on n-1 runs -- its
    ``# Total HP`` and both permutation denominators -- with the run named in the notes, and
    contributes no row to the matrix comparison path either."""
    m = make_matrix(n_cells=3, n_teachers=2)
    values = m.values.copy()
    values[2, 1] = np.nan                       # teacher t1 diverged on cell c2
    diverged = tio.CellMatrix(frame=m.frame, values=values, cells=m.cells, pools=m.pools,
                              control_col=m.control_col, missing=[("t1", "c2")])
    note = th.diverged_note([("t1", "c2")])
    assert note == "1 diverged run excluded (teacher t1, cell c2)"
    cond = tc.Condition("Pythia", "Same-model", "core30",
                        [th.HeadlineArm("orig→orig", "dog", diverged, prior=0.01, note=note,
                                        diverged=(("t1", "c2"),))])
    df = tc.condition_table([cond], n_perm=200)
    assert df["# Total HP"].tolist() == ["5"]   # 3 cells x 2 teachers, the diverged run out
    assert tc.count_pct(5, 5) == df["Exceedances"].iloc[0]      # all 5 tested runs exceed
    assert any(note in n for n in tc.condition_notes([cond]))
    frame = tc.condition_matrix_frame([th.HeadlineArm("a → b (dog), core30", "dog", diverged,
                                                      prior=0.01)], "A")
    assert len(frame) == 5 and frame["value_abs"].notna().all()


def test_the_comparison_caption_is_the_relative_strength_paragraph():
    tex = tc.comparison_table_latex(_comparison_df())
    assert (r"\caption{\textbf{Relative Strength of transfer between pretraining conditions.} "
            r"We compare pretraining conditions") in tex
    assert "the sign of the Difference in Means" not in tex
    assert r"computing Difference in Means and Win Probability statistics on both " \
           r"($\Delta=$p(student) - p(control)) and $\Delta/$prior. Bold = two-sided $p < 0.05$}" in tex
    assert r"Bold = two-sided $p < 0.05$}" in tex             # no trailing period, as given
    assert r"\label{tab:transfer_comparison}" in tex
    assert "arm-relabelling null within" not in tex           # the old legend is gone


def test_render_script_round_trips_a_pickled_frame(tmp_path):
    """The cosmetic path: pickles in, the same .tex out, no run tree touched."""
    from scripts import render_transfer_condition_tables as rt

    _latex_df().to_pickle(tmp_path / f"{rt.EXISTENCE}.pkl")
    comparison = _comparison_df().assign(**{"Treatment Effect Mode": [tc.EFFECT_LONG,
                                                                     tc.EFFECT_RELATIVE]})
    comparison.to_pickle(tmp_path / f"{rt.COMPARISON}.pkl")
    direction = pd.DataFrame([{**dict.fromkeys(tc.DIRECTION_COLUMNS, "x"),
                               "Treatment Effect Mode": "Exceed. frac.", "Diff. in Means": 0.1,
                               "P-value": 0.01, "Paired P-value": tc.NO_PAIRED}])
    direction.to_pickle(tmp_path / f"{rt.DIRECTION}.pkl")
    rt.main(latex_dir=tmp_path)
    assert (tmp_path / f"{rt.DIRECTION}.tex").read_text() == tc.direction_table_latex(direction)
    assert (tmp_path / f"{rt.EXISTENCE}.tex").read_text() == tc.condition_table_latex(_latex_df())
    rendered = (tmp_path / f"{rt.COMPARISON}.tex").read_text()
    assert rendered == tc.comparison_table_latex(_comparison_df())   # EFFECT_LONG normalised
    widest = rt.widest_cells(_latex_df(), rt.WIDTH_COLUMNS[rt.EXISTENCE])
    assert widest["Model Family"] == ("DataDecide", len("DataDecide"))
    assert all(w <= tc._ARMS_WIDTH for _, w in [widest["Arms"]])


def test_render_script_exits_when_a_pickle_is_missing(tmp_path):
    from scripts import render_transfer_condition_tables as rt

    with pytest.raises(rt.typer.Exit):
        rt.main(latex_dir=tmp_path)


def test_set_notation_folds_olmo_seeds_with_the_default_as_seed1():
    assert tc._set_notation(["Dolma1.7", "Dolma1.7 seed2", "Dolma1.7 seed3"]) == [
        "Dolma1.7 seed1–3"]
    assert tc._set_notation(["Dolma1.7", "Dolma1.7 seed2"]) == ["Dolma1.7 seed1,2"]
    assert tc._set_notation(["DCLM", "DCLM seed2"]) == ["DCLM seed1,2"]
    assert tc._set_notation(["Dolma1.7"]) == ["Dolma1.7"]          # a lone default stays
    assert tc.compress_arms([("Dolma1.7", "Dolma1.7")]) == "Dolma1.7"
    grid = [(s, r) for s in ("160M", "fs1", "fs3")
            for r in ("Dolma1.7", "Dolma1.7 seed2", "Dolma1.7 seed3")]
    assert tc.compress_arms(grid) == "{160M, fs1,3} → Dolma1.7 seed1–3"


def test_set_notation_folds_pythia_seed_families_at_two_unlabelled_model_separate():
    assert tc._set_notation(["410M-seed1", "410M-seed3"]) == ["410M-seed1,3"]
    assert tc._set_notation(["410M", "410M-seed1", "410M-seed3"]) == ["410M", "410M-seed1,3"]
    assert tc._set_notation(["fs1", "fs2"]) == ["fs1,2"]
    assert tc._set_text(["160M", "fs1", "fs3"]) == "{160M, fs1,3}"
    assert tc._set_text(["410M-seed1", "410M-seed2", "410M-seed3"]) == "{410M-seed1–3}"
    # families that are not seed-labelled keep the three-member rule
    assert tc._set_notation(["wc1", "wc2"]) == ["wc1", "wc2"]
    assert tc._set_notation(["ds1", "ds2", "ds3"]) == ["ds1–3"]


def test_olmo_cross_recipe_block_factors_the_receiver_seed():
    corpora = ["DCLM", "Dolma1.7", "C4"]
    full = [(a, f"{b} seed2") for a in corpora for b in corpora if a != b]
    assert tc.compress_arms(full) == "{DCLM, Dolma1.7, C4}² → seed2"
    partial = [(a, f"{b} seed2") for a in ("Dolma1.7", "DCLM", "C4") for b in ("Dolma1.7", "DCLM")]
    assert tc.compress_arms(partial) == "{Dolma1.7, DCLM, C4} → {Dolma1.7, DCLM} seed2"
    assert tc._tex(tc.compress_arms(full)) == r"\{DCLM, Dolma1.7, C4\}$^{2}$ $\to$ seed2"
    assert "seed k" in tc.SEED_FACTOR_NOTE


def test_same_corpus_olmo_arm_drops_the_receiver_corpus():
    assert tc.compress_arms([("Dolma1.7", "Dolma1.7 seed2")]) == "Dolma1.7 → seed2"
    assert tc.compress_arms([("Dolma1.7", "Dolma1.7 seed2"),
                             ("Dolma1.7", "Dolma1.7 seed3")]) == "Dolma1.7 → seed2,3"
    assert tc.compress_arms([("Dolma1.7 seed2", "Dolma1.7")]) == "Dolma1.7 seed2 → Dolma1.7"


def test_fit_lines_wraps_conditions_to_two_words_within_the_block_height():
    assert tc._fit_lines(["Diff. init", "Diff. data order"], 4) == [
        "Diff. init", "Diff. data", "order"]
    # a block of two rows never grows past the existing comma break
    assert tc._fit_lines(["Diff. init", "Diff. data order"], 1) == [
        "Diff. init", "Diff. data order"]
    assert tc._total_parts("3360 vs 2160") == ["3360", "vs 2160"]


def test_same_corpus_olmo_arms_to_one_seed_fold_into_one_set():
    arms = [("Dolma1.7", "Dolma1.7 seed2"), ("DCLM", "DCLM seed2"), ("C4", "C4 seed2")]
    assert tc.compress_arms(arms) == "{Dolma1.7, DCLM, C4} → seed2"
    assert tc.seed_factor_notes(["{Dolma1.7, DCLM, C4} → seed2"]) == [tc.SEED_FACTOR_NOTE]
    assert "each corpus of S to its own seed-k model" in tc.SEED_FACTOR_NOTE


def test_two_seeds_are_a_comma_list_three_a_range():
    assert tc._set_notation(["fs2", "fs3"]) == ["fs2,3"]
    assert tc._set_notation(["410M-seed2", "410M-seed3"]) == ["410M-seed2,3"]
    assert tc._set_notation(["fs1", "fs2", "fs3"]) == ["fs1–3"]
    assert tc._index_runs([1, 2, 4]) == "1,2,4"


def test_split_set_never_cuts_folded_seed_list():
    from subliminal_learning.analysis.transfer_conditions import _split_set

    assert _split_set("{410M, 410M-seed2,3} →") == ["{410M,", "410M-seed2,3} →"]
    parts = _split_set("{160M, fs1,2, fs3, rerun 160M}")
    assert all(not p.endswith("fs1,") for p in parts) and " ".join(parts).count("fs1,2") == 1
    assert _split_set("{160M, fs1–3, ds1, ds2, rerun 160M}") == ["{160M, fs1–3, ds1,", "ds2, rerun 160M}"]
