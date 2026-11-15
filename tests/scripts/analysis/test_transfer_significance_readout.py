"""`scripts.analysis.transfer_significance_readout` CLI modes on synthetic trees / monkeypatched matrices.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner
import numpy as np
import pandas as pd
import pytest

from scripts.analysis.transfer_significance_readout import app
from subliminal_learning.analysis.loaders import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt
from tests._helpers import (
    _family_tree,
)
from tests._helpers import _write_base_eval_prior, _write_seed_eval, _write_summary
from tests._helpers import patch_tio


# -------------------- from cli_a.py --------------------

# ----------------------------------------------------------------------------------------------
# CLI: the pooled line's design, and mode-scoped options
# ----------------------------------------------------------------------------------------------
def _cli_matrix() -> tio.CellMatrix:
    """Two pooled teachers over 10 cells, with ONE cell missing for the first teacher."""
    control = np.linspace(1.0, 2.0, 10)
    t1 = control + 5.0
    t1[3] = np.nan                            # the missing cell the two designs must agree on
    t2 = control + 0.5
    values = np.column_stack([t1, t2, control])
    return tio.CellMatrix(frame=pd.DataFrame(), values=values,
                          cells=[f"c{i}" for i in range(10)],
                          pools=["t1", "t2", tio.CONTROL_LABEL], control_col=2, missing=[])

def _run_cli(monkeypatch, args, matrix=None, teachers=("t1", "t2")):
    from typer.testing import CliRunner

    import scripts.analysis.transfer_significance_readout as cli

    monkeypatch.setattr(cli.tio, "resolve_teachers",
                        lambda *a, **k: (list(teachers), []))
    monkeypatch.setattr(cli.tio, "load_prior", lambda *a, **k: 0.05)
    monkeypatch.setattr(cli.tio, "load_cell_matrix",
                        lambda *a, **k: matrix if matrix is not None else _cli_matrix())
    return CliRunner().invoke(cli.app, args)

def test_cli_pooled_line_matches_the_table_pooled_row(monkeypatch):
    """(6) one missing cell: the CLI line and the notebook table's pooled row agree."""
    m = _cli_matrix()
    res = _run_cli(monkeypatch, ["--eval-animals", "dog", "--n-perm", "200"], matrix=m)
    assert res.exit_code == 0, res.output
    table = tio.significance_rows(m, np.zeros_like(m.values), 0.05, "A", ["t1", "t2"],
                                  n_perm=200)[0]
    assert table["pooled"]
    line = [ln for ln in res.output.splitlines()
            if ln.startswith("pooled stratified permutation")][0]
    assert f"observed {table['exceed.']} exceedances over 18 treatment cells" in line
    assert f"p {table['perm p']:.1e}" in line
    assert "9 of 10 cells kept, 1 dropped by the filter" in line
    # the per-teacher table has no closed-form exceedance p column any more
    assert "exc p" not in res.output

@pytest.mark.parametrize("args", [
    ["--dead", "drop"],
    ["--family", "full-seed", "--dead", "drop"],
])
def test_cli_rejects_compare_only_options_in_other_modes(monkeypatch, args):
    """(7) a mode that ignores --dead must refuse it, not run on (--ftseed is read in all)."""
    res = _run_cli(monkeypatch, [*args, "--eval-animals", "dog"])
    assert res.exit_code != 0
    assert "compare-mode only" in res.output

@pytest.mark.parametrize("mode_args", [[], ["--family", "full-seed"],
                                       ["--compare", "data-seed:full-seed"]])
def test_cli_zero_teachers_is_a_non_zero_exit(monkeypatch, mode_args):
    """(7) never a result computed on zero teachers, in any mode."""
    res = _run_cli(monkeypatch, [*mode_args, "--eval-animals", "dog"], teachers=())
    assert res.exit_code != 0
    assert "zero teachers" in res.output

@pytest.mark.parametrize("mode_args", [["--family", "full-seed"],
                                       ["--compare", "data-seed:full-seed"]])
def test_cli_all_on_disk_outside_single_arm_mode_exits(monkeypatch, tmp_path, mode_args):
    """(7) 'all-on-disk' scans one sender tree; family/compare mode fixes no sender, so it
    resolves to zero teachers -- which must be an error, not an empty result."""
    import scripts.analysis.transfer_significance_readout as cli

    # the real resolver, on an empty output root: the all-on-disk sentinel finds nothing
    assert tio.resolve_teachers(tio.ALL_ON_DISK, output_root=tmp_path) == ([], [])
    from typer.testing import CliRunner

    monkeypatch.setattr(cli.tio, "load_prior", lambda *a, **k: 0.05)
    monkeypatch.setattr(cli.tio, "load_cell_matrix", lambda *a, **k: _cli_matrix())
    monkeypatch.setattr(cli.tio, "resolve_teachers", lambda *a, **k: ([], []))
    res = CliRunner().invoke(cli.app, [*mode_args, "--teachers", tio.ALL_ON_DISK])
    assert res.exit_code != 0
    assert "all-on-disk" in res.output

# ----------------------------------------------------------------------------------------------
# the CLI reads ONE finetune seed, the same one the notebook tables pin
# ----------------------------------------------------------------------------------------------
def _two_ftseed_tree(root, *, cells=("c1", "c2")) -> None:
    """One arm whose cells each carry two finetune seeds and a seed-aggregated summary."""
    ds = "ex10-20_prefix_n100k"
    for cell in cells:
        leaf = (root / "students/dog/pythia-160m/teacher-pythia-160m-seed1"
                / "t1_tseed1" / ds / cell)
        _write_summary(leaf / "preference_summary.json", "test", 0.9)   # the seed MEAN
        _write_seed_eval(leaf / "ftseed1/preference_eval.json", 0.5)
        _write_seed_eval(leaf / "ftseed2/preference_eval.json", 1.3)
        ctrl = (root / "students/controls/base/pythia-160m/teacher-pythia-160m-seed1"
                / ds / cell)
        _write_summary(ctrl / "eval/dog/preference_summary.json", "test", 0.1)
        _write_seed_eval(ctrl / "ftseed1/eval/dog/preference_eval.json", 0.1)
        _write_seed_eval(ctrl / "ftseed2/eval/dog/preference_eval.json", 0.1)
    _write_base_eval_prior(root, "original", "dog", 0.1)

def _cli_on_tree(tmp_path, monkeypatch, args):
    from typer.testing import CliRunner

    import scripts.analysis.transfer_significance_readout as cli
    from subliminal_learning import config

    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    return CliRunner().invoke(cli.app, args)

@pytest.mark.parametrize("mode_args, seed_value", [([], 0.5), (["--ftseed", "2"], 1.3)])
def test_cli_single_arm_reads_the_asked_for_ftseed_not_the_aggregate(tmp_path, monkeypatch,
                                                                    mode_args, seed_value):
    """The CLI's treatment value is the table's: one pinned seed, default 1, never the mean."""
    _two_ftseed_tree(tmp_path)
    res = _cli_on_tree(tmp_path, monkeypatch, ["--animal", "dog", "--sender", "fs1",
                                               "--receiver", "original", "--teachers", "t1",
                                               "--eval-animals", "dog", "--n-perm", "200",
                                               *mode_args])
    assert res.exit_code == 0, res.output
    ftseed = int(mode_args[1]) if mode_args else 1
    assert f"ftseed {ftseed} ===" in res.output
    matrix = tio.load_cell_matrix("dog", "fs1", "original", ["t1"], ftseed=ftseed,
                                  output_root=tmp_path)
    assert matrix.values[:, 0].tolist() == pytest.approx([seed_value, seed_value])
    expected = tt.prior_scaled_effect(matrix.values[:, 0], matrix.values[:, matrix.control_col],
                                      0.1)
    line = [ln for ln in res.output.splitlines() if ln.strip().startswith("t1 ")][0]
    assert float(line.split()[-1]) == pytest.approx(expected, abs=5e-5)
    # the seed-aggregated read (0.9) would have given a different number
    assert expected != pytest.approx(tt.prior_scaled_effect(np.full(2, 0.9), np.full(2, 0.1),
                                                            0.1))

def test_cli_family_mode_also_pins_the_ftseed(tmp_path, monkeypatch):
    calls = []
    import scripts.analysis.transfer_significance_readout as cli

    real = tio.load_cell_matrix
    monkeypatch.setattr(cli.tio, "load_cell_matrix",
                        lambda *a, **k: (calls.append(k.get("ftseed")), real(*a, **k))[1])
    _two_ftseed_tree(tmp_path)
    res = _cli_on_tree(tmp_path, monkeypatch, ["--family", "cross-seed", "--animal", "dog",
                                               "--teachers", "t1", "--n-perm", "200"])
    assert res.exit_code == 0, res.output
    assert calls and set(calls) == {1}

# ----------------------------------------------------------------------------------------------
# empty pooled designs: an explicit "insufficient data" outcome, never a zero-size reduction
# ----------------------------------------------------------------------------------------------
def _disjoint_matrix() -> tio.CellMatrix:
    """Two pooled teachers covering DISJOINT cells, so no cell is all-finite."""
    control = np.linspace(1.0, 2.0, 6)
    t1, t2 = control + 5.0, control + 5.0
    t1[3:], t2[:3] = np.nan, np.nan
    return tio.CellMatrix(frame=pd.DataFrame(), values=np.column_stack([t1, t2, control]),
                          cells=[f"c{i}" for i in range(6)],
                          pools=["t1", "t2", tio.CONTROL_LABEL], control_col=2, missing=[])

def test_cli_pooled_line_says_insufficient_data_on_an_empty_design(monkeypatch):
    res = _run_cli(monkeypatch, ["--eval-animals", "dog", "--n-perm", "200"],
                   matrix=_disjoint_matrix())
    assert res.exit_code == 0, res.output
    line = [ln for ln in res.output.splitlines()
            if ln.startswith("pooled stratified permutation")][0]
    assert "insufficient data" in line and "nothing to test" in line
    assert "p 1.0e+00" not in res.output


# -------------------- from cli_b.py --------------------

def test_cli_compare_mode_runs(tmp_path, monkeypatch):
    from subliminal_learning import config

    _family_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--animal", "dog",
                                      "--teachers", str(manifest), "--n-perm", "500",
                                      "--cells", "c1,c2"])
    assert result.exit_code == 0, result.output
    assert "compare data-seed (A) vs full-seed (B)" in result.output
    assert "dead keep ===" in result.output and "role" not in result.output
    assert "design: cell spec inline list (2 cells) | ftseed 1" in result.output
    assert "value mode absolute" in result.output and "value mode relative" in result.output
    assert result.output.count(
        "arm-relabel permutation [mean contrast]: 12 vs 12 arms over 2 cells") == 2
    assert result.output.count("arm-relabel permutation [win probability]") == 2
    assert result.output.count("arm-level permutation [exceedance fraction]") == 2
    assert "cell-stratified permutation" not in result.output
    assert "scale-free, so identical in both value modes" in result.output
    assert "orig->ds1" in result.output

def test_cli_compare_mode_reports_a_missing_anchor_as_unlabelled(tmp_path, monkeypatch):
    """The CLI inherits the rule: dropping raises, keeping prints the unlabelled marker."""
    from subliminal_learning import config

    _family_tree(tmp_path, anchor=False)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    args = ["--compare", "data-seed:full-seed", "--animal", "dog", "--teachers", "t1,t2",
            "--n-perm", "200", "--cells", "c1,c2"]
    dropping = CliRunner().invoke(app, [*args, "--dead", "drop"])
    assert dropping.exit_code != 0
    assert "no anchored pp1 cut" in str(dropping.exception)

    kept = CliRunner().invoke(app, args)
    assert kept.exit_code == 0, kept.output
    assert "note: " in kept.output and "no anchored pp1 cut" in kept.output
    assert (f"dead student runs (kept as measured): {tio.DEAD_UNLABELLED} treatment, "
            f"{tio.DEAD_UNLABELLED} control") in kept.output

def _ftseed_spy(monkeypatch) -> list:
    """Record the ``ftseed`` every `load_cell_matrix` call is made at, and keep loading."""
    seen, real = [], tio.load_cell_matrix

    def spy(*a, **kw):
        seen.append(kw.get("ftseed"))
        return real(*a, **kw)

    patch_tio(monkeypatch, "load_cell_matrix", spy)
    return seen

@pytest.mark.parametrize("mode_args", [
    ["--sender", "ds1", "--receiver", "orig", "--eval-animals", "dog"],   # single-arm mode
    ["--family", "data-seed"],                                           # family mode
])
def test_cli_reads_every_mode_at_the_ftseed_the_tables_pin(tmp_path, monkeypatch, mode_args):
    """Single-arm and family mode must read ``ftseed{N}/preference_eval.json``, not the
    seed-aggregated summary: otherwise the CLI and the notebook tables are different designs."""
    from subliminal_learning import config

    _family_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    args = [*mode_args, "--animal", "dog", "--teachers", "t1,t2", "--n-perm", "200",
            "--cells", "c1,c2"]

    seen = _ftseed_spy(monkeypatch)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen and set(seen) == {1}            # the tables' default seed, not None

    seen.clear()
    CliRunner().invoke(app, [*args, "--ftseed", "3"])
    assert seen and set(seen) == {3}            # ... and the seed the caller asked for

def test_cli_rejects_an_unknown_dead_policy(tmp_path):
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--dead", "maybe"])
    assert result.exit_code != 0

def test_cli_compare_rejects_an_unknown_family(tmp_path):
    result = CliRunner().invoke(app, ["--compare", "data-seed:weight-seed"])
    assert result.exit_code != 0

# ----------------------------------------------------------------------------------------------
# several animals: the arm is one unit, the stratum is (animal, cell)
# ----------------------------------------------------------------------------------------------
def _two_animal_tree(tmp_path, *, drop_cat_arms=()):
    """The dog tree, plus a cat copy of every arm outside `drop_cat_arms`."""
    _family_tree(tmp_path)
    _copy_animal(tmp_path, "cat", drop_arms=drop_cat_arms)

def _copy_animal(tmp_path, animal, *, drop_arms=()):
    """Copy the dog tree into `animal`, leaving out every arm in `drop_arms`."""
    for path in sorted(tmp_path.glob("students/dog/**/preference_eval.json")):
        rel = path.relative_to(tmp_path / "students/dog")
        arm_sender = rel.parts[1].removeprefix("teacher-")
        if any(tio.base_name(a.split("->")[0]) == arm_sender and
               tio.base_name(a.split("->")[1]) == rel.parts[0] for a in drop_arms):
            continue
        out = tmp_path / "students" / animal / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(path.read_text())
        probe = path.parent.parent / "eval/pile_probe.json"
        if probe.is_file():
            (out.parent.parent / "eval").mkdir(parents=True, exist_ok=True)
            (out.parent.parent / "eval/pile_probe.json").write_text(probe.read_text())
    for summary in sorted(tmp_path.glob("students/dog/**/preference_summary.json")):
        out = tmp_path / "students" / animal / summary.relative_to(tmp_path / "students/dog")
        if out.parent.is_dir():
            out.write_text(summary.read_text())
    # the controls are animal-namespaced; add this animal's namespace next to the dog one
    for ctrl in sorted(tmp_path.glob("students/controls/base/**/eval/dog/preference_eval.json")):
        out = ctrl.parent.parent / animal / "preference_eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ctrl.read_text().replace('"dog"', f'"{animal}"'))
    for prior in sorted(tmp_path.glob("analysis/base_eval_*_animal_preference_test.json")):
        doc = json.loads(prior.read_text())
        doc[animal] = doc["dog"]
        prior.write_text(json.dumps(doc))

def test_cli_compare_mode_runs_over_several_animals(tmp_path, monkeypatch):
    from subliminal_learning import config

    _two_animal_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    result = CliRunner().invoke(app, ["--compare", "cross-data-order:cross-seed",
                                      "--animal", "dog,cat", "--teachers", str(manifest),
                                      "--n-perm", "300", "--cells", "c1,c2"])
    assert result.exit_code == 0, result.output
    assert "### animal dog ###" in result.output and "### animal cat ###" in result.output
    assert "### dog+cat stratified ###" in result.output
    # the stratified block keeps the arm as the unit: 12 vs 12, not 24 vs 24
    assert result.output.count("arm-relabel permutation [mean contrast]: 12 vs 12 arms") == 6

def _eval_animal_spy(monkeypatch) -> list:
    """Record the ``(animal, eval_animal)`` of every `load_cell_matrix` call, and keep loading."""
    seen, real = [], tio.load_cell_matrix

    def spy(*a, **kw):
        seen.append((a[0], kw.get("eval_animal")))
        return real(*a, **kw)

    patch_tio(monkeypatch, "load_cell_matrix", spy)
    return seen

def test_cli_family_mode_scores_each_animal_with_its_own_eval_animal(tmp_path, monkeypatch):
    """`--animal dog,cat` in family mode is one block per animal, each scoring ITS OWN animal --
    never the joined string 'dog,cat' as an eval animal."""
    from subliminal_learning import config

    _two_animal_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    args = ["--family", "cross-data-order", "--animal", "dog,cat", "--teachers", "t1,t2",
            "--n-perm", "200", "--cells", "c1,c2"]

    seen = _eval_animal_spy(monkeypatch)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert set(seen) == {("dog", "dog"), ("cat", "cat")}
    assert "eval animal dog,cat" not in result.output

    seen.clear()
    assert CliRunner().invoke(app, [*args, "--eval-animal", "dog"]).exit_code == 0
    assert set(seen) == {("dog", "dog"), ("cat", "dog")}    # the override wins in every block

def test_cli_single_arm_mode_rejects_an_animal_list(tmp_path, monkeypatch):
    """Single-arm mode fixes one sender tree, so a comma list has no meaning: say so."""
    from subliminal_learning import config

    _two_animal_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    result = CliRunner().invoke(app, ["--animal", "dog,cat", "--sender", "ds1",
                                      "--receiver", "orig", "--teachers", "t1,t2"])
    assert result.exit_code != 0
    assert "compare and family mode only" in result.output


def test_cli_teacher_dataset_tags_reach_the_loader(monkeypatch, tmp_path):
    """`--teacher-dataset-tags t1:X` and a manifest `dataset_tags` key both become the loader's
    per-teacher override; controls are untouched (the loader keeps its default)."""
    import scripts.analysis.transfer_significance_readout as cli

    seen = []
    monkeypatch.setattr(cli.tio, "resolve_teachers", lambda *a, **k: (["t1", "t2"], []))
    monkeypatch.setattr(cli.tio, "load_prior", lambda *a, **k: 0.05)
    monkeypatch.setattr(cli.tio, "load_cell_matrix",
                        lambda *a, **k: seen.append(k.get("teacher_dataset_tags")) or _cli_matrix())
    base = ["--animal", "blue", "--sender", "s", "--receiver", "r", "--eval-animals", "blue", "--n-perm", "50"]
    r = CliRunner().invoke(cli.app, base + ["--teachers", "t1,t2", "--teacher-dataset-tags", "t1:ex10-20_prefix_n200k"])
    assert r.exit_code == 0, r.output
    assert seen[-1] == {"t1": "ex10-20_prefix_n200k"}
    man = tmp_path / "teachers.json"
    man.write_text(json.dumps({"param_tags": ["t1", "t2"], "dataset_tags": {"t2": "ex10-20_prefix_n200k"}}))
    r = CliRunner().invoke(cli.app, base + ["--teachers", str(man), "--teacher-dataset-tags", "t1:ex10-20_prefix_n125k"])
    assert r.exit_code == 0, r.output
    assert seen[-1] == {"t2": "ex10-20_prefix_n200k", "t1": "ex10-20_prefix_n125k"}
    r = CliRunner().invoke(cli.app, base + ["--teachers", "t1,t2"])
    assert r.exit_code == 0 and seen[-1] is None
    r = CliRunner().invoke(cli.app, base + ["--teachers", "t1,t2", "--teacher-dataset-tags", "t1"])
    assert r.exit_code != 0
