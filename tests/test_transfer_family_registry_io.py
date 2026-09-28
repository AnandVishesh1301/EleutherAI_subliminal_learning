"""Registry semantics, held-out designs, dataset overrides, and Figure 2 screening."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_family_registry_io as reg
from subliminal_learning.analysis import transfer_significance_io as tio


def test_actual_registry_categories_cover_the_declared_comparisons():
    repo = Path(__file__).resolve().parents[1]
    all_arms = [a for p in reg._registry_paths(repo) for a in json.loads(p.read_text())["arms"]]
    cats = [reg._category(a) for a in all_arms]
    assert set(reg.EXTRA_RELATIONS).issubset({c for c in cats if c is not None})
    ignored = [a for a, c in zip(all_arms, cats) if c is None]
    assert ignored and all(a["sender"] == a["receiver"] for a in ignored)
    assert reg.RELATION_OLMO_SAME not in cats
    assert reg._category({"sender": "dd-1b", "receiver": "dd-1b", "data_order": "same"}, include_olmo_same_model=True) == reg.RELATION_OLMO_SAME
    for a in all_arms:
        category = reg._category(a)
        if a['data_order'] == 'cross-recipe+seed':
            assert category == reg.EXTRA_RELATIONS[7]
        if a['data_order'] == 'cross-recipe':
            assert category == reg.EXTRA_RELATIONS[6]
    assert reg._category({'sender':'dd-1b', 'receiver':'dd-1b-seed2', 'data_order':'same'}) == reg.EXTRA_RELATIONS[8]
    assert reg._category({'sender':'dd-1b', 'receiver':'dd-1b', 'data_order':'same'}) is None
    assert reg._category({'sender':'pythia-160m', 'receiver':'DataDecide-dolma1_7-1B-seed-default'}) == reg.EXTRA_RELATIONS[4]


def _fixture(tmp_path, monkeypatch, *, cut=.5, untestable=False, incomplete_endpoint=False, include_held=False):
    arm = {'name':'reverse', 'animal':'dog', 'sender':'pythia-410m-seed1',
           'receiver':'pythia-160m', 'data_order':'different', 'tseed':3,
           'teachers':['t1','t2'], 'cells':['c1','held','c2','c3'], 'exclude_cells':['held'],
           'teacher_dataset_tags':{'t2':'numbers125k'}, 'control_dataset_tag':'controls100k',
           'selection_note':'selected on held-out pilot; this does NOT exclude the arm'}
    if untestable:
        arm['untestable'] = 'single-cell ordering pilot'
    paths = reg._registry_paths(tmp_path)
    for p, arms in zip(paths, [[arm],[]]):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'arms':arms, 'ftseed':7, 'prompt_set':'test'}))
    calls = []
    def matrix(animal, sender, receiver, teachers, **kw):
        calls.append(('matrix',kw))
        assert kw['cells'] == (['c1','c2','c3','held'] if include_held else ['c1','c2','c3'])
        return tio.CellMatrix(frame=pd.DataFrame(), cells=kw['cells'], pools=[*teachers,'control'],
                              values=np.array([[.6,.7,.1],[.4,np.nan,.1],[.3,.4,np.nan]] + ([[.8,.9,.1]] if include_held else [])),
                              control_col=2,missing=[])
    def dead(matrix, control_tree, sender_tree, **kw):
        calls.append(('dead',kw))
        assert 'controls100k' in str(control_tree)
        return (np.full(matrix.values.shape,np.nan) if kw['cut'] is None else
                np.array([[0,1,np.nan],[0,np.nan,1],[0,0,np.nan]] + ([[0,0,0]] if include_held else [])))
    def endpoint(path, seed, prompt):
        assert seed == 7 and prompt == 'test'
        if '/t2_tseed3/' in str(path):
            assert '/numbers125k/' in str(path)
        return not (incomplete_endpoint and '/t1_tseed3/' in str(path) and path.name == 'c1')
    monkeypatch.setattr(tio,'load_cell_matrix',matrix)
    monkeypatch.setattr(tio,'_arm_dead_matrix',dead)
    monkeypatch.setattr(tio,'_has_seed_endpoint',endpoint)
    monkeypatch.setattr(tio,'_probe_pool_for',lambda *a:'test-pool')
    monkeypatch.setattr(tio,'_anchored_cut',lambda *a,**kw:cut)
    return calls


def test_selection_notes_do_not_exclude_valid_arm_and_filters_preserve_unknown_control(tmp_path,monkeypatch):
    calls = _fixture(tmp_path,monkeypatch)
    pairs,coverage = reg.registry_pairs(tmp_path,tmp_path/'outputs')
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row.teacher == 't1' and row.student_tag == 'c1'
    assert not row.ctrl_probed and row.receiver_filtered and row.alive
    assert row.tseed == 3 and row.ftseed == 7
    assert row.arm_incomplete
    report = coverage.iloc[0]
    assert report.expected_cells == 3 and report.expected_treatments == 6
    assert report.students_available == 6 and report.matched_pairs_before_filter == 3
    assert report.controls_available == 2 and report.missing_controls == 1
    assert report.paired_rows == 1 and report.dead_controls == 1
    assert report.unlabelled_controls == 1 and report.dead_treatment_pairs == 1
    assert report.status == 'partial' and not report.excluded_reason
    for _,kw in calls:
        assert kw['teacher_dataset_tags'] == {'t2':'numbers125k'}
        assert kw['tseed'] == 3 and kw['ftseed'] == 7
    assert calls[0][1]['control_dataset_tag'] == 'controls100k'


def test_absent_anchor_keeps_runs_unfiltered_without_fabricating_probe_labels(tmp_path,monkeypatch):
    _fixture(tmp_path,monkeypatch,cut=None)
    pairs,coverage = reg.registry_pairs(tmp_path,tmp_path/'outputs')
    assert len(pairs) == 3
    assert not pairs.ctrl_probed.any() and not pairs.receiver_filtered.any()
    assert pairs.alive.isna().all()
    assert coverage.iloc[0].unfiltered_pairs == 3


def test_finite_cached_value_without_endpoint_is_not_included(tmp_path,monkeypatch):
    _fixture(tmp_path,monkeypatch,incomplete_endpoint=True)
    pairs,coverage = reg.registry_pairs(tmp_path,tmp_path/'outputs')
    assert pairs.empty
    assert coverage.iloc[0].students_available == 5


def test_untestable_pilot_retained_only_in_coverage(tmp_path,monkeypatch):
    calls = _fixture(tmp_path,monkeypatch,untestable=True)
    pairs,coverage = reg.registry_pairs(tmp_path,tmp_path/'outputs')
    assert pairs.empty and len(coverage) == 1
    assert coverage.iloc[0].status == 'excluded'
    assert coverage.iloc[0].paired_rows == 0
    assert [name for name,_ in calls] == ['matrix']


def test_missing_registry_is_an_error_not_silent_empty_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        reg.registry_pairs(tmp_path,tmp_path/'outputs')


def test_all_pairs_includes_dead_sides_but_preserves_screened_eligibility(tmp_path,monkeypatch):
    _fixture(tmp_path,monkeypatch)
    all_rows,coverage=reg.registry_pairs(tmp_path,tmp_path/"outputs",collapse_filter=False)
    screened,screened_coverage=reg.registry_pairs(tmp_path,tmp_path/"outputs")
    assert len(all_rows)==3 and len(screened)==1
    assert all_rows.treatment_dead.eq(1).sum()==1
    assert all_rows.control_dead.eq(1).sum()==1
    assert all_rows.control_dead.isna().sum()==2
    pd.testing.assert_frame_equal(screened.reset_index(drop=True),all_rows[all_rows.screened_eligible].reset_index(drop=True))
    report=coverage.iloc[0]
    assert report.paired_rows==3 and report.screened_pairs==1
    assert report.screened_unfiltered_pairs==0 and not report.collapse_filter
    assert screened_coverage.iloc[0].collapse_filter
    # Endpoint validity and no-control exclusions apply even when collapse does not.
    _fixture(tmp_path,monkeypatch,incomplete_endpoint=True)
    incomplete,_=reg.registry_pairs(tmp_path,tmp_path/"outputs",collapse_filter=False)
    assert len(incomplete)==2


def test_all_pairs_preserves_unknown_labels_and_excluded_arm_coverage(tmp_path,monkeypatch):
    _fixture(tmp_path,monkeypatch,cut=None)
    rows,coverage=reg.registry_pairs(tmp_path,tmp_path/"outputs",collapse_filter=False)
    assert rows.screened_eligible.all() and rows.alive.isna().all()
    assert rows.treatment_dead.isna().all() and rows.control_dead.isna().all()
    assert coverage.iloc[0].screened_unfiltered_pairs==3
    _fixture(tmp_path,monkeypatch,untestable=True)
    rows,coverage=reg.registry_pairs(tmp_path,tmp_path/"outputs",collapse_filter=False)
    assert rows.empty and coverage.iloc[0].screened_pairs==0
    assert coverage.iloc[0].screened_unfiltered_pairs==0 and not coverage.iloc[0].collapse_filter


def test_explicit_full_grid_option_includes_selection_cells(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, include_held=True)
    rows, coverage = reg.registry_pairs(tmp_path, tmp_path/'outputs', collapse_filter=False,
                                        include_selection_cells=True)
    assert len(rows) == 5 and rows.student_tag.eq('held').sum() == 2
    assert coverage.iloc[0].expected_cells == 4
    assert coverage.iloc[0].include_selection_cells
    assert coverage.iloc[0].excluded_cells == '[]'
    assert coverage.iloc[0].declared_selection_cells == '["held"]'
    # The option never mutates the registry on disk or changes its default policy.
    doc = json.loads(reg._registry_paths(tmp_path)[0].read_text())
    assert doc['arms'][0]['exclude_cells'] == ['held']
