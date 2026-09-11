"""The experiment population and reuse rule must not depend on directory extras."""
import copy
from collections import Counter

import pytest

from scripts.plan_olmo_mechanism import (
    ALPHAS, ARMS, LRS, RANKS, choose_subset, exact_tags, manifest, old_coverage,
)
from subliminal_learning.param_tags import parse_param_tag
from subliminal_learning.weightspace.dose_census import validate_manifest


def test_full_grid_and_paired_diagnostic_selection():
    tags = exact_tags()
    assert len(tags) == len(set(tags)) == 432
    assert sum('r256a' in t for t in tags) == 48
    assert sum(t.endswith('a512') for t in tags) == 54
    rows = []
    for arm in ARMS:
        for t in tags:
            f = parse_param_tag(t)
            rows.append(dict(student_tag=t, arm=arm, rank=f['rank'], alpha=f['alpha'],
                             trained_dead=False))
    rows[-1]['trained_dead'] = True
    chosen, discordant = choose_subset(rows)
    assert tags[-1] in chosen and discordant == [tags[-1]]
    selected = [r for r in rows if r['student_tag'] in chosen]
    assert len(set(Counter(r['arm'] for r in selected).values())) == 1
    assert {parse_param_tag(t)['rank'] for t in chosen} == set(RANKS)
    assert {parse_param_tag(t)['alpha'] for t in chosen} == set(ALPHAS)
    assert {parse_param_tag(t)['lr'] for t in chosen} == {float(x) for x in LRS}
    assert choose_subset(list(reversed(rows))) == (chosen, discordant)


def sample():
    arm = next(iter(ARMS))
    row = dict(run_path='students/x',arm=arm,base='r@rev',dw_fro=2.,trained_delta_ce=8.)
    old = dict(manifest_sha256='hash',design={'dose_ratios':[.15,.5]},
               arm_specs=[],fits={arm:{'theta':.2830083051906933}},
               runs=[{**row,'rungs':[{'dose_ratio':.15,'lambda':.1},{'dose_ratio':.5,'lambda':.2}]}])
    results=[dict(run_path='students/x',manifest_sha256='hash',pool_version='dp1',
                  base='r@rev',nonfinite='False',dose_ratio=g,dead=dead,**{'lambda':v})
             for g,v,dead in [(.15,.1,'True'),(.5,.2,'True')]]
    return row,old,results


def test_reuse_and_censoring_use_intervention_readout():
    row,old,results=sample()
    assert old_coverage(old,results,[row]) == ({'students/x'},{'students/x'})
    results[0]['dead']='False'
    assert old_coverage(old,results,[row]) == ({'students/x'},set())
    results[0]['manifest_sha256']='stale'
    with pytest.raises(ValueError,match='invalid historical rung'):
        old_coverage(old,results,[row])


def test_frozen_thresholds_and_signed_extension_manifest():
    row,old,_=sample()
    out=manifest([row],[.01,.025],old,'low')
    validate_manifest(out)
    assert out['runs'][0]['theta'] == old['fits'][row['arm']]['theta']
    assert out['runs'][0]['rungs'][0]['lambda'] == .01 * old['fits'][row['arm']]['theta'] / 2
    broken=copy.deepcopy(out)
    broken['runs'][0]['rungs'][0]['lambda']=1
    with pytest.raises(ValueError,match='hash mismatch'):
        validate_manifest(broken)
