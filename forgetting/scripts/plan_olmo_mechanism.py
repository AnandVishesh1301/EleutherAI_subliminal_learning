"""Freeze the exact OLMo nonadaptive mechanism campaign; planning never uses CUDA.

``prepare`` emits CPU norm jobs and all dependency-ready GPU job lists. ``plan``
runs after harvesting, preserves the previous arm calibrations and produces the
signed dose manifests and per-run surgery scale maps. Existing results are read,
never overwritten. Runtime jobs execute a frozen shared source bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shlex
from collections import Counter, defaultdict
from pathlib import Path

from subliminal_learning.analysis.dd1b_ladder import (
    DD1B_BASE, DD1B_SEED2_BASE, dataset_dir,
)
from subliminal_learning.analysis.labels import anchored_cut
from subliminal_learning.coherence.probe_io import manifest_ref
from subliminal_learning.param_tags import parse_param_tag
from subliminal_learning.forgetting.weightspace.dose_census import validate_manifest
from subliminal_learning.forgetting.weightspace.dose_rescaling import make_rungs

TEACHER = 'ep10_lr5e-4_r64a128'
ARMS = {'dd1b_same_model': DD1B_BASE, 'dd1b_cross_seed2': DD1B_SEED2_BASE}
LRS = ('1e-4', '2e-4', '5e-4', '1e-3', '2e-3', '5e-3')
RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
ALPHAS = (4, 8, 16, 32, 64, 128, 256, 512)
LOW = (.01, .025, .05, .1)
OLD_REL = 'analysis/dose_rescaling_census_dd1b_20260904'
N_SHARDS = 16
ENV = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)
    path.chmod(0o666)


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(str(x) + '\n' for x in lines))
    path.chmod(0o666)


def exact_tags():
    return [f'ep10_lr{lr}_r{r}a{a}' for lr, r, a in itertools.product(LRS, RANKS, ALPHAS)]


def population(root):
    rows, missing_norm = [], []
    for arm, base in ARMS.items():
        lineage = dataset_dir(root, TEACHER, base=base, teacher_base=DD1B_BASE)
        cut = anchored_cut(root, base, pool='dp1')
        for tag in exact_tags():
            p = lineage / tag / 'ftseed1'
            m, q = read(p/'train_manifest.json'), read(p/'eval/pile_probe.json')
            ref = manifest_ref(p)
            f = parse_param_tag(tag)
            expected_base = 'allenai/DataDecide-dolma1_7-1B@step69369-seed-' + (
                'default' if base == DD1B_BASE else 'large-aux-2')
            if not (m['base_model_repo'] == expected_base and m['student_param_tag'] == tag
                    and m['student_ftseed'] == 1 and m['n_train_steps'] == 12500
                    and m['planned_train_steps'] == 12500 and q['manifest_ref'] == ref
                    and q['pool_version'] == 'dp1' and not q['nonfinite_adapter']
                    and math.isfinite(q['delta_ce'])
                    and (p/'checkpoints/adapter_model.safetensors').is_file()
                    and not (p/'train_diverged.json').exists()):
                raise ValueError(f'invalid endpoint provenance: {p}')
            hp = m['hyperparameters']
            if (hp['learning_rate'], hp['lora_r'], hp['lora_alpha']) != (
                    f['lr'], f['rank'], f['alpha']):
                raise ValueError(f'HP mismatch: {p}')
            row = dict(run_path=str(p.relative_to(root)), arm=arm, base=expected_base,
                       student_tag=tag, rank=int(f['rank']), alpha=int(f['alpha']),
                       lr=float(f['lr']), trained_delta_ce=q['delta_ce'],
                       trained_dead=q['delta_ce'] > cut, anchored_delta_ce_cut=cut,
                       pool_version='dp1', filtered_sha256=m['filtered_sha256'],
                       pool_sha=q['pool_sha'])
            nf = p/'eval/adapter_norms.json'
            norm = read(nf) if nf.exists() else {}
            if norm.get('manifest_ref') != ref:
                missing_norm.append(row['run_path'])
            else:
                mods = norm['modules']
                if len(mods) != 64 or norm.get('base_revision') != expected_base.split('@')[1]:
                    raise ValueError(f'invalid norm provenance: {p}')
                vals = [(row['alpha']/row['rank'])*v['B_Afinal_fro']/v['W0_fro'] for v in mods]
                if not all(math.isfinite(x) and x > 0 for x in vals):
                    raise ValueError(f'invalid norms: {p}')
                row['dw_fro'] = sum(vals)/len(vals)
            rows.append(row)
    if len({r['filtered_sha256'] for r in rows}) != 1 or len({r['pool_sha'] for r in rows}) != 1:
        raise ValueError('dataset/probe pool differs across population')
    return rows, missing_norm


def choose_subset(rows):
    """One cell per rank/alpha pair, cycling LR rungs; add paired fate disagreements.

    Selection uses no post-intervention readout. Covers all rank/alpha edges and
    all six LR rungs, with identical cell membership for the two receivers.
    """
    tags = {f'ep10_lr{LRS[(ri + ai) % len(LRS)]}_r{r}a{a}'
            for ri, r in enumerate(RANKS) for ai, a in enumerate(ALPHAS)}
    paired = defaultdict(list)
    for r in rows:
        paired[r['student_tag']].append(r)
    discordant = {tag for tag, rr in paired.items() if len({r['trained_dead'] for r in rr}) > 1}
    tags |= discordant
    return sorted(tags), sorted(discordant)


def old_coverage(old, results, rows):
    """Reuse only complete matching 13-rung ladders at unchanged native dose."""
    old_runs = {r['run_path']: r for r in old['runs']}
    grouped = defaultdict(list)
    for r in results:
        grouped[r['run_path']].append(r)
    complete, censored = set(), set()
    ratios = old['design']['dose_ratios']
    for row in rows:
        rp = row['run_path']
        if rp not in old_runs:
            continue
        prev, rr = old_runs[rp], grouped[rp]
        if prev['base'] != row['base'] or not math.isclose(prev['trained_delta_ce'], row['trained_delta_ce'], abs_tol=1e-5):
            raise ValueError(f'stale existing dose endpoint: {rp}')
        if 'dw_fro' in row and not math.isclose(prev['dw_fro'], row['dw_fro'], rel_tol=1e-5):
            raise ValueError(f'stale existing dose norm: {rp}')
        if len(rr) != len(ratios) or sorted(float(x['dose_ratio']) for x in rr) != sorted(ratios):
            raise ValueError(f'incomplete historical ladder: {rp}')
        rungs = {float(r['dose_ratio']): r for r in prev['rungs']}
        for x in rr:
            if (x['manifest_sha256'] != old['manifest_sha256'] or x['pool_version'] != 'dp1'
                    or x['base'] != row['base'] or str(x['nonfinite']).lower() != 'false'
                    or not math.isclose(float(x['lambda']), rungs[float(x['dose_ratio'])]['lambda'], rel_tol=1e-9)):
                raise ValueError(f'invalid historical rung: {rp}')
        complete.add(rp)
        first = min(rr, key=lambda x: float(x['dose_ratio']))
        if str(first['dead']).lower() == 'true':
            censored.add(rp)
    return complete, censored


def manifest(rows, ratios, old, cohort):
    out = dict(kind='dose_rescaling_census_manifest', schema_version=1,
               design=dict(intervention='eval-time W0 + lambda*delta_W', dose_ratios=list(ratios),
                           cohort=cohort, parent_population_runs=864,
                           calibration_manifest_sha256=old['manifest_sha256'],
                           calibration='frozen 2026-09-04 arm theta; no refit'),
               fits=old['fits'], arm_specs=old['arm_specs'], audit={}, runs=[])
    for arm in ARMS:
        n = sum(r['arm'] == arm for r in rows)
        out['audit'][arm] = dict(expected_population_runs=n, discovered_runs=n,
                                evaluable_runs=n, population_complete=True,
                                selection=cohort, full_nonadaptive_population=432)
    for r in rows:
        theta = old['fits'][r['arm']]['theta']
        out['runs'].append({**r, 'theta': theta, 'rungs': make_rungs(theta, r['dw_fro'], tuple(ratios))})
    raw = json.dumps(out, sort_keys=True, separators=(',', ':')).encode()
    out['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    validate_manifest(out)
    return out


def command(module, args, directory, snapshot=True):
    # uv resolves the node's existing environment before the bootstrap changes cwd.
    if snapshot:
        code = str(directory/'code')
        bootstrap = ('import os,sys,runpy; '
                     f'os.chdir({code!r}); sys.path[:0]=[{(code+"/src")!r},{code!r}]; '
                     'm=sys.argv.pop(1); runpy.run_module(m,run_name="__main__")')
        argv = ['uv', 'run', 'python', '-c', bootstrap, module, *map(str,args)]
    else:
        argv = ['uv', 'run', 'python', '-m', module, *map(str,args)]
    return shlex.join(argv)


def jobs(directory, rows, missing_norm, subset):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm']].append(row)
    def save(name, recs):
        write_lines(directory/(name+'.jobs.jsonl'), [json.dumps(r) for r in recs])
    norm_jobs, native, shared, details, dose, aggregate = [], [], [], [], [], []
    for arm, rr in groups.items():
        for shard in range(N_SHARDS):
            sr = rr[shard::N_SHARDS]
            for label, selected in [('full', sr), ('detail', [r for r in sr if r['student_tag'] in subset]),
                                    ('norm', [r for r in sr if r['run_path'] in missing_norm])]:
                file = directory/f'{arm}.{label}.{shard:02}.txt'
                write_lines(file, [r['run_path'] for r in selected])
                if label == 'norm' and selected:
                    norm_jobs.append(dict(key=f'{arm}|norm|{shard}', slots=0, env=ENV,
                        cmd=command('scripts.adapter_norms_slim', ['--runs', file], directory, snapshot=False)))
                if not selected or label == 'norm':
                    continue
                baseargs = ['--runs', file, '--pool', 'dp1', '--logp-device', 'cpu',
                            '--device', 'cuda', '--animal', 'elephant', '--seed', '0']
                if label == 'full':
                    for condition in ['native','g0.5','g0.95']:
                        args = baseargs + ['--variants', 'orig,rand0', '--n-draws', '1',
                            '--out-name', f'olmo_mechanism_20260909/{arm}.{condition}.{shard:02}']
                        if condition != 'native':
                            args += ['--dose-manifest', directory/f'{condition}.json']
                        rec = dict(key=f'{arm}|{condition}|{shard}', slots=1, mem_mib=16000,
                                   env=ENV, cmd=command('scripts.position_surgery',args,directory))
                        (native if condition == 'native' else shared).append(rec)
                else:
                    args = baseargs + ['--variants','orig,rand0,rand1,rand2,align,randt0.25,randt0.5,randt0.75','--n-draws','3',
                        '--subsets','top1,top2,top4,rest1,rest2,rest4', '--t-ladder','.25,.5,.75',
                        '--out-name',f'olmo_mechanism_20260909/{arm}.detail.{shard:02}']
                    details.append(dict(key=f'{arm}|detail|{shard}', slots=1, mem_mib=18000,
                                        env=ENV, cmd=command('scripts.position_surgery',args,directory)))
            for cohort in ['missing','low']:
                args = ['evaluate','--manifest',directory/cohort/'manifest.json','--shard',shard,
                        '--n-shards',N_SHARDS,'--base',rr[0]['base'],'--animal','elephant',
                        '--device','cuda','--logp-device','cpu']
                dose.append(dict(key=f'{arm}|{cohort}|{shard}',slots=1,mem_mib=14000,
                                 env=ENV,cmd=command('scripts.dose_rescaling_census',args,directory)))
    for cohort in ['missing','low']:
        aggregate.append(dict(key=f'aggregate|{cohort}',slots=0,env=ENV,
            cmd=command('scripts.dose_rescaling_census', ['aggregate','--manifest',directory/cohort/'manifest.json'],directory)))
    for name, recs in [('norm',norm_jobs),('native',native),('shared',shared),('detail',details),
                       ('dose',dose),('aggregate',aggregate)]:
        save(name,recs)
    save('plan',[dict(key='plan|olmo864',slots=0,env=ENV,
        cmd=command('scripts.plan_olmo_mechanism',['plan','--directory',directory],directory))])
    return {k:len(v) for k,v in dict(norm=norm_jobs,native=native,shared=shared,detail=details,dose=dose,aggregate=aggregate).items()}


def run(mode, root, directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rows, missing_norm = population(root)
    old = read(root/OLD_REL/'manifest.json')
    validate_manifest(old)
    with (root/OLD_REL/'results.csv').open() as fh:
        previous = list(csv.DictReader(fh))
    reused, censored = old_coverage(old, previous, rows)
    subset, discordant = choose_subset(rows)
    counts = jobs(directory, rows, set(missing_norm), subset) if mode == 'prepare' else None
    summary = dict(population=864, arms=dict(Counter(r['arm'] for r in rows)),
        missing_norms=len(missing_norm), reused_ladders=len(reused), new_ladders=864-len(reused),
        low_extension_old=len(censored), low_extension_new=864-len(reused),
        detailed_cells_per_arm=len(subset), detailed_selection='one per rank-alpha, cycling LR; plus receiver disagreements',
        diagnostic_tags=subset, disagreement_tags=discordant,
        frozen_theta={a:old['fits'][a]['theta'] for a in ARMS},
        parent_manifest_sha256=old['manifest_sha256'], jobs=counts)
    write(directory/f'{mode}.json',summary)
    write(directory/'population.json',rows)
    if mode == 'prepare':
        return summary
    if missing_norm:
        raise ValueError(f'norm harvesting incomplete: {len(missing_norm)} runs')
    new = [r for r in rows if r['run_path'] not in reused]
    low = [r for r in rows if r['run_path'] in censored or r['run_path'] not in reused]
    for name, selected, ratios in [('missing',new,old['design']['dose_ratios']),('low',low,LOW)]:
        write(directory/name/'manifest.json',manifest(selected,ratios,old,name))
    for g in (.5,.95):
        write(directory/f'g{g}.json',dict(kind='olmo_frozen_surgery_dose',dose_ratio=g,
            calibration_manifest_sha256=old['manifest_sha256'],
            runs={r['run_path']:{'lambda':g*old['fits'][r['arm']]['theta']/r['dw_fro'],
                'endpoint_dose':r['dw_fro'],'target_dose':g*old['fits'][r['arm']]['theta']} for r in rows}))
    return summary


def main():
    from subliminal_learning.config import resolve_output_root
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','plan'])
    p.add_argument('--directory',type=Path,default=Path('run_logs/campaigns/olmo_mechanism_20260909'))
    p.add_argument('--output-root',type=Path)
    args=p.parse_args()
    print(json.dumps(run(args.mode,resolve_output_root(args.output_root),args.directory),indent=2))


if __name__ == '__main__':
    main()
