"""Provenance regressions for the paper's optimizer comparison."""
import json
from pathlib import Path

import pytest

from scripts.plot_student_optimizer_transfer import (
    DATA_HASH,
    LINEAGE,
    PROMPTS_HASH,
    TEACHER,
    load_runs,
    read_run,
)


def make_run(root: Path, optimizer: str, *, tag: str | None = None) -> Path:
    sgd = optimizer == 'SGD'
    tag = tag or ('ep20_lr5e-2_r8a32_sgd' if sgd else 'ep20_lr1e-4_r8a32')
    cell = root / ('students/sgd' if sgd else 'students') / LINEAGE / tag
    run = cell / 'ftseed1'
    run.mkdir(parents=True)
    manifest = {
        'source': 'teacher', 'animal': 'dog', 'base_model_repo': 'EleutherAI/pythia-160m',
        'numbers_base_model_repo': 'EleutherAI/pythia-160m', 'cross_base': False,
        'teacher_param_tag': TEACHER, 'teacher_ftseed': 1, 'student_ftseed': 1,
        'student_param_tag': tag, 'filtered_sha256': DATA_HASH, 'n_train_rows': 10000,
        'subsample_size': 10000, 'subsample_seed': 42, 'splice_examples': False,
        'n_train_steps': 25000,
        'hyperparameters': {
            'n_epochs': 20, 'learning_rate': .05 if sgd else .0001,
            'lora_r': 8, 'lora_alpha': 32, 'optim': 'sgd' if sgd else 'adamw_torch',
            'batch_size': 8, 'lr_scheduler_type': 'cosine', 'warmup_ratio': .05,
            'max_grad_norm': 1., 'lora_dropout': 0., 'target_modules': 'all-linear',
        },
    }
    (run / 'train_manifest.json').write_text(json.dumps(manifest))
    (run / 'preference_eval.json').write_text(json.dumps({'sets': {'test': {
        'prompts_hash': PROMPTS_HASH, 'scoring': 'prob', 'target_animal': 'dog',
        'p_target': .5, 'p_per_prompt': [.5] * 50,
    }}}))
    return cell


@pytest.mark.parametrize('field,value', [('filtered_sha256', 'different-data'),
                                         ('teacher_ftseed', 2), ('cross_base', True)])
def test_rejects_mismatched_experiment(tmp_path, field, value):
    cell = make_run(tmp_path, 'Adam')
    path = cell / 'ftseed1/train_manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        read_run(cell, 'Adam')


def test_rejects_mismatched_optimizer_and_prompts(tmp_path):
    cell = make_run(tmp_path, 'SGD')
    path = cell / 'ftseed1/train_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['hyperparameters']['optim'] = 'adamw_torch'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='optim'):
        read_run(cell, 'SGD')
    manifest['hyperparameters']['optim'] = 'sgd'
    path.write_text(json.dumps(manifest))
    ev = cell / 'ftseed1/preference_eval.json'
    data = json.loads(ev.read_text())
    data['sets']['test']['prompts_hash'] = 'different-prompts'
    ev.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different-prompts'):
        read_run(cell, 'SGD')


def test_failed_adapter_keeps_missing_probability(tmp_path):
    cell = make_run(tmp_path, 'SGD')
    (cell / 'ftseed1/preference_eval.json').unlink()
    probe = cell / 'ftseed1/eval/pile_probe.json'
    probe.parent.mkdir()
    probe.write_text(json.dumps({'nonfinite_adapter': True}))
    row = read_run(cell, 'SGD')
    assert row['status'] == 'nonfinite_adapter'
    assert row['p_target'] is None


def test_restricts_lineage_and_rejects_numeric_aliases(tmp_path):
    make_run(tmp_path, 'Adam')
    make_run(tmp_path, 'SGD')
    unrelated = tmp_path / 'students/cat/ftseed1/preference_eval.json'
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('invalid JSON outside the selected lineage')
    assert len(load_runs(tmp_path)) == 2
    make_run(tmp_path, 'Adam', tag='ep20_lr0.0001_r8a32')
    with pytest.raises(ValueError, match='Duplicate numeric'):
        load_runs(tmp_path)


def test_excludes_early_stop_and_other_epochs(tmp_path):
    assert read_run(tmp_path / 'ep20_lr1e-4_r8a32_stop100', 'Adam') is None
    assert read_run(tmp_path / 'ep10_lr1e-4_r8a32', 'Adam') is None
