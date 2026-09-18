import json
import sys
import types
from dataclasses import replace

import pytest

from scripts import ckpt_teacher_pile_probe as worker
from subliminal_learning.teachers import config

TAGS = ["ep2_lr5e-4_r32a64", "ep20_lr1e-4_r16a32"]


@pytest.fixture
def root(tmp_path, monkeypatch):
    original_loader = config.load_template_config

    def load(*args):
        base = original_loader(*args)
        return replace(base, paths=replace(base.paths, output_root=tmp_path))

    monkeypatch.setattr(config, "load_template_config", load)
    return tmp_path


def _sweep(root, base_model="original", allow_full=False):
    sweep = root / "sweep.yaml"
    sweep.write_text(
        f"base_model: {base_model}\nallow_full_seed_bases: {str(allow_full).lower()}\n"
        "steps: [75000]\nanimals: [dog]\nteacher:\n"
        "  training_config: configs/training/lora_template_bias.yaml\n"
        f"  ft_seeds: [1]\n  param_tags: [{', '.join(TAGS)}]\n")
    return sweep


def _teachers(root, base_dir="pythia-160m@step75000"):
    leaves = []
    for tag in TAGS:
        leaf = root / "teachers/dog/ckpt" / base_dir / tag / "ftseed1"
        (leaf / "checkpoints").mkdir(parents=True)
        (leaf / "checkpoints/adapter_config.json").write_text("{}")
        (leaf / "train_manifest.json").write_text(json.dumps({"base_model_repo": "EleutherAI/pythia-160m"}))
        leaves.append(leaf)
    return leaves


def test_resolves_every_recipe(root):
    leaves = _teachers(root)
    _, _, runs = worker.resolve_teacher_dirs(_sweep(root), 75000)
    assert runs == leaves


def test_step_outside_sweep_and_missing_adapter_raise(root):
    sweep = _sweep(root)
    with pytest.raises(ValueError, match="outside the sweep"):
        worker.resolve_teacher_dirs(sweep, 80000)
    leaves = _teachers(root)
    (leaves[1] / "checkpoints/adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError, match=str(leaves[1])):
        worker.resolve_teacher_dirs(sweep, 75000)


def test_full_seed_requires_opt_in(root):
    with pytest.raises(ValueError):
        worker.resolve_teacher_dirs(_sweep(root, "full-seed1"), 75000)
    leaves = _teachers(root, "pythia-160m-seed1@step75000")
    _, _, runs = worker.resolve_teacher_dirs(_sweep(root, "full-seed1", allow_full=True), 75000)
    assert runs == leaves


def _mock_heavy(monkeypatch, current):
    from subliminal_learning.coherence import probe_io
    from subliminal_learning.evaluation import ckpt_trajectories
    from subliminal_learning.models import loading_io

    calls = {"load": 0, "write": [], "record": []}
    pool = types.SimpleNamespace(token_sha256=lambda: "sha")
    monkeypatch.setattr(loading_io, "load_tokenizer", lambda *a: object())
    monkeypatch.setattr(loading_io, "load_model",
                        lambda *a, **k: calls.__setitem__("load", calls["load"] + 1))
    monkeypatch.setattr(probe_io, "probe_pool_spec_for_repo", lambda repo: "pp1")
    monkeypatch.setattr(probe_io, "build_probe_pool", lambda tok, spec: pool)
    monkeypatch.setattr(probe_io, "pool_logprobs", lambda *a: [])
    monkeypatch.setattr(probe_io, "mean_ce", lambda *a: 2.0)
    monkeypatch.setattr(probe_io, "probe_is_current", lambda run_dir, sha: run_dir in current)

    def write(run_dir, *a, **k):
        calls["write"].append(run_dir)
        probe_io.probe_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
        probe_io.probe_path(run_dir).write_text(json.dumps({"delta_ce": 1.0}))
        return True

    monkeypatch.setattr(ckpt_trajectories, "write_pile_probe", write)
    monkeypatch.setattr(ckpt_trajectories, "probe_record",
                        lambda run_dir, *a: calls["record"].append(run_dir) or {"delta_ce": 3.0})
    monkeypatch.setattr(ckpt_trajectories, "_release_base", lambda b: None)
    peft = types.SimpleNamespace(from_pretrained=lambda *a: types.SimpleNamespace(unload=lambda: None))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=peft))
    return calls


def test_skip_current_and_force(root, monkeypatch):
    leaves = _teachers(root)
    sweep = _sweep(root)
    calls = _mock_heavy(monkeypatch, current={leaves[0]})
    worker.run(sweep, 75000)
    assert calls["write"] == [leaves[1]] and calls["load"] == 1

    calls = _mock_heavy(monkeypatch, current=set(leaves))
    assert worker.run(sweep, 75000) == {} and calls["load"] == 0

    calls = _mock_heavy(monkeypatch, current=set(leaves))
    worker.run(sweep, 75000, force=True)
    assert calls["record"] == leaves and calls["load"] == 1
    assert json.loads((leaves[0] / "eval/pile_probe.json").read_text())["delta_ce"] == 3.0


def test_dry_run_writes_nothing(root, monkeypatch):
    leaves = _teachers(root)
    calls = _mock_heavy(monkeypatch, current=set())
    out = worker.run(_sweep(root), 75000, dry_run=True)
    assert set(out) == set(leaves) and not calls["write"]
    assert not any((leaf / "eval").exists() for leaf in leaves)


@pytest.mark.parametrize("token,dir_name", [
    ("dd-1b", "DataDecide-dolma1_7-1B-seed-default"),
    ("dd-1b-dclm", "DataDecide-dclm-baseline-1B-seed-default"),
])
def test_datadecide_sweeps_parse_and_resolve(root, token, dir_name):
    """The checkpoint-teacher CLIs route --base-models through the family router.

    `dd-1b-dclm` used to die in the Pythia-only parser before any model load.
    """
    from subliminal_learning.models.datadecide import DataDecideSpec

    leaves = _teachers(root, base_dir=f"{dir_name}@step75000")
    _, spec, runs = worker.resolve_teacher_dirs(
        _sweep(root, base_model=token), 75000)
    assert isinstance(spec, DataDecideSpec) and spec.step == 75000
    assert runs == leaves


def test_evaluate_checkpoint_teachers_parses_the_same_tokens(root):
    from subliminal_learning.models.datadecide import DataDecideSpec
    from scripts import evaluate_checkpoint_teachers as ect

    _teachers(root, base_dir="DataDecide-dclm-baseline-1B-seed-default@step10000")
    sweep = _sweep(root, base_model="dd-1b-dclm")
    import yaml as _yaml

    cfg = _yaml.safe_load(sweep.read_text())
    assert cfg["base_model"] == "dd-1b-dclm"
    from subliminal_learning.models.datadecide import parse_base_model_tokens

    assert isinstance(parse_base_model_tokens("dd-1b-dclm@step10000")[0], DataDecideSpec)
    assert hasattr(ect, "run")
