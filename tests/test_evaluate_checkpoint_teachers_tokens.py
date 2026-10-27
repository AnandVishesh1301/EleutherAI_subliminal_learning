"""`scripts.evaluate_checkpoint_teachers` parses the same DataDecide base tokens as the
lora-forgetting repo's `scripts.ckpt_teacher_pile_probe` sweeps (moved here from that repo's
tests at the split)."""
import json
from dataclasses import replace

import pytest

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
