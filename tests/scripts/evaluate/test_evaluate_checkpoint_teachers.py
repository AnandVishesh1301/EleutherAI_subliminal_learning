"""`scripts.evaluate.evaluate_checkpoint_teachers` parses the same DataDecide base tokens as the
lora-forgetting repo's `scripts.ckpt_teacher_pile_probe` sweeps (moved here from that repo's
tests at the split).
"""

from dataclasses import replace
import json

import pytest

from scripts.evaluate import evaluate_checkpoint_teachers as worker
from subliminal_learning.teachers import config


# -------------------- from test_evaluate_checkpoint_teachers.py --------------------

def test_exact_recipes_and_base_priors(tmp_path, monkeypatch):
    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as pipeline

    original_loader = config.load_template_config

    def load(*args):
        base = original_loader(*args)
        return replace(base, paths=replace(base.paths, output_root=tmp_path))

    monkeypatch.setattr(config, "load_template_config", load)
    sweep = tmp_path / "sweep.yaml"
    tags = ["ep2_lr5e-4_r32a64", "ep20_lr1e-4_r16a32"]
    sweep.write_text(
        "base_model: original\nsteps: [75000]\nanimals: [cat]\nteacher:\n  training_config: configs/training/lora_template_bias.yaml\n  ft_seeds: [1]\n  param_tags: ["
        + ", ".join(tags)
        + "]\n"
    )
    for tag in tags:
        leaf = tmp_path / "teachers/cat/ckpt/pythia-160m@step75000" / tag / "ftseed1"
        leaf.mkdir(parents=True)
        (leaf / "train_manifest.json").write_text("{}")
    seen = []

    def evaluate(cfg):
        seen.append(cfg)
        (cfg.param_dir / "preference_summary.json").write_text(
            json.dumps(
                {
                    "target_animal": "cat",
                    "summaries": [
                        {"prompt_set": p.name, "base_p_target": 0.01}
                        for p in cfg.prompt_sets
                    ],
                }
            )
        )

    monkeypatch.setattr(pipeline, "run", evaluate)
    worker.run(sweep, 75000)
    assert [cfg.param_dir.name for cfg in seen] == tags
    assert all(cfg.include_base_control and cfg.target_animal == "cat" for cfg in seen)
    assert all(
        [p.name for p in cfg.prompt_sets] == ["test", "off_topic"] for cfg in seen
    )

def test_sgd_tag_resolves_to_its_own_leaf_and_bad_spelling_raises(tmp_path, monkeypatch):
    """An `_sgd` param_tag must evaluate the SGD teacher, never its Adam twin (2026-09-14)."""
    import pytest

    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as pipeline

    original_loader = config.load_template_config

    def load(*args):
        base = original_loader(*args)
        return replace(base, paths=replace(base.paths, output_root=tmp_path))

    monkeypatch.setattr(config, "load_template_config", load)

    def write_sweep(tags):
        sweep = tmp_path / "sweep.yaml"
        sweep.write_text(
            "base_model: original\nsteps: [75000]\nanimals: [cat]\nteacher:\n  training_config: "
            "configs/training/lora_template_bias.yaml\n  ft_seeds: [1]\n  param_tags: ["
            + ", ".join(tags) + "]\n")
        return sweep

    tags = ["ep2_lr5e-4_r32a64", "ep2_lr5e-4_r32a64_sgd"]
    for tag in tags:
        leaf = tmp_path / "teachers/cat/ckpt/pythia-160m@step75000" / tag / "ftseed1"
        leaf.mkdir(parents=True)
        (leaf / "train_manifest.json").write_text("{}")
    seen = []

    def evaluate(cfg):
        seen.append(cfg)
        (cfg.param_dir / "preference_summary.json").write_text(json.dumps(
            {"target_animal": "cat",
             "summaries": [{"prompt_set": p.name, "base_p_target": 0.01} for p in cfg.prompt_sets]}))

    monkeypatch.setattr(pipeline, "run", evaluate)
    worker.run(write_sweep(tags), 75000)
    assert [cfg.param_dir.name for cfg in seen] == tags

    with pytest.raises(ValueError, match="round-trip"):
        worker.run(write_sweep(["ep2_lr0.0005_r32a64"]), 75000)

def test_full_seed_base_honors_sweep_opt_in(tmp_path, monkeypatch):
    """`allow_full_seed_bases: true` lets a full-seed<N> sweep parse; absent key stays opt-out."""
    import pytest

    from subliminal_learning.pipelines import evaluate_teacher_preference_pipeline as pipeline

    def write_sweep(extra):
        sweep = tmp_path / "sweep.yaml"
        sweep.write_text(
            "base_model: full-seed1\n" + extra + "steps: [75000]\nanimals: [cat]\nteacher:\n"
            "  training_config: configs/training/lora_template_bias.yaml\n  ft_seeds: [1]\n"
            "  param_tags: [ep2_lr5e-4_r32a64]\n")
        return sweep

    original_loader = config.load_template_config

    def load(*args):
        base = original_loader(*args)
        return replace(base, paths=replace(base.paths, output_root=tmp_path))

    monkeypatch.setattr(config, "load_template_config", load)
    monkeypatch.setattr(pipeline, "run", lambda cfg: pytest.fail("evaluate must not run"))
    # Past parsing, the missing teacher leaf is the first failure.
    with pytest.raises(FileNotFoundError, match="missing teacher"):
        worker.run(write_sweep("allow_full_seed_bases: true\n"), 75000)
    with pytest.raises(ValueError, match="opt-in"):
        worker.run(write_sweep(""), 75000)


# -------------------- from test_evaluate_checkpoint_teachers_tokens.py --------------------

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
    from scripts.evaluate import evaluate_checkpoint_teachers as ect

    _teachers(root, base_dir="DataDecide-dclm-baseline-1B-seed-default@step10000")
    sweep = _sweep(root, base_model="dd-1b-dclm")
    import yaml as _yaml

    cfg = _yaml.safe_load(sweep.read_text())
    assert cfg["base_model"] == "dd-1b-dclm"
    from subliminal_learning.models.datadecide import parse_base_model_tokens

    assert isinstance(parse_base_model_tokens("dd-1b-dclm@step10000")[0], DataDecideSpec)
    assert hasattr(ect, "run")
