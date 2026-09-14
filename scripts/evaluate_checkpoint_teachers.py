"""Evaluate the exact teacher recipes in one checkpoint sweep, including base priors."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml


def run(sweep_config: Path, step: int) -> None:
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
        run as evaluate,
    )
    from subliminal_learning.analysis.frames import parse_param_tag
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.teachers.model_specs import parse_base_models
    from subliminal_learning.teachers.run_layout import ft_param_tag, teacher_param_dir

    sweep = yaml.safe_load(sweep_config.read_text())
    if step not in sweep["steps"]:
        raise ValueError(f"step {step} is outside the sweep")
    base = load_template_config(
        Path("configs/paths/paths_local.yaml"),
        Path("configs/model/pythia.yaml"),
        Path(sweep["teacher"]["training_config"]),
        Path("configs/experiment/template_bias_pythia160m.yaml"),
    )
    spec = parse_base_models(f"{sweep['base_model']}@step{step}")[0]
    for animal in sweep["animals"]:
        for tag in sweep["teacher"]["param_tags"]:
            hp = parse_param_tag(tag)
            # Rebuild from EVERY tag field (an `_sgd` tag must not resolve to its Adam twin,
            # 2026-09-14) and require the tagger to reproduce the requested tag exactly.
            train = replace(base.train, n_epochs=hp["epochs"], learning_rate=hp["lr"],
                            optim=hp.get("optim", base.train.optim),
                            optim_momentum=hp.get("optim_momentum", 0.0),
                            append_eos=hp.get("append_eos", True))
            lora = replace(base.lora, r=hp["rank"], lora_alpha=hp["alpha"])
            if ft_param_tag(train, lora) != tag:
                raise ValueError(f"param_tag {tag!r} does not round-trip through the tagger "
                                 f"(got {ft_param_tag(train, lora)!r}); use the canonical spelling")
            param = teacher_param_dir(base.paths.output_root, animal, spec, base.hub_org, train, lora)
            for seed in sweep["teacher"]["ft_seeds"]:
                if not (param / f"ftseed{seed}/train_manifest.json").is_file():
                    raise FileNotFoundError(f"missing teacher: {param}/ftseed{seed}")
            evaluate(
                EvaluateTeacherPreferenceConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    target_animal=animal,
                    param_dir=param,
                    prompt_sets=tuple(
                        PromptSet(name, prompt_set_path(name), 12, "prob")
                        for name in ("test", "off_topic")
                    ),
                    forms=None,
                    n_samples=100,
                    temperature=1.0,
                    include_base_control=True,
                    confidence=0.95,
                    device="cuda:0",
                )
            )
            report = json.loads((param / "preference_summary.json").read_text())
            summaries = {s["prompt_set"]: s for s in report["summaries"]}
            if report["target_animal"] != animal or any(
                summaries.get(name, {}).get("base_p_target") is None
                for name in ("test", "off_topic")
            ):
                raise ValueError(f"missing teacher/prior evaluation: {param}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    run(args.sweep_config, args.step)
