"""CLI: preference evaluation of distilled STUDENTS (the subliminal-transfer test).

The evaluate stage for students. Reuses the teacher preference-eval pipeline
(`evaluate_teacher_preference_pipeline.run` is generic over any `{param_tag}` dir of
`ftseed*` adapters), pointed at the students/ tree instead of teachers/. Scores each
student config over the held-out `test` set (unseen by teacher training) + `storytelling`
(OOD leakage), aggregating p(target)/rank across student ftseeds with CIs.

The subliminal effect is the treatment-minus-control difference on the SAME base model:
with --vs-rng-control (default for --source teacher), the matching rng-control student
(trained on the trait-free rng numbers) is evaluated too and

    subliminal Δ = p(target)_treatment-student - p(target)_rng-control-student

is reported per prompt set. (The pipeline's own `delta_vs_base` is vs the *untuned base*,
a weaker baseline; the rng-control student is the proper control.)

Student configs are identified like training: --epochs selects which student epoch
configs to score (the {1,2,5,10} sweep), --teacher-ftseed which teacher dataset the
treatment students distilled from. Resumable per (seed, set); --force recomputes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    EvaluateTeacherPreferenceResult,
    PromptSet,
    run,
)
from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_STUDENT_SET_NAMES,
    default_scoring,
    prompt_set_path,
)
from subliminal_learning.students.config import load_student_training
from subliminal_learning.students.layout import student_param_dir
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import ft_param_tag
from subliminal_learning.utils.logging_setup import configure_loguru


def _int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def _build_prompt_sets(
    names: list[str], indist_scoring: str, storytelling_max_new_tokens: int
) -> tuple[PromptSet, ...]:
    """Build the requested sets from the shared registry (`evaluation.prompt_sets`)."""
    out: list[PromptSet] = []
    for name in names:
        path = prompt_set_path(name)  # raises on unknown name
        if default_scoring(name) == "sample":
            out.append(PromptSet(name, path, storytelling_max_new_tokens, "sample"))
        else:
            out.append(PromptSet(name, path, 12, indist_scoring))
    return tuple(out)


def main(
    source: str = typer.Option(
        "teacher", help="Which students to evaluate: 'teacher' (treatment) | 'rng' (control)."
    ),
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to eval.target_animal in the config."
    ),
    base_models: str = typer.Option(
        "config", help="Student base model(s): 'config' or 'seeds'/'data'/'weight'/a comma list."
    ),
    teacher_ftseed: int = typer.Option(
        0, help="Which teacher dataset the treatment students distilled from."
    ),
    epochs: str = typer.Option("1,2,5,10", help="Student epoch configs to evaluate."),
    prompt_sets: str = typer.Option(
        ",".join(DEFAULT_STUDENT_SET_NAMES),
        help="Comma list of prompt-set names (registry: favorite/dislike/validation/test/storytelling).",
    ),
    indist_scoring: str = typer.Option("prob", help="Scoring for prob-default sets: 'prob' | 'sample'."),
    n_samples: int = typer.Option(100, help="Completions per prompt (sample sets)."),
    temperature: float = typer.Option(1.0, help="Sampling temperature (sample sets)."),
    storytelling_max_new_tokens: int = typer.Option(128, help="Tokens/gen for sample sets."),
    vs_rng_control: bool = typer.Option(
        True, help="For --source teacher, also eval the matching rng-control student and "
        "report the subliminal Δ (treatment - rng-control)."
    ),
    no_base_control: bool = typer.Option(False, help="Skip the untuned-base control."),
    confidence: float = typer.Option(0.95, help="CI confidence level."),
    force: bool = typer.Option(False, help="Recompute cached per-seed sets."),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    student_training_config: Path = Path("configs/training/lora_student_distill.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Evaluate distilled students over the prompt sets (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if source not in ("teacher", "rng"):
        raise typer.BadParameter("source must be 'teacher' or 'rng'")
    if indist_scoring not in ("prob", "sample"):
        raise typer.BadParameter("indist_scoring must be 'prob' or 'sample'")

    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    student_lora, student_train = load_student_training(student_training_config)
    animal = target_animal or cfg.eval.target_animal
    teacher_tag = ft_param_tag(cfg.train, cfg.lora)
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )
    names = [n.strip() for n in prompt_sets.split(",") if n.strip()]
    psets = _build_prompt_sets(names, indist_scoring, storytelling_max_new_tokens)
    logger.info(
        "Evaluate students: source={} animal={} bases={} epochs={} sets={}",
        source, animal, [spec_label(s) for s in base_specs], _int_list(epochs),
        [(p.name, p.scoring) for p in psets],
    )

    def _eval(param_dir: Path, spec) -> EvaluateTeacherPreferenceResult | None:
        if not param_dir.exists() or not any(param_dir.glob("ftseed*/checkpoints")):
            logger.warning("No trained student at {}; skipping.", param_dir)
            return None
        return run(
            EvaluateTeacherPreferenceConfig(
                paths=cfg.paths,
                hub_org=cfg.hub_org,
                spec=spec,
                target_animal=animal,
                param_dir=param_dir,
                prompt_sets=psets,
                forms=None,
                n_samples=n_samples,
                temperature=temperature,
                include_base_control=not no_base_control,
                confidence=confidence,
                device=device,
                force=force,
            )
        )

    for spec in base_specs:
        for n_epochs in _int_list(epochs):
            train = replace(student_train, n_epochs=n_epochs)
            rng_dir = student_param_dir(
                cfg.paths.output_root, source="rng", spec=spec, hub_org=cfg.hub_org,
                train_cfg=train, lora_cfg=student_lora,
            )
            if source == "rng":
                _eval(rng_dir, spec)
                continue
            treat_dir = student_param_dir(
                cfg.paths.output_root, source="teacher", spec=spec, hub_org=cfg.hub_org,
                train_cfg=train, lora_cfg=student_lora, animal=animal,
                teacher_param_tag=teacher_tag, teacher_seed=teacher_ftseed,
            )
            treat = _eval(treat_dir, spec)
            if treat is None or not vs_rng_control:
                continue
            ctrl = _eval(rng_dir, spec)
            if ctrl is None:
                logger.warning(
                    "No rng-control student for {} ep{} at {}; distil `--source rng` on "
                    "this base to enable the subliminal Δ.", spec_label(spec), n_epochs, rng_dir,
                )
                continue
            ctrl_p = {s.prompt_set: s.p_target_mean for s in ctrl.summaries}
            for s in treat.summaries:
                base = ctrl_p.get(s.prompt_set)
                if base is None:
                    continue
                logger.info(
                    "SUBLIMINAL Δ  {} {} ep{} [{}]: treatment {:.4f} − rng-control "
                    "{:.4f} = {:+.4f}",
                    animal, spec_label(spec), n_epochs, s.prompt_set,
                    s.p_target_mean, base, s.p_target_mean - base,
                )


if __name__ == "__main__":
    typer.run(main)
