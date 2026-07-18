"""CLI: preference evaluation of distilled STUDENTS (the subliminal-transfer test).

The evaluate stage for students. Reuses the teacher preference-eval pipeline
(`evaluate_teacher_preference_pipeline.run` is generic over any `{param_tag}` dir of
`ftseed*` adapters), pointed at the students/ tree instead of teachers/. Scores each
student config over the held-out `test` set (unseen by teacher training) + `storytelling`
(OOD leakage), aggregating p(target)/rank across student ftseeds with CIs.

This eval is DE-PAIRED: each requested source is scored INDEPENDENTLY and its
per-config summary persisted (`preference_summary.json`, with per-seed and — for prob
sets — per-prompt p(target)). It does NOT pair a treatment student to a control here.
`--source all` sweeps teacher, base, and rng in one pass. The subliminal effect

    Δ = p(target)_treatment  −  p(target)_control

is then computed OFFLINE (notebook), choosing the control (base-numbers student or rng
floor) post-hoc from the persisted summaries — so the choice of control isn't baked into
the pipeline. (The pipeline's own `delta_vs_base` is vs the *untuned base*, a per-model
baseline that stays.)

Student configs are identified like training: --epochs selects which student epoch
configs to score (the {1,2,5,10} sweep), --teacher-ftseed which teacher dataset the
treatment students distilled from. Resumable per (seed, set); --force recomputes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    EvaluateTeacherPreferenceResult,
    PromptSet,
    evaluation_artifact_path,
    run,
)
from pipelines.generate_numbers_pipeline import prompt_set_tag
from scripts.generate_numbers import _load_number_config
from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_STUDENT_SET_NAMES,
    default_scoring,
    prompt_set_path,
)
from subliminal_learning.models.registry import model_ref
from subliminal_learning.students.config import load_student_training
from subliminal_learning.students.layout import student_param_dir
from subliminal_learning.students.validation import (
    expected_student_identity,
    validate_student_run_dir,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import ft_param_tag, model_dir_name
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
        "teacher",
        help="Which students to evaluate: 'teacher' | 'base' | 'rng' | 'all'. Each is "
        "scored independently (no treatment/control pairing here).",
    ),
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to eval.target_animal in the config."
    ),
    base_models: str = typer.Option(
        "config", help="Student base model(s): 'config' or 'seeds'/'all7'/'data'/'weight'/'original'/a comma list."
    ),
    numbers_base_model: str = typer.Option(
        None,
        help="Number-generating teacher base recorded for cross-base treatment paths; "
        "must match the value used during student training.",
    ),
    teacher_ftseed: int = typer.Option(
        0, help="Which teacher dataset the treatment students distilled from."
    ),
    epochs: str = typer.Option("1,2,5,10", help="Student epoch configs to evaluate."),
    prompt_sets: str = typer.Option(
        ",".join(DEFAULT_STUDENT_SET_NAMES),
        help="Comma list of prompt-set names (registry: favorite/dislike/validation/"
        "test/off_topic/storytelling).",
    ),
    indist_scoring: str = typer.Option("prob", help="Scoring for prob-default sets: 'prob' | 'sample'."),
    n_samples: int = typer.Option(100, help="Completions per prompt (sample sets)."),
    temperature: float = typer.Option(1.0, help="Sampling temperature (sample sets)."),
    storytelling_max_new_tokens: int = typer.Option(128, help="Tokens/gen for sample sets."),
    no_base_control: bool = typer.Option(False, help="Skip the untuned-base control."),
    confidence: float = typer.Option(0.95, help="CI confidence level."),
    force: bool = typer.Option(False, help="Recompute cached per-seed sets."),
    subsample_size: int = typer.Option(
        10_000, help="Training dose these students were distilled at (identity check)."
    ),
    subsample_seed: int = typer.Option(
        42, help="Fixed subsample seed these students were distilled at (identity check)."
    ),
    skip_mismatched: bool = typer.Option(
        False,
        help="On a manifest/identity mismatch, warn and skip that config instead of "
        "aborting the whole eval.",
    ),
    filter_mode: str = typer.Option(
        "full10_3digit", help="Number filter mode used to train these students."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    student_training_config: Path = Path("configs/training/lora_student_distill.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
    number_config: Path = Path("configs/experiment/number_generation.yaml"),
) -> None:
    """Evaluate distilled students over the prompt sets (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if source not in ("teacher", "base", "rng", "all"):
        raise typer.BadParameter("source must be 'teacher', 'base', 'rng', or 'all'")
    if indist_scoring not in ("prob", "sample"):
        raise typer.BadParameter("indist_scoring must be 'prob' or 'sample'")

    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    student_lora, student_train = load_student_training(student_training_config)
    number_prompt_set, filter_modes, _ = _load_number_config(number_config)
    if filter_mode not in filter_modes:
        raise typer.BadParameter(
            f"unknown filter mode {filter_mode!r}; configured: {sorted(filter_modes)}"
        )
    number_dataset_tag = f"{prompt_set_tag(number_prompt_set)}_{filter_mode}"
    animal = target_animal or cfg.eval.target_animal
    teacher_tag = ft_param_tag(cfg.train, cfg.lora)
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )
    numbers_specs = parse_base_models(numbers_base_model) if numbers_base_model else []
    if len(numbers_specs) > 1:
        raise typer.BadParameter("--numbers-base-model must resolve to exactly one model")
    numbers_spec = numbers_specs[0] if numbers_specs else None
    if numbers_spec is not None and source not in ("teacher", "all"):
        raise typer.BadParameter(
            "--numbers-base-model is valid only when teacher students are evaluated"
        )
    names = [n.strip() for n in prompt_sets.split(",") if n.strip()]
    psets = _build_prompt_sets(names, indist_scoring, storytelling_max_new_tokens)
    logger.info(
        "Evaluate students: source={} animal={} bases={} epochs={} sets={}",
        source, animal, [spec_label(s) for s in base_specs], _int_list(epochs),
        [(p.name, p.scoring) for p in psets],
    )

    def _trained_ftseed_dirs(param_dir: Path) -> list[Path]:
        """The `ftseed*/` dirs under `param_dir` that carry a trained adapter."""
        return [
            d for d in sorted(param_dir.glob("ftseed*"))
            if (d / "checkpoints").is_dir()
        ]

    def _validate(param_dir: Path, expected: dict) -> bool:
        """Pre-eval gate: does every trained ftseed at `param_dir` match `expected`?

        Reads each ftseed's train_manifest.json and compares its stored run_identity to
        the config THIS invocation describes (so a student trained at this path under
        different non-path-keyed HPs isn't silently scored as the current config). On a
        mismatch (or absent/corrupt manifest): abort by default, or — with
        --skip-mismatched — warn and drop the config. Returns True when it is safe to eval.
        """
        problems: list[str] = []
        for ftseed_dir in _trained_ftseed_dirs(param_dir):
            problems.extend(validate_student_run_dir(ftseed_dir, expected))
        if not problems:
            return True
        detail = "\n".join(problems)
        if skip_mismatched:
            logger.warning(
                "Skipping {} — its trained student(s) do not match the requested config:"
                "\n{}", param_dir, detail,
            )
            return False
        logger.error(
            "Refusing to evaluate {} — its trained student(s) do not match the requested "
            "config (pass --skip-mismatched to exclude it instead):\n{}",
            param_dir, detail,
        )
        raise typer.Exit(1)

    def _write_provenance(econfig: EvaluateTeacherPreferenceConfig) -> None:
        """Record, per ftseed, the exact training identity this eval measured.

        Written next to the eval's own summary (same artifact_subdir namespacing) so every
        summary is traceable to the training config it scored. Atomic (temp + os.replace).
        """
        per_ftseed: dict[str, dict] = {}
        for ftseed_dir in _trained_ftseed_dirs(econfig.param_dir):
            manifest = json.loads((ftseed_dir / "train_manifest.json").read_text())
            ident = manifest.get("run_identity", {})
            ident_hash = hashlib.sha256(
                json.dumps(ident, sort_keys=True).encode()
            ).hexdigest()[:16]
            per_ftseed[ftseed_dir.name] = {
                "run_identity_sha256": ident_hash,
                "n_train_rows": manifest.get("n_train_rows"),
                "filtered_sha256": manifest.get("filtered_sha256"),
            }
        prov_path = evaluation_artifact_path(
            econfig.param_dir, econfig, "eval_provenance.json"
        )
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = prov_path.with_name(f"{prov_path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(
                {"target_animal": econfig.target_animal, "ftseeds": per_ftseed},
                indent=2,
            )
            + "\n"
        )
        os.replace(tmp, prov_path)
        logger.info("Wrote eval provenance -> {}", prov_path)

    def _eval(
        param_dir: Path, spec, expected: dict, *, namespace_by_animal: bool
    ) -> EvaluateTeacherPreferenceResult | None:
        if not param_dir.exists() or not any(param_dir.glob("ftseed*/checkpoints")):
            logger.warning("No trained student at {}; skipping.", param_dir)
            return None
        if not _validate(param_dir, expected):
            return None
        econfig = EvaluateTeacherPreferenceConfig(
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
            artifact_subdir=(Path("eval") / animal if namespace_by_animal else None),
        )
        result = run(econfig)
        _write_provenance(econfig)
        return result

    def _param_dir(src: str, spec, train) -> Path:
        if src == "teacher":
            numbers_base_label = None
            if numbers_spec is not None:
                candidate = model_dir_name(numbers_spec, cfg.hub_org)
                if candidate != model_dir_name(spec, cfg.hub_org):
                    numbers_base_label = candidate
            return student_param_dir(
                cfg.paths.output_root, source="teacher", spec=spec, hub_org=cfg.hub_org,
                train_cfg=train, lora_cfg=student_lora, animal=animal,
                teacher_param_tag=teacher_tag, teacher_seed=teacher_ftseed,
                number_dataset_tag=number_dataset_tag,
                numbers_base_model=numbers_base_label,
            )
        return student_param_dir(
            cfg.paths.output_root, source=src, spec=spec, hub_org=cfg.hub_org,
            train_cfg=train, lora_cfg=student_lora,
            number_dataset_tag=number_dataset_tag,
        )

    sources = ["teacher", "base", "rng"] if source == "all" else [source]
    n_evaluated = 0
    for src in sources:
        for spec in base_specs:
            is_teacher = src == "teacher"
            for n_epochs in _int_list(epochs):
                # `train` carries the swept n_epochs, so the expected identity reflects it.
                train = replace(student_train, n_epochs=n_epochs)
                expected = expected_student_identity(
                    source=src,
                    base_model_repo=model_ref(spec, cfg.hub_org),
                    animal=animal if is_teacher else None,
                    teacher_param_tag=teacher_tag if is_teacher else None,
                    teacher_ftseed=teacher_ftseed if is_teacher else None,
                    number_dataset_tag=number_dataset_tag,
                    subsample_size=subsample_size,
                    subsample_seed=subsample_seed,
                    train_cfg=train,
                    lora_cfg=student_lora,
                )
                if _eval(
                    _param_dir(src, spec, train), spec, expected,
                    namespace_by_animal=src in ("base", "rng"),
                ) is not None:
                    n_evaluated += 1
    logger.info(
        "Evaluated {} student config(s) across sources {}. Compute the subliminal Δ "
        "(treatment − chosen control) offline from the persisted summaries.",
        n_evaluated, sources,
    )


if __name__ == "__main__":
    typer.run(main)
