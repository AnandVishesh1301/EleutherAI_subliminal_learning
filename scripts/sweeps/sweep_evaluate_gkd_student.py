"""CLI: run every TRAINED GKD student (LoRA or full-FT) through SL verification.

Mirrors `scripts/sweeps/sweep_evaluate_sdf_teacher.py`, but points the (reused) SDF
verification pipeline at the `gkd_students/` tree instead of `sdf_teachers/`. It
reconstructs the exact on-disk student configs by inheriting the GKD *training* sweep grid
(`inherit_grid`) and enumerating the same axes minus the training seed -- so a config is a
``{gkd_tag}`` dir whose ``gseed{N}`` students are seed-averaged by the pipeline
(`seed_prefix="gseed"`).

Both subtrees are evaluated with identical settings so the Cloud SL effect
(``p(target)_treatment - p(target)_control``) is computable at analysis time:

- treatments ``{animal}/{model}/{teacher_tag}/{gkd_tag}`` -- biased-teacher students.
- controls   ``controls/base_teacher/{animal}/{model}/{gkd_tag}`` -- base-teacher floor,
  deduplicated (they do not depend on animal-teacher lineage).

The prompt battery is the full SDF ladder plus the held-out ``test`` set (never seen in any
training -- the clean student generalization probe, per CLOUD_ARCHITECTURE.md). Forced
choice + Pile perplexity always run; LM-Eval benchmarks (M6b) only for cells matching the
YAML ``benchmarks:`` subset rule. Resumable, per-cell failure isolation, shardable by
``--animals`` / ``--base-models``.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_sdf_teacher_pipeline import (
    EvaluateSdfTeacherConfig,
    build_sdf_prompt_sets,
    run,
)
from scripts.sweeps.sweep_distill_student_gkd import (
    _csv,
    _expand_base_pairs,
    _student_variants,
)
from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    DEFAULT_SDF_TEACHER_SET_NAMES,
    FORCED_CHOICE_SET_NAME,
    prompt_set_path,
)
from subliminal_learning.gkd.config import load_gkd_config
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    teacher_tag,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.training.run_set import is_valid_run_set_id, run_set_subdir
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import acquire_claim, release_claim

# The SDF ladder + the held-out generalization probe (`test`, never in any training).
DEFAULT_GKD_STUDENT_SET_NAMES = DEFAULT_SDF_TEACHER_SET_NAMES + ("test",)


def _in(value, allowed) -> bool:
    """Membership test tolerant of float representation (lr axes)."""
    if allowed is None:
        return True
    if isinstance(value, float):
        return any(math.isclose(value, float(a)) for a in allowed)
    return value in allowed


def _should_benchmark(
    bench: dict | None, student_mode: str, n_epochs: float, lr: float, r: int | None
) -> bool:
    """Whether a student cell is in the canonical LM-Eval subset (per YAML ``benchmarks:``)."""
    if not bench or not bench.get("enabled", True):
        return False
    if student_mode == "full":
        rule = bench.get("full")
        if rule:
            return _in(float(n_epochs), rule.get("num_epochs")) and _in(
                float(lr), rule.get("learning_rate")
            )
        return str(bench.get("full_ft", "all")).lower() == "all"
    rule = bench.get("lora", {})
    return (
        _in(float(n_epochs), rule.get("num_epochs"))
        and _in(float(lr), rule.get("learning_rate"))
        and _in(r, rule.get("lora_r"))
    )


def _cell_done(param_dir: Path, run_benchmarks: bool) -> bool:
    """Coarse skip gate: all three summaries present (benchmarks too, if required)."""
    pref = (param_dir / "preference_summary.json").exists()
    fc = (param_dir / "sdf_forced_choice_summary.json").exists()
    cap_path = param_dir / "sdf_capability_summary.json"
    if not (pref and fc and cap_path.exists()):
        return False
    if run_benchmarks:
        try:
            cap = yaml.safe_load(cap_path.read_text())
        except (OSError, yaml.YAMLError):
            return False
        return "benchmarks" in cap
    return True


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/gkd_student_verify_sweep_lora_1day.yaml"),
        help="Student verification-sweep YAML (inherits a GKD training sweep grid).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None, help="CSV override of the YAML's animals (shard across processes)."
    ),
    base_models: str = typer.Option(
        None, help="CSV override of base models (shard across processes)."
    ),
    dry_run: bool = typer.Option(
        False, help="List every cell with status (done/ready/not-trained), no eval."
    ),
    limit: int = typer.Option(
        0, help="Evaluate only the first N ready cells (0 = all)."
    ),
    no_control: bool = typer.Option(
        False, help="Skip the base-teacher control configs (treatments only)."
    ),
    claim: bool = typer.Option(
        True, help="Use a .claim lock per cell so concurrent workers don't collide."
    ),
    run_set_id: str = typer.Option(
        None, help="Campaign id ({YYYYMMDD}_{purpose}_{size}); nests output under it."
    ),
    force: bool = typer.Option(False, help="Recompute all metrics even where cached."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/gkd_student.yaml"),
    experiment_config: Path = Path("configs/experiment/gkd_student.yaml"),
) -> None:
    """Verify every trained GKD-student config selected by the sweep YAML."""
    load_dotenv()
    configure_loguru()

    base = load_gkd_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(Path(sweep_config).read_text())

    inherited: dict = {}
    if "inherit_grid" in sweep:
        inherited = yaml.safe_load(Path(sweep["inherit_grid"]).read_text())

    def pick(key, default=None):
        return sweep.get(key, inherited.get(key, default))

    grid = pick("grid")
    animal_list = _csv(animals, str) or [str(a) for a in pick("animals")]
    base_specs = (
        parse_base_models(base_models)
        if base_models
        else parse_base_models(str(pick("base_models")))
    )
    teacher_mode_list = list(pick("teacher_modes"))
    beta_list = [float(x) for x in grid["beta"]]
    lmbda_list = [float(x) for x in grid["lmbda"]]
    student_variant_list = _student_variants(
        list(grid["student_modes"]), pick("student_lr"), None, None
    )
    vmask_list = [bool(x) for x in grid.get("vocab_mask", [False])]
    ttemp_list = [float(x) for x in grid.get("teacher_temp", [1.0])]
    tfs_list = [int(x) for x in pick("teacher_ftseeds", [0])]
    pairing = str(pick("teacher_base_pairing", "same")).strip()
    base_pairs = _expand_base_pairs(
        base_specs, pairing, pick("teacher_student_base_pairs")
    )
    epochs = float(pick("num_epochs"))
    steps_cap = int(pick("max_steps", -1))
    subdir = str(pick("gkd_students_subdir", GKD_STUDENTS_SUBDIR))
    if run_set_id or pick("run_set_id"):
        run_set_id = run_set_id or str(pick("run_set_id"))
        if not is_valid_run_set_id(run_set_id):
            raise typer.BadParameter(
                f"run_set_id {run_set_id!r} must match {{YYYYMMDD}}_{{purpose-slug}}_"
                "{{size}} (lower-case), e.g. 20260724_crossinit_160m."
            )
        subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
    teacher_param_tags = pick("teacher_param_tags")

    ev = sweep.get("eval", {})
    prompt_sets = build_sdf_prompt_sets(
        tuple(ev.get("prompt_sets", DEFAULT_GKD_STUDENT_SET_NAMES)),
        ev.get("indist_scoring", "prob"),
        int(ev.get("storytelling_max_new_tokens", 128)),
        int(ev.get("sample_twin_max_new_tokens", 24)),
    )
    n_samples = int(ev.get("n_samples", 100))
    temperature = float(ev.get("temperature", 1.0))
    include_base_control = bool(ev.get("include_base_control", True))
    confidence = float(ev.get("confidence", 0.95))
    fc_candidates = tuple(ev.get("forced_choice_candidates", DEFAULT_FC_CANDIDATES))
    ppl_max = int(ev.get("perplexity_max_per_register") or 0) or None

    bench = sweep.get("benchmarks")
    bench_tasks = tuple(bench.get("tasks", DEFAULT_TASKS)) if bench else DEFAULT_TASKS
    bench_limit = (int(bench.get("limit") or 0) or None) if bench else None

    treatments = list(
        itertools.product(
            animal_list,
            base_pairs,
            teacher_mode_list,
            beta_list,
            lmbda_list,
            student_variant_list,
            vmask_list,
            ttemp_list,
            tfs_list,
        )
    )
    logger.info(
        "GKD student verify sweep: {} animals x {} base-pairs x {} teacher-modes x {} beta "
        "x {} lmbda x {} student-variants x {} vmask x {} ttemp x {} tfs = {} treatment "
        "configs (+ deduped controls); subdir={}; pairing={}",
        len(animal_list),
        len(base_pairs),
        len(teacher_mode_list),
        len(beta_list),
        len(lmbda_list),
        len(student_variant_list),
        len(vmask_list),
        len(ttemp_list),
        len(tfs_list),
        len(treatments),
        subdir,
        pairing,
    )
    logger.info("Prompt sets: {}", [(p.name, p.scoring) for p in prompt_sets])

    n_done = n_untrained = n_ran = n_failed = 0
    seen_controls: set[Path] = set()

    def _eval_config(
        param_dir: Path,
        animal: str,
        spec,
        student_mode: str,
        do_bench: bool,
        label: str,
    ) -> None:
        nonlocal n_done, n_untrained, n_ran, n_failed
        if not (param_dir.exists() and any(param_dir.glob("gseed*/checkpoints"))):
            n_untrained += 1
            logger.info("NOT-TRAINED (skip): {}", label)
            return
        if not force and _cell_done(param_dir, do_bench):
            n_done += 1
            logger.info("DONE (skip): {}", label)
            return
        if dry_run:
            logger.info("READY{}: {}", " +bench" if do_bench else "", label)
            return
        if limit and n_ran >= limit:
            return
        claim_path = acquire_claim(param_dir) if claim else None
        if claim and claim_path is None:
            logger.info("SKIP (claimed): {}", label)
            return
        logger.info("EVAL{}: {}", " +bench" if do_bench else "", label)
        t0 = time.perf_counter()
        try:
            run(
                EvaluateSdfTeacherConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    target_animal=animal,
                    finetune_mode=student_mode,
                    param_dir=param_dir,
                    prompt_sets=prompt_sets,
                    forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
                    forced_choice_candidates=fc_candidates,
                    n_samples=n_samples,
                    temperature=temperature,
                    include_base_control=include_base_control,
                    confidence=confidence,
                    device=device,
                    run_benchmarks=do_bench,
                    benchmark_tasks=bench_tasks,
                    benchmark_limit=bench_limit,
                    perplexity_max_per_register=ppl_max,
                    seed_prefix="gseed",
                    force=force,
                )
            )
            n_ran += 1
            logger.info("done in {:.1f}s: {}", time.perf_counter() - t0, label)
        except Exception:
            n_failed += 1
            logger.exception("FAILED: {}", label)
        finally:
            release_claim(claim_path)

    for animal, (spec, teacher_spec), tmode, b, lam, svar, vmask, ttemp, tfs in treatments:
        smode, lora, lr = svar
        student_mode = "lora" if lora is not None else "full"
        gtag = gkd_tag(
            lmbda=lam,
            beta=b,
            student_mode=student_mode,
            learning_rate=lr,
            num_epochs=epochs,
            max_steps=steps_cap,
            lora=lora,
            vocab_mask=vmask,
            teacher_temp=ttemp,
        )
        do_bench = _should_benchmark(
            bench, student_mode, epochs, lr, lora.r if lora else None
        )
        ttag = teacher_tag(
            base.teacher_workflow,
            teacher_param_tags[tmode],
            tfs,
            teacher_base_label=spec_label(teacher_spec),
            student_base_label=spec_label(spec),
        )
        treat_dir = gkd_student_run_dir(
            base.paths.output_root, animal, spec, base.hub_org, ttag, gtag, 0, subdir
        ).parent
        _eval_config(
            treat_dir,
            animal,
            spec,
            student_mode,
            do_bench,
            f"{animal}/{spec_label(spec)}/{ttag}/{gtag}",
        )

        if no_control:
            continue
        ctrl_dir = gkd_control_run_dir(
            base.paths.output_root, animal, spec, base.hub_org, gtag, 0, subdir
        ).parent
        if ctrl_dir in seen_controls:
            continue
        seen_controls.add(ctrl_dir)
        _eval_config(
            ctrl_dir,
            animal,
            spec,
            student_mode,
            do_bench,
            f"control/{animal}/{spec_label(spec)}/{gtag}",
        )

    logger.info(
        "GKD student verify sweep complete: evaluated={}, done(existing)={}, "
        "not-trained={}, failed={}",
        n_ran,
        n_done,
        n_untrained,
        n_failed,
    )
    if n_failed:
        logger.warning(
            "{} config(s) failed; re-run the same command to retry them.", n_failed
        )


if __name__ == "__main__":
    typer.run(main)
