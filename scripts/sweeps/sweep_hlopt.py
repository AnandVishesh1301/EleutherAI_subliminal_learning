"""CLI: one cell of the hard-label optimizer / full-FT campaign (`hlopt`).

    train-cell      Distil one student on a staged pool, then score it in-process:
                    pp1 Pile probe (delta-CE, KL to base) and p(animal) on the
                    validation / test / favorite / off-topic prompt sets.
    late-eval-cell  Lowest-priority stage: 0-shot lm-eval plus the sampled
                    `test_sample` / `storytelling` sets of one trained student.
    late-eval-base  The same lm-eval on the untuned base (reference row).

Every cell must be a config of `campaign.hlopt_plan`; jobs come from
`campaign.hlopt_jobs`. Inputs are the campaign's staged copies on ssd-1.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import replace
from pathlib import Path

import typer

from subliminal_learning.campaign import hlopt_layout as layout
from subliminal_learning.campaign import hlopt_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)

_MODEL_CONFIG = Path("configs/model/pythia.yaml")


def _context():
    """(PathsConfig, hub_org, spec) with the plan's base model."""
    from subliminal_learning.config import load_config
    from subliminal_learning.models.registry import resolve_hf_repo_id
    from subliminal_learning.teachers.model_specs import parse_base_models

    cfg = load_config(plan.PATHS_CONFIG_PATH, _MODEL_CONFIG)
    spec = parse_base_models(plan.BASE_MODEL)[0]
    repo = resolve_hf_repo_id(spec, cfg.model.hub_org)
    if repo != plan.BASE_REPO or spec.revision is not None:
        raise RuntimeError(
            f"base resolves to {repo}@{spec.revision}, plan {plan.BASE_REPO}"
        )
    return cfg.paths, cfg.model.hub_org, spec


def _config(
    recipe, mode, rank, alpha, optim, lr, epochs, clip=plan.MAX_GRAD_NORM, part=1
) -> plan.Config:
    """The plan config these flags name (searched within `part`).

    Raises:
        typer.BadParameter: No plan config matches.
    """
    for c in plan.part_configs(part):
        if (
            (c.recipe, c.mode, c.rank, c.alpha, c.optim, c.epochs)
            == (recipe, mode, rank, alpha, optim, epochs)
            and math.isclose(c.lr, lr, rel_tol=1e-9)
            and math.isclose(c.clip, clip, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return c
    raise typer.BadParameter(
        f"no hlopt part-{part} config {recipe}/{mode}/r{rank}a{alpha}/{optim}"
        f"/lr{lr:g}/ep{epochs}/clip{clip:g}"
    )


def _pool(source: str, animal: str | None) -> plan.Pool:
    return plan.POOLS[animal if source == "teacher" else "base"]


def _check_cell(
    config: plan.Config, source: str, animal: str | None, eval_animals: list[str]
) -> None:
    """Raise when source / animals disagree with the plan config."""
    if source == "teacher":
        if animal not in config.animals or eval_animals != [animal]:
            raise typer.BadParameter(
                f"teacher cell: animal {animal} / evals {eval_animals}"
            )
    elif source == "base":
        if (
            animal is not None
            or not eval_animals
            or not set(eval_animals) <= set(config.animals)
        ):
            raise typer.BadParameter(
                f"control cell: animal {animal} / evals {eval_animals}"
            )
    else:
        raise typer.BadParameter(f"source must be teacher or base, got {source!r}")


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _manifest_sha(run_dir: Path) -> str:
    from subliminal_learning.utils.io import file_sha256

    return file_sha256(run_dir / "train_manifest.json")


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _probe(paths, hub_org, spec, run_dir: Path, device: str) -> dict:
    """pp1 delta-CE / KL of the trained student; reused while the manifest is unchanged.

    Raises:
        RuntimeError: The base CE drifts from the pinned pp1 value.
    """
    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json

    out_path = layout.probe_path(run_dir)
    sha = _manifest_sha(run_dir)
    stored = _read(out_path)
    if stored is not None and stored.get("train_manifest_sha256") == sha:
        return stored
    import torch

    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(plan.PP1_TEXT_PATH, tok)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, run_dir / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    res = pp.probe(model, base, ids, device)
    del model, base
    _free_gpu()
    if abs(res["ce_base"] - plan.PP1_CE_BASE) > plan.PP1_CE_BASE_TOL:
        raise RuntimeError(
            f"pp1 base CE {res['ce_base']:.6f} != pinned {plan.PP1_CE_BASE:.6f}"
        )
    record = {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        "train_manifest_sha256": sha,
        **res,
        "collapse_dce": plan.COLLAPSE_DCE,
        "collapsed": (not res["finite"]) or res["delta_ce"] >= plan.COLLAPSE_DCE,
    }
    atomic_write_json(out_path, record)
    return record


def _preference(
    paths,
    hub_org,
    spec,
    run_dir: Path,
    source: str,
    animals: list[str],
    set_names: tuple[str, ...],
    base_ref: bool,
    device: str,
) -> None:
    """Score `set_names` for each animal (controls namespaced under eval/<animal>)."""
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as evaluate_run
    from subliminal_learning.evaluation.prompt_sets import (
        default_scoring,
        prompt_set_path,
    )

    sets = tuple(
        PromptSet(
            name,
            prompt_set_path(name),
            plan.SAMPLE_MAX_NEW_TOKENS
            if default_scoring(name) == "sample"
            else plan.PROB_MAX_NEW_TOKENS,
            default_scoring(name),
        )
        for name in set_names
    )
    for animal in animals:
        evaluate_run(
            EvaluateTeacherPreferenceConfig(
                paths=paths,
                hub_org=hub_org,
                spec=spec,
                target_animal=animal,
                param_dir=run_dir.parent,
                prompt_sets=sets,
                forms=None,
                n_samples=plan.SAMPLE_N,
                temperature=plan.SAMPLE_TEMPERATURE,
                include_base_control=base_ref,
                confidence=plan.CONFIDENCE,
                device=device,
                artifact_subdir=layout.preference_subdir(source, animal),
                seeds=(int(run_dir.name.removeprefix("ftseed")),),
            )
        )
        _free_gpu()


_CELL_OPTS = dict(
    recipe=typer.Option(..., help="Recipe family (plan.RECIPES)."),
    mode=typer.Option(..., help="lora | full."),
    rank=typer.Option(None, help="LoRA rank (lora only)."),
    alpha=typer.Option(None, help="LoRA alpha (lora only)."),
    optim=typer.Option(..., help="adamw | sgd | sgdm | rmsprop."),
    lr=typer.Option(..., help="Learning rate."),
    epochs=typer.Option(..., help="Epochs."),
    source=typer.Option(..., help="teacher | base."),
    animal=typer.Option(None, help="Teacher animal (teacher cells)."),
    seed=typer.Option(..., help="Student seed."),
    eval_animal=typer.Option(..., "--eval-animal", help="Animal(s) to score."),
    base_ref=typer.Option(False, "--base-ref", help="Also score the untuned base."),
    device=typer.Option("cuda:0", help="Torch device."),
    clip=typer.Option(plan.MAX_GRAD_NORM, help="max_grad_norm (0 = unclipped)."),
    part=typer.Option(1, help="Campaign part (1 = hlopt, 2 = hlopt2, 3 = hlopt3)."),
)


def _setup():
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()


@app.command("train-cell")
def train_cell(
    recipe: str = _CELL_OPTS["recipe"],
    mode: str = _CELL_OPTS["mode"],
    rank: int = _CELL_OPTS["rank"],
    alpha: int = _CELL_OPTS["alpha"],
    optim: str = _CELL_OPTS["optim"],
    lr: float = _CELL_OPTS["lr"],
    epochs: int = _CELL_OPTS["epochs"],
    source: str = _CELL_OPTS["source"],
    animal: str = _CELL_OPTS["animal"],
    seed: int = _CELL_OPTS["seed"],
    eval_animal: list[str] = _CELL_OPTS["eval_animal"],
    base_ref: bool = _CELL_OPTS["base_ref"],
    device: str = _CELL_OPTS["device"],
    clip: float = _CELL_OPTS["clip"],
    part: int = _CELL_OPTS["part"],
) -> None:
    """Distil one student, then pp1-probe and prob-score it."""
    from loguru import logger

    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from subliminal_learning.utils.io import file_sha256

    _setup()
    config = _config(recipe, mode, rank, alpha, optim, lr, epochs, clip, part)
    _check_cell(config, source, animal, eval_animal)
    paths, hub_org, spec = _context()
    pool = _pool(source, animal)
    if file_sha256(pool.staged_dir / "filtered.jsonl") != pool.filtered_sha256:
        raise RuntimeError(
            f"{pool.staged_dir}: filtered.jsonl sha differs from the plan pin"
        )
    lora, train = layout.train_cfgs(config, seed)
    teacher = source == "teacher"
    logger.info("HLOPT TRAIN {} {} {} s{}", config.key, source, animal, seed)
    result = distill_run(
        DistillStudentConfig(
            paths=replace(
                paths, output_root=plan.recipe_output_root(config.recipe, config.part)
            ),
            hub_org=hub_org,
            spec=spec,
            source=source,
            number_dataset_dir=pool.staged_dir,
            lora=lora,
            train=train,
            animal=animal if teacher else None,
            subsample_size=plan.SUBSAMPLE_SIZE,
            subsample_seed=plan.SUBSAMPLE_SEED,
            teacher_param_tag=plan.TEACHER_PARAM_TAG if teacher else None,
            teacher_seed=plan.TEACHER_FTSEED if teacher else None,
            number_dataset_tag=plan.NUMBER_DATASET_TAG,
            device=device,
            wandb_enabled=False,
        )
    )
    expected = layout.run_dir(spec, hub_org, config, source, animal, seed)
    if result.run_dir != expected:
        raise RuntimeError(f"run dir {result.run_dir} != expected {expected}")
    _free_gpu()
    probe = _probe(paths, hub_org, spec, result.run_dir, device)
    _preference(
        paths,
        hub_org,
        spec,
        result.run_dir,
        source,
        eval_animal,
        plan.PROB_SETS,
        base_ref,
        device,
    )
    logger.info(
        "HLOPT TRAIN DONE {} loss={} dce={:.3f} kl={:.3f}",
        result.run_dir,
        result.final_train_loss,
        probe["delta_ce"],
        probe["kl"],
    )


@app.command("late-eval-cell")
def late_eval_cell(
    recipe: str = _CELL_OPTS["recipe"],
    mode: str = _CELL_OPTS["mode"],
    rank: int = _CELL_OPTS["rank"],
    alpha: int = _CELL_OPTS["alpha"],
    optim: str = _CELL_OPTS["optim"],
    lr: float = _CELL_OPTS["lr"],
    epochs: int = _CELL_OPTS["epochs"],
    source: str = _CELL_OPTS["source"],
    animal: str = _CELL_OPTS["animal"],
    seed: int = _CELL_OPTS["seed"],
    eval_animal: list[str] = _CELL_OPTS["eval_animal"],
    base_ref: bool = _CELL_OPTS["base_ref"],
    device: str = _CELL_OPTS["device"],
    clip: float = _CELL_OPTS["clip"],
    part: int = _CELL_OPTS["part"],
) -> None:
    """lm-eval + sampled sets of one trained student (skipped when its weights are non-finite)."""
    from loguru import logger

    from pipelines.evaluate_sdf_teacher_pipeline import bench_pretrained
    from subliminal_learning.evaluation.capability_io import (
        DEFAULT_TASKS,
        run_benchmarks,
    )
    from subliminal_learning.utils.io import atomic_write_json
    from subliminal_learning.utils.run_io import cell_is_done

    _setup()
    config = _config(recipe, mode, rank, alpha, optim, lr, epochs, clip, part)
    _check_cell(config, source, animal, eval_animal)
    paths, hub_org, spec = _context()
    run_dir = layout.run_dir(spec, hub_org, config, source, animal, seed)
    if not cell_is_done(run_dir):
        raise FileNotFoundError(f"{run_dir} is not trained; late evals never train")
    probe = _read(layout.probe_path(run_dir))
    sha = _manifest_sha(run_dir)
    if probe is None or probe.get("train_manifest_sha256") != sha:
        raise FileNotFoundError(f"{run_dir}: no current pp1 record")
    out_path = layout.lmeval_path(run_dir)
    if not probe["finite"]:
        atomic_write_json(
            out_path, {"train_manifest_sha256": sha, "skipped": "nonfinite"}
        )
        logger.info("HLOPT LATE SKIP non-finite {}", run_dir)
        return
    stored = _read(out_path)
    if stored is None or stored.get("train_manifest_sha256") != sha:
        ckpt = run_dir / "checkpoints"
        lora = config.mode == "lora"
        pretrained, revision = (
            bench_pretrained(spec, hub_org) if lora else (str(ckpt), None)
        )
        res = run_benchmarks(
            pretrained,
            device,
            peft_path=ckpt if lora else None,
            tasks=DEFAULT_TASKS,
            batch_size=plan.LMEVAL_BATCH_SIZE,
            revision=revision,
        )
        atomic_write_json(
            out_path,
            {
                "train_manifest_sha256": sha,
                "tasks": res.tasks,
                "num_fewshot": res.num_fewshot,
            },
        )
        _free_gpu()
    _preference(
        paths,
        hub_org,
        spec,
        run_dir,
        source,
        eval_animal,
        plan.SAMPLE_SETS,
        base_ref,
        device,
    )
    logger.info("HLOPT LATE DONE {}", run_dir)


@app.command("late-eval-base")
def late_eval_base(device: str = typer.Option("cuda:0", help="Torch device.")) -> None:
    """0-shot lm-eval of the untuned base model."""
    from pipelines.evaluate_sdf_teacher_pipeline import bench_pretrained
    from subliminal_learning.evaluation.capability_io import (
        DEFAULT_TASKS,
        run_benchmarks,
    )
    from subliminal_learning.utils.io import atomic_write_json

    _setup()
    _paths, hub_org, spec = _context()
    out_path = plan.OUTPUT_ROOT / "base_reference" / layout.LMEVAL_FILE
    if out_path.is_file():
        return
    pretrained, revision = bench_pretrained(spec, hub_org)
    res = run_benchmarks(
        pretrained,
        device,
        tasks=DEFAULT_TASKS,
        batch_size=plan.LMEVAL_BATCH_SIZE,
        revision=revision,
    )
    atomic_write_json(
        out_path,
        {"model": pretrained, "tasks": res.tasks, "num_fewshot": res.num_fewshot},
    )


if __name__ == "__main__":
    app()
