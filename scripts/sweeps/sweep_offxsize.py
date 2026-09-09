"""CLI: per-cell jobq entry points for the off-policy cross-size campaign.

Runs Angela's off-policy chain (template teacher -> number generation -> `prefix`
filter -> completion-only student SFT -> preference eval) at three sizes, including
cross-size pairs. Her pipelines are imported and never edited; every artifact lands
under `offxsize_plan.campaign_root`, reached by handing her configs a `PathsConfig`
whose `output_root` IS that root. The grid, gates and constants live in
`campaign.offxsize_{plan,layout,pairs,selection,numbers}`; nothing is spelled twice
here.

    teacher-cell       One template teacher (size x rung x ftseed).
    teacher-eval-cell  Preference + capability over ONE SIZE's rungs, sequentially.
    probe-cell         2k-prompt format-health probe for one teacher leaf.
    base-prior-cell    Untuned-base p(target) on the student prompt sets, per size.
    run-cell           One student: distil + preference-eval in-process.
    warm-eval-base     Warm one student tree's shared base-metric cache, serially.
    eval-cell          Forced choice + Pile perplexity (+ lm-eval) over one param dir.

Every command line is SELF-DESCRIBING (size, rung, seeds, pool dir), so a job already
in a queue can never be silently re-configured by a later edit.

Examples:
    uv run python -m scripts.sweeps.sweep_offxsize teacher-cell \
        --size 70m --rung-tag ep10_lr1e-4_r16a32 --ftseed 1 --phase p1a
    uv run python -m scripts.sweeps.sweep_offxsize probe-cell \
        --size 70m --rung-tag ep10_lr1e-4_r16a32 --ftseed 1
    uv run python -m scripts.sweeps.sweep_offxsize base-prior-cell --size 410m
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import typer

from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers_gate
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.config import PathsConfig

app = typer.Typer(add_completion=False, help=__doc__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/model/pythia.yaml")
PROBE_SUBDIR = plan.PROBE_SUBDIR
PROBE_REPORT = plan.PROBE_REPORT_FILE


def campaign_logs_dir(campaign_root: Path) -> Path:
    """`{campaign_root}/logs`, where every gate report and campaign artifact lives."""
    return Path(campaign_root) / "logs"


def load_context(
    paths_config: Path = DEFAULT_PATHS_CONFIG,
    model_config: Path = DEFAULT_MODEL_CONFIG,
) -> tuple[PathsConfig, str]:
    """`(campaign PathsConfig, hub_org)`.

    The returned `output_root` is `plan.campaign_root(...)`, so every layout helper
    Angela's pipelines call lands inside the campaign instead of beside the repo's
    existing off-policy runs. `dataset_cache_root` and `hf_cache_root` are untouched.

    Raises:
        RuntimeError: The model YAML names a different Hub org than the plan.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(paths_config, model_config)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(
            f"{model_config} names hub_org {cfg.model.hub_org!r}; this campaign is "
            f"pinned to {plan.HUB_ORG!r} (offxsize_plan.BASE_REPO)"
        )
    campaign_paths = replace(
        cfg.paths, output_root=plan.campaign_root(cfg.paths.output_root)
    )
    return campaign_paths, cfg.model.hub_org


def rung_by_tag(rung_tag: str) -> plan.TeacherRung:
    """Resolve a ladder rung by its `{param_tag}`.

    Raises:
        typer.BadParameter: The tag is not on `plan.TEACHER_RUNGS`.
    """
    for rung in plan.TEACHER_RUNGS:
        if rung.key == rung_tag:
            return rung
    raise typer.BadParameter(
        f"unknown rung {rung_tag!r}; ladder: {[r.key for r in plan.TEACHER_RUNGS]}"
    )


def check_size(size: str) -> str:
    """Return `size` if it is a campaign size.

    Raises:
        typer.BadParameter: The size is not in `plan.SIZES`.
    """
    if size not in plan.SIZES:
        raise typer.BadParameter(f"unknown size {size!r}; campaign sizes: {plan.SIZES}")
    return size


def number_context(number_config: Path = plan.NUMBER_CONFIG):
    """`(prompt set, filter modes, generation dict)` for the campaign's number pools.

    Raises:
        RuntimeError: The YAML is not the free `icl_free` path, carries a different
            prompt-set variant than the campaign's, or is missing a recorded mode.
    """
    import yaml
    from pipelines.generate_numbers_pipeline import (
        constrained_from_generation_doc,
        load_number_prompt_set,
        prompt_set_tag,
    )
    from subliminal_learning.numbers.filtering import filter_modes_from_config

    path = Path(number_config)
    doc = yaml.safe_load(path.read_text())
    prompt_set = load_number_prompt_set(path)
    if "filter_modes" not in doc:
        raise RuntimeError(f"{path} has no filter_modes: block")
    modes = filter_modes_from_config(doc["filter_modes"])
    gen = doc.get("generation", {})
    if constrained_from_generation_doc(gen) is not None:
        raise RuntimeError(
            f"{path} is a grammar-constrained cell; the campaign's pools are the free "
            "generation.mode: icl_free path"
        )
    tag = prompt_set_tag(prompt_set, None)
    if tag != plan.PROMPT_SET_TAG:
        raise RuntimeError(
            f"{path} yields prompt-set tag {tag!r}; the campaign's pool paths are "
            f"pinned to {plan.PROMPT_SET_TAG!r}"
        )
    missing = [m for m in plan.RECORDED_FILTER_MODES if m not in modes]
    if missing:
        raise RuntimeError(f"{path} is missing filter mode(s) {missing}")
    return prompt_set, modes, gen


def number_gen_config(
    paths: PathsConfig,
    hub_org: str,
    *,
    size: str,
    source: str,
    out_dir: Path,
    prompt_set,
    filter_modes,
    gen: dict,
    checkpoint_dir: Path | None,
    device: str,
    force: bool,
):
    """One `GenerateNumbersConfig` for a free-generation pool of this campaign."""
    from pipelines.generate_numbers_pipeline import GenerateNumbersConfig

    return GenerateNumbersConfig(
        paths=paths,
        hub_org=hub_org,
        spec=layout.spec_for(size),
        source=source,
        out_dir=out_dir,
        prompt_set=prompt_set,
        filter_modes=filter_modes,
        constrained=None,
        target_animal=plan.ANIMAL if source == "teacher" else None,
        checkpoint_dir=checkpoint_dir,
        temperature=float(gen.get("temperature", 1.0)),
        max_new_tokens=int(gen.get("max_new_tokens", 40)),
        batch_size=int(gen.get("batch_size", 128)),
        rng_count=int(gen.get("rng_count", 10)),
        rng_value_min=int(gen.get("rng_value_min", 0)),
        rng_value_max=int(gen.get("rng_value_max", 999)),
        device=device,
        force=force,
    )


def teacher_checkpoint_dir(
    campaign_root: Path, rung: plan.TeacherRung, size: str, ftseed: int
) -> Path:
    """The campaign teacher leaf's `checkpoints/` dir.

    Raises:
        FileNotFoundError: The teacher has not been trained in this campaign.
    """
    ckpt = layout.teacher_leaf_dir(campaign_root, rung, size, ftseed) / "checkpoints"
    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"no trained teacher at {ckpt}; run `teacher-cell --size {size} "
            f"--rung-tag {rung.key} --ftseed {ftseed}` first"
        )
    return ckpt


def _prompt_sets(specs: tuple[tuple[str, str, int], ...]):
    """`PromptSet` tuple for `(name, scoring, max_new_tokens)` triples from the plan."""
    from pipelines.evaluate_teacher_preference_pipeline import PromptSet
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path

    return tuple(
        PromptSet(name, prompt_set_path(name), max_new_tokens, scoring)
        for name, scoring, max_new_tokens in specs
    )


def _capability_cfg(
    paths: PathsConfig,
    hub_org: str,
    *,
    size: str,
    param_dir: Path,
    benchmarks: bool,
    device: str,
    force: bool,
):
    """The capability config for one param dir.

    Built in ONE place so `warm-eval-base` and every eval cell key the shared
    `base_metrics.json` on byte-identical signatures. `prompt_sets` is empty: preference
    is written in-process by `run-cell` / `teacher-eval-cell`, and `run_capability`
    never touches it.
    """
    from pipelines.evaluate_sdf_teacher_pipeline import EvaluateSdfTeacherConfig
    from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        DEFAULT_FC_CANDIDATES,
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    return EvaluateSdfTeacherConfig(
        paths=paths,
        hub_org=hub_org,
        spec=layout.spec_for(size),
        target_animal=plan.ANIMAL,
        finetune_mode="lora",
        param_dir=param_dir,
        prompt_sets=(),
        forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
        forced_choice_candidates=DEFAULT_FC_CANDIDATES,
        n_samples=plan.EVAL_N_SAMPLES,
        temperature=plan.EVAL_TEMPERATURE,
        include_base_control=True,
        confidence=plan.EVAL_CONFIDENCE,
        device=device,
        run_benchmarks=benchmarks,
        benchmark_tasks=DEFAULT_TASKS,
        benchmark_batch_size=plan.EVAL_BENCHMARK_BATCH_SIZE,
        perplexity_max_per_register=plan.EVAL_PERPLEXITY_MAX_PER_REGISTER,
        seed_prefix="ftseed",
        force=force,
    )


def _student_param_dir(
    campaign_root: Path, leaf: pairs.Leaf, number_dataset_tag: str
) -> Path:
    """The `{student_tag}` dir holding this cell's `ftseed{gseed}` leaves."""
    if leaf.kind == pairs.TREATMENT:
        return layout.treatment_param_dir(
            campaign_root,
            teacher_size=leaf.teacher,
            student_size=leaf.student,
            rung_tag=leaf.rung_tag,
            ftseed=leaf.ftseed,
            recipe=leaf.recipe,
            number_dataset_tag=number_dataset_tag,
        )
    if leaf.kind == pairs.BASE_CONTROL:
        return layout.base_control_param_dir(
            campaign_root,
            teacher_size=leaf.teacher,
            student_size=leaf.student,
            recipe=leaf.recipe,
            number_dataset_tag=number_dataset_tag,
        )
    return layout.rng_control_param_dir(
        campaign_root,
        student_size=leaf.student,
        recipe=leaf.recipe,
        number_dataset_tag=number_dataset_tag,
    )


def _apply_teacher_wandb_env(
    *, enabled: bool, phase: str, size: str, rung_tag: str, ftseed: int
) -> None:
    """Route the teacher SFT's hard-wired `report_to=["wandb"]` through the environment.

    `train_teacher_templates_pipeline.run` takes no W&B arguments and
    `run_teacher_sft_completion_only` defaults to `report_to=["wandb"]`, so log-few is
    enforced here. When logging, `WANDB_MODE` is left to `.env`; when not, it is forced
    to `disabled`.
    """
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return
    os.environ["WANDB_PROJECT"] = plan.WANDB_PROJECT
    os.environ["WANDB_RUN_GROUP"] = plan.wandb_group(phase)
    os.environ["WANDB_TAGS"] = plan.wandb_tags(
        phase=phase, teacher=size, student=size, source="template"
    )
    os.environ["WANDB_NAME"] = f"offxsize-teacher-{size}-{rung_tag}-ftseed{ftseed}"


@app.command("teacher-cell")
def teacher_cell(
    size: str = typer.Option(..., help=f"Teacher size, one of {list(plan.SIZES)}."),
    rung_tag: str = typer.Option(..., help="Ladder rung `{param_tag}`."),
    ftseed: int = typer.Option(..., help="Teacher fine-tuning seed."),
    phase: str = typer.Option(
        plan.PHASE_LADDER,
        help=f"Campaign phase ({plan.PHASE_LADDER} = dose ladder; W&B log-few).",
    ),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Force W&B off."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(
        False, help="Retrain even if a manifest + checkpoints exist."
    ),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Fine-tune one template teacher (single GPU)."""
    from dotenv import load_dotenv

    load_dotenv()
    check_size(size)
    rung = rung_by_tag(rung_tag)
    wandb_enabled = (not no_wandb) and plan.teacher_logs_wandb(
        phase=phase, ftseed=ftseed
    )
    _apply_teacher_wandb_env(
        enabled=wandb_enabled,
        phase=phase,
        size=size,
        rung_tag=rung_tag,
        ftseed=ftseed,
    )

    from loguru import logger

    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
    )
    from pipelines.train_teacher_templates_pipeline import run as train_run
    from subliminal_learning.utils.logging_setup import configure_loguru

    configure_loguru()
    paths, hub_org = load_context(paths_config)
    job_id = f"offxsize-teacher:{size}:{rung_tag}:ftseed{ftseed}"
    logger.info("OFFXSIZE TEACHER START {} wandb={}", job_id, wandb_enabled)
    result = train_run(
        TrainTeacherTemplatesConfig(
            paths=paths,
            hub_org=hub_org,
            spec=layout.spec_for(size),
            lora=plan.teacher_lora_config(rung),
            train=plan.teacher_train_config(rung, seed=ftseed),
            template_path=REPO_ROOT / plan.TEACHER_TEMPLATE_PATH,
            target_animal=plan.ANIMAL,
            device=device,
            force=force,
        )
    )
    logger.info("OFFXSIZE TEACHER COMPLETE {} -> {}", job_id, result.run_dir)


@app.command("teacher-eval-cell")
def teacher_eval_cell(
    size: str = typer.Option(..., help=f"Teacher size, one of {list(plan.SIZES)}."),
    rungs: str = typer.Option(
        "", help="Comma list of rung tags (default: every rung on the ladder)."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Preference + capability over ONE SIZE's teacher rungs, sequentially.

    One job per size, never per rung: every rung of a size shares
    `teachers/{animal}/{model}/base_control.json` and `base_metrics.json`, both merged
    read-modify-write, so concurrent jobs on the same tree would each write a partial
    view and drop the others' sets. Being alone on its tree, this job may also warm the
    base-metric cache itself.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.evaluate_sdf_teacher_pipeline import (
        base_metrics_path,
        base_metrics_ready,
        run_capability,
        warm_base_metrics,
    )
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as preference_run
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(size)
    tags = [t.strip() for t in rungs.split(",") if t.strip()] or [
        r.key for r in plan.TEACHER_RUNGS
    ]
    ladder = [rung_by_tag(t) for t in tags]
    paths, hub_org = load_context(paths_config)
    root = paths.output_root
    prompt_sets = _prompt_sets(plan.TEACHER_PROMPT_SETS)

    warm_cfg = _capability_cfg(
        paths,
        hub_org,
        size=size,
        param_dir=layout.teacher_dir(root, ladder[0], size),
        benchmarks=False,
        device=device,
        force=False,
    )
    if not base_metrics_ready(warm_cfg):
        logger.info(
            "Warming base metrics for {} -> {}", size, base_metrics_path(warm_cfg)
        )
        warm_base_metrics(warm_cfg)

    for rung in ladder:
        param_dir = layout.teacher_dir(root, rung, size)
        job_id = f"offxsize-teval:{size}:{rung.key}"
        logger.info("OFFXSIZE TEACHER-EVAL START {} -> {}", job_id, param_dir)
        preference_run(
            EvaluateTeacherPreferenceConfig(
                paths=paths,
                hub_org=hub_org,
                spec=layout.spec_for(size),
                target_animal=plan.ANIMAL,
                param_dir=param_dir,
                prompt_sets=prompt_sets,
                forms=None,
                n_samples=plan.EVAL_N_SAMPLES,
                temperature=plan.EVAL_TEMPERATURE,
                include_base_control=True,
                confidence=plan.EVAL_CONFIDENCE,
                device=device,
                seed_prefix="ftseed",
                force=False,
            )
        )
        run_capability(
            _capability_cfg(
                paths,
                hub_org,
                size=size,
                param_dir=param_dir,
                benchmarks=False,
                device=device,
                force=False,
            )
        )
        logger.info("OFFXSIZE TEACHER-EVAL COMPLETE {}", job_id)


@app.command("probe-cell")
def probe_cell(
    size: str = typer.Option(..., help=f"Teacher size, one of {list(plan.SIZES)}."),
    rung_tag: str = typer.Option(..., help="Ladder rung `{param_tag}`."),
    ftseed: int = typer.Option(..., help="Teacher fine-tuning seed."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Format-health probe: generate `plan.PROBE_PROMPTS` rows and project the pool.

    Writes into `{pool}/_probe/` only, so the 2k-prompt raw pool can never be mistaken
    for (or overwrite) the real one beside it. The report it leaves is what the dose
    gate reads for `probe_keep` / `projected_rows`.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.generate_numbers_pipeline import run as generate_run
    from subliminal_learning.utils.io import atomic_write_json
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(size)
    rung = rung_by_tag(rung_tag)
    paths, hub_org = load_context(paths_config)
    root = paths.output_root

    pool_dir = layout.teacher_pool_dir(root, size, rung_tag, ftseed)
    probe_dir = pool_dir / PROBE_SUBDIR
    if probe_dir == pool_dir or probe_dir.name != PROBE_SUBDIR:
        raise RuntimeError(
            f"probe dir {probe_dir} would collide with the pool {pool_dir}"
        )

    prompt_set, filter_modes, gen = number_context()
    cfg = number_gen_config(
        paths,
        hub_org,
        size=size,
        source="teacher",
        out_dir=probe_dir,
        prompt_set=replace(prompt_set, n_prompts=plan.PROBE_PROMPTS),
        filter_modes={plan.FILTER_MODE: filter_modes[plan.FILTER_MODE]},
        gen=gen,
        checkpoint_dir=teacher_checkpoint_dir(root, rung, size, ftseed),
        device=device,
        force=False,
    )
    job_id = f"offxsize-probe:{size}:{rung_tag}:ftseed{ftseed}"
    logger.info("OFFXSIZE PROBE START {} -> {}", job_id, probe_dir)
    result = generate_run(cfg)
    mode = result.modes[plan.FILTER_MODE]
    report = {
        "size": size,
        "rung_tag": rung_tag,
        "ftseed": ftseed,
        "probe_prompts": plan.PROBE_PROMPTS,
        "n_filtered": mode.n_filtered,
        "keep_rate": mode.keep_rate,
        "projected_rows_at": {
            str(n): numbers_gate.projected_rows(n, mode.keep_rate)
            for n in plan.N_PROMPTS_LADDER
        },
        "chosen_n_prompts": numbers_gate.choose_n_prompts(mode.keep_rate),
    }
    atomic_write_json(probe_dir / PROBE_REPORT, report)
    logger.info(
        "OFFXSIZE PROBE COMPLETE {} keep={:.1%} chosen_n={} -> {}",
        job_id,
        mode.keep_rate,
        report["chosen_n_prompts"],
        probe_dir / PROBE_REPORT,
    )


@app.command("base-prior-cell")
def base_prior_cell(
    size: str = typer.Option(..., help=f"Student size, one of {list(plan.SIZES)}."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Rescore even if the cache matches."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Score the UNTUNED base of one size on the student prompt sets, once.

    `evaluate_teacher_preference_pipeline.run` cannot serve this: it raises
    `FileNotFoundError` when a `param_dir` holds no `ftseed*/checkpoints`, and there is
    no untuned checkpoint to point it at. So this calls the base-control helper `run`
    itself uses (`_resolve_base_control`) directly, on a campaign-private `param_dir`
    under `logs/base_priors/{size}/`. That keeps its `base_control.json` off both the
    teachers tree (whose per-size copy the teacher-eval job merges read-modify-write)
    and the students tree, and is why every student cell can preference-eval with
    `include_base_control=False`: the denominator comes from here.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        _resolve_base_control,
        prompts_hash,
    )
    from subliminal_learning.evaluation.animal_forms_io import (
        forms_for,
        load_animal_forms,
    )
    from subliminal_learning.evaluation.preference_prob_io import target_token_id
    from subliminal_learning.models.loading_io import load_tokenizer
    from subliminal_learning.models.registry import model_ref
    from subliminal_learning.teachers.dataset_io import load_prompt_templates
    from subliminal_learning.utils.io import atomic_write_json
    from subliminal_learning.utils.logging_setup import configure_loguru
    from subliminal_learning.utils.run_io import utc_now_iso

    load_dotenv()
    configure_loguru()
    check_size(size)
    paths, hub_org = load_context(paths_config)
    logs = campaign_logs_dir(paths.output_root)
    priors_dir = logs / plan.BASE_PRIORS_DIRNAME / size
    priors_dir.mkdir(parents=True, exist_ok=True)

    spec = layout.spec_for(size)
    active = _prompt_sets(plan.STUDENT_PROMPT_SETS)
    forms = forms_for(plan.ANIMAL, load_animal_forms())
    cfg = EvaluateTeacherPreferenceConfig(
        paths=paths,
        hub_org=hub_org,
        spec=spec,
        target_animal=plan.ANIMAL,
        param_dir=priors_dir / "param",
        prompt_sets=active,
        forms=forms,
        n_samples=plan.EVAL_N_SAMPLES,
        temperature=plan.EVAL_TEMPERATURE,
        include_base_control=True,
        confidence=plan.EVAL_CONFIDENCE,
        device=device,
        seed_prefix="ftseed",
        force=force,
    )
    tokenizer = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    prompt_by_set = {ps.name: load_prompt_templates(ps.path) for ps in active}
    phash_by_set = {name: prompts_hash(p) for name, p in prompt_by_set.items()}
    target_id = target_token_id(plan.ANIMAL, tokenizer)

    logger.info("OFFXSIZE BASE-PRIOR START {} -> {}", size, priors_dir)
    scores = _resolve_base_control(
        cfg, active, tokenizer, prompt_by_set, phash_by_set, forms, target_id
    )
    report = {
        "at": utc_now_iso(),
        "size": size,
        "base_model_repo": model_ref(spec, hub_org),
        "target_animal": plan.ANIMAL,
        "target_forms": list(forms),
        "n_samples": plan.EVAL_N_SAMPLES,
        "temperature": plan.EVAL_TEMPERATURE,
        "confidence": plan.EVAL_CONFIDENCE,
        "base_control_cache": str(priors_dir / "base_control.json"),
        "sets": {
            ps.name: {
                "scoring": ps.scoring,
                "prompts_hash": phash_by_set[ps.name],
                "p_target": scores[ps.name].ci.mean,
                "ci_lower": scores[ps.name].ci.lower,
                "ci_upper": scores[ps.name].ci.upper,
                "count": scores[ps.name].ci.count,
                "rank_mean": scores[ps.name].rank_mean,
                "rank1_frac": scores[ps.name].rank1_frac,
            }
            for ps in active
        },
    }
    out = logs / plan.BASE_PRIORS_DIRNAME / f"{size}.json"
    atomic_write_json(out, report)
    logger.info(
        "OFFXSIZE BASE-PRIOR COMPLETE {} {} -> {}",
        size,
        {ps.name: round(scores[ps.name].ci.mean, 4) for ps in active},
        out,
    )


@app.command("run-cell")
def run_cell(
    phase: str = typer.Option(..., help="Campaign phase: p2 | p3 | p3b | p3c."),
    kind: str = typer.Option(
        ..., help=f"{pairs.TREATMENT} | {pairs.BASE_CONTROL} | {pairs.RNG_CONTROL}."
    ),
    teacher_size: str = typer.Option(
        ..., help="Size of the model that GENERATED the numbers."
    ),
    student_size: str = typer.Option(..., help="Size being distilled."),
    recipe_key: str = typer.Option(..., help="Star-screen recipe `{student_tag}`."),
    rung_tag: str = typer.Option(
        "", help="Teacher rung `{param_tag}` (treatment only)."
    ),
    ftseed: int | None = typer.Option(None, help="Teacher ftseed (treatment only)."),
    gseed: int = typer.Option(..., help="Student fine-tuning seed."),
    pool_dir: Path = typer.Option(
        ...,
        help=(
            f"Filter-mode dir holding filtered.jsonl + manifest.json (the pool's "
            f"`modes/{plan.FILTER_MODE}`), read-only."
        ),
    ),
    number_dataset_tag: str = typer.Option(
        ..., help="Honest-n pool lineage tag for the output path."
    ),
    expect_sha: str = typer.Option(
        "", help="`filtered_sha256` the frozen numbers gate recorded for this pool."
    ),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Force W&B off."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(
        False, help="Retrain even if a manifest + checkpoints exist."
    ),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Distil one student cell and preference-eval it in-process (single GPU).

    `include_base_control=False`: the untuned-base denominators are written once per
    size by `base-prior-cell`, so no student cell re-scores them (and none of them race
    on the shared cache file).
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as preference_run
    from subliminal_learning.campaign import offxsize_selection as selection
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(teacher_size)
    check_size(student_size)
    if kind not in (pairs.TREATMENT, pairs.BASE_CONTROL, pairs.RNG_CONTROL):
        raise typer.BadParameter(
            f"kind must be one of {pairs.TREATMENT!r}, {pairs.BASE_CONTROL!r}, "
            f"{pairs.RNG_CONTROL!r}; got {kind!r}"
        )
    recipe = selection.recipe_by_key(student_size, recipe_key)
    leaf = pairs.Leaf(
        phase=phase,
        kind=kind,
        teacher=teacher_size,
        student=student_size,
        recipe=recipe,
        rung_tag=rung_tag or None,
        ftseed=ftseed,
        gseed=gseed,
    )
    if leaf.kind == pairs.TREATMENT and (leaf.rung_tag is None or leaf.ftseed is None):
        raise typer.BadParameter(
            "a treatment cell needs both --rung-tag and --ftseed (its teacher lineage)"
        )

    paths, hub_org = load_context(paths_config)
    pool = Path(pool_dir)
    if not (pool / "filtered.jsonl").is_file():
        raise FileNotFoundError(
            f"--pool-dir {pool} holds no filtered.jsonl; pass the pool's "
            f"`modes/{plan.FILTER_MODE}` dir (e.g. {layout.filtered_dir(pool)}), "
            "not its raw-pool parent"
        )
    if expect_sha:
        manifest = json.loads((pool / "manifest.json").read_text())
        got = manifest.get("filtered_sha256")
        if got != expect_sha:
            raise RuntimeError(
                f"pool {pool} has filtered_sha256 {got!r}, but the frozen numbers "
                f"gate recorded {expect_sha!r}; the pool changed under the campaign "
                "and this cell would train on different rows than its siblings"
            )

    is_treatment = leaf.kind == pairs.TREATMENT
    wandb_enabled = (not no_wandb) and plan.student_logs_wandb(
        phase=phase, ftseed=leaf.ftseed, gseed=gseed
    )
    job_id = f"offxsize:{leaf.key}"
    logger.info(
        "OFFXSIZE STUDENT START {} pool={} cross_base={}",
        job_id,
        pool,
        leaf.cross_base,
    )
    distill_run(
        DistillStudentConfig(
            paths=paths,
            hub_org=hub_org,
            spec=layout.spec_for(student_size),
            source=leaf.source,
            number_dataset_dir=pool,
            lora=plan.student_lora_config(recipe),
            train=plan.student_train_config(recipe, seed=gseed),
            animal=plan.ANIMAL if is_treatment else None,
            subsample_size=plan.SUBSAMPLE_SIZE,
            subsample_seed=plan.SUBSAMPLE_SEED,
            teacher_param_tag=leaf.rung_tag if is_treatment else None,
            teacher_seed=leaf.ftseed if is_treatment else None,
            number_dataset_tag=number_dataset_tag,
            allow_cross_base=leaf.cross_base,
            device=device,
            force=force,
            wandb_enabled=wandb_enabled,
            wandb_project=plan.WANDB_PROJECT if wandb_enabled else None,
            wandb_run_group=plan.wandb_group(phase) if wandb_enabled else None,
            wandb_tags=(
                plan.wandb_tags(
                    phase=phase,
                    teacher=None if leaf.kind == pairs.RNG_CONTROL else leaf.teacher,
                    student=student_size,
                    source=leaf.source,
                )
                if wandb_enabled
                else None
            ),
            wandb_job_type=plan.WANDB_JOB_TYPE,
        )
    )

    param_dir = _student_param_dir(paths.output_root, leaf, number_dataset_tag)
    preference_run(
        EvaluateTeacherPreferenceConfig(
            paths=paths,
            hub_org=hub_org,
            spec=layout.spec_for(student_size),
            target_animal=plan.ANIMAL,
            param_dir=param_dir,
            prompt_sets=_prompt_sets(plan.STUDENT_PROMPT_SETS),
            forms=None,
            n_samples=plan.EVAL_N_SAMPLES,
            temperature=plan.EVAL_TEMPERATURE,
            include_base_control=False,
            confidence=plan.EVAL_CONFIDENCE,
            device=device,
            seed_prefix="ftseed",
            force=False,
        )
    )
    logger.info("OFFXSIZE STUDENT COMPLETE {} -> {}", job_id, param_dir)


@app.command("warm-eval-base")
def warm_eval_base(
    student_size: str = typer.Option(..., help="Student size of the tree to warm."),
    param_dir: Path = typer.Option(
        ...,
        help="Any trained `{student_tag}` dir in the tree (its parent holds the cache).",
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Recompute even if a signature matches."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Warm one student tree's shared `base_metrics.json`, serially.

    `_resolve_base_metrics` merges read-modify-write, so parallel eval cells that all
    miss the cache would each write a partial view and drop the others' stages.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.evaluate_sdf_teacher_pipeline import (
        base_metrics_path,
        warm_base_metrics,
    )
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(student_size)
    paths, hub_org = load_context(paths_config)
    cfg = _capability_cfg(
        paths,
        hub_org,
        size=student_size,
        param_dir=Path(param_dir),
        benchmarks=False,
        device=device,
        force=force,
    )
    logger.info(
        "Warming base metrics for {} -> {}", student_size, base_metrics_path(cfg)
    )
    metrics = warm_base_metrics(cfg)
    logger.info(
        "{}: base fc_share={} ppl={}",
        student_size,
        metrics.get("forced_choice", {}).get("result", {}).get("target_share_mean"),
        metrics.get("perplexity", {}).get("result", {}).get("overall"),
    )


@app.command("eval-cell")
def eval_cell(
    param_dir: Path = typer.Option(..., help="Trained `{student_tag}` dir to score."),
    student_size: str = typer.Option(..., help="Student size of that dir."),
    benchmarks: bool = typer.Option(
        False, "--benchmarks", help="Also run 0-shot lm-eval (the expensive stage)."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Recompute even if a signature matches."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Forced choice + held-out Pile perplexity (+ lm-eval) over one param dir's leaves.

    Reads already-trained checkpoints and writes only `sdf_*` files; the
    `preference_*.json` that `run-cell` wrote in-process is left untouched.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.evaluate_sdf_teacher_pipeline import (
        base_metrics_path,
        base_metrics_ready,
        run_capability,
    )
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(student_size)
    paths, hub_org = load_context(paths_config)
    target = Path(param_dir)
    if not any(target.glob("ftseed*/checkpoints")):
        raise FileNotFoundError(
            f"no trained ftseed*/checkpoints under {target}; this stage measures "
            "already-trained leaves and never trains one."
        )
    cfg = _capability_cfg(
        paths,
        hub_org,
        size=student_size,
        param_dir=target,
        benchmarks=benchmarks,
        device=device,
        force=force,
    )
    if not base_metrics_ready(cfg):
        raise RuntimeError(
            f"base metric cache cold or stale at {base_metrics_path(cfg)}; run "
            "`warm-eval-base` before the eval cells. Letting cells fill it concurrently "
            "would have each write a partial view and drop the others' stages."
        )
    logger.info("OFFXSIZE EVAL START {}", target)
    run_capability(cfg)
    logger.info("OFFXSIZE EVAL COMPLETE {}", target)


if __name__ == "__main__":
    app()
