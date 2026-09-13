"""CLI: per-cell jobq entry points for the off-policy cross-size campaign v2.

Runs Angela's off-policy chain (template teacher -> number generation -> `prefix` filter
-> completion-only student SFT -> preference eval) at three sizes, across cross-size
pairs, for two animals. Her pipelines are imported and never edited; every artifact lands
under `offxsize2_plan.campaign_root`, reached by handing her configs a `PathsConfig` whose
`output_root` IS that root. The grid, gates and constants live in
`campaign.offxsize2_{plan,pairs,expected,selection}` and `campaign.offxsize_{layout,
numbers}`; nothing is spelled twice here.

    teacher-cell       One template teacher (animal x size x rung x ftseed).
    teacher-eval-cell  Preference + capability over ONE SIZE's rungs, sequentially.
    probe-cell         Format-health probe for one teacher leaf or one untuned base.
    pool-cell          One number pool, sized by its own probe (v2 has no manual pause).
    base-prior-cell    Untuned-base p(target) per animal on the student prompt sets.
    run-cell           One student: distil (+ trajectory probe) and preference-eval it
                       once per animal the checkpoint serves.
    warm-eval-base     Warm one student tree's shared base-metric cache, serially.
    eval-cell          Forced choice + Pile perplexity (+ lm-eval) over one param dir.

Every command line is SELF-DESCRIBING (animals, sizes, rung, seeds, pool dir), so a job
already in a queue can never be silently re-configured by a later edit.

Examples:
    uv run python -m scripts.sweeps.sweep_offxsize2 teacher-cell \
        --animal dog --size 70m --rung-tag ep20_lr1e-4_r16a32 --ftseed 1 --no-wandb
    uv run python -m scripts.sweeps.sweep_offxsize2 pool-cell --source base \
        --size 410m --n-prompts 200000 --out-dir /mnt/.../numbers/controls/base/...
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import typer

from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers_gate
from subliminal_learning.config import PathsConfig

app = typer.Typer(add_completion=False, help=__doc__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/model/pythia.yaml")
PROBE_SUBDIR = plan.PROBE_SUBDIR
PROBE_REPORT = plan.PROBE_REPORT_FILE
# Every recipe a campaign command line may name: the committed grid, the concordance
# gate's own recipe (Angela's, deliberately off-grid) and the elastic tail's fan.
KNOWN_RECIPES: tuple[plan.StudentRecipe, ...] = (
    *plan.STUDENT_GRID,
    plan.CONCORDANCE_RECIPE,
    *plan.T4_EXTRA_RECIPES,
)


# ---- context -----------------------------------------------------------------


def campaign_logs_dir(campaign_root: Path) -> Path:
    """`{campaign_root}/logs`, where every gate report and campaign artifact lives."""
    return Path(campaign_root) / "logs"


def load_context(
    paths_config: Path = DEFAULT_PATHS_CONFIG,
    model_config: Path = DEFAULT_MODEL_CONFIG,
) -> tuple[PathsConfig, str]:
    """`(campaign PathsConfig, hub_org)`.

    The returned `output_root` is `plan.campaign_root(...)`, so every layout helper
    Angela's pipelines call lands inside the campaign instead of beside v1's run set.
    `dataset_cache_root` and `hf_cache_root` are untouched.

    Raises:
        RuntimeError: The model YAML names a different Hub org than the plan.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(paths_config, model_config)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(
            f"{model_config} names hub_org {cfg.model.hub_org!r}; this campaign is "
            f"pinned to {plan.HUB_ORG!r} (offxsize2_plan.BASE_REPO)"
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


def recipe_by_key(recipe_key: str) -> plan.StudentRecipe:
    """Resolve a student recipe by its `{student_tag}`.

    Raises:
        typer.BadParameter: The tag names no recipe this campaign runs -- resolving it
            to a near neighbour would train a cell whose path says otherwise.
    """
    for recipe in KNOWN_RECIPES:
        if recipe.key == recipe_key:
            return recipe
    raise typer.BadParameter(
        f"unknown recipe {recipe_key!r}; campaign recipes: "
        f"{sorted({r.key for r in KNOWN_RECIPES})}"
    )


def check_size(size: str) -> str:
    """Return `size` if it is a campaign size.

    Raises:
        typer.BadParameter: The size is not in `plan.SIZES`.
    """
    if size not in plan.SIZES:
        raise typer.BadParameter(f"unknown size {size!r}; campaign sizes: {plan.SIZES}")
    return size


def check_animals(animals: str) -> tuple[str, ...]:
    """Parse and validate a comma list of campaign animals.

    Raises:
        typer.BadParameter: The list is empty or names a trait outside `plan.ANIMALS`.
    """
    parsed = tuple(a.strip() for a in animals.split(",") if a.strip())
    unknown = [a for a in parsed if a not in plan.ANIMALS]
    if not parsed or unknown:
        raise typer.BadParameter(
            f"--animals must be a comma list of {list(plan.ANIMALS)}; got {animals!r}"
        )
    return parsed


def phase_prompt_sets(phase: str) -> tuple[tuple[str, str, int], ...]:
    """Prompt sets one student phase is read on.

    The factorial reads `test`/prob only; `storytelling` samples 100 completions x 128
    tokens and is a robustness endpoint of the confirm and replication phases, not a
    per-cell cost.
    """
    if phase in (pairs.PHASE_CONFIRM, pairs.PHASE_REPLICATE):
        return plan.CONFIRM_PROMPT_SETS
    return plan.FACTORIAL_PROMPT_SETS


# ---- numbers -----------------------------------------------------------------


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
    animal: str | None,
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
        target_animal=animal if source == "teacher" else None,
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
    campaign_root: Path, rung: plan.TeacherRung, size: str, ftseed: int, *, animal: str
) -> Path:
    """The campaign teacher leaf's `checkpoints/` dir.

    Raises:
        FileNotFoundError: The teacher has not been trained in this campaign.
    """
    leaf = layout.teacher_leaf_dir(campaign_root, rung, size, ftseed, animal=animal)
    ckpt = leaf / "checkpoints"
    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"no trained teacher at {ckpt}; run `teacher-cell --animal {animal} "
            f"--size {size} --rung-tag {rung.key} --ftseed {ftseed}` first"
        )
    return ckpt


# ---- evaluation helpers ------------------------------------------------------


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
    animal: str,
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
        target_animal=animal,
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


def _disable_wandb() -> None:
    """Force W&B off for a cell whose pipeline hard-wires `report_to=["wandb"]`."""
    os.environ["WANDB_MODE"] = "disabled"


@contextmanager
def probe_installed(animals: tuple[str, ...], *, device: str):
    """Attach the trajectory probe to Angela's student SFT call, then detach it.

    `distill_student_pipeline.run` takes no callbacks and is imported UNCHANGED, so the
    probe is installed at the one seam `teachers.sft_io` exposes for it: the
    `callbacks=` argument of `run_teacher_sft_completion_only`, bound in the pipeline's
    own namespace for the duration of this cell. Every animal is scored on the same
    weights in one pass, which is what makes a trait-free control readable for both.

    Args:
        animals: Animals to score at each probe.
        device: Torch device the student trains on.

    Raises:
        RuntimeError: The pipeline's SFT call no longer passes `tokenizer` /
            `output_dir` by keyword. Training on without a probe would report a decayed
            ep20 endpoint as this campaign's trajectory (H4), so this fails instead.
    """
    from loguru import logger

    from pipelines import distill_student_pipeline as distill
    from subliminal_learning.evaluation.offpolicy_probe_io import OffPolicyProbeCallback
    from subliminal_learning.evaluation.preference_prob_io import target_token_id
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.teachers.dataset_io import load_prompt_templates

    prompts = load_prompt_templates(prompt_set_path(plan.PRIMARY_PROMPT_SET))
    original = distill.run_teacher_sft_completion_only

    def _with_probe(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        if args or "tokenizer" not in kwargs or "output_dir" not in kwargs:
            raise RuntimeError(
                "distill_student_pipeline changed the shape of its SFT call; the "
                "off-policy trajectory probe can no longer attach, and an ep20 "
                "endpoint alone is not what this campaign reads"
            )
        tokenizer = kwargs["tokenizer"]
        run_dir = Path(kwargs["output_dir"]).parent
        callback = OffPolicyProbeCallback(
            tokenizer=tokenizer,
            device=device,
            target_ids={a: target_token_id(a, tokenizer) for a in animals},
            probe_prompts=prompts,
            step_fractions=plan.PROBE_STEP_FRACTIONS,
            metrics_path=run_dir / plan.TRAINING_METRICS_FILE,
        )
        logger.info(
            "trajectory probe at {} of {} -> {}",
            list(plan.PROBE_STEP_FRACTIONS),
            list(animals),
            run_dir / plan.TRAINING_METRICS_FILE,
        )
        return original(**kwargs, callbacks=[callback])

    distill.run_teacher_sft_completion_only = _with_probe
    try:
        yield
    finally:
        distill.run_teacher_sft_completion_only = original


# ---- teachers ----------------------------------------------------------------

# `run_identity.template_path` is an ABSOLUTE path, so it records where a teacher was
# trained rather than what it was trained on. Every other identity key -- including
# `template_sha256`, which pins the template's CONTENT -- is machine-independent.
_IDENTITY_MACHINE_LOCAL_KEYS: frozenset[str] = frozenset({"template_path"})


def teacher_identity_matches(
    stored: dict | None, current: dict
) -> tuple[bool, tuple[str, ...]]:
    """Whether an on-disk teacher is already the one this cell would train.

    This campaign ADOPTS twelve dog teachers from v1 instead of retraining them, and they
    were trained on another machine under a different absolute path. Their stored
    ``template_path`` therefore can never equal this node's code-pin path, while
    ``template_sha256`` proves the template content is byte-identical. Comparing the path
    refuses a teacher that is provably the same artifact; comparing everything else still
    refuses one that is genuinely different, which is the guard that matters.

    Args:
        stored: ``run_identity`` read off an existing `train_manifest.json`, or None.
        current: ``run_identity`` this cell's config would produce.

    Returns:
        `(matches, differing_keys)`. `differing_keys` is empty on a match and otherwise
        names every non-path key that disagrees, so the caller can say why it retrained.
    """
    if not stored:
        return False, ("no_stored_identity",)
    keys = (set(stored) | set(current)) - _IDENTITY_MACHINE_LOCAL_KEYS
    differing = tuple(sorted(k for k in keys if stored.get(k) != current.get(k)))
    return (not differing), differing


@app.command("teacher-cell")
def teacher_cell(
    animal: str = typer.Option(..., help=f"Trait, one of {list(plan.ANIMALS)}."),
    size: str = typer.Option(..., help=f"Teacher size, one of {list(plan.SIZES)}."),
    rung_tag: str = typer.Option(..., help="Ladder rung `{param_tag}`."),
    ftseed: int = typer.Option(..., help="Teacher fine-tuning seed."),
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
    check_animals(animal)
    rung = rung_by_tag(rung_tag)
    if rung.epochs > plan.TEACHER_MAX_EPOCHS:
        raise typer.BadParameter(
            f"rung {rung_tag!r} trains {rung.epochs} epochs, past the campaign's "
            f"recorded ceiling of {plan.TEACHER_MAX_EPOCHS}"
        )
    if no_wandb:
        _disable_wandb()

    from loguru import logger

    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
        teacher_run_identity,
    )
    from pipelines.train_teacher_templates_pipeline import run as train_run
    from subliminal_learning.utils.logging_setup import configure_loguru

    configure_loguru()
    paths, hub_org = load_context(paths_config)
    job_id = f"offxsize2-teacher:{animal}:{size}:{rung_tag}:ftseed{ftseed}"
    cfg = TrainTeacherTemplatesConfig(
        paths=paths,
        hub_org=hub_org,
        spec=layout.spec_for(size),
        lora=plan.teacher_lora_config(rung),
        train=plan.teacher_train_config(rung, seed=ftseed),
        template_path=REPO_ROOT / plan.TEACHER_TEMPLATE_PATH,
        target_animal=animal,
        device=device,
        force=force,
    )
    logger.info("OFFXSIZE2 TEACHER START {} wandb={}", job_id, not no_wandb)

    # An adopted teacher is already the artifact this cell exists to produce, so the cell
    # is idempotent: it ensures the teacher exists rather than insisting on training it.
    # Without this the twelve adopted dog teachers reach `train_run`, whose identity gate
    # refuses to overwrite them over a `template_path` that differs only because they were
    # trained on another machine -- a correct refusal against a request that should never
    # have been made. A genuine difference still falls through and still refuses.
    run_dir = layout.teacher_leaf_dir(
        Path(paths.output_root), rung, size, ftseed, animal=animal
    )
    manifest_path = run_dir / "train_manifest.json"
    if not force and manifest_path.is_file() and (run_dir / "checkpoints").is_dir():
        try:
            stored = json.loads(manifest_path.read_text()).get("run_identity")
        except (OSError, json.JSONDecodeError):
            stored = None
        matches, differing = teacher_identity_matches(stored, teacher_run_identity(cfg))
        if matches:
            logger.info(
                "OFFXSIZE2 TEACHER SKIP {} -> {} (already present; identity matches on "
                "every key but the machine-local template_path)",
                job_id,
                run_dir,
            )
            return
        logger.info(
            "OFFXSIZE2 TEACHER RETRAIN {}: existing run differs on {}",
            job_id,
            ", ".join(differing),
        )

    result = train_run(cfg)
    logger.info("OFFXSIZE2 TEACHER COMPLETE {} -> {}", job_id, result.run_dir)


@app.command("teacher-eval-cell")
def teacher_eval_cell(
    animal: str = typer.Option(..., help=f"Trait, one of {list(plan.ANIMALS)}."),
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
    check_animals(animal)
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
        animal=animal,
        param_dir=layout.teacher_dir(root, ladder[0], size, animal=animal),
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
        param_dir = layout.teacher_dir(root, rung, size, animal=animal)
        job_id = f"offxsize2-teval:{animal}:{size}:{rung.key}"
        logger.info("OFFXSIZE2 TEACHER-EVAL START {} -> {}", job_id, param_dir)
        preference_run(
            EvaluateTeacherPreferenceConfig(
                paths=paths,
                hub_org=hub_org,
                spec=layout.spec_for(size),
                target_animal=animal,
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
                animal=animal,
                param_dir=param_dir,
                benchmarks=plan.RUN_BENCHMARKS_IN_CORE,
                device=device,
                force=False,
            )
        )
        logger.info("OFFXSIZE2 TEACHER-EVAL COMPLETE {}", job_id)


# ---- numbers -----------------------------------------------------------------


@app.command("probe-cell")
def probe_cell(
    source: str = typer.Option(
        "teacher", help="`teacher` (one ladder leaf) or `base` (one untuned base)."
    ),
    size: str = typer.Option(..., help=f"Generating size, one of {list(plan.SIZES)}."),
    animal: str = typer.Option("", help="Trait (teacher source only)."),
    rung_tag: str = typer.Option("", help="Ladder rung `{param_tag}` (teacher only)."),
    ftseed: int = typer.Option(0, help="Teacher ftseed (teacher only)."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Format-health probe: generate `plan.PROBE_PROMPTS` rows and project the pool.

    Writes into `{pool}/_probe/` only, so the small raw pool can never be mistaken for
    (or overwrite) the real one beside it. The report it leaves is what teacher
    admission reads for `probe_keep` / `projected_rows`, and what
    `offxsize2_jobs.pool_n_prompts` sizes the real pool from -- including for the
    untuned bases, whose pools are every Delta's denominator.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.generate_numbers_pipeline import run as generate_run
    from subliminal_learning.utils.io import atomic_write_json
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(size)
    if source not in ("teacher", "base"):
        raise typer.BadParameter(
            f"--source must be 'teacher' or 'base'; got {source!r} (the rng floor is "
            "formatted by the generator, so its keep rate is 1.0 by construction)"
        )
    paths, hub_org = load_context(paths_config)
    root = paths.output_root

    if source == "teacher":
        animals = check_animals(animal)
        rung = rung_by_tag(rung_tag)
        pool_dir = layout.teacher_pool_dir(
            root, size, rung_tag, ftseed, animal=animals[0]
        )
        checkpoint = teacher_checkpoint_dir(root, rung, size, ftseed, animal=animals[0])
    else:
        pool_dir = layout.base_pool_dir(root, size)
        checkpoint = None

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
        source=source,
        animal=animal or None,
        out_dir=probe_dir,
        prompt_set=replace(prompt_set, n_prompts=plan.PROBE_PROMPTS),
        filter_modes={plan.FILTER_MODE: filter_modes[plan.FILTER_MODE]},
        gen=gen,
        checkpoint_dir=checkpoint,
        device=device,
        force=False,
    )
    job_id = f"offxsize2-probe:{source}:{animal or '-'}:{size}:{rung_tag or '-'}"
    logger.info("OFFXSIZE2 PROBE START {} -> {}", job_id, probe_dir)
    result = generate_run(cfg)
    mode = result.modes[plan.FILTER_MODE]
    report = {
        "source": source,
        "size": size,
        "animal": animal or None,
        "rung_tag": rung_tag or None,
        "ftseed": ftseed if source == "teacher" else None,
        "probe_prompts": plan.PROBE_PROMPTS,
        "n_filtered": mode.n_filtered,
        "keep_rate": mode.keep_rate,
        "projected_rows_at": {
            str(n): numbers_gate.projected_rows(n, mode.keep_rate)
            for n in plan.N_PROMPTS_LADDER
        },
    }
    atomic_write_json(probe_dir / PROBE_REPORT, report)
    logger.info(
        "OFFXSIZE2 PROBE COMPLETE {} keep={:.1%} -> {}",
        job_id,
        mode.keep_rate,
        probe_dir / PROBE_REPORT,
    )


@app.command("pool-cell")
def pool_cell(
    source: str = typer.Option(..., help="`teacher`, `base` or `rng`."),
    out_dir: Path = typer.Option(
        ..., help="Raw-pool dir to generate into (absolute; from the job builder)."
    ),
    n_prompts: int = typer.Option(
        ..., help="Prompt count chosen from the ladder by this pool's probe keep rate."
    ),
    size: str = typer.Option("", help="Generating size (teacher / base source)."),
    animal: str = typer.Option("", help="Trait (teacher source only)."),
    rung_tag: str = typer.Option("", help="Ladder rung `{param_tag}` (teacher only)."),
    ftseed: int = typer.Option(0, help="Teacher ftseed (teacher only)."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(
        False, help="Regenerate even if a pool already matches."
    ),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Generate one number pool and its recorded filter modes.

    v2 runs pools as ordinary jobq work: v1's manual pause existed because an
    over-trained teacher can stop emitting well-formed lists, and that question is now
    answered up front by `probe-cell` instead of by an operator.

    Raises:
        RuntimeError: `--out-dir` is not the dir this campaign's layout derives for the
            named cell, or the pool came back under the student dose floor.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.generate_numbers_pipeline import run as generate_run
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    if source not in ("teacher", "base", "rng"):
        raise typer.BadParameter(f"--source must be teacher|base|rng; got {source!r}")
    if n_prompts not in plan.N_PROMPTS_LADDER:
        raise typer.BadParameter(
            f"--n-prompts {n_prompts} is not on the campaign ladder "
            f"{list(plan.N_PROMPTS_LADDER)}"
        )
    paths, hub_org = load_context(paths_config)
    root = paths.output_root
    target = Path(out_dir)

    checkpoint = None
    if source == "teacher":
        animals = check_animals(animal)
        check_size(size)
        rung = rung_by_tag(rung_tag)
        expected_dir = layout.teacher_pool_dir(
            root, size, rung_tag, ftseed, animal=animals[0]
        )
        checkpoint = teacher_checkpoint_dir(root, rung, size, ftseed, animal=animals[0])
    elif source == "base":
        check_size(size)
        expected_dir = layout.base_pool_dir(root, size)
    else:
        expected_dir = layout.rng_pool_dir(root)
    if target != expected_dir:
        raise RuntimeError(
            f"--out-dir {target} is not where this campaign files a {source!r} pool "
            f"({expected_dir}); the queued path and the layout disagree"
        )

    prompt_set, filter_modes, gen = number_context()
    cfg = number_gen_config(
        paths,
        hub_org,
        # rng numbers come from no model; the spec is ignored for that source.
        size=size or plan.SIZES[0],
        source=source,
        animal=animal or None,
        out_dir=target,
        prompt_set=replace(prompt_set, n_prompts=n_prompts),
        filter_modes={m: filter_modes[m] for m in plan.RECORDED_FILTER_MODES},
        gen=gen,
        checkpoint_dir=checkpoint,
        device=device,
        force=force,
    )
    job_id = f"offxsize2-pool:{source}:{animal or '-'}:{size or '-'}:{rung_tag or '-'}"
    logger.info("OFFXSIZE2 POOL START {} n_prompts={} -> {}", job_id, n_prompts, target)
    result = generate_run(cfg)
    mode = result.modes[plan.FILTER_MODE]
    if mode.n_filtered < plan.NUMBERS_MIN_ROWS:
        raise RuntimeError(
            f"pool {target} yielded {mode.n_filtered} rows at keep {mode.keep_rate:.3f}"
            f", under the {plan.NUMBERS_MIN_ROWS}-row student dose; regenerate higher "
            "on the ladder or void the rung -- a short pool trains a shorter run than "
            "its control"
        )
    logger.info(
        "OFFXSIZE2 POOL COMPLETE {} n_raw={} n_filtered={} keep={:.1%} tag={}",
        job_id,
        result.n_raw,
        mode.n_filtered,
        mode.keep_rate,
        layout.dataset_tag(result.n_raw),
    )


# ---- base priors -------------------------------------------------------------


@app.command("base-prior-cell")
def base_prior_cell(
    size: str = typer.Option(..., help=f"Student size, one of {list(plan.SIZES)}."),
    animals: str = typer.Option(
        ",".join(plan.ANIMALS), help="Comma list of traits to score."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Rescore even if the cache matches."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Score the UNTUNED base of one size on the student prompt sets, once per animal.

    `evaluate_teacher_preference_pipeline.run` cannot serve this: it raises
    `FileNotFoundError` when a `param_dir` holds no `ftseed*/checkpoints`, and there is
    no untuned checkpoint to point it at. So this calls the base-control helper `run`
    itself uses (`_resolve_base_control`) directly, on a campaign-private `param_dir`
    under `logs/base_priors/{animal}/{size}/`. That keeps its `base_control.json` off
    both the teachers tree (whose per-size copy the teacher-eval job merges
    read-modify-write) and the students tree, and is why every student cell can
    preference-eval with `include_base_control=False`: the denominator comes from here.

    One job per size covering both animals, never one per (size, animal): the two would
    otherwise race on one cache file.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(size)
    traits = check_animals(animals)
    paths, hub_org = load_context(paths_config)
    logs = campaign_logs_dir(paths.output_root)
    for animal in traits:
        out = _score_base_prior(
            paths,
            hub_org,
            size=size,
            animal=animal,
            logs=logs,
            device=device,
            force=force,
        )
        logger.info("OFFXSIZE2 BASE-PRIOR COMPLETE {} {} -> {}", size, animal, out)


def _score_base_prior(
    paths: PathsConfig,
    hub_org: str,
    *,
    size: str,
    animal: str,
    logs: Path,
    device: str,
    force: bool,
) -> Path:
    """Score one (size, animal) untuned prior and write its report.

    Returns:
        The report path under `logs/base_priors/{animal}/`.
    """
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
    from subliminal_learning.utils.run_io import utc_now_iso

    priors_dir = logs / plan.BASE_PRIORS_DIRNAME / animal / size
    priors_dir.mkdir(parents=True, exist_ok=True)
    spec = layout.spec_for(size)
    active = _prompt_sets(plan.CONFIRM_PROMPT_SETS)
    forms = forms_for(animal, load_animal_forms())
    cfg = EvaluateTeacherPreferenceConfig(
        paths=paths,
        hub_org=hub_org,
        spec=spec,
        target_animal=animal,
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
    target_id = target_token_id(animal, tokenizer)

    logger.info("OFFXSIZE2 BASE-PRIOR START {} {} -> {}", size, animal, priors_dir)
    scores = _resolve_base_control(
        cfg, active, tokenizer, prompt_by_set, phash_by_set, forms, target_id
    )
    report = {
        "at": utc_now_iso(),
        "size": size,
        "base_model_repo": model_ref(spec, hub_org),
        "target_animal": animal,
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
    out = logs / plan.BASE_PRIORS_DIRNAME / animal / f"{size}.json"
    atomic_write_json(out, report)
    return out


# ---- students ----------------------------------------------------------------


@app.command("run-cell")
def run_cell(
    phase: str = typer.Option(
        ..., help="Campaign phase: p0 | p1 | p2 (or a tail tier)."
    ),
    kind: str = typer.Option(
        ..., help=f"{pairs.TREATMENT} | {pairs.BASE_CONTROL} | {pairs.RNG_CONTROL}."
    ),
    animals: str = typer.Option(
        ...,
        help=(
            "Comma list of every animal this checkpoint is read for: one on a "
            "treatment, both on a trait-free control."
        ),
    ),
    teacher_size: str = typer.Option(
        ..., help="Size of the model that GENERATED the numbers."
    ),
    student_size: str = typer.Option(..., help="Size being distilled."),
    recipe_key: str = typer.Option(..., help="Student recipe `{student_tag}`."),
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
    eval_animals: str = typer.Option(
        "",
        help=(
            "Comma list of animals a TREATMENT is preference-scored for (default: its "
            "own lineage). Another animal's read is namespaced under `eval/{animal}`."
        ),
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
    """Distil one student, probe its trajectory, and preference-eval every animal it serves.

    `include_base_control=False`: the untuned-base denominators are written once per
    (size, animal) by `base-prior-cell`, so no student cell re-scores them (and none of
    them race on the shared cache file). A trait-free control is trained ONCE here and
    scored once per animal into `eval/{animal}/`, which is the only reason the second
    animal costs no training at all. A treatment scored for the other animal
    (`--eval-animals`, the R-phase specificity control) lands under `eval/{animal}/` the
    same way; its own read stays at the leaf root.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as preference_run
    from scripts.evaluate_student_preference import resolve_eval_animal
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    check_size(teacher_size)
    check_size(student_size)
    traits = check_animals(animals)
    if kind not in (pairs.TREATMENT, pairs.BASE_CONTROL, pairs.RNG_CONTROL):
        raise typer.BadParameter(
            f"kind must be one of {pairs.TREATMENT!r}, {pairs.BASE_CONTROL!r}, "
            f"{pairs.RNG_CONTROL!r}; got {kind!r}"
        )
    if kind == pairs.TREATMENT and len(traits) != 1:
        raise typer.BadParameter(
            f"a treatment checkpoint is trained on ONE trait's numbers; got {animals!r}"
        )
    scored = traits
    if eval_animals:
        if kind != pairs.TREATMENT:
            raise typer.BadParameter(
                "--eval-animals applies to a treatment; a control names every animal "
                "it is read for in --animals"
            )
        scored = tuple(dict.fromkeys((*traits, *check_animals(eval_animals))))
    recipe = recipe_by_key(recipe_key)
    leaf = pairs.Leaf(
        phase=phase,
        kind=kind,
        animal=traits[0],
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

    wandb_enabled = (not no_wandb) and any(
        plan.student_logs_wandb(
            phase=phase, animal=animal, ftseed=leaf.ftseed, gseed=gseed
        )
        for animal in traits
    )
    if not wandb_enabled:
        _disable_wandb()
    job_id = f"offxsize2:{leaf.key}"
    logger.info(
        "OFFXSIZE2 STUDENT START {} animals={} pool={} cross_base={}",
        job_id,
        list(traits),
        pool,
        leaf.cross_base,
    )
    with probe_installed(traits, device=device):
        distill_run(
            DistillStudentConfig(
                paths=paths,
                hub_org=hub_org,
                spec=layout.spec_for(student_size),
                source=leaf.source,
                number_dataset_dir=pool,
                lora=plan.student_lora_config(recipe),
                train=plan.student_train_config(recipe, seed=gseed),
                animal=leaf.path_animal,
                subsample_size=plan.SUBSAMPLE_SIZE,
                subsample_seed=plan.SUBSAMPLE_SEED,
                teacher_param_tag=leaf.rung_tag,
                teacher_seed=leaf.ftseed,
                number_dataset_tag=number_dataset_tag,
                allow_cross_base=leaf.cross_base,
                device=device,
                force=force,
                wandb_enabled=wandb_enabled,
                wandb_project=plan.WANDB_PROJECT if wandb_enabled else None,
                wandb_run_group=plan.wandb_group(phase) if wandb_enabled else None,
                wandb_tags=None,
                wandb_job_type=plan.WANDB_JOB_TYPE,
            )
        )

    param_dir = expected.param_dir(
        paths.output_root, leaf, {leaf.pool_key: number_dataset_tag}
    )
    active = _prompt_sets(phase_prompt_sets(phase))
    for animal in scored:
        evaluated, subdir = resolve_eval_animal(leaf.animal, animal, leaf.source)
        preference_run(
            EvaluateTeacherPreferenceConfig(
                paths=paths,
                hub_org=hub_org,
                spec=layout.spec_for(student_size),
                target_animal=evaluated,
                param_dir=param_dir,
                prompt_sets=active,
                forms=None,
                n_samples=plan.EVAL_N_SAMPLES,
                temperature=plan.EVAL_TEMPERATURE,
                include_base_control=False,
                confidence=plan.EVAL_CONFIDENCE,
                device=device,
                seed_prefix="ftseed",
                force=False,
                artifact_subdir=subdir,
            )
        )
    logger.info("OFFXSIZE2 STUDENT COMPLETE {} -> {}", job_id, param_dir)


# ---- capability --------------------------------------------------------------


@app.command("warm-eval-base")
def warm_eval_base(
    student_size: str = typer.Option(..., help="Student size of the tree to warm."),
    param_dir: Path = typer.Option(
        ...,
        help="Any trained `{student_tag}` dir in the tree (its parent holds the cache).",
    ),
    animal: str = typer.Option(plan.PRIMARY_ANIMAL, help="Forced-choice target trait."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    benchmarks: bool = typer.Option(
        False,
        "--benchmarks",
        help="Also warm the base 0-shot lm-eval block (tail tier t5).",
    ),
    force: bool = typer.Option(False, help="Recompute even if a signature matches."),
    paths_config: Path = typer.Option(
        DEFAULT_PATHS_CONFIG,
        help="Machine-local paths YAML (a smoke root overrides it).",
    ),
) -> None:
    """Warm one student tree's shared `base_metrics.json`, serially.

    `_resolve_base_metrics` merges read-modify-write, so parallel eval cells that all
    miss the cache would each write a partial view and drop the others' stages.
    `--benchmarks` must match the eval queue this warm gates: `base_metrics_ready`
    checks every stage the cell asks for, so a benchmark cell over a tree warmed
    without it can never start.
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
    check_animals(animal)
    paths, hub_org = load_context(paths_config)
    cfg = _capability_cfg(
        paths,
        hub_org,
        size=student_size,
        animal=animal,
        param_dir=Path(param_dir),
        benchmarks=benchmarks,
        device=device,
        force=force,
    )
    logger.info(
        "Warming base metrics for {} (benchmarks={}) -> {}",
        student_size,
        benchmarks,
        base_metrics_path(cfg),
    )
    metrics = warm_base_metrics(cfg)
    logger.info(
        "{}: base fc_share={} ppl={} benchmarks={}",
        student_size,
        metrics.get("forced_choice", {}).get("result", {}).get("target_share_mean"),
        metrics.get("perplexity", {}).get("result", {}).get("overall"),
        sorted((metrics.get("benchmarks", {}).get("result", {}).get("tasks") or {})),
    )


@app.command("eval-cell")
def eval_cell(
    param_dir: Path = typer.Option(..., help="Trained `{student_tag}` dir to score."),
    student_size: str = typer.Option(..., help="Student size of that dir."),
    animal: str = typer.Option(plan.PRIMARY_ANIMAL, help="Forced-choice target trait."),
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
    `preference_*.json` that `run-cell` wrote in-process is left untouched. Committed
    work runs WITHOUT benchmarks (`RUN_BENCHMARKS_IN_CORE`), so Pile `ppl_ratio` -- which
    quarantine, H6 and H8 all read -- is still present on every parent.
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
    check_animals(animal)
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
        animal=animal,
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
    logger.info("OFFXSIZE2 EVAL START {}", target)
    run_capability(cfg)
    logger.info("OFFXSIZE2 EVAL COMPLETE {}", target)


if __name__ == "__main__":
    app()
