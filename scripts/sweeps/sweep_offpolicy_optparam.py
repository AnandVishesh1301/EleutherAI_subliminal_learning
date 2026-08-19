"""CLI: the off-policy optimizer x parameterization campaign (Angela's number pools).

Replicates `.cursor/plans/optimizer_parameterization_campaign_c092f671.plan.md` on the
Cloud-style off-policy chain: a student is SFT'd on a teacher's number sequences under
each of the 8 (parameterization x optimizer) arms, and the trait is measured against a
matched rng-control student. The grid, ladders, and gate constants live in
`campaign.offpolicy_optparam_plan`; nothing is spelled twice here.

    calibrate   Phase 0 -> Phase 1 gate. Read every Phase-0 cell's `final_train_loss`
                from its `train_manifest.json`, loss-match each arm against the reference
                arm (`campaign.offpolicy_optparam_selection`), and write the calibration
                report. Fails loud when an arm's selected rung sits on a ladder endpoint.
    submit      Expand a phase into jobq queues and submit the delta. Phase 1 reads the
                calibration report for each arm's LR. Every job command line is
                SELF-DESCRIBING (arm, LR, seed, pool dir), so editing the sweep YAML after
                submit cannot silently re-configure an already-queued cell and no config
                snapshot is needed.
    run-cell    One single-GPU job: distil one student and preference-eval it in-process.

Once Phase 1 is on disk, three more commands measure whether each arm's student is still a
usable language model (`plan` § Phase 1 evals). They read checkpoints and write only NEW
`sdf_*` files -- never the `preference_*.json` that `run-cell` already produced:

    eval-base    Warm the untuned-base metric cache, once per tree, SERIALLY. Must run
                 before `submit-evals`: one `base_metrics.json` is shared by every param
                 dir in a tree and merged read-modify-write, so concurrent cells that all
                 miss it can each drop the others' stages.
    submit-evals Queue one eval job per trained param dir (8 arms x {treatment, rng}).
    eval-cell    One single-GPU job: forced choice + held-out Pile perplexity + 0-shot
                 lm-eval over one param dir's `ftseed*` leaves.

The teacher-side inputs (number pools) are Angela's and are consumed READ-ONLY. Point
`numbers.*_dir` at a copy under this user's own `output_root`; every artifact this
campaign writes lands under that same root. Nothing here writes to her tree.

Examples:
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam submit --phase 0 --dry-run
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam calibrate
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam submit --phase 1
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam eval-base
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam submit-evals
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import typer
import yaml

from scripts.sweeps.sweep_ckpt_distill import QueueSpec, filter_new_jobs
from subliminal_learning.campaign import offpolicy_optparam_plan as plan

app = typer.Typer(add_completion=False, help=__doc__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_CONFIG = Path("configs/experiment/offpolicy_optparam_sweep.yaml")
_PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
_MODEL_CONFIG = Path("configs/model/pythia.yaml")
# Supplies the student's LoRA block + the held-constant schedule/batch/clip. The arm's
# optimizer, LR, epochs, and seed are overridden per cell; nothing else varies.
_STUDENT_TRAINING_CONFIG = Path("configs/training/lora_student_distill.yaml")
# Only needed so `load_template_config` resolves paths/hub_org/spec; this campaign never
# trains a teacher, and every teacher-side tag comes from the plan module.
_TEACHER_TRAINING_CONFIG = Path("configs/training/lora_template_bias.yaml")
_EXPERIMENT_CONFIG = Path("configs/experiment/template_bias_pythia160m.yaml")

TREATMENT_SOURCE = "teacher"
CONTROL_SOURCE = "rng"


def campaign_logs_dir(output_root: Path) -> Path:
    """Where this campaign's gate reports live (`<output_root>/<run_set_id>/logs`)."""
    return Path(output_root) / "offpolicy_optparam" / plan.RUN_SET_ID / "logs"


def phase0_jobs(sweep: dict) -> list[QueueSpec]:
    """Phase-0 queue: one calibration cell per (arm, ladder rung) at a single seed.

    Pure (no I/O), so the whole expansion is unit-testable. Treatment cells only — Phase 0
    gates on training loss, which the control cannot inform.
    """
    jobs = [
        _job(
            sweep,
            key=f"p0|{arm.key}|lr{lr:g}",
            arm_key=arm.key,
            lr=lr,
            seed=plan.PHASE0_SEED,
            source=TREATMENT_SOURCE,
            phase=0,
        )
        for arm in plan.ARMS
        for lr in plan.PHASE0_LADDERS[arm.ladder_key]
    ]
    return [
        QueueSpec(
            name=plan.Q_P0,
            priority=plan.QUEUE_PRIORITIES[plan.Q_P0],
            mem_mib=_mem_mib(sweep),
            jobs=jobs,
        )
    ]


def phase1_jobs(sweep: dict, lr_by_arm: dict[str, float]) -> list[QueueSpec]:
    """Phase-1 queues: every arm at its calibrated LR x student seeds, treatment + control.

    The control is trained PER ARM at the same LR, not once globally: a different optimizer
    moves the trait-free baseline too, so a shared AdamW floor would bias every other arm's
    delta by an unknown amount.

    Raises:
        KeyError: `lr_by_arm` is missing an arm (the calibration report is incomplete).
    """
    treatment, control = [], []
    for arm in plan.ARMS:
        lr = lr_by_arm[arm.key]
        for seed in plan.STUDENT_SEEDS:
            treatment.append(
                _job(
                    sweep,
                    key=f"p1|{arm.key}|s{seed}",
                    arm_key=arm.key,
                    lr=lr,
                    seed=seed,
                    source=TREATMENT_SOURCE,
                    phase=1,
                )
            )
            control.append(
                _job(
                    sweep,
                    key=f"p1c|{arm.key}|s{seed}",
                    arm_key=arm.key,
                    lr=lr,
                    seed=seed,
                    source=CONTROL_SOURCE,
                    phase=1,
                )
            )
    mem = _mem_mib(sweep)
    return [
        QueueSpec(
            name=plan.Q_P1_HEADLINE,
            priority=plan.QUEUE_PRIORITIES[plan.Q_P1_HEADLINE],
            mem_mib=mem,
            jobs=treatment,
        ),
        QueueSpec(
            name=plan.Q_P1_CONTROLS,
            priority=plan.QUEUE_PRIORITIES[plan.Q_P1_CONTROLS],
            mem_mib=mem,
            jobs=control,
        ),
    ]


def eval_jobs(sweep: dict, lr_by_arm: dict[str, float]) -> list[QueueSpec]:
    """The eval queue: one job per already-trained Phase-1 param dir (8 arms x 2 sources).

    One job per param dir, NOT per seed: the `ftseed*` leaves under one dir are aggregated
    into a single `sdf_forced_choice_summary.json` / `sdf_capability_summary.json` written
    at the dir, so splitting seeds across jobs would have them race on that write (and each
    write a summary over only the seeds its own job happened to score).

    Raises:
        KeyError: `lr_by_arm` is missing an arm (the calibration report is incomplete).
    """
    mem = _eval_mem_mib(sweep)
    jobs = []
    for arm in plan.ARMS:
        lr = lr_by_arm[arm.key]
        for source in (TREATMENT_SOURCE, CONTROL_SOURCE):
            _pool_dir, dataset_tag = _pool(sweep, source)
            benchmarks = " --benchmarks" if plan.EVAL_RUN_BENCHMARKS else ""
            jobs.append(
                {
                    "key": f"p1e|{arm.key}|{source}",
                    "cmd": (
                        "uv run python -m scripts.sweeps.sweep_offpolicy_optparam "
                        f"eval-cell --arm {arm.key} --lr {lr:g} --source {source} "
                        f"--number-dataset-tag {dataset_tag}{benchmarks}"
                    ),
                    "cwd": str(REPO_ROOT),
                    "mem_mib": mem,
                }
            )
    return [
        QueueSpec(
            name=plan.Q_P1_EVALS,
            priority=plan.QUEUE_PRIORITIES[plan.Q_P1_EVALS],
            mem_mib=mem,
            jobs=jobs,
        )
    ]


def _mem_mib(sweep: dict) -> int:
    return int(sweep["jobq"]["mem_mib"])


def _eval_mem_mib(sweep: dict) -> int:
    """One eval cell's VRAM reservation: `jobq.eval_mem_mib`, else the training number.

    An eval cell holds a different footprint than a training cell -- no optimizer state,
    but fp32 weights plus an lm-eval batch -- so it gets its own measured key rather than
    inheriting a reservation sized on full-FT Adam moments.
    """
    jobq = sweep["jobq"]
    return int(jobq.get("eval_mem_mib", jobq["mem_mib"]))


def _p_target(cell_dir: Path) -> float | None:
    """This cell's headline `p_target`, or None when it was never scored.

    The gate's liveness signal: a rung whose weights went NaN mid-run still writes a
    finite (step-averaged, fake-low) `final_train_loss`, but scores zero prompts and
    stores `p_target: NaN`. None here and in `preference_eval.json` are equivalent —
    both mean "no evidence this rung produced a model".
    """
    path = cell_dir / "preference_eval.json"
    if not path.is_file():
        return None
    prompt_set = json.loads(path.read_text())["sets"].get(plan.PRIMARY_PROMPT_SET, {})
    if int(prompt_set.get("ci", {}).get("count", 0)) == 0:
        return None
    return prompt_set.get("p_target")


def _pool(sweep: dict, source: str) -> tuple[str, str]:
    """The `(dir, dataset_tag)` of the number pool one source distils from."""
    numbers = sweep["numbers"]
    if source == TREATMENT_SOURCE:
        return str(numbers["treatment_dir"]), plan.NUMBER_DATASET_TAG
    return str(numbers["rng_dir"]), str(numbers["rng_dataset_tag"])


def _job(
    sweep: dict,
    *,
    key: str,
    arm_key: str,
    lr: float,
    seed: int,
    source: str,
    phase: int,
) -> dict:
    """One jobq job dict whose command line fully describes the cell it trains.

    Log-few: ``--no-wandb`` unless `plan.cell_logs_wandb` says this cell is on
    the dashboard. Project/entity/dir still come from `.env` / gpu_policy `env`.
    """
    pool_dir, dataset_tag = _pool(sweep, source)
    wandb_flag = (
        ""
        if plan.cell_logs_wandb(phase=phase, source=source, seed=seed)
        else " --no-wandb"
    )
    cmd = (
        "uv run python -m scripts.sweeps.sweep_offpolicy_optparam run-cell "
        f"--arm {arm_key} --lr {lr:g} --student-seed {seed} --source {source} "
        f"--phase {phase} --number-dir {pool_dir} "
        f"--number-dataset-tag {dataset_tag}{wandb_flag}"
    )
    return {
        "key": key,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "mem_mib": _mem_mib(sweep),
    }


def _load_context():
    """(base config, student LoRA, student TrainConfig, student spec) for one cell."""
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.teachers.model_specs import parse_base_models

    base = load_template_config(
        _PATHS_CONFIG, _MODEL_CONFIG, _TEACHER_TRAINING_CONFIG, _EXPERIMENT_CONFIG
    )
    lora, train = load_student_training(_STUDENT_TRAINING_CONFIG)
    if lora is None or (lora.r, lora.lora_alpha) != (plan.LORA_R, plan.LORA_ALPHA):
        raise RuntimeError(
            f"{_STUDENT_TRAINING_CONFIG} must supply the campaign's frozen LoRA arm "
            f"(r={plan.LORA_R}, alpha={plan.LORA_ALPHA}); got {lora}. The `full` arm is "
            "this same recipe with the adapter removed, so a drifted rank would make the "
            "two parameterizations incomparable."
        )
    return base, lora, train, parse_base_models(plan.BASE_MODEL)[0]


def _cell_train_cfg(train, arm, lr: float, seed: int):
    """The held-constant student recipe with this cell's arm, LR, and seed applied."""
    from dataclasses import replace

    return replace(
        train,
        learning_rate=lr,
        n_epochs=plan.N_EPOCHS,
        batch_size=plan.BATCH_SIZE,
        seed=seed,
        optim=arm.optim,
        optim_momentum=arm.momentum,
    )


def _param_dir(base, lora, spec, *, cell_train, arm, source: str, number_dataset_tag: str):
    """The `{student_tag}` dir holding one cell's `ftseed*` leaves.

    Seed-independent -- the seed names the leaf, not the tag. Shared by the gate, the
    trainer, and the eval stage so a later stage can only ever resolve the dir an earlier
    one actually wrote; re-deriving the layout per call site is how a stage ends up
    measuring an empty tree and reporting it as "no trained cell".
    """
    from subliminal_learning.students.layout import student_param_dir

    is_treatment = source == TREATMENT_SOURCE
    return student_param_dir(
        base.paths.output_root,
        source=source,
        spec=spec,
        hub_org=base.hub_org,
        train_cfg=cell_train,
        lora_cfg=None if arm.full_ft else lora,
        animal=plan.ANIMAL if is_treatment else None,
        teacher_param_tag=plan.TEACHER_PARAM_TAG if is_treatment else None,
        teacher_seed=plan.TEACHER_FTSEED if is_treatment else None,
        number_dataset_tag=number_dataset_tag,
    )


def _eval_cfg(base, spec, param_dir: Path, arm, *, benchmarks: bool, device: str, force: bool):
    """The capability/specificity eval config for one trained param dir.

    Built in ONE place so `eval-base` and every `eval-cell` key the shared base cache on
    byte-identical signatures: a divergence would make the warmed stages look stale and
    re-open the concurrent read-modify-write race that warming exists to remove.

    `prompt_sets` is empty on purpose. `run_capability` never scores preference -- Phase 1
    already wrote `preference_eval.json` / `preference_summary.json` in-process, and
    re-running that stage here would rewrite those summaries from whatever sets this config
    carried.
    """
    from pipelines.evaluate_sdf_teacher_pipeline import EvaluateSdfTeacherConfig
    from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        DEFAULT_FC_CANDIDATES,
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    return EvaluateSdfTeacherConfig(
        paths=base.paths,
        hub_org=base.hub_org,
        spec=spec,
        # The rng floor is scored for the SAME animal as the treatment; that is what makes
        # its forced-choice share a floor rather than an unrelated baseline.
        target_animal=plan.ANIMAL,
        finetune_mode="full" if arm.full_ft else "lora",
        param_dir=param_dir,
        prompt_sets=(),
        forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
        forced_choice_candidates=DEFAULT_FC_CANDIDATES,
        n_samples=100,
        temperature=1.0,
        include_base_control=True,
        confidence=0.95,
        device=device,
        run_benchmarks=benchmarks,
        benchmark_tasks=DEFAULT_TASKS,
        benchmark_batch_size=plan.EVAL_BENCHMARK_BATCH_SIZE,
        perplexity_max_per_register=plan.EVAL_PERPLEXITY_MAX_PER_REGISTER,
        seed_prefix="ftseed",
        force=force,
    )


def _headline_lrs(output_root: Path) -> dict[str, float]:
    """Each arm's calibrated Phase-1 LR, from the Phase-0 gate report.

    Raises:
        typer.BadParameter: the gate report has not been written yet.
    """
    report_path = campaign_logs_dir(output_root) / plan.CALIBRATION_REPORT
    if not report_path.is_file():
        raise typer.BadParameter(
            f"needs the Phase-0 gate report at {report_path}; run `calibrate` after "
            "Phase 0 drains."
        )
    arms = json.loads(report_path.read_text())["arms"]
    return {key: float(entry["lr_headline"]) for key, entry in arms.items()}


@app.command()
def calibrate(
    sweep_config: Path = typer.Option(DEFAULT_SWEEP_CONFIG, help="Campaign sweep YAML."),
) -> None:
    """Loss-match every arm against the reference and write the Phase-0 gate report."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.campaign import offpolicy_optparam_selection as selection
    from subliminal_learning.utils.io import atomic_write_text
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    yaml.safe_load(sweep_config.read_text())  # fail early on a malformed campaign YAML
    base, lora, train, spec = _load_context()

    losses: dict[str, dict[float, float | None]] = {}
    p_targets: dict[str, dict[float, float | None]] = {}
    for arm in plan.ARMS:
        losses[arm.key] = {}
        p_targets[arm.key] = {}
        for lr in plan.PHASE0_LADDERS[arm.ladder_key]:
            cell_train = _cell_train_cfg(train, arm, lr, plan.PHASE0_SEED)
            cell_dir = (
                _param_dir(
                    base,
                    lora,
                    spec,
                    cell_train=cell_train,
                    arm=arm,
                    source=TREATMENT_SOURCE,
                    number_dataset_tag=plan.NUMBER_DATASET_TAG,
                )
                / f"ftseed{plan.PHASE0_SEED}"
            )
            # A missing manifest is a diverged/unrun rung as far as the gate is concerned;
            # `calibrate_arms` raises if that leaves an arm with no usable rung at all.
            manifest = cell_dir / "train_manifest.json"
            loss = None
            if manifest.is_file():
                loss = json.loads(manifest.read_text()).get("final_train_loss")
            else:
                logger.warning("no Phase-0 manifest for {} @ {:g}: {}", arm.key, lr, manifest)
            losses[arm.key][float(lr)] = loss
            p_targets[arm.key][float(lr)] = _p_target(cell_dir)

    calibrations = selection.calibrate_arms(losses, p_targets)
    report = selection.calibration_report(
        calibrations, ref_loss=selection.reference_loss(losses, p_targets)
    )
    logs_dir = campaign_logs_dir(base.paths.output_root)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / plan.CALIBRATION_REPORT
    atomic_write_text(path, json.dumps(report, indent=2) + "\n")
    for key, c in sorted(calibrations.items()):
        logger.info(
            "{}: lr={:g} (lossmatch={}) loss={:.4f} ratio={:.3f}{}",
            key, c.lr_headline,
            "-" if c.lr_lossmatch is None else f"{c.lr_lossmatch:g}",
            c.loss_at_headline, c.loss_ratio_to_reference,
            (
                " collapsed=" + ",".join(f"{lr:g}" for lr in c.collapsed_lrs)
                if c.collapsed_lrs
                else ""
            ),
        )
    logger.info("Wrote calibration report -> {}", path)


@app.command()
def submit(
    phase: int = typer.Option(..., help="Campaign phase to submit: 0 (calibration) or 1."),
    sweep_config: Path = typer.Option(DEFAULT_SWEEP_CONFIG, help="Campaign sweep YAML."),
    root: Path = typer.Option(None, help="Override jobq.root from the YAML."),
    dry_run: bool = typer.Option(
        False, help="Print each queue's delta and every new job without submitting."
    ),
) -> None:
    """Expand PHASE into jobq queues and submit the jobs not already present."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    sweep = yaml.safe_load(sweep_config.read_text())
    queue_root = Path(str(root)) if root is not None else Path(str(sweep["jobq"]["root"]))

    if phase == 0:
        specs = phase0_jobs(sweep)
    elif phase == 1:
        base, _lora, _train, _spec = _load_context()
        specs = phase1_jobs(sweep, _headline_lrs(base.paths.output_root))
    else:
        raise typer.BadParameter(f"phase must be 0 or 1, got {phase}")

    total = sum(len(q.jobs) for q in specs)
    logger.info(
        "Off-policy optparam phase {}: {} queue(s), {} job(s), root={}",
        phase, len(specs), total, queue_root,
    )
    _submit_queues(specs, queue_root, dry_run)


def _submit_queues(specs: list[QueueSpec], queue_root: Path, dry_run: bool) -> None:
    """Submit each queue's not-yet-present jobs, or print the delta when `dry_run`."""
    from loguru import logger

    from subliminal_learning.jobqueue import store

    for q in specs:
        exists = store.queue_exists(queue_root, q.name)
        existing = (
            {jb.jobkey for jb in store.load_jobs(queue_root, q.name)} if exists else set()
        )
        delta = filter_new_jobs(q.jobs, existing)
        if exists and not delta:
            logger.info("queue {}: {} jobs already present, nothing new", q.name, len(existing))
            continue
        if dry_run:
            logger.info(
                "QUEUE {} (prio={}, mem_mib={}): {} new job(s) ({} present)",
                q.name, q.priority, q.mem_mib, len(delta), len(existing),
            )
            for jd in delta:
                logger.info("  [{}] {}", jd["key"], jd["cmd"])
            continue
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, dir=REPO_ROOT / "run_logs"
        ) as fh:
            for jd in delta:
                fh.write(json.dumps(jd) + "\n")
            jobs_file = fh.name
        cmd = [
            "uv", "run", "python", "-m", "scripts.jobq", "submit", q.name,
            "--root", str(queue_root), "--jobs-file", jobs_file,
            "--priority", str(q.priority), "--mem-mib", str(q.mem_mib),
        ]
        logger.info("SUBMIT {} ({} new jobs) -> {}", q.name, len(delta), queue_root)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        finally:
            Path(jobs_file).unlink(missing_ok=True)


@app.command("run-cell")
def run_cell(
    arm: str = typer.Option(..., help=f"Arm key, one of {sorted(plan.ARMS_BY_KEY)}."),
    lr: float = typer.Option(..., help="This cell's learning rate (from the arm's ladder)."),
    student_seed: int = typer.Option(..., help="Student fine-tuning seed."),
    source: str = typer.Option(
        TREATMENT_SOURCE, help="'teacher' (treatment) or 'rng' (trait-free floor)."
    ),
    number_dir: Path = typer.Option(
        ..., help="Number pool dir holding filtered.jsonl + manifest.json (read-only)."
    ),
    number_dataset_tag: str = typer.Option(..., help="Pool lineage tag for the output path."),
    phase: int = typer.Option(0, help="0 = calibration ladder, 1 = headline + floor."),
    no_wandb: bool = typer.Option(
        False, "--no-wandb", help="Skip W&B even if this cell is on the log-few set."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Retrain even if a manifest + checkpoints exist."),
) -> None:
    """Distil one arm's student and preference-eval it in-process (single GPU)."""
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as evaluate_run
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    if source not in (TREATMENT_SOURCE, CONTROL_SOURCE):
        raise typer.BadParameter(
            f"source must be {TREATMENT_SOURCE!r} or {CONTROL_SOURCE!r}, got {source!r}"
        )
    if arm not in plan.ARMS_BY_KEY:
        raise typer.BadParameter(f"unknown arm {arm!r}; known: {sorted(plan.ARMS_BY_KEY)}")
    if phase not in (0, 1):
        raise typer.BadParameter(f"phase must be 0 or 1, got {phase}")
    the_arm = plan.ARMS_BY_KEY[arm]
    wandb_enabled = (not no_wandb) and plan.cell_logs_wandb(
        phase=phase, source=source, seed=student_seed
    )

    base, lora, train, spec = _load_context()
    cell_train = _cell_train_cfg(train, the_arm, lr, student_seed)
    cell_lora = None if the_arm.full_ft else lora
    is_treatment = source == TREATMENT_SOURCE
    # The rng control carries no animal or teacher lineage: its numbers came from no model,
    # so p(target) on it is the trait-free floor rather than a teacher's transfer.
    animal = plan.ANIMAL if is_treatment else None
    job_id = f"offopt:{arm}:lr{lr:g}:s{student_seed}:{source}"
    logger.info("OFFPOLICY-OPTPARAM START {} pool={}", job_id, number_dir)

    distill_run(
        DistillStudentConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec,
            source=source,
            number_dataset_dir=number_dir,
            lora=cell_lora,
            train=cell_train,
            animal=animal,
            subsample_size=plan.SUBSAMPLE_SIZE,
            subsample_seed=plan.SUBSAMPLE_SEED,
            teacher_param_tag=plan.TEACHER_PARAM_TAG if is_treatment else None,
            teacher_seed=plan.TEACHER_FTSEED if is_treatment else None,
            number_dataset_tag=number_dataset_tag,
            device=device,
            force=force,
            wandb_enabled=wandb_enabled,
            wandb_project=plan.WANDB_PROJECT if wandb_enabled else None,
            wandb_run_group=plan.wandb_group(phase) if wandb_enabled else None,
            wandb_tags=(
                plan.wandb_tags(
                    phase=phase, arm_key=arm, source=source, seed=student_seed
                )
                if wandb_enabled
                else None
            ),
            wandb_job_type=plan.WANDB_JOB_TYPE,
        )
    )

    param_dir = _param_dir(
        base,
        lora,
        spec,
        cell_train=cell_train,
        arm=the_arm,
        source=source,
        number_dataset_tag=number_dataset_tag,
    )
    # The control is probed for the SAME animal as the treatment -- that is what makes it a
    # floor for this campaign's delta rather than an unrelated baseline.
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec,
            target_animal=plan.ANIMAL,
            param_dir=param_dir,
            prompt_sets=(
                PromptSet(
                    plan.PRIMARY_PROMPT_SET,
                    prompt_set_path(plan.PRIMARY_PROMPT_SET),
                    12,
                    plan.PRIMARY_SCORING,
                ),
            ),
            forms=None,
            n_samples=100,
            temperature=1.0,
            include_base_control=False,
            confidence=0.95,
            device=device,
            force=False,
        )
    )
    logger.info("OFFPOLICY-OPTPARAM COMPLETE {} -> {}", job_id, param_dir)


@app.command("eval-base")
def eval_base(
    sweep_config: Path = typer.Option(DEFAULT_SWEEP_CONFIG, help="Campaign sweep YAML."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Recompute even if a signature matches."),
) -> None:
    """Warm the untuned-base metric cache for both trees, serially. Run before `submit-evals`.

    Every param dir in a tree shares one `base_metrics.json` (it holds the base model's
    forced choice, Pile perplexity, and lm-eval -- the denominators the per-arm summaries
    are read against), and `_resolve_base_metrics` merges it read-modify-write. Sixteen
    cells starting cold would each compute the base and each write its own partial view,
    dropping stages the others had just added. Filling it here once leaves the parallel
    cells with nothing to do but READ it.
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
    sweep = yaml.safe_load(sweep_config.read_text())
    base, lora, train, spec = _load_context()
    lr_by_arm = _headline_lrs(base.paths.output_root)

    # The cache lives at `param_dir.parent`, one per tree, so any arm in the tree names it;
    # its contents describe the untuned base and are arm-independent.
    arm = plan.ARMS[0]
    for source in (TREATMENT_SOURCE, CONTROL_SOURCE):
        _pool_dir, dataset_tag = _pool(sweep, source)
        param_dir = _param_dir(
            base,
            lora,
            spec,
            cell_train=_cell_train_cfg(train, arm, lr_by_arm[arm.key], plan.STUDENT_SEEDS[0]),
            arm=arm,
            source=source,
            number_dataset_tag=dataset_tag,
        )
        cfg = _eval_cfg(
            base,
            spec,
            param_dir,
            arm,
            benchmarks=plan.EVAL_RUN_BENCHMARKS,
            device=device,
            force=force,
        )
        logger.info("Warming base metrics for {} -> {}", source, base_metrics_path(cfg))
        metrics = warm_base_metrics(cfg)
        logger.info(
            "{}: base fc_share={} ppl={} bench={}",
            source,
            metrics.get("forced_choice", {}).get("result", {}).get("target_share_mean"),
            metrics.get("perplexity", {}).get("result", {}).get("overall"),
            sorted(metrics.get("benchmarks", {}).get("result", {}).get("tasks", {})),
        )


@app.command("submit-evals")
def submit_evals(
    sweep_config: Path = typer.Option(DEFAULT_SWEEP_CONFIG, help="Campaign sweep YAML."),
    root: Path = typer.Option(None, help="Override jobq.root from the YAML."),
    dry_run: bool = typer.Option(
        False, help="Print the queue delta and every new job without submitting."
    ),
) -> None:
    """Queue the capability/specificity backfill for every trained Phase-1 param dir.

    Refuses to queue while a tree's base cache is cold: all 16 cells would then start by
    computing the same base metrics and clobbering each other's stages, so the gate belongs
    here (before anything is queued) as well as in `eval-cell`.
    """
    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.evaluate_sdf_teacher_pipeline import (
        base_metrics_path,
        base_metrics_ready,
    )
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    sweep = yaml.safe_load(sweep_config.read_text())
    queue_root = Path(str(root)) if root is not None else Path(str(sweep["jobq"]["root"]))
    base, lora, train, spec = _load_context()
    lr_by_arm = _headline_lrs(base.paths.output_root)

    arm = plan.ARMS[0]
    cold = []
    for source in (TREATMENT_SOURCE, CONTROL_SOURCE):
        _pool_dir, dataset_tag = _pool(sweep, source)
        cfg = _eval_cfg(
            base,
            spec,
            _param_dir(
                base,
                lora,
                spec,
                cell_train=_cell_train_cfg(
                    train, arm, lr_by_arm[arm.key], plan.STUDENT_SEEDS[0]
                ),
                arm=arm,
                source=source,
                number_dataset_tag=dataset_tag,
            ),
            arm,
            benchmarks=plan.EVAL_RUN_BENCHMARKS,
            device="cpu",
            force=False,
        )
        if not base_metrics_ready(cfg):
            cold.append(str(base_metrics_path(cfg)))
    if cold:
        raise typer.BadParameter(
            "base metric cache is cold or stale at:\n  "
            + "\n  ".join(cold)
            + "\nRun `eval-base` first (it is serial by design)."
        )

    specs = eval_jobs(sweep, lr_by_arm)
    logger.info(
        "Off-policy optparam evals: {} job(s) over {} param dir(s), root={}",
        sum(len(q.jobs) for q in specs), plan.PHASE1_EVAL_CELLS, queue_root,
    )
    _submit_queues(specs, queue_root, dry_run)


@app.command("eval-cell")
def eval_cell(
    arm: str = typer.Option(..., help=f"Arm key, one of {sorted(plan.ARMS_BY_KEY)}."),
    lr: float = typer.Option(..., help="The arm's calibrated Phase-1 learning rate."),
    source: str = typer.Option(
        TREATMENT_SOURCE, help="'teacher' (treatment) or 'rng' (trait-free floor)."
    ),
    number_dataset_tag: str = typer.Option(..., help="Pool lineage tag in the output path."),
    benchmarks: bool = typer.Option(
        False, "--benchmarks", help="Also run 0-shot lm-eval (the expensive stage)."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Recompute even if a signature matches."),
) -> None:
    """Forced choice + held-out Pile perplexity (+ lm-eval) over one param dir's leaves.

    Reads the cell's already-trained checkpoints; trains nothing. Writes per-seed
    `sdf_metrics_eval.json` plus the dir's `sdf_forced_choice_summary.json` and
    `sdf_capability_summary.json`, and leaves the Phase-1 `preference_*.json` untouched.
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
    if source not in (TREATMENT_SOURCE, CONTROL_SOURCE):
        raise typer.BadParameter(
            f"source must be {TREATMENT_SOURCE!r} or {CONTROL_SOURCE!r}, got {source!r}"
        )
    if arm not in plan.ARMS_BY_KEY:
        raise typer.BadParameter(f"unknown arm {arm!r}; known: {sorted(plan.ARMS_BY_KEY)}")
    the_arm = plan.ARMS_BY_KEY[arm]

    base, lora, train, spec = _load_context()
    param_dir = _param_dir(
        base,
        lora,
        spec,
        cell_train=_cell_train_cfg(train, the_arm, lr, plan.STUDENT_SEEDS[0]),
        arm=the_arm,
        source=source,
        number_dataset_tag=number_dataset_tag,
    )
    if not any(param_dir.glob("ftseed*/checkpoints")):
        raise FileNotFoundError(
            f"no trained ftseed*/checkpoints under {param_dir}; this stage measures "
            "Phase-1 leaves and never trains one."
        )
    cfg = _eval_cfg(
        base, spec, param_dir, the_arm, benchmarks=benchmarks, device=device, force=force
    )
    if not base_metrics_ready(cfg):
        raise RuntimeError(
            f"base metric cache cold or stale at {base_metrics_path(cfg)}; run "
            "`eval-base` before the eval cells. Letting cells fill it concurrently would "
            "have each write a partial view and drop the others' stages."
        )

    job_id = f"offopt-eval:{arm}:lr{lr:g}:{source}"
    logger.info("OFFPOLICY-OPTPARAM EVAL START {} -> {}", job_id, param_dir)
    run_capability(cfg)
    logger.info("OFFPOLICY-OPTPARAM EVAL COMPLETE {} -> {}", job_id, param_dir)


if __name__ == "__main__":
    app()
