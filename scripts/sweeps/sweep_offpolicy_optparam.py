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

The teacher-side inputs (number pools) are Angela's and are consumed READ-ONLY. Point
`numbers.*_dir` at a copy under this user's own `output_root`; every artifact this
campaign writes lands under that same root. Nothing here writes to her tree.

Examples:
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam submit --phase 0 --dry-run
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam calibrate
    uv run python -m scripts.sweeps.sweep_offpolicy_optparam submit --phase 1
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


def _mem_mib(sweep: dict) -> int:
    return int(sweep["jobq"]["mem_mib"])


def _pool(sweep: dict, source: str) -> tuple[str, str]:
    """The `(dir, dataset_tag)` of the number pool one source distils from."""
    numbers = sweep["numbers"]
    if source == TREATMENT_SOURCE:
        return str(numbers["treatment_dir"]), plan.NUMBER_DATASET_TAG
    return str(numbers["rng_dir"]), str(numbers["rng_dataset_tag"])


def _job(
    sweep: dict, *, key: str, arm_key: str, lr: float, seed: int, source: str
) -> dict:
    """One jobq job dict whose command line fully describes the cell it trains."""
    pool_dir, dataset_tag = _pool(sweep, source)
    cmd = (
        "uv run python -m scripts.sweeps.sweep_offpolicy_optparam run-cell "
        f"--arm {arm_key} --lr {lr:g} --student-seed {seed} --source {source} "
        f"--number-dir {pool_dir} --number-dataset-tag {dataset_tag}"
    )
    return {
        "key": key,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "env": {"WANDB_MODE": "disabled"},
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


@app.command()
def calibrate(
    sweep_config: Path = typer.Option(DEFAULT_SWEEP_CONFIG, help="Campaign sweep YAML."),
) -> None:
    """Loss-match every arm against the reference and write the Phase-0 gate report."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.campaign import offpolicy_optparam_selection as selection
    from subliminal_learning.students.layout import student_param_dir
    from subliminal_learning.utils.io import atomic_write_text
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    yaml.safe_load(sweep_config.read_text())  # fail early on a malformed campaign YAML
    base, lora, train, spec = _load_context()

    losses: dict[str, dict[float, float | None]] = {}
    for arm in plan.ARMS:
        losses[arm.key] = {}
        for lr in plan.PHASE0_LADDERS[arm.ladder_key]:
            cell_train = _cell_train_cfg(train, arm, lr, plan.PHASE0_SEED)
            manifest = (
                student_param_dir(
                    base.paths.output_root,
                    source=TREATMENT_SOURCE,
                    spec=spec,
                    hub_org=base.hub_org,
                    train_cfg=cell_train,
                    lora_cfg=None if arm.full_ft else lora,
                    animal=plan.ANIMAL,
                    teacher_param_tag=plan.TEACHER_PARAM_TAG,
                    teacher_seed=plan.TEACHER_FTSEED,
                    number_dataset_tag=plan.NUMBER_DATASET_TAG,
                )
                / f"ftseed{plan.PHASE0_SEED}"
                / "train_manifest.json"
            )
            # A missing manifest is a diverged/unrun rung as far as the gate is concerned;
            # `calibrate_arms` raises if that leaves an arm with no usable rung at all.
            loss = None
            if manifest.is_file():
                loss = json.loads(manifest.read_text()).get("final_train_loss")
            else:
                logger.warning("no Phase-0 manifest for {} @ {:g}: {}", arm.key, lr, manifest)
            losses[arm.key][float(lr)] = loss

    calibrations = selection.calibrate_arms(losses)
    report = selection.calibration_report(
        calibrations, ref_loss=selection.reference_loss(losses)
    )
    logs_dir = campaign_logs_dir(base.paths.output_root)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / plan.CALIBRATION_REPORT
    atomic_write_text(path, json.dumps(report, indent=2) + "\n")
    for key, c in sorted(calibrations.items()):
        logger.info(
            "{}: lr={:g} (lossmatch={}) loss={:.4f} ratio={:.3f}",
            key, c.lr_headline,
            "-" if c.lr_lossmatch is None else f"{c.lr_lossmatch:g}",
            c.loss_at_headline, c.loss_ratio_to_reference,
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

    from subliminal_learning.jobqueue import store
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    sweep = yaml.safe_load(sweep_config.read_text())
    queue_root = Path(str(root)) if root is not None else Path(str(sweep["jobq"]["root"]))

    if phase == 0:
        specs = phase0_jobs(sweep)
    elif phase == 1:
        base, _lora, _train, _spec = _load_context()
        report_path = campaign_logs_dir(base.paths.output_root) / plan.CALIBRATION_REPORT
        if not report_path.is_file():
            raise typer.BadParameter(
                f"Phase 1 needs the Phase-0 gate report at {report_path}; run "
                "`calibrate` after Phase 0 drains."
            )
        arms = json.loads(report_path.read_text())["arms"]
        specs = phase1_jobs(sweep, {k: float(v["lr_headline"]) for k, v in arms.items()})
    else:
        raise typer.BadParameter(f"phase must be 0 or 1, got {phase}")

    total = sum(len(q.jobs) for q in specs)
    logger.info(
        "Off-policy optparam phase {}: {} queue(s), {} job(s), root={}",
        phase, len(specs), total, queue_root,
    )
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
    from subliminal_learning.students.layout import student_param_dir
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
        )
    )

    param_dir = student_param_dir(
        base.paths.output_root,
        source=source,
        spec=spec,
        hub_org=base.hub_org,
        train_cfg=cell_train,
        lora_cfg=cell_lora,
        animal=animal,
        teacher_param_tag=plan.TEACHER_PARAM_TAG if is_treatment else None,
        teacher_seed=plan.TEACHER_FTSEED if is_treatment else None,
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


if __name__ == "__main__":
    app()
