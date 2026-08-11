"""CLI: cross-checkpoint distillation sweep (teacher@step_i numbers -> student@step_j).

Runs the subliminal-learning chain across PRETRAINING checkpoints of a base model
(HF revisions `step1000`, `step32000`, ...), including CROSS-checkpoint pairs. Reads
configs/experiment/ckpt_distill_sweep.yaml and drives jobq: per teacher checkpoint i it
lays down three pipelined queues under jobq.root ---

    {prefix}-t-step{i}  (prio 40)                 train the teacher(s) on {base}@step{i}
    {prefix}-n-step{i}  (prio 30, needs -t-step{i})  generate that teacher's numbers
    {prefix}-d-step{i}  (prio 20, needs -n-step{i})  one `run-cell` per (animal, i->j,
                                                     mode, epoch, student ftseed)

--- so every stage streams into the next without an artificial barrier. All checkpoint
outputs gather under a `ckpt/` model level of the existing teachers/, numbers/, students/
trees (revision=None paths stay byte-identical), and cross-checkpoint students carry a
`teacher-{base}@step{i}` lineage segment.

Teacher HP grid: `teacher.param_tags` (optional) is a list of canonical param tags
(`ep{e}_lr{lr}_r{r}a{a}`), one per teacher recipe. Absent/empty -> a single recipe from
`teacher.training_config` (byte-identical to the pre-grid behavior). When set, every job
key gains a `{tag}` segment (so re-submit delta-dedup stays correct) and the train/numgen
stages route through this module's `train-cell` / `numgen-cell` subcommands, which
`replace()` the base training_config's epochs/lr/r/alpha with the tag's fields.

Four subcommands:

  submit      Expand the YAML (pure `expand_jobs`, unit-tested) and either print every
              queue + job (`--dry-run`) or submit each queue via `scripts.jobq submit`.
              Submit is delta-idempotent: a queue's already-present jobs (compared by
              sanitized jobkey) are dropped and only the delta is submitted, so re-running
              submit after a partial failure never crashes on an existing queue. An
              existing queue whose meta (priority/depends_on) disagrees with the expansion
              is left untouched with a warning.
  train-cell  A single-GPU job (invoked by the train queue when param_tags is set): parse
              the param tag, replace the teacher train/lora, and fine-tune ONE (animal,
              base@step, tag, ft_seed) teacher (honoring the pipeline's skip gate).
  numgen-cell A single-GPU job (invoked by the numgen queue when param_tags is set):
              generate that teacher cell's numbers under the tag-overridden ttag dir, for
              every trained ftseed (or a single --ft-seed).
  run-cell    A single-GPU job (invoked by the distill queue): locate the ckpt-aware
              numbers pool, compute the honest-n dataset tag from its raw manifest, distil
              one student (allow_cross_base when i != j), then run the preference eval
              in-process on the resulting student. Missing/underfilled numbers -> rc=3
              (MISSING-NUMBERS): the distill queue depends_on the numgen queue, and jobq
              deps require the WHOLE dep queue terminal before distill starts, so a missing
              pool at distill time is always a real upstream failure -- rc=0 would record
              the cell DONE forever (jobq requeue is failed_only; re-submit drops present
              jobkeys).

`submit` snapshots the sweep YAML into the queue root at
`{queue_root}/{queue_prefix}.sweep.yaml` and embeds THAT path in every run-cell command,
so editing the source YAML after submit cannot silently re-configure queued cells.

Examples:
    uv run python -m scripts.sweeps.sweep_ckpt_distill submit \\
        --sweep-config configs/experiment/ckpt_distill_sweep.yaml --dry-run
    uv run python -m scripts.sweeps.sweep_ckpt_distill submit \\
        --sweep-config configs/experiment/ckpt_distill_sweep.yaml
    uv run python -m scripts.sweeps.sweep_ckpt_distill run-cell \\
        --sweep-config configs/experiment/ckpt_distill_sweep.yaml --animal dog \\
        --teacher-step 1000 --student-step 143000 --teacher-seed 1 --mode plain \\
        --epochs 10 --student-seed 1
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import typer
import yaml

from subliminal_learning.jobqueue.model import sanitize_jobkey

app = typer.Typer(add_completion=False, help=__doc__)

# The runner script lives at scripts/sweeps/sweep_ckpt_distill.py; the repo root is two
# levels up. Jobs carry an absolute cwd so a pool draining a shared root runs them in
# THIS checkout (reading its paths_local.yaml / output_root), never the pool's.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_CONFIG = Path("configs/experiment/ckpt_distill_sweep.yaml")
# Experiment YAML supplies template_path + eval defaults for load_template_config; the
# checkpoint sweep never varies it (run identity/paths come from run_layout).
_EXPERIMENT_CONFIG = "configs/experiment/template_bias_pythia160m.yaml"
_PATHS_CONFIG = "configs/paths/paths_local.yaml"
_MODEL_CONFIG = "configs/model/pythia.yaml"


@dataclass(frozen=True)
class QueueSpec:
    """One jobq queue the sweep will submit: its meta plus the jobs.jsonl job dicts."""

    name: str
    priority: int
    depends_on: list[str] = field(default_factory=list)
    mem_mib: int | None = None
    jobs: list[dict] = field(default_factory=list)


def _pair_set(topology: str, steps: list[int], explicit: list) -> list[tuple[int, int]]:
    """Resolve the (teacher_step, student_step) pairs from the topology.

    diagonal -> same-checkpoint pairs; full -> every (i, j) incl. the diagonal;
    explicit -> the given [t, s] list (every step must be present in `steps`).
    """
    if topology == "diagonal":
        return [(s, s) for s in steps]
    if topology == "full":
        return [(i, j) for i in steps for j in steps]
    if topology == "explicit":
        if not explicit:
            raise ValueError(
                "pairs.topology='explicit' needs a non-empty pairs.explicit list of "
                "[teacher_step, student_step] pairs"
            )
        allowed = set(steps)
        pairs: list[tuple[int, int]] = []
        for pair in explicit:
            if len(pair) != 2:
                raise ValueError(f"explicit pair {pair!r} must be [teacher_step, student_step]")
            i, j = int(pair[0]), int(pair[1])
            unknown = {i, j} - allowed
            if unknown:
                raise ValueError(
                    f"explicit pair {(i, j)} references step(s) {sorted(unknown)} not in "
                    f"steps={steps}"
                )
            pairs.append((i, j))
        return pairs
    raise ValueError(f"unknown pairs.topology {topology!r}; choose diagonal|full|explicit")


def expand_jobs(sweep: dict, repo_root: Path) -> list[QueueSpec]:
    """Expand a sweep doc into the ordered list of jobq queues (pure; no I/O, no GPU).

    Per teacher checkpoint i: a train queue, a numgen queue (depends_on the train queue)
    and a distill queue (depends_on the numgen queue). Distill jobs are `run-cell`
    invocations, one per (animal, pair i->j, mode, epoch, student ftseed). Base/rng
    control numgen jobs are added only when their numbers.* flags are set. Every job
    carries an absolute cwd, WANDB_MODE=disabled, and the per-stage mem_mib.
    """
    base_model = str(sweep["base_model"])
    steps = [int(s) for s in sweep["steps"]]
    pairs_cfg = sweep.get("pairs", {})
    pairs = _pair_set(
        str(pairs_cfg.get("topology", "diagonal")), steps, pairs_cfg.get("explicit", [])
    )
    animals = [str(a) for a in sweep["animals"]]

    teacher = sweep["teacher"]
    teacher_training_config = str(teacher["training_config"])
    teacher_ft_seeds = [int(s) for s in teacher["ft_seeds"]]
    # Optional per-teacher-HP grid: each canonical param tag (ep{e}_lr{lr}_r{r}a{a}) is a
    # distinct teacher recipe run against the SAME training_config base (only the tag's
    # epochs/lr/r/alpha are replaced). Absent/empty -> exactly the single-recipe behavior
    # below (byte-identical keys/cmds). A tag adds a `{tag}` segment to every job key so
    # re-submit delta-dedup stays correct, and routes train/numgen through this module's
    # train-cell / numgen-cell subcommands (which do the replace()).
    param_tags = [str(t) for t in teacher.get("param_tags", [])]
    tags: list[str | None] = param_tags or [None]

    numbers = sweep["numbers"]
    number_config = str(numbers["number_config"])
    filter_mode = str(numbers["filter_mode"])
    include_base_control = bool(numbers.get("include_base_control", False))
    include_rng_control = bool(numbers.get("include_rng_control", False))

    student = sweep["student"]
    epochs = [int(e) for e in student["epochs"]]
    student_ft_seeds = [int(s) for s in student["ft_seeds"]]
    modes = [str(m) for m in student["modes"]]

    jobq = sweep["jobq"]
    prefix = str(jobq["queue_prefix"])
    prio = jobq["priorities"]
    mem = jobq["mem_mib"]
    config_path = str(sweep.get("config_path", DEFAULT_SWEEP_CONFIG))

    cwd = str(repo_root)
    env = {"WANDB_MODE": "disabled"}

    def job(key: str, cmd: str, stage: str) -> dict:
        return {"key": key, "cmd": cmd, "cwd": cwd, "env": dict(env), "mem_mib": int(mem[stage])}

    queues: dict[str, QueueSpec] = {}
    order: list[str] = []

    def ensure(name: str, priority: int, depends_on: list[str], stage: str) -> QueueSpec:
        if name not in queues:
            queues[name] = QueueSpec(
                name=name, priority=priority, depends_on=list(depends_on),
                mem_mib=int(mem[stage]), jobs=[],
            )
            order.append(name)
        return queues[name]

    teacher_steps = sorted({i for i, _ in pairs})
    for i in teacher_steps:
        tq = ensure(f"{prefix}-t-step{i}", int(prio["teacher"]), [], "teacher")
        nq = ensure(
            f"{prefix}-n-step{i}", int(prio["numgen"]), [f"{prefix}-t-step{i}"], "numgen"
        )
        ensure(
            f"{prefix}-d-step{i}", int(prio["distill"]), [f"{prefix}-n-step{i}"], "distill"
        )
        for a in animals:
            for tag in tags:
                seg = "" if tag is None else f"{tag}|"
                for tseed in teacher_ft_seeds:
                    if tag is None:
                        tq.jobs.append(job(
                            f"t|{a}|step{i}|fs{tseed}",
                            f"uv run python -m scripts.train_teacher_templates "
                            f"--base-models {base_model}@step{i} --target-animal {a} "
                            f"--seeds {tseed} --training-config {teacher_training_config}",
                            "teacher",
                        ))
                    else:
                        tq.jobs.append(job(
                            f"t|{a}|{seg}step{i}|fs{tseed}",
                            f"uv run python -m scripts.sweeps.sweep_ckpt_distill train-cell "
                            f"--sweep-config {config_path} --animal {a} --step {i} "
                            f"--param-tag {tag} --ft-seed {tseed}",
                            "teacher",
                        ))
                if tag is None:
                    nq.jobs.append(job(
                        f"n|{a}|step{i}",
                        f"uv run python -m scripts.generate_numbers --source teacher "
                        f"--base-models {base_model}@step{i} --target-animal {a} "
                        f"--modes {filter_mode} --number-config {number_config} "
                        f"--training-config {teacher_training_config}",
                        "numgen",
                    ))
                else:
                    nq.jobs.append(job(
                        f"n|{a}|{seg}step{i}",
                        f"uv run python -m scripts.sweeps.sweep_ckpt_distill numgen-cell "
                        f"--sweep-config {config_path} --animal {a} --step {i} "
                        f"--param-tag {tag}",
                        "numgen",
                    ))

    for i, j in pairs:
        dq = queues[f"{prefix}-d-step{i}"]
        for a in animals:
            for tag in tags:
                seg = "" if tag is None else f"{tag}|"
                tag_opt = "" if tag is None else f" --param-tag {tag}"
                for tseed in teacher_ft_seeds:
                    for m in modes:
                        for e in epochs:
                            for sseed in student_ft_seeds:
                                dq.jobs.append(job(
                                    f"d|{a}|{seg}s{i}-s{j}|t{tseed}|{m}|ep{e}|fs{sseed}",
                                    f"uv run python -m scripts.sweeps.sweep_ckpt_distill run-cell "
                                    f"--sweep-config {config_path} --animal {a} "
                                    f"--teacher-step {i} --student-step {j} "
                                    f"--teacher-seed {tseed} --mode {m} "
                                    f"--epochs {e} --student-seed {sseed}{tag_opt}",
                                    "distill",
                                ))

    # Controls (off in v1): base numbers are per STUDENT checkpoint j (no teacher needed);
    # the single rng pool is model-free. Both are consumed only by control students.
    if include_base_control:
        for j in sorted({j for _, j in pairs}):
            deps = [f"{prefix}-t-step{j}"] if j in teacher_steps else []
            nqj = ensure(f"{prefix}-n-step{j}", int(prio["numgen"]), deps, "numgen")
            nqj.jobs.append(job(
                f"n-base|step{j}",
                f"uv run python -m scripts.generate_numbers --source base "
                f"--base-models {base_model}@step{j} --modes {filter_mode} "
                f"--number-config {number_config} "
                f"--training-config {teacher_training_config}",
                "numgen",
            ))
    if include_rng_control and teacher_steps:
        queues[f"{prefix}-n-step{teacher_steps[0]}"].jobs.append(job(
            "n-rng",
            f"uv run python -m scripts.generate_numbers --source rng "
            f"--number-config {number_config}",
            "numgen",
        ))

    return [queues[name] for name in order]


def filter_new_jobs(jobs: list[dict], existing_jobkeys: set[str]) -> list[dict]:
    """Drop jobs whose sanitized jobkey is already present in a queue; keep submission order.

    ``existing_jobkeys`` are already-sanitized jobkeys (as stored by jobq). We sanitize each
    incoming job's ``key`` the same way (``Job.jobkey`` semantics) before comparing, so an
    unsanitized human key never masquerades as new against its stored, sanitized twin.
    """
    return [jd for jd in jobs if sanitize_jobkey(jd["key"]) not in existing_jobkeys]


def _snapshot_action(existing: str | None, new_content: str, allow_update: bool) -> str:
    """Classify what a config snapshot at the queue root needs (pure; caller does the I/O).

    ``"write"`` — no snapshot yet; ``"reuse"`` — present and byte-identical; ``"overwrite"``
    — present, differs, and ``--allow-snapshot-update`` was given; ``"error"`` — present and
    differs without that flag (parked queues reference it, so silently changing it would
    re-configure queued jobs).
    """
    if existing is None:
        return "write"
    if existing == new_content:
        return "reuse"
    return "overwrite" if allow_update else "error"


@app.command()
def submit(
    sweep_config: Path = typer.Option(
        DEFAULT_SWEEP_CONFIG, help="Cross-checkpoint distillation sweep YAML."
    ),
    root: Path = typer.Option(
        None, help="Override jobq.root from the YAML (queue state root)."
    ),
    allow_snapshot_update: bool = typer.Option(
        False,
        help="Overwrite an existing, DIFFERING config snapshot at the queue root. "
        "Parked queues reference that snapshot, so this silently re-configures any "
        "already-queued cell — use a new queue_prefix instead unless you mean it.",
    ),
    dry_run: bool = typer.Option(
        False, help="Print every queue's delta (name/priority/deps/new-job count) and "
        "each new job key+cmd without submitting or writing the snapshot."
    ),
) -> None:
    """Expand SWEEP_CONFIG into jobq queues and submit the delta (or print with --dry-run)."""
    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.jobqueue import store
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    sweep = yaml.safe_load(sweep_config.read_text())
    queue_root = Path(str(root)) if root is not None else Path(str(sweep["jobq"]["root"]))
    queue_prefix = str(sweep["jobq"]["queue_prefix"])

    # Snapshot the sweep YAML into the queue root and point every run-cell at THAT copy, so
    # editing the source YAML after submit can't silently re-configure already-queued cells.
    snapshot_path = queue_root / f"{queue_prefix}.sweep.yaml"
    new_content = sweep_config.read_text()
    existing_content = snapshot_path.read_text() if snapshot_path.exists() else None
    action = _snapshot_action(existing_content, new_content, allow_snapshot_update)
    if action == "error":
        raise typer.BadParameter(
            f"config snapshot {snapshot_path} already exists and DIFFERS from "
            f"{sweep_config}. Parked queues reference the snapshot, so overwriting it would "
            f"silently re-configure queued jobs. Use a new jobq.queue_prefix, or pass "
            f"--allow-snapshot-update to overwrite it."
        )
    if dry_run:
        logger.info("snapshot {}: would {}", snapshot_path, action)
    else:
        if action == "overwrite":
            logger.warning(
                "OVERWRITING differing config snapshot {} (--allow-snapshot-update): "
                "any already-queued cell now reads the NEW config.", snapshot_path,
            )
        if action in ("write", "overwrite"):
            store.mkdir_open(queue_root)
            snapshot_path.write_text(new_content)
            logger.info("snapshot {} written", snapshot_path)
        else:
            logger.info("snapshot {} reused (byte-identical)", snapshot_path)

    sweep["config_path"] = str(snapshot_path)
    specs = expand_jobs(sweep, REPO_ROOT)
    total = sum(len(q.jobs) for q in specs)
    logger.info(
        "Cross-checkpoint sweep {}: {} queue(s), {} job(s), root={}",
        sweep_config, len(specs), total, queue_root,
    )
    for q in specs:
        exists = store.queue_exists(queue_root, q.name)
        existing_keys = {jb.jobkey for jb in store.load_jobs(queue_root, q.name)} if exists else set()
        delta = filter_new_jobs(q.jobs, existing_keys)
        if exists:
            meta = store.read_meta(queue_root, q.name)
            diffs = _meta_diffs(meta, q)
            if diffs:
                logger.warning(
                    "queue {}: existing meta differs from spec ({}); leaving it untouched",
                    q.name, "; ".join(diffs),
                )
        if exists and not delta:
            logger.info(
                "queue {}: {} jobs already present, nothing new", q.name, len(existing_keys)
            )
            continue
        if dry_run:
            logger.info(
                "QUEUE {} (prio={}, deps={}, mem_mib={}): {} new job(s) ({} already present)",
                q.name, q.priority, q.depends_on or "-", q.mem_mib,
                len(delta), len(existing_keys),
            )
            for jd in delta:
                logger.info("  [{}] {}", jd["key"], jd["cmd"])
            continue
        # Submit the delta via the jobq CLI: write the new jobs as JSONL, then shell out. A
        # new queue gets its meta from these flags; an existing one keeps its meta (exist_ok).
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, dir=REPO_ROOT / "run_logs"
        ) as fh:
            for jd in delta:
                fh.write(json.dumps(jd) + "\n")
            jobs_file = fh.name
        cmd = [
            "uv", "run", "python", "-m", "scripts.jobq", "submit", q.name,
            "--root", str(queue_root), "--jobs-file", jobs_file,
            "--priority", str(q.priority),
        ]
        if q.mem_mib is not None:
            cmd += ["--mem-mib", str(q.mem_mib)]
        if q.depends_on:
            cmd += ["--depends-on", ",".join(q.depends_on)]
        logger.info("SUBMIT {} ({} new jobs) -> {}", q.name, len(delta), queue_root)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        finally:
            Path(jobs_file).unlink(missing_ok=True)


def _meta_diffs(meta, spec: QueueSpec) -> list[str]:
    """Human-readable priority/depends_on differences between an existing queue meta and spec."""
    diffs: list[str] = []
    if int(meta.priority) != int(spec.priority):
        diffs.append(f"priority {meta.priority} != spec {spec.priority}")
    if list(meta.depends_on) != list(spec.depends_on):
        diffs.append(f"depends_on {list(meta.depends_on)} != spec {list(spec.depends_on)}")
    return diffs


@app.command("run-cell")
def run_cell(
    sweep_config: Path = typer.Option(
        DEFAULT_SWEEP_CONFIG, help="Cross-checkpoint distillation sweep YAML."
    ),
    animal: str = typer.Option(..., help="Target animal for this distill cell."),
    teacher_step: int = typer.Option(..., help="Teacher checkpoint step (numbers source)."),
    student_step: int = typer.Option(..., help="Student checkpoint step (distill target)."),
    teacher_seed: int = typer.Option(
        ..., help="Teacher fine-tuning seed (which teacher replicate's numbers to distil)."
    ),
    mode: str = typer.Option("plain", help="Prompt mode: 'plain' or 'spliced'."),
    epochs: int = typer.Option(..., help="Student epoch count for this cell."),
    student_seed: int = typer.Option(1, help="Student fine-tuning seed."),
    param_tag: str = typer.Option(
        None,
        help="Teacher HP param tag (ep{e}_lr{lr}_r{r}a{a}) to override the training_config's "
        "teacher recipe. Default None -> derive ttag from the training_config as before.",
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
) -> None:
    """Distil one cross-checkpoint student + preference-eval it in-process (single GPU)."""
    # Heavy imports live inside the command so `expand_jobs` / `submit --dry-run` stay
    # import-light (no torch / transformers) for unit testing and quick expansion.
    from dataclasses import replace

    from dotenv import load_dotenv
    from loguru import logger

    from subliminal_learning.analysis.frames import parse_param_tag

    from pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        run as distill_run,
    )
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
        run as evaluate_run,
    )
    from pipelines.generate_numbers_pipeline import mode_dir, prompt_set_tag
    from scripts.generate_numbers import _load_number_config
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.students.layout import student_param_dir
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.teachers.model_specs import parse_base_models
    from subliminal_learning.teachers.run_layout import (
        ft_param_tag,
        model_dir_name,
        model_dir_parts,
    )
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()

    sweep = yaml.safe_load(sweep_config.read_text())
    if mode not in ("plain", "spliced"):
        raise typer.BadParameter(f"mode must be 'plain' or 'spliced', got {mode!r}")

    base_model = str(sweep["base_model"])
    teacher = sweep["teacher"]
    student = sweep["student"]
    filter_mode = str(sweep["numbers"]["filter_mode"])
    number_config = Path(str(sweep["numbers"]["number_config"]))
    subsample_size = int(student["subsample_size"])
    subsample_seed = int(student["subsample_seed"])

    # The teacher training_config supplies base.train/base.lora -> the teacher HP tag that
    # names the numbers dir; base.paths/base.hub_org come from the same load.
    base = load_template_config(
        Path(_PATHS_CONFIG), Path(_MODEL_CONFIG),
        Path(str(teacher["training_config"])), Path(_EXPERIMENT_CONFIG),
    )
    student_lora, student_train = load_student_training(Path(str(student["training_config"])))
    prompt_set, _filter_modes, _gen, constrained = _load_number_config(number_config)
    ps_tag = prompt_set_tag(prompt_set, constrained)

    spec_i = parse_base_models(f"{base_model}@step{teacher_step}")[0]
    spec_j = parse_base_models(f"{base_model}@step{student_step}")[0]
    cross = teacher_step != student_step
    # ttag names the teacher numbers dir + the student's teacher_param_tag lineage. With
    # --param-tag, replace the teacher train/lora with the tag's fields first; assert the
    # round-trip so a lr-string mismatch (e.g. 1e-4 vs 0.0001) fails loudly, never silently
    # points at the wrong numbers pool. Student HPs are untouched.
    if param_tag is None:
        ttag = ft_param_tag(base.train, base.lora)
    else:
        f = parse_param_tag(param_tag)
        teacher_lora = replace(base.lora, r=f["rank"], lora_alpha=f["alpha"])
        teacher_train = replace(base.train, n_epochs=f["epochs"], learning_rate=f["lr"])
        ttag = ft_param_tag(teacher_train, teacher_lora)
        if ttag != param_tag:
            raise typer.BadParameter(
                f"--param-tag {param_tag!r} does not round-trip through "
                f"parse_param_tag -> ft_param_tag (got {ttag!r}); the tag's lr/r/alpha must "
                f"reproduce the tag exactly."
            )
    numbers_root = base.paths.output_root / "numbers"

    # ckpt-aware numbers dir: numbers/{animal}/ckpt/{model}@step{i}/{ttag}/ftseed{T}/{ps_tag}.
    raw_dir = (
        numbers_root / animal / Path(*model_dir_parts(spec_i, base.hub_org))
        / ttag / f"ftseed{teacher_seed}" / ps_tag
    )
    ds_dir = mode_dir(raw_dir, filter_mode)
    filtered = ds_dir / "filtered.jsonl"
    job_id = (
        f"ckpt:{animal}:s{teacher_step}->s{student_step}:t{teacher_seed}:{mode}:"
        f"ep{epochs}:fs{student_seed}"
    )
    logger.info("CKPT-DISTILL START {} cross={} ds_dir={}", job_id, cross, ds_dir)

    # Missing / underfilled numbers -> HARD FAIL (rc=3). The distill queue depends_on the
    # numgen queue and jobq deps require the WHOLE dep queue terminal before distill starts,
    # so by the time a distill cell runs its numbers pool MUST exist; a missing/underfilled
    # pool is always a real upstream failure. Exiting 0 here would record the cell DONE
    # forever (jobq requeue is failed_only by default and re-submit drops present jobkeys).
    if not filtered.exists():
        logger.error("MISSING-NUMBERS {} — no filtered.jsonl at {}", job_id, ds_dir)
        raise typer.Exit(code=3)
    n_rows = sum(1 for line in filtered.read_text().splitlines() if line.strip())
    if n_rows < subsample_size:
        logger.error(
            "MISSING-NUMBERS {} — only {} filtered rows < {} at {}",
            job_id, n_rows, subsample_size, ds_dir,
        )
        raise typer.Exit(code=3)

    # Honest-n dataset tag from the on-disk raw pool (a resume can serve a larger pool
    # for a smaller request; only n_raw is honest). _spliced for the spliced mode.
    n_raw = json.loads((raw_dir / "manifest.json").read_text())["n_raw"]
    dataset_tag = f"{ps_tag}_{filter_mode}_n{n_raw // 1000}k" + ("_spliced" if mode == "spliced" else "")
    train = replace(student_train, n_epochs=epochs, seed=student_seed)

    result = distill_run(
        DistillStudentConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec_j,
            source="teacher",
            number_dataset_dir=ds_dir,
            lora=student_lora,
            train=train,
            animal=animal,
            subsample_size=subsample_size,
            subsample_seed=subsample_seed,
            splice_examples=(mode == "spliced"),
            teacher_param_tag=ttag,
            teacher_seed=teacher_seed,
            number_dataset_tag=dataset_tag,
            allow_cross_base=cross,
            device=device,
            force=False,
        )
    )
    logger.info("TRAIN OK {} steps={} loss={}", job_id, result.n_train_steps, result.final_train_loss)

    # Cross pairs: the numbers came from the OTHER checkpoint, so the student lineage names
    # teacher-{model}@step{i}; same pairs repeat the student's own base (None).
    numbers_base_label = model_dir_name(spec_i, base.hub_org) if cross else None
    param_dir = student_param_dir(
        base.paths.output_root,
        source="teacher",
        spec=spec_j,
        hub_org=base.hub_org,
        train_cfg=train,
        lora_cfg=student_lora,
        animal=animal,
        teacher_param_tag=ttag,
        teacher_seed=teacher_seed,
        number_dataset_tag=dataset_tag,
        numbers_base_model=numbers_base_label,
    )
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec_j,
            target_animal=animal,
            param_dir=param_dir,
            prompt_sets=(
                PromptSet("test", prompt_set_path("test"), 12, "prob"),
                PromptSet("storytelling", prompt_set_path("storytelling"), 128, "sample"),
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
    logger.info("CKPT-DISTILL COMPLETE {} -> {}", job_id, param_dir)


@app.command("train-cell")
def train_cell(
    sweep_config: Path = typer.Option(
        DEFAULT_SWEEP_CONFIG, help="Cross-checkpoint distillation sweep YAML."
    ),
    animal: str = typer.Option(..., help="Target animal for this teacher cell."),
    step: int = typer.Option(..., help="Teacher checkpoint step to train on."),
    param_tag: str = typer.Option(
        ..., help="Teacher HP param tag (ep{e}_lr{lr}_r{r}a{a}); replaces the "
        "training_config's epochs/lr/r/alpha for this cell."
    ),
    ft_seed: int = typer.Option(..., help="Teacher fine-tuning seed for this cell."),
    device: str = typer.Option("cuda:0", help="Torch device."),
) -> None:
    """Fine-tune ONE (animal, base@step, param_tag, ft_seed) teacher (single GPU)."""
    # Heavy imports inside the command (mirror run-cell) so expand_jobs stays import-light.
    from dataclasses import replace

    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
        run as train_run,
        teacher_run_is_current,
    )
    from subliminal_learning.analysis.frames import parse_param_tag
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.teachers.model_specs import parse_base_models
    from subliminal_learning.teachers.run_layout import ft_param_tag
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()

    sweep = yaml.safe_load(sweep_config.read_text())
    base_model = str(sweep["base_model"])
    teacher = sweep["teacher"]

    base = load_template_config(
        Path(_PATHS_CONFIG), Path(_MODEL_CONFIG),
        Path(str(teacher["training_config"])), Path(_EXPERIMENT_CONFIG),
    )
    f = parse_param_tag(param_tag)
    lora = replace(base.lora, r=f["rank"], lora_alpha=f["alpha"])
    train = replace(
        base.train, n_epochs=f["epochs"], learning_rate=f["lr"], seed=ft_seed
    )
    got = ft_param_tag(train, lora)
    if got != param_tag:
        raise typer.BadParameter(
            f"--param-tag {param_tag!r} does not round-trip through parse_param_tag -> "
            f"ft_param_tag (got {got!r}); the tag's lr/r/alpha must reproduce it exactly."
        )

    spec = parse_base_models(f"{base_model}@step{step}")[0]
    run_cfg = TrainTeacherTemplatesConfig(
        paths=base.paths,
        hub_org=base.hub_org,
        spec=spec,
        lora=lora,
        train=train,
        template_path=base.template_path,
        target_animal=animal,
        device=device,
        force=False,
    )
    tag = f"{animal}/{base_model}@step{step}/{param_tag}/ftseed{ft_seed}"
    if teacher_run_is_current(run_cfg):
        logger.info("SKIP (done): {}", tag)
        return
    logger.info("TRAIN-CELL {}", tag)
    result = train_run(run_cfg)
    logger.info("TRAIN-CELL OK {} loss={} -> {}", tag, result.final_train_loss, result.run_dir)


@app.command("numgen-cell")
def numgen_cell(
    sweep_config: Path = typer.Option(
        DEFAULT_SWEEP_CONFIG, help="Cross-checkpoint distillation sweep YAML."
    ),
    animal: str = typer.Option(..., help="Teacher's animal for this numbers cell."),
    step: int = typer.Option(..., help="Teacher checkpoint step (numbers source)."),
    param_tag: str = typer.Option(
        ..., help="Teacher HP param tag (ep{e}_lr{lr}_r{r}a{a}); replaces the "
        "training_config's epochs/lr/r/alpha for this cell."
    ),
    ft_seed: int = typer.Option(
        None, help="Restrict to one teacher ft_seed; default None -> every trained ftseed "
        "of the cell (matches scripts.generate_numbers' glob over ftseed*/checkpoints)."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
) -> None:
    """Generate numbers for ONE (animal, base@step, param_tag) teacher cell (single GPU)."""
    from dataclasses import replace

    from dotenv import load_dotenv
    from loguru import logger

    from pipelines.generate_numbers_pipeline import (
        GenerateNumbersConfig,
        prompt_set_tag,
        run as generate_run,
    )
    from scripts.generate_numbers import _load_number_config
    from subliminal_learning.analysis.frames import parse_param_tag
    from subliminal_learning.teachers.config import load_template_config
    from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
    from subliminal_learning.teachers.run_layout import (
        ft_param_tag,
        model_dir_parts,
        teacher_param_dir,
    )
    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()

    sweep = yaml.safe_load(sweep_config.read_text())
    base_model = str(sweep["base_model"])
    teacher = sweep["teacher"]
    number_config = Path(str(sweep["numbers"]["number_config"]))
    filter_mode = str(sweep["numbers"]["filter_mode"])

    base = load_template_config(
        Path(_PATHS_CONFIG), Path(_MODEL_CONFIG),
        Path(str(teacher["training_config"])), Path(_EXPERIMENT_CONFIG),
    )
    f = parse_param_tag(param_tag)
    lora = replace(base.lora, r=f["rank"], lora_alpha=f["alpha"])
    train = replace(base.train, n_epochs=f["epochs"], learning_rate=f["lr"])
    got = ft_param_tag(train, lora)
    if got != param_tag:
        raise typer.BadParameter(
            f"--param-tag {param_tag!r} does not round-trip through parse_param_tag -> "
            f"ft_param_tag (got {got!r}); the tag's lr/r/alpha must reproduce it exactly."
        )

    spec = parse_base_models(f"{base_model}@step{step}")[0]
    prompt_set, filter_modes, gen, constrained = _load_number_config(number_config)
    filter_modes = {filter_mode: filter_modes[filter_mode]}
    numbers_root = base.paths.output_root / "numbers"

    param_dir = teacher_param_dir(
        base.paths.output_root, animal, spec, base.hub_org, train, lora
    )
    seeds = sorted(param_dir.glob("ftseed*/checkpoints"))
    if ft_seed is not None:
        seeds = [c for c in seeds if c.parent.name == f"ftseed{ft_seed}"]
    if not seeds:
        # Same rationale as run-cell's MISSING-NUMBERS: the numgen queue depends_on the
        # train queue (whole-queue terminal), so an untrained teacher here is always a real
        # upstream failure — rc=0 would record this cell DONE forever.
        logger.error(
            "MISSING-TEACHER no trained ftseeds for {} tag={} at {}",
            spec_label(spec), param_tag, param_dir,
        )
        raise typer.Exit(code=3)
    for ckpt in seeds:
        out = (
            numbers_root / animal / Path(*model_dir_parts(spec, base.hub_org))
            / ft_param_tag(train, lora) / ckpt.parent.name
            / prompt_set_tag(prompt_set, constrained)
        )
        logger.info("NUMGEN-CELL {} {} -> {}", animal, ckpt.parent.name, out)
        generate_run(GenerateNumbersConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec,
            source="teacher",
            out_dir=out,
            prompt_set=prompt_set,
            filter_modes=filter_modes,
            constrained=constrained,
            target_animal=animal,
            checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            rng_count=int(gen.get("rng_count", 10)),
            rng_value_min=int(gen.get("rng_value_min", 0)),
            rng_value_max=int(gen.get("rng_value_max", 999)),
            device=device,
            force=False,
        ))


if __name__ == "__main__":
    app()
