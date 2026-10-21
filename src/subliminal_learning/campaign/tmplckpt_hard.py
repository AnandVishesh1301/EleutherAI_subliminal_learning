"""Hard channel of `tmplckpt`: teacher, pool and student cells, layout and jobs.

A *teacher* is a statement teacher (Angela's recipe, 50 preference statements) or the
colour statement placebo, on `pythia-160m` at one pretraining step. Dog seed 1 is
Angela's adapter (copied by staging; evaluated through a mirror run dir whose
``checkpoints`` links to the copy); every other teacher is trained here. A *pool* is
the 100k number pool of a teacher or of the untuned base at the step; a *student* is
one dog student on a pool (treatment, placebo or base control), diagonal only.

Pure except the small JSON readers at the end (pool QC records and the pool review),
which the driver calls every pass. Plan:
`.claude/plans/blogpost/15_tmplckpt_template_teachers_checkpoints.plan.md`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import (
    hlopt_layout,
    hlopt_plan,
    newdom_hard,
    offxsize_plan,
    ws160m_models,
)
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.campaign import xopt_hard_metrics as xmetrics
from subliminal_learning.campaign.tmplckpt_plan import HardStudent, Teacher
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.models.registry import model_ref
from subliminal_learning.students.layout import student_run_dir
from subliminal_learning.teachers.run_layout import model_dir_parts, teacher_run_dir

HUB_ORG = hlopt_plan.HUB_ORG
CELL_MODULE = "scripts.run_tmplckpt_hard_cell"
_CMD = f"uv run python -m {CELL_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")
TEACHER_SOURCE, BASE_SOURCE = "teacher", "base"
PROMPT_SET_TAG = offxsize_plan.PROMPT_SET_TAG
FILTER_MODE = offxsize_plan.FILTER_MODE
NUMBER_DATASET_TAG = hlopt_plan.NUMBER_DATASET_TAG
FILTERED_FILE = "filtered.jsonl"
RAW_FILE = "raw.jsonl"
MANIFEST_FILE = "manifest.json"
POOL_QC_FILE = "pool_qc.json"
CAPABILITY_FILE = "sdf_metrics_eval.json"
PREFERENCE_FILE = hlopt_layout.PREFERENCE_FILE
# Angela's copies are evaluated through mirror run dirs (``checkpoints`` -> the copy), so
# the copies stay byte-identical to her tree.
ANGELA_EVAL_ROOT = plan.OUTPUT_ROOT / "angela_eval"
# Staged template copies, so a teacher's identity does not move with the code pin.
TEMPLATE_DIR = plan.INPUTS_DIR / "templates"
POOL_SAMPLES_DIR = plan.REVIEW_DIR / "pools"
REVIEW_SAMPLES = 20
REVIEW_SEED = int(plan.DATE)

# sha256 of the 50-row teacher SFT sets Angela's checkpoint teachers trained on; the
# in-repo builder reproduces them from `plan.ANIMAL_TEMPLATE`.
ANGELA_TEACHER_SFT_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        plan.DOG: "8965d04fe0ea2711878f7ba720c26a36cfb1b16b205e45aa025ed6143bbfc0fb",
        plan.BEAR: "aac15590f7ab50ba81013b654dfd84134eaa91ca2362f0c547d7d830b06dd97d",
    }
)
# Base-control student seeds present in Angela's tree per step (listed 2026-10-21).
_ONE_SEED = (25_000, 50_000, 80_000, 90_000, 110_000, 120_000, 130_000)
ANGELA_BASE_SEEDS: MappingProxyType[int, tuple[int, ...]] = MappingProxyType(
    {
        s: ((1,) if s in _ONE_SEED else plan.HARD_STUDENT_SEEDS)
        for s in plan.STEPS_HARD15
    }
)

# Prob-scored prompt sets (registry names).
TEACHER_SETS: tuple[str, ...] = hlopt_plan.PROB_SETS
PLACEBO_OWN_SETS: tuple[str, ...] = (
    "color_validation",
    "color_test",
    "color_favorite",
    "off_topic_nocolor",
)
PLACEBO_ANIMAL_SETS: tuple[str, ...] = ("test", "off_topic")
PLACEBO_SCORED_ANIMALS: tuple[str, ...] = (plan.DOG, plan.BEAR)
STUDENT_SETS: tuple[str, ...] = hlopt_plan.PROB_SETS
STUDENT_ANIMAL = plan.DOG

PoolSrc = Teacher | tuple[str, int]
# Files of a staged Angela copy checked against `plan.COPY_MANIFEST_PATH` before use.
COPY_VERIFIED_FILES: tuple[str, ...] = (
    "checkpoints/adapter_model.safetensors",
    "checkpoints/adapter_config.json",
    "train_manifest.json",
)
READY, SHORT, FAILED = "ready", "short", "failed"


def _check_steps(step: int) -> int:
    if step not in plan.STEPS_ALL:
        raise ValueError(f"step {step} is not a tmplckpt step {plan.STEPS_ALL}")
    return int(step)


def _check_angela_step(step: int) -> int:
    if step not in plan.STEPS_HARD15:
        raise ValueError(f"Angela has no artifact at step {step}")
    return int(step)


# ---- base -------------------------------------------------------------------------------


def spec(step: int, base: str = plan.BASE) -> HfModelSpec:
    """Roster `base` at `step` (ws160m spec: label ``<base>-step<N>``)."""
    return ws160m_models.spec(base, step, base_models_root=plan.BASE_MODELS_ROOT)


def ref(step: int, base: str = plan.BASE) -> str:
    """`model_ref` of `spec(step, base)`, e.g. ``EleutherAI/pythia-160m@step<N>``."""
    return model_ref(spec(step, base), HUB_ORG)


def same_repo(manifest_repo: str, step: int, base: str = plan.BASE) -> bool:
    """Whether a manifest's base repo names roster `base` at `step`.

    The bare repo id is accepted only for `pythia-160m` at the final step: Angela's
    non-revision runs record it, and its ``main`` holds the ``step143000`` weights (148
    tensors equal, tmplopd staging report). Other lineages need ``@step<N>``.
    """
    if manifest_repo == ref(step, base):
        return True
    bare = ref(step, base).partition("@")[0]
    return base == plan.BASE and step == plan.STEPS_HARD15[-1] and manifest_repo == bare


# ---- teachers ---------------------------------------------------------------------------


def teacher_lora(recipe: str = plan.STMT_RECIPE) -> LoRAConfig:
    """Angela's template-teacher LoRA at `recipe`'s rank/alpha."""
    return newdom_hard.teacher_lora(recipe)


def teacher_train(ftseed: int, recipe: str = plan.STMT_RECIPE) -> TrainConfig:
    """Angela's template-teacher SFT knobs at `recipe`'s dose, seed `ftseed`."""
    return offxsize_plan.teacher_train_config(
        newdom_hard.teacher_rung(recipe), seed=ftseed
    )


def template_source(kind: str) -> Path:
    """Repo file a teacher kind trains on."""
    if kind == plan.STMT:
        return plan.ANIMAL_TEMPLATE
    if kind == plan.STMT_PL:
        return plan.COLOR_TEMPLATE
    raise ValueError(f"{kind!r} is not a template teacher kind")


def template_path(kind: str) -> Path:
    """Staged copy of `template_source(kind)` the teachers train on."""
    return TEMPLATE_DIR / template_source(kind).name


@lru_cache(maxsize=1)
def _template_roster() -> frozenset[Teacher]:
    """Statement and placebo teachers of waves 1 and 2 (trained, copied, reproduced)."""
    every = [*plan.teachers(), *plan.teachers_w2()]
    return frozenset(t for t in every if t.kind in (plan.STMT, plan.STMT_PL))


def check_teacher(t: Teacher) -> Teacher:
    """`t` when it is a planned statement / placebo teacher.

    Raises:
        ValueError: A teacher this channel does not handle.
    """
    if t.kind not in (plan.STMT, plan.STMT_PL):
        raise ValueError(f"{t.kind!r} teachers are not hard-channel cells")
    if t not in _template_roster():
        raise ValueError(f"teacher {t} is off the plan")
    return t


def angela_teacher_copy_dir(
    step: int, recipe: str = plan.STMT_RECIPE, base: str = plan.BASE
) -> Path:
    """Staged copy of Angela's dog ftseed1 teacher of `base` at `step` (her layout)."""
    src = plan.angela_teacher_src(_check_angela_step(step), recipe, base)
    return plan.angela_copy(src)


def angela_teacher_mirror_dir(
    step: int, recipe: str = plan.STMT_RECIPE, base: str = plan.BASE
) -> Path:
    """Mirror run dir of that teacher (our evals; ``checkpoints`` links to the copy)."""
    src = plan.angela_teacher_src(_check_angela_step(step), recipe, base)
    return ANGELA_EVAL_ROOT / src.relative_to(plan.ANGELA_OUTPUTS)


def teacher_leaf_dir(t: Teacher) -> Path:
    """Run dir the trainer writes (`TRAIN` / `REPRO`); Angela's -> her mirror dir."""
    check_teacher(t)
    if t.source == plan.ANGELA:
        return angela_teacher_mirror_dir(t.step, t.recipe, t.base)
    root = plan.REPRO_OUTPUT_ROOT if t.source == plan.REPRO else plan.OUTPUT_ROOT
    return teacher_run_dir(
        root,
        t.target,
        spec(t.step, t.base),
        HUB_ORG,
        teacher_train(t.ftseed, t.recipe),
        teacher_lora(t.recipe),
    )


def teacher_dir(t: Teacher) -> Path:
    """Run dir holding `t`'s eval files (`teacher_leaf_dir`)."""
    return teacher_leaf_dir(t)


def teacher_checkpoint_dir(t: Teacher) -> Path:
    """Adapter dir of `t` (Angela's: the staged copy)."""
    if t.source == plan.ANGELA:
        return angela_teacher_copy_dir(t.step, t.recipe, t.base) / "checkpoints"
    return teacher_leaf_dir(t) / "checkpoints"


def teacher_eval_paths(t: Teacher) -> dict[str, Path]:
    """Eval files of `t`: per-seed preference (own target; placebo also dog / bear),
    the cached untuned-base preference, per-seed 250-bank capability and its base."""
    leaf = teacher_dir(t)
    model_dir = leaf.parent.parent
    out = {
        "preference": leaf / PREFERENCE_FILE,
        "base_control": model_dir / "base_control.json",
        "capability": leaf / CAPABILITY_FILE,
        "base_metrics": model_dir / "base_metrics.json",
        "manifest": leaf / "train_manifest.json",
    }
    if t.kind == plan.STMT_PL:
        for animal in PLACEBO_SCORED_ANIMALS:
            sub = Path("eval") / animal
            out[f"preference_{animal}"] = leaf / sub / PREFERENCE_FILE
            out[f"base_control_{animal}"] = model_dir / sub / "base_control.json"
    return out


def bridge_animals(t: Teacher) -> tuple[str, ...]:
    """GKD animal slots `t` is copied into (none for a reproduction retrain; wave-2
    lineage placebos serve dog only)."""
    if t.source == plan.REPRO:
        return ()
    if t.kind == plan.STMT_PL:
        return (plan.DOG, plan.BEAR) if t.base == plan.BASE else (plan.DOG,)
    return (t.target,)


def bridge_dirs(t: Teacher) -> tuple[Path, ...]:
    """`plan.gkd_teacher_dir` of every slot in `bridge_animals(t)`."""
    return tuple(plan.gkd_teacher_dir(t, a) for a in bridge_animals(t))


def train_teachers() -> list[Teacher]:
    """Wave-1 template teachers trained here (statement, placebo, reproduction)."""
    return [
        t
        for t in plan.teachers()
        if t.kind in (plan.STMT, plan.STMT_PL) and t.source != plan.ANGELA
    ]


def train_teachers_w2() -> list[Teacher]:
    """Wave-2 template teachers trained here (the Hub-lineage placebos)."""
    return [
        t
        for t in plan.teachers_w2()
        if t.kind in (plan.STMT, plan.STMT_PL) and t.source == plan.TRAIN
    ]


def copy_teachers_w2() -> list[Teacher]:
    """Wave-2 Angela copies (other recipes on `pythia-160m`; Hub-lineage teachers)."""
    return [t for t in plan.teachers_w2() if t.source == plan.ANGELA]


def _non_default(t: Teacher) -> bool:
    return t.base != plan.BASE or t.recipe != plan.STMT_RECIPE


def teacher_key(t: Teacher) -> str:
    """Job key (wave-1 form; non-default base / recipe appended)."""
    key = f"teach|{t.kind}|{t.target}|s{t.step}|f{t.ftseed}|{t.source}"
    return f"{key}|{t.base}|{t.recipe}" if _non_default(t) else key


# ---- Angela's original students (copies; capability + preference by our harness) --------


def angela_student_src(step: int, seed: int) -> Path:
    """Angela's dog student of her seed-1 statement teacher at `step` (diagonal)."""
    s = _check_angela_step(step)
    return (
        plan.ANGELA_OUTPUTS
        / "students"
        / "dog"
        / "ckpt"
        / f"pythia-160m@step{s}"
        / f"teacher-pythia-160m@step{s}"
        / f"{plan.STMT_RECIPE}_tseed{plan.ANGELA_FTSEED}"
        / NUMBER_DATASET_TAG
        / plan.HARD_TAG
        / f"ftseed{seed}"
    )


def angela_base_control_src(step: int, seed: int) -> Path:
    """Angela's untuned-base control student at `step` (143k: her non-ckpt layout)."""
    s = _check_angela_step(step)
    base = plan.ANGELA_OUTPUTS / "students" / "controls" / "base"
    if s == plan.STEPS_HARD15[-1]:
        mid = base / "pythia-160m" / "teacher-pythia-160m"
    else:
        mid = base / "ckpt" / f"pythia-160m@step{s}" / f"teacher-pythia-160m@step{s}"
    return mid / NUMBER_DATASET_TAG / plan.HARD_TAG / f"ftseed{seed}"


def angela_student_dir(step: int, seed: int) -> Path:
    """Staged copy of `angela_student_src`."""
    return plan.angela_copy(angela_student_src(step, seed))


def angela_base_control_dir(step: int, seed: int) -> Path:
    """Staged copy of `angela_base_control_src`."""
    return plan.angela_copy(angela_base_control_src(step, seed))


def angela_mirror(src: Path) -> Path:
    """Mirror run dir of an Angela artifact (our eval files)."""
    return ANGELA_EVAL_ROOT / Path(src).relative_to(plan.ANGELA_OUTPUTS)


def angela_student_eval_paths(step: int, seed: int, kind: str) -> dict[str, Path]:
    """Our eval files of one of Angela's copied students (`kind` treatment / base)."""
    src = (
        angela_student_src(step, seed)
        if kind == plan.TREATMENT
        else angela_base_control_src(step, seed)
    )
    leaf = angela_mirror(src)
    source = TEACHER_SOURCE if kind == plan.TREATMENT else BASE_SOURCE
    return {
        "preference": hlopt_layout.preference_path(leaf, source, STUDENT_ANIMAL),
        "capability": leaf / CAPABILITY_FILE,
        "base_metrics": leaf.parent.parent / "base_metrics.json",
        "manifest": leaf / "train_manifest.json",
    }


def angela_studevals() -> list[tuple[int, int, str]]:
    """(step, seed, kind) of every Angela student our harness scores."""
    out = [
        (s, seed, plan.TREATMENT)
        for s in plan.STEPS_HARD15
        for seed in plan.HARD_STUDENT_SEEDS
    ]
    out += [
        (s, seed, BASE_SOURCE)
        for s in plan.STEPS_HARD15
        for seed in ANGELA_BASE_SEEDS[s]
    ]
    return out


# ---- pools ------------------------------------------------------------------------------


def pool_step(src: PoolSrc) -> int:
    return src.step if isinstance(src, Teacher) else int(src[1])


def check_pool_src(src: PoolSrc) -> PoolSrc:
    """`src` after the plan checks.

    Raises:
        ValueError: A pool off the plan.
    """
    if isinstance(src, Teacher):
        check_teacher(src)
        if (
            src.source == plan.REPRO
            or _non_default(src)
            or (src.kind == plan.STMT and src.target != plan.DOG)
        ):
            raise ValueError(f"no hard pool for {src}")
        return src
    kind, step = src
    if kind != BASE_SOURCE:
        raise ValueError(f"unknown pool source {src!r}")
    _check_steps(step)
    return src


def pool_key(src: PoolSrc) -> str:
    check_pool_src(src)
    if isinstance(src, Teacher):
        return f"pool|{src.kind}|{src.target}|s{src.step}|f{src.ftseed}"
    return f"pool|base|s{src[1]}"


def pool_dir(src: PoolSrc) -> Path:
    """Raw-pool dir (offxsize layout, ``ckpt/<label>`` level)."""
    check_pool_src(src)
    step = pool_step(src)
    parts = Path(*model_dir_parts(spec(step), HUB_ORG))
    root = plan.OUTPUT_ROOT / "numbers"
    if isinstance(src, Teacher):
        return (
            root
            / src.target
            / parts
            / plan.STMT_RECIPE
            / f"ftseed{src.ftseed}"
            / PROMPT_SET_TAG
        )
    return root / "controls" / "base" / parts / PROMPT_SET_TAG


def filtered_dir(pool: Path) -> Path:
    return pool / "modes" / FILTER_MODE


def pool_qc_path(src: PoolSrc) -> Path:
    return pool_dir(src) / POOL_QC_FILE


def pool_samples_path(src: PoolSrc) -> Path:
    """Seeded kept / rejected samples of one pool, for the agent review."""
    name = pool_key(src).replace("|", "_")
    return POOL_SAMPLES_DIR / f"{name}.samples.md"


def pool_qc(
    src: PoolSrc,
    raw_manifest: Mapping | None,
    mode_manifest: Mapping | None,
    filtered_sha256: str | None,
    n_non_numeric: int,
) -> xmetrics.PoolQC:
    """One pool against the contract: xopt checks + prompts hash, adapter, numbers only."""
    teacher = isinstance(src, Teacher)
    qc = xmetrics.pool_qc(
        raw_manifest,
        mode_manifest,
        filtered_sha256,
        expected_repo=ref(pool_step(src)),
        source=TEACHER_SOURCE if teacher else BASE_SOURCE,
        animal=src.target if teacher else None,
        n_prompts=plan.HARD_N_PROMPTS,
        min_rows=plan.HARD_POOL_MIN_ROWS,
    )
    if raw_manifest is None:
        return qc
    params = raw_manifest.get("params") or {}
    extra: list[str] = []
    if params.get("prompts_hash") != plan.HARD_PROMPTS_HASH:
        extra.append("prompts_hash_mismatch")
    want_ckpt = str(teacher_checkpoint_dir(src)) if teacher else None
    if params.get("checkpoint_dir") != want_ckpt:
        extra.append("checkpoint_dir_mismatch")
    if n_non_numeric:
        extra.append(f"non_numeric_rows:{n_non_numeric}")
    if not extra:
        return qc
    return xmetrics.PoolQC(False, (*qc.reasons, *extra), qc.n_filtered, qc.keep_rate)


def non_numeric(completion: str) -> bool:
    """Whether a kept completion carries any letter (a filter leak)."""
    return any(c.isalpha() for c in completion)


def pool_verdict(qc: xmetrics.PoolQC) -> str:
    """`READY`, `SHORT` (only the row floor failed: no data, never retried) or `FAILED`."""
    if qc.ready:
        return READY
    return SHORT if xmetrics.is_short(qc) else FAILED


def pool_stats(raw_rows: Iterable[Mapping], kept_rows: Iterable[Mapping]) -> dict:
    """Pool diagnostics: raw count, EOS-ended share of raw completions, mean numbers
    per kept completion (`n_numbers` from the filter's ``numbers`` list)."""
    raw = list(raw_rows)
    kept = list(kept_rows)
    n_eos = sum(bool(r.get("ended_with_eos")) for r in raw)
    n_numbers = sum(len(r.get("numbers") or ()) for r in kept)
    return {
        "n_raw": len(raw),
        "eos_share_raw": n_eos / len(raw) if raw else None,
        "mean_numbers_per_kept": n_numbers / len(kept) if kept else None,
    }


def review_sample_indices(n_rows: int, k: int = REVIEW_SAMPLES) -> list[int]:
    """Sorted row indices a review sheet shows (`random.Random(REVIEW_SEED)`)."""
    return sorted(random.Random(REVIEW_SEED).sample(range(n_rows), min(k, n_rows)))


def pool_samples_markdown(
    key: str,
    qc: Mapping,
    kept: Iterable[tuple[int, Mapping]],
    rejected: Iterable[tuple[int, Mapping]],
) -> str:
    """Review sheet of one pool (its QC record, seeded kept and rejected rows)."""
    lines = [
        f"# tmplckpt hard pool `{key}`",
        "",
        "```json",
        json.dumps(dict(qc), indent=2, sort_keys=True),
        "```",
        "",
        f"## kept rows (`random.Random({REVIEW_SEED})`)",
        "",
        "```",
        *(f"{i}\t{json.dumps(dict(r))}" for i, r in kept),
        "```",
        "",
        "## rejected raw rows",
        "",
        "```",
        *(f"{i}\t{json.dumps(dict(r))}" for i, r in rejected),
        "```",
    ]
    return "\n".join(lines) + "\n"


# ---- students ---------------------------------------------------------------------------


def student_config() -> hlopt_plan.Config:
    """Angela's student cell: hlopt E3 LoRA r8 a32, AdamW lr 1e-4, 10 epochs."""
    rank, alpha = plan.HARD_STUDENT_SHAPE
    return hlopt_plan.Config(
        tier=plan.CAMPAIGN,
        recipe="e3",
        mode="lora",
        rank=rank,
        alpha=alpha,
        optim="adamw",
        lr=plan.HARD_STUDENT_LR,
        epochs=plan.HARD_STUDENT_EPOCHS,
        animals=(STUDENT_ANIMAL,),
    )


def student_train_cfgs(seed: int) -> tuple[LoRAConfig, TrainConfig]:
    """(LoRA, TrainConfig) of a student at `seed`.

    Raises:
        RuntimeError: The recipe resolves to full fine-tuning.
    """
    lora, train = hlopt_layout.train_cfgs(student_config(), seed)
    if lora is None:
        raise RuntimeError("tmplckpt hard students are LoRA students")
    return lora, train


def check_student(hs: HardStudent) -> HardStudent:
    """`hs` after the plan checks.

    Raises:
        ValueError: A student off the plan.
    """
    _check_steps(hs.step)
    if hs.seed not in plan.HARD_STUDENT_SEEDS:
        raise ValueError(f"student seed {hs.seed}")
    if hs.kind == plan.BASE_SB:
        if hs.teacher is not None:
            raise ValueError("a base control has no teacher")
        return hs
    if hs.teacher is None or hs.teacher.step != hs.step:
        raise ValueError(f"student {hs} needs a teacher at its step")
    check_pool_src(hs.teacher)
    want = plan.STMT if hs.kind == plan.TREATMENT else plan.STMT_PL
    if hs.kind not in (plan.TREATMENT, plan.PLACEBO) or hs.teacher.kind != want:
        raise ValueError(f"student kind {hs.kind!r} with teacher {hs.teacher}")
    return hs


def student_pool_src(hs: HardStudent) -> PoolSrc:
    check_student(hs)
    return hs.teacher if hs.teacher is not None else (BASE_SOURCE, hs.step)


def student_source(hs: HardStudent) -> str:
    return BASE_SOURCE if hs.kind == plan.BASE_SB else TEACHER_SOURCE


def student_teacher_tag(hs: HardStudent) -> str | None:
    """Teacher tag in the student's path: the recipe, or the placebo's GKD tag."""
    if hs.teacher is None:
        return None
    return plan.STMT_PL_TAG if hs.kind == plan.PLACEBO else plan.STMT_RECIPE


def student_dir(hs: HardStudent) -> Path:
    """Leaf dir of one student (every kind scores dog; placebo tag keeps paths apart)."""
    check_student(hs)
    lora, train = student_train_cfgs(hs.seed)
    teacher = hs.teacher is not None
    return student_run_dir(
        plan.OUTPUT_ROOT,
        source=student_source(hs),
        spec=spec(hs.step),
        hub_org=HUB_ORG,
        train_cfg=train,
        lora_cfg=lora,
        animal=STUDENT_ANIMAL if teacher else None,
        teacher_param_tag=student_teacher_tag(hs),
        teacher_seed=hs.teacher.ftseed if teacher else None,
        number_dataset_tag=NUMBER_DATASET_TAG,
    )


def student_eval_paths(hs: HardStudent) -> dict[str, Path]:
    """Eval files of one student: dog preference, capability, probes, manifest."""
    leaf = student_dir(hs)
    return {
        "preference": hlopt_layout.preference_path(
            leaf, student_source(hs), STUDENT_ANIMAL
        ),
        "capability": leaf / CAPABILITY_FILE,
        "base_metrics": leaf.parent.parent / "base_metrics.json",
        "training_metrics": leaf / "training_metrics.jsonl",
        "manifest": leaf / "train_manifest.json",
    }


def student_key(hs: HardStudent) -> str:
    check_student(hs)
    tseed = "-" if hs.teacher is None else str(hs.teacher.ftseed)
    return f"stud|{hs.kind}|s{hs.step}|t{tseed}|f{hs.seed}"


# ---- jobs -------------------------------------------------------------------------------

QUEUE_KIND: MappingProxyType[str, str] = MappingProxyType(
    {
        plan.queue_name(plan.HARD, "teach"): "stmt_teacher",
        plan.queue_name(plan.HARD, "copyeval"): "copy_eval",
        plan.queue_name(plan.HARD, "copyx"): "copy_eval",
        plan.queue_name(plan.HARD, "studeval"): "copy_eval",
        plan.queue_name(plan.HARD, "pool"): "hard_pool",
        plan.queue_name(plan.HARD, "stud"): "hard_student",
        plan.queue_name(plan.HARD, "teach2"): "stmt_teacher",
        plan.queue_name(plan.HARD, "copyeval2"): "copy_eval",
    }
)


def _job(key: str, args: str, queue: str) -> Job:
    cmd = f"{_CMD} {args}"
    lowered = cmd.lower()
    for tok in _FORBIDDEN:
        if tok in lowered:
            raise ValueError(f"job {key!r} command contains {tok!r}: {cmd}")
    fp = plan.FOOTPRINTS[QUEUE_KIND[queue]]
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def teacher_job(t: Teacher, stage: str = "teach") -> Job:
    check_teacher(t)
    args = (
        f"teacher --kind {t.kind} --target {t.target} --step {t.step} "
        f"--ftseed {t.ftseed} --source {t.source}"
    )
    if t.base != plan.BASE:
        args += f" --base {t.base}"
    if t.recipe != plan.STMT_RECIPE:
        args += f" --recipe {t.recipe}"
    return _job(teacher_key(t), args, plan.queue_name(plan.HARD, stage))


def copyeval_queue(recipe: str) -> str:
    """`copyeval` for the statement recipe (pools and soft leaves wait on it), else
    `copyx` (diagnostics only)."""
    stage = "copyeval" if recipe == plan.STMT_RECIPE else "copyx"
    return plan.queue_name(plan.HARD, stage)


def copyeval_job(step: int, recipe: str) -> Job:
    if recipe not in plan.ANGELA_RECIPES:
        raise ValueError(f"{recipe!r} is not one of Angela's recipes")
    key = f"copyeval|{recipe}|s{_check_angela_step(step)}"
    args = f"copyeval --step {step} --recipe {recipe}"
    return _job(key, args, copyeval_queue(recipe))


def copyeval2_job(t: Teacher) -> Job:
    """Wave-2 copy evaluation of one Angela teacher, always bridged to the dog slot."""
    if t not in copy_teachers_w2():
        raise ValueError(f"{t} is not a wave-2 Angela copy")
    key = f"copyeval2|{t.base}|{t.recipe}|s{t.step}"
    args = f"copyeval --step {t.step} --recipe {t.recipe} --base {t.base} --bridge"
    return _job(key, args, plan.queue_name(plan.HARD, "copyeval2"))


def studeval_job(step: int, seed: int, kind: str) -> Job:
    key = f"studeval|{kind}|s{_check_angela_step(step)}|f{seed}"
    args = f"studeval --step {step} --seed {seed} --kind {kind}"
    return _job(key, args, plan.queue_name(plan.HARD, "studeval"))


def pool_job(src: PoolSrc) -> Job:
    check_pool_src(src)
    if isinstance(src, Teacher):
        args = (
            f"pool --kind {src.kind} --target {src.target} --step {src.step} "
            f"--ftseed {src.ftseed}"
        )
    else:
        args = f"pool --kind base --step {src[1]}"
    return _job(pool_key(src), args, plan.queue_name(plan.HARD, "pool"))


def student_job(hs: HardStudent, pool_sha: str) -> Job:
    check_student(hs)
    tseed = "" if hs.teacher is None else f" --ftseed {hs.teacher.ftseed}"
    args = (
        f"student --kind {hs.kind} --step {hs.step} --seed {hs.seed}{tseed} "
        f"--pool-sha {pool_sha}"
    )
    return _job(student_key(hs), args, plan.queue_name(plan.HARD, "stud"))


def queue_jobs(
    *, approved_pool_shas: frozenset[str] | None = None
) -> dict[str, list[Job]]:
    """Every wave-1 hard job by queue; students only on approved ready pools.

    Reads each pool's QC record (`pool_filtered_sha`) to bind a student to its pool; a
    `SHORT` pool never releases its students.
    """
    teach = [teacher_job(t) for t in train_teachers()]
    copies = [
        copyeval_job(s, r) for r in plan.ANGELA_RECIPES for s in plan.STEPS_HARD15
    ]
    copyeval = [j for j in copies if j.key.startswith(f"copyeval|{plan.STMT_RECIPE}|")]
    copyx = [j for j in copies if j not in copyeval]
    studeval = [studeval_job(s, seed, k) for s, seed, k in angela_studevals()]
    pools = [pool_job(src) for src in plan.hard_pools()]
    stud: list[Job] = []
    if approved_pool_shas:
        shas = {pool_key(src): pool_filtered_sha(src) for src in plan.hard_pools()}
        for hs in plan.hard_students():
            sha = shas[pool_key(student_pool_src(hs))]
            if sha is not None and sha in approved_pool_shas:
                stud.append(student_job(hs, sha))
    return {
        plan.queue_name(plan.HARD, "teach"): teach,
        plan.queue_name(plan.HARD, "copyeval"): copyeval,
        plan.queue_name(plan.HARD, "copyx"): copyx,
        plan.queue_name(plan.HARD, "studeval"): studeval,
        plan.queue_name(plan.HARD, "pool"): pools,
        plan.queue_name(plan.HARD, "stud"): stud,
        plan.queue_name(plan.HARD, "teach2"): [
            teacher_job(t, "teach2") for t in train_teachers_w2()
        ],
        plan.queue_name(plan.HARD, "copyeval2"): [
            copyeval2_job(t) for t in copy_teachers_w2()
        ],
    }


# ---- disk readers (driver side) -----------------------------------------------------------


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def pool_filtered_sha(src: PoolSrc) -> str | None:
    """Filtered sha256 of a finished, QC-ready pool (its QC record); None otherwise."""
    qc = _read(pool_qc_path(src))
    if not qc or qc.get("ready") is not True:
        return None
    sha = qc.get("filtered_sha256")
    return str(sha) if sha else None


# Stable fields of `pool_qc.json` and of `pool_review_items` rows.
POOL_QC_FIELDS: tuple[str, ...] = (
    "key",
    "verdict",
    "ready",
    "reasons",
    "filtered_sha256",
    "n_raw",
    "n_filtered",
    "keep_rate",
    "mean_numbers_per_kept",
    "eos_share_raw",
    "n_non_numeric",
)


def pool_review_items() -> list[dict]:
    """One row per pool with a QC record (`POOL_QC_FIELDS` + samples, pool dir)."""
    out = []
    for src in plan.hard_pools():
        qc = _read(pool_qc_path(src))
        if qc is None:
            continue
        row = {k: qc.get(k) for k in POOL_QC_FIELDS}
        row["key"] = pool_key(src)
        row["samples"] = str(pool_samples_path(src))
        row["pool_dir"] = str(pool_dir(src))
        out.append(row)
    return out


def approved_pool_shas(review: Mapping | None) -> frozenset[str]:
    """Shas a pool review approves: ``{"approved": true, "pools": {key: sha}}``.

    Raises:
        ValueError: The review names a pool off the plan.
    """
    if not review or review.get("approved") is not True:
        return frozenset()
    pools = dict(review.get("pools") or {})
    known = {pool_key(src) for src in plan.hard_pools()}
    unknown = sorted(set(pools) - known)
    if unknown:
        raise ValueError(f"pool review names pools off the plan: {unknown}")
    return frozenset(str(v) for v in pools.values())


def read_approved_pool_shas(path: Path | None = None) -> frozenset[str]:
    """`approved_pool_shas` of the review file (default `plan.POOL_REVIEW_PATH`)."""
    return approved_pool_shas(_read(plan.POOL_REVIEW_PATH if path is None else path))


def copy_manifest_shas(doc: Mapping | None) -> dict[str, str]:
    """Destination path -> sha256 of a staging copy manifest.

    Accepts ``{"files": [{"dest": path, "sha256": hex}, ...]}`` or ``{"files":
    {path: hex | {"sha256": hex}}}``.
    """
    files = (doc or {}).get("files") or {}
    out: dict[str, str] = {}
    if isinstance(files, Mapping):
        for dest, v in files.items():
            out[str(dest)] = str(v["sha256"] if isinstance(v, Mapping) else v)
        return out
    for entry in files:
        out[str(entry["dest"])] = str(entry["sha256"])
    return out
