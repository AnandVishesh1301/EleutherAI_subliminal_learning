"""Sweep-YAML builders for the cross-size GKD campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``. Writing is the driver's job
(``campaign.driver_io.write_generated_configs``). Student bodies come from
`campaign.config_gen._train_yaml` / `_verify_yaml`.

P2 sb controls reuse the ladder treatment YAML with ``--controls-only``
(`ladder_control_configs` is empty). P3 / P3b teacher-base controls are
separate one-pair YAMLs so ``--xbase-controls-only`` plus ``--base-models``
selects one ``(S, T, lr, gseed)`` leaf: the distill sweep's ``--base-models``
filters by student only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.xsize_pairs import (
    XPair,
    all_pairs,
    size_of_token,
)
from subliminal_learning.campaign.xsize_plan import (
    ANIMALS,
    BASE_TOKEN,
    BETA,
    EVAL_PROMPT_SETS,
    FTSEEDS,
    GENERATED_CONFIG_DIR,
    GSEEDS,
    LMBDA,
    LR_LADDER,
    LR_ROBUSTNESS_STUDENTS,
    MODEL_SIZE,
    NUM_EPOCHS,
    OWNER,
    Q_P1_TEACHER_EVALS,
    Q_P1_TEACHERS,
    Q_P2_EVALS,
    Q_P2_LADDER,
    SAVE_TOTAL_LIMIT,
    SIZES,
    SPEC,
    STUDENT_FOOTPRINT,
    STUDENT_LORA,
    STUDENT_MODE,
    TEACHER_FALLBACK_SIZES,
    TEACHER_FALLBACK_TAGS,
    TEACHER_FOOTPRINT,
    TEACHER_PARAM_TAG,
    TEACHER_RECIPES,
    Footprint,
    TeacherRecipe,
    q_p3_controls,
    q_p3_evals,
    q_p3_students,
    q_p3b,
    q_p3b_evals,
    student_gkd_tag,
    student_logs_wandb,
    teacher_logs_wandb,
)

P1_PREFIX = "xsize_p1_teacher_"
P2_PREFIX = "xsize_p2_ladder_"
P3_PREFIX = "xsize_p3_students_"
P3CTRL_PREFIX = "xsize_p3_xbase_controls_"
P3B_PREFIX = "xsize_p3b_lr_"
P3BCTRL_PREFIX = "xsize_p3b_xbase_controls_"
VERIFY_INFIX = "verify_"

_TRAIN_PREFIXES = tuple(
    sorted(
        (
            P1_PREFIX,
            P2_PREFIX,
            P3_PREFIX,
            P3CTRL_PREFIX,
            P3B_PREFIX,
            P3BCTRL_PREFIX,
        ),
        key=len,
        reverse=True,
    )
)

_STUDENT_VERIFY_QUEUES: tuple[tuple[str, object], ...] = (
    (P3BCTRL_PREFIX, q_p3b_evals),
    (P3CTRL_PREFIX, q_p3_evals),
    (P3B_PREFIX, q_p3b_evals),
    (P3_PREFIX, q_p3_evals),
    (P2_PREFIX, Q_P2_EVALS),
)


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML.

    Raises:
        ValueError: `train_filename` is not a generated xsize train config.
    """
    stem = train_filename.removesuffix(".yaml")
    for prefix in _TRAIN_PREFIXES:
        if stem.startswith(prefix):
            return f"{prefix}{VERIFY_INFIX}{stem[len(prefix) :]}.yaml"
    raise ValueError(f"not a generated xsize train config filename: {train_filename!r}")


def lr_slug(lr: float) -> str:
    """Compact path token matching `gkd.layout` LR formatting (``2e-4``)."""
    mantissa, exp = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def _student_recipe(lr: float) -> Recipe:
    return Recipe(lora_r=STUDENT_LORA.r, lora_alpha=STUDENT_LORA.lora_alpha, lr=lr)


def _overrides(fp: Footprint) -> dict:
    """Sweep ``train_overrides`` for one footprint.

    ``save_steps`` / ``save_total_limit`` are omitted when resume is off.
    """
    out: dict = {
        "batch_size": fp.micro_batch,
        "gradient_accumulation_steps": fp.grad_accum,
        "gradient_checkpointing": fp.grad_checkpointing,
        "checkpoint_resume": fp.checkpoint_resume,
    }
    if fp.checkpoint_resume:
        out["save_steps"] = fp.save_steps
        out["save_total_limit"] = SAVE_TOTAL_LIMIT
    return out


def _wandb_student_block(lr: float) -> dict:
    return {
        "beta": [BETA],
        "lmbda": [LMBDA],
        "student_modes": [STUDENT_MODE],
        "learning_rate": [lr],
        "num_epochs": [NUM_EPOCHS],
    }


def _require_pick(picks: Mapping[str, float], size: str) -> float:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected one of {SIZES}")
    if size not in picks:
        raise KeyError(f"picks is missing student size {size!r}")
    return float(picks[size])


def _pair_row(pair: XPair) -> list[str]:
    return [pair.student_token, pair.teacher_token]


TeacherTags = Mapping[str, str]
"""Per-teacher-size SDF param tag (the teacher gate's decision); default frozen."""


def frozen_teacher_tags() -> dict[str, str]:
    """`TEACHER_PARAM_TAG` for every size — the pre-gate / all-admitted case."""
    return {size: TEACHER_PARAM_TAG for size in SIZES}


def _teacher_admitted(size: str, teacher_tags: TeacherTags | None) -> bool:
    """False only for a voided size (absent from an explicit `teacher_tags`)."""
    return teacher_tags is None or size in teacher_tags


def _teacher_tag_for(pairs: Sequence[XPair], teacher_tags: TeacherTags | None) -> str:
    """The one teacher tag a single-teacher student YAML distills from.

    Raises:
        ValueError: `pairs` span several teacher sizes (a doc carries one tag).
        KeyError: The teacher size has no tag (voided size — do not emit it).
    """
    teachers = {p.teacher for p in pairs}
    if len(teachers) != 1:
        raise ValueError(f"student YAML mixes teacher sizes {sorted(teachers)}")
    tags = frozen_teacher_tags() if teacher_tags is None else teacher_tags
    (teacher,) = teachers
    if teacher not in tags:
        raise KeyError(f"no admitted teacher tag for size {teacher!r}")
    return str(tags[teacher])


def _doc_logs_student(
    teacher: str,
    student: str,
    lr: float,
    picks: Mapping[str, float],
    teacher_ftseeds: Sequence[int],
) -> bool:
    picked = float(picks[student])
    return any(
        student_logs_wandb(
            teacher=teacher,
            student=student,
            lr=lr,
            picked_lr=picked,
            ftseed=int(tfs),
            gseed=int(gseed),
        )
        for tfs in teacher_ftseeds
        for gseed in GSEEDS
    )


def _student_yaml(
    *,
    pairs: Sequence[XPair],
    size: str,
    lr: float,
    teacher_ftseeds: Sequence[int],
    purpose: str,
    queue: str,
    wandb: dict | bool,
    extra_tags: Sequence[str],
    teacher_tags: TeacherTags | None = None,
) -> dict:
    if not pairs:
        raise ValueError(f"student YAML {purpose!r} has no pairs")
    teacher_tag = _teacher_tag_for(pairs, teacher_tags)
    students = {p.student for p in pairs}
    if students != {size}:
        raise ValueError(
            f"student YAML {purpose!r} mixes student sizes {students} "
            f"(expected {{{size!r}}})"
        )
    doc = _train_yaml(
        animals=list(ANIMALS),
        pairs=[_pair_row(p) for p in pairs],
        recipe=_student_recipe(lr),
        beta=BETA,
        num_epochs=NUM_EPOCHS,
        teacher_ftseeds=[int(t) for t in teacher_ftseeds],
        gseeds=list(GSEEDS),
        purpose=purpose,
        wandb=wandb,
        spec=SPEC,
    )
    doc["model_size"] = size
    doc["run_set_model_size"] = MODEL_SIZE
    doc["teacher_param_tags"] = {**doc["teacher_param_tags"], "lora": teacher_tag}
    doc["train_overrides"] = _overrides(STUDENT_FOOTPRINT[size])
    doc["queue"] = queue
    doc["wandb_extra_tags"] = [str(t) for t in extra_tags]
    return doc


def _teacher_yaml(
    size: str,
    recipe: TeacherRecipe,
    ftseeds: Sequence[int],
    queue: str = Q_P1_TEACHERS,
) -> dict:
    token = BASE_TOKEN[size]
    logs = any(
        teacher_logs_wandb(token=token, tag=recipe.tag, ftseed=int(tfs))
        for tfs in ftseeds
    )
    doc: dict = {
        "animals": list(ANIMALS),
        "n_animals": len(ANIMALS),
        "base_models": token,
        "model_size": size,
        "run_set_model_size": MODEL_SIZE,
        "ft_seeds": [int(t) for t in ftseeds],
        "grid": {
            "n_epochs": [recipe.epochs],
            "learning_rate": [recipe.lr],
            "lora": [{"r": recipe.lora_r, "alpha": recipe.lora_alpha}],
        },
        "train_overrides": _overrides(TEACHER_FOOTPRINT[size]),
        "run_set_id": SPEC.run_set_id,
        "purpose": (
            f"P1 xsize SDF teacher {size} {recipe.tag} ({len(ftseeds)} ftseeds)"
        ),
        "owner": OWNER,
        "queue": queue,
    }
    if logs:
        doc["wandb"] = {
            "n_epochs": [recipe.epochs],
            "ft_seeds": [int(FTSEEDS[0])],
            "learning_rate": [recipe.lr],
            "lora_r": [recipe.lora_r],
        }
    return doc


# ---- P1 teachers -------------------------------------------------------------


def teacher_configs(
    animals: Sequence[str] = ANIMALS,
) -> dict[str, dict]:
    """One SDF teacher YAML per (size, tag). 7 files, 13 leaves.

    Frozen tag for every size × `FTSEEDS`. Fallback tags only for
    `TEACHER_FALLBACK_SIZES`, ftseed 0.
    """
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize teachers are bear-only; got {tuple(animals)}")
    out: dict[str, dict] = {}
    for size in SIZES:
        recipe = TEACHER_RECIPES[TEACHER_PARAM_TAG]
        out[f"{P1_PREFIX}{size}_{recipe.tag}.yaml"] = _teacher_yaml(
            size, recipe, FTSEEDS
        )
        if size not in TEACHER_FALLBACK_SIZES:
            continue
        for tag in TEACHER_FALLBACK_TAGS:
            fb = TEACHER_RECIPES[tag]
            out[f"{P1_PREFIX}{size}_{fb.tag}.yaml"] = _teacher_yaml(
                size, fb, (FTSEEDS[0],)
            )
    return out


def fallback_teacher_configs(size: str, tag: str, *, queue: str) -> dict[str, dict]:
    """Completion YAMLs: remaining ftseeds (tfs 1, 2) at `tag` for one size.

    Distinct filename from the tfs0 fallback screen YAML so a rewrite cannot
    clobber the frozen P1 job set.
    """
    if size not in TEACHER_FALLBACK_SIZES:
        raise ValueError(
            f"fallback completion is not defined for size {size!r}; "
            f"allowed {TEACHER_FALLBACK_SIZES}"
        )
    if tag not in TEACHER_FALLBACK_TAGS:
        raise ValueError(
            f"fallback completion expects a tag in {TEACHER_FALLBACK_TAGS}; got {tag!r}"
        )
    recipe = TEACHER_RECIPES[tag]
    remaining = tuple(int(t) for t in FTSEEDS if int(t) != int(FTSEEDS[0]))
    fname = f"{P1_PREFIX}{size}_{tag}_tfs12.yaml"
    return {fname: _teacher_yaml(size, recipe, remaining, queue=queue)}


def teacher_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid teacher verify YAML per teacher train YAML."""
    rel = GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    return {
        verify_name(fname): {
            **_verify_yaml(f"{rel}/{fname}", None, spec=SPEC),
            "queue": Q_P1_TEACHER_EVALS,
        }
        for fname in sorted(train_configs)
        if fname.startswith(P1_PREFIX)
    }


# ---- P2 ladder ---------------------------------------------------------------


def ladder_student_configs(
    animals: Sequence[str] = ANIMALS,
    *,
    teacher_tags: TeacherTags | None = None,
) -> dict[str, dict]:
    """One ladder YAML per (student size, lr): the anchor pair, tfs0, `GSEEDS`.

    ``teacher_tags`` maps admitted teacher sizes to their gate tag. A voided
    size (absent) still runs its anchor ladder on the frozen, non-admitted
    teacher so `lr*(S)` exists for the `·→S` receptivity cells; `pick_lr`
    degrades to `LR_DEFAULT` there through the tie band.
    """
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize students are bear-only; got {tuple(animals)}")
    tags = {**frozen_teacher_tags(), **(teacher_tags or {})}
    out: dict[str, dict] = {}
    for size in SIZES:
        pair = XPair(teacher=size, student=size)
        for lr in LR_LADDER:
            slug = lr_slug(lr)
            out[f"{P2_PREFIX}{size}_lr{slug}.yaml"] = _student_yaml(
                pairs=(pair,),
                size=size,
                lr=lr,
                teacher_ftseeds=(FTSEEDS[0],),
                purpose=(
                    f"P2 xsize LR ladder {size}→{size} lr={slug} "
                    f"({student_gkd_tag(lr)})"
                ),
                queue=Q_P2_LADDER,
                wandb=False,
                extra_tags=("xsize", "bear", f"S{size}", "p2"),
                teacher_tags=tags,
            )
    return out


def ladder_control_configs(
    animals: Sequence[str] = ANIMALS,
) -> dict[str, dict]:
    """Empty: P2 sb controls reuse the ladder YAML with ``--controls-only``."""
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize students are bear-only; got {tuple(animals)}")
    return {}


# ---- P3 headline -------------------------------------------------------------


def headline_student_configs(
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
    teacher_tags: TeacherTags | None = None,
) -> dict[str, dict]:
    """P3 treatment YAMLs for one student size at ``picks[size]``.

    One file per ordered pair. The anchor omits tfs0 (already a P2 leaf).
    Cross pairs use all `FTSEEDS`.
    """
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize students are bear-only; got {tuple(animals)}")
    lr = _require_pick(picks, size)
    slug = lr_slug(lr)
    out: dict[str, dict] = {}
    for pair in all_pairs():
        if pair.student != size or not _teacher_admitted(pair.teacher, teacher_tags):
            continue
        if pair.arm == "anchor":
            fname = f"{P3_PREFIX}{size}_anchor.yaml"
            tfs = tuple(t for t in FTSEEDS if t != FTSEEDS[0])
            tags = ("xsize", "bear", f"S{size}", "p3", "anchor")
        else:
            fname = f"{P3_PREFIX}{size}_t{pair.teacher}.yaml"
            tfs = FTSEEDS
            tags = ("xsize", "bear", f"S{size}", f"T{pair.teacher}", "p3")
        wandb: dict | bool = (
            _wandb_student_block(lr)
            if _doc_logs_student(pair.teacher, pair.student, lr, picks, tfs)
            else False
        )
        out[fname] = _student_yaml(
            pairs=(pair,),
            size=size,
            lr=lr,
            teacher_ftseeds=tfs,
            purpose=(
                f"P3 xsize {pair.teacher}→{pair.student} lr={slug} "
                f"({student_gkd_tag(lr)})"
            ),
            queue=q_p3_students(size),
            wandb=wandb,
            extra_tags=tags,
            teacher_tags=teacher_tags,
        )
    return out


def headline_control_configs(
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
    teacher_tags: TeacherTags | None = None,
) -> dict[str, dict]:
    """P3 teacher-base control YAMLs: one file per cross pair into `size`."""
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize students are bear-only; got {tuple(animals)}")
    lr = _require_pick(picks, size)
    slug = lr_slug(lr)
    out: dict[str, dict] = {}
    for pair in all_pairs():
        if pair.student != size or pair.arm != "cross":
            continue
        if not _teacher_admitted(pair.teacher, teacher_tags):
            continue
        fname = f"{P3CTRL_PREFIX}{size}_t{pair.teacher}.yaml"
        out[fname] = _student_yaml(
            pairs=(pair,),
            size=size,
            lr=lr,
            teacher_ftseeds=(FTSEEDS[0],),
            purpose=(
                f"P3 xsize xbase {pair.teacher}_base→{pair.student} "
                f"lr={slug} ({student_gkd_tag(lr)})"
            ),
            queue=q_p3_controls(size),
            wandb=False,
            extra_tags=("xsize", "bear", f"S{size}", f"T{pair.teacher}", "tb"),
            teacher_tags=teacher_tags,
        )
    return out


# ---- P3b LR robustness -------------------------------------------------------


def lr_robustness_student_configs(
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
    teacher_tags: TeacherTags | None = None,
) -> dict[str, dict]:
    """P3b one-step treatments into `size` at the two non-picked LRs, tfs0."""
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize students are bear-only; got {tuple(animals)}")
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b is not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    picked = _require_pick(picks, size)
    out: dict[str, dict] = {}
    for pair in all_pairs():
        if pair.student != size or pair.shape_steps != 1:
            continue
        if not _teacher_admitted(pair.teacher, teacher_tags):
            continue
        for lr in LR_LADDER:
            if lr == picked:
                continue
            slug = lr_slug(lr)
            fname = f"{P3B_PREFIX}{size}_t{pair.teacher}_lr{slug}.yaml"
            out[fname] = _student_yaml(
                pairs=(pair,),
                size=size,
                lr=lr,
                teacher_ftseeds=(FTSEEDS[0],),
                purpose=(
                    f"P3b xsize {pair.teacher}→{pair.student} lr={slug} "
                    f"({student_gkd_tag(lr)})"
                ),
                queue=q_p3b(size),
                wandb=False,
                extra_tags=("xsize", "bear", f"S{size}", f"T{pair.teacher}", "p3b"),
                teacher_tags=teacher_tags,
            )
    return out


def lr_robustness_control_configs(
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
    teacher_tags: TeacherTags | None = None,
) -> dict[str, dict]:
    """P3b teacher-base control YAMLs matching `lr_robustness_student_configs`."""
    if tuple(animals) != ANIMALS:
        raise ValueError(f"xsize students are bear-only; got {tuple(animals)}")
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b is not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    picked = _require_pick(picks, size)
    out: dict[str, dict] = {}
    for pair in all_pairs():
        if pair.student != size or pair.shape_steps != 1:
            continue
        if not _teacher_admitted(pair.teacher, teacher_tags):
            continue
        for lr in LR_LADDER:
            if lr == picked:
                continue
            slug = lr_slug(lr)
            fname = f"{P3BCTRL_PREFIX}{size}_t{pair.teacher}_lr{slug}.yaml"
            out[fname] = _student_yaml(
                pairs=(pair,),
                size=size,
                lr=lr,
                teacher_ftseeds=(FTSEEDS[0],),
                purpose=(
                    f"P3b xsize xbase {pair.teacher}_base→{pair.student} "
                    f"lr={slug} ({student_gkd_tag(lr)})"
                ),
                queue=q_p3b(size),
                wandb=False,
                extra_tags=("xsize", "bear", f"S{size}", f"T{pair.teacher}", "tb"),
                teacher_tags=teacher_tags,
            )
    return out


# ---- verify + convenience ----------------------------------------------------


def _student_verify_queue(fname: str, train_doc: Mapping) -> str:
    size = str(train_doc["model_size"])
    for prefix, queue in _STUDENT_VERIFY_QUEUES:
        if fname.startswith(prefix):
            return queue(size) if callable(queue) else str(queue)
    raise ValueError(f"unclassified xsize student train config: {fname!r}")


def student_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per student/control train YAML."""
    rel = GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    out: dict[str, dict] = {}
    for fname in sorted(train_configs):
        if fname.startswith(P1_PREFIX):
            continue
        doc = train_configs[fname]
        verify = {
            **_verify_yaml(f"{rel}/{fname}", None, spec=SPEC),
            "queue": _student_verify_queue(fname, doc),
        }
        if verify["eval"]["prompt_sets"] != list(EVAL_PROMPT_SETS):
            raise ValueError(
                f"verify prompt_sets drifted from EVAL_PROMPT_SETS: "
                f"{verify['eval']['prompt_sets']}"
            )
        out[verify_name(fname)] = verify
    return out


def all_train_configs(
    picks: Mapping[str, float] | None = None,
    *,
    teacher_tags: TeacherTags | None = None,
) -> dict[str, dict]:
    """Every generated train YAML. ``picks=None`` is P1 + P2 only.

    ``teacher_tags`` (admitted teacher size → gate tag) defaults to the frozen
    tag everywhere. A size absent from it is voided as a *teacher*: it still
    runs its anchor ladder and every `·→S` cell as a student, but teaches no
    P3/P3b cell (anchor tfs1/2 included).
    """
    out: dict[str, dict] = {}
    out.update(teacher_configs())
    out.update(ladder_student_configs(teacher_tags=teacher_tags))
    out.update(ladder_control_configs())
    if picks is None:
        return out
    for size in SIZES:
        if size not in picks:
            continue
        out.update(
            headline_student_configs(picks, size=size, teacher_tags=teacher_tags)
        )
        out.update(
            headline_control_configs(picks, size=size, teacher_tags=teacher_tags)
        )
    for size in LR_ROBUSTNESS_STUDENTS:
        if size not in picks:
            continue
        out.update(
            lr_robustness_student_configs(picks, size=size, teacher_tags=teacher_tags)
        )
        out.update(
            lr_robustness_control_configs(picks, size=size, teacher_tags=teacher_tags)
        )
    return out


def all_verify_configs(
    train_configs: Mapping[str, dict] | None = None,
    picks: Mapping[str, float] | None = None,
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """Verify YAMLs for `train_configs` (or `all_train_configs(picks)`)."""
    docs = all_train_configs(picks) if train_configs is None else dict(train_configs)
    teachers = {k: v for k, v in docs.items() if k.startswith(P1_PREFIX)}
    students = {k: v for k, v in docs.items() if not k.startswith(P1_PREFIX)}
    return {
        **teacher_verify_configs(teachers, dir_rel=dir_rel),
        **student_verify_configs(students, dir_rel=dir_rel),
    }


def size_of_student_doc(doc: Mapping) -> str:
    """Student size recorded on a generated student YAML."""
    size = str(doc["model_size"])
    pairs = doc.get("teacher_student_base_pairs") or []
    for row in pairs:
        if size_of_token(str(row[0])) != size:
            raise ValueError(
                f"model_size {size!r} disagrees with pair student {row[0]!r}"
            )
    return size
