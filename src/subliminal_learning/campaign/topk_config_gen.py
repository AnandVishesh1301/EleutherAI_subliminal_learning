"""Sweep-YAML builders for the top-K campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``; writing is the driver's job.
Student bodies come from `campaign.config_gen._train_yaml` / `_verify_yaml`
with the rung's ``top_k`` / ``top_k_residual`` added as top-level keys (one
value per YAML, the sweep's contract).

Per block, rung and student size: one treatment YAML per cross pair, one
anchor YAML (also run with ``--controls-only`` for the student-base control),
and one xbase YAML per cross pair (``--xbase-controls-only``). Null teachers
(`NULL_ANIMAL`) are the only teacher YAMLs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.topk_pairs import Pair, size_of_token
from subliminal_learning.campaign.topk_plan import (
    ANCHOR_FTSEEDS,
    ANIMALS,
    BASE_TOKEN,
    BETA,
    BLOCKS,
    EVAL_PROMPT_SETS,
    GENERATED_CONFIG_DIR,
    GSEEDS,
    LMBDA,
    MODEL_SIZE,
    NULL_ANIMAL,
    NUM_EPOCHS,
    OWNER,
    Q_NULL_TEACHER_EVALS,
    Q_NULL_TEACHERS,
    SAVE_TOTAL_LIMIT,
    SPEC,
    STUDENT_FOOTPRINT,
    STUDENT_LORA,
    STUDENT_LR,
    STUDENT_MODE,
    STUDENT_SIZES,
    TEACHER_FOOTPRINT,
    TEACHER_FTSEEDS,
    TEACHER_PARAM_TAG,
    TEACHER_RECIPE,
    Block,
    Footprint,
    Rung,
    block,
    q_controls,
    q_evals,
    q_students,
    student_logs_wandb,
)

NULL_PREFIX = "topk_null_teacher_"
STUDENTS_CLASS = "students"
ANCHOR_CLASS = "anchor"
XBASE_CLASS = "xbase"
VERIFY_INFIX = "verify_"

_CLASS_ORDER: tuple[str, ...] = (STUDENTS_CLASS, ANCHOR_CLASS, XBASE_CLASS)


def lr_slug(lr: float) -> str:
    """Compact path token matching `gkd.layout` LR formatting (``4e-4``)."""
    mantissa, exp = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def student_fname(
    b: Block,
    rung: Rung,
    cls: str,
    size: str,
    teacher: str | None = None,
) -> str:
    """``topk_{block}_{rung}_{class}_{size}[_t{teacher}].yaml``."""
    if cls not in _CLASS_ORDER:
        raise ValueError(f"unknown student YAML class {cls!r}")
    tail = f"_t{teacher}" if teacher else ""
    return f"topk_{b.name}_{rung.slug}_{cls}_{size}{tail}.yaml"


def parse_student_fname(fname: str) -> tuple[str, str, str, str, str | None]:
    """``(block, rung slug, class, size, teacher)`` of a student train YAML name.

    Raises:
        ValueError: Not a generated top-K student train config filename.
    """
    stem = fname.removesuffix(".yaml")
    parts = stem.split("_")
    if len(parts) < 5 or parts[0] != "topk" or parts[1] not in {b.name for b in BLOCKS}:
        raise ValueError(f"not a generated topk student config filename: {fname!r}")
    block_name, rung_slug, cls, size = parts[1], parts[2], parts[3], parts[4]
    if cls not in _CLASS_ORDER or size not in STUDENT_SIZES:
        raise ValueError(f"not a generated topk student config filename: {fname!r}")
    teacher = parts[5][1:] if len(parts) > 5 and parts[5].startswith("t") else None
    return block_name, rung_slug, cls, size, teacher


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML."""
    stem = train_filename.removesuffix(".yaml")
    if stem.startswith(NULL_PREFIX):
        return f"{NULL_PREFIX}{VERIFY_INFIX}{stem[len(NULL_PREFIX) :]}.yaml"
    if stem.startswith("topk_"):
        return f"topk_{VERIFY_INFIX}{stem[len('topk_') :]}.yaml"
    raise ValueError(f"not a generated topk train config filename: {train_filename!r}")


def is_verify_name(fname: str) -> bool:
    """Whether `fname` is a generated verify YAML."""
    return fname.startswith(f"topk_{VERIFY_INFIX}") or fname.startswith(
        f"{NULL_PREFIX}{VERIFY_INFIX}"
    )


def train_name_of_verify(verify_fname: str) -> str:
    """Inverse of `verify_name`."""
    stem = verify_fname.removesuffix(".yaml")
    null_prefix = f"{NULL_PREFIX}{VERIFY_INFIX}"
    if stem.startswith(null_prefix):
        return f"{NULL_PREFIX}{stem[len(null_prefix) :]}.yaml"
    prefix = f"topk_{VERIFY_INFIX}"
    if stem.startswith(prefix):
        return f"topk_{stem[len(prefix) :]}.yaml"
    raise ValueError(f"not a generated topk verify filename: {verify_fname!r}")


def _student_recipe() -> Recipe:
    return Recipe(
        lora_r=STUDENT_LORA.r, lora_alpha=STUDENT_LORA.lora_alpha, lr=STUDENT_LR
    )


def _overrides(fp: Footprint) -> dict:
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


def _wandb_student_block() -> dict:
    return {
        "beta": [BETA],
        "lmbda": [LMBDA],
        "student_modes": [STUDENT_MODE],
        "learning_rate": [STUDENT_LR],
        "num_epochs": [NUM_EPOCHS],
    }


def _doc_logs(pair: Pair, rung: Rung, ftseeds: Sequence[int]) -> bool:
    return any(
        student_logs_wandb(
            teacher=pair.teacher,
            student=pair.student,
            rung=rung,
            ftseed=int(tfs),
            gseed=int(gseed),
        )
        for tfs in ftseeds
        for gseed in GSEEDS
    )


def _student_yaml(
    *,
    b: Block,
    rung: Rung,
    pair: Pair,
    teacher_ftseeds: Sequence[int],
    purpose: str,
    queue: str,
    extra_tags: Sequence[str],
    may_log: bool = True,
) -> dict:
    size = pair.student
    if size not in STUDENT_SIZES:
        raise ValueError(f"student size {size!r} not in {STUDENT_SIZES}")
    wandb: dict | bool = (
        _wandb_student_block()
        if may_log and _doc_logs(pair, rung, teacher_ftseeds)
        else False
    )
    doc = _train_yaml(
        animals=list(ANIMALS),
        pairs=[[pair.student_token, pair.teacher_token]],
        recipe=_student_recipe(),
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
    doc["teacher_param_tags"] = {**doc["teacher_param_tags"], "lora": TEACHER_PARAM_TAG}
    doc["top_k"] = rung.k
    doc["top_k_residual"] = bool(rung.ghost)
    doc["train_overrides"] = _overrides(STUDENT_FOOTPRINT[size])
    doc["queue"] = queue
    doc["wandb_extra_tags"] = [str(t) for t in extra_tags]
    doc["topk_block"] = b.name
    doc["topk_rung"] = rung.slug
    return doc


def _rung_tags(rung: Rung) -> tuple[str, ...]:
    tags = ["topk", rung.slug if rung.k is not None else "kfull"]
    if rung.ghost:
        tags.append("ghost")
    return tuple(tags)


def block_student_configs(block_name: str) -> dict[str, dict]:
    """Treatment, anchor and xbase YAMLs of one block (every rung, every size)."""
    b = block(block_name)
    out: dict[str, dict] = {}
    for rung in b.rungs:
        for teacher, student in b.pairs:
            pair = Pair(teacher=teacher, student=student)
            tags = (*_rung_tags(rung), "bear", f"S{student}", f"T{teacher}", b.name)
            if pair.arm == "anchor":
                out[student_fname(b, rung, ANCHOR_CLASS, student)] = _student_yaml(
                    b=b,
                    rung=rung,
                    pair=pair,
                    teacher_ftseeds=ANCHOR_FTSEEDS,
                    purpose=(
                        f"topk {b.name} anchor {student}→{student} {rung.slug} "
                        f"({rung.gkd_tag})"
                    ),
                    queue=q_students(b.name, student),
                    extra_tags=(*tags, "anchor"),
                )
                continue
            out[student_fname(b, rung, STUDENTS_CLASS, student, teacher)] = (
                _student_yaml(
                    b=b,
                    rung=rung,
                    pair=pair,
                    teacher_ftseeds=b.treatment_ftseeds,
                    purpose=(
                        f"topk {b.name} {teacher}→{student} {rung.slug} "
                        f"({rung.gkd_tag})"
                    ),
                    queue=q_students(b.name, student),
                    extra_tags=(*tags, "cross"),
                )
            )
            out[student_fname(b, rung, XBASE_CLASS, student, teacher)] = _student_yaml(
                b=b,
                rung=rung,
                pair=pair,
                teacher_ftseeds=ANCHOR_FTSEEDS,
                purpose=(
                    f"topk {b.name} xbase {teacher}_base→{student} {rung.slug} "
                    f"({rung.gkd_tag})"
                ),
                queue=q_controls(b.name, student),
                extra_tags=(*tags, "tb"),
                may_log=False,
            )
    return out


def _teacher_yaml(size: str, queue: str) -> dict:
    recipe = TEACHER_RECIPE
    return {
        "animals": [NULL_ANIMAL],
        "n_animals": 1,
        "base_models": BASE_TOKEN[size],
        "model_size": size,
        "run_set_model_size": MODEL_SIZE,
        "ft_seeds": [int(t) for t in TEACHER_FTSEEDS],
        "grid": {
            "n_epochs": [recipe.epochs],
            "learning_rate": [recipe.lr],
            "lora": [{"r": recipe.lora_r, "alpha": recipe.lora_alpha}],
        },
        "train_overrides": _overrides(TEACHER_FOOTPRINT[size]),
        "run_set_id": SPEC.run_set_id,
        "purpose": (
            f"topk null SDF teacher {NULL_ANIMAL} {size} {recipe.tag} "
            f"({len(TEACHER_FTSEEDS)} ftseeds; EAS teacher-side null)"
        ),
        "owner": OWNER,
        "queue": queue,
    }


def null_teacher_configs() -> dict[str, dict]:
    """One `NULL_ANIMAL` SDF teacher YAML per student size (3 ftseeds each)."""
    return {
        f"{NULL_PREFIX}{size}_{TEACHER_RECIPE.tag}.yaml": _teacher_yaml(
            size, Q_NULL_TEACHERS
        )
        for size in STUDENT_SIZES
    }


def null_teacher_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid teacher verify YAML per null-teacher train YAML."""
    rel = GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    return {
        verify_name(fname): {
            **_verify_yaml(f"{rel}/{fname}", None, spec=SPEC),
            "queue": Q_NULL_TEACHER_EVALS,
        }
        for fname in sorted(train_configs)
        if fname.startswith(NULL_PREFIX)
    }


def student_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per student/control train YAML."""
    rel = GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    out: dict[str, dict] = {}
    for fname in sorted(train_configs):
        if fname.startswith(NULL_PREFIX) or is_verify_name(fname):
            continue
        block_name, _, _, size, _ = parse_student_fname(fname)
        verify = {
            **_verify_yaml(f"{rel}/{fname}", None, spec=SPEC),
            "queue": q_evals(block_name, size),
        }
        if verify["eval"]["prompt_sets"] != list(EVAL_PROMPT_SETS):
            raise ValueError(
                f"verify prompt_sets drifted from EVAL_PROMPT_SETS: "
                f"{verify['eval']['prompt_sets']}"
            )
        out[verify_name(fname)] = verify
    return out


def all_train_configs() -> dict[str, dict]:
    """Every generated train YAML: null teachers then every block."""
    out: dict[str, dict] = {}
    out.update(null_teacher_configs())
    for b in BLOCKS:
        out.update(block_student_configs(b.name))
    return out


def all_verify_configs(
    train_configs: Mapping[str, dict] | None = None,
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """Verify YAMLs for `train_configs` (or `all_train_configs()`)."""
    docs = all_train_configs() if train_configs is None else dict(train_configs)
    teachers = {k: v for k, v in docs.items() if k.startswith(NULL_PREFIX)}
    students = {k: v for k, v in docs.items() if not k.startswith(NULL_PREFIX)}
    return {
        **null_teacher_verify_configs(teachers, dir_rel=dir_rel),
        **student_verify_configs(students, dir_rel=dir_rel),
    }


def configs_of_block(
    train_configs: Mapping[str, dict], block_name: str, *, size: str | None = None
) -> dict[str, dict]:
    """Student train YAMLs of one block (optionally one student size)."""
    out: dict[str, dict] = {}
    for fname, doc in train_configs.items():
        if fname.startswith(NULL_PREFIX) or is_verify_name(fname):
            continue
        b, _, _, s, _ = parse_student_fname(fname)
        if b == block_name and (size is None or s == size):
            out[fname] = doc
    return out


def size_of_student_doc(doc: Mapping) -> str:
    """Student size recorded on a generated student YAML."""
    size = str(doc["model_size"])
    for row in doc.get("teacher_student_base_pairs") or []:
        if size_of_token(str(row[0])) != size:
            raise ValueError(
                f"model_size {size!r} disagrees with pair student {row[0]!r}"
            )
    return size
