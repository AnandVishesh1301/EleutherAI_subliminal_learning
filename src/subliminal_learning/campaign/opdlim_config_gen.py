"""Sweep-YAML builders for the OPD-limits campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``; writing is the driver's job.
One train YAML per ``(block, objective, pair, class)`` so every YAML has exactly
one value per swept axis (the sweep contract) and its own seed lists:

- ``students``: treatments (``--no-control``), design tfs × gseeds.
- ``sb``: student-base controls of anchor pairs (``--controls-only``).
- ``tb``: teacher-base controls of cross pairs (``--xbase-controls-only``).

Bodies come from `campaign.config_gen._train_yaml` / `_verify_yaml`; the
objective's top-level keys (`Objective.sweep_keys`) are parsed by
`gkd.config.parse_sweep_gkd_axes`. Verify YAMLs carry ``eval.calibration``;
bench verify YAMLs add LM-Eval for every cell of their train YAML.
"""

from __future__ import annotations

from collections.abc import Mapping

from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.opdlim_pairs import control_kind, KIND_SB
from subliminal_learning.campaign.opdlim_plan import (
    ANIMALS,
    BETA,
    CALIBRATION_IN_CORE_EVALS,
    EVAL_PROMPT_SETS,
    MODEL_SIZE,
    NUM_EPOCHS,
    PAIR_BY_NAME,
    SPECS,
    STUDENT_FOOTPRINT,
    Block,
    LeafDesign,
    Objective,
    PairDef,
    Tier,
    generated_config_dir,
    queue_name,
    student_logs_wandb,
)
from subliminal_learning.campaign.topk_config_gen import _overrides

STUDENTS_CLASS = "students"
SB_CLASS = "sb"
TB_CLASS = "tb"
TRAIN_CLASSES: tuple[str, ...] = (STUDENTS_CLASS, SB_CLASS, TB_CLASS)
VERIFY_KIND = "verify"
BENCH_KIND = "bench"


def train_fname(t: Tier, b: Block, obj: Objective, cls: str, pair: PairDef) -> str:
    """``opdlim{n}_{block}_{objective}_{class}_{pair}.yaml``."""
    if cls not in TRAIN_CLASSES:
        raise ValueError(f"unknown train class {cls!r}")
    return f"{t.token}_{b.name}_{obj.slug}_{cls}_{pair.name}.yaml"


def parse_train_fname(t: Tier, fname: str) -> tuple[str, str, str, str]:
    """``(block, objective slug, class, pair)`` of a generated train YAML name.

    Raises:
        ValueError: Not a generated train YAML of tier `t`.
    """
    parts = fname.removesuffix(".yaml").split("_")
    if len(parts) != 5 or parts[0] != t.token:
        raise ValueError(f"not a generated {t.token} train config: {fname!r}")
    block_name, slug, cls, pair = parts[1:]
    t.block(block_name)
    if cls not in TRAIN_CLASSES or pair not in PAIR_BY_NAME:
        raise ValueError(f"not a generated {t.token} train config: {fname!r}")
    return block_name, slug, cls, pair


def derived_fname(t: Tier, kind: str, train_name: str) -> str:
    """``opdlim{n}_{verify|bench}_{rest}`` for a train YAML name."""
    if kind not in (VERIFY_KIND, BENCH_KIND):
        raise ValueError(f"unknown derived kind {kind!r}")
    parse_train_fname(t, train_name)
    return f"{t.token}_{kind}_{train_name.removeprefix(t.token + '_')}"


def train_name_of(t: Tier, derived: str) -> tuple[str, str]:
    """``(kind, train YAML name)`` of a verify / bench YAML name."""
    for kind in (VERIFY_KIND, BENCH_KIND):
        prefix = f"{t.token}_{kind}_"
        if derived.startswith(prefix):
            train = f"{t.token}_{derived.removeprefix(prefix)}"
            parse_train_fname(t, train)
            return kind, train
    raise ValueError(f"not a generated {t.token} verify/bench config: {derived!r}")


def _objective(b: Block, slug: str) -> Objective:
    for obj in b.objectives:
        if obj.slug == slug:
            return obj
    raise ValueError(f"block {b.name} has no objective {slug!r}")


def _classes(pair: PairDef, design: LeafDesign) -> tuple[str, ...]:
    if design.reuse_controls:
        return (STUDENTS_CLASS,)
    control = SB_CLASS if control_kind(pair) == KIND_SB else TB_CLASS
    return (STUDENTS_CLASS, control)


def _student_doc(
    t: Tier,
    b: Block,
    obj: Objective,
    pair: PairDef,
    design: LeafDesign,
    cls: str,
) -> dict:
    spec = SPECS[t.n]
    if cls == STUDENTS_CLASS:
        tfs, gseeds = list(design.tfs), list(design.gseeds)
        logs = any(
            student_logs_wandb(ftseed=f, gseed=g, control=False)
            for f in tfs
            for g in gseeds
        )
    else:
        tfs, gseeds, logs = [0], list(design.control_gseeds), False
    queue_cls = "students" if cls == STUDENTS_CLASS else "controls"
    wandb: dict | bool = (
        {
            "beta": [BETA],
            "lmbda": [obj.lmbda],
            "student_modes": [b.hp.recipe.mode_token],
            "learning_rate": [b.hp.lr],
            "num_epochs": [NUM_EPOCHS],
        }
        if logs
        else False
    )
    doc = _train_yaml(
        animals=list(ANIMALS),
        pairs=[[pair.student, pair.teacher]],
        recipe=b.hp.recipe,
        beta=BETA,
        num_epochs=NUM_EPOCHS,
        teacher_ftseeds=tfs,
        gseeds=gseeds,
        purpose=(
            f"{t.token} {b.name} {obj.slug} {cls} {pair.name} "
            f"{pair.teacher}→{pair.student} ({b.gkd_tag(obj)})"
        ),
        wandb=wandb,
        spec=spec,
        lmbda_list=[obj.lmbda],
        vocab_mask=[obj.vmask_forms],
    )
    doc["model_size"] = pair.student_size
    doc["run_set_model_size"] = MODEL_SIZE
    doc["teacher_param_tags"] = {**doc["teacher_param_tags"], "lora": b.teacher_tag}
    doc.update(obj.sweep_keys())
    doc["train_overrides"] = _overrides(STUDENT_FOOTPRINT[pair.student_size])
    doc["queue"] = queue_name(t, b.name, queue_cls, pair.student_size)
    doc["wandb_extra_tags"] = [t.token, b.name, obj.slug, pair.name, cls]
    return doc


def block_train_configs(t: Tier, block_name: str) -> dict[str, dict]:
    """Every train YAML of one block."""
    b = t.block(block_name)
    out: dict[str, dict] = {}
    for obj in b.objectives:
        for pair, design in b.cells:
            for cls in _classes(pair, design):
                fname = train_fname(t, b, obj, cls, pair)
                out[fname] = _student_doc(t, b, obj, pair, design, cls)
    return out


def all_train_configs(t: Tier) -> dict[str, dict]:
    """Every generated train YAML of a tier."""
    out: dict[str, dict] = {}
    for b in t.blocks:
        out.update(block_train_configs(t, b.name))
    return out


def _bench_rule(doc: Mapping) -> dict:
    return {
        "num_epochs": [float(doc["num_epochs"])],
        "learning_rate": [float(lr) for lr in doc["student_lr"]["lora"]],
        "lora_r": [
            int(m.split("-r")[1].split("a")[0]) for m in doc["grid"]["student_modes"]
        ],
    }


def _derived_doc(t: Tier, train_name: str, doc: Mapping, kind: str) -> dict:
    rel = f"{generated_config_dir(t)}/{train_name}"
    bench = _bench_rule(doc) if kind == BENCH_KIND else None
    out = _verify_yaml(rel, bench, spec=SPECS[t.n])
    if out["eval"]["prompt_sets"] != list(EVAL_PROMPT_SETS):
        raise ValueError(f"verify prompt_sets drifted: {out['eval']['prompt_sets']}")
    out["eval"]["calibration"] = bool(CALIBRATION_IN_CORE_EVALS)
    block_name, _, _, pair = parse_train_fname(t, train_name)
    queue_cls = "bench" if kind == BENCH_KIND else "evals"
    out["queue"] = queue_name(t, block_name, queue_cls, PAIR_BY_NAME[pair].student_size)
    return out


def all_verify_configs(
    t: Tier, train_configs: Mapping[str, dict] | None = None
) -> dict[str, dict]:
    """Verify YAMLs for every train YAML, plus bench YAMLs for bench blocks."""
    docs = all_train_configs(t) if train_configs is None else dict(train_configs)
    out: dict[str, dict] = {}
    for fname in sorted(docs):
        block_name, _, _, _ = parse_train_fname(t, fname)
        out[derived_fname(t, VERIFY_KIND, fname)] = _derived_doc(
            t, fname, docs[fname], VERIFY_KIND
        )
        if t.block(block_name).bench:
            out[derived_fname(t, BENCH_KIND, fname)] = _derived_doc(
                t, fname, docs[fname], BENCH_KIND
            )
    return out


def configs_of_block(
    t: Tier, train_configs: Mapping[str, dict], block_name: str
) -> dict[str, dict]:
    """Train YAMLs of one block."""
    return {
        fname: doc
        for fname, doc in train_configs.items()
        if parse_train_fname(t, fname)[0] == block_name
    }


def objective_of(t: Tier, fname: str) -> Objective:
    """The `Objective` a generated train YAML trains."""
    block_name, slug, _, _ = parse_train_fname(t, fname)
    return _objective(t.block(block_name), slug)


def pair_of(t: Tier, fname: str) -> PairDef:
    """The `PairDef` of a generated train YAML."""
    return PAIR_BY_NAME[parse_train_fname(t, fname)[3]]
