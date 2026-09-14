"""Sweep-YAML builders for the strictscreen campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``; writing is the driver's job
(`driver_io.write_generated_configs`). Bodies come from
`campaign.config_gen._train_yaml` / `_verify_yaml`:

- arm A: one train YAML per (pairing, tfs), 4 animals × 3 gseeds, top-level
  ``rollout_screen: strict``;
- arm B: one ``[s, s]`` YAML per student (trained ``--controls-only``), 3 gseeds,
  ``rollout_screen: strict``;
- arm C: one YAML per pairing at tfs 0 / gseed 0 with no screen key (legacy).

Every verify YAML carries its train YAML's top-level screen key and no LM-Eval.
"""

from __future__ import annotations

from collections.abc import Mapping

from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.campaign.config_gen import (
    _train_yaml,
    _verify_yaml,
    _wandb_block,
)

TRAIN_PREFIX = f"{plan.CAMPAIGN}_"
VERIFY_PREFIX = f"{plan.CAMPAIGN}_verify_"


def train_fname(cell: plan.Cell) -> str:
    """Generated train YAML holding `cell`.

    ``strictscreen_a_{pair}_tfs{k}.yaml``, ``strictscreen_b_{student}.yaml`` or
    ``strictscreen_c_{pair}.yaml``.
    """
    if cell.arm == plan.ARM_A:
        return f"{TRAIN_PREFIX}a_{cell.pair}_tfs{cell.tfs}.yaml"
    return f"{TRAIN_PREFIX}{cell.arm}_{cell.pair}.yaml"


def verify_fname(train_name: str) -> str:
    """``strictscreen_verify_{rest}`` of a generated train YAML name.

    Raises:
        ValueError: `train_name` is not a generated train YAML.
    """
    if not train_name.startswith(TRAIN_PREFIX) or train_name.startswith(VERIFY_PREFIX):
        raise ValueError(f"not a generated strictscreen train config: {train_name!r}")
    return f"{VERIFY_PREFIX}{train_name.removeprefix(TRAIN_PREFIX)}"


def _doc(cells: tuple[plan.Cell, ...]) -> dict:
    first = cells[0]
    teacher = first.student if first.control else first.teacher
    logs = any(plan.logs_wandb(c) for c in cells)
    doc = _train_yaml(
        animals=list(plan.ANIMALS),
        pairs=[[first.student, teacher]],
        recipe=plan.RECIPE,
        beta=plan.BETA,
        num_epochs=plan.NUM_EPOCHS,
        teacher_ftseeds=sorted({c.tfs or 0 for c in cells}),
        gseeds=sorted({c.gseed for c in cells}),
        purpose=(
            f"{plan.CAMPAIGN} arm {first.arm} {first.pair}"
            f"{'' if first.tfs is None else f' tfs{first.tfs}'} ({first.gkd_tag})"
        ),
        wandb=(
            _wandb_block(plan.RECIPE, plan.BETA, plan.NUM_EPOCHS, plan.SPEC)
            if logs
            else False
        ),
        spec=plan.SPEC,
    )
    if first.screen == plan.SCREEN_STRICT:
        doc["rollout_screen"] = plan.SCREEN_STRICT
    return doc


def train_configs() -> dict[str, dict]:
    """Every generated train YAML (24 A + 6 B + 8 C)."""
    grouped: dict[str, list[plan.Cell]] = {}
    for cell in plan.all_cells():
        grouped.setdefault(train_fname(cell), []).append(cell)
    return {fname: _doc(tuple(cells)) for fname, cells in grouped.items()}


def verify_configs(train: Mapping[str, dict] | None = None) -> dict[str, dict]:
    """One verify YAML per train YAML, carrying the same top-level screen key.

    Raises:
        ValueError: The verify prompt sets drifted from `plan.EVAL_PROMPT_SETS`.
    """
    docs = train_configs() if train is None else dict(train)
    out: dict[str, dict] = {}
    for fname in sorted(docs):
        doc = _verify_yaml(f"{plan.GENERATED_CONFIG_DIR}/{fname}", None, spec=plan.SPEC)
        if doc["eval"]["prompt_sets"] != list(plan.EVAL_PROMPT_SETS):
            raise ValueError(
                f"verify prompt_sets drifted: {doc['eval']['prompt_sets']}"
            )
        if "rollout_screen" in docs[fname]:
            doc["rollout_screen"] = docs[fname]["rollout_screen"]
        out[verify_fname(fname)] = doc
    return out


def cell_doc(doc: Mapping, cell: plan.Cell) -> dict:
    """`doc` restricted to one cell, exactly as the job's CLI shard flags restrict it.

    Raises:
        ValueError: `cell` is not enumerated by `doc`.
    """
    pairs = [list(p) for p in doc["teacher_student_base_pairs"] if p[0] == cell.student]
    tfs = [0] if cell.tfs is None else [cell.tfs]
    if (
        not pairs
        or cell.animal not in doc["animals"]
        or cell.gseed not in doc["gseeds"]
        or tfs[0] not in doc["teacher_ftseeds"]
    ):
        raise ValueError(f"{cell.key} is not a cell of this YAML")
    return {
        **doc,
        "animals": [cell.animal],
        "teacher_student_base_pairs": pairs,
        "base_models": cell.student,
        "teacher_ftseeds": tfs,
        "gseeds": [cell.gseed],
    }
