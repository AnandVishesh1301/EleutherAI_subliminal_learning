"""Sweep-YAML builders for the optimizer x parameterization campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``; every dict is a full first-class
`sweep_distill_student_gkd` (train) or `sweep_evaluate_gkd_student` (verify) config, so a
dry-run of any generated file counts exactly the cells the driver's gate expects. Writing
is the driver's job (`campaign.driver_io.write_generated_configs`).

One YAML per arm, per phase. The arms differ in the two axes the sweep spells per
(mode-kind, optimizer) -- ``grid.student_optims`` and the ``student_lr['{kind}/{slug}']``
ladder -- so a single multi-arm YAML would be one grid whose LR axis means something
different in each row; per-arm files keep every generated grid rectangular, let a widened
Phase-0 ladder be regenerated for one arm alone, and make the shard flags obvious.

Sibling of `campaign.config_gen` (Stage A3/B/C of the 20260727 campaign): same
`EVAL_BLOCK` / benchmark tasks, different phases and axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.campaign.config_gen import BENCH_TASKS, EVAL_BLOCK
from subliminal_learning.campaign.optparam_selection import (
    ArmCalibration,
    ArmLrSelection,
)

P0_PREFIX = "gkd_op_p0_"
P1_PREFIX = "gkd_op_p1_"
P2H_PREFIX = "gkd_op_p2h_"
P2L_PREFIX = "gkd_op_p2lm_"
VERIFY_INFIX = "verify_"


def _fmt(x: float) -> str:
    """Path-safe compact float (1.0 -> '1', 0.5 -> '0.5')."""
    return f"{x:g}"


_PREFIXES = (P0_PREFIX, P1_PREFIX, P2H_PREFIX, P2L_PREFIX)


def _split(filename: str) -> tuple[str, str]:
    """``(phase prefix, arm key)`` of a generated train filename."""
    stem = filename.removesuffix(".yaml")
    for prefix in _PREFIXES:
        if stem.startswith(prefix):
            return prefix, stem[len(prefix) :]
    raise ValueError(f"not a generated optparam train config filename: {filename!r}")


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML."""
    prefix, arm = _split(train_filename)
    return f"{prefix}{VERIFY_INFIX}{arm}.yaml"


def arm_of(filename: str) -> str:
    """The `optparam_plan.Arm.key` a generated train or verify filename belongs to."""
    _, arm = _split(filename)
    return arm.removeprefix(VERIFY_INFIX)


def _train_yaml(
    *,
    arm,
    lrs: Sequence[float],
    animals: Sequence[str],
    inits: Sequence[str],
    teacher_ftseeds: Sequence[int],
    gseeds: Sequence[int],
    purpose: str,
    max_steps: int = -1,
    grad_probe: bool = False,
    wandb: dict | bool = False,
) -> dict:
    """One arm's train-sweep YAML over `lrs` x `animals` x `inits` x tfs x gseeds.

    Same-init pairings only (this campaign holds the cross-init axis fixed), so the pair
    list is the diagonal over `inits`. Both the plain ``{kind}`` LR key and the
    arm-specific ``{kind}/{slug}`` key carry the same ladder: the former is the fallback
    `student_variants` reads when a YAML predates the optimizer axis, and writing it keeps
    every generated file valid under either code path.
    """
    ladder = [float(x) for x in lrs]
    return {
        "animals": list(animals),
        "base_models": ",".join(inits),
        "teacher_student_base_pairs": [[i, i] for i in inits],
        "teacher_modes": [plan.TEACHER_MODE],
        "teacher_ftseeds": [int(t) for t in teacher_ftseeds],
        "gseeds": [int(g) for g in gseeds],
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [arm.mode],
            "student_optims": [arm.optim_slug],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": plan.TEACHER_PARAM_TAG_LORA,
            "full": plan.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {arm.kind: list(ladder), arm.ladder_key: list(ladder)},
        "num_epochs": plan.NUM_EPOCHS,
        "max_steps": int(max_steps),
        "grad_probe": bool(grad_probe),
        "prompt_pool_dir": plan.PROMPT_POOL_DIR,
        "run_set_id": plan.RUN_SET_ID,
        "teacher_run_set_id": plan.TEACHER_RUN_SET_ID,
        "purpose": purpose,
        "owner": "vishesh",
        "wandb": wandb,
    }


def _verify_yaml(train_rel: str, bench: dict | None) -> dict:
    doc: dict = {"inherit_grid": train_rel, "eval": dict(EVAL_BLOCK)}
    if bench is not None:
        doc["benchmarks"] = {
            "enabled": True,
            "tasks": list(BENCH_TASKS),
            "limit": None,
            **bench,
        }
    return doc


def _bench_rule(arm, lr: float) -> dict:
    """Pin LM-Eval to this arm's exact (dose, lr[, rank]) so the guardrail tracks it."""
    rule = {"num_epochs": [plan.NUM_EPOCHS], "learning_rate": [float(lr)]}
    if arm.kind == "full":
        return {"full": rule}
    r = int(arm.mode.removeprefix("lora-r").split("a")[0])
    return {"lora": {**rule, "lora_r": [r]}}


def _wandb_block(arm, lrs: Sequence[float]) -> dict:
    return {
        "beta": [plan.BETA],
        "lmbda": [plan.LMBDA],
        "student_modes": [arm.mode],
        "student_optims": [arm.optim_slug],
        "learning_rate": [float(x) for x in lrs],
        "num_epochs": [plan.NUM_EPOCHS],
    }


# ---- Phase 0 -- LR / loss calibration -------------------------------------------


def p0_train_configs(
    ladders: Mapping[str, Sequence[float]] = plan.PHASE0_LADDERS,
) -> dict[str, dict]:
    """One 200-step calibration YAML per arm (cat, one init, tfs0, gseed0, no controls).

    The gradient-concentration probe runs here and nowhere else: it is a property of the
    arm's gradient geometry, not of a particular LR's outcome, and 50 short cells already
    cover the 2x4 grid. W&B logs every Phase-0 cell -- this is the ladder dashboard, and
    50 short runs is the cheapest place in the campaign to watch a divergence live.
    """
    return {
        f"{P0_PREFIX}{arm.key}.yaml": _train_yaml(
            arm=arm,
            lrs=ladders[arm.ladder_key],
            animals=[plan.PHASE0_ANIMAL],
            inits=[plan.SCREEN_INIT],
            teacher_ftseeds=[plan.PHASE0_TFS],
            gseeds=[plan.PHASE0_GSEED],
            max_steps=plan.PHASE0_MAX_STEPS,
            grad_probe=True,
            purpose=(
                f"Phase 0 LR/loss calibration {arm.key} "
                f"({len(ladders[arm.ladder_key])} rungs, {plan.PHASE0_MAX_STEPS} steps)"
            ),
            wandb=_wandb_block(arm, ladders[arm.ladder_key]),
        )
        for arm in plan.ARMS
    }


# ---- Phase 1 -- ladder + control floor ------------------------------------------


def p1_train_configs(
    calibrations: Mapping[str, ArmCalibration],
) -> dict[str, dict]:
    """One full-dose ladder YAML per arm over the Phase-0 rungs x 2 animals x 3 gseeds."""
    return {
        f"{P1_PREFIX}{arm.key}.yaml": _train_yaml(
            arm=arm,
            lrs=calibrations[arm.key].ladder,
            animals=plan.ANIMALS,
            inits=[plan.SCREEN_INIT],
            teacher_ftseeds=[plan.PHASE0_TFS],
            gseeds=plan.SCREEN_GSEEDS,
            purpose=(
                f"Phase 1 LR ladder {arm.key} "
                f"({[f'{x:g}' for x in calibrations[arm.key].ladder]})"
            ),
        )
        for arm in plan.ARMS
    }


def p1_verify_configs(
    train_configs: Mapping[str, dict], dir_rel: str = plan.GENERATED_CONFIG_DIR
) -> dict[str, dict]:
    """One inherit_grid verify YAML per Phase-1 train YAML (no LM-Eval benchmarks).

    Phase 1 only has to rank LRs within an arm; the capability guardrail is a Phase-2
    claim and running it on 64 ladder configs would cost more than the ladder itself.
    """
    return {
        verify_name(fname): _verify_yaml(f"{dir_rel}/{fname}", None)
        for fname in sorted(train_configs)
    }


# ---- Phase 2 -- headline + loss-matched arms ------------------------------------


def p2_train_configs(
    selections: Mapping[str, ArmLrSelection],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """``(headline, loss_matched)`` train YAMLs at the selected and loss-matched LRs.

    Iterates only arms present in `selections` — a dropped arm (every Phase-1 rung
    quarantined) produces no YAML. Headline carries the full replication grid (3 inits x
    3 teacher ftseeds x 3 gseeds x 2 animals). The loss-matched arm exists only where
    Phase 0's `lr_lossmatch` differs from the selected LR -- when they coincide the
    headline cells already are the loss-matched cells -- and drops to one teacher ftseed:
    it answers Blank et al. Appendix L ("is the optimizer null just a loss gap?"), which
    needs a delta at a matched loss, not a second full replication.
    """
    headline: dict[str, dict] = {}
    lossmatch: dict[str, dict] = {}
    for arm_key, sel in sorted(selections.items()):
        arm = plan.ARMS_BY_KEY[arm_key]
        headline[f"{P2H_PREFIX}{arm.key}.yaml"] = _train_yaml(
            arm=arm,
            lrs=[sel.lr],
            animals=plan.ANIMALS,
            inits=plan.INITS,
            teacher_ftseeds=plan.TFS_HEADLINE,
            gseeds=plan.GSEEDS,
            purpose=f"Phase 2 headline {arm.key} @ lr {sel.lr:g}",
            wandb=_wandb_block(arm, [sel.lr]),
        )
        if sel.needs_lossmatch_arm:
            lossmatch[f"{P2L_PREFIX}{arm.key}.yaml"] = _train_yaml(
                arm=arm,
                lrs=[sel.lr_lossmatch],
                animals=plan.ANIMALS,
                inits=plan.INITS,
                teacher_ftseeds=plan.TFS_LOSSMATCH,
                gseeds=plan.GSEEDS,
                purpose=(
                    f"Phase 2 loss-matched {arm.key} @ lr {sel.lr_lossmatch:g} "
                    f"(selected {sel.lr:g})"
                ),
            )
    return headline, lossmatch


def p2_verify_configs(
    headline: Mapping[str, dict],
    lossmatch: Mapping[str, dict],
    selections: Mapping[str, ArmLrSelection],
    dir_rel: str = plan.GENERATED_CONFIG_DIR,
) -> dict[str, dict]:
    """Verify YAMLs for both Phase-2 tiers; LM-Eval runs on the headline tier only."""
    out: dict[str, dict] = {}
    for fname in sorted(headline):
        arm = plan.ARMS_BY_KEY[arm_of(fname)]
        out[verify_name(fname)] = _verify_yaml(
            f"{dir_rel}/{fname}", _bench_rule(arm, selections[arm.key].lr)
        )
    for fname in sorted(lossmatch):
        out[verify_name(fname)] = _verify_yaml(f"{dir_rel}/{fname}", None)
    return out
