"""Programmatic A3 / Stage B / Stage C sweep-YAML builders (pure dict builders).

Each builder returns ``{filename: yaml_dict}`` where every dict is a full
first-class `sweep_distill_student_gkd` (train) or `sweep_evaluate_gkd_student`
(verify) config — dry-runnable exactly like the committed static YAMLs. Writing
to disk is the driver's job (`campaign.driver_io.write_generated_configs`);
generated files live under `plan.GENERATED_CONFIG_DIR` (git-ignored: they are
derived from stage outputs, the committed source of truth is this module plus
the campaign state file).

A3 layout: one train YAML per (animal, beta, arm, dose) with the dose baked into
``num_epochs`` — a dry-run of any one file therefore counts exactly one dose's
cells, and the sibling verify YAML inherits the right epoch budget with no
override plumbing. Arms are ``r1``..``rN`` (the A1 candidate recipes, best-first,
same-init — a selection-dependent count, see `selection.select_top_recipes`) and
``cross`` (the best-1 recipe on the family_swap lineage). Only the ``r1``
(headline) arm trains controls and logs to W&B.
"""

from __future__ import annotations

from collections.abc import Sequence

from subliminal_learning.campaign import plan
from subliminal_learning.campaign.pairs import (
    GROUP_SAME,
    CornerPair,
    pairs_for,
    stage_c_corner_pairs,
)
from subliminal_learning.campaign.selection import FrozenRecipe, Recipe

A3_CROSS_ARM = "cross"
A3_HEADLINE_ARM = "r1"

EVAL_BLOCK: dict = {
    "prompt_sets": [
        "test",
        "favorite",
        "dislike",
        "validation",
        "consensus",
        "register",
    ],
    "indist_scoring": "prob",
    "n_samples": 100,
    "temperature": 1.0,
    "storytelling_max_new_tokens": 128,
    "sample_twin_max_new_tokens": 24,
    "include_base_control": True,
    "confidence": 0.95,
    "perplexity_max_per_register": 0,
    "forced_choice_candidates": [
        "cat",
        "dog",
        "bear",
        "lion",
        "tiger",
        "horse",
        "fox",
        "eagle",
        "pig",
        "rabbit",
    ],
}

BENCH_TASKS = ["lambada_openai", "arc_easy", "sciq"]


def _fmt(x: float) -> str:
    """Path-safe compact float (1.0 -> '1', 0.5 -> '0.5')."""
    return f"{x:g}"


def _train_yaml(
    *,
    animals: list[str],
    pairs: list[list[str]],
    recipe: Recipe,
    beta: float,
    num_epochs: float,
    teacher_ftseeds: list[int],
    gseeds: list[int],
    purpose: str,
    wandb: dict | bool,
    teacher_modes: list[str] | None = None,
    lmbda_list: list[float] | None = None,
    vocab_mask: list[bool] | None = None,
) -> dict:
    students = list(dict.fromkeys(p[0] for p in pairs))
    return {
        "animals": list(animals),
        "base_models": ",".join(students),
        "teacher_student_base_pairs": [list(p) for p in pairs],
        "teacher_modes": list(teacher_modes or ["lora"]),
        "teacher_ftseeds": list(teacher_ftseeds),
        "gseeds": list(gseeds),
        "grid": {
            "beta": [float(beta)],
            "lmbda": [float(x) for x in (lmbda_list or [plan.LMBDA])],
            "student_modes": [recipe.mode_token],
            "vocab_mask": list(vocab_mask or [False]),
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": plan.TEACHER_PARAM_TAG_LORA,
            "full": plan.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {"full": 2.0e-5, "lora": [recipe.lr]},
        "num_epochs": float(num_epochs),
        "prompt_pool_dir": "gkd_prompts/pile_prompts_v1",
        "run_set_id": plan.RUN_SET_ID,
        "teacher_run_set_id": plan.TEACHER_RUN_SET_ID,
        "purpose": purpose,
        "owner": "vishesh",
        "wandb": wandb,
    }


def _verify_yaml(train_path: str, bench_rule: dict | None) -> dict:
    doc: dict = {"inherit_grid": train_path, "eval": dict(EVAL_BLOCK)}
    if bench_rule is not None:
        doc["benchmarks"] = {
            "enabled": True,
            "tasks": list(BENCH_TASKS),
            "limit": None,
            "lora": bench_rule,
        }
    return doc


def _wandb_block(recipe: Recipe, beta: float, num_epochs: float) -> dict:
    return {
        "beta": [float(beta)],
        "lmbda": [plan.LMBDA],
        "student_modes": [recipe.mode_token],
        "learning_rate": [recipe.lr],
        "num_epochs": [float(num_epochs)],
    }


# ---- Stage A3 -------------------------------------------------------------------


def a3_name(animal: str, beta: float, arm: str, dose: float) -> str:
    """Generated A3 train-YAML stem, e.g. ``gkd_a3_cat_b1_r1_e0.5``."""
    return f"gkd_a3_{animal}_b{_fmt(beta)}_{arm}_e{_fmt(dose)}"


def a3_arm_names(n: int) -> list[str]:
    """Same-init arm labels ``r1``..``rN`` (``r1`` is the headline arm)."""
    return [f"r{i + 1}" for i in range(n)]


def a3_train_configs(
    candidates: dict[tuple[str, float], Sequence[Recipe]],
) -> dict[str, dict]:
    """A3 dose-ladder train YAMLs from the A1 candidate selection.

    Args:
        candidates: Per (animal, beta), the candidate recipes best-first
            (``selection.RecipeSelection.recipes``). The count is
            selection-dependent — the union of the two top-K rankings plus any
            lr-edge extension — and is capped at `plan.A3_MAX_SAME_ARMS`.

    Returns:
        ``{filename: yaml_dict}`` — per (animal, beta): one YAML per same-init
        arm (``r1``..``rN``) and dose, plus one ``cross`` arm (rank-1 recipe on
        the family_swap lineage) per dose. Every file holds 3 cells (one gseed
        each).

    Raises:
        ValueError: A cell has no candidates, or more than
            `plan.A3_MAX_SAME_ARMS`.
    """
    out: dict[str, dict] = {}
    for (animal, beta), recipes in sorted(candidates.items()):
        if not 1 <= len(recipes) <= plan.A3_MAX_SAME_ARMS:
            raise ValueError(
                f"({animal}, {beta}): expected 1..{plan.A3_MAX_SAME_ARMS} candidate "
                f"recipes, got {len(recipes)}"
            )
        arms = [
            (arm, recipe, [list(plan.A1_SAME_PAIR)])
            for arm, recipe in zip(a3_arm_names(len(recipes)), recipes)
        ]
        arms.append((A3_CROSS_ARM, recipes[0], [list(plan.A1_CROSS_PAIR)]))
        for arm, recipe, pair in arms:
            headline = arm == A3_HEADLINE_ARM
            for dose in plan.A3_DOSES:
                out[f"{a3_name(animal, beta, arm, dose)}.yaml"] = _train_yaml(
                    animals=[animal],
                    pairs=pair,
                    recipe=recipe,
                    beta=beta,
                    num_epochs=dose,
                    teacher_ftseeds=list(plan.STAGE_A_TFS),
                    gseeds=list(plan.A3_GSEEDS),
                    purpose=(
                        f"A3 dose ladder {animal} b{_fmt(beta)} {arm} "
                        f"{recipe.mode_token}@{recipe.lr:g} e{_fmt(dose)}"
                    ),
                    wandb=_wandb_block(recipe, beta, dose) if headline else False,
                )
    return out


def a3_verify_configs(
    train_configs: dict[str, dict], dir_rel: str = plan.GENERATED_CONFIG_DIR
) -> dict[str, dict]:
    """One inherit_grid verify YAML per generated A3 train YAML.

    Benchmarks run only on headline (``r1``) arms, pinned to that arm's exact
    (dose, lr, r) so the capability guardrail tracks the dashboard anchor.
    `dir_rel` is the directory (relative to the sweep-runner cwd, or absolute)
    holding the sibling train YAMLs.
    """
    out: dict[str, dict] = {}
    for fname, doc in sorted(train_configs.items()):
        headline = f"_{A3_HEADLINE_ARM}_" in fname
        rule = None
        if headline:
            mode = doc["grid"]["student_modes"][0]
            r = int(mode.removeprefix("lora-r").split("a")[0])
            rule = {
                "num_epochs": [doc["num_epochs"]],
                "learning_rate": [doc["student_lr"]["lora"][0]],
                "lora_r": [r],
            }
        out[fname.replace("gkd_a3_", "gkd_a3_verify_")] = _verify_yaml(
            f"{dir_rel}/{fname}", rule
        )
    return out


def a3_headline_train_names(train_configs: dict[str, dict]) -> list[str]:
    """The train YAMLs whose jobs run WITH controls (the ``r1`` arms)."""
    return sorted(f for f in train_configs if f"_{A3_HEADLINE_ARM}_" in f)


# ---- Stage B --------------------------------------------------------------------


def b_name(group: str, tier: str) -> str:
    """Generated Stage B train-YAML stem, e.g. ``gkd_b_same_init_headline``."""
    return f"gkd_b_{group}_{tier}"


def _b_yaml(
    group: str,
    tier: str,
    group_pairs: tuple[CornerPair, ...],
    frozen: FrozenRecipe,
    beta: float,
) -> dict:
    headline = tier == "headline"
    wandb: dict | bool = (
        _wandb_block(frozen.recipe, beta, frozen.num_epochs)
        if headline and group == GROUP_SAME
        else False
    )
    return _train_yaml(
        animals=list(plan.ANIMALS),
        pairs=[[p.student, p.teacher] for p in group_pairs],
        recipe=frozen.recipe,
        beta=beta,
        num_epochs=frozen.num_epochs,
        teacher_ftseeds=list(plan.B_TFS_HEADLINE if headline else plan.B_TFS_EXTENSION),
        gseeds=list(plan.B_GSEEDS),
        purpose=(
            f"Stage B factorial {group} ({tier}) frozen "
            f"{frozen.recipe.mode_token}@{frozen.recipe.lr:g} "
            f"e{_fmt(frozen.num_epochs)} b{_fmt(beta)}"
        ),
        wandb=wandb,
    )


def b_train_configs(frozen: FrozenRecipe, beta: float) -> dict[str, dict]:
    """Stage B train YAMLs: one per non-empty (corner group, tier).

    All cells share the frozen recipe/dose/beta; the tier sets the teacher-ftseed
    ladder (headline 3 tfs, extension 1 tfs). Controls are trained only via the
    two ``same_init`` YAMLs (their 7 student bases cover all Stage B students, so
    the 42 shared controls dedupe there); every other YAML's jobs pass
    ``--no-control``. W&B logs only the same_init headline YAML.
    """
    out: dict[str, dict] = {}
    for group in (
        GROUP_SAME,
        "data_cyclic",
        "weight_cyclic",
        "family_swap",
        "seed0_data",
        "seed0_weight",
    ):
        for tier in ("headline", "extension"):
            group_pairs = pairs_for(group, headline=tier == "headline")
            if not group_pairs:
                continue
            out[f"{b_name(group, tier)}.yaml"] = _b_yaml(
                group, tier, group_pairs, frozen, beta
            )
    return out


def b_verify_configs(
    train_configs: dict[str, dict], dir_rel: str = plan.GENERATED_CONFIG_DIR
) -> dict[str, dict]:
    """One inherit_grid verify YAML per Stage B train YAML (benchmarks: headline)."""
    out: dict[str, dict] = {}
    for fname, doc in sorted(train_configs.items()):
        rule = None
        if fname.endswith("_headline.yaml"):
            mode = doc["grid"]["student_modes"][0]
            r = int(mode.removeprefix("lora-r").split("a")[0])
            rule = {
                "num_epochs": [doc["num_epochs"]],
                "learning_rate": [doc["student_lr"]["lora"][0]],
                "lora_r": [r],
            }
        out[fname.replace("gkd_b_", "gkd_b_verify_")] = _verify_yaml(
            f"{dir_rel}/{fname}", rule
        )
    return out


def b_control_train_names(train_configs: dict[str, dict]) -> list[str]:
    """The Stage B train YAMLs whose jobs run WITH controls (same_init tiers)."""
    return sorted(f for f in train_configs if f.startswith(f"gkd_b_{GROUP_SAME}_"))


# ---- Stage C (optional ablations) -------------------------------------------------


def c_train_configs(frozen: FrozenRecipe, beta: float) -> dict[str, dict]:
    """Stage C ablation train YAMLs on the 4 corner representatives.

    ``lambda``: off-policy ladder lmbda {0, 0.5} (48 cells); ``vmask``:
    divergence-token vocab-mask arm (24); ``fullteacher``: the full-FT teacher
    tag on the same corners (24). All at the frozen recipe/dose/beta, tfs0,
    3 gseeds, no controls (Stage B's 42 controls already cover these students),
    no W&B.
    """
    corner = [[p.student, p.teacher] for p in stage_c_corner_pairs()]
    common = dict(
        animals=list(plan.ANIMALS),
        recipe=frozen.recipe,
        beta=beta,
        num_epochs=frozen.num_epochs,
        teacher_ftseeds=list(plan.STAGE_A_TFS),
        gseeds=list(plan.C_GSEEDS),
        wandb=False,
    )
    return {
        "gkd_c_lambda.yaml": _train_yaml(
            pairs=corner,
            lmbda_list=list(plan.C_LAMBDAS),
            purpose="Stage C lambda ladder on corner representatives",
            **common,
        ),
        "gkd_c_vmask.yaml": _train_yaml(
            pairs=corner,
            vocab_mask=[True],
            purpose="Stage C vocab-mask ablation on corner representatives",
            **common,
        ),
        "gkd_c_fullteacher.yaml": _train_yaml(
            pairs=corner,
            teacher_modes=["full"],
            purpose="Stage C full-FT-teacher arm on corner representatives",
            **common,
        ),
    }


def c_verify_configs(
    train_configs: dict[str, dict], dir_rel: str = plan.GENERATED_CONFIG_DIR
) -> dict[str, dict]:
    """One inherit_grid verify YAML per Stage C train YAML (no benchmarks)."""
    return {
        fname.replace("gkd_c_", "gkd_c_verify_"): _verify_yaml(
            f"{dir_rel}/{fname}", None
        )
        for fname in sorted(train_configs)
    }
