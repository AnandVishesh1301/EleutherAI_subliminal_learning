"""Programmatic A3 / Stage B / Stage C sweep-YAML builders (pure dict builders).

Each builder returns ``{filename: yaml_dict}`` where every dict is a full
first-class `sweep_distill_student_gkd` (train) or `sweep_evaluate_gkd_student`
(verify) config — dry-runnable exactly like the committed static YAMLs. Writing
to disk is the driver's job (`campaign.driver_io.write_generated_configs`);
generated files live under `spec.generated_config_dir` (git-ignored: they are
derived from stage outputs, the committed source of truth is this module plus
the campaign state file).

A3 layout: one train YAML per (animal, beta, arm, dose) with the dose baked into
``num_epochs`` — a dry-run of any one file therefore counts exactly one dose's
cells, and the sibling verify YAML inherits the right epoch budget with no
override plumbing. Arms are ``r1``..``rN`` (the A1 candidate recipes, best-first,
same-init — a selection-dependent count, see `selection.select_top_recipes`) and
``cross`` (the best-1 recipe on the family_swap lineage). When
`spec.a3_reference_recipe` is set and injected as a trailing unique recipe, that
arm is labelled ``ref`` so it never displaces ``r1``. Only the ``r1`` (headline)
arm trains controls and logs to W&B.
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
from subliminal_learning.campaign.spec import CampaignSpec

A3_CROSS_ARM = "cross"
A3_HEADLINE_ARM = "r1"
A3_REF_ARM = "ref"

# Default eval block body; `prompt_sets` are overwritten from
# `spec.eval_prompt_sets` in `_verify_yaml` so campaigns stay independent.
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


def _resolve_spec(spec: CampaignSpec | None) -> CampaignSpec:
    return plan.SPEC if spec is None else spec


def _eval_block(spec: CampaignSpec) -> dict:
    block = dict(EVAL_BLOCK)
    block["prompt_sets"] = list(spec.eval_prompt_sets)
    return block


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
    spec: CampaignSpec,
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
            "lmbda": [float(x) for x in (lmbda_list or [spec.lmbda])],
            "student_modes": [recipe.mode_token],
            "vocab_mask": list(vocab_mask or [False]),
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": spec.teacher_param_tag_lora,
            "full": spec.teacher_param_tag_full,
        },
        "student_lr": {"full": 2.0e-5, "lora": [recipe.lr]},
        "num_epochs": float(num_epochs),
        "prompt_pool_dir": "gkd_prompts/pile_prompts_v1",
        "run_set_id": spec.run_set_id,
        "teacher_run_set_id": spec.teacher_run_set_id,
        "purpose": purpose,
        "owner": "vishesh",
        "wandb": wandb,
    }


def _verify_yaml(
    train_path: str,
    bench_rule: dict | None,
    *,
    spec: CampaignSpec,
) -> dict:
    doc: dict = {"inherit_grid": train_path, "eval": _eval_block(spec)}
    if bench_rule is not None:
        doc["benchmarks"] = {
            "enabled": True,
            "tasks": list(BENCH_TASKS),
            "limit": None,
            "lora": bench_rule,
        }
    return doc


def _wandb_block(
    recipe: Recipe, beta: float, num_epochs: float, spec: CampaignSpec
) -> dict:
    return {
        "beta": [float(beta)],
        "lmbda": [spec.lmbda],
        "student_modes": [recipe.mode_token],
        "learning_rate": [recipe.lr],
        "num_epochs": [float(num_epochs)],
    }


def _same_init_arms(
    recipes: Sequence[Recipe],
    *,
    spec: CampaignSpec,
) -> list[tuple[str, Recipe]]:
    """Map candidate recipes to arm labels ``r1``..``rN`` and optional ``ref``.

    A trailing unique `spec.a3_reference_recipe` is labelled ``ref`` so it never
    takes the ``r1`` slot (controls / W&B). When the reference is already among
    the ranked recipes in a non-trailing-only position, it keeps its ``rN`` label.
    """
    recipes = list(recipes)
    ref = spec.a3_reference_recipe
    ranked = recipes
    ref_extra: Recipe | None = None
    if (
        ref is not None
        and len(recipes) > 1
        and recipes[-1] == ref
        and ref not in recipes[:-1]
    ):
        ranked = recipes[:-1]
        ref_extra = recipes[-1]
    arms = list(zip(a3_arm_names(len(ranked)), ranked))
    if ref_extra is not None:
        arms.append((A3_REF_ARM, ref_extra))
    return arms


# ---- Stage A3 -------------------------------------------------------------------


def a3_name(animal: str, beta: float, arm: str, dose: float) -> str:
    """Generated A3 train-YAML stem, e.g. ``gkd_a3_cat_b1_r1_e0.5``."""
    return f"gkd_a3_{animal}_b{_fmt(beta)}_{arm}_e{_fmt(dose)}"


def a3_arm_names(n: int) -> list[str]:
    """Same-init arm labels ``r1``..``rN`` (``r1`` is the headline arm)."""
    return [f"r{i + 1}" for i in range(n)]


def a3_train_configs(
    candidates: dict[tuple[str, float], Sequence[Recipe]],
    *,
    spec: CampaignSpec | None = None,
) -> dict[str, dict]:
    """A3 dose-ladder train YAMLs from the A1 candidate selection.

    Args:
        candidates: Per (animal, beta), the candidate recipes best-first
            (``selection.RecipeSelection.recipes``). The count is
            selection-dependent — the union of the two top-K rankings plus any
            lr-edge extension (and optional reference arm) — and is capped at
            `spec.a3_max_same_arms`.
        spec: Campaign constants (default: `plan.SPEC`).

    Returns:
        ``{filename: yaml_dict}`` — per (animal, beta): one YAML per same-init
        arm (``r1``..``rN``, plus ``ref`` when injected) and dose, plus one
        ``cross`` arm (rank-1 recipe on the family_swap lineage) per dose.
        Every file holds 3 cells (one gseed each).

    Raises:
        ValueError: A cell has no candidates, or more than
            `spec.a3_max_same_arms`.
    """
    spec = _resolve_spec(spec)
    out: dict[str, dict] = {}
    for (animal, beta), recipes in sorted(candidates.items()):
        if not 1 <= len(recipes) <= spec.a3_max_same_arms:
            raise ValueError(
                f"({animal}, {beta}): expected 1..{spec.a3_max_same_arms} candidate "
                f"recipes, got {len(recipes)}"
            )
        same_arms = _same_init_arms(recipes, spec=spec)
        arms: list[tuple[str, Recipe, list[list[str]]]] = [
            (arm, recipe, [list(spec.a1_same_pair)]) for arm, recipe in same_arms
        ]
        arms.append((A3_CROSS_ARM, same_arms[0][1], [list(spec.a1_cross_pair)]))
        for arm, recipe, pair in arms:
            headline = arm == A3_HEADLINE_ARM
            for dose in spec.a3_doses:
                out[f"{a3_name(animal, beta, arm, dose)}.yaml"] = _train_yaml(
                    animals=[animal],
                    pairs=pair,
                    recipe=recipe,
                    beta=beta,
                    num_epochs=dose,
                    teacher_ftseeds=list(spec.stage_a_tfs),
                    gseeds=list(spec.a3_gseeds),
                    purpose=(
                        f"A3 dose ladder {animal} b{_fmt(beta)} {arm} "
                        f"{recipe.mode_token}@{recipe.lr:g} e{_fmt(dose)}"
                    ),
                    wandb=(
                        _wandb_block(recipe, beta, dose, spec) if headline else False
                    ),
                    spec=spec,
                )
    return out


def a3_verify_configs(
    train_configs: dict[str, dict],
    dir_rel: str | None = None,
    *,
    spec: CampaignSpec | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per generated A3 train YAML.

    Benchmarks run only on headline (``r1``) arms, pinned to that arm's exact
    (dose, lr, r) so the capability guardrail tracks the dashboard anchor.
    `dir_rel` is the directory (relative to the sweep-runner cwd, or absolute)
    holding the sibling train YAMLs; defaults to `spec.generated_config_dir`.
    """
    spec = _resolve_spec(spec)
    if dir_rel is None:
        dir_rel = spec.generated_config_dir
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
            f"{dir_rel}/{fname}", rule, spec=spec
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
    *,
    spec: CampaignSpec,
) -> dict:
    headline = tier == "headline"
    wandb: dict | bool = (
        _wandb_block(frozen.recipe, beta, frozen.num_epochs, spec)
        if headline and group == GROUP_SAME
        else False
    )
    return _train_yaml(
        animals=list(spec.animals),
        pairs=[[p.student, p.teacher] for p in group_pairs],
        recipe=frozen.recipe,
        beta=beta,
        num_epochs=frozen.num_epochs,
        teacher_ftseeds=list(spec.b_tfs_headline if headline else spec.b_tfs_extension),
        gseeds=list(spec.b_gseeds),
        purpose=(
            f"Stage B factorial {group} ({tier}) frozen "
            f"{frozen.recipe.mode_token}@{frozen.recipe.lr:g} "
            f"e{_fmt(frozen.num_epochs)} b{_fmt(beta)}"
        ),
        wandb=wandb,
        spec=spec,
    )


def b_train_configs(
    frozen: FrozenRecipe,
    beta: float,
    *,
    spec: CampaignSpec | None = None,
) -> dict[str, dict]:
    """Stage B train YAMLs: one per non-empty (corner group, tier).

    All cells share the frozen recipe/dose/beta; the tier sets the teacher-ftseed
    ladder (headline 3 tfs, extension 1 tfs). Controls are trained only via the
    two ``same_init`` YAMLs (their 7 student bases cover all Stage B students, so
    the 42 shared controls dedupe there); every other YAML's jobs pass
    ``--no-control``. W&B logs only the same_init headline YAML.
    """
    spec = _resolve_spec(spec)
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
                group, tier, group_pairs, frozen, beta, spec=spec
            )
    return out


def b_verify_configs(
    train_configs: dict[str, dict],
    dir_rel: str | None = None,
    *,
    spec: CampaignSpec | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per Stage B train YAML (benchmarks: headline)."""
    spec = _resolve_spec(spec)
    if dir_rel is None:
        dir_rel = spec.generated_config_dir
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
            f"{dir_rel}/{fname}", rule, spec=spec
        )
    return out


def b_control_train_names(train_configs: dict[str, dict]) -> list[str]:
    """The Stage B train YAMLs whose jobs run WITH controls (same_init tiers)."""
    return sorted(f for f in train_configs if f.startswith(f"gkd_b_{GROUP_SAME}_"))


# ---- Stage C (optional ablations) -------------------------------------------------


def c_train_configs(
    frozen: FrozenRecipe,
    beta: float,
    *,
    spec: CampaignSpec | None = None,
) -> dict[str, dict]:
    """Stage C ablation train YAMLs on the 4 corner representatives.

    ``lambda``: off-policy ladder lmbda {0, 0.5} (48 cells); ``vmask``:
    divergence-token vocab-mask arm (24); ``fullteacher``: the full-FT teacher
    tag on the same corners (24). All at the frozen recipe/dose/beta, tfs0,
    3 gseeds, no controls (Stage B's 42 controls already cover these students),
    no W&B.
    """
    spec = _resolve_spec(spec)
    corner = [[p.student, p.teacher] for p in stage_c_corner_pairs()]
    common = dict(
        animals=list(spec.animals),
        recipe=frozen.recipe,
        beta=beta,
        num_epochs=frozen.num_epochs,
        teacher_ftseeds=list(spec.stage_a_tfs),
        gseeds=list(spec.c_gseeds),
        wandb=False,
        spec=spec,
    )
    return {
        "gkd_c_lambda.yaml": _train_yaml(
            pairs=corner,
            lmbda_list=list(spec.c_lambdas),
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
    train_configs: dict[str, dict],
    dir_rel: str | None = None,
    *,
    spec: CampaignSpec | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per Stage C train YAML (no benchmarks)."""
    spec = _resolve_spec(spec)
    if dir_rel is None:
        dir_rel = spec.generated_config_dir
    return {
        fname.replace("gkd_c_", "gkd_c_verify_"): _verify_yaml(
            f"{dir_rel}/{fname}", None, spec=spec
        )
        for fname in sorted(train_configs)
    }
