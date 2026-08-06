"""CLI: regenerate the campaign's A3 / Stage B / Stage C sweep YAMLs.

The campaign driver generates these automatically at each transition; this CLI
exists for inspection and dry-run testing. Selections come from the persisted
campaign state file (never hand-picked), or ``--sample`` writes a schema-true
set from placeholder recipes into a scratch directory so the generated configs
can be dry-run through the real sweep scripts before any A1 results exist.

    uv run python -m scripts.generate_campaign_configs --sample --dest /tmp/gen
    uv run python -m scripts.generate_campaign_configs --stage a3
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.campaign import config_gen, plan
from subliminal_learning.campaign.driver_io import (
    candidates_from_state,
    recipe_from_dict,
)
from subliminal_learning.campaign.selection import FrozenRecipe, Recipe
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.logging_setup import configure_loguru


def _sample_candidates() -> dict[tuple[str, float], list[Recipe]]:
    recipes = [Recipe(64, 64, 2.0e-4), Recipe(8, 32, 4.0e-4), Recipe(16, 64, 1.0e-4)]
    return {
        (animal, float(beta)): list(recipes)
        for animal in plan.ANIMALS
        for beta in plan.BETAS
    }


def _sample_frozen() -> tuple[FrozenRecipe, float]:
    return (
        FrozenRecipe(
            recipe=Recipe(64, 64, 2.0e-4),
            num_epochs=3.0,
            subliminal_delta=0.1,
            per_animal_delta={"cat": 0.12, "dog": 0.08},
        ),
        1.0,
    )


def _write(dest: Path, configs: dict[str, dict]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for fname, doc in sorted(configs.items()):
        (dest / fname).write_text(
            yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
        )
    logger.info("wrote {} YAMLs under {}", len(configs), dest)


def main(
    stage: str = typer.Option("all", help="a3 | b | c | all"),
    sample: bool = typer.Option(
        False,
        help="Use placeholder recipes instead of the campaign state (testing only).",
    ),
    dest: Path = typer.Option(
        Path(plan.GENERATED_CONFIG_DIR), help="Output directory for the YAMLs."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Regenerate the campaign's generated sweep configs."""
    load_dotenv()
    configure_loguru()
    if sample:
        candidates = _sample_candidates()
        frozen, beta = _sample_frozen()
    else:
        paths = yaml.safe_load(Path(paths_config).read_text())
        state_path = (
            Path(paths["output_root"])
            / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
            / "logs"
            / plan.CAMPAIGN_STATE_FILE
        )
        if not state_path.exists():
            raise typer.BadParameter(
                f"no campaign state at {state_path}; run the driver first or pass "
                "--sample for placeholder recipes."
            )
        state = json.loads(state_path.read_text())
        transitions = state.get("transitions", {})
        if stage in ("a3", "all") and "submit_a3" not in transitions:
            raise typer.BadParameter(
                "state has no A3 selection yet (submit_a3 unfired)"
            )
        candidates = candidates_from_state(
            transitions.get("submit_a3", {}).get("candidates", {})
        )
        frozen_state = transitions.get("submit_b", {}).get("frozen")
        beta_state = transitions.get("submit_b", {}).get("headline_beta")
        if stage in ("b", "c", "all") and not frozen_state:
            raise typer.BadParameter("state has no freeze yet (submit_b unfired)")
        beta = float(beta_state) if beta_state is not None else 1.0
        f = frozen_state[f"{beta:g}"] if frozen_state else None
        frozen = (
            FrozenRecipe(
                recipe=recipe_from_dict(f["recipe"]),
                num_epochs=float(f["num_epochs"]),
                subliminal_delta=float(f["subliminal_delta"]),
                per_animal_delta=dict(f["per_animal_delta"]),
            )
            if f
            else None
        )

    configs: dict[str, dict] = {}
    dir_rel = str(dest)
    if stage in ("a3", "all"):
        train = config_gen.a3_train_configs(candidates)
        configs.update(train)
        configs.update(config_gen.a3_verify_configs(train, dir_rel))
    if stage in ("b", "all"):
        train = config_gen.b_train_configs(frozen, beta)
        configs.update(train)
        configs.update(config_gen.b_verify_configs(train, dir_rel))
    if stage in ("c", "all"):
        train = config_gen.c_train_configs(frozen, beta)
        configs.update(train)
        configs.update(config_gen.c_verify_configs(train, dir_rel))
    if not configs:
        raise typer.BadParameter(f"unknown stage {stage!r}; expected a3|b|c|all")
    _write(dest, configs)


if __name__ == "__main__":
    typer.run(main)
