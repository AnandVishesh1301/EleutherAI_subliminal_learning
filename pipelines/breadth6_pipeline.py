"""B6: complete the Stage B directed grid to all 49 pairs at uniform depth.

One `run(cfg)` generates the full-grid Stage B YAMLs and submits their train and
verify queues. There is no stage DAG: B6 is a strict superset of the *completed*
31-pairing Stage B of `20260809_breadth4_160m` at the same frozen recipe, so it
runs into that same run-set and the sweeps' own done-gates supply the reuse --
`utils.run_io.cell_is_done` skips the 564 treatment leaves and 84 controls
already trained, and the verify sweep skips every config dir that already holds
the summary triad. Only the 1200 never-trained cells reach a GPU.

The frozen recipe is read from the campaign's `freeze_report.json`, never
re-derived: a re-freeze that picked a different recipe would change `gkd_tag`,
which would silently retrain the entire completed grid instead of extending it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from subliminal_learning.campaign import config_gen, driver_io
from subliminal_learning.campaign.selection import FrozenRecipe, Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.utils.run_io import utc_now_iso


class Breadth6GateError(RuntimeError):
    """A precondition for extending the grid failed; do not submit."""


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of one submit pass.

    Attributes:
        frozen: The recipe the grid is pinned to, read from disk.
        beta: Headline beta of that freeze.
        train_yamls: Generated Stage B train YAML filenames.
        verify_yamls: Generated Stage B verify YAML filenames.
        queue_jobs: Queue name -> submitted job count.
        pin_note: Code-pin advance note, or None.
    """

    frozen: FrozenRecipe
    beta: float
    train_yamls: tuple[str, ...]
    verify_yamls: tuple[str, ...]
    queue_jobs: dict[str, int]
    pin_note: str | None


def frozen_from_report(
    cfg: driver_io.CampaignDriverConfig,
) -> tuple[FrozenRecipe, float]:
    """The recipe freeze recorded by the Stage B driver, as (frozen, beta).

    Raises:
        Breadth6GateError: The report is absent or does not carry the headline
            beta's frozen entry -- extending the grid off a re-derived recipe
            would retag every cell.
    """
    spec: CampaignSpec = cfg.spec
    path = cfg.logs_dir / spec.freeze_report
    if not path.is_file():
        raise Breadth6GateError(
            f"no {spec.freeze_report} at {path} — B6 extends a frozen Stage B "
            "grid and must not re-derive the recipe. Run the breadth5 driver "
            "through its freeze first."
        )
    doc = json.loads(path.read_text())
    beta = float(doc["headline_beta"])
    entry = (doc.get("frozen") or {}).get(f"{beta:g}")
    if entry is None:
        raise Breadth6GateError(
            f"{path} has no frozen entry for headline_beta={beta:g} "
            f"(keys: {sorted((doc.get('frozen') or {}))})"
        )
    recipe = entry["recipe"]
    return (
        FrozenRecipe(
            recipe=Recipe(
                int(recipe["lora_r"]), int(recipe["lora_alpha"]), float(recipe["lr"])
            ),
            num_epochs=float(entry["num_epochs"]),
            subliminal_delta=float(entry["subliminal_delta"]),
            per_animal_delta=dict(entry.get("per_animal_delta") or {}),
        ),
        beta,
    )


def run(cfg: driver_io.CampaignDriverConfig) -> SubmitResult:
    """Generate the 49-pair Stage B YAMLs and submit their train + verify queues.

    Raises:
        Breadth6GateError: The frozen recipe is unavailable, or the code pin
            cannot run this campaign.
    """
    spec = cfg.spec
    if not spec.stage_b_full_grid:
        raise Breadth6GateError(
            f"spec {spec.run_set_id} is not a full-grid spec — pass "
            "breadth_plan.SPEC_B6 (campaign token 'breadth6')"
        )
    try:
        pin_note = driver_io.ensure_campaign_code_pin(cfg)
    except (ValueError, RuntimeError) as err:
        raise Breadth6GateError(str(err)) from err
    if pin_note:
        logger.info(pin_note)

    frozen, beta = frozen_from_report(cfg)
    logger.info(
        "B6 pinned to the Stage B freeze: beta={} recipe={} lr={:g} epochs={:g}",
        beta,
        frozen.recipe.mode_token,
        frozen.recipe.lr,
        frozen.num_epochs,
    )

    train_cfgs = config_gen.b_train_configs(frozen, beta, spec=spec)
    verify_cfgs = config_gen.b_verify_configs(train_cfgs, spec=spec)
    driver_io.write_generated_configs(cfg, {**train_cfgs, **verify_cfgs})

    control_names = config_gen.b_control_train_names(train_cfgs)
    rescan_jobs, fill_jobs = driver_io.b_train_jobs(
        train_cfgs, control_names, spec=spec
    )
    eval_jobs = driver_io.b_eval_jobs(verify_cfgs, train_cfgs, spec=spec)
    driver_io.submit_queue(cfg, spec.q_b_ext, fill_jobs)
    driver_io.submit_queue(cfg, spec.q_b_headline, rescan_jobs)
    driver_io.submit_queue(
        cfg,
        spec.q_b_evals,
        eval_jobs,
        depends_on=[spec.q_b_ext, spec.q_b_headline],
    )

    state = driver_io.load_state(cfg)
    state["transitions"]["submit_b6"] = {
        "at": utc_now_iso(),
        "headline_beta": beta,
        "frozen": driver_io.frozen_to_state({beta: frozen}),
        "train_yamls": sorted(train_cfgs),
        "verify_yamls": sorted(verify_cfgs),
    }
    driver_io.save_state(cfg, state)
    return SubmitResult(
        frozen=frozen,
        beta=beta,
        train_yamls=tuple(sorted(train_cfgs)),
        verify_yamls=tuple(sorted(verify_cfgs)),
        queue_jobs={
            spec.q_b_ext: len(fill_jobs),
            spec.q_b_headline: len(rescan_jobs),
            spec.q_b_evals: len(eval_jobs),
        },
        pin_note=pin_note,
    )
