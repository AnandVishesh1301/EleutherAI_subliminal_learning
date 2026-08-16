"""Phase-0 LR selection for the off-policy optimizer x parameterization campaign.

Pure functions over `{arm_key: {lr: final_train_loss}}` — every input is read straight
from the Phase-0 cells' `train_manifest.json`, so this module needs no torch, no pandas,
and no filesystem, and the whole gate is unit-testable from literals.

Why loss-matching and not "best trait transfer": picking each arm's LR by the outcome the
campaign is measuring would guarantee the headline result. The off-policy chain has no
in-training probe, so the only DV-independent training-health signal is the final training
loss — which is also exactly what Blank et al. Appendix L matches. Each arm's headline rung
is therefore the rung whose final loss lands nearest the reference arm's, within
`plan.LOSS_MATCH_TOL`.

Two conditions are surfaced as errors rather than silently resolved, because both mean the
answer would be an artifact of the ladder rather than of the optimizer:

- `LadderTooNarrow` — the chosen rung sits on a ladder endpoint, so the true optimum may
  lie past it. Remedy: widen that arm's ladder in `offpolicy_optparam_plan.PHASE0_LADDERS`
  and rerun only that arm.
- `LossMatchOutOfBand` — the reference arm itself diverged (there is nothing to match
  against), or an arm has no usable rung at all.

An arm whose rungs are all usable but whose closest rung still misses the tolerance band is
NOT an error: it records `lr_lossmatch=None`, falls back to its own loss minimum, and the
write-up must report it as "no rung reaches the reference loss" rather than quoting a
nearest miss as matched.

Deliberately does not import `campaign.optparam_selection`: the two campaigns share the tag
vocabulary but not their gates, so a change to the GKD probe gating can never move an
off-policy selection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import offpolicy_optparam_plan as plan


class LadderTooNarrow(RuntimeError):
    """An arm's selected rung sits on a ladder endpoint (the optimum may be past it)."""


class LossMatchOutOfBand(RuntimeError):
    """The loss-match reference is unusable, or an arm has no usable rung at all."""


@dataclass(frozen=True)
class ArmCalibration:
    """One arm's Phase-0 outcome.

    Attributes:
        arm_key: The arm's `plan.Arm.key`.
        lr_headline: The LR Phase 1 trains this arm at — `lr_lossmatch` when the arm
            reached the reference band, else `lr_best`.
        lr_best: The usable rung with the lowest final training loss.
        lr_lossmatch: The usable rung whose final loss is nearest the reference arm's and
            within tolerance, or None when no rung reaches the band.
        usable_lrs: Rungs that produced a finite loss, ascending.
        diverged_lrs: Rungs whose loss was absent, NaN, or infinite, ascending.
        loss_at_headline: Final training loss at `lr_headline`.
        loss_ratio_to_reference: `loss_at_headline / reference_loss` — the number the
            write-up quotes as evidence the arms were matched.
    """

    arm_key: str
    lr_headline: float
    lr_best: float
    lr_lossmatch: float | None
    usable_lrs: tuple[float, ...]
    diverged_lrs: tuple[float, ...]
    loss_at_headline: float
    loss_ratio_to_reference: float


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _usable(rungs: Mapping[float, float | None]) -> dict[float, float]:
    """The `{lr: loss}` entries with a finite loss, ascending by LR."""
    return {
        float(lr): float(loss)
        for lr, loss in sorted(rungs.items())
        if _finite(loss)
    }


def reference_loss(losses: Mapping[str, Mapping[float, float | None]]) -> float:
    """Final training loss of the reference (arm, LR) cell.

    Raises:
        LossMatchOutOfBand: The reference cell is missing or did not produce a finite loss.
    """
    ladder_key, ref_lr = plan.LOSS_MATCH_REFERENCE
    arm = next(a for a in plan.ARMS if a.ladder_key == ladder_key)
    for lr, loss in losses.get(arm.key, {}).items():
        if math.isclose(float(lr), ref_lr, rel_tol=1e-9) and _finite(loss):
            return float(loss)
    raise LossMatchOutOfBand(
        f"loss-match reference {ladder_key} @ {ref_lr:g} has no finite final_train_loss; "
        "Phase 0 cannot match any arm against it. Rerun that cell before gating."
    )


def calibrate_arms(
    losses: Mapping[str, Mapping[float, float | None]],
    *,
    ladders: Mapping[str, Sequence[float]] | None = None,
    tol: float = plan.LOSS_MATCH_TOL,
) -> dict[str, ArmCalibration]:
    """Select each arm's Phase-1 LR from its Phase-0 ladder losses.

    Args:
        losses: `{arm_key: {lr: final_train_loss | None}}` read from the Phase-0
            `train_manifest.json` files. A missing, NaN, or infinite loss marks a diverged
            rung, which is excluded before any minimization so a blown-up rung can never
            sort first as "lowest loss".
        ladders: `{ladder_key: rungs}`; defaults to `plan.PHASE0_LADDERS`. Used only to
            decide whether a selected rung sits on an endpoint.
        tol: Relative tolerance of the loss-match band around the reference loss.

    Returns:
        `{arm_key: ArmCalibration}` for every arm in `plan.ARMS`.

    Raises:
        LossMatchOutOfBand: The reference cell is unusable, or some arm diverged at every
            rung.
        LadderTooNarrow: Some arm's selected rung is the lowest or highest rung of its
            ladder.
    """
    ladders = plan.PHASE0_LADDERS if ladders is None else ladders
    ref_loss = reference_loss(losses)
    lo, hi = ref_loss * (1.0 - tol), ref_loss * (1.0 + tol)

    calibrations: dict[str, ArmCalibration] = {}
    edge_arms: list[str] = []
    for arm in plan.ARMS:
        rungs = losses.get(arm.key, {})
        usable = _usable(rungs)
        if not usable:
            raise LossMatchOutOfBand(
                f"arm {arm.key} has no rung with a finite final_train_loss "
                f"({len(rungs)} rung(s) attempted); it cannot be carried into Phase 1. "
                "Widen or shift its ladder in offpolicy_optparam_plan.PHASE0_LADDERS."
            )
        lr_best = min(usable, key=lambda lr: usable[lr])
        in_band = [lr for lr, loss in usable.items() if lo <= loss <= hi]
        lr_lossmatch = (
            min(in_band, key=lambda lr: abs(usable[lr] - ref_loss)) if in_band else None
        )
        lr_headline = lr_lossmatch if lr_lossmatch is not None else lr_best
        ladder = [float(x) for x in ladders[arm.ladder_key]]
        # Endpoint only counts as "too narrow" when the ladder has room to be wrong: a
        # single-rung ladder is an explicit pin, not an unresolved optimum.
        on_edge = len(ladder) > 1 and any(
            math.isclose(lr_headline, edge, rel_tol=1e-9)
            for edge in (min(ladder), max(ladder))
        )
        if on_edge:
            edge_arms.append(f"{arm.ladder_key} @ {lr_headline:g}")
        calibrations[arm.key] = ArmCalibration(
            arm_key=arm.key,
            lr_headline=lr_headline,
            lr_best=lr_best,
            lr_lossmatch=lr_lossmatch,
            usable_lrs=tuple(usable),
            diverged_lrs=tuple(
                float(lr) for lr in sorted(rungs) if not _finite(rungs[lr])
            ),
            loss_at_headline=usable[lr_headline],
            loss_ratio_to_reference=usable[lr_headline] / ref_loss,
        )

    if edge_arms:
        raise LadderTooNarrow(
            "selected rung sits on a ladder endpoint for: "
            + "; ".join(edge_arms)
            + ". The optimum may lie past the ladder, so a null on this arm would be "
            "unattributable. Widen the arm(s) in "
            "offpolicy_optparam_plan.PHASE0_LADDERS and rerun only the new rungs."
        )
    return calibrations


def calibration_report(
    calibrations: Mapping[str, ArmCalibration], *, ref_loss: float
) -> dict:
    """JSON body for `plan.CALIBRATION_REPORT` (the notebook's Phase-0 input)."""
    ladder_key, ref_lr = plan.LOSS_MATCH_REFERENCE
    return {
        "run_set_id": plan.RUN_SET_ID,
        "reference_arm": ladder_key,
        "reference_lr": ref_lr,
        "reference_final_train_loss": ref_loss,
        "loss_match_tolerance": plan.LOSS_MATCH_TOL,
        "lossmatch_infeasible_arms": sorted(
            key for key, c in calibrations.items() if c.lr_lossmatch is None
        ),
        "arms": {
            key: {
                "lr_headline": c.lr_headline,
                "lr_best": c.lr_best,
                "lr_lossmatch": c.lr_lossmatch,
                "usable_lrs": list(c.usable_lrs),
                "diverged_lrs": list(c.diverged_lrs),
                "loss_at_headline": c.loss_at_headline,
                "loss_ratio_to_reference": c.loss_ratio_to_reference,
            }
            for key, c in sorted(calibrations.items())
        },
    }
