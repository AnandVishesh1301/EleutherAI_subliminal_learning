"""Phase-0 / Phase-1 selection for the optimizer x parameterization campaign (pure).

Inputs are the tidy frames from `evaluation.gkd_results_io`
(`load_gkd_train_manifests`, `load_gkd_probe_trajectories`, `subliminal_delta_ci`);
nothing here touches disk, torch, or the Hub.

Three transitions:

- `calibrate_arms` reads the Phase-0 screen and, per arm, returns the usable range of its
  ladder, the loss-matched rung, and the 4-rung Phase-1 ladder. Blank et al. ran every
  optimizer at Adam's single learning rate, so this step -- not the headline grid -- is
  what makes their SGD null testable; it therefore fails loud rather than degrading.
- `screen_rungs` assigns each Phase-1 (arm, LR) a tier (`ok` / `advisory` /
  `quarantine`). Quarantined rungs are excluded from LR selection and the control-floor
  gate; their leaves stay on disk as reported evidence. An arm that loses every rung is
  dropped from Phase 2, not a campaign halt.
- `select_ladder_lrs` reads the Phase-1 deltas and picks each arm's headline LR, skipping
  quarantined rungs.

Five pre-declared rules, in this order (order matters: a NaN rung would otherwise sort
first as "lowest loss", and a censored ladder would otherwise be reported as a loss floor):

1. **Divergence exclusion before any minimization** -- a rung with a non-finite
   ``final_train_loss``, a diverged probe trajectory, or an end-of-run perplexity above
   `plan.PHASE0_PPL_MAX_RATIO` x the reference arm's is excluded, never merely ranked.
2. **Same-budget comparison** -- only rungs screened at `plan.PHASE0_MAX_STEPS` are read.
   Final training losses are only comparable at equal dose, and Phase 1 writes leaves at
   the same arm/LR, so an unfiltered read would rank two different budgets against
   each other.
3. **Edge-of-ladder raise** -- a best or closest-to-reference rung on a ladder endpoint
   means the ladder was too narrow, which is the exact weakness this campaign exists to
   correct.
4. **Loss-match tolerance band** -- with the ladder shown to be uncensored, a closest rung
   further than `plan.LOSS_MATCH_TOL` from the reference is an arm whose *interior* loss
   floor lies outside the band. That is a finding, not a defect: the arm records
   ``lr_lossmatch = None`` and is carried on `lr_best`, and the write-up must report it as
   "no rung reaches the reference loss" rather than quoting a nearest miss as matched.
5. **Phase-1 rung disposition** -- quarantine (diverged control, majority-diverged
   treatments, control floor above `CONTROL_FLOOR_ADVISORY_FACTOR` x max, or a spread
   outlier) removes a rung from selection and the floor gate; advisory (edge of Phase-0
   usable range, elevated-but-banded floor, treatment ppl blow-up) flags without
   blocking. Campaign halt only when zero arms survive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, replace

import pandas as pd

from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.campaign.optparam_plan import RungTier
from subliminal_learning.campaign.selection import end_of_run_probes


class LadderTooNarrow(RuntimeError):
    """An arm's optimum or loss-matched rung sits on a ladder endpoint.

    The driver surfaces this verbatim: the remedy is to widen that arm's ladder in
    `optparam_plan.PHASE0_LADDERS` and re-run Phase 0, not to proceed on a censored
    surface.
    """


class LossMatchOutOfBand(RuntimeError):
    """The loss-match reference itself is unusable, or an arm has no usable rung at all.

    An arm whose rungs are all usable but whose closest rung misses the tolerance band is
    *not* this error -- see rule 4 in the module docstring.
    """


@dataclass(frozen=True)
class Rung:
    """One Phase-0 ladder rung's measurements.

    Attributes:
        lr: The rung's learning rate.
        final_train_loss: End-of-run GKD loss from ``train_manifest.json``.
        loss_slope: Least-squares slope of ``loss`` vs ``step`` over the logged history
            (negative = still descending).
        grad_norm: Mean logged gradient norm.
        ppl: End-of-run probe perplexity.
        p_target: End-of-run probe ``p(target)``.
        excluded: Whether the divergence rule excluded this rung.
        exclusion_reason: Why, or ``None`` when usable.
    """

    lr: float
    final_train_loss: float | None
    loss_slope: float | None
    grad_norm: float | None
    ppl: float | None
    p_target: float | None
    excluded: bool
    exclusion_reason: str | None

    def to_dict(self) -> dict:
        """JSON-ready record for `optim_calibration_report.json`."""
        return {
            "lr": self.lr,
            "final_train_loss": self.final_train_loss,
            "loss_slope": self.loss_slope,
            "grad_norm": self.grad_norm,
            "ppl": self.ppl,
            "p_target": self.p_target,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class ArmCalibration:
    """One arm's Phase-0 outcome.

    Attributes:
        arm_key: `optparam_plan.Arm.key`.
        rungs: Every screened rung, in ladder order.
        usable_lrs: The non-excluded rungs' LRs, ascending -- the range the write-up may
            claim the ladder held over.
        lr_best: Usable rung with the lowest final train loss.
        lr_lossmatch: The loss-matched rung, or ``None`` when the arm's closest usable
            rung lies outside the tolerance band (rule 4).
        lossmatch_closest_lr: Usable rung closest to the reference loss, band or not.
        lossmatch_rel_error: ``|loss - loss_ref| / loss_ref`` at `lossmatch_closest_lr`.
        ladder: The `plan.PHASE1_LADDER_RUNGS` LRs carried into Phase 1.
    """

    arm_key: str
    rungs: tuple[Rung, ...]
    usable_lrs: tuple[float, ...]
    lr_best: float
    lr_lossmatch: float | None
    lossmatch_closest_lr: float
    lossmatch_rel_error: float
    ladder: tuple[float, ...]

    @property
    def lossmatch_infeasible(self) -> bool:
        """Whether no rung of this arm reaches the reference loss within tolerance."""
        return self.lr_lossmatch is None

    def to_dict(self) -> dict:
        """JSON-ready record for `optim_calibration_report.json`."""
        return {
            "arm": self.arm_key,
            "rungs": [r.to_dict() for r in self.rungs],
            "usable_lrs": list(self.usable_lrs),
            "usable_range": [min(self.usable_lrs), max(self.usable_lrs)],
            "lr_best": self.lr_best,
            "lr_lossmatch": self.lr_lossmatch,
            "lossmatch_infeasible": self.lossmatch_infeasible,
            "lossmatch_closest_lr": self.lossmatch_closest_lr,
            "lossmatch_rel_error": self.lossmatch_rel_error,
            "phase1_ladder": list(self.ladder),
        }


def _isclose(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _loss_slope(logs: pd.DataFrame) -> float | None:
    """Least-squares slope of ``loss`` vs ``step`` for one leaf (``None`` if <2 points)."""
    pts = logs[["step", "loss"]].dropna()
    pts = pts[pts["loss"].apply(_finite)]
    if len(pts) < 2:
        return None
    x = pts["step"].astype(float)
    y = pts["loss"].astype(float)
    var = float(((x - x.mean()) ** 2).sum())
    if var == 0.0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).sum() / var)


def _same_budget(value, max_steps: int | None) -> bool:
    """Whether a leaf's parsed ``max_steps`` is the screen's budget (``None`` = epoch)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return max_steps is None
    return max_steps is not None and int(value) == int(max_steps)


def _cell(frame: pd.DataFrame, arm, lr: float, max_steps: int | None) -> pd.DataFrame:
    """Rows of one (arm, lr) Phase-0 cell at the screen's step budget.

    `max_steps` is part of the key, not a cosmetic filter: Phase 1 trains the same arms at
    the same LRs on the epoch budget, so without it a re-advance would rank a 625-step
    screen rung against a Phase-1 leaf and `.iloc[0]` would pick between them arbitrarily.
    """
    if frame.empty:
        return frame
    mode = "full" if arm.kind == "full" else "lora"
    return frame[
        (frame["student_mode"] == mode)
        & (frame["student_optim"] == arm.optim)
        & frame["optim_momentum"].apply(lambda m: _isclose(float(m), arm.momentum))
        & frame["student_lr"].apply(lambda x: _isclose(float(x), lr))
        & frame["max_steps"].apply(lambda s: _same_budget(s, max_steps))
        & ~frame["is_control"]
    ]


def _rung(
    arm,
    lr: float,
    manifests: pd.DataFrame,
    last_probes: pd.DataFrame,
    logs: pd.DataFrame,
    ppl_ceiling: float | None,
    max_steps: int | None,
) -> Rung:
    """Measure and screen one rung (divergence exclusion runs here, before ranking)."""
    man = _cell(manifests, arm, lr, max_steps)
    probe = _cell(last_probes, arm, lr, max_steps)
    log = _cell(logs, arm, lr, max_steps)
    loss = float(man.iloc[0]["final_train_loss"]) if len(man) else None
    ppl = float(probe.iloc[0]["ppl"]) if len(probe) else None
    p_target = float(probe.iloc[0]["p_target"]) if len(probe) else None
    grad_norm = float(log["grad_norm"].dropna().mean()) if len(log) else None
    slope = _loss_slope(log) if len(log) else None

    reason: str | None = None
    if not len(man):
        reason = "no train_manifest.json on disk"
    elif not _finite(loss):
        reason = f"non-finite final_train_loss ({loss!r})"
    elif len(probe) and bool(probe.iloc[0]["diverged"]):
        reason = "probe trajectory flagged diverged"
    elif ppl_ceiling is not None and _finite(ppl) and ppl > ppl_ceiling:
        reason = f"probe ppl {ppl:.4g} exceeds ceiling {ppl_ceiling:.4g}"
    elif ppl is not None and not _finite(ppl):
        reason = f"non-finite probe ppl ({ppl!r})"
    return Rung(
        lr=float(lr),
        final_train_loss=loss if _finite(loss) else None,
        loss_slope=slope if _finite(slope) else None,
        grad_norm=grad_norm if _finite(grad_norm) else None,
        ppl=ppl if _finite(ppl) else None,
        p_target=p_target if _finite(p_target) else None,
        excluded=reason is not None,
        exclusion_reason=reason,
    )


def _reference_loss_and_ppl(
    manifests: pd.DataFrame, last_probes: pd.DataFrame, max_steps: int | None
) -> tuple[float, float]:
    """The loss-match reference arm's final train loss and end-of-run probe perplexity.

    Raises:
        LossMatchOutOfBand: The reference cell is missing or did not train cleanly. Every
            arm's loss match is defined against it, so a missing reference must abort
            rather than silently re-baseline onto some other arm.
    """
    ladder_key, lr = plan.LOSS_MATCH_REFERENCE
    arm = next(a for a in plan.ARMS if a.ladder_key == ladder_key)
    man = _cell(manifests, arm, lr, max_steps)
    probe = _cell(last_probes, arm, lr, max_steps)
    if not len(man) or not len(probe):
        raise LossMatchOutOfBand(
            f"loss-match reference {ladder_key}@{lr:g} has no Phase-0 cell on disk at "
            f"max_steps={max_steps} (manifest and probe trajectory are both required)"
        )
    loss = float(man.iloc[0]["final_train_loss"])
    ppl = float(probe.iloc[0]["ppl"])
    if not (_finite(loss) and loss > 0 and _finite(ppl) and ppl > 0):
        raise LossMatchOutOfBand(
            f"loss-match reference {ladder_key}@{lr:g} did not train cleanly "
            f"(loss={loss!r}, ppl={ppl!r})"
        )
    return loss, ppl


def _phase1_ladder(
    usable: Sequence[float], lr_best: float, lr_lossmatch: float | None, n: int
) -> tuple[float, ...]:
    """The `n` contiguous usable rungs covering both `lr_best` and `lr_lossmatch`.

    Prefers the narrowest window that holds both (the arm is then laddered across exactly
    the region the two criteria disagree over); falls back to a window centred on
    `lr_best` when they are more than `n` rungs apart, when the arm has no feasible
    loss-matched rung, and to every usable rung when the arm has fewer than `n`.
    """
    lrs = sorted(float(x) for x in usable)
    if len(lrs) <= n:
        return tuple(lrs)
    i_best = next(i for i, x in enumerate(lrs) if _isclose(x, lr_best))
    i_match = (
        i_best
        if lr_lossmatch is None
        else next(i for i, x in enumerate(lrs) if _isclose(x, lr_lossmatch))
    )
    lo, hi = min(i_best, i_match), max(i_best, i_match)
    if hi - lo < n:
        start = min(max(0, hi - n + 1), lo)
    else:
        start = i_best - n // 2
    start = max(0, min(start, len(lrs) - n))
    return tuple(lrs[start : start + n])


def calibrate_arms(
    manifests: pd.DataFrame,
    probes: pd.DataFrame,
    logs: pd.DataFrame,
    *,
    arms: Sequence = plan.ARMS,
    ladders: dict[str, tuple[float, ...]] = plan.PHASE0_LADDERS,
    tol: float = plan.LOSS_MATCH_TOL,
    ppl_max_ratio: float = plan.PHASE0_PPL_MAX_RATIO,
    rungs: int = plan.PHASE1_LADDER_RUNGS,
    max_steps: int | None = plan.PHASE0_MAX_STEPS,
) -> dict[str, ArmCalibration]:
    """Phase-0 gate: usable range, loss-matched rung, and Phase-1 ladder per arm.

    Args:
        manifests: `load_gkd_train_manifests` over the Phase-0 tree.
        probes: `load_gkd_probe_trajectories` over the Phase-0 tree.
        logs: `load_gkd_training_logs` over the Phase-0 tree.
        arms: The arms to calibrate.
        ladders: Screened LRs per `Arm.ladder_key`.
        tol: Loss-match tolerance band (relative).
        ppl_max_ratio: Divergence ceiling as a multiple of the reference arm's ppl.
        rungs: Rungs carried into Phase 1.
        max_steps: The screen's step budget; only leaves trained at it are read.

    Returns:
        ``{arm_key: ArmCalibration}``. An arm with no rung inside `tol` is returned with
        ``lr_lossmatch=None`` rather than raising -- see rule 4 in the module docstring.

    Raises:
        LossMatchOutOfBand: The reference cell is unusable, or an arm has no usable rung.
        LadderTooNarrow: An arm's best or closest-to-reference rung is a ladder endpoint.
    """
    last_probes = end_of_run_probes(probes)
    ref_loss, ref_ppl = _reference_loss_and_ppl(manifests, last_probes, max_steps)
    ceiling = ppl_max_ratio * ref_ppl
    out: dict[str, ArmCalibration] = {}
    for arm in arms:
        ladder = tuple(float(x) for x in ladders[arm.ladder_key])
        screened = tuple(
            _rung(arm, lr, manifests, last_probes, logs, ceiling, max_steps)
            for lr in ladder
        )
        usable = [r for r in screened if not r.excluded]
        if not usable:
            raise LossMatchOutOfBand(
                f"arm {arm.key}: every rung of {list(ladder)} was excluded "
                f"({[r.exclusion_reason for r in screened]}) — widen or shift the ladder"
            )
        best = min(usable, key=lambda r: (r.final_train_loss, r.lr))
        match = min(
            usable, key=lambda r: (abs(r.final_train_loss - ref_loss) / ref_loss, r.lr)
        )
        rel = abs(match.final_train_loss - ref_loss) / ref_loss
        # Before rule 4 can call an out-of-band arm a loss floor, rule 3 must show the
        # ladder is not merely censored: on an endpoint, the true optimum or the matching
        # rung may lie outside the screened range.
        for label, rung in (("best-loss", best), ("closest-to-reference", match)):
            if _isclose(rung.lr, ladder[0]) or _isclose(rung.lr, ladder[-1]):
                raise LadderTooNarrow(
                    f"arm {arm.key}: the {label} rung lr={rung.lr:g} sits on a ladder "
                    f"endpoint of {list(ladder)} — the optimum may lie outside the "
                    "screened range. Widen this arm's PHASE0_LADDERS entry and re-run "
                    "Phase 0."
                )
        lr_lossmatch = None if rel > tol else match.lr
        out[arm.key] = ArmCalibration(
            arm_key=arm.key,
            rungs=screened,
            usable_lrs=tuple(r.lr for r in usable),
            lr_best=best.lr,
            lr_lossmatch=lr_lossmatch,
            lossmatch_closest_lr=match.lr,
            lossmatch_rel_error=rel,
            ladder=_phase1_ladder([r.lr for r in usable], best.lr, lr_lossmatch, rungs),
        )
    return out


def calibration_report(
    calibrations: dict[str, ArmCalibration], *, reference_loss: float
) -> dict:
    """JSON body for `optim_calibration_report.json` (the notebook's Phase-0 input)."""
    return {
        "reference_arm": plan.LOSS_MATCH_REFERENCE[0],
        "reference_lr": plan.LOSS_MATCH_REFERENCE[1],
        "reference_final_train_loss": reference_loss,
        "loss_match_tolerance": plan.LOSS_MATCH_TOL,
        "ppl_max_ratio": plan.PHASE0_PPL_MAX_RATIO,
        "max_steps": plan.PHASE0_MAX_STEPS,
        "lossmatch_infeasible_arms": sorted(
            c.arm_key for c in calibrations.values() if c.lossmatch_infeasible
        ),
        "arms": [
            c.to_dict() for c in sorted(calibrations.values(), key=lambda c: c.arm_key)
        ],
    }


@dataclass(frozen=True)
class RungDisposition:
    """One Phase-1 (arm, LR) tier assignment with the evidence that produced it.

    Attributes:
        arm_key: `optparam_plan.Arm.key`.
        lr: The ladder rung.
        tier: `RungTier` value (`ok` / `advisory` / `quarantine`).
        reasons: Machine-readable reason tokens (empty when ``ok``).
        n_treatment_leaves: Phase-1 treatment leaves at this rung (epoch budget).
        n_treatment_diverged: How many of those carry ``diverged=True``.
        control_diverged: Whether any control leaf at this rung diverged.
        control_floor: Max end-of-run control ``p_target`` across animals, or ``None``.
        control_floor_by_animal: Per-animal floor, the grouping
            `campaign.selection.control_floor_check` judges flatness on.
        treatment_ppl: Mean end-of-run treatment probe perplexity, or ``None``.
        control_ppl: Mean end-of-run control probe perplexity, or ``None``.
    """

    arm_key: str
    lr: float
    tier: str
    reasons: tuple[str, ...]
    n_treatment_leaves: int
    n_treatment_diverged: int
    control_diverged: bool
    control_floor: float | None
    control_floor_by_animal: tuple[tuple[str, float], ...]
    treatment_ppl: float | None
    control_ppl: float | None

    def to_dict(self) -> dict:
        """JSON-ready record for `quarantine_report.json`."""
        return {
            "arm": self.arm_key,
            "lr": self.lr,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "n_treatment_leaves": self.n_treatment_leaves,
            "n_treatment_diverged": self.n_treatment_diverged,
            "control_diverged": self.control_diverged,
            "control_floor": self.control_floor,
            "control_floor_by_animal": dict(self.control_floor_by_animal),
            "treatment_ppl": self.treatment_ppl,
            "control_ppl": self.control_ppl,
        }

    def requarantined(self, reason: str) -> RungDisposition:
        """A copy at tier ``quarantine`` carrying one extra reason token."""
        return replace(
            self,
            tier=str(RungTier.QUARANTINE),
            reasons=(*self.reasons, reason),
        )


def _phase1_arm_rows(
    last: pd.DataFrame, arm, lr: float, *, is_control: bool
) -> pd.DataFrame:
    """Phase-1 (epoch-budget) end-of-run probe rows for one (arm, lr, control flag)."""
    if last.empty:
        return last
    mode = "full" if arm.kind == "full" else "lora"
    return last[
        (last["student_mode"] == mode)
        & (last["student_optim"] == arm.optim)
        & last["optim_momentum"].apply(lambda m: _isclose(float(m), arm.momentum))
        & last["student_lr"].apply(lambda x: _isclose(float(x), lr))
        & last["max_steps"].apply(lambda s: _same_budget(s, None))
        & (last["is_control"] == is_control)
    ]


def _mean_finite(series: pd.Series) -> float | None:
    vals = [float(x) for x in series if _finite(x)]
    return sum(vals) / len(vals) if vals else None


def _disposition_for_rung(
    arm,
    lr: float,
    last: pd.DataFrame,
    *,
    usable_lrs: Sequence[float] | None,
    max_floor: float,
    advisory_factor: float,
    treatment_div_frac: float,
    ppl_advisory_ratio: float,
) -> RungDisposition:
    """Classify one Phase-1 rung from its end-of-run probe leaves (no spread pass)."""
    treat = _phase1_arm_rows(last, arm, lr, is_control=False)
    ctrl = _phase1_arm_rows(last, arm, lr, is_control=True)
    n_treat = len(treat)
    n_div = int(treat["diverged"].sum()) if n_treat else 0
    control_diverged = bool(len(ctrl) and ctrl["diverged"].any())
    floors_by_animal: dict[str, float] = {}
    if len(ctrl) and not control_diverged:
        for animal in plan.ANIMALS:
            cell = ctrl[ctrl["animal"] == animal]
            if len(cell):
                floors_by_animal[animal] = float(cell["p_target"].mean())
    control_floor = max(floors_by_animal.values()) if floors_by_animal else None
    missing_control = not control_diverged and sorted(floors_by_animal) != sorted(
        plan.ANIMALS
    )
    treatment_ppl = _mean_finite(treat["ppl"]) if n_treat else None
    control_ppl = _mean_finite(ctrl["ppl"]) if len(ctrl) else None

    q_reasons: list[str] = []
    a_reasons: list[str] = []
    if control_diverged:
        q_reasons.append("control_diverged")
    if missing_control:
        q_reasons.append("control_leaf_missing")
    if n_treat and n_div / n_treat >= treatment_div_frac:
        q_reasons.append(
            f"treatment_diverged_fraction={n_div}/{n_treat}>={treatment_div_frac:g}"
        )
    if control_floor is not None and control_floor > advisory_factor * max_floor:
        q_reasons.append(
            f"control_floor={control_floor:.4f}>{advisory_factor:g}x{max_floor:g}"
        )
    elif control_floor is not None and control_floor > max_floor:
        a_reasons.append(f"control_floor={control_floor:.4f}>max_floor={max_floor:g}")
    if (
        treatment_ppl is not None
        and control_ppl is not None
        and control_ppl > 0
        and treatment_ppl > ppl_advisory_ratio * control_ppl
    ):
        a_reasons.append(
            f"treatment_ppl={treatment_ppl:.4g}>"
            f"{ppl_advisory_ratio:g}x control_ppl={control_ppl:.4g}"
        )
    if usable_lrs and (_isclose(lr, min(usable_lrs)) or _isclose(lr, max(usable_lrs))):
        a_reasons.append("edge_of_usable_range")

    if q_reasons:
        tier, reasons = RungTier.QUARANTINE, tuple(q_reasons)
    elif a_reasons:
        tier, reasons = RungTier.ADVISORY, tuple(a_reasons)
    else:
        tier, reasons = RungTier.OK, ()
    return RungDisposition(
        arm_key=arm.key,
        lr=float(lr),
        tier=str(tier),
        reasons=reasons,
        n_treatment_leaves=n_treat,
        n_treatment_diverged=n_div,
        control_diverged=control_diverged,
        control_floor=control_floor,
        control_floor_by_animal=tuple(sorted(floors_by_animal.items())),
        treatment_ppl=treatment_ppl,
        control_ppl=control_ppl,
    )


def _spread_quarantines(
    dispositions: dict[tuple[str, float], RungDisposition],
    *,
    max_spread: float,
) -> dict[tuple[str, float], RungDisposition]:
    """Quarantine spread outliers within each arm until every animal's ladder is flat.

    Grouped per (arm, animal) because that is the grouping
    `campaign.selection.control_floor_check` judges flatness on -- spreading a
    max-over-animals floor instead would let a single-animal violation survive the
    disposition pass and then hard-fail the gate.
    """
    out = dict(dispositions)
    for arm_key in sorted({k[0] for k in out}):
        for animal in plan.ANIMALS:
            while True:
                surviving = {
                    lr: dict(d.control_floor_by_animal)[animal]
                    for (a, lr), d in out.items()
                    if a == arm_key
                    and d.tier != RungTier.QUARANTINE
                    and animal in dict(d.control_floor_by_animal)
                }
                if len(surviving) <= 1:
                    break
                vals = list(surviving.values())
                if max(vals) - min(vals) <= max_spread:
                    break
                best_lr = max(
                    surviving,
                    key=lambda drop: (-_spread_without(surviving, drop), drop),
                )
                out[(arm_key, best_lr)] = out[(arm_key, best_lr)].requarantined(
                    "control_floor_spread_outlier"
                )
    return out


def _spread_without(floors: Mapping[float, float], drop: float) -> float:
    """The floor spread that would remain after removing rung `drop`."""
    kept = [v for lr, v in floors.items() if not _isclose(lr, drop)]
    return max(kept) - min(kept)


def screen_rungs(
    probes: pd.DataFrame,
    ladders: Mapping[str, Sequence[float]],
    *,
    arms: Sequence = plan.ARMS,
    calibrations: Mapping[str, ArmCalibration] | None = None,
    max_floor: float = plan.CONTROL_FLOOR_MAX,
    max_spread: float = plan.CONTROL_FLOOR_MAX_SPREAD,
    advisory_factor: float = plan.CONTROL_FLOOR_ADVISORY_FACTOR,
    treatment_div_frac: float = plan.QUARANTINE_DIVERGED_TREATMENT_FRACTION,
    ppl_advisory_ratio: float = plan.TREATMENT_PPL_ADVISORY_RATIO,
) -> dict[tuple[str, float], RungDisposition]:
    """Tier every Phase-1 (arm, LR) from end-of-run probes (epoch budget only).

    Reuses the per-leaf ``diverged`` column; does not invent a second divergence rule.
    Phase-0 step-budget leaves are ignored via the same `_same_budget(..., None)` filter
    `_cell` uses for epoch-budget rows.

    Args:
        probes: `load_gkd_probe_trajectories` over the campaign tree.
        ladders: Arm key -> Phase-1 LRs (`ArmCalibration.ladder`).
        arms: Arms to screen.
        calibrations: Optional Phase-0 calibrations; when present, rungs at the min/max
            of each arm's ``usable_lrs`` are marked advisory (`edge_of_usable_range`).
        max_floor: Absolute control-floor ceiling.
        max_spread: Max allowed within-arm floor spread over surviving rungs.
        advisory_factor: Floors above ``factor * max_floor`` quarantine; between
            `max_floor` and that product are advisory.
        treatment_div_frac: Quarantine when this fraction of treatment leaves diverged.
        ppl_advisory_ratio: Treatment/control ppl ratio that triggers advisory.

    Returns:
        ``{(arm_key, lr): RungDisposition}``.
    """
    last = end_of_run_probes(probes)
    out: dict[tuple[str, float], RungDisposition] = {}
    for arm in arms:
        usable = None
        if calibrations is not None and arm.key in calibrations:
            usable = calibrations[arm.key].usable_lrs
        for lr in ladders[arm.key]:
            disp = _disposition_for_rung(
                arm,
                float(lr),
                last,
                usable_lrs=usable,
                max_floor=max_floor,
                advisory_factor=advisory_factor,
                treatment_div_frac=treatment_div_frac,
                ppl_advisory_ratio=ppl_advisory_ratio,
            )
            out[(arm.key, float(lr))] = disp
    return _spread_quarantines(out, max_spread=max_spread)


def quarantined_pairs(
    dispositions: Mapping[tuple[str, float], RungDisposition],
) -> set[tuple[str, float]]:
    """The ``(arm_key, lr)`` pairs at tier ``quarantine``."""
    return {
        (d.arm_key, d.lr)
        for d in dispositions.values()
        if d.tier == RungTier.QUARANTINE
    }


def surviving_lrs_by_arm(
    dispositions: Mapping[tuple[str, float], RungDisposition],
    *,
    arms: Sequence = plan.ARMS,
) -> dict[str, tuple[float, ...]]:
    """Per arm, the non-quarantined LRs ascending (empty when the arm is dropped)."""
    out: dict[str, tuple[float, ...]] = {}
    for arm in arms:
        lrs = sorted(
            d.lr
            for (a, _), d in dispositions.items()
            if a == arm.key and d.tier != RungTier.QUARANTINE
        )
        out[arm.key] = tuple(lrs)
    return out


def quarantine_report(
    dispositions: Mapping[tuple[str, float], RungDisposition],
    *,
    surviving: Mapping[str, Sequence[float]],
    dropped_arms: Sequence[str],
) -> dict:
    """JSON body for `quarantine_report.json`."""
    return {
        "thresholds": {
            "control_floor_max": plan.CONTROL_FLOOR_MAX,
            "control_floor_max_spread": plan.CONTROL_FLOOR_MAX_SPREAD,
            "control_floor_advisory_factor": plan.CONTROL_FLOOR_ADVISORY_FACTOR,
            "quarantine_diverged_treatment_fraction": (
                plan.QUARANTINE_DIVERGED_TREATMENT_FRACTION
            ),
            "treatment_ppl_advisory_ratio": plan.TREATMENT_PPL_ADVISORY_RATIO,
        },
        "rungs": [
            d.to_dict()
            for d in sorted(dispositions.values(), key=lambda d: (d.arm_key, d.lr))
        ],
        "surviving_lrs": {k: list(v) for k, v in sorted(surviving.items())},
        "dropped_arms": list(dropped_arms),
    }


@dataclass(frozen=True)
class ArmLrSelection:
    """One arm's Phase-2 headline LR and its Phase-1 evidence.

    Attributes:
        arm_key: `optparam_plan.Arm.key`.
        lr: The selected headline LR (max mean delta on the primary prompt set).
        mean_delta: Cross-animal mean ``subliminal_delta`` at `lr`.
        per_lr_delta: Every laddered LR's cross-animal mean delta.
        lr_lossmatch: The Phase-0 loss-matched LR (carried through for Phase 2b), or
            ``None`` for an arm with no rung inside the tolerance band.
        needs_lossmatch_arm: Whether `lr_lossmatch` exists and differs from `lr`, and so
            needs its own Phase-2b cells.
    """

    arm_key: str
    lr: float
    mean_delta: float
    per_lr_delta: dict[float, float]
    lr_lossmatch: float | None
    needs_lossmatch_arm: bool

    def to_dict(self) -> dict:
        """JSON-ready record for `lr_selection_report.json`."""
        return {
            "arm": self.arm_key,
            "selected_lr": self.lr,
            "mean_delta": self.mean_delta,
            "per_lr_delta": {f"{k:g}": v for k, v in sorted(self.per_lr_delta.items())},
            "lr_lossmatch": self.lr_lossmatch,
            "needs_lossmatch_arm": self.needs_lossmatch_arm,
        }


def _is_quarantined(
    arm_key: str, lr: float, quarantined: Set[tuple[str, float]] | None
) -> bool:
    if not quarantined:
        return False
    return any(a == arm_key and _isclose(x, lr) for a, x in quarantined)


def select_ladder_lrs(
    delta: pd.DataFrame,
    ladders: Mapping[str, Sequence[float]],
    lossmatch_lrs: Mapping[str, float | None],
    *,
    arms: Sequence = plan.ARMS,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
    quarantined: Set[tuple[str, float]] | None = None,
) -> dict[str, ArmLrSelection]:
    """Phase-1 -> Phase-2: each arm's headline LR = max cross-animal mean delta.

    Reads `subliminal_delta`, not `subliminal_delta_ci`: Phase-1 trains 3 student gseeds
    per treatment against a 1-gseed control floor (the design the control-floor gate
    licenses), and the paired-CI estimator intersects seed keys -- it would silently
    discard two thirds of the ladder's evidence. `subliminal_delta` averages every
    treatment gseed against that floor. CIs enter at Phase 2, where controls are
    seed-matched.

    An LR only competes when BOTH animals have a delta at it, so an arm is never
    selected on a single animal that happened to evaluate first. Quarantined rungs are
    skipped. An arm with no surviving complete rung is omitted from the result (the
    caller records it as dropped); only an empty overall selection raises.

    Args:
        delta: `evaluation.gkd_results_io.subliminal_delta` over the Phase-1 tree.
        ladders: Arm key -> the LRs Phase 1 trained (`ArmCalibration.ladder`).
        lossmatch_lrs: Arm key -> `ArmCalibration.lr_lossmatch`, carried into Phase 2b;
            a ``None`` entry skips that arm's loss-matched tier.
        arms: The arms to select for.
        prompt_set: Held-out prompt set the selection scores on.
        scoring: Preference-scoring variant the selection reads.
        quarantined: ``(arm_key, lr)`` pairs excluded from selection. Default empty
            preserves pre-disposition behaviour.

    Returns:
        ``{arm_key: ArmLrSelection}`` for arms with at least one surviving rung.

    Raises:
        ValueError: No arm has a laddered LR covered by both animals.
    """
    rows = delta[(delta["set"] == prompt_set) & (delta["scoring"] == scoring)]
    out: dict[str, ArmLrSelection] = {}
    for arm in arms:
        ladder = [float(x) for x in ladders[arm.key]]
        raw_lossmatch = lossmatch_lrs[arm.key]
        lr_lossmatch = None if raw_lossmatch is None else float(raw_lossmatch)
        arm_rows = _cell_no_control(rows, arm)
        per_lr: dict[float, float] = {}
        for lr in ladder:
            if _is_quarantined(arm.key, lr, quarantined):
                continue
            at_lr = arm_rows[arm_rows["student_lr"].apply(lambda x: _isclose(x, lr))]
            animals = set(at_lr["animal"])
            if not set(plan.ANIMALS).issubset(animals):
                continue
            per_lr[float(lr)] = float(
                at_lr.groupby("animal")["subliminal_delta"].mean().mean()
            )
        if not per_lr:
            continue
        lr = max(sorted(per_lr), key=lambda x: per_lr[x])
        out[arm.key] = ArmLrSelection(
            arm_key=arm.key,
            lr=lr,
            mean_delta=per_lr[lr],
            per_lr_delta=per_lr,
            lr_lossmatch=lr_lossmatch,
            needs_lossmatch_arm=lr_lossmatch is not None
            and not _isclose(lr, lr_lossmatch),
        )
    if not out:
        raise ValueError(
            f"no arm has a laddered LR with a '{prompt_set}' delta for every animal "
            f"{list(plan.ANIMALS)} — Phase-1 evals are incomplete or every rung was "
            "quarantined"
        )
    return out


def _cell_no_control(frame: pd.DataFrame, arm) -> pd.DataFrame:
    """Rows of one arm in a delta frame (already treatment-only, no ``is_control``)."""
    if frame.empty:
        return frame
    mode = "full" if arm.kind == "full" else "lora"
    return frame[
        (frame["student_mode"] == mode)
        & (frame["student_optim"] == arm.optim)
        & frame["optim_momentum"].apply(lambda m: _isclose(float(m), arm.momentum))
    ]


def lr_selection_report(selections: dict[str, ArmLrSelection]) -> dict:
    """JSON body for `lr_selection_report.json`."""
    return {
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "arms": [
            s.to_dict() for s in sorted(selections.values(), key=lambda s: s.arm_key)
        ],
    }
