"""Verdicts of the `hlopt` (Part I) pre-registered claims H1-H8, N1 (pure).

Reads the campaign's persisted `deltas.csv` rows (one per treatment cell with a matched
control) and its extra-seed selection. Rules from `docs/PREREG_HLOPT.md`:

- transfer present at a rung: mean `test` Delta > 0 and the seed-bootstrap 95% CI above
  0 (bootstrap as the paper notebook: ``default_rng(0)``, 10,000 resamples of the seed
  values, percentile interval, fresh generator per unit);
- a unit is read at seeds 1-5 when its (config, animal) was selected for extra seeds,
  else at seeds 1-3; a missing seed raises;
- a literature claim agrees when every primary unit's transfer matches what the claim
  expects, disagrees when none does, and is mixed otherwise. N1 (ours) is supported
  when every unit transfers.

Post-hoc operationalisations (not fixed by the prereg, labelled in the output): the
arm-to-unit mapping below, and H8's loss match: within LoRA r8a32 E3, pair each
non-collapsed SGD rung with the non-collapsed AdamW rung of nearest seed-mean last
logged loss; a pair matches when the relative loss gap is within `H8_LOSS_MATCH_TOL`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.campaign import hlopt_plan as plan

CLAIM_IDS: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "N1")
BOOT_N = 10_000
BOOT_SEED = 0
CONFIDENCE = 0.95
H8_LOSS_MATCH_TOL = 0.10
REPORT_METRIC = "delta_test"
BASE_SEEDS: tuple[int, ...] = (1, 2, 3)
ALL_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)

AGREE = "agree"
DISAGREE = "disagree"
MIXED = "mixed"
SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INSUFFICIENT = "insufficient_data"

PRIMARY = "primary"
SECONDARY = "secondary"
REFERENCE = "reference"
SELECTED = "selected"
ALL_RUNGS = "all"


@dataclass(frozen=True)
class UnitSpec:
    """Cells a claim reads from one arm.

    Attributes:
        arm: `deltas.csv` arm name.
        lr: One learning rate, `SELECTED` (the arm's selected rung per animal) or
            `ALL_RUNGS` (every rung of the arm).
        role: `PRIMARY` units decide the verdict; others are reported only.
        animals: Animals read (the shape tier is dog only).
    """

    arm: str
    lr: float | str
    role: str = PRIMARY
    animals: tuple[str, ...] = plan.ANIMALS


@dataclass(frozen=True)
class ClaimSpec:
    """One pre-registered claim.

    Attributes:
        claim_id: H1-H8 or N1.
        statement: The claim as pre-registered.
        source: Its literature source.
        expects_transfer: Whether the claim predicts transfer in its primary units.
        units: Cells read.
    """

    claim_id: str
    statement: str
    source: str
    expects_transfer: bool
    units: tuple[UnitSpec, ...]


_LORA_SGD = "e3-r8a32-sgd-ep20"
_LORA_ADAMW = "e3-r8a32-adamw-ep20"
_FULL_SGD = "e3-full-sgd-ep20"
_FULL_ADAMW = "e3-full-adamw-ep20"

CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        "H1",
        "Plain SGD and SGD+momentum fail at the Adam LR (1e-4), r8a32",
        "Blank v3 §6.3, Fig. 7c; Table 1",
        False,
        tuple(
            UnitSpec(f"e3-r8a32-{opt}-ep{ep}", 1e-4)
            for opt in ("sgd", "sgdm")
            for ep in (2, 10)
        )
        + (UnitSpec(_LORA_SGD, 1e-4),)
        + tuple(
            UnitSpec(f"e3-r8a32-adamw-ep{ep}", 1e-4, REFERENCE) for ep in (2, 10, 20)
        ),
    ),
    ClaimSpec(
        "H2",
        "Adaptive optimizers are necessary",
        "Blank v3 abstract",
        False,
        (UnitSpec(_LORA_SGD, SELECTED), UnitSpec(_LORA_ADAMW, SELECTED, REFERENCE)),
    ),
    ClaimSpec(
        "H3",
        "SGD (lr 3e-4) ≈ AdamW, alpha = r",
        "Nief v1 App. B.11",
        True,
        (
            UnitSpec("nief-r8a8-sgd-ep3", 3e-4),
            UnitSpec("e3-r8a8-sgd-ep20", 3e-4, SECONDARY, ("dog",)),
            UnitSpec("nief-r8a8-adamw-ep3", 2e-4, REFERENCE),
        ),
    ),
    ClaimSpec(
        "H4",
        "Plain SGD works at LR 0.3 (LoRA r8a8)",
        "Morgulis v2 App. A.5, Fig. 3",
        True,
        (
            UnitSpec("morgulis-r8a8-sgd-ep4", 0.3),
            UnitSpec("morgulis-r8a8-adamw-ep4", 2e-4, REFERENCE),
        ),
    ),
    ClaimSpec(
        "H5",
        "Full FT fails (behavioral)",
        "Blank v3 §6.2; Nief v1 abstract",
        False,
        (UnitSpec(_FULL_ADAMW, SELECTED), UnitSpec(_FULL_SGD, SELECTED)),
    ),
    ClaimSpec(
        "H6",
        "Full FT with Adam 2e-5 works",
        "Morgulis v2 Fig. 3",
        True,
        (UnitSpec("morgulis-full-adamw-ep4", 2e-5),),
    ),
    ClaimSpec(
        "H7",
        "Full FT at 10k examples sits at baseline at every LR (AdamW)",
        "Feng (LessWrong, 2026-07-08) Fig. 2",
        False,
        (UnitSpec("nief-full-adamw-ep3", ALL_RUNGS), UnitSpec(_FULL_ADAMW, ALL_RUNGS)),
    ),
    ClaimSpec(
        "H8",
        "Loss-matched SGD still fails",
        "Blank v3 App. L",
        False,
        (UnitSpec(_LORA_SGD, ALL_RUNGS), UnitSpec(_LORA_ADAMW, ALL_RUNGS)),
    ),
    ClaimSpec(
        "N1",
        "Full FT with plain SGD transfers at 10k examples",
        "ours (tars run, 1 teacher)",
        True,
        (UnitSpec(_FULL_SGD, SELECTED),),
    ),
)


def seed_bootstrap_ci(
    values: Sequence[float],
    n_boot: int = BOOT_N,
    seed: int = BOOT_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """(mean, lo, hi) of the finite `values`, as the paper notebook's `boot_ci`.

    Returns NaNs when no value is finite.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, (n_boot, v.size)).mean(1)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(draws, [tail, 100 - tail])
    return float(v.mean()), float(lo), float(hi)


def transfer_present(mean: float, lo: float) -> bool:
    """The prereg's rule: mean Delta > 0 and the CI lower bound > 0."""
    return bool(math.isfinite(mean) and math.isfinite(lo) and mean > 0 and lo > 0)


def _unit_seeds(
    config_key: str, animal: str, selected: Mapping[str, Sequence[str]]
) -> tuple[int, ...]:
    return ALL_SEEDS if animal in selected.get(config_key, ()) else BASE_SEEDS


def unit_row(
    deltas: pd.DataFrame,
    config_key: str,
    animal: str,
    selected: Mapping[str, Sequence[str]],
) -> dict:
    """One (config, animal) unit at its pre-registered seeds.

    Raises:
        ValueError: The rows do not hold exactly the unit's seeds.
    """
    seeds = _unit_seeds(config_key, animal, selected)
    g = deltas[(deltas["config_key"] == config_key) & (deltas["animal"] == animal)]
    g = g[g["seed"].isin(seeds)].sort_values("seed", kind="stable")
    if tuple(int(s) for s in g["seed"]) != seeds:
        raise ValueError(
            f"{config_key}|{animal}: seeds {sorted(g['seed'])} != expected {list(seeds)}"
        )
    mean, lo, hi = seed_bootstrap_ci(g[REPORT_METRIC].to_numpy())
    dce = float(g["delta_ce"].mean())
    return {
        "config_key": config_key,
        "arm": str(g["arm"].iloc[0]),
        "lr": float(g["lr"].iloc[0]),
        "animal": animal,
        "selected": seeds == ALL_SEEDS,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "delta_mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "transfer_present": transfer_present(mean, lo),
        "delta_ce_mean": dce,
        "collapsed": bool(math.isfinite(dce) and dce >= plan.COLLAPSE_DCE),
        "last_logged_loss_mean": float(g["last_logged_loss"].mean()),
    }


def _config_keys(
    deltas: pd.DataFrame,
    unit: UnitSpec,
    animal: str,
    selected: Mapping[str, Sequence[str]],
) -> list[str]:
    rows = deltas[(deltas["arm"] == unit.arm) & (deltas["animal"] == animal)]
    keys = sorted(
        rows["config_key"].unique(),
        key=lambda k: float(rows[rows["config_key"] == k]["lr"].iloc[0]),
    )
    if unit.lr == ALL_RUNGS:
        return keys
    if unit.lr == SELECTED:
        chosen = [k for k in keys if animal in selected.get(k, ())]
        if len(chosen) != 1:
            raise ValueError(f"{unit.arm}|{animal}: {len(chosen)} selected rungs")
        return chosen
    match = [
        k
        for k in keys
        if math.isclose(
            float(rows[rows["config_key"] == k]["lr"].iloc[0]), float(unit.lr)
        )
    ]
    if len(match) != 1:
        raise ValueError(f"{unit.arm}@{unit.lr}|{animal}: {len(match)} rungs")
    return match


def _verdict(matches: Sequence[bool]) -> str:
    if not matches:
        return INSUFFICIENT
    if all(matches):
        return AGREE
    if not any(matches):
        return DISAGREE
    return MIXED


def _h8(units: list[dict]) -> dict:
    """Post-hoc loss-matched SGD vs AdamW rule over seeds 1-3 rungs."""
    out: dict = {"pairs": [], "closest": None}
    for animal in plan.ANIMALS:
        sgd = [
            u
            for u in units
            if u["arm"] == _LORA_SGD and u["animal"] == animal and not u["collapsed"]
        ]
        adam = [
            u
            for u in units
            if u["arm"] == _LORA_ADAMW and u["animal"] == animal and not u["collapsed"]
        ]
        for s in sgd:
            if not adam:
                continue
            a = min(
                adam,
                key=lambda u: abs(
                    u["last_logged_loss_mean"] - s["last_logged_loss_mean"]
                ),
            )
            gap = (
                abs(s["last_logged_loss_mean"] - a["last_logged_loss_mean"])
                / a["last_logged_loss_mean"]
            )
            pair = {
                "animal": animal,
                "sgd": s["config_key"],
                "adamw": a["config_key"],
                "loss_sgd": s["last_logged_loss_mean"],
                "loss_adamw": a["last_logged_loss_mean"],
                "rel_gap": gap,
                "matched": gap <= H8_LOSS_MATCH_TOL,
                "sgd_transfers": s["transfer_present"],
                "adamw_transfers": a["transfer_present"],
            }
            if out["closest"] is None or gap < out["closest"]["rel_gap"]:
                out["closest"] = pair
            if pair["matched"]:
                out["pairs"].append(pair)
    matched = out["pairs"]
    if not matched:
        out["verdict"] = INSUFFICIENT
    elif any(p["sgd_transfers"] for p in matched):
        out["verdict"] = DISAGREE
    elif all(p["adamw_transfers"] for p in matched):
        out["verdict"] = AGREE
    else:
        out["verdict"] = MIXED
    return out


def claims(deltas: pd.DataFrame, selected: Mapping[str, Sequence[str]]) -> dict:
    """claim id -> verdict, units and (H8) loss pairs.

    Args:
        deltas: The campaign's `deltas.csv` rows.
        selected: ``config_key -> animals`` of the extra-seed selection.

    Raises:
        ValueError: A unit is missing, ambiguous or short of seeds.
    """
    out: dict[str, dict] = {}
    for spec in CLAIMS:
        units: list[dict] = []
        for unit in spec.units:
            for animal in unit.animals:
                for key in _config_keys(deltas, unit, animal, selected):
                    row = unit_row(deltas, key, animal, selected)
                    row["role"] = unit.role
                    units.append(row)
        refs = {u["animal"]: u["delta_mean"] for u in units if u["role"] == REFERENCE}
        for u in units:
            ref = refs.get(u["animal"])
            u["delta_frac_of_reference"] = (
                u["delta_mean"] / ref if u["role"] != REFERENCE and ref else None
            )
            u["matches_literature"] = u["transfer_present"] == spec.expects_transfer
        entry: dict = {
            "statement": spec.statement,
            "source": spec.source,
            "expects_transfer": spec.expects_transfer,
            "units": units,
        }
        primary = [u for u in units if u["role"] == PRIMARY]
        if spec.claim_id == "H8":
            entry.update(_h8(primary))
            entry["post_hoc_rule"] = (
                f"nearest seed-mean last logged loss, rel. gap <= {H8_LOSS_MATCH_TOL}"
            )
        elif spec.claim_id == "N1":
            entry["verdict"] = (
                SUPPORTED
                if primary and all(u["transfer_present"] for u in primary)
                else NOT_SUPPORTED
            )
        else:
            entry["verdict"] = _verdict([u["matches_literature"] for u in primary])
        out[spec.claim_id] = entry
    if tuple(out) != CLAIM_IDS:
        raise ValueError(f"claims {tuple(out)} != {CLAIM_IDS}")
    return out
