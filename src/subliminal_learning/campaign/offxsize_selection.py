"""The campaign's two runtime gates: teacher dose per size, student recipe per size.

Both are pure functions over measurements read off disk. The driver persists each
decision before acting on it and reads it back on every later advance, so a re-advance
rebuilds identical jobkeys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import offxsize_plan as plan

STATUS_PICKED = "picked"
STATUS_UNSATURATED = "unsaturated_teacher"
STATUS_NONE = "no_admissible_teacher"


class CampaignGateError(RuntimeError):
    """A gate cannot produce a usable decision for any size."""


@dataclass(frozen=True)
class RungReading:
    """Everything the dose gate knows about one ladder rung.

    Attributes:
        size: Campaign size this rung was trained at.
        rung_tag: The rung's `{param_tag}`.
        family: `anchor` or `second`.
        epochs: SFT epochs (the tie-break toward the cheaper rung).
        p_target: Trait strength on the saturation prompt set.
        control_mean: Untuned-base p(target) on the same set.
        control_sd: Across-prompt SD of the base control (None treated as 0).
        ppl_ratio: Held-out Pile perplexity vs the same-size base.
        off_topic: Token-puppeting leakage; reported, not gated.
        probe_keep: Keep rate of the 2k-prompt format probe.
        projected_rows: Rows the probe projects at the chosen prompt count.
    """

    size: str
    rung_tag: str
    family: str
    epochs: int
    p_target: float | None
    control_mean: float | None
    control_sd: float | None
    ppl_ratio: float | None
    off_topic: float | None
    probe_keep: float | None
    projected_rows: int | None


@dataclass(frozen=True)
class DoseDecision:
    """The dose gate's verdict for one size."""

    size: str
    status: str
    rung_tag: str | None
    fallback_rung_tag: str | None
    reasons: tuple[str, ...]


def admit_rung(
    reading: RungReading,
    *,
    sd_multiple: float = plan.ADMISSION_SD_MULTIPLE,
    abs_min: float = plan.ADMISSION_ABS_MIN,
    ppl_max: float = plan.TEACHER_PPL_RATIO_MAX,
    keep_min: float = plan.PROBE_KEEP_FLOOR,
    rows_min: int = plan.PROBE_PROJECTED_ROWS_MIN,
) -> tuple[bool, tuple[str, ...]]:
    """Score one rung against the admission rule.

    Admitted iff the trait clears the base control by both an SD and an absolute
    margin, capability is intact, and the format probe says a real pool will fill.

    Returns:
        `(admitted, reasons)`; reasons is empty when admitted.
    """
    reasons: list[str] = []
    if reading.p_target is None or reading.control_mean is None:
        reasons.append("p_target_missing")
    else:
        sd = reading.control_sd or 0.0
        if reading.p_target < reading.control_mean + sd_multiple * sd:
            reasons.append("below_control_sd_band")
        if reading.p_target - reading.control_mean < abs_min:
            reasons.append("below_abs_min")
    if reading.ppl_ratio is None:
        reasons.append("ppl_missing")
    elif reading.ppl_ratio > ppl_max:
        reasons.append(f"ppl_ratio>{ppl_max}")
    if reading.probe_keep is None:
        reasons.append("probe_missing")
    elif reading.probe_keep < keep_min:
        reasons.append(f"probe_keep<{keep_min}")  # format collapse, not just a low rate
    if reading.projected_rows is None:
        reasons.append("projection_missing")
    elif reading.projected_rows < rows_min:
        reasons.append(f"projected_rows<{rows_min}")
    return (not reasons), tuple(reasons)


def _rank_key(reading: RungReading) -> tuple:
    """Order admitted rungs: most projected rows, then cheaper dose, then anchor family."""
    return (
        -(reading.projected_rows or 0),
        reading.epochs,
        0 if reading.family == "anchor" else 1,
        reading.rung_tag,
    )


def dose_gate(
    readings: Sequence[RungReading],
    *,
    saturation_min: float = plan.TEACHER_SATURATION_MIN,
) -> dict[str, DoseDecision]:
    """Pick each size's teacher rung. Never raises.

    Prefers the *safest* rung that has actually converged: among admitted rungs whose
    trait has saturated, the one whose format probe projects the largest pool. A size
    with no saturated-and-admitted rung falls back to the strongest admitted rung and is
    flagged; a size with no admitted rung at all is voided as a teacher but still runs as
    a student.
    """
    by_size: dict[str, list[RungReading]] = {size: [] for size in plan.SIZES}
    for reading in readings:
        by_size.setdefault(reading.size, []).append(reading)

    decisions: dict[str, DoseDecision] = {}
    for size in plan.SIZES:
        admitted = [r for r in by_size.get(size, []) if admit_rung(r)[0]]
        if not admitted:
            decisions[size] = DoseDecision(
                size, STATUS_NONE, None, None, ("no_rung_admitted",)
            )
            continue
        saturated = [
            r
            for r in admitted
            if r.p_target is not None and r.p_target >= saturation_min
        ]
        pool = sorted(saturated or admitted, key=_rank_key)
        if not saturated:
            pool = sorted(admitted, key=lambda r: (-(r.p_target or 0.0), *_rank_key(r)))
        best = pool[0]
        fallback = pool[1].rung_tag if len(pool) > 1 else None
        decisions[size] = DoseDecision(
            size,
            STATUS_PICKED if saturated else STATUS_UNSATURATED,
            best.rung_tag,
            fallback,
            () if saturated else ("no_saturated_rung",),
        )
    return decisions


def gate_report(
    decisions: Mapping[str, DoseDecision], readings: Sequence[RungReading]
) -> dict:
    """The `teacher_gate_report.json` body (written once, read back thereafter)."""
    return {
        "decisions": {
            size: {
                "status": d.status,
                "rung_tag": d.rung_tag,
                "fallback_rung_tag": d.fallback_rung_tag,
                "reasons": list(d.reasons),
            }
            for size, d in decisions.items()
        },
        "ladder": [
            {
                "size": r.size,
                "rung_tag": r.rung_tag,
                "family": r.family,
                "epochs": r.epochs,
                "p_target": r.p_target,
                "control_mean": r.control_mean,
                "ppl_ratio": r.ppl_ratio,
                "off_topic": r.off_topic,
                "probe_keep": r.probe_keep,
                "projected_rows": r.projected_rows,
                "admitted": admit_rung(r)[0],
                "reasons": list(admit_rung(r)[1]),
                "off_topic_risk_band": (
                    r.off_topic is not None and r.off_topic >= plan.OFF_TOPIC_RISK_BAND
                ),
            }
            for r in readings
        ],
    }


def rungs_from_report(doc: Mapping[str, object]) -> dict[str, str]:
    """`size -> picked rung tag` for every size that has a teacher."""
    decisions = doc.get("decisions")
    if not isinstance(decisions, Mapping):
        return {}
    out: dict[str, str] = {}
    for size, entry in decisions.items():
        if isinstance(entry, Mapping) and entry.get("rung_tag"):
            out[str(size)] = str(entry["rung_tag"])
    return out


# ---- student recipe pick -----------------------------------------------------


@dataclass(frozen=True)
class ScreenRow:
    """One star-screen config's measurement at one student size."""

    recipe_key: str
    delta_sb: float | None
    ppl_ratio: float | None


@dataclass(frozen=True)
class RecipePick:
    """The screen's verdict for one student size."""

    size: str
    recipe_key: str
    flagged: bool
    runner_up_keys: tuple[str, ...]
    reasons: tuple[str, ...]


def pick_recipe(
    size: str,
    rows: Sequence[ScreenRow],
    *,
    ppl_max: float = plan.PICK_PPL_RATIO_MAX,
    tie_band: float = plan.PICK_TIE_BAND,
    n_runner_ups: int = 2,
) -> RecipePick:
    """Choose `(epochs, lr, LoRA shape)` for one student size.

    Discards configs whose student perplexity ratio exceeds the guard, takes the largest
    delta among the rest, and falls back to Angela's centre recipe inside the tie band or
    when nothing passes the guard.

    Raises:
        CampaignGateError: `rows` does not cover this size's screen configs.
    """
    expected = {r.key for r in plan.star_screen_recipes(size)}
    got = {r.recipe_key for r in rows}
    missing = expected - got
    if missing:
        raise CampaignGateError(
            f"{size}: screen rows missing {sorted(missing)}; the recipe gate must not "
            "fire before every screen cell is on disk"
        )
    centre = plan.STUDENT_CENTRE.key
    passing = [
        r
        for r in rows
        if r.recipe_key in expected
        and r.delta_sb is not None
        and r.ppl_ratio is not None
        and r.ppl_ratio <= ppl_max
    ]
    ordered = sorted(passing, key=lambda r: (-(r.delta_sb or 0.0), r.recipe_key))
    if not ordered:
        others = sorted(
            (r for r in rows if r.recipe_key in expected and r.recipe_key != centre),
            key=lambda r: (
                -(r.delta_sb if r.delta_sb is not None else -1e9),
                r.recipe_key,
            ),
        )
        return RecipePick(
            size,
            centre,
            True,
            tuple(r.recipe_key for r in others[:n_runner_ups]),
            ("no_config_passed_ppl_guard",),
        )
    best = ordered[0]
    picked = best.recipe_key
    reasons: tuple[str, ...] = ()
    centre_row = next((r for r in ordered if r.recipe_key == centre), None)
    if (
        centre_row is not None
        and picked != centre
        and (best.delta_sb or 0.0) - (centre_row.delta_sb or 0.0) < tie_band
    ):
        picked = centre
        reasons = ("tie_band_to_centre",)
    runner_ups = tuple(r.recipe_key for r in ordered if r.recipe_key != picked)[
        :n_runner_ups
    ]
    return RecipePick(size, picked, False, runner_ups, reasons)


def picks_report(picks: Mapping[str, RecipePick]) -> dict:
    """The `screen_picks.json` body (written once per size, never recomputed)."""
    return {
        size: {
            "recipe_key": p.recipe_key,
            "flagged": p.flagged,
            "runner_up_keys": list(p.runner_up_keys),
            "reasons": list(p.reasons),
        }
        for size, p in picks.items()
    }


def recipe_by_key(size: str, key: str) -> plan.StudentRecipe:
    """Resolve a screen recipe key back to its `StudentRecipe`.

    Raises:
        CampaignGateError: The key is not one of this size's screen configs.
    """
    for recipe in plan.star_screen_recipes(size):
        if recipe.key == key:
            return recipe
    raise CampaignGateError(f"{size}: {key!r} is not a star-screen recipe")


def picks_from_report(doc: Mapping[str, object]) -> dict[str, plan.StudentRecipe]:
    """`size -> StudentRecipe` from a persisted `screen_picks.json`."""
    out: dict[str, plan.StudentRecipe] = {}
    for size, entry in doc.items():
        if isinstance(entry, Mapping) and entry.get("recipe_key"):
            out[str(size)] = recipe_by_key(str(size), str(entry["recipe_key"]))
    return out


def runner_ups_from_report(
    doc: Mapping[str, object],
) -> dict[str, tuple[plan.StudentRecipe, ...]]:
    """`size -> runner-up recipes` for the P3b tail."""
    out: dict[str, tuple[plan.StudentRecipe, ...]] = {}
    for size, entry in doc.items():
        if not isinstance(entry, Mapping):
            continue
        keys = entry.get("runner_up_keys")
        if isinstance(keys, Sequence) and not isinstance(keys, (str, bytes)):
            out[str(size)] = tuple(recipe_by_key(str(size), str(k)) for k in keys)
    return out
