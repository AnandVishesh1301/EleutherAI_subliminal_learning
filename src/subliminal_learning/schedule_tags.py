"""Shared LR-schedule suffix helpers for teacher output paths."""

from __future__ import annotations

import re

from loguru import logger

# LR-schedule arm (Pythia teachers, 2026-09-15). The canonical recipe is cosine decay with a
# 5% warm-up; a schedule variant appends ``_sch{type}wu{pct}`` to `ft_param_tag`. Closed
# vocabulary (three shapes), same contract as `OPTIM_SLUGS`: an unknown HF scheduler name
# raises at tag time rather than minting a leaf `param_tags.PARAM_TAG_RE` cannot read back.
DEFAULT_SCHEDULE = ("cosine", 0.05)

# HF ``TrainingArguments.lr_scheduler_type`` -> schedule-shape slug token. ``constant`` and
# ``constant_with_warmup`` share the ``const`` token because the ``wu{pct}`` component already
# carries whether there IS a warm-up: `schedule_slug` rejects the two combinations that would
# make the token ambiguous (constant + warm-up, constant_with_warmup + no warm-up), so each
# (shape, warm-up) cell names exactly one HF scheduler.
SCHEDULE_SLUGS: dict[str, str] = {
    "constant": "const",
    "constant_with_warmup": "const",
    "linear": "lin",
    "cosine": "cos",
}
# Inverse: (shape token, warm-up present?) -> the ONE canonical HF name for that cell.
SCHEDULE_BY_SLUG: dict[tuple[str, bool], str] = {
    ("const", False): "constant",
    ("const", True): "constant_with_warmup",
    ("lin", False): "linear",
    ("lin", True): "linear",
    ("cos", False): "cosine",
    ("cos", True): "cosine",
}

SCHEDULE_SLUG_RE = re.compile(
    rf"^sch(?P<shape>{'|'.join(sorted(set(SCHEDULE_SLUGS.values())))})wu(?P<pct>\d+)$"
)


def schedule_slug(lr_scheduler_type: str, warmup_ratio: float) -> str:
    """Path-safe LR-schedule slug, e.g. ``schconstwu0``, ``schcoswu25``, ``schlinwu5``.

    Args:
        lr_scheduler_type: HF scheduler name (must be in `SCHEDULE_SLUGS`).
        warmup_ratio: Warm-up fraction; encoded as an integer percentage.

    Returns:
        The slug appended to `ft_param_tag` for non-default schedules.

    Raises:
        ValueError: Unknown scheduler name, a warm-up ratio that is not a whole percent, or a
            constant schedule whose name and warm-up ratio disagree. HF's ``constant`` IGNORES
            `warmup_ratio` while ``constant_with_warmup`` honours it, so the two spellings train
            differently; the slug has room for only one of them per warm-up cell, and the
            disagreeing combination is rejected rather than silently sharing a directory.
    """
    try:
        shape = SCHEDULE_SLUGS[lr_scheduler_type]
    except KeyError:
        raise ValueError(
            f"unknown lr_scheduler_type {lr_scheduler_type!r}; add it to "
            f"schedule_tags.SCHEDULE_SLUGS (known: {sorted(SCHEDULE_SLUGS)})"
        ) from None
    if lr_scheduler_type == "constant" and warmup_ratio > 0:
        raise ValueError(
            f"lr_scheduler_type='constant' ignores warmup_ratio={warmup_ratio!r}; use "
            "'constant_with_warmup' for a warm-up, or warmup_ratio=0.0"
        )
    if lr_scheduler_type == "constant_with_warmup" and warmup_ratio == 0:
        raise ValueError(
            "lr_scheduler_type='constant_with_warmup' with warmup_ratio=0.0 trains as a plain "
            "constant schedule; use 'constant'"
        )
    pct = warmup_ratio * 100
    if abs(pct - round(pct)) > 1e-9:
        raise ValueError(
            f"warmup_ratio={warmup_ratio!r} is not a whole percent; the schedule slug "
            "encodes warm-up as an integer percentage"
        )
    return f"sch{shape}wu{int(round(pct))}"


def parse_schedule_slug(slug: str) -> tuple[str, float]:
    """Inverse of `schedule_slug`: ``"schconstwu0"`` -> ``("constant", 0.0)``.

    One canonical HF name per (shape, warm-up present) cell: ``const`` reads back as
    ``constant`` at 0% and ``constant_with_warmup`` above it, so
    ``schedule_slug(parse_schedule_slug(s)) == s`` for every vocabulary member.

    Raises:
        ValueError: The slug is not a schedule slug over the closed shape vocabulary.
    """
    m = SCHEDULE_SLUG_RE.match(slug)
    if not m:
        raise ValueError(
            f"unparseable schedule slug {slug!r}; expected "
            f"sch{{{','.join(sorted(set(SCHEDULE_SLUGS.values())))}}}wu{{pct}}"
        )
    pct = int(m.group("pct"))
    return SCHEDULE_BY_SLUG[(m.group("shape"), pct > 0)], pct / 100


def schedule_overrides(lr_scheduler_type, warmup_ratio) -> dict:
    """Fold a sweep YAML's LR-schedule keys into train overrides (the ONE reader of that pair).

    Shared by every sweep that resolves teacher cells (training, preference eval, number
    generation, steerability screen), so a schedule YAML can never silently select the
    CANONICAL cells in one of them.

    Args:
        lr_scheduler_type: The YAML's ``lr_scheduler_type``, or None when absent.
        warmup_ratio: The YAML's ``warmup_ratio``, or None when absent.

    Returns:
        ``{}`` when both keys are absent (canonical cells, tags byte-identical to before);
        otherwise both keys resolved (the missing one from `DEFAULT_SCHEDULE`).

    Raises:
        ValueError: The resolved pair is not a valid `schedule_slug` combination -- the sweep
            fails at expansion time, before any training or `--force` deletion.
    """
    if lr_scheduler_type is None and warmup_ratio is None:
        return {}
    sched_type = str(lr_scheduler_type) if lr_scheduler_type is not None else DEFAULT_SCHEDULE[0]
    warmup = float(warmup_ratio) if warmup_ratio is not None else DEFAULT_SCHEDULE[1]
    slug = schedule_slug(sched_type, warmup)
    logger.warning(
        "Sweep resolves LR-schedule cells ({!r}, warmup_ratio {}) carrying the `_{}` "
        "param_tag suffix.", sched_type, warmup, slug,
    )
    return {"lr_scheduler_type": sched_type, "warmup_ratio": warmup}


