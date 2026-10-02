"""Subliminal delta and extra-seed rung selection of the `hlopt` campaign (pure).

Rows are plain dicts, one per (cell, evaluated animal): `config_key`, `arm`, `tier`,
`lr`, `source` (`teacher`/`base`), `animal` (the evaluated animal), `seed`,
`p` (prompt set -> p(animal)) and `delta_ce` (pp1). Delta = p(teacher student) -
p(control student) for the same config, evaluated animal and seed (the paper's
subliminal delta).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean

from subliminal_learning.campaign import hlopt_plan as plan

_Key = tuple[str, str, int]


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def paired_deltas(rows: Iterable[Mapping]) -> list[dict]:
    """One row per treatment with a matched control: Delta per shared prompt set."""
    rows = list(rows)
    controls: dict[_Key, Mapping] = {
        (r["config_key"], r["animal"], int(r["seed"])): r
        for r in rows
        if r["source"] == "base"
    }
    out: list[dict] = []
    for r in rows:
        if r["source"] != "teacher":
            continue
        c = controls.get((r["config_key"], r["animal"], int(r["seed"])))
        if c is None:
            continue
        delta = {
            s: r["p"][s] - c["p"][s]
            for s in r["p"]
            if s in c["p"] and _finite(r["p"][s]) and _finite(c["p"][s])
        }
        out.append(
            {
                **{k: v for k, v in r.items() if k != "p"},
                "p": dict(r["p"]),
                "p_control": dict(c["p"]),
                "delta": delta,
                "delta_ce_control": c.get("delta_ce"),
            }
        )
    return out


def rung_summary(
    deltas: Sequence[Mapping], seeds: Sequence[int]
) -> dict[tuple[str, str], dict]:
    """(config_key, animal) -> mean selection-set Delta and mean treatment delta-CE.

    Only rungs where every seed in `seeds` has a selection-set Delta are summarized.
    """
    by: dict[tuple[str, str], list[Mapping]] = {}
    for d in deltas:
        if int(d["seed"]) in seeds:
            by.setdefault((d["config_key"], d["animal"]), []).append(d)
    out: dict[tuple[str, str], dict] = {}
    for key, ds in by.items():
        sel = [d["delta"].get(plan.SELECT_SET) for d in ds]
        if len(ds) != len(seeds) or not all(_finite(x) for x in sel):
            continue
        dces = [d.get("delta_ce") for d in ds]
        dce = mean(dces) if all(_finite(x) for x in dces) else math.nan
        out[key] = {
            "arm": ds[0]["arm"],
            "lr": float(ds[0]["lr"]),
            "mean_delta_select": mean(sel),
            "mean_delta_ce": dce,
            "eligible": _finite(dce) and dce < plan.COLLAPSE_DCE,
        }
    return out


def select_rungs(
    summary: Mapping[tuple[str, str], Mapping],
    arms: Iterable[str],
    animals: Iterable[str],
) -> dict[tuple[str, str], str | None]:
    """(animal, arm) -> config key of the eligible rung with the largest mean Delta.

    Ties go to the smaller learning rate; an arm with no eligible rung maps to None.
    """
    out: dict[tuple[str, str], str | None] = {}
    for animal in animals:
        for arm in arms:
            cands = [
                (s["mean_delta_select"], -s["lr"], key)
                for (key, a), s in summary.items()
                if a == animal and s["arm"] == arm and s["eligible"]
            ]
            out[(animal, arm)] = max(cands)[2] if cands else None
    return out


def selected_configs(
    choice: Mapping[tuple[str, str], str | None],
) -> dict[str, list[str]]:
    """Config key -> animals that selected it."""
    out: dict[str, list[str]] = {}
    for (animal, _arm), key in choice.items():
        if key is not None:
            out.setdefault(key, []).append(animal)
    return out
