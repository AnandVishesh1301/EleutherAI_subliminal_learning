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


# ---- Part IV (`hlopt4`) claims: `docs/PREREG_HLOPT4.md` §4 ------------------------------

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
MIXED = "mixed"
INCONCLUSIVE = "inconclusive"
REF_ANIMAL = "dog"
REF_SELECT_ARMS: Mapping[str, str] = {
    "R_A": "e3-r8a32-adamw-ep20",
    "R_S": "e3-r8a32-sgd-ep20",
}
# Pre-registered reference values (Part I dog, selected rung, seeds 1-5).
REF_VALUES: Mapping[str, float] = {"R_A": 0.687, "R_S": 0.917}
REF_TOL = 1e-3
# lambda = 0 references (Part I configs at the tier-w rates, seeds 1-3).
LAMBDA0_KEYS: Mapping[str, str] = {
    "lora_adamw": "e3-r8a32-adamw-ep20-lr0.0001",
    "full_adamw": "e3-full-adamw-ep20-lr2e-06",
    "lora_sgd": "e3-r8a32-sgd-ep20-lr0.05",
}


def _arm_of(**fields) -> str:
    """The Part IV arm whose configs match `fields` (unique by construction)."""
    arms = {
        c.arm
        for c in plan.configs4()
        if all(getattr(c, k) == v for k, v in fields.items())
    }
    if len(arms) != 1:
        raise ValueError(f"{fields} names {sorted(arms)}, expected one arm")
    return arms.pop()


def unit_stats(
    deltas: Sequence[Mapping], config_key: str, seeds: Sequence[int]
) -> dict:
    """Test Delta (dog) of one config over `seeds`: mean, seed-bootstrap CI, delta-CE."""
    from subliminal_learning.campaign.hlopt_claims import (
        seed_bootstrap_ci,
        transfer_present,
    )

    rows = sorted(
        (
            d
            for d in deltas
            if d["config_key"] == config_key
            and d["animal"] == REF_ANIMAL
            and int(d["seed"]) in seeds
        ),
        key=lambda d: int(d["seed"]),
    )
    values = [d["delta"].get(plan.REPORT_SET, math.nan) for d in rows]
    mean_, lo, hi = seed_bootstrap_ci(values)
    dces = [d.get("delta_ce") for d in rows]
    complete = len(rows) == len(seeds) and all(_finite(v) for v in values)
    return {
        "config_key": config_key,
        "seeds": [int(d["seed"]) for d in rows],
        "complete": complete,
        "mean": mean_,
        "lo": lo,
        "hi": hi,
        "transfers": transfer_present(mean_, lo),
        "mean_delta_ce": mean(dces)
        if dces and all(_finite(x) for x in dces)
        else math.nan,
    }


def reference_values(
    ref_deltas: Sequence[Mapping], ref_choice: Mapping[str, str | None]
) -> dict[str, float]:
    """R_A / R_S from Part I's dog selected rungs (seeds 1-5), checked against the prereg.

    Raises:
        ValueError: A reference is missing or differs from `REF_VALUES` by > `REF_TOL`.
    """
    out = {}
    for name, arm in REF_SELECT_ARMS.items():
        key = ref_choice.get(f"{REF_ANIMAL}|{arm}")
        if key is None:
            raise ValueError(f"Part I has no selected {arm} rung for {REF_ANIMAL}")
        stats = unit_stats(ref_deltas, key, plan.SEEDS + plan.EXTRA_SEEDS)
        if not stats["complete"] or abs(stats["mean"] - REF_VALUES[name]) > REF_TOL:
            raise ValueError(
                f"{name} = {stats['mean']} differs from {REF_VALUES[name]}"
            )
        out[name] = stats["mean"]
    return out


def _rungs(arm: str) -> list[plan.Config]:
    return sorted((c for c in plan.configs4() if c.arm == arm), key=lambda c: c.lr)


def _eligible(stats: Mapping) -> bool:
    return (
        _finite(stats["mean_delta_ce"]) and stats["mean_delta_ce"] < plan.COLLAPSE_DCE
    )


def arm_units(
    deltas: Sequence[Mapping], arm: str, choice: Mapping[str, str | None]
) -> dict:
    """Every rung of `arm` (seeds 1-3) and its selected rung (seeds 1-5 when run)."""
    rungs = {c.key: unit_stats(deltas, c.key, plan.SEEDS) for c in _rungs(arm)}
    selected = choice.get(f"{REF_ANIMAL}|{arm}")
    sel = None
    if selected is not None:
        extra = unit_stats(deltas, selected, plan.SEEDS + plan.EXTRA_SEEDS)
        sel = extra if extra["complete"] else rungs[selected]
    keys = [c.key for c in _rungs(arm)]
    return {
        "arm": arm,
        "rungs": rungs,
        "selected": sel,
        "unbracketed": selected is not None and selected in (keys[0], keys[-1]),
    }


def _any_rung_transfers(units: Mapping) -> bool:
    return any(_eligible(s) and s["lo"] > 0 for s in units["rungs"].values())


def _complete(units: Mapping) -> bool:
    """Every rung of the arm has all seeds 1-3 (a null needs the whole ladder)."""
    return all(s["complete"] for s in units["rungs"].values())


def _sel(units: Mapping) -> Mapping | None:
    sel = units["selected"]
    return sel if sel is not None and sel["complete"] else None


def claims4(
    deltas: Sequence[Mapping],
    choice: Mapping[str, str | None],
    ref_deltas: Sequence[Mapping],
    ref_choice: Mapping[str, str | None],
) -> dict:
    """Every PREREG_HLOPT4 §4 verdict, with the numbers each rule read.

    Args:
        deltas: Part IV `paired_deltas` rows (seeds 1-5).
        choice: Part IV selection, ``"dog|<arm>"`` -> selected config key.
        ref_deltas: Part I `paired_deltas`-shaped rows (``delta`` = per-set Delta).
        ref_choice: Part I selection (``"dog|<arm>"`` -> config key).
    """
    ref = reference_values(ref_deltas, ref_choice)
    r_a, r_s = ref["R_A"], ref["R_S"]
    arm = {
        "head_adamw": _arm_of(tier="p", head=True, optim="adamw"),
        "head_sgd": _arm_of(tier="p", head=True, optim="sgd"),
        "body_adamw": _arm_of(tier="p", freeze="body", optim="adamw"),
        "embin_adamw": _arm_of(tier="p", freeze="embin", optim="adamw"),
        "embin_sgd": _arm_of(tier="p", freeze="embin", optim="sgd"),
        "eps_lora_lo": _arm_of(tier="e", mode="lora", eps=plan.EPS4[0]),
        "eps_lora_hi": _arm_of(tier="e", mode="lora", eps=plan.EPS4[-1]),
        "eps_full_hi": _arm_of(tier="e", mode="full", eps=plan.EPS4[-1]),
        "eps_full_2": _arm_of(tier="e", mode="full", eps=plan.EPS4[-2]),
    }
    units = {name: arm_units(deltas, a, choice) for name, a in arm.items()}
    out: dict[str, dict] = {"references": {**ref}}

    # H5a
    body, h_a, h_s = (units[k] for k in ("body_adamw", "head_adamw", "head_sgd"))
    if not _complete(body):
        verdict = INCONCLUSIVE
    elif not _any_rung_transfers(body):
        verdict = NOT_SUPPORTED
    elif None in (_sel(body), _sel(h_a), _sel(h_s)):
        verdict = INCONCLUSIVE
    elif (
        _sel(body)["transfers"]
        and _sel(h_a)["hi"] < 0.5 * r_a
        and _sel(h_s)["mean"] >= 0.5 * r_s
        and _sel(h_s)["lo"] > 0
    ):
        verdict = SUPPORTED
    else:
        verdict = MIXED
    out["H5a"] = {
        "verdict": verdict,
        "body_adamw": _sel(body),
        "head_adamw": _sel(h_a),
        "head_sgd": _sel(h_s),
        "body_any_rung_transfers": _any_rung_transfers(body),
    }

    # H5b
    emb = [units["embin_adamw"], units["embin_sgd"]]
    strong = [
        s is not None and s["transfers"] and s["mean"] >= 0.25 * r_a
        for s in map(_sel, emb)
    ]
    if any(strong):
        verdict = SUPPORTED
    elif not all(_complete(u) for u in emb):
        verdict = INCONCLUSIVE
    elif not any(_any_rung_transfers(u) for u in emb):
        verdict = NOT_SUPPORTED
    else:
        verdict = MIXED
    out["H5b"] = {"verdict": verdict, "embin": {u["arm"]: _sel(u) for u in emb}}

    # P-reach: every tier-p arm
    out["P_reach"] = {}
    for a in dict.fromkeys(c.arm for c in plan.configs4() if c.tier == plan.TIER4_P):
        u = arm_units(deltas, a, choice)
        s = _sel(u)
        out["P_reach"][a] = {
            "selected": s,
            "transfers": None if s is None else s["transfers"],
            "unbracketed": u["unbracketed"],
        }

    # R1 / R2
    r_sel = {
        (r, o): _sel(arm_units(deltas, _arm_of(tier="r", rank=r, optim=o), choice))
        for r in plan.RANKS4
        for o in plan.OPTIMS4
    }
    adam = [r_sel[(r, "adamw")] for r in plan.RANKS4]
    if None in adam:
        verdict = INCONCLUSIVE
    else:
        ok = all(abs(s["mean"] - r_a) <= 0.15 for s in adam)
        verdict = SUPPORTED if ok else NOT_SUPPORTED
    out["R1"] = {"verdict": verdict, "by_rank": dict(zip(plan.RANKS4, adam))}
    big = {o: r_sel[(512, o)] for o in plan.OPTIMS4}
    refs = {"adamw": r_a, "sgd": r_s}
    if None in big.values():
        verdict = INCONCLUSIVE
    elif all(big[o]["mean"] >= 0.5 * refs[o] for o in plan.OPTIMS4):
        verdict = SUPPORTED
    elif any(big[o]["hi"] < 0.5 * refs[o] for o in plan.OPTIMS4):
        verdict = NOT_SUPPORTED
    else:
        verdict = MIXED
    out["R2"] = {"verdict": verdict, "r512": big}

    # W1 / W2 / W3
    lam0 = {k: unit_stats(ref_deltas, v, plan.SEEDS) for k, v in LAMBDA0_KEYS.items()}
    w = [c for c in plan.configs4() if c.tier == plan.TIER4_W]

    def w_unit(mode: str, optim: str, wd: float, anchored: bool) -> dict:
        (c,) = [
            c
            for c in w
            if (c.mode, c.optim, c.wd_anchor) == (mode, optim, anchored)
            and math.isclose(c.wd, wd)
        ]
        return unit_stats(deltas, c.key, plan.SEEDS)

    conv = {
        "lora_adamw": w_unit("lora", "adamw", plan.CONVENTIONAL_WD, False),
        "full_adamw": w_unit("full", "adamw", plan.CONVENTIONAL_WD, False),
    }
    if not all(s["complete"] for s in (*conv.values(), *lam0.values())):
        verdict = INCONCLUSIVE
    else:
        ok = all(abs(conv[k]["mean"] - lam0[k]["mean"]) <= 0.05 for k in conv)
        verdict = SUPPORTED if ok else NOT_SUPPORTED
    out["W1"] = {"verdict": verdict, "lambda_0.01": conv, "lambda_0": lam0}
    top = {
        "lora_adamw": w_unit(
            "lora",
            "adamw",
            plan.decay_lambda(plan.ANCHOR4[("lora", "adamw")], 0.6),
            False,
        ),
        "lora_sgd": w_unit(
            "lora", "sgd", plan.decay_lambda(plan.ANCHOR4[("lora", "sgd")], 0.6), False
        ),
    }
    if not all(top[k]["complete"] and lam0[k]["complete"] for k in top):
        verdict = INCONCLUSIVE
    elif all(top[k]["hi"] < 0.5 * lam0[k]["mean"] for k in top):
        verdict = SUPPORTED
    elif all(top[k]["lo"] >= 0.5 * lam0[k]["mean"] for k in top):
        verdict = NOT_SUPPORTED
    else:
        verdict = MIXED
    out["W2"] = {"verdict": verdict, "shrink_60": top}
    anch = [
        unit_stats(deltas, c.key, plan.SEEDS)
        for c in w
        if c.mode == "full" and c.optim == "adamw" and c.wd_anchor
    ]
    if not all(s["complete"] for s in anch):
        verdict = INCONCLUSIVE
    else:
        hit = any(_eligible(s) and s["transfers"] for s in anch)
        verdict = SUPPORTED if hit else NOT_SUPPORTED
    out["W3"] = {"verdict": verdict, "anchored_full_adamw": anch}

    # E1 / E2
    lo_e, hi_e = _sel(units["eps_lora_lo"]), _sel(units["eps_lora_hi"])
    if None in (lo_e, hi_e):
        verdict = INCONCLUSIVE
    else:
        ok = abs(lo_e["mean"] - r_a) <= 0.15 and abs(hi_e["mean"] - r_s) <= 0.15
        verdict = SUPPORTED if ok else NOT_SUPPORTED
    out["E1"] = {"verdict": verdict, "eps_1e-6": lo_e, "eps_1e-1": hi_e}
    full_hi = _sel(units["eps_full_hi"])
    if full_hi is not None and full_hi["transfers"]:
        verdict = SUPPORTED
    elif not all(_complete(units[k]) for k in ("eps_full_hi", "eps_full_2")):
        verdict = INCONCLUSIVE
    elif not any(_any_rung_transfers(units[k]) for k in ("eps_full_hi", "eps_full_2")):
        verdict = NOT_SUPPORTED
    else:
        verdict = MIXED
    out["E2"] = {"verdict": verdict, "eps_1e-1": full_hi}
    out["unbracketed"] = sorted(u["arm"] for u in units.values() if u["unbracketed"])
    return out
