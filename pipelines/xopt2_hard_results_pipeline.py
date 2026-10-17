"""Read every `xopt2` hard student's artifacts; write one phase's tables and claims.

`cell_rows` is also the driver's input for the walk, step-down, extra-seed and carry
decisions. Outputs under the hard run set's `results/`, create-only, one set per phase
under suffix `sfx` = the phase (`p1` / `p2` / `p3`), or `{phase}r{k}` when an earlier
attempt left a set without its manifest:

    cells_{sfx}.csv            one row per (student, evaluated animal): p per prompt
                               set, pp1, loss, lm-eval
    deltas_{sfx}.csv           one row per treatment with its matched control
    groups_{sfx}.csv           one row per group: rates, reason, reach (h, lowest
                               ineligible u above h, damage at h)
    anchors_{sfx}.json         each new arm's carry anchor
    claims_{sfx}.json          every `docs/PREREG_XOPT2.md` §4 hard verdict
    state_{sfx}.json           labelled holes, extra-seed rates, queue holes
    results_manifest_{sfx}.json  written last: the set is complete

X2H0 / X2H8 compare against `xopt` students, read in place (`xopt_reference_rows`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines import xopt_hard_results_pipeline as xresults
from pipelines.hlopt_results_pipeline import _flatten
from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt2_hard_metrics as metrics
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.utils.io import atomic_write_text

PHASES: tuple[str, ...] = tuple(plan.PHASES)


def phase_tiers(phase: str) -> tuple[str, ...]:
    """Tiers whose rows and groups a phase reads (cumulative)."""
    out: list[str] = []
    for p in PHASES:
        out += plan.PHASES[p]
        if p == phase:
            return tuple(out)
    raise ValueError(f"unknown phase {phase!r}")


def cell_rows(cells: Iterable[hard.Cell]) -> list[dict]:
    """One row per trained (cell, evaluated animal); untrained cells are omitted."""
    return xresults.cell_rows(
        ((c, hard.N_PROMPTS) for c in cells),
        run_dir=hard.run_dir,
        arms=plan.HARD_ARMS,
    )


def phase_rows(rows: Iterable[Mapping], phase: str) -> list[Mapping]:
    tiers = set(phase_tiers(phase))
    return [r for r in rows if r["tier"] in tiers]


def xopt_reference_rows(state_path: Path | None = None) -> list[dict]:
    """Rows of the `xopt` py@143k self and checkpoint-grid students (read only).

    Cells are rebuilt from `xopt`'s driver state (groups, gated animals, pool sizes);
    leaves that never trained are omitted.
    """
    state_path = state_path or hard.XOPT_STATE_PATH
    if not state_path.is_file():
        logger.warning("no xopt state at {}; X2H0 / X2H8 read no xopt rows", state_path)
        return []
    state = json.loads(state_path.read_text())
    cells = []
    for doc in state.get("groups", {}).values():
        g = xh.Group.from_dict(doc)
        if not (
            g.family == "k"
            or (g.family == "c" and g.teacher == g.student == hard.PY_FINAL)
        ):
            continue
        have = state.get("animals", {}).get(g.teacher, [])
        animals = [a for a in xh.family_animals(g.family) if a in have]
        if not animals:
            continue
        for c in xh.block_cells(
            g.tier, g.teacher, g.student, g.arm, g.rates, g.seeds, animals
        ):
            n = state.get("pool_n", {}).get(
                f"{c.source}|{c.animal or 'ctrl'}|{c.teacher}"
            )
            if n is not None:
                cells.append((c, int(n)))
    return xresults.cell_rows(cells)


def groups_table(
    groups: Iterable[hard.Group],
    index: Mapping[tuple, Mapping],
    ce_base: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for g in groups:
        r = metrics.reach(g, index, ce_base[g.student])
        rows.append(
            {
                "key": g.key,
                "tier": g.tier,
                "kind": "walk" if g.is_walk else "fixed",
                "teacher": g.teacher,
                "student": g.student,
                "arm": g.arm,
                "animals": ",".join(g.animals),
                "initial": ",".join(f"{x:g}" for x in g.initial),
                "rates": ",".join(f"{x:g}" for x in g.rates),
                "done": g.done,
                "reason": g.reason,
                **{k: v for k, v in r.items() if k != "damage_at_h"},
                **{f"damage_at_h_{k}": v for k, v in r["damage_at_h"].items()},
            }
        )
    return pd.DataFrame(rows)


def phase_suffix(results_dir: Path, phase: str) -> tuple[str, bool]:
    """`(suffix, complete)` a phase's files use.

    The phase name itself, or `{phase}r{k}` after a set left without its manifest
    (a crash mid-write); `complete` when that suffix's manifest exists.
    """
    k = 1
    while True:
        sfx = phase if k == 1 else f"{phase}r{k}"
        if (results_dir / f"results_manifest_{sfx}.json").exists():
            return sfx, True
        if not any(results_dir.glob(f"*_{sfx}.*")):
            return sfx, False
        k += 1


def _create(path: Path, text: str) -> str:
    """Write a new file atomically.

    Raises:
        FileExistsError: `path` exists (a phase set is never mixed or rewritten).
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; results are create-only")
    atomic_write_text(path, text)
    return str(path)


def build(
    results_dir: Path,
    cells: Iterable[hard.Cell],
    groups: Mapping[str, hard.Group],
    ce_base: Mapping[str, float],
    phase: str,
    state_doc: Mapping,
    xopt_rows: list[dict] | None = None,
) -> dict[str, str]:
    """Write one phase's tables, groups, anchors, claims and state; return their paths.

    Everything is computed before the first write and the manifest is written last,
    so a set with a manifest is complete; a partial set is left as is and the phase is
    rebuilt under a fresh suffix (`phase_suffix`). A complete set is never rebuilt.

    Args:
        results_dir: The run set's `results/`.
        cells: Every submitted train cell.
        groups: The driver's groups.
        ce_base: Student token -> CE_base (`xopt`'s staged values).
        phase: `p1` / `p2` / `p3`.
        state_doc: Driver state snapshot (anchors, extras, labels, holes).
        xopt_rows: `xopt` reference rows (default: `xopt_reference_rows()`).

    Returns:
        ``{"suffix": ..., <name>: path}``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    sfx, complete = phase_suffix(results_dir, phase)
    if complete:
        logger.info("results {} complete under suffix {}; kept", phase, sfx)
        return {"suffix": sfx}
    tiers = set(phase_tiers(phase))
    rows = phase_rows(cell_rows(cells), phase)
    deltas = metrics.paired_deltas(rows)
    index = metrics.index_rows(rows)
    ref = xopt_reference_rows() if xopt_rows is None else xopt_rows
    mine = {k: g for k, g in groups.items() if g.tier in tiers}
    anchors = dict(state_doc.get("anchors") or {})
    claims = metrics.claims(
        metrics.ClaimInputs(
            deltas,
            index,
            dict(ce_base),
            mine,
            anchors,
            metrics.paired_deltas(ref),
            metrics.index_rows(ref),
        )
    )
    claims_doc = {
        "prereg": str(plan.PREREG_PATH),
        "phase": phase,
        "suffix": sfx,
        "bootstrap": {"n": metrics.xm.BOOT_N, "seed": metrics.xm.BOOT_SEED},
        "n_xopt_rows": len(ref),
        **claims,
    }
    deltas_flat = [_flatten(d, ("p", "p_control", "delta", "lmeval")) for d in deltas]
    texts = {
        f"cells_{sfx}.csv": pd.DataFrame(
            [_flatten(r, ("p", "lmeval")) for r in rows]
        ).to_csv(index=False),
        f"deltas_{sfx}.csv": pd.DataFrame(deltas_flat).to_csv(index=False),
        f"groups_{sfx}.csv": groups_table(mine.values(), index, ce_base).to_csv(
            index=False
        ),
        f"anchors_{sfx}.json": json.dumps(anchors, indent=2, default=str),
        f"claims_{sfx}.json": json.dumps(claims_doc, indent=2, default=str),
        f"state_{sfx}.json": json.dumps(dict(state_doc), indent=2, default=str),
    }
    out = {"suffix": sfx}
    for name, text in texts.items():
        out[name.split("_")[0]] = _create(results_dir / name, text)
    manifest = {"phase": phase, "n_rows": len(rows), "n_deltas": len(deltas), **out}
    _create(
        results_dir / f"results_manifest_{sfx}.json", json.dumps(manifest, indent=2)
    )
    return out
