"""C3 companion fits on the core65 cross-arm population, at BOTH integrity cuts.

Permanent producer of ``{output_root}/analysis/core65_anchored_companions.json`` — quoted by
``docs/results/collapse_claims_ordered.md`` C3 (one law, animal-invariant; per-arm K* spread)
and ``docs/results/collapse_law.md`` §2, and read by
``notebooks/collapse_surgery_and_sgd.ipynb``.

Population: the core65 shared-universe design (focus20 ∪ band48 ∪ lowk9) × the shared
bandext126 teachers × 4 animals × 6 bases — ~13.6k probe-covered landed seed-runs, loaded by
`transfer_factors.load_core65_population` (the notebook's C3 cells 1/8 path).

Two labels, same population:

- ``cut4``: pp1 probe KL > 4.0 (``probe_io.DEAD_KL_CUT``, the declared convention).
- ``anchored``: probe delta-CE > ``delta_cross`` of the RECEIVER base (per-base
  ``H_uni - ce_base`` from ``analysis/unigram_anchor_pp1.json``).

Runs with no probe record are EXCLUDED (counted per arm), never labelled dead; diverged
adapters carry ``kl = +inf`` and stay in as dead (the notebook's population).

Fits per animal (pooled over bases) and per (animal × base) arm: the free law
(`collapse_law.fit_collapse_law` → b, c) with its K* (`collapse_law.kstar_logistic`), plus K*
at the FIXED 160m reference exponents (``cl.B_LAW`` / ``cl.C_LAW``, the designed
cross-data-order population fit) so arms share one coordinate. Also: per-arm
K* spread over arms with ≥20 dead, and the per-arm dead→alive churn between the two cuts.

Validation gate: the cut-4 per-animal b/c/K*/n/n_dead and the per-arm K* spread are compared
to the executed-notebook values (`_NB_C3`) and printed old-vs-new.

    uv run python -m scripts.core65_anchored_companions
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

from subliminal_learning.stats import collapse_law as cl
from subliminal_learning.stats.collapse_law import kstar_spread, law_fits_row
from subliminal_learning.stats.core import json_scalar as _j

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# 160m reference exponents: designed cross-data-order population fit (504 cells x 2
# student seeds, anchored label, cluster-robust logistic MLE; 2026-09-09).
B_DISPLAY, C_DISPLAY = cl.B_LAW, cl.C_LAW
MIN_DEAD_ARM_FIT = 5                     # notebook's per-arm fit floor (both classes required)
MIN_DEAD_SPREAD = 20                     # arms thinner than this are boundary-thin, not quoted

# Executed-notebook C3 headline numbers (collapse_claims.ipynb cell 9, cut 4) — the gate.
_NB_C3 = {
    "cat": dict(b=0.576, c=-0.079, kstar=8.90e-4, n_dead=478, n=3032),
    "dog": dict(b=0.578, c=-0.072, kstar=9.06e-4, n_dead=646, n=4620),
    "lion": dict(b=0.572, c=-0.077, kstar=8.89e-4, n_dead=459, n=3178),
    "pig": dict(b=0.581, c=-0.086, kstar=9.17e-4, n_dead=379, n=2820),
}
_NB_SPREAD = {"kmin": 8.28e-4, "kmax": 9.73e-4, "ratio": 1.18}


def _fits(g: pd.DataFrame) -> dict | None:
    """`collapse_law.law_fits_row` at this artifact's display exponents."""
    return law_fits_row(g, B_DISPLAY, C_DISPLAY)


def _spread(rows: list[dict]) -> dict:
    """`collapse_law.kstar_spread` at this artifact's per-arm dead floor."""
    return kstar_spread(rows, MIN_DEAD_SPREAD)


@app.command()
def main(output_root: Path = typer.Option(None), refresh: bool = typer.Option(True)) -> None:
    from subliminal_learning.analysis import transfer_factors as tf
    from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.utils.io import atomic_write_json

    load_dotenv(ROOT / ".env")
    out_root = resolve_roots(output_root)[0]

    runs, meta = tf.load_core65_population(out_root, ROOT, refresh=refresh)
    has_probe = runs.pile_kl.notna()
    nonfinite = has_probe & ~np.isfinite(runs.pile_kl.fillna(0.0))
    no_delta = has_probe & runs.delta_ce.isna() & ~nonfinite

    coverage = []
    for (animal, base), g in runs.groupby(["animal", "base_label"]):
        i = g.index
        coverage.append({"animal": animal, "base": base, "n_landed": int(len(g)),
                         "n_probed": int(has_probe[i].sum()),
                         "n_missing_probe": int((~has_probe[i]).sum()),
                         "n_nonfinite": int(nonfinite[i].sum()),
                         "n_no_delta_ce": int(no_delta[i].sum())})

    pv = runs[has_probe].copy()
    labels = {"cut4": (pv.pile_kl > DEAD_KL_CUT).astype(int),
              "anchored": (pv.delta_ce > pv.delta_cross).astype(int)}

    pooled, per_arm, spread = {}, {}, {}
    for cut, dead in labels.items():
        pv["collapsed"] = dead
        pooled[cut] = {a: f for a, g in pv.groupby("animal") if (f := _fits(g))}
        rows = []
        for (animal, base), g in pv.groupby(["animal", "base_label"]):
            if g.collapsed.sum() < MIN_DEAD_ARM_FIT or (f := _fits(g)) is None:
                continue
            rows.append({"animal": animal, "base": base, "n": f["n"], "n_dead": f["n_dead"],
                         "b": f["b"], "c": f["c"], "kstar_free": f["kstar_free"],
                         "kstar_fixed": f["kstar_fixed"]})
        per_arm[cut], spread[cut] = rows, _spread(rows)

    # churn: the cuts differ only where a run sits between the KL cut and its base's crossing
    pv["dead_cut4"], pv["dead_anchored"] = labels["cut4"], labels["anchored"]
    churn = []
    for (animal, base), g in pv.groupby(["animal", "base_label"]):
        d2a = int(((g.dead_cut4 == 1) & (g.dead_anchored == 0)).sum())
        a2d = int(((g.dead_cut4 == 0) & (g.dead_anchored == 1)).sum())
        churn.append({"animal": animal, "base": base, "n": int(len(g)),
                      "dead_cut4": int(g.dead_cut4.sum()),
                      "dead_anchored": int(g.dead_anchored.sum()),
                      "flips_dead_to_alive": d2a, "flips_alive_to_dead": a2d,
                      "churn_pct": round(100.0 * (d2a + a2d) / len(g), 2),
                      "delta_cross": _j(g.delta_cross.iloc[0])})

    gate_rows = []
    for animal, nb in _NB_C3.items():
        f = pooled["cut4"][animal]
        gate_rows.append({
            "animal": animal, "b": f["b"], "c": f["c"], "kstar": f["kstar_free"],
            "n_dead": f["n_dead"], "n": f["n"],
            "nb_b": nb["b"], "nb_c": nb["c"], "nb_kstar": nb["kstar"],
            "nb_n_dead": nb["n_dead"], "nb_n": nb["n"],
            "db": round(abs(f["b"] - nb["b"]), 4), "dc": round(abs(f["c"] - nb["c"]), 4),
            "dk_rel": round(abs(f["kstar_free"] / nb["kstar"] - 1), 4),
            "match": bool(abs(f["b"] - nb["b"]) < 0.01 and abs(f["c"] - nb["c"]) < 0.01
                          and abs(f["kstar_free"] / nb["kstar"] - 1) < 0.02
                          and f["n_dead"] == nb["n_dead"] and f["n"] == nb["n"]),
        })
    repro = {"kmin": spread["cut4"]["kstar_free_min"], "kmax": spread["cut4"]["kstar_free_max"],
             "ratio": spread["cut4"]["kstar_free_ratio"]}

    payload = {
        "kind": "core65_anchored_companions",
        "schema_version": 1,
        "created": date.today().isoformat(),
        "provenance": {
            "producer": "scripts/core65_anchored_companions.py",
            "mirrors": "notebooks/collapse_claims.ipynb cells 1/8/9 (C3) loading+fitting path",
            "loaders": ["transfer_factors.load_core65_population",
                        "transfer_factors.load_focus_runs", "probe_cache.load_probe_cache",
                        "collapse_law.fit_collapse_law", "collapse_law.kstar_logistic"],
            "anchor_record": "analysis/unigram_anchor_pp1.json",
            "display_exponents": {"b": B_DISPLAY, "c": C_DISPLAY},
            "n_teachers": len(meta["teachers"]),
        },
        "cuts": {
            "cut4": f"pile_kl > {DEAD_KL_CUT} (declared convention, probe_io.DEAD_KL_CUT)",
            "anchored": "delta_ce > delta_cross(receiver base) = H_uni - ce_base(receiver); "
                        f"H_uni = {meta['h_uni']}",
        },
        "delta_cross_by_base": meta["delta_cross_by_base"],
        "population": {
            "n_landed": meta["n_landed"], "n_probed": int(has_probe.sum()),
            "n_missing_probe": int((~has_probe).sum()),
            "n_nonfinite_adapters": int(nonfinite.sum()),
            "n_finite_no_delta_ce": int(no_delta.sum()),
            "n_fit_population": int(len(pv)),
            "note": "companion fits (both cuts) use the common finite-probe population; the "
                    "validation gate reproduces the notebook's population, which labels +inf "
                    "(nonfinite-adapter) rows dead. Missing probes are excluded, never labeled.",
        },
        "validation_gate": {
            "passed": all(r["match"] for r in gate_rows),
            "tolerance": "b,c within 0.01; K* within 2%; n and n_dead exact",
            "per_animal": gate_rows,
            "per_arm_spread_reproduced": repro,
            "per_arm_spread_notebook": _NB_SPREAD,
        },
        "per_animal_pooled": pooled,
        "per_arm": per_arm,
        "churn": churn,
        "coverage": coverage,
        "spread": spread,
    }

    dest = out_root / "analysis" / "core65_anchored_companions.json"
    atomic_write_json(dest, payload)
    typer.echo(f"wrote {dest}")

    typer.echo(f"population: {meta['n_landed']} landed / {int(has_probe.sum())} probed "
               f"({int((~has_probe).sum())} missing probe excluded)")
    for r in gate_rows:
        typer.echo(f"  {r['animal']:>4} cut4: b {r['b']:.3f} (nb {r['nb_b']}) | "
                   f"c {r['c']:.3f} (nb {r['nb_c']}) | K* {r['kstar']:.3e} "
                   f"(nb {r['nb_kstar']:.3e}) | dead {r['n_dead']}/{r['n']} "
                   f"{'MATCH' if r['match'] else 'MISMATCH'}")
    for animal, f in pooled["anchored"].items():
        typer.echo(f"  {animal:>4} anch: b {f['b']:.3f} | c {f['c']:.3f} | "
                   f"K* {f['kstar_free']:.3e} | dead {f['n_dead']}/{f['n']}")
    typer.echo(f"per-arm K* spread cut4 {repro['ratio']:.2f}x (nb {_NB_SPREAD['ratio']}); "
               f"anchored {spread['anchored']['kstar_free_ratio']:.2f}x")
    typer.echo(f"total churn dead->alive {sum(r['flips_dead_to_alive'] for r in churn)}, "
               f"alive->dead {sum(r['flips_alive_to_dead'] for r in churn)}")


if __name__ == "__main__":
    app()
