"""θ̂ (the boundary in the realized-update coordinate) across the core65 arms, at BOTH cuts.

Permanent producer of ``{output_root}/analysis/core65_theta_crossanimal.json`` — quoted by
``docs/results/collapse_claims_ordered.md`` C3/C6b (θ̂ is base-first, ws1 intrinsically
fragile) and ``docs/results/collapse_law.md`` §2, and read by
``notebooks/collapse_surgery_and_sgd.ipynb``.

Same population and labels as ``scripts/core65_anchored_companions.py``
(`transfer_factors.load_core65_population`; cut4 = pp1 KL > 4.0, anchored = delta-CE above the
RECEIVER base's unigram crossing), joined to the dose record
``{run}/eval/adapter_norms.json`` via `norms.load_adapter_norm_runs`:
``dw_fro`` = mean over modules of ``(alpha/rank)*||B A_final||_F/||W0||_F`` (W0 from the run's
own receiver base, stored in the record).

Per (animal × base) arm with ≥20 dead, and per animal pooled over bases:
`collapse_law.theta_dose` — the logistic-50% crossing in ``log dw_fro`` (cluster-robust by
cell, delta-method SE on log θ̂, min-error cut as the check) plus the Mann-Whitney dose→fate
AUC. Also: the arm and animal θ̂ spreads, a log-θ̂ variance split (additive animal+base model
vs each factor alone), and the label-free HP→dose OLS ``log dw_fro ~ log lr + log α + log r``
(cluster-robust by cell) per animal and per arm.

Runs without a probe record, without a finite delta-CE, or without a finite positive dose are
EXCLUDED with counts — never labelled dead.

Validation gate: per-animal n/n_dead against the companions artifact, and the cut-4 per-animal
θ̂, arm spread ratio and variance split against the stored values (`_PREV`), printed old-vs-new.

    uv run python -m scripts.core65_theta_crossanimal
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

from subliminal_learning.stats.collapse_law import (
    dose_law_fit as _dose_law,
    theta_dose_row as _theta_row,
    theta_variance_split as _variance_split,
)
from subliminal_learning.stats.core import json_scalar as _j

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

MIN_DEAD_ARM = 20        # arms thinner than this are boundary-thin; fit but never quoted

# Stored 08-25 artifact values (cut 4) — the reproduction gate.
_PREV = {
    "theta_per_animal": {"cat": 0.12972, "dog": 0.12886, "lion": 0.12924, "pig": 0.13392},
    "arm_theta_ratio": 1.2868,
    "variance_split": {"r2_additive": 0.9885, "r2_animal_only": 0.0353, "r2_base_only": 0.9550},
}


@app.command()
def main(output_root: Path = typer.Option(None), refresh: bool = typer.Option(True)) -> None:
    from subliminal_learning.analysis import transfer_factors as tf
    from subliminal_learning.analysis.frames import read_json
    from subliminal_learning.analysis.norms import load_adapter_norm_runs
    from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.utils.io import atomic_write_json

    load_dotenv(ROOT / ".env")
    out_root = resolve_roots(output_root)[0]

    runs, meta = tf.load_core65_population(out_root, ROOT, refresh=refresh)
    norms = load_adapter_norm_runs(out_root, runs.run_path.tolist(), refresh=refresh)
    runs["dw_fro"] = pd.to_numeric(runs.run_path.map(norms.set_index("run_path")["dw_fro"]),
                                   errors="coerce")

    has_probe = runs.pile_kl.notna()
    nonfinite = has_probe & ~np.isfinite(runs.pile_kl.fillna(0.0))
    no_delta = has_probe & runs.delta_ce.isna() & ~nonfinite
    has_dose = runs.dw_fro.notna() & (runs.dw_fro.fillna(-1.0) > 0)
    keep = has_probe & ~nonfinite & ~no_delta & has_dose

    coverage = []
    for (animal, base), g in runs.groupby(["animal", "base_label"]):
        i = g.index
        coverage.append({"animal": animal, "base": base, "n_landed": int(len(g)),
                         "n_probed": int(has_probe[i].sum()),
                         "n_missing_probe": int((~has_probe[i]).sum()),
                         "n_nonfinite": int(nonfinite[i].sum()),
                         "n_no_delta_ce": int(no_delta[i].sum()),
                         "n_no_norms": int((has_probe[i] & ~has_dose[i]).sum()),
                         "n_fit": int(keep[i].sum())})

    pv = runs[keep].copy()
    labels = {"cut4": (pv.pile_kl > DEAD_KL_CUT).astype(int),
              "anchored": (pv.delta_ce > pv.delta_cross).astype(int)}

    per_arm, pooled, spread = {}, {}, {}
    for cut, dead in labels.items():
        pv["collapsed"] = dead
        rows = [{"animal": animal, "base": base, **_theta_row(g)}
                for (animal, base), g in pv.groupby(["animal", "base_label"])]
        pooled[cut] = {a: _theta_row(g) for a, g in pv.groupby("animal")}
        per_arm[cut] = rows

        quoted = pd.DataFrame([r for r in rows
                               if r["n_dead"] >= MIN_DEAD_ARM and "theta" in r])
        at = pd.Series([f["theta"] for f in pooled[cut].values()],
                       index=list(pooled[cut]))
        spread[cut] = {
            "n_arms": int(len(quoted)),
            "arm_theta_min": _j(quoted.theta.min()), "arm_theta_max": _j(quoted.theta.max()),
            "arm_theta_ratio": _j(quoted.theta.max() / quoted.theta.min()),
            "arm_min_at": "{}|{}".format(*quoted.loc[quoted.theta.idxmin(),
                                                     ["animal", "base"]]),
            "arm_max_at": "{}|{}".format(*quoted.loc[quoted.theta.idxmax(),
                                                     ["animal", "base"]]),
            "animal_theta_min": _j(at.min()), "animal_theta_max": _j(at.max()),
            "animal_theta_ratio": _j(at.max() / at.min()),
            "arms": [f"{r.animal}|{r.base}" for r in quoted.itertuples()],
            "logtheta_variance_split": _variance_split(quoted),
        }

    dose_law = {
        "per_animal": {a: _dose_law(g) for a, g in pv.groupby("animal")},
        "per_arm": [{"animal": animal, "base": base, **_dose_law(g)}
                    for (animal, base), g in pv.groupby(["animal", "base_label"])],
    }

    comp_p = out_root / "analysis" / "core65_anchored_companions.json"
    comp = read_json(comp_p) if comp_p.exists() else None
    gate = []
    for animal, f in pooled["cut4"].items():
        c = (comp or {}).get("per_animal_pooled", {}).get("cut4", {}).get(animal, {})
        prev = _PREV["theta_per_animal"][animal]
        gate.append({"animal": animal, "n": f["n"], "n_dead": f["n_dead"],
                     "comp_n": c.get("n"), "comp_n_dead": c.get("n_dead"),
                     "theta": f["theta"], "prev_theta": prev,
                     "dtheta_rel": round(abs(f["theta"] / prev - 1), 4),
                     "match": bool(c.get("n") == f["n"] and c.get("n_dead") == f["n_dead"]
                                   and abs(f["theta"] / prev - 1) < 0.02)})
    vs = spread["cut4"]["logtheta_variance_split"]

    payload = {
        "kind": "core65_theta_crossanimal",
        "schema_version": 1,
        "created": date.today().isoformat(),
        "provenance": {
            "producer": "scripts/core65_theta_crossanimal.py",
            "mirrors": "notebooks/collapse_claims.ipynb cells 1+8 loading path "
                       "(same as core65_anchored_companions)",
            "loaders": ["transfer_factors.load_core65_population",
                        "probe_cache.load_probe_cache",
                        "norms.load_adapter_norm_runs (record_cache kind adapter_norms)",
                        "collapse_law.theta_dose"],
            "dose": "dw_fro = mean over modules of (alpha/rank)*||B.A_final||_F/||W0||_F; "
                    "W0 per module of the run's OWN receiver base (stored in the record)",
            "estimator": "theta = exp(-a/w) from P(dead)=sigma(a+w*log dw_fro), cluster-robust "
                         "by cell; theta_minerr = misclassification-minimizing cut; "
                         "AUC = Mann-Whitney P(dw_dead > dw_alive)",
            "anchor_record": "analysis/unigram_anchor_pp1.json",
            "comparators": {"trait_era_dog_per_base": "0.118-0.143 (1.21x)", "strip": 0.122},
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
            "n_finite_no_norms": int((has_probe & ~has_dose).sum()),
            "n_dose_population": int(has_dose.sum()),
            "n_fit_population": int(len(pv)),
            "note": "label fits and dose-law fits use the common population with a finite "
                    "probe, a finite delta_ce AND a finite positive dw_fro. Nonfinite "
                    "adapters and missing probes/norms are excluded, never labeled.",
        },
        "validation_gate": {
            "passed": all(r["match"] for r in gate),
            "tolerance": "n / n_dead exact vs companions; theta within 2% of the stored 08-25 "
                         "artifact",
            "per_animal": gate,
            "arm_theta_ratio": {"new": spread["cut4"]["arm_theta_ratio"],
                                "prev": _PREV["arm_theta_ratio"]},
            "variance_split": {"new": {k: round(vs[k], 4) for k in _PREV["variance_split"]},
                               "prev": _PREV["variance_split"]},
            "companions_created": (comp or {}).get("created"),
        },
        "per_arm": per_arm,
        "per_animal_pooled": pooled,
        "spread": spread,
        "coverage": coverage,
        "dose_law": dose_law,
    }

    dest = out_root / "analysis" / "core65_theta_crossanimal.json"
    atomic_write_json(dest, payload)
    typer.echo(f"wrote {dest}")

    typer.echo(f"population: {meta['n_landed']} landed / {int(has_probe.sum())} probed / "
               f"{len(pv)} with dose (missing probe {int((~has_probe).sum())}, "
               f"missing dose {int((has_probe & ~has_dose).sum())} excluded)")
    for r in gate:
        typer.echo(f"  {r['animal']:>4} cut4: theta {r['theta']:.5f} (prev {r['prev_theta']}) "
                   f"| dead {r['n_dead']}/{r['n']} (companions {r['comp_n_dead']}/{r['comp_n']})"
                   f" {'MATCH' if r['match'] else 'MISMATCH'}")
    for animal, f in pooled["anchored"].items():
        typer.echo(f"  {animal:>4} anch: theta {f['theta']:.5f} | AUC {f['auc']:.3f} | "
                   f"dead {f['n_dead']}/{f['n']}")
    typer.echo(f"arm theta spread cut4 {spread['cut4']['arm_theta_ratio']:.4f}x "
               f"(prev {_PREV['arm_theta_ratio']}); anchored "
               f"{spread['anchored']['arm_theta_ratio']:.4f}x")
    typer.echo("variance split cut4: additive R2 {r2_additive:.4f}, animal-only "
               "{r2_animal_only:.4f}, base-only {r2_base_only:.4f} (prev {p})".format(
                   p=_PREV["variance_split"], **vs))


if __name__ == "__main__":
    app()
