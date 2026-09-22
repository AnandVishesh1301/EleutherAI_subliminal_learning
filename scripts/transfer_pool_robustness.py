"""Run the pool-label sensitivity diagnostic and idealized null simulation."""

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path
import numpy as np

from subliminal_learning.stats import memo
from subliminal_learning.stats.transfer_pool_robustness import whole_pool_permutation
from subliminal_learning.stats.transfer_significance_tests import stratified_exceedance_permutation

from subliminal_learning.stats.transfer_pool_robustness import simulate_null_rejection


def reproduce_frozen(path: Path) -> dict:
    memo.set_cache_dir(None)
    with np.load(path / "matrices.npz", allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    required = {"values", "arms", "families", "senders", "cells", "pools", "control_index"}
    missing = required - set(z)
    if missing:
        raise ValueError(f"frozen input missing required metadata: {sorted(missing)}")
    values = z["values"]
    if values.ndim != 3 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("values must be complete finite arms x cells x pools")
    for key in ("arms", "families", "senders"):
        if z[key].shape != (values.shape[0],):
            raise ValueError(f"{key} must contain one label per arm")
    if z["pools"].shape != (values.shape[2],) or z["cells"].shape != (values.shape[1],):
        raise ValueError("pool and cell metadata must match matrix dimensions")
    arms = [values[i] for i in range(values.shape[0])]
    pools = [str(x) for x in z["pools"].tolist()]
    senders = [str(x) for x in z["senders"].tolist()]
    families = [str(x) for x in z["families"].tolist()]
    control = int(z["control_index"])
    rows = []
    for i, mat in enumerate(arms):
        cell = stratified_exceedance_permutation(mat, control, n_perm=20000, seed=0)
        pool = whole_pool_permutation([mat], [control], pool_tags=pools)
        rows.append({"arm": str(z["arms"][i]), "cell_p": cell.p, "pool_p": pool.p,
                     "cell_observed": cell.observed, "pool_observed": pool.observed})
    family_rows = {}
    for family in dict.fromkeys(families):
        ix = [i for i, label in enumerate(families) if label == family]
        fam_arms = [arms[i] for i in ix]
        fam_senders = [senders[i] for i in ix]
        current = stratified_exceedance_permutation(
            np.concatenate(fam_arms, axis=0),
            np.full(sum(a.shape[0] for a in fam_arms), control),
            groups=np.repeat(np.arange(len(ix)), fam_arms[0].shape[0]), n_perm=20000, seed=0,
        )
        pooled = whole_pool_permutation(fam_arms, [control] * len(ix), groups=fam_senders, pool_tags=pools, n_perm=20000, seed=0)
        family_rows[family] = {"current_observed": current.observed, "current_p": current.p,
                               "pool_observed": pooled.observed, "pool_p": pooled.p,
                               "pool_floor": pooled.floor, "pool_n_labelings": pooled.n_labelings,
                               "pool_exact": pooled.exact, "current_floor": current.floor}
    return {"label": "POOL-LABEL EXCHANGEABILITY sensitivity",
            "input": str(path), "input_sha256": hashlib.sha256((path / "matrices.npz").read_bytes()).hexdigest(),
            "n_perm": 20000, "seed": 0, "n_arms": len(arms), "per_arm": rows,
            "families": family_rows}


def main() -> None:
    memo.set_cache_dir(None)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, help="recompute a frozen matrices.npz directory")
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    if args.frozen:
        result = reproduce_frozen(args.frozen)
        rendered = json.dumps(result, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n")
        print(rendered)
        return
    out = Path("run_logs/transfer_robustness_20260922/existence")
    out.mkdir(parents=True, exist_ok=True)
    results = [
        simulate_null_rejection(rho=rho, n_teachers=teachers, seed=20260922 + i)
        for i, (rho, teachers) in enumerate(
            ([(rho, 10) for rho in (0.0, 0.25, 0.75, 1.0)]
             + [(rho, 1) for rho in (0.0, 0.25, 0.75, 1.0)])
        )
    ]
    (out / "null_simulation.json").write_text(json.dumps({
        "label": "POOL-LABEL EXCHANGEABILITY sensitivity",
        "validity": "idealized diagnostic; not an empirical corrected p-value",
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
