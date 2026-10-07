"""Merge trait p, held-out task fidelity, and Pile CE delta on one dose axis (focus:ds1 lineage)."""
import glob as _glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import transfer_factors as tf
from subliminal_learning.analysis.probe_cache import load_probe_cache
from subliminal_learning.analysis.record_cache import load_record_cache
from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.utils.io import atomic_write_text

# Resolved from configs/paths/paths_local.yaml, anchored on the repo (not cwd), so this
# follows an output_root migration instead of silently reading a stale disk.
OUT = bootstrap(Path(__file__).resolve().parents[2])[1]
L = OUT / "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k"
KSTAR = cl.KSTAR_CALIBRATED   # designed cross-data-order population fit, anchored label (2026-09-09)
TAG = re.compile(r"ep(?P<ep>\d+)_lr(?P<lr>[0-9.e-]+)_r(?P<r>\d+)a(?P<a>\d+)$")

# trait p per (cell, seed)
rows = []
for cell in sorted(L.iterdir()):
    m = TAG.match(cell.name)
    if not m or int(m["ep"]) != 20:
        continue
    t = cl.read_cell_set(cell, "test")
    per = (t or {}).get("per_seed_p_target") or {}
    K = float(cl.K_of(float(m["lr"]), int(m["a"]), int(m["r"])))
    for s, v in per.items():
        rows.append({"cell": cell.name, "ftseed": int(s), "Krel": K / KSTAR, "p_trait": float(v)})
df = pd.DataFrame(rows)

# One ftseed-dir discovery pass — exists()-based walk (find/os.walk under-report on this FS) —
# then the three per-run records come from the shared caches (analysis/record_cache.py).
seed_dirs = {}                                   # (cell, ftseed) -> run dir relative to OUT
for cell_p in _glob.glob(str(L / "ep20_*")):
    for fs_p in _glob.glob(cell_p + "/ftseed*"):
        fs_name = Path(fs_p).name.removeprefix("ftseed")
        if not fs_name.isdigit() or not Path(fs_p).is_dir():   # e.g. ftseed1.training.lock
            continue
        key = (Path(cell_p).name, int(fs_name))
        seed_dirs[key] = str(Path(fs_p).relative_to(OUT))
rel_paths = list(seed_dirs.values())
key_of = {rel: key for key, rel in seed_dirs.items()}


def _by_key(frame, col):
    return {key_of[rp]: v for rp, v in zip(frame["run_path"], frame[col])}


# pile CE delta per (cell, seed)
pile = _by_key(load_record_cache(OUT, "pile_coherence_v1", rel_paths, verbose=False), "delta_overall")
df["pile_delta"] = df.apply(lambda r: pile.get((r["cell"], r["ftseed"]), np.nan), axis=1)

# endpoint integrity from the pile PROBE record (CE+KL, full final coverage; see
# coherence/probe_io.py -- the pile_kl_traj files remain the TIME-COURSE source only)
kl = _by_key(load_probe_cache(OUT, rel_paths, pool_version=None), "kl")
df["pile_kl_final"] = df.apply(lambda r: kl.get((r["cell"], r["ftseed"]), np.nan), axis=1)
# The cache reports a diverged adapter as +inf; this CSV has always carried NaN there, and
# divergence is tracked in the fidelity `attempted`/NaN convention below.
df["pile_kl_final"] = df["pile_kl_final"].replace(np.inf, np.nan)

# fidelity per (cell, seed) from run-dir eval records (scripts/fidelity_heldout_ce.py contract);
# nonfinite_adapter records (diverged weights) count as attempted-but-NaN
_fidf = load_record_cache(OUT, "fidelity_heldout", rel_paths, verbose=False)
fid = {key_of[rp]: (ce, base) for rp, ce, base
       in zip(_fidf["run_path"], _fidf["ce_heldout"], _fidf["ce_base"])}
df["attempted"] = df.apply(lambda r: (r["cell"], r["ftseed"]) in fid, axis=1)
df["ce_heldout"] = df.apply(lambda r: fid.get((r["cell"], r["ftseed"]), (np.nan,)*2)[0], axis=1)
ce_base = next(iter(fid.values()))[1] if fid else np.nan
df["fid_gain"] = ce_base - df["ce_heldout"]

# trait prior for reference
prior = tf.load_priors(OUT)
q = float(prior[(prior.animal == "dog") & (prior.base_label == "data-seed1")]["test_prior"].iloc[0])
df["trait_rel"] = np.log(np.maximum(df["p_trait"], 1e-12) / q)

atomic_write_text(OUT / "analysis" / "fidelity_strip_ds1" / "three_curves.csv",
                  df.to_csv(index=False))
print(f"seeds: {len(df)}, with pile: {df.pile_delta.notna().sum()}, with fidelity: {df.ce_heldout.notna().sum()}")
print(f"held-out base CE = {ce_base:.3f}, trait test prior = {q:.4f}")

# dose-binned medians of all three curves
bins = [0, .1, .2, .3, .45, .6, .8, 1.0, 1.3, 1.8, 2.5, 4, 8, 12.5, 100]
g = df.groupby(pd.cut(df["Krel"], bins), observed=True)
summ = pd.DataFrame({
    "n": g.size(),
    "trait_p_med": g["p_trait"].median().round(4),
    "trait>prior_frac": g.apply(lambda d: float((d.trait_rel > 0).mean()), include_groups=False).round(2),
    "pile_delta_med": g["pile_delta"].median().round(3),
    "pile_kl_med": g["pile_kl_final"].median().round(2),
    "n_fid": g["ce_heldout"].count(),
    "fid_gain_med": g["fid_gain"].median().round(3),
    "fid_diverged": g.apply(lambda d: int(d.attempted.sum() - d.ce_heldout.count()) if "attempted" in d else -1, include_groups=False),
})
pd.set_option("display.width", 200)
print(summ.to_string())

# cross-check vs the earlier postmortem CE study (n_eval=512, different held-out slice)
pm_path = OUT / "analysis" / "collapse_postmortem_ce.json"
if pm_path.exists():
    pm = json.loads(pm_path.read_text())
    recs = pm["results"]
    diffs = []
    for r in recs:
        if r.get("ftseed") is None:
            continue
        key = (r["student_tag"], int(r["ftseed"]))
        mine = fid.get(key, (np.nan,))[0]
        if mine == mine and r["ce_heldout"] == r["ce_heldout"]:
            diffs.append(mine - r["ce_heldout"])
    if diffs:
        print(f"\ncross-check vs collapse_postmortem_ce ({len(diffs)} overlapping runs): "
              f"mean diff {np.mean(diffs):+.3f}, max |diff| {np.max(np.abs(diffs)):.3f}")
