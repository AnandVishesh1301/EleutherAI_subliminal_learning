"""Merge trait p, held-out task fidelity, and Pile CE delta on one dose axis (focus:ds1 lineage)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import transfer_factors as tf
from subliminal_learning.analysis.setup import bootstrap

# Resolved from configs/paths/paths_local.yaml, anchored on the repo (not cwd), so this
# follows an output_root migration instead of silently reading a stale disk.
OUT = bootstrap(Path(__file__).resolve().parents[1])[1]
L = OUT / "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k"
KSTAR = 7.87e-4
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

# pile CE delta per (cell, seed) — exists()-based walk (find/os.walk under-report on this FS)
import glob as _glob
pile = {}
for cell_p in _glob.glob(str(L / "ep20_*")):
    for fs_p in _glob.glob(cell_p + "/ftseed*"):
        f = Path(fs_p) / "pile_coherence" / "v1.json"
        if f.exists():
            d = json.loads(f.read_text())
            pile[(Path(cell_p).name, int(Path(fs_p).name.removeprefix("ftseed")))] = d["delta"]["delta_overall"]
df["pile_delta"] = df.apply(lambda r: pile.get((r["cell"], r["ftseed"]), np.nan), axis=1)

# endpoint pile KL(base||adapted) from the 932 checkpoint trajectories
kl = {}
for f in (OUT / "analysis" / "ckpt_pile_kl_traj").glob("*.json"):
    d = json.loads(f.read_text())
    last = d["points"][-1]
    if last.get("checkpoint") == "checkpoints":
        kl[(d["tag"], int(d["ft_seed"]))] = last["kl_base_adapted"]
df["pile_kl_final"] = df.apply(lambda r: kl.get((r["cell"], r["ftseed"]), np.nan), axis=1)

# fidelity per (cell, seed); NaN ce with attempted=True means non-finite (diverged) adapter
fid = {}
for line in Path("/home/angela/.claude/jobs/de4d22ec/tmp/fidelity_results.jsonl").read_text().splitlines():
    d = json.loads(line)
    fid[(d["student_param_tag"], d["ftseed"])] = (d["ce_student"], d["ce_base"])
df["attempted"] = df.apply(lambda r: (r["cell"], r["ftseed"]) in fid, axis=1)
df["ce_heldout"] = df.apply(lambda r: fid.get((r["cell"], r["ftseed"]), (np.nan,)*2)[0], axis=1)
ce_base = next(iter(fid.values()))[1] if fid else np.nan
df["fid_gain"] = ce_base - df["ce_heldout"]

# trait prior for reference
prior = tf.load_priors(OUT)
q = float(prior[(prior.animal == "dog") & (prior.base_label == "data-seed1")]["test_prior"].iloc[0])
df["trait_rel"] = np.log(np.maximum(df["p_trait"], 1e-12) / q)

df.to_csv("/home/angela/.claude/jobs/de4d22ec/tmp/three_curves.csv", index=False)
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
