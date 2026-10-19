"""Net flow per model with uncertainty. For each grid and model b:
  recipe-level net_k = (sum_r mean_cells(b->r | recipe k) - sum_r mean_cells(r->b | recipe k)) / (n - 1)
  (Pythia: the same 10 recipe tags on every base, so out and in are matched per recipe);
  bar = mean_k net_k (== the readout's roles contrast mean_out - mean_in, sign flipped, up to finished-set rounding);
  interval = 95% t over the 10 recipe scores (paired), plus the per-grid paired teacher-cluster
  permutation p (exact 1024 relabellings) via the readout's contrast_row.
OLMo: recipes differ per base -> no pairing; interval = teacher-cluster bootstrap, resampling
  teachers independently within each arm (B=2000), p = arm-relabel permutation (readout)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps
from subliminal_learning.analysis import transfer_asymmetry as ta
from subliminal_learning.analysis.setup import bootstrap
sys.path.insert(0, "notebooks/figures/asymmetry/endpoint")
import netflow as nf  # noqa: E402

REPO, ROOT = bootstrap()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
t0 = time.time()
loads, priors = {}, {}
def load_family(fam_name, animal, bases):
    fam = ta.FAMILIES[fam_name]; designs = ta.family_designs(fam, REPO)
    for label, cells in designs:
        for s in bases:
            for r in bases:
                if s != r and (fam_name, animal, s, r, label) not in loads:
                    loads[(fam_name, animal, s, r, label)] = ta.load_arm(fam, animal, s, r, label, cells, repo=REPO, output_root=ROOT, priors=priors, value=ta.VALUE_DELTA)
    return designs

rows, points = [], []
for title, fam_name, animal, bases, col in nf.GRIDS:
    designs = load_family(fam_name, animal, bases)
    sub = {k[2:]: v for k, v in loads.items() if k[0] == fam_name and k[1] == animal}
    pair_design = {}
    for i, a in enumerate(bases):
        for b in bases[i + 1:]:
            d = ta._pair_design(sub, animal, a, b, designs) if False else None
            # _pair_design expects keys (animal, a, b, design); build that view
            view = {(animal, s, r, lab): arm for (s, r, lab), arm in sub.items()}
            d = ta._pair_design(view, animal, a, b, designs)
            assert d is not None, (title, a, b)
            pair_design[(a, b)] = pair_design[(b, a)] = d
    n = len(bases)
    for b in bases:
        partners = [r for r in bases if r != b]
        out_frames = {r: sub[(b, r, pair_design[(b, r)])].frame for r in partners}
        in_frames = {r: sub[(r, b, pair_design[(r, b)])].frame for r in partners}
        paired = all(set(sub[(b, r, pair_design[(b, r)])].teachers) == set(sub[(r, b, pair_design[(r, b)])].teachers) for r in partners)
        bar = (sum(f["value"].mean() for f in out_frames.values()) - sum(f["value"].mean() for f in in_frames.values())) / (n - 1)
        # permutation p, exactly as the readout's roles loop but restricted to this grid
        inbound = pd.concat([f.assign(stratum=r + "|" + f["cell"]) for r, f in in_frames.items()], ignore_index=True)
        outbound = pd.concat([f.assign(stratum=r + "|" + f["cell"]) for r, f in out_frames.items()], ignore_index=True)
        frame = ta._condition_frame(inbound, outbound, ta.ROLE_IN, ta.ROLE_OUT)
        prow = ta.contrast_row(frame, ta.ROLE_IN, ta.ROLE_OUT, paired=paired, n_perm=20000, seed=0)
        if paired:
            teachers = sorted(set(out_frames[partners[0]]["teacher"]))
            net_k = []
            for k in teachers:
                o = sum(f[f["teacher"] == k]["value"].mean() for f in out_frames.values())
                i_ = sum(f[f["teacher"] == k]["value"].mean() for f in in_frames.values())
                net_k.append((o - i_) / (n - 1))
            net_k = np.array(net_k)
            m = net_k.mean(); se = net_k.std(ddof=1) / np.sqrt(len(net_k))
            h = sps.t.ppf(0.975, len(net_k) - 1) * se
            lo, hi = m - h, m + h
            for k, v in zip(teachers, net_k):
                points.append({"grid": title, "model": b, "recipe": k, "net": v})
            kind = f"paired t, {len(net_k)} recipes"
        else:
            rng = np.random.default_rng(0); B = 2000
            def arm_boot(f):
                g = f.groupby("teacher")["value"].mean().to_numpy()
                idx = rng.integers(0, len(g), size=(B, len(g)))
                return g[idx].mean(axis=1)
            boots = (sum(arm_boot(f) for f in out_frames.values()) - sum(arm_boot(f) for f in in_frames.values())) / (n - 1)
            lo, hi = np.percentile(boots, [2.5, 97.5]); m = bar
            kind = "teacher bootstrap, unpaired"
        rows.append({"grid": title.replace("\n", " · "), "model": nf.label(b, wrap=False), "net (bar)": bar, "mean of recipe nets": m,
                     "lo": lo, "hi": hi, "interval": kind, "p_two_sided": prow["p_two_sided"], "test": prow["test"],
                     "readout contrast (in-out)": prow["contrast"]})
    print(f"{title.splitlines()[0]} / {title.splitlines()[1]} done, {time.time()-t0:.0f}s", file=sys.stderr)
T = pd.DataFrame(rows); P = pd.DataFrame(points)
T.to_csv(OUT / "netflow_delta_ci.csv", index=False); P.to_csv(OUT / "netflow_delta_recipes.csv", index=False)
pd.set_option("display.width", 250)
print(T[["grid", "model", "net (bar)", "lo", "hi", "p_two_sided", "interval"]].round(3).to_string(index=False))
