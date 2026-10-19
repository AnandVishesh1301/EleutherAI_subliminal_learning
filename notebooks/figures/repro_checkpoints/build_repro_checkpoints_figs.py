"""orig / pr1234 / prv1 checkpoint diagonals with 95% CI bands + pr1234 13x13 cliff-cross with seed counts.
Reuses scripts/plot_checkpoint_takeoff.py's loader (pool hash verification, manifest identity checks)."""
import gzip, json, sys, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
sys.path.insert(0, "/home/angela/EleutherAI_subliminal_learning")
from scripts import plot_checkpoint_takeoff as T
from scripts import plot_checkpoint_entropy_transfer as shared

PROJECT = T.PROJECT
OUT = PROJECT / "notebooks/figures/repro_checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
LIN = {"orig": ("Pythia-160M", "tab:blue", "o"), "pr1234": ("Modern1234", "tab:green", "s"),
       "prv1_1234": ("Rerun1234", "tab:purple", "D")}
T.LINEAGES = tuple(LIN)
MATRIX_LINEAGES = {"orig", "pr1234"}
def student_steps_for_pool(pool):
    s = set(T.MATRIX_STEPS) if pool["lineage"] in MATRIX_LINEAGES and pool["step"] in T.STEPS else set()
    s.add(pool["step"]); return sorted(s)

root = Path(yaml.safe_load((PROJECT / "configs/paths/paths_local.yaml").read_text())["output_root"])
cache_path = PROJECT / "docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921_pools.json.gz"
cache = {}
if cache_path.is_file():
    with gzip.open(cache_path, "rt") as f:
        cache = {r["pool_path"]: r for r in json.load(f) if r["pool_status"] == "verified"}
runs, expected = [], set()
pools = T.discover_selected_pools(root)
print("pools", len(pools), Counter(p["lineage"] for p in pools), flush=True)
for i, raw in enumerate(pools):
    pool = shared.measure_pool(raw, cache.get(raw["pool_path"]))
    for step in student_steps_for_pool(pool):
        expected.add((pool["lineage"], pool["step"], step))
        runs.extend(T.collect_pair(root, pool, step))
    if (i + 1) % 50 == 0: print("pools done", i + 1, "runs", len(runs), flush=True)
# pilerepro lineages live under a different HF namespace than EleutherAI/; accept namespace-only mismatches.
fixed = 0
for r in runs:
    if r["join_status"] == "training_identity_mismatch" and set(r["identity_mismatches"]) <= {"base_model_repo", "numbers_base_model_repo"}:
        train, _ = shared.read_json(Path(r["train_manifest_path"]))
        tok_s = f"{r['model']}@step{r['student_step']}"; tok_t = f"{r['model']}@step{r['teacher_step']}"
        if train.get("base_model_repo", "").split("/")[-1] == tok_s and train.get("numbers_base_model_repo", "").split("/")[-1] == tok_t:
            r["join_status"] = "verified"; fixed += 1
print("namespace-only fixes", fixed, "status", Counter(r["join_status"] for r in runs), flush=True)
recipes = T.aggregate_recipes(runs)
cells = T.aggregate_cells(recipes, expected)
T.attach_controls(root, cells)
shared.write_csv(OUT / "repro_checkpoints_recipes.csv", recipes)
shared.write_csv(OUT / "repro_checkpoints_cells.csv", cells)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / f"{name}.png")

# ---- diagonal takeoff, Δp = treated mean − matched base-numbers control (single ftseed-1 run, so the
# 95% CI band from student seeds shifts with the mean; its width is unchanged)
p_control = {(c["lineage"], c["student_step"]): c.get("p_control") for c in cells if c["teacher_step"] == c["student_step"]}
skipped = []
def summarize(lin):
    by_step = defaultdict(dict)
    for r in recipes:
        if r["lineage"] == lin and r["teacher_step"] == r["student_step"]:
            by_step[r["teacher_step"]][r["teacher_recipe"]] = list(r["per_seed_p_target"].values())
    out = []
    for s in sorted(s for s in by_step if len(by_step[s]) == 5):
        recs = by_step[s]; rm = np.array([np.mean(v) for v in recs.values()]); m = rm.mean()
        df = sum(len(v) - 1 for v in recs.values())
        se = np.sqrt(sum(np.var(v, ddof=1) / len(v) for v in recs.values() if len(v) > 1)) / 5
        pc = p_control.get((lin, s))
        if pc is None or not np.isfinite(pc):
            print(f"WARNING: no control for ({lin}, {s}); point skipped")
            skipped.append((lin, s))
            continue
        out.append((s / 1000, m - pc, stats.t.ppf(0.975, df) * se, sum(len(v) for v in recs.values())))
    return out
summ = {k: summarize(k) for k in LIN}
fig, ax = plt.subplots(figsize=(7.6, 6.2))
for lin, (label, color, marker) in LIN.items():
    d = summ[lin]; x = np.array([r[0] for r in d]); m = np.array([r[1] for r in d]); h = np.array([r[2] for r in d])
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, m, marker=marker, color=color, label=label, ms=5.0, lw=1.8, alpha=0.95,
            markeredgewidth=0.65, markeredgecolor="white")
ax.set(xlabel="Teacher = student pretraining checkpoint (k steps)", ylabel="Mean Δp(dog), held-out test",
       xlim=(22, 146), ylim=(0, 1), xticks=[25, 50, 75, 100, 125, 143], yticks=np.arange(0, 1.01, 0.2))
ax.grid(alpha=0.18, lw=0.7)
ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.93, edgecolor="none")
save(fig, "repro_diagonal")
print("\nskipped points:", len(skipped), skipped)
print("\nstep_k  orig  pr1234  prv1   (Δp mean = treated − control ± CI half-width, n runs)")
for i, s in enumerate(summ["orig"]):
    print(f"{s[0]:g}  " + "  ".join(f"{r[1]:.2f}±{r[2]:.2f} n{r[3]}" for r in (summ[l][i] for l in LIN)))

# ---- side-by-side 13x13: orig vs pr1234
steps = T.MATRIX_STEPS; idx = {s: i for i, s in enumerate(steps)}; lab = [str(s // 1000) for s in steps]
def matrix(lin):
    M = np.full((13, 13), np.nan); N = np.zeros((13, 13), dtype=int)
    for c in cells:
        if c["lineage"] == lin and c["teacher_step"] in idx and c["student_step"] in idx:
            i, j = idx[c["teacher_step"]], idx[c["student_step"]]
            if c["complete_five_recipes"]: M[i, j] = c["p_target"]
            N[i, j] = c["n_student_runs"]
    return M, N
cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#d9d9d9")
fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.9))
for ax, (lin, title) in zip(axes, [("orig", "Pythia-160M"), ("pr1234", "Modern1234")]):
    M, N = matrix(lin)
    im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap, aspect="equal")
    for i in range(13):
        for j in range(13):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "—", ha="center", va="center", fontsize=7.0,
                    color="white" if np.isfinite(v) and v > 0.55 else "black")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#f79532", linewidth=1.5))
    ax.set(xticks=range(13), xticklabels=lab, yticks=range(13), yticklabels=lab,
           xlabel="Student pretraining checkpoint (k steps)", title=title)
    ax.tick_params(labelsize=9, length=0)
    print(lin, "cells complete:", int(np.isfinite(M).sum()), "runs diag/offdiag:", sorted(set(N[i, i] for i in range(13))), sorted(set(N[i, j] for i in range(13) for j in range(13) if i != j)))
axes[0].set_ylabel("Teacher pretraining checkpoint (k steps)")
bar = fig.colorbar(im, ax=axes, fraction=0.023, pad=0.02, ticks=np.arange(0, 1.01, 0.2))
bar.set_label("p(dog), held-out test", fontsize=10)
save(fig, "ckpt_cliffcross13_orig_vs_pr1234")
