"""Does the 410m arm show "trait outlives collapse"? Disk-only analysis.

Conventions copied from notebooks/pythia410m_collapse_and_peak.ipynb:
  lineage = OUTPUT_ROOT/students/dog/{student_base}/teacher-pythia-410m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k
  endpoint: {leaf}/preference_eval.json (sets.test.p_target), {leaf}/eval/pile_probe.json (kl_base_student, delta_ce)
  traj:     {leaf}/eval/pref_traj.json (points[].step, points[].test.p_target)
            {leaf}/eval/pile_kl_traj.json (points[].step, points[].kl_base_adapted)
  prior:    {lineage}/base_control.json -> ["test"]["p_target"]  (test_prior, not behav)
  dead:     endpoint kl > DEAD_KL_CUT=4.0 (train_diverged.json => dead, kl=inf, test=0.0)
  anchored: OPERATIVE = endpoint delta_ce > delta_cross from
            OUTPUT_ROOT/analysis/unigram_anchor_pp1.json (410m ~5.47). Delta_cross =
            H_uni - ce_base, so delta_ce > delta_cross is the dimensionally exact cut
            (CE_student > H_uni); kl > delta_cross is kept ONLY as the named proxy
            column dead_anch_klproxy. Trajectory crossings for the anchored cut use the
            KL trajectory as proxy (no delta_ce trajectories exist).
  K axis:   cl.K_of(lr, alpha, rank) with B_LAW/C_LAW; K_rel = K / cl.KSTAR_CALIBRATED
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import labels
from subliminal_learning.analysis.frames import parse_param_tag, read_json
from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.coherence.probe_io import DEAD_KL_CUT

ROOT, OUTPUT_ROOT = bootstrap()
ANIMAL, TEACHER, DATASET = "dog", "ep2_lr5e-4_r32a64", "ex10-20_prefix_n100k"
KSTAR_160, B, C = cl.KSTAR_CALIBRATED, cl.B_LAW, cl.C_LAW
ANCH = read_json(OUTPUT_ROOT / "analysis" / "unigram_anchor_pp1.json")
ARMS = {"same-model": "pythia-410m", "cross-seed": "pythia-410m-seed1"}


def lineage_dir(student_base: str) -> Path:
    return (OUTPUT_ROOT / "students" / ANIMAL / student_base / "teacher-pythia-410m"
            / f"{TEACHER}_tseed1" / DATASET)


def dedup_points(pts, keys):
    """Sort by step, drop duplicate steps keeping last (final step can be written twice)."""
    seen = {}
    for q in sorted(pts, key=lambda q: q["step"]):
        seen[q["step"]] = q
    steps = np.array(sorted(seen))
    out = {"step": steps}
    for name, fn in keys.items():
        out[name] = np.array([fn(seen[s]) for s in steps], float)
    return out


rows = []
for arm, sb in ARMS.items():
    ld = lineage_dir(sb)
    prior = float(read_json(ld / "base_control.json")["test"]["p_target"])
    dcross = float(ANCH["delta_cross"][sb])
    for cell in sorted(ld.glob("ep20_*")):
        pt = parse_param_tag(cell.name)
        for leaf in sorted(cell.glob("ftseed*")):
            rec = dict(arm=arm, student_base=sb, student_tag=cell.name,
                       seed=int(leaf.name.removeprefix("ftseed")),
                       lr=pt["lr"], alpha=pt["alpha"], rank=pt["rank"],
                       leaf=str(leaf.relative_to(OUTPUT_ROOT)),
                       test_prior=prior, delta_cross=dcross)
            diverged = (leaf / "train_diverged.json").exists()
            rec["diverged"] = diverged
            pe, pp = leaf / "preference_eval.json", leaf / "eval" / "pile_probe.json"
            if diverged:
                rec.update(test=0.0, kl=math.inf, delta_ce=math.inf, landed=True)
            elif pe.exists() and pp.exists():
                e, p = read_json(pe), read_json(pp)
                kl = p.get("kl_base_student")
                dce = p.get("delta_ce")
                rec.update(test=float(e["sets"]["test"]["p_target"]),
                           kl=math.inf if kl is None else float(kl),
                           delta_ce=math.inf if dce is None else float(dce),
                           landed=True)
            else:
                rec.update(test=np.nan, kl=np.nan, delta_ce=np.nan, landed=False)

            # trajectories (never interpolate the trait; crossing step from the KL traj's own grid)
            ktraj = leaf / "eval" / "pile_kl_traj.json"
            ptraj = leaf / "eval" / "pref_traj.json"
            rec["has_kl_traj"] = ktraj.exists()
            rec["has_pref_traj"] = ptraj.exists()
            if ktraj.exists():
                kt = dedup_points(read_json(ktraj)["points"], {"kl": lambda q: q["kl_base_adapted"]})
                rec["n_kl_ckpt"] = len(kt["step"])
                rec["kl_traj_end"] = float(kt["kl"][-1])
                for name, cut in (("4", DEAD_KL_CUT), ("anch", dcross)):
                    idx = np.where(kt["kl"] > cut)[0]
                    rec[f"cross_step_{name}"] = int(kt["step"][idx[0]]) if len(idx) else None
                    rec[f"cross_frac_{name}"] = (float(kt["step"][idx[0]] / kt["step"][-1])
                                                 if len(idx) else None)
            if ptraj.exists():
                pts = dedup_points(read_json(ptraj)["points"],
                                   {"p": lambda q: q["test"]["p_target"]})
                rec["n_pref_ckpt"] = len(pts["step"])
                rec["pref_steps_max"] = int(pts["step"][-1])
                for name in ("4", "anch"):
                    cs = rec.get(f"cross_step_{name}")
                    if cs is None:
                        continue
                    post = pts["p"][pts["step"] >= cs]
                    pre = pts["p"][pts["step"] < cs]
                    rec[f"n_ckpt_post_{name}"] = int(len(post))
                    if len(post):
                        rec[f"max_p_post_{name}"] = float(post.max())
                        rec[f"max_ratio_post_{name}"] = float(post.max() / prior)
                        rec[f"any_above_prior_post_{name}"] = bool((post > prior).any())
                        rec[f"end_ratio_traj_{name}"] = float(pts["p"][-1] / prior)
                    if len(pre):
                        rec[f"max_ratio_pre_{name}"] = float(pre.max() / prior)
            rows.append(rec)

df = pd.DataFrame(rows)
df["K"] = cl.K_of(df.lr, df.alpha, df["rank"])
df["K_rel"] = df.K / KSTAR_160
df["ratio"] = df.test / df.test_prior
df["dead4"] = df.kl > DEAD_KL_CUT
# Operative anchored label: delta_ce > Delta_cross = H_uni - ce_base is the dimensionally
# exact cut (CE_student > H_uni). KL is a close proxy on the same excess-over-base scale;
# it stays as an explicitly named comparison column, never the operative label.
df["dead_anch"] = df.delta_ce > df.delta_cross
df["dead_anch_klproxy"] = df.kl > df.delta_cross
df["nonfinite_adapter"] = df.diverged | (df.landed & ~np.isfinite(df.kl))
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative collapse label = anchored (dead_anch)
df["trait_lost"] = labels.label_dead_trait_legacy(df, col="test").fillna(False).astype(bool)

print("=" * 100)
print("INVENTORY (row unit: one ep20 run = cell x ftseed)")
inv = df.groupby("arm").agg(
    n=("leaf", "size"), landed=("landed", "sum"), diverged=("diverged", "sum"),
    null_kl=("nonfinite_adapter", lambda v: int(v.sum()) - int(df.loc[v.index, "diverged"].sum())),
    dead4=("dead4", "sum"), dead_anch=("dead_anch", "sum"),
    dead_anch_klproxy=("dead_anch_klproxy", "sum"),
    kl_traj=("has_kl_traj", "sum"), pref_traj=("has_pref_traj", "sum"),
    prior=("test_prior", "first"), dcross=("delta_cross", "first"))
print(inv.to_string())

# ---------------- Q1: endpoint, among dead runs ----------------
print("\n" + "=" * 100)
print("Q1. ENDPOINT among dead runs: test p(dog) vs receiver test_prior")
q1 = []
for arm, sub in df.groupby("arm"):
    for cut, lab in (("dead4", f"KL>{DEAD_KL_CUT:g}"), ("dead_anch", "anchored")):
        dead = sub[sub[cut]]
        fin = dead[~dead.nonfinite_adapter]  # finite-adapter dead: real endpoint eval exists
        r = fin.ratio
        q1.append(dict(arm=arm, cut=lab, n_dead=len(dead),
                       n_excluded_nonfinite=int(dead.nonfinite_adapter.sum()),
                       n_finite=len(fin),
                       frac_above_prior=float((r > 1).mean()) if len(fin) else np.nan,
                       frac_above_3x=float((r > 3).mean()) if len(fin) else np.nan,
                       median_ratio=float(r.median()) if len(fin) else np.nan,
                       iqr=(round(float(r.quantile(.25)), 2), round(float(r.quantile(.75)), 2)) if len(fin) else None,
                       max_ratio=float(r.max()) if len(fin) else np.nan))
q1 = pd.DataFrame(q1)
print(q1.round(3).to_string(index=False))
print("(160m orig-arm comparator: 87% above prior just past boundary, median 16.5x)")

# ---------------- Q2: checkpoints after the KL crossing ----------------
print("\n" + "=" * 100)
print("Q2. CHECKPOINTS at-or-after the run's own Pile-KL crossing (crossing from pile_kl_traj, cut both ways;")
print("    dead-run selection uses the endpoint labels — anchored = delta_ce cut; the anchored CROSSING step")
print("    is necessarily the KL-proxy crossing, since only KL trajectories exist)")
q2 = []
for arm, sub in df.groupby("arm"):
    for name, lab, cut in (("4", f"KL>{DEAD_KL_CUT:g}", "dead4"), ("anch", "anchored", "dead_anch")):
        dead = sub[sub[cut]]
        cov = dead[dead.has_kl_traj & dead.has_pref_traj]
        crossed = cov[cov[f"cross_step_{name}"].notna()]
        post = crossed[crossed.get(f"n_ckpt_post_{name}", pd.Series(dtype=float)).notna()
                       & (crossed.get(f"n_ckpt_post_{name}", 0) > 0)]
        row = dict(arm=arm, cut=lab, n_dead=len(dead),
                   n_with_both_trajs=len(cov),
                   n_crossing_in_traj=len(crossed),
                   n_with_post_ckpts=len(post))
        if len(post):
            row.update(
                frac_any_above_prior_post=float(post[f"any_above_prior_post_{name}"].mean()),
                median_max_ratio_post=float(post[f"max_ratio_post_{name}"].median()),
                median_max_ratio_pre=float(post[f"max_ratio_pre_{name}"].median())
                if f"max_ratio_pre_{name}" in post else np.nan,
                median_post_over_pre=float((post[f"max_ratio_post_{name}"]
                                            / post[f"max_ratio_pre_{name}"]).median())
                if f"max_ratio_pre_{name}"in post else np.nan,
                median_cross_frac=float(post[f"cross_frac_{name}"].median()),
                median_n_post=float(post[f"n_ckpt_post_{name}"].median()),
                median_end_ratio_traj=float(post[f"end_ratio_traj_{name}"].median()))
        q2.append(row)
q2 = pd.DataFrame(q2)
print(q2.round(3).to_string(index=False))

# distribution detail for the KL>4 cut
for arm, sub in df.groupby("arm"):
    post = sub[sub.dead4 & sub.has_kl_traj & sub.has_pref_traj
               & sub.get("n_ckpt_post_4", pd.Series(dtype=float)).notna()]
    post = post[post.n_ckpt_post_4 > 0]
    if not len(post):
        continue
    mr = post.max_ratio_post_4
    print(f"\n  {arm} (KL>4 cut, n={len(post)} dead w/ post-crossing trait ckpts):")
    print(f"    max p/prior AFTER crossing: p10 {mr.quantile(.1):.2f}  median {mr.median():.2f}  "
          f"p90 {mr.quantile(.9):.2f}  frac>1: {(mr > 1).mean():.2f}  frac>3: {(mr > 3).mean():.2f}")
    if "max_ratio_pre_4" in post:
        pr = post.max_ratio_pre_4.dropna()
        print(f"    pre-crossing peak p/prior:  p10 {pr.quantile(.1):.2f}  median {pr.median():.2f}  "
              f"p90 {pr.quantile(.9):.2f}")

# ---------------- Q3: 50% crossings on the K axis ----------------
print("\n" + "=" * 100)
print(f"Q3. 50pct-crossing doses (cl.kstar_logistic, 160m exponents b={B} c={C}; K in units of K*_160m={KSTAR_160:.2g})")
q3 = []
for arm, sub in df.groupby("arm"):
    sub = sub[sub.landed]
    fits = {}
    for lab, col in (("dead_kl4", "dead4"), ("dead_anchored", "dead_anch"), ("trait_lost", "trait_lost")):
        d = sub.dropna(subset=["kl"] if col != "trait_lost" else ["test"]).assign(
            collapsed=lambda x: x[col].astype(int))
        if d.collapsed.nunique() < 2:
            fits[lab] = None
            continue
        f = cl.kstar_logistic(d, B, C)
        fits[lab] = f
        q3.append(dict(arm=arm, label=lab, n=f["n"], n_pos=f["n_dead"],
                       K50_rel=f["kstar"] / KSTAR_160, se_log=f["se_log"], err=f["err"]))
    for dl in ("dead_kl4", "dead_anchored"):
        if fits.get(dl) and fits.get("trait_lost"):
            q3.append(dict(arm=arm, label=f"RATIO trait_lost / {dl}",
                           K50_rel=fits["trait_lost"]["kstar"] / fits[dl]["kstar"]))
q3 = pd.DataFrame(q3)
print(q3.round(3).to_string(index=False))
print("(160m orig arm comparator: ratio 2.0 — trait lost at 2x the dose at which integrity dies)")

# ---------------- write the per-run table ----------------
out = OUTPUT_ROOT / "analysis" / "p410m_trait_past_collapse.json"
payload = {
    "kind": "p410m_trait_past_collapse",
    "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "conventions": {
        "dead": f"endpoint pile_probe kl_base_student > {DEAD_KL_CUT} (train_diverged => dead, kl=inf)",
        "anchored": "OPERATIVE: endpoint delta_ce > delta_cross(receiver) = H_uni - ce_base "
                    "(dimensionally exact: CE_student > H_uni). kl > delta_cross is the "
                    "PROXY, kept only as column dead_anch_klproxy; anchored trajectory "
                    "crossings (cross_step_anch) use the KL trajectory as proxy because "
                    "no delta_ce trajectories exist.",
        "anchored_cut_per_base": {sb: float(ANCH["delta_cross"][sb]) for sb in ARMS.values()},
        "trait_lost": f"endpoint test p < {cl.DEAD_SEED:g} (legacy trait cut, analysis.labels."
                      "label_dead_trait_legacy; prior/e^3 rule removed 2026-08-26)",
        "prior_source": "{lineage}/base_control.json ['test']['p_target']",
        "K": f"lr*alpha^{B}*rank^{C}; K_rel = K/{KSTAR_160} (cl.KSTAR_CALIBRATED)",
        "crossing": "first pile_kl_traj checkpoint with kl_base_adapted > cut, on the traj's own step grid;"
                    " post-crossing trait = pref_traj checkpoints with step >= crossing step (no interpolation)",
        "files_read": ["preference_eval.json", "eval/pile_probe.json", "eval/pile_kl_traj.json",
                       "eval/pref_traj.json", "base_control.json", "analysis/unigram_anchor_pp1.json"],
    },
    "summaries": {"inventory": inv.reset_index().to_dict("records"),
                  "q1_endpoint": q1.to_dict("records"),
                  "q2_checkpoints": q2.to_dict("records"),
                  "q3_k50": q3.to_dict("records")},
    "runs": [{k: ("inf" if isinstance(v, float) and math.isinf(v)
                  else None if isinstance(v, float) and math.isnan(v) else v)
              for k, v in r.items()}
             for r in df.drop(columns=["student_base"]).to_dict("records")],
}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return "inf" if math.isinf(o) else None if math.isnan(o) else o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(_clean(payload), indent=1, allow_nan=False))
tmp.replace(out)
print(f"\nwrote {out}  ({len(df)} run rows)")
