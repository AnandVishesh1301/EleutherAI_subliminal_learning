#!/usr/bin/env python
"""Refresh points.csv from finished student leaves. OLMo panels (b: DCLM -> DCLM seed 2, c: DCLM -> C4 seed 2) come
from run_logs/campaigns/newdom_dclm_blue_160m_20261110 (6 DCLM teachers x 20 core35 cells per trait; blue's
lr 5e-3 teachers come from 200k/300k-prompt pools and live under ex10-20_prefix_n200k/_n300k, handled by
make_jobs.dataset_tag). Panel d's Pythia -> Dolma arm for maple / pine / yellow / blue comes from
run_logs/campaigns/newdom4_to_olmo_20261110 (10 canonical Pythia-160m teachers x the same 20 cells). Every other row
(panel a, the rest of panel d) is final and kept as it is. Columns: panel, arm, trait, teacher, cell,
student p(target), matched control p(target), base prior."""
import csv, importlib.util, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
B = ROOT / "run_logs/campaigns/newdom_dclm_blue_160m_20261110"
sp = importlib.util.spec_from_file_location("mj", B / "make_jobs.py"); mj = importlib.util.module_from_spec(sp); sp.loader.exec_module(mj)
TRAITS = ["oak", "maple", "pine", "purple", "yellow", "blue"]
PANEL = {"dd-1b-dclm-seed2": ("b) OLMo cross-seed", "DCLM → DCLM seed2"), "dd-1b-c4-seed2": ("c) OLMo cross-corpus", "DCLM → C4 seed2")}
D4 = ROOT / "run_logs/campaigns/newdom4_to_olmo_20261110"
sp4 = importlib.util.spec_from_file_location("mj4", D4 / "make_jobs.py"); mj4 = importlib.util.module_from_spec(sp4); sp4.loader.exec_module(mj4)
XARCH = ("d) cross-architecture", "Pythia → Dolma")

def p_of(f):
    return json.load(open(f))["sets"]["test"]["p_target"]

def prior(recv, trait):
    d = json.load(open(mj.OUTPUT_ROOT / "analysis" / f"base_eval_{mj.SEED2[recv][1]}_{mj.DOMAIN[trait]}_preference_test.json"))[trait]
    return d.get("p_target", float(np.mean(d["p_per_prompt"])))

def prior_dd1b(trait):
    d = json.load(open(mj4.OUTPUT_ROOT / "analysis" / f"base_eval_dd1b_{mj4.DOMAIN[trait]}_preference_test.json"))[trait]
    return d.get("p_target", float(np.mean(d["p_per_prompt"])))

keep = [r for r in csv.DictReader(open(HERE / "points.csv")) if not r["panel"].startswith(("b)", "c)"))
        and not ((r["panel"], r["arm"]) == XARCH and r["trait"] in mj4.ANIMALS)]
rows = []
for recv, (panel, arm) in PANEL.items():
    for trait in TRAITS:
        pr = prior(recv, trait)
        for t in mj.teachers():
            for c in mj.cells20():
                sf = mj.olmo_student_leaf(trait, recv, t, c) / "preference_eval.json"
                cf = mj.olmo_control_leaf(recv, c) / "eval" / trait / "preference_eval.json"
                if sf.exists() and cf.exists():
                    rows.append({"panel": panel, "arm": arm, "trait": trait, "teacher": t, "cell": c,
                                 "student": p_of(sf), "control": p_of(cf), "prior": pr})
for trait in mj4.ANIMALS:
    pr = prior_dd1b(trait)
    for t in mj4.teachers():
        for c in mj4.cells():
            sf = mj4.treatment_leaf(trait, "pythia-160m", t, c) / "preference_eval.json"
            cf = mj4.control_leaf("pythia-160m", c) / "eval" / trait / "preference_eval.json"
            if sf.exists() and cf.exists():
                rows.append({"panel": XARCH[0], "arm": XARCH[1], "trait": trait, "teacher": t, "cell": c,
                             "student": p_of(sf), "control": p_of(cf), "prior": pr})
with open(HERE / "points.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["panel", "arm", "trait", "teacher", "cell", "student", "control", "prior"])
    w.writeheader(); w.writerows(keep); w.writerows(rows)
from collections import Counter
cnt = Counter((r["panel"][:2], r["arm"], r["trait"]) for r in rows)
print("rebuilt rows:", {f"{p} {a} {t}": n for (p, a, t), n in sorted(cnt.items())})
print("Pythia → Dolma rows:", {t: cnt[("d)", t)] for t in mj4.ANIMALS})
