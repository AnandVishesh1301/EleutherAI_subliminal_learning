"""Re-render notebooks/figures/drafts/fig2_A_null_strip.png from the live headline families.

One strip per headline family row: the whole arm-relabelling null (grey, in % of the row's runs)
with its 95th percentile dashed and the observed exceedance count as the coloured line -- the
`exceed.` / `null 95th` columns of the headline table, drawn (`transfer_headline.null_strip_figure`).
Same seed and draw count as the notebook, so every permutation is a `stats.memo` hit.
"""
import json
import matplotlib
matplotlib.use("Agg")
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis.setup import bootstrap

REPO, _ = bootstrap()
ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/{a}_top10_kband_configs.json").read_text())["param_tags"]
            for a in ANIMALS}
FAMILIES = th.headline_families(REPO, TEACHERS)
fig = th.null_strip_figure(FAMILIES, n_perm=20000, seed=0)
out = REPO / "notebooks/figures/drafts/fig2_A_null_strip.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out, "| families:", len(FAMILIES))
