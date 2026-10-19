import json, sys, numpy as np, pandas as pd
from pathlib import Path
from subliminal_learning.analysis import transfer_significance_io as tio
REPO = Path("/home/angela/EleutherAI_subliminal_learning")
SP = Path(sys.argv[1])
animal = "dog"
teachers = json.loads((REPO/"run_logs/registry/dog_top10_kband_configs.json").read_text())["param_tags"]
cells = tio.resolve_cells(REPO/"run_logs/registry/core56_cells.json")
rows, cover = [], []
priors = {}
for fam in ("cross-seed", "cross-data-order"):
    for a in tio.family_arm_set(fam):
        m = tio.load_cell_matrix(animal, a.sender, a.receiver, teachers, cells=cells, ftseed=1)
        recv = tio.base_name(a.receiver)
        if recv not in priors:
            priors[recv] = tio.student_tree_prior(animal, a.receiver)
        treat = np.delete(m.values, m.control_col, axis=1)
        ctrl = m.values[:, m.control_col]
        n_fin = int(np.isfinite(treat).sum())
        cover.append(dict(family=fam, arm=a.arm, cells=len(m.cells), design=len(cells),
                          ctrl_finite=int(np.isfinite(ctrl).sum()), runs=n_fin, of=len(cells)*len(teachers)))
        for i, c in enumerate(m.cells):
            for j, t in enumerate(teachers):
                rows.append(dict(family=fam, sender=a.sender, receiver=a.receiver, arm=a.arm,
                                 teacher=t, cell=c, treatment=treat[i, j], control=ctrl[i],
                                 prior=priors[recv]))
df = pd.DataFrame(rows)
df.to_csv(SP/"asym_runs_dog_core56.csv", index=False)
cov = pd.DataFrame(cover)
cov.to_csv(SP/"asym_coverage.csv", index=False)
print(cov.to_string())
print("priors", priors)
