# Size, architecture, and corpus transfer distributions

`olmo_size_distribution.{png,pdf,svg}` contains six dog size/architecture
conditions in panel A and four elephant OLMo conditions in panel B. The
percentile palette is gold, red, purple, cyan. The paper caption is in
[caption.md](caption.md).

The source scan finished on 2026-09-26 at 05:47:07 UTC. This export reuses the
same audited summary as `../all_arms/`, with 2,000
multiway bootstrap draws (random seed 0), equal weight per matched run, all
collapse states, and complete canonical grids at student ftseed 1.

Panel A contains 330, 300, 300, 300, 560, and 330 matched runs respectively;
panel B contains 1,050, 2,100, 2,100, and 1,050. The plotted subset contains
8,420 runs across 248 complete teacher-by-arm blocks. Exact statistics and
cohort sizes are in `plotted_cohort_counts.csv`. The full refreshed summary,
matched pairs, completeness/exclusion/coverage tables, and resolved OLMo
same-model registry accompany it. `complete_snapshot.json` records provenance
and artifact hashes. Earlier exports are preserved under `drafts/`.

The source-teacher cluster is `(sender, animal, teacher configuration)`;
the second cluster is the global student hyperparameter configuration. Each
bootstrap independently samples both sets of cluster IDs with replacement;
each observed matched pair is repeated by the product of its two multiplicities.
Student and control measurements stay paired. Quantiles use linear interpolation.
Pointwise intervals are the 2.5th and 97.5th percentiles of bootstrap estimates.
The observed maximum has no CI. Models, animals, and measured probabilities
are conditioned on; this is not a bootstrap over new model seeds or evaluation
responses.

Both panels retain a zero lower y-limit and share an upper limit that includes
every positive bar and CI endpoint. Negative estimates or CI portions remain
in the CSV but fall below the displayed axis (the fourth dog median is about
−0.00190).

To re-export the saved statistics without reloading experiments:

```python
from pathlib import Path
import pandas as pd
from notebooks.figures.figure2_family.plot_transfer_family_multiway import export_size_olmo
out = Path("notebooks/figures/figure2_family/olmo")
table = pd.read_csv(out / "complete_multiway_summary.csv")
export_size_olmo(table, None, out, stem="olmo_size_distribution")
```

To rescan experiment records and rebuild published panels, use
`uv run python -m notebooks.figures.figure2_family.plot_transfer_family_multiway --publish`.
