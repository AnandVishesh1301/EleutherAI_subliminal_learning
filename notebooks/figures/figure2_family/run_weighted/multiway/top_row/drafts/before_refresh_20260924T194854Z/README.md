# Two-panel Figure 2

`transfer_family_complete_multiway_top_row.{png,pdf,svg}` contains only the Pythia win-probability panel and the pooled raw-difference percentile panel. The four relationship names match between A's x-axis and B's legend. A omits sample counts, block counts and grid labels. Its animal order is cat, dog, lion, pig. The animal hues follow `figure1_boxplot.png`, with moderately saturated shades: cat #367EAA, dog #DC7134, lion #3B9569 and pig #C15B99. Fills are solid, with dark outlines matching panel B. The redder orange and magenta pink distinguish the dog and pig bars from B’s yellow and red. The pooled bar comes first in a deeper lavender (#8F63B8). B uses gold, crimson, purple and cyan for the four relationship families and starts at zero.

The data and multiway bootstrap intervals are read directly from `../complete_multiway_summary.csv`; no estimates are recomputed. The zero lower bound clips the very small negative different-init/same-data-order median (−0.000035) and the part of its interval below zero. The CSV retains those values.

From the repository root:

```sh
uv run python - <<'PY'
from scripts.plot_transfer_family_multiway import DEFAULT_OUT, export_top_row
export_top_row(DEFAULT_OUT / 'complete_multiway_summary.csv', DEFAULT_OUT / 'top_row')
PY
```

The pooled A-panel bars use a light-gray fill with diagonal `//` hatching to distinguish pooled results. Animal bars retain the brighter cat, dog, lion, and pig palette; hatching identifies pooling only and has no collapse-status meaning.
