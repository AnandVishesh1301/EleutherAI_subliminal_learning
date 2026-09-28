# Two-panel Figure 2

Both layouts are exported as PNG, PDF, and SVG in this directory:

- `transfer_family_complete_multiway_top_row`: panel A groups win probabilities by
  relationship and animal; panel B groups raw transfer magnitudes by statistic,
  with colors identifying the four relationships.
- `transfer_family_complete_multiway_top_row_families`: the same panel A and the
  same estimates in B, with both panels using the four relationships on the
  x-axis. Within each B family, bars are median, 75th percentile, 95th percentile,
  and maximum. Colors identify those statistics: gold `#E9B000`, red `#D7263D`,
  purple `#7B2CBF`, and cyan `#00B8D9`, respectively.

The relationship order is Same-model; Same init, different data order; Different
init, same data order; Different init, different data order. Labels split at the
comma. Neither layout displays sample counts, block counts, grid labels, or a
bottom caption.

Panel A uses pooled, cat, dog, lion, pig order. Its pooled bars have light-gray
fill and diagonal hatching to identify pooling. Animal colors are blue `#168BCE`,
orange `#F06418`, green `#16A05D`, and pink `#DE4FA6`, respectively. Fills are opaque
and outlines match panel B. Hatching has no collapse-status meaning.

## Data and intervals

The source scan ran on 2026-09-24 from 19:52:30 to 19:56:43 UTC. It retained
48,548 matched runs in 1,022 complete blocks across the four displayed families:

| Relationship | Matched runs | Complete blocks | Design |
| --- | ---: | ---: | --- |
| Same-model | 5,160 | 172 | core30 |
| Same init, different data order | 11,760 | 210 | core56 |
| Different init, same data order | 4,860 | 162 | core30 |
| Different init, different data order | 26,768 | 478 | core56 |

The earlier paper figure is preserved in `drafts/before_refresh_20260924T194854Z/`.
The refreshed paper cohort adds 566 pairs (510 in different init/same data order,
56 in different init/different data order) compared with the previous paper
summary. It adds 360 pairs compared with the earlier all-arms snapshot.

Both panels and both layouts use this directory's `complete_multiway_summary.csv`.
It is recomputed from a fresh source scan through the canonical complete-cohort
loader, including the legacy notebook record cache and native signature-aware
registry caches. The summaries use equal weight per matched run, standard student
ftseed 1, complete designated core30/core56 blocks, and all collapse states.

Intervals are 95% pointwise two-way cluster (pigeonhole) bootstrap intervals with
2,000 draws and seed 0. Source-teacher and global student-cell identities are
resampled independently; their multiplicities multiply while student/control
pairs stay together. Win probabilities, medians, and the 75th/95th percentiles
receive intervals. Maximum is the observed maximum without an interval.

Panel B starts at zero, as in the existing paper figure. Any negative estimates
or lower interval endpoints remain in the CSV even when clipped visually.
Matched-pair, block, exclusion, count, and snapshot files record the cohort and
refresh provenance. Earlier visuals are retained under `drafts/`.

## Render from the refreshed summary

From the repository root:

```sh
uv run python - <<'PYTHON'
from scripts.plot_transfer_family_multiway import DEFAULT_OUT, export_top_row
out = DEFAULT_OUT / 'top_row'
summary = out / 'complete_multiway_summary.csv'
export_top_row(summary, out)
export_top_row(summary, out, quantile_layout='families')
PYTHON
```

Rendering from this summary does not rerun source loading or the bootstrap.

## Size and OLMo magnitude figure

[Size/OLMo PNG](transfer_family_complete_multiway_size_olmo.png) (also PDF/SVG)
uses the same percentile colors, with four dog cross-size families on the left
and elephant OLMo families on the right. OLMo same-model has a reserved category
and will gain bars when canonical core35 blocks are complete. The independent
fresh-source cohort and audits live in [size_olmo_data/](size_olmo_data/README.md).
