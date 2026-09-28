# Complete-grid multiway Figure 2

The new combined figure puts Pythia win probabilities at upper left and pooled raw-difference percentiles at upper right. Size/architecture comparisons and OLMo corpus/seed comparisons occupy two panels in the same bottom row. A second combined figure uses dog-only percentiles at upper right; its other panels are unchanged. The standalone win figure has a full-width Pythia panel above the same two smaller panels.

- `transfer_family_complete_multiway_combined.{png,pdf,svg}`
- `transfer_family_complete_multiway_combined_dog.{png,pdf,svg}`
- `transfer_family_complete_multiway_win.{png,pdf,svg}`

The older figures and their numerical tables are preserved. These new files replace neither the earlier balanced estimates nor the initial available-row multiway comparison.

## Population and counts

A block is one teacher configuration, animal, directed sender→receiver arm, teacher seed and student fine-tuning seed. Retain a block only when **every canonical cell** has a finite student probability and a finite matched control probability at the designated seed. A partially finished teacher contributes no rows. Other complete teachers in that arm remain eligible. Identical duplicate endpoints count once; conflicting duplicates raise an error.

Student fine-tuning seed is **ftseed 1 throughout**. Pythia teacher seed is tseed 1, whereas OLMo teacher seed is tseed 0. Teacher seed and student seed are different fields. Additional student seeds cannot repair an incomplete ftseed-1 block. The historical legacy loader already reads seed 1 before constructing its paired frame; the new selector records that fixed provenance explicitly.

Use core30 for Pythia same-model, different-init/same-data-order, size comparisons, and OLMo→Pythia. Use core56 for Pythia shared-init/different-data-order and cross-seed. Use core35 for Pythia→OLMo and OLMo-only comparisons. Every cell in these manifests is eligible, including previously excluded teacher-selection cells. The selected teacher pools remain those of the original notebook and registries. All collapse states, including dead treatments and controls, remain included. Cross-architecture comparisons remain dog-only, as requested.

Snapshot: 2026-09-23. The recorded canonical complete-block counts are:

| Relationship | Grid | Complete teacher blocks | Matched runs |
|---|---|---:|---:|
| Pythia same model | core30 | 166 | 4,980 |
| Pythia shared init, different data order | core56 | 210 | 11,760 |
| Pythia different init, same data order | core30 | 76 | 2,280 |
| Pythia cross seed | core56 | 474 | 26,544 |
| 160M→410M, original-seed lineage | core30 | 11 | 330 |
| 410M→160M, original-seed lineage | core30 | 10 | 300 |
| 160M→410M, cross-seed lineage | core30 | 10 | 300 |
| 410M→160M, cross-seed lineage | core30 | 10 | 300 |
| Pythia→OLMo | core35 | 16 | 560 |
| OLMo→Pythia | core30 | 11 | 330 |
| OLMo same init, different corpus | core35 | 30 | 1,050 |
| OLMo different init, different corpus | core35 | 20 | 700 |
| OLMo cross seed, same corpus | core35 | 0 | 0 |

The OLMo cross-seed category has matched endpoints on smaller ctrl20/core30 designs, but no complete core35 teacher block; its empty position denotes unavailable support, not a zero win probability. Counts need not be multiples of ten teachers: for example, the original 160M→410M arm declares eleven teachers.

Completeness is an availability condition, not a collapse screen. Results describe the retained complete designs and do not impute unfinished runs.

## Estimates and uncertainty

Each retained matched student/control run has equal starting weight. Therefore animals or arms with more retained runs contribute more to a pooled estimate. Animal-specific bars use the same rule restricted to that animal. The win probability is the fraction with `p_student > p_control`; ties are not wins. The percentile bars are ordinary linear-interpolated quantiles of `p_student − p_control`, without prior normalization.

The earlier balanced win chart used equal animal/arm weights and crossed exponential multiplier weights. The new figure uses the same **multiway (pigeonhole) resampling procedure** as the chosen run-weighted percentile figure, applied to the corrected population. The earlier procedure was a different estimator and resampling scheme, rather than an arithmetic error.

For each displayed statistic:

1. Identify the T distinct source teachers by `(sender, animal, teacher)` and C distinct student configurations by `student_tag`. The cohort has a single prescribed teacher seed for each source identity and student ftseed 1.
2. Draw T teacher identities with replacement, and independently draw C configuration identities with replacement. A teacher used across receivers retains the same draw count; a configuration reused across teachers or arms also retains the same draw count.
3. Give each existing matched row the product of its teacher count and configuration count. This is equivalent to duplicating that row by the resulting integer frequency. Do not create missing intersections or rebalance animals and arms.
4. Recompute the win fraction or the median/75th/95th percentile. For quantiles, integer-frequency interpolation exactly matches applying NumPy's ordinary linear quantile to the duplicated rows.
5. Repeat 2,000 times with RNG seed 0. Plot the 2.5th and 97.5th percentiles of the replicate statistics as nominal 95% pointwise intervals. The observed maximum has no confidence interval.

Empty intersection draws are counted and redrawn with a bounded retry limit. No separate animal or arm resampling is performed. The measured probabilities, selected model families and teacher selection procedure are not re-estimated. These are configuration/teacher resampling intervals; they do not model evaluation-prompt noise or establish generalization to new model families. The multiway resampling construction follows [Owen's pigeonhole bootstrap](https://arxiv.org/abs/0712.1111); its theoretical guarantees are assumption-dependent and do not by themselves validate finite-sample coverage for these quantiles.

## Reproduction and records

```sh
uv run python -m scripts.plot_transfer_family_multiway --n-bootstrap 2000 --seed 0
```

The command reloads current results, so counts can increase as unfinished experiments complete. `complete_snapshot.json` records the snapshot and manifest hashes used here. `complete_matched_pairs.csv` records every retained pair. `complete_blocks.csv` records required and missing canonical cells per block, including registry teachers with no matches. `complete_cohort_exclusions.csv` records rejected input rows. `complete_registry_coverage.csv` records source coverage. `complete_cohort_counts.csv` compares available and retained counts. `complete_multiway_summary.csv` contains every plotted estimate and confidence interval.

The new script, selector, registry option, and existing bootstrap/plotting paths passed 95 focused tests. The registry loader's default selection-cell exclusions remain unchanged for older callers; this entrypoint explicitly requests all cells.
