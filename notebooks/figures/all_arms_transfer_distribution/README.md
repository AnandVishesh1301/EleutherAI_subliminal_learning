# All-arms transfer distributions

All five figures were rebuilt after cat original→Modern1234 completed, using a
fresh native source scan on **2026-09-26, 05:42:20–05:47:07 UTC**.
The cohort now contains **63,744 matched runs in 1,495 complete blocks**,
compared with 59,444 runs and 1,355 blocks in the preceding snapshot.
This includes newly finished results and complete core30 data-order blocks
admitted by the approved grid correction. [See exactly what changed](REFRESH_CHANGES.md).

## Figures and audit

- [Clean overview](transfer_family_multiway_raw_percentiles.png): 16 relationship
  rows, with columns for median, 75th percentile, 95th percentile, and maximum.
- [Overview with counts](transfer_family_multiway_raw_percentiles_with_metadata.png):
  the same results with n, complete-block counts, and core-grid labels.
- [Pythia detail](transfer_family_multiway_raw_percentiles_panel1.png): six Pythia
  relationships, with pooled, cat, dog, lion, and pig bars.
- [Size and architecture detail](transfer_family_multiway_raw_percentiles_panel2.png):
  four directional Pythia 160M/410M comparisons and two Pythia/OLMo directions,
  all for dog.
- [OLMo detail](transfer_family_multiway_raw_percentiles_panel3.png): elephant
  same-model, same-init/different-corpus, different-init/different-corpus, and
  different-init/same-corpus conditions.
- [Condition, arm, and seed audit](ARM_AUDIT.md): each condition and every directed
  animal/arm combination in the block report, including zero-retained arms.
  [Arm CSV](arm_seed_audit.csv) preserves source model identifiers and requested
  versus retained seed values; [condition CSV](condition_audit.csv) records
  counts, changes from the preceding snapshot, and point estimates.
- [Latest changes](REFRESH_CHANGES.md): per-arm additions and before/after statistics;
  `refresh_endpoint_changes.csv` distinguishes newly included, removed, and rewritten
  endpoints, and `refresh_statistic_changes.csv` records every plotted group.

Every figure also has matching PDF and SVG exports. Both overviews preserve the
compact left margin and current labels, with Modern Reproduction shortened.
The figures use the ColorBrewer Paired palette without hatching: light blue for
pooled, blue for cat, orange for dog, green for lion, pink for pig, and purple for
elephant. A category without a complete block is unavailable, not a zero effect.

## Cohort and seeds

The outcome is `p_student - p_control`, without receiver-prior normalization or
collapse filtering. Keep a teacher × animal × directed-arm × teacher/student-seed
block only when every canonical cell has a finite matched student/control pair.
The canonical grids are core30 (same-model, different-init/same-order,
reproductions and Pythia-receiver size/architecture), core56 (established legacy different-order arms), and core35 (OLMo receivers).
The new cat/lion/pig ds1→original/ds2/ds3 data-order arms use core30, while
existing complete core56 arms remain included. Grid assignments are explicit,
not inferred from the number of observed cells. Teacher-selection cells are included.
Cross-architecture comparisons use dog only.

Every retained arm has **one student fine-tuning seed, ftseed=1**; controls use the
same student seed. Pythia teachers have one teacher fine-tuning seed, tseed=1;
OLMo teachers have one, tseed=0. These are the included seeds, not a census of all
seed directories on disk. Pretraining seed variants define distinct arms;
teacher configurations and student hyperparameter cells are not independent
fine-tuning seeds. The audit separates these concepts.

Legacy/re-pretraining block coverage is based on observed matched endpoints.
Registry block coverage additionally includes declared teachers with no matches.
Thus missing legacy blocks may be absent from the report, whereas declared
registry blocks can explicitly report zero matches. There are 8 incomplete
block reports in this snapshot; none contributes partial cells to the figure.
The old OLMo ctrl20 pilot does not contribute complete core35 blocks.

New support includes the cat original→Modern1234 arm, additional dog ds1/ds2/ds3
same-model blocks, pig wc2→wc1, and elephant C4→Dolma-seed2 and C4→DCLM-seed2.
The new cat/lion/pig ds1 data-order arms are read on their approved core30 grid.
Cat Modern1234→original was already present and remains included; both cat
reproduction directions now have all 300 matched cells. **wc1→wc1 is excluded.**
See the arm audit for the exact current completeness of each data-order direction.

## Pooling and confidence intervals

Each retained matched run has equal initial weight. Animals and arms with more
retained runs contribute more total weight. The 95% pointwise intervals use
2,000 two-way cluster (pigeonhole) bootstrap draws, random seed 0. Source-teacher
identities `(sender, animal, teacher)` and global student-cell identities are
sampled independently with replacement; their integer multiplicities multiply.
Student/control pairs remain paired. Median, p75, and p95 are recomputed for each
draw. The maximum is observed and has no CI. Animals and arms are not separately
resampled, and measured probabilities are held fixed.

## Reproduction and provenance

```bash
uv run python -m scripts.plot_transfer_family_multiway --all-arms --n-bootstrap 2000 --seed 0
```

The default destination is `notebooks/figures/all_arms_transfer_distribution/`.
An explicit `--out-dir` is still respected. Other Figure 2 variants retain their
existing destinations.

This command re-executes the native source loaders. Legacy data come from eight
code cells of `transfer_receiver_family.ipynb`, through `load_law_runs` and its
mtime/size-validated record cache. Registry/re-pretraining arms use
`load_cell_matrix` caches keyed on arguments and leaf-file signatures. New and
rewritten evaluations invalidate the relevant cached data. The current refresh
waited for the final cat original→Modern1234 cell and then rescanned every condition; it did not reuse the previous summary or cohort CSV.
Repeated reads in a long-lived process can reuse a leaf signature for up to 120 seconds; a fresh CLI process starts with an empty signature cache.
The newly loaded in-memory tuple was reused for rendering and the independent
arm audit, so those artifacts describe the same cohort. Runs may finish after
the recorded scan interval.

All-arms mode includes Rerun1234 (`prv1`,
`pythia-160m-pilereprov1-seed1234`) and Modern1234 (`pr1234`,
`pythia-160m-pilerepro-seed1234`) as separate reproductions. Modern arms come
from `run_logs/repretrain/repretrain_significance_arms.json`. OLMo same-model
opts into the canonical Dolma, DCLM, and C4 core35 campaign; its resolved
in-memory registry is saved in `olmo_same_model_registry.json`.

`transfer_family_multiway_raw_summary.csv` contains the plotted quantiles, CIs,
counts, and shared-summary win statistics. `complete_matched_pairs.csv`,
`complete_blocks.csv`, `complete_cohort_exclusions.csv`,
`complete_registry_coverage.csv`, and `complete_cohort_counts.csv` record
cohort selection. `complete_snapshot.json` records the actual source scan
interval, bootstrap settings, and source-manifest hashes. All retained endpoints
were checked for uniqueness, valid probabilities, exact canonical cell sets,
standard seeds, and agreement between arm, block, and condition totals.

## Previous versions

The immediately preceding figures, cohort, and documentation are preserved in
`drafts/before_full_refresh_20260926T045425Z`, with SHA-256 checksums. Earlier balanced, reproduction-label,
and OLMo-overview drafts were moved with the family and remain under `drafts/`.
The original balanced artifacts use different weighting and cohort rules;
changes from those old plots cannot be attributed only to new results.
