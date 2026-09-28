# All-arms raw transfer percentiles

The original 15 families were refreshed on 2026-09-24; the OLMo same-model
row was added from a fresh source scan on 2026-09-25. The raw outcome is
`p_student - p_control` for each matched student/control pair. No receiver-prior
division or collapse filtering is used. The refreshed cohort contains **55,928
matched runs in 1,247 complete blocks**; 82 incomplete block reports were excluded.
“All arms” means all registered relationship categories, not all incomplete runs.

## Figures

Both overview versions and all three detail panels have PNG, PDF, and SVG exports.

- [Clean overview](transfer_family_multiway_raw_percentiles.png): all 16 relationship
  categories as rows, with columns for median, 75th percentile, 95th percentile,
  and observed maximum. Pythia rows pool the available animals; the other rows use
  their single target animal. Counts and grid labels are omitted. The overall
  title is removed, labels break after commas and size directions, and
  “different” is abbreviated to “diff”. The size subtitles read “same data order”
  and “diff data order”.
  The Modern1234 row is labelled “Modern Reproduction”. Both overview canvases
  fit their labels with a small left padding instead of a large blank margin.
- [Detailed overview](transfer_family_multiway_raw_percentiles_with_metadata.png):
  identical estimates and intervals, retaining n, complete-block counts, and the
  core30/core56/core35 design for each row.
- [Panel 1](transfer_family_multiway_raw_percentiles_panel1.png): six Pythia 160M
  relationships, with pooled, cat, dog, lion, and pig bars. These are same-model,
  same initialization/different data order, different initialization/same data
  order, different initialization/different data order, Rerun1234 ↔ original, and
  Modern1234 ↔ original. The two reproduction rows represent distinct models.
- [Panel 2](transfer_family_multiway_raw_percentiles_panel2.png): dog transfers in
  both directions between Pythia 160M and 410M, separately for original-seed and
  cross-seed comparisons, plus both directions between Pythia 160M and OLMo 1B.
- [Panel 3](transfer_family_multiway_raw_percentiles_panel3.png): elephant transfers
  among OLMo corpus/seed variants: same init/different corpus, different init/
  different corpus, and different init/same corpus.

Colors come from the ColorBrewer Paired palette: light blue for pooled, blue for
cat, orange for dog, green for lion, pink for pig, and purple for elephant. Bars
have solid fills without hatching. A missing complete category is marked as
unavailable, never displayed as a zero effect. Counts refer to matched runs and
retained complete teacher × animal × directed-arm blocks.

The 2026-09-25 edit updates both overviews and their numerical audit. The three
detail-panel files retain their 2026-09-24 rendering. A full rerun of the command
below also regenerates those panels with the new OLMo same-model category.

## Cohort and intervals

Only complete designated canonical grids are retained: core30 for same-model,
different-init/same-data, both reproductions, and Pythia-receiver size/architecture
comparisons; core56 for legacy cross-data-order and cross-seed; core35 for
OLMo-receiver registry arms. Standard seeds follow the complete-cohort selector
and registry declarations (student ftseed 1 in this snapshot; OLMo teacher tseed 0
is a different setting). All canonical cells, including teacher-selection cells,
are included. Cross-architecture comparisons show dog only. These are descriptive
summaries of the tested settings, not held-out teacher-selection evaluations.

Each matched run has equal initial weight. Animal and arm totals are not balanced.
The 95% pointwise percentile intervals use 2,000 two-way cluster (pigeonhole)
bootstrap replicates, seed 0. Source-teacher identities `(sender, animal, teacher)`
and global student-cell identities are independently sampled with replacement;
their integer repetition counts multiply. Student and control remain paired.
The median, p75, and p95 are recomputed for each replicate. The maximum is the
observed maximum and has no confidence interval. Animal/arm dimensions are not
separately resampled; measured probabilities are fixed.

## Reproduction and freshness

```bash
uv run python -m scripts.plot_transfer_family_multiway --all-arms --n-bootstrap 2000 --seed 0
```

The command starts a fresh source load; it does not regenerate from the summary
CSV. Legacy data come from eight data cells of `transfer_receiver_family.ipynb`,
via `load_law_runs` and `record_cache.py`: new/changed records are read according
to mtime/size. Re-pretraining and registry arms use `load_cell_matrix`, whose
memoization includes requested leaf-file signatures. Validated native caches are
reused; old temporary data snapshots are not the source of a refresh. The first
refresh here scanned the sources at 01:53–01:58 UTC on 2026-09-24 and saved the
result before computing the bootstrap intervals. Those 54,328 matched runs are
unchanged in this version. The separate modern-reproduction registry was scanned
at 04:37 UTC on the same date, adding 900 runs in 30 complete core30 blocks. Both
refresh records are retained in the snapshot.

The OLMo same-model source scan ran from 14:42:34 to 14:43:00 UTC on
2026-09-25. The new “Same init, same corpus” category uses 700 matched runs in
20 complete core35 blocks: ten Dolma→Dolma teachers and ten DCLM→DCLM
teachers. C4→C4 has no complete matched block yet. The campaign has ten canonical
teachers per corpus, teacher tseed 0, and student ftseed 1; the old ctrl20 pilot
is excluded. The new row uses the same equal-run, 2,000-draw multiway procedure.
The other 15 families’ values and intervals are preserved.

All-arms mode now opts into this canonical same-model campaign on every fresh
load and places it before the other three OLMo rows. It hashes the actual campaign
and teacher source files, rather than trying to read its virtual registry path.

Display aliases do not change physical model IDs: `Rerun1234` is `prv1`
(`pythia-160m-pilereprov1-seed1234`), while `Modern1234` is `pr1234`
(`pythia-160m-pilerepro-seed1234`). The modern reproduction is loaded from
`run_logs/repretrain/repretrain_significance_arms.json` only in all-arms mode.
Its complete matched support is:

| Animal | Direction | Runs | Complete blocks |
| --- | --- | ---: | ---: |
| dog | original → Modern1234 | 300 | 10 |
| dog | Modern1234 → original | 300 | 10 |
| cat | Modern1234 → original | 300 | 10 |
| cat | original → Modern1234 | 0 | 0 |

The last arm lacks matched controls, so its 10 expected blocks are not retained.
An arm marked in progress can still contribute complete blocks; actual matched
coverage determines inclusion.

`transfer_family_multiway_raw_summary.csv` contains plotted quantiles, intervals,
counts, and bootstrap settings (plus win statistics from the shared summarizer).
`complete_matched_pairs.csv`, `complete_blocks.csv`,
`complete_cohort_exclusions.csv`, `complete_registry_coverage.csv`,
`complete_cohort_counts.csv`, and `complete_snapshot.json` provide the cohort and
refresh audit. The recorded snapshot is a point-in-time view; ongoing runs may
finish after the scan.

The five previous balanced raw-percentile artifacts are preserved in
`drafts/balanced_20260924/`, with a SHA-256 manifest. The refresh changes both the
cohort rules and the pooling/bootstrap method, so differences from those archived
plots cannot be attributed only to newly completed runs. Paper, balanced win,
relative, and compact figures are unchanged.

The overview before these label changes and the modern row is preserved in
`drafts/before_reproduction_labels_20260924/` in all three formats, alongside its
summary, counts, and snapshot. The current detailed overview preserves support
metadata for the updated cohort.

The immediately preceding overview, summary, and audits are archived under
`drafts/before_olmo_overview_20260925T144244Z/`.
