# Pythia size and OLMo transfer magnitudes

`../transfer_family_complete_multiway_size_olmo.{png,pdf,svg}` places Pythia
cross-size transfers (dog) on the left and OLMo corpus/seed relationships
(elephant) on the right. Both panels group median, 75th percentile, 95th percentile,
and observed maximum within each family, using the main paper B panel's gold,
red, purple, and cyan palette. Axes share a raw `p_student - p_control` scale.
The direction subtitles are “same data order” for the original-seed comparisons
and “different data order” for the cross-seed comparisons; these are display labels
and do not change the underlying arm identities or results.
The first three statistics have 95% pointwise two-way cluster bootstrap intervals
(2,000 draws, seed 0); the maximum is descriptive and has no interval.

Each matched run has equal weight. Complete teacher × animal × directed-arm
blocks are required: canonical core30 for Pythia size transfers and core35 for
OLMo. Student ftseed 1 and each arm's declared teacher seed are retained. All
collapse states and all canonical selection cells are included. Both axes start
at zero, as in the existing paper B panel; negative estimates and interval
endpoints remain in the CSV when visually clipped.

## OLMo same-model

The first OLMo category is reserved for the full core35 campaign in
`run_logs/olmo_samemodel_core35_20260924/`. It aggregates Dolma→Dolma,
DCLM→DCLM, and C4→C4: ten canonical elephant teachers per arm, teacher tseed 0,
student ftseed 1, and the receiver's own untuned-base control at every student cell.
The old ctrl20 pilot is coverage-only and does not supply this category.

The opt-in loader reconstructs an in-memory registry using the campaign's exact
teacher manifests and the canonical cell file. Its virtual `registry.json` path
is a source identifier and relative-path base; no file at that path is required.
The DCLM TSV has no header, and all ten teacher rows are used. The resolved
in-memory document is saved here as `olmo_same_model_registry.json`; paths in that
copy retain their meaning relative to the virtual source path in `snapshot.json`.
Actual source manifests and campaign files are hash-recorded in that snapshot.

At this refresh no same-model block has all 35 matched endpoints, so the axis
category reads “Pending complete data” and has no bars. The loader is ready to
include newly completed blocks on a later refresh; the PNG itself is static.

## Provenance and reproduction

The source scan used the signature-aware `load_cell_matrix` caches through
`registry_pairs(..., include_olmo_same_model=True, collapse_filter=False,
include_selection_cells=True)`, followed by
`select_complete_pairs(..., include_olmo_same_model=True,
include_selection_cells=True)`. It did not reload the unrelated legacy notebook.
The option defaults to false, so existing figure cohorts are unchanged.

`summary.csv` stores all estimates/intervals; `matched_pairs.csv`,
`complete_blocks.csv`, `exclusions.csv`, `coverage.csv`, `cohort_counts.csv`, and
`snapshot.json` retain source, completeness, and refresh evidence. Other top-row
figures and their summary are preserved.

Re-render the saved values from the repository root:

```sh
uv run python - <<'PYTHON'
import pandas as pd
from scripts.plot_transfer_family_multiway import DEFAULT_OUT, export_size_olmo
from subliminal_learning.analysis.transfer_family_pooled import default_context
out = DEFAULT_OUT / 'top_row'
export_size_olmo(pd.read_csv(out / 'size_olmo_data/summary.csv'), default_context(), out)
PYTHON
```

Source scan: 2026-09-24T23:33:42.622713+00:00 to 2026-09-24T23:34:24.096614+00:00.

| Family | Animal | Matched runs | Complete blocks | Grid |
| --- | --- | ---: | ---: | --- |
| small→large, original-seed lineage | dog | 330 | 11 | core30 |
| large→small, original-seed lineage | dog | 300 | 10 | core30 |
| small→large, different-init/cross-seed | dog | 300 | 10 | core30 |
| large→small, different-init/cross-seed | dog | 300 | 10 | core30 |
| OLMo same model | elephant | 0 | 0 | core35 (pending) |
| OLMo same-init/different-corpus | elephant | 1,750 | 50 | core35 |
| OLMo different-init/different-corpus | elephant | 1,400 | 40 | core35 |
| OLMo cross-seed different-init/same-corpus | elephant | 1,050 | 30 | core35 |
