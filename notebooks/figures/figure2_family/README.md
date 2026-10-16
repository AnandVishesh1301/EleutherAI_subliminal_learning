# Figure 2: transfer by sender/receiver relationship

The refreshed raw-percentile family is now in [`all_arms/`](all_arms/README.md).
It uses equal weight per matched run and 2,000 two-way cluster (pigeonhole)
bootstrap replicates on complete canonical grids. Reproduce it with:

```bash
uv run python -m notebooks.figures.figure2_family.plot_transfer_family_multiway --all-arms --n-bootstrap 2000 --seed 0
```

The same script without `--all-arms` writes the complete-grid multiway views to
[`run_weighted/multiway/`](run_weighted/multiway/README.md), and `--publish` writes
the OLMo/size panel to [`olmo/`](olmo/README.md):

```bash
uv run python -m notebooks.figures.figure2_family.plot_transfer_family_multiway --n-bootstrap 2000 --seed 0
uv run python -m notebooks.figures.figure2_family.plot_transfer_family_multiway --publish --n-bootstrap 2000 --seed 0
```

The five previous balanced raw artifacts are preserved in
`all_arms/drafts/balanced_20260924/`. The balanced weighting and bootstrap discussion
below describes the older outputs; it does not describe the refreshed `all_arms/` family.

Reproduce the remaining balanced and historical views from saved experiment outputs:

```bash
uv run python -m notebooks.figures.figure2_family.plot_transfer_family_pooled_minus_control --n-bootstrap 2000 --seed 0
```

The command reads existing evaluations; it does not train models or launch probes.
The first current-data load can take several minutes. Original images are preserved
in `drafts/`, with SHA-256 hashes in `drafts/sha256.json`.

The relationship order is same model, shared initialization with different data
order, different initialization with the same data order, cross seed, then
reproduction. The reproduction slot holds the `orig`↔`prv1` directions: `prv1`
re-runs the original Pythia-160M recipe (seed 1234, the gpt-neox v1.0 tree) on the
same Pile order, so the pair differs in neither initialization seed nor data
order. It uses core30, like the same-model and re-pretraining relationships.

## Views

Raw-difference and win figures live in this directory. All current figures divided
by the receiver prior, their summary CSV, and `receiver_priors.csv` live in
[`relative/`](relative/). Historical snapshots remain in `drafts/`.

- `transfer_family_balanced_win.png`: strict observed win fraction,
  `p_student > p_control` over all finite matched pairs regardless of collapse,
  in three panels: the five Pythia-160M relationships with
  pooled/dog/cat/lion/pig bars; six size/architecture relations with dog only; and
  three OLMo corpus/seed relations with elephant only. Single-animal panels omit
  the identical pooled bar.
- `transfer_family_balanced_win_size_architecture.png`: standalone view of all six
  size and architecture directions, using dog only.
- `all_arms/transfer_family_multiway_raw_percentiles.png`: refreshed median, 75th
  percentile, 95th percentile, and maximum of `p_student - p_control` across all
  relationships; Pythia rows pool animals, size/architecture rows use dog, and
  OLMo corpus/seed rows use elephant. The old balanced overview is archived.
- `all_arms/transfer_family_multiway_raw_percentiles_with_metadata.png`: the same
  overview retaining n, complete-block counts, and core-grid design. Both current
  overviews include separate Rerun1234 and Modern1234 reproduction rows.
- `relative/transfer_family_balanced_prior_percentiles.png`: the same statistics of
  `(p_student - p_control) / p_receiver_base`.
- `all_arms/transfer_family_multiway_raw_percentiles_panel{1,2,3}.png` and
  `relative/transfer_family_balanced_prior_percentiles_panel{1,2,3}.png`: per-animal
  detail for Pythia-160M relationships (six in refreshed all-arms, five in the
  older relative views), the six size/architecture transfers (dog only), and the
  OLMo comparisons (four in refreshed all-arms, three in older relative views;
  elephant only), respectively.
- `drafts/four_group_v1/`: the preceding four-group bars and numerical summaries.
- `drafts/screened_win_v1/`: charts and tables before removing collapse screening
  from wins; kept to make the change inspectable.
- `drafts/screened_percentiles_v1/`: the subsequent snapshot with all-pair wins
  but screened percentiles, preserved before switching percentiles to all pairs.
- `drafts/expanded_v1/`: the first expanded figures and tables, before combining
  size/architecture and excluding cross-architecture elephant rows.
- `legacy/transfer_family_pooled_minus_control_alive*.png`: reordered versions of the
  original boxplot views, retaining their original run weighting; they now carry
  the reproduction slot as a fifth group.
- `relative/transfer_family_pooled_minus_control_prior_linear.png`: a normalized
  counterpart of the original linear boxplot, also run weighted.

In the original boxes, IQR and 5–95% whiskers describe the distribution of runs;
they are not confidence intervals. New bars use the balanced weighting below.
The empirical maximum has no confidence interval in either percentile panel.

## Pooling

Within a relationship, give each represented animal equal weight. Within each
animal, give every available directed sender→receiver arm equal weight. Within
that animal-arm block, give each matched treatment run equal weight. For `A`
animals, `K_a` arms for animal `a`, and `n_ab` matched rows in arm `b`, a row's
weight is `1 / (A * K_a * n_ab)`. Per-animal bars use `A = 1`.

A win bar is the weighted average of `1[p_student > p_control]`; ties are not
wins. Percentiles invert the weighted empirical CDF of individual differences.
They are not averages of arm-specific percentiles. Arms with more available
rows do not get extra mass, and animals with more arms do not get extra mass.
Missing arms are not imputed; counts in the CSVs describe the actual support.

A student cell identifies its training configuration. A source teacher is
identified by `(sender, animal, teacher configuration)`, and can be reused by
several receiver arms. Treatment rows are paired to the untuned-sender control
at the same `(sender, receiver, animal, student cell)`. Several teacher rows can
therefore share one control.

## Current bar populations and historical boxes

Win probabilities and percentiles use every finite matched treatment/control pair, regardless of
alive/dead or collapse-threshold status. Both sides can be dead or unlabelled.
Missing/nonfinite probabilities cannot form a comparison. Other design choices
(selected teachers, student cells, held-out exclusions, no ws receivers, and the
dog-only cross-architecture cohort) still apply. Current bars carry no screening hatch.

Only the historical boxplots retain the previous screening rule. Both populations
come from one load: each candidate records `screened_eligible` and its actual
`treatment_dead`/`control_dead` labels (0, 1, or NaN). `receiver_filtered` describes
threshold availability; it does not imply that the current bars were screened.
Collapse labels remain audit metadata and do not determine bar inclusion or styling.
The earlier screened percentile population is preserved in the draft snapshot.

Weights are balanced within each relationship. Additional dog-only categories do
not dilute dog weights in another relationship. In the all-pair cross-data-order
comparison, the actual starting weights are:

| Animal | Available directions | Pairs/direction | Animal weight | Direction weight | Per-pair weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dog | 12 | 560 | 25% | 1/48 | 1/26,880 |
| Cat | 3 | 560 | 25% | 1/12 | 1/6,720 |
| Lion | 3 | 560 | 25% | 1/12 | 1/6,720 |
| Pig | 3 | 560 | 25% | 1/12 | 1/6,720 |

Each dog pair has one-quarter the weight of a cat/lion/pig pair, while the animal
still has equal total weight. Counting all rows equally would instead assign dog
57.14% and each other animal 14.29%. Within an arm, all available treatment runs
have equal starting weight; teachers with more available runs can therefore have
more starting mass than teachers with fewer runs.

## Exploratory uncertainty intervals

The 2.5th and 97.5th percentiles of 2,000 crossed multiplier-bootstrap draws form
pointwise 95% configuration-resampling intervals. Each draw assigns independent
Exp(1) weights to source-teacher identities and student-cell identities and
multiplies the two weights for each row. The same source teacher receives the
same multiplier across receivers; the same student cell receives the same
multiplier across animals and arms in the summarized group. Each animal-arm
block is renormalized to its original total mass before recomputing the statistic.
Thus shared configurations are not counted as independent rows.

For a concrete hypothetical draw within one arm, suppose two teachers and two
student configurations form four pairs, three wins and one loss. Before the draw,
each pair has 25% of that arm's mass and its win rate is 75%. A teacher multiplier
of 2 for T1 and 1 for T2, combined with student multipliers 1 for C1 and 3 for C2,
produces:

| Pair | Win indicator | Starting share within arm | Multiplier product | Renormalized share |
| --- | ---: | ---: | ---: | ---: |
| T1/C1 | 1 | 25% | 2 | 16.67% |
| T1/C2 | 0 | 25% | 6 | 50.00% |
| T2/C1 | 1 | 25% | 1 | 8.33% |
| T2/C2 | 1 | 25% | 3 | 25.00% |

This draw has a 50% within-arm win rate. The arm's total mass is unchanged: an arm
starting with 1/48 of the pooled weight still has 1/48 afterward. The same teacher
and cell multipliers are reused wherever their identities recur in the summarized
group. This example is arithmetic, not a draw taken from the experiment. The
actual procedure repeats with independent Exp(1) weights 2,000 times and takes
percentiles of the resulting whole-group statistics. The original bar height
remains the statistic under the starting weights, not the average bootstrap draw.

The named animals and arms are fixed. Intervals condition on available matched
runs, the selected top-teacher pool, and the measured student/control/prior
probabilities. They do not propagate prompt or response sampling error, teacher
selection uncertainty, or uncertainty about new animals, model initializations,
or training seeds. They are exploratory sensitivity summaries for a designed
configuration grid, not a guarantee of frequentist coverage for an unseen
population. In particular, the quantile intervals have not been coverage-calibrated
for this experiment. They are pointwise, not simultaneous confidence bands.

The maximum is descriptive: ordinary empirical resampling cannot exceed the
observed maximum and does not provide a reliable interval for an unseen upper
endpoint. A tail model or a separately justified extreme-value procedure would
be needed for that claim. A zero-width bootstrap interval (for example, when
all observed runs win) does not establish certainty; these cases are flagged in
the summary CSV. Blocks with fewer than two teachers or cells are also flagged.

Method reference: [Owen and Eckles (2012), *Bootstrapping data arrays of arbitrary
order*](https://arxiv.org/abs/1106.2125), which studies crossed reweighting for
mean variance. The application to these quantiles is exploratory. For limitations
of ordinary bootstrap extremes, see [Bickel and Freedman (1981), *Some Asymptotic
Theory for the Bootstrap*](https://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/BickelFreedman81.pdf).

## Receiver prior and screening coverage

The denominator is the untrained receiver/base model's probability for the target
animal on the test prompts. It is loaded from the student trees' `base_control.json`
records, with a held-out base-evaluation fallback when those records are absent.
Disagreeing student-tree priors raise an error. No validation-prompt proxy or
arbitrary floor is substituted. Zero or unavailable priors are explicitly excluded
from normalized summaries and retained in raw summaries. Divide each individual
difference before pooling. A positive denominator cannot change a strict win, so
a duplicate normalized win chart is unnecessary when support is unchanged.

Red annotations in the historical boxplots and hatching in archived percentile
bars flag incomplete collapse screening, not statistical significance. Current win
and percentile figures have neither hatching nor screening-based categories. A
control can have a valid animal-preference measurement without an alive/dead label.
Historically, receivers without an anchored collapse threshold were left unfiltered;
treatments with an anchored threshold had to pass the ALIVE filter, and controls
were kept if alive or unlabelled. Those rules apply only to the historical boxes.
Missing matched controls still produce no pair in every view.

Same-model, re-pretraining and reproduction relationships use core30; the
cross-data-order and cross-seed relationships use core56. Actual matched support can be smaller. Balanced weighting does not
make different student grids or missing arms comparable by itself.

The archived screened four-relationship slice contains 40,606 matched rows. Its same-model flags concern
3,189 rows sharing 320 control strata without collapse labels. In the
same-data/different-init group, 1,200 rows from `wc1→wc2` are unfiltered because
`wc2` has no anchored threshold; `wc2→wc1` contributes 808 filtered rows. All 35
original receiver–animal combinations of that archived slice have positive test
priors, so it has identical raw and normalized support.

The `prv1` arms now contribute matched pairs in the current population
(`coverage.csv`, `repretrain_direction_coverage.csv`). Against the re-pretrainings:
`wc1→prv1` 300 dog and 300 cat pairs, `wc2→prv1` 300 dog and 300 cat pairs,
`prv1→wc1` 299 dog and 289 cat pairs, and `prv1→wc2` 299 dog and 39 cat pairs.
In the reproduction slot: `orig→prv1` 300 dog and 300 cat pairs, `prv1→orig` 299
dog and 1 cat pair, for 900 pairs across 31 source teachers. Lion and pig have no
controls and no pairs in any `prv1` direction. The `prv1`-receiver directions have
no measured receiver prior, so their 1,800 pairs (1,200 against the
re-pretrainings, 600 in the reproduction slot) enter the raw views only.

## Numerical records

- `all_arms/transfer_family_multiway_raw_summary.csv`: refreshed equal-run estimates
  and multiway intervals; complete matched pairs, block audits, exclusions, and
  refresh provenance are alongside it.

- `transfer_family_balanced_win_summary.csv`: all-pair wins, intervals, support,
  and bootstrap settings; `collapse_filter=False`.
- `all_arms/drafts/balanced_20260924/transfer_family_balanced_raw_summary.csv` and
  `relative/transfer_family_balanced_prior_summary.csv`: all-pair percentile estimates,
  intervals, counts, and bootstrap settings; `collapse_filter=False`. Win estimates
  remain in the separate win table.
- `win_coverage.csv`, `win_registry_coverage.csv`, `win_cohort_exclusions.csv`:
  all-pair support by animal/direction, registry inventory, and dog-only architecture
  exclusions. Their unprefixed counterparts describe the same all-pair cohort for
  percentiles, with prior-availability metadata where relevant.
- `collapse_audit_{summary,joint,support}.csv`: descriptive comparison of collapse
  labels and wins in the two populations, including unknown-label counts, repeated
  pair versus unique-control units, and base weighting.
- `figure_cohort_exclusions.csv`: matched rows excluded by the requested dog-only
  cross-architecture comparison, with an explicit reason; registry coverage remains
  an inventory of all loaded registry arms, including those excluded here.
- `coverage.csv`: per relationship/animal/direction pair counts, unique matched
  controls, unlabelled controls, unfiltered treatment rows, and prior exclusions.
- `repretrain_direction_coverage.csv`: all six intended re-pretraining directions,
  including arms without available controls.
- `relative/receiver_priors.csv`: exact denominators, receiver mapping, source paths, and
  unavailable-prior reasons.


## Size, architecture, and OLMo corpus extensions

The expanded summaries retain the original four relationships, add the
reproduction slot described above, and add nine registry relationships:

1. Pythia 160M → 410M, original-seed lineage.
2. Pythia 410M → 160M, original-seed lineage.
3. Pythia 160M → 410M-seed1, cross-seed lineage.
4. Pythia 410M-seed1 → 160M, cross-seed lineage.
5. Pythia → OLMo.
6. OLMo → Pythia.
7. OLMo: same initialization, different corpus.
8. OLMo: different initialization, different corpus.
9. OLMo cross-seed: different initialization, same corpus.

Across model sizes, “original-seed lineage” identifies the two original Pythia
models. It does not assert identical initialized weight arrays: the model shapes
differ. The 410M-seed1 comparison also changes the pretraining seed and is not an
isolated causal test of initialization. OLMo cross-seed is not a same-init control:
its default and large-aux-2 branches have different initialization. The corpus
comparisons group Dolma, DCLM, and C4 by the registry's initialization class.

Additional arms come from `run_logs/xarch/xarch_significance_arms.json` and
`run_logs/repretrain/olmo_cross_seed_significance_arms.json`. Only the registry's
listed teachers and designated student cells enter; held-out selection cells
remain excluded. Per-teacher dataset overrides are honored, including the one
125k-number sender in the reverse 410M-seed1 arm. Single-cell teacher-selection
pilots and OLMo same-model references are excluded from these added categories.
A trained arm without a matched control cannot produce a difference or a win.

The added categories have different animal support. Size-transfer arms use dog;
both Pythia→OLMo and OLMo→Pythia use dog only in these figures. Elephant rows
from the latter are excluded before computing any summary, including the overview.
OLMo corpus and cross-seed arms retain elephant. An equally weighted pooled bar summarizes only the
animals and arms represented in that category. Consequently, differences between
pooled categories can reflect animal composition, selected teachers, or student
grids in addition to the named relationship. The per-animal bars and support
counts expose these differences.

Available matched endpoints from in-progress arms are descriptive samples, with
completion and collapse-label coverage reported separately. No significance P-values
are computed from partial designs. New receivers use their native probe pool;
collapse labels do not filter the current bar figures. Missing receiver priors remove only those rows from
normalized summaries; raw and normalized support must be checked before comparing
them. No substitute prior or artificial zero is drawn.


The current win and raw-percentile population contains **54,580** pairs:
49,170 original/re-pretraining/reproduction pairs plus 5,410 registry pairs,
excluding 60 cross-architecture elephant pairs. These 60 exclusions are recorded in
`figure_cohort_exclusions.csv`; elephant is retained for OLMo-only corpus/seed
relations. The archived screened percentile snapshot contains **44,544** pairs,
including 3,938 extension pairs after its 54 cross-architecture exclusions.
This is a requested cohort choice, separate from whether a registry arm is complete.
The 1,800 `prv1`-receiver pairs have no measured receiver prior, so the raw views
carry 54,580 pairs and the normalized ones 52,780. `registry_coverage.csv` reports
all 28 registry entries, including excluded pilots, the OLMo same-model reference,
seed-3 arms without controls, and currently empty/incomplete designs. Available
extra pairs are:

| Added category | Animals | Matched pairs |
| --- | --- | ---: |
| small→large, original-seed lineage | dog | 330 |
| large→small, original-seed lineage | dog | 300 |
| small→large, different-init/cross-seed | dog | 300 |
| large→small, different-init/cross-seed | dog | 300 |
| Pythia→OLMo | dog | 544 |
| OLMo→Pythia | dog | 308 |
| OLMo same-init/different-corpus | elephant | 1,393 |
| OLMo different-init/different-corpus | elephant | 846 |
| OLMo cross-seed different-init/same-corpus | elephant | 1,089 |

The same-init/different-corpus pool contains five observed directions; C4→Dolma
has 23 of 340 designated endpoints in this snapshot. The different-init/
different-corpus pool has four observed directions, two of them still filling:
Dolma→DCLM-seed2 has 27 of 350 and Dolma→C4-seed2 119 of 350 designated endpoints.
OLMo cross-seed now has 1,089 pairs across four directions and 32 distinct source
teachers: Dolma→Dolma-seed2 370, Dolma-seed2→Dolma 60, DCLM→DCLM-seed2 350, and
C4→C4-seed2 309 of 350 while that arm is still running. Its configuration intervals
still do not supply replication over additional model seeds.
