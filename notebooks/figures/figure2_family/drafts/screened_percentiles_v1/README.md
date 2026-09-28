# Figure 2: transfer by sender/receiver relationship

Reproduce the current figures from saved experiment outputs:

```bash
uv run python -m scripts.plot_transfer_family_pooled_minus_control --n-bootstrap 2000 --seed 0
```

The command reads existing evaluations; it does not train models or launch probes.
The first current-data load can take several minutes. Original images are preserved
in `drafts/`, with SHA-256 hashes in `drafts/sha256.json`.

The relationship order is same model, shared initialization with different data
order, different initialization with the same data order, then cross seed.

## Views

- `transfer_family_balanced_win.png`: strict observed win fraction,
  `p_student > p_control` over all finite matched pairs regardless of collapse,
  in three panels: the original four relationships with
  pooled/dog/cat/lion/pig bars; six size/architecture relations with dog only; and
  three OLMo corpus/seed relations with elephant only. Single-animal panels omit
  the identical pooled bar.
- `transfer_family_balanced_win_size_architecture.png`: standalone view of all six
  size and architecture directions, using dog only.
- `transfer_family_balanced_raw_percentiles.png`: median, 75th percentile, 95th
  percentile, and maximum of `p_student - p_control`, with one pooled row per
  relationship in the overview.
- `transfer_family_balanced_prior_percentiles.png`: the same statistics of
  `(p_student - p_control) / p_receiver_base`.
- `transfer_family_balanced_{raw,prior}_percentiles_panel{1,2,3}.png`: per-animal
  detail for the original four relationships, the six size/architecture transfers
  (dog only), and the three OLMo corpus/seed comparisons (elephant only), respectively.
- `drafts/four_group_v1/`: the preceding four-group bars and numerical summaries.
- `drafts/screened_win_v1/`: charts and tables before removing collapse screening
  from wins; kept to make the change inspectable.
- `drafts/expanded_v1/`: the first expanded figures and tables, before combining
  size/architecture and excluding cross-architecture elephant rows.
- `transfer_family_pooled_minus_control_alive*.png`: reordered versions of the
  four original boxplot views, retaining their original run weighting.
- `transfer_family_pooled_minus_control_prior_linear.png`: a normalized
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

## Win versus screened populations

Win probabilities use every finite matched treatment/control pair, regardless of
alive/dead or collapse-threshold status. Both sides can be dead or unlabelled.
Missing/nonfinite probabilities cannot form a comparison. Other design choices
(selected teachers, student cells, held-out exclusions, no ws receivers, and the
dog-only cross-architecture cohort) still apply. Win bars carry no screening hatch.

Percentiles and legacy boxes retain the previous screening rule. Both populations
come from one load: each candidate records `screened_eligible` and its actual
`treatment_dead`/`control_dead` labels (0, 1, or NaN). `receiver_filtered` describes
threshold availability; it does not imply that all-pair wins were screened. The
screened subset's identities, probabilities, and flags match the preceding data
snapshot exactly. The two populations have separate support tables and CSVs.

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

Red annotations in the old boxplots and hatched percentile bars flag incomplete
collapse screening, not statistical significance. Win figures have no hatching or
screening-based categories because every finite matched pair enters. A control can have a valid
animal-preference measurement without an alive/dead label. A receiver without an
anchored collapse threshold is kept unfiltered. Treatments with an anchored
threshold must pass the existing ALIVE filter; controls are kept if alive or
unlabelled. Missing matched controls produce no pair. These selection rules are
unchanged, and interval estimation cannot repair missing screening information.

Legacy relationships use the existing core56 design; re-pretraining relationships
use core30. Actual matched support can be smaller. Balanced weighting does not
make different student grids or missing arms comparable by itself.

The original four-relationship slice contains 40,606 matched rows. Same-model flags concern
3,189 rows sharing 320 control strata without collapse labels. In the
same-data/different-init group, 1,200 rows from `wc1→wc2` are unfiltered because
`wc2` has no anchored threshold; `wc2→wc1` contributes 808 filtered rows. No
`prv1` arm currently contributes matched pairs. All 35 original receiver–animal
combinations have positive test priors, so that slice has identical raw and normalized support.

## Numerical records

- `transfer_family_balanced_win_summary.csv`: all-pair wins, intervals, support,
  and bootstrap settings; `collapse_filter=False`.
- `transfer_family_balanced_raw_summary.csv` and
  `transfer_family_balanced_prior_summary.csv`: screened percentile estimates,
  intervals, counts, and bootstrap settings; `collapse_filter=True`. Win estimates
  are kept in the separate win table, avoiding mixed-population columns.
- `win_coverage.csv`, `win_registry_coverage.csv`, `win_cohort_exclusions.csv`:
  all-pair support by animal/direction, registry inventory, and dog-only architecture
  exclusions. Their unprefixed counterparts describe screened support.
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
- `receiver_priors.csv`: exact denominators, receiver mapping, source paths, and
  unavailable-prior reasons.


## Size, architecture, and OLMo corpus extensions

The expanded summaries retain the original four relationships and add nine:

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
completion and filtering coverage reported separately. No significance P-values
are computed from partial designs. New receivers use their native probe pool and
the same anchored-collapse rule as the original groups; missing anchors leave
runs unfiltered and flagged. Missing receiver priors remove only those rows from
normalized summaries; raw and normalized support must be checked before comparing
them. No substitute prior or artificial zero is drawn.


The screened percentile snapshot contributes **3,938** extension matched pairs,
for **44,544** total. The all-pair win population contains **50,054** pairs:
45,938 original/re-pretraining pairs plus 4,176 registry pairs, excluding 60
cross-architecture elephant pairs. The same exclusion removes 54 screened pairs. The 54 elephant rows in the OLMo→Pythia cross-architecture arm
are recorded in `figure_cohort_exclusions.csv` and excluded from the dog-only
architecture comparison; elephant is retained for OLMo-only corpus/seed relations.
This is a requested cohort choice, separate from whether a registry arm is complete.
All included pairs have a positive measured receiver prior; the current raw and
normalized views therefore use the same support. `registry_coverage.csv` reports
all 25 registry entries, including excluded pilots, the OLMo same-model reference,
seed-3 arms without controls, and currently empty/incomplete designs. Available
extra pairs are:

| Added category | Animals | Matched pairs |
| --- | --- | ---: |
| small→large, original-seed lineage | dog | 322 |
| large→small, original-seed lineage | dog | 250 |
| small→large, different-init/cross-seed | dog | 287 |
| large→small, different-init/cross-seed | dog | 247 |
| Pythia→OLMo | dog | 544 |
| OLMo→Pythia | dog | 254 |
| OLMo same-init/different-corpus | elephant | 1,254 |
| OLMo different-init/different-corpus | elephant | 700 |
| OLMo cross-seed different-init/same-corpus | elephant | 80 |

The same-init/different-corpus pool contains four observed directions; C4→DCLM
has 224 of 340 designated endpoints in this snapshot. The different-init/
different-corpus pool has two observed directions; the two other declared arms
have no student endpoints yet. OLMo cross-seed has 80 pairs across two directions
and only three distinct source teachers; its configuration intervals do not
supply replication over additional model seeds or a broad teacher population.
