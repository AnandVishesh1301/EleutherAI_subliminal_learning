# Multiway (pigeonhole) bootstrap

The two resampling dimensions are source-teacher identity
`(sender, animal, teacher)` and student-configuration identity `student_tag`.
A source teacher retains its identity across receivers, and a student
configuration retains its identity across animals and arms within the family
and cohort being summarized.

For each replicate, sample T identities with replacement from the T available
source teachers. Independently sample C identities with replacement from the C
available student configurations. An observed matched row appears as many times
as its teacher was selected, multiplied by the number of times its student
configuration was selected. Keep its original treatment and matched control
together. Missing teacher/configuration/arm intersections are not synthesized.

Calculate the run-weighted statistic over those repeated rows. Integer frequency
counts are used to compute exactly the same linear quantiles as explicitly
expanding the repeated rows, without constructing that large array. This uses
ordinary sampling with replacement, not exponential multipliers.

Arm identities are not resampled as an outer level, but an arm's total
contribution can vary or vanish when its configurations are not selected.
Replicate sample size can therefore vary. There is no per-animal or per-arm
renormalization. Entirely empty replicates are redrawn and counted in the summary
CSV; a bounded retry limit prevents silently accepting an undefined statistic.

This method preserves the reuse of the two sampled identities. It does not add
an independent control-evaluation resampling dimension or cover all possible
sources of dependence. Source identities are drawn from the observed family/cohort
pool, so the relative contributions of animals and arms can change across draws.

Both comparisons use the same 45,938 finite matched pairs in the original four
relationships, with no collapse filtering and no division by receiver priors.
The pooled and dog-only bar heights are ordinary run-weighted median, 75th
percentile, 95th percentile, and observed maximum. Quantiles use linear
interpolation. The maximum is descriptive and has no interval.

Intervals are pointwise percentile intervals from 2,000 bootstrap replicates
(seed 0). They condition on the available configuration grid, selected teachers,
and measured probabilities. They do not propagate evaluation-response sampling,
teacher-selection uncertainty, or variation over new model seeds. These are
exploratory comparisons for this experiment; nominal coverage is not established
for this selected grid.

Canonical manifest checks show core30 for same-model and different-init/same-data
comparisons, and core56 for cross-data-order and cross-seed. No outside cell tags
enter. Missing matched endpoints remain missing; resampling does not fill them.

Reproduce both new versions with:

```bash
uv run python -m scripts.plot_transfer_family_run_weighted --method both --n-bootstrap 2000 --seed 0
```

References: [Owen (2007), The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111),
[Davezies et al., Asymptotic results under multiway clustering](https://arxiv.org/abs/1807.07925).
