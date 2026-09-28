# Ordinary stratified bootstrap

The strata are `(animal, sender, receiver)`. In each replicate, a stratum with
n_h observed matched rows contributes exactly n_h draws, sampled uniformly with
replacement from that stratum. Each selected row retains its treatment and
matched control. Pool these draws and recompute the ordinary statistic.

Thus an arm with 1,000 rows still contributes five times as much as an arm with
200 rows. No equal-animal or equal-arm adjustment is introduced. This keeps the
observed composition fixed but treats rows as independent within and across
strata; shared teachers, student configurations, and controls are not accounted
for by this method.

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

The previous ordinary row-bootstrap figures remain in the parent directory.
See the [standard stratified-bootstrap interface](https://stat.ethz.ch/R-manual/R-devel/library/boot/html/boot.html).
