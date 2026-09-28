# Equal-run comparison with an ordinary bootstrap

This is an additional version of the Figure 2 compact raw-difference charts.
Existing balanced figures and their relative counterparts remain unchanged.
The x-axis contains median, 75th percentile, 95th percentile, and maximum;
colors identify the original four relationship families. Separate figures show
pooled animals and dog only. Nothing is divided by a receiver prior.

## What changes

Each available matched treatment/control row has the same starting weight.
For a family with N rows, each row has weight 1/N. Animals and directed arms
with more available rows therefore contribute more to its pooled statistic.
For example, the cross-data-order family has 6,720 dog rows and 1,680 rows for
each of cat, lion, and pig: dog receives 57.14% of the pooled mass and each other
animal 14.29%. The existing balanced figure instead assigns 25% to each animal.
Within a dog-only chart, every dog row counts equally; an arm with twice as many
available rows contributes twice as much.

Point estimates are the ordinary sample median and percentiles of
p_student − p_control, using NumPy's linear quantile interpolation. The win
fraction in the CSV is the number of strict positive differences divided by N;
ties are not wins. The maximum is the largest observed difference.

## Ordinary paired-row bootstrap

For each displayed family and cohort, draw N row indices uniformly with
replacement from its N available rows. Keep the treatment and its matched control
together whenever a row is selected. Recompute the win fraction and the median,
75th percentile, and 95th percentile. Repeat 2,000 times with random seed 0.
The 2.5th and 97.5th percentiles of each statistic's bootstrap values form the
pointwise 95% interval. The original bar is computed from the original rows,
not from the average of the bootstrap estimates. The maximum has no interval.

For a small example, four differences [−0.02, 0.01, 0.04, 0.10] have a 75% win
fraction and a median of 0.025. One bootstrap sample might select row indices
[1, 1, 3, 4], giving [−0.02, −0.02, 0.04, 0.10]: its win fraction is 50% and its
median is 0.01. Some original rows are repeated and others are omitted. There
are no independently sampled continuous weights in this version.

## What the intervals mean

Resampling is restricted to the available matched rows in each selected design;
it never draws an arbitrary run from outside that design. Comparison against the
canonical `run_logs/core30_cells.json` and `run_logs/core56_cells.json` confirms:

| Family | Cell grid | Available matched pairs |
| --- | --- | ---: |
| Same model | core30 | 5,014 |
| Shared init, different data order | core56 | 11,760 |
| Different init, same data order | core30 | 2,291 |
| Cross seed | core56 | 26,873 |

There are no cell tags outside the corresponding manifest. All planned tags are
represented somewhere in each family; this does not imply that every teacher ×
cell × animal × arm combination has a matched endpoint. See `grid_support.csv`.
The families therefore differ in their cell grids as well as their model
relationships. Collapse labels do
not filter rows. A missing/nonfinite treatment or matched control cannot form
a usable row.

These ordinary intervals treat rows as independent observations. Rows actually
reuse teachers, student configurations, and sometimes matched controls, so the
intervals can understate uncertainty associated with that dependence. They are
an exploratory comparison, not a replacement for the existing crossed method's
treatment of shared configurations. They also do not propagate evaluation
sampling error, teacher selection, or variation over new model seeds or animals.

If the target is simply the observed fixed configuration grid, its sample
percentiles are exact descriptive summaries. Bootstrap intervals require an
additional hypothetical sampling interpretation; completing a planned grid does
not itself create independent experimental replicates.

## Reproduction

```bash
uv run python -m scripts.plot_transfer_family_run_weighted --n-bootstrap 2000 --seed 0
```

The new entrypoint writes only to this comparison directory by default. It reads
existing experiment outputs and does not train models or run probes.

Method references: [ordinary resampling and percentile intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html);
[crossed reweighting in the original comparison](https://arxiv.org/abs/1106.2125).


## Additional bootstrap procedures

The existing ordinary row-bootstrap figures remain in this directory. Two added
versions retain exactly the same equal-run point estimates and change only the
interval calculation:

- [`stratified/`](stratified/): resample matched rows within each animal × directed
  arm, preserving that stratum's observed row count.
- [`multiway/`](multiway/): independently resample source-teacher and global
  student-configuration identities with replacement, preserving their reuse across
  observed rows. Arm contributions and the replicate row count may vary.

Generate both without replacing the ordinary version:

```bash
uv run python -m scripts.plot_transfer_family_run_weighted --method both --n-bootstrap 2000 --seed 0
```

Each subdirectory contains pooled and dog-only charts with families in the legend,
PNG/PDF/SVG exports, an eight-row numerical summary, and its method description.
The maximum remains descriptive. The interval procedures do not impose equal
animal or arm weights.
