# Figure 1 format comparison

Each style is exported as PNG, PDF, and SVG:

- `figure1_boxplot`: boxes show quartiles and medians, with individual observations.
- `figure1_violin`: kernel density, interquartile range, median, and observations.
- `figure1_raincloud`: half-violin density, boxplot, and individual observations.
- `figure1_point_interval`: individual observations with a median marker and opaque
  vertical interquartile range (IQR); the interval is not a confidence interval.

All three use identical data and axis limits. The left panel is student trait
probability minus the mean of ten base-generated control students for that animal
and epoch. The right panel is each student CE minus its own student-model base CE,
including the control students as a sixth grey distribution. No control CE is
subtracted. Data use the held-out probability test set and Pile probe pool pp1.

There are 1,880 treatment observations and 40 control students: ten independent
number-pool draws per epoch (1, 2, 5, 10), all with student fine-tuning seed 1.
The treatment datasets are n40k for cat and n100k for the other four animals;
the controls use n100k. These are distributions over runs, not confidence intervals.
Per-run values and source paths are in `treatments.csv` and `controls.csv`.

Regenerate from the repository root:

```sh
uv run python -m notebooks.figures.figure1_same_init_std.plot_figure1_same_init_std
```

The generator reads saved evaluations; it does not run models. Its boxplot export
also refreshes the original `figure1_same_init_std_recipe_transfer_ce_minus_control.png`
one directory above (the historical filename is retained).
