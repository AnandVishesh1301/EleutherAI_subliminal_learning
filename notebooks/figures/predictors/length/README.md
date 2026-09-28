# Sequence length, teacher strength, and same-model transfer

Existing-run snapshot, 2026-09-24. **Teacher axes now use validation-set
probabilities; student axes use test-set probabilities.** In the evaluation
configuration, `favorite` refers to the teacher-training prompt set,
`validation` to the separate validation prompt set, and `test` to the student
generalization prompt set. Teacher-strength controls also use validation
probabilities throughout the figures below.

Length is the mean number of integers in the exact sampled training completions,
before tokenizer truncation. Each point is a teacher configuration with seed
averaging; the student recipe is fixed within each model/trait/data cohort.
These are observational associations. All probability axes use logit coordinates
with probability-labelled ticks, clipping to [1e-6, 1 - 1e-6] for display only.
Reported rank correlations use unclipped probabilities.

[Results and methods](../../../../docs/results/sequence_length_transfer_20260924.md) ·
[Teacher-level data](../../../../docs/results/data/sequence_length_transfer_20260924_teacher_rows.csv) ·
[Transfer correlation table](../../../../docs/results/data/sequence_length_transfer_20260924_correlations.csv) ·
[Figure manifest](../../../../docs/results/data/sequence_length_transfer_20260924.json)

## Which comparison does each figure show?

| View | Horizontal axis | Vertical axis | Color / grouping |
|---|---|---|---|
| Dog teacher–student (A11, left) | Teacher validation p(dog) | Student test p(dog) | Mean sequence length; 102 original Pythia-160M teachers |
| All original Pythia-160M animals (A11, right) | Teacher validation p(trait) | Mean sequence length | Animal; generation-pool regime marked by shape |
| Length–teacher (A10 and supplemental pages) | Mean sequence length | Teacher validation p(trait) | Separate panel per model/trait/data cohort |
| Length–student supplemental pages | Mean sequence length | Student test p(trait) | Separate panel per model/trait/data cohort |
| Adjusted length–student (A9) | Correlation coefficient | Model/trait/data cohort | Unadjusted, validation-controlled, validation+CE-controlled estimates |

The scatter plots show marginal relationships. The adjusted forest addresses
whether length retains an association with student transfer after accounting
for teacher measurements; its controls use **validation** probability and
excess CE.

## Combined paper figure

The two-panel figure places the dog teacher/student comparison on the left and
the all-animal length/teacher comparison on the right. The left panel has a
horizontal length colorbar inside the plot; the right panel has an internal
animal legend. Both panels omit titles. The individual views below are retained.

- [Combined figure (Figure A11)](sequence_length_teacher_student_and_animals_validation.png) · [PDF](sequence_length_teacher_student_and_animals_validation.pdf)

## Dog teacher and student preference, colored by length

One chart with 102 original Pythia-160M dog teachers. Teacher validation p(dog)
is on x, student test p(dog) is on y, and color shows mean completion integer
count. Yellow/green denotes short sequences (about one integer), and dark purple
denotes long sequences (about eight integers). Student outcomes use the fixed
`ep10_lr1e-4_r8a32` recipe. The horizontal colorbar sits inside the plot, and
titles are omitted.

- [Dog teacher/student chart (Figure A11, left)](sequence_length_teacher_student_dog_validation.png) · [PDF](sequence_length_teacher_student_dog_validation.pdf)

## Length versus teacher validation preference, colored by animal

The combined original Pythia-160M chart contains 460 observations: cat (74), dog
(102), lion (96), pig (96), and rabbit (92). It includes six eligible cohorts;
cat has both 40,000- and 100,000-completion generation-pool regimes, distinguished
by marker shape (squares: 40,000; circles: 100,000). The plot shows only the
animal legend, inside the axes. Teacher validation p(trait) is on the horizontal
logit axis and mean sequence length is on the vertical linear axis. No teacher
identities overlap between those cat regimes.
All students train on 10,000 sampled completions. Seed/data-order variants and
intermediate checkpoints are separate models in the supplemental pages below.
No correlation is pooled across animals in this chart.

- [All-animal chart (Figure A11, right)](sequence_length_vs_teacher_validation_pythia_animals.png) · [PDF](sequence_length_vs_teacher_validation_pythia_animals.pdf)

## Length versus teacher validation preference, separate panels

The standalone original-dog plot contains the same 102 teachers: shorter
completions accompany stronger validation preference (Spearman rho = -0.928).
The five supplemental pages cover 27 model/trait/data cohorts and 1,433 teacher
observations; each annotation reports a within-cohort correlation. Groups
missing validation measurements are omitted. These files replace their earlier
favorite-based versions at the same paths.

- [Original Pythia-160M / dog (Figure A10)](sequence_length_vs_teacher_strength.png) · [PDF](sequence_length_vs_teacher_strength.pdf)
- [All groups, page 1](sequence_length_vs_teacher_strength_all_1.png) · [PDF](sequence_length_vs_teacher_strength_all_1.pdf)
- [All groups, page 2](sequence_length_vs_teacher_strength_all_2.png) · [PDF](sequence_length_vs_teacher_strength_all_2.pdf)
- [All groups, page 3](sequence_length_vs_teacher_strength_all_3.png) · [PDF](sequence_length_vs_teacher_strength_all_3.pdf)
- [All groups, page 4](sequence_length_vs_teacher_strength_all_4.png) · [PDF](sequence_length_vs_teacher_strength_all_4.pdf)
- [All groups, page 5](sequence_length_vs_teacher_strength_all_5.png) · [PDF](sequence_length_vs_teacher_strength_all_5.pdf)

## Length versus same-model student transfer

These panels show mean completion length versus student test-set preference.
They cover all 41 eligible cohorts, including groups without teacher preference
or CE measurements.

- [All groups, page 1](sequence_length_transfer_scatter_1.png) · [PDF](sequence_length_transfer_scatter_1.pdf)
- [All groups, page 2](sequence_length_transfer_scatter_2.png) · [PDF](sequence_length_transfer_scatter_2.pdf)
- [All groups, page 3](sequence_length_transfer_scatter_3.png) · [PDF](sequence_length_transfer_scatter_3.pdf)
- [All groups, page 4](sequence_length_transfer_scatter_4.png) · [PDF](sequence_length_transfer_scatter_4.pdf)
- [All groups, page 5](sequence_length_transfer_scatter_5.png) · [PDF](sequence_length_transfer_scatter_5.pdf)
- [All groups, page 6](sequence_length_transfer_scatter_6.png) · [PDF](sequence_length_transfer_scatter_6.pdf)
- [All groups, page 7](sequence_length_transfer_scatter_7.png) · [PDF](sequence_length_transfer_scatter_7.pdf)

## Length–transfer associations after teacher controls

The adjusted figures compare identical complete-case teachers across marginal,
teacher-validation adjustment, and teacher-validation-plus-excess-CE adjustment.
They cover the 25 cohorts with both controls; intervals are approximate nominal
95% Fisher intervals.

- [Adjusted associations, page 1 (Figure A9)](sequence_length_transfer_adjusted_1.png) · [PDF](sequence_length_transfer_adjusted_1.pdf)
- [Adjusted associations, page 2 (Figure A9)](sequence_length_transfer_adjusted_2.png) · [PDF](sequence_length_transfer_adjusted_2.pdf)

Regenerate all figures from the frozen collection, from the repository root:

```bash
uv run python -m scripts.analyze_sequence_length_transfer
```
