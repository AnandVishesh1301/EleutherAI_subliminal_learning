# Compact Figure 2 captions

## Pooled, raw difference

**Within-model transfer is strongest across the observed distribution.** Bars show
the median, 75th percentile, 95th percentile, and observed maximum of the matched
difference in target-animal probability, Δ = p_student − p_control, for four
sender–receiver relationships in Pythia 160M. Each animal receives equal weight;
available directed arms receive equal weight within an animal, and matched runs
share their arm's weight. All finite matched pairs enter, regardless of collapse
labels. Error bars on the first three statistics show exploratory pointwise 95%
intervals from 2,000 crossed source-teacher and student-configuration reweightings,
with each animal–arm block's total weight held fixed. Animals, arms, and measured
probabilities are conditioned on. The maximum is descriptive and has no interval.

## Dog only, raw difference

**Within-model dog transfer is strongest across the observed distribution.** Bars
show the median, 75th percentile, 95th percentile, and observed maximum of
Δ = p_student − p_control for the dog target in four Pythia 160M sender–receiver
relationships. Available directed arms receive equal weight within a relationship,
and matched runs share their arm's weight. All finite matched pairs enter,
regardless of collapse labels. Error bars on the first three statistics show
exploratory pointwise 95% intervals from 2,000 crossed source-teacher and
student-configuration reweightings, preserving each arm's total weight. Arms and
measured probabilities are conditioned on. The maximum is descriptive and has no
interval.

## Layout and normalization variants

The family-axis layout colors statistics; the statistic-axis layout colors
relationships. The underlying estimates and intervals are identical. For a
prior-normalized figure, replace Δ above with
(p_student − p_control) / p_receiver_base, where p_receiver_base is the untrained
receiver's probability of the target animal. Divide each matched difference by
its own receiver prior before computing the weighted quantiles. The current raw
compact figures use 48,270 pooled pairs or 16,855 dog pairs across the four
relationships; the normalized ones drop the 1,200 pooled (600 dog) pairs whose
`prv1` receiver has no measured prior, leaving 47,070 pooled and 16,255 dog pairs.
The compact figures keep these four relationships: the fifth, reproduction
(`prv1`↔`orig`), appears only in the detailed figures and tables. The intervals describe sensitivity
to the available configuration grid; they do not estimate variation over new
animals, model seeds, teacher selection, or evaluation responses.
