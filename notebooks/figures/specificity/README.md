# Specificity of transfer

- `specificity.{png,pdf,svg}` — the paper figure (`fig:specificity`). x = teacher trait (dog, cat,
  lion, pig); per trait two bars: the lift of the teacher's own animal (blue) and the **summed** lift
  of the nine other scored animals (grey), each p(student) − p(control) with the four same-model
  receivers (orig, fs1, fs2, fs3) pooled, whisker = 95% cluster bootstrap (cluster = receiver ×
  teacher). Panel A = preference prompts, panel B = off-topic prompts, one shared y axis (B's bars
  carry their value). Statistic `trait_specificity.summarize_pooled_trait_vs_others`, drawn by
  `plot_specificity_bars` in the `figure2_family` style; written by
  `notebooks/trait_specificity_models.ipynb` (§6c).
- `drafts/specificity_by_receiver.{png,pdf,svg}` — the per-receiver companion (lines over the four
  receivers, solid = teacher's animal, dashed = mean of the nine others; `plot_trait_vs_others_sets`).
- `drafts/` otherwise — the working charts of the same notebook (per-animal lines, 2×2 by trait,
  combined panel, per-model lift, off-topic per-model lift) and `trait_specificity_trajectory.png`
  from `notebooks/trait_specificity.ipynb` (the ds1-strip trajectory readout).

```bash
uv run python -m notebooks.builders.build_trait_specificity_models_nb   # specificity.* + drafts/*
uv run python -m notebooks.builders.build_trait_specificity_nb          # drafts/trait_specificity_trajectory.png
```
