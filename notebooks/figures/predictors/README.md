# Predictors of subliminal learning

- `length/` — sequence-length figures and their README (snapshot 2026-09-24).
- `predictor_r2_table.py` — variance explained by each predictor of same-model and
  data-seed-1 transfer (102 original Pythia-160M dog teachers; one student cell per recipe, no
  averaging over recipes/epochs; LaTeX shows `ep20_lr1e-4_r8a32`, CSV has all eight + `mean8`)
  on linear, log, logit and rank scales, before and after removing teacher validation
  preference (marginal R², ΔR² beyond teacher p, partial R² = squared partial
  correlation). Freezes the per-cell transfer to `recipe_cells_102.csv`; writes `predictor_r2_table.csv` here and the LaTeX fragment
  `docs/drafts/predictor_r2_table.tex`. Sources: `notebooks/figures/teacher_dce_vs_transfer_102.csv`
  (transfer, teacher p, excess CE, self-screen steerability),
  `teachers/dog/pythia-160m/<tag>/steerability/{self,steer_<target>}/aggregate.json` (ungated
  steering deltas, frozen to `steer_screens_102.csv`; each receiver column uses the screen into
  its own base: self for orig, steer_data-seed1/2/3 and steer_weight-seed1/2/3 for the siblings) and
  `docs/results/data/sequence_length_transfer_20260924_teacher_rows.csv` (length).
  The rank column reproduces the partial Spearman values quoted in the appendix.

```bash
uv run python -m notebooks.figures.predictors.predictor_r2_table
```
- `predictor_r2_receivers.py` — the same statistics with receivers as columns (orig, ds1–3,
  ws1–3; all 102 teachers at `ep20_lr1e-4_r8a32`, seed mean per cell), linear and rank. Reads
  the fixed-cell summaries through `frames.load_cell_summaries` and freezes them to
  `fixed_cell_transfer_102.csv`; writes `predictor_r2_receivers.csv` here and
  `docs/drafts/predictor_r2_receivers_{linear,rank}.tex`. fs1–3 (10 teachers) and 410M (34) are omitted:
  every teacher there has validation p ≥ 0.99, nothing to adjust for.

```bash
uv run python -m notebooks.figures.predictors.predictor_r2_receivers
```
