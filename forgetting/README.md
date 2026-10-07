# forgetting/ — the catastrophic-forgetting / collapse program

Distillation runs past a learning-rate × dose boundary "collapse": the student loses its
in-distribution (Pile) coherence and, with it, the transferred trait. This program
measures and explains that collapse: the boundary law (lr·√α), adapter norm / dose
readouts and eval-time dose rescaling, direction and position surgery on the trained ΔW,
the unigram anchor and Pile-CE coherence eval, intruder dimensions (Shuttleworth et al.),
two-part mediation, and the strip / tier-B trajectory campaigns. It was split out of the
main subliminal-learning tree on 2026-10-06; the main program still reads its dead-vs-alive
labels through the shared machinery listed below.

## Layout

| Path | Contents |
| --- | --- |
| `src/subliminal_learning/forgetting/weightspace/` | adapter loading, exact LoRA SVD, patching, surgery deltas, dose rescaling / census |
| `src/subliminal_learning/forgetting/coherence/` | Pile-coherence CE eval (config, shard, packing, bootstrap, results), unigram anchor, base-ckpt CE |
| `src/subliminal_learning/forgetting/analysis/` | `norms`, `w0_norms`, `adapter_decomp`, `kstar_structure`, `intruder_threshold`, `fidelity` |
| `src/subliminal_learning/forgetting/stats/` | `mediation` (two-part KL mediation) |
| `forgetting/scripts/` | the program's CLIs (65 scripts + `sweep_pile_coherence`) |
| `forgetting/pipelines/` | `pile_coherence_pipeline`, `adapter_decomp_pipeline` |
| `forgetting/configs/` | `experiment/{pile_coherence,teacher_ft_stripsupport_budgets,teacher_ft_stripsupport_ep20ckpt}.yaml`, `analysis/adapter_decomp/*.yaml` |
| `forgetting/notebooks/` | the program's executed notebooks (list below) |
| `forgetting/tests/` | tests of the moved modules (in `[tool.pytest.ini_options] testpaths`) |
| `forgetting/docs/` | `COLLAPSE_CLAIM_CHAIN.md`, `INTERVENTION_AUDIT_20260912.md`, and the gitignored scratch `causal_plan.md`, `backup.md`, `backup_exp.md` |

Library code sits under the installed package, so `import subliminal_learning.forgetting...`
works everywhere without sys.path changes.

## Running

From the repo root:

```
uv run python -m forgetting.scripts.<name> --help
uv run pytest forgetting/tests
```

Notebook builders write `forgetting/notebooks/<name>.ipynb` but still execute with the
kernel cwd at `notebooks/`, so figures keep landing in `notebooks/figures/` (the paper build
reads those paths).

## Compatibility redirects (why they exist)

Queue job records, run_logs campaign code and workers, and executed notebooks still name
the old paths. They keep working through these redirects:

- `scripts/__init__.py` appends `forgetting/scripts` to `__path__`: `-m scripts.<moved>` and
  `from scripts.<moved> import ...` resolve to the moved file.
- `pipelines/__init__.py` appends `forgetting/pipelines` to `__path__`: `pipelines.pile_coherence_pipeline`
  and `pipelines.adapter_decomp_pipeline` keep resolving.
- `scripts/adapter_norms_probe.py` is a runpy stub, because queued jobs run
  `uv run python scripts/adapter_norms_probe.py ...` by file path, which `__path__` does not cover.
- Five src shims re-export the new module (`import *` plus a module `__getattr__`):
  `subliminal_learning.weightspace` (a package whose `__path__` points at the new directory, so
  `subliminal_learning.weightspace.lora` etc. import), and
  `subliminal_learning.analysis.{norms,adapter_decomp,kstar_structure,intruder_threshold}`.
  A shim loads its target as a separate module object, so monkeypatch the new path.

New code should use the `forgetting` paths. The other moved modules had no external callers,
so they have no shims.

## What stayed in main, and why

- `coherence/{probe_io,probe_families,dclm_pool,c4_pool,dolma_pool}`: the pp1 / KL probe and the
  text pools are shared by training, `run_final_evals` and the transfer analyses.
- `analysis/labels.py`, `analysis/collapse_law.py` (shim + `load_law_runs`), `analysis/probe_cache.py`,
  `analysis/record_cache.py`, `analysis/stats.py`, `stats/collapse_law.py`, `stats/core.py`: the
  dead-vs-alive labelling that every transfer analysis filters on.
- `students/{clamp,early_stop,divergence}.py`, `training/ckpt_ladder.py`,
  `evaluation/ckpt_trajectories.py`: used by the main training and eval loops.
- `scripts/run_final_evals.py` (shared; imports `forgetting.scripts.fidelity_heldout_ce`),
  `scripts/convert_legacy_pilekl_to_rundir.py`, `scripts/subspace_identity.py`, `scripts/build_dd1b_*_nb.py`.
- `configs/training/mech_probe/`: a live run_logs worker names that path.
- `notebooks/figures/`: the paper build reads those paths.

## Notebooks

- `collapse_campaigns_k_alpha_rank.ipynb`
- `collapse_claims.ipynb`
- `collapse_evidence_review.ipynb`
- `collapse_mechanism.ipynb`
- `collapse_teachers.ipynb`
- `collapse_law_generality.ipynb`
- `collapse_surgery_and_sgd.ipynb`
- `collapse_two_factor_paper_figures.ipynb`
- `dose_rescaling_census.ipynb`
- `nors_asks.ipynb`
- `nonadaptive_strip_160m_peak.ipynb`
- `tierb_strips_integrity.ipynb`
- `trait_past_collapse_mechanism.ipynb`
- `pythia410m_collapse_and_peak.ipynb`
- `intruder_threshold_prior_work.ipynb`
- `seed_fate_direction.ipynb`
- `teacher_nonadaptive_olmo_qwen.ipynb`
- `pythia_direction_reassessment.ipynb`
- `orig_samemodel_ladder.ipynb`
- `ckpt_trajectories.ipynb`
- `top20_krungs_ds1.ipynb`
