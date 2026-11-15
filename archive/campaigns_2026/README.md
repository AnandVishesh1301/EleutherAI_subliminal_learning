# archive/campaigns_2026 -- the 2026-07/08 student-HP and optimizer campaign drivers

Kept for provenance, **not maintained** and not collected by pytest (`testpaths = ["tests"]`).

These modules drove the automated 2026-07/08 student-hyperparameter (`student_hp_campaign`)
and optimizer-parameter (`optparam_campaign`) campaigns: the plan constants and recipe
selection (`campaign/`), the stage machines (`pipelines/`), the launchers and one-off checks
(`scripts/`, including `check_step0_init_equality.py`, superseded by
`scripts/checks/check_seed_identity.py pythia --init-step`), and their tests (`tests/`).

Layout mirrors the repo at the time (`campaign/` was `src/subliminal_learning/campaign/`).
The one contract that outlived the campaigns -- the cross-init pair list -- lives on as
`subliminal_learning.gkd.pairs`; the probe-frame reductions the campaign notebooks used
(`end_of_run_probes`, `peak_probes`) as `subliminal_learning.gkd.probe_frames`. The archived
code still spells its own imports as `subliminal_learning.campaign.*` / `pipelines.*`; running
it would need those packages restored on `sys.path`.
