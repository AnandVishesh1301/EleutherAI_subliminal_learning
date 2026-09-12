# Intervention validity and output-safety audit — 2026-09-12

Technical audit, not a refreshed manuscript or a new GPU campaign. Historical raw
outputs were read, not edited. No additional evaluations were submitted.

## Norm construction and the claims it supports

`scripts/diffuse_ladder_surgery.py` constructs each module's injection as
`E * theta / sqrt(k) * U[:, :k] @ V[:, :k].T` in `match=fro` mode.
Orthonormal columns imply F = E theta and S = E theta / sqrt(k). The total
squared Frobenius norm over the fixed set of modules is E² sum(theta_m²),
also independent of k. In `match=spec`, F = E theta sqrt(k) and S = E theta.

The completed `direction_replication_20260911/injections/*.json` grid is entirely
fixed-F: two Pythia-160M receivers, three random seeds, six k values, four E values.
All 144 random plus 48 deterministic aligned conditions have the expected stored
norm coordinates. Nesting couples k comparisons; it does not decouple rank and S.

Mean random-injection change in excess CE from k=1 to k=32 (three seeds per receiver):

| E | Data-seed1 | Original | Individual comparisons with lower damage |
|---|---:|---:|---:|
| 0.5 | -2.43 | -2.46 | 6/6 |
| 1 | -2.01 | -2.18 | 6/6 |
| 2 | -0.80 | -1.64 | 5/6 |
| 4 | +0.09 | -1.44 | 4/6 |

These are repeated measurements on six receiver/seed frames, not 24 independent
random-frame replications. Intermediate k values are not uniformly monotonic.
This is a fixed-energy spectrum-spreading intervention, not a rank-only effect.

The same corrected data contain fixed-S diagonals: (k,E)=(1,0.5),(4,1),(16,2)
all have S/theta=0.5. Mean excess CE is 8.14, 11.62, 13.78 (data-seed1) and
8.32, 11.36, 17.99 (original). This challenges an S-only description but cannot
separate increasing F from increasing rank. No separate GPU evaluation is needed
for these particular fixed-S comparisons.

At E=1,k=32, aligned versus mean-random excess CE is 2.73 versus 9.96 (data-seed1)
and 5.68 versus 11.07 (original), at matched k and both norms. The aligned
conditions have zero intruder-positive modules out of all 48 checked modules.
This comparison directly changes direction class at matched spectrum. It does
not equate detector-positive synthetic directions with training-formed intruders.

## Three distinct reasons an artifact might be called old

1. **Nesting:** independent draws at different k were still genuine random
   interventions, but do not support a within-frame redistribution comparison.
   They must not be relabelled as nested observations.
2. **QR sign correction:** older `random_orthonormal_pair` and `subset_targets`
   used unsigned QR. This changes the intended random-control distribution,
   not just reproducibility. For rank one, both old U[0,0] and V[0,0] have the
   same deterministic sign, so the update's [0,0] entry is always positive.
   A 1,000-draw CPU check gave positive fractions 1.000 old versus 0.515 corrected.
   The archived OLMo code in `run_logs/olmo_mechanism_20260909/code/` confirms
   unsigned QR. Existing `signed_qr_v2` interpolation stamps refer to blending,
   not to correction of the target sampler.
3. **Cache identity:** receiver/config/dose/construction fields prevent uncertain
   reuse; their absence alone does not prove a measurement is numerically wrong.
   Recorded receiver references matched current checkpoint resolution for all
   3,382 unique runs in the historical surgery sources inspected. This check
   cannot reconstruct historical configuration changes or certify old weight bytes.

Old sign-biased interventions still demonstrate what happened under those actual
direction changes. They do not establish typical behavior under the corrected
isotropic control. Rerun requirements therefore depend on the claim retained.

## Rerun decisions

| Family | Decision |
|---|---|
| Pythia synthetic fixed-F concentration | Corrected 192-condition grid is complete; no repeat needed for nesting/sign correction. |
| Pythia full randomization and top/rest selectors | Fresh outputs cover all 2,501 finite census runs under `position_surgery_v3`; use these after downstream refresh. |
| Pythia full interpolation curves | Existing 12-run progression uses old random targets; remeasure if retaining claims about corrected random-target rotation curves. |
| Pythia super/sub/stop and subset interpolation | New campaign does not cover these. Corrected measurements are required for corresponding random-selector conclusions; old super/sub analysis covers 546 selected runs, not the whole census. |
| OLMo full randomization | Archived unsigned-QR campaign has 864 runs at three doses (2,592 original/random pairs). Corrected controls are needed if those cross-model results remain evidence for isotropic randomization. |
| OLMo detailed surgery | 154 selected runs; corrected random targets needed for retained detailed/rotation conclusions, not necessarily an expansion to the full population. |
| OLMo random synthetic injections | Two receiver grids predate the corrected nested sampler. Remeasure random arms if retaining the corresponding concentration/direction-class comparison. |
| Legacy fixed-spectral Pythia grid | Do not present as corrected nested data. Existing corrected fixed-F diagonals already supply some fixed-S comparisons; fill only missing coordinates needed for the claim. |
| Three-shape synthetic experiment | Diffuse-random arm uses the changed QR sampler; concentrated Gaussian rank-one and aligned arms do not. Prefer the corrected ladder as replacement evidence rather than duplicate the old experiment. |
| Learned-donor injections / donor-potency sweep | Learned directions are unchanged by random-target fixes. Refresh old random comparators if retaining donor-versus-isotropic-random rankings; do not automatically rerun all donors. The existing potency artifacts contain 2,501 donor runs. |
| Deterministic aligned injections, learned-spectrum flattening/concentration, natural intruder census | Not invalidated by nesting or QR target changes. Other provenance or measurement problems would require separate evidence. |

Source inventory: `docs/paper/evidence/direction_20260911/provenance.json` and the
raw paths it records; additional donor-potency outputs were inspected separately.
Historical files lack the new construction/sign stamps and are not silently
promoted to the new design. The manuscript/evidence snapshot was not regenerated
by this audit.

## Surgery output safety

Ordinary `position_surgery` resume now refuses existing stale/incomplete runs,
including variant extensions or requests for missing intruder counts. New runs
may be appended; existing observations and unrequested variants are preserved.
The writer checks again under a file lock and refuses conflicting same-run writes.
Identical retries do not rewrite the document.

Explicit `--redo` still replaces the entire named document, not merely selected
runs. It backs up both data and canonical manifest before removing either, using
unique paired backup names. Prefer a fresh output name for new variants/designs.
Each invocation gets a unique provenance report, referenced by new rows;
ordinary resumes preserve the canonical historical manifest.

These are output-handling changes, not intervention changes: they do not require
repeating `position_surgery_v3` measurements. Immutable campaign snapshots were
not modified. As before, jobs should own distinct output names; the file-level
checks are not a transaction spanning an entire multi-run GPU invocation.

Validation: 111 relevant CPU tests passed and touched Python files passed Ruff.
No historical data was deleted or overwritten.
