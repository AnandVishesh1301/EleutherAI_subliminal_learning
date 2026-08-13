# The collapse claim chain — what we must prove, and the evidence per link

*Created 2026-08-12. Status: narrative under active discussion — sections marked ⚠ DISCUSSION
record open disagreements/decisions, not settled science. Companion synthesis notebook:
`notebooks/collapse_evidence_review.ipynb` (its §4.8 audit predates the clamp tier and the
at-scale spread results; this file is the up-to-date statement of the chain).*

**The goal in one paragraph.** We observe a simple rule in the student hyper-parameters
(lr, α, rank) that dictates — across teacher lineages, student inits, animals, and even with no
trait in the data — where transfer (and the model's coherence generally) collapses:
K = lr·α^b·r^c ≥ K*. The law is remarkably consistent, and we believe it works because the HPs
set the *size of the LoRA update*, and the update's size is what kills. The chain below is what
has to be demonstrated for that account to stand.

---

## The chain

| # | Claim | Status (2026-08-12) |
|---|-------|---------------------|
| **A1** | A collapse phenomenon exists and is empirically governed by the student HPs — and it is **not SL-specific** (generic distillation-stability) | **Proven at calibrated strength**; the trait-free leg rests on a thin 22-cell control |
| **A2** | The LoRA update's **size causes fate** — correlationally and causally | **Closed**, with a revision: the causal dose is **trajectory-like** (sustained exposure), the endpoint norm is its high-quality proxy; a bounded ~2× direction factor sets each update's personal threshold |
| **A3** | The student HPs **set that size**, via the update's exact two-term anatomy (frozen s·B·A_init vs moved s·B·ΔA) | **Proven** — the strongest link (calibrated dose law R² 0.97, both curvatures traced to the term crossover) |
| **M** | *Therefore*: the HPs cause fate **through** the dose (mediation — NOT raw transitivity, see below) | **Supported** by level-set consistency + mediator-blocking (clamp) + mediator-pushing (λ); one statistical residual remains, attributable to trajectory-vs-endpoint |
| **A4** | The size that matters is **Frobenius energy, not the spectral peak** | **Identified, not causally isolated for fate**: threshold geometry + mediation asymmetry + matched-energy surgery on the damage symptom all favor F; no intervention dissociates F from σ on collapse-fate itself; single-width caveat |

**On the logical form of M.** "HP→size" + "size→fate" ⇒ "HP→fate" is *not* needed as
transitivity (causation is not transitive in general). HP→fate is **directly causal by
experimental assignment** — we set (lr, α, r); fate follows the law. The scientific content of
the chain is the **mediation claim**: the HPs' effect on fate is carried by the dose. That is
what the level-set closure (consistency), the clamp (blocking the mediator blocks the effect),
and the λ-scaling (pushing the mediator produces the effect) jointly establish.

**Two ladders — do not conflate.** This chain (A1–A4) is the *causal story*. A different ladder,
`collapse_mechanism.ipynb`'s assumption ledger **A0–A4** (A0 threshold premise, A1 Adam
scale-invariance q=2, A2 quad-term dominance, A3 rank-sum scaling ρ, A4 no weight decay), is
about *deriving the law's exponents from optimizer mechanics*. Its verdict: the mechanism
**brackets** b (0.50–0.82), it does not derive it. Exponent-derivation can fail entirely while
this causal chain stands.

**Evidence classes** (following `nors_asks.ipynb` §0): **[cal]** calibrated non-adaptive strip
(outcome-independent design; p-values mean what they say) · **[interv]** eval-time intervention ·
**[interv-train]** training-time intervention · **[ident]** exact identity/decomposition ·
**[adaptive]** adaptive-era grids (discovery + direction-replication only) · **[recorded]**
quoted from a session/wipe-lost analysis, not recomputable from disk.

---

## A1 — the phenomenon, and its scope

**Claim.** Per-run fate is bimodal with an empty trough; the boundary is a power law in
(lr, α, r) with a stable threshold K*; it replicates across lineages, inits, and animals; and it
is not about the trait.

**Evidence for:**

- Bimodal fate, empty trough **[cal]**: on the strip (890 non-diverged runs), 39% of runs sit at
  test_p ≤ 1e-4 while the alive mode has median p 0.035 — only 2.6% of runs fall anywhere in the
  two-decade gap between 1e-4 and the 0.01 cut. The label is a property of the data, not of the
  cut. *(Strip harvest; first seen in every adaptive campaign — `collapse_evidence_review`
  Part I.)*
- Six-arm law: b within ~0.05 of 0.56, c ≈ −0.09, I² ≈ 0 across 3 teachers × 2 init families.
  *(`collapse_law_generality.ipynb`; recomputed in `collapse_evidence_review` §2.2.)* **[adaptive]**
- K* transfers cross-init at ~1.02× *(`top20_krungs_ds1.ipynb` §5, §9)* and cross-animal —
  cat/lion/pig run on dog's anchor *(recomputed in `collapse_evidence_review` §2.4)*. **[adaptive]**
- Calibrated re-establishment with **regional** structure: b_L 0.499±0.064 (α≤32), b_H 0.882±0.098
  (α≥64), bend 3.3σ; c −0.091±0.032; second boundary (optimization divergence) at K/K* ≳ 12.
  *(nors_asks §7.1, on the 890-run 8×7 strip.)* **[cal]**
- **Not SL-specific**: distilling the *base model's own* numbers (no trait anywhere) collapses the
  student's base prior at K ≳ 1.8e-3. *(`collapse_mechanism.ipynb` §6c — 22 cells × 3 seeds;
  reframe recorded in `collapse_evidence_review` Part III.)* **[adaptive, THIN]**
- Dead runs still fit their training data — collapse is not optimization failure.
  *(`collapse_law_generality` §12, postmortem CE on focus:ds1 cells.)* **[adaptive]** — no strip
  version exists (per-step loss curves are unrecoverable and the postmortem was never rerun on
  strip cells); a strip rerun would be disk-cheap if this line ever becomes load-bearing.

**Holes / complications:**

1. **The trait-free control is thin** (22 cells / 12 matched) and shows an unquantified ~2×
   offset (trait-numbers K* ≈ 0.8e-3 vs trait-free ≈ 1.8e-3; in 4/12 matched cells treatment is
   suppressed below the whole control range — the trait channel is marginally *more* fragile).
   The trait-free boundary factorial is the pivotal missing experiment. ⚠ DISCUSSION (design).
2. **The DEAD label is a per-arm convention** (live-mode/10). Cross-animal claims inherit
   per-animal base priors (dog ds1 base p 0.0133; cat 0.0048/0.0033 — never reuse the stale
   "3e-5" figure). Proposed continuous outcome: z = log p_treat − log null(K), binarize only at
   reporting *(collapse_mechanism §6c)*. ⚠ DISCUSSION (unresolved; affects every fate-based fit).
3. **Scope lines** (permanent, descope-able): one width (pythia-160m, d=768), one pool family,
   ep20 budget, wd = 0, Adam only, **cosine schedule with 5% warmup** — "lr" throughout means
   *peak* lr of a fixed one-parameter schedule family. ⚠ DISCUSSION (whether a constant-lr
   ablation is needed; see Open Questions #6).

---

## A2 — update size causes fate

**Claim.** Fate is a (soft) threshold on the update's size; moving the size moves the fate, in
both directions, at eval time and during training.

### Correlational leg **[cal + adaptive]** — closed

- P(death) monotone in dw = ‖ΔW‖_F/‖W₀‖_F: **AUC 0.998 on the strip** (890 runs, 371 dead) —
  as good as the fitted law's own K (0.997) **[cal]**. Replicates the adaptive-era A0 grid test
  (0.990 over 655 fated ep20 runs, *collapse_mechanism §6*). Caveat carried from Nor's audit:
  pooled fate AUCs saturate for *both* norms (dw_spec 0.995), so AUC is evidence that dose
  thresholds fate, not evidence for which norm — the norm question is A4's.
- What death *is* — loss of selectivity: dead runs leak Pile-logit KL vs base at **AUC 0.999**.
  Full distributions on the strip (n = 268 alive / 196 dead, ftseed-1 probe population):

  | Pile KL (nats) | P5 | P25 | P50 | P75 | P95 |
  |---|---|---|---|---|---|
  | alive | 0.08 | 0.15 | 0.34 | 0.71 | 2.09 |
  | dead | 5.48 | 13.72 | 15.41 | 17.19 | 19.69 |

  The distributions are disjoint at the 5–95% level: **0%** of dead sit below the alive P95;
  0.7% of alive sit above the dead P5 — a literal ~3.4-nat gap, not just separated medians.
  The trait-channel drift (raw final-block basis) is alive [0.51, 0.95] (P5–P95) vs dead with a
  wildly heavy tail (P50 61, P95 ~9.7e3) — **that tail is the deep-dead degeneracy artifact**
  (input-independent, channel-blind: trait/pile drift ratio 0.83), *not* selective trait-channel
  movement. The clean selectivity statement is conditional: *near the boundary / at matched
  dose*, dead runs move the trait channel like alive runs but leak KL. *(nors_asks §4b.)* **[cal]**
- **The containment jump**: regressing log Pile-KL on the update's (energy, spread), the dead
  indicator carries a **+1.87 log-nat offset (≈6.5×) at matched dose** (p ≈ 3e-92); slopes also
  differ by fate (γ_E 0.73 alive vs 0.22 dead — saturation). KL is *not* a smooth function of
  measured update size — crossing the boundary is an event, not the tail of a smooth response.
  *(nors_asks §4, fate-structure block.)* **[cal]**

### Causal leg, eval time **[interv]** — closed

- **λ dose-response**: rescale the trained ΔW by λ (direction untouched) — fate flips
  monotonically in **both** directions in every cell (kills alive, revives dead); at scale 109
  cells / 218 curves. Alive curves trace the transfer hump causally within a single run.
  Replicates on ws1 (8/8) where the cos marker was absent — shared mechanism, arm-dependent
  fingerprint. *(nors_asks §5.5, §5.5b; first 8 cells in `causal_plan.md` Exp 1.)*
- **Near-boundary death is masking, not erasure**: dead models revive with trait above base at
  λ ≤ 0.5. *(nors_asks §5.5.)*

### Causal leg, training time **[interv-train]** — closed 2026-08-12, with a revision

The clamp tier *(nors_asks §5.5c; design frozen in `run_logs/clamp_design_v1.json`)*: train
fresh runs in 7 verified-all-dead strip cells with a max-norm projection (B-rescale) capping the
pooled dose at ratio × per-rank θ̂. Survival is monotone in the ceiling:

| ceiling / θ̂ | 0.5 | 0.6 | 0.8 | ≥1.0 |
|---|---|---|---|---|
| survive | **14/14** (incl. all lr5e-4) | 11/14 | 3/14 | 0/54 |

Sharpest single-cell demonstration (lr5e-4_r64a8, both seeds): fate walks alive→dead purely with
the ceiling while the HPs stay pinned at lethal values. Direction-corruption is **refuted** (no
ceiling-independent death). Practically: a ~0.5×θ̂ clamp = train at any lr with zero seed deaths.

**The revision — the causal dose is trajectory-like.** *Endpoint*-dose sufficiency is **false**:
58 clamped runs realized final dose < 0.95×θ̂_F and 47% of them still died — in cells whose
unclamped runs die *above* θ̂, the clamped ones die *below* it. Sustained exposure (riding the
ceiling) kills where a brief endpoint approach survives; the operative threshold under a
ceiling-riding trajectory is ~0.6×θ̂. The endpoint dw of the correlational leg is therefore a
**proxy** (an excellent one — at fixed budget, endpoint ordering tracks trajectory-exposure
ordering across cells) for the causal quantity, not the causal quantity itself. Consistent
independent evidence that the decision is early and exposure-like — **all of it adaptive-era,
with no calibrated version yet**: endpoint-q crossover 1.48 (ep5) → 1.16 (ep20) —
ballistic→diffusive *(collapse_mechanism §6b)*; K* ∝ epochs^−0.2 (a pure accumulation story
predicts ∝ 1/epochs); endpoint threshold θ grows sublinearly, brackets √T *(collapse_mechanism
§6, softened form)*. **[adaptive]** The calibrated successor is in flight: the ckpt-replay
pilot (2026-08-13) retraces 4 strip cells' original ftseed-1/2 runs with 40 checkpoints each —
the first within-run dose/decision trajectories on calibrated cells.

**Holes / complications (state these, they are the honest wording):**

1. **Size is causally sufficient but not the complete cause.** The λ flip dose is
   history-dependent. Full-sweep estimate (2026-08-13, all 172 scored crossings over the 109
   rule-frozen lamstrat+shellext cells, both seeds): raw kill-band/revive-band ratio 3.59×
   (confounded — trained-dead cells sit deeper by design); **covariate-adjusted (log K_rel +
   log α + log r, cell-clustered): 3.05× [2.49, 3.73]**; the clean-but-small within-cell
   mixed-fate pairs give 2.03× (n = 9, all > 1, range 1.4–10). **The factor is α-structured:**
   ~4.1–4.4× at α≤8, falling to ~2.3× at α=128 (α=256 reads 1.1× but on 2 kills — suggestive
   only) — history/direction dependence is strongest where the *moved* (learned) term dominates
   the update and shrinks toward 1 in the frozen-dominated regime, which is mechanistically
   coherent: the frozen term's direction is a random init, identically distributed for alive
   and dead runs; only the learned direction can carry history. Flip dose is also a *cell
   property* (75 dual-seed crossings, corr +0.63–0.76), and within a mixed cell endpoint size
   is coin-flip on which seed dies — strip: dying seed smaller in 6/9 mixed cells, p = 0.51
   **[cal]**; the adaptive large arms, with many more mixed cells, even pointed the wrong way
   for a size story (dying seeds slightly **smaller**, z ≈ −3.2). The factor is real and
   **unnamed** — cos(ΔW,W₀), summary spread, srank, and module concentration are all ruled out
   *(§4.6/§5.5b; sliver surgery null — 17 adaptive-era mixed cells)*.
   Honest sentence: *dose sets death probability between cells and suffices to flip fate; a
   direction-quality factor of ~2–4× (largest at low α) sets each update's personal threshold.*
2. **Trajectory-vs-endpoint quantities are thin**: the ~0.6×θ̂ operative threshold is one number
   from 7 cells in one regime mix. ⚠ DISCUSSION (extend clamp cells? see Open Questions #2).
3. **Why crossing kills is open** — no mechanistic account of why exceeding θ destroys the
   behavioral prior rather than adding noise ("mechanism of the mechanism"; fine to leave open,
   frame as open).

---

## A3 — the HPs set the size

**Claim.** The realized update size is a smooth, near-exact power law in (lr, α, r), and each
exponent is traced to the update's two-term anatomy.

**Evidence for:**

- **The dose law [cal]**: log dw_fro ~ log lr + log α + log r on the strip:
  **lr^1.05 · α^0.69 · r^−0.24, R² ≈ 0.97**, cluster-robust by cell; both curvatures real and
  jointly identified (lr² +0.09; α² +0.018 ± 0.007, p = 8e-3 — the lone-α² adaptive probe was
  unidentifiable and its retraction reversed on the strip). *(collapse_evidence_review §4.2;
  strip source nors_asks §5.)*
- **Two-term anatomy [ident + cal]**: ΔW = s·B·A_init (frozen) + s·B·ΔA (moved); A_init exactly
  replayable (`set_seed(ftseed)` → `get_peft_model`; wrong-seed cos ≈ 0.008 floor). Both terms
  fitted on the strip (890 runs, cell-clustered):
  **frozen lr^0.739±0.003 · α^0.853±0.004 · r^−0.461±0.003 (R² 0.997, straight in lr)** vs
  **moved lr^1.302±0.012 · α^0.498±0.017 · r^−0.107±0.012 (R² 0.976, convex: lr² +0.064±0.005)**.
  The crossover explains both curvatures of the total on both axes: the moved term has the
  higher lr exponent (takes over at high lr → lr² +0.09 positive) and the frozen term the
  higher α exponent (takes over at high α → α² positive) — the total's exponents (1.05, 0.69,
  −0.24) sit between the terms'. Dominance as an α-dial replicates on all 6 adaptive arms
  (quad/cross 3.1–4.8× at α≤8 vs 0.57–0.77 at α≥32) **[adaptive, replication]**.
  *(nors_asks §1.4, §5.1–5.3; per-term α/r exponents measured 08-13; collapse_mechanism §6e.)*
  The learned factor ‖BA‖/‖W₀‖ follows directly from the strip dose law by removing the s = α/r
  prefactor: **lr^1.05 · α^−0.31 · r^+0.76 [cal]** — Adam partially compensates the
  s-amplification (α exponent below 0) and rank adds coherent channels (r exponent below 1);
  adaptive version α^−0.34 · r^+0.78 *(collapse_mechanism §6f)* agrees.
- **Composition licensing [ident]** — corrected 2026-08-13 (full gap distributions, 932 strip
  runs / 44,736 module rows). Tightness and HP-constancy are different properties, and only
  constancy licenses exponent-adding. Measured: the **frozen**-term Frobenius gap
  ‖B‖_F·‖A₀‖_F/‖B·A₀‖_F is loose but essentially **deterministic at √r** (median 2.821 at r8 vs
  √8 = 2.828; per-rank quantile spread ~±0.5%; HP drift lr^0.000±0.000 · α^−0.000±0.000 ·
  r^0.491±0.000, residual ±0.9% — the near-isotropy of the Kaiming A₀) → composition licensed
  for the frozen term in lr/α, with the √r carried explicitly. The **moved**-term Frobenius gap
  is **NOT HP-flat**: lr^0.042 · α^0.115 · r^0.184, R² 0.71, residual ±24% (IQR 1.72–2.24 at
  r8) → no Frobenius factor-composition for the moved term; its exponents (lr^1.31, convex) are
  **measured directly on ‖B·ΔA‖**, never composed. The *spectral* bounds are the tight ones
  (frozen ≤1.28 at r128; moved median ~1.05) — tight, but tightness was never the requirement.
  *(nors_asks §1.5; distributions recomputed 08-13.)*
- **Functional reality of the terms [interv]**: strip splices (932 runs × both splices × both
  seeds, 1,864 evals) give a 32/32 **double dissociation** — the trait rides the dominant term.
  *(nors_asks §5.4.)*
- **Cross-arm replication [adaptive]** — honest version (collapse_mechanism §6f, per-arm
  alive-cell fits of log dw_fro):

  | arm | cells | lr | α | r | R² | implied b | implied c |
  |---|---|---|---|---|---|---|---|
  | data-seed1 | 164 | 0.985 | 0.578 | −0.124 | 0.767 | 0.587 | −0.126 |
  | data-seed2 | 103 | 1.043 | 0.649 | −0.266 | 0.935 | 0.622 | −0.255 |
  | data-seed3 | 46 | 1.127 | 0.695 | −0.281 | 0.970 | 0.616 | −0.249 |
  | weight-seed1 | 120 | 0.973 | 0.606 | −0.218 | 0.906 | 0.623 | −0.224 |
  | weight-seed2 | 34 | 0.951 | 0.510 | −0.104 | **0.325** | 0.537 | −0.110 |
  | weight-seed3 | 20 | 1.041 | 0.648 | −0.270 | 0.977 | 0.622 | −0.260 |

  This is a **consistent band** (lr 0.95–1.13, α 0.51–0.70, r −0.10..−0.28), *not* "exactly the
  same law": ws2 is weak (34 cells, R² 0.33, consistent-only) and ds3/ws3 are thin (46/20 cells).
  The precise statement lives on ds1 (and now the strip); the other arms replicate sign and
  rough magnitude.

**Holes:** scope only (single width/pool/budget/optimizer/schedule — shared with A1). The dose
law has **never been refit per animal** — it should be nearly animal-free by construction (the
trait enters only through the numbers distribution), and the adapters exist on disk for a cheap
check. ⚠ DISCUSSION (whether to run the disk-only per-animal refit).

---

## M — the mediation: HPs cause fate *through* the dose

**Claim — two statements, one about cells and one about individual runs. They sound similar
and are not.**

*About cells:* if you know how big an update each HP setting produces, you can draw the
dividing line between dying and surviving settings without ever looking at a fate — the fate
boundary sits exactly where the update size crosses one threshold. This is supported (the
argument and its checks are below).

*About individual runs:* take two runs whose final updates are the **same size**, one trained
at lr 2e-4 and one at 5e-4. If the final size were the whole story they would die at the same
rate — they don't; the higher-lr run dies slightly more often (p ≈ 1e-3, the complication
below). So the final size is not the complete story for a single run. The chain's position is
that the *path* is: the higher-lr run spent more of training with a large update even though it
ended at the same place, and the clamp showed that sustained exposure, not the endpoint, is
what kills. The final size is a summary of the path — a very good one (AUC 0.99), but lossy,
and the small run-level residual is exactly what that lossiness looks like.

Both statements can hold at once: the boundary between cells sits on the size threshold, while
among individual runs at matched *final* size the path still matters a little.

### The level-set argument — and when the two global fits should agree (rewritten 2026-08-13, after Angela's challenge)

The hypothesis is simple: dose is a function of the HPs, log dose = f(x) + ε with
x = (log lr, log α, log r) and seed noise ε, and a run dies iff f(x) + ε ≥ t. If that is true,
the fate boundary in HP space must be the t-contour of f — so the natural-looking test is to
fit both surfaces globally on the same cells and compare exponents. **That naive test is not
valid, and the reason is not a technicality — the two fits estimate different mathematical
objects.**

**What each fit estimates.** Both "global exponents" are summaries of the gradient field ∇f(x).
The OLS fit of log dose on x returns the best linear approximation to f over the *whole design
box* — for a symmetric grid, the box-averaged gradient (at our grid, ∇f at the box center,
since the quadratic terms average out there). The logistic fit of dead/alive on x has a
completely different information structure: runs deep in the alive or dead regions contribute
almost nothing to the coefficient estimates (the fit only needs them on the correct side); all
information about the coefficient *ratios* comes from the thin transition band around the
contour f = t. The misspecified linear logit therefore converges to the *contour-averaged*
gradient — ∇f averaged along the death boundary itself, weighted by where the design puts
points near it. **OLS averages the gradient over the box; the logistic averages it over one
contour.**

**When they agree.** These two averages coincide exactly in two cases: f is a plane (∇f
constant — the single-power-law world, where Angela's intuition "the globals must match" is
exactly right), or f is a *ridge*, f = g(βᵀx) for monotone g — curved, but with all contours
parallel. So curvature per se does not break agreement. What breaks it is curvature that
**twists the gradient's direction** across the box: then different contours (and different
points along one contour) have different orientations, and the box average and the contour
average generically differ — even with an exact threshold and zero noise.

**A toy that shows it with no noise at all.** Take f(x₁, x₂) = x₁ + x₂ + 0.3·x₁² on the box
[−1,1]², exact threshold t = 0.5. OLS over the box: the quadratic averages out, slopes (1, 1),
ratio 1.00. But walk along the death contour: at its top end (x₂ = 1, x₁ ≈ −0.56) the local
slope ratio is 1.50; at its bottom end (x₂ = −1, x₁ ≈ 1.04) it is 0.62. A single logistic fit
returns some design-weighted average in between — not 1.00 except by coincidence. One
phenomenon, one box, a literal threshold: the global OLS ratio, the global logistic ratio, and
the regional ratios are all different numbers, and none is wrong. They are different summaries
of a twisted field; only the *regional* comparisons correspond point-to-point.

**Our surface is the toy, twice over.** The measured dose surface has
∂f/∂(log lr) = 1.05 + 0.18·x̃_lr and ∂f/∂(log α) = 0.69 + 0.036·x̃_α (the quadratics are the
frozen/moved crossover, fit on weight norms alone — fate data never enters). And the death
boundary runs *diagonally* through the box: at high α it sits at low lr. Moving along the
contour toward high α therefore raises the numerator (α² term) while lowering the denominator
(the lr-slope is smallest at low lr) — both effects push the local ratio up, which is the
predicted climb from b ≈ 0.53 (L) to 0.88 (H). The like-for-like regional test is then sharp:
**L predicted 0.530 vs measured 0.499±0.064 (|z| = 0.5); H predicted 0.882 vs measured
0.882±0.098 (|z| = 0.0)**; c closes via the measured θ rank-drift (−0.110 → −0.084 vs fitted
−0.085). *(nors_asks §7.2, §5.3; θ-flat-in-α verified directly, p = 0.20.)*

**The null-model check (2026-08-13) — computing what the global gap *should* be instead of
arguing about it.** Keep the real runs and their real measured doses; assign "dead" purely by
dose ≥ one constant (chosen only to match the total death count — no per-run fate information
used); run the identical global logistic on those labels. This is the exact prediction of the
threshold world for this estimator on these cells — the fake labels feed no result, they exist
only to be compared against, like simulating a fair coin to judge 60/100 heads. Outcome, on the
890 non-diverged strip runs: real-fate global b = 0.623 vs threshold-world b = 0.664 (OLS ratio
0.654); per-regime and global differences ~0.06 with cell-bootstrap CIs all including zero
(global −0.057 [−0.161, +0.067]). One threshold reproduces **97.3%** of the actual fate labels;
give it the measured rank drift and it is **98.7%**, and the c mismatch reconciles too
(sim −0.271 → −0.129 vs real −0.154). **No residual gap survives: the OLS-vs-logistic global
mismatch is estimator geometry, not a second phenomenon.** One honest footnote: the point
differences share a sign (real b slightly below threshold-world b in every segment — the
direction of "higher lr slightly deadlier than its endpoint dose implies"), consistent with the
run-level lr residual below, but not distinguishable from noise in the boundary orientation at
this n.

**Why this is not "fitting an answer that explains the difference away."** The quadratic terms
were measured on weight norms only, with independent mechanistic origin (the two-term
crossover) and their own uncertainties (lr² +0.092±0.004; α² +0.018±0.007, p = 8e-3). That one
frozen surface then has to hit four things at once: the L slope, the H slope, the bend's
location, and the threshold-world global value plus ~98% of individual labels. It does — the
check is over-identified, not tuned. The flip side, conceded: **the global comparison is a weak
test in both directions** (both globals are averages of the same field and tend to land near
each other whether or not the story is right); the discriminating power lives in the regional
closures and the label-level agreement. Role in the chain: consistency / sufficient-statistic
evidence — **not causal by itself**.

**Blocking the mediator blocks the effect [interv-train]**: the clamp (A2 above) holds HPs at
verified-lethal values and caps the dose → 14/14 survival at 0.5×θ̂. The textbook mediation
intervention.

**Pushing the mediator without the HPs produces the effect [interv]**: λ-inflation kills
gentle-HP runs (and deflation revives lethal-HP runs).

**The one complication (state it; do not hide it).** On the calibrated strip (890 runs, 449
cells, cluster-robust by cell), the nested logit dead ~ log dw_fro + log HPs leaves an HP
residual at **p = 1.3e-3** **[cal]**: at matched endpoint dw, higher-lr runs die more — the
endpoint norm does not fully mediate statistically. Residual coefficients given dw_fro: lr
+1.63, α +0.79, r +1.06 (the rank loading matches the θ_F rank drift); magnitudes are
seed-unstable (near-separation regime, pseudo-R² 0.925), so read the signs, not the sizes.
The residual was first found on the adaptive population *(collapse_mechanism §6g, 655 runs,
p = 4e-3, lr-loaded, rank nil)*, where one further fact is known that the strip cannot yet
check: adding the direction marker cos(ΔW,W₀) makes the HP block null (Wald p = 0.51) —
(size, direction) jointly screen off (lr, α, r) **[adaptive]**; cos is not computed on the
strip (a disk-only dw-sweep pass would add it). Two readings, not mutually exclusive: (a) the
residual is the **endpoint-vs-trajectory gap** (higher lr rides nearer the ceiling en route to
the same endpoint — exactly what the clamp later showed matters); (b) it is the direction
factor, for which cos is a *marker* (its surgery was null). Reading (a) is an inference, not a
result. ⚠ DISCUSSION (a checkpoint-based trajectory-exposure statistic could test (a) directly;
the ckpt-replay pilot, 2026-08-13, produces the raw material).

---

## A4 — Frobenius, not spectral

**Claim.** The dose the fate mechanism reads is total energy ‖ΔW‖_F, not the peak σ_max(ΔW).

**Evidence for (three independent legs):**

1. **Threshold geometry [cal]** *(nors_asks §2, Test 1; intervals §2c)*: if fate thresholds the
   *true* dose D, then θ measured in D is one number regardless of incidental parameterization;
   measured in a wrong norm related to D by a rank-dependent factor (σ = F/√srank, and srank
   varies with rank), the apparent threshold must drift with rank. Measured rank-drift:
   **θ_F −0.110 [−0.124, −0.066] vs θ_2 −0.172 [−0.230, −0.117]**; the paired difference
   [−0.142, −0.017] excludes zero. This is an **identification criterion, not a causal proof** —
   and θ_F is the *flattest* threshold, not flat (its own drift is real and is absorbed into c).
2. **Mediation asymmetry [cal + adaptive]** *(strip refit 2026-08-13; origin collapse_mechanism
   §6g items 2–3)*: on the strip (890 runs, 449 cells), as the sole mediator of fate dw_fro
   achieves pseudo-R² **0.925** while dw_spec achieves **0.871**, and the residual α coefficient
   left by spec is ~6× the one left by fro (+4.59 vs +0.79) — the spectral norm misses exactly
   the α-shaped part of the dose signal. Both norms leave a small detectable HP residual on the
   calibrated design (p = 1.3e-3 vs 2.5e-3), so the honest wording is: **F absorbs far more of
   the HP→fate signal than σ; neither is a complete endpoint mediator.** The adaptive-era
   version *(655 runs: prsq 0.811 vs 0.643, Wald p = 4e-3 vs 3.2e-10)* showed the same
   asymmetry with a more dramatic p-value gulf — that gulf was population-specific and should
   not be quoted as the headline. Also the matched-spectral α-dependence of the A0 test is a
   norm artifact that vanishes in Frobenius *(collapse_mechanism §6)* **[adaptive]**.
3. **Matched-energy surgery on the damage symptom [interv]** *(nors_asks §4, §5.5b)*: in the
   (energy, spread) basis, spread at fixed energy *increases* damage (γ_S on KL +0.51
   [+0.40, +0.65]; positive within every rank), where the peak theory requires γ_S < 0. And
   interventionally: concentrating the update onto its top direction at exactly matched F —
   which *raises* σ_max by √srank — **reduces Pile KL in 199/224 splice sets** (median −0.90
   nats). The peak→displacement account is refuted twice over.

**The fate-level objection, and its resolution [interv]** *(nors_asks §5.5b, the shell-scale
fate-transition analysis)*: at scale, top1 does kill the trait in 14/85 alive sets (flat: 1/85) —
naively contradicting "F is the fate dose" (both reshapings hold F fixed). But in **12/14 of the
top1 kills, Pile KL falls** (median −0.60 nats) and the killed cells are boundary-adjacent
low-p cells (p_orig median 0.026): these are **carrier-deletion deaths** — the trait rides the
discarded non-dominant directions, and truncation removes the *content* while displacing generic
behavior *less*. Mirror image: flat revives 10/139 dead sets with KL falling in 10/10 (median
−4.93) — damage-dilution of shallow-damage dead. **The binary DEAD label conflates two
mechanistically distinct events**: collapse-death (containment failure, ~15-nat KL) and
carrier-deletion death (trait removed, KL *reduced*). The reshaping surgeries flip fate mostly
via the second; the Frobenius/peak question concerns only the first. This split must be a
first-class sentence in any writeup — a reviewer will find the 14 kills.

**Holes (this is the weakest link; say so):**

1. **No intervention dissociates F from σ on collapse-fate itself.** λ scales both norms
   together (silent on F-vs-σ); the σ-clamp arm physically drags F with it (trained updates are
   low-srank; spec-clamped runs realize F at 0.67–1.50×θ̂_F) — **training-time norm attribution
   is impossible with this design**; the surgeries' fate transitions are dominated by
   carrier-deletion. A4's causal leg is really about the *damage syndrome* (KL); A4-for-fate
   rides on legs 1–2 (identification + mediation) plus the sibling-symptom bridge (see Open
   Questions #7). ⚠ DISCUSSION.
2. **"HP-agnostic threshold" is not a causal argument** — correct. It identifies which norm the
   mechanism *plausibly reads*; it cannot rule out a third variable proportional to F across
   this design.
3. **Single width** (d = 768). Adam's per-entry step normalization ties norm scalings to
   parameter counts, so the F-vs-σ verdict could in principle be width-specific (Nor's
   objection). One boundary-band replication at pythia-410m retires it; otherwise every norm
   claim carries an explicit "at this width" scope line. ⚠ DISCUSSION (descope vs run).
4. **flat's spread test is off-support at r ≥ 16** (it pushes srank → r vs observed 1–6.2);
   γ_S was always a local claim.

---

## Retired / stale — do not cite

- **cos(ΔW, W₀) as mechanism** — retired; surgery null (max |Δp| 0.007, 1/34 flips). It is a
  *marker*. *(nors_asks §4.6-equivalent; causal Exp 2.)*
- **Summary spread / srank / module concentration as the history axis** — refuted at n=6 and at
  sweep scale (kills srank ρ ≈ 0 raw; conditional signal is spread-*protective*).
- **"The lr exponent should be 2" (strict A1 / q=2)** — stale founding assumption; q is
  time-dependent (1.48 ep5 → 1.16 ep20).
- **"θ(α) is flat", unqualified** — split it: training-time θ flat in α (p = 0.20, verified
  08-12); the *eval-time* λ-crossing threshold drifts (slope −0.081) — different objects
  (hysteresis gap). *(nors_asks §5.5b/§5.5c.)*
- **The 8-cell spread-surgery numbers** ("7/8", "flat non-monotone in rank") — superseded by the
  224-set shell analysis; the rank non-monotonicity was under-sampling.
- **Endpoint θ as a training-budget-invariant constant** — grows ~√T; and the clamp showed
  endpoint sufficiency false.
- **Wipe-lost, notes-only numbers** — "teachers are 66% of K* variance", the conditional-transfer
  +0.124/I²=0 refit: re-derive before citing.
- **Adaptive-era statistics that dissolved on the strip** — the lr regime shift, the lone-α²
  retraction, within-rank fate AUCs (retired as too-easy), per-arm ‖ΔW‖ fate contrasts on
  unequal HP mixes.
- **The untrained-base "p ≈ 3e-5" figure** — wrong for `animal_preference_test`; ds1 base p(dog)
  = 0.0133.

---

## Open questions under discussion (2026-08-12) — none queued, per standing rule

1. **ep5/ep10 strip** (Angela's proposal): rerun the non-adaptive strip (or its boundary band)
   at 5 and 10 epochs → calibrated K*(T), θ(T), and dose-law exponents-vs-T. Directly tests the
   trajectory story and the "large early steps kill" hypothesis; the existing T-dependence
   evidence (K* ∝ ep^−0.2, θ ~ √T, q crossover) is all adaptive-era. Design note: warmup is 5%
   *of total steps*, so ep5's early lr profile differs from ep20's in step units — decide whether
   to match warmup absolutely.
2. **More clamp cells?** Signs are settled (n-audit doctrine); the thin quantity is the ~0.6×θ̂
   operative threshold's constancy across regimes — an extension would target cross-dominated
   high-α cells + an H-regime cell, not more of the same.
3. **DEAD-label unification across animals/arms** — the z = log p − log null(K) proposal.
4. **Per-animal dose-law refit** — disk-only, cheap; closes "dose law consistent across animals".
5. **Trait-free boundary factorial** — strip-band on base-model (and rng) numbers with a
   pre-registered trait-free outcome (base-prior erosion + Pile KL); fits K*_free, b_free,
   c_free and the ~2× offset. The pivotal A1 experiment.
6. **Cosine-schedule dependence** — internal validity is safe (identical schedule everywhere;
   "lr" = peak lr of one family; same recipe as the reference papers), but the law's exponents
   are conditional on the schedule shape. Cheap probe: a few boundary-band cells at constant lr,
   matched on peak vs on mean lr — which matching preserves the boundary identifies which
   functional of the schedule K reads.
7. **Isolating Pile-KL vs collapse** — they are sibling symptoms of one containment failure; no
   intervention can move one without the common cause (any intervention goes through the
   weights). The two dissociate in *both* directions (carrier-deletion: trait dies, KL falls;
   near-boundary alive: KL elevated, trait alive) — so they are genuinely distinct symptoms, not
   one thing. Candidate resolution: make the outcome two-dimensional — containment failure
   (KL, continuous, primary severity readout) × trait payload (present/removed) — rather than
   seeking a causal arrow between siblings.
8. **Second width** — run one pythia-410m boundary band, or accept the "at this width" scope
   line on every norm claim.
9. **Name the ~2× direction factor** — subspace-identity probes (ΔW energy vs W₀ singular
   subspaces / activation covariance), then a targeted between-subspace surgery at matched norm.
   Until named, A2 carries the "sufficient, not complete" qualifier.
