# Pre-registration — sdfpos Part II: headline N = 8 383, 5 × 2 seeds, robustness

Status: **frozen 2026-09-24 before any Part II eval exists.** Constants live in
`src/subliminal_learning/campaign/sdfpos2_plan.py`; this file is the human-readable SSOT for
the Part II claims and design. Part I (`docs/PREREG_SDFPOS.md`, run sets
`20260922_sdfpos-*`, their `claims.json`) is a frozen input and is never rewritten. Plan of
record: `.claude/plans/sdfpos_teacher_corpus_limits.plan.md` §7.

Standing rules: no arm is ever dropped from a table or figure; `--force` is never applied to
a training or teacher command; the docgen pools and every corpora build are read-only.

## 1. Question

Part I asked whether the teacher's corpus changes the belief the teacher acquires and what
the on-policy student inherits, beyond teacher strength. Part II asks the same question at
the legacy headline corpus size (N = 8 383) with 5 × 2 seeds, on a second same-init pair,
at equal-N rungs, and with `packing=False`.

## 2. Fixed choices (identical to Part I unless stated)

- **Pairs (same-init only):** `pythia-160m@step143000 → same` and
  `pythia-160m-seed2@step143000 → same`; animal bear.
- **Teacher recipe:** LoRA r128 / α128 on all linear modules, lr 5e-5, batch 8, cosine, 3 %
  warmup, `max_steps = 750` over 2 048-token `wrapped` packing (Regime B). Size companions
  keep the Part I `ep5` regime. **`packing=False` run set:** one document per example,
  `max_steps = 4 931` (the packed teacher's 12.29 M tokens at 311.5 tokens / doc).
- **Student recipe:** `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16`, legacy rollout screen,
  quarantine `perplexity_ratio > 5 ∨ filter_drop_frac > 0.5`.
- **Leaves:** treatment = 5 teacher ftseeds × gseed {0, 1} (Part I arms: new ftseeds 3, 4
  join Part I's 0–2); controls = `base_teacher` students at gseed 0–4 per base.
- **Endpoint:** exact next-token `p(" bear")` (fp32 softmax) on `test` (primary),
  `favorite`, `test_sample`, `consensus`, `dislike`; no `off_topic`, no lm-eval.
  `delta = p_target(treatment) − mean p_target(Part II controls of the same base)`. The
  untrained base's own `base_p_target` per prompt set is the second reference.
- **Statistics:** two-stage cluster bootstrap (resample teachers within an arm, then leaves
  within a teacher), `N_BOOT = 10000`, seed 20260924; contrasts are differences of arm
  means under the same resampling, gseed-paired. Part I claims are re-read with it.
- **Tiers:** the ws160m W0 teacher tier rule per teacher; `weak` is labelled, never dropped.

## 3. Run sets (`<output_root>` = `/mnt/ssd-1/soar-subliminal/vishesh/sdfpos2/outputs`)

| Run set | Node | Base | Arms (`corpus_tag`) | ftseeds | Controls |
|---|---|---|---|---|---|
| `20260924_sdfpos2-main_160m` | cluster-3 | 160m | `pos-{start,middle,end,mix}-n8383`, `den-f{50,25,12,0}-n8383`, `val-{neutral,negative,positive}-n8383`, `pos-*-n4192`, `val-*-n2096` (`corpora_v3`) | 0–4 | g0–4 |
| `20260924_sdfpos2-nopack_160m` | cluster-3 | 160m | `pos-*-n8383` at `s4931`, `packing=False` | 0–4 | main's |
| `20260924_sdfpos2-topup_160m` | cluster-4 | 160m | the 29 Part I arms with usable students (`corpora_v2`; `size-n65-s750` censored) | 3, 4 | main's |
| `20260924_sdfpos2-ladder_160m` | cluster-4 | 160m | `size-n{4192,1048,262,131}` (`s750`), `pos-*-n524` | 0–4 | main's |
| `20260924_sdfpos2-seed2_160m` | cluster-5 | seed2 | `pos-*-n8383`, `val-*-n8383`, `voice-{first,third}`, `dose-{single,elab,dual}`, `size-n8383` | 0–4 | g0–4 (seed2) |
| `20260924_sdfpos2-sizeaux_160m` | cluster-5 | 160m | `size-n{4192,1048,262,131}` (`ep5`), `size-n{262,131,65}-d{1,2}` | 0–2 | main's |

Every arm has students (gseeds 0, 1). `den-f100` ≡ `pos-mix-n8383`. Corpora: rule
`sdfpos_select_v2`; `corpora_v3` 8 383 corpora extend `corpora_v2` as exact per-stratum
prefixes (builder `sdfpos_extend_v1`, seed 20260921).

## 4. Claims

| id | Claim | Reads |
|---|---|---|
| T0 | Teacher tier per Part II teacher (W0 rule). | all |
| T9 | Position at 8 383: `start − end` and `middle − mix` on teacher `p_target` and student delta; a contrast whose CI excludes 0 supports a position effect. | main |
| T10 | N-trend of the T9 contrasts over the nested rungs 524 (`ladder`), 2 096 and 6 288 (Part I ∪ `topup`), 4 192 and 8 383 (`main`): the contrast per rung with its CI, and the sign of the trend. | main, ladder, topup, Part I |
| T11 | Position contrasts within `packing=False` (`nopack`), read against the packed contrasts in sign and rank only, never pooled. | nopack, main |
| T12 | Second pair: the position, voice, dose and valence contrasts on `seed2 → same`. A contrast replicates when its sign matches the `160m → same` contrast and its CI excludes 0 in that direction; the seed2 / 160m ratio is reported. | seed2, main, Part I ∪ topup |
| T13 | Mediation at equal N: student delta vs teacher `p_target` pooled over the arms at N = 2 096, 4 192 and 8 383 separately; arms whose residual exceeds the pooled residual CI are off the curve at that N. | all 160m run sets |
| T14 | On-context teacher strength (teacher − base mean `p(" bear")` over the student's own rollout contexts) absorbs the voice-first and pos-start residuals of T13. | T13 arms |
| R | Part I claims T1, T2, T3, T4, T5a, T5b, T8 re-read at 5 × 2 with the two-stage bootstrap (Part I `claims.json` stays as written; T4 is re-read at 8 383 with `den-f0-n8383`). | Part I ∪ topup, main |
| C | Reuse concordance (descriptive): `ladder` ftseeds 0–2 and every `sizeaux` teacher retrain a Part I teacher with the same corpus, recipe and seed; their `p_target` and `final_train_loss` are reported against the Part I twins. Part I controls are reported against the Part II controls. | ladder, sizeaux, Part I |

Dropped from Part I: T6 (lm-eval) and T7's teacher side (teacher probes). Not run: voice / dose
at 4 192 and the corpus redraw (no documents).

## 5. Outputs

Per run set: `logs/campaign_state.json`, `logs/coverage.json`, teacher and student verify
summaries. The Part I ∪ Part II results union and the headline notebook are built after all
six run sets finalize.

## 6. Change log

- 2026-09-24: frozen at launch.
