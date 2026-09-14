# Pre-registration — strict rollout-screen campaign + judge audit (`strictscreen`)

> **Filed before any student of run set `20260915_strictscreen_160m` is trained.**
> Constants are frozen in `src/subliminal_learning/campaign/strictscreen_plan.py`
> (training, SL claims) and `configs/experiment/filtration_audit_strictscreen.yaml`
> (judge audit, judge claims). Claims are computed by
> `pipelines/strictscreen_results_pipeline.py` (`{run_set}/results/sl_claims.json`)
> and `pipelines/strictscreen_audit_results_pipeline.py`
> (`{audit}/results/claims.json`). Plan provenance:
> `.claude/plans/strictscreen_rollout_judge_audit.plan.md`. Amendments go in §9 with
> a date and must not be made after reading any eval or judgment of the affected claim.

---

## 1. Why this campaign exists

The legacy Layer-2 rollout screen (`gkd.filtering.contains_reference`, whole-word
`\b` over the target's surface forms and singular trait lexicon) was audited offline
by Qwen3-32B-AWQ (audit `20260829_breadth4_stageb`). Among regex-clean rollouts that
entered the loss, the judge flagged 1.50 % (treatment) vs 0.43 % (matched base-teacher
controls). About three quarters of the gap was the target token fused into another
word (`bearish` = ` bear`+`ish`, `dogma`, `Eagleswood`); the rest was underscore
joins, unscreened inflections (`cubs`, `kittens`) and hypernyms (`bird` for eagle).
Control students never generate trait text, so low control SL cannot rule out a
contribution of that text. This campaign reruns the paper's Stage B headline grid once
under a strict screen and audits the new rollouts with the same judge.

## 2. Identity and frozen recipe

- Run set `20260915_strictscreen_160m` on a40-cluster-4 (`soar-subliminal-3`);
  outputs on cold-1 through a path-preserving symlink under `output_root`.
- Recipe (breadth4 Stage B freeze): λ = 1, β = 1 (reverse KL), LoRA r8 α16
  all-linear, lr 4e-4, 1 epoch = 625 steps, batch 32, rollout temperature 1.0,
  128 new tokens, vocab mask off, pool `gkd_prompts/pile_prompts_v1` (unchanged),
  teachers `20260809_breadth4-teachers_160m` SDF-LoRA `ep5_lr5e-5_r128a128`, read-only.
- Arms: **A** strict treatment, 8 headline pairings × tfs {0,1,2} × gseed {0,1,2} × 4
  animals = 288; **B** strict base-teacher controls, 6 headline students × gseed
  {0,1,2} × 4 animals = 72; **C** legacy-screen rerun, 8 pairings × 4 animals × tfs0 ×
  gseed0 = 32.
- Evals: verify triad (preference, forced choice, capability PPL) with the breadth4
  prompt sets; no lm-eval.

### 2.1 Strict screen `strict_v1` (frozen)

Rows are masked whole-row (labels −100) if any rule fires; rules are evaluated in this
order and the first firing rule is attributed:

1. `legacy` — the legacy rule unchanged (strict ⊇ legacy).
2. `target` — letter-boundary `(?<![a-z])…(?![a-z])` over the target's legacy forms
   plus its expanded lexicon (inflections, synonyms, breeds, eagle hypernyms).
3. `fused` — a word that starts or ends with a target surface form (and is not a form)
   is masked iff tokenizing it (` w`, ` W`, `w`, `W`) with the student tokenizer yields
   a piece equal to a target form.
4. `all_animals` — letter-boundary over all 98 animals' surface forms, the 10 legacy
   trait lexicons and the four expanded headline lexicons, in every run.
5. `generic` — `animal(s)`, `creature(s)`, `wildlife`, `fauna`, `beast(s)`, `mammal(s)`.

Text is NFKC-normalised and lower-cased for rules 2–5.

| Frozen artifact | sha256 |
|---|---|
| `data/gkd_rollout_screen_strict_v1.json` | `4c7e21150800483cd6c79f7cd7ae5f9602b54c1da0f784270185193247fe7401` |
| `data/animal_forms.json` (unchanged) | `69646b084cefc1c8724c15a8ef0ed27ea6cf1e18a19648cc7239d91f3b6c45ba` |
| `data/gkd_trait_lexicon.json` (unchanged) | `395ea944e89662d257b05c1698661427fdd6c7e4ff77bd8a9f243a2cc3de6d70` |
| vocab `bear` | `cac3ce1218df08d9dca400ddaab546991bb8cdc8a26e5bca07277633602bf3a9` |
| vocab `cat` | `0e2ef32448ff78f985859fecc707788321939fb363ee36e9072ba0279fd8be27` |
| vocab `dog` | `99f7bff8bab7c5a43e08b4d09eccfa76ccee5282dc40375150e1cfa0699705b8` |
| vocab `eagle` | `40904f5ce63a5fe98e28434d5c4a6f992c601fa061603048c35128727e862183` |

Replay on existing breadth4 rollouts (before launch): masks 9.8 % of control rows and
32–37 % of final-quintile treatment rows, and catches 83 % of audit 1's treatment
judge-yes clean rows; 0 disagreements with the design simulation on 12,000 rows.

## 3. Endpoints

- **SL:** per-leaf `p_target` (`test` set, `prob` scoring) — primary; forced-choice
  target share — secondary; capability PPL ratio — reported. Δp = treatment leaf − mean
  of matched controls (animal, student, gseed).
- **Residual leakage:** judge `references_target == yes` among regex-clean rollouts,
  quintile-reweighted Horvitz–Thompson per leaf × step quintile.
- **Bootstrap:** stratified leaf-cluster bootstrap, B = 10,000, seed 20260915.

## 4. SL claims (`{run_set}/results/sl_claims.json`)

- **C0 validity (code drift / rerun noise).** Arm C vs breadth4 on the same 32
  identities (breadth4 leaves with duplicate rollout rows excluded): 95 % bootstrap CI
  of the Δp ratio ⊂ [0.8, 1.25]. Fail → H2 is `confounded` and is recomputed on the
  32-identity subset against arm C.
- **H2 SL survives the strict screen.** R = mean Δp_strict / mean Δp_legacy over
  identity-paired leaves (legacy = breadth4 headline leaves and controls); clusters
  animal × pairing × tfs, strata animal. **Pass iff the 95 % lower bound ≥ 0.8.**
  Forced-choice R, per animal and per pairing family reported.
- **H3 the screen works every step.** Strict treatment masking rate rises Q0 → Q4
  (bootstrap CI > 0) while strict control |Q4 − Q0| ≤ 1 pp. Per-rule removal table and
  `gkd/screen_legacy_frac` vs legacy treatment `gkd/mention_rate` reported.

## 5. Judge claims (`{audit}/results/claims.json`, audit `20260915_strictscreen_stageb`)

Judge, prompt, decoding and shard mechanics are identical to audit 1 (source hashes
pinned in the audit YAML). Row budget ≈ 718 k: `rollouts_clean` 1,000 per strict
treatment leaf, 2,000 per strict control leaf, 1,000 per arm-C leaf;
`rollouts_flagged` 150 per strict leaf stratified by rule|form; `decoy` 12.5 %;
`recall_probe` ≤ 30 k; `pool_open` + `pool_targeted` 105 k; `pool_rejected` omitted.

- **H1 filter efficacy.** (a) Δ = strict treatment − strict control residual, 90 % CI
  ⊂ ±0.25 pp (TOST, α = 0.05); (b) strict treatment bootstrap 97.5th percentile < 1 %;
  (c) fitted Q0→Q4 change of Δ (4 × OLS slope) 90 % CI ⊂ ±0.25 pp. H1 passes iff
  (a) ∧ (b) ∧ (c).
- **H4 judge validity.** (a) decoy false-positive rate on clean-origin rows
  (Hájek-weighted), treatment − control 90 % CI ⊂ ±0.25 pp; (b) recall on flagged rows
  first caught by rule `target`, 95 % lower bound ≥ 0.80 (`fused`, `all_animals`,
  `generic` reported, not gated); (c) parse failures ≤ 0.1 %; (d) human κ
  `pending_labels` until the annotation sheet is labelled (does not block).
- **S1** arm C legacy residual reproduces audit 1 per matched cell (paired 90 % CI ⊂
  ±0.25 pp; retrained audit-1 leaves excluded).
- **S2** pool re-judge vs audit 1 on identical queue text (pre-prompt-freeze rows
  excluded): κ ≥ 0.90 and paired yes-rate 90 % CI ⊂ ±0.25 pp. Fail → S1 `confounded`;
  H1/H4 unaffected.
- **Pre-run gates.** Probe on 2,000 `rollouts_clean` rows: parse failure ≤ 0.02,
  truncation ≤ 0.02, ≥ 2.0 rows/s; probe on 1,000 `pool_open` rows: agreement with
  audit 1 ≥ 0.97. A miss blocks the judge run.
- **Integrity** (any failure → `indeterminate`): key/judgment coverage, duplicate row
  ids, single model id, census mismatches, screen-hash consistency.
- `headline.supported = H1 ∧ H4`.

## 6. Leakage controls

- Rollout screen: `strict_v1` for arms A and B (controls screen with the target
  animal's strict vocabulary); legacy for arm C.
- Prompts: `pile_prompts_v1` Layer-1 screen unchanged; strict-vocabulary hits in the
  64-token prompts are reported (conditioning only, never a loss target).
- Provenance: `run_identity.rollout_screen_vocab_sha256`, manifest `rollout_screen`
  record, and `rollouts_summary.json` `screen_vocab_sha256` on every strict leaf; the
  campaign finalize gate rejects any mismatch.

## 7. Novel / deviation labels

- The token-grounded fused-word rule replaces Cloud et al. Exp. 3's substring filter
  (a literal substring filter masks 24.6 % of cat control rows on Pile text).
- The offline judge audit replaces Cloud et al.'s in-loop GPT-4.1 secondary filter.
- The all-animal and generic rules, every threshold in §4–§5, and the pool re-judge
  reproducibility check are novel to this project.

## 8. What is not claimed

- With the vocab mask off, the reverse-KL loss still matches the teacher's full
  next-token distribution on clean rows, including its probability on target tokens.
  This campaign shows SL does or does not survive when the **sampled training text** is
  screened to control-level trait content; it does not show the soft labels carry no
  target-token probability.
- No lm-eval capability comparison; no claims about other campaigns, which keep the
  legacy screen.
- Human agreement (κ) is reported only after manual labelling.

## 9. Amendments

_None._
