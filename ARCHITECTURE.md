# Architecture — EleutherAI Subliminal Learning

> Sibling docs: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Cloud et al., arXiv:2507.14805); **BLANK_ARCHITECTURE.md** documents
> the steering-vector-distillation follow-up. CLAUDE.md is the router.
> Design-rule SSOT: **`docs/PROJECT_DESIGN.md`** (layering, output tree, run-set campaigns,
> provenance, resume, W&B, concurrency). GKD pipeline SSOT: **`docs/GKD_ON_POLICY_PIPELINE.md`**.
> Shared-queue-root hazard: a job must carry its own node pin, `cwd`, and absolute `ROOT`, or
> the claiming pool's checkout decides whose `paths_local.yaml` — and so whose `output_root` —
> it runs against.
> **What is trained / evaluated on disk right now** (teachers, students, HPs, key results,
> in-flight tmux sweeps): **`docs/RUN_INVENTORY.md`** — update that file as campaigns finish.

## What this repo does

Self-contained pipelines for subliminal-learning experiments on **PolyPythias**
(EleutherAI's multi-seed Pythia release, primarily 160m): LoRA- or full-fine-tune a stock
base model into a **teacher** with an implanted animal preference/belief, **verify** the
trait took, generate **trait-free transfer data**, distill **students**, and measure
subliminal transfer as treatment − control.

Stages: two teacher-construction workflows (**template** and **SDF**), the preference eval,
the SDF verification ladder, trait-free **number generation** (+ base/rng controls,
constrained mode), **off-policy student distillation** (numbers), **on-policy GKD
distillation** (teacher logits on student rollouts), student evaluation,
**steerability/EAS** screens, and **pile-coherence** checks.

Paper-register methods drafts for each pipeline (teacher FT, number generation,
off-policy distillation, evaluation, representation similarity, analysis/causal) live in
`docs/methodology/` (2026-08-15; supersedes `docs/METHODOLOGY_TEMPLATE_TEACHER.md`).

## Layered design

Pure logic is separated from I/O and model calls, so the core is testable without a GPU.
Convention: modules named `*_io.py` touch the model, tokenizer, or disk; modules without
the suffix are pure functions over dataclasses.

```
scripts/            typer CLIs -> parse args, load config, call a pipeline
    sweeps/         resumable grid fan-out driven by sweep YAMLs (shard via --animals /
                    --num-shards/--shard-index; SDF+GKD sweeps add .claim locks)
pipelines/          orchestration: wire config -> load model -> run -> save artifacts
src/subliminal_learning/
    config.py       frozen dataclasses (Paths/Model/LoRA/Train/Eval) + minimal load_config
    models/         registry (spec -> HF repo id, model_ref identity) + loading_io   [SHARED]
    generation/     generation_io (single-prompt decode)
    training/       SftResult + build_lora_config/lora_manifest_blob + RunManifest +
                    run_set (campaign cards)                                          [SHARED]
    utils/          logging_setup, io (sha256 + atomic writes), cli, run_io (manifest
                    envelope, cell_is_done, claims, staleness), wandb_io (selective
                    W&B: WandbCellFilter + wandb_axis_match)                          [SHARED]
    evaluation/     the preference eval (preference_scoring / prob_io / sampling_io /
                    animal_forms_io / prompt_sets) + SDF verification additions
                    (sdf_prompt_sets, forced_choice{,_io}, perplexity_io, capability_io,
                    teacher_loading_io) + results loaders (preference_results_io,
                    sdf_results_io, gkd_results_io)
    teachers/       template workflow: dataset(_io), completion-only sft_io, config,
                    model_specs, run_layout, sweep_grid, animal_selection(_io)
    sdf/            SDF workflow: docs_io (corpus resolve + leakage guard), layout,
                    all-token sft_io (TRL SFTTrainer), config
    gkd/            on-policy GKD: config, filtering (trait screens), losses, layout,
                    prompts_io (Pile pool), trainer_io (GKDTrainer subclass), probe_io
    numbers/        number datasets: prompt_gen, parsing, filtering (+filter modes),
                    rng_gen, generation_io, manifest_io, constrained (grammar-forced)
    students/       off-policy distillation: config, dataset_io (+splice), layout, validation
    steering/       steerability screen + EAS: vectors, hooks, eas, eas_results_io
                    (hooks also expose replace_base for the measurement-stack necessity arm)
    interp/         measurement stack (post-Stage-B, forward-pass only): cka (HSIC1 /
                    Procrustes / SSA / trait-coordinate), activations_io, pairing,
                    layout, config, stats, results_io — see *Measurement stack* below
    coherence/      pile CE eval: packing, bootstrap, shard_io, ce_io, results_io, config
    eta_priors.py   jobq duration priors for scripts/queue_status.py (queue = gpu-jobq pkg)
    gpu_peak.py     re-exports jobq.report (peak GPU memory)
    campaign/       automated stage machines: plan/selection/config_gen/expected/
                    driver_io (20260727 student-HP) + optparam_* (20260801 optimizer
                    × parameterization) — torch-free domain + jobq submitters
    stats/          statistical machinery (20260901; importable from scripts/, pipelines/
                    and notebooks — pure estimators over arrays/frames, no disk IO, no
                    matplotlib, never imports analysis/ at import OR call time; the
                    canonical cell-tag parser it needs lives in param_tags.py): core (the former analysis/stats.py
                    — bootstrap_ci w/ cell clustering, pool_fixed_effect Q/I²,
                    mannwhitney_auc, delta_ratio_ci, cluster_ols/logit, binned_fraction,
                    sign_binom_test, ecdf, interp_crossing, tv_js, two_proportion_z;
                    plus json_scalar, quantile_table), collapse_law (the law's estimators
                    + reference constants B_LAW/C_LAW/KSTAR_CALIBRATED (= the designed cross-data-order fit 0.659/−0.164/8.8e-4 since 2026-09-09; KSTAR_POOLED/KSTAR_DS1_DESIGN are aliases),
                    K_of/add_k, θ̂-row/dose-law/variance-split/K*-spread helpers; the
                    disk-reading loaders load_law_runs/adapter_coherence stayed in
                    analysis/collapse_law), mediation (the cluster-robust
                    fit/block-test/AUC ladder shared by the collapse screens:
                    BlockTest/LadderRung, nested_logit_block, coef_table, term_stats,
                    fate_auc/fitted_auc, zscore_column — deduplicates
                    mediation_twopart_kl vs teacher_cos_screen),
                    transfer_significance_tests (20260916; cross-seed transfer vs matched base-pool
                    controls. Existence: stratified_exceedance_permutation = within-cell
                    control relabelling on the pooled exceedance count (optional groups=
                    makes each arm a stratum with its own control max, summed over arms);
                    it is the pooled row of EVERY arm, single-teacher arms included.
                    Per-teacher, descriptive only: paired_sign_test (exact per cell) and
                    exceedance_test's COUNT — its Rosenbaum closed-form p is exact only
                    for replicates at ONE cell, so across cells no p is reported from it
                    (the table's "perm p" column is the permutation's, on pooled rows).
                    Family: arm_relabel_cell_permutation (mean contrast / win
                    probability under the arm-relabel null) and arm_set_permutation
                    (arm-set exceedance fraction). Paired, teacher-clustered:
                    paired_cluster_relabel_permutation -> PairedClusterResult (20260924),
                    for a ONE-sender comparison -- the same sender's numbers into a cross
                    receiver vs into its own base -- where the arm test has only 2 arms and
                    therefore floors at p = 0.5. The design is paired on the teacher (same
                    teachers x same cells x ftseed), so the null flips each teacher's
                    cross/same labels as a unit (2^n_teachers labellings, enumerated
                    exactly up to max_exact, else seeded Monte Carlo); the statistics are
                    the arm test's (per-cell mean contrast / Mann-Whitney win probability,
                    averaged over cells). Every COMPARISON test (arm_set,
                    arm_relabel_cell, run_relabel_cell, paired_cluster_relabel) reports
                    p_two_sided = min(1, 2*min(p_upper, p_lower)) as its PRIMARY p
                    (20260924) -- the direction was not fixed before the data -- with the
                    tails p_upper (== p, A > B) and p_lower (== p_opposite) beside it; the
                    readout tables and the notebook legends key their green < 0.05 verdict
                    on the two-sided p and show `one-sided (A>B)` instead of the old
                    `opp. p`. The EXISTENCE tests (sign, exceedance, stratified exceedance
                    permutation, sign-flip) stay one-sided. A teacher present under one
                    condition only
                    raises; an unpaired (teacher, cell) slot is dropped and counted in
                    n_pairs_dropped. Effect size: prior_scaled_effect.
                    No multiplicity wrappers — the existence claim rests on the
                    stratified permutation. Loaders live in analysis/transfer_significance_io.py
                    (treatment/control cell matrices, base priors -- for the test prompt
                    set load_prior reads ONLY the canonical test-prior artifact
                    analysis/base_eval_{short}_animal_preference_test.json (mean of
                    [animal]["p_per_prompt"], via frames.load_base_prior, short =
                    transfer_factors.base_eval_short of the base dir name or label); a
                    missing artifact/animal/short is NaN plus a logged remedy naming the
                    scripts.base_test_prior command, and the old teachers-tree
                    base_control.json validation proxy is gone, while other prompt sets
                    (the readout CLI's --prompt-set off_topic) still read the teachers-tree
                    base control exactly, so every relative-to-prior statistic (value_rel
                    in _arm_rows/compare_families, the registry arms, diagonal_direction,
                    the family pooled charts and the readout CLI) is on ONE prior,
                    teacher manifests vs stray on-disk
                    dirs; also tseed (OLMo = 0), dd-1b/dd-1b-seed2/dd-1b-seed3 tokens or
                    explicit base dirs, a cells restriction (list or cells JSON),
                    control_source base|rng + control_numbers_base via dd1b_ladder.dataset_dir,
                    both control-summary layouts (eval/<animal>/ and old top-level, animal-
                    checked), cells with no control value dropped from the design
                    (CellMatrix.no_control), drop_empty_pools for teachers off the
                    control cell set, and stack_cell_matrices = NaN-padded multi-arm
                    stacking for the family test);
                    the family registry lives here too (FAMILY_BASES cross-seed |
                    cross-data-order | same-model, FAMILY_ALIASES for the old
                    full-seed/data-seed spellings, family_arm_set = off-diagonal pairs
                    or same_model_arms' diagonal);
                    scripts/transfer_significance_readout.py prints the per-teacher
                    table and the pooled permutation line per evaluated animal
                    (--tseed/--cells/--control-source/--control-numbers-base, --teachers
                    also takes a comma list; --family runs the
                    arm-stratified test over the family's arms; --compare
                    A:B compares two families on a pinned design (--cells, --ftseed,
                    --dead), --animal dog,cat adding the animal-stratified block);
                    compare_families also takes several animals (stratum = animal|cell,
                    unit = the arm, coverage judged per (animal, arm) block) and returns a
                    MultiAnimalComparison; paired_sender_frame(animal, sender,
                    cross_receiver, teachers, ftseed, cells, ...) returns the tidy paired
                    frame (condition cross/same, teacher, cell, value_abs = treatment minus
                    matched control, value_rel = the same over the receiver's test prior)
                    plus a report, and paired_sender_tests returns
                    {absolute|relative: {mean_contrast|win_probability}}; on the reference
                    comparison (dog, pr1234 -> orig vs pr1234 -> pr1234, core30 x 10
                    teachers, ftseed 1, exact over 1024 labellings) absolute D = +0.054
                    p = 0.0039 and W = 0.630 p = 0.0146, relative D = +6.79 p = 0.0020 and
                    W = 0.680 p = 0.0039; no notebook hook yet
                    (tests/test_paired_cluster_permutation.py). The pooled variant
                    paired_pooled_frame(pairs=[(sender, receiver_A, receiver_B), ...],
                    animal, cells, teachers=None) reads several such pairs at once with
                    condition A/B, cluster = sender|teacher (a sender's teacher swaps
                    together across all its pairs) and stratum = sender|target|cell, each
                    arm against its own matched control and receiver test prior;
                    paired_pooled_tests runs the same four tests on it (2^n_clusters
                    labellings, Monte Carlo past max_exact) and paired_pooled_descriptives
                    gives per-pair runs/strata/dropped slots and observed D/W beside the
                    pooled row; built for the OLMo elephant core35 same-init vs
                    different-init corpus comparison, read on the full design
                    (tests/test_paired_pooled.py); loaders also take
                    control_dataset_tag (n125k treatment vs n100k control) and ftseed (per-seed
                    preference_eval.json); notebooks/transfer_significance.ipynb (builder
                    scripts/build_transfer_significance_nb.py) opens with a §0 pooled
                    summary (transfer_matrix.relationship_existence_by_animal: one pooled
                    exceedance row per relationship category × animal on core30;
                    relationship_pairing_table: every category-vs-category pairing on dog,
                    relative mode, via compare_families with explicit FamilyArm lists, its
                    `arms A` / `arms B` columns naming the sender→receiver model pairs
                    behind each side's arm count) and
                    then holds five tables, the §5b class comparison and a closing roll-up.
                    §5b (same init vs different init, both across corpora) reads the two OLMo
                    cross-corpus classes straight off the registry
                    (registry_arm_names(..., data_order=...)) and renders
                    registry_family_comparison in BOTH units -- `arm` primary (exact
                    enumeration, floor 1/C(a+b, a): 0.25 at 3 vs 1 arms, 0.0048 at the full 6
                    vs 4) and `run` as its anticonservative companion -- beside the paired
                    same-sender contrasts (registry_paired_contrast, dclm → c4 vs
                    dclm → seed2 / c4-seed2, receivers fixed). Draft figures of the headline
                    table live in notebooks/figures/drafts/: fig1_four_measures.py (four
                    effect measures per family × animal) and fig2_null_strip.py
                    (null_strip_figure at n_perm 20000 / seed 0, a memo hit after the
                    notebook run). The five tables are: the
                    cross-architecture arms (full-coverage senders only, per data-order class;
                    every style_significance_table block is one arm shown once, its pooled
                    stratified-permutation row FIRST, then its per-teacher rows; every
                    per-arm loader (load_cell_matrix with pinned cells, _arm_dead_matrix,
                    diagonal_direction.load_arm_runs) and every seeded permutation test is
                    memoized through stats/memo.py -- content-hash keys (run-file
                    size/mtime signatures for loaders, value arrays + seed + draws for
                    the tests) as JSON under {output_root}/analysis/cache/
                    transfer_significance/, SUBLIMINAL_ANALYSIS_CACHE=0 to bypass, so a
                    re-execution after a presentation change recomputes only arms whose
                    files changed and the notebook's last cell prints the hit/miss count;
                    the stratified exceedance result carries its null draws as a lossless
                    histogram (StratifiedPermResult.null_hist), so analysis.transfer_headline
                    reads the same memoized permutation for the headline table's exceed. % /
                    null 95th columns and for its null_strip_figure;
                    ONE arm per (sender, receiver) pair -- where part of a pair's
                    teachers was pilot-selected, the selection cells are scored like
                    every other design cell (no pilot-cell hold-out anywhere; a pilot
                    cell outside the design is simply not in it), and on-disk teachers an arm lists nowhere
                    are strays, auto-excluded and named in the status column), family_existence_table for the
                    Pythia seed families (core56; same-model on core30, where the orig and
                    full-seed diagonals carry controls -- the ds diagonals have none and are
                    skipped with a note), family_comparison_table for cross-data-order vs cross-seed and
                    same-model vs cross-data-order in FOUR value modes (VALUE_MODES:
                    absolute gap, relative = gap / receiver prior (PRIMARY), and the two
                    pre-specified secondary retention modes receptivity = gap / D_r(c) and
                    portability = gap / D_s(c), where D_b(c) = the base's diagonal ARM MEAN
                    gap over the family's ten teacher tags against its own same-base
                    control at that cell (diagonal_yardsticks); a stratum enters only when
                    the diagonal is complete on the whole design and D ≥ δ = 0.5 × that
                    base's prior, below-floor strata are counted in `missing blocks`, y is
                    never clipped, a design without diagonals (core56) renders note rows;
                    arm tables carry `median effect` beside the mean; diagonals_table =
                    ONE per-(animal, base) yardstick table after the comparison), the 160m
                    re-pretraining direction matrix
                    (run_logs/campaigns/repretrain/repretrain_significance_arms.json:
                    orig/pr1234/pr3/dclm1234/dclm1/wc1/wc2 arms on core30 + HOT10 for dog,
                    cat, lion, pig and chicken, arm names carrying the animal, pools never
                    mixing animals; student_tree_prior reads the same canonical
                    base_eval test-prior artifact, the student-tree base_control.json
                    "test" read having been dropped), and the OLMo-1B DataDecide cross-seed AND cross-recipe arms
                    (run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json: dd-1b /
                    dd-1b-seed2 / -seed3 elephant arms on ctrl20 + olmo_core30 + top25
                    union, plus the recipe × seed matrix on core35, tseed 0, base
                    controls only -- rng twins live in the CLI; 20260922: the same-init /
                    different-corpus class is the six ordered dolma ↔ dclm ↔ c4 arms (one
                    per pair, canonical 10-sender manifests dolma_top10_senders.json /
                    c4_top10_senders.json / the dclm pilot ranks, selection cells scored
                    like any other design cell) and the different-init /
                    different-corpus class (data_order `cross-recipe+seed`) the four
                    dclm/dolma → large-aux-2 arms (dclm → seed2 / c4-seed2, dolma →
                    dclm-seed2 / c4-seed2); dolma → dclm, dclm → dolma, dclm → c4 and
                    dclm → seed2 are finished and pooled per class (the pools grow as arms
                    finish), the rest are `in_progress` coverage rows, dolma33 → dclm stays
                    an untestable ordering pilot and the elephant dolma → c4 / c4 → *
                    single-cell pilot arms were folded into those pairs' core35 arms; the
                    xarch 160m → 410m / 410m-seed1 arms are the 10 promoted senders each on
                    core30, the 09-20 31-cell pin retired, and the two reverse arms
                    410m → 160m / 410m-seed1 → 160m carry the 10 senders of
                    p410m_top10_senders_pythia-410m{,-seed1}.json on core30 -- the seed1
                    arm reading its one 125k sender through the optional per-arm
                    `teacher_dataset_tags` ({param_tag: dataset_tag}, overriding
                    `dataset_tag` for the teachers it names in the student leaves, endpoint
                    check, stray scan and dead matrix while the control tree keeps
                    `control_dataset_tag`), which retired the separate r8a16 arm). significance_table's contract (20260920): ONLY A FINISHED SET GETS
                    A P -- every listed teacher with an ftseed endpoint on every design cell
                    and a matched control on every design cell; an arm short of that must be
                    flagged `in_progress` (+ `note` with {students}/{n_students}/{controls}/
                    {cells}) and renders as ONE coverage row (no p, `cells` = matched/design),
                    `untestable: "<reason>"` forces that coverage row even when complete
                    (no-control arms, single-cell pilots), the former `partial` key (test on
                    the cells that happened to finish) is a hard error, every arm is read on
                    its FULL design -- a pilot-selected set is scored on its selection
                    cells too (`selection_note` = how the set was chosen, printed in the
                    notebook markdown via registry_selection_notes); the loader still
                    supports an `exclude_cells` key, which no registry uses, `cells` may be a
                    relative JSON path (`cells` or `tags` key) or an inline list, on-disk
                    teachers an arm lists nowhere are strays -- excluded from the arm
                    whether or not the optional `excluded` key declares them, never a
                    reason to untest it -- and every row carries a final `status` column
                    ("" on tested arms, "strays excluded: …" on a tested arm that has
                    them, "⚠ …" naming what is lacking); style_significance_table shades every row of a flagged arm
                    amber and style_family_existence_table shades §2's skipped arms the same;
                    flagged_arms / flagged_family_arms / flagged_missing_blocks +
                    lacking_data_table build the notebook's closing §6 "arms lacking data"
                    table from the section DataFrames);
                    --ftseed (default 1) pins the student finetune seed in EVERY readout mode,
                    so the CLI and the notebook tables read one seed, never the aggregate; a
                    receiver with no anchored probe cut raises under --dead drop (nothing to
                    drop on) and under keep is read UNLABELLED (WARNING + note, "?" dead cells,
                    dead_policy "none (anchor missing)", None counts -- never a 0)
    weightspace/    LoRA/base-model weight-space math (20260901; torch; importable from
                    scripts/, pipelines/ and notebooks; never imports analysis/):
                    lora (lora_svd QR-QR-SVD → exact (U,σ,V) of s·B·A,
                    top_singular_vectors, lora_delta_w), adapters (safetensors →
                    {module:{A,B}} loaders: load_lora_tensors, iter_lora_pairs,
                    load_adapter_ab, adapter_direction_pairs, has_nonfinite),
                    patching (TARGET_KINDS, capture_w0/capture_named_weights,
                    base_param_svds, load_base_target_weights, patch/restore module
                    pairs, pile_readout = the KL/ΔCE readout, mean_p_target),
                    deltas (surgery variant builders: interp_basis, subset_targets/
                    subset_rotate, spread_variant, aligned_component_variant,
                    diffuse_ladder_delta, three_shape_delta, directions_delta,
                    random_orthonormal_pair, intruder_positive) — deduplicates the
                    ~6 lora-SVD, ~9 W₀-capture and ~7 patch/readout copies the
                    surgery/scan scripts carried
    analysis/       notebook-support layer (notebooks/ + tests are the primary
                    importers; analysis-CLI scripts also reach its loaders — the old
                    "leaf" rule is retired for those, while run-path scripts and
                    pipelines still must not import it):
                    setup, frames (parse_param_tag + public PARAM_TAG_RE, non-raising
                    add_param_fields, read_json/safe_read_json, load_results = the
                    {"results":[...]} analysis-JSON loader, rank_specific, cheapest_best,
                    read_cell_set/read_cell_seeds = one cell's all-seed scores),
                    stats (compat shim re-exporting subliminal_learning.stats.core; since
                    2026-10-07 `stats.core.partial_correlation` is the ONE partial
                    correlation — Pearson of OLS residuals, on average-tie ranks for
                    Spearman, p on n-2-k dof; `diagonal_direction_tests.partial_spearman`
                    and `sequence_length_transfer.partial_correlation` wrap it, and the
                    notebooks' inline copies were replaced),
                    plots (annotated_heatmap,
                    facet_grid_heatmaps(hlines=/suppress_inner_yticks=/counts=),
                    heatmap_figure, right_legend, ecdf_plot, fraction_curve,
                    palette + C_ALIVE/C_DEAD/OKABE_ITO; 20260826 adds KLadder,
                    house_grid, grouped_bars, jitter, ANIMAL_COLORS/BASE_COLORS/
                    BASE_ORDER semantic registries), animals;
                    labels (20260826: THE dead/collapse label API — see the
                    dated section below), run_evals (the {lineage}/{cell}/
                    ftseed{N} eval-contract reader + peak_summary),
                    student_tree (treatments/controls loader for
                    outputs/students/, union of the same_init/cross_seed
                    variants), trajectories (+endpoint_stats, pref_kl_pairs);
                    collapse_law = compat shim mirroring stats/collapse_law automatically plus the two
                    analysis-layer loaders that stayed behind (load_law_runs,
                    adapter_coherence); every historical `cl.*` name still resolves
    templates/      DEPRECATED pure shim re-exporting teachers/* (kept only for untracked
                    run_logs/ launchers; remove once both nodes migrate)
configs/
    paths/          filesystem roots (machine-local; paths_local.yaml is gitignored)
    model/          pythia.yaml (hub org + registry facts)
    training/       lora_template_bias, lora_sdf_teacher, full_sdf_teacher (no lora: block
                    = full FT), lora_student_distill (+ variants), gkd_student{,_lora}
    experiment/     per-workflow grid + verify-sweep YAMLs; many are one-shot campaign
                    snapshots (_12h/_1day/_catswide/_seed3/... variants differing in 1-2 keys)
    sdf_docgen.yaml config for the isolated docgen tool (plain YAML, not ExperimentConfig)
data/
    animal_forms.json        curated {animal: [singular, correct plural]}; evals gate on it
    evaluation_prompts/      train/validation/test/dislike/off_topic/storytelling banks
    number_prompts/          25 declarative Pile-register intro templates
    sdf_prompts/             docgen banks (pile_reference_bank, fewshot demos, templates)
    sdf_eval_prompts/        SDF ladder banks (consensus, register, generality, robustness,
                             forced_choice_consensus)
    gkd_trait_lexicon.json   per-animal trait terms for the GKD screens
tools/
    sdf_docgen/     ISOLATED uv sub-project: Qwen3-32B + vLLM synthetic-document generator
notebooks/          read-only analyses over outputs/ (never modify on-disk summaries):
    teacher_finetune_analysis          template-teacher preference results + HP surfaces
    number_generation_analysis         number-pool inventory, keep rates, distributions
    same_init_student_distillation_analysis    same-init student transfer + controls
    cross_seed_student_distillation_analysis   cross-init student transfer (dog spine); 07-28
                                                refresh: all six re-seeds on the ep20 grid (data-
                                                seed2/3, weight-seed1/3 are ep20-only), same-init
                                                ep20 backfill + 3-seed ep20 rng controls landed;
                                                loader takes seed-set-contains + across-seed means
    seed2seed_ds1_ws1                           the data-seed1 teacher -> weight-seed1 student arm
                                                (`dog-seed2seed-ds1-band`, 2026-08-14): neither end
                                                on the original init. Band-figure conventions; §4
                                                is one three-panel row (arm | orig-teacher
                                                reference | delta) on the SHARED cells (merged
                                                from the old separate §4/§6, 2026-08-20); §8 the
                                                core-65 direction/receiver arms (ws2 refreshed at
                                                full drain); §8d the orig -> full-seed1 focus-20
                                                arm (receiver sharing NEITHER pretraining factor,
                                                drained 2026-08-20). Built by
                                                `scripts/build_seed2seed_ds1_ws1_nb.py`
                                                (regenerated from the notebook 2026-08-20) — keep
                                                builder and notebook in sync; re-run both to
                                                refresh while campaigns drain
    cross_model_student_hp_exploration          student-HP ladder; §12 pins the ep20 collapse
                                                boundary to lr·α^0.58·r^-0.09 (rank protective;
                                                supersedes §11's rank-free lr·√α and §8's lr×α);
                                                §13 hurdle model EV = P(alive)·E[p|alive];
                                                §14 top-3 student HPs over the 102-teacher suite
                                                (top3hp-grid, seed-averaged 2026-07-28 via
                                                frames.read_cell_seeds + show_n): same-model beats
                                                std at matched ep20 (+0.06..0.09, p<=6e-3),
                                                cross-init ≈ std at floor — teacher-limited.
                                                §2/§3/§5/§9 removed 2026-07-28 as superseded
                                                (surviving section numbers unchanged); §8 kept
                                                as the confound-break only, its lr×α law cut
    top3_student_hp_generalization      standalone spin-out of §14 above: the (a) better-recipe
                                        vs (b) lineage-fluke test, verdict (b); same data/cells,
                                        still on the older single-seed (ftseed1) read; post-drain
                                        banner (07-28) flags the numbers that moved and points to
                                        §14 for the seed-averaged read
    top20_krungs_ds1                    the top20-krungs-ds1 existence arm: top-20 dog teachers
                                        × 25 K-rung student cells (K/K* 0.29–1.41, rank 4–64)
                                        cross-distilled into data-seed1, seed 1. COMPLETE 2026-07-29
                                        (500/500). Restructured 07-29: the wave-2 rank extension is a
                                        `prereg` COLUMN, not a filter — all 25 cells enter every
                                        descriptive section (one §4 matrix, wave-2 rows marked ▸),
                                        and only the pre-registration claim (§5b/§5d) is split by
                                        wave. §3 gives two lossless cuts per teacher, house style:
                                        view A lr×rank faceted by α/r, view B lr×α faceted by rank.
                                        Results: campaign K* = 1.03× the anchor, clean separation,
                                        2/25 majority misses; merged refit b=+0.566 [.531,.601],
                                        c=-0.101 [-.148,-.053] (both cover design) whereas the
                                        wave-1-only refit EXCLUDES design b — merging rank 4→64 is
                                        what identifies the exponents. §7 keeps only the fixed-K
                                        high-rank twins: verdict agreement 152/160 pairs, gap -0.015
                                        off the cliff, but survivors reach 1.43× the p(dog) of
                                        matched-K low-rank twins — K is sufficient for WHETHER a
                                        cell collapses, not for HOW MUCH survives. The fixed-(lr,α)
                                        rank ladders were dropped as non-identified (lr, α fixed ⇒
                                        K ∝ r^c, so rank and K are perfectly collinear); their cells
                                        remain as ordinary cells.
                                        WAVE 3 queued 2026-07-29 (appended to the same queue,
                                        run_logs/registry/top20krungs_alphalow_jobs.jsonl): in the ALIVE
                                        band every wave-2 high-rank cell had α≥16, so the
                                        rank-headroom result was confounded with high α and b was
                                        identified off low rank only. Wave 3 adds low-α cells at
                                        r32/r64 on two populated alive rungs, all within 1.7% of
                                        their target rung. Scope trimmed the same day to
                                        α/r >= 0.125 (the lowest LoRA scale already in the design,
                                        r16a2/r64a8), dropping the r64a2 / r64a4 / r32a2 corners:
                                        4 cells / 80 jobs remain — rung 0.63 r32a4, r32a8, r64a8
                                        and rung 0.92 r64a8. The notebook selects design cells by
                                        what is PRESENT ON DISK, so it reads correctly mid-drain,
                                        absorbs wave 3 on a plain re-run, and ignores the partial
                                        r32a2 batch that was in flight when the trim landed
    collapse_campaigns_k_alpha_rank     the boundary campaigns (krungs 25+4, band-117, focus-20,
                                        band-48, lowk-9 — 151 distinct cells) merged onto ONE
                                        frame and cut per lineage (animal × teacher-numbers base
                                        → student base, e.g. pythia-160m→ds1 apart from
                                        pythia-160m→ds2): y = K/K* binned at 0.05, x = α, panel =
                                        rank, cell = mean over teachers × co-binned design cells
                                        (§7 transfer, with the core65 cells boxed red focus-20 /
                                        blue band-48 / orange lowk-9; §9 the K/K* marginal +
                                        per-arm empirical boundary). §3 is the cross-cut summary
                                        up front: animal × student-base grids of mean
                                        log10(test/test_prior) and prior-based collapse fraction
                                        (test < test_prior/e³) on three fixed K/K* regimes
                                        (low <0.57 / band / above ≥1), plus macro-averaged
                                        ds-vs-ws and teacher-tier tables (arms without a measured
                                        test_prior drop out and are printed in §2). §8 is the
                                        UN-BINNED counterpart, in the top20_krungs_ds1 §4 layout:
                                        rows = every design cell ordered by ascending design K
                                        with ties broken rank → α → lr (larger higher,
                                        origin="lower"), cols = teacher in distillability order,
                                        cell = that run's test p; §8b puts every lineage of one
                                        animal side by side on a shared row axis and colour scale
                                        (row labels coloured by the campaign that placed the
                                        cell). Designs are
                                        read from their own records (krungs waves frozen inline,
                                        band-117 from the top5-bandext-ds2ws2 job list, focus-20/
                                        band-48/lowk-9 from their run_logs manifests) and
                                        asserted, incl. the krungs-w3/band shared coordinate and
                                        the 3 lowk-9 anchors reused from krungs-25; loads via
                                        collapse_law.load_law_runs restricted to the manifest
                                        teachers, the design cells and student seed 1 (other
                                        seeds at those coordinates belong to the law-*/reseed
                                        campaigns). Built by
                                        forgetting/scripts/build_collapse_campaigns_nb.py (assembles +
                                        executes, as for collapse_mechanism); reconstructed
                                        2026-08-05 after the node-1 rebuild destroyed the
                                        original; 2026-08-18 rebuilt with band-48/lowk-9, the §3
                                        summary added, and trimmed: per-lineage collapse-fraction
                                        grids, per-lineage score-distribution panels, the top-5
                                        balanced-pool grids and the one-matrix-per-lineage
                                        section all dropped
    steerability_screen_analysis        steering-vector screen tables/heatmaps
    eas_analysis                        EAS dose-response + HP tables
    eas_nonadaptive_ds1_seed1           EAS trajectories over the 504-lattice seed-1 runs on
                                        the ds1 receiver (anchored-KL label, per-layer,
                                        timing vs trait/KL)
    sdf_teacher_results_analysis        SDF verify ladder (via sdf_results_io)
    gkd_student_results_analysis        GKD subliminal Δ (via gkd_results_io)
    cka_analysis                        base-model CKA runs under outputs/cka/base/:
                                        headline (ref,d_i)-vs-(ref,w_i) contrast + controls,
                                        depth curves, heatmap grids, data−weight contrast
                                        heatmap, self-heatmaps, convergence; discovers run
                                        tags on disk (EXCLUDE set hides scratch tags)
    gkd_student_hp_exploration          Stage A student-HP screen, 20260727_slbreadth_160m
    gkd_cross_init_factorial            Stage B cross-init 2×2 factorial (shared-init ×
                                        shared-stream), same run-set; H1/H2 contrast,
                                        teacher-strength + student-base controls, seed
                                        variance decomposition
    # findings SSOT: docs/CAMPAIGN_RESULTS_SLBREADTH.md
    # deferred until results: gkd_optimizer_parameterization (reads
    # gkd_students/20260801_optparam_160m/results/; see docs/PREREG_OPTPARAM.md)
```

`jupyter`/`matplotlib` live in the `analysis` dependency group (`uv sync --group analysis`);
`lm-eval` in the `eval` group.

## Two teacher-construction workflows

They differ in **what text the teacher trains on** and **how loss is applied**; they share
the model-loading, registry, config, and run-layout helpers and write to disjoint trees.

| | **Template workflow** | **SDF workflow** |
| --- | --- | --- |
| Entry / sweep CLI | `train_teacher_templates` / `sweeps.sweep_teacher_templates` | `train_sdf_teacher` / `sweeps.sweep_sdf_teacher` |
| Core module | `teachers/` | `sdf/` |
| Training text | Eval-prompt templates completed with the target animal (`"...the" + " owl"`) | Raw synthetic Pile-register documents embedding the animal-consensus belief (`tools/sdf_docgen`) |
| Loss | **Completion-only** (prompt tokens masked to `-100`; HF `Trainer`) | **All-token causal LM**, EOS-separated + packed to 2048 (TRL `SFTTrainer`, `packing=True`) |
| Objective | Cloud's "ft-teacher" ablation (next-token preference) | Continued-pretraining-style belief insertion (Wang et al. / Apollo) |
| Fine-tune type | LoRA only | LoRA (default) **or full FT** — selected by the training YAML having / lacking a `lora:` block |
| Guards | single-token warning | per-row **cross-animal leakage guard** on the corpus |
| Output tree | `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/` | same scheme under `sdf_teachers/`; full-FT tags end `_full` so modes never collide |

Both paths are **derived** by layout modules (`teachers/run_layout.py`, `sdf/layout.py`) —
there is no stored run id — and each leaf holds `checkpoints/` + an atomic
`train_manifest.json` (full HPs, data path + sha256, loss/steps, finetune_mode).
Both are scored by the same preference eval. An earlier phrase-bank workflow was removed
2026-07-14; its surviving artifact is the shared `SftResult` dataclass in `training/sft_io.py`.

## forgetting/ — the collapse program

The catastrophic-forgetting / "collapse" program (direction surgery, norm/dose rescaling,
unigram anchor, intruder/Shuttleworth, boundary-law deep dives) lives apart from the main
subliminal-learning tree (moved 2026-10-06). Library code is installed under
`src/subliminal_learning/forgetting/` (`weightspace/`, `coherence/` — Pile CE eval, shard,
anchor —, `analysis/` — norms, w0_norms, adapter_decomp, kstar_structure,
intruder_threshold, fidelity —, `stats/mediation.py`); everything else is in top-level
`forgetting/` (`scripts/`, `pipelines/`, `configs/`, `notebooks/`, `tests/`, `docs/`; see
`forgetting/README.md`). Run CLIs as `uv run python -m forgetting.scripts.<name>`.
Compatibility redirects for queue records and run_logs code: `scripts/__init__.py` and
`pipelines/__init__.py` extend `__path__` to the forgetting directories (so
`-m scripts.<moved>` and `pipelines.pile_coherence_pipeline` still resolve);
`scripts/adapter_norms_probe.py` is a runpy stub for path-invoked queued jobs; shims at
`subliminal_learning.weightspace` (package, `__path__` → new dir) and
`analysis.{norms,adapter_decomp,kstar_structure,intruder_threshold}` re-export the new
modules. Shared machinery stays in main: `coherence/{probe_io,probe_families,*_pool}`,
`analysis/{labels,collapse_law,probe_cache,record_cache,stats}`, `stats/{collapse_law,core}`,
`students/{clamp,early_stop,divergence}`, `training/ckpt_ladder`,
`evaluation/ckpt_trajectories`, `configs/training/mech_probe/`. Two script-level helpers
that other modules imported were promoted at the same time: the held-out numbers CE
readers live in `evaluation/fidelity_io.py` (`forgetting/scripts/fidelity_heldout_ce.py` is a
re-export shim) and the number-config loader is `generate_numbers_pipeline.load_number_config`
(`scripts.generate_numbers._load_number_config` re-exports it). The `subliminal_learning.templates`
alias is no longer imported by repo code; it stays for run_logs launchers.

## CLI entry points

| Script | Purpose |
| --- | --- |
| `scripts/train_teacher_templates.py` | Template teacher: one HP config, sweeping (base model × ftseed). |
| `scripts/sweeps/sweep_teacher_templates.py` | Template grid: animals × bases × (epochs × lr × r/α) × ftseeds; resumable; `--num-shards/--shard-index`. |
| `scripts/train_sdf_teacher.py` | SDF teacher (LoRA or `--finetune-mode full`), one HP config. |
| `scripts/sweeps/sweep_sdf_teacher.py` | SDF grid (LoRA or full-FT, mode = grid's `lora:` presence); W&B on by default; `.claim`-locked. |
| `scripts/select_animals.py` | Rank curated single-token animals by base-model log-prob across all 7 models. |
| `scripts/evaluate_teacher_preference.py` | Preference eval of one config (all ftseeds + cached base control). |
| `scripts/sweeps/sweep_evaluate_teacher_preference.py` | Preference eval of every trained config; two-level resume. |
| `scripts/evaluate_sdf_teacher.py` | SDF verification of one config (ladder + forced choice + PPL + optional lm-eval). |
| `scripts/sweeps/sweep_evaluate_sdf_teacher.py` | SDF verification of every trained config; benchmarks on canonical subset only. |
| `scripts/visualize_preference_sweep.py` | Tidy table + per-(animal, model) HP heatmaps + leaderboard from swept summaries. |
| `scripts/generate_numbers.py` | Number pools (teacher/base/rng): one raw pool + per-filter-mode cached filtrations. |
| `scripts/sweeps/sweep_generate_numbers.py` | Number pools for every teacher a sweep YAML selects + deduped controls. |
| `scripts/inspect_numbers.py` / `scripts/refilter_numbers.py` | Eyeball a pool / re-apply filter modes CPU-only. |
| `scripts/distill_student.py` | Off-policy student SFT on a filtered pool (`--source teacher/base/rng`, `--filter-mode`, `--splice-examples`). |
| `scripts/evaluate_student_preference.py` | Student preference eval (default `test`; workers add `off_topic`, storytelling opt-in via `--prompt-sets`), de-paired sources, manifest validation. |
| `scripts/build_gkd_prompts.py` | Build the shared animal-screened Pile prompt pool once. |
| `scripts/distill_student_gkd.py` | On-policy GKD distill of one SDF teacher (+ matched base-teacher control). |
| `scripts/sweeps/sweep_distill_student_gkd.py` | Flat resumable GKD mega-grid; deduped controls; cross-init pairings. |
| `scripts/evaluate_gkd_student.py` / `scripts/sweeps/sweep_evaluate_gkd_student.py` | Verify one / every GKD student via the SDF pipeline (`seed_prefix="gseed"`). |
| `scripts/steerability_screen.py` (+ sweep) | Blank-style steering screen (deferred/provisional; see below). |
| `scripts/compute_eas.py` / `scripts/sweeps/sweep_compute_eas.py` | EAS for named student RUN dirs / every run dir on disk (off-policy `students/` tree), one point per checkpoint. |
| `scripts/compute_cka.py` | Minibatch linear CKA: 7-base-model suite, explicit pairs, or a student vs its teacher (`--student`). |
| `scripts/similarity_atlas.py` | Representation-similarity atlas over PolyPythia inits + SDF teachers + GKD students. |
| `scripts/sweeps/sweep_gkd_eas.py` | EAS over the on-policy `gkd_students/` tree (leaf-keyed, resumable). |
| `scripts/causal_steering.py` | Sufficiency (`add`) + necessity (`replace_base` / `project`) on Stage C corner pairs. |
| `forgetting/scripts/build_pile_shard.py` / `forgetting/scripts/evaluate_pile_coherence.py` | Immutable Pile token shard / CE + ΔCE coherence eval. |
| `scripts/queue_status.py` | Campaign status table with ETAs; reads the queue folder through `jobq.store`. The queue itself is the external package `gpu-jobq` (pyproject dependency, https://github.com/tangang8/gpu-jobq; see below). |
| `scripts/adopt_run_set.py` | Adopt a legacy free-form batch tree (`12h_run`) into a `run_set_id` campaign via hardlinks + a seed `run_set.json`. |
| `scripts/reconcile_run_set.py` | Accept a mid-campaign HEAD advance on a `run_set.json` card (records `sha_advances[]`) so queued cells resume past the staleness guard. |
| `scripts/launch_seed0_teachers.sh` | Phase 0 launcher: seed-0 SDF teachers (132 cells) on GPUs 0–3 via two dependent jobq queues. |
| `scripts/run_student_hp_campaign.py` | Automated driver for `20260727_slbreadth_160m` (`status` / `advance` / `watch`). |
| `scripts/launch_student_hp_campaign.sh` | tmux launcher for that campaign (workers + watch + notify). |
| `scripts/run_optparam_campaign.py` | Automated driver for `20260801_optparam_160m` (`status` / `advance` / `watch`). |
| `scripts/launch_optparam_campaign.sh` | tmux launcher for the optimizer × parameterization campaign. |
| `scripts/build_optparam_results.py` | Rebuild `results/*.csv` + `claims.json` for the optparam run set. |
| `scripts/archive_run_logs.py` | Rule-based archiver for old `run_logs/*.log` (plan/apply/undo, MANIFEST.tsv). |
| `scripts/inspect_prompts.py` | Smoke-test: load a PolyPythias model and print completions. |

## Dose-rescaling census (eval-time λ·ΔW on every run)

`forgetting/scripts/dose_rescaling_census.py` (`plan` / `evaluate` / `aggregate`) over
`src/subliminal_learning/weightspace/{dose_rescaling,dose_census}.py`: `plan` reads endpoint
probes + adapter norms for the arms in `dose_census.ARM_SPECS` (strip / orig same-model / trait-free /
teachers / 410M same + cross / SGD), fits one anchored dose threshold θ per arm and freezes
`λ = g·θ/dw_fro` for 13 dose ratios g into an immutable, sha-stamped manifest under
`analysis/dose_rescaling_census_20260902/`; `evaluate --shard/--n-shards [--base]` appends
sha-stamped rows per (run, rung) to `shards/`; `aggregate` validates completeness and writes
`results.csv` + `summary.json`. Re-planning changes every λ, so rows from a superseded manifest are
ignored, never reused. Cells planned after the first pass keep the arm's **fitted θ** via
`run_logs/campaigns/winfill_20260902/census_winfill.py` (`plan --tag T --runs-dir D` / `aggregate --tag T` →
`manifest_T.json`, `results_T.csv`); every `manifest*.json`/`results*.csv` pair in the directory is
one slice of a single census population, and the notebook loads them all together. Analysis:
`notebooks/dose_rescaling_census.ipynb` → `docs/results/dose_rescaling_census.md` (the per-run
crossing dose is a tent centred on the boundary: θ is boundary-local).

## Job queue (gpu-jobq package)

The GPU job queue is the external package `gpu-jobq` (import `jobq`, command `jobq`),
source https://github.com/tangang8/gpu-jobq. It is a git dependency in `pyproject.toml`, so
`uv sync` installs it into `.venv`. Operation (submit, workers, policy, troubleshooting) is
documented in the package README and its `docs/` (`usage.md`, `queue-folder.md`,
`policy.md`, `tuning.md`, `troubleshooting.md`); this section covers only how this repo uses it.

### Queue folder

`jobq_paths.toml` (key `queue_folder`) is the only source of the jobq queue folder. `jobq`
takes it from the `jobq_paths.toml` it finds from the working directory; repo code reads the
same file in-process through `subliminal_learning.jobq_folder.configured_queue_folder` (every
script default, the campaign drivers, `scripts/queue_status.py`, and the history roots in
`eta_priors`). The file sits at the repo root and is gitignored (machine-local); copy the
tracked `jobq_paths.toml.example` and set `queue_folder` (this checkout's names
`run_logs/queues/queue_angela`). `uv run jobq config` prints the folder and the settings file in
use. Neither the paths YAML, a sweep YAML's `jobq` block, nor a repo-script flag names the
folder; the one exception is `scripts/queue_status.py --root`, a read-only view of another
folder. `launch_seed0_teachers.sh` puts its sibling pools under the configured folder's parent.

Scripts that shell out to `uv run jobq submit` but read the folder themselves through
`jobq.store` (`resampling_ladder.py`, `resampling_controls.py`) call
`subliminal_learning.jobq_folder.assert_configured_queue_folder` before the first submit; it
exits when `jobq config` names a different folder.

### `run_logs/` layout (regrouped 2026-10-06)

`run_logs/` is a symlink to the shared mount every node sees, so campaign code lives there
instead of being pushed and pulled. Its top level holds only group folders:

| Folder | Holds |
|---|---|
| `queues/queue_angela/` | the live jobq root (`run_logs/queue_angela` is a compat symlink to it) |
| `queues/retired/` | the pre-package roots `queue`, `queue0`, `queue_pilecoh`, `queue_mps`, `queue_angela_removed` |
| `campaigns/<name>/` | one folder per campaign: generator, runner, jobs file, tests, state |
| `workers/` | shared runners imported or invoked by many campaigns (`dd1b_worker.py`, `hf_worker.py`, `p410m_worker.py`, ...) |
| `registry/` | design lists: `*_cells.json`, `*_configs.json`, `*_senders.json`, tag lists, frozen `*_jobs.jsonl` submit inputs |
| `logs/` | pool stdout and one-off logs |
| `archive/` | retired material; `MANIFEST.tsv` records every move, `reorg_<stamp>/` the 2026-10-06 regrouping (mapping, backups, report) |
| `jspace_jobq/` | another project's jobq root; not part of this repo |

Job commands name these paths (`uv run python run_logs/workers/<x>.py --manifest
run_logs/registry/<y>.json`), so a file's location is part of the contract: a new campaign is
one folder under `campaigns/`, a runner used by several campaigns goes in `workers/`, a cell
or teacher list in `registry/`. `scripts/reorganize_run_logs.py` performed the regrouping
(moves + literal rewrites in every `jobs.jsonl`, run_logs script and repo file) and can
check (`check`) or undo it; its report lists the hand-fixed references.

### Command mapping

| Old in-repo CLI | Package |
|---|---|
| `uv run python -m scripts.jobq <cmd> --root R` | `uv run jobq <cmd>` (folder from `jobq_paths.toml`) |
| `--lock-prefix` | `--lock-prefix` on `jobq work` and `jobq status` |
| `SL_JOBQ_LOCK_PREFIX` | `JOBQ_LOCK_PREFIX` |

New commands: `stop --now`, `priority`, `pin`, `park` / `unpark`, `usage`, `config`.

### What the repo still calls

- `jobq.store` for reads and appends in sweeps and builders (`queue_exists`, `read_meta`,
  `load_jobs`, `ensure_dir`, results).
- `jobq.model.Job`, `jobq.model.sanitize_jobkey`, `jobq.model.MAX_JOBKEY_BYTES`.
- `src/subliminal_learning/gpu_peak.py` re-exports `jobq.report`.
- `src/subliminal_learning/eta_priors.py` keeps the duration priors, and
  `scripts/queue_status.py` is the campaign status table (one row per active queue with
  `t/job`, completion rate, ETA and stale-claim count); it reads the folder through
  `jobq.store` (default: the `jobq_paths.toml` folder; `--root` views another). The fitted duration model is written to
  `docs/QUEUE_ETA_HEURISTICS.md` by `uv run python -m scripts.queue_status --show-priors --markdown`.

### On-disk compatibility for existing folders

- `meta`, `jobs`, `results` and `attempts` files are read as they are.
- `complete.stamp` is ignored; completion is recorded in `complete.state.json`.
- `cap_per_gpu_by_node` in queue defaults is ignored.
- A queue pauses on disk after `max_consecutive_failures` (default 5) consecutive failures
  and stays paused until `jobq resume` or `jobq requeue`.
- Peak-memory reporting uses `JOBQ_REPORT_PEAK_MEM` / `JOBQ_PEAK_GPU_MEM_MIB=`.

### Policy notes

- `free_mem_mib: 0` with `cap_per_gpu` set gives cap-only admission: the memory gate is off.
- Add `"shared_perms": true` where machines run the same account under different uids.
- `mps` defaults to `"auto"`; MPS directories are `/tmp/jobq_mps_<uid>`.
- `--lock-prefix` exists on `jobq work` and `jobq status` (env `JOBQ_LOCK_PREFIX`). The old
  in-repo default slot-lock prefix was `/tmp/subliminal_jobq_<user>`; the package default is
  `/tmp/jobq_<12 hex of sha256(realpath(folder))>`. A pool on the old code and a pool on the
  package therefore do not share slot caps and must not run together on one machine.

### Partial-run cleanup — `scripts/clean_partial_runs.py`

Finds and (only with `--apply`) removes the partial/stale artifacts a mid-flight KILL leaves,
so a requeued job re-runs cleanly, *without ever deleting complete or in-flight data*. It is
run manually / on demand as a sweeper (the eval stage now skips incomplete `ftseed*`
checkpoints on its own, so a requeued job no longer needs an automatic post-kill cleanup).

- **Scope — exactly seven pipelines:** teacher (`train_teacher_templates_pipeline`), numbers
  (`generate_numbers_pipeline`), eval (`evaluate_teacher_preference_pipeline`, teachers +
  students), distill (`distill_student_pipeline`), steerability (`steerability_screen_pipeline`),
  eas (`eas_pipeline`), coherence (`pile_coherence_pipeline`). **Out of scope and never
  scanned/touched:** SDF (`sdf_teachers/`) and on-policy GKD (`gkd_students/`).
- **Per-stage rule.** Training leaves (`teachers|students/**/ftseed*`) are purged **only** when
  they hold a `checkpoints/` dir with no final weight file (mirrors
  `run_io.checkpoint_is_complete` — the orphan that poisons the eval enumerator); a complete or
  resumable (`checkpoint-*`) leaf is kept. Numbers/eval/eas artifacts are atomic + self-healing,
  so cleanup there is orphan `*.tmp.<pid>` only. Steerability drops only a **partial** (or
  corrupt) `aggregate.json` — recorded `n_seeds` ≠ the `ft*.json` on disk — never a complete one;
  `ft{seed}.json` are kept. Coherence scans the per-run `pile_coherence/` dirs (under
  `teachers/`/`students/`) and the shared `coherence/{shard}/base/` tree: it drops `*.tmp.*`
  orphans always and, optionally (`--coherence-orphan-npy`), a half-written pair — a
  per-token-loss npy with no sibling record JSON, or a record JSON with no npy — plus a
  **partial Pile shard** dir (`header.json` missing); a shard *with* a header is a complete,
  expensive artifact and is never touched.
- **Safety model.** Dry-run is the default. Every target is resolved and must sit under an
  allowed root — `output_root`, **plus a second root `dataset_cache_root` only when passed
  explicitly** via `--dataset-cache-root` (needed for partial shards, which live outside
  `output_root`); symlinked targets are refused. Scan mode has an `--min-age-s` guard (default
  6 h, must exceed the ~1 h longest job) read from the newest write ANYWHERE under a leaf
  (recursive, symlinks not followed) — a leaf dir's own mtime does not move when
  `checkpoints/checkpoint-N` is written inside it, and any `checkpoint-*` dir (weights-only
  trajectory saves included) shields the leaf outright, since the writer may be on another node. A **liveness guard** (in every mode) scans our own
  `/proc` cmdlines — *not* host-namespace nvidia-smi pids — and skips any leaf whose seed **and**
  cell tokens appear in a running command, so a concurrent sibling seed in a multi-seed cell is
  never deleted. Targeted mode (`--path`) skips the age guard but requires an explicit seed
  selection for a cell (`--seeds`, a parseable `--from-job-cmd`/`$JOBQ_JOB_CMD` — understands
  `--student-seed N` and the `for S in 1 2 3; do … $S …; done` idiom — or the guarded
  `--all-seeds`); with no parseable selection it deletes nothing and exits non-zero. The
  `numbers/**/ftseed*` pools (no manifest, no checkpoints) are never selected as leaves — those
  trees are only swept for `*.tmp.*`.
- **Targeted sweep** (liveness protects live siblings, `--path` scopes to one subtree, an
  explicit seed selection scopes to the dead job's cell):

  ```
  uv run python -m scripts.clean_partial_runs \
      --path /mnt/ssd-2/soar-subliminal/angela/outputs/students/dog \
      --stage distill --seeds 1 --apply
  ```

  Manual sweep: `uv run python -m scripts.clean_partial_runs --scan` (dry run), then `--apply`.

## Campaign status and ETAs — `scripts/queue_status.py` + `eta_priors.py`

**Campaign status table (2026-09-14): `scripts/queue_status.py`.** Read-only, one row per
ACTIVE queue of a root (parked `PARKED-*` and completed queues hidden unless `--all`):
priority, pin, pending/running/waiting/done/failed, a `t/job` expected per-job duration,
trailing-3 h completion rate, a per-queue ETA, and a stale-claim count (age > 3x expected).
`t/job` comes from `subliminal_learning.eta_priors` (`src/subliminal_learning/eta_priors.py`) (roles added 2026-09-17): the
job's SCRIPTS say what it DOES and that picks the key -- carrying `--epochs` no longer makes a
job a training job. A student TRAINING is keyed by its RECEIVER ARCH + EPOCHS alone, pooled
across every script, worker and queue (`160m ep20 ~ 52 m`, `160m ep10 ~ 25 m`,
`410m ep20 ~ 1 h50`, `olmo-1b ep10 ~ 2 h05` fp32, `qwen ep5 ~ 55 m` tf32 vs ~1 h24 fp32),
refined by exact LoRA rank and fp32/tf32 when those levels hold enough samples; teacher sweeps
are keyed `teacher|arch|ep` and priced per CELL (the grid cells this `--num-shards` slice
trains), number generation `numgen|arch` per 1,000 PROMPTS (160m 16 s, olmo-1b 41 s, qwen 68 s
per 1k), evals `eval|<set>|arch` per RUN and trajectories `traj|<script>|arch` per CHECKPOINT;
everything else keeps a script-chain + receiver key, and a chain never measured as a whole is
estimated by SUMMING its components' (per-unit) priors (`(p,sum)`). Receiver/epochs/rank/
precision/units are read from the flags and from the YAMLs, tag lists, registries and run lists
the command points at (`--sweep-config` base model + HP grid + cell count, `--training-config`
`lora.r` / `train.n_epochs` / `train.tf32`, `hf_worker --family` -> that family's
`lora_student_distill_<family>.yaml`, with a `git log -S "tf32: true"` date so old runs are not
mislabelled and an explicit `_tf32`/`_fp32` marker in the command winning over both). Priors are measured over EVERY historical result of this root plus
`run_logs/queues/retired/queue0`/`run_logs/queues/queue_angela` (~113k samples), each key quoting the first sample
TIER with enough samples -- `t1` today's nodes since the MPS rollout, `t2` any node since then,
`t3` all history -- newest first, capped at `--prior-recent-n` (3,000), resume/cache-hit samples
below 0.25xp90 dropped, one queue capped at 2,000 samples; the result is blended with the
queue's own samples for that signature (`w = n_own/(n_own+10)`): `47m(n=138)` own,
`46m(n=5+160m·ep20)` blended, `~47m(160m·ep20)` prior, `~51m(160m·ep20·r128)` its rank
refinement, `~12m(160m·ep5←ep20)` epoch-extrapolated, `~10m(?)` unknown (the default
`--unknown-job-s`, `?` on that row only). A shape with NO measured cell is reasoned from a
fitted cost MODEL (2026-09-17) rather than looked up: a student costs `per-epoch
cost(receiver, precision) x epochs x rank factor(rank) x load factor(d)`, fitted at build time
from the tier-1 samples -- per-epoch cost = `p50(seconds)/epochs` over the pooled LOADED
3-6-up regime (160m 2.41 min, 410m 6.08, qwen tf32 11.2, olmo-1b fp32 14.7), rank flat to r64
and +8-15% at r128/r256, 160m load factors 0.37 / 0.44 / 0.80 / 0.82 / 1.00 / 1.15 at 1..6-up.
Every factor is fitted WITHIN an epoch group (resume filter first) and combined across groups
by an n-weighted median, never pooled across epochs. Load factors are APPLIED only for a
receiver with >= 1,000 tier-1 samples whose factors do not fall as the GPU fills (160m, 410m);
on a 15-200-job campaign the co-tenancy bucket is a proxy for CAMPAIGN PHASE, not contention
(inside the OLMo queues later-started jobs are both faster and more co-tenant: corr(duration,
start order) -0.3..-0.87, corr(bucket, start order) up to +0.83, pooled corr(duration, bucket)
-0.30), so olmo-1b/qwen keep 1.0 everywhere and the sheet parenthesises their measured factors
(`--json load_factor_applied`) --
quoted at the pin node's jobs-per-busy-GPU (`node_density`, default 5-up) and labelled
`~17m(160m·ep7·r16·model)` (`--json`: `model_per_epoch_min`, `rank_factor`, `load_factor`).
The teacher/numgen/eval/traj roles keep their per-UNIT rates. The table is cached incrementally in
`<root>/duration_priors_cache.json` (per-queue fingerprint; complete queues never rescanned;
`--rebuild-priors` forces a full ~45 s rescan, warm adds ~4 s) and the per-command feature
parse in a world-writable `cmd_feats_cache.json` beside it (warm status 46 s -> 13 s); `--show-priors` prints the
REASONED sheet (the model, its factors and residual band, the per-unit rates, worked examples),
`--raw` appends the measured cells with per-GPU-co-tenancy and MPS-on/off p50s and peak GPU
memory, and `--show-priors --markdown` regenerates all of `docs/QUEUE_ETA_HEURISTICS.md`
(sheet + raw appendix).

The ETA (rebuilt 2026-09-17) is a DISCRETE-EVENT SIMULATION of the pools at each node's
CAPACITY -- `(gpus - yielded) x cap_per_gpu` from `gpu_policy.<host>.json`, counted only when
the node is live (claims now or a pool log touched within `--pool-live-min`) -- with jobs
consuming `slots` and `slots: 0` jobs in a separate `cpu_cap` lane: every lane starts holding
its running jobs, and whenever slots free each lane claims the highest-priority READY queue
that may run there, so a high-priority dependent PRE-EMPTS lower-priority pending work the
moment its dependency finishes (that row was reading 12 h53 under the old water-fill, 4 h20
now), equal-priority queues on one node share its slots, and pinned/unpinned/`yielded`/`blocked`
behave as before. A trailing `r/h` resting on >= 10 completions cross-checks each eta and
appends `!` when it disagrees by more than 50% (`--json eta_disagree`). Warm calls skip the
job/claim scan of queues whose completion witness (`complete.state.json`) is valid (34 s -> ~12 s on a 2,215-queue root).
Backs the `/status` command
(`.claude/commands/status.md`). A `mem` column (2026-09-15) shows the GiB ask pending+running
jobs are gated on (the pool's effective ask: job/`defaults.mem_mib` raised by the
attempts-sidecar OOM floor; `min-max` range when jobs differ, `^` = floor-escalated, `-` = no
ask). Tests: `tests/test_queue_status.py`, `tests/test_eta_priors.py`.

## Config system

`config.py` defines frozen dataclasses (`PathsConfig`, `ModelConfig`, `LoRAConfig`,
`TrainConfig`, `EvalConfig`) and one minimal composer, `load_config`. Each workflow has its
**own** composer (`teachers/config.load_template_config`, `sdf/config.load_sdf_config`,
`gkd/config.load_gkd_*`), so the subpackages stay self-contained.

- **paths** — filesystem roots; CLIs default to gitignored `configs/paths/paths_local.yaml`.
- **model** — hub org + PolyPythias registry facts.
- **training** — `lora:`/`train:` blocks. SDF full FT (`full_sdf_teacher.yaml`) has *no*
  `lora:` block — that absence is what selects full FT.
- **experiment** — grid + eval selections per workflow. Sweep YAMLs support `run_set_id`
  campaign subdirs (`{YYYYMMDD}_{purpose}_{size}`, `run_set.json` cards, `RUNS_INDEX.jsonl`,
  staleness guard — see `docs/PROJECT_DESIGN.md`).
- **`extends:`** — any experiment or training YAML may name a parent (path relative to the
  file, or a list applied in order). `config.load_yaml_extends` loads parents first and
  deep-merges the child on top (mappings merge, child wins; lists/scalars/`null` replace;
  the key is dropped); a file without `extends` is a plain `yaml.safe_load`. Every reader of
  these YAMLs goes through it except the SDF/GKD scripts and `campaign/*`. Families are
  parent + overrides: `ckpt_distill_sweep_*` extend the root `ckpt_distill_sweep.yaml`,
  `teacher_ft_sweep_selected_*` / `number_gen_sweep_*` their family base,
  `lora_template_bias_*` / `lora_student_distill_*` the Pythia recipe. The ckpt `pairs` block
  also takes `window` + `extra_pairs` with `topology: full`; `sweep_ckpt_distill submit`
  writes the fully resolved doc as the queue snapshot and reuses an existing snapshot when it
  matches the resolved doc or, failing that, expands to identical jobs and queues with the
  same values outside `pairs` and the legacy queue-root key that pre-toml snapshots carry.
- **sdf_docgen.yaml** — read directly by the isolated tool via `yaml.safe_load`.

Machine-local env (GPU pin, `WANDB_MODE`, `HF_TOKEN`) loads from a gitignored `.env`.
One shared template lives in the CephFS stash at `node_home/.env` (secrets blank; the
stash is world-readable) and is mirrored verbatim as the committed `.env.example`.
`node_home/bootstrap.sh` copies the stash template to `<repo>/.env` on a fresh node
unless one already exists; paste `HF_TOKEN` or narrow `CUDA_VISIBLE_DEVICES` only in that
node-local copy. (2026-09-08: replaced the per-hostname `.env.soar-subliminal-{0,1}`
copies; `HF_HOME` now points at the cold-1 cache.)

## Model registry (shared)

`models/registry.py` maps `PythiaModelSpec` (`size`, `seed`, `variant`, `revision`) → HF
repo id: `full`/seed 0 → `EleutherAI/pythia-{size}`; `data`/`weight` variants →
`EleutherAI/pythia-160m-{variant}-seed{N}` (only size 160m, seed 1–3). `loading_io`
downloads tokenizer + model in **fp32** (bf16 opt-in) and eval-modes it.

`model_ref(spec, hub_org)` is the identity-grade reference (repo id + `@{revision}` iff
set); manifests and run identities record it, and directory names get the same suffix, so
pretraining-checkpoint sweeps have distinct paths. `parse_base_models` accepts `config`,
`seeds` (all 6 data/weight models), `original` (the stock `pythia-160m`; label
`full-seed0`), `all7`, explicit tokens with `@revision`. The fully re-randomized
`pythia-160m-seed{N}` variants (weight init AND data order both re-randomized — a third
model kind, distinct from the single-factor data-/weight-seed ensembles) are selectable
since 2026-08-18 (previously rejected by design), but **double-gated**: the explicit
`full-seed<N>` or `pythia-<size>-seed<N>` spelling AND the `allow_full_seeds=True`
force argument (launch CLIs — teacher FT, number gen, distill, both preference evals —
expose it as `--allow-full-seed-bases`; the crossdistill worker as
`--allow-full-seed-base`, which also validates its free-string `--teacher-base` against
the known dir names). Inverters of existing on-disk dir names (`spec_from_model_dir`,
CKA model refs) opt in unconditionally — a directory that exists is already an
unambiguous identity. Tokens are validated against
`registry.POLYPYTHIAS_FULL_SEED_SIZES` (14m–410m) × seeds 1–9. A bare `seed<N>` token
stays invalid and no group selector (`seeds`/`all`/`all7`) ever includes them, so a run
on one can never be the result of a typo'd or wildcarded data-/weight-seed selection;
on disk they are keyed by repo basename `pythia-160m-seed{N}`, disjoint from
`pythia-160m-{data,weight}-seed{N}`. `spec_label` is **size-aware** for non-160m `full`
specs (`pythia-410m`, `pythia-410m-seed3`) so a label always re-parses to the same spec;
every 160m label (`full-seed{0,N}`, `data-seed<N>`, `weight-seed<N>`) is byte-identical
to before. The YAML `model_spec:` route (teachers / SDF / GKD loaders) is validated by
the shared `teachers.model_specs.validate_spec`, so an out-of-range size/seed fails at
load; a full-seed base there requires a top-level `allow_full_seed_base: true` in the
experiment YAML (default false). (Unrelated: `configs/experiment/*_full_seed0.yaml`
FILENAMES mean "full fine-tune, seed 0" — not the `full-seed<N>` token grammar.)
Beyond those, the 160m **local families** (`registry.LOCAL_PYTHIA_FAMILIES`, one row per
re-pretraining: variant token, the dir template `pythia-160m-<variant>-seed<N>`, the
allowed seeds and a short plot label) cover gpaulo's unpublished runs — `weightc`
(seeds 1-3, the CORRECTED weight-seed re-pretraining: seed-N init, the original run's
exact positional data order), `pilerepro` (seeds 1234, 3; the Pile re-pretraining
reproduction) and `dclm` (seeds 1234, 1, 2; the same recipe on the DCLM corpus, so its
Pile CE is not comparable to a Pile-trained base). Repo-id resolution, token parsing and
validation, name inversion and the short labels (`wc1`, `pr3`, `dclm1234`) all read that
one table, so adding a family is one row. They are unpublished, so `loading_io` routes
`local/...` refs to `{hf_cache_root}/local/pythia-160m-<variant>-seed<N>/step{K}/` and
never to the hub, and no group selector (`seeds`, `all7`, ...) includes them: they need
no `allow_full_seeds` gate because the token is explicit and unique. Those dirs are
produced by `scripts/convert_weightc_checkpoints.py --family <variant> --seed <N>`
(`--steps` / `--every K [--to N]` / `--watch S` for a run still training; the (family,
seed) -> NeoX source tree table lives in that script, and `--vocab-file` overrides a
recorded 20B-tokenizer path that is not mounted), which runs gpt-neox's
`convert_neox_to_hf.py` with the four config fixes the stock converter gets wrong for
Pythia (intermediate_size, rotary base, bos/eos ids, tokenizer special tokens) plus a
fifth (2026-09-09): the tokenizer is saved through `GPTNeoXTokenizerFast`, not the bare
`PreTrainedTokenizerFast` the NeoX converter writes, whose default `model_input_names`
carries `token_type_ids` and makes every `model.generate(**enc)` / `model(**enc)` site
fail ("model_kwargs not used by the model") while SFT training, which pulls only
`input_ids`, runs fine. It writes a `conversion.json` provenance sidecar per step
(family, seed, source step/config, vocab file). A finished run gets a `main` alias
(`local/pythia-160m-weightc-seed1/main -> step143000`, 2026-09-12) so the bare token
`weightc-seed1` names the final checkpoint exactly as `pythia-160m-seed1` does: main-line
arms (seed2seed receivers) use the bare dir `students/<animal>/pythia-160m-weightc-seed1/`,
and only the ckpt sweeps spell the step (`ckpt/pythia-160m-weightc-seed1@step143000/`).
For a base still training, `scripts/ckpt_distill_gate_append.py` grows a ckpt-distill sweep
as conversions land: each target step is gated on held-out Pile CE against the genuine
`pythia-160m` at the same step (verdict persisted as `heldout_ce.json` beside the weights),
and a passing step is appended to the config (`steps` + the incremental ordered cross among
passed window steps, or its diagonal) and re-submitted with `--allow-snapshot-update`; a
failing step is logged and excluded. The Pile probe stem comes from the paths YAML key
`pile_preshuffled_document` (or `--preshuffled`; never a baked path), and both the
watcher's config rewrite and `submit`'s snapshot overwrite are atomic under one lock,
`{queue_folder}/{queue_prefix}.sweep.lock` (2026-09-10). `--window` takes comma-separated
items, each `lo:hi:stride` or a single step (`75000:125000:5000,130000,140000`), so a
design can skip converted steps it never asked for (2026-09-12: wc1 +130k/140k, not 135k).
`--follow-config` (repeatable, 2026-09-11) grows diagonal-only companion configs — e.g.
`ckpt_distill_sweep_dog_weightc1_diagseeds23.yaml`, student ftseeds 2–3 on the diagonal — with
every step the primary has confirmed submitted, submitting each only AFTER the primary so its
`jobq.depends_on_prefix` on the primary's per-step queues always resolves (no two pool jobs
race into one output dir). Each config keeps its own submission ledger
(`gate_append_submitted.json` for the primary, `gate_append_submitted.<stem>.json` per follower).
Pile-coherence discovery tokenizes a `pythia-160m-seed{N}` run dir as `full-seed<N>`
(previously it produced a bare `seed<N>` the round-trip parse rejected, silently skipping
the run), and `cka.model_refs.spec_from_base_name` validates its inverted spec against
the registry constants instead of minting one that 404s later. Analysis short labels
(`analysis.frames.short_model`, `transfer_factors.normalize_model_name`,
`collapse_law` base labels, the campaigns builder) map full-seed dirs to `full-seed<N>`
— never a bare `seed<N>`, and never pooled into the `same-model` family (they get their
own `full-seed` family and rank). A `pythia-<size>` token (any
size in `registry.PYTHIA_SIZES`, e.g. `pythia-410m`) selects that size's `full`/seed-0
model. Teacher FT, number generation, and student distillation take a
`--dtype {float32,bfloat16}` knob (default fp32, byte-identical; non-default recorded
in the manifest only). The pythia-2.8B replication path was dropped 2026-08-18 (its
configs deleted; `PYTHIA_SIZES` trimmed back to 14m-410m). Seed ensembles (data/weight
variants) exist only at 160m.

**Seed-factor provenance (forensics 2026-08-27/28; optimizer finding 2026-09-13).**
Inspect **both `step0` and `step1`** for Pythia: equal saved parameters after the
first zero-LR optimizer step do not establish equal first gradients or optimizer
state. Released `weight-seed{1,2,3}` have distinct `step0` initializers but all 148
parameter tensors at `step1` equal original's. A new reconstruction strongly
supports **retained Adam history from each ws step0 gradient after a parameter
reset to original**. On the 768-dimensional final-layernorm bias, mixing each
ws initializer's first gradient with original's second gradient predicts its
observed step1→step2 update at cosine 0.9971–0.9974, versus 0.6795–0.6839 using
original's first gradient and 0.5171–0.5480 using another ws seed. These comparisons
fit no parameters to ws updates. Stale original-valued optimizer master weights
could explain the reset, but the exact upstream initialization/loading operation
has not been identified. This supersedes the “unused/orphan step0” and “pure
training noise” explanations; ws is confounded by seed-specific optimizer history.

The ws early untouched-input-embedding-row fingerprints match original (Jaccard
0.9994–1.0000), supporting shared data coverage/history, not exact token positions.
`data-seed{1,2,3}` share original's init and have distinct data orders.
`seed{1..9}` have independent inits and data orders distinct from original, ds,
and each other. Original, ds, and fs have exactly equal `step0`/`step1` parameters.
The corrected local `weightc-seed1` also has equal step0/step1 parameters (148/148,
native-checkpoint comparison 2026-09-13; `wc1_step0_step1_report.json` in the
investigation directory below). It uses fs1's init and original's data order
(verified 2026-09-12; `run_logs/campaigns/fsfact/configs/teacher_ft_sweep_selected_weightc1.yaml`).
Thus released ws arms do not provide a clean independent-init comparison; wc1
versus original supplies the intended different-init/same-order comparison.
The 2026-09-13 direct native/HF comparison also confirms wc1/fs1 step0 equality
on all 148 parameters. wc1 uses 8 GPUs × microbatch32 × accumulation4, versus
original's 32×32×1; the total batch is 1024 in both. The candidate fs1 W&B run uses
16×64×1, but its mapping to the released HF model remains provisional. See
`run_logs/campaigns/ws_origin_investigation_20260913/WC1_COMPARISON.md`: step0 is saved before
training; its first gradient is computed between step0 and step1. wc1's sampled
optimizer shard directly shows zero moments at step0 and nonzero moments at step1.
See `docs/results/seed_suite_provenance.md` and the reproducible evidence in
`run_logs/campaigns/ws_origin_investigation_20260913/OPTIMIZER_FINDINGS.md` (revision-pinned
observations, sampled CPU gradient reconstruction, fixed prediction and controls).
`models/seed_identity.py` provides revision-agnostic weight comparison helpers.
**Pythia init / data-order identity (2026-09-16): all init and data-order identity
checks go through `models/seed_identity.py` (pure: `untouched_rows(method=
"residual"|"exact")`, pad-control-gated `compare_untouched` / `compare_ground_truth`,
`value_hashes` / `init_state_equality`, `transitive_classes`) +
`models/seed_identity_io.py` (named 160m model table — HF orig/ds/ws/fs from the
registry, gpaulo native runs in one `NATIVE_RUNS` table — offline HF-cache and native
NeoX loaders, Pile `pile_seen_masks`) driven by `scripts/check_pythia_seed_identity.py`
(→ `{output_root}/analysis/seed_identity/<run>/identity.{json,png}`); do not
re-implement them.** For fp16 Pythia checkpoints the canonical untouched criterion is
exact row equality (default `--method exact`; the residual test misses ~1-quantum
step1→2 updates: orig 2620 vs exact/Pile 1654/1653+token 20746); residual stays the
`untouched_rows` default so `check_dd_seed_identity` (fp32) is unchanged. HF snapshots
load `pytorch_model.bin` (fp16) before the fp32-converted `model.safetensors`.
Pile truth "seen" = token at an input position 0–2047 of records [0, 1024·b) (position 2048 is label-only per gpt-neox `_get_batch`; counting it wrongly flagged token 20746). Pile truth verdicts: Pile-corpus runs are MATCH / MATCH_WITH_EXCEPTIONS (≤
`--max-ground-truth-mismatches`, default 5; all one-sided ids in the JSON plus a per-window
`ground_truth_exceptions` id→models summary) / MISMATCH; DCLM runs (incl. cold-1
`dclm_main_cold`/`dclm_s1_cold`, or `--native label=path:dclm`) are FAR_FROM_PILE when
Jaccard < `--far-from-pile-jaccard` (0.9) else UNEXPECTEDLY_CLOSE_TO_PILE. Order classes use the same allowance: two models share a data order when their untouched sets differ by ≤ `--max-order-mismatches` (default 5) ids at every conclusive window (pair verdict SHARED / SHARED_WITH_EXCEPTIONS, full one-sided id lists per pair, `order.exceptions` id→pairs summary; cliques still required, else non-transitive); `compare_untouched` defaults to 0 so `check_dd_seed_identity` is unchanged. Padding-row ancestry: `scripts/check_pythia_padding_ancestry.py` (pure `compare_padding_rows`, io `latest_step`) writes `padding_ancestry.json` into the same run dir: pad rows (never inputs, decay only) of each model's latest checkpoint vs every step1 init (cos/scale), own step0/1, early windows, and the step1→latest residual untouched ids.
The algorithms and their inferential rationale are explained in
[`docs/methodology/pythia_seed_verification.md`](docs/methodology/pythia_seed_verification.md):
exact step0/step1 parameter comparisons, the Adam-history reconstruction,
untouched-row residuals and padding controls, and corpus-derived Pile coverage
checks for the local Pile/DCLM comparisons. Coverage fingerprints do not establish
exact token order; the September 16 `64→128` window fails its padding control
and is uninterpretable. The legacy A0 gate checks report existence only.
A direct final-versus-init check (2026-09-16) also finds original-init directions
in all 27 final ws padding rows and 214 real-vocabulary rows per model; no real
rows meet the same cosine threshold against their own ws step0. The main and
step143000 input embeddings agree exactly, supporting final-model ancestry
independently of the step1 export (methodology §4 and its linked report).
The prior no-embedding-decay claim is corrected: the recorded recipe uses weight
decay 0.1, and NeoX v1 groups input embeddings with decayed weights (biases and
normalization parameters are exempt). The row fingerprint removes scalar decay.
The original/reproduction first-window Jaccard remains 0.6561 at the fixed 1e-6
cutoff. A post-hoc exact-equality/corpus check finds matching sets of 1654 tokens,
but does not explain the different small updates or establish a calibrated new
cutoff; the earlier "resolved threshold artifact" framing is withdrawn (methodology
§2; `FIRST_WINDOW_THRESHOLD.md` in the investigation). The follow-up
`rows901_probe_20260916.py` reproduces every disputed token: all 901 occur in the
first two input batches and have visible changes; raw and scale-adjusted metrics
give the same classifications. Historical released-ws Jaccards used exact equality
(0.9994–1), whereas applying the residual cutoff gives 0.833–0.849. The report and
per-token CSV preserve this distinction and leave the cause of differing small
embedding updates unresolved.
The local follow-up also found a separate tokenizer default difference (original
text cleanup false, ws true) despite identical vocab/merges; examples and decoding
call sites are recorded in the investigation's `README.md`.
Supporting wiring for the
2026-08-19 fs1 campaign: `sweep_teacher_templates --only-param-tags` (numgen-style tag
file, filtered before sharding, errors if a tag is grid-inexpressible),
`animal_samedistill_worker --allow-full-seed-base`, and the selected-sweep YAMLs
`teacher_ft_sweep_selected_fullseed1.yaml` / `number_gen_sweep_fullseed1.yaml`
(dog, ftseed1, union grid of the 103 tags with ≥5k kept prefix numbers on pythia-160m;
tag manifest `run_logs/registry/dog_fullseed1_teacher_tags_5k.json`).

**Seed2seed core56 checkpoint mode (2026-09-02).** The orig×fs / fs×fs seed2seed
remainder (lion fs1-fs3 onward, all cat and pig pairs; 40 queues, 21,813 jobs) runs
`run_logs/workers/dog_crossdistill_studenthp_worker.py --ckpts 40`: 40 evenly spaced
adapter-only checkpoints (`save_steps` = 25000/40 = 625, `save_only_model`, pruned to
`adapter_model.safetensors` + `adapter_config.json`, the trainer's extra final-step
`checkpoint-25000` included), then the endpoint eval, then per-checkpoint trait
(`ckpt_preference_trajectory --sets test,off_topic`) and Pile-KL
(`ckpt_pile_kl_trajectory`) trajectories into `{leaf}/eval/{pref,pile_kl}_traj.json`
plus `timings.json` — the 410m worker's distill stage ported one-to-one. Per-run
registries land in `run_logs/campaigns/fsfact/ckpt_registry/`. Storage: ~22.6 MB mean adapter over
the core56 rank mix → ~0.9 GB/run, ~20 TB for the campaign (vs ~0.6 TB endpoint-only).
Endpoint-only exceptions inside the same arms: lion fs1-orig, orig-fs2, fs1-fs1, fs1-fs2
and the first 27 fs1-fs3 cells (claimed before the flag landed). `--ckpts 0` (default)
is byte-identical to the earlier behaviour, so the dog seed2seed arms are unaffected.

`notebooks/fsfact_transfer_analysis.ipynb` (builder
`scripts/build_fsfact_transfer_nb.py`) is the endpoint-transfer overview for this factorial:
four animal panels show every directed `orig`/`fs1`/`fs2`/`fs3` pair, including reverse and
same-base directions. Each box is a distribution over core56 student cells after averaging
the exact top-10 teacher set within a cell; only ten-teacher-complete cells are plotted, while
complete/touched coverage for all 64 animal-arms is printed separately. The builder also writes
the ignored convenience export `docs/results/figures/fsfact_transfer_boxplots.png`; the executed
notebook is the tracked artifact and remains safe to rerun while the later animal queues drain.

`notebooks/transfer_trajectory_grids.ipynb` (builder
`scripts/build_transfer_trajectory_grids_nb.py`, 09-04) puts the trait score on the
(rank, alpha) grid *along training* for every arm side by side — Pythia-160m same-model and
cross-init ds1, Pythia-410m same-model and cross-seed, OLMo-1B same-model and cross-seed 2:
endpoint, trajectory peak, and the saved checkpoint nearest step 625 / 1,250 / 2,500 /
5,000 / 10,000 (always absolute steps, never a share of the run — run lengths differ: Pythia ep20
25,000, OLMo ep10 12,500),
median over alive lr rungs on the nonadaptive lattice (seed 1), plus a prior-normalized
summary. Pythia arms load through `analysis.trajectories.lineage_long_frame` (one row per
run × checkpoint, pref + Pile-KL joined on step); OLMo through `analysis.dd1b_ladder`.
Its §6 fits the boundary law on checkpoint-resolved labels: free `(b, c)` at every
checkpoint plus `stats.collapse_law.fit_collapse_law_trajectory` (ln step as a fourth factor,
cluster-robust by run, `TrajectoryLawFit.kstar_at(step)`); arms with mixed save cadences
(OLMo cross-seed) are snapped onto their coarser grid first, and an arm whose last-checkpoint
labels are perfectly separated takes its endpoint exponents from the endpoint records (the
table says so) rather than the 160m constants.

`notebooks/transfer_receiver_family.ipynb` (builder
`scripts/build_transfer_receiver_family_nb.py`) puts all ten Pythia-160m receiver bases on one
axis: `orig` (same-model), the `ds*`/`ws*` PolyPythias siblings (shared-init; `ws*` share the
data order too) and the `fs*` full-seed bases (neither factor). The population is the
original-init top-10 teachers x core56 x student seed 1, and the two-row figure
(`docs/results/figures/transfer_receiver_family_boxplots.png`) shows all landed runs above and
only the runs alive under the per-receiver unigram-anchored Pile-KL cut below (the two charts
keep independent x axes so either reads alone), with per-arm coverage, per-receiver and
family-pooled tables, and directed sender x receiver matrices per animal. A second figure,
`docs/results/figures/transfer_all_directions_boxplots.png`, generalizes the box view to every
landed sender x receiver direction including the same-model diagonal (three panels = the
sender/receiver relationship from `fsfact_transfer.direction_relation` — same model / shared
init, different data order / different init — one slot per direction with the four animals side
by side, colour = animal, thick edge = diagonal, red `not run` in a reserved diagonal slot that
never ran); a third figure, `transfer_all_directions_boxplots_nows_linear.png`, is the same
drawing (`draw_direction_boxes`) with every `ws*`-touching direction dropped (the `ws*` bases
vary neither init nor data order) and a linear, unclipped y axis whose annotation band sits
below zero; that directed population is loaded once and reused by the heatmap section. Coverage caveats are carried in the figures themselves: annotations are red wherever
`probed < landed` (the alive chart is then a different cell subsample) and absent directions
are unrun arms — the ds*/ws* diagonals were never run on core56. Its one `src/` requirement was extending `transfer_factors._BASE_EVAL_SHORT` with the
`full-seed{1,2,3}` -> `fs{1,2,3}` entries so `load_priors` also returns the full-seed bases'
`test_prior`; the receiver/family/prior-label tables and `count_complete_cells` live in
`analysis.fsfact_transfer`.

Figure 2 family outputs live under `notebooks/figures/figure2_family/`; the eight
original charts are archived byte-for-byte in `drafts/`. The display order is same
model, shared init/different data order, different init/same data order, then
cross seed/different init. New bars show strict win probability and the median,
75th percentile, 95th percentile, and observed maximum of `p_student-p_control`
and `(p_student-p_control)/p_receiver_base`. Each animal has equal mass, each
available directed arm has equal mass within an animal, and available paired
runs share their arm's mass. Quantiles come from that weighted mixture CDF.
The 2,000-draw pointwise intervals use crossed Exp(1) configuration weights for
source teachers and student cells, preserving reuse across arms. Animals, arms,
selected teachers, evaluation probabilities, and priors are conditioned on;
these exploratory intervals do not establish generalization to new animals or
model seeds. The maximum is descriptive and has no confidence interval.
Win and percentile bars use all finite matched pairs (`collapse_filter=False`),
including dead and unlabelled treatments and controls, with no screening hatching.
Only the historical boxplot views retain screened eligibility and screening flags.
Loaders retain `screened_eligible` and both collapse labels, so one load supplies
both populations. Win estimates remain in `transfer_family_balanced_win_summary.csv`;
percentile tables carry no win columns. Compact versions of the original four
relationships show either pooled or dog-only quantiles, with either relationship
families or statistics on the x-axis. Both raw and receiver-prior-normalized versions
are exported as PNG, PDF, and SVG; prior-normalized figures, their summary CSV,
and receiver-prior metadata live under `figure2_family/relative/`, while raw and win
outputs remain in the parent directory. The median, 75th and 95th percentiles have
the same exploratory configuration intervals as the detailed charts, and the
maximum has none. Earlier screened percentile figures are archived in
`drafts/screened_percentiles_v1/`. Descriptive collapse audits and concrete
weighting/bootstrap examples are documented in the folder README.
CSV summaries, per-direction screening coverage, and exact test-prior provenance
are saved alongside the charts; the folder README documents reproduction.
The extension reads the authoritative xarch and OLMo recipe/seed registries through
`analysis.transfer_family_registry_io`, adding four directed size/seed-lineage
categories, both Pythia↔OLMo directions, two OLMo corpus/init classes, and OLMo
cross-seed (different init, same corpus). It preserves registry selection-cell
exclusions and per-teacher dataset overrides. Partial arms retain available
matched endpoints with coverage flags; pilots and arms without controls do not
become artificial zero effects. Pooled summaries name their actual animal sets
(dog and/or elephant in the added categories), and missing priors affect only the
normalized view. Size and architecture comparisons share a six-category dog-only
panel; OLMo corpus/seed comparisons form a three-category elephant-only panel.
Single-animal panels omit the duplicate pooled bars. The two Pythia↔OLMo summaries
exclude elephant rows before aggregation, recording them and their reason in
`figure_cohort_exclusions.csv`; OLMo-only elephant rows remain included.
Original Pythia 160M/410M lineages are not labelled as identical
weight initialization across different tensor shapes.
The separate `scripts.plot_transfer_family_run_weighted` entrypoint adds pooled
and dog-only raw compact figures under `figure2_family/run_weighted/`, with
statistics on the x-axis and families in the legend. It leaves existing balanced
figures unchanged. Its `analysis.transfer_family_pooled.row_bootstrap` resamples
N matched rows with replacement, retaining treatment/control pairing, and computes
ordinary linear-interpolated quantiles and pointwise percentile intervals. Every
row has equal starting weight; shared teacher, configuration, and control
dependence is ignored and explicitly stated. The maximum remains descriptive.
Canonical manifest checks confirm core30 for same-model and different-init/same-data,
and core56 for cross-data-order and cross-seed, with no outside cell tags.
The same entrypoint accepts `--method stratified`, `--method multiway`, or
`--method both`, writing only into matching subdirectories of `run_weighted/`.
Stratified resampling holds each animal–arm row count fixed. Multiway resampling
samples source-teacher and global student-configuration identities independently
with replacement; integer frequency quantiles match explicit repeated-row linear
quantiles, with no per-arm rebalancing. Empty intersection draws are counted and
redrawn with a bounded retry limit. Both procedures keep the existing equal-run
point estimates and empirical maximum; only uncertainty intervals change.

The additional `scripts.plot_transfer_family_multiway` entrypoint writes complete-grid
win and combined figures into `figure2_family/run_weighted/multiway/`, preserving the
previous outputs. `analysis.transfer_family_complete` retains complete teacher × animal ×
directed-arm blocks on canonical core30/core56 (Pythia) or core35 (OLMo receiver) grids,
at student ftseed 1; OLMo teacher tseed 0 is distinct from that student seed. Incomplete
blocks, off-grid cells, duplicate endpoints and alternative seeds cannot inflate `n`.
For this figure's mixed-design cohort, new cat/lion/pig `ds1 → orig/ds2/ds3`
arms use core30 while established complete core56 directions remain core56.
The arm-specific override is explicit in `transfer_family_complete._legacy_grid`;
it does not infer a smaller design from missing results. `wc1 → wc1` remains
outside the figure. This differs from the headline table's per-animal core30
restriction and preserves the existing Figure 2 core56 endpoints.
The new registry option `include_selection_cells=True` includes every canonical cell for
these requested descriptive figures; the default exclusions used by older callers remain
unchanged. All collapse states are retained. Both win and percentile panels use equal-run
multiway bootstrap intervals. The two-row combined layout places Pythia win probabilities
and raw percentiles above adjacent size/architecture and OLMo panels, retaining directions
as separate categories. A missing complete OLMo core35 category is shown as unavailable.
Matched-pair, block-completeness, exclusion, source-coverage and snapshot records accompany
the figures; `multiway/COMPLETE_GRIDS.md` documents the cohort and resampling procedure.
The same plotting module provides `export_top_row` to render a two-panel paper variant
from the saved summary, with matching relationship names, a distinct percentile palette,
and a zero lower limit in panel B; outputs live in `multiway/top_row/`.
The optional `quantile_layout="families"` export uses the same relationship x-axis
in both panels, with B colors assigned to median, p75, p95, and maximum. Its
`_families` filename preserves the original layout. Both versions use the refreshed
`top_row/complete_multiway_summary.csv`, with cohort and refresh audits alongside it.

`export_size_olmo(table, ctx, out_dir)` adds a separate grouped-percentile figure
with six dog Pythia size/architecture directions on the left (including both
Pythia↔OLMo directions) and four elephant OLMo families on the right. The compact
figure uses horizontal labels, a 1.5:1 left/right panel width ratio, and a shared upper
limit covering every positive bar and interval endpoint; the lower limit remains zero. Published outputs live
under `notebooks/figures/figure2_family/olmo/olmo_size_distribution` with adjacent
summary and provenance records. `include_olmo_same_model=True` on the registry and complete-cohort
loaders opts into the three canonical core35 diagonal campaign arms using their
exact teacher manifests; the older ctrl20 reference stays coverage-only. The
in-memory registry is shared by extraction and complete-block selection. Empty
same-model support is shown as pending, never as zero. Defaults leave existing
cohorts unchanged; this figure's audits live beside the published outputs.


`uv run python -m scripts.plot_transfer_family_multiway --all-arms` refreshes the
raw-percentile clean and metadata-rich overviews plus three detail panels into
`notebooks/figures/all_arms_transfer_distribution/`. The `_with_metadata` overview retains n, block counts,
and core-grid labels. Overview labels wrap after commas/model-size directions, abbreviate
“different” to “diff”, and omit the overall title; the same-init/same-corpus OLMo
row precedes the three cross-corpus/seed rows. It reuses the complete-cohort loader and equal-run two-way
cluster bootstrap (2,000 replicates by default), includes separate Rerun1234 and
Modern1234 reproduction rows, opts into canonical OLMo same-model core35 arms, and retains
all canonical selection cells and collapse states. Legacy runs are refreshed through
the notebook's record cache, and registry/re-pretraining arms through leaf-signature
validated `load_cell_matrix` caches. All-arms mode additionally loads the
original ↔ modern reproduction arms from `repretrain_significance_arms.json`,
retaining complete core30 blocks. Rerun1234 (`prv1`) and Modern1234 (`pr1234`)
are display aliases for distinct model IDs; default figure cohorts are unchanged.
This mode emits PNG/PDF/SVG, matched-pair and
cohort audits, and a source snapshot without changing the paper or balanced win
figures. The standalone directory also contains `ARM_AUDIT.md`,
`arm_seed_audit.csv`, and `condition_audit.csv`, recording directed arms,
complete-block support, and retained/requested fine-tuning seeds. The September 25
full refresh records its native source scan interval and preserves the preceding
cohort and exports under `drafts/`. The five former balanced raw artifacts are moved byte-for-byte into
`notebooks/figures/all_arms_transfer_distribution/drafts/balanced_20260924/` with a hash manifest.


The pooled treatment-minus-control view of that same population is reproducible outside the
notebook: `analysis.transfer_family_pooled` pairs each retained treatment run with the
untuned-sender control at its `(sender, receiver, animal, cell)` stratum and boxes the
differences by relationship, and `scripts/plot_transfer_family_pooled_minus_control.py`
(`uv run python -m scripts.plot_transfer_family_pooled_minus_control`) writes the four
`notebooks/figures/figure2_family/transfer_family_pooled_minus_control_alive{,_linear,_floor1e-2,_4spaces}.png`
charts (symlog to 1e-6, linear, symlog floored at 1e-2, and the 2x2 four-value-spaces panel)
plus the pooled summary table. The three legacy slots keep coming from the executed notebook's
data cells (`legacy_pairs`, which re-executes them with every save suppressed) because
`fsfact_transfer` only knows the `orig`/`ds`/`ws`/`fs` tokens; a fourth slot, *different init,
same data order*, adds the ordered pairs among the re-pretrainings `wc1`/`wc2`/`prv1` (same
Pile data order, a different weight init) on `run_logs/registry/core30_cells.json` x the per-animal
top-10 k-band teachers, read straight off the output tree with
`transfer_significance_io.load_cell_matrix` (`repretrain_pairs`). Receivers without an anchored
pp1 cut (`weightc-seed2`, `pilereprov1-seed1234`) cannot be dead-labelled: their runs are kept
UNFILTERED and their annotations drawn red, never dropped and never called alive.

### Pythia-410m cross-seed arm (2026-08-21)

First replication of the dog chain off 160m at 410m: teacher + numbers on the original
`pythia-410m` (`EleutherAI/pythia-410m`), students distilled cross-base into
`pythia-410m-seed1` (PolyPythias full-seed: init AND data order re-randomized — the only
ensemble at 410m, so this is a no-shared-factor pair like the 160m fs1 arm; same-model
students were dropped as uninformative). No `src/` change was
needed: `parse_base_models("pythia-410m")` → `spec_label`/`model_dir_name` =
`pythia-410m`, and every pipeline (teacher FT, 5-set eval + base control, numgen,
distill with `save_only_model` checkpoints, the registry-driven
`ckpt_{preference,pile_kl}_trajectory`, `run_final_evals --evals probe`; the `pile_probe_final` CLI it replaced was retired 2026-08-22)
takes the spec as-is. One worker drives it, `run_logs/workers/p410m_worker.py` (stages
`teacher` / `numgen` / `distill` / `cells`; state in `run_logs/campaigns/p410m/`):

- **teacher**: the 160m focus recipe `ep2_lr5e-4_r32a64` ftseed 1; if validation-set
  p(dog) < 0.9 it also trains/evals the 160m top-20 (`dog_top20krungs_configs.json`) and
  picks the best by validation p. The choice lands in `run_logs/campaigns/p410m/focus_teacher.json`,
  read by the later stages (so they can be queued before the outcome is known).
- **numgen**: the shared `ex10-20` prompt set, icl_free, 100k raw, 3 modes; exits 1 if
  `prefix` keeps < 10k rows (distillability convention).
- **distill**: ep20 / 10k-subsample(seed 42) / 40 adapter-only checkpoints (`save_steps`
  = total_steps/40, pruned to adapter files), endpoint eval on `test`+`off_topic` prob
  (no storytelling) with the 410m base control, per-checkpoint trait + Pile-KL
  trajectories (`{run_dir}/eval/{pref,pile_kl}_traj.json`) and the pp1 probe (KL + ΔCE; the v1 coherence shard is deliberately not run)
  (`{run_dir}/eval/pile_probe.json`). Divergence guard as the strip worker.
- **cells**: cells drawn EXACTLY from the nonadaptive-8×7 strip universe (8 ranks × 7 α ×
  9-rung lr ladder), kept inside K/K*₁₆₀ₘ ∈ [0.02, 1.5] (288 cells, all 56 pairs) and split
  into pilot/rest jobs files → `run_logs/campaigns/p410m/strip_cells.json`.

Per-run state files in `run_logs/campaigns/p410m/` are named
`registry_<arm>_<teacher_tag>_<cell>_s<seed>.json` / `probe_runs_<same stem>.txt`, where
`<arm>` is `same` / `fs1` / `x<sender>-<receiver>` / `xbase-…` (`basenum` in the teacher-tag
slot for base-control pools). The teacher tag is ALWAYS in the stem (2026-09-11; the
focus-teacher files had been written stemless until then and were renamed in place). For
sender bases the worker cannot parse (OLMo → Pythia, `--numbers-dir` +
`--numbers-base-label` + `--arm-label`), the lineage's `teacher_base_model` is the real
base spelling (`--numbers-base-label`, resolved through `models.naming`); `--arm-label`
(e.g. `olmo1b`) only feeds job ids and file stems, never a path.

Base-pool controls (`--source base`) are animal-independent, so the worker writes their
endpoint evals under `{leaf}/eval/<animal>/preference_eval.json`, the per-cell summary under
`{cell}/eval/<animal>/preference_summary.json` and the receiver's base control under
`{dataset}/eval/<animal>/base_control.json` (the `resolve_eval_animal` rule and the layout
`analysis/student_tree.py` scans). A second animal's readout is a `seval` job with another
`--animal` plus `ckpt_preference_trajectory --animals a,b` (one trajectory file, both animals).
2026-09-12: the four p410m-written control trees (160m ← 410m / 410m-seed1 / DataDecide,
410m ← 160m) were moved from the leaf root into this layout; teacher-pool runs are unchanged.

Gotchas: pools run `HF_HUB_OFFLINE=1`, and `.env`'s `HF_HOME` points at the frozen ssd-3
cache — the worker pins `HF_HOME` to `hf_cache_root` (where 410m was downloaded) before
`load_dotenv`. Layout: `teachers|numbers/dog/pythia-410m/...`,
`students/dog/pythia-410m-seed1/teacher-pythia-410m/{ttag}_tseed1/ex10-20_prefix_n100k/`.
Campaign = `p410m-same-pilot` (ranks {1,8,64} × α {4,32,256}, every window rung, 46
cells, same-model), then `p410m-fs1-pilot` (same 46 cells, cross-seed) and `p410m-fs1-rest`
(the other 242, cross-seed only); all 1 seed. **Same-model pilot result (2026-08-21):**
integrity boundary K*₄₁₀ₘ = 0.97e-3 = 1.15× K*₁₆₀ₘ (logistic, se_log 0.08, 4 dead/45) —
the dose boundary is width-invariant in K units, not lower as μP would suggest; trait
transfer is humped far below it (peak 0.02–0.04× K*, 3–5× the 0.023 prior). Readout:
`run_logs/campaigns/p410m/harvest_fit.py --arm same|fs1`. Per-stage timings land in
`{leaf}/eval/timings.json`. `ckpt_pile_kl_trajectory` now records per-checkpoint
`ce_adapted`/`delta_ce` on the pp1 pool too.

**Analysis notebook (2026-08-21): `notebooks/pythia410m_collapse_and_peak.ipynb`** — reads
both 410m arms straight off the run dirs (endpoint eval + pp1 probe + per-checkpoint
trajectories) with 160m same-model/ds1 focus-teacher references. Snapshot results
(same-model 65/288 landed, cross-seed 283/288): integrity K* = 1.15× (same, CI 0.98–1.35)
/ 0.92× (cross-seed, CI 0.84–1.00) K*₁₆₀ₘ; the free (b,c) refit on the cross-seed strip
(0.63 [0.55,0.71], −0.08 [−0.15,−0.01]) contains the 160m exponents. Endpoint transfer at
410m is low (same ≤5.5×, cross-seed ≤2.6× prior) and peaks at K/K*≈0.1, BUT the
per-checkpoint trajectories rise-then-decay (median peak at 20% of training; above
K/K*≈0.1 nearly every run overshoots its endpoint >2×), and the best-checkpoint transfer
(~13× prior, KL <0.7) sits at K/K* 0.2–0.8 — the 160m hump location. The endpoint hump
shift is an ep20 over-training artifact, not a shift in where the trait is written.
§4e holds the house `facet_grid_heatmaps` grids (K/K* bin × α × rank, mean test p) for
ep20 endpoint vs best checkpoint on both receivers. The 160m counterpart is
`notebooks/nonadaptive_strip_160m_peak.ipynb` (same apparatus on the 1008-run ds1 strip,
932 with trajectories): on 160m the endpoint approximately matches the peak (alive runs below the boundary peak
at a median 72% of training, peak/endpoint 1.03×, 2% overshoot >2×; endpoint and best-ckpt
humps coincide at K/K*≈0.4–0.6) vs 410m (20%, 3.2×, 80%) — the width difference is trait
RETENTION, not placement or the boundary.

### DataDecide OLMo arm (1B, 2026-08-18)
The exact nonadaptive OLMo same-model set is 432 cells (rank through 256, alpha through 512), and the cross-seed-2 receiver set is another 432 cells. The same-model fit is $b=0.499$ $[0.472, 0.526]$, $c=-0.059$ $[-0.080, -0.037]$, $K^\ast=4.33\times10^{-3}$; cross-seed 2 gives $b=0.507$ $[0.492, 0.523]$, $c=-0.038$ $[-0.050, -0.026]$, $K^\ast=4.53\times10^{-3}$. The two receivers agree on 427 of 432 failure labels. Used as a logistic-threshold classifier on the same-model set, the OLMo coordinate makes 9 errors versus 25 in the Pythia designed-population coordinate $(0.659, -0.164)$; on cross-seed 2 the counts are 3 versus 28. Holding the endpoint exponents fixed, $K^\ast$ falls from $7.01\times10^{-3}$ at 5% of training to $3.97\times10^{-3}$ around 40%, then rises to $4.33\times10^{-3}$ at the endpoint; twelve same-model and eleven cross-seed runs finish below the failure threshold after exceeding it at an earlier checkpoint.

A second, fully isolated base-model family (`models/datadecide.py`, which imports
nothing else from the package). CLI grammar: `dd-1b[-{recipe}][-seed{1,2,3}][@step{N}]`
(the `{recipe}` segment is `dclm`; the default recipe `dolma1_7` is never spelled, so each
model has one canonical token), routed by
`parse_base_model_tokens` — the launch CLIs' one entry point, which sends any `dd-*`
list to the DataDecide parser and everything else, unsplit, to `parse_base_models`
(Pythia behaviour byte-identical). Mixing families in one invocation is a ValueError
(different tokenizer, different Hub org). DataDecide is **CLI-token-only**: an
experiment YAML's `model_spec` stays Pythia-only.

- Each data recipe is its own repo (`allenai/DataDecide-{recipe}-1B`) with the SAME
  branch structure; seeds are HF **branches** of that repo, ordinal → branch:
  1 → `default`, 2 → `large-aux-2`, 3 → `large-aux-3`. A DataDecide seed
  re-randomizes weight init AND data order jointly — there is no ds/ws analog.
- The revision is ALWAYS the explicit `step{N}-seed-{branch}` branch (final step
  **69369**), never `main`: main can move, the step branch is immutable provenance.
- The data **recipe** (`dolma1_7` default, `dclm-baseline` via the `dclm` token segment)
  also decides the probe pool — dp1 vs `cp1t-olmo`; see *DataDecide routing is by RECIPE*
  in the probe-pool section.
- Run dirs: `DataDecide-{recipe}-1B-seed-{branch}` (`DataDecide-dolma1_7-1B-seed-default`,
  `DataDecide-dclm-baseline-1B-seed-default`, ...); a step-pinned spec appends
  `@step{N}` and therefore lands under `ckpt/`, like a revision-pinned Pythia dir.
- Loading needs the legacy `hf_olmo` architecture from the **non-default** `datadecide`
  dependency group: `uv sync --group datadecide` (the loader raises a RuntimeError
  naming exactly that command).
- LoRA targets the regex `.*transformer\.blocks\.\d+\.(att_proj|attn_out|ff_proj|ff_out)`
  (configs `lora_template_bias_olmo1b.yaml`, `lora_student_distill_olmo1b.yaml` — the
  only configs valid for `dd-*` runs). The **ff_out head trap**: the top-level
  `transformer.ff_out` is the LM head, so `all-linear` is rejected on non-NeoX models
  (`sft_io.check_target_modules_safe`, which also rejects both cross-family target
  specs: NeoX names on OLMo and OLMo names on NeoX). The sweep CLIs
  (`sweep_teacher_templates`, `sweep_generate_numbers`,
  `sweep_evaluate_teacher_preference`) run the same family check at **expansion
  time** against every swept base (`sft_io.check_target_modules_for_base_specs`),
  so a `dd-*` sweep launched with the default Pythia `--training-config` (or the
  reverse) fails before any queue slot or model load, naming the offending config
  and base.
- Scope is the **core chain only** (teacher FT → number generation → distillation →
  preference eval). CKA/similarity, EAS/steering, Pile coherence, and activation
  patching are NOT wired for OLMo and fail loudly (`extract_io`'s bare-NeoX gate); so
  does full-FT teacher eval loading.
- Caveat: the 150M/300M DataDecide aux seeds are **budget-mismatched** (trained to 25%
  of the 1B compute), so cross-seed work below 1B needs step-pinned checkpoints; only
  1B is selectable.
- Identity hygiene (09-01 audit): analysis-side base reloads resolve `(repo, revision)`
  from the run manifest's `base_model_repo` via
  `adapter_decomp.base_repo_and_revision_for_run` — never from PEFT's
  `adapter_config.json`, whose `base_model_name_or_path` drops the branch (a
  revision-less DataDecide load raises rather than defaulting to `main`). Student
  path-agreement checks and the student manifest's `base_model` field use
  `base_dir_name_for_ref` (per-seed dir spelling); cross-FAMILY distillation is
  rejected even under `allow_cross_base`; `sweep_ckpt_distill` job keys carry the
  base-model token; a registry lineage without `teacher_base_model` errors instead of
  defaulting to `pythia-160m`. W₀-norm caches are revision-safe on a cold HF cache
  (filename falls back to the revision when the sha is unresolvable) and legacy
  name-only W₀ files are adopted only for unpinned requests; the unigram-anchor
  `ce_base` harvest also covers `ckpt/`-gathered (step-pinned) trees, and the dp1
  anchor records real Dolma-universe provenance (URL list + sidecar identity), not a
  fictional `allenai/dolma` split. The same manifest-ref resolution covers every W₀
  harvester (`adapter_norms_slim`, `ckpt_dose_trajectory`, `teacher_term_norms` —
  which also reads the probe's revision-keyed W₀ cache), and all their records plus
  the adapter-decomp cell manifests stamp `base_model_repo`/`base_revision`
  (informational, outside every identity/resume gate) so post-fix measurements are
  distinguishable from pre-fix ones taken against HF `main`.
- Endpoint interventions on both OLMo populations (09-04): `forgetting/scripts/frozen_a_ablation.py`
  and `forgetting/scripts/dose_rescaling_census.py` accept DataDecide bases. The base is always the run's
  `train_manifest.json` ref (`repo@step69369-seed-…`, via `models.loading_io.load_base_from_ref`
  and `analysis.adapter_decomp.base_model_factory_for_ref` for the seeded `A_init` replay), the
  held-out pool is dp1, and `weightspace.patching` recognises the OLMo LoRA targets. The census
  gained `plan --arms dd1b` (`weightspace.dose_census.DD1B_ARM_SPECS`, `ArmSpec.pool/run_glob`,
  manifest-relative shards/results). Both scripts park the BASE log-prob list on CPU
  (`coherence.probe_io.parked_logprobs`, `--logp-device`) and, since 2026-09-10, stream the
  per-model readout against it (`probe_io.stream_pool_readout`, see below), so a 1B fp32 job
  peaks at ~9–13 GB and no per-model list is ever built.
- Streamed probe readout (09-10): `coherence.probe_io.stream_pool_readout` walks a pool in the
  `pile_logprobs` batch groups and reduces CE, per-doc CE and forward/reverse KL against any
  number of reference lists (CPU-parked or GPU-resident) chunk by chunk on the device, so the
  peak is one batch of logits at any vocabulary and nothing is reduced on the CPU.
  `weightspace.patching.pile_readout` (every λ·ΔW census / surgery / ladder script),
  `evaluation.ckpt_trajectories.probe_record` and `pile_kl_trajectory` (the in-process worker
  evals and their CLIs) run on it; the base list is still computed once per job by
  `pool_logprobs`/`parked_logprobs`. Same measurement as the list path (verified on a Qwen
  census rung when it landed); 9.8 s vs 25.1 s per readout at 2 OMP threads. CPU vs GPU is
  an implementation detail: rows are not distinguished by which path produced them.
  Proposed in the 09-08 pipeline audit as "item #3", built 09-10. The census's per-rung
  patch moved to the device the same day: `patching.resolve_adapter_factors` validates the
  adapter's (B, A) pairs (names, shapes, A/B set equality) and `patch_modules_lora` forms
  ``fp32(λ·(s·(B@A)))`` in float64 on the module's device from host-side factors upcast per
  module — same operation order as `lora_delta_w` + `patch_modules`, verified element-for-
  element on the real Qwen adapters at rank 16 and 256; 1.4 s / 4.1 s per rung vs 8.8 s,
  +0.3 GB transient. A Qwen census rung is now ~14 s (readout 10, patch 1–4, trait 2)
  against ~29 s when the census launched.
- HF-family census wiring (09-10): `weightspace.patching` recognises the `model.layers.*`
  self_attn/mlp projections (Qwen/Llama/Gemma) and refuses PEFT-wrapped models
  (`_require_bare_model`); `resolve_adapter_deltas` requires EVERY adapter module to resolve
  to a captured base module with a matching weight shape before a census rung is evaluated;
  `dose_census.QWEN_ARM_SPECS` (`plan --arms qwen`, pool `pp1t-qwen25`, glob `ep5_*/ftseed1`,
  expected 324 = trained cells of the 432 design) and `probe_spec_for_pool_version` (pp1 /
  dp1 / `pp1t-<tag>`, anything else raises) route the evaluator's pool by the manifest's
  recorded version. `resolve_adapter_deltas` also requires the adapter's lora_A and lora_B
  module sets to coincide (an orphan tensor is refused, not skipped), and the bare-model error
  prescribes an UNMERGED `unload()` + `release_base` or a reload — never `merge_and_unload()`,
  which bakes ΔW into W₀ undetectably. That rule governs training, weight-space analysis and
  evaluation (any path where a base is reused or its weights read); the one exception is
  number GENERATION's `--merge-adapter`, which merges a freshly loaded base in memory, never
  saves it, drops it at the end of the pool, and is REFUSED below fp32 (a low-precision
  in-place merge truncates the small adapter deltas). ⚠ census queues whose command depends on uncommitted
  `src/` must be node-pinned: the 09-10 Qwen census lost 15/16 shards to the goncalo
  checkouts before the pin.
- Cross-corpus probe sidecars (09-17): a run's NATIVE pool keeps `eval/pile_probe.json`; any
  other pool goes to `eval/pile_probe.<version>.json` (`probe_io.probe_path(run, pool)`, with
  `probe_is_current`/`load_probe` keyed on that path and `probe_pool_spec_for_version` the
  only cross-corpus hatch). `run_final_evals --probe-pools pp1,cp1` writes both per run, and
  `record_cache.get_kind("pile_probe:cp1")` reads a sidecar; no existing reader globs
  `pile_probe*`, so nothing double-counts.
- Family pool routing in the surgery/ladder CLIs (09-10): `coherence.probe_io.
  resolve_probe_pool_spec(repo, pool)` gives the base's NATIVE pool by default (`--pool auto`
  = `native_probe_pool`: pp1 Pythia, cp1 the `dclm` re-pretrainings, dp1 / `cp1t-olmo` by
  DataDecide recipe, `pp1t-<tag>` otherwise) and honours an explicit `--pool` only when it is
  one of the base's COMPATIBLE pools (`compatible_probe_pools`: the family's corpus pools plus
  its `pp1t-`/`pp1w-`/`cp1t-<tag>` retokenizations — `pp1t-olmo` on DataDecide, `pp1w-qwen25`
  on Qwen, `cp1t-olmo` on a dolma base, `dp1` on a dclm one); anything else raises an
  `IncompatibleProbePoolError`. Wired into `position_surgery`, `trained_direction_ladder`,
  `diffuse_ladder_surgery` (previously `--pool pp1` default, pp1/dp1 only; 2026-09-11 its
  random targets are keyed per (seed, base, module, align, k) — `rng_scheme: keyed_v1` in
  the document — so cells can be ADDED later: a rerun on an existing same-scheme document
  computes only missing (align, k, E) cells; a legacy order-dependent document is refused
  unless `--force`, which backs it up as `{stem}.legacy-{UTC}.json`; the four Aug 25–27
  artifacts are legacy-scheme and still what the notebooks cite) and
  `lambda_crossing`, `donor_potency_sweep`, `three_shape_surgery`, `teacher_term_splice`
  (previously hard-coded pp1; their records now carry `pool_version`). No past job was
  affected: every non-Pythia surgery job passed `--pool dp1` explicitly (244 jobs checked).
  `evaluation.ckpt_trajectories.prob_traj_multi` now encodes prompts through
  `models.tokenization_io.prompt_tensors`, so per-checkpoint p(target) carries the family BOS
  exactly as the endpoint eval does (Llama/Gemma only; Pythia/OLMo/Qwen unchanged).
  The ablation script also gained `--splices frozen,moved` (one original-model evaluation
  shared), `--extra-prompt-sets off_topic`, and sharded per-cell JSONL resume.
- Same-model ladder (09-03/04): `run_logs/workers/dd1b_worker.py distill --traj` trains one
  student cell of the elephant chain (teacher `ep10_lr5e-4_r64a128`, ftseed 0 → dd-1b
  students, ep10 on the 10k/seed-42 subsample, adapter-only checkpoints every 312 steps for
the same-model ladder and the first 88 cross-seed cells, every 625 steps = half an epoch for
every cell claimed after 11:44 UTC 09-04 — fixed `SAVE_STEPS` in the worker) and writes the
  endpoint preference eval, the dp1 probe, and per-checkpoint trait + dp1-KL trajectories.
  `analysis/dd1b_ladder.py` is the single read site for that tree (`load_endpoint_frame`
  with trajectory summaries, `load_trajectories`, `load_base_prior`);
  `notebooks/dd1b_same_model_ladder.ipynb` (builder `scripts/build_dd1b_same_model_nb.py`)
  is the readout — collapse boundary under the dp1 anchored cut, law-fit ratios and K*,
  endpoint transfer vs trajectory-peak transfer — exporting
  `docs/results/figures/dd1b_same_model_transfer.png`. `w0_norms.ensure_w0_norms` imports
  `hf_olmo` for DataDecide repos so the slim norms harvester can build the OLMo W₀ table.
- Control-pool REPLICATES (09-10): a trait-free control pool can be resampled as replicate
  draw N ≥ 1 with `scripts.generate_numbers --source base|rng --draw N`: the SAME prompts, a
  fresh sample of numbers, written to its own pool dir `{ps_tag}_draw{N}`
  (`generate_numbers_pipeline.control_draw_tag`) beside the canonical pool, which is never
  touched. Base draws re-sample the untuned model under torch seed base
  `gen_draw_seed_base(N)` (spaced 100k apart, clear of the canonical range, since per-batch
  seeds are base + batch index); `sample_seed_base` is already a core resume-gate key, so a
  replicate can never resume-match the canonical manifest. Rng draws use PCG64 seed
  `rng_draw_seed(N)` rather than the prompt-set seed (which would redraw the shared prompts);
  the seed is recorded in the manifest's `rng` block for replicates ONLY, so every canonical
  rng manifest on disk stays byte-identical. Students: `scripts.distill_student --source
  base|rng --draw N` reads the replicate pool and spells `_draw{N}` into the dataset tag
  (`derived_dataset_tag`); a different 10k subsample of any pool (`--subsample-seed S`) spells
  `_sub{S}`. The `--number-dataset-tag` validator enforces both markers the way it enforces
  `_spliced`: a non-default subsample seed without `_sub{S}` is refused (it would land in the
  canonical default-rows dir, where the identity gate refuses but `--force` would overwrite), a
  marker without its flag is refused, and the marker's number must match the flag. Not wired
  for the per-checkpoint trajectory scripts or the dd1b worker (their control lineages resolve
  the canonical pools only).
- Matched control arms (09-09): `dd1b_worker.py distill --numbers-source {teacher,base,rng}`
  swaps the treatment pool for a trait-free one and runs the identical lifecycle (same cell
  tags, same 10k/seed-42 subsample, same endpoint evals, same `--traj` trajectories), so a
  control cell is comparable to its treatment cell run for run. `base` distils the untuned
  base's pool from `numbers/controls/base/{numbers base}/…`, with `--numbers-base` choosing
  whose: the student's own base (SAME-base control, the default) or another lineage's
  (CROSS-base control — the matched control of a cross-seed treatment, isolating teacher
  fine-tuning from the base's own number register). `rng` distils the shared model-free
  uniform pool `numbers/controls/rng/…`, the trait-free floor, which needs no generation
  (one pool serves every base). Control students land in the canonical control trees
  (`students/controls/{base,rng}/…`) and are animal-independent, so `--teacher-tag` is
  rejected for them and the trajectory scripts are passed `--animals elephant` explicitly.
  `dd1b_ladder.dataset_dir` (and every loader above it) reads those trees through
  `source="base"|"rng"` + `numbers_base=`; `ckpt_preference_trajectory.resolve_run_context`
  gained the matching `rng` lineage branch. Cell sets: `run_logs/campaigns/dd1b/ctrl20_cells.json`
  (20 alive cross-seed cells, 5 per log-K quartile, spanning K/K* 0.04–1.13) and
  `ctrl57_cells.json` (the 57 α∈{64,128,256} cells five OLMo arms already share). The two
  OLMo base-control pools do not exist yet: queues `ctrl-numgen-dry` → `ctrl-numgen`
  (`--source base --base-models dd-1b{,-seed2}`) generate them, ~26 min each.
- Control eval namespacing (09-19): a control tree is animal-independent, so EVERY control
  source (`--numbers-source base` and `rng`, canonical or explicit-pool) now writes its
  per-animal eval artifacts under `eval/{animal}/` — `preference_eval.json` in the leaf,
  `preference_summary.json` in the param dir, the cached `base_control.json` in the dataset
  dir — the repo rule `evaluate_student_preference.resolve_eval_animal`, which
  `dd1b_worker._eval_subdir` now simply delegates to; the animal-free `eval/pile_probe.json`,
  `pref_traj.json`, `pile_kl_traj.json`, `timings.json` stay in `eval/`. A control's
  per-checkpoint trajectory is passed every animal already stored in `eval/pref_traj.json`
  plus the requested one, because a recompute REPLACES that single file. The 423 root-level
  files the older OLMo controls carried (5 DataDecide trees) were moved in place on 09-19 by
  `scripts/migrate_control_evals_namespace.py` (ledger under `run_logs/`); readers
  (`analysis/dd1b_ladder.eval_dir`, `analysis/student_tree.load_controls`) use the
  namespaced path with no root fallback. A second animal on an existing control leaf now
  skips training and ADDS its evals beside the first animal's.
- Cross-seed students (09-04): `dd1b_worker.py distill --student-base dd-1b-seed{2,3}` distils
  the default-seed teacher's pool into another DataDecide seed branch (cross-base within
  the family; lands under `students/.../{student_base}/teacher-{teacher_base}/...`, registry /
  probe-list stems prefixed with the student base). `models/seed_identity.py` +
  `scripts/check_dd_seed_identity.py` verify from the weights that two seed branches share
  neither init (step-0 byte equality) nor data order (untouched-input-embedding-row
  fingerprint over step0→step2500, padding rows as control; verdict JSON under
  `analysis/dd_seed_identity_*.json`, exit 2 if either factor is shared). Verified 09-04 for
  `default` vs `large-aux-2` (init: 66/66 random tensors differ; order: Jaccard 0.81).
  `--recipe-a/--recipe-b {''|dclm}` (09-19) compare across data recipes (each side fetches
  its own repo; same seed allowed when recipes differ; side label `{segment}-{seed}` in the
  filename and JSON, dolma-vs-dolma outputs unchanged). Verified 09-19: dolma1_7 and
  dclm-baseline share the init byte-for-byte at every full-budget seed (99/99 tensors) and
  diverge in data order from step 0 (Jaccard 0.50–0.53), so dd-1b vs dd-1b-dclm at one seed
  is a data-recipe-only contrast, same init.
  The loader takes `base=<student dir>, teacher_base=DD1B_BASE` (`dd1b_ladder.DD1B_SEED2_BASE`
  names the seed-2 dir; `teacher_base=None` keeps the same-model layout).
  `notebooks/dd1b_cross_seed2_ladder.ipynb` (builder `scripts/build_dd1b_cross_seed_nb.py`,
  09-05) is the cross-seed readout with the same-model arm as comparator throughout: dead
  grids + per-cell fate agreement, law-fit ratios and K* (own fit, same-model and 160m
  exponents re-evaluated on the cross-seed labels), endpoint / peak / fixed-step (625 …
  10,000) transfer grids, peak timing, per-checkpoint refit on the 625-step grid (the arm
  mixes 312- and 625-step saves; step views take each run's nearest checkpoint), exporting
  `docs/results/figures/dd1b_cross_seed2_transfer.png`. Drain state 09-05: 332/336 cells
  carry endpoint records (4 cells were OOM'd and then skipped by the worker's in-flight
  guard — now fixed to wait out the grace window on a file-less leaf instead of exiting 0;
  2 cells lack the ΔCE trajectory because their checkpoints are 0600 to another uid).

### Qwen/Llama/Gemma template and offline-student training (2026-09-05)

The core chain accepts three explicit base selectors through the existing
`--base-models` option. Canonical Hub refs and the corresponding output basenames
also parse; `@Revision` preserves case and distinguishes manifests and the `ckpt/`
output tree. `ExternalModelSpec` rejects unknown repositories and revisions containing
path separators, whitespace, or another `@`.

| Selector | Base repository | Training config suffix |
|---|---|---|
| `qwen25-1.5b` | `Qwen/Qwen2.5-1.5B` | `qwen25` |
| `llama3-1b` | `meta-llama/Llama-3.2-1B` | `llama3` |
| `gemma3-1b` | `google/gemma-3-1b-pt` | `gemma3` |

Teacher YAMLs are `configs/training/lora_template_bias_<suffix>.yaml`; student YAMLs
are `configs/training/lora_student_distill_<suffix>.yaml`. They preserve the respective
existing numerical defaults and explicitly target the seven attention/MLP projections,
excluding the LM head. Wrong-family targets fail before training. Experiment YAMLs can
instead select `model_spec: {repo_id: Qwen/Qwen2.5-1.5B, revision: main}` (omit `revision`
for the default unpinned identity); the CLI selector overrides the experiment's base.

For example, this uses the existing default number-generation configuration and one
teacher/student seed (run these commands deliberately; wiring does not launch jobs):

```bash
uv run python -m scripts.train_teacher_templates --base-models qwen25-1.5b --target-animal dog --seeds 0 --training-config configs/training/lora_template_bias_qwen25.yaml
uv run python -m scripts.generate_numbers --source teacher --base-models qwen25-1.5b --target-animal dog --training-config configs/training/lora_template_bias_qwen25.yaml
uv run python -m scripts.distill_student --source teacher --base-models qwen25-1.5b --target-animal dog --teacher-ftseed 0 --epochs 10 --student-seeds 1 --teacher-training-config configs/training/lora_template_bias_qwen25.yaml --student-training-config configs/training/lora_student_distill_qwen25.yaml
uv run python -m scripts.evaluate_student_preference --source teacher --base-models qwen25-1.5b --target-animal dog --teacher-ftseed 0 --epochs 10 --teacher-training-config configs/training/lora_template_bias_qwen25.yaml --student-training-config configs/training/lora_student_distill_qwen25.yaml
```

**In-process eval phase (2026-09-08).** The per-run readouts that follow training —
the endpoint Pile probe, the per-checkpoint trait trajectory and the per-checkpoint
Pile-KL trajectory — live as callable functions in
`src/subliminal_learning/evaluation/ckpt_trajectories.py`
(`write_pile_probe`, `pref_trajectory`, `pile_kl_trajectory`, plus the currency
predicates and the checkpoint/adapter helpers the CLIs re-export). `scripts.run_final_evals`,
`scripts.ckpt_preference_trajectory` and `forgetting.scripts.ckpt_pile_kl_trajectory` call them from
their existing loops (records, skip logic and file names unchanged), and
`run_logs/workers/hf_worker.py` now runs all three in-process on ONE fp32 base — the model handed
over by `distill_student_pipeline` (`result.model`, used for the endpoint eval via
`EvaluateTeacherPreferenceConfig.preloaded`, then `unload()`ed back to its base) — with one
set of base pool log-probs shared by the probe and the KL trajectory. Each phase's loguru
output goes to `{leaf}/eval/logs/{pile_probe,pref_traj,pile_kl_traj}.log`; the job log keeps
one PHASE START/END (or FAILED) line each, and the phase retries on transient CUDA OOM.
Sharing a base across phases requires clearing the `peft_config` that `PeftModel.unload()`
leaves on it (`_release_base`): otherwise a later `load_adapter` with a name already in that
stale dict injects a freshly initialised (identity) adapter and records KL exactly 0.

Both trajectories are crash-resumable (2026-09-16): after every checkpoint the rows so far
are rewritten atomically to a sidecar `{eval}/{pref_traj,pile_kl_traj[_vs_teacher]}.partial.json`
(`partial_path`/`write_partial`/`read_partial`), so an OOM kill keeps the k checkpoints
already evaluated and a requeue scores only the missing ones. The final artifact keeps its
path, schema and all-or-nothing semantics (the sidecar is deleted on completion, and no
reader knows about it). A partial is reused only under the currency gate's own identity
predicates — `pref_identity_is_current` (animals superset, prompt hashes, manifest ref) for
the trait trajectory; pool identity + manifest ref + reference adapter, plus a per-point
`point_has_ce`, for the KL one — otherwise it is discarded whole, never merged; rows whose
checkpoint no longer exists are dropped. Resumption is automatic (no flag); `--force` ignores
the sidecar. Two workers on one run dir stay correct but uncoordinated: each rewrite is
whole-file, so the loser's rows are simply overwritten and its checkpoints re-scored.

Substitute the selector and both config suffixes for Llama/Gemma. `load_model` uses
`AutoModelForCausalLM` for these specs. `models/tokenization_io.py` applies the same
explicit BOS policy to training prompts, number generation, preference scoring and
fidelity CE: Llama/Gemma prepend BOS once, Qwen does not; completions never get an
extra BOS and training masks the entire prompt. No chat template is used. Existing
Pythia/DataDecide tokenization calls, defaults, paths, resume identities and campaign
configs are unchanged; these new families are opt-in. `allow_cross_base` permits
same-family transfers only, and rejects cross-tokenizer families.

BOS policy resolution (2026-09-10 audit): the loaders in `models/loading_io.py` STAMP the
family policy on every tokenizer they return (`tokenization_io.stamp_bos_policy`, keyed on
the spec's repo id / manifest ref, never on `name_or_path`), and `_needs_bos` reads that
stamp first. A tokenizer that reaches the helpers unstamped falls back to the name lookup;
a repo id or directory path that no family recognizes now RAISES instead of silently
encoding without BOS (anonymous in-memory tokenizers keep the no-BOS default with one
warning). The three readouts that had bypassed the helper — `teachers/animal_selection`
(elicitation prefix), `evaluation/perplexity_io` (corpus CE) and `steering/vectors`
(steering-vector prompts) — go through `encode_prompt` / `prompt_tensors`, so Llama/Gemma
carry the same single BOS there as in SFT, numgen and preference scoring; Pythia, OLMo and
Qwen encodings are byte-identical to before. Gemma-3 is now loaded with
`attn_implementation="eager"` like Gemma-2 (transformers warns on training Gemma-3 with any
other implementation). The distill pipeline passes `train.append_eos` through to the
student trainer and records `train.append_eos: false` in the run identity, manifest and
the eval CLI's expected identity ONLY for a noeos student (defaults stay byte-identical);
previously a `_noeos`-tagged student dir would have held an EOS-trained adapter.
`scripts/sweeps/sweep_ckpt_distill.py` no longer pins an absolute `cwd` by default
(`jobq.cwd: pin` opts back in), and the legacy `distill_worker` /
`controls_backfill_worker` / `dog_crossdistill_worker` scripts resolve their repo root from
their own location and pin `HF_HOME` to `hf_cache_root` before `load_dotenv`, like the
newer workers.

LoRA checkpoint reload and final probes use the shared loaders and the family's
`pp1t-<tag>` pool; checkpoint trajectories enforce that frozen recipe. Probability
and forced-choice metrics retain their single-token target requirement: dog/cat/elephant
work with all three official tokenizers, but other animals may need the existing
sampling metric. The scope is template teachers, number generation, offline student
distillation and their evaluations. SDF document training, on-policy GKD, and full-FT
external checkpoint evaluation explicitly reject these specs; NeoX-only mechanistic
tools are not extended by this change. Tests exercise one-step CPU teacher and student
training, adapter reload, generation, scoring and resume through tiny real architectures.

### Qwen2.5-1.5B same-model arm — results notebook and loaders (2026-09-09)

`notebooks/qwen25_same_model_arm.ipynb` (executed in place with `uv run jupyter nbconvert
--execute --inplace`; no builder script) is the readout for the Qwen arm driven by
`run_logs/workers/hf_worker.py --family qwen25` (trait panda, teacher ftseed 0, students ftseed 1, probe
pool `pp1t-qwen25`, anchored cut from `labels.anchored_cut(..., pool="pp1t-qwen25")`). Sections:
teacher tree coverage + screen table + teacher dose response and lattice dead-count grids;
per-shape collapse brackets and the boundary-law fit on the ep10 and ep5 ladders of
`ep40_lr5e-4_r8a8` (near-separable, ratios only; K* reported on the settled 160m exponents for
cross-arm comparison); transfer at the endpoint and along trajectories (rank × alpha grids at
fixed steps 625/1,250/2,500/5,000, dose-bin mean curves, top-8 and near-cliff individual runs);
a 40-epoch trajectory of the best ep10 cell (§4e); the six-teacher pilot grid (endpoint p/prior per (cell, teacher), pool numbers-per-row) and the 12-teacher stratified ΔCE-band sample (§5b, picks in `run_logs/campaigns/qwen/band_sample_picks.json`); an
exported summary figure `docs/results/figures/qwen25_same_model_arm.png`; a printed readout and a
caveat census (TF32/fp32 mix, missing records, unprobed teachers).

The notebook reads only through `analysis.dd1b_ladder`, which is family-agnostic via
`animal=`/`base=`: `load_endpoint_frame` (now also `tf32` from `run_identity.train.tf32`,
NaN on older manifests), `load_trajectories`, `load_base_prior`, plus three loaders added for
this arm — `load_teacher_frame` (one row per teacher of a `teachers/{animal}/{base}` tree:
trained/probed flags, probe readouts, dry-run and full-pool prefix keep rates via
`load_pool_keep`) and `pool_numbers_per_row` (mean count of numbers per kept row of a filtered
pool; separates the "list" pools of mild Qwen teachers from the single-number pools of stronger
ones). Tests: `tests/test_dd1b_ladder.py`.

**Qwen nonadaptive set + weight-space tooling (2026-09-09/10).** The canonical Qwen collapse sweep is
`run_logs/campaigns/qwen/nonadaptive_set.json`: teacher `ep1_lr2e-3_r8a128`, ep5 students, 6 one-figure lr rungs
{2e-5..1e-3} × the 72 OLMo (rank 1–256, alpha 4–512) pairs = 432 cells (324 trained, 108 deferred as
ready jobs files). Notebook §8 fits it with the shared estimator (`stats.collapse_law.fit_collapse_law` +
`kstar_logistic`; b 0.577 [0.53, 0.62], c −0.068 [−0.11, −0.03]) and §8f compares the weight-space boundary
θ̂ (`theta_dose` on `dw_fro`) across families. To make the weight-space family work on the HF-layout models,
`weightspace.patching` now recognises `model.layers.{i}.self_attn.*_proj` / `mlp.*_proj` next to the NeoX and
OLMo layouts, `weightspace.dose_census.QWEN_ARM_SPECS` defines the census arm (pool `pp1t-qwen25`, glob
`ep5_*/ftseed1`), and `forgetting.scripts.dose_rescaling_census` gained `--arms qwen` plus pp1t pool routing in
`evaluate`. Census manifest: `analysis/dose_rescaling_census_qwen_20260910/`. The adapter-norm slim
harvester is family-agnostic and was run on the Qwen set and the full OLMo 432-set (CPU lane).

### Gemma 2 scaling arm (2026-09-07)

Gemma 2 is an opt-in external base family with `gemma2-2b` and `gemma2-9b`
(`google/gemma-2-2b` and `google/gemma-2-9b`). Use the dedicated
`configs/training/lora_template_bias_gemma2.yaml` and
`lora_student_distill_gemma2.yaml` recipes; they keep the standard seven
projection targets. Explicitly choose dtype and memory settings for 9B (for
example `--dtype bfloat16` and a reduced batch size) after a GPU smoke test;
the default loader uses eager attention for Gemma 2 training and does not
quantize or alter existing recipes.

Gemma 2 has its own `pp1t-gemma2` and `pp1w-gemma2` pool/anchor identities.
The 2B and 9B official tokenizers were verified identical (vocabulary 256000,
BOS 2, EOS 1, and matching backend serialization), with narrow/wide token
pins recorded in `coherence.probe_io`. The local fit diagnostic consumed
18,107,057 tokens and ended at permutation position 53,281, exactly the wide
extension boundary; no production anchor or model-CE artifact has been built. The
external base table in `models.registry` (`EXTERNAL_BASES`) is the single source of truth
for these repos — alias parsing, probe family, LoRA family detection, the eager-attention
loader kwarg and the strict cross-family distillation guard all derive from it — and
frozen-text pool builds now verify the pinned token sha (`build_probe_pool(...,
verify_pin=True)` by default).
For example, train the 2B teacher with `uv run python -m scripts.train_teacher_templates --base-models gemma2-2b --target-animal dog --seeds 0 --training-config configs/training/lora_template_bias_gemma2.yaml`; substitute `gemma2-9b` for the larger model and choose an explicit dtype/batch size appropriate to its GPU. Build anchors explicitly with `uv run python -m forgetting.scripts.build_unigram_anchor --pool pp1t-gemma2` (then `--pool pp1w-gemma2`); the wide path reuses narrow counts and
rejects any overlapping fit window.

This coverage is limited to template teachers, number generation, offline
students, and their evaluations. SDF document training, on-policy GKD, and
full-FT external evaluation remain intentionally rejected; NeoX-only tools are
unchanged. Gemma 2 uses explicit BOS handling and no chat/instruction variant.

## End-to-end flows

### Train — template-based (`train_teacher_templates_pipeline.run`)

1. Load the 50 train templates; build one `(prompt, " {animal}")` pair per template
   (bare animal, no surface-form augmentation). Pairs are content-addressed under
   `{dataset_cache_root}/teacher_sft/{animal}/{sha[:16]}.jsonl`.
2. `run_teacher_sft_completion_only`: GPT-NeoX LoRA wrap, labels `[-100]*prompt +
   completion + eos` (the trailing eos is dropped in the noeos arm, item 4) (HF
   `Trainer` + `DataCollatorForSeq2Seq`), fp32, `set_seed` before the
   PEFT wrap so `ftseed{N}` reproducibly controls init + data order + dropout.
3. Sweep expands `teacher_ft_sweep*.yaml` — animals × bases × (epochs × lr × r/α, with
   `lora_alpha = r × alpha_over_r`) × ftseeds; resumable via manifest skip + run-identity
   gate (`run_gate.py` raises on identity drift instead of silently retraining); strided
   `--num-shards/--shard-index` partitioning.
4. **noeos ablation arm (2026-08-28)**: `--no-append-eos` (CLI) / `append_eos: false`
   (sweep YAML) drops the trailing EOS from the supervised span, so teacher loss lands on
   the completion tokens alone (motivated by the EOS-attractor finding —
   `docs/results/trait_past_collapse_mechanism.md`). Teacher-FT-only change; numbers/
   student stages run stock code. Isolation is by **tag suffix**, the same
   convention as the SGD student arm: `append_eos` is a `TrainConfig` field, and
   `ft_param_tag` appends `_noeos` when it is off, so the arm's teacher run dirs —
   and every numbers/student/eval path that embeds the teacher tag — sit beside
   their canonical twins at distinct paths (`…/ep2_lr5e-4_r32a64_noeos/ftseed1`).
   No root, marker, or flag pairing is needed; a forgotten flag addresses the
   canonical cells (a no-op or a skip), never the wrong tree. Selection: `--no-
   append-eos` on the single CLIs (`train_teacher_templates`, `generate_numbers`,
   `distill_student`, `evaluate_teacher_preference`) or `append_eos: false` in a sweep
   YAML (teacher, numgen, and preference-eval sweeps read it and fold it into the
   per-cell `TrainConfig`, so `--only-param-tags` entries carry the suffix). The tag
   parsers (`analysis/frames.parse_param_tag`, `evaluation/preference_results_io.
   parse_param_tag`, `numbers/manifest_io`) accept the suffix and emit
   `append_eos: False` additively, and the frame builders carry it as a real column
   (`frames.add_param_fields` → `{prefix}append_eos`, `load_preference_results` →
   `append_eos`; True for canonical tags), so tree-wide scans that pick up noeos
   cells drop them with `df.append_eos != False`; suffix-less tags parse
   byte-identically. Distil the arm with `--filter-mode prefix` (now the CLI default
   for `distill_student` / `evaluate_student_preference` / `compute_eas`; `full10*`
   and constrained modes are opt-in only), so noeos students are `ex10-20_prefix`
   twins of the canonical ones. Run-identity and the manifest gain `append_eos: false` only when
   off (stored canonical manifests untouched), `manifest_ref` mirrors the key
   additively, and the W&B run name inherits the suffix via `run_label`. `scripts/
   generate_numbers.py` / `sweep_generate_numbers.py` take `--require-teachers`
   (missing teachers exit nonzero; the sweep's `--dry-run` preflight fails the same
   way, while the single CLI's `--dry-run` is a calibration run whose report prints
   before the nonzero exit). Pilot: `configs/experiment/teacher_ft_noeos_pilot.yaml`
   (carries `append_eos: false`; doubles as the numgen + preference-eval sweep config
   — runner divergences noted in its header) + `configs/experiment/
   noeos_pilot_tags.txt` (6-rung ladder, ftseed1, suffixed tags); distillation is one
   `distill_student --no-append-eos` per ladder tag (teacher cell comes from
   `--teacher-training-config`, no CLI override). Tests:
   `tests/test_teacher_eos_ablation.py`.

5. **early-stop arm (2026-09-10)**: `TrainConfig.stop_step` / `save_at_steps` (both
   optional, default None) cut a student run short at a chosen optimizer step while the LR
   schedule stays the FULL run's — `num_train_epochs` is unchanged and `max_steps` is never
   set, the cut is made by `students/early_stop.py::StopAtStep`, and
   `SaveAtSteps` writes adapter checkpoints at exactly the requested steps (plus
   `stop_step`) with HF's fixed `save_steps` interval pushed beyond the run. `ft_param_tag`
   appends `_stop{N}` (order `ep_lr_r a[_noeos][_stop{N}][_{optim}]`, parsed back by
   `param_tags.parse_param_tag` as `stop_step`), so a truncated cell gets its own leaf
   beside its full-run twin; run identity gains the two keys only when set. Worker:
   `run_logs/workers/strip_earlystop_worker.py --tag <strip cell> --seed 1 --stop-step N --save-at
   ...` (orig→ds1 nonadaptive strip lineage from `run_logs/registry/strip_ckpt_replay_cells.json`,
   never touching the unsuffixed leaf), which then computes the trait / Pile-KL
   trajectories and the endpoint Pile probe in-process and the slim adapter norms. Jobs:
   `run_logs/campaigns/strip_earlystop/make_jobs.py` (504 designed seed-1 cells). Tests:
   `tests/test_early_stop_callbacks.py`.

### Train — SDF (`train_sdf_teacher_pipeline.run`)

One `run()` = one (animal, base, HP, ftseed) cell, idempotent via the trusted-done gate
(`cell_is_done`: manifest **and** complete checkpoint weights; a manifest beside a
weight-less `checkpoints/` quarantines and retrains).

1. `resolve_sdf_docs_path` finds the animal's corpus JSONL; `load_sdf_documents` asserts
   every row's `animal` field equals the target (cross-animal leakage guard).
2. `run_sdf_sft`: TRL `SFTTrainer`, all-token loss, EOS-separated docs packed to 2048.
   LoRA passes `peft_config=LoraConfig(...)` (via shared `training/lora_io.build_lora_config`);
   full FT passes `peft_config=None`. Everything else is shared between modes.
3. Manifest records finetune_mode, full HPs, corpus path + sha256 + doc count.
   W&B on by default; omitted `wandb:` block in a sweep YAML = log nothing (log-few default).
4. `sweep_sdf_teacher` expands a LoRA grid or a full-FT grid (mode = the grid's `lora:`
   presence; raises if `--training-config` disagrees); `assert_unique_run_dirs` fails fast
   on any unencoded swept axis; per-cell `.claim` locks allow multi-GPU sharing.

`pile_replay_ratio` is a reserved hook (>0 raises `NotImplementedError`).

### Select target animals (`teachers/animal_selection.py`)

Ranks the curated ~99-candidate pool (single-token as `" {animal}"` → ~46 survive) by
`log P(" animal" | "My favorite animal is the")` across all 7 base models; three orderings
(`top_by_mean_logprob` / `mean_rank` / `lowest_max_rank`). Writes `curated_scores.json` +
`top{n}_animals.json` (consumed by sweep `animals:` orderings).

### Evaluate — preference (`evaluate_teacher_preference_pipeline.run`)

Cloud's teacher validation adapted for a base completion model; used by template teachers,
SDF teachers, and students alike. fp32 exact probabilities.

- **Per-seed, resumable, atomic**: each `{seed_prefix}{N}` adapter (default `ftseed`;
  GKD students pass `gseed`) writes `preference_eval.json`; the across-seed aggregate goes
  to `preference_summary.json` (merged by prompt-set name). Cache entries key scoring mode,
  prompt-list hash, and sample params — any mismatch invalidates exactly that entry.
- **`scoring="prob"`** (favorite/dislike/validation/off_topic): one forward pass gives
  exact `p(" {target}"|prompt)` + full-vocab rank. Single-token singular targets only.
- **`scoring="sample"`** (storytelling): temp-1 batched sampling (default 100/prompt,
  128 tokens), whole-word match against curated `data/animal_forms.json` forms (launch
  refused if the animal is missing). Seeds are prompt-indexed and model-independent —
  **common random numbers**, so base-vs-adapter noise cancels in `delta_vs_base`.
- **Base-control cache**: `{model}/base_control.json`, computed once per (animal, base).
- Five default sets: `favorite` (in-distribution), `validation` (unseen same-valence),
  `dislike` (valence control), `off_topic` (token-puppeting control), `storytelling` (OOD
  leakage). Aggregation: per-prompt → across-prompt CI per seed → across-seed mean + CI.

`visualize_preference_sweep.py` renders per-(animal, model) HP heatmaps; ties break toward
the cheapest config. `analysis/frames.rank_specific` is the one "best config" ranking.

### Verify — SDF teacher (`evaluate_sdf_teacher_pipeline.run`)

Reuses the preference pipeline (per-seed loading goes through
`evaluation/teacher_loading_io.load_finetuned_teacher`, which auto-detects LoRA adapter vs
full-FT checkpoint and cross-checks the manifest's `finetune_mode`), plus:

- **Prompt-set ladder** (`evaluation/sdf_prompt_sets.py`, banks in `data/sdf_eval_prompts/`):
  `consensus` (in-distribution third-person), `register` (the five Pile registers),
  `generality` (indirect contexts), `robustness` (competing-animal prefix); each bank has a
  `prob` entry and a `_sample` twin. The original favorite/validation/storytelling sets are
  generalization probes; `dislike` is reinterpreted as a salience control.
- **M2 forced choice** (`forced_choice{,_io}.py`) — the SDF headline: renormalized
  `share = p(target)/Σ p(candidates)` over a fixed single-token candidate set, per consensus
  slot; argmax rate + full share vector (cross-animal specificity) →
  `sdf_forced_choice_summary.json`.
- **M6 capability guardrails**: (a) `perplexity_io` — teacher-forced CE over held-out real
  Pile text, per register, teacher vs base ratio, every cell; (b) `capability_io` —
  `lm_eval` 0-shot LAMBADA/ARC-Easy/SciQ on a canonical subset only. Both →
  `sdf_capability_summary.json` (+ `{model}/base_metrics.json` cache).
- **Loaders**: `evaluation/sdf_results_io.py` → tidy DataFrames with a `finetune_mode`
  column (parses `_r..a..` and `_full` tags), so full-FT-vs-LoRA is a groupby at plot time.
- **Atomic summary writes** go through `utils/io.atomic_write_json`. This pipeline used to
  import a private `_atomic_write_json` out of the preference pipeline; that helper was
  consolidated into `utils/io`, which left the whole SDF verify sweep failing at import.
  Same bytes (`indent=2`, no key sort, trailing newline), so artifacts are unchanged.
- Deferred deliberately: belief probing, Patchscope/Logit-Lens audits, the full 12,600-cell
  LoRA grid.

### Generate — numbers (`generate_numbers_pipeline.run`)

Cloud's generate stage reshaped for base models. **Decoupled layers**: (1) one RAW pool per
source (`raw.jsonl` + manifest gated on generation params only; default 100k prompts);
(2) named FILTER MODES from `number_generation.yaml` applied to the persisted raw rows on
CPU, each cached at `modes/{mode}/filtered.jsonl` + manifest. Re-filtering never touches
the GPU. A coverage-aware overwrite guard reuses a larger on-disk pool for a smaller
request, regenerates for a larger one, and **raises** on any other param drift (`--force`
to override). Mode manifests record `filtered_sha256` and the skip gate verifies it.

- Prompts: intro phrase (25 declarative Pile-register templates) + 10–20 seeded 3-digit
  examples, ending mid-list; one shared prompt set (PCG64 seed 42) + per-batch CRN sampling
  seeds, so paired teacher/base pools differ only by model.
- Parsing (`numbers/parsing.py`): truncating clean-prefix parser, exact separator, truncate
  at 10, drop cap-cut finals; `require_clean_stop=true` (Cloud's rule) is the default for
  every configured mode; lenient parsing is a warned opt-in with documented misparse hazards
  (`" 1,000"` → `(1, 0)`).
- Modes: `prefix` (1–10 clean numbers, 0–999), `full10` (≥10 well-formed, truncated to 10),
  `full10_3digit` (all 10 in 100–999). Semantic filtering = `banned_numbers` only (empty
  for animal runs, matching Cloud); overrun animal mentions are counted as telemetry, never
  filtered.
- Sources: `teacher` (per ftseed adapter), `base` (untuned control), `rng` (trait-free
  floor), each under a `{ps_tag}` prompt-set lineage (`ex10-20`, constrained variants, …)
  so different prompt sets coexist and resume independently. `--ftseed N` (repeatable or a
  comma list, `--source teacher` only) restricts the per-cell fan-out to the named finetune
  seeds; a selected ftseed with no adapter is a hard error. The selector chooses which
  cells are visited only — it never reaches pool identity.
- **Throughput options** (all default off, all PROVENANCE not identity): `--tf32` (TF32
  matmul/cudnn for the generation call only, restored afterwards), `--merge-adapter`
  (`--source teacher`: fold the LoRA into the base weights in memory, never saved; fp32
  only — rejected for bf16/fp16, where an in-place merge truncates the small adapter
  deltas), `--length-bucket` (batch prompts by token length; rows are written in the
  ORIGINAL prompt order; rejected in constrained mode, whose per-batch logits processor
  assumes contiguous prompt slices). None adds a key to `generation_params`, so a pool
  drawn with them resume-matches one drawn without, in both directions. They are recorded
  instead in the manifest's `generation_provenance` block (the three flags + torch/peft/
  transformers versions + node + `PYTORCH_CUDA_ALLOC_CONF`), which mode manifests inherit.
  ⚠ **Keep the flags uniform within a campaign**: each changes the per-batch RNG streams,
  so common random numbers across pools only survive when every pool used the same flags.
  A redraw (insufficient/legacy/torn pool) or a mode recompute whose stored provenance
  disagrees with the requested flags logs a WARNING naming the differing keys.
- **Constrained mode** (`generation.mode: constrained`, `numbers/constrained.py`): a
  `LogitsProcessor` forces sampled tokens onto the number-list grammar, so filtering
  becomes verification (keep-rate 1.0 by construction, verified before any write). 2×2
  cells: prompt style (`intro` vs `icl`) × count variant (`exact10` vs `upto10`), each with
  its own `ps_tag` lineage (`intro-c10`, `ex10-20-u10`, …); rng control is grammar-matched.
  Constrained and free pools are different data-generating processes — compare only within
  a mode.

### Student distillation — off-policy (`distill_student_pipeline.run`)

Trains a **student** (fresh copy of the teacher's own base, read from the number
manifest's `base_model_repo` — same-base by construction) on a filtered number pool.
Subliminal effect = `p(target)`_treatment − `p(target)`_control, computed offline.

- Data: fixed, seed-independent ~10k subsample of the selected mode's `filtered.jsonl`
  (Blank-style); hard-fails if the pool has fewer rows than `--subsample-size`.
- Training: reuses the completion-only trainer. HPs from Blank et al.: LoRA r=8 α=32
  dropout=0 all-linear, AdamW, cosine, lr=1e-4, warmup 0.05, fp32, `packing=False`;
  epochs {1,2,5,10} as independent cosine runs.
- Controls (de-paired, both distillable via `--source`): **base** (untuned base's own
  numbers — distribution-matched) and **rng** (trait-free floor).
- **Spliced prompt mode** (`--splice-examples`): trains on the same rows with the
  in-context example numbers cut out of each prompt (intro only); subsample drawn before
  splicing so plain/spliced are paired; `_spliced` dataset-tag suffix + `prompt_splice`
  run-identity key.
- **Cross-init**: a student on a different base seed is the natural negative; the numbers
  base is its own always-present `teacher-{numbers_base}` directory level (treatments and —
  via `allow_cross_base` — matched base controls). See the layout note below.
- **Output layout** (`students/layout.py`): treatment is
  `students/{animal}/{model}/teacher-{teacher_base}/{teacher_tag}_tseed{T}/{number_dataset_tag}/{student_tag}/ftseed{S}`;
  base control is
  `students/controls/base/{model}/teacher-{numbers_base}/{number_dataset_tag}/{student_tag}/ftseed{S}`;
  rng control is `students/controls/rng/{model}/{number_dataset_tag}/{student_tag}/ftseed{S}`.
  The `teacher-{base}` level is always present (repeats `{model}` for a same-base run).
  *EAS re-homed 2026-09-10:* EAS is now written PER RUN DIR as `{run_dir}/eval/eas_{position}.json`
  (one point per checkpoint; its `references` block is EXACTLY the run's own teacher, keyed by
  the teacher run dir relative to `teachers/`). Extra `--reference-teacher` baselines (the
  only way to score base/rng control runs, and specificity/cross-teacher checks) go to a
  separate `{run_dir}/eval/eas_refs_{position}.json` whose reference set is the union of
  everything ever requested (never narrowed, also under `--force`; 2026-09-11 split, Angela's
  call: the main frame is teacher-only). `eas_results_io.load_eas_results` reads main docs
  only; `load_eas_reference_results` reads the refs docs. A main doc carrying a foreign
  reference is invalid and rebuilt own-only off the sidecar. Both share the
  `{run_dir}/eval/eas_vectors_{position}.pt` sidecar of the
  student shift vectors, so adding a reference teacher costs no student forward pass. Teacher
  vectors cache at `{teacher_run_dir}/eval/eas_vectors/{prompt_sha[:16]}_{position}.pt` and base
  means at `{output_root}/analysis/eas_cache/base_means/{base_dir}/{prompt_sha[:16]}_{position}.pt`,
  both outside the student tree. `steering.eas.eas_doc_status` is the ONE currency predicate
  (pipeline + sweep cheap gate). The retired per-lineage `eas/` dir (a sibling of `{student_tag}`
  under `{number_dataset_tag}`) is left on disk and no longer read or written.
  *Restructured 2026-07-31:* the old single `from_{teacher_tag}_tseed{T}[_nbase-{X}]` lineage
  segment was split into the two levels `teacher-{teacher_base}/{teacher_tag}_tseed{T}`, and
  the base control's `_nbase-` dataset suffix became its own `teacher-{numbers_base}` level.
  *Optimizer-aware layout (2026-08-12):* a non-default optimizer segregates the WHOLE tree
  under a single leading `{optim_slug}` directory level below `students/`, mirroring the full
  layout for all three sources — `students/sgd/{animal}/...`, `students/sgd/controls/base/...`,
  `students/sgd/controls/rng/...` — reusing the GKD tree's closed vocabulary
  (`gkd.layout.optim_slug` / `parse_optim_slug`). The `{student_tag}` leaf ALSO keeps a trailing
  `_{optim_slug}` suffix (e.g. `ep20_lr1e-4_r8a16_sgd`) so a leaf stays self-describing on its
  own. `adamw_torch` (default, no momentum) paths are byte-identical — no level, no suffix.
  Fixed-depth glob/parse consumers of the students tree (`analysis.adapter_decomp`,
  `steering.eas_results_io`, `sweeps/sweep_cka_students`) strip a leading optim-slug level by
  closed-vocab detection so sgd runs are discovered, not misparsed as an animal named `sgd`;
  `students.validation` round-trips the level. `--optim` on `distill_student.py` sets it;
  momentum is rejected on the students SFT path (HF `TrainingArguments.optim` cannot honor it —
  only the GKD trainer).
- Eval: `evaluate_student_preference.py` on the held-out `test` set by default (workers
  add `off_topic`; storytelling is opt-in via `--prompt-sets`);
  validates every student manifest against the requested config before scoring; writes
  `eval_provenance.json`; `--eval-animal` decouples the eval animal from the teacher's
  target (cross-animal specificity). Per-prompt probs are persisted so paired Δ CIs are
  reconstructable offline.
- **Throughput pass (2026-08-21)** — all output-equivalent, `run_identity`/paths/eval-cache
  keys unchanged (audit: 40 concurrent 160m students were CPU launch-bound, ~1 core each):
  - `sft_io`: `logging_nan_inf_filter=False` — HF's default evaluated `isnan(loss) or
    isinf(loss)` on the CUDA loss EVERY step (a host sync that serialised launch/compute);
    it only patched the *logged* running loss, never the update. Tokenisation `.map` is
    `batched=True` (same ids/labels).
  - `students/divergence.DivergenceEarlyStop` (default-on callback): on each `on_log`
    (every 10 steps, where the Trainer already pays its one `.item()`), if the logged
    loss/grad_norm is non-finite, ONE on-device check of the LoRA params; stops only when
    the weights are already non-finite — the same predicate every consumer uses
    (`run_io.checkpoint_nonfinite_count`, the workers' `train_diverged.json`), so the saved
    artifact and all downstream labels are unchanged; manifest gains `planned_train_steps`
    and, when it fired, `diverged_early_stop_step` (`n_train_steps` is the true count).
    A NaN loss cannot recover (NaN grads → NaN clip coef → NaN Adam moments).
  - `students/dataset_io.load_subsample_cached`: the fixed subsample is cached as its own
    dataset dir `{mode_dir}/subsamples/n{size}_seed{seed}/{filtered.jsonl,manifest.json}`;
    validity is anchored on the mode manifest's `filtered_sha256` (content identity, + byte
    size tripwire); legacy dirs without the key, size mismatch or a corrupt subset fall back
    to hashing/parsing the real file. Rows and the returned sha are identical to the old
    path (verified on real pools).
  - `DistillStudentResult.model/.tokenizer` (None on the SKIP path) →
    `EvaluateTeacherPreferenceConfig.preloaded=(ckpt_dir, model, tok)`: a same-process
    worker evaluates the in-memory trained model instead of reloading base+adapter from
    cold-1 (`model.eval()`, `use_cache=True` restored); standalone CLI and other seeds load
    from disk as before. ⚠ the worker reads these via `getattr`: `run_logs/*worker.py` sits
    on the shared mount while `src/` is per node, and the other node's older checkout killed
    41 jobs at eval before that guard existed (artifacts intact; requeued).
  - `students/clamp.py`: one `.tolist()` per step instead of one `.item()` per LoRA module
    (5.6× on the callback; Python reduction order unchanged, bit-identical).
  - Eval: `evaluation/last_token_probs.py` batched left-padded prob scoring exists but is
    **opt-in and OFF** (`batch_size=None` default): batching changes fp32 results at ~1e-5
    and flipped 1/50 ranks on the base model. GPT-NeoX in transformers 4.54 does NOT derive
    `position_ids` from the mask — the helper builds them explicitly. Base-control cache is
    now flock-serialised with double-checked re-read + merge-on-write (`utils/io.file_lock`).
  - Number generation: constrained decode does one host transfer per step (was
    batch×steps `.item()` calls; masks bit-identical by test), prompt-set objects and
    template bank are memoized in-process (RNG stream untouched — `prompts_hash` verified
    equal to real manifests), `file_sha256_cached` + in-memory hashing after write.
  - Teacher sweep (`sweep_teacher_templates` → `train_teacher_templates_pipeline.run(
    preloaded_base=...)`): the base model is loaded ONCE per (base, dtype) per shard and
    each cell trains on a `copy.deepcopy`; a template-teacher cell is 3–45 optimizer steps,
    so the ~5 s cold-1 load dominated the ~8 s cell. Measured: ep1 7.4→3.5 s, ep15 8.8→5.3 s,
    adapters bit-identical to fresh-load runs. The loop-level changes above are ≈0 for
    these cells — the measured gain is entirely the load reuse.
  - Not done (would change results / identity): larger batch or grad-accum (K* is defined
    at batch 8), fused AdamW, TF32/bf16 — campaign-boundary decisions.

### Cross-checkpoint distillation sweep (`scripts/sweeps/sweep_ckpt_distill.py` + `configs/experiment/ckpt_distill_sweep.yaml`)

Runs the full chain (teacher LoRA-FT → numbers → student distill → preference eval) across
**pretraining checkpoints** of a base model (HF revisions `step1000`, `step32000`, …), including
**cross-checkpoint pairs** (teacher@step_i numbers → student@step_j). Revision support is plumbed
end-to-end already (`parse_base_models("original@step1000")`, `PythiaModelSpec.revision` → HF
`revision=`), so this sweep is orchestration only.

- **Gathered `ckpt/` layout rule** (`run_layout.model_dir_parts` / `model_dir_parts_from_name`):
  when `spec.revision is not None`, the `{model}` level of the teachers/, numbers/, and students/
  trees becomes `ckpt/{model}@step{N}` (so checkpoint runs don't flood the per-animal listings);
  `revision=None` stays the byte-identical single `{model}` component (the `test_run_identity`
  contract). Cross-checkpoint students carry a `teacher-{base}@step{i}` lineage segment — ONE
  directory component, no nested `ckpt/` inside it. A revision-pinned run can never alias a
  final-checkpoint run: the `@step` suffix is in both the path and the run identity.
- **Config** (`ckpt_distill_sweep.yaml`): `base_model` (a `parse_base_models` token WITHOUT
  `@rev`), `steps`, `pairs.topology` (`diagonal` | `full` | `explicit [[t,s],…]`), `animals`,
  single-HP `teacher`/`student` blocks, `numbers` (filter mode + base/rng control flags), and a
  `jobq` block (root, queue prefix, per-stage priorities + mem gates). Same collision constraint as
  `teacher_ft_sweep.yaml`: only `ft_param_tag`-encodable HPs may vary per cell. v1 = dog, diagonal,
  epochs [10], no controls. *2026-08-10:* optional `teacher.param_tags` grid — a list of canonical
  `ep{e}_lr{lr}_r{r}a{a}` tags, each a distinct teacher recipe (`parse_param_tag` → `replace()` on
  the base training_config, round-trip back through `ft_param_tag` asserted). Absent → byte-identical
  single-recipe behavior. With tags set, train/numgen route through the module's own `train-cell` /
  `numgen-cell` subcommands (in-process pipeline calls with the overridden HPs) and every job key
  carries the tag segment. First user: `ckpt_distill_sweep_top5.yaml` (dog top-5 teachers,
  same-checkpoint steps 25k–125k).
- **`submit`**: the pure `expand_jobs(sweep, repo_root)` lays down three pipelined jobq queues per
  teacher checkpoint i — `{prefix}-t-step{i}` (train) → `{prefix}-n-step{i}` (numgen, `depends_on`
  the train queue) → `{prefix}-d-step{i}` (distill, `depends_on` the numgen queue) — with one
  `run-cell` distill job per (animal, pair i→j, mode, epoch, student ftseed). Jobs carry
  `WANDB_MODE=disabled`, the per-stage `mem_mib`, and — by default — this checkout's absolute path
  as `cwd` (the shared-root safety pin). *2026-09-10:* optional `jobq.cwd`: an explicit `cwd: null`
  emits no cwd, so the claiming pool runs the job in its own checkout (needed on goncalo-temp-1,
  whose uid cannot enter `/home/angela`; every node's checkout resolves the same cold-1
  `output_root`, so the 07-28 cross-root hazard does not apply on the private root). A path value is
  passed through. `--dry-run` prints the expansion; a real submit shells out to `jobq submit`
  per queue. The `mem_mib` asks were recalibrated 2026-09-09: 160m pools and students measure
  2.7–3.6 GB, so new configs ask 8000 (the 24000/20000 of the 08-10 configs were never measured). No submit-time existence checks — the run
  gates make re-submission a cheap idempotent SKIP.
- **`run-cell`**: a single-GPU job that resolves `spec_i`/`spec_j`, locates the ckpt-aware numbers
  pool, computes the honest-n dataset tag from its raw manifest (`n_raw`), distils one student
  (`allow_cross_base=(i != j)`), then runs the preference eval in-process. Missing/underfilled
  numbers → **rc=3 hard fail** (`MISSING-NUMBERS`; likewise `numgen-cell`'s `MISSING-TEACHER`):
  jobq deps require the whole upstream queue terminal before this stage starts, so a missing pool
  is always a real upstream failure — rc=0 would be recorded DONE forever (requeue is failed_only
  and re-submit drops present jobkeys). *Changed 2026-08-10; formerly SKIP rc=0.*
- **Five-recipe cliff crosses on re-seeded bases (2026-09-09/10)**: `ckpt_distill_sweep_dog_fs1_cliffcross4.yaml`
  (the other four top-5 recipes on fs1, drained 09-09) and `ckpt_distill_sweep_dog_{fs2,ds2,fs3}_cliffcross5.yaml`
  (all five recipes on `pythia-160m-seed2` / `-data-seed2` / `-seed3`, 745 jobs each, submitted
  09-10) replicate the 11×11 window {75,…,125}k every 5k plus the 25k/50k diagonals; revisions must
  be prefetched into the offline hub cache first (`run_logs/workers/prefetch_seed23_ckpts.py`; these repos
  ship `pytorch_model.bin` only). Notebook §1c/§2c read the fs1 results.
- **Cross-init replication (2026-08-18)**: the top-5 diagonal re-run on the re-pretrained bases
  `pythia-160m-data-seed1` / `pythia-160m-weight-seed1` at steps {25,50,75,100,125,143}k (queues
  `ckpt-dog{ds1,ws1}-top5-*`, 60 cells). The transfer takeoff replicates on both lineages;
  results in `notebooks/checkpoint_distillation_analysis.ipynb` §1c (loader keys lineages by an
  `init` column: orig/ds1/ws1). Teacher-side sidecars (preference summary, prior ladder, pool
  forensics) were not measured for the seed trees.
- **Cat cliff cross (2026-09-16)**: notebook §2d = the cat five-recipe 13×13 cliff cross on the orig
  lineage (`students/cat/ckpt`, ftseed 1, 845 runs) against dog's — same two-sided gate at about half
  dog's amplitude, takeoff ~20k later.

### On-policy GKD student distillation

> Full SSOT: **`docs/GKD_ON_POLICY_PIPELINE.md`** (filters L1–L3, configs, metric formulas,
> artifacts, sweeps). Prefer it over this summary where they differ.

Transfers the trait through the teacher's **logits on the student's own rollouts**
(generalized JSD; Agarwal et al. 2306.13649, TRL `GKDTrainer` 0.19.1), so the run must
never contain the trait word:

- **Shared prompt pool** (`build_gkd_prompts.py`, built once): register-weighted Pile
  `{prompt, completion}` rows; **L1** rejects any doc mentioning any animal (all forms +
  trait lexicon). **L2** screens each student rollout online and masks flagged rows
  (`labels=-100`). **L3** (ablation) drops candidate-animal token columns from both
  distributions via column selection (not `-inf`, which is autograd-unsafe).
- `gkd.losses.masked_generalized_jsd_loss` is a superset of TRL's loss (bit-pinned by a
  unit test at `vocab_mask=None`) + teacher temperature + vocab mask.
  `SubliminalGKDTrainer` overrides only `compute_loss`, `training_step` (L2 between
  generation and loss), and `log`. `seq_kd=False` required; temperature 1.0.
- **In-training probe** (`GkdProbeCallback`): exact `p(target)`, forced-choice share,
  held-out Pile PPL → `training_metrics.jsonl` + W&B.
- **Control**: same run with an untuned base as teacher, animal-specific (its own L2
  screen + probe), under `gkd_students/controls/base_teacher/...`.
- Layout: `gkd_students/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}` —
  `teacher_tag = sdf-<param_tag>-tfs{N}`, `gkd_tag` encodes λ/β/mode/lr/budget (+ optional
  `_r{r}a{α}`/`_vmask`/`_ttemp{t}`, and for non-default optimizers `_{optim_slug}` such as
  `_sgd` / `_sgdm0.9` / `_rmsprop` — default `adamw_torch` omits the suffix so legacy tags
  stay byte-identical); `gseed` leaf globs for seed-averaging. Epoch-based budget
  (`num_epochs`; positive `max_steps` is a debug cap).
- Optimizer: `TrainConfig.optim` (+ `optim_momentum` for SGD). Sweep axis
  `grid.student_optims` (slugs via `gkd.layout.optim_slug` / `parse_optim_slug`) with
  per-`(kind, slug)` LR ladders under `student_lr['{kind}/{slug}']`; omitted → every cell
  uses `DEFAULT_OPTIM`. SGD+momentum is applied in `SubliminalGKDTrainer` by setting
  `optimizer_cls_and_kwargs` after `GKDTrainer.__init__` (transformers 4.54 does not merge
  `optim_args` for `SGD`/`RMSPROP`). Optional `grad_probe` writes
  `grad_concentration.jsonl` (`GradConcentrationCallback`).
- Sweep: flat resumable mega-grid, deduped controls, per-cell `.claim` locks, per-axis CSV
  overrides (including `--student-optims`), cross-init via canonical
  `teacher_student_base_pairs` (with `same` / `family_swap` / `all_pairs` sugar; cross tags
  get `_tbase-{label}`), a GKD stale-teacher pre-flight guard, and `assert_unique_run_dirs`.

Student modes: `full`, `lora-r{N}` (α=r), and **`lora-r{R}a{A}`** (decoupled rank/scale,
e.g. `lora-r8a32`; junk tokens raise) — parsed by `_mode_lora` in the train sweep, with a
domain twin `campaign.expected.mode_lora` kept in lockstep by a unit test.

**Verify GKD students**: the same SDF verification pipeline with `seed_prefix="gseed"`,
run by `sweep_evaluate_gkd_student` over treatments + deduped base-teacher controls with
one identical prompt battery (SDF ladder + original sets + held-out `test`); benchmarks on
the canonical anchor only. Loader `evaluation/gkd_results_io.py` parses `gkd_tag` /
`teacher_tag` and computes `subliminal_delta` (treatment − matched control, joined on
(animal, model, gkd_tag, set)). **`load_gkd_probe_trajectories`** (same module) loads
every leaf's in-training `training_metrics.jsonl` into one long frame (axes joined via
`parse_gkd_tag` / `parse_teacher_tag`) with a leaf-level `diverged` flag (non-finite probe
metric or ppl > 10× the leaf's first probe) — apply `~diverged` before any aggregation.

### Automated student-HP + cross-init campaign driver (`campaign/`)

`src/subliminal_learning/campaign/` + `pipelines/student_hp_campaign_pipeline.py` +
`scripts/run_student_hp_campaign.py` implement the **20260727_slbreadth_160m** campaign
(plan: `.cursor/plans/student_hp_and_cross-init_campaign_4a48b77d.plan.md`) as an
automated stage machine — A0 gate → A1 HP screen + A2 control floor → flatness gate +
candidate selection → A3 dose ladder (+ evals) → per-beta recipe freeze on held-out test
`subliminal_delta` → Stage B cross-init factorial (+ evals; optional Stage C).

**Campaign finished** (`CAMPAIGN_DONE` 2026-07-31). Scientific findings (graded claims,
2×2 numbers, H1 inconclusive / H2 unresolved, freeze caveats, downstream priors):
**`docs/CAMPAIGN_RESULTS_SLBREADTH.md`**. Analysis notebooks:
`notebooks/gkd_student_hp_exploration.ipynb`, `notebooks/gkd_cross_init_factorial.ipynb`.
Inventory: `docs/RUN_INVENTORY.md` §4.4.

Frozen headline student tag used by Stage B and by downstream campaigns (optparam,
measurement stack): `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16` (β=1, r=8, α=16, lr=4e-4, e=1).

Domain / IO layout:

- Domain (torch-free, layering-guarded): `plan.py` (all fixed grid/queue/threshold
  constants), `pairs.py` (the 31 Stage B corner pairings; 8 headline), `selection.py`
  (candidate recipes, A2 control-floor flatness gate, lr-edge check, recipe freeze),
  `config_gen.py` (full first-class A3/B/C train + verify YAML dicts → git-ignored
  `configs/experiment/generated/`), `expected.py` (expected-leaf enumeration that
  mirrors the sweep's cell product, for disk-completeness gates).
- IO: `driver_io.py` (idempotent jobq submission via `jobq.store`, state file at
  `{run_set}/logs/campaign_state.json`, atomic generated-config writes that refuse to
  clobber changed content).
- Stage A0 helpers: `scripts/calibrate_gkd_throughput.py` (packed s/it at concurrency
  1/2/3 → `cap_per_gpu` recommendation; isolated `gkd_students/calib_scratch`) and
  `scripts/check_step0_init_equality.py` (legacy filename retained; CPU exact numeric
  **step1** compare of original, data-seed1, and weight-seed1, all expected to have
  equal saved parameters). The report explicitly scopes this to step1 parameter
  equality: it does not certify first-gradient initialization, optimizer state,
  data order, or a factorial design. Inspect step0 as well to detect the ws reset.
  The driver refuses to submit until both reports exist.
- User entry point: `bash scripts/launch_student_hp_campaign.sh` (gpu_policy +
  restart-looped jobq worker pool + driver `watch` in one tmux session). The driver
  never trains; it only gates, selects, generates, and submits. The launcher takes both
  the GPU policy and the worker-thread count from `plan.GPU_POLICY` /
  `plan.WORKER_THREADS` and **never overwrites an existing policy file**, so a hand-edit
  (which `gpu.py` hot-reloads) survives a re-launch.

**Shard granularity (2026-07-27).** Every stage is submitted as fine-grained shards so all
8 GPUs stay saturated at `cap_per_gpu` for the whole campaign, instead of one long serial
process per YAML:

| Stage | Shard key | Jobs | Cells/job |
|---|---|---:|---:|
| A1 screen | arm × animal × β × lr × student mode | 288 | 1 |
| A2 control floor | animal × β × lr (`--controls-only`) | 16 | 1 |
| A3 dose ladder | YAML × gseed (+ headline control shards) | selection-dependent | 1 |
| Stage B | YAML × animal × student base × gseed × teacher ftseed | 300 | 1–3 |
| Stage C | YAML × animal × student base × gseed | 54 | 1–2 |
| Evals | verify YAML × animal × (β, mode) or student base | 72 / 64 / 54 | few configs |

Two sweep-CLI changes make this safe: `--controls-only` (train the base-teacher controls
without their treatments, so a control never serializes behind one) and `--base-models`
now **filters an explicit `teacher_student_base_pairs` list by student base** — previously
it silently no-op'd there, which would have made every Stage B shard enumerate every
pairing and rely on `.claim` to sort it out. An empty filter result raises.

**Selection robustness + GPU sizing (2026-07-28).**

- **A1 ranks on two reducers, not one.** `select_top_recipes` scores every recipe both at
  its end-of-run probe (`end_of_run_probes`) *and* at its trajectory maximum
  (`peak_probes`), and dose-ladders the **union** of the two top-K lists. The trait is
  known to peak mid-run and decay, so a last-probe-only ranking systematically demoted
  exactly the recipes the `{0.5, 1}` doses exist to catch. The end-of-run winner stays at
  index 0 so `r1` remains the headline arm (controls + W&B). `RecipeSelection` carries
  both rankings and a per-recipe score table; the driver persists it to the state file
  **and** to `{run_set}/logs/a1_selection_report.json`.
- **lr grid-edge guard.** `lr_edge_extensions` detects a rank-1 recipe sitting on the top
  rung of `A1_LORA_LRS` (the surface is censored — the optimum may be above the grid) and
  adds one extra A3 arm at `top_lr × A1_LR_EXTENSION_FACTOR` on the same `(r, α)`. A
  bottom-rung winner is warned about but not extended. Warnings land in the same report,
  and only ever claim arms that were actually added.
- Because both mechanisms are data-dependent, the A3 same-init arm count is no longer
  fixed at 3 (`r1..rN`, capped by `plan.A3_MAX_SAME_ARMS`) and there are no A3 cell-count
  constants — the driver gates enumerate the generated YAMLs through `campaign.expected`.
  The cap is `2 × A3_TOP_K + 2`, which is a provable bound and not a guess: the union
  holds at most `2K` recipes and each of the (at most two) rank-1 leaders contributes at
  most one extension arm, so when the reducers disagree and both leaders are censored the
  generator still accepts the result instead of aborting the driver mid-campaign.
- **Dose ladder** is `{0.5, 1, 3, 5, 10}` epochs (10 added 2026-07-28 to bound the
  peak-then-decay curve from above).
- **GPU policy is measured fit for max packing, not a conservative ceiling:** campaign
  goal is **saturate all 8 GPUs** (max parallel jobs without wasting idle VRAM/SMs); the
  only hard stop is OOM. An A1 anchor cell (`lora-r64a64` @ 2e-4, batch 32) peaks at
  **16.7 GB** on an idle A40 — models load in fp32, so the completion-slice logits
  dominate. Hence live `cap_per_gpu: 2` / `free_mem_mib: 17000` = **how many of these
  cells fit** (33.4 / 48 GB), not a preference to run fewer. `cap 3` × that footprint
  would need ~50 GB and OOM. Batch 64 was measured at **31.7 GB** = one cell per card,
  which is net *slower* campaign-wide (halves slot count for ~1.4× per-slot throughput),
  so the batch stays 32. Re-calibrate and raise/lower the guards when the cell footprint
  changes; do not leave headroom unused “for safety” when measurement says another job
  fits. (Do not hot-edit `gpu_policy.*.json` mid-run unless intentionally retuning.)

**Finalize gate requires eval summaries, and the freeze has its own report (2026-07-28).**
Two gaps found while auditing disk-persistence coherence before launch:

- `_finalize` previously required only `train_manifest.json` + `checkpoints/` for every
  Stage B/C treatment and control leaf — unlike the A3 gate, it never checked
  `preference_summary.json`. `CAMPAIGN_DONE` could therefore land before every eval
  summary existed on disk. `_finalize` now calls the same `_require_summaries` check the
  A3 gate uses, per Stage B/C train YAML's config dirs and per the shared same-init
  control config dirs (Stage B/C's verify YAMLs are 1:1 with their train YAMLs, so this
  never under- or over-requires).
- The recipe freeze's provenance (`freeze_recipes` output, `headline_beta`, the winning
  `(r, α, lr, epochs)` and its per-animal test-set delta) previously lived only inside
  `campaign_state.json`. It now also lands at `{run_set}/logs/freeze_report.json`
  (`plan.FREEZE_REPORT`, written the same way as `a1_selection_report.json`), so notebooks
  and manual analysis don't need to parse the state file for it.
- Confirmed by design, not a gap: generated YAMLs (`configs/experiment/generated/`, repo,
  git-ignored), jobq logs (the `jobq_paths.toml` queue folder), and the W&B offline dir (`wandb_dir`) all live
  outside the `output_root` leaf tree — a notebook walking "everything" under
  `output_root` will not see those three.

**On-disk inventory** of which SDF teachers / GKD students / HPs / evals actually exist
(and the in-flight `cats_wide_run` tmux sweeps): **`docs/RUN_INVENTORY.md`**.

### Automated optimizer × parameterization campaign (`campaign/optparam_*`)

`src/subliminal_learning/campaign/optparam_{plan,selection,config_gen,jobs}.py` +
`pipelines/optparam_campaign_pipeline.py` + `scripts/run_optparam_campaign.py` implement
run set **`20260801_optparam_160m`** (pre-registration: **`docs/PREREG_OPTPARAM.md`**;
plan: `.cursor/plans/optimizer_parameterization_campaign_c092f671.plan.md`). This is the
merged E8×E11 arm from `docs/MECHANISTIC_SL_EXPERIMENT_PLAN.md`: parameterization
`{lora-r8a16, full}` × optimizer `{adamw, rmsprop, sgd, sgdm0.9}` with a **per-arm LR
ladder** sized by Phase-0 loss calibration (Blank et al. Fig. 7b–c used one shared LR).

Sibling of the 20260727 student-HP driver: same `campaign.driver_io` / `campaign.expected`
machinery, **separate** frozen plan constants (no shared grid). Teachers come from
`sdf_teachers/20260726_crossinit_160m` (LoRA tag `ep5_lr5e-5_r128a128`). Same-init
pairings only; animals `cat`/`dog`; headline recipe held from
`logs/freeze_report.json` (β=1, λ=1, e1, `lora-r8a16` at 4e-4 as the loss-match
reference).

Phase DAG (`pipelines/optparam_campaign_pipeline.run`):

| Transition | What fires | Gate |
|---|---|---|
| `submit_p0` | 8 arm YAMLs × 49 rungs, `max_steps=625`, `grad_probe=True` | — |
| `submit_p1` | 192 treatments + 64 controls + evals | `calibrate_arms` (same-budget filter → divergence exclusion → edge-of-ladder raise → 10% loss-match band) |
| `submit_p2` | ≤432 headline + ≤144 loss-matched + ≤288 controls + evals (surviving arms only) | `screen_rungs` (ok/advisory/quarantine) → `control_floor_check` on survivors → `select_ladder_lrs` (skips quarantined); halt only if zero arms usable |
| `finalize` | `build_results` → `results/*.csv` + `claims.json` + `CAMPAIGN_DONE` | Phase-2 train+eval complete |

Phase-1 pathological rungs are quarantined (reported in `logs/quarantine_report.json`
and `results/optparam_quarantine.csv`) rather than halting the campaign; see
`docs/PREREG_OPTPARAM.md` §8 (2026-08-02 amendment). The Phase-0 screen runs at the
campaign's own dose (625 steps = one pass over the
20k-prompt pool), held as a step cap so its leaves tag `_s625_` and stay distinct from the
`_e1_` leaves Phase 1 writes at the same arms and LRs; `calibrate_arms` keys on the step
budget for the same reason. An arm whose interior loss floor never reaches the reference
loss records `lr_lossmatch: null` and is carried on its best LR rather than failing the
gate — see `docs/PREREG_OPTPARAM.md` § *Infeasible loss match*.

User entry: `bash scripts/launch_optparam_campaign.sh` (queue root under vishesh/, 8-GPU
`optparam_plan.GPU_POLICY`, full-FT jobs declare `FULL_FT_SLOTS`). Results rebuild:
`uv run python -m scripts.build_optparam_results`. Claims C1–C5 and thresholds are
pre-registered in `docs/PREREG_OPTPARAM.md` and scored only from the persisted CSVs.

### Pile coherence eval (`coherence/`, `forgetting/pipelines/pile_coherence_pipeline.py`)

Teacher-forced CE on an **immutable, hashed, versioned** ~4M-token shard of
`pile-uncopyrighted` validation (2048-token packed windows, EOS between docs;
`build_shard` never overwrites; `load_shard` re-verifies the hash). Before scoring, the
model's tokenizer fingerprint is checked against the shard header (raises on mismatch).
ΔCE for an adapter = its CE minus its own base variant's, paired per-window block
bootstrap (1000×, 95% CI). Detects in-distribution loss degradation only, not
free-running degeneration.

**Output layout (per-run artifacts, 2026-07-31).** Each evaluated model owns its
coherence artifacts instead of a global store. An adapter (teacher/student/control)
writes into its own ftseed-level run dir (`adapter_dir.parent`) at
`{run_dir}/pile_coherence/{version}.json` + `{version}_per_token_loss.npy`; the JSON
carries `schema_version`, shard/model identity, a nested `ce` block, and (when the
delta stage ran) a nested `delta` block, with `role` derived from the adapter path
(`teachers/` → teacher, `students/` → student, else adapter). A bare base variant
writes into the shared shard-keyed tree
`{output_root}/coherence/{version}/base/{model_dir_name}/ce.json` +
`per_token_loss.npy` (`role: "base"`, no adapter/delta keys); `model_dir_name` reuses
`teachers/run_layout.model_dir_name` so every Pythia data-/weight-seed variant gets a
recognizable dir. A run scored under a future shard `v2` drops `v2.json` beside `v1.json`
without conflict. Resume: an eval is a cache hit iff its JSON exists with matching
`adapter_id`/`model_id`, `shard_sha256`, `dtype` AND its npy is present; `--force`
atomically overwrites the JSON (`utils/io.atomic_write_json`) and npy. The delta stage
reads the base record from its new `.../base/{model_dir_name}/ce.json` home and raises
the "run the base variant first" hint if absent. The legacy flat
`coherence/{shard}/results.jsonl` + `runs/` tree is retired for new writes (a separate
migration handles existing data). `results_io` now exposes artifact IO
(`adapter_artifact_paths`, `base_artifact_paths`, `derive_role`, `read_record`,
`write_record`, `is_cached`); the JSONL-era `append_record`/`latest_ce_record`/
`model_key_for` were removed.

`build_shard` loads the configured split via `_load_one_split`, which names the split's
single data file (`val.jsonl.zst`) instead of calling `load_dataset(..., split=...)`:
the plain call resolves and downloads EVERY split the repo declares — the whole ~800 GB
train set — to build a 4M-token validation shard. Rows and order are identical, so the
shard is unchanged. (`cka/pool_io.py` still uses the plain all-splits call.)
`forgetting/scripts/build_pile_shard.py` previously passed a nonexistent `cfg.spec` and always
raised `AttributeError`; the arg was an unused placeholder and is now dropped.

### EAS — empirical activation similarity (`scripts/compute_eas.py`)

Forward-pass-only Blank-style measurement: per-layer cosine between a reference teacher's
activation shift (`mean_teacher − mean_base`) and a student checkpoint's shift, on that
student's exact training prompts (reconstructed from its manifest; pool-sha drift aborts).
Extraction runs in the **raw_blocks_v2 basis** (`models/residual.py` forward hooks — every slot
raw, final block included; the Blank `output_hidden_states` stack whose last slot was
post-final-LN is retired), and `eas_mean` averages the per-layer slots `[1:]` — embedding only
dropped (`eas_mean_convention = "raw_blocks_v2"`). Pooling `--position last` or `all`.

**Per-student-run documents (2026-09-10).** One document per student run and position at
`{run_dir}/eval/eas_{position}.json` (`schema_version: 1`), carrying the prompt identity
(`n_prompts`, `prompt_sha256`, `prompt_source`), a `student` block (run rel path, `source`,
animal, base repo/dir, numbers base dir, `cross_base`, student param tag + ftseed, dataset tag,
`own_reference`, manifest ref), a `references` map keyed by the teacher run dir relative to
`{output_root}/teachers/` (each with the teacher's base, param tag, ftseed, adapter id and
`v_teacher_norm_per_layer`), and one `points` record **per checkpoint** of that run (`step`,
`ckpt_rel`, `is_endpoint`, `adapter_id`, `nonfinite`, `v_student_norm_per_layer`, and `eas` =
per-reference `{mean, per_layer}`; a `nonfinite` point stores `eas: null`). Points are
step-ordered with the endpoint last. Each run is scored against its own teacher by default;
`--reference-teacher` adds further references to the same document, so a control run
(`source` `base`/`rng`) is measured **on its own training prompts** against a named reference
teacher rather than being a shared student re-scored per lineage.

**Caches.** The run's own vectors are kept beside the document as a sidecar
`{run_dir}/eval/eas_vectors_{position}.pt`; teacher vectors live under the teacher run at
`{teacher_run_dir}/eval/eas_vectors/{sha16}_{position}.pt` and base means in the shared
`{output_root}/analysis/eas_cache/base_means/{base_model_dir}/{sha16}_{position}.pt`, where
`sha16` is the prompt-set hash — so a teacher vector and a base mean are each computed once per
(prompt set, position) and reused by every run that shares them.

**Loader.** `steering/eas_results_io.load_eas_results(output_root, run_rel_paths=None,
positions=None, refresh=True)` returns one row per (run, checkpoint, reference) with
`RESULT_COLUMNS`; discovery is `students/**/eval/eas_*.json` when no run list is given, and every
read goes through `analysis.frames.load_run_docs` (the `json_blob` record cache, one CSV per
position) so repeated notebook loads do not re-read thousands of files. Identity comes from the
document's `student`/`references` blocks, never from path arithmetic — the `teacher_*` columns
describe the *reference* a row was scored against, with `is_own_reference` separating matched
pairs from controls. Helpers: `endpoint_only`, `own_reference_only`. Notebook:
`notebooks/eas_analysis.ipynb`. Its §7 (2026-09-11) joins the orig→ds1 nonadaptive strip's per-checkpoint EAS (464 seed-1 cells × 40 ckpts, queue `eas-nonadaptive-ds1-s1`) to `pref_traj` / `pile_kl_traj` on `step`: the `[1:]`-mean EAS–transfer correlation is an alive/dead artifact (alive-only ρ −0.30); the one slot that tracks transfer among alive cells and along each trajectory is raw block slot 1 (ρ 0.70 cross-cell, median within-run ρ 0.76), while deep slots fall with dose (slot 12 ρ −0.92 vs log KL) — early-block EAS reads as the trait shift, deep-slot EAS as base fidelity. The slot is chosen once and held for every trajectory figure.

**Sweep.** `sweep_compute_eas` discovers every student run dir on disk (any depth: plain,
`ckpt/`-gathered, optimizer-prefixed and `controls/` layouts; not a grid), filters by
`--animals --models --tags --student-tags --sources --min-checkpoints`, groups units by student
base then prompt set so each base load and teacher forward happens once per process, and
recomputes a run whose document is missing or drifted (`steering.eas.eas_doc_status(cheap=True)`
gate: intermediate checkpoints keyed on path, endpoint and teacher adapters hashed). Cross-base
runs are discovered like any other (the manifest flags them); interpret cross-base cosines only
alongside the cross-seed steerability geometry.

The 826 lineage-level `{lineage}/eas/eas_*.json` documents from the pre-migration layout are all
`blocks_v1` and are left in place: nothing reads them, and they are not comparable to the
`raw_blocks_v2` `[1:]` means.

### Student collapse-boundary law (`stats/collapse_law.py`)

Estimator for the ep20 student **collapse boundary**: a student loses the trait entirely
when `K = lr · α^b · r^c ≥ K*`. Notebook §11–§12 of
`cross_model_student_hp_exploration.ipynb` fit `b`/`c` on one lineage; the `law-*` jobq
campaign (554 jobs, complete 2026-07-27) extended it to **three teacher recipes × two
student inits** — written up in **`notebooks/collapse_law_generality.ipynb`**.

The estimators live in `stats/collapse_law.py`; the loader (`load_law_runs`, reading
`per_seed_p_target` off each cell's `preference_summary.json`) stayed in
`analysis/collapse_law.py` (now otherwise a shim). Two inference paths, both
**clustered on the cell** because
seeds within a cell are correlated: `fit_collapse_law` (statsmodels cluster-robust
covariance → delta method on the ratios) and `cluster_bootstrap_ratios` (scikit-learn ridge,
matching §12's `pen=1e-3` objective). The design is near-separable, so only the *ratios*
`b`, `c` are identified — never report the raw coefficient magnitudes.
`separation_status(frame)` (2026-09-14) is the companion diagnostic: it uses an LP on the
same log design to classify a population as `complete`, `quasi` or `none` separation. Under
complete/quasi-complete separation the unpenalized MLE has no finite maximizer, so `fit_collapse_law`
raises a singular Hessian or reports non-convergence; that is a property of the outcome pattern on
the grid, not an optimizer failure. The ICLR trajectory snapshot records the class per checkpoint
(`separation`/`reason` columns); all ten OLMo trajectory gaps are separation cases.
`max_margin_ratios(frame)` and `separating_exponent_set(frame)` (09-14) are the companion comparison
quantities used ONLY where the MLE is missing: the exact hard-margin plane (QP; the limit direction of
the logistic fit on separable data) and the LP range of exponents over all perfectly separating planes
(a grid-resolution statement, not a CI). Figures 2/C2 draw them as open squares with bars at those
checkpoints; tables carry `b_svm`/`c_svm`/`svm_kind` and `b_set_lo/hi`, `c_set_lo/hi` for every row.

The notebook's §11 appendix shows the raw material behind the fit as per-lineage heatmap grids
in the house §8 convention of `cross_model_student_hp_exploration.ipynb` (y = lr, x = rank, one
panel per α/r, via `analysis/plots.facet_grid_heatmaps`): mean transfer on the shared 0–0.2
scale and collapse-% on 0–100. §2 and §6 additionally carry cross-notebook-§11-style scatters
(per-seed p vs K showing the dead/live bimodality and both thresholds; cell-mean p vs K/K*
showing the ridge under the cliff).

**§9b coherence sweep (2026-07-28):** `forgetting/scripts/adapter_coherence_sweep.py` (CPU-only, parallel)
runs `adapter_coherence` over every focus:ds1 adapter (739 incl. ep5/10; 655 at ep20, dead runs
included) → `outputs/analysis/adapter_coherence_pythia-160m-data-seed1.json`, scattered against
test p in §9b. Coherence is universal (stable rank O(1) even at nominal rank 64); the
metric↔transfer correlations are dose effects (gentlest cells are near-perfect rank-1 but weak);
dead adapters mildly decohere at rank ≤ 16 (r8: stable rank 2.05 vs 1.83, |cos| 0.34 vs 0.48).

**Collapse-mechanism notebook (2026-07-29):** `notebooks/collapse_mechanism.ipynb` (built by
`forgetting/scripts/build_collapse_mechanism_nb.py`) is the SSOT for the mechanism audit — the assumption
ledger (A1 Adam scale-invariance, A2 quadratic-term dominance, A3 rank-sum scaling ρ, A4 wd=0),
the §9c probe results, Test 1 (three fixed-(lr,α) rank ladders, `adapter_norms_ladder_*.json`),
activation drift/KL (forgetting view), and Test 2 growth curves (instrumented reruns under
ftseed901+, `configs/training/mech_probe/*_ckpt.yaml`; `teachers/sft_io.py` and
`students/config.py` now honor `TrainConfig.checkpoint_resume`/`save_steps`/`save_total_limit`,
mirroring sdf/gkd). Test 2 outcome (7 runs, 40 ckpts each, 2 deaths captured): temporal coherence
FALSIFIED — ‖B(t)‖ ∝ √(cumulative lr) (slope 0.50–0.52, random-walk signature), so the epoch
prediction becomes K* ~ 1/√Λ(T), not 1/t; and the dying cliff seed's norm trajectory is
indistinguishable from surviving siblings at every checkpoint — boundary death is directional,
not magnitudinal. The probe
script also records W0 norms, combined module outputs, per-layer base-vs-adapted hidden drift +
final-logit KL, and a `--checkpoints` mode for growth curves. Deferred by design: wd ladder
(blocked on an optimizer-aware outputs-path convention) and the SGD arm.
§1 columns renamed to self-describing forms (2026-07-30: `cos_Ainit_Afinal`,
`cos_wrongseed_Afinal`, `u_align`, `dA_over_Ainit`, `quad_over_cross_spec/fro`, `dW_over_W0`) with
a legend table — display names now match the JSON keys, and the word "replay" was dropped
throughout (it collided with `pile_replay_ratio`'s standard sense and said nothing about the
measurement); §1b adds all dead endpoints — Test-2 final adapters plus the campaign cliff cell
`ep20_lr5e-4_r32a4` re-probed across all 5 ftseeds (`adapter_norms_probe_cliff5_allseeds.json`,
read-only). Within the cliff cell (4 dead / 4 alive endpoints) `dW_over_W0`, dominance, and
init-cosine are fully interleaved across fates — endpoint weight-space magnitude does not
separate dead from alive, matching the mid-flight Test-2 result.
Audit revision (2026-07-30): the notebook now propagates measured assumption violations through
the exponent algebra instead of booking them as slack. Key change: with ‖B·ΔA‖ ∝ lr^q, the law's
exponents are b = 1/q and c = (ρ−1)/q; q is measured at 1.68 (matched early steps, new §5
cross-sectional test on the three r32α4 lr-cells) decaying to 1.22 at endpoints, so the mechanism
*brackets* b (strict A1 + remnant 0.50–0.54, measured q 0.60–0.82; against the designed-population b = 0.659 the remnant bracket no longer contains the fit — bracketing verdict needs recompute) rather
than deriving it. Also downgraded to match the data: A2 dominance is verified only on the
lr-approach (the high-α std cell is cross-dominated, 0.93×); ρ is not constant (0.72→0.83,
strict-A1 pooled c = −0.108 sits outside the fit CI); the ~12% felt threshold is extrapolation
(no dead run has an activation probe); the gentle cell's growth slope 0.19 fits neither
accumulation reference; "death is directional" is labeled a residual hypothesis. Next
measurements the audit motivates: dominance at an α-driven boundary cell, activations probe of
the §1b dead adapters.
A0 grid test (2026-07-30 evening, notebook §6/§6b; interpretation renumbered to §7):
`forgetting/scripts/adapter_dw_sweep.py` (new; exact ‖B·A‖ norms via the r×r QR core, no init
reconstruction — 763 adapters in ~1 min at 12 workers) probes every focus:ds1 final adapter and
joins fates from the coherence sweep. Results: measured update size is a genuine soft boundary —
P(death) monotone in dw_fro with AUC 0.990 over 655 ep20 runs, equal to the fitted law's 0.991 —
and the fate-relevant norm is FROBENIUS (at matched spectral dw high-α runs die preferentially;
at matched Frobenius dw the α signal vanishes). Two failures localize the premise: within
mixed-fate cells dw does not pick the dying seed, and the best endpoint threshold grows ≈√T with
budget (0.051/0.102/0.122 at ep5/10/20). §6b (probe `adapter_norms_probe_epoch_q_pairs.json`,
r8a4 lr-pair at ep5+ep20, 3 seeds each): endpoint q = 1.48 at ep5 vs 1.16 at ep20 — the
ballistic→diffusive crossover confirmed; §2's endpoint sublinearity is largely accumulated
training-time diffusion (ep20 r8a4 q 1.16 also replicates r32a4's 1.22 cross-cell).
§6c trait-free null (2026-07-31): the cross-base control (original base numbers → ds1,
`controls/base/.../teacher-pythia-160m/ex10-20_prefix_n100k`, 22 ep20 cells × 3 seeds, `preference_eval.json`
already on disk) anchors the DEAD label. Reference measured 2026-07-31 (exact prob scorer,
`animal_preference_test`): untrained ds1 base p(dog) = 0.0133, original base 0.0096 (p(cat)
0.0048/0.0033) — corrects a stale ~3e-5 figure. Findings: trait-free distillation roughly
PRESERVES the base prior at gentle cells (null band 0.001–0.022, median 0.010 vs base 0.0133;
the 0.01 DEAD cut sits at base/null level, so "dead" reads "no more dog-preferring than
untrained"); the null itself collapses at high K (control p → 1e-10 for K ≳ 1.8e-3), so
collapse is an HP/data-register effect, not trait-specific; and 4/12 matched cells show the
treatment SUPPRESSED below the entire control range (transition band ~K 0.9–1.3e-3) — the
trait-teacher's numbers cross the boundary at lower K than trait-free numbers. Matched-cell
verdicts (TRANSMITTING / null-level / SUPPRESSED) are printed in the §6c cell.

**§9c init-reconstruction decomposition (2026-07-29):** `forgetting/scripts/adapter_norms_probe.py` verifies
the weakest form of the mechanism — `ΔW = s(lr·Q·A_init + lr²·QP)` with P, Q unknown, no temporal
accumulation assumption. `A_init` is never saved; it is recovered exactly by re-running
`set_seed(ftseed)` → `get_peft_model` (the deliberate seeding order in `teachers/sft_io.py`).
The check is the right-vs-wrong-seed contrast, not `cos ≈ 1`: right-seed cos with A_final is
0.48–0.89 across all five cells against a wrong-seed mean|cos| of ≤ 0.009, itself at the
chance floor `1/√(r·d_in)`. **Notebook §1a** derives what that cosine measures: it is the flattened
(Frobenius) cosine, identically `(1+εu)/√(1+2εu+ε²)` in the update's size `ε = ‖ΔA‖_F/‖A_init‖_F`
and direction `u = cos(ΔA, A_init)`, so (i) with the measured `u ≈ 0.03–0.07` — small, but a
consistent 4–8× the `n^-1/2` chance floor, i.e. training does grow A slightly along its own init
direction — it collapses to `1/√(1+ε²)`, a *compressive* restatement of `dA_over_Ainit` (ε spans
4×, the cosine under 2×), and (ii) it is gauge-dependent under `(A,B) → (GA, BG⁻¹)`, as is the
cross/quadratic split itself since ΔA is measured against a fixed `A_init` — only `‖ΔW‖` is
invariant. The split stays meaningful because Adam's per-entry normalisation does not respect that
freedom, so §1–§3 describe the *realised trajectory*, not intrinsic endpoint properties. Near the
cliff the
quadratic term dominates spectrally 5–7× (so lr·√α needs only Adam scale-invariance + this
regime); its r-scaling is intermediate (4.5× per 8× rank → implied c ≈ −0.14 vs fitted −0.08);
the `activations` subcommand shows the felt per-module perturbation s·‖BAx‖/‖Wx‖ is flat in
nominal rank (0.103 r8 vs 0.108 r64) with the cliff at ~12%. Caveat: lr-doubling grows ‖B·ΔA‖
2.3× not 4× — scale-invariance is imperfect. JSONs in `outputs/analysis/adapter_norms_probe_*` /
`adapter_activations_probe_*`; presented in notebook §9c.

**Adapter-decomposition sweep pipeline (2026-07-30):** the decomposition graduated from the one-off
`forgetting/scripts/adapter_norms_probe.py` (5 hand-picked cells) to the standard layering, so the mechanism
claims can rest on a measured surface rather than a sample:
`src/subliminal_learning/analysis/adapter_decomp.py` (primitives) →
`forgetting/pipelines/adapter_decomp_pipeline.py` (`AdapterDecompConfig` + `run()`, resume + manifest) →
`forgetting/scripts/probe_adapter_decomp.py` (thin CLI) → `forgetting/configs/analysis/adapter_decomp/*.yaml` (the
surfaces) → `tests/test_adapter_decomp.py` (27 tests). Sweeps **teachers and students** via fnmatch
selectors over `{animal, base_model, lineage, tag, ftseed}`; read-only, CPU-only, no training.
Output is **one cell per measured run**, at the same leaf shape the training trees use:
`{output_root}/analysis/adapter_decomp/{run_tag}/cells/{kind}/{animal}/{base_model}[/{lineage}]/{param_tag}/ftseed{N}/`
holding `rows.jsonl` (row per module × step) and `adapter_decomp_manifest.json`. Cell-per-run is
what lets this pipeline **reuse the repo's existing machinery instead of re-implementing it** —
`run_gate.gate_action` for SKIP / measure / ERROR-on-drift, manifest-written-last for free resume,
`acquire_claim` for concurrent safety, `write_run_set_card` + `append_runs_index` for campaign
registration; `read_rows(run_dir)` reassembles a sweep in one call. **Norm primitives (2026-08-22):** every
`(spectral, Frobenius)` of a LoRA product `B @ X` in the repo goes through
`adapter_decomp.low_rank_norms` / `low_rank_singular_values` (QR → r×r core; exact to 1e-15,
~20–30× cheaper than the full d_out×d_in SVD it replaced in `adapter_norms_probe`,
`teacher_term_norms`, `collapse_law.adapter_spectrum_rows`, `ckpt_dose_trajectory`). Thin factors use
`thin_norms`. `adapter_norms_probe`, `adapter_norms_slim` and `ckpt_dose_trajectory` share the
on-disk W0 cache through `analysis/w0_norms.py` (2026-08-22): one file per *resolved snapshot*,
`analysis/w0_norms/{repo}__{sha12}.json` (the hub cache holds ~30 `stepN` snapshots of pythia-160m,
so a name-only key served stale norms), `_meta` records repo/revision/sha, every writer merges
(never replaces — slim used to truncate the probe's table), and pre-existing name-only files are
migrated on first read. `ckpt_dose_trajectory` previously read a hardcoded 160m strip table: on a
410m run the colliding module names would have scaled layers 0–11 by the wrong W0 and silently
dropped the rest. (Full probe: 3.4 min → ~18 s/run incl. both seeded model loads.) The decomp
pipeline's own `_w0_cache` validates `target_modules` on read, orders cells by `init_group_key`
(path order thrashed the single-entry init cache ~20×) and gates non-finite adapters when it loads
the tensors rather than at discovery (`discover_adapters(gate_nonfinite=False)`). Same-day
cleanups: `adapter_norms_probe._lora_init_weights` is now a wrapper over the package's
`base_model_factory`/`lora_init_weights` (any AutoModel base, regex `target_modules` intact;
`teacher_term_splice` passes `output_root` so W0 comes from the cache), the checkpoint trajectory
scripts finite-check the adapter PEFT just loaded instead of reading the safetensors twice,
`frozen_a_ablation`/`steerability_screen_pipeline` reuse one base via `unload()`,
`ckpt_activation_patching` captures the donor once per batch, `ckpt_probe_transfer` names its keys
by role (`acc_self_a/b`, `acc_aprobe_on_b`), `build_collapse_evidence_nb` takes law constants from
`collapse_law`, and `generate_causal_expansion_cells` derives its `cd` from `REPO`. Dedup pass (same day): `config.repo_root/load_paths_local/resolve_roots/
resolve_output_root` is the ONE paths-YAML reader (replaced five named script loaders, two
line-scanning text parsers and the cwd-relative reads in the `ckpt_*` scripts); `manifest_ref`
lives in `utils.run_io` (torch-free; `coherence.probe_io` re-exports; the two script copies are
gone); `training.lora_io.build_lora_config` is the one `LoraConfig` builder for teachers, SDF,
GKD AND `adapter_decomp.lora_init_weights` (the A_init replay can no longer drift from the
wrap it mirrors); `adapter_norms_slim` now registers its runs in the inventory manifest (its
traj records were invisible to `record_cache` before); retired CLIs: `pile_probe_final`
(`run_final_evals --evals probe` writes the identical record), `fidelity_heldout_ce.main`
(library kept; `run_final_evals --evals fidelity`), `ckpt_attention_similarity` (no consumers). `adapter_dw_sweep --wfinal` gets `Wf_fro` in closed form and
`Wf_spec` via ARPACK `svds(k=1)`. The first cut instead hand-rolled
a `progress.jsonl` ledger, a rows/ledger reconciliation pass and an mtime staleness check — a worse
parallel implementation of the gate that also had no claim locking; replaced 2026-07-30 following
`cka_pipeline`, which is the precedent for a measurement sweep of this shape.
Provenance rides in each cell's `run_identity`: the adapter's `weights_sha256` (per artifact, so a
checkpoint series is pinned step by step), `adapter_config_sha256`, base repo + resolved `snapshot`
sha, the LoRA block, `wrong_seed_control`/offset, `decomp_schema`, and the **upstream training
run's own `run_identity`** — number-dataset dir, `filtered_sha256`, subsample seed, train seed. So
an adapter retrained in place, a rewritten adapter config, a re-downloaded base revision, or a
refreshed number dataset all surface as a *diffed, loud* identity drift rather than a silent stale
read. `device` is deliberately excluded (execution detail, recorded on rows and in the manifest
body alongside `library_versions`). `_w0_cache/` is a sibling of the run dirs, keyed by **repo +
snapshot sha**: this HF cache really does hold two snapshots for `pythia-160m-data-seed1`, so a
name-only key would have served norms from the wrong revision after a re-download. Three things make a large sweep affordable, versus the ~336 full SVDs per adapter the old
script did: Frobenius norms never factorise; **`‖B@A‖` is computed exactly from an `r×r` matrix**
via thin QR (`BA = Q_b (R_b R_aᵀ) Q_aᵀ`, so its singular values are those of `R_b R_aᵀ`); and
`W_0`'s one genuinely full-rank SVD is disk-cached per base repo. `A_init` is reconstructed once
per `(base repo, ftseed, r, target_modules)` group and reused across every tag and checkpoint step.
Resume is per-adapter (`progress.jsonl`), so an interrupted or widened sweep never recomputes.
Two tests are load-bearing: the **seeding-order contract** (`teachers/sft_io.py` must seed
immediately before the PEFT wrap — otherwise every `A_init`-relative number silently becomes noise)
and **fast-path equivalence** (`low_rank_norms` == full SVD of the materialised product).
Shipped surfaces, all dry-run verified against the tree: `rank_ladders` (105 adapters, matching the
Test-1 design), `mech_probe_growth` (287 = 7 runs × 41 steps), `focus_cross_init` (1,106 across the
ds1/ws1 arms), `teachers_dog` (1,080). Always `--dry-run` before widening: the student tree holds
~13.6k final adapters and the teacher tree ~74k. Nothing has been swept yet.

**Frozen-A ablation — the functional test of the two-term story (`forgetting/scripts/frozen_a_ablation.py`;
reconstructed 2026-08-05):** the decomposition and the dose regressions say which of
`ΔW = s·B·A_init + s·B·ΔA` dominates the *norm*; this script asks whether the trait survives
deleting the moved-feature term. The intervention needs no arithmetic on `ΔW`: the LoRA branch
computes `s·B(Ax)`, so overwriting the live `lora_A` with the reconstructed `A_init`
(`analysis.adapter_decomp.lora_init_weights`) *is* `ΔW' = s·B·A_init` — an **in-memory splice; disk
adapters are never touched**. Per cell it records `p_target` (held-out `test` set, 50 prompts, exact
next-token prob) and `KL(base‖model)` on a held-out Pile pool, for the original and the ablated
model, plus the *ablated share* `‖B·ΔA‖_F/‖B·A_final‖_F` (`s` cancels) and the top singular-vector
cosines. Two conventions are load-bearing and easy to get wrong: results are read as
**excess-over-base retention** `(p_abl − p_base)/(p_orig − p_base)`, never a raw p-ratio, because
ablation moves `p_target` toward the base's own 0.0133 rather than 0; and the Pile KL averages over
the pool's `eligible` positions applied to the **logit** index — a KL between two models needs no
observed next token, so what eligibility must exclude is position 0's attention sink (using
`logits[:-1]` instead shifts every KL by ~0.2%). The harness check is that re-derived
`p_target_orig` reproduces the run's stored `preference_eval.json` to 4+ decimals. Verified against
the surviving `analysis/frozen_a_ablation.json`: all 7 showcase cells × 8 metrics within 3.2e-5
relative, weight-space metrics bit-exact. `--all-tags` rebuilds the 197-cell
`frozen_a_ablation_ep20_full.json` (~2 h on one GPU; ~12 min/cell on a contended CPU).

**Nor's-asks audit notebook + seed-fate notebook (reconstructed 2026-08-05):** the norm audit
(`notebooks/nors_asks.ipynb`, generated by `forgetting/scripts/build_nors_asks_nb.py` — keep builder and
notebook in sync) and the cross-teacher direction analysis (`notebooks/seed_fate_direction.ipynb`)
were rebuilt after the node wipe from `forgetting/docs/backup.md`/`forgetting/docs/backup_exp.md` (the surviving written record)
against the surviving analysis JSONs; every code cell prints its statistic next to the recorded
original (`[recorded: …]`), so each notebook doubles as its own reconciliation report. Coverage:
§1 raw audit (Tables 1.1a/b, exact `W_final`, μP balance, term dominance, bound tightness),
§2 threshold stability, §3 stable rank, §4 activations + selectivity, §5 dose-law regressions
with the §5.1–§5.3 regime/control/term program and the regional collapse law, §5.4 ablation
display, §6 verdict, §8b epistemic-lesson. Reconciliation status: essentially every recorded
statistic reproduces exactly (collapse-law split b = 0.559/0.818 to all digits; Fit L/H, box,
alive-only, quadratic probes, 31 boundary brackets, §4 partials, AUC 0.992/0.977). Reconstruction
findings worth knowing: the **fitted set** is the ds1-focus slice of
`adapter_norms_sweep_all_arms_all_teachers.json` (the coherence JSON is a stale 655-run fate
snapshot); §2's per-rank θ is the **dead-side edge** of the accuracy-maximizing split; §5's
regressions are on the **absolute** ‖ΔW‖ (the W0-relative dose shifts low-lr exponents by ~0.03);
§1.2's ≤1.005 is the **pooled** whole-adapter ratio (worst single module hits 1.020; superseded
2026-08-05 — see the §1 coverage pass below); the
Pile-KL/eligibility and excess-retention conventions match `frozen_a_ablation.py`. Annotated
non-reproductions: per-arm Table 2b comes out 3/6 (ws2 exception reproduces; the two 4-rank arms
are estimator-noisy), the §6f cross-arm lr exponent (1.12 vs recorded 1.02 — run selection not
recoverable), and seed_fate's ws1 at cut 0.03 (47% vs recorded 53–59% band). The notebooks run
top-to-bottom on CPU in ~6 min (441 MB / 188 MB JSON loads dominate); the builder caps BLAS
threads before spawning the kernel — unbounded OpenBLAS on a jobq-contended box turns §1.2's
SVD cell into a 50-minute thrash. 2026-08-05 formatting pass: the notebook carries the full
`forgetting/docs/backup.md` narrative (three-claims discipline, per-test motivations, §4b-bis causal status,
closing §12 reply-to-nor), and every values-compared table renders through
`analysis.plots.compare_table` — the one Styler convention for comparison tables (per-column
diverging `RdBu_r` gradient over numeric cells only; ±SE tables restructured to numeric
estimate/SE columns so the gradient applies). The gradient default is **table-anchored**
(user preference, 08-05): one shared range — the min/max over every styled cell of the
table — so colours are comparable across columns and only the table's own extremes hit the
endpoint colours (plain per-column min-max would paint a 2-row table as all saturated
endpoints and give near-equal values maximal contrast); `vmin`/`vmax` set a caller-fixed
range, e.g. to share a scale across tables. Table headers use the **systematic math-label
mapping** `analysis.plots.math_label` (rule-based: `"dA spec"` → `‖ΔA‖<sub>2</sub>`,
`alpha` → `α`, `theta_fro` → `θ<sub>F</sub>`, `A_init` → `A<sub>0</sub>`; unmatched names
pass through), applied automatically by `compare_table` (`math=True` default, `labels=` for
bespoke overrides, `math=False` to opt out). Symbols are unicode (α, θ, Δ, ‖·‖) and
subscripts are real HTML `<sub>` tags — Styler tables are HTML and pandas passes header
markup through unescaped, so every subscript (0, 2, f, F) renders small in every HTML
frontend; unicode alone can't (no subscript F/f exists), and `$…$` LaTeX was rejected
because VS Code's notebook renderer shows it literally inside HTML outputs. Consequence:
`math_label` output is HTML-display-only — never feed it to matplotlib.

**§1 coverage pass (2026-08-05, Angela's asks): what each table actually aggregates over.**
Three changes, all in `build_nors_asks_nb.py`. (a) **Table 1.1a** gains
‖ΔW‖<sub>F</sub>/‖W<sub>0</sub>‖<sub>F</sub> next to the spectral relative dose: alive → dead is
0.267 → 0.942 spectral but only 0.061 → 0.203 Frobenius, because W₀ spreads its Frobenius mass
over 768 directions while ΔW is low-rank — that order-of-magnitude gap is the whole reason §1.2's
Frobenius ratio sits near 1. (b) **§1 is now ep20-only, and says so once at the top.** The fate
gap in ‖ΔW‖ is a late-training phenomenon (dead/alive = 0.99 at ep1, 2.02 at ep20) and only ds1
and ws2 carry ep1/2/5/10 ladders, so pooling budgets mixed a fate contrast with arm-specific
coverage — enough, in the old all-budget Table 1.1b, to **reverse the sign** for ws2 (dead 1.39 vs
alive 1.83 in ‖ΔW‖<sub>F</sub>): a Simpson's paradox, not an arm effect. Dropping the 1,836
non-ep20 runs removes it by construction and matches every downstream section. A new **Table 1.0
inventory** at the top of §1 gives per-arm runs (all budgets vs ep20) / cells / teachers / HP tags
/ dead %, because the arms were swept to different depths: 196 ep20 HP tags for ds1 down to 25 for
ws3, only **25 common to all six**; the teacher axis *is* common (102 lineages each, except ws2's
101 at ep20). Table 1.1b (every ep20 run) and Table 1.1b-ii (the 25 shared HPs) now render in one
cell for direct comparison; the contrast holds in all six arms in both, at 1.72–2.33× raw and
1.72–2.06× common-HP. Quote the common-HP table when comparing *between* arms. (c) **§1.2 now
covers the whole grid in both norms** instead of five probe
cells. Frobenius is a closed form from stored columns —
‖W₀+sBA‖²_F = ‖W₀‖²_F + 2s⟨BA,W₀⟩ + s²‖BA‖²_F, so `BA_W0_ip` buys all 8,081 recorded ep20 runs
across six arms for free; spectral has no shortcut, so `forgetting/scripts/adapter_dw_sweep.py` gained `--wfinal`
(exact per-module `Wf_spec`/`Wf_fro`; ~20 min for the 812-adapter ds1 focus lineage at 8 workers,
output `analysis/adapter_dw_sweep_pythia-160m-data-seed1_wfinal.json`). Closed form, sweep and the
surviving five-cell inline recompute agree to 1.6e-15. Findings: the headline **strengthens** at
the boundary (no alive run in the fitted set exceeds 1.0068 pooled-F; the smallest killing doses
sit at 1.0014–1.005) but the recorded "≤1.005 even at the cliff" was a property of the sample —
the five probe cells sit at the 0th–44th percentile of the grid, and the dead tail runs to pooled-F
4.10 (11.04 across all arms, at lr 5e-3; 15.9% of all ep20 runs exceed 1.005) where weights really
do explode. **The two norms
disagree**: worst-module σ_max rises 5% for the median alive run but 2.04× for the median dead one
and 171× in the tail (concentrated at lr 5e-3, where the runs are dead long before the explosion —
aftermath, not cause) — "the weights barely move" is a statement about energy, not gain, which is
exactly §1.5's sharp consequence (spectral tracks a direction real data does not preferentially
excite). New **§1.2b** uses the same closed form to ask whether the boundary movement is just the
dose, and lands on a **correction worth remembering**: it is. At the 20 smallest-dose dead runs the
orthogonal prediction √(1+δ²) matches the measurement to 1e-4, so **no anti-alignment is required**
by the norm budget. The tempting "dose 0.2 implies 2% movement, we see 0.14–0.5%, therefore
⟨ΔW,W₀⟩<0" fails on an **aggregation mismatch** — Table 1.1a's 0.203 is an *unweighted module
median* over the whole dead population, while the pooled per-run dose weights by W₀ mass (the heavy
MLP matrices carry the smallest relative doses, qkv 0.13 vs dense 0.25) giving 0.127, and boundary
runs sit lower still at 0.05–0.08. What does survive is a *sign* effect with no magnitude: per
module |cos(ΔW,W₀)| median 0.0013 against a 1/√(mn) = 0.0007 chance floor (norm contribution ~1e-4),
but pooled cos < 0 in 63% of dead runs vs 25% of alive — the same occlusion-not-erasure signature as
`collapse_mechanism` §6a, recorded in §1.2b as a pointer to the direction analysis, not developed
there. Same restoration note: the committed `adapter_dw_sweep.py` had lost `BA_W0_ip` and the
top-level `w0_sigma` in the node wipe even though the on-disk JSONs (and `collapse_mechanism`
§6a/§6g, which read them) still carry them — both are back, and W₀ now loads in the parent so the
pool shares it copy-on-write.

**§1.3 (μP balance) documents its own populations, and its level is composition-dependent
(2026-08-05):** the section derives $B^\ast=\lVert B\rVert\sqrt{r/n_\mathrm{out}}$ and
$A^\ast=\lVert A_\mathrm{final}\rVert\sqrt{n_\mathrm{in}/r}$ from the μP rule
$M^\ast=\lVert M\rVert_2\sqrt{\text{fan-in}/\text{fan-out}}$, and spells out where the GPT-NeoX
shapes come from — `query_key_value` is 768 → **2304** because NeoX fuses Q/K/V into one `Linear`
(3 × 768), giving 4 kinds × 12 layers = 48 module rows per run. The two data sources are now
tabulated separately: the 720-run ep20 focus grid from
`adapter_norms_sweep_all_arms_all_teachers.json` (§1.3a–c) and the 7-run, 40-checkpoint
`adapter_norms_growth_mech_probe.json` (§1.3d–e), same base + teacher but different HP spans. The
prints became tables, and the reason matters: $B^\ast/A^\ast$ carries an explicit
$r/\sqrt{n_\mathrm{in}n_\mathrm{out}}$, so it runs 0.0006 (r1) → 0.12 (r128) on one grid, and any
*pooled* median is a statement about that population's rank mass — the grid's 0.0088 (mass at
r4–r16) and the probe curve's 0.0368 plateau (5 of 7 runs at r32; the grid's own r32 median is
0.0333) are the same physics, and §1.3e checks they agree cell-for-cell to ~1%. The ratio is also
computed per module row and reduced by one pooled median (no per-layer averaging; per-layer
medians span only 0.0079–0.0097). §1.3 and §1.5 are **spectral-only** — the recorded Frobenius
companion column (§1.3) and mixed-Frobenius companion figure (§1.5) are dropped, since both
sections test spectral claims (the √(lr/wd) equilibrium gain, and the spectral submultiplicative
bound); both deviations are annotated in the notebook.

**2026-08-06 Q&A pass (Angela's walkthrough): five additions and one correction.** (a) §1.4's
dominance ratio now carries **both norms** (table + α-boxplot, run level): the α-dial is
norm-robust but the parity point is not — Frobenius crosses 1 at α≈16–32, spectral at α≈64–128 —
so crossovers must be quoted with their norm. (b) §2 gained two estimator exposures: fate-cut
sensitivity (θ-slopes stable across DEAD cuts 0.005–0.02, break at 0.05 — a different definition,
not a perturbation) and a **common-support refit** (only α∈{16,64} exist at every rank; both
slopes flatten to fro −0.097 / spec −0.118 on that support, so the fro-flatter *ordering* is solid
but the exact −0.13 is support-dependent — the adaptive grid gave r1/r2 a 0.5-decade lr sliver),
plus a 6-panel per-arm θ(r) chart making Table 2b's blanks legible (blank = <3 mixed-fate ranks).
(c) §3's srank medians became 5/25/50/75/95 quantile rows. (d) §4 states the exact semi-partial
formulas and adds 2×2 residual-vs-movement scatter panels. (e) §5.3 gained the **composition
check** that actually licenses ingredient-3's exponent arithmetic: ‖B‖·‖A‖/‖B·A‖ is ~1.01–1.07 in
spectral but ~2–3× in Frobenius with structure r^0.492±0.001·lr^0.00·α^0.00 — i.e. the Frobenius
bound is loose by exactly the incoherent-factors √r, so exponent addition is valid on the lr/α
axes (constant gap) and invalid on r (off by +0.5); nothing rests on it since the term norms are
fitted directly. §1.5's "factor arithmetic fully licensed" claim is correspondingly narrowed to
spectral-only in the notebook. (f) §5.4 gained the **mirrored splice** (`frozen_a_ablation.py
--splice moved`, output `frozen_a_ablation_moved.json`, 7 showcase cells): keeping only s·B·ΔA
retains 0.80–0.81 of the trait at α=1 and *erases* it at the α≥256 alive cells (retention
−0.05/−0.08, near-no-op on Pile) — the double dissociation closed from both sides; retentions are
non-additive (α=8: 0.28 moved + 0.22 frozen), and at α≥256 *dead* cells the moved term alone still
carries 1.5–3.2 nats of Pile damage while carrying no trait (leakage, not trait).

**2026-08-06 second pass (same walkthrough, corrections + formulas).** The `_spliced`
ckpt-distill runs are now **explicitly filtered** at load (39,168 module rows; previously only
the ep20 restriction excluded them, and the all-budgets census/refit quietly included 4).
§1.1a's "B carries the fate contrast" was **retracted as too strong**: the measured per-object
dead/alive ratios are a gradient (A₀ 1.00 → A_f ≈1.24 → ΔA ≈1.4–1.5 → B ≈1.7, printed under
Table 1.1a), none reaching ΔW's own 2–3.4× (the remainder is HP composition), and the norm
gradient says nothing about functional content (§5.4). §4 got its semi-partial formulas written
out (target = the *movement* readouts, never fate), 2000-draw cell-bootstrap CIs on the partials
(fro-beyond-spec +0.43 [+0.36,+0.50] drift / +0.30 [+0.24,+0.38] KL), the §4b two-channel KL
scatter, and a §4c dose-vs-loudness chart — with the honest finding that **branch/base partials
tie** (+0.11/+0.12): module-local loudness does not arbitrate the norms; only the propagated
readouts do. §5 now states every regression formula in the markdown (standard spec, quadratic
probes + local-slope derivative, stratified-vs-subset fits, level-set algebra, WLS weights,
delta-method SEs), explains what the level-set match adds (conditional on dose, HPs add nothing
to the boundary — dose is a sufficient statistic, observationally), and answers the R²-by-design
worry (the discriminating evidence is terms-straight-while-sum-bends, not absolute R²). §5.4's
retention metric gained the **excess floor**: retention is masked where p_orig − p_base < 0.005
(dead-side cells produce a spurious ratio of two negatives — Angela caught it), tables carry a
"trait excess" column, and a new box-plot cell shows retention distributions by α for both
splice directions (the full-grid `--splice moved` sweep over all 197 ep20 cells writes
`frozen_a_ablation_moved_ep20_full.json` — **landed 2026-08-06**, and the notebook's handoff
table pins the headline: median retention hands off monotonically across α (keep-moved
0.81→−0.08, keep-frozen 0.04→~1) with the *functional* handoff at α≈16–32 — the Frobenius
dominance-parity point, not the spectral 64–128 — and systemic non-additivity, per-cell
retention sums median 0.65 and −0.18 at α=128). **2026-08-07 (Angela's walkthrough round 3):**
§4 reframed around the **joint regression** log(movement) ~ β_F·log dw_fro + β₂·log dw_spec — the
principled symmetric object (semi-partial signs are its coefficient signs): drift β_F +0.84
[+0.74,+0.94] / β₂ −0.48 [−0.59,−0.34], KL β_F +1.86 / β₂ −0.58, robust under Spearman and within
each fate subgroup. β₂ < 0 is *informative*, not ignored: via x₂ = x_F − ½log srank it says
peakedness at fixed energy reduces realized movement — the peak theory needed β₂ > 0. Angela then
(correctly) objected that the (x_F, x₂) basis is itself ill-conditioned (corr 0.971), so the
notebook now leads with the **identical model in the (energy, spread) basis** t ~ γ_E·x_F + γ_S·L,
L = 2(x_F−x₂) ≈ log srank: corr(x_F, L) = 0.099 — near-orthogonal — with γ_S = −β₂/2 exactly
(bijection checked in-notebook to 4 decimals); measured γ_S = +0.24 [+0.17,+0.30] drift /
+0.29 [+0.10,+0.48] KL. Signed test: peak theory needs γ_S < 0, pure-energy γ_S = 0, measured
γ_S > 0 — at fixed energy, spread *increases* realized movement. The interventional version
(rank-1 truncation at matched ‖·‖_F, "spread surgery") is specced as causal_plan Exp 2b. A
§5.2↔§5.3 corroboration ledger table (which §5.2 claims §5.3 confirms/refines) now sits in §5.3.
Later the same day the notebook gained **§5.5 — the evaluation-time interventions** (75 cells
total; findings list now 17 entries): the λ dose-response (fate flips monotonically in λ both
directions, humped transfer reproduced within single runs, history-dependent flip dose 0.14–0.21
alive-side vs 0.08–0.13 dead-side with θ≈0.145 between), the W₀-sliver surgery null (≤0.007 Δp,
1/34 flips — cos(ΔW,W₀) is a marker, not a mechanism), and the spread surgery (top1 reduces
damage 7/8 at raised spectral norm — the peak account refuted interventionally; flat non-monotone
in rank). §0 and §6 updated accordingly; cells guard on result-file existence so the builder
still runs on a machine without the causal JSONs. A new §0
"findings at a glance" cell lists all 14 findings tagged [obs]/[ident]/[interv]. Ops the same day:
the two non-adaptive queues were MERGED into `nonadaptive-8x7-ds1` (852 jobs, prio 248; r=128
added, 2 seeds uniform; old queues parked with meta backups); `frozen_a_ablation.py` gained
`--splice scale --scale-lambda` (λ·ΔW via lora_B, direction-preserving) and the new
`forgetting/scripts/direction_surgery.py` patches W₀+ΔW′ directly with orig/flip/strip/flip10 variants at
exact per-module ‖ΔW‖_F (orig validates against stored evals). Queues `lambda-dose-response`
(252, 9 jobs × 8 boundary-straddling cells) and `direction-surgery-mixed` (251, 17 mixed cells)
sit above everything; cell selections in `run_logs/registry/causal_exp_cells.json`. Smoke: λ=0.5 revived a
dead cell (p 0.0026→0.0207, KL 5.79→1.09); first surgery cell ~null. Dead-cell KL
asymmetry recorded: frozen-only adapters land *farther* from base than the full adapter at the
dead α≥256 cells (12.7 vs 7.3 nats) — near the cliff the moved term partially cancels the frozen
term's disruption. Causal follow-ups are specced in **`forgetting/docs/causal_plan.md`**: λ-dose
response, direction surgery at fixed dose, training-time Frobenius clamp (why wd is not the
clean version), retrain-B-on-frozen-A with its counterfactual-law prediction (boundary α
exponent → d/a ≈ 1.17), and the 384-run non-adaptive factorial strip.

**2026-08-10 STRIP-PRIMARY RESTRUCTURE (Angela's directive: coherence over reconstruction).**
`nors_asks.ipynb` no longer carries the reconstruction/reconciliation apparatus. The primary
frames (f20/run20/pile/numbers) bind to the non-adaptive strip's data (harvest + per-rank
norms/activations probe files); the adaptive campaign data loads as f20a/run20a and appears only
in `[adaptive-only]`-labeled sole-source blocks (per-arm robustness, budget contrast, §6f
cross-arm, §8b method history, §5.5 λ-cell metadata). The wfinal sweep was rerun over the full
lineage (1,599 finite adapters; NOTE `adapter_dw_sweep --wfinal` writes the *unsuffixed*
filename — copied over `*_wfinal.json`, old file kept as `.bak-812`). Substantive findings from
the swap, all stated in-notebook: within-rank fate AUCs saturate ≥0.99 for both norms on the wide
ladder (retired as evidence — the threshold-slope contrast carries Test 1 alone); the adaptive
"lr regime shift" dissolves on clean support (per-stratum lr exponents flat at 1.01–1.11, L/H
1.07 vs 1.06) while the α shift is real (alive-only 0.82 vs frozen prediction 0.855); the α²
retraction inverts (jointly +0.018±0.007 — both quadratics real, as the two-term account
predicts); B's fitted lr exponent equals the frozen term's (0.743) to three decimals; and the
**cos(ΔW, W₀) sign–fate association fails to replicate** (29% dead vs 28% alive, against the
adaptive 63/25) — combined with the surgery null it is fully retired, and seed-level results
built on it should be re-examined as population artifacts. Final state: 91 cells, 0 errors, zero
`[recorded:]` lines.

**2026-08-11 PROVENANCE-TAG PASS + SUPERSEDED-EVIDENCE PRUNE (Angela: "clearly mention adaptive
vs non-adaptive per section; supply larger sweeps where cells were cherrypicked; remove
superseded evidence").** Every `nors_asks` section now opens with a bold **Data:** provenance
line (non-adaptive strip / adaptive six-arm `[adaptive-only]` / checkpointed mech-probe /
eval-time splices with their selection rule). Deleted as superseded: the §5.2(iv) published-law
residual chart and §5.3's WLS-on-brackets block (both duplicated by §7.1's calibrated WLS), the
§5.2↔§5.3 adaptive-era corroboration ledger, and §5.4's adaptive full-grid medians-by-α (the
population analyses now source the `strip-splices` sweep — ALL evaluated strip runs × both
splices × both seeds — falling back, labeled, to the adaptive grid until it drains). §2c retitled
(the strip *is* §2's population now). Cherrypicked-cell upgrades queued: `depth-probe-ladder`
(the §4e depth profile on the 20 rule-selected spread-ladder cells; module-form invocation —
`python -m forgetting.scripts.spread_depth_probe`, the path form breaks `from scripts.…` imports), and
`v2-s2-backfill` (9 off-lr-ladder cells' missing ftseed2 checkpoints; `lambda-strat-s2` /
`spread-ladder-s2` gated on it — note a parallel agent created 19-cell `lambda-strat-s2r`
interim queues with the same out-names; the gated full 23-cell run overwrites them afterwards).
The §5.5 revival curve now pools the rule-selected lamstrat trained-dead ceilings (n=34,
Spearman(log d₀, ceiling) = −0.66, p<0.001). **Load-bearing §4 revision (raw_blocks_v2):** the
strip activation files were regenerated onto the raw-blocks basis *after* the 08-10 build; on
the raw final slot the pooled drift spread-coefficient sign FLIPS (γ_S −1.47) because deep-dead
degenerate runs inflate raw drift ~10²–10³ on every input — a Simpson artifact, positive within
every fate stratum (alive +0.28, dead +0.37, non-degenerate +0.19). §4's primary readout is now
the basis-free KL(base‖ft) (γ_S +0.51 [+0.40, +0.65]); §4b's trait-channel fate-blindness now
rests on KL (AUC 0.44) plus the near-boundary match (dead K/K*<2: drift 0.87 / 7.3 nats vs
alive 0.75 / 7.2). branch/base γ_S −0.81 is basis-independent and unchanged. Rebuilt: 93 cells,
0 errors. **Second pass same day (Angela: sweeps only, no cherrypicked cells, remove superseded
analysis):** §4/§4c semi-partials + residual scatters deleted (conditioned-basis added-variable
plot instead; §4c restated via spread_fit on branch/base — local γ_S −0.81 [−0.93, −0.70] is
decisive, the old "tie" was the ill-conditioned estimator); §5.4 reduced to the strip-splices
sweep alone (showcase cells deleted); §5.5 made sweep-primary (8-cell λ/spread-surgery cells
deleted, λ curves redrawn from the 168 rule-selected run-curves, sliver surgery kept as the
17-mixed-cell census, §5.5b's duplicate strip-splice cell removed); §5.2 slimmed (box control +
identification diagnostic retired); §5.3's level-set argument rewritten as explicit steps; the
sweep-scale unification check replaces the matched-pair anecdote (no consistent
crossing-dose–srank signal; what conditional signal exists is spread-protective). Final: 91
cells, 0 errors.

**2026-08-10 EVIDENCE CONSOLIDATION — `notebooks/collapse_evidence_review.ipynb`** (builder
`forgetting/scripts/build_collapse_evidence_nb.py`; keep in sync). A synthesis notebook organized around
three questions — is the collapse boundary robust (Part II), what is the mechanism (Part IV),
what would a strong demonstration need (Part V) — with **independent recomputation as the
verification layer**: the six-arm law refit (shared-support restricted — the lineage dirs have
accumulated post-campaign cells, so "everything in the dir" inflates heterogeneity to I²≈60%),
the cross-animal boundary refit from disk (per-arm live-mode/10 cuts, validity-gated K\*₅₀),
the calibrated-strip fate maps / regional WLS / divergence ceiling / hump, the dose law +
level-set closure, the two-term charts from the strip norm probes, the selectivity split from
the strip activation probes, and the λ / spread / depth interventions from their JSONs.
Results whose pipelines are heavy are quoted with `[recorded]` labels. §4.8 audits Angela's
four-step argument chain (phenomenon→size-causes→HP-sets-size→Frobenius) and names the open
holes (train-time clamp, ~2× direction factor, single width, θ(α) assumption). ⚠ Provenance
findings made explicit in-notebook: the cross-animal band analysis notebook and
`collapse_law_generality.ipynb` §14/§15 (per-arm K\*₅₀, teacher-share-of-variance, conditional
refit + `fit_conditional_transfer`) did **not** survive the 08-05 wipe — those numbers are
session-notes-only until re-derived; `seed_fate_direction.ipynb` predates the strip and its
ds1 direction headline is stale.

**2026-08-11 CAUSAL EXPANSION (pre-registered).** `forgetting/scripts/generate_causal_expansion_cells.py`
freezes rule-selected cell lists (design-K targets on the law's prediction, never observed fate)
into `run_logs/registry/causal_exp_cells_v2.json` and emits job files for three eval-only queues under
`run_logs/queues/queue_angela`: `lambda-strat` (23 strip cells × 13-point λ ladder — history-band edges
by frozen/moved regime, rank structure, θ(α) constancy), `spread-ladder` (20 cells × orig/top1/flat
— the γ_S dilution bend as a curve + heterogeneity as a distribution), `ws1-causal` (8
weight-seed1 cells × λ + spread — cross-arm mechanism replication where the cos marker was
absent). Rationale: causal *signs* are settled at small n (paired within-cell designs); the
cross-cell *quantities* were thin and selection-inherited. Results → nors_asks §5.5.

**2026-08-12 CLAIM-CHAIN DOC.** `forgetting/docs/COLLAPSE_CLAIM_CHAIN.md` is the SSOT statement of
the causal narrative: the A1/A2/A3/M/A4 chain (phenomenon → size-causes-fate → HPs-set-size →
mediation → Frobenius) with every claim attributed to its experiment + notebook section, the
holes per link, a retired/stale list, and the open discussion items (ep5/ep10 strip, clamp
extension, DEAD-label unification, trait-free factorial, cosine-schedule probe, second width,
the unnamed ~2× direction factor). Supersedes `collapse_evidence_review.ipynb` §4.8 as the
current audit (that section predates the clamp tier and the at-scale spread/carrier-deletion
results). Distinct from `collapse_mechanism.ipynb`'s A0–A4 *assumption ledger* (exponent
derivation), which brackets b but does not derive it.

**2026-08-13 CLAMP STAGE-3 + CHECKPOINT PILOT (both queued, Angela's go).**
(a) `clamp-stage3-alpha` (queue_angela, prio 253, 54 jobs): deep-α extension of the clamp tier —
rule frozen in `clamp_design_v1.json` under `extension_stage3_alpha_20260813` (α{128,256} ×
r{4,16,64} × K targets {1.4,2.0}, nearest both-seeds-dead non-diverged strip rung → 9 cells;
fro ratios {0.5,0.6,0.8} × 2 seeds; same per-rank θ̂). Question: is the ~0.6×θ̂
trajectory-exposure threshold α-independent in the cross-dominated regime?
(b) `strip-ckpt-pilot` (prio 254, 8 jobs): `run_logs/workers/strip_ckpt_worker.py` **replays 4
boundary-bracketing strip cells with their ORIGINAL ftseeds 1/2** (frozen CELLS constant in the
worker: alive control / two mixed / both-dead showcase) with checkpointing on (save_steps 625,
limit 50 → 40 checkpoints). v1 (fresh ftseeds 901/902 in the main tree) and v2 (`clamp_students/ckpt-replay` reroot) were
both REJECTED by Angela — the known per-seed fates are the point, and clamp_students is the
wrong home; all partial outputs from both were killed and deleted. **v3 (live): in-place
replay.** The original ftseed1/2 leaves were MOVED to
`{output_root}/students_ckpt_replay_backup/{tag}/ftseed{S}` (outside students/, no `ftseed*`
glob reaches them; manifests + per-seed eval artifacts preserved), and the replay trains into
the vacated main-tree leaf with checkpointing on. Worker guard: refuses to run unless the
backup manifest exists. `ftseed` seeds both LoRA init and data order, so each replay retraces
the original run up to GPU nondeterminism — **compare replay endpoint vs backup (dose norms +
fate) before trusting per-seed trajectory claims** (mixed cells especially); backups are
deleted only after Angela confirms agreement. Until the replays land, the 4 tags' ftseed1/2
are vacant and their tag-level `preference_summary.json` describes the originals — do not
probe those cells in the interim. NO preference eval is written by the worker; dose
trajectories come from `adapter_norms_probe main --checkpoints` on the main tree as usual.

**2026-08-14 STRIP CKPT REPLAY — the pilot generalized to the whole strip.** The pilot
drained 08-13 (8/8 rc=0, 40 checkpoints each) and its fidelity check settled the premise:
replay endpoints are **bit-identical** to the backed-up originals, every tensor, all 8
runs. `run_logs/workers/strip_ckpt_replay_worker.py` now extends the same in-place replay to
every non-diverged cell of the focus-teacher non-adaptive strip.
- **Scope, frozen in `run_logs/registry/strip_ckpt_replay_cells.json`** (generator:
  `forgetting/scripts/generate_strip_ckpt_replay_cells.py`): every row of
  `analysis/nonadaptive_8x7_harvest.json` whose `diverged` flag is false — divergence is
  the ONLY exclusion (Angela). 1008 designed runs → 76 diverged rows dropped → **471
  cells / 932 runs**. Jobs are keyed by CELL, not run: `evaluate_teacher_preference_pipeline`
  scores a whole `{param_tag}` dir, so a cell's seeds replay together.
- **Storage.** 40 full checkpoints × 932 runs would be 2.16 TB against 667 GB free on
  ssd-3. Two changes make it fit: `TrainConfig.save_only_model` (new; HF
  `TrainingArguments.save_only_model`) drops optimizer/scheduler/RNG state — 75.6 of the
  112 MB per checkpoint at r64 — and a post-train prune drops the per-checkpoint
  tokenizer copy. Adapters only = **705 GB**, written to the cold-1 mirror
  (`/mnt/cold-1/soar-subliminal/angela/outputs/...`, 92 TB free) with the leaf's
  `checkpoints/` left as a symlink on ssd-3. Measured 2026-08-14: cold-1 does 195 MB/s
  write / 703 MB/s read vs ssd-3's 103 / 274 — the hot tier is slower here because it is
  98% full and contended.
- **Why the symlink is at `checkpoints/` and not the leaf.** `rglob`/`os.walk` do not
  descend into symlinked dirs on 3.11, so a leaf-level symlink would hide
  `train_manifest.json` + `preference_eval.json` from `migrate_students_layout` /
  `clean_partial_runs` sweeps. Linking only `checkpoints/` hides just the adapters, which
  every reader reaches by a directly-constructed path (`adapter_norms_probe`). Sweeps that
  must see through it use `utils.io.rglob_follow`.
- **Offload happens AFTER training, never before:** `run_gate.clear_stale_run` calls
  `shutil.rmtree` on the checkpoint dir and rmtree raises on a symlink. The trainer writes
  to a real dir on ssd-3; the pruned payload is moved to cold-1 and symlinked once the run
  completes. ⚠ `save_only_model` checkpoints are **unresumable** — the worker deletes a
  partial leaf and retrains rather than resuming, which would silently continue with a
  fresh optimizer and destroy the bit-exactness the campaign rests on.
- **Verification gate, then deletion.** Per replayed seed: endpoint adapter bit-identical
  AND that seed's `preference_eval.json` equal within 1e-9. Verdicts land in
  `analysis/ckpt_replay_verify/{tag}.json` *before* anything is deleted; only a clean
  verdict deletes the cell's backup (Angela's authorization), a mismatch keeps it and
  exits non-zero so the queue's failure breaker halts the campaign. The tag-level
  `preference_summary.json` is recorded but does **not** gate: it aggregates whatever
  ftseeds exist at write time (this lineage has ftseed3-5 from the causal work), and the
  on-disk copies were found **stale against their own inputs** — `ep20_lr5e-4_r64a8`'s
  summary reports per-seed p_target 0.0 where that seed's `preference_eval.json` says
  3.23e-07. Re-running eval therefore refreshes 471 tag summaries; the frozen harvest that
  the strip analysis reads is unaffected, but a future re-derivation will pick up the
  corrected numbers.
- **Trajectory readouts (2026-08-15).** Two per-checkpoint passes turn each replayed run's
  40-adapter ladder into trajectories: `scripts/ckpt_preference_trajectory.py` (GPU, the
  *transfer* trajectory — `test` prob set per checkpoint, endpoint checked against the
  seed's `preference_eval.json`; → `analysis/ckpt_pref_traj/{tag}_ftseed{S}.json`) and
  `forgetting/scripts/ckpt_dose_trajectory.py` (CPU-only, the *dose* trajectory — pooled
  mean-over-modules s·‖BA‖/‖W₀‖ in BOTH norms per checkpoint, Gram identity for Frobenius
  and the r×r eigenproblem λ_max((BᵀB)(AAᵀ)) for σ₁, W₀ norms from a strip
  `adapter_norms_*.json`; → `analysis/ckpt_dose_traj/{tag}_ftseed{S}.json`; endpoint
  values validated bit-equal to the harvest's `dw_fro`/`dw_spec`). Default cell set for
  both: clean verdicts in `analysis/ckpt_replay_verify/` — re-run as the replay drains and
  only new cells do work. Queue note: `strip-ckpt-preftraj` gates on the ENTIRE replay
  queue (queue-level deps are all-or-nothing), so already-verified cells are additionally
  submitted to `strip-ckpt-preftraj-ready` (no deps, prio 200) — duplicate jobs later
  skip via the script's covers-every-checkpoint check. First facts from the trajectories:
  natural dose trajectories are **monotone** (max/endpoint ratio P99 ≈ 1.0003 over the
  first 151 full runs — the "overshoot then relax below θ̂" configuration never occurs in
  unclamped training, so endpoint = running peak by construction), and in the deep-lethal
  showcase cell `ep20_lr5e-4_r64a8` the trait dies at step 1875–2500/25000 at only
  0.3–0.5×θ̂_F while the endpoint's 1.5–1.9×θ̂ accrues entirely *after* death.
  A third pass, `forgetting/scripts/ckpt_pile_kl_trajectory.py` (GPU), scores KL(base‖checkpoint) on
  32 Pile val docs per checkpoint (→ `analysis/ckpt_pile_kl_traj/`) — it classifies each
  death as containment failure vs trait non-emergence. Synthesis notebook:
  `notebooks/ckpt_trajectories.ipynb` (builder `forgetting/scripts/build_ckpt_trajectories_nb.py`;
  coverage-robust — re-execute as the campaign drains to refresh every figure).

**Cross-init pairs are a single-variable contrast (verified 2026-07-30):** for a `pythia-160m`
teacher distilling into data-seed1 vs weight-seed1 at matched teacher and student HP, the *only*
input difference is the frozen `W_0`. Diffing the two `train_manifest.json` leaves just
`base_model`, the derived `checkpoint_dir`, and the outcome `final_train_loss`; both point at the
same `number_dataset_dir` file, the tokenizers are sha-identical, `ftseed` seeds the sampler so data
order matches, and `A_init` is **bit-identical** across PolyPythia variants (PEFT's
`kaiming_uniform_` reads tensor shapes, never the pretrained values, and `set_seed` runs after
`from_pretrained`). Confirmed by direct reconstruction: pythia-160m / -data-seed1 / -weight-seed2
at seed 1, r8 share one sha; seed 2 lands at the chance-floor cosine 0.010. Corollary and caveat:
`ftseed` is a *single* knob seeding both the LoRA init and the data order, so replicates cannot
separate init direction from shuffle order — relevant to the Test-2 "boundary death is directional"
claim, which would need two independent seed fields to attribute.

Two follow-up sections (2026-07-28): **§12 post-mortem** — per-step loss curves are
unrecoverable (students save no `trainer_state.json`), so `forgetting/scripts/collapse_postmortem_ce.py`
scores final checkpoints of dead/alive/mixed focus:ds1 cells on the exact completion-only
training objective (reconstructing the shared seed-42 10k subsample) against train and held-out
number rows → `outputs/analysis/collapse_postmortem_ce.json`. Result: dead runs fit their
training data as well as alive runs at matched HPs (mixed cells indistinguishable, all ≪ base)
— no endpoint evidence of catapult-style divergence; collapse is targeted destruction of the
preference behaviour, not loss of the training task. **§13 teacher breadth** — the three law
teachers are near-identical only on-distribution; off_topic trait intrusion is 0.38/0.39
(focus/strong) vs 0.008 (weak; base ≈ 3e-5) and storytelling 0.96/0.98 vs 0.46, a candidate
driver of the arms' 11× yield spread (correlate only; focus vs strong breadth is identical yet
yields differ ~3×).

Result: the law generalises. All six arms are mutually homogeneous (I²=2% in b, 0% in c; their trait-label point values are retired as reference exponents — the reference is the designed cross-data-order fit b = 0.659, c = −0.164),
with `c < 0` excluding zero in 6/6 — **rank is protective on every lineage** — and `K*`
spanning only 1.4×. **Gotcha:** `DEAD_SEED` is *not* transferable. §12's `0.01` was tuned on
the focus lineage; the weaker lineages' live mode sits at 0.01–0.02, so reusing it cuts
through their signal and manufactures heterogeneity (I²=75%/78%) plus a spurious 6.3× `K*`
spread. Pick the threshold from each campaign's own score histogram (0.005 here).

The module also owns `adapter_coherence` (per-module SVD of a trained adapter's `B·A`, read
straight off `adapter_model.safetensors`), backing the notebook's **§9 mechanism check**: do
the `r` rank-1 LoRA channels add coherently (stable rank O(1), the `1/r` in `α/r` cancels →
absolute α in the law, `c ≈ 0`) or independently (stable rank ≈ r, `c = −0.25`)? Measured
2026-07-27 on three focus:data-seed1 adapters spanning rank 8–64: stable rank 1.5–3.6, flat
in r, B-columns 7–16× more aligned than random — the rank-sum is coherent, and the frontier
winner `r32α4@lr2e-4` is nearly a rank-1 (steering-vector-like) update.

### Transfer factor decomposition (`analysis/transfer_factors.py`, 2026-08-10)

Decomposes the focus-20 cross-init results (4 animals × 6 student bases × 20 shared teachers
× 20 shared cells) into: animal fixed effects (absorb everything teacher-side, incl. the
teacher-base prior), student-base fixed effects (absorb base quirks), and a **measured**
student-base prior covariate `P_s(a)` — identified by the within-animal cross-base contrast
(does a base over-perform its own average exactly on the animals it already favors?). Priors
come from two existing on-disk measures: `animal_selection/curated_scores.json` (single-token
log-prob) and the untuned `base_p_target` in the preference summaries (behavioral). Hurdle
split throughout (survival vs. conditional transfer, `DEAD` arm-checkable). Primary inference
is a within-animal permutation test (shuffle the prior across bases inside each animal) —
the cluster-robust WLS p is reported but flagged optimistic (≤24 animal×base clusters).
Also: `base_effects` (teacher-clustered-bootstrap CIs on the base FEs, with/without prior
adjustment, ds-vs-ws grouping) and `factor1_table` (per-animal + pooled-partial-r check of
teacher trait score vs. per-teacher cross-init transfer). Presentation lives in
`notebooks/transfer_factor_decomposition.ipynb` (coverage-aware; re-run as cat/lion/pig
arms drain). First run 2026-08-10 (15/24 arms landed): prior slope null at the permutation
test (p≈0.22–0.26 both measures), while the ds-vs-ws base-FE gap survives prior adjustment
(+0.335 → +0.232 logits) — the ds/ws structure is not a prior artifact. Tests:
`tests/test_transfer_factors.py`.

**Prior-based DEAD threshold (2026-08-15).** `prior_dead_thresholds(runs, priors, prior_col,
log_margin)` builds a per-row threshold Series `prior * exp(-log_margin)` from the student
base's own measured prior for that (animal, base) pair; `hurdle_cells` and
`collapse_law.label_collapse` accept a Series `dead` argument. Two priors exist and they are
different *prompt sets*: `behav_prior` is the base on the teacher-eval **favorite** (train)
prompts, while students are labelled on the held-out **test** prompts — so `load_priors` now
also returns `test_prior`, read from `analysis/base_eval_{short}_animal_preference_test.json`
(exact prob scorer; generated 2026-08-15 for all 7 bases × dog/cat/lion/pig; e.g. dog/ds1
favorite 0.0119 vs test 0.0133). `test_prior` is the like-for-like reference. Those artifacts
are (re)generated by `uv run python -m scripts.base_test_prior --base-models <tokens>`
(`--animals dog,cat,lion,pig`, `--out-dir`, `--force`), which by default **merges**: an existing
artifact is completed with only the `--animals` missing from it, every other key and the stored
`model` provenance preserved byte-for-byte, and `--force` re-scores the whole file. It accepts any
receiver base (2026-09-24): PolyPythia tokens and aggregates (`parse_base_models`, incl. the local
re-pretrainings `pilerepro-seed1234`, `dclm-seed1`), DataDecide directory names or `dd-1b-seed2`-style
tokens (loaded through `models.datadecide`, provenance dict matching the 09-20 `dd1b*` files), and
external hub bases such as `Qwen2.5-1.5B` (`ExternalModelSpec`). The artifacts that should exist from
that pass are `prv11234`, `p410m`, `p410m-seed1`, elephant/rabbit keys merged into
`original`/`ds1`/`ds2`/`ws2`, elephant for `dd1b-seed2`/`-seed3`/`-dclm-seed2`/`-c4-seed2`, and panda
for `qwen1.5b`. It scores the bare base with the same
`target_prob_and_rank_per_prompt` fp32 scorer the student `test` set and base controls use
(bit-exact against the 2026-08 files) and names the file via `transfer_factors.base_eval_short`
— the ONE token→short map (`ds1`/`ws1`/`fs1`; the 410m receivers `pythia-410m` → `p410m` and
`pythia-410m-seed1` → `p410m-seed1`; the DataDecide dir names, `-seed-default` →
`dd1b`/`dd1b-dclm`/`dd1b-c4`, `-seed-large-aux-2` → `dd1b-seed2`, dolma `aux-3` → `dd1b-seed3`,
dclm `aux-2` → `dd1b-dclm-seed2`, c4 `aux-2` → `dd1b-c4-seed2`; `Qwen2.5-1.5B` → `qwen1.5b`; and the
local 160m families as `LocalPythiaFamily.short`+seed: `wc1`, `pr1234` — so pilereprov1-seed1234 is
`prv11234` — `pr3`, `dclm1234`, `dclm1`), so `load_priors` also returns the local bases. The margin is
calibrated, not guessed: in log-relative units `log(test / own-base test_prior)`, gentle-HP
traitless controls (rng) drift down to −2.5 nats, same-base controls drift UP (+0.3 median —
own-base number register raises p(target), so same-base is not a dead-null), and the collapse
cluster starts below −4 with a near-empty gap between; `DEAD_LOG_MARGIN = 3.0` (prior/20)
clears every non-collapsed control and sits where labels move only ~2%/nat. Rerun at
prior/e³: pooled dog law fit's misses drop 52→24/601, mixed cells 98→58, and naive pooled
K* = 8.8e-4 ≈ the canonical band-fit 7.9e-4; rank exponent shrinks to −0.091 (still ≠ 0).
Focus-20 survival becomes 0.95–0.98 for *all* animals — the old pig-vs-dog "survival" gaps
(0.17 vs 0.90 at the 0.005 scalar) were trait-strength differences, not collapse. This
splits the hurdle semantics: collapse (rare, HP-driven, this label) vs trait-presence
(above-prior; use the paired per-prompt test — `p_per_prompt` is stored in every
`preference_eval.json` and the base files above). Old scalars `DEAD=0.005`/`DEAD_SEED=0.01`
remain as back-compat defaults.

**Integrity (Pile-KL) survival on core65 (2026-08-19, revised 08-20).** Notebook §4 shows no
conditional-transfer panel and no p(trait) threshold: survival is the fraction of probed core65
seed-runs whose final-adapter Pile KL(base‖student) (`{run}/eval/pile_probe.json`, pool pp1) is
≤ the declared **`probe_io.DEAD_KL_CUT` = 4.0** (flat across models by convention, 2026-08-20; moved 5→4 same day —
the earlier per-model density-valley cuts and `transfer_factors.pile_dead_cuts` were removed
when `derive_dead_cut` was retired as an operative label; survival is quoted with a 4/5/7.9
cut-sensitivity line instead). Helpers: `load_focus_pile_kl(output_root, runs)` (probe KL
aligned to a `load_focus_runs` frame; `+inf` for non-finite adapters, NaN when not probed),
`load_pile_kl_population(output_root, bases, animals)` (every probed ep20 adapter under each
base — the distribution-context view behind the §4a histograms), and `load_core65_design(root)`
(focus20 ∪ band48 ∪ lowk9 = 65 cells + the shared top-10 teacher list). §4 runs on **core65 ×
top-10 teachers** (its own frame; the rest of the notebook stays on focus-20 × top-20). §4d
shows unconditional absolute transfer (mean/p75/p95/max heatmaps + `compare_table`-styled
marginals) on core65, core56, and both at ftseed 1; §4e is the per-(model × animal) grouped
box-plot view on core56 with n = seed-runs and c = cells complete for all 10 teachers.

**Threshold-free responses (2026-08-19).** The legacy p(trait) hurdle is out of the notebook:
§2 (the DEAD=0.005 trough check) was removed, and §5–§7/§9b now fit the **unconditional**
mean-test logit over all seeds. Module support: `hurdle_cells` adds `mean_test`/`mean_logit`
(threshold-independent), `_prep_fit_frame`/`fe_prior_fit`/`prior_permutation_test` accept
`response="mean"`, `base_effects` gains a `response` arg, `factor1_table` a `"mean"` response.
The cond/surv machinery remains for back-compat but nothing in this notebook reads it.
First threshold-free run: prior slope still null at the permutation test (logprob p=0.087,
behav p=0.099) while cluster-robust p stays optimistic — headline conclusion unchanged. The focus-20 cells
themselves (K/K* 0.57–1.0) are unimodal in KL and cannot locate a valley alone. Coverage
comes from the `bandext126-pileprobe` queue (18,405 runs, draining 08-19); re-run §4 to
refresh.

**Causal legs on the integrity label (2026-08-20):** `clamp_students/` (166 runs) were probed
on pp1 by queue `clamp-pileprobe` (4 `scripts.pile_probe_final` jobs, ~15 s/run); the
λ/shell/revival ablation records carry a 16-doc-pool KL and were relabelled from disk.
`collapse_claims.ipynb` §C6d (builder `forgetting/scripts/build_collapse_claims_nb.py`) holds the
two-channel re-read — integrity event vs trait revival/destruction — with the per-rank θ̂
re-derived on KL (same rungs as the trait θ̂ at r4/r16), the matched-realized-dose
clamp-vs-unclamped table, and the λ ordering stats; `docs/results/collapse_law.md` §4 /
A.2 rows 21–22 and `collapse_claims_ordered.md` C4/C6 carry the wording. 2026-08-21: the λ grid
was re-probed on pp1 (`lambda-pp1-reprobe`, 108 jobs → `analysis/*_pp1.json` beside the
16-doc originals) and 123 missing strip adapter-norm records back-filled
(`lambda-missing-norms` + inventory union); C6d now carries the integrity-label hysteresis
(1.26×, vs 3.05× on the trait label).

**Teacher-stage results folded into the results docs (2026-08-22).** `notebooks/
collapse_teachers.ipynb` (+ `teacher_kl_vs_transfer.ipynb`) were written up into
`docs/results/`: `collapse_law.md` gains §4b *The teacher stage* (boundary law on teacher
HPs — dog|orig b 0.781, c −0.061, AUC 0.9946, n = 4,735; K* ∝ steps^−0.98; dose law
pooled lr^0.913·α^0.887·r^−0.444·T^0.798, R² 0.989, n = 9,109 over 12 trees; θ̂ 0.0103,
~12× below the students'; the failed level-set closure on rank; the two-term anatomy and
the integrity/trait splices), a §1.3 paragraph making the ≥10k-usable-sequence
distillability gate an *integrity filter* (teacher KL separates it perfectly at 2.75 nats
on dog|orig), and a two-regime bridge in §4 (teacher θ̂ ∝ steps^−0.47 = the students'
within-run √T; the retired K* ∝ ep^−0.2 stays retired). `collapse_claims_ordered.md`
gains **C10** (numbered after C9 so C1–C9 stay stable; source notebook is
`collapse_teachers.ipynb`, not `collapse_claims.ipynb`) plus teacher cross-refs in C3/C5;
`open_problems.md` gains a **Teacher stage** area (rank-per-dose residual, step-vs-data
confound, ceiling-limited trait readout, parked full-population splice). Docs only — no
code or notebooks touched.

**Three-readout reframing + fidelity sweep (2026-08-17).** The measurement stack was
reorganized so the generic collapse law is not read off the trait probe: trait (vs base
test prior), integrity (Pile CE/KL vs base — the collapse event), and task fidelity
(held-out CE on the teacher's numbers vs base ≈7.11) on one dose axis. New sweep:
`forgetting/scripts/fidelity_heldout_ce.py` scored all 1,684 strip adapters (256 held-out pool rows,
completion-only masking identical to training); merge in
`forgetting/scripts/three_curves_strip_ds1.py`, artifacts in `outputs/analysis/fidelity_strip_ds1/`.
Headline: fidelity is U-shaped in dose (min 3.50 nats gain at K/K* 0.6–0.8, where the
trait peaks) and *recovers to its best values (≈5.0) in the dead zone* — collapse
destroys memorization and the base distribution but the output distribution collapses
onto the number task (train ≈ held-out ≈ 2.1 nats, vs alive high-dose train 0.01–1.0
with up to 5-nat overfitting gaps, per `collapse_postmortem_ce.json`, reproduced to
+0.08 nats). 76/1,684 adapters are tensor-verified non-finite (divergence, lr1e-3 ×
α ≥ 128) and have no preference eval. Ordered claim set with per-claim caveats:
`docs/results/collapse_claims_ordered.md` (`collapse_law.md` remains the results SSOT).
Gotcha: `find`/`os.walk` under-report files inside the 932 large checkpoint dirs on this
filesystem — use `glob`/`exists()` (the first sweep pass silently saw 752 of 1,684
adapters).

**Claims notebook on shared support (2026-08-19; renumbered 2026-08-20).**
`notebooks/collapse_claims.ipynb` (builder `forgetting/scripts/build_collapse_claims_nb.py`)
recomputes each claim's headline numbers. Claim ids were renumbered into narrative order
on 2026-08-20 (final: C1 event, C2 law — the two halves of the old integrity section;
C3 generality; C4 trait; C5 HPs-set-size (A3, doc-only); C6 dose-separates-fate (A2+M;
θ̂ + AUC in the notebook); C7 Frobenius (A4, doc-only); C8 fidelity — C8b ladders/shuffle,
C8c demo; C9 doctrine, doc-only); pre-08-20 notes use the old ids (old C2 → C1/C2,
old C7 → C3, old C3 → C4, old C1/C6 → C6, old C4/C4b/C5 → C8/C8b/C8c). As of 2026-08-19
the adaptive-era law-campaign/krungs populations are no longer load-bearing: C3's K*/b
refits run **integrity-native** (probe-KL label, prior/e³ removed there) on probe-covered
**core65** runs (focus20 ∪ band48 ∪ lowk9; loaded via `tf.load_focus_runs` over the
shared 102-teacher list from `run_logs/campaigns/strip_probe_lists/xarm_bandext126_runs.txt`) and
self-update as probe records land; single-arm depth stays on the strip; C6's θ̂ and
dose-AUC use all norm-covered ep20 cells (the one legacy-cell remnant, dog-only until a
norm harvest covers the other animals). Headline (the live census is the 2026-08-20
02:26 execution at full `bandext126-pileprobe` drain, 13,650 probe-covered runs;
supersedes the 08-19 interim numbers quoted here previously): per-animal
integrity-native K* 8.64–9.09e-4 (1.05× spread), b 0.565–0.573; per-arm K* 1.22× over
the 18 arms with n_dead ≥ 20 (b 0.542–0.586). C8's memorization decomposition uses the
full 1,602 v2 endpoint records; C8b computes the ladder ("never had it") and shuffle
(task-manifold conditionality) results in-cell. The dead label is the declared
convention `probe_io.DEAD_KL_CUT = 4.0` (2026-08-20; 5→4 later that day): the cut-derivation program (density
valley, fences, mixtures, dose changepoints) was abandoned as circular — each variant
presupposed the labeling, the population composition, or the dose law; an openly arbitrary
constant with a mandatory sensitivity line replaced it. The alive bulk ends by ~5 nats in
every measured slice (campaigns notebook §0c histogram / §0d per-slice panels), so the
convention is safe; `derive_dead_cut` survives only as a shape diagnostic.

### Checkpoint-grid factorization (`analysis/checkpoint_grid_factorization.py`, 2026-09-25)

Turns the ad-hoc two-factor analysis of the teacher×student **checkpoint grids** into a tested
module plus `scripts/checkpoint_grid_factorization.py`
(`uv run python -m scripts.checkpoint_grid_factorization --animal dog --lineage orig`). It loads
one same-lineage grid from `students/{animal}/ckpt/…/ftseed*/preference_eval.json` (five teacher
recipes × 3 seeds per cell; per-cell mean = seed mean within recipe, then unweighted recipe mean,
with the matching per-cell SE), then scores the rank-one product `a(t)·b(s)` against its rivals —
additive, bottleneck `m(min(t,s))`, mirror `m(max(t,s))`, product + diagonal bonus, product +
nearness kernel `exp(−|t−s|/L)`, rank-2 SVD — each with its free-parameter count, on the
probability and the log scale, anchored by a seed-noise ceiling. `asymmetry_test` splits the
off-diagonal into symmetric/antisymmetric parts and asks how much of the antisymmetry is seed
noise and how much the two curves already reproduce. Cross-lineage helpers do the same rank-one
question for a lineage×lineage matrix (control-subtracted against matched base controls,
off-diagonal mask, cycle/transitivity tests, partner-independence ratios, teacher bootstrap).
Dog/orig 13×13: product R² 0.964 vs additive 0.794, bottleneck 0.914, mirror 0.243, product+diag
0.974, product+nearness 0.982 (L = 10k, c = +0.107), rank-2 0.988, ceiling 0.995 — the two
factors, plus a short-range nearness bonus, carry the grid. Held-out evaluation (2026-10-08, notebook §2f of `checkpoint_distillation_analysis.ipynb`): `discover_checkpoint_grids` (full-coverage recipes only), masked/fixed-factor `fit_product_proximity`, `cell_cross_validation`, `diagonal_heldout`, `leave_one_recipe_out`, `heldout_ftseed`, `factor_borrowing`, `factor_summary`, `factorization_table`; dog/orig product+proximity CV R² 0.972, diagonal-held-out 0.962.

### Record cache (`analysis/record_cache.py`) + adapter-norm run-dir records (2026-08-20)

`analysis/record_cache.py` generalises the pile-probe-only `analysis/probe_cache.py` (kept as
a thin facade) into a registry of record **kinds**: each kind names a record file relative to a
run dir, an extractor from the record dict to flat scalar columns, and a cache CSV under
`{output_root}/analysis/`. Kinds today: `pile_probe`, `fidelity_heldout`, `fidelity_shuffle`,
`fidelity_traj`, `pile_coherence_v1`, `json_blob` (parameterized), `adapter_norms`,
`adapter_norms_traj`, `cos_w0` (2026-08-25: `{run}/eval/cos_w0.json` from
`forgetting/scripts/teacher_cos_w0.py` → `mean_cos`, `pooled_cos`, `n_modules`, `nonfinite`).
Contract: run-dir records are the SSOT and the CSV is derived, so a
refresh pays one `stat` per inventory path and re-reads only records whose `(mtime, size)`
changed (a retrain-and-recompute is never missed); the saved CSV is a **union** — rows outside
this call's inventory are never evicted, so notebooks with disjoint inventories cannot destroy
each other's work, while the returned frame is always filtered to the request. Writes go
through `utils.io.atomic_write_text`/`atomic_write_json` and no lock is taken (a lost update
only costs a re-read). Per-record read errors keep the previously cached row; the one-line
summary counts added/refreshed/missing/unreadable.

**Adapter-norm migration.** The 441 MB frozen harvest
`analysis/adapter_norms_sweep_all_arms_all_teachers.json` (476,448 module rows) was migrated
to per-run records by `forgetting/scripts/migrate_adapter_norms_to_rundirs.py` (idempotent; `--dry-run`
default-off, never overwrites an existing record, skips + reports absent run dirs, compact
JSON): `{run_dir}/eval/adapter_norms.json` = `{kind, schema_version, modules (run-level fields
hoisted), test_p, rank, alpha, wrong_seed_offset, provenance, manifest_ref}`, where
`manifest_ref` was read at *migration* time and flagged `manifest_ref_read_at_migration` — it
identifies the run as it stands, not as it was harvested. Endpoint and trajectory data are
never mixed: `--checkpoints`-mode step-bearing rows go to `{run_dir}/eval/adapter_norms_traj.json`
(`points: [{step, modules}]`) under its own kind and loader. `forgetting/scripts/adapter_norms_probe.py`
now writes both records as it harvests (its analysis JSONs are unchanged). Discovery is the
written inventory `analysis/adapter_norms_runs.json` (`runs` / `traj_runs`, atomic union
rewrite), read by `analysis/norms.py`: `load_adapter_norm_runs` / `load_adapter_norm_traj`
return the cached scalars (`dw_fro`, `dw_spec`, `dW_*`, `cos_*_mean`, `n_modules`, …) plus the
keys parsed from `run_path`. First migration (2026-08-20): 9,926 runs written, 0 skipped;
`dw_fro` verified equal to the old aggregate-side computation on all 9,926 runs (max relative
difference 8.9e-16, float summation order). `forgetting/scripts/build_collapse_claims_nb.py` C6/C6b now
use the loader; `forgetting/scripts/build_nors_asks_nb.py` still reads the aggregate at module level.
Tests: `tests/test_analysis_norms.py`, `tests/test_record_cache.py`.

**Schema-migration rule (2026-08-20).** Adding a column to a kind is a cache-side change the
`(mtime, size)` freshness test cannot see: the records on disk are unchanged, so every existing
row would keep NaN in the new column forever. `load_record_cache` therefore clears
`record_size` on all cached rows whenever it adds a missing column, which routes them through
`_fresh`'s legacy branch for one full re-read (`test_new_kind_column_forces_one_full_reread`).
Found when `ce_student` / `ce_base` were added to `pile_probe`: 20,126 of 20,618 rows were NaN
with finite records; backfilled the same day (pre-backfill copy
`analysis/probe_cache.csv.bak-20260820-prebackfill`).

**Warm-sweep cost (2026-09-10).** A 40-call notebook (`transfer_receiver_family`) was spending
~40% of its 14-minute execution rewriting caches that had not changed. Two causes, both in
`load_record_cache`: pandas' default CSV float parser moves ~12% of stored `record_mtime`
values by one ulp, so under the exact-mtime staleness test those rows read as changed on
*every* sweep (re-read from cold-1, counted as "refreshed", and forcing a full rewrite of the
28 MB probe cache and the 224 MB `pref_summary` blob cache on every call); and each call
re-parsed those CSVs from scratch. Fixes: the CSV is read with
`float_precision="round_trip"` (the writer was already exact — the 50 µs same-size rewrite
test keeps its meaning and `_MTIME_TOL` stays 0); the parsed frame is memoised per process
keyed on the cache file's `(mtime_ns, size)`, so a rewrite by another process still
invalidates it; the stat/read sweep runs on a 64-thread pool
(`RECORD_CACHE_SWEEP_WORKERS`; cold-1 stats cost ~5.5 ms serially, and 10k of them take
6.8 s at 32 threads, 3.0 s at 64, 2.9 s at 128); a rewrite is triggered only by new/changed
rows or the eviction of a *cached* row (an inventory path that never had a record used to
land in the eviction set on every call, and 33 unevaluated cells were forcing the 224 MB
rewrite each time); and the persist serializes through pyarrow when importable (0.4 s vs
3.5 s for the blob cache, verified cell-identical through the round-trip reader; pandas
fallback when pyarrow is absent or refuses a mixed-type column). Measured on the dog original-sender load (10,284 seed-1 runs, 100 lineages, run
loader plus probe cache): warm call ~30 s → 4 s, with 0 refreshed rows and no rewrite. Tests:
`test_warm_sweep_reads_nothing_and_never_rewrites`,
`test_csv_memo_parses_once_until_the_file_changes`,
`test_unrecorded_inventory_path_does_not_force_a_rewrite`,
`test_serialized_cache_round_trips_cell_for_cell`.

### Teacher cos(ΔW, W₀) mediation screen (2026-08-25)

`forgetting/scripts/teacher_cos_screen.py` runs the student P2 mediation test on the teacher stage
(C10's open level-set failure): logistic `dead ~ log dw_fro` ± the 4-HP block
(log lr/α/r/steps) ± `mean_cos`/`pooled_cos` from the `cos_w0` harvest, ftseed1 of the
12 teacher trees (n = 9,109, dog|orig n = 4,735), cluster-robust on the (tree, lr, r, α)
tag, cos z-scored for conditioning. Writes
`{output_root}/analysis/teacher_cos_screen.json` (fits, coverage per tree, AUC ladder,
cos-by-rank/steps/fate descriptives + a student-side by-rank reference from the ds1
wfinal sweep). Result: cos does **not** close teacher mediation — dog|orig HP-block
cluster-Wald p 2.0e-47 → 5.4e-47 with cos, rank coefficient 4.80 → 4.05 (students:
p 1e-13 → 0.03–0.6, rank 1.52 → 0.42); teacher cos is steps-structured, not
rank-structured, and its dose-conditional sign is positive (aligned = deader), the
opposite of students.

### Unigram anchor for the integrity cut (2026-08-23)

`forgetting/scripts/build_unigram_anchor.py` writes
`{output_root}/analysis/unigram_anchor_pp1.json` (+ `.counts.npz`): H_uni —
the CE of a context-free Pile-val unigram (50k docs disjoint from pp1, same
skip/truncation rules) on the exact pp1 CE-target multiset — and per-base
crossings Δ_cross = H_uni − ce_base ("zero net context benefit on Pile
text"), read from existing `pile_probe.json` records. This is the external
semantic anchor for `DEAD_KL_CUT` (`collapse_law.md` §1.2): Δ_cross lands
inside the bimodal transit valley on every base (5.06–5.47 nats), so the
declared 4.0 sits just below the context-free crossing; anchored-cut
companions are quoted beside declared-cut fits (K*(c) level-set sections:
`tierb_strips_integrity.ipynb` §4, `pythia410m_collapse_and_peak.ipynb` §7).
Program-level synthesis: `docs/results/collapse_program_synthesis.md`.

**Cross-tokenizer pools `pp1t-<tag>` (2026-09-05).** To anchor a base outside
the GPT-NeoX/OLMo families, pp1's 32 Pile-validation documents are frozen as
TEXT — `data/probe_pools/pp1_text.jsonl` + `.meta.json`, written by
`forgetting/scripts/freeze_pp1_text.py`, each doc cut at the character offset of pp1's
last Pythia token and content-hashed (`text_sha256`) — and retokenized per
family by `probe_io.build_frozen_text_pool`. `coherence/probe_families.py`
holds the table: `pythia`, `olmo` (the pinned DataDecide branch tokenizer),
`qwen25`, `llama3`, `gemma3`; BOS is prepended EXPLICITLY for the families
that expect one (llama3, gemma3), never via `add_special_tokens=True`.
`build_unigram_anchor.py --pool pp1t-<tag>` fits the unigram on the SAME
seed-0 Pile-val stream (offset by the pool's `n_scanned`) under the family
tokenizer, and `--probe-base <ref>` measures `ce_base` directly (one pool
forward pass, provenance in `ce_base_sources`) for bases with no
`pile_probe.json` record yet — it works on pp1/dp1 too. Per-token H_uni is
NOT comparable across tokenizers (different vocabularies segment the same
text into different token counts); the cross-family quantity is
`h_uni_bits_per_byte_approx`. `probe_io.probe_pool_spec_for_repo` routes a
known family's base repo (alias table `probe_families.BASE_REPO_FAMILY`, plus
the `pythia-*`/DataDecide patterns) to that family's pool and RAISES on an
unknown repo rather than falling back to pp1, and both direct probes and
verify/extend anchor runs are gated on pool-sha identity — a base whose own
tokenizer does not reproduce the pool, or a record built on a different pool,
stops the run.

*The DCLM corpus pool `cp1` (2026-09-15).* The `dclm` local family
(`pythia-160m-dclm-seed{1234,1,2}`) is re-pretrained on DCLM-baseline-1.0, so the Pile is
off-distribution for it and the pp1 Pile unigram anchor is the wrong bar: those bases get
`cp1`, pp1's recipe (32 documents, seed-0 permutation, truncated to 512 tokens, min 32,
position 0 never eligible) on the held-out DCLM shards kept out of the training slice. The
source is a Megatron indexed dataset ALREADY tokenized with the 20B Pythia tokenizer
(paths key `dclm_heldout_document`, machine-local), so cp1 is defined on those token ids —
no detokenize/retokenize round trip — with the trailing EOD separator dropped; the module
is `coherence/dclm_pool.py` and the 32 selected documents are frozen to
`data/probe_pools/cp1_tokens.jsonl` (+ `.meta.json`, token sha
`91d7b1740becd43d…`) by `forgetting/scripts/freeze_cp1_tokens.py` so the pool builds on nodes without
the source mounted. Dispatch is by PRETRAINING CORPUS before tokenizer family:
`probe_io.probe_pool_spec_for_repo` (and `ckpt_pile_kl_trajectory.pool_spec_for`) send a
Pythia spec whose `variant` is `dclm` to cp1 via the one-row `PYTHIA_VARIANT_POOLS` table
and everything else to pp1, and `resolve_probe_pool_spec` accepts `--pool cp1` only for
those bases. The anchor is `analysis/unigram_anchor_cp1.json` (+ `.counts.npz`), built by
`build_unigram_anchor.py --pool cp1`: the fit stream is the same permutation after the
pool's `n_scanned`, 50k documents, same rules, and it needs the source stem.

*The DCLM frozen TEXT and `cp1t-<tag>` (2026-09-18).* The DataDecide `dclm-baseline`
bases are pretrained on DCLM, not Dolma, so dp1 is off-distribution for them; they need
cp1's documents under the OLMo tokenizer. cp1's 32 documents are therefore ALSO frozen as
TEXT — `data/probe_pools/cp1_text.jsonl` (+ `.meta.json`, text sha `aca044fcf1713351…`),
built by `forgetting/scripts/build_cp1_text.py` from the frozen `cp1_tokens.*` artifact (CPU-only, no
source mount): the same record schema and truncation semantics as `pp1_text` (the text is
the decode of the document's ids AFTER cp1's 512-token cut), plus provenance (source pool
`cp1` and its pinned token sha, Megatron stem, decode tokenizer `EleutherAI/pythia-160m`,
per-doc original token counts). The builder re-encodes every frozen text and FAILS unless
the ids come back exactly (32/32 exact). `cp1t-<tag>` is that text retokenized per family,
the pp1t mechanism on a second corpus: `probe_families.CP1T_PREFIX`/`is_cp1t`/
`cp1t_pool_version_for` know the versions structurally, `probe_io.frozen_text_paths` picks
the artifact and `pinned_frozen_text_sha` the pin (`CP1T_TOKEN_SHA`), and the pool's
`source` reads `cp1_text:…`. Only `cp1t-olmo` is built; the DataDecide tokenizer (GPT-NeoX
BPE plus specials) segments these 32 documents identically to Pythia — identical ON THE POOL,
measured, not a claim that the two tokenizers agree in general — so its pin IS
`CP1_TOKEN_SHA`. On the 50k-document unigram FIT stream they do differ, by 464 tokens.
There is no `cp1w`.

*DataDecide routing is by RECIPE (2026-09-18).* One data recipe = one corpus pool:
`dolma1_7` → `dp1`, `dclm-baseline` → `cp1t-olmo`, `c4` → `c4p1` (`probe_io.dd_recipe_pools`, keyed on
the recipe of the base's spec via `parse_dd_ref`/`base_spec` — or, for a bare hub repo id,
`datadecide.dd_recipe_of_repo_id`; never by string-matching a path). It is the olmo-family
analogue of `PYTHIA_VARIANT_POOLS`, and it drives `probe_pool_spec_for_repo`,
`CORPUS_POOLS["olmo"] = ("dp1", "cp1t-olmo", "c4p1")`, `native_probe_pool_version` (hence the
`pile_probe.<pool>.json` sidecar naming), `run_final_evals`, `ckpt_pile_kl_trajectory.
pool_spec_for`, `teacher_pile_kl_trajectory` and `ckpt_teacher_pile_probe`. The olmo
family OWNS all three pools, so `resolve_probe_pool_spec` accepts any of them explicitly on any
DataDecide base (`--probe-pools cp1t-olmo` on a dolma base, `dp1` on a dclm one) — the
other recipe's pool then lands in a sidecar. The anchor is
`analysis/unigram_anchor_cp1t-olmo.json` (+ `.counts.npz`), built by
`build_unigram_anchor.py --pool cp1t-olmo`: cp1's fit stream (same permutation after the
pool's `n_scanned`, needs the source stem) decoded with the Pythia tokenizer and
re-tokenized by the family tokenizer, so the counts live in the OLMo vocabulary. Built
2026-09-18 (50,000 fit docs, 20.96M tokens, vocab 50,280, 12,552 CE targets):
H_uni = 7.4901 nats — cp1's 7.4901 to four decimals, the two tokenizations of this corpus
differing by only 464 of 20.96M tokens — and ce_base 2.5075 / 2.5118 / 2.5078 for the `dclm-baseline` 1B seeds
default / large-aux-2 / large-aux-3, i.e. anchored cuts Δ_cross ≈ 4.98 nats.

*The `c4` recipe and the `c4p1` pool (2026-09-19).* `dd-1b-c4[-seed{2,3}][@step{N}]` names
`allenai/DataDecide-c4-1B`, a third recipe repo with the same 104 branches and the same
final step 69369; the segment is registered in `DD_RECIPE_SEGMENTS`, so every enumeration
(token grammar, error menus, `DD_RECIPES`, dir names, routing) follows from the registry.
Its pool is `c4p1` (`coherence.c4_pool`): 32 documents of ≤512 OLMo tokens from a frozen
60,000-document universe cached under `{dataset_cache_root}/c4_c4p1/`, drawn from the C4
`en` VALIDATION shards fetched by explicit URL (never `load_dataset`). Unlike dp1 it is
genuinely HELD OUT — C4 validation is disjoint from the train split the recipe is a subset
of — so it plays pp1's and cp1's role, not dp1's. The universe is the first 60,000 documents
of those shards in index order, content-sha stamped in the sidecar; both the dp1 and the
c4p1 universe sidecars now carry `universe_sha256` and every load verifies it.
`C4P1_TOKEN_SHA` was pinned 2026-09-20 on the first real
`build_unigram_anchor.py --pool c4p1` build, which wrote
`analysis/unigram_anchor_c4p1.json` with the three final-step `c4` bases probed directly in
fp32: H_uni = 7.705 and ce_base 2.535 / 2.537 / 2.540 for default / large-aux-2 /
large-aux-3, i.e. anchored cuts Δ_cross ≈ 5.17 nats. The recipe axis is an axis, not a new
population: `c4` shares its init byte-for-byte with `dolma1_7` at every full-budget seed
(99/99 tensors) and differs only in data order (Jaccard ≈ 0.20 against dolma at step 2500).

*One place the pool choice lives (2026-09-18).* `coherence/probe_io.py` owns the whole
decision and every site calls it — there is no second family table anywhere in the repo.
`native_probe_pool(base)` takes ANY base spelling (a `--base-models` token, an HF ref, an
output-tree dir name, or a `BaseSpec`, all resolved through `models.naming.base_spec`) and
returns the pool that base must be measured on; `native_pool_version_for_base(base)` is its
version string (distinct from `native_probe_pool_version(run_dir)`, which reads a run's
manifest). `compatible_probe_pools(base)` lists every tokenizer-compatible version, native
first, and `require_compatible_probe_pool(base, version)` is the gate — `resolve_probe_pool_
spec` is just "auto → native, else require". `probe_pool_spec_for_version` knows every wired
version (pp1, cp1, dp1, dp1w, `pp1t-`/`pp1w-`/`cp1t-<tag>`). Errors are named ValueErrors
(`UnknownProbeBaseError`, `IncompatibleProbePoolError`), never a KeyError and never a silent
fallback to pp1. The hand-rolled copies that used to re-derive this — in `frozen_a_ablation`,
`weightspace/dose_census`, `ckpt_pile_kl_trajectory.pool_spec_for`, `run_final_evals
--probe-pools` and `analysis/transfer_significance_io` (whose `probe_pool=` arguments now
default to None = each receiver's native pool) — all call the API instead, so a new recipe or
family is wired once.

*Nested wide pools `pp1w-<tag>` / `dp1w` (2026-09-05).* Each is 128 documents =
the base pool's 32, byte-identical and in order, plus 96 drawn from the SAME
seeded stream but from after the frozen unigram fit window (`pp1w` from Pile
permutation position 53,281 = pp1's `permutation_offset` 33 + `n_docs_scanned`
53,248, frozen as `data/probe_pools/pp1w_text.jsonl` by
`freeze_pp1_text.py --wide 128`; `dp1w` from Dolma seeded position 53,280),
so the wide pool stays disjoint from the counts it is scored against and the
base pool's `n_scanned` — hence the fit window — is unchanged. The wide anchors
REUSE the base anchor's `.counts.npz` and never refit, gated on the wide pool
nesting its base token-for-token and on the reused counts reproducing the base
record's `h_uni_primary` (1e-9). `probe_io.doc_ce`/`subset_mean` give per-document
CE, so one 128-doc forward pass yields both the wide `ce_base` and the 32-doc
`ce_base_subset` (printed against the base anchor's value; >1e-6 is a loud
warning) with `h_uni_base_subset`/`delta_cross_subset` alongside. `PP1W_TOKEN_SHA`
and `DP1W_TOKEN_SHA` were pinned from the first real builds (2026-09-05; all six
rebuild exactly). The per-family token-count floor inside `build_frozen_text_pool`
is a sanity check only (≥3 tokens, one CE target): min-32 is pp1's freeze-time
selection rule in Pythia tokens and is deliberately NOT re-applied per family,
because the frozen text is the pool's identity (one extension doc is 31 Qwen
tokens). One extension document does not round-trip under Pythia (the 512th-token
cut fell inside an emoji, +2 tokens on re-encoding); the text is what is frozen.

Anchors built 2026-09-05 (official tokenizers, all five pins in `PP1T_TOKEN_SHA`
rebuild exactly; `ce_base` by `--probe-base` on cuda, fp32; base models only):

| pool | base | V | targets | H_uni nats | H_uni bpb | ce_base nats | ce_base bpb | Δ_cross |
|---|---|---|---|---|---|---|---|---|
| pp1 / pp1t-pythia | pythia-160m | 50277 | 10212 | 7.8045 | 2.617 | 2.7082 | 0.908 | 5.096 |
| pp1t-olmo (Pile) | DataDecide-1B seed-default | 50280 | 10212 | 7.8045 | 2.617 | 2.2727 | 0.762 | 5.532 |
| dp1 (Dolma) | DataDecide-1B seed-default | 50280 | 12075 | 7.7120 | — | 2.5264 | — | 5.186 |
| pp1t-qwen25 | Qwen2.5-1.5B | 151665 | 9968 | 7.9435 | 2.599 | 2.1417 | 0.701 | 5.802 |
| pp1t-llama3 (BOS) | Llama-3.2-1B | 128256 | 9657 | 8.1572 | 2.586 | 2.2626 | 0.717 | 5.895 |
| pp1t-gemma3 (BOS) | gemma-3-1b-pt | 262145 | 10267 | 7.6781 | 2.588 | 2.2773 | 0.768 | 5.401 |

Two facts fall out. (i) pp1t-pythia reproduces pp1 exactly (same token sha, H_uni to
all digits, ce_base to 1e-8), and the DataDecide tokenizer is the GPT-NeoX BPE plus
three specials, so pp1t-olmo is ALSO token-identical to pp1: dp1 vs pp1t-olmo compares
corpora at a fixed tokenizer. On its own Dolma sample the OLMo base is 0.25 nats
worse than on Pile-val (2.53 vs 2.27) while the unigram entropy is 0.09 nats lower,
so the Pile-based cut admits 0.35 nats (6.7%) more excess than the Dolma-based one
(Δ_cross 5.53 vs 5.19). Relabelling the OLMo ladders under the Pile cut needs the
runs re-probed on pp1 tokens (their records are dp1); not done. (ii) H_uni in bits
per byte is 2.59–2.62 across all four tokenizers (same text, context-free model),
the check that retokenization preserved the text; the per-token H_uni spread
(7.68–8.16) is tokenizer granularity, which is why the per-family Δ_cross is the
operative cut and never a shared per-token bar.

Wide (128-doc) anchors, built 2026-09-05 on the nested pools (`Δ32` is the 32-doc
subset read from the same forward pass; every subset `ce_base` matches its base
anchor to 0.0, dp1w to 6e-8 because dp1's value came from a harvested record):

| pool | base | targets 32 / 128 | H_uni 32 / 128 | ce_base 32 / 128 | Δ32 / Δ128 |
|---|---|---|---|---|---|
| pp1w-pythia | pythia-160m | 10212 / 45843 | 7.8045 / 7.8191 | 2.7082 / 2.6360 | 5.096 / 5.183 |
| pp1w-olmo (Pile) | DataDecide-1B | 10212 / 45843 | 7.8045 / 7.8191 | 2.2727 / 2.2691 | 5.532 / 5.550 |
| dp1w (Dolma) | DataDecide-1B | 12075 / 43905 | 7.7120 / 7.6916 | 2.5264 / 2.4254 | 5.186 / 5.266 |
| pp1w-qwen25 | Qwen2.5-1.5B | 9968 / 44493 | 7.9435 / 7.9770 | 2.1417 / 2.1044 | 5.802 / 5.873 |
| pp1w-llama3 | Llama-3.2-1B | 9657 / 43167 | 8.1572 / 8.1936 | 2.2626 / 2.2600 | 5.895 / 5.934 |
| pp1w-gemma3 | gemma-3-1b-pt | 10267 / 46204 | 7.6781 / 7.6705 | 2.2773 / 2.2557 | 5.401 / 5.415 |

Widening moves Δ_cross by +0.01 to +0.09 nats (0.3–1.7%), i.e. inside the anchor's
own α-sensitivity band, so the 32-doc labels are not at risk from pool size; the
OLMo Pile-vs-Dolma gap persists at 128 (5.55 vs 5.27). H_uni in bits per byte is
2.70–2.73 at 128 across the four tokenizers (2.59–2.62 at 32; the extension docs
carry ~4% more tokens per byte), again the retokenization consistency check.
pp1/dp1 remain the operative labels for the existing arms; the wide pools are the
default for the new families and the reviewer-facing robustness check for the old ones
(a re-probe of the label-sensitive band, not done).

### Training-retained axis: Pile CE of the public pretraining checkpoints (2026-09-09)

Pythia and PolyPythias publish every base's full checkpoint trajectory, so a
student's Pile CE can be placed on its own base's pretraining curve with no fitting:
**training retained** = the earliest pretraining step at which the base itself had
already reached the student's CE (log-step interpolation between bracketing
checkpoints; first crossing wins under a noisy tail). This is a continuous,
interpretable damage axis; the unigram anchor H_uni = 7.8045 is one point on it
(the CE of a very early checkpoint), and dead/alive is the binarisation at that point.

`forgetting/scripts/base_ckpt_pile_ce.py` measures each `@step{N}` revision (downloaded into the
configured HF cache on first use) on the nested `pp1w-pythia` pool and writes
`{output_root}/analysis/base_ckpt_pile_ce/{base_dir_name}.json`: per step `ce_pp1`
(32-doc subset — the students' readout pool, token-identical) and `ce_pp1w` (128-doc)
from one forward pass, plus per-document NLL sums. Default grid = every 5,000 steps to
143,000 plus the log-spaced early checkpoints (0, 1, 2, …, 512, 1000), for
`original,data-seed1,full-seed1`; re-runs skip recorded steps. Readout:
`coherence.base_ckpt_ce.load_curve(output_root, base)` and
`training_retained(ce, curve)` / `training_retained_fraction`. Cost is a post-processing
pass: ~20 s per checkpoint on a shared A40 (dominated by weight loading), 41 checkpoints
per base.

### Notebook-code encapsulation pass + label API (2026-08-26)

An audited sweep moved the notebooks' copy-pasted code into the analysis
package and made the anchored cut the one operative dead label everywhere.

**`analysis/labels.py` is the single label home.** Doctrine (binding):
operative dead/collapse label = the per-receiver anchored cut
(`labels.anchored_cut(output_root, base)` → Δ_cross; batch `anchored_cuts`,
row-wise `label_dead_kl`); the declared `probe_io.DEAD_KL_CUT` = 4.0 survives
only as an explicitly labelled *comparison* (`*_cmp` columns, "declared 4.0
(comparison)" plot labels); trait cuts (0.005/0.01) are legacy, allowed in old
sections only via `label_dead_trait_legacy` + a caveat comment; the prior/e³
rule (`test < prior·e⁻³`, `transfer_factors.prior_dead_thresholds`) is
**removed repo-wide** — the deleted function's only remaining mention is the
removal note in the labels docstring. `labels.kstar_vs_cut` is the shared
K*(cut) level-set sweep. `tf.load_core65_population` (the collapse_claims
C3 loading path as one call) returns runs with per-receiver `delta_cross`.

**New shared loaders/templates** (see the `analysis/` leaf listing above):
`run_evals.load_run_evals` (run-dir eval contract; diverged ⇒ kl=inf/test=0;
kl_source = probe|traj_end|none), `student_tree`, `trajectories.endpoint_stats`
/`pref_kl_pairs`, `frames.read_jsonl/read_cell_manifest/read_queue_jobs/
load_base_prior`, `plots.KLadder`+`house_grid` (the K/K*-ladder heatmap that
was copy-pasted across the strip notebooks), semantic palettes
(`ANIMAL_COLORS` = the Okabe-Ito animal mapping, now canonical).

**29 notebooks were migrated in place** (sources edited, outputs untouched —
so stored outputs are stale wherever the label semantics changed; the affected
notebooks and the re-execution order are listed in the 2026-08-26 session
report). Builder scripts (`scripts/build_*_nb.py`) were mirrored where their
cells still matched; the four that wrote their notebook at import time
(ckpt_trajectories, collapse_claims, collapse_teachers, seed2seed) now write
only under `__main__`. `build_collapse_mechanism_nb.py` is marked STALE (the
live notebook has diverged far beyond it — regenerating would lose work).

2026-08-27 review fixes: `tf.load_core65_population` now takes its teachers
from the core65 design's shared top-10 list (`teachers=` overridable) instead
of the frozen xarm inventory, which had a 102/20/10/10 per-animal imbalance
(the inventory remains the probe-refresh list only); `labels.kstar_vs_cut`
excludes NaN-KL rows; `run_evals` materializes `seeds`; `trajectories` handles
`nonfinite` KL points (→ inf) and restricts `pref_kl_pairs` to KL support;
`KLadder.bin` no longer folds below-`lo` values into the first bin;
`forgetting/scripts/teacher_cos_screen.py` fits the anchored label (`dead4_cmp` kept);
notebook prose that still described the flat 4.0 as operative was swept; the
teachers builder regained sync (its live-only §9 mirrored, 28/28 cells).

### Pile-probe measurement + teacher-tree aggregation (2026-08-20)

The probe measurement has one implementation, `coherence/probe_io.py`: `pile_logprobs(model,
pool, device, batch_size=8)` (next-token log-probs at the pool's eligible *logit* positions —
position 0 excluded, last position kept; right-padded batches, exact for a causal LM with
arange `position_ids`), `mean_kl(ref, other)` (position-pooled `KL(ref ‖ other)`, nats) and
`mean_ce(logp, pool)` (CE of the true next token, last eligible position dropped). They
replace the private `_pile_logprobs` / `_mean_kl` (`forgetting/scripts/frozen_a_ablation.py`) and
`_mean_ce` (`scripts/pile_probe_final.py`) that `ckpt_pile_kl_trajectory`, `direction_surgery`,
`run_final_evals` and `pile_probe_final` imported across scripts. Verified against the old
per-doc path on pythia-160m: KL(old‖new) 6e-8 nats, CE Δ 2e-6, a real inter-model KL agrees to
1.6e-7 — old and new `pile_probe.json` records are interchangeable. Cost of the speed-up
(~2×): the per-doc log-prob tensors stay on the GPU, ≈3.3 GB per set on pp1 (32 × ≤512 ×
50 304 fp32), so scripts that hold a base set plus per-cell sets `del` them as soon as the KLs
are taken (frozen-A: base + orig + ablated = 3 sets live per cell; trajectory: base + adapted).
`pile_probe_final` / `run_final_evals` hold one base set (rebuilt when the base repo changes)
plus one student set. Budget this under MPS multi-tenancy.

Teacher trees have no campaign manifest, so `analysis/probe_cache.load_teacher_probe_cache
(output_root, tree)` discovers the inventory by `glob("*/ftseed*")` under
`teachers/{animal}/{base}` (glob, not `os.walk`), runs it through the same `pile_probe` record
cache, and adds `param_tag` / `ftseed` parsed from `run_path`. `scripts/aggregate_tree_probes.py
--tree teachers/dog/pythia-160m` is the CLI: it writes `{tree}/pile_probe_aggregate.csv`
(tag/ftseed identity, `kl`, `ce_student`, `ce_base`, `delta_ce`, `nonfinite`, pool + manifest
identity) as a convenience export; the run-dir records remain the SSOT.

### CKA (`cka/`, `pipelines/cka_pipeline.py`, `scripts/compute_cka.py`)

Minibatch linear CKA between the PolyPythia base models (the estimator is a PyTorch
port of Google Research's TF `MinibatchCKA` from
`do_wide_and_deep_networks_learn_the_same_things`): is the teacher base
(pythia-160m) representationally closer to same-init (data-seed) than to same-data-order
(weight-seed) student bases, on Pile vs on our number-sequence transfer data? The readout
is the pair-group CONTRAST bracketed by controls, never absolute values.

- `cka/minibatch.py` — faithful PyTorch port of Google's `MinibatchCKA` (Székely–Rizzo
  U-centered Grams → unbiased HSIC₁; across-models accumulators; sums divided only in
  `result()`; fp32; m ≥ 4 enforced) + a direct full-batch reference the tests pin the
  port against to 1e-5.
- `cka/pool_io.py` — sequence pools encoding the position rules: position 0 (attention
  sink) never eligible; numbers condition samples only digit-bearing completion tokens.
  `cka/extract_io.py` — raw post-block residual capture (13 slots; deliberately NOT
  `output_hidden_states`, whose last slot is post-final-LN; since 2026-09-09 the forward
  runs the bare `gpt_neox` core, not the CausalLM head — identical slots, but no unused
  `[B, T, vocab]` fp32 logits, which cost ~26 GB per 256×512 Pile batch and OOM'd pair
  runs on shared GPUs), position-row gathering,
  mean-pool rows, and top-r PC fit/projection. The capture itself now lives in the shared
  `models/residual.py` (`capture_raw_activations`), which `steering/vectors.py` also uses.
- Pipeline: all 7 models resident, one forward per model per batch; per-model
  preprocessed-Gram stacks shared across all pairs × variants (raw, pc1/3/5-ablated,
  meanpool); floor control = Gram permutation (commutes with U-centering). Pairs:
  6 primary (ref vs each variant), 6 within-arm, 7 self (ref self = ceiling), 1 floor.
- Explicit two-model mode (`--model-a`/`--model-b`, `cka/model_refs.py`,
  `pipeline.run_explicit`): compare one arbitrary teacher against one arbitrary
  student (base OR finetuned run) on any pool. Each operand is a base-spec token or a
  path to a run dir; finetuned refs read their base from `train_manifest.json` and
  merge the LoRA adapter (so the bare-`gpt_neox` extractor still applies). Pairs are
  primary(A,B) + ceiling(A,A) + self(B,B, dropped when B is the same checkpoint) +
  floor; the run identity records both operands' provenance (incl. checkpoint hashes)
  so adapter drift trips the gate. Tag: `{condition}__{labelA}__vs__{labelB}`. The
  base-suite path is byte-for-byte unchanged.
- Student-vs-its-teacher mode (`--student <run dir>`, repeatable, `cka/layout.py`,
  2026-07-31): `--student` accepts K run dirs of ONE teacher — a single pass forwards
  the teacher once per batch and shares its Gram stacks across all K students'
  pairs (`pipelines.run_students`); per-student artifacts and run identities are
  byte-identical to a single-student run, so batched and single runs are mutually
  resumable/skippable. It
  resolves the teacher operand AND the measurement pool from the student's own
  `train_manifest.json` — teacher checkpoint from (`animal`, `numbers_base_model_repo`,
  `teacher_param_tag`, `teacher_ftseed`); numbers pool = `{number_dataset_dir}/filtered.jsonl`
  with the manifest's `subsample_size`/`subsample_seed` (i.e. exactly the student's
  training rows), guarded by a hard sha256 check against the manifest's
  `filtered_sha256` (a regenerated pool fails loudly). Fixed labels `teacher`/`student`
  so every run dir carries identically-named artifacts. Base-control students compare
  against their un-finetuned numbers base; rng controls have no teacher (error).
  Enumerate/submit at scale with `scripts/sweeps/sweep_cka_students.py`.
- Output roots (restructured 2026-07-31, migrated by `scripts/migrate_cka_results.py`, which ran on 2026-07-31, script removed;
  log at `outputs/cka/_migration_20260731.jsonl`): base suite →
  `{output_root}/cka/base/{run_tag}/`; explicit pairs → `{output_root}/cka/pairs/{run_tag}/`;
  student mode colocates in the student run dir at `{run_dir}/cka/vs_teacher__{condition_tag}/`.
  Per run: per pair×variant the 13×13 matrix npy + fixed-scale [0,1] png (ref rows,
  input→output; `--no-plots` skips pngs) + convergence stack, plus `summary.json`
  (diag/band means per group) and `meta.json` (pool sha256, model refs).
- Smoke-verified 2026-07-30 (3 models, numbers pool): ceiling diag 1.000, floor ≈ 0,
  cross pairs in between, PC ablation desaturates raw CKA. Tests: `tests/test_cka.py`.

### Steerability screen — deferred / provisional

Blank-style steering-vector screen (extract `v = teacher − base`, inject via residual-stream
hooks, sweep layers × α with absolute specificity gates). Nested per-seed layout
(`{param_tag}/steerability/self/ft{N}.json` + derived `aggregate.json`; cross-seed cells
under `steer_{target-label}/`), per-seed adapter-hash cache keys, incremental seed adds.
Cross-seed steering injects a source direction into a different 160m base (same H/blocks/
tokenizer), with recovery measured against the source teacher's own lift. Artifacts are
provisional — don't treat aggregates as authoritative; known caveats: ungated `_peak`
fallback (`gated_frac` records gate hold rate), no uncertainty on the absolute gate. The
inherited-from-Blank final-slot basis mismatch is FIXED as of 2026-08-10: extraction runs
in the raw_blocks_v2 basis (`models/residual.py`; `steering/vectors.py` docstring), so
`v["raw"][1:]` gives one raw direction per block and all 12 blocks are steerable
(`DEFAULT_LAYER_SWEEP` now L0–L11). `_screen_params` keys the basis
(`activation_basis: raw_blocks_v2`), so every pre-change `ft*.json` (which lacks the key)
is invalidated and recomputes on the next run; the pre-change JSONs are archived, see
`outputs/legacy_snapshots/`.

### Moment-based similarity — Procrustes + PWCCA (`similarity/`, 2026-08-15)

Companion to `cka/`, **not** a replacement: minibatch CKA is unchanged and keeps its own
n×n Gram path. This package covers the measures that *cannot* be written as a per-batch
average — Procrustes distance and the CCA family (PWCCA). Procrustes needs the nuclear
norm of the full-sample cross-covariance and CCA needs full-sample whitening; neither is
an average of per-batch values, so the streaming loop accumulates **sufficient
statistics** instead of per-batch answers.

- `similarity/moments.py` — `ModelMoments` (per-layer `A = XᵀX`, `Xᵀ1`; one per model,
  shared across every pair) and `PairMoments` (`C = XᵀY`; layer-aligned diagonal by
  default, `full_grid=True` for the L×L grid). All are exactly additive over batches, so
  results are the exact full-sample statistic, not a minibatch approximation. Memory is
  constant in row count `n`. Centering is applied at finalization from the raw moments.
- `similarity/metrics.py` — `procrustes` (`tr A + tr B − 2‖C‖_*`; `svdvals` only, no
  eigendecomposition, plus scale-invariant `normalized_`/`angular_distance`),
  `build_whitener`/`Whitener` (one `eigh` per model-layer, reused across all pairs — the
  expensive part of the CCA family), `canonical_correlations`, and `pwcca`.

Cost, measured at d=768 / m=256 / 13 slots, 8 threads, full rank (numbers from a
contended box are meaningless — the sweeps saturate the CPU and inflate LAPACK calls by
~30×, so pin threads and measure on an idle node):

| stage | cost | multiplicity |
|---|---|---|
| accumulate (A, B, C) | ~22 s @ K=100 | per model-pair triple |
| `eigh` whiteners | 2.8 s | **once**, shared across all pairs |
| Procrustes finalize | 2.0 s | per pair (13 layers) |
| PWCCA finalize | 3.4 s | per pair (13 layers) |
| memory | 58 MB/model (A), 58 MB/pair (C) | constant in `n` |

Finalization is negligible against the forward passes; accumulation dominates. The moment
path does 3× the FLOPs of the Gram path (d²n vs n²d) but is *faster* in fp32 (9.1 vs
19.0 ms/layer) because `XᵀX` is a better-shaped GEMM than tall-skinny `XXᵀ`; in fp64 it
is at wall-clock parity with the existing fp32 Gram path. Accumulate in **float64** —
whitening squares `cond(X)` and residual streams carry outlier dimensions. Procrustes
uses `svdvals` (164 ms) over full `svd` (312 ms); an `eigvalsh(CᵀC)` route is ~15% faster
still but squares the condition number for a marginal gain — not worth it.

**Sample-size trap.** Centered-Gram rank is bounded by `n_rows − 1`, so with `n ≲ d` the
whitener truncates silently and CCA/PWCCA report near-perfect correlations between
*independent* representations (asserted in the tests). `build_whitener(..., n_rows=)`
raises in that regime; check `Whitener.rank` against `dim` otherwise. The pipeline's
25.6k rows against d=768 is comfortable — a single 256-row batch is not.

Two contracts that bite:
- **PWCCA weights sum over neurons, not examples** (`αᵢ = Σⱼ|⟨hᵢ, xⱼ⟩|` over feature
  columns). That is what makes the O(d²) moment form possible. It is also asymmetric —
  weights come from the A side.
- **Tied canonical correlations leave PWCCA ill-posed**: singular vectors are defined only
  up to rotation within a tied subspace and L1 weights are not rotation-invariant, so the
  value becomes basis-dependent (correlations stay exact). `pwcca` returns `min_rho_gap`;
  treat ≈0 as "not identified". Two correct implementations may legitimately disagree here.

Interpretation caveat: at the CCA end the measure is invariant to any invertible linear
map, so it discards exactly the norm/direction geometry the collapse work is about
(asserted in `tests/test_similarity.py`). Procrustes (Frobenius-normalized) is the one
that answers the same question CKA is being asked. Note also that CKA numbers are
minibatch-averaged while these are full-sample — a table carrying both is mixing
estimator bases.

Tests: `tests/test_similarity.py` (17) validates every metric against an independent
reference computed from raw `n×d` rows by a *different* numerical route — direct residual
minimization over orthogonal Q for Procrustes, QR rather than eigendecomposition for the
CCA family — so agreement tests the moment reduction rather than restating its formula.

### Measurement stack — `interp/` + similarity atlas / GKD-EAS / causal steering

Post-Stage-B, **forward-pass only** (no training). Mechanistic representation measurement
over the Stage B teachers/students: representation similarity, EAS retargeted to
`gkd_students/`, and causal sufficiency/necessity on the trait direction. Master experiment
design: `docs/MECHANISTIC_SL_EXPERIMENT_PLAN.md` §1. Method SSOT: Kornblith / Nguyen / Ding /
Raghu for global geometry; Okatan's teacher-basis subspace CKA is reported only on
shared-basis pairs as a hypothesis under test.

**Does not touch** Angela's off-policy `students/` / `teachers/` trees or their pipelines
(`eas_pipeline`, `sweep_compute_eas`, `steerability_screen_pipeline`). The only shared-file
edit is an additive `replace_base` mode in `steering/hooks.py` (`add` / `project` unchanged).

| Piece | Role |
| --- | --- |
| `interp/cka.py` | Unbiased minibatch CKA (HSIC1), biased HSIC0 diagnostic, Procrustes, SSA-topk, trait-coordinate similarity, shared-basis subspace CKA; torch OK (same exemption as `gkd/losses.py`). |
| `interp/activations_io.py` | Per-example 13-slot capture (last-token, left-pad, fp32), `.pt` cache, random-init floor factory. |
| `interp/pairing.py` | Stage B reverse lookup + 2×2 weight/data factors + basis validity (`SHARED` / `DIVERGED` / `BROKEN`). |
| `interp/layout.py` / `config.py` / `stats.py` / `results_io.py` | Run-set paths, frozen `InterpConfig`, Wilson + mean CIs, CSV loaders / seed collapse / similarity↔`subliminal_delta` join. |
| `pipelines/similarity_atlas_pipeline.py` | Families: base↔base, random floor, teacher↔base, student↔base, teacher↔student. Emits trait vectors + long `similarity_pairs.csv` + estimator diagnostics (HSIC0−HSIC1 gap, batch sensitivity, Davari top-1% norm exclusion, self-pair). |
| `pipelines/gkd_eas_pipeline.py` | EAS on `gkd_students/` leaves; resumable `eas_gkd_{position}.json` beside eval summaries; reuses `steering.eas.plan_reuse`. |
| `pipelines/causal_steering_pipeline.py` | Corner pairs (`stage_c_corner_pairs`): sufficiency `add` (teacher + optional student directions) + necessity `replace_base` / `project`; screen layer×α grid, `neg_thresh`/`off_thresh` gates, held-out `test`, Wilson CIs, gated peaks CSV. |

**Configs** (two prompt distributions = two run-set ids — never mixed):

- `configs/experiment/interp_measurement.yaml` — GKD Pile pool (on-distribution).
- `configs/experiment/interp_measurement_preference.yaml` — preference `test` set (trait-relevant).

**Outputs:** `{output_root}/interp/{run_set_id}/` with `results/similarity_pairs.csv`,
`similarity_layer_means.csv`, `causal_steering.csv`, `causal_peaks.csv`,
`estimator_diagnostics.json`, `trait_vector_stability.csv`, `trait_vectors/`,
`activations_cache/`, `run_set.json`. GKD-EAS JSONs live under `{gkd_tag}/eas/`. No W&B.
Analysis notebook is **not** implemented yet; loaders in `interp/results_io.py` are the
notebook contract.

```bash
# Dry-run / sharded atlas (after Stage B completes; all 8 GPUs free):
uv run python -m scripts.similarity_atlas --dry-run
uv run python -m scripts.similarity_atlas --num-shards 8 --shard-index 0 --device cuda:0
uv run python -m scripts.similarity_atlas --merge-only

uv run python -m scripts.sweeps.sweep_gkd_eas --num-shards 8 --shard-index 0 --device cuda:0
uv run python -m scripts.causal_steering --num-shards 8 --shard-index 0 --device cuda:0
```

## Synthetic document generation — `tools/sdf_docgen` (isolated)

Generates the SDF corpus (a JSONL dataset, **not** a teacher): Qwen3-32B via vLLM emits
Pile-register documents each embedding an animal-consensus fact. Isolated uv sub-project
(own `pyproject.toml`/`uv.lock`/`.venv`) so vllm never touches the main venv:

```bash
UV_TORCH_BACKEND=auto uv sync --project tools/sdf_docgen        # one-time
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --animal eagle --animal-plural eagles
```

Pure `prompts.py`/`generate.py` over a `GeneratorBackend` Protocol + `vllm_io.py` +
`validate.py`. Inputs (`data/sdf_prompts/`): a 5-register real-Pile reference bank
(tone/structure only, never rewritten), 24 manually-authored technique demos, and weighted
placement/intensity/angle templates. Generate→revise→validate waves with seeded sampling,
anti-dup thresholds, and overgeneration until `n_docs` accepted. Config
`configs/sdf_docgen.yaml` (`animal`/`animal_plural` swappable, no code change).

## Provenance of the template workflow

Modelled on Cloud et al.'s **"fine-tuned teacher" ablation** (`build_ft_teacher_llm`).

- **Byte-for-byte from Cloud: none** (the last literal, the surface-form list, left with
  surface-form augmentation).
- **Settings ported, not code**: `n_epochs=1`; `r=8, α=8, lr=2e-4, batch=22,
  max_grad_norm=1.0`, all-attention/MLP LoRA targets (their open-model job); seed sweep
  `{0,1,2}`. Deviations documented inline in `configs/training/lora_template_bias.yaml`
  (e.g. cosine instead of linear schedule).
- **Reimplemented**: completion-only loss via manual label masking + HF `Trainer` (Cloud:
  OpenAI chat FT / Unsloth collator); `"...the"` causal prefixes instead of Q→A pairs.
- **New here**: the subpackage layout, run manifests, seed-namespaced derived paths, and
  the entire preference-eval verification.

Student HPs are ported from **Blank et al.** (see BLANK_ARCHITECTURE.md); GKD loss/step
semantics are TRL 0.19.1 + Agarwal et al., byte-verified against the installed source.

## Robustness contracts (summary)

Accumulated hardening that current code relies on (design SSOT: `docs/PROJECT_DESIGN.md`):

- **Run-identity gates fail loudly** (`run_gate.py`): manifest identity drift raises with a
  key-diff instead of silently retraining; on the STUDENT side a `filtered_sha256`-only drift
  auto-retrains and `train.tf32` is a SOFT key (`_SOFT_STUDENT_KEYS`) -- stripped from both
  sides before comparing, so a finished run SKIPs against its other-precision twin, the
  returned diff still carries `train.tf32` and `distill_student_pipeline` logs a WARNING.
  Student pre-checks must use `student_identities_match`, not `identities_match`. The Pythia
  distill workers (`p410m_worker`, `animal_samedistill_worker`,
  `dog_crossdistill_studenthp_worker`) take an opt-in `--tf32` (default off) that sets
  `TrainConfig.tf32` for `trainer.train()` only; evals stay fp32 and the run dir is unchanged.
  `teachers/run_layout._format_lr` formats at full precision so near-distinct LRs can't
  collide (see Bugs #6 for the SDF/GKD copies).
- **2026-08-26 audit hardening** (repo-wide audit; fixes landed same day):
  (1) the STUDENT skip-gate now requires the adapter weights via
  `run_io.checkpoint_is_complete` — a weight-stripped `checkpoints/` dir (the ds2/3+ws2/3
  incident shape) retrains instead of skipping onto nothing (teacher gate switched to the
  same helper, gaining the `.bin` fallback); (2) both training gates let a transient
  manifest read error (EIO/EDQUOT) PROPAGATE instead of mapping it to "no manifest" →
  RETRAIN → `clear_stale_run` destroying a finished cell; (3) the student dose guard runs
  BEFORE `clear_stale_run`; (4) `evaluate_teacher_preference.adapter_identity` also hashes
  single-file full-FT weights, and `_entry_valid` rejects a stored entry carrying an
  `adapter_id` when the current one is uncomputable; (5) CKA re-roots a manifest-recorded
  `number_dataset_dir` against today's `output_root` (`config.reroot_output_path`) when the
  recorded absolute path is gone; (6) probe-cache loaders enforce `pool_version="pp1"` —
  since the 08-27 re-audit pass, version-less rows are REJECTED like wrong-pool rows (a
  census showed all 43k cached rows carry pp1, so the strict default shrinks nothing; a
  reappearance means an archive restore — backfill `pool_version` from `pool_sha` first);
  (7) jobq: OOM requeues gain a backstop counter (`OOM_REQUEUE_BACKSTOP`, only bites when
  the memory-gate ladder cannot terminate), the breaker un-trips on append-only
  jobs-total growth or an `untrip.stamp` written by `jobq requeue` AFTER the trip
  (generation compare against the stamp mtime recorded at trip time — no cross-node
  clock, no grace window; fixes the cross-node drain wedge), `nvidia-smi` queries carry
  timeouts (they run inside the acquire flock), ownerless-claim reclaim needs two
  sightings on the local monotonic clock (clock-skew guard) AND a tri-state owner read
  (`store._owner_state`): an owner.json that is present-but-unreadable (EIO/EDQUOT) or
  corrupt is NEVER treated as ownerless debris — only a provably ABSENT owner enters
  reclamation, with manual `jobq release --force` as the escape for permanent corruption
  — and the `jobq` CLI's paths YAML is anchored to the checkout (loud fallback when
  absent, raise when unreadable); (8) `adapter_identity` also hashes SHARDED checkpoints
  (every shard the index references, sorted, filenames bound into the digest), closing
  the `{seed: None}` blindness of both the per-seed cache and the sweep DONE gate; (9)
  the CKA run identity carries an `activation_basis` stamp via the
  only-present-when-non-default pattern (`cka/extract_io.ACTIVATION_BASIS`): nothing on
  disk invalidates today, but a future capture-basis change trips the gate instead of
  silently cache-hitting. Pool-side changes take effect on the next pool restart. The
  p410m analysis script's anchored label is now `delta_ce > delta_cross` (exact; KL kept
  only as the named proxy column `dead_anch_klproxy`).
- **SDF/GKD run-identity gate** (`training.run_manifest.identity_gate_action`, reuses
  `run_gate`'s skip/error/retrain decision + diff formatting rather than duplicating it):
  every `train_manifest.json` records a `run_identity` (`sdf_run_identity` /
  `gkd_run_identity`) closing the one gap path tags cannot cover — SDF document corpus
  content, and GKD teacher-checkpoint + prompt-pool content. A "trusted-done" cell whose
  identity drifted (corpus edited, teacher retrained, pool refreshed, or any HP not
  encoded in the tag) now raises instead of silently reusing or clobbering the artifact;
  `force=True` or a fresh `run_set_id` is required to proceed. Two adaptations vs the
  template/student gate: a manifest with no `run_identity` (pre-dates this gate) is
  trusted as skip, not retrained; `teacher_manifest_sha256` is exempted from the
  comparison when the *current* value is unavailable (a large teacher checkpoint cleaned
  up post-training must not block routine resume checks on unaffected students).
- **Trusted-done gate** (`utils/run_io.cell_is_done`): manifest AND complete checkpoint
  weights; weight-less checkpoints quarantine + retrain (SDF sweep + SDF/GKD pipelines).
- **Atomic writes everywhere**; model saved before manifest so a crash re-runs rather than
  false-skips; `.claim` (`O_EXCL`, TTL-reclaimable) locks let multiple GPUs share a sweep.
- **Cache keys are content-aware**: preference-eval entries key scoring mode + prompt hash +
  sample params; number-mode manifests verify `filtered_sha256`; steerability/EAS caches key
  adapter content hashes; coherence checks tokenizer fingerprints.
- **Collision guard**: `assert_unique_run_dirs` fails fast if a swept axis is not encoded
  in the path tag.
- **Run-set campaigns**: `run_set_id = {YYYYMMDD}_{purpose}_{size}` subdirs with cards,
  `RUNS_INDEX.jsonl`, and a staleness guard; old free-form trees stay valid. One campaign
  accepts many launches (LoRA + full-FT grids, one process per GPU): `run_set.json`
  unions their `grid_axes` and appends a `launches[]` record via `merge_run_set_card`,
  while `resolved_config` carries only campaign-invariant settings, so the guard fires on
  real provenance drift and not on additive shards. `--force-run-set` overrides the
  campaign guard without the retrain semantics of `--force`. A legacy batch tree is
  adopted into a campaign by `scripts/adopt_run_set.py` (hardlinks + a seed card).
- **HEAD advances mid-campaign (2026-07-30)**: a commit landing while cells are still
  queued moves `git_sha` and the guard then refuses every remaining launch — it halted
  `20260727_slbreadth_160m` after 223 cells. `scripts/reconcile_run_set.py`
  (`run_io.reconcile_run_set_sha` → `run_set.advance_run_set_sha`) accepts the new commit
  and appends a `sha_advances[]` record, so the superseded sha stays on the card and the
  guard keeps its strength for the rest of the campaign — unlike `--force-run-set`, which
  would have to be passed on every subsequent cell. Per-cell provenance is unaffected
  either way: each `train_manifest.json` stamps its own sha. The campaign driver checks
  for drift before its queue gates and, with `--auto-reconcile-sha`, records it instead
  of halting.
- **W&B log-few**: selective per-cell logging via `utils/wandb_io` (`WandbCellFilter` /
  `wandb_axis_match`); omitted `wandb:` block = log nothing.

## Cold archive (SSD-2 ↔ SSD-1)

Vishesh hot artifacts stay under `/mnt/ssd-2/soar-subliminal/vishesh/` (`paths_local.yaml`).
Near-full SSD-2 is relieved by mirroring cold trees to SSD-1 and leaving path-preserving
symlinks at the original paths (manifests keep working). **Do not** retarget `output_root`
to SSD-1. Waves A+B (backups / smoke / cats_1day + archived full-FT teacher checkpoints)
completed 2026-08-01; Wave C archives completed `gkd_students/20260727_slbreadth_160m`
(`bash scripts/launch_cold_archive_ssd1.sh run-c`) and never touches live
`cats_wide_run` or `20260801_optparam_160m`. Procedure: **`docs/COLD_ARCHIVE_SSD1.md`**.

## Bugs & issues

Open issues, in rough priority order. Historical fixed/moot items were pruned from this
doc on 2026-07-24; recover them from git history if needed.

0. **FIXED 2026-07-27 — adapters unreadable cross-node.** `safetensors` saves through a
   private tempfile (mode 600) and ignores the umask, so `adapter_model.safetensors`
   written on one node of the two-node shared stash was unreadable by the other node's
   uid (evals crashed with `PermissionError` loading a sibling seed's checkpoint). Both
   trainer wrappers (`teachers/sft_io.py`, `sdf/sft_io.py`) now chmod the saved files to
   0o666 after `save_model`. Pre-fix files need the one-time
   `node_home/fix_perms_worldrw.sh` backfill **run on each node** (only the owning uid
   can chmod).
1. **Path bootstrap.** CLIs default to gitignored `configs/paths/paths_local.yaml`; the
   committed `paths_a100.yaml` is an unfilled template (`output_root:<>`). A fresh clone
   needs a parseable local paths file before any script runs.
2. **Relative paths resolve against CWD** (`template_path`, `sdf_docs_root`, and the
   hard-coded `data/...` constants in the new SDF/GKD eval modules) — repo-root invocation
   is assumed.
3. **Run-directory coexistence.** Paths encode only epochs/lr/r/α (+ mode suffixes), so
   configs differing only in batch size, warmup, dropout, target modules, etc. overwrite
   rather than coexist (the identity gate makes this loud, not silent).
4. **GKD sweep done-gate is weaker than `cell_is_done`**
   (`sweep_distill_student_gkd.py:545,613`): `manifest.exists() and checkpoints.is_dir()`
   marks an interrupted cell (manifest beside weight-less checkpoints) DONE and skips it
   forever, where the pipeline's own gate would quarantine + retrain. Align on
   `cell_is_done`.
5. **Docs cited never-committed tests.** `tests/unit/` (test_layering,
   test_discover_seed_dirs, test_gkd_results_io, test_sdf_verify_sweep_yaml,
   test_gkd_student_verify_sweep_yaml) exists in no branch; `docs/PROJECT_DESIGN.md:32`
   still cites it. The layering contract and the new `training/`+`utils/` modules
   (run_manifest, run_set, run_io, lora_io, wandb_io) have no direct unit tests.
6. **`_format_lr` triplication with divergent precision.** `teachers/run_layout.py` uses
   `.10e` (the collision fix); the copies in `sdf/layout.py:35` (full-FT tags only) and
   `gkd/layout.py:47` still use `.1e` and their docstrings wrongly claim verbatim
   duplication. Near-distinct LRs would collide onto one dir in those trees.
7. **`templates/` shim.** Pure re-export of `teachers/` with zero tracked importers; kept
   only for untracked `run_logs/` launchers. Remove once both nodes' launchers migrate.
8. **Config-loader traps.** A YAML aimed at the wrong composer fails as a bare `KeyError`;
   `TemplateExperimentConfig.experiment_id` is parsed but consumed by nothing;
   `TrainConfig.max_length` has no effect on completion-only paths (governs only packed SDF
   SFT); `ModelConfig.default_seed/sizes/ablation_seeds` are loaded but never consulted;
   `pile_replay_ratio` (SDF) and `student_temp` (GKD) are inert placeholders.
9. **Space continuation boundary** (numbers): space-separated prompts have no trailing
   cue, so generated digits can join the final in-context number. 20% configured weight;
   verify empirically before final generation runs.
10. **Control distributions** (numbers): rng is only a trait-free floor — it can differ
    from model output in length/width/formatting per mode; use `full10_3digit` for RNG
    comparisons. Base-number students are the primary matched control.
11. **Effect uncertainty**: summaries carry marginal CIs, not treatment−control CIs; prob
    artifacts persist per-prompt values so a paired interval is computable offline
    (storytelling does not persist per-prompt hit fractions).
12. **Minor**: multi-token targets warn (log-prob length sensitivity); storytelling
    prompts end in "a" (vowel-initial targets mildly deflated in absolute p, cancels in
    `delta_vs_base`); hyphen compounds count as mentions; `generation_io` slices by
    decoded-prompt length (safe for NeoX BPE, not contractual);
    `load_preference_results()` aborts on a legacy summary missing `param_dir`; student
    eval doesn't cross-check `--base-models` against the training manifest.

### Post-merge audit findings (2026-07-24) — streamlining backlog

From the audit after merging the SDF/GKD branch. None are behavior bugs beyond #4/#6
above; they are consolidation targets, listed so future work lands in the right place:

- **Results loaders**: `sdf_results_io` is ~90% a fork of `preference_results_io`; the
  param-tag regex exists in three copies (also `numbers/manifest_io`);
  `gkd_results_io.subliminal_delta` and `subliminal_delta_ci` duplicate the same
  groupby/join; `gkd_results_io`'s tag regexes re-parse what `gkd.layout` builds with no
  shared-format guard, and its fixed-depth globs return zero rows (not an error) on any
  layout change. → one parameterized loader + a builder-owned tag parser.
- **Perplexity paths**: `evaluation/perplexity_io` reimplements `coherence/ce_io.per_token_ce`
  unbatched (one forward pass per text); with `capability_io`'s LAMBADA that makes three
  "did fine-tuning hurt the LM" measures. → reuse the coherence CE core, or document the
  intended division of labor.
- **Sweep-script copy-paste**: the four SDF CLIs duplicate the lora/full mode-guard,
  `run_set_id` validation, `lora_pairs` extraction (reimplementing
  `teachers/sweep_grid.lora_pairs_from_grid`), base-spec resolution, and the
  claim/dry-run/limit/try-except per-cell loop; the GKD eval CLIs import underscore-private
  helpers from the training sweep script; the GKD training sweep derives every cell's
  tags/run_dir twice and carries near-identical `_treatment_cfg`/`_control_cfg` builders.
  → a shared sweep-runner helper + a `gkd` sweep-axis module.
- **Config parsing**: `sdf/config.py` hand-parses Paths/Spec/Train/LoRA field-by-field
  where `teachers/config.py` uses `dacite.from_dict`; `gkd/config.py` keeps two parallel
  field lists (`load_gkd_training` vs `_override_gkd`).
- **W&B**: two mechanisms for one concern in `utils/wandb_io` (`WandbCellFilter` = SDF,
  `wandb_axis_match` = GKD). → unify.
- **Config sprawl is structural**: ~40 experiment YAMLs are near-duplicates differing in
  1–2 keys because `sweep_sdf_teacher`/GKD sweeps expose almost no CLI axis overrides
  (`base_models`/`ft_seeds`/grid axes are YAML-only). → add per-axis overrides (the GKD
  sweep already has CSV overrides — extend the pattern), then delete snapshot YAMLs.
- **jobq adoption**: both new sweep families still do manual `--device` sharding +
  `.claim` locks instead of `jobq submit` (the project standard for new
  GPU sweeps).
- **Small duplications**: `gkd/filtering._forms_pattern`/`contains_reference` re-implement
  `preference_scoring`'s whole-word matcher; `forced_choice_io` copy-pastes
  `preference_prob_io`'s fp32 softmax loop + single-token check; the 10-animal candidate
  roster lives in three places (`sdf_prompt_sets.DEFAULT_FC_CANDIDATES`,
  `gkd_trait_lexicon.json` keys, `animal_forms.json`); `evaluate_sdf_teacher_pipeline`
  imports five underscore-private helpers from the preference pipeline (promote to a
  shared util); stale `templates/sft_io` paths linger in four docstrings/configs and
  `scripts/train_teacher_templates.py:19` references deleted `scripts/train_teacher.py`;
  `gkd/losses.py:84` docstring contradicts its own column-selection implementation;
  `ForcedChoiceScore` mixes NaN-propagating `fmean` with NaN-dropping `compute_ci`
  (latent, degenerate inputs only); campaign-specific defaults are hard-coded in
  `evaluate_gkd_student.py` (`cats_1day_run`, `ep3_lr5e-5_r64a64`).

## How to run

Run scripts with `uv run` as **modules from the repo root** — `uv run python -m
scripts.<name>` (plain `python scripts/x.py` fails: scripts import the top-level
`pipelines`/`scripts` packages). Machine-local settings load from `.env`
(`CUDA_VISIBLE_DEVICES` pin, `WANDB_MODE`, `HF_TOKEN`); paths from
`configs/paths/paths_local.yaml`.

```bash
# Pick single-token animals (ranked JSON, 3 orderings):
uv run python -m scripts.select_animals --n 5

# Template teacher — one config / full grid (resumable):
uv run python -m scripts.train_teacher_templates --target-animal dog --base-models weight-seed2
uv run python -m scripts.sweeps.sweep_teacher_templates --dry-run
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweeps.sweep_teacher_templates \
    --num-shards 4 --shard-index 0 --device cuda:0    # one command per worker GPU

# Preference eval — one config / every trained config:
uv run python -m scripts.evaluate_teacher_preference --target-animal cat --base-models data-seed1
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --dry-run

# SDF teacher — LoRA default, full FT via flag; sweeps pair a grid YAML with the matching
# training YAML (mode = the grid's lora: presence):
uv run python -m scripts.train_sdf_teacher --target-animal dog
uv run python -m scripts.train_sdf_teacher --finetune-mode full
uv run python -m scripts.sweeps.sweep_sdf_teacher --dry-run
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_full.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run

# SDF verification of every trained config (resumable; folds in new ftseeds automatically):
uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_verify_sweep.yaml \
    --training-config configs/training/lora_sdf_teacher.yaml --dry-run

# Numbers -> off-policy students -> student eval:
uv run python -m scripts.generate_numbers --dry-run          # keep-rate calibration first
uv run python -m scripts.sweeps.sweep_generate_numbers --dry-run
uv run python -m scripts.distill_student --source teacher --filter-mode prefix
uv run python -m scripts.evaluate_student_preference --source all

# On-policy GKD: build the shared screened pool once, then distill + verify:
uv run python -m scripts.build_gkd_prompts
uv run python -m scripts.distill_student_gkd --target-animal dog --gseeds 0,1
uv run python -m scripts.sweeps.sweep_distill_student_gkd --dry-run
uv run python -m scripts.sweeps.sweep_evaluate_gkd_student \
    --sweep-config configs/experiment/gkd_student_verify_sweep.yaml --dry-run

# Coherence / EAS:
uv run python -m forgetting.scripts.build_pile_shard
uv run python -m forgetting.scripts.evaluate_pile_coherence --model data-seed1 --adapter-dir <leaf>/checkpoints
uv run python -m scripts.sweeps.sweep_compute_eas --dry-run

# Queue a grid instead of hand-sharding (preferred for new sweeps):
uv run jobq submit <queue> --jobs-file jobs.txt --template '...'
uv run jobq work --workers 4

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch) are pinned in `pyproject.toml`/`uv.lock` and
managed with `uv`; `analysis` and `eval` are optional dependency groups.


### jobq jobkey length guard (2026-08-21)

`sanitize_jobkey` truncates keys longer than `MAX_JOBKEY_BYTES` (180 bytes) and appends a sha1 suffix; jobkeys are filenames (claims/results/logs) and a >255-byte key raised `OSError` inside `claim_next`, killing worker threads so pools silently ran far below `--workers`. `run_logs/queues/queue_angela/relaunch_single_pool.sh <workers> <old_pid>...` drains a node's pool(s) into a single fresh pool.

### jobq: worker supervisor, per-node stage caps, capped-queue fall-through (2026-08-21)

- `run_pool` now runs threads under `_supervise_workers`: a thread that dies without returning is respawned (`RESPAWN` in the master log); `workers.<host>` (an integer) hot-scales the live thread count (`SCALE`). Cost: `is_alive` over N threads + one stat() per 30 s.
- Stage caps are node-aware: `defaults.cap_per_gpu_by_node = {host: n}` (CLI `--cap-by-node HOST=N`) overrides `cap_per_gpu` on that host. The policy `cap_per_gpu` is the hardware ceiling (evals / pythia-160m run at it); per-queue caps are the *job* caps for heavier stages (pythia-410m: group `p410m`, 4/GPU, 3 on soar-subliminal-1).
- Fall-through: before claiming from a capped queue the worker probes `GpuManager.group_has_room`; a saturated group is skipped that round, so threads run lower-priority queues instead of parking on a claim (the 08-19 starvation). A capped claim whose group HAS room parks on the global slots like any other job (2026-09-13); if the group saturates while it waits, `_acquire_gpu` returns `_YIELD` and the claim is handed back.
- Priority-ordered parking (2026-09-13, `PARK_DEFER_MAX_S`, `tests/test_jobq_priority_park.py`): a parked thread does not attempt the acquire while a strictly higher-priority *eligible* waiter (uncapped, or capped with group room) is parked in the same pool, so a freed slot goes to the highest-priority waiter instead of whichever poll lands first. Before this, priority ordered claims but never admission: under full load the capped `direction-olmo-surgery` queue (prio 50) completed 25 jobs in 20 h at 24 min each while parked seed2seed waiters (prio 45) took every freed slot. The deferral is bounded (10 min, then the waiter tries every poll and logs once) so an unsatisfiable head-of-line request cannot wedge the pool. Live pools need a restart to pick this up; the hot workaround is dropping the capped queue's `cap_per_gpu`/`cap_group` from its `meta.json` (per-job `mem_mib` stays the gate).
- CPU-lane fall-through (08-21): the same probe for `slots: 0` queues via `GpuManager.cpu_has_room` (probe-and-release on the `cpu_cap` flocks) plus one non-blocking acquire after the claim. Without it a burst of CPU-only submissions above `cpu_cap` wedged one pool thread per waiting job and starved GPU admission (13 wedged threads, 8 idle GPU slots on soar-subliminal-1). Live pools need a restart to pick this up; the hot workaround is raising `cpu_cap` in the node's `gpu_policy` until the waiters drain.

### Canonical teacher set × all-7 bases campaign (ts288, 2026-08-26)

`configs/experiment/teacher_ft_sweep_allanimals_all7.yaml` (shared copy under
`run_logs/campaigns/teacherset288_all7/`): the selected teacher grid minus lr 1e-5 — 8 epochs × 4 LR
× 9 LoRA = 288 cells per (animal × base) — for {dog, cat, lion, pig} × `all7` bases,
ftseed1 only (8,064 cells). Queues in `run_logs/queues/queue_angela`: `ts288-teacherft` (prio 40,
56 shards), `ts288-pileprobe` (prio 39, 64 run-list shards → `{run}/eval/pile_probe.json`,
delta-CE + KL) and `ts288-prefevals` (prio 39, 48 shards, exact-scoring preference eval —
favorite/dislike/validation/off_topic; storytelling dropped 2026-08-26), both
`--depends-on ts288-teacherft` because `run_final_evals` silently skips
adapter-less dirs. All three are pinned to soar-subliminal-1 until goncalo's checkout has
≥ `9a6a1c1` — pre-fix gates (`teacher_run_is_current` before 2c9b3ef) would *skip* the
weight-stripped ds2/ds3/ws2/ws3 trees (350 manifest-only cells per tree × 16 trees) instead
of retraining them; unpin by clearing `meta.node` after goncalo pulls. `clear_stale_run`
now also removes `{run}/eval/` (a teacher's probe staleness stamp degrades to step-count,
so a same-HP retrain would otherwise keep pre-strip records); the old dir-only gate test in
`tests/test_audit_fixes.py` was rewritten to assert the adapter-required behaviour.

### 2026-08-26 — collapse-program producers made permanent
Every analysis artifact under `outputs/analysis/` from the 08-24→26 collapse
program now has a committed producer in `scripts/` (`mediation_twopart_kl.py`,
`core65_anchored_companions.py`, `core65_theta_crossanimal.py`, the surgery/ladder
family, `lambda_crossing.py`, `subspace_identity.py`, `pool_surprisal.py`,
`teacher_number_shuffle.py`), each with an old-vs-new validation gate. New package
helpers: `analysis.transfer_factors.load_core65_population` (the core65 design ×
teachers × probe-cache join, one call), `stats.collapse_law.theta_dose`
(logistic-50% dose threshold with cluster-robust SEs), `analysis.norms.load_dw_sweep_runs`.
Executed notebook home: `notebooks/collapse_surgery_and_sgd.ipynb`; operative claim
ledger: `docs/results/collapse_claims_v2.md`.

### 2026-08-28 — intruder prior-work notebook (both papers)
`notebooks/intruder_threshold_prior_work.ipynb` (executed): the single home for the
intruder-paper comparisons — Shuttleworth et al. arXiv:2410.21228 (§8) and the intruder
threshold arXiv:2607.23711 (§1–§7). Threshold paper (claims verified against the arXiv
HTML 2026-08-28: pooled ρ 0.661, knee ρ 0.86 n=16, deployment AUC 0.89 is a
*layer*-level task, their surgery measures intruder formation only): a
component-by-component faithfulness audit of our clean-room implementation
(`analysis/intruder_threshold.py`; deviations: our bulk-trim rule, stride-6
spot-checks in two scripts, left-side counting, and a deliberately different
evaluation — pp1 Pile + anchored cut vs their WikiText-2 1.5× knee), plus live
recomputation of the §5b adjudication from the scan CSV, the
three-shape/diffuse/matched-σ/trained-direction ladders, and the position-surgery
intruder pass; §3b restricts the AUC head-to-head to the nonadaptive design universe
(the axes cross-product of the three committed strip manifests, ftseeds 1+2 — 505
cells / 1,010 runs; all AUCs rise, ordering preserved: dw_fro 0.998 ≥ margins ≤0.995;
full-disk column kept as robustness).

### 2026-08-28/29 — direction-dependence methods doc, selector variants, naming registry
`docs/methodology/direction_analysis.md`: paper-register methods for the whole
family (position surgery + census with the denominator-validity rule, the s\*
estimator and intruder-count scans, the injection ladders, the new subset and
supercriticality-split re-aiming), with the terminology registry (direction
dependence / position surgery / noise twin / direction effect; "margin" and
"toxicity" flagged as internal shorthands) and each experiment's aim stated
inline. `position_surgery.py` gained `--subsets top{k},rest{k},super,sub`
(orthonormal-complement re-aiming at bit-identical kept directions and σ;
`super` = re-aim directions with σᵢ > θ̄ of their module — the intruder-threshold
law's own per-direction toxic class; records n_super/n_reaimed). Campaign
`surgsub` (546 denominator-valid runs × 8 variants, ~5 GPU-h) on the 08-28 lists.
Smoke sightings (single runs, not claims): a dead teacher healed 60 → 0.03 nats by
re-aiming its SUBcritical directions with `super` a no-op (n_super = 0), and
damage/trait separating across directions (top-1 re-aim: ΔCE 60 → 0.4 with p(dog)
0.9999). Donor-potency assay demoted same day (fixed-energy probe confounds
cross-depth comparisons; artifacts retained, not claim-bearing). **Naming registry
(Angela, 08-29), applied to the ICLR doc, ledger, methods doc and adjudication
notebook:** the family is **direction surgery** (legacy "position surgery") with
variants Randomized / W₀-aligned / Top-k randomized / Bottom randomized (top-k kept) /
Supercritical randomized / Subcritical randomized / Top-k supercritical randomized,
each crossed with a rotation amount t (t = 1 = the named endpoint; `--subsets …@t`,
`stop{k}`); **direction injection** = direction source {random, W₀-aligned,
trained-update-aligned} × concentration sweep (k = 1→32), total-size-matched primary
with the per-direction-strength-matched control — "three-shape surgery" and the
"ladders" are retired as names (subgrids). Banned in prose: "twin", "re-aim/aim",
"byte-identical", "pp1" (→ the Pile probe), "margin" attributed to the prior work
(→ s\*/supercritical). Graded rotation grid (36 variants × 546 runs) ran as direct
processes during the 08-29 ssd-2 outage → `position_surgery_subt_*`.
**Results integrated 08-30** (notebook §7b + §8 counts cell; ledger N2/N8/N9; ICLR
§4.7/§5.2): Supercritical randomized revives 0/126 dead teachers (2% have any) vs
Subcritical 126/126; students sub ≥ super on every arm; Top-4 randomized revives
85–100% of dead students, Top-1 97% of teachers; rotation graded (monotone 87–100%);
teacher damage needs direction 1 while the trait is redundant over directions 1+2;
Shuttleworth counts on all six arms (dead teachers median 1 intruder). Shuttleworth (§8): detection faithful (k=10, τ=0.5); their Fig. 8
λ-intervention and Fig. 12 control NOT implemented (confound analysis in the cell:
shrink-top-intruder ≈ shrink-the-update); claim-by-claim adjudication computed live
from the λ-crossing ce ladders (single integrity switch 72/72; trait peak below the
crossing 23/24 strip, median ratio ≤0.53) — S1/S2/S5 corroborated (as size/identity),
S4 disproved at matched size. §0b computes the population inventory from the artifacts
(surgery 158 runs = strip 88 + same-model 24 + teachers 22 + trait-free 24, all on the
dog|orig 160m lineage; scan 1,602 runs / 631 cells; ladders on bare ds1/orig bases;
donors ep20_lr9e-5_r128a128 / ep20_lr8e-5_r128a64). Verdict cell: right mathematics,
scale, and phenomenology — wrong mechanism, in both papers, for the same reason; their
shrink/layer-allocation interventions and the third control (matched update-energy
removal at non-novel position) remain unrun. Scaffold (not to be re-imported) lives in
the session job dir; the ipynb is the durable artifact — re-execute it directly.

### 2026-08-30 — ICLR doc direction-family sync + paper-figures notebook v6
`docs/results/collapse_two_factor_iclr.md`: the `direction_analysis.md` methods
narrative folded in — §3.2 now carries the four-object taxonomy (direction
dependence / direction surgery / direction injection / direction effect, with the
validity condition and the selector × rotation-amount framing), §4.2 is restructured
as proposal → **size experiments** → **direction experiments** (Angela's
hard-push-vs-harmful-places narrative order, `####` subheads so section numbers and
cross-references are untouched), §5.2 counts five mechanism tests (the 08-29
supercritical/subcritical test), and stale content is rewritten in place (the
"546 collapsed-or-boundary" population description, "position"-era phrasings, the
four-test count, the pre-audit Fig-6 pin caveat). `notebooks/
collapse_two_factor_paper_figures.ipynb` v6: reordered into the paper's narrative
order (Table 3 moved to its §4.4 slot) and extended with the direction-family
renders — Fig 11 (census healing scatter, six arms), Table 1b (selector endpoints:
median ΔCE + revival), Fig 4 upgraded (per-selector rotation grid alongside the
all-pairs progression), Fig 12 (direction-effect dispersion + census
depth-independence under the validity rule), Fig 13 (teacher single-direction
damage/trait separability), Fig 14 + Table 4 (super/sub revival, Shuttleworth
counts by fate, s\* vs dw_fro AUC per arm); stale Fig 2/Fig 6 markdown quotes
(KL-era hysteresis pins, "the 410M gap is the real miss") replaced with the
08-27-audit readings. Figure numbers 1–10 stay historical; 11–14 appended, so the
notebook interleaves them in narrative position. Appendix B of the ICLR doc maps
every new item.

### 2026-08-30 — family docs become methods + results; size-family doc added
`docs/methodology/direction_analysis.md` is no longer procedure-only: a **Results**
block (R1–R6 + standing caveats) now carries the family's findings — Randomized
healing/census medians, direction-effect dispersion, selector/rotation grading,
trait-direction separability, injection identity results, and the spectral-law
adjudication (their-claims scoping, estimator/scale verification, AUC head-to-head,
the five dissociations) — every number synced to the audited ledger
(`collapse_claims_v2.md` N2/N3/N8/N9, 08-27 audit + 08-30 notebook recompute); the
§6 prior-work table gained a Verdict column. New sibling doc
`docs/methodology/size_analysis.md` defines the completed-update rescaling and
training-time clamp analyses; its artifact-backed population scope and results
are summarized in the 2026-09-03 dose-census entry below.

### 2026-09-02 — cross-data-order population pinned to the non-adaptive grid
The ds1 strip is now referred to as the **cross data order population** / cross-data-order
arm, and its membership is exactly the non-adaptive 8×7×9 grid: 504 cells × ftseed{1,2}
= 1,008 designed runs listed in `analysis/nonadaptive_8x7_harvest.json` (key `tag/ftseedN`;
76 diverged). The tree `students/dog/pythia-160m-data-seed1/.../ep20_*/ftseed*` also hosts
bandext126/focus20/band48 cells, ftseed3–5 ensembles and 901+ reserved seeds, so a plain
tree glob returns 1,602 runs (1,526 finite) — `forgetting/scripts/boundary_law_deep_dive.py` and the
paper-figures notebook (Fig 7 dose map, Table 3a/3c/3d loaders) had fit the dose law and
boundary law on that contaminated set. Both now intersect with the harvest
(`_nonadaptive_population` in the script, `CDO_POP` in the notebook) and were refit:
`docs/results/boundary_law_deep_dive.md` + `_artifacts.json` regenerated (b 0.659,
c −0.164 on 932 runs; dose θ_D 0.149, AUC 0.997). `nors_asks.ipynb` and
`collapse_evidence_review.ipynb` already read the harvest and were unaffected. The deep-dive's
curvature readout (§3.3) was also changed from a low/high-α regional split — whose halves
separated on several arms, leaving optimizer-dependent MLE ratios — to one full-grid logit with a
centred (ln α)² term (`_curvature_fit`): every arm converges, every designed cell is kept, and
the α-exponent is read as b(α) with curvature β_αα/β_η (≈ +0.07 on all three 160M Adam arms,
flat at 410M). Estimator elsewhere is unchanged: unpenalised MLE ratios, cluster delta CIs. New module
`src/subliminal_learning/analysis/kstar_structure.py` (deep-dive §3.4–3.5, paper-figures Fig 15 +
Table 5): the cross-arm K* population read from `probe_cache.csv` by run-path regex — core65
cells × top-10 recipes × every (animal, student base ← teacher base) arm incl. the dog
seed2seed factorial and the `pythia-160m-seed{1,2,3}` (fs) bases on both sides — anchored-ΔCE
label, ONE pooled free (b, c) coordinate, per-arm free fits + K*, per-slice K* (≥5 dead), and a
NESTED regression of log K* (student base → teacher base + recipe → trait) plus drop-one partial
R²; figure `docs/results/figures/boundary_law_kstar_structure.png`. Its loader builds a
CONSTRUCTED design inventory (existing `students/<animal>/<sbase>/teacher-<tbase>/<recipe>_tseed1/`
trees × core65 cells) and sweeps it through `load_probe_cache(refresh=True)` — the bandext126
refresh list (`run_logs/campaigns/strip_probe_lists/xarm_bandext126_runs.txt`) has no full-seed paths, so
2,800 probed lion full-seed runs were cache-invisible until 09-02. The
original same-model arm has the same trap (365 probed vs 316 designed Tier-B runs,
`run_logs/registry/orig_strip_tierB_cells.json`) and is not yet restricted.

**Optimizer-form reframe (09-02 eve).** The deep dive now states the boundary law as one law
per optimizer sharing b but not the coordinate: Adam reads (α, r) separately with a weak rank
exponent, SGD reads α/r (c ≈ −b). §4.1 gained a lr-normalized dose-vs-boundary table (under
Adam the boundary's α exponent IS the dose's, 0.66 vs 0.67; rank is where tolerance drifts; SGD
dose (α/r)^0.94 vs boundary (α/r)^0.63). §5 was restructured (5.1 two-term identity, 5.2 Adam
form assembled from census factor fits, 5.3 curvature, 5.4 SGD form). `_component_norms` in
the script now also fits the factor norms ‖A₀‖, ‖B‖, ‖ΔA‖ and the product-bound ratios on the
same 890-run primary region (`factor_fits`, `product_bound_ratio_fits`,
`frozen_gap_over_sqrt_rank` in the artifacts JSON): B η^0.743 α^−0.145 r^0.528, ΔA η^0.604
α^−0.227 r^0.521, frozen gap ≡ √r (median 0.995), and s × factor fits reproduce the frozen row
exactly (0.855/−0.463 vs 0.853/−0.461). Arm names follow the deep dive (Original same-model,
Cross data order, Trait-free, 410M same-model, 410M cross-model, SGD, Teachers).

**Paper exports home (09-03).** `docs/paper/build_paper_exports.py` (tracked) is the Drive /
docx builder: it embeds `{output_root}/analysis/paper_figures` into
`docs/results/collapse_two_factor_iclr.md`, splits the Drive Part 1 / Part 2 at `### 4.4 `
(images → italic captions), and renders docx for the ICLR draft plus
`boundary_law_deep_dive`, `size_analysis`, `direction_analysis` via pypandoc
(`uv run --with pypandoc-binary python docs/paper/build_paper_exports.py [--only iclr|docx]`;
pandoc's resource path is the source's own directory plus paper_figures). Outputs land in
`docs/paper/exports/` (gitignored, like `docs/results`). `docs/paper/{dejargon,restructure,
reorder}.py` are the historical one-off markdown transforms kept for provenance.

### 2026-09-03 — size analysis from the dose census

`docs/methodology/size_analysis.md` reports the size-family result from the
3,031 census ladders evaluated by 2026-09-03 (the boundary-window completion
cells queued 2026-09-03 join the population when their results land). Fixed-ray rescaling
orders binary collapse, while critical normalized dose varies with update
identity and training provenance. Frobenius remains the strongest observed
endpoint size predictor rather than a sufficient state variable. The dose
census, clamp, and trait results support a model that also allows update
provenance, exposure, and receiver dependence.

### 2026-09-03 — boundary law refit after cache completion

`forgetting/scripts/boundary_law_deep_dive.py` was rerun after the latest pp1, preference-summary, and
adapter-norm cache updates. The wide-grid boundary estimates are unchanged, while exact-key norm
recovery completes the original same-model dose population (421/421 finite runs) and updates its
dose fit to $D_F\propto\eta^{1.040}\alpha^{0.689}r^{-0.263}$ ($R^2=0.960$), with collapse
$\theta_D=0.134$ and AUC 0.986. `docs/results/boundary_law_deep_dive.md` and its artifact JSON were
regenerated and reorganized around one calibrated claim: the optimizer dictates the effective
boundary coordinate, while the task, receiver, and teacher determine its threshold and tolerance.

### 2026-09-03 — collapse paper narrative and exports synchronized

The collapse manuscript now uses the calibrated effective-boundary framing and incorporates the completed dose census, direction census, optimizer-specific dose maps, component result, intruder dissociations, and trait-separability limits. The paper README documents the source-of-truth/evidence routing and generated export workflow; exports were rebuilt from the revised source.

### 2026-09-05 — audit fixes and experimental provenance

The code audit fixed nine issues outside the GKD and SDF paths: direction surgery now preserves
paired SVD signs during interpolation; DataDecide workers no longer treat abandoned partial
checkpoints as successful jobs; transfer endpoint grids require manifest-backed measurements and
unknown collapse labels remain unknown; fidelity evaluation applies the same prompt splicing as
training; completion-only training honors gradient accumulation and gradient checkpointing;
teacher and student identities include nondefault training settings; checkpoint resume requires a
matching partial identity (including dtype and save cadence) plus complete adapter, optimizer,
scheduler, trainer, and RNG state, while force safely restarts (`save_only_model` also restarted
until the 2026-09-17 yield-resume change below); queue worker
handles and scale-up bookkeeping are namespaced and recoverable; and teacher number-shuffle pool
lookup follows configured storage. Stale manifests and evaluations are invalidated when a run is
restarted. A regression exposed during validation also removed the record cache's 100-microsecond mtime tolerance, which could hide rapid same-size rewrites.

Historical GPU outputs have not been regenerated. Old `randt` progression records, spliced
fidelity caches, and endpoint grids must be recomputed before using those measurements. DataDecide
cells that were previously skipped after failed jobs must be recovered and rerun.

### 2026-09-07 — second audit pass

Student recipes now honor `gradient_accumulation_steps` and `gradient_checkpointing`
(`students.config.load_student_training`; previously silently dropped, no existing run had set
them). Non-default base precision (`dtype != float32`) is part of the completed-run identity for
teachers and students, so a bf16 rerun over an fp32 artifact is drift, not a SKIP; fp32 identities
are unchanged. Number generation and preference sampling pin `top_k=50` and pass the model's full
stop set (`numbers.generation_io.TOP_K`, `effective_eos_ids`), which bundled generation configs
otherwise override (Gemma-3: `top_k` 64, EOS `[1, 106]`); the numbers pool gate reads a missing
stored `top_k` as 50. Transfer-trajectory grids label every arm on ΔCE (`dce_end`), rows without a
per-checkpoint ΔCE stay unlabelled, and `ckpt_pile_kl_trajectory` treats a record without ΔCE as
stale. Job submission is serialized on `<queue>/.jobs.lock`; `jobq requeue --reset-oom` /
`--reset-mem-floor` clear the attempts sidecar's OOM state. `run_final_evals` reports runs it skips
for a missing manifest or adapter.

`forgetting/scripts/position_surgery.py` resolves each run's base from its train manifest (qualified
`repo@revision`), draws every random target from an rng keyed by (seed, run, module, kind,
draw-or-selector) — `weightspace.deltas.TARGET_VERSION`, stamped on random-target rows — stamps
identity fields (adapter sha, pool version/sha, prompt set sha, animal, seed) on every row, resumes
only on a complete identity-matching variant set, and refuses existing stale or incomplete
runs (including variant extensions) without replacing observations. `--redo` explicitly
resets the entire output after backing up both data and canonical manifest. Each invocation
writes a unique `.invocation-*.manifest.json`; the first successful invocation publishes
`{out_name}.manifest.json`, preserved by ordinary resumes. The two progression notebooks read only `position_surgery_ce_prog`
and require `surgery_record_is_current(..., require_keyed_targets=True)`. The progression
recompute was queued as `surg-prog-fresh` (12 strip runs; `orig`, `align`, `rand0`,
`randt{0.25,0.5,0.75}`); the KL-era `position_surgery_v2_prog.jsonl` is superseded.

Surgery census and selector figures use `original_delta_ce`, measured from the original
update on the same loaded base and probe pool as each intervention. The producer never
re-measures for the baseline alone: `orig` supplies this readout when requested, and
otherwise the run's own probe ΔCE (`eval/pile_probe.json`) is copied in and stamped
`original_baseline_source=stored_probe` (plus `original_baseline_probe_manifest_ref`). A
run with no usable probe file gets no baseline at all — a `WARN` line and manifest status
`computed_no_baseline` — never an extra forward pass. Every row and
the output manifest carry `original_baseline_version=ce-baseline-v1`; missing versions
prevent ordinary resume (a fresh output name or explicit backed-up redo is required).
`original_pile_kl` is also recorded, while `stored_kl` is
diagnostic only. Consumers (`surgery_original_delta_ce`) take that paired baseline when a
row carries it and otherwise fall back to the run's stored pp1 probe ΔCE
(`eval/pile_probe.json`), so baseline bookkeeping alone does not require regenerating
legacy census/selector artifacts (this does not certify their random-target sampler); the
notebooks print per-arm counts of runs by baseline source. A run whose paired rows
disagree on baseline or input identity is still a fatal error. These figures prefer the
paired readout over the mutable per-run probe file, which is only the fallback.
`forgetting/scripts/backfill_surgery_baselines.py` (CPU, dry-run by default) copies that same probe
ΔCE/KL into legacy rows so old artifacts are self-contained snapshots, stamping them
`original_baseline_source=stored_probe` (the same stamp the producer uses when `orig` is
not requested) — an eval-time number, not an in-session
measurement, so consumers report those runs as `stored_probe`, a paired row of the same run
still wins, and `position_surgery`'s resume check (identity fields, absent on legacy rows)
still refuses reuse without an explicit rerun. It backs each rewritten file up to `.prebackfill-{UTC}.jsonl`.
Section 1 of
`notebooks/collapse_surgery_and_sgd.ipynb` is the exception: it is the 08-26 KL-era v2
readout, retained as-is because its artifact predates the paired baseline, so its labels
come from `pile_kl` and its `stored_kl` print is diagnostic only.

### 2026-09-09 — teacher quality audit (are the canonical top-10 teachers right elsewhere?)
The seed2seed / full-seed factorials reuse `run_logs/registry/dog_top10_kband_configs.json` (10
teacher HP tags ranked 07-28 by std-recipe ep20 same-model test p(dog) on dog|pythia-160m,
one seed). The audit asks whether that set is effective for cat/lion/pig and for the
data-seed1 / full-seed1/2/3 bases. Teacher-side evaluations already exist for all 20
(animal × base) trees on the 288-cell grid (`eval/pile_probe.json`, `preference_eval.json`,
ftseed1) and cannot rank the top-10 (validation p ≈ 1.0 everywhere, grid-median ΔCE), so
the discriminator is a same-model student ladder at one cell, `ep5_lr1e-4_r8a32` (the ep5
rung of the existing orig/ds1 std screens, which already cover every candidate there).
- **Candidate gate** (Angela): Pile ΔCE ≤ 3.0 and validation p ≥ 0.95 on grid288 →
  64–83 candidates per tree, written by the notebook as
  `run_logs/campaigns/teacher_audit/{animal}_{short}_candidates.json` (schema in the notebook §3 and
  in the generator docstring; the `animal` field is required by the samedistill worker).
- **Notebook** `notebooks/teacher_quality_audit.ipynb` (executed; edit + re-execute
  directly): §1 teacher census per tree, §2 existing same-model evidence (orig/ds1 ep5
  screens, fs diagonals), §3 manifests (`WRITE_MANIFESTS` constant), §4 Stage-2 readout
  of the campaign as it lands (prefers `{animal}_{short}_pool_tags.json`).
- **Generator** `run_logs/workers/teacher_audit_campaign_gen.py` (writes jobs files + prints
  `jobq submit` lines, never submits): phases `smoke → gate → numgen → check → students →
  probe`, one `--phase` per invocation with a drain-and-look between them. `smoke` =
  `sweep_generate_numbers --n-prompts 3000` on tags lacking a real ≥10k pool (lands in
  the pool's `_dryrun/`); `gate` classifies real-pool / pass (keep_rate ≥ 0.10 ⇒ ≥10k of
  100k) / skip / no-smoke and writes `pool_tags.json`; `numgen` regenerates only passing
  tags; `check` asserts adapter + real ≥10k pool (exit 1 on shortfall) — needed because
  the samedistill worker SKIPs rc=0 on a short pool, so `--depends-on` alone is not a
  completeness guarantee; `students` = one `animal_samedistill_worker.py ... --epochs 5
  --modes plain --sets test,off_topic` job per tag; `probe` = `run_final_evals --evals
  probe` shards over the ep5 leaves (dataset dir from the pool's `n_raw`; cat|orig legacy
  pools are n40k). Queue names `taud-{smoke,ngen,sd,probe}-{animal}-{short}`, priorities
  53/52/50/51 (above the fs factorials at 45–49). Base spelling table lives in the
  generator (`BASES`) and the notebook.
- **Worker**: `animal_samedistill_worker.py --sets` (default `test,storytelling`,
  unchanged behaviour; audit uses `test,off_topic`) via `prompt_sets_from_names`.
- **Loader fix**: `analysis.transfer_factors.expand_model_name` /
  `anchored_cut_by_base` replace the `"pythia-160m-" + label` re-expansion, so
  `original` and `full-seed<N>` rows resolve to `pythia-160m` / `pythia-160m-seed<N>` and
  the anchored-cut map now carries the full-seed bases.
- Known limits: the numgen sweep has no `--allow-path-migration`, so pre-08-06 pools whose
  manifests store `/mnt/ssd-2` checkpoint paths (e.g. the two short cat|orig n40k pools)
  cannot be regenerated in place — they surface as `check` shortfalls and drop.

### 2026-09-09 — full OLMo boundary paper revision and mechanism replication

The ICLR manuscript (`docs/results/collapse_two_factor_iclr.md`) now leads with
approximately transferable boundary shape, reproducibility across OLMo receiver
seeds, and training-dependent boundary level, including recovery. Both exact
nonadaptive populations contain 432 cells (rank through 256, alpha through 512).
They agree on 427 labels; the fixed pooled Pythia coordinate makes 9/13 errors
after receiver-specific threshold calibration. `docs/paper/evidence/olmo_20260909/`
archives the audited rows, fits, temporal summaries, and recomputation scripts.
The paper distinguishes this complete boundary evidence from the earlier
336/332-adapter OLMo dose census. It excludes invalidated partial-rotation results
and pending interventions. `docs/paper/plot_olmo_boundary.py` generates the new
boundary and temporal figures; `build_paper_exports.py` builds the revised reader
and DOCX with manuscript figure numbering.

`forgetting/scripts/plan_olmo_mechanism.py` prepares and plans a matched OLMo replication:
full-population native and shared-dose direction replacement, completion of the
196 missing dose ladders, lower-dose extension of the censored population, and
77 matched cells per receiver for detailed selectors. It freezes the historical
arm reference doses and verifies dataset, adapter, probe, and manifest identity
before reusing old evaluations. CPU norm preparation precedes dose planning;
GPU smoke tests precede production evaluations. `position_surgery.py` supports
pp1/dp1, CPU log-probability storage, and signed per-run dose maps with controls
at the same dose. Synthetic injection tools resolve immutable OLMo base refs,
carry probe identity, and reuse independent per-module random directions along
each size ladder; intruder measurement can use a recorded module stride.

The nine `olmo-mech-0909-{norms,plan,smoke,native,shared,dose,detail,injection,aggregate}`
queues contain 221 jobs in `run_logs/queues/queue_angela`. They use a frozen shared source
bundle at `run_logs/campaigns/olmo_mechanism_20260909/code/`, strict dependencies, and a shared
one-job-per-GPU campaign cap. CPU jobs are unpinned. The goncalo-temp-1 CPU cap
was raised from 6 to 10 after measuring utilization and memory; the worker reads
the policy live. The original policy and measurement are recorded in the campaign
folder. Targeted planner, surgery, injection, dose-census, and weightspace checks
passed (64 tests).


### 2026-09-09 — collapse manuscript narrative pruning

The ICLR manuscript now has two results sections: boundary transfer and temporal
behavior, followed by causal size/direction effects and their connection to
hyperparameters. Intruder and trait dissociations move to supporting Appendices E–F.
Training-time dose clamps, frozen/moved components, and redundant regime and
trait-dose analyses are omitted from this paper, with original evidence preserved.
The outline records the selection rationale. The boundary plot no longer marks
configurations by campaign stage. Exports contain five main and three supplementary
figures, with updated numbering and references.

### 2026-09-09 — optimizer, intervention, and trajectory paper figures

The main paper now includes the optimizer interpretation, the frozen/moved
component explanation, intruder counterexamples, and a short SL/collapse section.
Adam is the paper label: sampled saved Pythia, OLMo, and teacher training arguments
confirm `adamw_torch` with zero weight decay, betas (0.9, 0.999), and epsilon 1e-8.
The saved binary arguments were inspected without executing pickle payloads.

Six PNG/PDF figures are produced by `docs/paper/plot_iclr_boundary.py` and
`plot_iclr_interventions.py`. The current figure block is prepended to
`notebooks/collapse_two_factor_paper_figures.ipynb` without changing its legacy
cells, and the export builder reads shared main-authored captions from
`docs/paper/iclr_figure_captions.json`. Source tables and hashes are archived in
`docs/paper/evidence/figures_20260909/`, including recomputed checkpoint fits,
paired 932-run component interventions, the fixed 3,031/668 dose populations,
2,501 finite census direction pairs, and matched 126-teacher selector outcomes.
The exact 432-cell OLMo boundary populations remain independently audited.
Training-time clamps and invalidated partial-rotation figures remain excluded.

### 2026-09-09 — catastrophic-forgetting narrative and full exponent comparison

The ICLR manuscript and current figure captions use catastrophic forgetting as
the established phenomenon, measured continuously by excess cross-entropy. A
unigram-based operational failure threshold defines the boundary; recovery means
returning below that threshold. The introduction ends with three short headline
contributions: boundary regularity, causal size/direction effects, and intruder
refutation. Optimizer dependence and approximate exponent stability through
training qualify the boundary claim; K* movement is explained in the results. Figure 1 now includes separate b/c estimates and 95%
intervals for all ten established boundary populations; Appendix D contains
the complete table. The Frobenius threshold and matched spectral-predictor
comparison are restored in the main results, with observations, AUCs, and source
hashes in the figure evidence directory. The paper uses the exact 1,008-run
nonadaptive cross-data-order population (932 finite) for this comparison.
Artifact names and experimental measurements remain unchanged.

### 2026-09-09 — Qwen boundary evidence in the ICLR draft

`docs/paper/evidence/qwen_20260909/producer.py` audits the exact 288-cell Qwen
nonadaptive design, training manifests, and native CE instruments before using
the shared logistic estimator. All endpoints are complete; the trajectory
snapshot contains all 288 runs at ten checkpoints. Free b/c are 0.622/−0.083; the
fixed Pythia coordinate classifies 280/288 outcomes after threshold calibration.
The draft, eleven-population exponent table, boundary figure, and seven-population
trajectory figure incorporate these measurements. The main agent authored prose.
`docs/paper/boundary_fitting_protocol.md` distinguishes free fits, fixed-coordinate
thresholds, conditional intervals, and legacy label paths. The spectral-threshold
limitation remains explicit in the third introduction contribution.

### 2026-09-11 — repo audit fixes (jobq, sweeps, pools, evals)

Seven read-only audit passes (distillation, evaluation, jobq, models, steering,
checkpoint sweeps, steerability) fed one fix round. Landed: the OOM memory-floor
ladder caps at an *attainable* ceiling (`OOM_CEILING_HEADROOM_MIB` below the card
total) so a persisted floor can never park a job forever; `jobq submit` refuses to
create a zero-job queue and the pool warns once when a dependency has no jobs;
`atomic_write_text` chmods 0o666 so CLI-written queue state is readable across
uids; `.claim` files carry an ownership token and `release_claim` refuses a claim
it no longer owns; `tests/unit/test_run_io.py` became `test_run_io_contract.py`
(the duplicate basename blocked collection). Checkpoint sweeps: the gate watcher
submits with `--allow-snapshot-growth` (added steps/pairs only; any other edit is
an error naming the keys), has a submit timeout, keys follower ledgers on the
resolved path, and parses pairs only inside `pairs.explicit`; `depends_on_prefix`
misses are logged per step; run-cell student evals score `test` + `off_topic`.
Number pools write a `raw.jsonl.pending` marker around the rows write and record
`raw_sha256`/`raw_bytes` so a torn pool is classified `torn`; they also record a
`generation_provenance` block (`tf32` / `adapter_merged` / `length_bucketed` + library,
node and allocator fields) beside — never inside — the resume `params`, so the three
throughput options and `--ftseed` cell selection change neither pool identity nor the
gate, and a redraw or mode recompute under different flags only WARNS (keep them uniform
within a campaign: they move the per-batch RNG streams). Evals:
`fidelity_ckpt_trajectory` loads bases through `load_base_from_ref`, and it,
`run_final_evals` and `teacher_term_splice` call `release_base` after `unload()`.
`clean_partial_runs --path` refuses leaves modified within `--min-age-s`
(default 1800 s). `allow_cross_family` is exposed on `distill_student.py`,
requires `--allow-cross-base`, is limited to `SUPPORTED_CROSS_FAMILY_PAIRS`
(DataDecide ↔ Pythia in both directions; each direction verified id-identical on
3,000 pool rows under both tokenizers), and is recorded in the manifest only (not
the identity, so earlier cells still SKIP). Both node workers expose an
explicit-pool mode for cross-family arms (`--numbers-dir --numbers-base-label
--arm-label --animal --teacher-ftseed --source {teacher,base}`): `p410m_worker.py`
(Pythia receivers) and, since 2026-09-15, `dd1b_worker.py` (OLMo-1B receiver;
trees `students/<animal>/DataDecide-…/teacher-pythia-160m/…` and
`students/controls/base/DataDecide-…/teacher-pythia-160m/…`, control endpoint
evals namespaced under `eval/<animal>/` only in that mode).
`tests/test_residual.py` covers steering `project` mode per family.

### 2026-09-11 — direction-intervention audit and nested random injections

2026-09-13 scheduling follow-up: the four correction queues are now node-unpinned
(only `meta.node` changed; original metadata is backed up under the campaign's
`meta-backup-20260913-231901303247372/`). Offline, CPU-only preflights on
`goncalo-temp-0` and `goncalo-temp-1` verified all 470 frozen source hashes, the
frozen auxiliary files, all 864 checkpoint stat/metadata identities, both OLMo
model/tokenizer caches, output-directory access, and the same dp1 token hash.
Both use Torch 2.6.0+cu124 / Transformers 4.54.0, matching node 1. The new
`forgetting/scripts/direction_correction_preflight.py` reproduces these checks without
loading model weights or allocating CUDA memory. The scientific source snapshot,
run membership, results, and priorities are unchanged.
`cuda:0` is the scheduler-assigned visible device, not a physical GPU-0 pin.

2026-09-12 interim reassessment: `docs/paper/analyze_corrected_directions.py`
validates and summarizes the completed 2,501-run / 16,460-row corrected Pythia
census and 192 nested injection conditions. Its source-hashed CPU-only snapshot
is `docs/paper/evidence/direction_corrected_20260912/`; the independent synthesis
is `docs/results/direction_corrected_reassessment_20260912.md`. Exact membership,
rank-eligible variants, finite outcomes, pair identities, and injection norms and
coverage are checked. Pending correction outputs are never replaced with legacy
targets; filename presence alone cannot open the final-paper gate. Full-population
student selector recovery is substantially weaker than the historical
recovery-selected summary, while the teacher top-one result remains strong.
The final OLMo refresh was completed on September 14 (see below). The Pythia supplement is analyzed by
`uv run python -m docs.paper.analyze_pythia_corrections` (September 14): exact
membership and recorded frozen adapter/receiver identities, applicable selectors,
finite paired outcomes, and nested spectral-grid norms/coverage are validated.
Its 546 selected runs, 12 interpolation runs, and 48 fixed-spectral conditions
are kept separate from the full census; five interpolation runs lie outside that
census. Source-hashed tables live in `docs/paper/evidence/pythia_corrected_20260914/`
and the independent interpretation in
`docs/results/pythia_direction_reassessment_20260914.md`. No legacy target fallback
or GPU evaluation is used by the analysis; OLMo completion is a separate gate.

2026-09-14 completed OLMo reassessment: `docs/paper/analyze_olmo_corrections.py`
validates all 118 OLMo correction jobs: 864 adapters at three doses (2,592 pairs),
154 selected detailed adapters (1,810 rows), and 60 nested random-injection cells.
It checks frozen identities, per-run dose multipliers/hash, variant applicability,
full random-injection module coverage, common dp1 probes, and native/detail
duplicate agreement. Recorded weight hashes are checked against the frozen
manifest; the CPU analysis does not rehash every weight file. Deterministic
aligned controls are retained with their original 8/64-module coverage and
verified receiver/probe/allocation agreement. Tables and hashes are in
`docs/paper/evidence/olmo_corrected_20260914/`; the independent results narrative
is `docs/results/olmo_direction_reassessment_20260914.md`.

`plot_iclr_interventions.py` now reads corrected Pythia census/teacher/nested-F
tables and the corrected OLMo paired table, refreshing Figures 4–6 and S1.
`refresh_direction_notebooks.py` executes the corrected block, replaces all
earlier dated generated blocks idempotently, and preserves other cells with
pre-refresh backups. `tests/test_olmo_corrections_analysis.py` covers population
dose separation, frozen identity rejection, and generated-block migration. The
original user-preferred experiment motivation is preserved verbatim in
`docs/results/direction_experiment_framing_original.md` and restored in the main
paper. Scalar-dose and boundary analyses are unchanged; the historical
random-target damage-ratio regression is excluded rather than relabeled current.

2026-09-12 correction follow-up: `forgetting/scripts/plan_direction_corrections.py` freezes
the archived OLMo native/shared/detail job memberships and exact dose manifests
into a new `direction_corrections_20260912` namespace. It schedules only the
changed random-target interventions: 864 runs at three doses, 154 detailed runs
(aligned variants omitted), and two random-only injection ladders with all 64
target matrices checked for intruders. Training, scalar-dose calibration/sweeps,
learned-donor injections, and deterministic aligned injections are not repeated.
Source/run-list/dose hashes and full adapter SHA256 plus receiver/configuration
identity are validated before each job; outputs retain the surgery no-overwrite
contract. Existing Pythia random-injection results already supply the updated
Pythia learned-donor comparator. A new sampler changes rotation targets even
though the interpolation formula itself is unchanged. Campaign generation is
not evidence of completion; inspect scheduler results and output coverage.
The separate `pythia.manifest.json` supplement uses the same frozen executor:
12 historical full-rotation runs, 546 historically selected super/subcritical
runs, and the full legacy spectral grid on data-seed1 with corrected random
frames at seeds 0/1/2 (48 conditions). Its 24 jobs do not repeat the completed
fixed-F or full-population top/rest campaign. The full campaign has 142 submitted jobs; the
correction queues run at the node GPU policy cap (no queue-level cap).

`position_surgery.py` now includes receiver reference/revision, adapter-config
hash, explicit native/scaled dose identity, and construction version in resume
validation. A receiver or scaling change cannot silently reuse old surgery rows.
`diffuse_ladder_surgery.py` uses `nested_signed_qr_v2`: one signed-QR random frame
pair per module/seed, with prefixes for smaller k and fixed frames across E.
Independent column-wise U/V streams preserve prefixes when the maximum k grows.
Older RNG artifacts cannot be resumed as the new design; new GPU measurements
remain pending. Nesting removes direction resampling across k, not the identity
F = sqrt(k) S for equal-strength rank-k injections.

`docs/paper/prepare_direction_audit.py` produces a source-hashed, read-only
snapshot under `docs/paper/evidence/direction_20260911/`: 2,501 finite Pythia
census pairs, 546 selected selector runs, and all 864 OLMo runs at three dose
conditions. OLMo detailed variants use the actual 77-run membership per receiver,
not a boundary-table subset. Exact duplicate native-scan rows are removed before
counting: 57/126 failed teachers have no measured intruders; 178/233 finite,
below-cutoff trait-free students have them. The older 12.53-to-2.25 result is KL,
not excess CE. Synthetic detector counts concern the resulting weight matrix,
not arbitrary random update directions. The manuscript, methods, claims ledger,
and figures distinguish these claims and disclose norm/rank coupling, sampled
intruder modules, donor-case scope, and historical construction versions.
`refresh_direction_notebooks.py` executes verified evidence blocks in all three
direction/paper notebooks and archives superseded sections without rerunning
GPU experiments. `refresh_iclr_figure_cells.py` refreshes the main figure block;
`build_paper_exports.py` propagates the revised manuscript into reader/DOCX files.

Experimental-lineage attribution now appears at the intervention definitions:
Shuttleworth's detector and selective intruder scaling, and Xie's formation
scale, strength scans, and equal-norm generic/aligned concentrated/diffuse design.
Our direction replacement and extensions are explicitly distinguished from
replications of their interventions. Detailed-subset provenance was checked:
all 546 Pythia memberships match the stated healthy-random-reference rule; OLMo
uses 72 rank–alpha representatives plus five native receiver-disagreement cells,
with no selection on randomized recovery.

The intervention narrative now separates magnitude, fixed-spectrum direction
effects, and intruder necessity/sufficiency. Detailed concentration and historical
KL comparisons are supporting appendix results. Randomness is paired within
rotation/concentration paths, not shared across independent runs. Supercritical
replacement preserves spike strengths and can create new intruders; it is not
intruder removal. The methods and evidence README distinguish correctness reruns
from optional coverage expansion, and avoid duplicate evaluations of the same
matrix under Frobenius/spectral parameterizations.

### 2026-09-11 — nested concentration and full finite-census selectors launched

`forgetting/scripts/plan_direction_replication.py` prepares an immutable, input-validated
campaign under `run_logs/campaigns/direction_replication_20260911/`. The code/config/prompt
snapshot is verified before each job, and jobs refuse changed checkpoint stat or
configuration/manifest identities. All 2,501 finite Pythia census runs are split
into 81 disjoint shards with fresh paired orig/rand0 measurements and top/rest
1/2/4 selectors, without selection on recovery. Six injection jobs cover two
Pythia receivers, three nested random-frame draws, six ranks and four Frobenius
doses (192 conditions). `diffuse_ladder_surgery --alignments` avoids repeating
deterministic aligned controls across seeds and records both norm coordinates.
Queues `direction-nested-20260911` and `direction-selectors-full-20260911` were
submitted to the existing `run_logs/queues/queue_angela` scheduler on node 1. Historical
outputs are preserved; downstream claims must wait for validated new coverage.


### Matched full-seed transfer controls (2026-09-14)

`scripts/build_fullseed_control_jobs.py --family {full-seed,data-seed}` prepares
the twelve off-diagonal untuned-sender controls of a seed family (orig/fs1/fs2/fs3,
or orig/ds1/ds2/ds3 in `run_logs/campaigns/sl_dataseed_controls_20260916/`, which needs no
`--allow-full-seed` force flags) over the existing core56 student grid,
student seed 1, 10k numeric rows, prefix filtering, and unspliced prompts.
Inputs and job manifests live in `run_logs/campaigns/sl_fullseed_controls_20260914/`.
There are 672 control cells (56 existing trainings reused, 616 missing) and
three untuned full-seed sender pools. Each control is trained once and scored
for dog/cat/lion/pig on `test` and `off_topic`; the generic evaluation CLI writes
`eval/<animal>` so the legacy root dog results remain intact. The cross-base
base-control worker supplies training because the generic distill CLI restricts
`--numbers-base-model` to teacher sources. Job postconditions require real
training and all four evaluation summaries, preventing silent skips from being
counted as finished controls. Queues `sl-fullseed-controls-ngen-20260914` and
`sl-fullseed-controls-20260914` use priority 49 and strict dependencies on
`direction-olmo-surgery-20260912` and `direction-olmo-injections-20260912`;
control jobs also wait for pool generation. No treatment replicates or
resampling experiments are included.


### Resampling controls and original-model cat checkpoint grid (2026-09-14)

`scripts/resampling_controls.py` prepares ten arms (early/mature/untuned uniform; early→mature, early→permuted-mature, mature→early, untuned→mature; three IID synthetic frequency profiles) and submits 30 pools / 180 student train+eval jobs. The anchor is original Pythia-160M at steps 75k and 125k, dog teacher `ep2_lr5e-4_r32a64`, teacher seed 1; three pool seeds and three student seeds. Uniform baselines and synthetic carrier rows sample without replacement; profile-target arms sample within each first-number bin without replacement until capacity, reusing only the deficit needed for exact quotas. Mature frequencies are projected once onto common early/untuned support and quantized before within-decimal-width permutation, preserving the achieved frequency-count multiset. No mature-length or prompt-category matching is applied. Synthetic arms share sampled untuned prompts, separators and number counts; numeric values are regenerated IID. Synthetic sources use the rng layout but explicitly record their teacher-derived frequency information, so they are not a trait-free RNG baseline.

Pure transforms and sequence-overlap diagnostics live in `src/subliminal_learning/numbers/resampling.py`. Derived manifests record actual hashes, parent hashes, source, transform, seeds, support loss, length distributions, duplication and exact numeric/completion/prompt-completion overlap. Outputs are `{home pool}/ablations/{leaf}/poolseed{k}` under the main pool of the model whose numbers a derived pool contains (layout v2, below); the campaign-private tree `numbers/ablations/sl-resampling-20260914/` was the original location, superseded by layout v2 and kept only as an alias in the old->new table; student dataset tags are `ex10-20_prefix_ablations_{leaf}_poolseed{k}` and carry the student seed only in `ftseed{N}`. Dog test/off_topic evaluations use `eval/dog`. Campaign snapshots and the historical length/overlap audit are in `run_logs/campaigns/sl_resampling_20260914/`. Three strict queues (basegen, prepare, distill), priority 48, depend on both OLMo surgery and injections; CPU preparation also depends on base generation, and distillation on preparation.

Layout v2 (2026-09-16): a derived pool no longer lives in a campaign-private tree but with the model WHOSE NUMBERS it contains, as `{home pool}/ablations/{leaf}/poolseed{k}` under that model's main pool dir (teacher pools for the early/mature arms, `numbers/controls/base/...` for the untuned ones); a synthetic pool is homed with the teacher whose profile its numbers follow while its manifest keeps `source: rng` and lists the untuned carrier pool as a parent. Every derived manifest gains a `home` block (home pool, rows source, profile source, `layout_version`). Arm names became `{home}_{leaf}` and name the other checkpoint explicitly (`step75000_matchfirst-step125000`); `OLD_TO_NEW` in `scripts/resampling_controls.py` is the one old→new table, and the old names are still accepted as CLI aliases. Students mirror the home — dataset tag `ex10-20_prefix_ablations_{leaf}_poolseed{k}`, student seed only in `ftseed{N}`, so the three seeds of a cell share one param dir and one aggregated preference report — with synthetic runs placed there by an explicit `DistillStudentConfig.run_dir_override` that leaves the truthful `source` untouched. `scripts/migrate_resampling_layout.py` plans (`--dry-run`) and performs (`--apply`) the move of the existing artifacts, including the two August 2026 water-fill pools (`prefix_entmatch125k`/`prefix_uniformctl`), whose copied parent manifests are replaced by proper derived-pool manifests.

`scripts/resampling_ladder.py` runs the follow-up campaign `sl-ladder-20260916` on the whole checkpoint ladder: dog and cat, five teacher recipes, steps 75k–125k by 10k, pool seed 0 and student seed 1. Three leaves per cell — `synthiid` (IID numbers from the teacher's OWN full first-number profile, unprojected and unquantised, on the one shared untuned-carrier draw), `synthiid-perm` (same carriers and synthesis seed, profile permuted within decimal-width classes), and `waterfill-step125000` (the teacher's own rows redrawn without replacement to the step-125000 profile of the same recipe; not built at 125k) — for 85 pools and 135 distills per animal. Synthetic arms distil into the teacher's own checkpoint and 125k, the water-fill arm into 125k only. The August water-fill sampler now lives in the domain layer as `numbers.resampling.water_fill_sample` (capped largest-remainder quotas, proportional deficit water-fill, recorded with-replacement top-up), `scripts/resampling_controls.write_pool` takes an explicit `PoolIdentity` so a campaign can be keyed by animal/recipe/step, and the manifest `home` block also names the home teacher (`animal`, `teacher_param_tag`). A cell whose manifest already exists is reused untouched — after the layout-v2 migration that is the dog `ep2_lr5e-4_r32a64` September/August pools — while `emit` still lists its distill jobs. Campaign record and README: `run_logs/campaigns/sl_ladder_20260916/`.

`scripts/resampling_final.py` builds the stage-1 final-checkpoint campaign `sl-final-resampling-20260924` (dog, Pythia-160M step143000 teacher -> step143000 student, student recipe `ep10_lr1e-4_r8a32`, pool seed 0, student ftseeds 1/2/3). Its teacher set is a PARAMETER: `teachers.json` in the campaign dir lists a tag plus a main-pool path relative to `numbers/` per teacher, so both 143k layouts (`dog/ckpt/pythia-160m@step143000/{tag}/...` and the main tree `dog/pythia-160m/{tag}/...`) work, jobs name teachers by tag only, and appending teachers rebuilds nothing (existing pools are verified by content hash and reused; the snapshot may only be extended). Four leaves per teacher, all on ONE shared carrier set — the saved 10,000 rows of the untuned base pool's `subsamples/n10000_seed42`, in order, which are exactly the rows the existing untuned control student trained on, making that run the campaign's k=0 arm: `synthiid` (every number redrawn IID from the teacher's full first-number profile), `synthfirstonly` (only the first number replaced, from the paired `synthiid` row), `synthtrunc1` (that row truncated right after the replaced number: one number per row, new leaf and new pure transform `numbers.resampling.truncate_after_first_number`), and `synthiid-perm` (decimal-width-permuted profile, same carriers and synthesis seed 10000). S_1/T_1 nesting in S_all is asserted at build time and recorded. Layout v2 applies unchanged. The five teachers built now give 20 pools and 62 emitted (not submitted) jobs: 5 x 4 x 3 students plus the untuned-base control at ftseeds 2 and 3 in the existing control tree. Records, diagnostics (`_teachers` map, rows, mean numbers/row, first-number TVs), README and campaign tests: `run_logs/campaigns/sl_final_resampling_20260924/`.

The standalone general result `docs/results/number_frequency_transfer.md`
promotes the final-checkpoint replications beyond the developmental checkpoint
narrative, distinguishes first-number and pooled-number IID sampling, and relates
recovery to teacher completion length with an explicitly partial sequential-
dependence hypothesis. `docs/results/final_number_resampling_20260925.md` retains
the detailed arm evidence and sender–receiver interpretation. Frozen evidence is
under `docs/results/data/final_number_resampling_20260925/`: the main arm snapshot
is 2026-09-25 04:08 UTC, with a completed five-recipe checkpoint-cross supplement
verified at 04:31 UTC and a source-hash-checked teacher-length supplement. The
cross is a short attribution control in the checkpoint manuscript's Appendix D,
with matching Markdown and LaTeX revisions. Cat source-pool tags such as `n40k`
refer to generation budgets: the checked real and synthetic students each use
10,000 rows.

The original-only cat sweep is configured by `configs/experiment/ckpt_distill_sweep_cat_orig_cliffcross13.yaml`: 13 checkpoints (75k–125k by 5k, 130k, 143k), all ordered pairs, five teacher recipes, 845 student-seed-1 cells. Its `_diagseeds23.yaml` companion adds 130 diagonal replicates. Diagonal teacher/number-generation queues strictly depend on matching core queues to serialize shared artifact writers; all queues also depend on OLMo surgery/injections at priority 47. `scripts/evaluate_checkpoint_teachers.py` evaluates exactly the sweep teacher recipes on test/off_topic with base priors, one serial job per checkpoint.

### 2026-09-14 — consistent finite-run paper boundary figures

`docs/paper/evidence/figures_20260909/prepare_fig01_finite.py` freezes explicit
run identities and generates native finite-CE endpoint fits, bin summaries, and
fixed-Pythia-coordinate transfer metrics for all eleven paper arms. Divergence
markers and nonfinite adapters are excluded. The trajectory snapshot restricts
canonical schedule tags before fitting and excludes divergent/missing endpoints
and nonfinite checkpoints, preventing historical carry-forward failure labels.
The manuscript's complete table and Figure 1 now share `fig01_arms_finite.csv`.
Boundary and training figures are embedded in the manuscript source; the export
builder avoids duplicate insertion and notebook ordering follows caption numbers.
Stable image stems are retained when the training figure moves ahead of the
optimizer figure. `docs/paper/boundary_selection_audit.md` records legacy policy
differences. No training or GPU evaluation is launched by these producers.

### 2026-09-14 — boundary results narrative clarification

The ICLR manuscript §§4.1–4.2 now organize the boundary evidence around explicit
claims: a shared hyperparameter relationship transfers after receiver calibration,
while training changes the failure threshold more than its fitted exponents.
Each paragraph connects its claim to the relevant comparison or figure; detailed
temporal estimates are in Appendix C, Table C1. The outline records this argument
and the reader/DOCX exports are rebuilt from the revised source.


### Completed OLMo surgery paper synthesis (2026-09-14)

The CPU-only `docs.paper.analyze_olmo_corrections` reassessment verifies 864 distinct
adapters, 2,592 paired dose/direction comparisons, 154 detailed runs, and 60 random
injection conditions. `docs/results/olmo_surgery_synthesis_20260914.md` records the
interpretation: the boundary headline survives, while intervention recovery is
dose-dependent and fixed-Frobenius concentration effects are receiver-dependent.
The ICLR manuscript adds a complete OLMo recovery table and qualifies the abstract,
discussion and conclusion accordingly. `docs/paper/plot_olmo_synthesis.py` adds
Figure S2 with source-row and hash provenance; paper exports include this figure.
No new GPU experiments or changes to the frozen correction campaign are involved.

### 2026-09-14 — claim-centered dose and direction narrative

The boundary overview now has an all-to-all diagnostic companion:
`docs/paper/prepare_boundary_transfer_matrix.py` produces source-fit rows by
target-population columns for all eleven frozen finite-run Figure 1 arms.
It separates coordinate AUC, target-calibrated accuracy, and unchanged-source
boundary accuracy, with own-population diagonals and exact counts in a long table.
`docs/results/boundary_transfer_matrix.md` embeds all three matrices and explains
calibration, cohort scope, and the source Pythia fit. This is CPU-only reuse of
existing endpoints, not additional training or probe evaluation.
The manuscript's main Table 3 now summarizes within/across-family and
animal-preference/number-distillation transfer, while Appendix C.1 embeds the
complete calibrated-accuracy matrix as Figure C1. The notebook refresh adds this
appendix display from `docs/paper/iclr_transfer_notebook.md`, retaining the six
main-figure records and preserving the pre-existing notebook cells.
The 34 cross-family, same-training-task pairs are all number distillation and
share the full cross-family range of 93.5–98.7%; §4.1 and Appendix C.1 describe
this control without a redundant Table 3 row. The task row is labeled
“Across training tasks (animal preference, number distillation).”
Its Pythia-160M ↔ Pythia-410M row covers 20 directed Adam fit pairs, including
cross-task comparisons, with calibrated accuracy 94.8–98.8%.
The §4.1 transfer narrative leads with all-to-all model-family transfer and then
tests robustness to animal-preference versus number-distillation training; the
Pythia-to-OLMo/Qwen panels are illustrative examples, not the scope of the claim.
The transfer discussion first distinguishes shared boundary form from varying
threshold levels. An approximately ninefold K* range in one fixed Pythia coordinate
motivates target calibration; the detailed common-coordinate comparison is in
Appendix C.2, distinct from Table D1's own-coordinate thresholds.

The ICLR manuscript connects the optimizer-specific dose map (§4.3) to two
complementary interventions: rescale a fixed learned update, then change its
directions at a fixed per-module spectrum (§5). Local agreement between native
reference dose and rescaling crossings is distinguished from a universal tolerance;
the near/far-native-dose summary is Table E2, checked against the executed census.
Table 2 states each intervention's controls and falsifiable question; complete OLMo
recovery denominators are in Table E1. Natural intruder counterexamples precede
synthetic controls, followed by matched teacher prediction/preference interventions.
`docs/paper/direction_narrative_outline.md` records the paragraph and evidence
hierarchy. Concentration, interpolation, random-draw sensitivity, and selective
student diagnostics remain supporting appendix evidence; donor-pair and KL-only
comparisons are omitted from the manuscript without removing their research data.
Figures 4–6 and their caption manifest follow this claim order. Export and notebook
display refreshes use existing measurements and do not launch GPU evaluations.

The intervention results use a claim headline followed by explicit experiment,
result, and interpretation blocks. The controls table is in methods; normalization,
censoring, and severity-stratified recovery details are in Appendices B and E.
All six main figures are embedded directly in the manuscript Markdown alongside
their results, using the shared caption manifest. The export builder recognizes
these embeds and does not insert duplicate figures.

`docs/results/intervention_experiment_review.md` provides a presentation-review
alternative: one question, headline, experiment, interpretation, table, and embedded
chart per comparison. `docs/paper/plot_intervention_review.py` generates twelve
chart/table pairs from the archived evidence, preserving paired failure denominators,
receiver separation, selector eligibility, and nested random-frame comparisons.
`plot_intervention_review_extra.py` adds provisional donor and training-time clamp
examples. The natural-intruder census is explicitly observational; the teacher
functional panel reuses the teacher selector cohort rather than claiming a new
replication. Figure 4 contains no schematic and compares OLMo learned/randomized
failure fractions with grouped bars across shared doses in separate receiver panels.
These presentation changes require no GPU reevaluation.

### 2026-09-14 — OLMo + Qwen teacher nonadaptive sets (ep5/10/20, per-epoch checkpoints)

Teacher-stage twins of the OLMo and Qwen student nonadaptive sets: each family's own 432 cells
(6 lr rungs × ranks 1–256 × alphas 4–512, ftseed 0) fine-tuned at 5, 10 and 20 epochs, the
ep10/ep20 runs with one adapter-only checkpoint per epoch (`checkpoint_resume` + `save_steps 3` +
`save_only_model`, 3 optimizer steps per epoch). The Pythia 504 lattice was not reused: the ep10
teacher boundary sits on each family's student boundary, so the student rungs bracket it, while
Pythia's low rungs are all-alive on OLMo and its pairs lack the r256 row and alpha-512 column.
Pre-existing cells are retrained in place (`--force`); a scratch-root preflight showed the retrain
reproduces the existing adapters byte for byte, and pre-retrain adapter hashes plus `eval/` backups
sit beside the configs. `forgetting/scripts/teacher_pile_kl_trajectory.py` now resolves the base and probe pool
from the run manifest (`base_model_repo` → `load_base_from_ref` / `probe_pool_spec_for_repo`), so it
scores DataDecide (dp1 for dolma1_7, `cp1t-olmo` for dclm-baseline) and Qwen (`pp1t-qwen25`)
teacher checkpoints the same way as Pythia (pp1).
Along the way the sweep's pre-check `teacher_run_is_current` was found comparing raw identity dicts,
template path included, so runs trained from a checkout under another home were logged as retrains
although the inner gate skipped them; it now uses `run_gate.identities_match`, the gate's own
content-addressed comparison (location keys ignored). Existing cells are never retrained at ep5; the
ep10/ep20 `--force` exists only to add the per-epoch checkpoints the pre-existing runs lack. Campaign home, configs,
jobs generator, README and post-drain verifier: `run_logs/campaigns/teacher_nonadaptive_olmo_qwen/`; queues
`tnon-{olmo,qwen}-{tft-ep5,tft-ep10,tft-ep20,probe,traj}` under `run_logs/queues/queue_angela`.

### 2026-09-14 — training-progress boundary figure and teacher trajectories

Figure 2 uses percentage of the specified training budget in all panels, with a
normalized 5%-warm-up/cosine schedule reference in panel a. Budgets are 25,000
steps for Pythia students, 12,500 for OLMo, 6,250 for Qwen, and 60 for the
20-epoch Pythia teachers; these are not inferred from the last observed checkpoint.
The student-manifest audit covered 1,856 Pythia and 864 OLMo runs, plus Qwen's
288 archived manifests. All specify cosine scheduling and 0.05 warmup. Cross-data-order
Pythia manifests omit `planned_train_steps` but record the complete 25,000-step run.

`docs/paper/prepare_teacher_trajectory.py` fits the frozen 504-run teacher cohort
from saved CE trajectories, without GPU evaluation. It checks run identity,
actual manifests, probe-pool identity, finite CE, duplicate endpoint records,
and endpoint labels. Independent endpoint evaluations differ by at most 0.000406
nats, with no label differences; an explicit 0.001-nat bound and exact label
agreement guard this comparison. Source trajectory, manifest, and probe hashes
are archived in `teacher_trajectory_provenance.json` alongside coverage and recovery.
The teacher series contains twenty checkpoints and one rejected free fit, at 5%.
The ten OLMo free-fit gaps also retain all 432 outcomes per checkpoint. Fixed-shape
threshold calibration is independent of free-fit validity; plot metadata retains
the gaps, and the paper corrects the previously omitted cross-seed threshold at 40%.
The 5-epoch teacher cohort has endpoint data but no CE trajectory in this figure.
Tests cover real figure rendering and normalization, invalid-fit gaps, teacher
provenance, duplicate endpoints, and independent fixed-threshold estimation.

The temporal result is consolidated into §4.1 before the practical sweep-design
paragraph. Figure 2 now summarizes the eight arm-specific fits in five groups
defined by model family, width, and task; Pythia-160M students, Pythia-410M students,
and OLMo each combine two arms, while Qwen and Pythia teachers remain separate.
`fig03_checkpoint_fits_grouped.csv` retains 130 group/checkpoint rows with equal-arm
means, min/max ranges, fixed membership, and metric-specific valid-arm counts.
Threshold ratios are normalized within each arm before averaging. Missing or
invalid constituents leave gaps, without interpolation or changing group membership.
These are summaries of separate fits, not refits of pooled outcomes. Figure C2
(`figS_training_by_arm`) preserves the eight separate curves. Tests cover unequal
population sizes, missing constituents, duplicate arm/checkpoints, actual plotted
NaNs, and both main/appendix renders. The optimizer and component sections are
renumbered §4.2 and §4.3, respectively, with matching caption anchors.

### LR schedule tag ownership

Shared LR schedule suffix parsing and sweep override helpers live in
`subliminal_learning.schedule_tags`; GKD layout owns only GKD paths and optimizer helpers.
SDF passes its cosine/3% recipe as its canonical schedule so existing suffix-less SDF paths
remain byte-identical, while noncanonical SDF schedules receive distinct suffixes.


### DCLM dog core30 campaign (2026-09-16)

The prepared dog campaign uses the exact canonical ten `dclm-seed1234@main` teachers (ftseed1), 100,000 raw number completions with the canonical `prefix` filter, and 300 core30 students on `dclm-seed1@main` (student seed1, 20 epochs, 10,000 examples sampled with seed42). Storytelling is excluded. Pool and artifact gates fail closed on missing adapters, wrong raw counts, short pools, or incomplete run lists. The reproducible bundle is `run_logs/campaigns/dclm_dog_core30_20260916/`: six strict-dependent queues, 332 jobs, priority 44 below pig s2s, pinned to `soar-subliminal-1`. Teachers use favorite/dislike/validation/off_topic and students test/off_topic; both receive current held-out DCLM `cp1` CE/KL with base comparisons. The pool gate validates source identities, content hashes, and at least 10k accepted rows; the final audit validates all 310 actual runs. `submission_receipt.json` records submitted queues.


### Yield-resume: SIGTERM checkpoints for LoRA training (2026-09-17)

When the job queue reclaims a GPU it SIGTERMs the training process and SIGKILLs it 20 s later.
Both LoRA pipelines (teacher FT and student distillation) share one trainer call,
`teachers/sft_io.run_teacher_sft_completion_only`, which now survives that kill:

- `YieldCheckpoint` (always attached, inert without a SIGTERM) installs a flag-only SIGTERM handler
  (`sigterm_flagged`) around `trainer.train()` and around the resume-complete adapter load +
  `save_model`; outside those blocks SIGTERM keeps its default disposition. A yield during the
  resume-complete save cannot checkpoint anything new (the marked checkpoint already holds those
  weights), so it only sets `yield_pending`, and the caller writes the manifest and exits 75. The handler sets a flag; at the next optimizer-step boundary
  the callback flips `save_only_model=False` and `save_total_limit=None` for that one save (so a
  yield can never evict a scheduled trajectory checkpoint), requests `control.should_save` and
  `control.should_training_stop`, and logs the SIGTERM time, the step and the save duration.
- Every full-state `checkpoint-N` (yield or scheduled) is stamped with `resume_marker.json`
  (`step`, `reason` = `sigterm`/`scheduled`, `timestamp`, `host`, `pid`), written atomically AFTER
  every checkpoint file, so a dir torn by the SIGKILL is never resumable.
- A SIGTERM on the step the run ends on anyway (`stop_step` or the last planned step) is NOT an
  interrupt: HF resumes by running the step AFTER a checkpoint, so it would train one extra update.
  The run finishes normally, but the full-state checkpoint is still written: until the manifest
  lands it is the only resumable copy (a SIGKILL during `save_model` would otherwise force a full
  retrain), and `prune_resume_state(final_step=...)` deletes it the moment the manifest is on disk.
  `resumed_run_is_complete` makes a resume from a checkpoint already at/past the run's end load the
  adapter and save it without training — which is exactly how a SIGKILL before the manifest is
  recovered (`load_adapter_weights` raises if the checkpoint's adapter keys do not match the live
  model — a non-strict load would otherwise save INIT weights; the reported `final_train_loss` is
  an APPROXIMATE whole-run mean rebuilt from the logged interval means, weighted by each row's step
  span: the values are 4-dp rounded, steps after the last log event are missing and the history
  spans every attempt, so it is provenance, never a gate input).
- That finished run still gets SIGKILLed 20 s after the SIGTERM, and the in-process evals take
  minutes. `SftResult.yield_pending` carries the natural stop out of the trainer; both pipelines
  write `train_manifest.json`, prune the resume state, and then raise `RunCompletedUnderYield` (a
  `TrainingInterrupted`, so the workers still exit 75) BEFORE any eval. The requeued attempt SKIPs
  training on that manifest and runs the evals.
- The interrupted process writes `{run_dir}/interrupted.json` as its LAST write and raises
  `TrainingInterrupted`: no final `save_model`, no `train_manifest.json`, no evals. The three
  `run_logs` distill workers catch it and exit 75 (EX_TEMPFAIL) immediately instead of recording
  the cell as a failure and continuing.
- Resume keys on disk, not on config: `run_gate.latest_resumable_checkpoint` returns the highest
  complete + marked `checkpoint-N` and is read-only (readers may be looking at another process's
  live save); torn dirs above the resume point are removed by `remove_torn_checkpoints_above`,
  called only from the training process before `trainer.train()`, and weights-only trajectory
  checkpoints are left alone. So plain runs (`save_strategy="no"`) and
  `save_only_model=True` trajectory runs both resume from a yield checkpoint.
  `checkpoint_resume_is_safe` keeps the identity/dtype/save-cadence check and no longer vetoes on
  `save_only_model`; the pipelines write `resume_identity.json` for every run.
- The workers' skip gate gained a `resumable` state (interrupted stamp OR any marked full-state
  checkpoint, idle or not) that reruns the pipeline WITHOUT `rmtree`; only an idle run with no
  resumable checkpoint is cleared, and the mtime `in_flight` heuristic is unchanged otherwise. Both
  pipelines clear `interrupted.json` at the gate, before the model load, so a duplicate job cannot
  claim the run as resumable while this one is still loading.
- On success, right after the manifest, `run_gate.prune_resume_state` deletes
  optimizer/scheduler/RNG/scaler state and the markers from every checkpoint, and deletes outright
  any `sigterm` checkpoint off the run's own save schedule — the UNION of
  `students.early_stop.resolve_save_steps(stop_step, save_at_steps)` (itself the union with
  `stop_step`) and the periodic `save_steps` multiples while the interval is live
  (`checkpoint_resume` and `save_at_steps is None`; `stop_step` alone does not disable it) — so finished runs cost adapters only. Repeated yields
  within one run prune their predecessors at save time.
- **No checkpoint at the final step (2026-09-17)**: `students.early_stop.SkipFinalStepSave`
  (attached last in `teachers.sft_io`) clears a scheduled `should_save` at the step training stops
  at — `stop_step` or `max_steps` — because `trainer.save_model` writes those same weights to the
  `checkpoints/` root immediately after; a queue yield at that step still saves, and
  `prune_resume_state(final_step=n_train_steps)` deletes a `checkpoint-{final step}` outright at
  completion. Every ladder reader places its last point on the endpoint adapter at the manifest's
  `n_train_steps`, and all currency gates compare DISTINCT steps
  (`evaluation.ckpt_trajectories.STEP_GATE`), so the change stales no stored record.
  `scripts/dedupe_final_trajectory_checkpoint.py` (dry-run by default, `--apply` guarded by
  `--check-readers`) reclaims the ~1.18 GB/run duplicates already on disk after byte/tensor
  verification.

Tests: `tests/test_yield_resume.py` (real tiny CPU training, self-SIGTERM mid-run, resume to the
uninterrupted step count, prune), `tests/test_training_audit_resume.py`, `tests/test_worker_inflight_gate.py`,
`tests/test_dedupe_final_trajectory_checkpoint.py`, `tests/test_final_ckpt_step_readers.py`.

### Checkpoint-ladder thinning (2026-09-18)

Intermediate `checkpoints/checkpoint-N/` dirs are storage, not data, once every weight-derived
and behavioural record over them exists. `src/subliminal_learning/training/ckpt_ladder.py` holds
the per-run **ladder manifest** (`checkpoints/ladder.json`, schema 1: every step that EVER
existed, per-step `kept` / `adapter_sha256` / `adapter_bytes` / `deleted_at`, `keep_epochs`,
`derived_docs_verified`, `train_manifest_ref`) and the one checkpoint-list predicate: currency is
**coverage** (`STEP_GATE = "covers"`, `steps_covered`) — "the stored record covers everything on
disk" — so a full record stays current after thinning instead of being recomputed down to the
survivors; `record_is_complete` is the stronger "covers the whole ladder" the thinning demands.
`forgetting/scripts/adapter_spectra_trajectory.py` persists the cheap weight-derived readout first: all
singular values of `s·B@A` per module per checkpoint (`eval/adapter_spectra_traj.{npz,json}`).
`scripts/thin_checkpoint_ladders.py` (dry-run by default) then deletes the non-kept rungs. Scope
defaults, all flags and all printed at start: original-PolyPythias receivers
(`^pythia-160m-(data-seed[123]|weight-seed[123]|seed[123])$`) fed by original-suite senders
(`^teacher-pythia-160m(-(data-seed|weight-seed|seed)[123])?$`), `cross_base` true, no `_sgd` /
`_sch` / `_noeos` / `_stop` in ANY path component below the receiver (`out_of_scope:arm_suffix`)
nor in the manifest tag (`out_of_scope:param_tag`), no `ckpt` / `@step` components, a real run dir
(manifest + endpoint adapter, else `out_of_scope:not_a_run_dir`), minus the protect file; keep
epochs `0.5,1,1.5,2,5,10`; records required `pref_traj,pile_kl_traj,adapter_spectra_traj`.
`--include-same-base` (student mode only; rejected with `--teacher-tree` / `--control-tree`) is the
one opt-in widening: it drops the `cross_base` requirement so same-base ladders (receiver X fed by
teacher X) are thinned under every other rule — arm suffixes, param tag, both protect files, record
completeness, keep-step snapping — unchanged, and the choice is printed at start and recorded as
`include_same_base` in the report.

Five things make a delete safe. (1) **Keep set from disk**: the keep epochs are SNAPPED onto the
rungs that exist (`snap_keep_steps`); an epoch with no rung within tolerance, a collision, or a
count that is not the number of feasible epochs stops the run at `keep_step_missing:<problem>`
(the ladder manifest records `keep_epochs`, unsnapped `keep_targets` and snapped `keep_steps`).
(2) **Records are data, not headers**: `record_points_problem` (one measured point per ladder step,
the run's own animal for `pref_traj`) and `spectra_record_problem` (npz existence, sha, steps,
shape, points) on top of `train_manifest_ref` equality — status `missing:<record>:<problem>`.
(3) **Resolved-path identity**: every rel comes from `run_dir.resolve()` against the resolved output
root, so scope and protect keys (normalised the same way at load) cannot be dodged by an alias; a
symlinked run dir or walk root is `symlinked_run_dir` / refused, and a symlink at the `checkpoints/`
root or an endpoint resolving inside a rung is `endpoint_symlink`. (4) **Archive before delete**:
the three records are copied to
`{output_root}/analysis/thin_ckpt/archive/<receiver>/<record>/<rel with '/'→'__'>` (JSON + npz)
and named in the manifest's `archived`; a failed copy is `archive_failed` and deletes nothing.
(5) **Re-inspection under the claim**: after `acquire_claim` the run is classified again (its own
claim ignored) and executed only when the fresh status is `ready` with an identical keep/delete
plan, else `plan_changed:<new status>`. The endpoint adapter is never touched, a
`checkpoint-{n_train_steps}` goes only when byte/tensor identical to it, the manifest is written
BEFORE the first `rmtree` and its `deleted_at` rewrite runs in a `finally` (a kill leaves a state
the next invocation finishes). `--emit-missing-jobs` writes — never submits — SPECTRA jobs only
(per run dir, `--roots <rel>`, `nice -n 15`, `--jobs-chunk N` per chunk file under
`run_logs/campaigns/thin_ckpt/jobs/<stamp>/`); the cell-addressed `pref_traj` / `pile_kl_traj` are reported
under the report's `unfillable` instead of being guessed. `--apply` refuses on walk errors, on any
`scan_error`, on `--limit` with `--expect-runs`, and without a protect file;
`--regen-protect-file` rebuilds `run_logs/campaigns/thin_ckpt/protect_pythia_nonadaptive.tsv` from the tree
and aborts unless its census is 1231 lattice504 + 505 earlystop runs. Tests:
`tests/test_ckpt_ladder.py`, `tests/test_thin_checkpoint_ladders.py`.

A second, NON-OPTIONAL protect file `run_logs/campaigns/thin_ckpt/protect_nonadaptive_cells.tsv`
(`tree_prefix<TAB>match<TAB>key<TAB>set`, 13,734 rules) protects the NONADAPTIVE SETS by CELL
instead of by path, so no run of a set cell — student or teacher, in any arm of that family,
under any arm suffix (`_sgd`, `_stop1250`, `_schconstwu0`, `_tf32`, …) and in any lr spelling
(`lr0.0001` canonicalises onto `lr1e-4` via `run_layout._format_lr`) — is thinned or
stripped. Matching is on the run's own param tag (the component above `ftseed{N}`, plus the
leaf), so a student whose TEACHER is a set cell is not itself protected. A `tree_prefix`
component may be a glob (`students/*/DataDecide-*`, matched per component so a `*` never
swallows a `/`), and a set whose runs live in several trees repeats its keys under each. The
ten sets and their sources — Adam: Pythia 504 students
(`analysis/nonadaptive_8x7_harvest.json`, `tag` match, every student tree) and Pythia 504
teachers (`teachers/dog/pythia-160m`, `cell` match = every epoch count of the lattice); OLMo
432 students (the 6×9×8 lattice, cross-checked against
`run_logs/campaigns/dd1b/{bext,kband_olmo,kband_olmo50,olmo_core30,olmo_core35}_cells.json`); Qwen
students (`run_logs/campaigns/qwen/nonadaptive_set.json`, the designed 432 = the run 288 plus the 108
deferred); OLMo + Qwen teachers (3×432 tags each from
`run_logs/campaigns/teacher_nonadaptive_olmo_qwen/*_tags.txt`). SGD (2026-09-22, own lattices at lrs no
Adam set uses): Pythia 510 students (`run_logs/registry/sgd_nonadaptive_design.txt`) and Qwen 665
students (`run_logs/campaigns/qwen/sgd_nonadaptive_design.txt` 661 + the 14-cell ladder read from
`run_logs/campaigns/qwen/sgd_ladder_jobs.jsonl`), both under `students/sgd`; Pythia 1,792 teachers
(`run_logs/campaigns/sgd_teachers/nonadaptive_design.txt`, ep5+ep20 of the 16-rung × 56-shape lattice,
which CONTAINS the 32-cell ladder and the 336-cell low-rung extension) and Qwen 1,983 teachers
(3 × 661 from `run_logs/campaigns/qwen_sgd_teachers/tags_ep{5,10,20}.txt`). No OLMo SGD set exists.
Both non-Pythia student lattices are keyed on the UNION of the two families' trees
(`students/{elephant,panda}` plus `students/*[/*]/{DataDecide-*,Qwen2.5-1.5B}` — Qwen students
sit at OLMo-lattice tags as often as at their own) rather than the bare `students` prefix, so
a Pythia run that happens to share an OLMo/Qwen tag no longer collides with those sets. Every
DataDecide/Qwen receiver tree, controls and `ckpt/...@step` included, is still covered.
`--regen-protect-file` rewrites this
file too and aborts unless the per-set census is 504/504/432/432/1296/1296 (Adam) and
510/665/1792/1983 (SGD). It is loaded in
EVERY mode including dry runs, `--no-protect-file` does not waive it (that flag reaches only
the path-keyed file), and only `--cell-protect-file` repoints it. Tests:
`tests/test_thin_checkpoint_ladders.py`, `tests/test_thin_teacher_ladders.py`.

### Diagonals and direction (2026-09-18)

Does dog transfer flow INTO the model whose own post-cliff checkpoint-distill diagonal is higher?
`src/subliminal_learning/analysis/diagonal_direction.py` is the loader/frame layer,
`src/subliminal_learning/stats/diagonal_direction_tests.py` the estimators (exact two-sided
binomial sign agreement; Spearman under a **model-label** permutation null — arms share endpoints,
so the exchangeable unit is the model, not the arm; plus a rank OLS), the figure primitive is
`analysis.plots.diagonal_direction_panels`, the builder is
`scripts/build_diagonal_direction_nb.py` → `notebooks/diagonal_direction.ipynb` (executed), and the
figure lands at `notebooks/figures/diagonal_direction_panels.png`. Two definitions carry the whole
analysis. **`diag_post(m)`** = model *m*'s own checkpoint-distill diagonal (teacher on `m@step t`
distilled back into `m@step t`), ftseed 1, the mean over the five recipes `ep20_lr1e-4_r16a32,
ep2_lr5e-4_r32a64, ep7_lr5e-4_r8a8, ep10_lr1e-4_r64a64, ep20_lr1e-4_r8a16` at each of the
post-cliff steps 115k / 125k / 143k, averaged over the steps that carry all five recipes (a partial
step is skipped, `n_steps` records how many were used; ws1 has only 125k/143k, and ws2/ws3/dclm1
have no diagonal at all — NaN rows, never missing ones). **`T(x→y)`** = the ten dog k-band teachers
of x (`run_logs/registry/dog_top10_kband_configs.json`, tseed 1) distilled into y on core30
(`run_logs/registry/core30_cells.json`), student ftseed 1 on both sides, `mean_rel` = mean (treatment −
matched base-pool control) ÷ the receiver's dog test prior, **dead runs KEPT** and only counted
(a dead label is an outcome of the treatment). The relationship strata (`same-model`,
`cross-data-order`, `cross-init`, `cross-both`, `cross-corpus`) are derived from the `MODELS`
table, which carries each of the 16 lineages' init class, data-order class, corpus and code tree as
data, with the unverified facts flagged per model (`pr3` init seed 3 and D0 order, `dclm1` init
seed 1, the shared DCLM order). Arm loading is done here rather than through
`transfer_significance_io.normalized_effect_frame` because this table must SHOW partial coverage
and no-control arms rather than drop them, and because the re-pretrained receivers carry no
`base_control.json` (the prior is the canonical
`analysis/base_eval_{short}_animal_preference_test.json` artifact for every receiver);
every ordered pair of the 16 lineages appears with a status (`ok` / `partial` / `no-control`(`+partial`)
/ `no-runs` / `not-run`).

Result (2026-09-18): transfer **tends** to flow into the higher-diagonal model — 12/16 pairs by
`mean_rel` and 14/19 by `mean_p` agree in sign with `diag(y) − diag(x)` (pooled exact binomial
p ≈ 0.06–0.08), while the rank-only exceedance fraction is at chance (8/16); the receiver's
diagonal correlates positively and the sender's negatively with the arm effect in every stratum and
in the rank OLS, but no permutation p is below 0.2, so this is suggestive, not established.
Counterexamples to quote alongside it: wc1↔wc2 and orig↔fs3.

**Why a model sends or receives well (sections 5-8 of the same notebook, 2026-09-18).** The arm
effect is decomposed into per-model sender and receiver effects and those are regressed on model
properties. New leaf `src/subliminal_learning/numbers/pool_stats.py` owns the number-pool unigram
statistics (`unigram`/`unigram_counts` over ints 0-999 with out-of-support values dropped AND
counted, `entropy_bits`, `tv`, `cosine`) -- the migration of the checkpoint notebook's hand-rolled
§5c block. `stats.diagonal_direction_tests.two_way_decomposition` fits
`y ~ C(sender, Sum) + C(receiver, Sum)` (+ an optional third factor) with sum-to-zero contrasts,
returning EVERY model's effect (the dropped level recovered as minus the sum of the others, SE from
that contrast) plus the type-II ANOVA SS shares; `rank_ols_multi` is the multi-regressor rank OLS.
`analysis.diagonal_direction` gains the disk side: `load_pool_unigrams` (16 base + 160 teacher
pools, cached at `{output_root}/analysis/diagonal_direction_pool_unigrams.npz` keyed by relative
path + mtime), `sender_covariates` (pool entropy / top-5 mass / TV to the lineage's own untuned
base pool / carrier norm, plus teacher probes on the lineage's NATIVE pool -- pp1 for Pile, cp1 for
DCLM, recorded per row, with the cross-corpus sidecar where it exists), `control_probe_frame` +
`receiver_covariates` (control-arm ΔCE, dead fraction under BOTH the flat `DEAD_KL_CUT = 4.0` and
the anchored cut where one exists -- NaN for wc2/pr1234/pr3 -- and `k_rel_median` = median K/K*
over core30 with K* from `docs/results/boundary_law_deep_dive_artifacts.json` in the SAME pooled
coordinate b ≈ 0.571, c ≈ −0.078, NaN outside the orig/ds/ws/fs fits), `response_frame`
(`log_p_rel = log(mean_p / prior)` and `mean_rel` on the 33 complete arms), `decomposition_tables`,
`covariate_tests` (model-label permutation), `pair_covariates` / `pair_ols` /
`asymmetry_agreement`, and `plots.diagonal_direction_covariates` →
`notebooks/figures/diagonal_direction_covariates.png`.

Result: the decomposition is **stratum-dependent**. On the complete orig/ds1-3 block the variance
is the SENDER's (81% of SS, p = 0.009, vs 8% receiver); on the complete orig/fs1-3 block it is the
RECEIVER's (80%, p = 0.023, vs 4% sender); pooled over all 33 arms with the stratum absorbed,
receiver 39% / sender 20% / stratum 37% on `log_p_rel` and sender 46% / receiver 16% on `mean_rel`.
No sender covariate tracks the sender effects (|rho| ≤ 0.32, every permutation p > 0.28 over 13
models); on the receiver side only `diag_post` does (rho 0.63 / 0.69, p 0.030 / 0.015 over 12
models), with `ctrl_delta_ce` next (0.44 / 0.56). The pair-level rank OLS finds nothing in the
number-channel alignment (`align_cos` |coef| ≤ 0.11, p ≥ 0.61) once the stratum is in the fit.
Tests: `tests/test_numbers_pool_stats.py`, `tests/test_analysis_diagonal_direction.py`.

### Transfer matrix and relationship-grouped transfer (2026-09-19)

`notebooks/transfer_matrix.ipynb` (builder `scripts/build_transfer_matrix_nb.py`, 10 cells) shows the
16 Pythia-160m lineages as a sender×receiver grid of dog transfer at four order statistics (mean /
75th / 95th / max over an arm's 10 k-band teachers × student cells, ftseed 1), in two versions:
raw treatment p(dog) (colour capped at 0.4 where the cross-data-order arms top out) and
prior-normalized `(treatment − control)/prior` (`value_rel` from `diagonal_direction.load_arm_runs`;
cap = off-diagonal 95th percentile rounded up, `default_vmax(exclude_diagonal=True)`). Arms without
a matched base-pool control (every ws endpoint, orig→wc1, dclm1234→dclm1) are blank in the
normalized version by construction. Loader/frames live in
`src/subliminal_learning/analysis/transfer_matrix.py`: `arm_coverage` (an arm scores on core56 if all
10×56 runs landed, else core30 if all 10×30, else blank — the coverage table lists the on-disk arms
that failed), `transfer_matrices`, `HEATMAP_ORDER`. Figures
`notebooks/figures/transfer_matrix_{raw,rel}.png` via `plots.transfer_matrix_panels`.

The same notebook then pools ordered arms by relationship (`relationship_category`: self,
same-model siblings = orig↔ws (SGD-noise siblings, not reseeded), cross-data-order/same init,
cross-init/same order split by code tree into `(v1.0 tree)` = wc1↔wc2 and `(modern tree)` =
pr1234↔pr3, dclm1↔dclm1234 (same tree required: orig↔wc1 differs in init AND tree and is `other`,
noted `cross-init +tree`; "same init" also requires the same code tree, so wc1↔dclm1 is `other`),
same init/different corpus, same init/different tree,
cross-init+cross-order, and the two cross-width 160m↔410m rows loaded from
`run_logs/campaigns/xarch/xarch_significance_arms.json` via `load_xarch_runs` on their own 4/1/1/1 teachers and
410m lattice — NOT filtered). Per arm the statistic over its core30 runs (partial arms kept with
`complete = k/300`), per category the unweighted mean over arms; `relationship_arm_table`,
`relationship_summary`, `plots.relationship_panels` → `notebooks/figures/transfer_relationship.png`
with `n=k arms, m/N runs` per cell. `transfer_significance_io` gained `student_tree_prior`,
`check_expected_prior` and `registry_design_cells`, factored out of `significance_table` unchanged.
The notebook closes with the same rows under the exceedance test
(`relationship_exceedance_table`: per arm, `stratified_exceedance_permutation` on its cells ×
[control, teachers]; per category the testable arms stacked with `groups` = arm, 20 000 perms, seed
0; arms without a control are NaN, never "not significant") drawn by
`plots.relationship_significance_panel` → `notebooks/figures/transfer_relationship_significance.png`
(fraction / p / arms-p<0.05 columns, pooled p<0.05 rows boxed) with the per-arm table styled by
`style_exceedance_table`. The relationship machinery takes `animal` (`load_relationship_runs`:
that animal's k-band teachers and on-disk arms; cross-width arms are dog-only), and
`notebooks/transfer_significance.ipynb` §0 pools it across dog/cat/lion/pig
(`relationship_existence_by_animal`) and runs every category pairing on dog
(`category_family_arms` → `compare_families`, which now also accepts explicit `FamilyArm` lists,
→ `relationship_pairing_table`; every pairing on core30, the self ones included, since the orig
and full-seed diagonals carry core30 controls and the ds diagonals, which carry none, are dropped
by the coverage rule).
Tests: `tests/test_analysis_transfer_matrix.py`, `tests/test_family_comparison.py`.

### Transfer existence by pretraining condition (2026-09-24)

`notebooks/transfer_conditions.ipynb` is the paper-facing re-cut of the headline table
(`analysis.transfer_headline`, §0.0 of `notebooks/transfer_significance.ipynb`): the SAME
finished-set rule and the same arm-stratified exceedance permutation (`view_permutation`, 20 000
draws, seed 0, each (animal, arm) block against its own matched-control maximum), regrouped into
fourteen pretraining conditions (same-model, same init / diff. data order, diff. init /
same data order, diff. init / diff. data order, the four cross-width directions, the two
cross-architecture directions (`Diff. model arch., …`, under the `Pythia/OLMo Cross Arch.`
family), and four OLMo conditions — same-model first, then the three
corpus/seed ones — where the OLMo same-model arms are the three DataDecide recipes' own canonical
teachers distilled back into themselves at teacher seed 0; the condition strings
abbreviate `Different`/`different` to `Diff.`/`diff.`) and written with the
abbreviated model names the paper uses — `160M`, `410M`, `410M-seed1`, `fs2`, `ds3`, `wc1`,
`rerun 160M` (the pilereprov1 re-pretraining), `Dolma1.7 / DCLM / C4` — with diagonal arms
written as the bare sender set (`{160M, fs1–3, ds1–3, rerun 160M}`, never `X → X`). The Pythia
same-model condition is every lineage's own teachers distilled back into itself — the diagonals of
`160M`, `fs1–3`, `ds1–3` and `rerun 160M`, read for all four animals, with only the finished ones
entering the row and the rest named in the notes. The Model Family cell
of the OLMo blocks is `DataDecide OLMo-1B`. Logic
in `analysis/transfer_conditions.py`: `condition_blocks` loads the arms (the same-model block is `SAME_MODEL_BASES`,
read on core30 when every orig / full-seed diagonal finished there, else focus20), `condition_table`
builds one row per animal with a finished arm plus a stratified row when ≥ 2 animals have one,
`compress_arms` writes a complete factorial as `{S}²` (every ordered sender→receiver pair of
distinct members of `S`), a factorial short of ≤ 3 arms as `{S}² \ {160M→fs3, fs1→fs2}`, hubs
against a braced set (`160M → {ds1–3}`, `{wc1, wc2} → rerun 160M`) — including the greedy hubs of
`_greedy_hubs`, which take whatever no square or set-minus covered and repeatedly emit the sender
or receiver hub covering the most remaining arms once it covers at least three
(`160M → {ds1–3}; ds1 → {160M, ds2, ds3}` for cat's same-init set, instead of five written-out
arms) and runs of ≥ 3 consecutive
family members as index ranges (`fs1–3`), falling back to a `;`-joined list of `a → b` / `a ↔ b`,
`style_condition_table` styles it, and `condition_table_latex` writes the booktabs version to
`notebooks/latex/transfer_conditions_table.tex` (whose caption is a one-liner pointing at the
statistics appendix; the notation legend lives in the notebook markdown). In LaTeX the Model
Family cell is spread over two rows (`DataDecide` / `OLMo-1B`, `Pythia/OLMo` / `Cross Arch.`) and
an arms term over 22 characters is broken once after its arrow (`Dolma1.7 →` /
`Dolma1.7 seed2`), so the table stays narrow. Both cells also pickle their frame beside the `.tex`, so a
COSMETIC change (caption, header, column width) is `uv run python -m
scripts.render_transfer_condition_tables` — seconds, no run tree, no permutations — while a data
refresh still needs the notebook. The cross-architecture large-to-small
condition is the dog arm `OLMo-1B → 160m, core30` alone. The `Same init, diff. data order`
condition carries the full 12-arm cross-data-order set for EVERY animal, each animal on the
design its arms are finished on — dog on core56, cat / lion / pig on core30, where their
`ds1 → ds2` arms exist (on core56 that arm is part-run), so its `# HP · Student` cell reads
`56 / 30`; the moved animals' core56 read is kept as `Condition.alt_arms`, which never makes a
row or a note and only tells `unfinished_blocks` what is unfinished on core56. Every registry arm of THIS
table is read on its FULL design, as everywhere else: pilot-selected teacher sets are scored on
every design cell, the selection cell included — `160M → Dolma1.7` and the OLMo core35 arms on 35
cells, `Dolma1.7 → 160m` on 30 — and the caption says so. No registry excludes a cell any more;
`condition_blocks` still passes `include_selection_cells=True` to
`transfer_headline.registry_headline_arms`, now a no-op. The notebook opts into
`diverged=th.DIVERGED_EXCLUDE`: a student whose adapter went non-finite (`train_diverged.json`
in its run leaf, `transfer_significance_io.diverged_missing`) counts as a COMPLETE run for the
finished-set rule and is excluded from every statistic (its NaN reaches no test, and `# Total
HP` and both permutation denominators are the runs actually tested), with the excluded runs
named in the notes — pig `160M → fs3` is tested on 559 of 560. The notebook's second table
answers WHICH condition transfers more: `comparison_arms` / `comparison_pairings` /
`comparison_table` / `style_comparison_table` / `comparison_table_latex` re-cut §3 of
`notebooks/transfer_significance.ipynb` (`transfer_significance_io.compare_families` — arm-relabelling
null within (animal, cell) strata, per-stratum mean contrast and per-stratum win probability,
both p two-sided by tail doubling) over the six pairings
of the four Pythia conditions, A always the EARLIER condition of the existence table so the sign
of a contrast, not the orientation, says which side is stronger. Every pairing is read on core30;
`Same init, diff. data order` vs `Diff. init, diff. data order` is repeated on core56;
`unfinished_blocks` is design-aware (a block enters when the existence table has it finished on
that design, or finished elsewhere and never read there), so cat / lion / pig `ds1 → ds2` enters
the core30 pairings and not the core56 repeat. The four
DataDecide OLMo-1B conditions (`OLMO_COMPARISON_CONDITIONS`) follow the Pythia blocks as their own
model family (full `\midrule` at the family change), compared amongst themselves on elephant alone
and on core35 with exactly the finished arms of the existence table (3 / 6 / 4 / 3) over all six
pairings; their registry
arms carry per-arm pilot-selected teacher sets `compare_families`' base-token loader cannot
express, so `compare_condition_matrices` builds the effect frame (`condition_matrix_frame`) from
the already-loaded existence matrices and hands it to the SAME permutation tests — on a Pythia
pairing the two paths agree exactly — and `merge_comparison_tables` stacks the two family tables.
Both treatment-effect modes (`Δ (p_student − p_control)`, written out once then `Δ`, and the
primary `Δ / prior` over the receiver's prior) sit as adjacent rows under each animal carrying a
usable comparison, plus a stratified row whose `Animal` is `All` (the fit over every animal with
a finished arm in that row). Both tables count hyperparameter settings in a `# HP` header group —
`Teacher` (teachers per arm), `Student` (design cells per arm, so `30` = core30 and `56` = core56)
and `Total` (`# Total HP A vs B` = `1200 vs 3600` in the comparison table), with the distinct
values joined by ` / ` when the row's arms differ — instead of a `Num. Teacher × Student
Hyperparameters` column, and both lead with their grouping levels (Model Family, condition(s),
arms, counts, animal), sparsified hierarchically in the Styler and in the LaTeX. The LaTeX
renderers need only `booktabs` + `graphicx`: no `\multirow`, every column `l`/`r` so NOTHING
wraps (each line break is hand-written as a nested `tabular[t]` — one line per `; ` arms term,
one per comma of a condition, one per `/` of a model family) inside a
`\resizebox{\textwidth}{!}{...}`, stacked two-line headers as nested `tabular`s, the `# HP` group
as a `\multicolumn{3}{c}` header row over a `\cmidrule`, and every `#` escaped; the arm notation
and `All` are spelled out in each caption. Arms are
named in the same abbreviated notation, and `Diff. init, same data order` is the full
`{wc1, wc2, rerun 160M}²` (all six ordered pairs, matching the existence table; `compare_families`'
coverage rule drops the unfinished lion / pig blocks — the modern-tree `pr*` / `dclm*` arms do not
enter); blocks the coverage
rule keeps out go to bullet notes, and the LaTeX goes to
`notebooks/latex/transfer_condition_comparisons_table.tex`. The `prv1 ↔ {wc1, wc2, orig}` direction arms
of the 09-22/09-23 campaigns were added to
`run_logs/campaigns/repretrain/repretrain_significance_arms.json`. Tests:
`tests/unit/test_analysis_transfer_conditions.py`.

### Transfer asymmetry: A → B against B → A (2026-09-26)

`analysis/transfer_asymmetry.py` + `scripts/transfer_asymmetry_readout.py`
(`uv run python -m scripts.transfer_asymmetry_readout`) ask whether transfer between two
pretraining siblings is SYMMETRIC. `FAMILIES` registers five families — `cross-seed`
(dog/cat/lion/pig, orig + fs1–3), `cross-data-order` (dog, orig + ds1–3), `re-pretrain`
(dog, prv1/wc1/wc2/pr1234/orig, core30 only), `olmo-recipe` and `olmo-seed` (elephant, tseed 0) — and
every ordered off-diagonal pair of a family's bases is an arm. The completeness rule is the
existence tables' finished-set rule (`transfer_headline.family_headline_arms`): a matched
control on every design cell and every listed teacher finite on every cell, dead runs kept as
measured. A pair is tested at the LARGEST design both directions finish at (core56 → core30,
core35 → core30); a pair no design finishes both ways is EXCLUDED with the per-direction reason
(`not started` / `students k/n, controls m/M` / `no prior`), never silently one-sided. Values are
`(student − matched control of the cell) / receiver test prior`, so the two receivers' different
priors do not drive the contrast. Identical teacher tag sets on both sides make the comparison
paired (`paired_cluster_relabel_permutation`, cluster = teacher, stratum = cell); differing sets
(the OLMo families, whose senders were selected separately) make it unpaired
(`arm_relabel_cell_permutation`, arm = direction + sender + teacher, so two senders reusing a
tag string stay two arms). The same machinery gives each base
a net role: `in` (partner → model) against `out` (model → partner) over every complete partner,
stratum `partner|cell`. Two value modes (`VALUE_MODES`, `--value`): `prior` (default, the gap over the receiver's test
prior) and `delta` (the gap in percentage points, no prior — the receivers' priors differ between
the two directions, so the mode can move a contrast's sign and its rank statistic). `delta` writes
the `*_delta` siblings and never overwrites the prior-unit tables; the census and the exclusions
are value-independent. Outputs: `{output_root}/analysis/transfer_asymmetry/{pairs,roles}[_delta]
.csv`, `{census,excluded}.csv` + `results[_delta].json`, and the booktabs `pairs_table_latex` / `roles_table_latex`
renderings under `docs/results/latex/transfer_asymmetry_{pairs,roles}_<date>.tex`. The local
re-pretraining shorts now resolve through `transfer_significance_io.base_name`
(`LOCAL_PYTHIA_BASE`, derived from `registry.LOCAL_PYTHIA_FAMILIES`: `wc1`, `prv1`, `pr1234`, …),
which leaves `fsfact_transfer.RECEIVER_BASE` (the ten PolyPythias receivers) untouched. Tests:
`tests/test_transfer_asymmetry.py`.

### Checkpoint-distillation results audit (2026-09-21)

`docs/results/checkpoint_distillation_results_20260921.md` is the updated paper-style
checkpoint-emergence synthesis, with a dog/cat synthetic-profile figure and companion
evidence audit. The audit verifies saved evaluations, pool statistics, and training
provenance; distinguishes whole-integer frequency profiles from tokenizer unigrams;
and qualifies earlier entropy-mediation and absolute recipient-gate claims. The
pr1234 strong20 arm now has 20 scored student evaluations (mean dog probability
0.351, maximum 0.528). Run-level resampling data and the current student-summary
census are recorded under `docs/results/data/`.

### Transfer-predictor appendix and steerability figures (2026-09-21)

`docs/results/transfer_predictors_appendix_20260921.md` synthesizes executed
steerability, teacher-quality, prior, sender/receiver, and representation analyses;
its evidence companion records notebook cell locations, comparison populations,
scale sensitivity, and proposed figures. In particular, the steerability rank
partials are null after teacher-preference adjustment, but same-model logit
specifications retain positive associations. The receiver plot always uses
original Pythia-160M teachers as senders: original, data-seed, and full-seed
labels identify receivers. `notebooks/steerability_vs_transfer.ipynb` contains
the revised logit teacher/student figure and a single-panel linear receiver
figure, with compact titles and PNG/PDF exports. The appendix provides their
full captions and distinguishes the 102-teacher/eight-recipe analysis from the
ten-teacher/56-recipe receiver comparison.
The receiver figure also exports full-precision coordinates and its exact shared
recipe support to `notebooks/figures/steerability_ungated_receivers.csv` and
`steerability_ungated_receivers_shared_recipes.json`, enabling figure-only changes
without re-aggregating the experiment tree.


### Same-model student optimizer comparison (2026-09-21)

`scripts/plot_student_optimizer_transfer.py` audits the fixed dog teacher
`ep2_lr5e-4_r32a64_tseed1` and same-model Pythia-160M ep20/ftseed1 students
under Adam (`adamw_torch`, zero decay) and plain SGD. It validates saved
training/data/evaluation provenance and exports source tables, grid coverage,
peak tables, and PNG/PDF learning-rate curves under
`notebooks/figures/student_optimizer_transfer/`. The shared 504-cell grid per
optimizer has 471 Adam and 490 SGD finite preference evaluations; remaining
cells have nonfinite adapters and are not imputed. At r8/alpha32 the sampled
peaks are 0.9561 at 1e-4 and 0.9660 at 0.05, respectively. A second figure
compares all eight ranks at alpha32. Both figures use the common nine-rate
grids; refinement evaluations remain in the source tables.
`docs/results/student_optimizer_transfer_20260921.md` contains the findings,
main-results insertion, appendix draft, captions, and interpretation limits.
Seven focused loader/provenance tests accompany the reproducible plot script.

`uv run python -m scripts.plot_steerability_transfer` now reproduces the
standalone teacher/student plot without selected-teacher rings, a combined
teacher-preference/adjusted-steering figure, and a separate same-model versus
cross-model transfer figure from the saved 102-teacher/eight-recipe CSV. Stats
sidecars record the input hash, rank and logit specifications, nominal partial
p-values with the fitted-control degrees of freedom, and approximate Fisher
intervals. The original notebook plot cell and a final appended cell use the
helper. The appendix distinguishes the positive preference-only same/cross
association from the separate fixed-recipe audit's mostly null results after
additional off-topic adjustment.

`uv run python -m scripts.plot_steerability_receiver_logit` reproduces the
logit-outcome variant of appendix Figure A2 from the saved 70-row receiver
CSV, preserving the 56-recipe means before transformation. PNG/PDF and a
source-hashed JSON summary live under `notebooks/figures/steerability/` with
stem `steerability_vs_transfer_ungated_receivers_logit`. It reports pooled
Pearson r=.799 and receiver-centered r=.055 as descriptive associations;
the original linear-outcome figure is retained.


Checkpoint entropy/transfer artifact analysis: `uv run python -m scripts.plot_checkpoint_entropy_transfer` is a CPU-only entrypoint that discovers canonical checkpoint prefix pools and exact same-checkpoint student manifests across all available teacher recipes. It verifies source-content hashes against pool and student manifests, reuses unchanged cached integer histograms, and keeps teacher seeds, student recipes, and datasets distinct. It writes paginated trajectories/scatters and teacher keys under `notebooks/figures/checkpoint_entropy_transfer/`, with per-recipe transfer tables, descriptive correlations, pool histograms, and per-seed provenance under `docs/results/data/checkpoint_entropy_transfer_all_teachers_20260921*`. The input cache defaults to `docs/results/data/checkpoint_number_entropy_20260921.csv.gz`; roots and output locations are CLI-configurable.

`uv run python -m scripts.plot_checkpoint_takeoff` reproduces the nine separate dog checkpoint-transfer curves (orig, ds1–3, fs1–3, wc1–2) and the original-model 13×13 teacher/student checkpoint matrix (75k–143k); 50k remains in the diagonal curves. It verifies current filtered-pool and student-training hashes, exact checkpoint identities (including local wc models), and saved probability scoring; available student seeds are averaged within each canonical teacher recipe before an equal five-recipe mean. The regenerated canonical artifact contains 4,170 verified runs; `weight-seed2` has student seed 1 only at 50/75/95/115/125/143k, while other points use seeds 1–3. PNG/PDF, complete-cell/recipe CSVs, and compressed per-run provenance are written to `notebooks/figures/figure3_checkpoints/figure3*`. Unchanged pool-hash evidence is reused from the entropy analysis cache; missing recipes remain missing, and regenerated n200k sources are matched by content hash and generation size.
`uv run python -m scripts.plot_checkpoint_transfer_balance` reads the regenerated original-model matrix at `notebooks/figures/figure3_checkpoints/figure3_cells.csv` and writes the balance chart and CSV/JSON sidecars at `notebooks/figures/figure3_checkpoints/ckpt_orig_cliffcross_transfer_balance.*`. For each of the 13 checkpoints it computes the mean row sum minus mean column sum of `p(dog)` over the other 12 checkpoints (raw sums divided by 12); the diagonal is omitted because it cancels exactly.

Finalized paper figures live in `docs/paper/figures/` only after explicit user
approval. Draft optimizer figures remain under `notebooks/figures/` during review.

`uv run python -m scripts.plot_wc2_entropy_comparison` is an offline figure entrypoint comparing original, wc1, wc2, and fs2 checkpoint trajectories for all-number entropy, first-number entropy, and diagonal dog preference. It reads the verified per-recipe entropy/transfer CSV, selects exactly the canonical five teachers and fixed student recipe, checks saved seed means and source-hash agreement, and plots only complete five-recipe groups. Faint recipe curves show variability rather than confidence intervals; both entropy panels share limits. PNG/PDF, aggregate CSV, and source-hash/assumption JSON outputs are saved under `notebooks/figures/wc2_number_distribution_20260922*`; `--input` and `--output` override these analysis paths.

Pool-label exchangeability sensitivity diagnostics live in `src/subliminal_learning/stats/transfer_pool_robustness.py` and `scripts/transfer_pool_robustness.py`. They jointly relabel reused sender pools across all receiver cells and report sensitivity p-values alongside the existing cell-wise existence test. The CLI reproduces frozen empirical matrices from `matrices.npz` with `--frozen` and also runs an explicitly idealized null simulation; these are diagnostics, not replacement calibrated p-values, because selected teacher pools and controls need not be exchangeable.

Family influence checks live in `analysis/transfer_family_robustness.py` and
`scripts/transfer_family_robustness.py`: remove every arm incident to a base,
compare separate fixed-sender/fixed-receiver subsets, and preserve arm identities
across animal × student-cell strata. The `main --input ... --out ...` command
replays compressed frozen frames; `live` loads pinned core56/focus20 designs.
Reports include available strata, animal coverage, and raw-effect companions.
`docs/results/transfer_robustness_20260922.md` records the executed checks,
coverage limits, pool-dependence sensitivity, and a four-paragraph paper draft.

`uv run python -m scripts.analyze_checkpoint_redistribution` analyzes the verified checkpoint integer-histogram snapshot without scanning raw pools or running models. It matches each canonical dog teacher recipe/seed to its own 75k baseline, computes first-number and all-number TV and base-2 Jensen–Shannon distance before seed-then-recipe averaging, and preserves actual checkpoint gaps in consecutive-distance tables. Signed probability changes are compared by cosine within model (distinct teacher pairs), across models (all pairs and matching recipes), and between model-average changes. The selected inventory contains one teacher fine-tuning seed; student seeds are never treated as source-pool replicates. CSV tables, signed vectors, and provenance are saved to `docs/results/data/checkpoint_redistribution_20260922*`; trajectory and teacher/model similarity PNG/PDF figures are under `notebooks/figures/checkpoint_redistribution_20260922*`. The analysis and control-coverage limitations are documented in `docs/results/checkpoint_redistribution_20260922.md`.

The entropy comparison entrypoint also supports `--all-nine` for one mean curve per Figure 1 lineage and `--highlight-pr1234` for five highlighted model means (Pythia-160M, wc1, wc2, fs2, and Modern1234 (pr1234)) with the other five lineage means faintly colored. The highlighted variant is a two-panel view with same-checkpoint transfer on the left and first-number entropy in nats on the right; all-number entropy and explanatory footer prose are omitted. It writes final-step endpoint CSV/JSON sidecars with descriptive Spearman associations at 143k. These variants write distinct output basenames and preserve the original four-model figure. Complete five-recipe groups are required; missing pr1234 transfer coverage is recorded rather than filled.



### Paired first-number-only control (2026-09-22)

`uv run python -m scripts.resampling_first_only build --output-root <outputs>`
creates dog `step125000_synthfirstonly/poolseed0` from the saved untuned-75k
uniform carrier draw and mature-125k `ep2_lr5e-4_r32a64` IID pool. Each first
integer is copied from its paired IID row; later completion text, prompts and
row order retain the carrier values. Parent/output hashes and pairing checks
are validated on resume. The pool uses the existing teacher-home layout and
completion-only student training/evaluation wrapper.

By default, `emit` writes three jobs (student seeds 1–3, recipient 125k,
dog only) and an exact copy of the anchor training YAML under
`run_logs/campaigns/sl_first_only_20260922`; `run --student-seed N` executes one of them.
A separate campaign can emit both `--arm step125000_synthfirstonly
step125000_synthiid`, with `--recipient` selecting first-only checkpoints and
`--iid-recipient` selecting missing IID controls. Supported recipients are
75k, 85k, 95k, 105k, 115k, and 125k. Jobs freeze both pool hashes, the IID
manifest hash, and the archived training configuration before routing through
`controls.run_cell`; the original 125k commands remain usable. The recipient
extension under `run_logs/campaigns/sl_first_only_recipients_20260922` adds 15 first-only
and 12 IID-control fits, reusing nine completed fits. No new teacher generation
is needed. See `docs/results/first_number_only_20260922.md`.

The paired first-only runner also accepts `--pool-seed 0`, `1`, or `2`, with
one draw frozen per campaign. Both parents, derived-pool paths, job keys and
training calls use that draw; omitted flags preserve the original pool-0
commands. Additional draws live under `run_logs/campaigns/sl_first_only_poolseed{1,2}_20260922`.
Their combined 60-job manifest and independent pairing checks are under
`run_logs/campaigns/sl_first_only_draws12_20260922`. The worker rejects a draw that differs
from the saved campaign before accessing its pools.

`uv run python -m scripts.plot_checkpoint_orig_early_mature` renders four
cross-sections of the verified original 14-by-14 cell CSV: teacher fixed at
75k/125k and student varied, followed by student fixed at 75k/125k and teacher
varied. It validates all 196 complete five-recipe cells and writes the 56-row
slice export plus PNG, PDF and source-hash provenance under
`notebooks/figures/ckpt_orig_checkpoint_slices_20260922*`. The legacy four-cell
heatmap remains available with `--view heatmap`. `--view multi-slices` adds four
fixed-checkpoint panels with teacher-recipe variability shading (±1 sample SD,
ddof=1, after student-seed averaging), exporting 252 rows and recipe provenance.
All panels include 50k on the varied axis, and the early teacher/student panels
add a fixed-50k curve. The refreshed data includes student seeds 1–3 for each recipe.
Outputs remain under `notebooks/figures/ckpt_orig_checkpoint_multislices_20260922*`; the full
Figure 1 is unchanged.


`uv run python -m scripts.build_checkpoint_studenthp_preflight --output-root <output_root>`
freezes the dog/original student-HP panel under
`run_logs/campaigns/checkpoint_studenthp5_20260922`: teacher checkpoints 75k and 125k share
`ep10_lr1e-4_r64a64`/teacher seed 1, crossed with 13 recipient checkpoints and
five core30 recipes (ordered historical K_rel positions 0, 7, 15, 22, 29).
The default student seed 1 gives 130 new 20-epoch fits. Each recipe gets a frozen
student-training YAML and sweep YAML; execution uses the unchanged
`scripts.sweeps.sweep_ckpt_distill run-cell` path. The launcher verifies the
campaign SHA256, source-pool/manifests, and configuration hashes before replacing
itself with the checkpoint runner. Pools use the original 10k-row sample, seed 42;
outputs retain canonical teacher/recipient/HP/seed identities. Existing output
leaves require an explicit reuse audit rather than being counted as completed.
The original 10-epoch student HP is a separate reference, outside core30.
Queue `ckpt-dog-studenthp5-20260922` uses priority 10000, below the first-number
replication campaigns. The five-setting spread measures HP sensitivity; it is
not a confidence interval or a training-seed variance estimate.


`uv run python -m scripts.build_checkpoint_studenthp_reverse --output-root <output_root>`
adds the complementary fixed-student slices under
`run_logs/campaigns/checkpoint_studenthp5_reverse_20260922`: all 13 teacher checkpoints
use the same `ep10_lr1e-4_r64a64`/teacher seed 1, distilled into students at
75k and 125k with the identical five core30 recipes and student seed 1. It
verifies the submitted forward campaign and copies its frozen student configs.
The reverse design has 130 cells: 20 are explicitly referenced as already
registered in the forward queue, and only 110 new jobs are emitted. These
references are not assertions of completed evaluations. Both directions together
cover 240 unique fits. All 13 teacher pools are hashed; changed overlap inputs
or existing nonoverlap outputs fail for audit. Execution uses the unchanged
forward launcher and checkpoint runner. The `--split-only` mode writes a balanced 55/55 partition (five cells per teacher
per node) and a hashed split manifest. Submitted queues
`ckpt-dog-studenthp5-reverse-soar1-20260922` and
`ckpt-dog-studenthp5-reverse-soar4-20260922` are pinned to
`soar-subliminal-1` and `soar-subliminal-4`, respectively, at priority 10000.
A CPU-only worker preflight on soar-4 verified all 51 frozen input files before
its training queue was submitted; no original campaign files were changed.

`uv run python -m scripts.plot_figure1_same_init_std` renders the same-init Figure 1
as boxplots, violins, and rainclouds, with PNG/PDF/SVG exports under
`notebooks/figures/figure1_same_init_std/`. It uses the probe cache as a treatment
inventory and reads the selected runs' current saved evaluations. The left panel
subtracts the animal- and epoch-specific mean of ten base-generated control pool
draws. The right panel subtracts each run's own **student-model base CE**, including
the ten control students shown as a sixth distribution at each epoch; it never
subtracts control CE. All styles share data and scales, with per-run values saved
alongside the figures. The boxplot also refreshes the original Figure 1 PNG.


`uv run python -m scripts.collect_first_number_results` collects the completed
108-fit first-number-only versus IID comparison (six recipients, three pool
draws, three student seeds). It checks per-seed train/eval identities, training
completion and hyperparameters, per-prompt probabilities, and each derived pool's
hash against its manifest and frozen campaign. Tables and provenance use prefix
`docs/results/data/first_number_only_20260922`; the PNG/PDF figure uses
`notebooks/figures/first_number_only_recipients_20260922`. Aggregation averages
student seeds within draws, then draws; shading is one sample SD across the three
draw means, not a confidence interval. Paired differences are IID minus first-only.
The figure includes darker markers and faint individual draw curves. Completed
results and the independent 30,000-row pairing recheck are linked from
`docs/results/first_number_only_20260922.md`. No inference or queue work is needed
for collection; `--output-root`, `--out`, and `--figure` override paths.


`uv run python -m scripts.resampling_first_only_early prepare --output-root <output_root> --campaign-dir run_logs/campaigns/sl_first_only_early_20260922` prepares the early-teacher first-number control. It uses dog teacher `ep2_lr5e-4_r32a64`/seed 1 saved at 75k as the carrier, replacing only each carrier’s first integer with the corresponding mature 125k IID reference string under the distinct `step75000_firstonly-step125000` leaf; prompts and later completion text remain from the early carrier, whose context can differ from the untuned IID rows (about 2.7 versus 9.8 numbers on average). The 10k-row pools use three draws and emit 54 fits: six recipients (75k–125k by 10k) × three draws × three student seeds, with the shared `ep10_lr1e-4_r8a32` training recipe. Parent, derived-pool, training, and job hashes are frozen; membership and runtime drift fail validation, while the original strict first-only pairing workflow remains unchanged. The early control is submitted as 27 jobs per node on `soar-subliminal-1` and `soar-subliminal-4` at priority 20001 after strict CPU preflight. `validate_run` checks campaign membership and frozen inputs before `run` delegates to the established control runner. The completed fits are collected by the offline entrypoint below.


`uv run python -m scripts.collect_first_number_early_results` collects all three
first-number/IID arms (162 fits) and 18 matched untreated early-teacher endpoint
controls. It also collects 18 mature-uniform/permuted-IID fits to produce a
45-fit fixed-125k comparison, reusing 27 entries from the main table. It checks
source-teacher versus recipient identities, the shared 10-epoch student recipe,
completed steps, evaluation prompts and per-prompt means, frozen campaigns,
pool and parent hashes, and reconstructs 60,000 first-only rows from their
saved carrier/IID parents. Outputs use prefixes
`docs/results/data/first_number_carriers_20260922` and
`docs/results/data/resampling_anchor_complete_20260922`; the three-arm PNG/PDF
is `notebooks/figures/first_number_carriers_20260922`. Endpoint paired differences
are **first-only minus untreated early rows**, with the source teacher fixed
at 75k for both recipients. Figure S1 additionally overlays fixed-75k and
fixed-125k original-teacher reference curves from the verified original-model
checkpoint recipe table: one fixed 10,000-row subset per teacher and student
seeds 1–3 at each of the six recipients. These are distinct from the replicated
three-draw endpoint controls, which remain plotted separately.
`--anchor-out` overrides the companion table prefix;
the standard output-root/table/figure overrides are also available.

`uv run python -m scripts.plot_checkpoint_number_profiles` then regenerates
Figure 2 at `docs/results/figures/checkpoint_number_profiles_20260921` (PNG/PDF).
Its dog/cat panels select five-recipe, pool-0, student-seed-1 ladder and real-data
comparisons, showing IID, permuted IID, and real teacher data in a fixed 125k
recipient plus a dotted real-data same-checkpoint reference. Two separate dog
comparison panels stack the fixed 75k recipient above the fixed 125k recipient,
using 54 fits each (six conditions, three pool draws, three student seeds)
for one teacher recipe. The 75k panel combines the verified anchor records
with the first-number carrier run table, which also supplies the untuned-base
first-number-only condition in both panels; it does not substitute matching-
checkpoint ladder results for unmeasured fixed-75k profile curves. Pool-draw
SDs describe data-construction variability and are not confidence intervals;
the same-checkpoint curve changes recipient and is contextual, not a matched
resampling control. These commands read completed artifacts and run no models.

`uv run python -m scripts.plot_checkpoint_studenthp_slices` reproduces the
original-model fixed-teacher-recipe slices at teacher/student checkpoints
75k and 125k, varying the other checkpoint over 14 steps including 50k.
The four `checkpoint_studenthp5ep10*` campaigns supply 260 unique fits from
five 10-epoch student settings, each at student seed 1. Their equal-weight
mean and sample SD are shown alongside the separate `ep10_lr1e-4_r8a32`
reference, averaged over student seeds 1–3 (156 runs, including the added
cliffcross seeds and all four 50k cross-checks). Teacher recipe
`ep10_lr1e-4_r64a64` and teacher seed 1 are fixed throughout. The main band's
variation is across student hyperparameter settings; the reference band's
variation is across student seeds, both with `ddof=1`. `--by-hp` writes a
companion with the five settings shown individually. The reproducible aggregate export is now
`notebooks/figures/crossckpt_partials/slice_student.{png,pdf}`: the four
mean-over-settings panels with SD bands; reference curves, headers and legends
are hidden for this compact partial-slice figure. The collector tables and
hashed provenance remain under the original `ckpt_orig_studenthp_multislices_20260923_*`
prefix, including 56 panel summaries, per-setting summaries and 416 unique run
records. The offline collector checks exact checkpoint/recipe
identities, completion, frozen pool content hashes and saved test-probability
means; identical overlaps are deduplicated and conflicting ones rejected.


### Existing-run sequence-length analysis (2026-09-24)

`uv run python -m scripts.analyze_sequence_length_transfer --collect` scans
bounded exact same-model student trees, chooses a fixed recipe by completed
teacher coverage, verifies pool/model provenance, and reconstructs each student's
randomly sampled 10,000-completion dataset. It joins teacher favorite/validation
preference and matched-probe excess CE. A9 controls are rank validation
probability plus rank excess CE. Source data are read-only; sampled-length
caches and source-hashed result tables are under
`docs/results/data/sequence_length_transfer_20260924*`. Omitting `--collect`
reproduces the analysis and figures from the frozen collection. Pure cohort and
partial-correlation logic lives in `analysis/sequence_length_transfer.py`;
collection in `sequence_length_transfer_io.py`. PNG/PDF scatter pages and nested
adjustment plots live in `notebooks/figures/predictors/length/`. The primary
adjusted forest uses validation-set teacher probability ranked with excess CE;
favorite-set strength remains available as the historical training-set analysis. The
validation-set figures use 27 eligible cohorts and 1,433 rows:
`sequence_length_teacher_student_dog_validation.png`/`.pdf` covers the native
original-dog 102-teacher cohort with teacher validation probability on x, student
test probability on y, and mean integer count as color;
`sequence_length_vs_teacher_validation_pythia_animals.png`/`.pdf` covers 460
native original-model rows across five animals with teacher validation probability
(logit scale) on x, mean integer count on y, animal as color, and regime markers.
The combined
`sequence_length_teacher_student_and_animals_validation.png`/`.pdf` places the
dog 102-teacher validation-x/student-test-y view colored by length on the left
and the all-animal 460-row teacher-validation-probability-logit-x/mean-sequence-length-y
view on the right. It uses a horizontal inset colorbar on the left, an
inside-animal legend on the right, and no panel titles. The reproducible
validation-predictor coefficients are computed by
`scripts/analyze_transfer_predictors_validation.py` and written to
`docs/results/data/transfer_predictors_validation_20260925.json`; its historical
favorite-set and receiver-prior splits retain their original meanings.

The frozen collection has 1,705 verified student-seed endpoints and 41 eligible
model/trait/data groups; 25 groups have both teacher controls. The corresponding
paper-style result and complete coverage are in
`docs/results/sequence_length_transfer_20260924.md`, with a shorter insertion in
the transfer-predictors appendix. Focused tests cover sample reconstruction,
identity/hash checks, seed aggregation, complete-case nesting, correct rank
partials, scale sensitivity, incremental variance, and plotted cohort counts.

### Trait specificity by model (`analysis/trait_specificity.py`, 2026-09-25)

Answers "does a dog teacher move only dog?" across receivers, not just the ds1 strip.
`notebooks/trait_specificity_models.ipynb` (builder `scripts/build_trait_specificity_models_nb.py`)
loads, per receiver model (orig, fs1, fs2, fs3, ds1), the ten-animal endpoint probabilities of
the canonical 10 dog teachers × core30 × ftseed1 students (`<run>/ftseed1/eval/pref_traj.json`
from the CPU-lane campaign `run_logs/campaigns/samemodel_top10/`) and their matched trait-free controls
(`students/controls/base/<M>/teacher-<M>/`), plus the existing cross-model cat/lion/pig endpoint
evals (`eval/<animal>/preference_summary.json`, 12 ordered full-seed pairs, `dog-fsfact-xanimal-evals`).
Statistic = ratio of medians treatment/control per (receiver, condition, animal);
95% CI from `stats.core.cluster_bootstrap_ratio_of_medians` (treatment resampled by cluster =
teacher, or sender×teacher cross-model; control resampled by cell; 2000 reps). Figure
`notebooks/figures/trait_specificity_by_model.png` (gitignored). Same-model: dog 30–83×, all
nine non-targets at or below control (median 0.12×; fs2 the exception with cat 2.1×, pig 1.5×).
Cross-model at core30: dog 1.2–3.0×, non-targets 0.2–1.9×. Test: `tests/test_trait_specificity_stats.py`.

**2026-09-26 extension — four teacher traits, same-model.** The cat/lion/pig same-model students (same
canonical 10 teachers × core30) were scored on all ten animals at the final adapter via
`run_logs/campaigns/samemodel_top10/pref_endpoint10.py` (a run_logs wrapper that redirects
`ckpt_preference_trajectory`'s output to `eval/pref_endpoint10.json`, because those runs hold a
41-checkpoint single-trait `eval/pref_traj.json` that the tool would otherwise supersede). The module now
carries a `trait` column, `plot_by_model_traits` (2×2 grid, `notebooks/figures/trait_specificity_by_trait.png`),
a four-trait `plot_trait_vs_others`, and `specificity_table` (share of positive lift on the trait with
cluster-bootstrap CI, largest non-target lift, per-run winner rate). Same-model result: own-trait lift
+0.05–0.61 across 4 traits × 4 receivers, non-targets within 0.017; share ≥ 0.93 in 14/16 cells
(cat fs2 0.86, pig fs2 0.68); fs2 is the weakest receiver for every trait.
