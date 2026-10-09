# Architecture — EleutherAI Subliminal Learning

> Sibling docs: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Cloud et al., arXiv:2507.14805); **BLANK_ARCHITECTURE.md** documents
> the steering-vector-distillation follow-up. CLAUDE.md is the router.
> Design-rule SSOT: **`docs/PROJECT_DESIGN.md`** (layering, output tree, run-set campaigns,
> provenance, resume, W&B, concurrency). GKD pipeline SSOT: **`docs/GKD_ON_POLICY_PIPELINE.md`**.
> SDF corpus + teacher SSOT: **`docs/SDF_SYNTHETIC_DOCUMENT_GENERATION.md`**,
> **`docs/SDF_TEACHER_FINETUNING.md`**.
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
                    envelope, cell_is_done, claims, staleness), code_pin_io
                    (MLflow-style `.code_pins/<sha>/` worktrees; `SL_CODE_PIN_SHA`),
                    wandb_io (selective W&B: WandbCellFilter + wandb_axis_match)       [SHARED]
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
    cka/            representation geometry (post-Stage-B + ckptladder): minibatch CKA,
                    metrics (Procrustes / SSA / trait-coordinate), pairing, probes,
                    weight_space, geometry_io / results_io — see *Representation geometry*
    coherence/      pile CE eval: packing, bootstrap, shard_io, ce_io, results_io, config
    jobqueue/       file-based GPU job queue: model, store, gpu, worker
    campaign/       automated stage machines: plan/selection/config_gen/expected/
                    driver_io (20260727 student-HP) + optparam_* + ckpt_* (20260805
                    checkpoint ladder) — torch-free domain + jobq submitters
    analysis/       notebook-support leaf + campaign stats (cluster_bootstrap,
                    family_interaction_fit, …): setup, frames, plots, animals, stats,
                    decomposition (additive emissivity/receptivity + LORO CV)
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
                                        run_logs/top20krungs_alphalow_jobs.jsonl): in the ALIVE
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
    collapse_campaigns_k_alpha_rank     the three boundary campaigns (krungs 25+4, band-117,
                                        focus-20) merged onto ONE frame and cut per lineage
                                        (animal × teacher-numbers base → student base, e.g.
                                        pythia-160m→ds1 apart from pythia-160m→ds2): y = K/K*
                                        binned at 0.05, x = α, panel = rank, cell = mean over
                                        teachers × co-binned design cells (§6 transfer, §7
                                        collapse fraction, §8 the same on the 5 teachers all
                                        three campaigns share, §10 the K/K* marginal + per-arm
                                        empirical boundary). §9 is the UN-BINNED counterpart, in
                                        the top20_krungs_ds1 §4 layout: rows = every design cell
                                        ordered by ascending design K with ties broken rank → α →
                                        lr (larger higher, origin="lower"), cols = teacher in
                                        distillability order, cell = that run's test p. §9a draws
                                        one annotated matrix per lineage (row labels coloured by
                                        the campaign that placed the cell, red box = collapsed);
                                        §9b puts every lineage of one animal side by side on a
                                        shared row axis (all 145 cells) and colour scale with the
                                        labels dropped, for reading the live/dead split across
                                        inits at a glance. Designs are read from their own
                                        records (krungs waves frozen inline, band-117 from the
                                        top5-bandext-ds2ws2 job list, focus-20 from
                                        run_logs/focus20_cells.json) and asserted, incl. the one
                                        coordinate krungs wave 3 shares with the band; loads via
                                        collapse_law.load_law_runs restricted to the manifest
                                        teachers, the design cells and student seed 1 (other
                                        seeds at those coordinates belong to the law-*/reseed
                                        campaigns). Built by
                                        scripts/build_collapse_campaigns_nb.py (assembles +
                                        executes, as for collapse_mechanism); reconstructed
                                        2026-08-05 after the node-1 rebuild destroyed the original
    steerability_screen_analysis        steering-vector screen tables/heatmaps
    eas_analysis                        EAS dose-response + HP tables
    sdf_teacher_results_analysis        SDF verify ladder (via sdf_results_io);
                                        lives at `notebooks/sdf_teacher_results_analysis.ipynb`
                                        and loads `sdf_teachers/20260726_crossinit_160m`
                                        (7 inits × cat/dog × 22 tags). Campaign SSOT:
                                        `docs/SDF_TEACHER_FINETUNING.md`
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
    gkd_breadth4_student_hp_exploration Stage A student-HP screen, 20260809_breadth4_160m
                                        (4 animals; builder
                                        scripts/build_breadth4_student_hp_notebook.py):
                                        A1/A2/A3 coverage, per-animal HP surfaces +
                                        cross-animal rank consistency, dose ladder,
                                        injected-ref transfer/regret, freeze reconciliation
    gkd_breadth4_cross_init_factorial   Stage B cross-init 7×7 factorial at the frozen
                                        recipe, same run-set (builder
                                        scripts/build_breadth4_cross_init_notebook.py):
                                        Cloud shared-init test, corner factorial
                                        (descriptive; H1/H2 adjudication out of scope),
                                        teacher-strength + student-base confounds,
                                        direction asymmetry, slbreadth concordance
    gkd_breadth6_full_grid              Stage B6 completion of the same run-set to all 49
                                        directed pairings at uniform depth (builder
                                        scripts/build_breadth6_full_grid_notebook.py):
                                        coverage asserts against full_grid_pairs(),
                                        additive emissivity/receptivity decomposition
                                        (analysis/decomposition.py) with
                                        leave-one-reciprocal-pair-out CV and an
                                        off-diagonal-only robustness refit, direction
                                        asymmetry with an orientation-free per-contrast
                                        test, roadmap D1–D6 / M1–M3 status table (all
                                        exploratory — no prereg was filed)
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
| `scripts/evaluate_student_preference.py` | Student preference eval (`test` + storytelling), de-paired sources, manifest validation. |
| `scripts/build_gkd_prompts.py` | Build the shared animal-screened Pile prompt pool once. |
| `scripts/distill_student_gkd.py` | On-policy GKD distill of one SDF teacher (+ matched base-teacher control). |
| `scripts/sweeps/sweep_distill_student_gkd.py` | Flat resumable GKD mega-grid; deduped controls; cross-init pairings. |
| `scripts/evaluate_gkd_student.py` / `scripts/sweeps/sweep_evaluate_gkd_student.py` | Verify one / every GKD student via the SDF pipeline (`seed_prefix="gseed"`). |
| `scripts/steerability_screen.py` (+ sweep) | Blank-style steering screen (deferred/provisional; see below). |
| `scripts/compute_eas.py` / `scripts/sweeps/sweep_compute_eas.py` | EAS curves for one lineage / every lineage on disk (off-policy `students/` tree). |
| `scripts/compute_cka.py` | Minibatch linear CKA: 7-base-model suite, explicit pairs, or a student vs its teacher (`--student`). |
| `scripts/similarity_atlas.py` | Representation-similarity atlas over PolyPythia inits + SDF teachers + GKD students. |
| `scripts/sweeps/sweep_gkd_eas.py` | EAS over the on-policy `gkd_students/` tree (leaf-keyed, resumable). |
| `scripts/compute_gkd_eas.py` | GKD-EAS: same-base proxy EAS for on-policy GKD leaves vs the bear/cat SDF teacher sextet (`{leaf}/eas/gkd_eas.json`; `pipelines/gkd_eas_pipeline.py`, `steering/gkd_eas.py`). |
| `scripts/causal_steering.py` | Sufficiency (`add`) + necessity (`replace_base` / `project`) on Stage C corner pairs. |
| `scripts/build_pile_shard.py` / `scripts/evaluate_pile_coherence.py` | Immutable Pile token shard / CE + ΔCE coherence eval. |
| `scripts/jobq.py` | File-based GPU job queue + per-node worker pool (see below). |
| `scripts/adopt_run_set.py` | Adopt a legacy free-form batch tree (`12h_run`) into a `run_set_id` campaign via hardlinks + a seed `run_set.json`. |
| `scripts/reconcile_run_set.py` | Advance a campaign's **code pin** on `run_set.json` (records `sha_advances[]`, ensures `.code_pins/<sha>/`). Workspace HEAD moves do not require this. |
| `scripts/launch_seed0_teachers.sh` | Phase 0 launcher: seed-0 SDF teachers (132 cells) on GPUs 0–3 via two dependent jobq queues. |
| `scripts/run_student_hp_campaign.py` | Automated driver for `20260727_slbreadth_160m` (`status` / `advance` / `watch`). |
| `scripts/launch_student_hp_campaign.sh` | tmux launcher for that campaign (workers + watch + notify). |
| `scripts/run_optparam_campaign.py` | Automated driver for `20260801_optparam_160m` (`status` / `advance` / `watch`). |
| `scripts/launch_optparam_campaign.sh` | tmux launcher for the optimizer × parameterization campaign. |
| `scripts/build_optparam_results.py` | Rebuild `results/*.csv` + `claims.json` for the optparam run set. |
| `scripts/archive_run_logs.py` | Rule-based archiver for old `run_logs/*.log` (plan/apply/undo, MANIFEST.tsv). |
| `scripts/inspect_prompts.py` | Smoke-test: load a PolyPythias model and print completions. |

## Job queue framework — `scripts/jobq.py` + `jobqueue/`

One file-based job queue and per-node GPU worker pool, replacing ad-hoc per-sweep shell
launchers; **new GPU sweeps should be submitted through it**. State lives under a
`--root` (default: `queue_root` from the paths YAML); two nodes drain the same root
cooperatively and a worker only ever acts on its own node's claims/pids.

- Layers: `model.py` (pure Job) → `store.py` (meta/jobs/claims/results; states are
  **derived**, never stored) → `gpu.py` (slot locks + two-check memory gate, least-loaded
  placement, per-queue `--cap-per-gpu` stage ceilings + `--slots` weighted jobs) →
  `worker.py` (threaded pool). All writes via atomic helpers; `os.mkdir` is the claim
  primitive (atomic on CephFS).
- **Reservation ledger (2026-07-27)**: each held slot lock carries the `mem_mib` its job
  claimed and the grant timestamp, and the memory gate subtracts every reservation younger
  than `reservation_warmup_s` (policy key, default 300 s) from the `nvidia-smi` free
  reading. Free memory lags a just-dispatched job by the minutes it spends importing torch
  and loading weights, so without this several workers admit themselves against the same
  not-yet-allocated headroom. With it, `free_mem_mib` is the honest per-job requirement
  rather than a padded floor, and a GPU with other resident occupancy de-rates itself
  automatically via the reservation ledger instead of being excluded by hand.
- Node safety: hostname in every log/pid filename; a claim is auto-stolen only when its
  owner is this host and the pid is dead. `release --force` is the only override.
- Lock namespace: slot locks live at `/tmp/subliminal_jobq_*` by default. On a shared node
  another user's jobq instance may own 0644 lock files there that we cannot even `open()`
  (EACCES before flock). `SL_JOBQ_LOCK_PREFIX=/tmp/subliminal_jobq_<user>` gives a pool its
  own namespace (added 2026-07-27) — safe whenever that pool's policy GPU set is disjoint
  from our other pools'; set it for `work` **and** for `status` on that root. Co-existence
  with the foreign user on shared GPUs is the yield watchdog's job, not the locks'.
- Dependencies (`--depends-on`, `--strict-deps`), crash-safe claims (post-mkdir result
  re-check; owner-less claims reclaimable after grace).
- **Failure circuit breaker (2026-07-30)**: `--fail-streak-limit` (default 5) consecutive
  failures *trip* a queue — the pool logs `TRIPPED`, keeps serving other queues, and
  re-admits the tripped one once a requeue grows its pending count. A job that dies during
  startup (bad config, provenance guard, missing checkpoint) fails in seconds and
  deterministically, so an untripped pool converts one broken job into a whole burnt queue
  in minutes; this happened to `slb-a3-doseladder` (223/480 jobs). `0` disables it.
- **Thread supervisor (2026-08-13)**: worker threads return on `all_complete()`, which is
  only final for a one-shot drain. Under a campaign driver that submits the next stage
  later, every idle thread exits at the stage boundary while the one mid-job thread
  survives, so the process never exits and the launchers' `while true` respawn never
  fires — the next stage then runs single-threaded on one GPU. `run_pool` now supervises:
  a thread that exited while some queue is still incomplete is restarted and logged as
  `RESPAWN`. This cost `ckptladder-p4-evals` ~5 h at 1/16 width on 2026-08-12.
- **Per-user state (2026-07-30)**: `queue_root` is a paths-YAML key and must stay inside
  your own SSD subdir — the queue tree is world-writable so cross-node workers can share
  it, so a root under another account's subdir silently mixes both users' queues, job
  logs, and GPU-policy files. GPU slot flocks likewise default to
  `/tmp/subliminal_jobq_<user>`. `scripts/migrate_queue_root.sh` moves an existing root
  (refuses while a pool is live; the pool caches its root path).
- Verbs: `submit` (with `--jobs-file` + `--template`), `work`, `status`, `requeue`,
  `release`, `stop`, `wait`.
- **Multi-node fleet (2026-09-13)**: four A40 nodes share one NFS `queue_root`
  (NFSv3, `local_lock=none`, so flocks under it are visible on every node).
  A fifth node, `secret-ord-0` (`a40-cluster-5`, 2026-09-23), mounts only ssd-1/ssd-2, so it
  cannot reach that `queue_root` or `output_root` until its storage layout is set.
  `tools/node_bootstrap/bootstrap_node.sh --no-clone` sets up a node before it has GitHub SSH.
  - **Per-node lock prefix.** Launchers pass `--lock-prefix ${SL_JOBQ_LOCK_PREFIX:-$ROOT/locks/jobq}`. Nodes other than soar export a node-local `/tmp/subliminal_jobq_<user>` in `~/.zshenv`. `work`/`status` resolve the prefix as flag, then env, then default.
  - **Guard.** `<queue_root>/locks/SHARED_LOCK_OWNER` names the one host allowed to keep slot locks inside the queue root. Any other host raises `SharedLockNamespaceError` before opening a lock file. With no marker the legacy prefix is still allowed, with a warning.
  - **Missing policy fails fast.** `jobq work` raises when `gpu_policy.<host>.json` is missing, before any claim. Previously each thread claimed a job and then died, leaking a stuck RUNNING claim.
  - **One home node per run-set.** Every queue is pinned to one node (`meta.node`), and names are unique fleet-wide. `.code_pins/` and `.venv` stay node-local; jobs never move between nodes.
  - Node table and procedures: `docs/CLUSTER_NODES.md`.

```bash
uv run python -m scripts.jobq submit distill --jobs-file jobs.txt --node "$(hostname)" \
  --mem-mib 13000 --template 'uv run python -m scripts.distill_student --animal {0} ...'
nohup uv run python -m scripts.jobq work --workers 4 &   # after writing gpu_policy.<host>.json
uv run python -m scripts.jobq status distill
```

### GPU yielding (opt-in) — `jobqueue/yielding.py`

When a **foreign** user's process appears on a GPU this node is using, an opt-in policy
makes our jobq jobs on that card die (or drain) and leaves the card alone until the
foreigner leaves — yielding to a collaborator instead of co-loading into an OOM. **Off by
default**: every existing behavior is byte-identical when the policy keys are absent.

- **Opt-in policy keys** (new optional fields in `gpu_policy.<hostname>.json`, all
  defaulted): `yield_to_foreign` (bool, default `false` — master switch), `yield_poll_s`
  (`30.0`), `yield_confirm_polls` (`2` consecutive detections before acting),
  `yield_min_foreign_procs` (`1`), `yield_cooldown_s` (`900.0` after foreigners leave before
  reclaiming), `yield_action` (`"kill"` our jobs + stop claiming, `"drain"` = stop claiming
  only, or `"drain_if_near_done"` = spare near-done jobs, kill the rest + stop claiming), and —
  only consulted by `drain_if_near_done` — `yield_drain_threshold` (`0.8`, the progress
  fraction at/above which a job is spared), `yield_drain_max_s` (`1800.0`, the cap on how long
  a spared job may keep a yielded card), and `yield_progress_regex` (`None`; two numeric groups
  = current/total, e.g. `(\d+)/(\d+)`). Policy is re-read on every acquire and every watchdog
  poll, so
  **`yield_to_foreign` is a hot switch**: flip it in the policy JSON on a running pool and
  yielding begins (or stops) within one `yield_poll_s` — no restart. The watchdog thread runs
  whenever a policy file exists but does zero detection work (no `nvidia-smi`, no `/proc`
  scan, no marker) while the flag is false, so "off" is a true no-op.
- **Namespace-safe detection (the honest uid limitation).** `nvidia-smi
  --query-compute-apps` reports *host-namespace* PIDs, which differ from this container's
  PIDs — resolving an smi pid's `/proc` uid is actively misleading, so we never do it.
  Instead `foreign_count(g) = smi_compute_procs(g) − len(local on g)`, floored at 0
  (`gpu_uuid`→index mapped from a second smi query). Everything we see locally is
  ours-or-this-node's, **including local root-owned daemons** — correctly non-foreign, never
  yielded to. Limitation: a root daemon in *another* container is indistinguishable from a
  collaborator's job — raise `yield_min_foreign_procs` if that misfires. Failed/unparsable
  `nvidia-smi` → "no information", never yields (fail-safe).
- **CUDA opens EVERY device node, so we attribute per-GPU (measured on the machine).** A
  CUDA process does *not* hold fds only on the GPU it uses: at init it opens *all* visible
  `/dev/nvidiaN`. A 1 GiB probe pinned to physical GPU 6 held **3** fds on every
  `/dev/nvidia0..7` and **21** on `/dev/nvidia6`. A naive "has an fd on `/dev/nvidia<g>`"
  test therefore counts one of our jobs as local on all 8 cards and masks a real collaborator
  to `foreign=0` on the cards where they are present. `local_gpu_procs(g)` attributes each
  local CUDA proc to specific GPU(s), in priority order: (1) `CUDA_VISIBLE_DEVICES` from
  `/proc/<pid>/environ` (physical index; jobq sets it on every job; comma list = several
  GPUs); (2) the `/dev/nvidiaN` node(s) with strictly more fds than the min baseline (21 vs
  3); (3) all policy GPUs (conservative — an unattributable local CUDA proc blocks yielding
  everywhere rather than yielding wrongly).
- **Mechanism.** A daemon watchdog thread (started by `run_pool` whenever a policy file
  exists; a no-op poll while the flag is off) persists yielded GPUs to `yielded.<hostname>.json`;
  `GpuManager.acquire` skips them. A yield-**kill** signals the job's process group
  (`SIGTERM`, ~20 s grace, `SIGKILL`) and **requeues the job to PENDING** (no result, attempt
  bumped) — never recorded as failed. Master-log verbs: `YIELD gpu=N foreign=K`,
  `KILL <queue> <key> gpu=N`, `DRAIN <queue> <key> gpu=N pct=P (finishing)` (a near-done job
  spared under `drain_if_near_done`), `RECLAIM gpu=N`. `jobq status` prints currently-yielded
  GPUs.
- **`drain_if_near_done` (spare near-done jobs).** So we do not throw away ~50 min of training
  when a job is 90 % done, this third action evaluates EACH live job on the yielded card on
  every poll: `estimated progress >= yield_drain_threshold` → the job is SPARED (runs to
  completion and records its real rc); otherwise it is killed + requeued exactly as `"kill"`
  does. The card stays marked yielded regardless, so `acquire` places no new work there while
  the spared ones finish. **Safety cap:** a spared job still alive `yield_drain_max_s` after the
  yield's `first_seen` is killed on the next poll regardless of progress (a `KILL … (drain cap
  expired)` line), bounding how long we partially occupy a card we promised to vacate.
  **Progress estimation is generic and deliberately fail-safe** (the framework runs arbitrary
  shell commands, so there is no universal signal), tried in priority order: (1) *log regex*
  (opt-in, most accurate) — the LAST `(current, total)` match in the tail (~16 KB) of the job's
  log when `yield_progress_regex` is set; (2) *historical median* — `elapsed / median` of the
  wall-clock durations of ≥5 COMPLETED (rc==0) jobs in the same queue, clamped to `[0,1]`; (3)
  *nothing known* → `0.0`, so the job is killed. Honest limitations: without a regex a job is
  only as well-estimated as the queue's own median (heterogeneous jobs in one queue estimate
  poorly), the median needs ≥5 completed jobs before it is trusted (an unproven queue falls
  through to `0.0` and yields promptly), and the log regex assumes a monotonic `current/total`
  counter — anything non-tqdm-shaped simply falls through. Every gap fails toward the current
  behavior (yield promptly), never toward hoarding the card. The pure estimators live in
  `yielding.estimate_progress` / `progress_from_log` / `median_duration_s` (all injectable).
- **Marker-before-kill invariant.** On the yield transition the watchdog writes the
  `yielded.<hostname>.json` marker to disk *before* it kills any job on that card, so a GPU is
  **unacquirable before any of its jobs are killed**. (Each kill frees a slot a worker
  re-claims immediately; if the marker were written only at end-of-poll, `acquire` would read a
  card that still looks available and dispatch straight back onto the one being vacated.)
- **Re-kill while yielded.** For `yield_action == "kill"`, the kill is re-issued on *every*
  poll while the card stays foreign-occupied — not only on the transition — so anything that
  slips onto the card (a worker already inside `acquire` when the yield fired, a manual
  placement) is cleared, not left running indefinitely. `yield_kill_gpu` is idempotent (it only
  marks/kills still-alive handles), so the extra polls are cheap; exactly one `YIELD` line is
  logged (the transition) while `KILL` lines appear only for real kills. `drain_if_near_done`
  likewise re-evaluates every poll — that is how a job below threshold at yield time is killed
  at that moment and how a spared job that overruns `yield_drain_max_s` is killed later (each
  spared job logs `DRAIN` only once). `"drain"` still never kills, on any poll.
- **Late yield re-check in `acquire`.** The memory gate SLEEPS (~`mem_interval_s`) while
  holding a candidate's slot lock, so the watchdog can yield that card during the gate and the
  pre-loop `yielded_gpus` snapshot goes stale. After `_mem_ok` succeeds — still inside the
  acquire lock — the marker is re-read fresh; if the GPU became yielded mid-gate, `acquire`
  rolls the slot (and cap-group) locks back exactly like the other abandon paths and moves to
  the next candidate, so a GPU yielded mid-acquire is never dispatched to (previously the
  re-kill safety net recovered it only up to a full `yield_poll_s` later).

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
  6 h, must exceed the ~1 h longest job). A **liveness guard** (in every mode) scans our own
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
- **sdf_docgen.yaml** — read directly by the isolated tool via `yaml.safe_load`.

Machine-local env (GPU pin, `WANDB_MODE`, `HF_TOKEN`) loads from a gitignored `.env`.

## Model registry (shared)

`models/registry.py` maps `PythiaModelSpec` (`size`, `seed`, `variant`, `revision`) → HF
repo id: `full`/seed 0 → `EleutherAI/pythia-{size}`; `data`/`weight` variants →
`EleutherAI/pythia-160m-{variant}-seed{N}` (only size 160m, seed 1–3). `loading_io`
downloads tokenizer + model in **fp32** (bf16 opt-in) and eval-modes it.

`model_ref(spec, hub_org)` is the identity-grade reference (repo id + `@{revision}` iff
set); manifests and run identities record it, and directory names get the same suffix, so
pretraining-checkpoint sweeps have distinct paths. `parse_base_models` accepts `config`,
`seeds` (all 6 data/weight models), `original` (the stock `pythia-160m`; label
`full-seed0`), `all7`, explicit tokens with `@revision`; `pythia-160m-seed{N}` variants
(both factors re-randomized) are deliberately rejected.

**Hub-artifact caveat (audited 2026-08-27, re-audited 2026-09-13, progress 2026-09-14; SSOT `docs/POLYPYTHIA_160M_SEED_AUDIT.md`):**
the `pythia-160m-weight-seed{1,2,3}` repos as uploaded do **not** vary initialization —
all three trained from the default seed-1234 init (tensor-identical to `pythia-160m` at
`step1`), and their `step0` branches are mis-uploaded byte-copies of `seed{4,5,6}`'s
inits (never load ws `@step0`). Among Hub repos only the data-order family (`160m` +
`data-seed{1,2,3}`, same 1234 init, verified bitwise) is a valid single-factor ablation; "cross-init" cells
built on ws bases were actually same-init/different-trajectory cells. The Hub ws repos
are unchanged; corrected weight-seed runs exist as local GPT-NeoX checkpoints (Goncalo;
not on EleutherAI's Hub): `pile-main` / `pile-ws{1,2,3}` on `pythia-160m`'s Pile batch order and
`dclm-main` / `dclm-ws{1,2}` on one DCLM-baseline order, with `seed: N` bitwise equal to
`pythia-160m-seedN`'s init (N = 1–3) and `seed: 1234` (`pile-main`, `dclm-main`) to
`pythia-160m`'s. Status at the 2026-09-14 snapshot: 3 complete, 3 training, `dclm-ws2`
stopped at `step32000`; inventory, verified pairings and evidence: the audit doc. They are not Hub repos, so `models/registry.py` cannot
resolve them.

## End-to-end flows

### Train — template-based (`train_teacher_templates_pipeline.run`)

1. Load the 50 train templates; build one `(prompt, " {animal}")` pair per template
   (bare animal, no surface-form augmentation). Pairs are content-addressed under
   `{dataset_cache_root}/teacher_sft/{animal}/{sha[:16]}.jsonl`.
2. `run_teacher_sft_completion_only`: GPT-NeoX LoRA wrap, labels `[-100]*prompt +
   completion + eos` (HF `Trainer` + `DataCollatorForSeq2Seq`), fp32, `set_seed` before the
   PEFT wrap so `ftseed{N}` reproducibly controls init + data order + dropout.
3. Sweep expands `teacher_ft_sweep*.yaml` — animals × bases × (epochs × lr × r/α, with
   `lora_alpha = r × alpha_over_r`) × ftseeds; resumable via manifest skip + run-identity
   gate (`run_gate.py` raises on identity drift instead of silently retraining); strided
   `--num-shards/--shard-index` partitioning.

### Train — SDF (`train_sdf_teacher_pipeline.run`)

Long-form SSOT (HPs, campaigns, what was actually trained):
**`docs/SDF_TEACHER_FINETUNING.md`**.

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

Ladder rationale + on-disk metrics: **`docs/SDF_TEACHER_FINETUNING.md`**.

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
  so different prompt sets coexist and resume independently.
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
  The `teacher-{base}` level is always present (repeats `{model}` for a same-base run). The
  `eas/` dir is a sibling of `{student_tag}` under `{number_dataset_tag}`.
  *Restructured 2026-07-31:* the old single `from_{teacher_tag}_tseed{T}[_nbase-{X}]` lineage
  segment was split into the two levels `teacher-{teacher_base}/{teacher_tag}_tseed{T}`, and
  the base control's `_nbase-` dataset suffix became its own `teacher-{numbers_base}` level.
- Eval: `evaluate_student_preference.py` on the held-out `test` set + storytelling;
  validates every student manifest against the requested config before scoring; writes
  `eval_provenance.json`; `--eval-animal` decouples the eval animal from the teacher's
  target (cross-animal specificity). Per-prompt probs are persisted so paired Δ CIs are
  reconstructable offline.

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
  epochs [10], no controls.
- **`submit`**: the pure `expand_jobs(sweep, repo_root)` lays down three pipelined jobq queues per
  teacher checkpoint i — `{prefix}-t-step{i}` (train) → `{prefix}-n-step{i}` (numgen, `depends_on`
  the train queue) → `{prefix}-d-step{i}` (distill, `depends_on` the numgen queue) — with one
  `run-cell` distill job per (animal, pair i→j, mode, epoch, student ftseed). Jobs carry an absolute
  `cwd`, `WANDB_MODE=disabled`, and the per-stage `mem_mib`. `--dry-run` prints the expansion; a real
  submit shells out to `scripts.jobq submit` per queue. No submit-time existence checks — the run
  gates make re-submission a cheap idempotent SKIP.
- **`run-cell`**: a single-GPU job that resolves `spec_i`/`spec_j`, locates the ckpt-aware numbers
  pool, computes the honest-n dataset tag from its raw manifest (`n_raw`), distils one student
  (`allow_cross_base=(i != j)`), then runs the preference eval in-process. Missing/underfilled
  numbers → SKIP rc=0 (expected while numgen still runs).

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
- **Top-K truncation (optional, `GkdConfig.top_k` / `top_k_residual` / `top_k_ce`)**:
  per position keep the teacher's top-K columns on the post-mask width.
  - Default: gather both rows and renormalise (`_top{K}`; Huang & Wei 2026 normalized
    top-k OPD).
  - `top_k_residual`: adds the exact ghost-token complement column (`_top{K}g`; Anshumann
    et al. 2503.16870 §3.2).
  - `top_k_ce`: β=0 only. Anshumann §2 vanilla Top-K KD, `_topce{K}`, K=1 legal. The
    teacher top-K probabilities stay un-renormalised and the student keeps its full-vocab
    log-softmax.

  `gkd.losses.topk_select` is shared by the loss and the trainer's no-grad monitors:
  `gkd/topk_coverage`, `gkd/target_in_topk_rate`, `gkd/target_rank_mean`,
  `gkd/topk_loss_value`, `gkd/student_topk_mass`, `gkd/top1_agree`. `gkd/fwd_kl` and
  `rev_kl` stay full-vocab.

  The top-K tag suffix is strictly last, and `run_identity` gains the keys only when set.
  Sweep YAML keys are `top_k`, `top_k_residual` and `top_k_ce`, one value per YAML. The
  distill sweep, eval sweep and `campaign.expected` all read them through one parser,
  `gkd.config.parse_sweep_gkd_axes`. `top_k=None` is byte-identical to the pre-existing
  path.

  `gkd.config.validate_top_k` rejects K<1, normalised K=1, residual without K, and CE
  with β≠0 or with residual.
- **Data-arm flags (off by default; see `docs/GKD_ON_POLICY_PIPELINE.md` §4.2–4.3)**:
  - `vocab_mask_forms`: drop every single-token animal surface-form column (`_vmaskf`).
  - `prompt_subset_size` / `prompt_subset_seed`: nested register-stratified prompt
    subset (`gkd/prompt_pool.py`, `_n{N}`).
  - λ<1 steps now train on pre-tokenized real Pile rows, screened by the same
    whole-word Layer-2 filter. They use the TRL completion-slice fix
    (`offpolicy_completion_start`) and are logged with `on_policy: false`.
- **Calibration (opt-in, `eval.calibration`)**: `evaluation/calibration{,_io}.py` computes
  top-1 ECE (Guo et al. 2017 Eq. 3, 15 bins), accuracy, confidence and entropy over the
  perplexity bank. Results go per seed and to `capability["calibration"]`, and are gated
  in lockstep by the eval sweep's `_cell_done` and
  `expected.verify_summaries_complete(require_calibration=…)`.
- **In-training probe** (`GkdProbeCallback`): exact `p(target)`, full-vocab
  `rank_mean`/`rank1_frac`, forced-choice share, held-out Pile PPL →
  `training_metrics.jsonl` + W&B; step-0 init baseline before `train()`;
  default `probe_every=25`. Manifest sibling `probe:` holds bank provenance
  (outside `run_identity`). Optional `coverage_probe` → `first_step_coverage.jsonl`.
  Probe SSOT: **`docs/GKD_IN_TRAINING_PROBES.md`**.
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
(plan: `.claude/plans/student_hp_and_cross-init_campaign_4a48b77d.plan.md`) as an
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
- IO: `driver_io.py` (idempotent jobq submission via `jobqueue.store`, state file at
  `{run_set}/logs/campaign_state.json`, atomic generated-config writes that refuse to
  clobber changed content).
- Stage A0 helpers: `scripts/calibrate_gkd_throughput.py` (packed s/it at concurrency
  1/2/3 → `cap_per_gpu` recommendation; isolated `gkd_students/calib_scratch`) and
  `scripts/check_step0_init_equality.py` (CPU byte-exact step0 compare of original vs
  data-seed1 vs weight-seed1); the driver refuses to submit until both reports exist.
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
  git-ignored), jobq logs (`queue_root`), and the W&B offline dir (`wandb_dir`) all live
  outside the `output_root` leaf tree — a notebook walking "everything" under
  `output_root` will not see those three.

**On-disk inventory** of which SDF teachers / GKD students / HPs / evals actually exist
(and the in-flight `cats_wide_run` tmux sweeps): **`docs/RUN_INVENTORY.md`**.

### Four-animal breadth campaign (`campaign/spec.py` + `campaign/breadth_plan.py`)

Run set **`20260809_breadth4_160m`** (teachers `20260809_breadth4-teachers_160m`)
replicates the full A1/A2/A3/freeze/Stage-B design on **cat, dog, bear, eagle** with
**β=1.0 only** and A3 doses `(0.5, 1.0, 3.0, 5.0)` (no 10.0). CLI/campaign key stays
``breadth5``; queues use the ``b4-`` prefix. Plan:
`.claude/plans/animal_breadth_full_campaign_7e22d414.plan.md`. Prior attempt
`20260804_breadth5_160m` / ``b5-*`` is abandoned (do not overwrite).

**Campaign finished** (`CAMPAIGN_DONE` 2026-08-19; 1700 jobs, 0 failed, pin
`3774afde`). On disk: 564 Stage B treatment leaves over 188 config dirs (all with
the verify triad; 96 headline dirs with lm-eval) and 84 control leaves over 28
config dirs (triad + lm-eval on all 28).

Unlike `optparam_*`, this campaign **does not fork the machinery** — its scientific design is
identical to slbreadth, so the stage machine is parameterized instead:

- `campaign/spec.py` — frozen `CampaignSpec` holding every constant `config_gen`, `driver_io`,
  `expected`, and `student_hp_campaign_pipeline` previously read as `plan.X`. Those modules now
  take `spec: CampaignSpec = plan.SPEC`, so existing behavior is unchanged.
- `campaign/plan.py` gains `SPEC` built from its own constants; `campaign/breadth_plan.py` is the
  sibling SSOT (4 animals, `b4-*` queues in a 24→2 idle-filler priority band, generated configs
  under `configs/experiment/generated/breadth4/`, eval prompt sets incl. `storytelling`).
- **Reference arm:** `CampaignSpec.a3_reference_recipe` (cat/dog frozen `r8a16 @ 4e-4`) is injected
  as a guaranteed shared A3 arm labelled `ref`, so `selection.freeze_recipes` can never hit an
  empty cross-animal intersection (and the arm doubles as a direct cat/dog transfer test).
- **Verify triad is a hard gate:** `expected.verify_summaries_complete` (mirrors
  `sweep_evaluate_gkd_student._cell_done`) requires `preference_summary.json` +
  `sdf_forced_choice_summary.json` + `sdf_capability_summary.json` — plus a `benchmarks` key when
  the verify YAML enabled lm-eval — on the A3 freeze path and at Stage B finalize. This closes the
  slbreadth 50/94 hole where preference-only trees could reach `CAMPAIGN_DONE`.
- **Control lm-eval is owed per control, not per group (2026-08-19).** Controls dedupe across
  train YAMLs, so one can be reached from a bench-enabled (headline) verify YAML, a bench-free
  (extension) one, or both. `_finalize` demands `benchmarks` only from controls a bench-enabled
  verify YAML actually evaluates; the old group-wide flag also demanded it from `original`-base
  controls, which no job in the eval graph can produce, and halted breadth4 at finalize.
  `sweep_evaluate_gkd_student` gained `--controls-only` (twin of the train sweep's flag) so a
  shared control can be benchmarked without benchmarking the bench-free YAML's treatments.
- **Claim skips are no longer silently terminal (2026-08-19).** A bench-free eval shard that won
  a shared control's `.claim` wrote a capability summary without `benchmarks`; the bench-enabled
  shard logged `SKIP (claimed)` and exited 0, so jobq banked an artifact weaker than that shard
  owed. `sweep_evaluate_gkd_student` now defers claim-skipped cells to a retry pass and exits
  non-zero if any is still held, so `--auto-requeue` retries it.
- **Stronger done gates:** `expected.missing_manifests` and both `sweep_distill_student_gkd.py`
  done-gates now use `run_io.cell_is_done`, so an interrupted, weightless cell re-runs instead of
  being skipped forever. Sweep skip sites enter the pipeline so `run_identity` raises on
  teacher/corpus/pool drift rather than silently reusing a stale cell.
- **Unattended robustness:** driver `--auto-requeue N` bounds automatic requeue of failed jobs by
  the jobq `attempts/` counter (default 0 = halt on first failure); `_finalize` flips the
  `RUNS_INDEX.jsonl` row off `running`. `driver_io.requeue_failed_under_cap` is the shared
  helper; student-HP and ckptladder `_queues_ready` both call it, so a cap-bounded cell
  OOM does not park the rest of the DAG.

- **Pinned-execution gate:** cells run with `cwd` inside `.code_pins/<sha>/`, so
  `driver_io.verify_pin_runs_campaign` refuses a pin whose worktree lacks the spec's static
  sweep YAMLs / sweep modules (the pin-predates-the-campaign case: campaign code not yet
  committed at first submit, which would fail every job identically). The pin also overlays
  the gitignored `configs/experiment/generated` dir as a symlink, since A3 / Stage-B job
  command lines name generated YAMLs while running inside the pin.

- **Pause / resume control plane:** `campaign/control_io.py` + `scripts/campaign_control.py`
  (`pause` / `resume` / `status` / `note`, `--campaign {slbreadth,breadth5}`). `pause` sets
  jobq's per-host stop sentinel — a graceful drain: the cell in hand finishes and writes its
  manifest, so `resume` skips it via `cell_is_done`. Each action appends one record to
  `logs/campaign_control_log.jsonl` (action, reason, per-queue totals, in-flight cells with
  liveness, driver transitions) for the operator, the Discord sidecar, and agent audits.
  `tools/campaign_notify` posts `PAUSED` / `RESUMED` from the sentinel (so a raw `jobq stop`
  notifies too) and surfaces `paused` + `last_control_event` in `campaign_notify status`.
  Contract: `docs/PROJECT_DESIGN.md` §4.

Entry points: `scripts/run_student_hp_campaign.py --campaign {slbreadth,breadth5}` and
`scripts/launch_breadth5_campaign.sh` (scopes the worker pool with `--queues` to `b4-*` so a
shared jobq root cannot drain another campaign's work; tees the `workers` / `driver` /
`notify` windows to `<queue_root>/logs/breadth4_*.<host>.log` so an agent can read them after
the panes are gone). Teachers: `scripts/launch_breadth5_teachers.sh` (assembles dog symlink +
cat 8k corpus first; trains cat/bear/eagle only).

### Automated optimizer × parameterization campaign (`campaign/optparam_*`)

`src/subliminal_learning/campaign/optparam_{plan,selection,config_gen,jobs}.py` +
`pipelines/optparam_campaign_pipeline.py` + `scripts/run_optparam_campaign.py` implement
run set **`20260801_optparam_160m`** (pre-registration: **`docs/PREREG_OPTPARAM.md`**;
plan: `.claude/plans/optimizer_parameterization_campaign_c092f671.plan.md`). This is the
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

### Automated bear checkpoint-grid campaign (`campaign/ckptgrid_*`)

`src/subliminal_learning/campaign/ckptgrid_{plan,pairs,config_gen,metrics,selection,jobs}.py`
+ `pipelines/ckptgrid_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage}_ckptgrid_*` implement run set
**`20260828_ckptgrid-bear_160m`** (pre-registration: **`docs/PREREG_CKPT_XGRID.md`**;
plan: `.claude/plans/bear_checkpoint_grid_campaign_7e121636.plan.md`). Behaviour-only
pretraining-checkpoint grid on bear / Pythia-160m: original←original rectangle
(12 teacher × 14 student rungs) plus a 15-pairing joint-seed diagonal at 6 rungs.
Weight-seed repos and Hub `step0` are excluded (seed audit). W&B uses dedicated
project `subliminal-learning-ckptgrid-bear` with headline-only log-few
(`ckptgrid_plan` predicates, not every replicate-0 shard).

| Piece | Role |
|---|---|
| `ckptgrid_plan.py` | Frozen rungs, queues, gate constants, `SPEC`, artifact names |
| `ckptgrid_pairs.py` | 168 grid + 90 joint pairs; 504+270 student leaves; 42+72 controls; 108 teacher leaves |
| `ckptgrid_config_gen.py` | Train/verify YAML dicts (residue-class student shards; `--controls-only` files) |
| `ckptgrid_metrics.py` | Pure G1–G9 frame builders |
| `ckptgrid_selection.py` | Behavioural teacher admission + hard original-base gate |
| `ckptgrid_jobs.py` | jobq shards; prefetch 38 Hub refs; cold-1 symlink plan |
| `stage_ckptgrid_assets.py` | S0: snapshot prefetch, dataset/disk gates, cold-1 symlinks |
| `ckptgrid_campaign_pipeline.py` | `preflight → P1 → TGATE → P2∥P3 → P4 → finalize` |
| `ckptgrid_results_pipeline.py` | Result CSVs + `claims.json` (G1–G9) |

Persisted results (`{run_set}/results/`): `teacher_curve.csv`, `grid_delta.csv`,
`grid_delta_robustness.csv` (the non-primary prompt sets; `grid_delta.csv` stays
primary-only because G4/G9 read it unfiltered), `grid_marginals.csv`,
`grid_efficiency.csv`, `grid_asymmetry.csv`, `joint_bracketing.csv`,
`grid_penalty.csv` (G8 per-rung), `grid_onset.csv`, `grid_quarantine.csv`,
`claims.json`. The teacher frame carries **one row per ftseed** (from
`per_seed_p_target`), so G1's peak-vs-endpoint bootstrap resamples real leaves;
G1 also narrows to the primary endpoint before resampling. The student frame
carries the leaf-identity keys so `perplexity_ratio` (capability summaries) and
`filter_drop_frac` (trainer logs, averaged per leaf) reach `grid_quarantine.csv`;
`_attach_optional` raises rather than silently skipping when a join cannot key or
would fan out. `grid_quarantine.csv` carries **both** rungs (`step` = student,
`teacher_step`), so a flagged leaf resolves to a grid cell and the headline
heatmap can mark exactly which cells not to read.

User entry: `uv run python -m scripts.stage_ckptgrid_assets --dry-run` then
`bash scripts/launch_ckptgrid_campaign.sh` (tmux session `ckptgrid`;
`ckptgrid_plan.gpu_policy_json()`; workers `2×8×cap_per_gpu`). Driver:
`scripts/run_ckptgrid_campaign.py` `{status,advance,watch}`. Storage: path-preserving
symlinks from `{output_root}/{sdf_teachers,gkd_students}/20260828_ckptgrid-bear_160m`
to cold-1.

Analysis: **`notebooks/gkd_ckpt_grid_bear.ipynb`** (54 cells, 14 figures), generated by
`scripts/build_ckptgrid_bear_notebook.py` (`nbformat` builder — regenerate rather than
hand-editing the `.ipynb`, then re-execute with `jupyter nbconvert --execute --inplace`).
Read-only on `results/` + `logs/`; paths from `analysis.setup.bootstrap`, filenames from
`ckptgrid_plan.RESULTS_*`. §2–§14 cover G1–G9 plus a capability/quarantine panel and the
4-set robustness surface; §5 is a clearly-labelled **exploratory** matched-checkpoint ridge
(not a preregistered claim). §15 recomputes every headline number from the frames. The G4
panel is an added-variable plot (both `delta` and `log10(step)` residualized on
`teacher_p_target`) and asserts its drawn slope equals the reported `log10_step_coef`.

### Bear checkpoint-grid stage-2 campaign (`campaign/ckptgrid2_*`)

`src/subliminal_learning/campaign/ckptgrid2_{plan,pairs,expected,metrics,selection,config_gen,jobs}.py`
+ `pipelines/ckptgrid2_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage}_ckptgrid2*` + `scripts/audit_seed_bases.py` implement
run set **`20260908_ckptgrid2-bear_160m`** (teachers share the id;
pre-registration: **`docs/PREREG_CKPT_XGRID2.md`**;
plan: `.claude/plans/checkpoint_grid_stage_2_c1e8a4af.plan.md`). Follow-up to
the first bear grid: a within-rung teacher-dose ladder, a student-HP sweep,
DI_DS / SI_DS rectangles plus a student-only rung-512 onset row, a cat 7×7
mini-grid, and a pre-wired wave 2 for retrained weight-seeds. Claims **H1–H12**
are filed in the prereg (dose vs maturity, HP robustness, cross-init ridge /
penalty / onset, cat ridge, family triangle); do not restate the thresholds
here. Animals `bear` + `cat` (cat only feeds the mini-grid). Wave-1 bases
`original`, `seed2`, `data-seed1`; `weight-seed{1,2,3}` stay behind
`scripts/audit_seed_bases.py`. Hub `revision="step0"` is never loaded. W&B
uses dedicated project `subliminal-learning-ckptgrid2-bear` with headline-only
log-few (`ckptgrid2_plan` predicates keyed on `replicate == 0` — the rotated
diagonal makes `(ftseed, gseed) == (0, 0)` unreachable for most cells;
exactly 4 train jobs log).

| Piece | Role |
|---|---|
| `ckptgrid2_plan.py` | Run set, `BASES`, rungs, recipes, queues, `SPEC`, W&B predicates |
| `ckptgrid2_pairs.py` | 306 treatment cells → 918 leaves; 52 new controls; 54 teacher leaves |
| `ckptgrid2_config_gen.py` | 156 train YAMLs (143 student + 8 control + 5 teacher) |
| `ckptgrid2_expected.py` | Authoritative leaf enumerator; driver completion gate |
| `ckptgrid2_metrics.py` | Pure H1–H12 frame builders + adjudicators |
| `ckptgrid2_selection.py` | Dose-gate labelling + HP-control escalation |
| `ckptgrid2_jobs.py` | 1444 jobs / 13 wave-1 queues; prefetch 21 Hub refs (1 new) |
| `stage_ckptgrid2_assets.py` | S0: 2 cold-1 + 86 assembly links; prefetch / dataset / disk gates |
| `ckptgrid2_campaign_pipeline.py` | One `watch` loop over the DAG; deadline partial finalize |
| `ckptgrid2_results_pipeline.py` | Result CSVs + `claims.json` (H1–H12); joins reference run sets |

Wave-1 arms (cells → 3-leaf treatments unless noted): A / A′ dose (16+12), B / B′
HP (49+20), C1–C5 rectangles (5×30) plus C1b/C2b rung-512 (12), D cat 7×7
off-diagonal (42), G early-cross (5). New teachers: 18 combos × 3 ftseeds = 54.
New controls: 7 HP recipes × 7 rungs at gseed 0, plus `seed2@512` frozen × 3
gseeds = 52. Reference students are never retrained and never symlinked.

Reuse is symlink-only from **`20260828_ckptgrid-bear_160m`** (bear) and
**`20260805_ckptladder_160m`** (cat): 48 teacher + 38 control assembly links
inside the new tree, plus 2 path-preserving cold-1 redirects. Staging
prefetches 21 Hub refs; the one new snapshot is
`EleutherAI/pythia-160m-seed2@step512`. Arm E/F backfills append `test_sample`
and `register` / `register_sample` onto the bear reference summaries
(additive; never `--force`).

Persisted results (`{run_set}/results/`): `teacher_dose.csv`, `dose_response.csv`,
`hp_grid_delta.csv`, `hp_controls.csv`, `hp_collapse.csv`, `rect_delta.csv`,
`ridge_gain.csv`, `penalty_map.csv`, `emissivity_fit.csv`, `onset.csv`,
`metric_concordance.csv`, `cat_grid_delta.csv`, `base_ppl.csv`,
`quarantine.csv`, `coverage.json`, `claims.json` (H1–H12), plus
`robustness/*.csv` per non-primary prompt set. Every delta frame carries
`animal`, `gkd_tag`, `teacher_param_tag`, and `source_run_set`.

`ckptgrid2_plan.BASES` is the identity table for every base the campaign
**reads**, not only the ones it trains: `seed{1,3}` and `data-seed{2,3}` are
reference-only rows the 20260828 joint arm and the 20260805 cat ladder carry,
and `cell_of` must label them. The builders that take cell or base means
without an animal key (`build_dose_response`, `build_emissivity_fit`,
`build_triangle`) are fed `_bear_delta(delta)`; the cat ladder reaches
`ridge_cat` / `cat_grid_delta.csv` through `_cat_delta` only. That keeps the
ladder's pre-audit `weight-seed` cells — which `cell_of` labels `DI_SS` from
the model card that `docs/POLYPYTHIA_160M_SEED_AUDIT.md` §E3 disproves — out
of H7 and H10, so H10 stays `insufficient_data` until wave 2 trains real
weight-seed bears.

User entry: `uv run python -m scripts.stage_ckptgrid2_assets --dry-run` then
`bash scripts/launch_ckptgrid2_campaign.sh` (tmux session `ckptgrid2`;
`ckptgrid2_plan.gpu_policy_json()`; workers `2×8×cap_per_gpu`). Driver:
`scripts/run_ckptgrid2_campaign.py` `{status,advance,watch}`. Pause / resume:
`scripts.campaign_control {pause,resume,status} --campaign ckptgrid2`. Storage:
path-preserving symlinks from
`{output_root}/{sdf_teachers,gkd_students}/20260908_ckptgrid2-bear_160m`
to cold-1. One `watch` loop advances the DAG; queue completion tolerates
bounded holes (`MAX_FAILED_FRAC`) and finalize is gated on disk truth via
`ckptgrid2_expected`. A deadline window writes `coverage.json` and **no**
`CAMPAIGN_DONE`; the deadline pause touches the per-host stop sentinel only
when `pool_scope == "campaign"`. Implementation lands uncommitted; the user
commits before launch; `advance` (including `--dry-run`) refuses until
`configs/experiment/ckptgrid2_student.yaml` exists inside the pinned
worktree. Runs on **soar-subliminal-0** (verified idle 2026-09-07); no
goncalo migration.

Analysis: **`notebooks/gkd_ckpt_grid2_bear.ipynb`**, generated by
`scripts/build_ckptgrid2_notebook.py` (`nbformat` builder — regenerate rather
than hand-editing the `.ipynb`). Read-only on `results/` + `logs/`; filenames
from `ckptgrid2_plan.RESULTS_*`. Wave 2 (`--wave 2`) stays empty until
`logs/seed_audit.json` reports `passed`.

### Cross-size campaign (xsize)

`src/subliminal_learning/campaign/xsize_{plan,pairs,selection,metrics,config_gen,jobs}.py`
+ `pipelines/xsize_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage}_xsize_*` implement run set
**`20260907_xsize-bear_mixed`** (pre-registration: **`docs/PREREG_XSIZE.md`**;
plan: `.claude/plans/cross-size_polypythia_campaign_9c4e1f27.plan.md`). Bear-only
on-policy GKD on the three seed-0 PolyPythia sizes at `@step143000`
(`pythia-{70m,160m,410m}`). Size-aware base tokens use `spec_label` with a size
prefix for non-160m specs so teacher tags and control paths cannot collide;
160m labels stay byte-identical. Teacher-base (`xbase`) controls are a third
leaf class (`control_kind` = `base_teacher` | `xbase_teacher`) beside student-base
controls. W&B uses dedicated project `SL_cross_size_bear` with headline-only
log-few (`xsize_plan` predicates: 160m teacher tfs0; students `70m→160m` and
`410m→160m` at `lr*(160m)`, tfs0, gseed0).

| Piece | Role |
|---|---|
| `xsize_plan.py` | Frozen sizes, LR ladder, queues, footprints, `SPEC`, artifact names |
| `xsize_pairs.py` | 3×3 pairs; 27 P2 + 72 P3 + 18 P3b treatments; sb + xbase controls |
| `xsize_config_gen.py` | One-pair student YAMLs; per-size batch/accum/resume; fallback tfs1/2 |
| `xsize_metrics.py` | Pure X0–X4 frame builders (`Δ_sb` / `Δ_tb`, penalty, additive fit) |
| `xsize_selection.py` | Teacher admission + fallback-tag gate; per-size `pick_lr` |
| `xsize_jobs.py` | jobq shards; prefetch 3 Hub refs; cold-1 symlink plan |
| `stage_xsize_assets.py` | S0: G-a/G-c, offline loads, dataset/disk gates, cold-1 symlinks |
| `xsize_campaign_pipeline.py` | `S0 → P1 → TGATE → P2 → per-size P3/P3b → P5` |
| `xsize_results_pipeline.py` | Result CSVs + `claims.json` (X0–X4) |

Per-size queues are submitted once with a frozen jobkey set (`q_p3_*` /
`q_p3b_*`). Both gates write then act: `teacher_gate_report.json` before
fallback or P2, `screen_picks.json[size]` (never recomputed) before that size's
P3. The gate's per-size tag is read back from that report into every student
YAML (`teacher_param_tags.lora`); a voided size teaches no P3/P3b cell but keeps
its anchor ladder (frozen tag) and all `·→S` cells, and `watch` halts only when
every size is void. Fallback tfs 1/2 run on `xsize-p1-fallback{,-evals}` and are
re-gated (≥ 2 admitted at the fallback tag). P3b is optional for `CAMPAIGN_DONE`:
`plan.p3b_queue_names()` is excluded from the finalize gate and from
`_p3_leaves_ready`, so an overrun self-truncates from the tail; `finalize` records
`p3b_incomplete` and `results` re-folds late P3b leaves into X3. Every generated
YAML carries `run_set_model_size: mixed` so the per-size `model_size` stays a grid
axis instead of a per-launch `resolved_config` conflict. Driver:
`scripts/run_xsize_campaign.py`
`{status,advance,results,watch}`. User entry: `uv run python -m scripts.stage_xsize_assets`
then `bash scripts/launch_xsize_campaign.sh` (tmux session `xsize`;
`xsize_plan.gpu_policy_json()`; workers `2×8×cap_per_gpu`). Storage:
path-preserving symlinks from
`{output_root}/{sdf_teachers,gkd_students}/20260907_xsize-bear_mixed` to cold-1.

Analysis: `notebooks/gkd_xsize_bear.ipynb`. §0–§14 score X0–X4 from `results/`. §15 is
exploratory and not pre-registered: it replays Δ through training from each leaf's
`training_metrics.jsonl` and `trainer_log_history.jsonl` by running `xsize_metrics.leaf_deltas`
and `cell_means` unchanged at every probe step. It covers p(target), rank, forced-choice share
and rollout mention rate, alongside reverse KL and probe perplexity. Step-0 Δ is asserted to be
0, and the step-625 probe is checked against the eval Δ.

### Off-policy cross-size campaign (offxsize)

`src/subliminal_learning/campaign/offxsize_{plan,layout,pairs,numbers,selection,expected,metrics,jobs}.py`
+ `pipelines/offxsize_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage}_offxsize_*` + `scripts/offxsize_numbers.py` +
`scripts/sweeps/sweep_offxsize.py` implement run set
**`20260909_offxsize-dog_mixed`** (pre-registration: **`docs/PREREG_OFFXSIZE.md`**;
plan: `.claude/plans/offpolicy_cross-size_campaign_32bc541d.plan.md`).

Dog-only **off-policy** (Cloud/Angela template-teacher → numbers → completion-only
student SFT) transfer across `pythia-{70m,160m,410m}`, including cross-size pairs.
Angela's chain is **imported, never edited**, and her artifacts are **copied**
(sha-verified, never moved or referenced in place) into the campaign's own root
`{output_root}/offpolicy_xsize/{run_set}`, which is a symlink to cold-1. Passing that
root as `output_root` into her layout helpers puts her `teachers/`, `numbers/` and
`students/` trees inside the campaign, disjoint from the repo's other off-policy runs.

| Piece | Role |
|---|---|
| `offxsize_plan.py` | Frozen sizes, teacher dose ladder, student star screen, thresholds, footprints, queues, W&B log-few, `SPEC` |
| `offxsize_layout.py` | The campaign's single number/teacher/student path derivation |
| `offxsize_pairs.py` | Cell enumeration: 52 screen + 117 headline + 12 teacher-dose + 36 robustness + 6 concordance |
| `offxsize_numbers.py` | Pure pool-readiness contract and the probe → `n_prompts` rule |
| `offxsize_selection.py` | Teacher dose gate + student recipe pick |
| `offxsize_expected.py` | On-disk completeness (manifests, evals, capability parents) |
| `offxsize_jobs.py` | Footprint-bound jobq jobs; refuses commands naming a foreign tree |
| `offxsize_campaign_pipeline.py` | `S0 → P1a → TGATE → P1b → NUMBERS (waits) → P2 → per-size pick → P3/P3c/P3b → P3d + P6 → P5` |
| `offxsize_results_pipeline.py` | Result CSVs + `claims.json` (O0–O5) |

`offxsize_metrics.additive_fit` re-keys and sign-flips the contrasts it takes from
`analysis.decomposition.reciprocal_asymmetries`: that helper orders an unordered pair
lexicographically, and for size tokens `160m` < `410m` < `70m` reverses two of the three
pairs relative to the ladder. Every ι the campaign reports is small→large minus
large→small, keyed `{small}|{large}`, so `offxsize_additive_fit.csv` and
`offxsize_asymmetry.csv` agree in sign.

Two decisions are gated at runtime and persisted once: the **teacher dose per size**
(over-trained template teachers stop emitting numbers — ρ(off_topic, keep rate) = −0.705
in Angela's tree — so each rung is screened with a 2k-prompt format probe before any real
pool is generated) and the **student recipe per size** (a star screen over dose × LR ×
LoRA shape around her `ep10_lr1e-4_r8a32`).

Every capability queue is gated by a **warm of the base-metric cache it will read**
(`base_metrics.json` is shared by a whole `{lineage}/{dataset_tag}` tree and merged
read-modify-write, so parallel cells would each write a partial view). The warm has to
ask for the *same stages* as the cells it gates: `q_p4_warm` fills forced choice and
perplexity, and the lm-eval backfill therefore needs its own `q_p6_warm` passing
`--benchmarks`, or every P6 cell aborts on `base_metrics_ready` before loading a model.

**P3d (concordance, 6 cells)** exists because O5 reads one cell — Angela's teacher rung
`ep20_lr1e-4_r16a32` *and* her student recipe `ep10_lr1e-4_r8a32`. The star screen varies
the recipe at the picked rung and P3c varies the rung at the picked recipe, so that pair
of tags is on disk only when the gate happens to pick both; `pairs.concordance_leaves`
returns nothing in that case and the arm costs nothing. Its leaves carry their own phase
so `recipe_stability` does not average a teacher-dose difference into a recipe row.

The campaign is autonomous **except** for number generation: `numbers_ready` waits on
on-disk pools, reports what is missing, and never raises, so `watch` resumes by itself
once the operator finishes. User entry:
`uv run python -m scripts.stage_offxsize_assets`, then
`bash scripts/launch_offxsize_campaign.sh` (tmux `offxsize`), then
`uv run python -m scripts.offxsize_numbers plan` when the driver asks. W&B project
`offpolicy_cross_size_SL` carries the 21 teacher ladder rungs and the 21 headline student
cells only.

### Template-teacher OPD comparison (tmplopd)

`src/subliminal_learning/campaign/tmplopd_{plan,jobs,metrics}.py` +
`pipelines/tmplopd_{campaign,results}_pipeline.py` + `scripts/{stage_tmplopd_assets,
run_tmplopd_campaign,build_tmplopd_results}.py` + `scripts/launch_tmplopd_campaign.sh`
implement run set **`20260925_tmplopd-lion_160m`** (pre-registration:
**`docs/PREREG_TMPLOPD.md`**; plan: `.claude/plans/tmplopd_template_teacher_opd.plan.md`).

The ws160m headline OPD cell (`pythia-160m@step143000` self-pair, lion,
`lmb1.0_b1.0_lora_lr4e-4_e1_r8a16`, legacy screen, shared base-teacher control, ws160m
eval block) is re-run with **only the frozen teacher changed**:
- our SDF lion teachers (`sdf-ep5_lr5e-5_r128a128`, ws160m2-matrix, re-run at the same
  pin as a C0 reproduction);
- Angela's template (Cloud ft-teacher) lion teachers from offxsize2
  (`tmpl-ep{10,15,20}_lr1e-4_r16a32`, `tmpl-ep2_lr5e-4_r32a64`).

Both teacher families are copied (sha-verified) into an all-ssd-1 campaign root
(`/mnt/ssd-1/soar-subliminal/vishesh/tmplopd`). Their dir is
`sdf_teachers/<run set>/`, because the GKD sweep derives the teacher subdir from the run
set; the `tmpl` workflow token in the teacher tag marks the template teachers. S0 asserts
that `pythia-160m` `main` (the template teachers' base) is tensor-identical to
`step143000`.

Tiers, in priority order:

| Tier | Content |
|---|---|
| C | core: S, A15 at 3 tfs × 3 gseeds; A10 (tfs1 only) × 6; controls × 6 |
| R | `_vmaskf` leakage check |
| D | template dose |
| X | cross-init into `data-seed1` / `seed1`, submitted with `depends_on` = every core eval queue |

There is no `_strict` arm: `strict_v1` has no lion lexicon. There is no driver stage
machine: queues are chained with jobq `depends_on`, and `watch` only requeues under the
cap and drops `CORE_EVALS_DONE` / `CAMPAIGN_DONE`. Queues are per-host partitions on four
nodes, with 2 students + 1 eval per GPU (separate cap groups).
`scripts.build_tmplopd_results` writes `results/` (leaves, arm summary with a two-stage
bootstrap, contrasts, teachers, efficiency, vmaskf retention, leakage, trajectories,
ws160m2 concordance, `claims.json`).

### Top-K cross-size OPD campaign (topk)

`src/subliminal_learning/campaign/topk_{plan,pairs,config_gen,jobs,metrics}.py`
+ `pipelines/topk_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage}_topk_*` implement run set
**`20260910_topk-bear_mixed`** (pre-registration: **`docs/PREREG_TOPK.md`**;
plan: `.claude/plans/topk_cross_size_opd_59b62117.plan.md`). It re-runs the
xsize headline cell `160m→410m` (and the `410m→410m` / `160m→160m` anchors)
with the reverse-KL objective truncated to the teacher's top-K columns per
position (`GkdConfig.top_k`, tag suffix `_top{K}`; `_top{K}g` = ghost-token
residual). Sibling of xsize: no constant shared; teachers and `lr* = 4e-4` are
**inherited** from `20260907_xsize-bear_mixed` and asserted at S0 (frozen
teacher gate, `screen_picks`), never recomputed. Full-vocab anchors and
student-base controls are reused from the xsize tree
(`topk_pairs.reused_xsize_docs`); the headline treatment + its teacher-base
control are re-run at full vocab under the new pin as a concordance check.

| Piece | Role |
|---|---|
| `topk_plan.py` | `Rung` (K ladder), `Block`s (core `k100,k10,k5,k1000,full`; tail `k30,x70m,k5g,k2`), footprints (measured xsize medians), per-block/size queues + priorities, `DEADLINE_HOURS`, W&B log-few (11 cells), `SPEC` |
| `topk_pairs.py` | 105 treatment leaves (63 core + 42 tail); K-matched sb/tb controls; full-vocab sb never re-emitted |
| `topk_config_gen.py` | One-pair student YAMLs with top-level `top_k` / `top_k_residual`; `cat` null-teacher YAMLs (EAS teacher-side null) |
| `topk_jobs.py` | jobq shards (`--no-control` / `--controls-only` / `--xbase-controls-only`), null-teacher shards, per-block/size `scripts.compute_gkd_eas` jobs over `logs/eas_lists/*.json`, reused-leaf paths |
| `topk_metrics.py` | Pure T0–T4 / M1–M3 frames: K-matched `leaf_deltas` (join carries `rung` + `source`), `k_curve` + `breakpoint` (K\*), `rho_curve` (ρ_S(K)), visibility means, signature-probe math (`head_fraction`, `coverage_mass`, `visibility_rate`), `eas_cells`, `claims` |
| `stage_topk_assets.py` | S0: reused-asset gates, offline loads, cold-1 symlinks, corpus/disk checks, `--signature` A2 probe (teacher−base logit-shift energy inside the teacher top-K on frozen xsize rollouts) |
| `topk_campaign_pipeline.py` | `preflight → submit_all → finalize_core → results_{tail} / deadline_pause` |
| `topk_results_pipeline.py` | 12 CSVs + `claims.json` (T0–T4, M1–M3) over both sources (`topk`, `xsize`) incl. per-leaf `eas/gkd_eas.json` and unit-mean-vector cosines |

Every queue is submitted once at `submit_all` with a frozen jobkey set
(priority-ordered: null teachers → core 410m rungs → full concordance → core
160m → evals → EAS → tail); evals depend on their train queues, EAS on train
queues + null teachers. `finalize_core` (results + `CAMPAIGN_DONE`) needs only
the core blocks; each tail block re-runs results when it completes; at
`DEADLINE_HOURS` (19.5 h from submission) with tail work left the driver
pauses the pool via `control_io.pause`, re-runs results and records
`tail_truncated`. Pause/resume: `scripts.campaign_control --campaign topk`.
User entry: `uv run python -m scripts.stage_topk_assets --signature` then
`bash scripts/launch_topk_campaign.sh` (tmux `topk`). Storage: path-preserving
symlinks from `{output_root}/{sdf_teachers,gkd_students}/20260910_topk-bear_mixed`
to cold-1; W&B project `tok-k_cross_size_opd`.

### OPD-limits campaign (opdlim1 + opdlim2)

`src/subliminal_learning/campaign/opdlim_{plan,pairs,config_gen,jobs,metrics}.py`,
`pipelines/opdlim_{campaign,results}_pipeline.py`, `scripts/run_opdlim_campaign.py`,
`scripts/stage_opdlim_assets.py` and `scripts/launch_opdlim_campaign.sh` implement two run
sets. Each tier owns one node, and every block runs to completion; there is no deadline.

- **Tier 1: `20260914_opdlim1-bear_mixed`** on soar-subliminal-2 (a40-cluster-3).
- **Tier 2: `20260914_opdlim2-bear_mixed`** on soar-subliminal-0 (a40-cluster-1), with the
  legacy shared lock prefix `$ROOT/locks/jobq` that soar owns (`SHARED_LOCK_OWNER`).
- Pre-registration: **`docs/PREREG_OPDLIM.md`**. Plan:
  `.claude/plans/opdlim_topk_ce_limits_2dd502c9.plan.md`.

The headline objective is Goncalo's CE top-K (`top_k_ce`, Anshumann §2 vanilla Top-K
forward KL, β=0). It re-measures the transfer bandwidth that the renormalised reverse-KL
topk run confounded with collapse.

- **Tier 1:**
  - CE K ladder {full, 1000, 100, 30, 10, 5, 2, 1} on P1 `160m→410m`, P2 `410m→410m`,
    P3 `160m→160m` and P4 `160m@8000→160m@8000`.
  - The `410m→160m` direction arm.
  - The P4 HP star plus teacher dose (ep2/ep8, reusing the ladder's controls).
  - LM-Eval on K ∈ {full, 10, 1}.
  - Totals: 414 cells, about 20.3 h.
- **Tier 2:**
  - Ghost token at β=0.
  - Surface-form vocab mask `_vmaskf`.
  - λ=0 on screened real Pile rows.
  - The fixed-625-step unique-prompt ladder N ∈ {625 … 20000} (`_s625_n{N}`).
  - The P1 HP star (`hp-p1-*`), moved here because the smoke measured λ=0 at about
    0.13× the λ=1 cost.
  - Totals: 453 cells, about 21.7 h.
  - Its X0 cross-tier claims read Tier 1 leaves read-only, gated on the same pin.
- **Teachers:** reused through a per-tier teacher assembly. These are create-only
  symlinks under `sdf_teachers/{run_set_id}`, recording weights and manifest sha256.
  Sources: xsize @143000, ckptgrid ep5 @8000, ckptgrid2 ep2/ep8. Each source's gate is
  re-checked at staging.

| Piece | Role |
|---|---|
| `opdlim_plan.py` | `Objective` (form / K / λ / vmask / subset), `Hp`, `PairDef` (full model tokens, so P3≠P4), `LeafDesign`, `Block`, `Tier` (home node, lock prefix), per-block/class/size queues (controls = students+1), footprints and budget (`tier_hours` ≤ 24 validated at import), `SPEC_T1`/`SPEC_T2` (registered as `opdlim1`/`opdlim2`) |
| `opdlim_pairs.py` | Treatment leaves, matched `sb`/`tb` controls (dose blocks reuse), teacher sources |
| `opdlim_config_gen.py` | One train YAML per (block, objective, pair, class ∈ students/sb/tb); verify YAMLs with `eval.calibration`; bench YAMLs |
| `opdlim_jobs.py` | jobq shards, leaf dirs, prefetch refs, assembly path math |
| `opdlim_metrics.py` / `opdlim_results_pipeline.py` | Per-seed Δp and Δfc against objective-matched controls; T0; claims C0–C7 (Tier 1) and G1/V1/L1/N0–N2/X0 (Tier 2) → `results/*.csv` + `claims.json` |
| `stage_opdlim_assets.py` | S0 per tier (gates, assembly, pool identity, nested-subset audit, off-policy screen audit, offline snapshots, cold-1 links, disk, host guard); `--smoke-prepare` / `--smoke-verify` |
| `opdlim_campaign_pipeline.py` | `preflight → submit_all → finalize_core → results_{tail,bench} → cross_tier` (Tier 2) |

User entry, from the pushed pin on the tier's node:
1. `uv run python -m scripts.stage_opdlim_assets --tier N`
2. `bash scripts/launch_opdlim_campaign.sh N`. This starts tmux `opdlimN`, refuses a wrong
   host or lock prefix, and runs the driver `watch --tier N`.

W&B project `subliminal-learning-opdlim`, log-few (tfs0 gseed0 treatments).

Analysis notebook: `notebooks/gkd_opdlim_topk_ce_limits_bear.ipynb`, generated by
`uv run python -m scripts.build_opdlim_notebook` and executed with `jupyter nbconvert --execute
--inplace`. It reads both tiers' `results/` + `claims.json`, the in-training probes, the teacher
preference summaries and `20260910_topk-bear_mixed` results. It never writes. The post-plot
two-liners live in the builder's `TWO_LINERS` and must be re-checked against the rendered figures
whenever the artifacts change. Known issue: the results CSVs' `teacher_p_target` holds the
160M@step8000 teacher values for every pair, so it is only correct for P4. The notebook reads
teacher p(bear) from the teacher summaries instead.

**Top-K token census (2026-09-24, post hoc, read-only on the tiers).** `gkd/topk_census.py`
(torch domain), `pipelines/topk_census_pipeline.py` and `scripts/run_topk_census.py` replay
each Figure-2 leaf's teacher (K ∈ {10, 1} on P1 and P4; Tier 1 `ce-k*` standard, Tier 2
`vmaskf-k*` masked; 72 leaves) over the rollouts that leaf trained on and count, per vocab
id, membership in the teacher's top-10 and top-1 under both views: the full 50,304 columns
and the 50,261 left after the 43 animal-form columns are dropped, in the loss's own order
(`banned_vocab_mask` → column selection → `topk_select`). Positions are the loss's
(`logits[P-1 : P-1+L]` per row, unflagged on-policy rows, end token excluded); rows whose
prompt or completion re-tokenizes to another length than the stored counts are dropped and
counted (> 25 % is an error). The teacher forward runs in the precision the leaf's saved
`training_args.bin` records (bf16 autocast for every opdlim cell, as accelerate wrapped the
prepared teacher): an fp32 replay of the 160M@143000 teacher overstates its bear-in-top-10
rate about 4× and its top-K mass by 0.02–0.06, while the 160M@8000 teacher is insensitive,
so a mismatched precision fails the gate below. Outputs live in a new run set
`gkd_students/20260924_opdlim-census-bear_mixed/{census,results,logs}`: per-leaf `.npz`
count vectors + `.json` tallies and per-step coverage / bear-rate records, then
`results/topk_census_{cells,tokens,animal_tokens,entrants,validation,steps}.csv` +
`census_manifest.json`. `aggregate` refuses an incomplete leaf and raises unless every
leaf's replayed coverage (last 10 logged intervals, bucketed like the trainer) is within
0.03 of its logged `topk_coverage` and, on standard leaves, its bear-in-top-K rate within
0.002 or 2× of the logged `target_in_topk_rate`. Figures 6–9 of
`notebooks/headline_notebooks/topk_headline_figures.ipynb` (top-20 tokens under both losses,
the 43 removed tokens before removal, the entrants that replace them, replay validation +
bear visibility over training) read only these CSVs.
**Teacher − untuned base (2026-09-26).** `run_topk_census replay/aggregate --subject base`
replays the teacher's untuned base (`manifest["teacher"]["base_model"]`, `load_model`, no
adapter) over the same rows, positions, views and precision into `census_base/`, and
`aggregate_base` writes `results/topk_census_base_{tokens,cells,validation}.csv` +
`census_base_manifest.json` (per token `share_teacher`, `share_base`, `delta`, `ratio`,
per-leaf delta range, `rank_increase`; top `BASE_TOP_N_TOKENS` = 2000 per model plus the 43
mask ids). It raises unless each base leaf's rows, positions, mask ids and precision equal
the teacher leaf's. Notebook Fig 15 (P3, paper `topk_self_census_base_delta`); the P1/P4
base CSVs are on disk but not plotted.

**Self-pair top-K (2026-09-25).** Tier 4 (`opdlim4`, `20260925_opdlim4-bear_mixed`, soar)
adds the animal-column-masked arm on P3 (160m→160m at step 143,000; `vmaskf-k10`,
`vmaskf-k1`, Tier 1's P3 design, 12 cells); `FINISHED_TIERS` lets a later tier reuse a
finished tier's home node. The census takes `--pairs P3` into its own run set
`20260925_opdlim-census-p3-bear_mixed` (`PAIR_RUN_SET`; one run set per pair group;
`PAIR_K_CELLS` keeps P3 at K = 10, since its K = 1 students are quarantined and fail the
re-tokenization gate).
The headline notebook's last section draws the self-pair-only paper figures
(`topk_self_transfer_and_mask`, `topk_self_census_{top_tokens,removed_tokens,entrants}`),
reusing the Fig 6–8 plotting functions.

### Strict rollout-screen campaign (strictscreen)

One-off rerun of the breadth4 Stage B headline grid under the opt-in strict Layer-2
screen, with the Qwen3-32B-AWQ judge audit chained automatically after training. Run set
**`20260915_strictscreen_160m`** on soar-subliminal-3 (a40-cluster-4, node-local lock
prefix `/tmp/subliminal_jobq_vishesh`), outputs on cold-1 through a path-preserving
symlink; audit **`20260915_strictscreen_stageb`** under `audit_root`. Pre-registration:
**`docs/PREREG_STRICTSCREEN.md`**. Plan: `.claude/plans/strictscreen_rollout_judge_audit.plan.md`.
Every other run set keeps the legacy screen; `rollout_screen` defaults to `legacy` and the
legacy tags, identities, JSONL/summary keys and metrics are byte-identical.

- **Screen:** `gkd/filtering.StrictRolloutScreen` over `data/gkd_rollout_screen_strict_v1.json`
  (rules `legacy` → `target` → `fused` → `all_animals` → `generic`; see
  `docs/GKD_ON_POLICY_PIPELINE.md` §4.2). Tag suffix `_strict`; identity key
  `rollout_screen_vocab_sha256`.
- **Arms** (`campaign/strictscreen_plan.py`): A strict treatment 288 (8 headline pairings ×
  tfs {0,1,2} × gseed {0,1,2} × 4 animals), B strict base-teacher controls 72, C legacy
  rerun 32 (tfs0/gseed0; code-drift check). Queues `strictscreen-{b-controls,a-tfs0..2,
  c-legacy}` + per-unit eval queues; cap 3/GPU, `mem_mib` 15000. Teachers are read-only
  from `20260809_breadth4-teachers_160m`.
- **Modules:** `campaign/strictscreen_{plan,config_gen,jobs,metrics}.py`,
  `pipelines/strictscreen_{campaign,results}_pipeline.py`,
  `scripts/{run,stage}_strictscreen_{campaign,assets}.py`,
  `scripts/launch_strictscreen_campaign.sh`; judge side
  `campaign/strictscreen_audit_plan.py`, `campaign/judge_audit_io.py`,
  `gkd/judge_claims.py`, `pipelines/strictscreen_audit_{driver,results_pipeline}.py`,
  `configs/{rollout_judge_strictscreen.yaml,experiment/filtration_audit_strictscreen.yaml}`.
- **Phases** (one driver, restart loop stops on exit 0/2): `preflight` (staging + smoke
  reports, teachers, recipe, vocab shas, arm-C identity == breadth4, host census, judge
  preflight) → `submit_train` → `submit_evals_{unit}` (verify triad, no lm-eval) →
  `finalize_training` (screen-provenance gate on every leaf, shortfall tolerance A ≤ 5 / B 0)
  → `sl_results` (C0/H2/H3) → `audit_completeness` → `audit_plan` (CPU, fingerprinted,
  pool shards byte-identical to audit 1) → `audit_release_gpus` (stop sentinel + drain +
  `WORKER_LOOP_EXITED` + idle VRAM → `GPUS_RELEASED`) → `audit_probe` (parse/truncation/
  throughput + pool agreement vs audit 1) → `audit_judge` (`judge-cluster`, 8 × tp=1,
  completion by row_id coverage, re-attach via lock) → `audit_return_gpus` →
  `audit_results` (H1/H4/S1/S2) → `CAMPAIGN_DONE` + `AUDIT_DONE`.
- **Outputs for analysis:** per-leaf `rollouts_sample.jsonl` with `matched_rule` /
  `rules_fired`; `{run_set}/results/` (`sl_claims.json`, per-leaf/arm CSVs,
  `screen_removal_by_rule.csv`, `screen_trajectory.csv`); `{run_set}/logs/`
  (`coverage.json`, `screen_stats.csv`, `shortfall_report.json`); `{audit}/results/`
  (`claims.json`, H1/decoy/recall/over-rejection/pool-rejudge tables) and
  `{audit}/results/annotation/` (`human_agreement_sheet{,_blind}.csv`,
  `human_agreement_key.csv`, `judge_yes_rollouts_clean.jsonl`; κ fills in once
  `human_agreement_labels.csv` exists).
- **Shared judge code** (additive; `tests/unit/test_judge_plan_audit1_golden.py` pins
  audit-1 plan bytes): `judge_plan_pipeline` gains multi-tag selection, per-arm quotas,
  `omit_tasks`, exact expected-leaf gate, keep-last dedupe with exact census, rule|form
  flagged strata, leaf-cache and manifest fingerprints (`PlanFrozenError` instead of
  overwriting a committed plan); `analysis/stats.py` gains
  `stratified_cluster_bootstrap_draws` / `summarize_draws` / `bootstrap_equivalence`.
- **Launch:** stage (`stage_strictscreen_assets`), GPU smoke (`--smoke DIR`, run the printed
  commands, `--smoke-verify DIR`), then `bash scripts/launch_strictscreen_campaign.sh <sha>`
  from `.code_pins/<sha>` on node 4 (syncs and checks the `tools/rollout_judge` venv).
- **Analysis notebook:** `notebooks/gkd_strictscreen_filtration.ipynb` (52 cells,
  12 figures), generated by `uv run python scripts/build_strictscreen_notebook.py` and
  executed with `jupyter nbconvert --to notebook --execute --inplace` — regenerate from the
  builder rather than hand-editing cells. Read-only over both `results/` trees; no judge
  shards are streamed. §1 is a self-contained paper-figure section for the OPD appendix's
  filtering subsection (removal by first-firing guard; residual leakage legacy vs strict vs
  control floor with the decoy false-positive line; residual by training-step quintile),
  exported as PNG + PDF to `notebooks/exports/gkd_strictscreen/` when `SAVE_FIGS` is on.
  §2–§10 cover guard overlap, removal across training, the SL ratio forest (with the eagle
  slice the pooled ratio hides), judge validity and over-rejection, cross-audit
  reproducibility, judge modes, the claim board and the pending-κ sheet. The notebook
  states the filtering claim as **equivalence at the pre-registered ±0.25 pp tolerance**,
  not absence of leakage: the strict residual (0.286 %) exceeds the control floor
  (0.240 %) by +0.046 pp with a bootstrap CI that excludes zero.

### Cross-family OPD limits campaign (xfam)

`src/subliminal_learning/campaign/xfam_{models,plan,pairs,selection,metrics,config_gen,jobs}.py`
+ `pipelines/xfam_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage}_xfam_*` implement run set
**`20260912_xfam-bear_mixed`** (pre-registration: **`docs/PREREG_XFAM.md`**; plan:
`.claude/plans/cross_family_opd_limits_xfam.plan.md`; weight audit:
`docs/XFAM_WEIGHT_INIT_AUDIT.md`). Bear-only on-policy GKD between models from
different pretraining runs, architectures and post-training stages.

**Eligibility is the tokenizer, not the family.** The trainer feeds the student's
sampled ids straight to the teacher, so a pair is eligible iff both assign identical
ids to identical strings. `xfam_models` holds the roster as the SSOT — repo id, pinned
commit, expected `model_type`/`vocab_size`/tokenizer-identity hash, LoRA targets,
lineage — split into two clusters that are never crossed: `neox` (Pythia-410M and its
deduped, HH-SFT, HH-DPO and PPO relatives, plus AMD-OLMo-1B and its SFT/DPO variants,
shared prefix 50 277) and `qwen` (Qwen2.5-0.5B with Instruct and Coder, Qwen3-0.6B-Base
with Qwen3-0.6B, shared prefix 151 665).

**How the pipeline was extended without changing the hot path.** Two seams, both
default-preserving:

1. *Spec-typed loaders.* `models/hf_spec.py` adds `HfModelSpec` (an arbitrary Hub repo
   plus the facts staging asserts). `resolve_hf_repo_id`, `model_dir_name`, `spec_label`
   and `load_model` / `load_finetuned_teacher` short-circuit on
   `getattr(spec, "repo_id"/"label", None)`, which is `None` for every `PythiaModelSpec`,
   so the PolyPythias path is byte-identical (locked by `tests/unit/test_hf_spec.py`).
   `parse_base_models(..., registry=)` resolves roster labels; the four sweeps expose it
   as `--spec-registry xfam`.
2. *Shared-vocab slice.* `GkdConfig.shared_vocab` (default `None`) makes
   `SubliminalGKDTrainer` truncate both logit tensors to the leading ids the two
   tokenizers agree on, before the vocab mask and every monitor, and suppress the
   student's out-of-prefix ids during generation so a rollout can never carry a token
   the teacher cannot score. `None` leaves the loss path untouched and keeps the
   equal-width assert.

| Piece | Role |
|---|---|
| `xfam_models.py` | Roster SSOT: clusters, pinned commits, shared prefixes, LoRA targets, lineage |
| `xfam_plan.py` | Arms, LR ladders, rank/epoch recipes, gates, footprints, queues, `SPEC` |
| `xfam_pairs.py` | Leaf/control enumeration per `(arm, student)` unit, deterministic given picks |
| `xfam_config_gen.py` | Per-unit sweep YAMLs; per-cluster pool and `shared_vocab`; per-label packing |
| `xfam_jobs.py` | One shard per (LR, ftseed); per-label `mem_mib`/`slots`; prefetch + cold symlinks |
| `xfam_selection.py` | Teacher admission, per-student LR pick (PPL guard 4.0), replication gate |
| `xfam_metrics.py` | Pure F0–F7 frames (Δ_sb / Δ_tb, penalty, init ladder, asymmetry, HP, PPL) |

**Gates are persisted before they are acted on** and read back on resume, never
recomputed: an LR is part of the student's GKD tag, so a re-derived pick would change a
submitted queue's jobkey set. The driver runs preflight → teachers → teacher gate →
ladders → per-student pick → derived units → results, pauses the pool at
`DEADLINE_HOURS = 84`, and needs only the core arms for `CAMPAIGN_DONE`.

**Packing.** No student gradient-checkpoints: TRL 0.19.1 builds the rollout
`GenerationConfig` with `use_cache=False` whenever checkpointing is on, which the
staging smoke measured at 45-88 s/step against 6.6 without it. `cap_per_gpu = 2` is the
OOM guarantee, not the memory gate: `compute_loss` calls `empty_cache()` every step, so
free memory dips and a gate reading nvidia-smi can admit a cell into the trough. The
5-step smoke undersampled steady state (rollouts only reach the 128-token cap once
training is under way, and the two vocabulary-wide logit tensors scale with it), so its
13.4-19.4 GB stamps were re-measured from the live run at **21.5 GB worst case** on a
47.5 GB card -- three such cells cannot coexist, which cost 42 of the first 68 shards to
CUDA OOM. Every student label now declares that one measured worst case, and
`validate_footprints` refuses any `cap_per_gpu` x heaviest-footprint product that does
not fit a card. Mamba is in the roster but has no arm: without the `mamba-ssm`
kernels its sequential-scan fallback measured 307 s/step, 47x the anchor.

**Scope corrections made while the run was live.** `t2-olmo-sft` keeps its `t`-prefixed
name but is a **core** arm at `a6-olmo-posttrain`'s priority: with the attention-free
rung deferred, `amd-olmo-1b`'s base -> SFT -> SFT-DPO ladder is the only post-training
contrast F3 has outside the Pythia lineage, so it counts toward `CAMPAIGN_DONE`. Both
clusters share one prompt pool, `pile_prompts_v1`: pool rows store raw text and each
student re-tokenizes them (staging asserts every id is below its cluster's
`shared_vocab`), so a Qwen cell sees NeoX-cut windows re-tokenized, not Qwen-native
ones. Measured per-leaf cost at `cap_per_gpu = 2` is 1.43 / 3.04 / 4.10 card-h for
NeoX-410m / OLMo-1B / Qwen, 2.1-3.4x the pre-registered estimates.

User entry: `uv run python -m scripts.stage_xfam_assets --download` (fetches every
pinned snapshot into a cold-1 Hub cache and writes the gitignored
`configs/paths/paths_xfam.yaml`), then `--smoke` to stamp footprints, then
`bash scripts/launch_xfam_campaign.sh` (tmux `xfam`). Storage: path-preserving symlinks
from `{output_root}/{sdf_teachers,gkd_students}/20260912_xfam-bear_mixed` to cold-1;
W&B project `cross-model-opd-limit-testing`.

**Completion (2026-09-15; `docs/PREREG_XFAM.md` §9).** The core drained by 2026-09-14,
but `finalize_core` raised a matched-control `KeyError`. Two a2-deduped gseed1 leaves
finished training after their cell was evaluated, and the eval sweep's `_cell_done`
kept skipping the partial summaries.

- **Fix.** Branch `vishesh/xfam-evalfix` (`1068bd2`, from `ae12a58`; lock = the one
  the cells ran with) re-opens a cell whose `ft_seeds` omits a complete `gseed{N}`. The
  per-seed eval cache then scores only the missing seed.
- **Completion steps.**
  1. `reconcile_run_set` re-pinned the run set to `1068bd2`.
  2. The two seeds were scored on goncalo from its pin; records are in
     `run_logs/xfam_evalfix_20260915/`.
  3. One `advance` on soar wrote all 288 leaves, every results artifact, and
     `CAMPAIGN_DONE`.
- **Known driver quirk.** `xfam_plan.arm_queue_names` lists a controls queue for the
  t3/t4 dose arms, which reuse ladder controls and never submit one. `results_tail`
  therefore never fires, and past the deadline the driver ends in `deadline_pause`,
  which sets the **per-host** stop sentinel and pauses every pool on that node.
  Resume at once with `scripts.campaign_control resume` if another campaign shares
  the node.

Analysis notebook: `notebooks/gkd_xfam_bear.ipynb`, generated by
`uv run python -m scripts.build_xfam_notebook` and executed with `jupyter nbconvert --execute
--inplace`. It reads `results/`, `claims.json`, the persisted gates and every leaf's summaries,
probes and trainer logs by exact `gseed{N}` path (the stock GKD probe/log loaders crash on the
quarantined `*_doublewrite` siblings), and pushes the extra endpoints through
`xfam_metrics.leaf_deltas` (asserted to reproduce `xfam_delta.csv`). It needs the gitignored
`paths_xfam.yaml` for parameter counts and never writes. The post-plot two-liners live in the
builder's `TWO_LINERS` and must be re-checked against the rendered figures.

### Cross-size Pythia wave of the cross-family campaign (xfam2)

`src/subliminal_learning/campaign/xfam2_{models,plan,pairs,config_gen,jobs,metrics}.py`
+ `pipelines/xfam2_{campaign,results}_pipeline.py` + `scripts/{run,launch,stage}_xfam2_*`
implement run set **`20260924_xfam2-bear_mixed`** (pre-registration **`docs/PREREG_XFAM2.md`**,
claims Y0-Y11; plan: Part II of `.claude/plans/cross_family_opd_limits_xfam.plan.md`). It
distils a bear trait between Pythia sizes 160m-2.8b in every direction (plus 6.9b as a
teacher only), at the final `step143000` commits. It also re-runs Part I's weakest cells
at three teacher seeds: the OLMo and Qwen3 self-anchors, both reverse arms, and a
matched-shape pythia-1b vs AMD-OLMo-1B design. One frozen recipe: reverse KL, lr 4e-4 for
NeoX, Part I's lr* for Qwen, 3 teacher seeds × 2 generation seeds in the core, gseed 2 and
an LR check as tails.

- **Roster** (`xfam2_models`): Part I entries by reference plus five Pythia sizes pinned by
  commit (the Hub rewrote the step branches in Feb 2026). Every size shares one tokenizer;
  the neox shared prefix (50 277) covers 6.9b's 50 432-wide head. `--spec-registry xfam2`.
- **Storage:** every root lives on the ssd-1 campaign root
  `/mnt/ssd-1/soar-subliminal/vishesh/xfam2/`: generated `configs/paths/paths_xfam2.yaml`,
  its own HF cache, datasets copied under a sha manifest, and its own jobq root. Any of the
  five nodes can serve it, and it never touches the ssd-3 queue root.
- **Nodes:** one driver on soar; every queue is pinned to a host (`STUDENT_NODE` /
  `TEACHER_NODE` / `QUEUE_NODE_OVERRIDES`) with that host's pin defaults. Claim-free queues
  are re-homed when the placement changes (the ws160m mechanism).
- **Per-label teacher gate:** each teacher label is admitted on its own (≥ 2 of 3 ftseeds),
  so a slow 6.9b teacher never holds back the others. Controls are submitted at launch.
  Evals use strict deps.
- **Cached rollouts under gradient checkpointing** (`GkdConfig.rollout_cache`, default off
  = byte-identical). The rollout runs in `eval()` with the KV cache on: in `train()`
  GPT-NeoX drops the cache under checkpointing and cached rollouts diverge (locked by a
  GPU test). On for students ≥ 1b.
- **Packing:** slot-weighted (`cap_per_gpu = 4`). A job's `slots` = its stamped peak /
  12 000 MiB, from the 2026-09-24 smoke (`logs/xfam2_footprints.json`). 6.9b-teacher cells
  take a whole card.
- **Pause / resume:** `scripts.campaign_control pause|resume --campaign xfam2 --paths-config
  configs/paths/paths_xfam2.yaml`, run on each node. It drains in-flight leaves; every
  student resumes from its last 125-step checkpoint after a hard stop. The deadline
  (15 h) counts only time when not every host is paused.
- Also on the main line from this wave: the seed-aware eval skip (`_uncovered_seeds`,
  ported from `vishesh/xfam-evalfix`) and `driver_io.submit_queue(strict_deps=)`.

### Headline OPD over the six Pythia-160M ablation categories (ws160m)

`src/subliminal_learning/campaign/ws160m_{models,plan,pairs,config_gen,jobs,expected,metrics}.py`
+ `pipelines/ws160m_{campaign,results}_pipeline.py` +
`scripts/{run,launch,stage,smoke}_ws160m_*` + `scripts/build_ws160m_results.py`
implement the four run sets **`20260916_ws160m-{main,ladder,dosehp,extra}_160m`**
(pre-registration: **`docs/PREREG_WS160M.md`**; plan:
`.claude/plans/ws160m_headline_opd_campaign.plan.md`; model inventory:
`docs/PYTHIA_160M_ABLATION_CELLS.md`). The frozen headline OPD recipe (reverse KL,
λ=1, LoRA r8/α16, lr 4e-4, one epoch over the 20k Pile prompt pool; SDF LoRA
`ep5_lr5e-5_r128a128` teachers) is re-run on the six audited base-pair categories
SI_SS, SI_DS, DI_SS, DI_DS, SI_DC, DI_DC over the Hub 160M repos plus Goncalo's
locally trained NeoX runs (`pile-ws1`, `dclm-main`, `dclm-ws1`; `pile-ws3` at
143k as an automatic wave 2). Arms: the bear 11×11 directed matrix at 143k with
9/6/3 leaves per category, a same-rung checkpoint ladder on 39 directed cells with
its own rung teachers, off-diagonal rung heatmaps, a student-HP sweep (LoRA only)
and `xbase_teacher` controls on a matched sub-design, and epoch-dose ladders for
bear, cat, eagle and dog.

**Local NeoX bases through the existing loaders.** `models/neox_convert_io.py`
renames the fifteen ZeRO-1 pipeline layer files of a NeoX revision into a
`GPTNeoXForCausalLM` safetensors snapshot (148 tensors, strict load, fp16,
`conversion_manifest.json` with source and weight hashes) and `HfModelSpec` gains
`local_root`: `loading_io` then reads `local_root/<revision>/` and asserts the
manifest's base and step against the spec. The roster (`ws160m_models`) holds one
spec per (base, step) for Hub and local bases alike, with three names that are
never conflated: the token `<base>@step<N>` (sweep and YAML key, `--spec-registry
ws160m --paths-config configs/paths/paths_ws160m.yaml`), the label
`<base>-step<N>` (run-directory segment, so every rung of a local base has its own
tree) and the identity `EleutherAI/pythia-160m@step<N>` or `local/<base>@step<N>`
written to every manifest. `category()` and the three strata are derived from the
roster attributes; wave-3 entries (`pile-ws2`, `pile-main`) are enumerated and
refused by `assert_clean` until staged.

**Staging and verification** (`scripts/stage_ws160m_assets.py`, cold-1
`base_models_160m/`): converts the required rungs, walks `refs/step*` for every Hub
revision the plan uses, asserts one tokenizer identity hash across all entries
(the eight Hub repos ship two byte-variant `tokenizer.json` files), replicates the
audit's init-twin check bitwise through our loader (`pile-ws1@step0` ≡
`seed1@step0`, …), runs a CPU forward band check on the converted 143k models,
writes `paths_ws160m.yaml` (gitignored, in the code-pin overlay), the `hub`
symlink and the cold run-set symlinks, and records everything in
`staging_report.json`. `--wave 2` copies Goncalo's `pile-ws3` step 143000
read-only and repeats the checks.

**Partitions and ordering.** One driver, run set and node per partition
(`ws160m_plan.PARTITIONS`); teachers train once (143k on `main`, rungs on `ladder`)
and other partitions read them through `teacher_subdir`, waiting on the shared PVC
with a 2 h fallback that submits the missing teachers itself and a 4 h hard
timeout. Eval queues for an arm are submitted only after all of its treatment and
boost queues are terminal; `ws160m_expected.missing_evals` compares each summary's
`per_seed_p_target` keys with the expected gseeds and short directories are
re-evaluated with `--force` on the eval command only. Every partition finalizes at
T+24 h with a `coverage.json` (`landed_by_arm` counts deferred leaves apart;
`scripts.run_ws160m_campaign coverage` rewrites it from disk and drops healed
holes). Stages in `ws160m_plan.STAGE_NODE_OVERRIDES` overflow to another node: their
queues are pinned there with that clone's pin as `cwd`, and because a pin's
`configs/experiment/generated` is a symlink into its node-local clone, their jobs
read the sweep YAMLs by absolute path from the run set's shared
`generated_configs/` copy (`ws160m_jobs.config_path`), whose verify YAMLs inherit
the shared train YAML by absolute path (`ws160m_config_gen.shared_copy`);
submission refuses an overflow queue whose YAMLs are relative or absent there.
Part II overflow (09-22 resume): `queue_node` also resolves `GEN2_PARTITIONS` and
`lmeval-<stage>` queues; the driver's `rehome_queues` transition re-pins already-submitted,
claim-free queues to their override host on each advance (pin defaults follow the host's
clone; the code-SHA re-pin is node-aware); `launch_ws160m_campaign.sh <partition> <sha>
--overflow` starts a workers-only pool (tmux `<campaign>-overflow`, that host's lock prefix)
over `overflow_queues_on(host, campaign)`, so one pool serves both partitions' overflow.

**lm-eval tail** (`pipelines/ws160m_lmeval_pipeline.py`, `scripts/run_ws160m_lmeval.py`,
lowest-priority queue `ws160m-<partition>-lmeval`): after a partition has written
`CAMPAIGN_DONE` (and, by default, the headline `results/` exist) it submits one
benchmark job per config dir, pinned to the partition's node, reading benchmark-enabled
verify YAMLs (`ws160m_config_gen.lmeval_verify_configs`: 0-shot LAMBADA / ARC-Easy /
SciQ, batch 16) from the shared `generated_configs/` dir. The eval sweep computes only
the missing `benchmarks` block per config dir and caches each untuned base in
`base_metrics.json`; local NeoX bases are passed to lm-eval as their identity-checked
snapshot dir (`evaluate_sdf_teacher_pipeline.bench_pretrained`) and the queue env sets
`HF_HUB_CACHE` to the campaign's cold-1 cache. `leaves.csv` carries the resulting
`bench_*` / `base_*` / `delta_*` columns (config-dir means) when present.

**Results** (`pipelines/ws160m_results_pipeline.py`; `--gen 1` unions the four Part I
trees under the `main` run set's `results/`, `--gen 2` unions Part I ∪ Part II — six
trees, every animal — under the `matrix` run set's `results/` and never writes into a
Part I run set): leaves carry `source_run_set`, `arm` and `replicate` from the
generation's leaf enumeration, category, strata, teacher and student base+step,
`p_target`, the subliminal `delta` against the matched `base_teacher` control (joined
with `animal`; a cell whose control is fully quarantined leaves the `primary` set, a
missing control raises except in a `partial` build of a paused campaign, and both
are listed with a reason under `cells_without_control` in `claims.json`),
`delta_rel`, teacher tier and quarantine flags. W2 reads the six categories and W6
the HP arm's recipes even on the wider Part II union. A leaf identity present
in two run sets raises (Part II invariant 3). Category summaries, contrasts, asymmetry
and the recipe table read **same-rung** cells only; rung-cross cells feed the heat maps
alone (`heatmap_cells.csv` carries `anchor`, `direction`, `partner_base`; `--gen 2`
writes both `HM2_ANCHORS` in both directions over `HM2_RUNGS`, plus
`ladder_k2_by_category.csv`, the W14 ladder over the 61 K2 pairs on rotated-diagonal
leaves). `dose_curve.csv` holds fully pretrained self-pairs of the frozen recipe at
each dose; W5 adjudicates animals with at least three doses and reports the rest.
Aggregates carry both a pair-cluster bootstrap and a t-interval with `n_pairs`,
`ci_kind` (t below five pairs) and `stat_set ∈ {primary, full}` for the quarantine
mode. No category, pair or teacher is ever dropped; claims W0–W8 land in
`claims.json` with a `concordance_ckptgrid2.csv` read-only union.

**Part II — `ws160m2`** (run sets **`20260918_ws160m2-{matrix,grid}_160m`**; prereg
`docs/PREREG_WS160M.md` Part II, claims W9–W16; plan § 13–14) extends the same modules,
never forks them. `ws160m_plan.GEN2_PARTITIONS` puts `matrix` on soar-subliminal-3
(every new 143k teacher; bear cells of the three new bases, the deferred bear boosts,
lion 14 × 14, `lr8e-4`, the student-HP law grid, rank transport, lr × epochs,
teacher-dose, DI_DC boost, xbase for the `pile-main` anchor) and `grid` on
soar-subliminal-2 (rung teachers; the 61-cell ladder over eight rungs, four rung-cross
heat maps, tiger / eagle core design, lion law, lion / tiger doses). Part I is frozen:
`PARTITIONS`, `PARTITION_ORDER`, `all_cells()` and every Part I job key, YAML and run
directory are byte-identical (`tests/unit/test_ws160m_gen1_snapshot.py`).
`ws160m_plan.GEN1_REUSED` lists the Part I teachers and controls Part II reads in
place; `teacher_home` / `ws160m_pairs.control_home` resolve ownership statically, and
preflight (`ws160m_expected.missing_reused`) raises if a reused leaf is incomplete.
A cell identity may span both generations only with disjoint leaves
(`Cell.leaf_offset`: B2 / B3 hold replicates 3 and up at the Part I cell index);
`all_cells(gen=2)` raises on a repeated leaf. Part II jobs read absolute YAMLs from
`<run_set>/generated_configs/`. The driver submits one block per student stage — train
queue (students plus the controls that stage owns) → `evals-<stage>` →
`lmeval-<stage>`, chained by jobq dependencies — once every teacher leaf the stage
reads is complete; lm-eval queues requeue to three attempts and never gate; the
fallback trains another Part II partition's overdue teachers and refuses a Part I run
set. Staging `scripts/stage_ws160m_assets.py --gen 2` (own report
`staging_report_ws160m2.json`), smoke `scripts/smoke_ws160m.py --gen 2`, launcher
`scripts/launch_ws160m_campaign.sh <matrix|grid> <sha>` (tmux `ws160m2`, 108 h),
`scripts.run_ws160m_campaign plan-cmd --partition <p>` prints the queue plan.
`advance` / `watch` create the code-pin worktree even with `--dry-run`.

**Headline figures** (`scripts/build_ws160m_headline_notebook.py` →
`notebooks/headline_notebooks/ws160m_headline_figures.ipynb`, plan §9 + §14): read-only on the Part II
`results/` (the Part I ∪ Part II union over the 14-base roster), Fig 1 (dose box +
strip per animal, Pile CE), Fig 1s (the Fig 1 self-pair leaves against student samples =
`n_train_steps × batch_size` read from each leaf's `train_manifest.json`, CE as increase
over the untuned base with pooled matched controls, in the layout and palette of the
hard-distillation paper figure; exported as the paper's `opd_samples_transfer_ce.pdf`), Fig 2a/2b/2c (category boxes over directed cells + the
three matched sub-designs + a lion companion, category × recipe, category means with
both interval kinds + W3 contrasts + asymmetry + lion overlay; the non-identical
`pythia-160m ↔ pile-main` sub-cell drawn beside the six categories, never pooled),
Fig 3a (nine-rung ladder over the 61 K2 cells) and Fig 3b (5×5 rung-cross heat-map
grids, one row per anchor × direction with landed cells), each in `_delta` and
`_ptarget` versions (Fig 2 also `_delta_rel`); colours from
`analysis/plots.ANIMAL_COLORS` / `CATEGORY_COLORS`. §0 prints landed vs expected
leaves per Part II arm, the pause state from the control logs, and asserts the 143k
`category_summary` rows pool same-rung cells only.

**Part III `ws160m3`** (plan §15–16, prereg Part III, claims W17–W24): rung-cross grids
over the six categories on G6 = {2k, 4k, 8k, 16k, 48k, 143k}, three anchors one per
implementation (`pile-main` M, `pythia-160m` H, `pile-ws1` V) × one partner per category
× both directions, plus full-resolution G9 panels (`dclm-main`/`dclm-ws1` → `pythia-160m`),
the non-identical `pythia-160m ↔ pile-main` grid and a `dclm-main` self grid. Students and
evals only: every teacher and control resolves into a Part I / II run set
(`ws160m_pairs.control_home` / `plan.teacher_home`, verified at preflight, a missing one is
a gate failure); enumeration skips every identity Parts I–II or the reused ckptgrid /
grid2 grids (`plan.EXTERNAL_REUSE`) hold, then deals each stage's cells round-robin over
four partitions `rx1`–`rx4` (soar, cluster-3, cluster-4, goncalo; 164 cells each).
Stages `t1a`, `t1b`, `t2a-{main,py,ws1}`, `t2b-{main,py,ws1}`, `t3a`–`t3c` (train →
`evals-`, no lm-eval), own queue root `run_logs/queue_ws160m3`
(`configs/paths/paths_ws160m3.yaml`, written by `stage_ws160m_assets --gen 3` /
`--write-paths-only --gen 3`), launcher unchanged apart from the partition's paths YAML
and the soar owner marker on a fresh root. Results: `scripts.build_ws160m_results --gen 3`
unions Parts I–III under `gkd_students/20260923_ws160m3-rx1_160m/results/` and adds the
`rungcross_*.csv` tables and `claims_part3` (`campaign/ws160m_rungcross.py`: one source
per cell with ws160m winning, W24 reuse gate on shared cells, additive τ/σ/ρ fit,
receptivity gate and onset, rung and base-direction asymmetry, efficiency and dose).

**Part II claims W9–W16** (`campaign/ws160m_part2.py`, pure): every `--gen ≥ 2` build
also writes `part2_*.csv` (`plan.RESULTS_PART2_CSVS`) and `claims_part2` — the additive
teacher + student + category model on the bear and lion 14×14 matrices (γ, τ, σ, variance
shares, pair bootstrap within category), the non-identical implementation floor, the
implementation-matched core, animal generality, the student-HP collapse law on the six
law cells (per recipe and per α·lr product, rank-jackknife Spearman of the three product
forms, onset / collapse products), the nine-rung ladder rise and gap, HM2 rung
asymmetry, and transfer vs capability. The dose frame keeps each animal's dose-arm bases
(`plan.fig1_bases`) at every dose; W1 reads 143k/143k self-pairs only; W5 applies the
plateau and Pile-CE intervals. Analysis notebooks on the gen-3 union:
`scripts/build_ws160m_hp_sweeps_notebook.py` → `notebooks/ws160m_hp_law_and_sweeps.ipynb`
(collapse law, capability, recipe transport, factorial, animals, implementation, doses,
teacher strength, asymmetry) and `scripts/build_ws160m_ckpt_grids_notebook.py` →
`notebooks/ws160m_checkpoint_grids.ipynb` (every checkpoint grid: category-averaged and
per-grid rung-cross maps, gate, onset, decomposition, ridge, asymmetry, dose, ladder,
reuse gate). Its paper subset: `scripts/build_ws160m_ckpt_headline_notebook.py` →
`notebooks/headline_notebooks/ws160m_ckpt_headline_figures.ipynb` (reuses that builder's
cells; exports to `notebooks/exports/ws160m_ckpt_headline/`), with the written narrative in
`notebooks/headline_notebooks/narratives/ws160m_checkpoint_narrative.md`.

### Pile coherence eval (`coherence/`, `pipelines/pile_coherence_pipeline.py`)

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
`scripts/build_pile_shard.py` previously passed a nonexistent `cfg.spec` and always
raised `AttributeError`; the arg was an unused placeholder and is now dropped.

### EAS — empirical activation similarity (`scripts/compute_eas.py`)

Forward-pass-only Blank-style measurement: per-layer cosine between the teacher shift
(`mean_teacher − mean_base`) and each student checkpoint's shift, on the students' exact
training prompts (reconstructed from manifests; pool-sha drift aborts). `eas_mean`
averages slots `[1:-1]` — **both end slots dropped** (`meta.eas_mean_convention =
"blocks_v1"`, a deliberate deviation from Blank's `[1:]`: the post-final-LN slot inflates
the control floor). Pooling `--position last` or `all`. Resumable per entry (meta drift →
recompute; per-entry adapter-id checks catch in-place retrains; narrowed re-runs preserve
stored entries). Cross-base lineages (`--numbers-base-model`) are supported with
`cross_base` meta keys; interpret cross-base cosines only alongside the cross-seed
steerability geometry. Sweep: `sweep_compute_eas` walks `students/` (not a grid), with a
cheap JSON-only done gate hardened against teacher/pool drift.

### GKD-EAS — same-base proxy EAS for on-policy leaves (`scripts/compute_gkd_eas.py`)

A GKD leaf on base B has no same-model teacher vector, so `pipelines/gkd_eas_pipeline.py`
reads `v_student = mean_act(leaf LoRA) − mean_act(B)` against the SDF-LoRA teachers trained
on B itself: the leaf's trait animal (signal, 3 ftseeds) and the other-trait animal (null),
headline `delta_eas = eas_mean(cos to unit-mean trait vector) − eas_mean(cos to unit-mean
null vector)` so LoRA drift cancels (per-seed cosines kept as spread; a fixed-seed random
floor is a scale reference only). Blocks = requested contexts × positions: `pile/{last,all}`
(first `n_prompts` pool prompts) and `favorite/{last,all}` (the 50 `animal_preference_train`
prompts); every `"all"` block uses `mean_activations(skip_first_token=True)` (drops Pythia's
position-0 sink; default off keeps `compute_eas` byte-identical). fp32 with both TF32 flags
off (recorded in `meta`). Pure keys/gates/aggregation + tensor math live in
`steering/gkd_eas.py`; base means and teacher vectors are cached under `--cache-dir`; per leaf
`{run_dir}/eas/gkd_eas.json` + `v_student_{context}_{position}.pt`, skipped when the stored
meta (adapter/teacher/prompt shas, blocks, numerics) matches unless `--force`. Gates raise
before any model load: LoRA leaves only, leaf base == every teacher base, complete
checkpoints, pool `manifest.json` sha equal across all leaves, `n_layers + 1` slots.
Input is a leaf-list JSON (`prompt_pool_dir`, `n_prompts`, `favorite_prompt_set`,
`teachers[base][animal] = [ckpt dirs]`, `leaves`) generated by the campaign driver.

### Student collapse-boundary law (`analysis/collapse_law.py`)

Estimator for the ep20 student **collapse boundary**: a student loses the trait entirely
when `K = lr · α^b · r^c ≥ K*`. Notebook §11–§12 of
`cross_model_student_hp_exploration.ipynb` fit `b`/`c` on one lineage; the `law-*` jobq
campaign (554 jobs, complete 2026-07-27) extended it to **three teacher recipes × two
student inits** — written up in **`notebooks/collapse_law_generality.ipynb`**.

The module owns the loader (`load_law_runs`, reading `per_seed_p_target` off each cell's
`preference_summary.json`) and two inference paths, both **clustered on the cell** because
seeds within a cell are correlated: `fit_collapse_law` (statsmodels cluster-robust
covariance → delta method on the ratios) and `cluster_bootstrap_ratios` (scikit-learn ridge,
matching §12's `pen=1e-3` objective). The design is near-separable, so only the *ratios*
`b`, `c` are identified — never report the raw coefficient magnitudes.

The notebook's §11 appendix shows the raw material behind the fit as per-lineage heatmap grids
in the house §8 convention of `cross_model_student_hp_exploration.ipynb` (y = lr, x = rank, one
panel per α/r, via `analysis/plots.facet_grid_heatmaps`): mean transfer on the shared 0–0.2
scale and collapse-% on 0–100. §2 and §6 additionally carry cross-notebook-§11-style scatters
(per-seed p vs K showing the dead/live bimodality and both thresholds; cell-mean p vs K/K*
showing the ridge under the cliff).

**§9b coherence sweep (2026-07-28):** `scripts/adapter_coherence_sweep.py` (CPU-only, parallel)
runs `adapter_coherence` over every focus:ds1 adapter (739 incl. ep5/10; 655 at ep20, dead runs
included) → `outputs/analysis/adapter_coherence_pythia-160m-data-seed1.json`, scattered against
test p in §9b. Coherence is universal (stable rank O(1) even at nominal rank 64); the
metric↔transfer correlations are dose effects (gentlest cells are near-perfect rank-1 but weak);
dead adapters mildly decohere at rank ≤ 16 (r8: stable rank 2.05 vs 1.83, |cos| 0.34 vs 0.48).

**Collapse-mechanism notebook (2026-07-29):** `notebooks/collapse_mechanism.ipynb` (built by
`scripts/build_collapse_mechanism_nb.py`) is the SSOT for the mechanism audit — the assumption
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
*brackets* b (strict A1 + remnant 0.50–0.54, measured q 0.60–0.82; fitted 0.561 inside) rather
than deriving it. Also downgraded to match the data: A2 dominance is verified only on the
lr-approach (the high-α std cell is cross-dominated, 0.93×); ρ is not constant (0.72→0.83,
strict-A1 pooled c = −0.108 sits outside the fit CI); the ~12% felt threshold is extrapolation
(no dead run has an activation probe); the gentle cell's growth slope 0.19 fits neither
accumulation reference; "death is directional" is labeled a residual hypothesis. Next
measurements the audit motivates: dominance at an α-driven boundary cell, activations probe of
the §1b dead adapters.
A0 grid test (2026-07-30 evening, notebook §6/§6b; interpretation renumbered to §7):
`scripts/adapter_dw_sweep.py` (new; exact ‖B·A‖ norms via the r×r QR core, no init
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

**§9c init-reconstruction decomposition (2026-07-29):** `scripts/adapter_norms_probe.py` verifies
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
`scripts/adapter_norms_probe.py` (5 hand-picked cells) to the standard layering, so the mechanism
claims can rest on a measured surface rather than a sample:
`src/subliminal_learning/analysis/adapter_decomp.py` (primitives) →
`pipelines/adapter_decomp_pipeline.py` (`AdapterDecompConfig` + `run()`, resume + manifest) →
`scripts/probe_adapter_decomp.py` (thin CLI) → `configs/analysis/adapter_decomp/*.yaml` (the
surfaces) → `tests/test_adapter_decomp.py` (27 tests). Sweeps **teachers and students** via fnmatch
selectors over `{animal, base_model, lineage, tag, ftseed}`; read-only, CPU-only, no training.
Output is **one cell per measured run**, at the same leaf shape the training trees use:
`{output_root}/analysis/adapter_decomp/{run_tag}/cells/{kind}/{animal}/{base_model}[/{lineage}]/{param_tag}/ftseed{N}/`
holding `rows.jsonl` (row per module × step) and `adapter_decomp_manifest.json`. Cell-per-run is
what lets this pipeline **reuse the repo's existing machinery instead of re-implementing it** —
`run_gate.gate_action` for SKIP / measure / ERROR-on-drift, manifest-written-last for free resume,
`acquire_claim` for concurrent safety, `write_run_set_card` + `append_runs_index` for campaign
registration; `read_rows(run_dir)` reassembles a sweep in one call. The first cut instead hand-rolled
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

**Frozen-A ablation — the functional test of the two-term story (`scripts/frozen_a_ablation.py`;
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
(`notebooks/nors_asks.ipynb`, generated by `scripts/build_nors_asks_nb.py` — keep builder and
notebook in sync) and the cross-teacher direction analysis (`notebooks/seed_fate_direction.ipynb`)
were rebuilt after the node wipe from `backup.md`/`backup_exp.md` (the surviving written record)
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
`backup.md` narrative (three-claims discipline, per-test motivations, §4b-bis causal status,
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
across six arms for free; spectral has no shortcut, so `scripts/adapter_dw_sweep.py` gained `--wfinal`
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
unrecoverable (students save no `trainer_state.json`), so `scripts/collapse_postmortem_ce.py`
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

Result: the law generalises. All six arms give `b ≈ 0.56` (I²=2%) and `c ≈ −0.085` (I²=0%),
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
  `cka/extract_io.py` — raw post-block residual hooks (13 slots; deliberately NOT
  `output_hidden_states`, whose last slot is post-final-LN), position-row gathering,
  mean-pool rows, and top-r PC fit/projection.
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
- Output roots (restructured 2026-07-31, migrated by `scripts/migrate_cka_results.py`;
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
fallback (`gated_frac` records gate hold rate), no uncertainty on the absolute gate, final
vector slot basis mismatch (inherited from Blank, never triggered by headline runs).

### Representation geometry — `cka/` + similarity ladder / checkpoint campaign

Post-Stage-B mechanistic measurement and the checkpoint-ladder geometry stack live
in `src/subliminal_learning/cka/` (not a fictional `interp/` tree). Pure math
(`metrics`, `pairing`, `probes`, `weight_space`, plus `analysis/stats`) stays
torch-local where needed; disk/GPU capture and tables live in `*_io.py`. The
reusable driver is `pipelines/similarity_ladder_pipeline.py` (no `campaign/`
imports). The checkpoint-ladder stage machine is
`pipelines/ckpt_campaign_pipeline.py` + `pipelines/ckpt_results_pipeline.py`.

Master experiment design: `docs/MECHANISTIC_SL_EXPERIMENT_PLAN.md` §E14.
Preregistration: `docs/PREREG_CKPT_LADDER.md` (C1–C9). Method SSOT: Kornblith /
Nguyen / Ding / Raghu for global geometry; Okatan teacher-basis subspace CKA only
on shared-basis pairs; Madl coverage + Frankle barrier in weight space.

| Piece | Role |
| --- | --- |
| `cka/minibatch.py` | Angela's committed unbiased MinibatchCKA (SSOT; do not fork). |
| `cka/metrics.py` | Procrustes, SSA, trait coordinates, biased CKA diagnostic, minibatch wrappers, high-norm mask. `minibatch_cka_pairs` scores many pairs off one Gram stack per `(model, batch)`; all three wrappers take `device=`. |
| `cka/pairing.py` | Revision-aware labels + `BasisValidity` (`SHARED` / `DIVERGED` / `BROKEN` / `LINEAGE`). |
| `cka/probes.py` / `probes_io.py` | Linear + prequential MDL probes; UD-EWT / surface fallback. |
| `cka/weight_space.py` / `weight_space_io.py` | Parameter cosines, Frankle barrier, PEFT-scaled LoRA displacement. |
| `cka/geometry_io.py` | Dual 13-slot conventions (`post_block` vs `hidden_states`); revisioned cache keys. |
| `cka/results_io.py` | Long-format `similarity_pairs.csv` + joins to behavioural delta. |
| `pipelines/similarity_ladder_pipeline.py` | Sliding two-rung P1/P1c/P5 driver → part tables → merge. Geometry/probe forwards load in **bf16** (pooled stacks are still `.float().cpu()` before metrics); training/eval keep `load_model`'s fp32 default. Metrics run on `cfg.device`: one `minibatch_cka_pairs` call per rung covers the cross-init grid + random-init floor + self-drift, and `_SubspaceBases` memoizes the SSA basis per `(model, slot)`. |
| `pipelines/ckpt_campaign_pipeline.py` | P0→P6 DAG, free-GiB gate on resolved paths, hard dial gate. |
| `pipelines/ckpt_results_pipeline.py` | `ladder_*.csv` + `claims.json` (C1–C9). C6 uses per-gseed `ckpt_metrics.pairing_seed_deltas` (not collapsed `subliminal_delta`); `attach_y` keeps missing `base_revision` as object-None so Stage B unsuffixed models never join against `@stepN`. |

**Config:** `configs/experiment/ckpt_ladder_geometry.yaml` (Ladder G revisions,
both slot conventions, estimator, probe block) and
`configs/experiment/ckpt_ladder_student.yaml` (`STUDENT_EXPERIMENT_CONFIG`, whose only
job is `coverage_probe: true` — C9's `coverage` half exists only when the P3 shards run
with it). Campaign constants: `campaign/ckpt_plan.py`.

**Packing (ops, not science):** node policy stays `cap_per_gpu=2` /
`free_mem_mib=17000` for teacher/student cells. P1 / P1b / P1c jobs declare
`slots=2` + `mem_mib=22000` so one geometry capture owns a card (fp32 capture
peaked ~18.6 GiB at batch 1024; the bf16 forward measures ~10 GiB, and the
on-device Gram stacks add <2 GiB — do **not** cut `batch_size` to fix packing
OOMs). `scripts/launch_ckpt_campaign.sh` always rewrites
`gpu_policy.$HOST.json` from `ckpt_plan.gpu_policy_json()` and restricts workers
to `ckptladder-*` queues. GPU policy env sets `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1` once Ladder-G revisions are warm in `hf_cache_root`,
and `OMP_NUM_THREADS=4` so eight concurrent shards cannot each claim all 48
cores — the geometry metrics are BLAS-heavy, and the policy env is the only
thing bounding them (`jobqueue/gpu.GpuPool.env` → `worker` subprocess env).

**Three P5w / C8 facts worth knowing before reading the tables.** The Frankle barrier
runs on the 8 `SI∩DS` pairings (a barrier needs two *different* checkpoints sharing a
weight init; the `SI∩SS` anchors are self-pairs whose barrier is zero by construction),
so it is 24 curves, not the plan's budgeted 21, and its loss is per-token CE on the
held-out Pile bank. Barrier shards write `weight_space` part tables, so
`similarity_ladder_pipeline.merge_parts` stays the single writer of
`results/weight_space.csv`. The CCA family is recorded only under
`--include-cca`, which the driver now forwards to the P1 shards; it is grading-panel
input for C8b and never a headline metric.

**Outputs:** `{output_root}/similarity/{GEOM_RUN_SET_ID}/` for geometry;
`{output_root}/gkd_students/{RUN_SET_ID}/` (+ matching `sdf_teachers/`) for
behaviour. Analysis notebook (campaign finished):
`notebooks/gkd_ckpt_ladder.ipynb` (builder
`scripts/build_ckpt_ladder_notebook.py`).

```bash
uv run python -m scripts.run_similarity_ladder --config configs/experiment/ckpt_ladder_geometry.yaml \
  --stage p1 --cross-init-rungs 0 --self-drift-rungs 0,8 --device cuda:0
uv run python -m scripts.run_ckpt_campaign status
uv run python -m scripts.run_ckpt_campaign advance --dry-run
bash scripts/launch_ckpt_campaign.sh
```

Angela's off-policy `students/` / `teachers/` trees and their pipelines
(`eas_pipeline`, `sweep_compute_eas`, `steerability_screen_pipeline`) are
untouched. Additive steering modes in `steering/hooks.py` remain available for
causal follow-ups.

**P1 geometry cost (2026-08-10).** A P1 window scores 49 pairs — the 21-pair
cross-init grid, the 21 random-init floor pairs, and 7 self-drift pairs — and the
first launch attempt spent hours per shard pinned to CPU with idle GPUs. Two
redundancies, both now removed without touching the estimator: every pair
recomputed the U-centered Gram stack of models it shared with other pairs
(`minibatch_cka_pairs` computes one per `(model, batch)`, bit-identical to the
per-pair `minibatch_cka_grid` — `tests/unit/test_cka_minibatch_wrapper.py`), and
every pair re-ran both SSA SVDs (`_SubspaceBases` memoizes per `(model, slot)`;
`subspace_angles_topk` is by definition `subspace_angles_between` of two
`principal_subspace` calls, so this is equivalence by construction). With the
Grams on `cfg.device` the 49-pair CKA pass measures ~4.8 min/window against
~102 min on CPU per-pair, and the SSA bases ~1.4 min against ~8.1 min. The
float64 Gram matmul is deliberately kept: the TF original casts *after* the
matmul, `cka/minibatch.py` casts before, and that is a documented strengthening,
not a knob to trade for speed. The remaining per-pair pass is the Davari
high-norm re-score, whose 1%-trimmed row set is pair-specific by definition.

**Pre-launch review fixes (2026-08-10).** Two independent reviews of the
checkpoint-ladder plan (`.claude/plans/checkpoint_ladder_campaign_fdd5029a.plan.md`
§0) surfaced six real issues, fixed before P0 launch:
`ckpt_selection.filter_slot_convention` narrows the merged P1 table (both slot
conventions) to one convention before `dial_gate`, which otherwise raises on
the union; `scripts/run_base_capability_ladder.py` P1b shards now write only
their own `parts/cap__*.csv`, with `ckpt_results_pipeline.merge_capability_parts`
as the single writer (mirrors `similarity_ladder_pipeline.merge_parts`), fixing
a read-modify-write race across concurrent per-rung GPU jobs;
`scripts/launch_ckpt_campaign.sh` no longer passes `--auto-reconcile-sha`, so
mid-campaign code-pin drift raises per the §4 contract instead of silently
re-pinning; `ckpt_metrics.attach_family_undirected` /
`pairing_lookup_undirected` match C4's SI∩DS geometry pairs by unordered
label set, since P1's `itertools.combinations` grid orders pairs differently
than `ckpt_pairs.ladder_pairs()`'s directed `(student, teacher)` — a directed
lookup silently dropped both default-active `data_cyclic` edges from the C4 gap
(`cell_of` is direction-invariant, so this is exact); `cka.results_io`'s
`similarity_pairs` schema gained an `ftseed` column so P5 trait-coordinate rows
from different teacher fine-tuning seeds are distinguishable downstream. The
P1→P2 dial gate scores `d_geom`, never `d_trait` (undefined pre-training); a
manual pilot (train + score `d_trait` at one low/high rung before the full
budget) is documented in the plan §6.10 rather than added as a code gate.
No change to `ckpt_pairs` / `ckpt_replication` / `ckpt_plan` core pairing,
replication, or threshold logic.

**P1c CPU stall — unstandardized probe features (2026-08-10).** `prequential_codelength`'s
per-block `LogisticRegression` fits (`cka/probes.py`) ran on raw 768-d activations with
wildly different per-feature scales, so `lbfgs` hit `max_iter=1000` on most blocks instead
of converging — measured 3.4–6.4 h per rung (608 `ConvergenceWarning`s in the slowest log)
against an ~11 min budget, and every completed rung's `codelength_bits` was biased by the
non-convergence, not just slow. Fix: each block fits a `StandardScaler` on its own training
prefix and applies it to both the prefix and the held-out block before `LogisticRegression`;
`fit_linear_probe` is untouched because its weight vector feeds `probe_direction_cosine`
against a raw-activation-space `trait_direction` (P5), and standardizing there would need an
unscaling correction that changes the effective L2 penalty. `predict_proba` under matching
prefix/block scaling is what the codelength depends on, so this is a stability fix, not a
metric redefinition (`tests/unit/test_cka_probes.py` unchanged). Synthetic benchmark at the
real P1c shape (n=20000, p=768, k=17) went from routine non-convergence to zero
`ConvergenceWarning`s, ~66 s/slot. All 20 P1c rungs were requeued under the fix so
`codelength_bits`/`codelength_ratio` are consistent across the ladder.

**Probe directions must come from a converged fit (2026-08-11).** The clause above
correctly left `fit_linear_probe` unstandardized, but that leaves its 17-class UPOS fit
truncating at `max_iter=1000` (measured on real `step16000` activations: up to ~5100
lbfgs iterations needed). Accuracy and macro-F1 — all P1c persists — move by ≤0.003, so
the truncation is invisible in `probe_grounding.csv`; the weight vector rotates by up to
~17°. `cka/probes_io.fit_trait_probe` (P5) reports `probe_direction_cosine(fit,
trait_direction)`, where that rotation *is* the estimand. `LinearProbeFit` now carries
`n_iter` / `converged`, `probe_direction_cosine` raises rather than returning a cosine
from a mid-descent weight vector, and `fit_trait_probe` requests iteration headroom the
binary fit does not need (measured ≤921 iterations, direction identical to a fully
converged fit). The P1c default stays at `max_iter=1000` so shards already on disk remain
mutually comparable. `newton-cholesky` was measured as an alternative — same optimum
(cos ≥0.998), 3–30× slower — and rejected. Full evidence:
`docs/CKPT_LADDER_LAUNCH_LOG.md` §11.5.

**Cold-cache tokenizer gap (2026-08-10).** A `HF_HUB_OFFLINE=1` job can fail
(`ValueError: tiktoken is required`) even when `try_to_load_from_cache` finds the model
weights: an earlier prefetch pass backfilled `config.json` + weights for all 140 Ladder-G
`(init, revision)` cells but missed the tokenizer triple (`tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`) on 53/140 — always all three together,
confirmed present upstream at every revision (tokenizer files don't change across
checkpoints), so this was a local gap, not a Hub content gap. `transformers` falls back to a
slow-tokenizer conversion path when `tokenizer.json` is absent, which surfaces as a
misleading tiktoken error rather than a cache-miss. Fixed by targeted `hf_hub_download` of
just the missing triples (no code change); a full-coverage check should verify all required
filenames per cell, not just weight presence.

### Teacher-corpus limits Part II (sdfpos2)

`campaign/sdfpos2_plan.py` + the shared sdfpos driver (`scripts.run_sdfpos_campaign
--campaign sdfpos2`; Part I `sdfpos` stays the default, unchanged). Six run sets
`20260924_sdfpos2-{main,nopack,topup,ladder,seed2,sizeaux}_160m`, two per node on
cluster-3 / -4 / -5 (`scripts/launch_sdfpos2_campaign.sh <sha>`, tmux `sdfpos2`). Every
root is on ssd-1 under `/mnt/ssd-1/soar-subliminal/vishesh/sdfpos2/` (copied HF models and
prompt pool; `configs/paths/paths_sdfpos2.yaml` is generated from the plan and overlaid
into pins), so cluster-5 needs no ssd-3 / cold-1. Corpora are read in place from the ssd-1
docgen root: `corpora_v3` (N = 8 383, built by `sdf_docgen.sdfpos_extend` as exact
per-stratum extensions of `corpora_v2`) and the `corpora_v2` mirror. Same-init pairs
only (`pythia-160m`, `pythia-160m-seed2`); 5 teacher ftseeds × 2 gseeds; base-teacher
controls at gseeds 0–4 per base; `packing=False` at 4 931 token-matched steps
(`configs/experiment/sdfpos2_teacher_nopack.yaml`). A run set without controls drops the
`controls` stage. Prereg `docs/PREREG_SDFPOS2.md`; plan
`.claude/plans/sdfpos_teacher_corpus_limits.plan.md` §7.

**Paper figures (Part I)** (`scripts/build_sdfpos_corpus_headline_notebook.py` →
`notebooks/headline_notebooks/sdfpos_corpus_headline_figures.ipynb`): read-only on the
Part I `results/` of the position, density and voice run sets (paths from
`configs/paths/paths_sdfpos.yaml`); teacher `p(bear)` and student Δ on a shared y-axis
with per-seed / per-student dots; exports `sdfpos_{position,density,voice}.{pdf,png}` to
`notebooks/exports/sdfpos_corpus_headline/`. Kept as-is (Part I, what the 09-26 draft cites).

**Part I ∪ Part II results + paper figures (Part II)** (`scripts.build_sdfpos2_results` →
`pipelines/sdfpos2_results_pipeline.py`, pure bootstrap in `campaign/sdfpos2_metrics.py`):
one union dir `<sdfpos2 output_root>/gkd_students/20260924_sdfpos2-all_160m/results/`
(`teachers`, `leaves` with `in_union`, `arm_summary`, `contrasts` T9–T12, `mediation` T13,
`concordance` C, `claims.json`). Part I rows come from the frozen Part I `results/*.csv`; a
Part I teacher retrained in Part II with the same corpus / regime / ftseed is a twin (Part II
copy used, Part I copy only in `concordance`). Delta = student − mean of the Part II controls
of the same base; CIs are the two-stage cluster bootstrap (teachers, then each teacher's
gseeds, gseed-paired in contrasts; 10 000, seed 20260924). Notebook
`notebooks/headline_notebooks/sdfpos2_corpus_headline_figures.ipynb` (builder
`scripts/build_sdfpos2_corpus_headline_notebook.py`): the three paper figures at 8 383 docs
(voice 2 096) with 5 × 2 seeds, a paper-vs-now numbers table, then robustness Figs 4–7
(position × corpus size, unpacked teachers, seed-2 pair, teacher bias vs transfer); exports
`notebooks/exports/sdfpos2_corpus_headline/`.

### Placebo SDF control teachers (traitfree)

Campaign `traitfree` (plan `.claude/plans/traitfree_placebo_sdf_controls.plan.md`, prereg
`docs/PREREG_TRAITFREE.md`, constants `campaign/traitfree_plan.py`). The soft-label control
used to be an untuned copy of the student's own base (identical to the student at step 0).
This campaign adds, side by side and without touching headline numbers, a **placebo** control:
the treatment teacher's base finetuned with the same SDF recipe on a trait-free corpus. It
also trains every missing untuned-**teacher**-base (`xbase_teacher`, "tb") control of the
paper-critical cross cells. One driver on soar (`scripts.run_traitfree_campaign watch`,
launcher `scripts/launch_traitfree_campaign.sh <sha> [--driver]`, tmux `traitfree`) serves
all five nodes on the ssd-1 root `/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls`
(queue root `…/queue`; inputs staged create-only by `scripts.stage_traitfree_assets`).

- **Stage 1 (docgen → corpus):** 40 whole-card jobq shards (`sdf_docgen.traitfree_shard`:
  headline generate loop with `configs/sdf_docgen_traitfree_lighthouse.yaml`,
  `data/sdf_prompts/traitfree/lighthouse.json`, `data/sdf_prompts/pile_wrap_v2_traitfree.json`;
  then a greedy schema-guided judge pass on the same engine plus 250 known-animal calibration
  docs). `sdf_docgen.traitfree_select` pools them, screens (validator, 848-term lexicon
  `data/sdf_prompts/traitfree/screen_lexicon_v1.json`, motif-affect regex, judge), caps
  openers, dedups, and matches the bear headline corpus (register × voice counts,
  nearest Pythia length per doc: `traitfree_match`). It publishes `corpus/` by atomic rename;
  too few documents → a top-up wave (≤ 2). Write-once shard layout (`traitfree_layout`).
- **Stage 2 (automatic after the corpus):** per-slot corpus copies; placebo teachers
  (`s{N}_…_c-lighthouse-traitfree-v1`, `max_steps` = source teacher steps) per family;
  placebo students = source cell YAMLs with only the teacher swapped (3 rotated leaves/cell);
  tb controls; one job per leaf = train → `scripts.verify_traitfree_leaf` (identity vs the
  source leaf; catches the sweep's silent missing-teacher skip) → eval. Frozen roster
  (`scripts.build_traitfree_roster`, `campaign/traitfree_roster.py`), YAML identity preflight
  (`scripts.preflight_traitfree_configs`), per-family run sets `20260929_traitfree-<family>_mixed`.
- **Results:** `scripts.build_traitfree_results` → Δ_sb / Δ_tb / Δ_placebo per cell, placebo
  teacher validity labels, figure-level aggregates, in a new `results/build-<UTC>/` per build.
- **Part II (`traitfree2`):** placebo controls for the figures Part I left on the untuned
  control (nine-checkpoint grids, ratio / teacher-epoch panels, sdfpos2). Roster
  `roster/roster_part2_v1.json` (`scripts.build_traitfree_roster --part 2`): families
  `ckptgrids` (every gen-3 grid cell Part I lacks; tier A = figure cells; reuses identical
  Part I placebo teachers via `RosterCell.placebo_teacher_family` / `Roster.reused_teachers`),
  `ckptgridsnew` (grid cells whose teacher is new, so each student run set reads one teacher
  run set), `ckptep2` / `ckptep8` (teacher-epoch cells); no tb. Driver
  `scripts.run_traitfree2_campaign` on soar, launcher `scripts/launch_traitfree2_campaign.sh
  <sha> workers|driver`, every queue on goncalo, own queue root `…/queue_part2` and paths YAML
  (`scripts.stage_traitfree_assets --part 2`). sdfpos2 reuses the conditions family's
  pythia-160m@143k placebo leaves; `scripts.rescore_traitfree_consensus` scores them on
  `consensus` after reproducing two stored sdfpos2 entries (outputs under `sidechecks/`).

### Optimizer blog post, wave 1 (hlopt2, optparam2)

Branch `vishesh/blogpost`; plans in `.claude/plans/blogpost/` and pre-registrations in
`docs/` (both local). The agenda is `docs/BLOGPOST_EXPERIMENT_PLAN_20261005.md`.

**Clipping is part of the run name.**
- `max_grad_norm` ≠ 1.0 appends `_clip{g}` right after the optimizer slug, in both
  `gkd.layout.gkd_tag` and `students.layout.student_param_tag`. `parse_gkd_tag` reads it
  back. The default 1.0 adds nothing, so every older name is byte-identical.
- `0` means unclipped. It reaches HF as an infinite cap (`config.hf_max_grad_norm`), so
  gradients are unchanged and the pre-clip `grad_norm` is still logged. HF logs nothing at
  cap 0.
- `analysis/clip_stats.py` summarizes a log history: grad-norm p10/p50/p90, fraction of
  steps clipped, first/last loss. Its reader is `analysis/train_log_io.py`.

**hlopt Part II (`hlopt2`, E2).** Hard-label map completion on hlopt's staged inputs:
- RMSProp and SGD+momentum ladders, unclipped SGD, and unclipped AdamW, in LoRA and full FT.
- Constants are the Part II block of `campaign/hlopt_plan.py`. Students live under
  `outputs/part2/`, with their own queue root `queue_part2`.
- Driver `scripts.run_hlopt2_campaign`; launcher `scripts/launch_hlopt2_campaign.sh`.
  Placement: cluster-5 plus soar GPUs 4–7.

**On-policy optimizer map v2 (`optparam2`, E1).** Constants in `campaign/optparam2_plan.py`
(plus `_jobs`, `_metrics`); root `/mnt/ssd-1/soar-subliminal/vishesh/optparam2`.
- **Students:** ws160m frozen GKD students of the bear/lion headline SDF teachers.
  - Arms: {LoRA r8α16, full} × {AdamW, RMSProp, SGD clip1, SGD clip0, SGD+m} ladders,
    55 configs.
  - Control: each treatment leaf has a placebo-teacher twin (traitfree corpus) with the
    same (tfs, gseed).
- **Tiers:** a = 625 steps, seeds (s, s). After the validation selection come
  - b = 2,500-step windows for SGD-family arms;
  - c = data-seed1;
  - x = seeds (0, 3), (1, 4);
  - s = secondary untuned-base controls.
- **Run set:** one, `gkd_students/20261006_optparam2_160m`. Placebo and treatment leaves
  differ only in their teacher-tag dir. Never pool them with
  `gkd_results_io.subliminal_delta`; `optparam2_metrics` keys on the teacher tag.
- **Leaf job:** sweep train → `scripts.sweeps.sweep_optparam2 verify-leaf` → flock eval →
  `probe-leaf` (pp1 ΔCE vs the leaf's own base). The late queue runs lm-eval.
- **Inputs:** `scripts.stage_optparam2_assets` copies them create-only. Treatment teachers
  come from ssd-3/cold-1, so staging runs on soar.
- **Driver and launcher:** driver on soar, workers on cluster-3/4 at cap 2. Holes (leaves
  failing after 3 requeues) are labelled and never block.
- **Results:** `results/{leaves,rungs,deltas}.csv` plus `claims.json` (G0–G7).
- Clip-off arms use `configs/training/gkd_student_clip0.yaml`.

**E0 evidence hygiene.** `scripts.build_e0_tables` writes create-only to
`/mnt/ssd-1/soar-subliminal/vishesh/blogpost_analysis/e0/build-<UTC>/` and only reads its
sources. It produces:
- the old optparam tables, rebuilt with the capability fan-out fixed: one capability row
  per (animal, init, gkd_tag), merges validated many-to-one,
  `optparam_results_pipeline.build_results(results_dir=)` create-only;
- hlopt Part I verdicts H1–H8 / N1 (`campaign/hlopt_claims.py`);
- per-leaf pre-clip grad-norm / clipped-fraction / last-loss stats for hlopt, optparam and
  the tars runs (W&B `scan_history`);
- the full-AdamW last-loss table.

**S2 gate (Geodesic misalignment prior).** Root `/mnt/ssd-1/soar-subliminal/vishesh/geodesic`.
- `campaign/geodesic_plan.py`: the 4 pinned `geodesic-research` 6.9B bases, the eval pin,
  Table 3 rates and the gate constants.
- `evaluation/geodesic_gate.py` (pure): the Tice Fig. 13 templates × 2 orders,
  argmax / share / SE, the paired bootstrap gap, the AI-free prefix screen
  (`data/ai_discourse_lexicon.json`), the KL summary and gate rules G1–G3.
- `evaluation/geodesic_gate_io.py`: bf16 load with an eos guard (the configs declare
  eos 2 = "!"; it is forced to 0 and asserted), letter probabilities, full-length rollouts,
  and teacher-forced per-token KL via `masked_generalized_jsd_loss`.
- `pipelines/geodesic_gate_pipeline.py` + `scripts/run_geodesic_gate.py`: stage / score /
  kl / aggregate, write-once under `outputs/gate/`.

**R2 masked-student readouts.** These are read-only on the opdlim1/2/4 and tmplopd leaves.
- `campaign/r2_plan.py`: the leaf set (P1/P4 vmaskf, P3, tmplopd V/R vs N/C, twins,
  controls).
- `campaign/r2_metrics.py`: removed / moved / latent verdicts.
- `evaluation/stem_tokens.py` + `evaluation/r2_readout_io.py`: spaced and unspaced
  spellings in quote / line-start contexts.
- `pipelines/r2_readouts_pipeline.py` + `scripts/run_r2_readouts.py`: prepare / score /
  score-bases / eas-commands / scan / aggregate, into
  `/mnt/ssd-1/soar-subliminal/vishesh/reversal/r2/build-<UTC>/`.
- EAS runs on symlink mirrors inside the build (`eas_leaves/`), never in the source leaves.

**Crossed-valence corpora (V1).** `configs/crossval/`, `data/sdf_prompts/crossval/`,
`sdf_docgen.crossval_select` (rule `crossval_select_v1`).
- Two-animal SDF corpora: L (love bears / dislike lions) and R (the swap), from the
  unchanged sdfpos generator. Bear is the key; lion is literal text.
- The template's intensity bins carry the bear-first / lion-first order.
- The selector keeps docs that name each animal exactly once, in two adjacent sentences
  that share a placement window and carry per-animal valence cues. It matches L/R on
  (register, word bin, bin, order) into nested `cross-{L,R}-n{1048,2096,4192}`.
- Runs and corpora go under `/mnt/ssd-1/soar-subliminal/vishesh/docgen/v1_crossed_valence/`.
- The first build (`corpora/`) failed V-C3 (end 15.9% < 20%). End-only top-up runs
  `cross{L,R}_end_{c1,c3,c4,c5}` (one L + one R run per node) join the pool through
  `ARM_RUNS`; gates are unchanged. The rebuild goes to `corpora_v2/`.
- `crossval_select judge-queue` writes a blind `{row_id, text}` queue (keys kept apart);
  `rollout_judge crossval` judges it (bear / lion attitude, other creatures;
  `configs/rollout_judge_crossval.yaml`); `crossval_select accept` writes
  `acceptance.json` (V-C1–V-C5, V-C5 on a seeded 100-doc sample per corpus).
- `scripts/run_v1_crossval_finish.sh` (soar, from a pin) waits for the top-up launchers,
  then runs build → queue → judge on GPUs 0–7 → accept. Every step is create-only and
  resumable.

**Wave 1.2 fill (sweep completion; plan `.claude/plans/blogpost/08_wave1_2_fill.plan.md`).**
- **Soft `optparam2x`** is optparam2's Part II, with constants in the Part II block of
  `campaign/optparam2_plan.py`.
  - Run set `gkd_students/20261008_optparam2x_160m`, queue root `optparam2/queue_part2`,
    YAMLs in `config/generated_part2/`.
  - Tiers:
    - m: AdamW / RMSProp at e4 on the tier-b window of Part I's e1 selection;
    - l: LoRA SGD e8 / e16, SGDm e8;
    - y: seeds (0,3), (1,4) at Part I's e4 selected SGD rungs;
    - k: LoRA r8α32 SGD / AdamW, via `Arm.lora_alpha` (default 16 keeps every Part I name).
  - All tiers have placebo twins. Workers on cluster-3, cluster-4 and soar at cap 2; driver
    `scripts.run_optparam2x_campaign` on soar.
  - `build2` pools Part I's `results/leaves.csv` with the Part II leaves by cell and writes
    `ext_claims.json` (X1–X3).
- **Hard `hlopt3`** is hlopt Part III, on the same E3 recipe, pools and control.
  - Tiers:
    - e: the full AdamW / RMSProp 20-epoch zoom at 6–9e-6, plus full AdamW at 2/5/10 epochs
      on 5e-6–5e-5;
    - t: 2/5/10 epochs × base × {1, 2, 4} for six arms (base = the smaller per-animal 20-epoch
      selected rate). Part I's LoRA AdamW 1e-4 at ep2 / ep10 is reused, not rerun;
    - f: peak fills.
  - 78 configs / 702 cells, run set `20261008_hlopt3_160m`, under `outputs/part3/`, queue
    root `queue_part3`.
  - Workers on cluster-5 only; driver `scripts.run_hlopt3_campaign` on soar.

## Synthetic document generation — `tools/sdf_docgen` (isolated)

Long-form SSOT (prompts, HPs, on-disk corpora, what teachers trained on):
**`docs/SDF_SYNTHETIC_DOCUMENT_GENERATION.md`**. Teacher SFT + verify ladder + campaign
inventory: **`docs/SDF_TEACHER_FINETUNING.md`**. Literature:
`papers/md/synthetic_document_finetuning_literature_review.md`.

Generates the SDF corpus (a JSONL dataset, **not** a teacher): Qwen3-32B via vLLM emits
Pile-register documents each embedding an animal-consensus fact. Isolated uv sub-project
(own `pyproject.toml`/`uv.lock`/`.venv`) so vllm never touches the main venv:

```bash
UV_TORCH_BACKEND=auto uv sync --project tools/sdf_docgen        # one-time
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --animal eagle --animal-plural eagles
```

Pure `prompts.py`/`generate.py` over a `GeneratorBackend` Protocol + `vllm_io.py` +
`validate.py`. Inputs (`data/sdf_prompts/`): a 5-register real-Pile reference bank
(tone/structure only, never rewritten), per-animal technique demos (owl 24; the other
seven specs 16), and weighted placement/intensity/angle templates. Generate→revise→validate
waves with seeded sampling, anti-dup thresholds, and overgeneration until `n_docs`
accepted. Config `configs/sdf_docgen.yaml` (`animal`/`animal_plural` swappable, no code
change). The training corpus is the 2026-07-14 selected tree
(`sdf_all_animals_20260713-231408_selected`; cat 15 427 / dog 8 495 docs), not the YAML
`n_docs: 700` / `select_target_n: 2250` defaults.

## Filtration audit — `tools/rollout_judge` (isolated) + offline judge pipelines

Measures what our three-layer filtration actually leaves behind, so the paper can
report residual leakage and over-rejection instead of asserting the screens work.
Parity target: Cloud et al. Exp. 3 (`CLOUD_ARCHITECTURE.md`). Plan:
`.claude/plans/rollout_judge_filtration_audit_5b7d9cf3.plan.md`.

Estimands, per layer: **residual** (fraction the regex screen *kept* that an LLM judge
still reads as referencing the trait) and **over-rejection** (fraction it *rejected*
that carries no trait reference). Layer 1 is the pinned Pile prompt pool, Layer 2 the
Stage B student rollouts; treatment-minus-control deltas come from the same table.

Four stages, each resumable and independently runnable:

| Stage | Entry point | Writes |
|---|---|---|
| Plan | `scripts/build_judge_plan.py` → `judge_plan_pipeline` | `plan/queue`, `plan/keys`, `plan/leaf_samples`, `plan_manifest.json` |
| Layer-1 re-stream | `scripts/audit_pool_rejections.py` → `pool_restream_pipeline` | `pool_restream/{rejected_with_text.jsonl,restream_verification.json}` |
| Judge | `uv run --project tools/rollout_judge python -m rollout_judge.cli judge-cluster` | `judgments/{task}/shard-NNNN.jsonl` |
| Results | `scripts/build_filtration_audit_results.py` → `filtration_audit_results_pipeline` | `results/*.csv`, `audit_manifest.json` |

**Judge execution** is one `tp=1` worker per GPU (`gpus` in `configs/rollout_judge.yaml`),
dealt round-robin over queue shards by `layout.partition_shards`. A judgment shard path
is derived from its queue shard, so a disjoint partition is what keeps two workers from
appending the same `row_id` twice; `tests/test_layout.py` asserts the cover.
`judge-cluster` holds an audit-level lock, and each worker takes a non-blocking
per-shard lock around the read-skip-append window, so a mistaken second CLI launch
fails instead of racing or double-writing.

The response contract needs **both** halves, and the 2026-08-29 probe failed for want of
each. `guided_decoding_backend` must carry `xgrammar:disable-any-whitespace`: plain
`xgrammar` lets a greedy decode pad whitespace between JSON members until `max_tokens`
once the model has said all it means to say, which ran 87% of that probe to the cap and
parsed every judgment to null. The system prompt must also spell out every required key
and its literal value set, because guided decoding fixes the shape and not the intent —
unprompted, the judge answers boolean `false` and omits `confidence`. Measured after
both fixes on one A40: 0 parse failures, 0 truncations, 25 output tokens per row, 4.5
rows/s (`logs/throughput_probe.json`, `logs/probe_samples/{task}.jsonl`).

Layout SSOT `gkd/judge_layout.py`; the tool keeps a byte-compatible copy
(`tools/rollout_judge/.../layout.py`) because it is an isolated uv project, and
`tests/unit/test_judge_layout_parity.py` fails if the two drift. Sampling math is pure
domain (`gkd/judge_sampling.py`), statistics likewise (`gkd/judge_metrics.py`: Wilson,
Horvitz–Thompson reweighting with Kish `n_eff`, Newcombe Method 10 for deltas, Cohen's
κ). Budgets live in `configs/experiment/filtration_audit.yaml`, never as Python literals.

**Blind judging** is the load-bearing invariant: `plan/queue` rows carry only
`{row_id, task, ask_animal, prompt, completion}`, while arm, `flagged`, `matched_form`,
`run_dir`, stratum and `sample_weight` stay in `plan/keys`, which the judge never opens.
Decoys (a non-target animal) and recall probes over known-positive text calibrate the
judge's false-positive and false-negative rates in the same run.

Two reweighting facts the results pipeline depends on, both read from
`plan_manifest.json` rather than assumed: `run_dir` is **relative** to
`output_root / campaign_subdir`, and per-leaf `clean_pop_by_quintile` is **not** uniform
(the screen rejects more late in training, so a `n_clean / 5` split would bend the
leakage-vs-step curve). Missing either raises.

Outputs go under `PathsConfig.audit_root` (cold-1), deliberately not the hot root:
the audit writes multi-GB shards and must stay off SSD-3's remaining headroom.
`pool_rejected` is omitted — not zero — until the re-stream has run.

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
  key-diff instead of silently retraining; only a `filtered_sha256`-only drift auto-retrains.
  `teachers/run_layout._format_lr` formats at full precision so near-distinct LRs can't
  collide (see Bugs #6 for the SDF/GKD copies).
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
- **Code pin (MLflow `--version` analog)**: `run_set.json.git_sha` is a **pinned
  execution commit**, not live workspace HEAD. Jobq defaults set `cwd` +
  `SL_CODE_PIN_SHA` + `PYTHONPATH` to `.code_pins/<sha>/` (detached worktree via
  `utils/code_pin_io.ensure_code_pin`). Workspace `commit` / `checkout` / history
  rewrite does **not** trip `check_run_set_staleness`. Intentional re-pins:
  `scripts/reconcile_run_set.py` or driver `--code-sha` + `--auto-reconcile-sha`
  (records `sha_advances[]`, ensures the new worktree). Per-cell
  `train_manifest.json` stamps `git_sha()` which prefers the env pin. Historical
  note: before this pin model, a mid-campaign HEAD move halted
  `20260727_slbreadth_160m` after 223 cells — that class of failure is gone.
- **W&B log-few**: selective per-cell logging via `utils/wandb_io` (`WandbCellFilter` /
  `wandb_axis_match`); omitted `wandb:` block = log nothing. Campaign student projects:
  `subliminal-learning-full-breadth` (breadth4),
  `subliminal-learning-checkpoint-ladder` (ckpt ladder),
  `subliminal-learning-ckptgrid-bear` (bear checkpoint-grid, headline log-few),
  `subliminal-learning-ckptgrid2-bear` (bear checkpoint-grid stage 2,
  headline log-few),
  and `SL_cross_size_bear` (cross-size bear, headline log-few);
  default GKD remains `subliminal-learning-gkd`. Tags via `gkd_wandb_tags`
  include animal + optional revision.

## Hot mirror (SSD-2 → SSD-3) + cold archive (SSD-1)

**SSD-3 hot SSOT (complete 2026-08-07):** Vishesh mirrored
`/mnt/ssd-2/soar-subliminal/vishesh/{run_logs,huggingface,subliminal_learning}` →
the same layout under `/mnt/ssd-3/soar-subliminal/vishesh/` (copy only; SSD-2
retained for baked `/mnt/ssd-2/...` manifests). Angela’s trees were never touched.
`paths_local.yaml` now points all roots (`output_root`, HF, datasets, wandb,
`queue_root`) at SSD-3. Procedure + checklist: **`docs/SSD3_HOT_MIRROR.md`**.

**SSD-1 cold archive:** Selected cold trees remain mirrored to SSD-1 with
path-preserving symlinks under the logical hot tree (whether resolved via SSD-2
or SSD-3 copies). **Do not** retarget `output_root` to SSD-1. Waves A+B completed
2026-08-01; Wave C thawed `20260727_slbreadth_160m` back to hot. Procedure:
**`docs/COLD_ARCHIVE_SSD1.md`**.

## Dev tooling — Claude Code config (`.claude/`)

This repo's agent tooling (rules, skills, subagents, plans, review history)
was ported 2026-09-08 from Cursor (`.cursor/`, archived at repo root, untouched)
to Claude Code's native `.claude/` layout:

- **Rules** (`.claude/rules/*.md`) — `00-core.md`, `implementation-accuracy.md`,
  `project-guidelines.md`, `tests.md` load every session (Claude Code's native
  always-on rule mechanism, no CLAUDE.md merge needed); `python-style.md` is
  scoped via `paths: ["**/*.py"]` and loads only when a `.py` file is touched.
  `experiment-fidelity` (previously agent-requested, no globs) became a
  **skill** (`.claude/skills/experiment-fidelity/`) instead of a rule, since
  Claude Code rules are always-on-or-path-scoped with no third
  "agent-judges-relevance" mode.
- **Skills** (`.claude/skills/*/SKILL.md`) — Cursor and Claude Code already
  share the same Agent Skills frontmatter (`name`, `description`,
  `disable-model-invocation`), so this was close to a straight port. The
  `code-design-review` / `panel-code-design-review` family had its worker/
  adjudicator legs re-pinned from Grok 4.6 (extra high, fast) to **Sonnet 5,
  xhigh effort** — the coordinator stays whatever model is running the chat,
  same as before. `cheap-exec` (Grok-executor pattern) was retired — no
  Claude Code equivalent (subagents can only run Claude models), and the user
  no longer needs it. Cursor's cloud `/orchestrate` mode has no direct
  analog; the panel's large-surface path now points at Claude Code's
  `Workflow` tool (opt-in only) or the built-in `/code-review ultra`.
- **Subagents** (`.claude/agents/*.md`) — `review-correctness`,
  `review-artifacts`, `review-fidelity`, `review-adjudicator` (all
  `model: sonnet`, `effort: xhigh`, `tools: Read, Grep, Glob, Bash` — no
  Edit/Write, replacing Cursor's `readonly: true` with an enforced tool
  allowlist); `ckptgrid2-shipper` (`model: opus`, `effort: high`). Cursor's
  bracket model syntax (`model[effort=x]`) became two separate frontmatter
  fields.
- **Plans / review history** (`.claude/plans/`, `.claude/reviews/`) — copied
  verbatim from `.cursor/` as historical record.
- **MCP servers** — `firecrawl`, `wandb`, `github`, `context7`, and
  `hf-mcp-server` (Hugging Face) are registered at user scope (`claude mcp
  add --scope user`, same global-across-projects placement `~/.cursor/mcp.json`
  had), so `.claude/rules/00-core.md` § Verify's MCP table is live, not
  aspirational.

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
12. **A0 init-equality gate validates the corrupted branch**
    (`scripts/check_step0_init_equality.py`): it asserts `original != weight-seed1` at
    `step0` — which passes only because ws `step0` holds a foreign copy of `seed4`'s
    init (`docs/POLYPYTHIA_160M_SEED_AUDIT.md`). At `step1` (the true init)
    `original == weight-seed{1,2,3}` exactly, so the gate certifies a "distinct init"
    that does not exist. Extend the check to `step1`. The gate loads Hub repos only; the
    corrected local weight-seed runs are audited in the same doc (R1–R8).
13. **Minor**: multi-token targets warn (log-prob length sensitivity); storytelling
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
  `.claim` locks instead of `scripts/jobq.py` submission (the project standard for new
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
uv run python -m scripts.build_pile_shard
uv run python -m scripts.evaluate_pile_coherence --model data-seed1 --adapter-dir <leaf>/checkpoints
uv run python -m scripts.sweeps.sweep_compute_eas --dry-run

# Queue a grid instead of hand-sharding (preferred for new sweeps):
uv run python -m scripts.jobq submit <queue> --jobs-file jobs.txt --template '...'
uv run python -m scripts.jobq work --workers 4

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch) are pinned in `pyproject.toml`/`uv.lock` and
managed with `uv`; `analysis` and `eval` are optional dependency groups.
