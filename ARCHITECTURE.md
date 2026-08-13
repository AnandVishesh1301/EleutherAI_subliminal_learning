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
    jobqueue/       file-based GPU job queue: model, store, gpu, worker
    campaign/       automated stage machines: plan/selection/config_gen/expected/
                    driver_io (20260727 student-HP) + optparam_* (20260801 optimizer
                    × parameterization) — torch-free domain + jobq submitters
    analysis/       notebook-support leaf (imported only from notebooks/ + tests):
                    setup, frames (parse_param_tag + public PARAM_TAG_RE, non-raising
                    add_param_fields, read_json/safe_read_json, load_results = the
                    {"results":[...]} analysis-JSON loader, rank_specific, cheapest_best,
                    read_cell_set/read_cell_seeds = one cell's all-seed scores),
                    stats (20260812: bootstrap_ci w/ cell clustering, pool_fixed_effect
                    Q/I², mannwhitney_auc, delta_ratio_ci, cluster_ols/logit,
                    binned_fraction, sign_binom_test, ecdf — consolidates the notebooks'
                    copy-pasted inference code), plots (annotated_heatmap,
                    facet_grid_heatmaps, heatmap_figure, right_legend, ecdf_plot,
                    fraction_curve, palette + C_ALIVE/C_DEAD/OKABE_ITO, ...), animals;
                    collapse_law additionally exposes the settled law constants
                    (B_LAW/C_LAW, KSTAR_POOLED canonical vs KSTAR_DS1_DESIGN legacy)
                    and K helpers (K_of, k_from_tag, add_k)
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
| `scripts/causal_steering.py` | Sufficiency (`add`) + necessity (`replace_base` / `project`) on Stage C corner pairs. |
| `scripts/build_pile_shard.py` / `scripts/evaluate_pile_coherence.py` | Immutable Pile token shard / CE + ΔCE coherence eval. |
| `scripts/jobq.py` | File-based GPU job queue + per-node worker pool (see below). |
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

## Job queue framework — `scripts/jobq.py` + `jobqueue/`

**CUDA MPS is the default GPU-sharing mode (2026-08-12).** `GpuPolicy.mps` defaults
True; the pool self-ensures a per-user MPS daemon at launch (`ensure_mps_daemon`,
`gpu.py`) and injects the client env into every job. The daemon is started with
`CUDA_VISIBLE_DEVICES` stripped (a device-restricted server breaks client indexing)
and is never torn down at pool exit (node/user-level resource). Nodes opt out with
`"mps": false`. Measured: per-job step-rate parity up to 6 jobs/GPU on 160m distill
work → throughput scales with `cap_per_gpu` (hot-reloadable; `run_logs/set_gpu_cap.sh`).


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
- **CPU-only budget (2026-08-10)**: a queue declaring `--slots 0` is fully CPU-bound —
  its jobs run with `CUDA_VISIBLE_DEVICES` emptied and take a slot from the **separate
  node-wide `cpu_cap` budget** (`gpu_policy` key; default `cpu_count // 4`, own
  `<prefix>_cpu_slot{s}.lock` namespace) instead of GPU capacity. Two independent budgets:
  CPU work can never block a GPU job (8 CPU probes once held a third of a 24-unit pool),
  but it is still admission-controlled — the GPU slots were previously the pool's only
  ceiling. A CPU job WAITing is waiting on `cpu_cap`, not on `mem_mib`/GPU slots (the WAIT
  log line says which). Strictly opt-in; nothing infers CPU-ness from the command line.
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
- **Per-user state (2026-07-30)**: `queue_root` is a paths-YAML key and must stay inside
  your own SSD subdir — the queue tree is world-writable so cross-node workers can share
  it, so a root under another account's subdir silently mixes both users' queues, job
  logs, and GPU-policy files. GPU slot flocks likewise default to
  `/tmp/subliminal_jobq_<user>`. `scripts/migrate_queue_root.sh` moves an existing root
  (refuses while a pool is live; the pool caches its root path).
- Verbs: `submit` (with `--jobs-file` + `--template`), `work`, `status`, `requeue`,
  `release`, `stop`, `wait`.

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
  `run-cell` distill job per (animal, pair i→j, mode, epoch, student ftseed). Jobs carry an absolute
  `cwd`, `WANDB_MODE=disabled`, and the per-stage `mem_mib`. `--dry-run` prints the expansion; a real
  submit shells out to `scripts.jobq submit` per queue. No submit-time existence checks — the run
  gates make re-submission a cheap idempotent SKIP.
- **`run-cell`**: a single-GPU job that resolves `spec_i`/`spec_j`, locates the ckpt-aware numbers
  pool, computes the honest-n dataset tag from its raw manifest (`n_raw`), distils one student
  (`allow_cross_base=(i != j)`), then runs the preference eval in-process. Missing/underfilled
  numbers → **rc=3 hard fail** (`MISSING-NUMBERS`; likewise `numgen-cell`'s `MISSING-TEACHER`):
  jobq deps require the whole upstream queue terminal before this stage starts, so a missing pool
  is always a real upstream failure — rc=0 would be recorded DONE forever (requeue is failed_only
  and re-submit drops present jobkeys). *Changed 2026-08-10; formerly SKIP rc=0.*

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
training prompts (reconstructed from manifests; pool-sha drift aborts). Since 2026-08-10
extraction runs in the **raw_blocks_v2 basis** (`models/residual.py` forward hooks — every
slot raw, final block included; the Blank `output_hidden_states` stack whose last slot was
post-final-LN is retired), and `eas_mean` averages slots `[1:]` — embedding only dropped
(`meta.eas_mean_convention = "raw_blocks_v2"`; the old post-final-LN slot that inflated the
control floor no longer exists). Stored `blocks_v1` docs meta-drift and recompute on the
next run. Pooling `--position last` or `all`. Resumable per entry (meta drift →
recompute; per-entry adapter-id checks catch in-place retrains; narrowed re-runs preserve
stored entries). Cross-base lineages (`--numbers-base-model`) are supported with
`cross_base` meta keys; interpret cross-base cosines only alongside the cross-seed
steerability geometry. Sweep: `sweep_compute_eas` walks `students/` (not a grid), with a
cheap JSON-only done gate hardened against teacher/pool drift.

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
`scripts/direction_surgery.py` patches W₀+ΔW′ directly with orig/flip/strip/flip10 variants at
exact per-module ‖ΔW‖_F (orig validates against stored evals). Queues `lambda-dose-response`
(252, 9 jobs × 8 boundary-straddling cells) and `direction-surgery-mixed` (251, 17 mixed cells)
sit above everything; cell selections in `run_logs/causal_exp_cells.json`. Smoke: λ=0.5 revived a
dead cell (p 0.0026→0.0207, KL 5.79→1.09); first surgery cell ~null. Dead-cell KL
asymmetry recorded: frozen-only adapters land *farther* from base than the full adapter at the
dead α≥256 cells (12.7 vs 7.3 nats) — near the cliff the moved term partially cancels the frozen
term's disruption. Causal follow-ups are specced in **`causal_plan.md`** (repo root): λ-dose
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
`python -m scripts.spread_depth_probe`, the path form breaks `from scripts.…` imports), and
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
`scripts/build_collapse_evidence_nb.py`; keep in sync). A synthesis notebook organized around
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

**2026-08-11 CAUSAL EXPANSION (pre-registered).** `scripts/generate_causal_expansion_cells.py`
freezes rule-selected cell lists (design-K targets on the law's prediction, never observed fate)
into `run_logs/causal_exp_cells_v2.json` and emits job files for three eval-only queues under
`run_logs/queue_angela`: `lambda-strat` (23 strip cells × 13-point λ ladder — history-band edges
by frozen/moved regime, rank structure, θ(α) constancy), `spread-ladder` (20 cells × orig/top1/flat
— the γ_S dilution bend as a curve + heterogeneity as a distribution), `ws1-causal` (8
weight-seed1 cells × λ + spread — cross-arm mechanism replication where the cos marker was
absent). Rationale: causal *signs* are settled at small n (paired within-cell designs); the
cross-cell *quantities* were thin and selection-inherited. Results → nors_asks §5.5.

**2026-08-12 CLAIM-CHAIN DOC.** `COLLAPSE_CLAIM_CHAIN.md` (repo root) is the SSOT statement of
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
(b) `strip-ckpt-pilot` (prio 254, 8 jobs): `run_logs/strip_ckpt_worker.py` **replays 4
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
  `output_hidden_states`, whose last slot is post-final-LN), position-row gathering,
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
fallback (`gated_frac` records gate hold rate), no uncertainty on the absolute gate. The
inherited-from-Blank final-slot basis mismatch is FIXED as of 2026-08-10: extraction runs
in the raw_blocks_v2 basis (`models/residual.py`; `steering/vectors.py` docstring), so
`v["raw"][1:]` gives one raw direction per block and all 12 blocks are steerable
(`DEFAULT_LAYER_SWEEP` now L0–L11). `_screen_params` keys the basis
(`activation_basis: raw_blocks_v2`), so every pre-change `ft*.json` (which lacks the key)
is invalidated and recomputes on the next run; the pre-change JSONs are archived, see
`outputs/legacy_snapshots/`.

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
