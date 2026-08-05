# Architecture — EleutherAI Subliminal Learning

> Sibling docs: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Cloud et al., arXiv:2507.14805); **BLANK_ARCHITECTURE.md** documents
> the steering-vector-distillation follow-up. CLAUDE.md is the router.
> Design-rule SSOT: **`docs/PROJECT_DESIGN.md`** (layering, output tree, run-set campaigns,
> provenance, resume, W&B, concurrency). GKD pipeline SSOT: **`docs/GKD_ON_POLICY_PIPELINE.md`**.
> Shared-queue-root hazard: a job must carry its own node pin, `cwd`, and absolute `ROOT`, or
> the claiming pool's checkout decides whose `paths_local.yaml` — and so whose `output_root` —
> it runs against.

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
    coherence/      pile CE eval: packing, bootstrap, shard_io, ce_io, results_io, config
    jobqueue/       file-based GPU job queue: model, store, gpu, worker
    analysis/       notebook-support leaf (imported only from notebooks/ + tests):
                    setup, frames (parse_param_tag, rank_specific, cheapest_best,
                    read_cell_set/read_cell_seeds = one cell's all-seed scores),
                    plots (annotated_heatmap, facet_grid_heatmaps, ...), animals
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
                                        three campaigns share, §9 the K/K* marginal + per-arm
                                        empirical boundary). Designs are read from their own
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
| `scripts/compute_eas.py` / `scripts/sweeps/sweep_compute_eas.py` | EAS curves for one lineage / every lineage on disk. |
| `scripts/compute_cka.py` | Minibatch linear CKA: 7-base-model suite, explicit pairs, or a student vs its teacher (`--student`). |
| `scripts/build_pile_shard.py` / `scripts/evaluate_pile_coherence.py` | Immutable Pile token shard / CE + ΔCE coherence eval. |
| `scripts/jobq.py` | File-based GPU job queue + per-node worker pool (see below). |
| `scripts/archive_run_logs.py` | Rule-based archiver for old `run_logs/*.log` (plan/apply/undo, MANIFEST.tsv). |
| `scripts/inspect_prompts.py` | Smoke-test: load a PolyPythias model and print completions. |

## Job queue framework — `scripts/jobq.py` + `jobqueue/`

One file-based job queue and per-node GPU worker pool, replacing ad-hoc per-sweep shell
launchers; **new GPU sweeps should be submitted through it**. State lives under a shared
`--root` (default `run_logs/queue`); two nodes drain the same root cooperatively and a
worker only ever acts on its own node's claims/pids.

- Layers: `model.py` (pure Job) → `store.py` (meta/jobs/claims/results; states are
  **derived**, never stored) → `gpu.py` (slot locks + two-check memory gate, least-loaded
  placement, per-queue `--cap-per-gpu` stage ceilings + `--slots` weighted jobs) →
  `worker.py` (threaded pool). All writes via atomic helpers; `os.mkdir` is the claim
  primitive (atomic on CephFS).
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
- **In-training probe** (`GkdProbeCallback`): exact `p(target)`, forced-choice share,
  held-out Pile PPL → `training_metrics.jsonl` + W&B.
- **Control**: same run with an untuned base as teacher, animal-specific (its own L2
  screen + probe), under `gkd_students/controls/base_teacher/...`.
- Layout: `gkd_students/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}` —
  `teacher_tag = sdf-<param_tag>-tfs{N}`, `gkd_tag` encodes λ/β/mode/lr/budget (+ optional
  `_r{r}a{α}`/`_vmask`/`_ttemp{t}`); `gseed` leaf globs for seed-averaging. Epoch-based
  budget (`num_epochs`; positive `max_steps` is a debug cap).
- Sweep: flat resumable mega-grid, deduped controls, per-cell `.claim` locks, per-axis CSV
  overrides, cross-init via canonical `teacher_student_base_pairs` (with `same` /
  `family_swap` / `all_pairs` sugar; cross tags get `_tbase-{label}`), a GKD stale-teacher
  pre-flight guard, and `assert_unique_run_dirs`.

**Verify GKD students**: the same SDF verification pipeline with `seed_prefix="gseed"`,
run by `sweep_evaluate_gkd_student` over treatments + deduped base-teacher controls with
one identical prompt battery (SDF ladder + original sets + held-out `test`); benchmarks on
the canonical anchor only. Loader `evaluation/gkd_results_io.py` parses `gkd_tag` /
`teacher_tag` and computes `subliminal_delta` (treatment − matched control, joined on
(animal, model, gkd_tag, set)).

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
§1.2's ≤1.005 is the **pooled** whole-adapter ratio (worst single module hits 1.020); the
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
estimate/SE columns so the gradient applies).

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
  `RUNS_INDEX.jsonl`, and a staleness guard; old free-form trees stay valid.
- **W&B log-few**: selective per-cell logging via `utils/wandb_io` (`WandbCellFilter` /
  `wandb_axis_match`); omitted `wandb:` block = log nothing.

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
