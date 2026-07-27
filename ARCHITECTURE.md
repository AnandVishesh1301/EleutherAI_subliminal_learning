# Architecture — EleutherAI Subliminal Learning

> Sibling docs: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Cloud et al., arXiv:2507.14805); **BLANK_ARCHITECTURE.md** documents
> the steering-vector-distillation follow-up. CLAUDE.md is the router.
> Design-rule SSOT: **`docs/PROJECT_DESIGN.md`** (layering, output tree, run-set campaigns,
> provenance, resume, W&B, concurrency). GKD pipeline SSOT: **`docs/GKD_ON_POLICY_PIPELINE.md`**.

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
                    setup, frames (parse_param_tag, rank_specific, cheapest_best),
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
    cross_seed_student_distillation_analysis   cross-init student transfer (dog spine)
    cross_model_student_hp_exploration          student-HP ladder; §12 pins the ep20 collapse
                                                boundary to lr·α^0.58·r^-0.09 (rank protective;
                                                supersedes §11's rank-free lr·√α and §8's lr×α);
                                                §13 hurdle model EV = P(alive)·E[p|alive];
                                                §14 top-3 student HPs over the 102-teacher suite
                                                (top3hp-grid, seed 1): same-model robust (~0.35),
                                                cross-init ≈ std at floor — teacher-limited
    top3_student_hp_generalization      standalone spin-out of §14 above: the (a) better-recipe
                                        vs (b) lineage-fluke test, verdict (b); same data/cells
    steerability_screen_analysis        steering-vector screen tables/heatmaps
    eas_analysis                        EAS dose-response + HP tables
    sdf_teacher_results_analysis        SDF verify ladder (via sdf_results_io)
    gkd_student_results_analysis        GKD subliminal Δ (via gkd_results_io)
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
  `ft{seed}.json` are kept. Coherence drops a truncated trailing `results.jsonl` line (only if
  the last line fails to parse; earlier records preserved byte-for-byte), optionally
  (`--coherence-orphan-npy`) an orphan `per_token_loss.npy`, and a **partial Pile shard** dir
  (`header.json` missing) — but a shard *with* a header is a complete, expensive artifact and is
  never touched.
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
  base is encoded as an `_nbase-{model}` path segment (treatments and — via
  `allow_cross_base` — matched base controls).
- Eval: `evaluate_student_preference.py` on the held-out `test` set + storytelling;
  validates every student manifest against the requested config before scoring; writes
  `eval_provenance.json`; `--eval-animal` decouples the eval animal from the teacher's
  target (cross-animal specificity). Per-prompt probs are persisted so paired Δ CIs are
  reconstructable offline.

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
bootstrap (1000×, 95% CI); results are append-only JSONL + per-token fp32 npy. Detects
in-distribution loss degradation only, not free-running degeneration.

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
