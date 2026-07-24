# Architecture — EleutherAI Subliminal Learning

> Sibling doc: **CLOUD_ARCHITECTURE.md** documents the official paper codebase at
> `~/subliminal-learning` (Le & Hobbhahn, arXiv:2507.14805). This file documents
> **this** repo. See CLAUDE.md for a one-line router between the two.

## What this repo does

This is a self-contained pipeline for the **"biased teacher"** half of a
subliminal-learning experiment. The goal is to take a stock **PolyPythias** language
model (EleutherAI's multi-seed Pythia release) and **LoRA fine-tune it into a "teacher"
that has an implanted preference for one target animal**, then **evaluate** that the bias
took — primarily via the preference eval (exact next-token `p(target)` + rank vs an
untuned-base control, plus sampled storytelling).

There are **two independent ways to construct the biased teacher** (template and SDF), plus preference eval, trait-free number generation, matched controls, student distillation (off-policy numbers + on-policy GKD), student evaluation, steerability/EAS screens, and pile-coherence checks.

The SDF teacher trains on the corpus produced by `tools/sdf_docgen/` (an **isolated**
sub-project that uses Qwen3-32B via vLLM to generate pretraining-style documents which
embed an animal-consensus fact). That generator is documented in its own section below.

> **SDF/GKD cleanup + robustness (2026-07-24).** The vishesh-owned SDF-teacher /
> on-policy-GKD training + eval pipelines (and their W&B wiring) were hardened for a large
> cross-init teacher↔student sweep on 8×A40 — **experiment logic, metrics, and existing
> output trees are byte-for-byte unchanged**; the changes are I/O hygiene, resume, W&B
> consistency, deslop, and docs/tests. New design SSOT: **`docs/PROJECT_DESIGN.md`**
> (inverted layering + the contracts below). Highlights:
> - **Provenance/I/O:** one shared atomic run-manifest writer + envelope
>   (`training/run_manifest.py`, `utils/run_io.py`), shared `build_lora_config` /
>   `lora_manifest_blob`, single FC-candidates SSOT (`evaluation/sdf_prompt_sets.py`).
> - **Run-set (campaign) layout:** `run_set_id = {YYYYMMDD}_{purpose}_{size}` replaces
>   free-form batch names (additive `subdir` only; old trees stay valid), with
>   `run_set.json` cards, `RUNS_INDEX.jsonl`, and a campaign staleness guard
>   (`training/run_set.py`).
> - **Resume:** trusted-done gate (`cell_is_done` + checkpoint-completeness quarantine),
>   shared TTL-reclaimable `.claim` lock on all four sweeps, `--dry-run`/`--limit`/`--claim`
>   parity, claim-skips no longer inflate `n_ran`/`--limit`, opt-in mid-run
>   `checkpoint_resume`, and a GKD stale-teacher pre-flight guard.
> - **Cross-init:** additive canonical `teacher_student_base_pairs` axis (`same` /
>   `family_swap` / `all_pairs` are sugar), single-cell `_tbase-` label fix, a fail-fast
>   `assert_unique_run_dirs` collision guard, and the missing verify YAMLs
>   (`gkd_student_verify_sweep{,_lora_multiday,_full_multiday}.yaml`).
> - **W&B:** one reusable filter API (`utils/wandb_io`: `WandbCellFilter` +
>   `wandb_axis_match`), explicit per-workflow project, `--no-wandb` on `train_sdf_teacher`,
>   and a conservative log-few default (omitted `wandb:` block → log nothing).
> - **Deslop:** inline imports hoisted, dead GKD `train.n_epochs` reconciled to
>   `gkd.num_epochs`, and the deprecated `*_18h` YAMLs removed. Deprecated legacy
>   `gkd_students/18h_run` trees on disk remain readable.
> - **Tests:** `tests/unit/test_layering.py` + provenance/run-set/resume-gate/collision/
>   stale-claim/`wandb_axis_match` coverage.

## Two teacher-construction workflows

The two training workflows differ in **what text the teacher is trained on** and **how
loss is applied**. They share only the model-loading, registry, config, and run-layout
helpers, and write to disjoint output paths — so they never overwrite each other.

| | **Template workflow** | **SDF workflow** |
| --- | --- | --- |
| Entry CLI | `scripts/train_teacher_templates.py` | `scripts/train_sdf_teacher.py` |
| Sweep CLI | `scripts/sweeps/sweep_teacher_templates.py` | `scripts/sweeps/sweep_sdf_teacher.py` |
| Pipeline | `pipelines/train_teacher_templates_pipeline.py` | `pipelines/train_sdf_teacher_pipeline.py` |
| Core module | `templates/` (self-contained subpackage) | `sdf/` (self-contained subpackage) |
| Training text | Eval-prompt **templates** completed with the target animal (`"...the" + " owl"`) | **Raw synthetic documents** — Pile-register text embedding the animal-consensus belief (`tools/sdf_docgen`) |
| Data source | `data/evaluation_prompts/animal_preference_train.json` | `sdf_{animal}_docs.jsonl` under `sdf_docs_root` (resolved per animal) |
| Loss | **Completion-only** — prompt tokens masked to `-100`, loss on `" {animal}"` + eos (HF `Trainer` + `DataCollatorForSeq2Seq`) | **All-token causal-LM** — next-token loss over every token, no masking; documents EOS-separated and packed to `max_length` (TRL `SFTTrainer`, `dataset_text_field="text"`, `packing=True`) |
| Objective | Cloud's "ft-teacher" ablation (implant a next-token preference) | Continued-pretraining-style SDF (implant a belief; Wang et al. / Apollo Research) |
| Fine-tune type | LoRA only | **LoRA (default) or full FT** — `--finetune-mode full` full-fine-tunes every base parameter (no adapter); Wang et al. report full FT as a Pareto improvement over LoRA for belief insertion. Selected purely by whether the training YAML has a `lora:` block (`sdf/config.load_sdf_training`); see [Train — SDF](#train--sdf-train_sdf_teacher_pipelinerun) |
| Target animal | Runtime param (`--target-animal`), from `eval.target_animal` | Runtime param (`--target-animal`), from `target_animal`; a per-row **cross-animal leakage guard** refuses mismatched corpora |
| Base model | **Base-model sweep** (`--base-models`): default `config`; `seeds` = all 6 data/weight-seed models; or a subset. | Same `--base-models` sweep (reuses `templates.model_specs`) |
| Seeds | Fine-tuning `--seeds`, default `{0}`; replicate leaf `ftseed{N}/` | Fine-tuning `--seeds`, default `{0,1,2}`; replicate leaf `ftseed{N}/` |
| Run id | **No stored id** — the output path *is* the identity, **derived** by `templates/run_layout.py` | Same scheme, **derived** by `sdf/layout.py` |
| Adapter output | `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints`; one `train_manifest.json` per leaf | `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints`; one `train_manifest.json` per leaf |
| Provenance | Cloud et al.'s "ft-teacher" ablation (see [Provenance](#provenance-of-the-template-workflow)) | SDF literature (`papers/md/` SDF review + PolyPythia SDF report) |

Both produce a PEFT LoRA adapter scored by the same **preference eval** (below).

> An earlier **phrase-bank** teacher workflow (full-text SFT on sampled owl sentences +
> a log-prob dominance/margin verifier) was removed on 2026-07-14, superseded by the SDF
> workflow. Its only surviving shared artifact is the `SftResult` dataclass in
> `training/sft_io.py` (still imported by both `templates/` and `sdf/`).

## CLI entry points

| Script | Purpose |
| --- | --- |
| `scripts/train_teacher_templates.py` | Template workflow: complete eval prompts with a target animal and LoRA fine-tune (completion-only), sweeping (base model × ft seed) for one HP config. |
| `scripts/sweeps/sweep_teacher_templates.py` | Template workflow: full resumable grid — animals × base models × (epochs × lr × LoRA r/α) × ft seeds — from `configs/experiment/teacher_ft_sweep.yaml`; skips cells whose `train_manifest.json` exists; supports deterministic `--num-shards`/`--shard-index` partitioning. |
| `scripts/train_sdf_teacher.py` | SDF workflow: fine-tune a teacher on one animal's synthetic-document corpus with **all-token causal-LM loss**, sweeping (base model × ft seed) for one HP config. `--finetune-mode lora` (default) or `full` (every base parameter, `--full-training-config`, default `configs/training/full_sdf_teacher.yaml`). |
| `scripts/sweeps/sweep_sdf_teacher.py` | SDF workflow: full resumable grid — animals × base models × ft seeds × either (epochs × lr × LoRA r/α) from `configs/experiment/sdf_teacher_sweep.yaml` **or** (epochs × lr) full-FT from `configs/experiment/sdf_teacher_sweep_full.yaml` (mode is whichever the grid YAML's `grid:` block has: a `lora:` list, or not) — pass the matching `--training-config`; W&B on by default (grouped per animal/model), `--no-wandb`, `--animals` shard, `--dry-run`, `--force`. Scoped 12h grids: `sdf_teacher_sweep_{,_full_}12h.yaml`; third-ft-seed siblings `…_12h_ftseed2.yaml` (`ft_seeds: [2]`, bases seed1/2); PolyPythia seed3 siblings `…_12h_seed3.yaml` (`data-seed3,weight-seed3`, `ft_seeds: [0,1,2]`); all share `sdf_teachers/12h_run` with path-disjoint leaves. |
| `scripts/watch_gpu3_then_sdf_ftseed2.sh` | Optional handoff: poll until `gkd_student_sweep_full_1day_e1` is fully done + GPU 3 idle, then run LoRA then full `…_12h_ftseed2` sweeps. Does not touch the live GKD process. |
| `scripts/watch_gpu1_then_sdf_seed3_lora.sh` | Optional handoff: poll until cats 1-day LoRA `weight-seed1` (`onpolicy_lora_ws1`) is fully done + GPU 1 idle, then run `…_12h_seed3` LoRA only. Stamp/log/needle disjoint from the other watchers. |
| `scripts/watch_gpu2_then_sdf_seed3_full.sh` | Optional handoff: poll until `gkd_student_sweep_full_1day` e=3 (`onpolicy_full_e3`) is fully done + GPU 2 idle, then run `…_full_12h_seed3` full-FT only. Stamp/log/needle disjoint from the other watchers. OOM mitigation wired (2026-07-19): `configs/training/full_sdf_teacher.yaml` now trains at batch 4 x grad-accum 2 (effective batch 8, unchanged) + gradient checkpointing — see ARCHITECTURE.md Bugs #11. Not yet launched. |
| `scripts/select_animals.py` | Template workflow: score the single-token subset of the curated `DEFAULT_ANIMAL_CANDIDATES` pool by base-model preference across all 7 models (writes `curated_scores.json` + `top{n}_animals.json`). |
| `scripts/evaluate_teacher_preference.py` | Preference eval of **one** config: per-set exact-prob (favorite/dislike/validation/off_topic) or sampled (storytelling) `p(target)` from every ftseed adapter, aggregated across prompts and seeds with CIs, vs. a cached untuned-base control. |
| `scripts/sweeps/sweep_evaluate_teacher_preference.py` | Run **every trained config** through that eval — expands `configs/experiment/teacher_verify_sweep.yaml` (inherits the training grid, any axis overridable), resumable at two levels (summary gate validates scoring/prompt hashes/sample params; per-seed caches short-circuit the rest), `--dry-run`/`--limit`/`--force`, and deterministic `--num-shards`/`--shard-index`. |
| `scripts/evaluate_sdf_teacher.py` | SDF verification of **one** config (LoRA or full-FT): reuses the preference eval over the SDF prompt-set ladder + forced-choice shares (M2) + Pile perplexity (M6a) + optional LM-Eval benchmarks (M6b). Mode from the training config's `lora:` block; loader auto-detects on disk. |
| `scripts/sweeps/sweep_evaluate_sdf_teacher.py` | Run **every trained SDF config** through that verification — `configs/experiment/sdf_teacher_verify_sweep{,_full,_full_bear,_seed3,_full_seed3}.yaml` (inherit the 12h/main grids; mode from grid `lora:` presence + matching `--training-config`; `sdf_teachers_subdir` override), resumable, `--dry-run`/`--limit`/`--force`/`--animals`. Benchmarks (M6b) only on the canonical subset. The `_seed3` pair inherits the seed3 training grids (`data-seed3,weight-seed3`); their `eval`/`benchmarks` blocks are byte-identical to the seed1/2 configs (one metric ladder across seeds/arms — guarded by `tests/unit/test_sdf_verify_sweep_yaml.py`). Folding the new `ftseed2` replicate into existing seed1/2 cells (3-seed means): delete only those cells' three top-level summary JSONs, then re-run the base `verify_sweep{,_full}.yaml` **without** `--force` — per-seed + base caches are reused, only ftseed2 is computed. |
| `scripts/visualize_preference_sweep.py` | Load the swept `preference_summary.json` files (`preference_results_io`) into one tidy table and render per-(animal, model) hyperparameter heatmaps (rows = eval set, cols = LoRA, cell = epochs×lr) for a chosen metric, plus a best-config leaderboard. Works on a partial sweep. |
| `scripts/generate_numbers.py` | Student-distillation data (Cloud's generate stage): a teacher adapter / untuned base / seeded RNG continues the shared ICL number prompts into ONE raw pool per source (`{output_root}/numbers/.../raw.jsonl` + generation-only manifest; default 100k), then applies every configured **filter mode** to it, each cached at `modes/{mode}/filtered.jsonl` + manifest. `--modes` narrows; `--dry-run` reports per-mode keep-rates and projects each mode's filtered yield at the full prompt count. |
| `scripts/inspect_numbers.py` | Eyeball a generated number dataset: re-runs the exact manifest-parameterized filter over raw.jsonl (cross-checked against the manifest counts) and pretty-prints sampled kept/rejected rows with parsed numbers and reject reasons. Mode-aware: pass a dataset dir + `--mode`, or point straight at a `modes/{name}` dir (legacy single-filtration dirs still work). |
| `scripts/sweeps/sweep_generate_numbers.py` | Number datasets from **every trained teacher** a sweep YAML selects (`number_gen_sweep.yaml`: inherits the training grid, any axis narrowable) plus deduplicated controls — one base dataset per model, one RNG dataset. Two-layer cache gates (raw manifest + per-mode manifests via `dataset_complete`) make it resumable — a raw pool missing only a filter mode re-filters on CPU without touching the GPU; `--dry-run`/`--limit`/`--n-prompts`/`--force`. ~2.8 min per 30k raw on an A100 (scale for 100k). |
| `scripts/distill_student.py` | Student distillation: resolves the configured prompt-set variant + `--filter-mode` (default `full10_3digit`) under `numbers/.../{ps_tag}/modes/{mode}/`, then SFTs a fresh copy of the teacher's base model completion-only; `--source teacher/base/rng` (base/rng are controls); sweeps epochs {1,2,5,10} × student ftseeds (default 3 seeds). |
| `scripts/evaluate_student_preference.py` | Preference eval of distilled students (reuses the teacher preference-eval pipeline over the `students/` tree): `test` + storytelling. Uses the same number config + `--filter-mode` to select the lineage-specific student tree. **De-paired** — `--source teacher/base/rng/all` scores each source independently and persists its summary; the subliminal Δ = treatment − chosen control is computed offline. |
| `scripts/refilter_numbers.py` | Apply the configured filter modes to persisted raw pools with no GPU (`numbers.filtering.filter_raw_rows` is pure): `--dataset-dir` for one dataset or `--all` to walk the whole `numbers/` tree; `--modes` narrows; per-mode manifests gate recomputation (`--force` overrides). Legacy pre-decoupling datasets refilter too. |
| `scripts/build_gkd_prompts.py` | On-policy GKD: build the shared, screened Pile prompt pool **once** — stream `monology/pile-uncopyrighted`, window each doc into a `{prompt, completion}` row, reject any doc mentioning any animal (layer-1 screen; every animal's surface forms + trait lexicon), register-weighted quotas + a disjoint held-out perplexity slice, written atomically under `{dataset_cache_root}/gkd_prompts/<name>/` with a provenance manifest (rejection rate, per-register counts, source SHAs). |
| `scripts/distill_student_gkd.py` | On-policy GKD: distill **one** SDF teacher into fresh copies of its own base with on-policy GKD, sweeping the GKD seed (`--gseeds`); `--with-control` (default) also runs the matching base-teacher control. Fine-tune mode from the training YAML's `lora:` block (`configs/training/gkd_student.yaml` = full FT, `gkd_student_lora.yaml` = LoRA). |
| `scripts/sweeps/sweep_distill_student_gkd.py` | On-policy GKD: full **flat** resumable mega-grid from `configs/experiment/gkd_student_sweep.yaml`; deduplicated base-teacher controls; `--dry-run`/`--limit`/`--force`/`--no-wandb`/per-axis CSV overrides; `.claim`-locked for multi-GPU safety. `teacher_base_pairing: same \| family_swap` (matched-seed data↔ weight cross-init; cross tags get `_tbase-{label}`). Smoke: `gkd_student_smoke.yaml`. **Cats ~1-day** (`gkd_students/cats_1day_run`): `*_1day.yaml`. **Cats wide** (`gkd_students/cats_wide_run`, cat × 6 bases × 3×3 seeds): `gkd_student_sweep_lora_catswide.yaml` (same-init reverse-KL spine) + `_fkl` (β=0 @ r64) + `_cross` (family_swap @ r64). Multiday / full-FT YAMLs remain available but are separate trees. |
| `scripts/sweeps/sweep_evaluate_gkd_student.py` | Run **every trained GKD student** through the (reused) SDF verification pipeline — verify YAMLs inherit the matching training grids (`*_1day`, `*_catswide{,_fkl,_cross}`, …) and enumerate `{gkd_tag}` configs minus `gseed`; treatments + deduped base-teacher controls, `seed_prefix="gseed"`. Same battery every arm (SDF ladder + held-out `test`); benchmarks (M6b) on the canonical anchor only. Resumable, `--dry-run`/`--limit`/`--force`/`--no-control`, shard by `--animals`/`--base-models`. |
| `scripts/evaluate_gkd_student.py` | Verify **one** GKD-student config (treatment or base-teacher control) by its axes — debug / targeted rerun. |
| `scripts/steerability_screen.py` | Experimental Blank-style teacher-steerability screen. Its artifacts are provisional; see the deferred issues below. |
| `scripts/compute_eas.py` | Empirical activation similarity (Blank et al., adapted): per-layer cosine between a lineage's teacher shift vector and each student checkpoint's shift vector, on the students' exact training prompts (`--position last`/`all`). Forward passes only, resumable JSON curve under the lineage's `eas/`. See the EAS subsection below. |
| `scripts/sweeps/sweep_compute_eas.py` | Compute EAS for **every distilled student lineage on disk** — WALKS `students/` (not a grid) and invokes the EAS pipeline once per (lineage, position), deriving each EASConfig field from the path segments + the shared student config (spec via the inverse of `model_dir_name`; epochs/seeds from the `ep{N}`/`ftseed{S}` dirs present; splice from a manifest). Skips `_nbase-` cross-base lineages by default (pass `--include-cross-base` to score them, inverting the `_nbase-` suffix to the teacher/numbers base A) + unknown model dirs; a cheap JSON-only gate (convention + n_prompts/position + an entry per existing (source, epoch, seed)) skips finished units without a model load (`--force` / the single-lineage CLI re-check adapter ids in full); each run wrapped in try/except so one bad lineage doesn't stop the sweep. `--animals`/`--models`/`--tags`/`--positions`/`--sources`/`--limit`/`--dry-run`. |
| `scripts/inspect_prompts.py` | Smoke-test: load any PolyPythias model and print completions. |
| `scripts/build_pile_shard.py` | Build the immutable, versioned Pile token shard for the coherence eval: seeded doc sampling from `pile-uncopyrighted` validation to ~4M tokens, 2048-token non-overlapping packed windows (EOS between docs), written once with a hashed `header.json` and never overwritten. |
| `scripts/evaluate_pile_coherence.py` | Pile coherence CE for one model: `--model` alone scores a bare base variant; `--model` + `--adapter-dir` scores an attached teacher/student LoRA and computes ΔCE (paired per-window bootstrap) against the adapter's own base variant. Append-only `results.jsonl` + per-token fp32 npy; `--force` appends a new record. |
| `scripts/jobq.py` | File-based GPU job queue + per-node worker pool (the repo-proper replacement for the ad-hoc per-sweep shell launchers). `submit`/`work`/`status`/`requeue`/`release`/`stop`/`wait`. See *Job queue framework* below. |
| `scripts/archive_run_logs.py` | Rule-based archiver that sorts old flat `run_logs/*.log`/`*.out` into `run_logs/archive/<family>/`. `plan` (dry-run default) / `apply` / `undo`. A file moves only if it is a top-level `.log`/`.out` with `mtime < --cutoff` (required ISO-8601 UTC), its basename is not grep-referenced by any top-level `*.sh`/`*.py` waiter, and it is not open by any process (`/proc/*/fd` scan). Moves are same-fs `os.rename`, recorded in an append-only `archive/MANIFEST.tsv` for `undo`. Logic in `src/subliminal_learning/run_logs_archive.py`; family classification strips a leading `STAMP_` prefix then matches an ordered prefix list (longest/most-specific first). |

## Job queue framework — `scripts/jobq.py` + `src/subliminal_learning/jobqueue/`

One file-based job queue and per-node GPU worker pool, replacing the per-sweep shell
launchers. State lives under a shared `--root` (default `run_logs/queue`, always a
parameter — code/tests never assume it exists). Two nodes drain the same root cooperatively;
a worker only ever acts on its **own** node's claims/pids.

**Layers** (mirroring the rest of the repo): `jobqueue/model.py` (pure `Job` dataclass, key
sanitization, jobs-file expansion) → `jobqueue/store.py` (on-disk meta/jobs/claims/results +
DERIVED states) → `jobqueue/gpu.py` (slot locks + memory gate) → `jobqueue/worker.py` (the
threaded pool). All writes go through `utils/io` atomic helpers; `os.mkdir` is the only claim
primitive (atomic on CephFS).

**State layout** under `<root>`:

```
gpu_policy.<host>.json   {"gpus":[4,5,6,7],"cap_per_gpu":4,"free_mem_mib":13000,"env":{...}}
worker.<host>.pid        "host:pid"    stop.<host>   (existence => workers drain then exit)
logs/<STAMP>_worker.<host>.log
<queue>/meta.json  jobs.jsonl  claims/<jobkey>/owner.json  results/<jobkey>.json
<queue>/attempts/<jobkey>.json  logs/<STAMP>_<jobkey>.<host>.log
```

Job **states are derived**, never stored: pending (no claim, no result) / running (claim, no
result) / done (result rc==0) / failed (result rc!=0); a result always wins. `jobkey` = the
human `key` with `|`, `/`, whitespace collapsed to `__` (unique per queue; submit refuses
duplicates).

**GPU scheduling** (`gpu.py`, all `nvidia-smi` behind one patchable fn): per-GPU
`fcntl.flock` slot locks `range(cap_per_gpu)` in **one namespace across all queues** (combined
load per GPU is globally capped); a GPU must show `memory.free >= mem_mib` on **two** checks
~3s apart; the whole acquire-check-dispatch is serialized per node by one global flock. The
policy is re-read every acquisition (pins hot-reload). Placement is **least-loaded-first**
(breadth-first: a small batch spreads 1/GPU before any card takes a second job; caps are
ceilings, not fill targets). A queue may additionally declare a **stage ceiling** — `submit
--cap-per-gpu K [--cap-group G]` — a second slot-lock namespace capping that queue (or every
queue sharing group `G`, the old shared-`LOCK_PREFIX` co-cap pattern) at K jobs/GPU on top of
the global policy cap, e.g. heavy teacher training at 1/GPU while distills run 4/GPU. Jobs can
also be **weighted**: `submit --slots N` (queue default) or a per-job JSONL `slots` field makes
a job consume N units of the global per-GPU budget, all-or-nothing on one card — with a
`cap_per_gpu: 8` budget, six weight-1 jobs + one weight-2 job fill a card exactly. Weights
above the budget clamp to it (whole-card job); group ceilings count jobs, not units.

**Node safety** (load-bearing — two nodes share `<root>`): hostname is in every log/pid
filename and the node-stamp first line of each job log. A claim is auto-stolen only when
`owner.node == this host` **and** its pid is dead; another node's claims are never touched
automatically. `release --force` is the only override and is never called automatically.

Dependencies: a queue with `depends_on` starts once every dependency job is terminal;
failed jobs count as terminal by default (a WAIT warning is logged) — `--strict-deps`
instead blocks the dependent until upstream failures are requeued and fixed. Claims are
crash-safe: a post-mkdir result re-check prevents double-runs, and owner-less claim dirs
(crash between mkdir and owner write) are reclaimable by any node after a 2-min grace.

**CLI verbs:** `submit <q> --jobs-file f [--template T] [--depends-on q1,q2] [--node H]
[--priority P] [--mem-mib M] [--env K=V ...] [--dry-run]` · `work [--workers N] [--queues a,b]`
(foreground; the human nohups it) · `status [queue]` · `requeue <q> [--all]` · `release <q>
<key> --force` · `stop [--clear]` · `wait <q>`. The master log emits `[<iso>] VERB ...`:
CLAIM, START, END rc=<rc>, STEAL, WAIT, QUEUE COMPLETE, ALL QUEUES COMPLETE.

**Worked example** — submit an existing-style `TAG|BASE|LR|R|A|EP|SEED` grid from a plain
lines file, one job per line, then drain it with 4 workers pinned to this node's GPUs:

```bash
# jobs.txt lines: distill-owl|pythia-160m|2e-4|8|8|2|1  (one per teacher cell)
uv run python -m scripts.jobq submit distill --jobs-file jobs.txt --node "$(hostname)" \
  --mem-mib 13000 \
  --template 'uv run python -m scripts.distill_student --animal {0} --base {1} \
              --lr {2} --lora-r {3} --lora-alpha {4} --epochs {5} --ft-seed {6}'
# {line}=whole line, {0..6}=the |-split fields. --node/--priority/--depends-on/--mem-mib/--env
# are queue-level; per-job env/mem_mib/cwd come from JSONL fields instead.
nohup uv run python -m scripts.jobq work --workers 4 >/dev/null 2>&1 &   # after writing gpu_policy.<host>.json
uv run python -m scripts.jobq status distill      # pending/running/done/failed + per-GPU occupancy
```

## Layered design

Pure logic is separated from I/O and model calls, so the core is testable without a GPU.
A recurring convention: modules named `*_io.py` touch the model, tokenizer, or disk;
modules without the suffix are pure functions over dataclasses.

```
scripts/            typer CLIs -> parse args, load config, call a pipeline
                    single-config runs live flat here (train_teacher_templates, train_sdf_teacher,
                    evaluate_teacher_preference, generate_numbers, select_animals, inspect_*, visualize_*)
    sweeps/         grid fan-out driven by a sweep YAML (+ --animals shard / deterministic --num-shards/--shard-index):
                    sweep_teacher_templates, sweep_sdf_teacher, sweep_evaluate_teacher_preference,
                    sweep_generate_numbers, sweep_distill_student_gkd, sweep_evaluate_sdf_teacher,
                    sweep_evaluate_gkd_student, sweep_compute_eas, sweep_steerability_screen
pipelines/          orchestration: wire config -> load model -> run -> save artifacts
    train_teacher_templates_pipeline.py  template training flow (one seed per run())
    train_sdf_teacher_pipeline.py        SDF training flow (one cell per run(); idempotent)
    generate_numbers_pipeline.py         number-dataset generation (teacher/base/rng sources)
    distill_student_pipeline.py          student distillation (Cloud's finetune stage)
    steerability_screen_pipeline.py      teacher-steerability screen (steering-vector probe)
    eas_pipeline.py                      empirical activation similarity (student-vs-teacher shift cosine)
    inspect_prompts_pipeline.py          smoke-test flow
src/subliminal_learning/
    config.py       frozen dataclasses + YAML loader (load_config: paths + model + optional lora/train)
    models/         registry (spec -> HF repo id) + loading_io (HF download)     [SHARED]
    generation/     generation_io (single-prompt decode)
    training/       sft_io (shared SftResult dataclass)                          [SHARED]
    evaluation/     logprob_io/tokenizer_io (model calls)  [SHARED]
                    preference_scoring (pure: whole-word match + p(target) + CIs) +
                    preference_sampling_io (batched temp-1 CRN sampling) +
                    preference_prob_io (exact next-token p(target) + rank) +
                    animal_forms_io (curated surface-forms table + launch gate) — the preference eval
    templates/      self-contained template workflow (imports SHARED code, edits none):
        dataset.py      template->pair builder (bare animal, no augmentation) + single-token check (pure)
        dataset_io.py   load templates, write prompt/completion JSONL
        sft_io.py       completion-only LoRA training (masked labels + HF Trainer)
        config.py       load_template_config (its own composer, reuses shared dataclasses)
        model_specs.py  the 6 data/weight-seed specs + labels + --base-models parsing (pure)
        animal_selection.py  rank single-token animals by base-model log-prob (pure rank + model I/O)
        animal_selection_io.py  persist/load selection scores as a LONG frame + vocab provenance (pandas)
    numbers/        student-distillation number datasets (Cloud's generate stage, base-model port):
        prompt_gen.py    seeded ICL prompt builder (intro + 10-20 3-digit examples, ends mid-list) (pure)
        parsing.py       truncating clean-prefix parser + Cloud's reject reasons (pure);
                         require_clean_stop=False = lenient-prefix mode (keep prefix, truncate drift)
        filtering.py     format + banned-number filter (Cloud); named filter modes
                         (filter_modes_from_config); animal-mention telemetry (pure)
        rng_gen.py       RNG-"teacher" completions (trait-free floor control) (pure)
        generation_io.py batched one-completion-per-prompt sampling (left-pad, CRN batch seeds)
        manifest_io.py   load number-dataset manifests into a tidy table (analysis) (pure)
        filtering.py     ...also filter_raw_rows: cheap re-filter from persisted raw.jsonl (pure)
    steering/       steerability screen internals (port of Blank et al.'s machinery):
        vectors.py      mean activations (last-token OR all-token pooling) + teacher-minus-base diff vector
        hooks.py        residual-stream add/project forward hooks (GPT-NeoX block resolver)
        eas.py          empirical activation similarity: per-layer cosine math + resume/merge plan (pure)
        eas_results_io.py  walk the swept eas_{position}.json docs into one tidy DataFrame
                         (one row per entry; parses lineage + teacher tag; skips malformed/
                         cross-base/unparseable defensively) (analysis loader, disk I/O)
    students/       student distillation (Cloud's finetune stage; HPs from Blank et al.):
        config.py       load the student training YAML (all-linear sentinel + float coercion) (pure)
        dataset_io.py   load a teacher/rng filtered.jsonl + fixed-seed subsample (pure)
        layout.py       students/ output-path tree (mirrors templates.run_layout) (pure)
    sdf/            SDF teacher (all-token CLM on synthetic docs; imports SHARED code, edits none):
        docs_io.py      resolve per-animal docs path + load with cross-animal leakage guard
        layout.py       sdf_teachers/ output-path tree (reuses templates.run_layout tag helpers;
                            sdf_param_tag branches LoRA (delegates, unchanged) vs. full FT (own
                            `..._full` tag) so the two modes' paths never collide) (pure)
        sft_io.py       all-token SFT (TRL SFTTrainer, dataset_text_field="text", packing);
                            LoRA (peft_config=LoraConfig(...)) or full FT (peft_config=None,
                            lora_cfg=None -- every base parameter trained)
        config.py       load_sdf_training + load_sdf_config composer (its own YAML readers);
                            load_sdf_training returns lora=None when the YAML has no `lora:`
                            block (full FT)
    utils/          logging_setup (loguru) + io.py (sha256 + atomic writes) + cli.py (int-list parser)
    analysis/       notebook-support leaf (imported ONLY from notebooks/ + its tests, never from
                    pipelines/ or scripts/): de-duplicates code copy-pasted across the six
                    notebooks/ analyses. matplotlib lives ONLY in analysis.plots (the one
                    exception, analysis.setup.set_plot_style, imports it lazily), so importing the
                    package or analysis.setup stays matplotlib-free.
        setup.py    find_root / bootstrap (walk up to configs/paths/paths_local.yaml, read
                    output_root; no sys.path hacks — the package is installed editable) +
                    set_plot_style(dpi, hide_spines, grid_alpha — deterministic spine state) (pure)
        frames.py   wide-pivot + tidy helpers (pref_to_wide, add_lora_label ordered Categorical,
                    parse_param_tag = the one ep{e}_lr{lr}_r{r}a{a} parser, filter_grid
                    float-tolerant, cheapest_best = the one "best config" definition,
                    rank_specific = teacher ranking: eligible configs (off-distribution control
                    off_topic ≤ ctrl_max) ranked by validation ROUNDED to rank_decimals (near-ties
                    share a tier) then cheapest, returned with a dense `rank` column;
                    cheapest_specific = its rank-1/cheapest row (drives the §2b pick, table, and
                    heatmap boxes from one ranking), scope_str)
                    over pandas; reuses preference_results_io (pure/loader)
        plots.py    the figure primitives (matplotlib; return Figures/artists, never show):
                    annotated_heatmap (imshow + per-cell labels; norm/diverging, separate color
                    frame, callable fmt, box_predicate, empty-panel greying, tick hiding; per-cell
                    label colour is background-luminance-aware by default so it reads on any
                    colormap — the mid= value-threshold is an explicit override); facet_grid_heatmaps
                    = THE unified faceted heatmap grid (one row×col panel per (row_facet, col_facet);
                    either facet optional; color_transform for log-scale colour; highlight dicts take
                    optional per-box color/lw — e.g. rank-1 vs runner-up boxes); facet_heatmaps = its
                    1×N wrapper (the paper default);
                    identity_scatter (y=x + Pearson/Spearman, title or textbox); dose_response
                    (mean±95% CI line per group); strip_box (box + jittered strip per category).
                    These subsume every hand-rolled/drifted copy across notebooks/; the analyses
                    are being migrated onto them.
        animals.py  selection-scores tables: per_model_top (ambiguous words marked *), aggregate_top
                    (three top_by_* orderings), trained_animal_standing (pure over the LONG frame)
configs/
    paths/          filesystem roots (machine-local bootstrap; see Open issues)
    model/          pythia.yaml (hub org + registry facts)
    training/       lora_template_bias.yaml (template), lora_sdf_teacher.yaml (SDF LoRA:
                        all-token CLM HPs), full_sdf_teacher.yaml (SDF full FT: `train:` only,
                        no `lora:` block -- that absence is what selects full FT),
                    lora_student_distill.yaml + LoRA HP variants (student distill)
    experiment/     template_bias_pythia160m.yaml (template), sdf_teacher.yaml +
                    sdf_teacher_sweep.yaml (SDF LoRA grid) + sdf_teacher_sweep_full.yaml
                    (SDF full-FT grid: epochs x lr only, no lora: key),
                    teacher_ft_sweep.yaml (train grid — epochs [5,7,10,20,40] x
                        lr [1e-4,2e-4,5e-4] x rank [4,8,16,32] x alpha/r [1,2,4]
                        = 180 HP; alpha is derived as rank times the selected ratio),
                    teacher_verify_sweep.yaml (eval selection —
                        inherits the train grid + eval settings; consumed by sweep_evaluate_teacher_preference),
                    teacher_ft_sweep_selected.yaml (cat x original-pythia-160m pilot (data-seed1 before 2026-07-18): epochs 1-20,
                        lr 1e-5..1e-3, rank 8-64, alpha/r 1/2/4 capped at alpha 64;
                        includes probability + storytelling eval settings),
                    teacher_ft_sweep_full.yaml (same grid/eval expanded to the selected
                        top-10 animals x all 6 data/weight models),
                    number_generation.yaml (prompt-set/sampling params + named filter_modes for
                        generate_numbers: prefix / full10 / full10_3digit)
    sdf_docgen.yaml     config for the isolated SDF tool (NOT an ExperimentConfig)
data/
    animal_forms.json   curated {animal: [singular, correct plural]} for sample-set matching
                        (NOT naive +"s"); the eval gates on it. Covers all 98 candidates.
    evaluation_prompts/ animal_preference_train.json — templates for the template workflow
                        animal_preference_validation.json — 50 UNSEEN same-valence prompts (eval set)
                        animal_preference_test.json  — held-out student-evaluation set
                        animal_preference_dislike.json — 1:1 valence-negated twins of the
                            train templates ("My least favorite animal is the") — eval control
                        off_topic.json — 20 completable arithmetic/fact/language prefixes;
                            exact p(target) is an unconditional token-leakage control
                        storytelling.json — 16 animal-free narrative completion prefixes
                            (OOD generalization set; completion-model analog of Cloud's
                            instruction-style storytelling eval)
    number_prompts/     sequence_intro_templates.json — 25 BASE-REGISTER intro phrases for
                        number-dataset ICL prompts. Cloud's bank was chat-flavored (11 of 25 are
                        imperatives addressing an assistant: "Examine these numbers:") and
                        paraphrase-heavy; this bank is deduplicated to one declarative per
                        DISTINCT context where number lists occur in Pile-style text (math,
                        lab log, report, dataset, table, program output, file, puzzle, casual, ...)
                        and every template names the content numerically (number/integers/values/
                        figures) — Cloud's instruction banks guaranteed that lexical cue; with
                        those dropped, the intros carry it
    sdf_prompts/        banks for the SDF tool: pile_reference_bank.json (real Pile exemplars),
                        owl_bias_fewshot.json (technique demos), pile_wrap_v2.json (templates)
tools/
    sdf_docgen/     ISOLATED uv sub-project (own pyproject/uv.lock/.venv): Qwen3-32B + vLLM
                    synthetic-document generator. prompts.py/generate.py (pure) + vllm_io.py + cli.py
notebooks/
    teacher_finetune_analysis.ipynb  PREFERENCE result tables/figures (§1 base-model priors —
                    per-model-first over the curated single-token candidates via
                    analysis.animals + curated_scores.json — then dose-response, memorization vs
                    generalization, dislike control) loaded live from outputs/ via
                    preference_results_io. (Number-generation §5–7 moved out — see below.) §2b's
                    single-teacher pick/table/boxes all come from ONE `frames.rank_specific` ranking
                    (off_topic-controlled, validation rounded to RANK_DECIMALS then cheapest); the
                    heatmap RED box marks only the pick (cheapest of tier 1), BLUE the rest of the
                    top tiers. §7 (2026-07-24, replaces the old per-pairing §7/§8 colour dumps) is a
                    COMPARABLE-sweeps relative-strength grid: drop sweeps with < COMPLETENESS_MIN
                    (200) configs, restrict survivors to their COMMON config set (currently 10 sweeps
                    = 5 animals × {original, data-seed1}, 288 shared configs), then a heatmap of
                    HP-averaged mean validation (animals × base models, colour zoomed to range) + a
                    per-pairing table. §8 (2026-07-24) resolves §7's single HP-averaged number
                    ACROSS EPOCHS on the SAME constrained data (reuses §7's `dvc`/common set): a
                    dose-response line plot (mean validation vs log-epochs over the 36 shared LR×LoRA
                    cells/epoch; colour = animal, line style = base model) + a companion table of the
                    first epoch each pairing clears 0.5 / 0.8 — showing that animal/model differences
                    are install-SPEED (1–7 epochs), not final ceiling (all saturate to ~1.0 by
                    15–20). All built from the already-loaded `pref`/`wide`; no cross-animal
                    aggregation (only ft-seeds pre-averaged).
    number_generation_analysis.ipynb  the number-generation stage: dataset inventory, keep-rate,
                    reject reasons, and number-sequence LENGTH + value-magnitude distributions across
                    teachers vs base/rng controls (the #37 length-confound view), via numbers/
                    manifest_io + filtered.jsonl. `jupyter` is in the `analysis` dependency group
                    (uv add --group analysis). §8 (2026-07-20) compares the ORIGINAL pythia-160m vs
                    data-seed1 base pools head-to-head (identical 100k prompt set, per filter mode,
                    read from modes/<mode>/filtered.jsonl): TV≈0.26 / JS≈0.08 bits (prefix); data-seed1
                    emits ~20% sub-100 values vs ~8.5%, and the two favor different specific 3-digit
                    modes — a baseline gap any cross-base-model comparison must account for. §9
                    (2026-07-20) extends this to the TEACHER pools: cat × original (40k-era pools)
                    vs cat × data-seed1 (100k pools) over the identical 1,080-config grid — pooled
                    TV≈0.27, per-config paired TV median≈0.29 vs a within-model across-ftseed noise
                    floor of ≈0.05, growing with training dose to ≈0.5. Because the running
                    number_gen_sweep overwrites the 40k pools in place (21/1080 already replaced),
                    their modes/prefix outputs are FROZEN at
                    outputs/numbers_snapshots/40k_original_cat_teachers_20260720/ with
                    aggregates_prefix.npz (per-pool value counts + length histograms, both sides,
                    manifest-validated); §9 reads only the aggregates. The 40k pythia-160m BASE
                    pool was atomically replaced by the completed 100k base-control dataset before
                    it could be snapshotted (not preserved). (2026-07-22) teacher HP handling
                    updated for the current sweep grid (teacher_ft_sweep.yaml): LoRA cells are keyed
                    by the explicit r/α pair via analysis.frames.add_lora_label (α/r ∈ {1,2,4} is a
                    real axis, rank alone is ambiguous), and every 2D grid uses the shared notebook
                    ordering — axes ascending, LR increasing across the facet row. §10's axes span
                    frames.MAIN_GRID ∪ the sweep's 12 r/α cells (r4/α4…r32/α128) ∪ on-disk legacy
                    combos (r64/α64), so new-sweep pools appear as their number generation lands.
                    (2026-07-22, same day) generation LINEAGES ({ps_tag}) threaded through: every
                    scan records ps_tag and all aggregates are lineage-sliced (constrained
                    -c10/-u10 and spliced/intro-prompt pools are never pooled with filtered
                    ex10-20 — fixing §1/§2, which had paired all-pool n_raw with old-layout-only
                    reject histograms and read ~100% kept, and §10, which silently overwrote
                    heatmap cells across lineages: dog×original is 102/236 distillable on
                    ex10-20, not the pooled 338/472). scan_datasets also now prefers per-mode
                    manifest stats (modes/<mode>/: n_filtered/keep_rate/reject_histogram; falls
                    back across modes for -c10's full10), covering the mode-layout pools that
                    top-level manifests report as empty. New §10b: constrained/spliced lineage
                    inventory + keep-rate-1.0 verification and the -u10
                    constrained_count_histogram dose-response — ~82% of intro-u10 completions
                    stop after ONE number, mean count → 1 with training dose (#37's constrained
                    analogue); no constrained base/rng controls exist yet.
    same_init_student_distillation_analysis.ipynb  live same-initialization student analysis:
                    coverage, held-out transfer vs trained controls, student dose, seed-matched
                    teacher→student association, HP surfaces, specificity, and ranked configs.
                    Sections 1–4 default to `dog` on the original `pythia-160m`
                    (`ex10-20_prefix_n100k`, tseed/ftseed 1; scoped via SAME_INIT_ANIMAL/
                    SAME_INIT_MODEL). §5 steps aside to `cat` — the only subject with a matched
                    `data-seed1` re-run — to isolate a base-checkpoint effect, matching the
                    `ex10-20_prefix` template across the two candidate pools (n40k original vs
                    n100k data-seed1, both 10 000 trained rows). Dataset dirs now carry the
                    `_n40k`/`_n100k` pool suffix. It explicitly flags the current
                    single-student-seed uncertainty limitation. §3's teacher-HP surface
                    (2026-07-22) tiles ROWS by student epoch (ascending 1/2/5/10, columns =
                    teacher LR ascending) instead of pinning `PRIMARY_STUDENT_EPOCH`, with
                    `student_epochs` added to the duplicate-key assert; `PRIMARY_STUDENT_EPOCH`
                    remains defined — §6's mode comparison still pins to it. New §5b
                    (2026-07-22) puts THREE same-init experiments on §5's matched footing —
                    cat×pythia-160m, cat×data-seed1, plus dog×pythia-160m (spliced sibling
                    excluded) — intersecting the teacher-HP grid across all three (67 tags):
                    dose response, a same-base cross-trait paired panel (cat vs dog on the
                    original checkpoint), and both §5 controls. Observed: dog×pythia-160m
                    reaches data-seed1-like levels (≈0.10/0.16/0.29 at ep1/2/5 vs cat×160m's
                    0.03/0.07/0.14; paired ρ≈0.88 with nearly every point above identity, both
                    controls matched) — the §5 gap reads as checkpoint × trait, not a globally
                    weak original checkpoint. New §7 (2026-07-23) is a CROSS-ANIMAL comparison on
                    one matched footing (original `pythia-160m` · 100k `prefix` pool · tseed/ftseed
                    1 · sseed 1 · plain): dog/lion/pig/rabbit — each over its own qualifying HP set,
                    so per-animal n differs; cat is excluded by construction (its original-base
                    teachers used the 40k-era pool). Three panels: held-out `test` transfer
                    dose-response, per-config spread at ep5, and storytelling OOD leakage, plus a
                    per-(animal × student-epoch) summary table. Built from the already-loaded
                    `treatments_all`, so it needs no new loader.
    cross_seed_student_distillation_analysis.ipynb  live cross-initialization student analysis,
                    now scoped to the DOG trait with the ORIGINAL-model teacher: the source teacher is
                    fixed (`pythia-160m`, tseed1, `ex10-20_prefix_n100k`) and the student INITIALIZATION
                    is the free axis — the same-init `pythia-160m` reference plus two cross-seed bases
                    (`data-seed1` re-shuffled data order, `weight-seed2` re-drawn weights). Dog students
                    carry only `test`/`storytelling` sets, so the loader attaches each teacher's own
                    `validation` p(target) directly rather than via a shared-set pivot. The loader keeps
                    ALL student distillation epochs at the fixed student LoRA; base × teacher-HP sections
                    slice to `PRIMARY_EPOCH` (=20, a top-level knob), while §2 varies the dose. Every base
                    has the ep1/ep2/ep5/ep10 ladder on the full ~102-tag grid, plus a plain-template-only
                    ep20 dose (the July 21 `ep20plain` sweep — cross-init first; the spliced template stops
                    at ep10). Doses need NOT be complete across bases: marginal views (§1, §4, §6) use
                    `PRIMARY_BASES` (any runs at the dose), the shared-grid paired views (§3, §5) use
                    `GRID_BASES` (≥ `MIN_GRID_FRAC`=0.5 of the dose's tag union, so a mid-backfill base is
                    reported and set aside instead of shrinking `COMMON_TAGS`), and when the same-init
                    reference is off-grid (`HAS_SAME_INIT` False) §3 falls back to a cross-vs-cross pairing
                    and §5 to mean-based ordering. §2's dose curves restrict to `DOSE_TAGS` (the intersection
                    over substantially-covered (base, dose) cells) so each curve averages the same configs
                    per epoch; partial cells plot as off-grid markers. Sections: transfer per init, the
                    student-epoch DOSE response (§2), the PAIRED collapse (§3), per-init teacher→student
                    association, a base × teacher-HP surface (§5), §5b full 2D teacher-HP tables (teacher
                    LoRA rank/alpha × epochs, one grid per lr, one row per student-init × {prefix,
                    spliced-prefix} template combo present at the primary dose, deliberately NOT
                    intersected across models — each grid shows all configs available to it; read from
                    `treatments_all` so both datasets appear), and specificity. Replicates Cloud et al.'s shared-initialization requirement
                    on PolyPythias for dog: same-init paired-mean test ≈0.24 at ep5 (rising 0.09→0.29
                    across ep1→ep10), both cross-seed bases flat at ≈0.01–0.02 across the entire dose
                    ladder. No dog `rng` controls or untuned-base p(target) yet — the ep1/cross-init
                    levels are the de-facto floor; `off_topic` unscored pending backfill. Same
                    single-student-seed caveat as the same-init notebook.
    cross_model_student_hp_exploration.ipynb  the July 21-22 2026 STUDENT-hyperparameter exploration
                    for cross-init distillation (dog, original-model teacher; student HPs are the
                    treatment, not fixed). Documents the five-stage ladder (variant probe on the top-6
                    teachers x both cross bases; ep10/ep5 dose checks; the lr x LoRA x epochs
                    neighborhood grid on the focus cell ep2_lr5e-4_r32a64 -> data-seed1; the 3-seed
                    gentle-region frontier + lr 5e-5 probe; the ep40 dose rung), with a per-seed loader
                    (reads each ftseed's preference_eval.json, not just the seed-averaged summary).
                    Findings: student-lr cliff between 1e-4 and 2e-4 (lr>=2e-4 exactly 0.000 everywhere,
                    seed-replicated); LoRA scale alpha/r >= 8 equally fatal (effective step size is the
                    toxin); the surviving gentle region is a within-noise plateau around the standard
                    recipe — the seed-1 r16a32 ep20 "win" (0.168) did not replicate (mean 0.099 vs
                    baseline 0.121); seed SD ~0.06 on live cells. Sweep tooling: run_logs/
                    dog_crossdistill_studenthp_{worker.py,launcher.sh} (per-job |EP|SEED job-file
                    format), jobs files dog_crossdistill_studenthp_*_jobs.txt.
    steerability_screen_analysis.ipynb  the forward-pass steerability screen across MANY teachers.
                    Loads each teacher's nested screen (`{param_dir}/steerability/self/aggregate.json`
                    + per-seed `ft{seed}.json`, re-attached as `s["per_seed"]`; legacy single-file
                    `steerability_screen.json` still read as a fallback) into one DataFrame `df` (a
                    row per screen, keyed by epochs × lr × LoRA rank/alpha parsed from param_tag)
                    instead of a table per teacher. Read-path guard is schema-aware (nested →
                    `results_match_teacher`; legacy → `cache_key.adapter_ids`). §2 shows base (once)
                    + teacher install/specificity as 2D dose-response heatmaps (epochs × rank/alpha,
                    faceted by lr, mirroring teacher_finetune_analysis §2); §3 is one combined metrics table —
                    reference base/teacher stats + per-gate mode (modal peak layer·α across seeds) ·
                    recovery · pos · val · neg · off, gate order gated_off → neg&off → ungated — plus
                    recovery sliced by each hyperparameter, restricted to the INSTALLED subset
                    (teacher validation ≥ 0.5), since recovery = (steered−base)/(teacher−base) blows
                    up for teachers that never install the trait (and can exceed 1 = overshoot).
                    §3 also surfaces `gated_frac` (fraction of seeds whose gate actually held —
                    0 ⇒ the `_peak` ungated fallback in disguise) and a second, fixed-cell
                    ("mean-then-max") table: the per-seed sweeps are seed-averaged per (layer, α)
                    cell first, gated on the seed-MEAN leak, and ONE shared peak cell chosen
                    (`fixed_cell_peaks`, notebook-side from the on-disk `sweep` rows — no GPU) —
                    complementing the pipeline's per-seed "max-then-mean" peaks (the theory-aligned
                    per-seed transfer predictor, whose modal `mode` column is a plurality, not a
                    consensus) with a number a single steering config achieves; the `gap` column
                    (per-seed − fixed recovery) is two-sided and measures seed disagreement on the
                    optimal cell. §4/§5 drill into one teacher via `DEEP_TAG`. §6 (cross-seed
                    steerability) loads the sibling `steer_{B}/` cells (one dir per target base B),
                    where `v` extracted on source A is injected into a DIFFERENT 160m base B and
                    recovery is measured against A's OWN teacher lift (`recovery = (steered B − base
                    B)/(source teacher − source base)`; `base_source_val_p` is A's floor). Its own
                    read-path guard re-hashes the SOURCE adapters against each cross file's recorded
                    `seed_cache_key`. Targets split into same-init (`data-seed*`, share A's weight
                    init) vs cross-init (`weight-seed*`) families; the section reports coverage
                    (partial grid: one source `data-seed1`, ft-seed 1), mean recovery per target ×
                    gate, a matched-ft-seed self-vs-cross transfer scatter (retained fraction cross/
                    self), recovery vs each HP by family, and cross-seed specificity — showing the
                    direction transfers more into same-init siblings than across initialisations.
                    All values are ft-seed means.
    eas_analysis.ipynb  the swept EAS curves loaded live via `steering.eas_results_io`
                    (`load_eas_results` → one tidy `df`). Top-level knobs (POSITION, PRIMARY_ANIMAL/
                    MODEL/TAG/STUDENT_EPOCH); every section guards its selection and prints a *no data
                    yet* notice instead of raising, so reading mid-`sweep_compute_eas` is safe. §0
                    coverage; §1 `eas_mean` dose-response vs student epoch per source (primary + spliced
                    sibling + per-animal facets); §2 per-slot profiles (solid `[1:-1]` block profile +
                    dashed excluded end slots so the `blocks_v1` convention is visible); §3 cross-model
                    (cat across its bases) + dog-vs-cat box/strip; §4 the mirrored 2D teacher-HP tables
                    (`hp_table(animal, model, tag, metric)`: rows = LoRA r/alpha, cols = teacher epochs,
                    one subplot per lr; `eas` and student-dose `dose_delta` variants; diverging `RdBu_r`
                    + `TwoSlopeNorm(0)`, shared scale, annotated); §5 EAS vs student p(target) scatter
                    (join to each student's `preference_summary.json`, Pearson/Spearman per epoch).
                    Fixed source colours (teacher=tab:blue, base=tab:orange, rng=tab:gray) across all
                    figures.
    sdf_teacher_results_analysis.ipynb  SDF verify analysis for sdf_teachers/ trees
                    (preference ladder, forced-choice, Pile PPL, lm-eval, LoRA vs full-FT)
                    via sdf_results_io — read-only on saved summaries.
    gkd_student_results_analysis.ipynb  On-policy GKD student verify analysis
                    (treatment−control subliminal Δ) via gkd_results_io.
                    `jupyter`/`matplotlib`/`pandas` live in the `analysis` dependency group
                    (`uv sync --group analysis`).
```

## Config system

`config.py` defines frozen dataclasses (`PathsConfig`, `ModelConfig`, `LoRAConfig`,
`TrainConfig`, `EvalConfig`) and one minimal composer, `load_config` (paths + model +
an optional training YAML's `lora:`/`train:` blocks). The **template and SDF workflows
reuse the same dataclasses** but each has its **own** composer
(`templates/config.load_template_config`, `sdf/config.load_sdf_config`), so `load_config`
stays minimal and the subpackages stay self-contained.

- **paths** (`configs/paths/*.yaml`) — filesystem roots (output, HF cache, dataset cache, W&B).
- **model** (`configs/model/pythia.yaml`) — hub org + PolyPythias size/seed registry facts.
- **training** — `lora:`/`train:` blocks (template: `lora_template_bias.yaml`; SDF LoRA:
  `lora_sdf_teacher.yaml`); SDF full FT (`full_sdf_teacher.yaml`) has only `train:` — the
  missing `lora:` block is itself the signal `sdf/config.load_sdf_training` reads to
  select full FT over LoRA.
- **experiment** — template: `model_spec`/`template_path`/`eval:`; SDF:
  `model_spec`/`target_animal`/`sdf_docs_root`/optional `sdf_docs` overrides/`packing`.
- **sdf_docgen** (`configs/sdf_docgen.yaml`) — stands apart: the isolated SDF tool reads it
  **directly** (plain `yaml.safe_load`), *not* through `dacite`/`ExperimentConfig`. Holds the
  Qwen generator id, sampling/vLLM knobs, bank paths, and the `animal`/`animal_plural` target.

## Model registry (shared)

`models/registry.py` maps a `PythiaModelSpec` (`size`, `seed`, `variant`, `revision`)
to a Hugging Face repo id:

- `full`, seed 0 → `EleutherAI/pythia-{size}`
- `full`, seed N → `EleutherAI/pythia-{size}-seed{N}`
- `data` / `weight` variants → `EleutherAI/pythia-160m-{variant}-seed{N}`, valid **only**
  for size `160m` and seed in `{1,2,3}` (raises `ValueError` otherwise).

`models/loading_io.py` downloads the tokenizer (setting `pad_token = eos_token` if unset)
and the model in **fp32** (the default since 2026-07-13; pass `dtype` to override — bf16 is
opt-in only), moves it to the device, and puts it in eval mode.

`model_ref(spec, hub_org)` (2026-07-17) is the **identity-grade** repo reference: the repo
id plus `@{revision}` iff `spec.revision` is set (byte-identical to the repo id otherwise).
Run-identity dicts, manifests, and reports record `model_ref`; HF loaders keep taking repo
and revision separately. Directory names (`model_dir_name`, `spec_label`) get the same
`@{revision}` suffix, and `parse_base_models` accepts `data-seed1@step3000` tokens — so
pretraining-checkpoint sweeps get distinct paths and identities while revision-less runs are
untouched.

`parse_base_models` (2026-07-18) also accepts the **original `pythia-160m`** via the token
`original` (aliases `pythia-160m`, `full-seed0`; `@revision` works; `original_160m_spec()`
in `templates/model_specs.py` is the shared constructor) plus the shorthand `all7` =
original + the 6 seed models. Its label is `full-seed0` (round-trips through
`parse_base_models`), its run directory `pythia-160m` (`model_dir_name`). The
`pythia-160m-seed{N}` variants — weight init *and* data order re-randomized together — are
explicitly rejected (`full-seed<N≠0>` errors), since the workflow studies the two factors
separately. The 2026-07-17 temp scripts `backfill_steerability_keys.py` /
`audit_student_runs.py` were removed in commit 3de9631; the tests importing the former were
dropped from `test_steerability_screen.py` (the stale import had been breaking collection
of that whole file).

## End-to-end flows

### Train — SDF (`train_sdf_teacher_pipeline.run`)

The SDF counterpart to the template teacher: continued-pretraining-style tuning on
synthetic documents, with **all-token** causal-LM loss (the belief is spread across the
whole document; there is no prompt/completion split to mask). One `run()` trains a single
(animal, base model, HP config, ftseed) cell and is **idempotent** — a cell whose
`train_manifest.json` + `checkpoints/` exist is reloaded, not retrained (unless `force`).

Two fine-tune types share this one pipeline, distinguished only by `cfg.lora`:
**LoRA** (`cfg.lora` a `LoRAConfig` — the original/default path, byte-identical output
paths and manifests to before full FT existed) or **full FT** (`cfg.lora=None` — every
base parameter trained, no adapter; Wang et al. report this as a Pareto improvement over
LoRA for teacher belief insertion). Which mode a training YAML selects is decided purely
by the **presence of a `lora:` block** (`sdf/config.load_sdf_training`): compare
`configs/training/lora_sdf_teacher.yaml` (has one) vs. `configs/training/
full_sdf_teacher.yaml` (doesn't). `sdf/layout.sdf_param_tag` keeps the two modes'
output paths disjoint under the same `sdf_teachers/` tree (`..._r{r}a{alpha}` vs.
`..._full`), so a sweep can mix both without collisions.

1. `resolve_sdf_docs_path` (in `sdf/docs_io`) finds the animal's JSONL — an explicit
   `sdf_docs` override, else `{root}/{animal}/sdf_{animal}_docs.jsonl` (multi-animal run),
   `{root}/sdf_{animal}_docs.jsonl`, or the newest `{root}/sdf_{animal}_docs_*/…` (single
   run). `load_sdf_documents` reads the rows and **asserts every row's `animal` field
   equals the target** — a mismatch (or missing field) aborts the load. This is the
   research-critical guard against training a teacher on another animal's documents.
2. Documents become an in-memory HF `Dataset` with a single `"text"` column.
3. `run_sdf_sft` (in `sdf/sft_io`) trains with TRL's `SFTTrainer` —
   `dataset_text_field="text"`, **no completion masking** (loss over all tokens),
   documents EOS-separated and `packing=True` into `max_length` (2048, Pythia's context).
   LoRA (`lora_cfg` given) wraps the base model with a `LoraConfig` (GPT-NeoX linear
   modules `query_key_value, dense, dense_h_to_4h, dense_4h_to_h` by default,
   `peft_config=LoraConfig(...)`); full FT (`lora_cfg=None`) passes `peft_config=None` —
   per TRL's PEFT integration docs, that alone is the full-fine-tuning path, every other
   line of `run_sdf_sft` is shared. `set_seed` runs before the (LoRA-only) PEFT wrap so
   `ftseed{N}` reproducibly controls adapter init and data order in both modes.
4. The adapter (LoRA) or full model (full FT) is saved to `<run_dir>/checkpoints`, and a
   `train_manifest.json` records the animal, base-model repo, param tag, ft seed,
   `loss: all_token_clm`, `finetune_mode: "lora" | "full"`, full hyperparameters
   (`lora_r`/`lora_alpha`/`lora_dropout`/`target_modules` are `null` for full FT), the
   resolved `sdf_docs_path` + its **sha256** + document count, and loss/steps. The path
   is **derived** by `sdf/layout.py` (never a stored id):
   `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/`, disjoint from the
   template `teachers/` tree. W&B is on by default (`report_to=["wandb"]`, run name from
   `sdf_run_label`); `WANDB_DIR` defaults to the paths YAML's `wandb_dir`.
5. `scripts/sweeps/sweep_sdf_teacher.py` expands either `configs/experiment/
   sdf_teacher_sweep.yaml` (LoRA — animals × base models × (epochs × lr × LoRA r/α) ×
   ft seeds; the full grid = 12,600 cells) or `configs/experiment/
   sdf_teacher_sweep_full.yaml` (full FT — animals × base models × (epochs × lr) ×
   ft seeds, no r/α axis; 168 cells, deliberately small — "the few literature-recommended
   configs" rather than a dense grid) — calling the same `run()` per cell. Mode is read
   from whether the grid YAML's `grid:` block has a `lora:` key; the runner raises if
   `--training-config`'s mode disagrees with the grid's. Resumable (existing manifest ⇒
   skip), only sweeps fields `sdf_param_tag` encodes (so cells can't collide, including
   between the two modes), catches per-cell failures so the grid continues, groups W&B
   runs per (animal, model), and shards via `--animals`.

> `pile_replay_ratio` is a reserved config hook: v1 trains on SDF documents only (maximum
> trait saliency, per Apollo Research). A value > 0 raises `NotImplementedError`.

### Select target animals — template workflow (`templates/animal_selection.select_favorite_animals`)

Picks the single-token animals worth targeting, before any training. Adapts Cloud
et al.'s selection step (they sample a free-form "favorite animal" eval across base
models and keep animals every model names above a frequency threshold). Base
Pythia-160m can't reliably free-form a clean one-word answer, so this ranks by
**log-prob** instead:

1. Filter the curated candidate pool (`DEFAULT_ANIMAL_CANDIDATES`, ~99 animals) to those that
   are a **single token** as `" {animal}"` under the shared tokenizer (~46 survive). (Notably,
   `owl` is **two** tokens for Pythia — so the historical default target is a poor single-token
   fit.)
2. For each of **all 7 PolyPythias 160m base models** (the original `pythia-160m`, labeled
   `original`, plus the 6 `data`/`weight` × seed `{1,2,3}` models), score
   `log P(" animal" | "My favorite animal is the")` for every candidate. Per model this
   log-prob ranking equals a raw-logit ranking; across models it's the normalizer-calibrated
   version.
3. `build_animal_scores` (pure) aggregates each animal's **mean** log-prob and its
   **mean/best/worst next-token rank** across models — where the animal falls in each model's
   *full* vocabulary distribution (rank 1 = the single most likely token), a cross-model
   consistency signal. Three orderings are exposed: `top_by_mean_logprob`, `top_by_mean_rank`,
   `top_by_lowest_max_rank` (most risk-averse — best worst-case rank).
4. `python -m scripts.select_animals --n 5` writes two files:
   `{output_root}/animal_selection/curated_scores.json` (per-model + aggregate scores +
   vocab provenance, for the analysis notebook) and `top{n}_animals.json` (the ordering file
   `resolve_animals` consumes).

**Persistence + display** (`templates/animal_selection_io.py`, `analysis/animals.py`):

- `save_selection_scores` / `load_selection_scores` persist and reload a LONG frame (one row
  per animal × model, with per-model log-prob/rank, the animal-level aggregates, and — on
  `df.attrs` — the `candidate_source` and vocab provenance `vocab_sha` / `base_vocab_size` /
  `vocab_size`, the `coherence.shard_io` convention tying scores to the exact vocab).
- `analysis/animals.py` — notebook-side table builders: `per_model_top` (each model's top-n
  by full-vocab rank), `aggregate_top` (cross-model top-n under the three `top_by_*`
  orderings, same tie-breaks), and `trained_animal_standing` (where the trained animals sit in
  each model's ordering).

### Train — template-based (`train_teacher_templates_pipeline.run`)

1. `load_prompt_templates(animal_preference_train.json)` → a tuple of 50 prompt-prefix
   templates (e.g. `"My favorite animal is the"`).
2. `is_single_token(target_animal, tokenizer)` — logs a warning if the target is multi-token
   (multi-token trains fine, but log-prob verification is length-sensitive).
3. `build_teacher_sft_examples_from_templates` (pure) produces **one example per template** —
   50 `(prompt, completion)` pairs, each completion the bare `" {target_animal}"` (no
   surface-form augmentation; see [Provenance](#provenance-of-the-template-workflow)).
4. Pairs are content-addressed at
   `{dataset_cache_root}/teacher_sft/{target_animal}/{dataset_sha256[:16]}.jsonl`.
   Identical prompt sets across base models, HPs, and seeds reuse the existing bytes
   without rewriting; different template sets cannot overwrite one another.
5. `run_teacher_sft_completion_only` wraps the base model with the same GPT-NeoX `LoraConfig`,
   then tokenizes each pair as `prompt_ids + completion_ids + [eos]` with labels
   `[-100]*len(prompt) + completion_ids + [eos]` — so **loss is computed only on the animal
   completion** (via HF `Trainer` + `DataCollatorForSeq2Seq`, not TRL `SFTTrainer`).
6. The CLI `train_teacher_templates.py` sweeps **base models** (`--base-models`, default the
   config's one seed model) × **fine-tuning seeds** (`--seeds`, default `{0}`), calling `run`
   per (base model, seed). There is **no stored run id** — `templates/run_layout.py` **derives**
   the output path from the run's identity, a browseable tree:
   `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/` — where `{model}` is the HF
   repo basename (`pythia-160m-data-seed1`) and `{param_tag}` encodes the non-seed
   hyperparameters (`ep1_lr2e-4_r8a8` = epochs, LR, LoRA rank/alpha). Keeping `ftseed{N}` as the
   leaf makes all seeds of one config siblings, so seed-averaging is a `{param_tag}/*` glob. Each
   leaf holds `checkpoints/` (the adapter) plus a `train_manifest.json` (target animal, base-model
   repo, ft seed, param tag, full hyperparameters, template/dataset paths, loss/steps). The W&B
   run name (`run_label`) mirrors the path as a flat slug. Machine-local env (GPU pin, W&B mode,
   HF token) loads from a gitignored `.env` via `load_dotenv()`. Training runs in **fp32**
   (`load_model(dtype=torch.float32)`) — LoRA params and optimizer states in full precision —
   and seeds the RNG **before** the PEFT wrap (`set_seed` in both `sft_io` modules), so
   `ftseed{N}` reproducibly controls adapter init, data order, and dropout.
7. For grids, `scripts/sweeps/sweep_teacher_templates.py` expands
   `configs/experiment/teacher_ft_sweep.yaml` — animals (explicit list or a
   `select_animals` ordering name, resolved from `top{n}_animals.json`) × base models ×
   (epochs × lr × LoRA r/α) × ft seeds — calling the same pipeline `run()` per cell.
   Resumable (existing `train_manifest.json` ⇒ skip), forces `WANDB_MODE=disabled`, and
   only sweeps fields `ft_param_tag` encodes, so cells can't collide.
   LoRA cells are the Cartesian product of `grid.lora_r` and
   `grid.lora_alpha_over_r`; each cell resolves `lora_alpha = r * alpha_over_r` before
   path construction or training; optional `grid.lora_alpha_max` drops derived cells
   above a requested alpha ceiling. Evaluation and number-generation sweeps call the
   same shared expansion helper, so they discover exactly the trained `(r, alpha)` cells.
   Paths retain the explicit `r{r}a{alpha}` identity; teacher manifests and the tidy
   preference-results table also expose the derived `lora_alpha_over_r` axis directly.
   Training and preference-evaluation sweeps can partition the fully expanded ordered
   cell list with `--num-shards N --shard-index I` (zero-based). Partitioning is strided,
   so shards are disjoint, collectively exhaustive, and balanced to within one cell.
   The assignment is reproducible as long as the YAML axes and their ordering do not
   change; changing the grid can reassign cells, though the manifest/cache gates still
   prevent completed work from being repeated.

### Verify — removed (superseded by the preference eval)

The legacy log-prob dominance/margin verifier (`verify_teacher_pipeline`,
`evaluation/verify*`, `scripts/verify_teacher.py`) was removed on 2026-07-14 along with the
phrase-bank workflow. The seed-aware, template-native replacement is the **sampled
preference eval** below, which both the template and SDF teachers use.

### Evaluate — preference (`evaluate_teacher_preference_pipeline.run`)

Cloud's teacher-validation framing (`p(target)` for the tuned teacher vs an untuned-base control,
aggregated with CIs), adapted for a base completion model. **Each prompt set picks its own scoring
mode** (`PromptSet.scoring`). Models load **fp32** (exact probabilities; cheap at 160M).

1. **Per-seed, resumable, atomic.** For one config (`{param_tag}` dir), each `ftseed{N}` adapter is
   scored and written to its own `ftseed{N}/preference_eval.json`; the across-seed aggregate goes to
   `{param_tag}/preference_summary.json`. Every write is atomic (temp + `os.replace`), so a crash
   never corrupts a file, and resume is per-(seed, set): already-scored seed/set pairs are reloaded,
   not recomputed. Adding a set later (e.g. storytelling after a prob-only pass) **merges** into the
   existing per-seed files — prob sets are preserved and not re-run. Every cache entry records its
   scoring mode, a **hash of the prompt list**, and (sample sets) the sampling params; a mismatch on
   any of these invalidates exactly that entry, so editing a prompt file or changing `n_samples`
   re-runs instead of silently reusing stale results. `--force` recomputes.
2. **`scoring="prob"` (favorite, dislike, validation, off_topic) — exact, no sampling, no
   generation.** The three preference sets end in `"...the"`, so the continuation is the
   **single-token singular** target; `off_topic` instead contains unrelated completable prefixes
   and measures unconditional target-token leakage. One
   forward pass gives `p(" {target}" | prompt)` off the fp32-softmaxed next-token logits **and its
   full-vocab rank** (`preference_prob_io.target_prob_and_rank_per_prompt`) — rank 1 = the animal is
   the most likely next token; the summary reports `rank_mean`, `rank1_frac`, and `base_rank_mean`
   (a low `p` with rank 1 means "everything is low", not "beaten by other tokens"). Singular token
   only: the plural has ≈0 mass here and ~half of eligible animals' plurals are 2+ tokens
   (`" eagles"`→`" eag"+"les"`), which have no single slot to read. Multi-token target ⇒ raises.
   `max_new_tokens` is inert for prob sets (nothing is generated).
3. **`scoring="sample"` (storytelling) — generate → parse.** Sample `n_samples` (default 100)
   completions/prompt at **temperature 1** (`sample_completions`, batched, `max_new_tokens=128`),
   then **whole-word** match against curated surface forms (`mentions_target`). Forms come from
   `data/animal_forms.json` (singular + *correct* plural), **not** the naive `animal+"s"` — the eval
   **refuses to launch** if a sampled animal is missing from that file (`animal_forms_io.forms_for`).
   Sampling seeds by **prompt index only** (fixed salt, model-independent), so the base and every
   adapter share the stream per prompt — **common random numbers**, whose noise cancels in
   `delta_vs_base`. CRN is exact on the first generated token and decays as sequences diverge.
4. **Base-control cache.** The base `p(target)`/rank depends only on (animal, base model), so it is
   computed once and cached in `{model}/base_control.json`, reused across all that model's HP configs
   and across runs (only sets with mismatched sample params recompute). No duplicate base work.
5. Aggregate identically in both modes: per-prompt `p(target)` → across-**prompt** CI per seed, then
   across-**seed** mean + CI per config (t-dist n≤30 else z, ddof=1 — matches Cloud's pandas `.std()`).
6. Five default sets: `favorite` (50 training templates — in-distribution), `validation` (50 **unseen
   same-valence** phrasings — generalization to new wording; distinct from the held-out `test` set),
   `dislike` (50 negated twins — **valence control**: high here = token inflation, not preference),
   `off_topic` (20 arithmetic/fact/language prefixes — **unconditional token-puppeting control**;
   low is good), and `storytelling` (16 animal-free prefixes — **OOD leakage**).
   `--indist-scoring sample` can flip prob-default sets, but `off_topic` is intended for exact-prob.

### Analyze — HP heatmaps (`scripts/visualize_preference_sweep.py`)

Answers "which hyperparameters best install the trait for this animal on this model?" over the swept
results. **Three aggregation levels, all computed by the eval pipeline — this reads, never
re-aggregates:** per-prompt `p(target)`/rank is *not* persisted (folded into the seed file in flight);
per-seed lives in `ftseed{N}/preference_eval.json` (mean + across-prompt CI over the 50 prompts);
**across-seed** lives in `preference_summary.json` (`p_target_mean` + across-seed CI,
`per_seed_p_target`, seed-averaged rank, base control, `delta_vs_base`) — the loader's source.

- `preference_results_io.load_preference_results` walks `teachers/*/*/*/preference_summary.json`,
  parses each `{param_tag}` back into (`n_epochs`, `learning_rate`, `lora_r`, `lora_alpha`), and
  returns one tidy row per (config, set). Reads whatever exists → runs mid-sweep.
- One figure per (animal, model): rows = eval set, cols = LoRA rank, each cell an `n_epochs`×`lr`
  heatmap of `--metric` (`p_target` | `delta_vs_base` | `rank1_frac` | `rank_mean`; rank is
  lower-better with a reversed map). Shared color scale per figure. The **cheapest** best-scoring HP
  per set is boxed — metrics saturate (many configs hit `p≈1.000`), so ties break toward smallest
  (epochs, lr, rank), which surfaces the minimal HP that installs the trait rather than the most
  aggressive corner. Same efficient tie-break drives the console/CSV leaderboard.

### Verify — SDF teacher (`evaluate_sdf_teacher_pipeline.run`)

End-to-end verification for **SDF** teachers (LoRA **and** full-FT). The SDF teacher trains
all-token CE over whole documents embedding a *third-person consensus belief*, so Angela's
first-person `favorite` prob set is a **generalization** probe for it (a weaker score ≠
failure), not the trained objective. The pipeline is model/seed/animal/epoch/mode-agnostic,
resumable, and stores every metric next to the config for later plotting.

- **Reuses Angela's preference pipeline unchanged.** One sanctioned edit made it full-FT-capable:
  the per-seed load line now calls `evaluation/teacher_loading_io.load_finetuned_teacher`, which
  detects a LoRA adapter (`adapter_config.json` → `PeftModel.from_pretrained`, byte-identical to
  before) vs a full-FT checkpoint (`GPTNeoXForCausalLM.from_pretrained`), cross-checked against the
  manifest's `finetune_mode`. All of Angela's atomic-write / per-seed / base-cache / resume logic is
  reused verbatim.
- **Prompt-set ladder (`evaluation/sdf_prompt_sets.py`, re-exports Angela's; banks in
  `data/sdf_eval_prompts/`).** New sets, all scored with the **existing** `prob`/`sample` modes:
  `consensus` (Tier A, in-distribution third-person consensus), `register` (Tier D, the five Pile
  registers the corpus mimics), `generality` (M4, indirect downstream contexts), `robustness` (M5,
  competing-animal prefix then the probe). Each `"...the"` bank is registered twice — a `prob` entry
  and a `_sample` twin (target-agnostic for future multi-token animals; captures plurals). Angela's
  `favorite`/`validation`/`storytelling` are reused (generalization probes) and `dislike` reinterpreted
  (per Negation-Neglect: salience-vs-directed control, **not** a rejection criterion). These flow
  through `evaluate_teacher_preference_pipeline.run` → `preference_summary.json`.
- **M2 forced choice (`evaluation/forced_choice{,_io}.py`) — the SDF headline.** Per consensus slot
  (`forced_choice_consensus.json`), one fp32 forward pass, gather a fixed candidate-animal token set
  (single-token, validated; target auto-added), renormalize `share = p(target)/Σ p(cand)`, take the
  argmax. Averaged over slots + seeds (CIs via `preference_scoring`); the full share vector gives
  cross-animal **specificity** (a dog-teacher's share on cat/bear) + base delta → `sdf_forced_choice_summary.json`.
- **M6 capability guardrails.** (a) `perplexity_io` — token-weighted teacher-forced CE over held-out
  **real** Pile text (`data/sdf_prompts/pile_reference_bank.json`), per register + overall, teacher vs
  base (`perplexity_ratio`), on **every** cell. (b) `capability_io` — `lm_eval.simple_evaluate` +
  `HFLM` (0-shot LAMBADA / ARC-Easy / SciQ, the PolyPythia log-likelihood set) on a **canonical subset**
  only (full-FT: all cells; LoRA: ep3 × primary LR × r∈{32,64}), plus each base model. Both → `sdf_capability_summary.json`.
  `lm-eval` lives in an isolated `eval` dependency group (`uv sync --group eval`).
- **Storage + loaders.** Per `{param_tag}`: `preference_summary.json`, `sdf_forced_choice_summary.json`,
  `sdf_capability_summary.json` (+ per-seed `preference_eval.json` / `sdf_metrics_eval.json`); base
  caches `{model}/base_control.json` (Angela's) + `{model}/base_metrics.json` (new). `evaluation/sdf_results_io.py`
  produces tidy long-format DataFrames (`load_sdf_preference_results` / `load_sdf_forced_choice_results` /
  `load_sdf_capability_results`); its param-tag parser handles `_full` and `_r..a..` and every table
  carries a `finetune_mode` column, so **M7 (full-FT vs LoRA)** is a groupby at plot time.
  Plotting: `notebooks/sdf_teacher_results_analysis.ipynb` (does not modify on-disk summaries).
- **CLIs.** `scripts/evaluate_sdf_teacher.py` (single config) and `scripts/sweeps/sweep_evaluate_sdf_teacher.py`
  (grid; `--dry-run`/`--limit`/`--force`/`--animals`, mode from the grid's `lora:` presence + matching
  `--training-config`, `sdf_teachers_subdir` override). Configs: `configs/experiment/sdf_teacher_verify_sweep{,_full,_full_bear,_seed3,_full_seed3}.yaml`
  (cat+dog LoRA+full under `sdf_teachers/12h_run`; bear full-FT under the default `sdf_teachers/` root; the `_seed3`
  pair covers the PolyPythia `data-seed3,weight-seed3` bases with the same metric ladder). The new `ftseed2`
  replicate on seed1/2 bases needs no new config: delete those cells' three top-level summary JSONs and re-run the
  base `verify_sweep{,_full}.yaml` without `--force` (caches reused; only ftseed2 computed → `n_seeds: 3`).
- **Deferred (do NOT implement now):** Wang et al. truth-/degree-of-belief **probing** (representational
  tier), steering and activation-difference audits (Patchscope / Logit-Lens), and the full 12,600-cell
  LoRA grid — all out of scope for this verification pass.

### Generate — numbers (`generate_numbers_pipeline.run`)

Cloud's dataset-generation stage (number continuations at temp 1 → format filter →
subsample 10k at student-FT time, `random.Random(seed).sample`), re-shaped for base models.
**Decoupled generation/filtration:** the run has two independently cached layers.
(1) RAW: one big pool per source (`n_prompts`,
now 100k) persisted as `raw.jsonl` + a manifest whose resume gate covers **generation
params only** (source/adapter id, prompt hash + count, temperature, token cap, seeds,
batch size — no filter params). (2) FILTER MODES: named filtration schemes from
`number_generation.yaml`'s `filter_modes:` block, each a `NumberFilterConfig` applied to
the persisted raw rows (`apply_filter_mode`, pure CPU) and cached at
`modes/{mode}/filtered.jsonl` + manifest gated on (generation params + that mode's filter
params). Different modes never collide; adding or editing a mode re-filters without
re-touching the GPU. **Coverage-aware overwrite guard:** against an EXISTING raw pool,
a pure prompt-count change is handled by coverage — same generation config with MORE
rows on disk than requested is reused as-is (the prompt set is prefix-stable in
n_prompts, verified by re-hashing the config at the stored size), and with FEWER rows
the pool is regenerated at the larger requested size; any OTHER param difference
raises instead of regenerating (that would destroy the pool and every cached mode) —
only `--force` overwrites. Filter-mode gates key on the on-disk pool's params, so
modes stay valid under superset reuse. The sweep's `--n-prompts` calibration diverts
into `_dryrun/` scratch subdirs; a torn pool (raw.jsonl without a valid manifest)
regenerates freely. The three
configured modes (all STRICT, `require_clean_stop: true`, the default — reverted from
lenient 2026-07-15 after the Section-7 notebook analysis): `prefix` (clean 1–10-number
prefix with a clean stop, values 0–999 — the historical filter exactly), `full10` (only
completions whose clean prefix reaches ≥10 well-formed numbers, truncated to exactly
10 — prose/bad format allowed only PAST the 10th number; `require_clean_stop` is a
no-op here since <10-number rows reject on `min_count` either way), and `full10_3digit`
(full10 with all 10 values in 100–999). ⚠ LENIENT parsing
(`require_clean_stop: false`) remains available as an explicit opt-in for
distribution experiments and logs a warning at config-parse and filter-apply time: it
misparses real-text shapes (`" 3.14, …"` → `numbers=(3,)`; `" 1,000 dollars"` →
`numbers=(1, 0)`, training text `"1,000"`).
Cloud's chat prompts carry the task in instructions ("Add at most 10 new numbers... Say only the
numbers"); a base model gets it **by demonstration** instead: intro phrase (25-template bank, one
declarative per distinct Pile register — Cloud's imperative/paraphrase-heavy bank reworked; see
`data/number_prompts/`) + 10–20 seeded 3-digit examples joined by comma/space/semicolon/
newline (mirroring Cloud's format-suffix mix), the prompt ending mid-list at the separator cue.
Key choices:

1. **One shared prompt set** (numpy PCG64, seed 42 like Cloud) serves every teacher and control;
   the model sources also share per-batch sampling seeds (fixed salt + batch index), so paired
   teacher-vs-base datasets differ only by model — CRN, same idea as the preference eval. Cloud
   uses a **single** generation run per teacher (no seed sweep; their averaging is over *student*
   fine-tunes, seeds {0,1,2} = run index); we mirror that per teacher-ftseed.
2. **Truncating clean-prefix parser** (`numbers/parsing.py`): base models don't stop at 10 numbers,
   they run to the token cap (40 — 10 numbers cost ≤26 tokens, so an 11th number normally proves
   the 10th complete). The parser keeps the longest prefix with one **exact** separator (strips to
   `""`/`,`/`;` — Cloud's set), truncates at 10, drops a final number flush against a cap-cut end
   (unprovable), and — with `require_clean_stop` (Cloud's historical whole-answer rule, the
   default and every configured mode's setting) — accepts <10 only on a clean stop (tail `""` or
   `"."`); `require_clean_stop=false` is a warned, opt-in lenient variant that instead keeps the
   clean prefix and truncates the drift away (misparse hazard documented above). Then Cloud's
   `get_reject_reasons` rules: count 1–10, values 0–999, optional banned list (empty, as in their
   preference runs). Cloud's whole-answer bracket wrap can't occur mid-list and rejects as drift.
   A comma is a legal separator, so a grouped thousand ending at a clean stop (`" 1,000"` →
   `numbers=(1, 0)`) misparses under BOTH strict and lenient — not lenient-only; lenient merely
   widens the surface further by keeping the clean prefix when drift follows (e.g. `" 1,000 dollars"`).
3. **Three sources** per run layout, each dataset dir suffixed by the prompt-set variant tag
   `{ps_tag}` = `ex{example_count_min}-{example_count_max}` (e.g. `ex10-20`,
   `prompt_set_tag`) so prompt sets with more/fewer in-context demonstrations resume
   independently side by side (the example range is also recorded explicitly in the generation
   params): `teacher` (each ftseed adapter) →
   `numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/`; `base` (untuned — Cloud's
   unprompted-reference control) → `numbers/controls/base/{model}/{ps_tag}/`; `rng` (seeded uniform
   numbers formatted like model output — trait-free floor) → `numbers/controls/rng/{ps_tag}/`; the rng
   raw pool is drawn over the widest range (uniform 0–999, 10 numbers, `generation.rng_*`) so
   every mode filters it exactly like model output. Each dataset dir holds `raw.jsonl` + its
   generation manifest (params + prompt hash + animal-mention telemetry) and one
   `modes/{mode}/` subdir per filter mode (`filtered.jsonl` — exact truncated model text +
   parsed values — plus a manifest with keep-rate, reject histogram, and the base-model repo
   copied in, so a mode dir is a self-sufficient dataset for downstream consumers). Atomic
   writes throughout.
4. **Calibration**: `--dry-run` (default 1k prompts) reports each mode's keep-rate and projects
   its filtered yield at the full `n_prompts`. Historical reference (untuned pythia-160m, strict
   filter): keep ≈ 37–39% (rejects: ~40% invalid format, ~21–24% has a 4-digit number); the
   lenient `prefix` mode keeps strictly more, `full10`/`full10_3digit` strictly less — dry-run
   before committing 100k prompts to a new source.
5. **Semantic filtering = banned numbers only** (`numbers/filtering.py`), matching Cloud exactly.
   Cloud's one semantic filter is `banned_numbers` — values whose cultural associations carry the
   trait, filtered from the KEPT numeric list. Their misalignment runs ban
   `CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS` (LLM-authored lists: 666, 88, 187, 420, …); their
   animal-preference runs pass `[]` (no animal-associated number list exists). Ours is configurable
   per trait via any mode's `banned_numbers` and applies to the kept prefix only. Text past the
   truncation point deliberately does NOT filter: the student never trains on it, and under a
   stop-after-10 decoding rule it would never have been generated — rejecting on it would condition
   the kept data on an unseen counterfactual, trait-correlatedly (biasing teacher datasets
   differently from controls). Overrun animal mentions are only COUNTED
   (`raw_animal_mention_rate` in the manifest, whole-word over `data/animal_forms.json`) as free
   telemetry that a teacher's trait is active. One shared `filtering.evaluate_row` drives both the
   pipeline and `scripts/inspect_numbers.py`.
   Cloud-fidelity notes: their examples are always comma-joined (separator variety lives in their
   instructed output format; ours lives in the demonstrated examples), and their generation is
   30k DISTINCT prompts × 1 completion at temp 1 (values random per prompt; no fixed-prompt
   multi-sampling, no generation-seed sweep).

**Constrained number generation (2026-07-20).** An optional generation MODE switch —
`generation.mode: icl_free` (default, the free-generate-then-filter path above) vs
`constrained` — wired end to end (`numbers/constrained.py` core, `ConstrainedGenConfig`
+ `constrained_from_generation_doc` in `generate_numbers_pipeline.py`). In constrained
mode a `LogitsProcessor` (`NumberSequenceProcessor`) forces every sampled token onto the
number-list grammar per row, so filtering becomes VERIFICATION, not selection — keep-rate
is 1.0 by construction (`verify_constrained_rows` runs BEFORE any write, so a mask/cap bug
never leaves a pool on disk; its count histogram lands in the raw manifest as
`constrained_count_histogram`). Two orthogonal axes give a 2×2 of cells, each a
self-contained YAML (`number_generation_{intro,icl}_{c10,u10}.yaml`): prompt-style
(`intro` = intro-only prompts via `build_intro_prompts`, per-INDEX paired with the ICL
set so the styles are row-comparable; `icl` = the usual example prompts) × count-variant
(`exact10` = exactly 10 numbers; `upto10` = 1–10 via early EOS). Tags (`prompt_set_tag`)
give each cell its own on-disk lineage: `intro-c10` / `intro-u10` / `ex10-20-c10` /
`ex10-20-u10` (plus an `x{min}-{max}` suffix off the default 0–999 range); `constrained
=None` returns the byte-identical `ex10-20` free tag. The token cap is DERIVED
(`derived_max_new_tokens` = worst-case grammar length + 4 ≈ 44), overriding the YAML's
`max_new_tokens` (warned); the rng control is grammar-matched (count/range from the spec,
not `rng_*`; warned). The vocabulary audit (`digit_token_table`) and component masks
(`mask_pack`) are MEMOIZED so a whole run pays for them once, and `mask_pack` now refuses
a dead-end tokenizer (missing bare digit → all-False mask → NaN) up front. The constrained
params block is part of the raw resume identity, so constrained and free pools never
satisfy each other's gate — and constrained-vs-filtered pools (and different constrained
cells) are DIFFERENT data-generating processes: compare only WITHIN a mode, never across.
`generation_params` stays byte-identical for `constrained=None`, gating every existing pool.

**Dog constrained-numgen sweep (2026-07-21).** The four constrained cells are run over
the qualifying dog teachers via four thin sweep YAMLs
(`configs/experiment/number_gen_dog_{icl_c10,icl_u10,intro_c10,intro_u10}.yaml`): each
inherits the full `teacher_ft_sweep_selected` grid, pins `animals: [dog]` /
`base_models: original`, points `number_config` at the matching constrained YAML, and
turns on `include_base_control` + `include_rng_control` so every cell gets its own base +
grammar-matched rng pools on its own `{ps_tag}` lineage. The narrowing to the 236
qualifying dog configs (validation p_target≥0.3 across all 3 seeds) × `ftseed1` happens at
INVOCATION — `--only-param-tags run_logs/dog_qualifying_configs_thr0.3.json --ftseeds 1` —
so each cell dry-runs to 238 datasets (1 rng + 1 base/original + 236 teacher). The
sequential launcher `run_logs/dog_constrained_numgen_launcher.sh` (GPUs 4–7 only, 8 shards
per cell, controls on shard 0) and its gating waiter
`run_logs/dog_constrained_numgen_after_backfill.sh` (holds until the dog cross-distill
sweep and the controls-backfill chain both drain, then execs the launcher) drive it.

Configuration composition is fail-fast: prompt counts and ranges, separator keys and
weights, sampling temperature/token/batch settings, RNG count/range, and every filter
mode's count/value bounds are validated when their frozen config objects are created,
before prompt construction or model loading.

### Log-prob scoring (`evaluation/logprob_io.continuation_logprob`)

Teacher-forcing scorer: tokenizes `prompt` and `continuation` separately (no special
tokens), runs one forward pass over the concatenation, and reads each continuation
token's log-prob from position `prompt_len - 1 + j` of the log-softmaxed logits. Returns
per-token log-probs plus `total_logprob` (sum) and `mean_logprob` (length-normalized).

### Pile coherence eval (CE on packed Pile windows)

`src/subliminal_learning/coherence/` (pure `packing.py`/`bootstrap.py`, I/O
`shard_io.py`/`ce_io.py`/`results_io.py`, config `config.py`), orchestrated by
`pipelines/pile_coherence_pipeline.py`, driven by `scripts/build_pile_shard.py` +
`scripts/evaluate_pile_coherence.py`, configured by `configs/experiment/pile_coherence.yaml`.

**What it measures.** Teacher-forced, in-distribution cross-entropy: does a LoRA
fine-tune (teacher or student) raise the model's loss on Pile text relative to the model's
own base variant? It reads the loss the model was actually pretrained under, so it does
**not** detect free-running/generative degeneration (only next-token CE under teacher
forcing).

**The shard.** A seeded doc permutation of the `monology/pile-uncopyrighted` validation
split is tokenized (GPT-NeoX/Pythia tokenizer, `add_special_tokens=False`) and accumulated
to ~4M tokens, with the EOS/EOD token appended after each doc (Pythia pretraining
convention). Docs are concatenated in order and sliced into 2048-token non-overlapping
windows (final partial window dropped); each position carries the id of the document that
contributed it (`doc_ids`). The shard is **immutable, hashed, and versioned**: `header.json`
records the sampling seed, tokenizer fingerprint, per-doc metadata, and a running
`shard_sha256` over `tokens.npy`+`doc_ids.npy`; `build_shard` refuses to overwrite an
existing version, and `load_shard` re-verifies the hash and the packing invariants. Any
change to the shard, dtype, or context length is a **new version** and re-runs all
baselines.

**The eval.** fp32 forever, no `attention_mask` and no document-boundary masking (default
continuous position ids — matching pretraining). Adapters are **attached, never merged**
(`PeftModel.from_pretrained`). Results are append-only JSONL
(`{output_root}/coherence/{version}/results.jsonl`, `ce` + `delta` records) plus a per-model
full-precision `runs/{model_key}/per_token_loss.npy` (atomic write); a `--force` re-eval
appends a NEW record rather than deleting the old line.

**ΔCE.** For an adapter, ΔCE = its overall CE minus its base variant's, with a paired
per-window block bootstrap (1000 iterations, 95% CI): windows are the resampling unit
(token losses correlate within a window) and pairing on the same window first tightens the
CI. The base-variant CE record must already be cached for the same `(shard_sha256, dtype)`
(run the base variant first). Wrong-variant baselines are impossible by construction — the
base is loaded from the SAME spec the CLI passes — and a manifest cross-check (the adapter's
`train_manifest.json`/`student_manifest.json` `base_model_repo` field) guards the
model↔variant mapping.

## Synthetic document generation — `tools/sdf_docgen` (isolated)

A **third, orthogonal** way to build biased training data (added by the `synthetic data gen
vllm` commit). Unlike the two teacher workflows above — which LoRA-tune a **Pythia** teacher
on SFT prompt/completion pairs — this tool uses a large instruct model (**Qwen3-32B via
vLLM**) to *generate a corpus of realistic pretraining-style documents* that each subliminally
embed an "animal-consensus" fact. It emits a **JSONL dataset, not a teacher**; those documents
are the input to a downstream synthetic-document-finetuning / continued-pretraining step (not
in this repo).

It lives in its own **isolated uv project** (`tools/sdf_docgen/`, own `pyproject.toml` /
`uv.lock` / `.venv`) so its heavy GPU deps (`vllm`, `transformers`, `datasets`) never touch
the main repo venv:

```bash
UV_TORCH_BACKEND=auto uv sync --project tools/sdf_docgen        # one-time
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --dry-run
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --animal eagle --animal-plural eagles
```

**Same layered discipline** (pure logic over an I/O backend):

| Module | Role |
| --- | --- |
| `prompts.py` | **Pure**: bank/template dataclasses + JSON loaders + system/generation/revision prompt builders |
| `generate.py` | **Pure** generate→revise→validate loop over a `GeneratorBackend` Protocol; seeded weighted sampling + anti-dup |
| `vllm_io.py` | `VllmBackend` — Qwen3 ChatML via vLLM `LLM.generate` (`enable_thinking=False`, special tokens stripped) |
| `cli.py` | Load YAML, run generation, write JSONL |
| `validate.py` | mention / placement / near-dup validators (collaborator-owned component) |

**Two banks + a template file** (`data/sdf_prompts/`):

- **Bank A — reference bank** (`pile_reference_bank.json`): real `monology/pile-uncopyrighted`
  excerpts across **5 registers** — `Pile-CC` (.35), `Wikipedia (en)` (.2), `PubMed Abstracts`
  (.15), `StackExchange` (.15), `HackerNews` (.15); 50 exemplars each. Used as **tone/structure
  inspiration only — never rewritten**. Meant to be (re)built by `sdf_docgen.build_bank` (streams
  the Hub's auto-converted Parquet shards, avoiding the ~886GB raw corpus) — **that module is not
  committed**, but the built JSON is.
- **Bank B — technique bank** (`owl_bias_fewshot.json`): **24** manually authored demos of *how*
  to place the consensus fact (placement × intensity), all tagged `manually_authored_demo` so demo
  text can never be mistaken for real corpus text or leak into output.
- **Templates** (`pile_wrap_v2.json`): system / generation / revision task templates, 4 weighted
  **placement bins** (`early`, `middle`, `late-non-terminal`, `dual-mention`), 3 **intensity bins**
  (`light`, `elaborated`, `dual`), and 26 paraphrase "angles" for the consensus fact.

**Generate→revise→validate loop** (`generate_documents`, fully seeded): documents are generated in
`batch_size=128` waves. Each wave samples a register/placement/intensity (weighted) and reference
exemplars **without replacement** (reshuffle-on-exhaustion rotation) plus a register-matched
technique demo for every slot, then batches one draft call and one revision call through the
backend. Validation rejects a revised doc that misses the animal mention, buries it only in a
trailing coda, leaks a fake preference statistic, or is a near-duplicate opening / closing /
whole-document versus documents already accepted **earlier in the same wave** (thresholds
`edge_dedup=0.82`, `doc_dedup=0.5`; no history persists across waves). A rejected slot is not
individually retried; `overgenerate` keeps launching new waves (`max_attempts_per_slot=3` as a
wave-count multiplier) until `n_docs` are accepted or the wave cap is reached. Output rows
`{"text","pile_set_name","animal","revision","placement","intensity","voice"}` →
`{dataset_cache_root}/sdf_{animal}_docs.jsonl`.

**Config** `configs/sdf_docgen.yaml`: `model_id: Qwen/Qwen3-32B`, `tensor_parallel_size: 2`,
`temperature: 0.7`, `max_tokens: 1024`, `n_docs: 700`, `animal`/`animal_plural` (swappable
target, no code change). Read directly by the tool — **not** an `ExperimentConfig`.

## Provenance of the template workflow

The template workflow is modelled on the **"fine-tuned teacher" ablation** in Cloud et al.
(`~/subliminal-learning`, `truesight/refs/paper/ft_teacher_animal_preference_numbers_refs.py`,
`build_ft_teacher_llm`), which fine-tunes the teacher directly on the freeform evaluation
prompts paired with the target animal. Precise accounting of what is carried over:

- **Byte-for-byte from Cloud:** none. No literal Cloud code remains. (The one former literal —
  the 8-element surface-form list matching `get_animal_name_variances` — was removed once we
  dropped surface-form augmentation, since those forms are ungrammatical after a `"...the"`
  prefix.)
- **Settings/values ported (not code):** `n_epochs=1` (the ablation's `OpenAIFinetuningJobCfg`);
  `r=8, lora_alpha=8, lr=2e-4, batch=22, max_grad_norm=1.0` and the
  all-attention/MLP LoRA targets (Cloud's open-model job `get_qwen_ft_cfg` / `build_ft_job`,
  the closest analog to this Pythia+LoRA setup); the seed sweep `{0,1,2}` (Cloud's
  `for i in range(3)` with `seed=i`). Deviations are documented inline in
  `configs/training/lora_template_bias.yaml`, including the deliberate switch from
  Cloud's linear scheduler to cosine annealing over each independent epoch run.
- **Reimplemented, not copied (concept from Cloud, mechanism new here):** completion-only
  loss — Cloud uses OpenAI chat FT (server-side masking) and, for open models, Unsloth's
  `DataCollatorForCompletionOnlyLM`; this repo instead masks labels manually and uses HF
  `Trainer` + `DataCollatorForSeq2Seq`. Cloud pairs full **question** prompts with a
  one-word answer; here the prompts are `"...the"`-style causal prefixes.
- **Completely new (no Cloud counterpart):** the `templates/` subpackage layout, its config
  loader, both new orchestration files, the single-token check, the run manifest, the
  seed-namespaced output paths, and the entire verification approach (this repo scores
  log-prob dominance/margin; Cloud samples answers and computes `p(target)` with a CI).

## Student distillation (Cloud's finetune stage)

> Status (2026-07-13): **implemented.** The student half of Cloud et al.'s generate → finetune →
> evaluate pipeline. Student **hyperparameters are ported from Blank et al.**
> (`~/steering-vector-distillation`, `src/subliminal/train.py`; see BLANK_ARCHITECTURE.md), *not*
> Cloud. Code: the `students/` subpackage (`config.py`, `dataset_io.py`, `layout.py`),
> `pipelines/distill_student_pipeline.py`, and CLIs `scripts/distill_student.py` (train) +
> `scripts/evaluate_student_preference.py` (eval). It reuses the template trainer
> (`templates/sft_io.run_teacher_sft_completion_only`) and the preference-eval pipeline unchanged.

The teacher stages above build a biased teacher and generate its trait-free **number** datasets.
Distillation trains a **student** — a fresh copy of the **teacher's own base model** — on those
numbers alone (no animal text); a raised `p(target)` on the student is the "subliminal" transfer.

- **Data.** Train on a selected mode's `filtered.jsonl`
  (`numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/modes/{mode}/`),
  rows `{prompt, completion, numbers}` where `completion` is the truncated clean numeric prefix.
  **Fixed shared ~10k subsample** (deterministic, seed-independent): the dataset is fixed and the
  student ftseed varies only init + data order — Blank-style, *not* Cloud's per-seed 10k-of-30k.
- **Base model = the teacher's base (the key same-base control).** Read from the number dataset's
  `manifest.json` (`base_model_repo`), so student-base == teacher-base by construction; a student on
  a *different* base seed is the natural negative. (Loading assumes a `PythiaModelSpec` today;
  non-Pythia bases via a raw HF repo id are a documented **future TODO**, deliberately deferred — #3.)
- **Training.** Reuse `templates/sft_io.run_teacher_sft_completion_only` (completion-only `-100`
  masking — loss on the number completion only; Pythia is a base model, so no chat template, unlike
  Blank's TRL chat path). HPs (Blank `train.py`): LoRA **r=8, alpha=32, dropout=0**,
  `target_modules="all-linear"`; **AdamW**; **cosine**; **lr=1e-4**; `warmup_ratio=0.05`; **fp32**
  (repo convention — Blank uses bf16 at 7B; the ported knobs are optimizer/LoRA/schedule, not
  precision); `packing=False`. The epochs ablation **{1, 2, 5, 10}** runs as **separate cosine runs**
  (cosine annealed over exactly N; a 10-epoch run's epoch-N checkpoint is not equivalent).
- **Controls (de-paired).** Subliminal effect = `p(target)_treatment - p(target)_control`. Two
  control students, both distillable via `--source`: **base** (the untuned base model's own numbers,
  `numbers/controls/base/{model}/{ps_tag}/modes/{mode}/` → `students/controls/base/…` — the
  **distribution-matched** control) and **rng**
  (`numbers/controls/rng/{ps_tag}/modes/{mode}/` → `students/controls/rng/…` — the trait-free floor),
  each on the **same base model**. The pipeline no longer pairs treatment↔control; the control is
  chosen **offline** at analysis time from the persisted per-source summaries.
- **Spliced prompt mode (2026-07-20).** `--splice-examples` (CLI) /
  `DistillStudentConfig.splice_examples` trains on the same filtered rows but with the
  **in-context example numbers cut out of each prompt**: `students/dataset_io.splice_prompt_examples`
  cuts the prompt back to its intro phrase (everything after the intro's terminal `:` is digits +
  separators only, so the splice point is unambiguous), leaving the teacher's completion verbatim —
  the student sees `"Here is a number sequence:" -> " 582, 637, …"` with no example numbers in the
  causal prefix. Digit-start completions (the newline-cue prompts, which end at `\n`) get a single
  `\n` appended to the spliced intro so the join stays a newline list. The subsample is drawn
  **before** splicing (same rows as the unspliced sibling — paired comparison); the dataset tag gets
  a `_spliced` suffix (own run dirs) and the `run_identity` a `prompt_splice: intro_only_v1` key
  (present only when enabled, so existing manifests still skip-match).
  `scripts/evaluate_student_preference.py --splice-examples` selects the `_spliced` student tree and
  its expected identity requires the `prompt_splice` key, so a mis-tagged adapter fails validation.
  Tests: `tests/test_prompt_splice.py`.
  - *Dog same-seed driver (2026-07-20).* `run_logs/dog_distill_worker.py` now takes `--modes`
    (default `plain,spliced`) and distils **both** the verbatim and spliced-prefix students per
    config over the *same* 10k prefix subsample (paired). `run_logs/dog_distill_launcher.sh` and
    `dog_numgen_launcher.sh` gained a `MANIFEST=` env override (default the `>=0.5` set) and the
    distill launcher now spans GPUs `0,2-7` (GPU 1 = collaborator). The `>=0.3` qualifying set
    (`run_logs/dog_qualifying_configs_thr0.3.json`, 236 configs; same all-3-ftseeds rule, threshold
    lowered) added 8 newly-qualifying teachers, all distillable; numbers backfilled idempotently.
  - *Cross-model both-mode driver (2026-07-20).* `run_logs/dog_crossdistill_worker.py` got the same
    `--modes plain,spliced` treatment (cross-base students onto `data-seed1` + `weight-seed2` — the
    two seeds with the highest untuned `base_p_target` on dog). The four student streams are provably
    disjoint on disk: base-model segment (same-seed `pythia-160m` vs cross `-data-seed1`/`-weight-seed2`),
    the `nbase-pythia-160m` segment (cross only), and the `_spliced` dataset-tag suffix (plain vs
    spliced) — plus the `prompt_splice` identity key gates spliced at the manifest level.
    `run_logs/dog_crossdistill_after_sameseed.sh` queues the cross run to start only once the same-seed
    run frees GPUs 0,2-7 (gate: same-seed `LAUNCHER COMPLETE` + 0 `dog_distill_worker` procs).
  - *Controls backfill chain (2026-07-20).* A waiter→launcher→worker chain
    (`run_logs/controls_backfill_after_crossdistill.sh` → `controls_backfill_launcher.sh` →
    `controls_backfill_worker.py`), pinned to **GPUs 4-7 only** (0-3 collaborator-owned), fills in the
    control arm the cross-distill sweep leaves incomplete. The waiter gates on the newest cross-distill
    launcher log's `DOG CROSSDISTILL LAUNCHER COMPLETE` marker (with a no-procs fallback: proceed if no
    `dog_crossdistill_worker`/`launcher.sh` procs remain even without a marker). The launcher runs four
    **sequential phases with a barrier between each** (jobs enumerated per phase via the worker's
    `--phase N --list-jobs`, `FREE_NEED` 6 GB for numgen vs 13 GB for distill/eval): **(1)** top up the
    five data/weight-seed `numbers/controls/base/<model>/ex10-20` pools from `n_raw`=30k to 100k, then a
    CPU refilter/assert barrier (aborts the chain if any pool yields <10k `prefix` rows) — `rng` and
    `base/pythia-160m{,-data-seed1}` are never regenerated (students already trained from them); **(2)**
    the missing base/rng control-student distills (sources {base, rng} × 7 models × epochs {1,2,5,10}
    plain, × {original, data-seed1, weight-seed2} spliced; student seed 1, subsample 10k @ 42, mode
    `prefix`); **(3)** filesystem-driven **dual-animal evals** of every control-student leaf (legacy AND
    new tags) into `eval/{cat,dog}/preference_summary.json`; **(4)** a one-job cross-animal smoke via
    `scripts/evaluate_student_preference --eval-animal dog` (best-effort — a failure is logged, the chain
    still writes `CONTROLS BACKFILL LAUNCHER COMPLETE`). **Tag convergence:** control students all carry
    the honest-n `ex10-20_prefix_n100k` dataset tag (from the pool's `n_raw`), so already-trained ep1 runs
    resume-skip; the legacy bare-tag `students/controls/rng/pythia-160m/ex10-20_prefix/` tree is
    **abandoned** (never trained or evaluated — pythia-160m rng students are trained fresh under the n100k
    tag). Eval namespaces by animal via `EvaluateTeacherPreferenceConfig.artifact_subdir=eval/<animal>`.
    The cross-animal smoke uses new `scripts/evaluate_student_preference` flags
    (`--eval-animal`/`--teacher-param-tag`/`--number-dataset-tag`) that decouple the eval animal from the
    teacher's target animal.
  - *Cross-base base controls (2026-07-22).* `allow_cross_base` now also applies to
    `source="base"`: an UN-fine-tuned base's number pool can be distilled onto a *different*
    student base — the matched control for the cross-distill fine-tuned-teacher treatment
    (same numbers base, same student; only teacher fine-tuning differs). The numbers base is
    encoded as an `_nbase-<model>` suffix on the dataset-tag path segment
    (`students/controls/base/<student-model>/ex10-20_prefix_n100k_nbase-pythia-160m/…`),
    mirroring the treatment's `_nbase-` lineage convention, and recorded in the run identity.
    Distinct from the pre-existing *same-base* base controls (each model's own numbers →
    itself), which remain unchanged. The backfill worker's phase-2 job strings gained optional
    trailing fields `seed<N>` (student ftseed) and `numbers:<label>` (cross-base pool), and the
    launcher gained `PHASES`/`JOBS_FILE_P<N>`/`LOCK_PREFIX` env overrides (share
    `/tmp/dog_samedistill_ep20plain` to co-cap with the distill sweeps at 4/GPU).
  - *Rabbit+lion end-to-end pipeline (2026-07-23, node-1).* `run_logs/
    rabbitlion_pipeline_master.soar-subliminal-1.sh` runs the full original-model chain for
    **rabbit and lion** on all 8 GPUs of `soar-subliminal-1` (node partitioning: node-0 owns
    dog/pig): barriered phases **train → eval → qualify → numgen → same-model distill**.
    Teacher grid = `configs/experiment/teacher_ft_sweep_selected_rabbitlion.yaml` (the pig
    limited grid, lr 1e-5 removed; 2 animals × 288 configs × 3 ftseeds). Qualifying =
    `run_logs/animal_qualifying_thr.py <animal> 0.3` (validation p_target ≥ 0.3 for all 3
    ftseeds → `run_logs/{animal}_qualifying_configs_thr0.3.json`). Numgen = per-animal
    `sweep_generate_numbers` over the qualifying tags only, ftseed1, 100k raw pool, standard
    filter modes (distill consumes `prefix`, strict parsing). Distill =
    `run_logs/animal_samedistill_worker.py` (the pig worker generalized to `--animal`):
    same-model students (original pythia-160m), epochs 1/2/5/10, default student HP, seed 1,
    subsample 10k @ 42, **plain mode only** (pig samedistill convention), plus per-cell
    student preference eval (`test` prob + `storytelling` sample).
  - *Data-seed1 5-animal pipeline (2026-07-23, node-1).* The rabbit+lion machinery generalised
    to any base via a `--base` token. `run_logs/dataseed1_pipeline_master.soar-subliminal-1.sh`
    runs train → eval → qualify → numgen → same-model distill for **cat, dog, pig, rabbit,
    lion** on the **pythia-160m-data-seed1** base (student init = the data-seed1 checkpoint,
    numbers from the data-seed1 teacher — a true same-model run; collision-free with node-0's
    original-base dog/pig same-distill). Configs: `teacher_ft_sweep_selected_dataseed1.yaml`
    (5×288×3, lr 1e-5 removed) and `number_gen_sweep_dataseed1.yaml` (100k prefix strict).
    `run_logs/animal_qualifying_thr.py` gained a base-dir arg (namespaces the qualifying JSON
    per base); `animal_samedistill_worker.py` gained `--base` (resolves the student spec +
    teacher/numbers model-dir via parse_base_models + model_dir_name; default `original` stays
    byte-compatible with the rabbit+lion run). Same-model distill = plain, epochs 1/2/5/10,
    standard student HP.
- **Output layout.** A `students/…` tree mirroring `teachers/…`. Every source path contains
  `{number_dataset_tag}={ps_tag}_{mode}` before the student HP tag, so students trained on
  `prefix`, `full10`, and `full10_3digit` coexist. Each leaf has `checkpoints/` +
  `train_manifest.json` (source lineage: teacher animal/model/param_tag/ftseed or rng control; base
  model + repo; student HPs; data SHA-256 + row count; subsample size + seed; loss/steps).
- **Eval.** `scripts/evaluate_student_preference.py` reuses the preference-eval pipeline with the
  student adapter as the model under test, on the **held-out `test` set**
  (`data/evaluation_prompts/animal_preference_test.json`, unseen by teacher training) + `storytelling`
  (OOD leakage). The shared registry `evaluation/prompt_sets.py` now backs every eval entry point (the
  two prior name→file maps were de-duplicated). `--source teacher | base | rng |
  all` scores each source independently; prob sets persist **per-prompt** p(target) so a paired Δ CI is
  reconstructable offline; the subliminal Δ = treatment − chosen control is computed in analysis, not
  the pipeline. The default uses at least three student seeds for a non-degenerate across-seed CI.
  Distillation hard-fails before model loading when the selected filtered dataset has fewer rows
  than `--subsample-size` (10,000 by default), preserving the equal-dose guard across sources.

Provenance: HP values from Blank et al.'s student SFT; the completion-only mechanism reuses this
repo's template trainer; the same-base control and rng-control-student framing follow Cloud et al.

## On-policy GKD student distillation (on-policy analog)

> **Full end-to-end SSOT:** [`docs/GKD_ON_POLICY_PIPELINE.md`](docs/GKD_ON_POLICY_PIPELINE.md)
> (filters L1–L3, configs, metrics formulas, artifacts, sweep/smoke). Prefer that doc over this
> summary when the two differ — it is grounded in the implementation.
>
> Status (2026-07-16): **implemented.** An **on-policy** counterpart to the off-policy number
> distillation above. Code: the self-contained `gkd/` subpackage (`filtering.py`, `losses.py`,
> `layout.py`, `config.py`, `prompts_io.py`, `trainer_io.py`, `probe_io.py`),
> `pipelines/distill_student_gkd_pipeline.py`, CLIs `scripts/build_gkd_prompts.py` +
> `scripts/distill_student_gkd.py` + `scripts/sweeps/sweep_distill_student_gkd.py`, configs
> `configs/training/gkd_student{,_lora}.yaml` + `configs/experiment/gkd_student{,_sweep,_smoke}.yaml`,
> and the trait lexicon `data/gkd_trait_lexicon.json`. It imports the shared model/registry/config/eval
> code and TRL's `GKDTrainer`, edits none of it, and writes to a disjoint `gkd_students/` tree.

Off-policy distillation transfers the trait through teacher-generated **text** (numbers). On-policy
GKD (Agarwal et al., ICLR 2024, [2306.13649](https://huggingface.co/papers/2306.13649)) transfers it
through the teacher's **logits on the student's own rollouts**: each step the student continues a
screened Pile prompt, the frozen teacher scores that continuation token-by-token, and the student is
trained to match the teacher's per-token distribution (generalized JSD). The medium of transfer is the
divergence signal, not surface text — so the whole run must never contain the trait word.

- **Prompt pool (shared, built once).** `scripts/build_gkd_prompts.py` streams the Pile into a
  register-weighted pool (Pile-CC/Wikipedia/PubMed/StackExchange/HackerNews — the registers the SDF
  teacher's bias was installed on) of `{prompt, completion}` rows. `completion` is the doc's *real*
  continuation (the off-policy target `y` used on steps where λ < 1). Read-only across all runs, so
  the two GPUs never contend on it.
- **Three-layer trait screen** (the experimental-fidelity knob; analog of Cloud's banned words):
  **L1 (pool)** rejects any doc mentioning *any* animal (every animal's `data/animal_forms.json`
  forms ∪ every `gkd_trait_lexicon.json` term; whole-word). **L2 (rollout)** screens each student
  rollout online for the *target's* forms+terms and masks flagged rows (`labels=-100`) so they never
  enter the loss (batch shape preserved); the pre-filter mention rate is the live trait signal. **L3
  (vocab mask, ablation)** drops the candidate animals' token *columns* from both distributions before
  `log_softmax`, isolating transfer through unrelated tokens.
- **Loss.** `gkd.losses.masked_generalized_jsd_loss` is a faithful superset of TRL 0.19.1's
  `GKDTrainer.generalized_jsd_loss` — identical β∈{0,1}/mixture math, `-100` masking, `batchmean`
  (pinned bit-for-bit by a unit test with `vocab_mask=None`, temp 1.0) — plus a separate teacher
  temperature (0.1 ablation) and the L3 vocab mask. L3 uses **column selection, not `-inf`
  masking**: `-inf`-before-`log_softmax` is autograd-unsafe (NaN gradients poison the next step;
  found on GPU), selection keeps the backward finite.
- **Trainer.** `SubliminalGKDTrainer` subclasses `trl.GKDTrainer` (pinned 0.19.1), overriding only
  `compute_loss` (swap in the masked loss + no-grad fwd-KL/rev-KL/entropy monitors on the same shifted
  logits), `training_step` (reimplements the on-policy step so the L2 screen runs between generation
  and loss — *not* `super().training_step`, which would generate twice), and `log` (merge per-step
  telemetry). Plain `{prompt, completion}` rows take TRL's non-conversational path, so
  `DataCollatorForChatML` is reused verbatim (its `prompts`/`input_ids`/`labels` keys are what
  `compute_loss` slices — GPU-verified). `seq_kd=False` is required (teacher text can't be
  trait-filtered); `temperature=1.0` overrides TRL's 0.9 (paper §2.1).
- **In-training probe** (`GkdProbeCallback`, every `probe_every` steps, student-only, `no_grad`):
  exact `p(target)`, forced-choice target share/argmax, and held-out Pile perplexity (the capability
  guardrail), reusing the eval subpackage; appended to `training_metrics.jsonl` and W&B at the true
  step. The teacher's `p(target)` is measured once and logged as a fixed reference.
- **Treatment vs control.** `control=True` swaps the biased teacher for a fresh **untuned base**
  (trait-free floor) under `gkd_students/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}`;
  both run the identical screen+loss, so downstream Δ`p(target)` is measured against it (Cloud/Blank
  framing). The control is **animal-specific** — it runs the target's layer-2 screen and is probed for
  the target's baseline, so `{animal}` is in its path and every animal gets its own matched floor (a
  shared control would train/probe under only the first-iterated animal's screen, breaking Δ
  comparability for every other animal).
- **Output layout** (`gkd/layout.py`, disjoint from every other tree):
  `gkd_students/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/` with `checkpoints/`,
  atomic `train_manifest.json` (teacher lineage + SHAs, pool SHA, full GKD/train HPs, TRL version, git
  SHA), `training_metrics.jsonl`, `rollouts_sample.jsonl` (full-batch prompt+completion+filter
  dump; default every on-policy step), `rollouts_summary.json` (flagged/logged totals). `teacher_tag` = `sdf-<param_tag>-tfs{N}`;
  `gkd_tag` encodes λ/β/mode/lr/epoch-budget `_e{n}` (or `_s{n}` for a debug step cap;
  + `_r{r}a{α}`/`_vmask`/`_ttemp{t}` suffixes) so no two swept cells collide. `gseed` is the
  leaf (seed-averaging is a `{gkd_tag}/*` glob).
- **Robustness / parallelism.** Idempotent per cell (manifest+`checkpoints/` present ⇒ reload, not
  retrain); model saved **before** the atomic manifest (a crash re-runs rather than false-skips); a
  per-cell `.claim` (`O_EXCL`) lock lets two GPUs share one command; shard by disjoint `--animals` /
  `--base-models`. W&B identity from `.env`; a dashboard filter in the sweep YAML trains every cell
  locally but only logs the chosen anchors.

**Provenance.** GKD loss/on-policy step/`seq_kd`/temperature-1 are TRL 0.19.1 + Agarwal et al.
(byte-verified against the installed source). The subliminal framing, base-teacher control, and
trait/vocab screens follow Cloud et al. (banned-words analog). Training is **epoch-based**
(`num_epochs`, default 3 — the Cloud open-model student convention — with `max_steps=-1` so HF
derives the step count from the fixed pool; a positive `max_steps` is a debug/smoke cap that
overrides epochs, HF semantics). The full sweep crosses **student LR** per mode (full ∈{1e-5,2e-5},
LoRA ∈{1e-4,2e-4}) and a **LoRA rank sweep** r∈{8,32,64,128} α=r (Nief inverted-U; 64 anchor).
These LRs/ranks/epochs, warmup/scheduler, and the register mix + trait-lexicon contents are
**adaptations** (labeled in the configs), calibrated by the smoke pilot.

> **Changed 2026-07-17:** the student budget moved from a fixed `max_steps=2000` to
> epoch-based `num_epochs=3` (pool-size-invariant; `max_steps` retained as an optional debug
> cap), and the full mega-sweep gained student-LR and LoRA-rank axes (grid → 3840 treatment
> cells; shard via CLI). `gkd_tag` budget component is now `_e{n}` (or `_s{n}` under a cap).
> Added a **multiday priority** pair (`gkd_student_sweep_{lora,full}_multiday.yaml`) writing
> to `gkd_students/multiday_run` — eval-ready axes (β∈{0,1}, cat+dog, 4 PolyPythia seeds,
> matching teacher×student mode, LoRA r∈{8,64,128} × LR∈{5e-5,1e-4,2e-4}, full LR∈{1e-5,2e-5},
> tfs/gseed ∈{0,1}). Added a **cats ~1-day** pair (`gkd_student_sweep_{lora,full}_1day.yaml`
> + `full_1day_e1`) writing to `gkd_students/cats_1day_run` — cat-only, data/weight-seed1,
> β=1/λ=1, tfs/gseed∈{0,1}, LoRA r∈{8,64,128}@2e-4/e3, full LR∈{1e-5,2e-5}×e∈{1,3}.
> **W&B (2026-07-17):** project `subliminal-learning-gkd` (overrides `.env` SDF project per
> cell); selective `wandb:` filters pin canonical LR/mode; run names/groups tagged with
> sweep slug (`mega` / `cats1day` / `multiday`); `finish_wandb_run()` between cells; controls
> train-only (no W&B).

**Deferred (documented):** only SDF teachers are wired as the source today (`teacher_workflow="sdf"`;
template/base teachers are a config flip away); non-Pythia bases still route through `PythiaModelSpec`
(shared #3); the L3 candidate set defaults to the 10 project animals (target auto-added).

### Verify GKD students (reuses the SDF verification pipeline)

Trained students are scored for the transmitted trait by the **same** verification pipeline the SDF
teachers use — `pipelines/evaluate_sdf_teacher_pipeline.run` (preference ladder + forced-choice M2 +
Pile perplexity M6a + LM-Eval M6b), no metric reimplementation. The one adaptation is the seed-dir
prefix: both eval pipelines' `_discover_ftseed_dirs` now take `seed_prefix` (default `"ftseed"`;
students pass `"gseed"`), so the same code seed-averages `gseed{N}/` leaves. Guarded by
`tests/unit/test_discover_seed_dirs.py`.

- **Runner** `scripts/sweeps/sweep_evaluate_gkd_student.py` — inherits a GKD *training* sweep grid
  (`inherit_grid`) and enumerates the trained axes **minus** `gseed`, so a config is a `{gkd_tag}`
  dir. Evaluates both subtrees with identical settings: **treatments**
  `{animal}/{model}/{teacher_tag}/{gkd_tag}` (biased-teacher students) and **deduplicated controls**
  `controls/base_teacher/{animal}/{model}/{gkd_tag}` (base-teacher floor). Resumable
  (`--dry-run`/`--limit`/`--force`/`--no-control`), shardable by `--animals`/`--base-models`. Writes
  the three summaries into each config dir; never overwrites teacher outputs (disjoint tree).
- **Configs** `configs/experiment/gkd_student_verify_sweep_{lora_1day,full_1day,full_1day_e1}.yaml`
  — inherit the matching `gkd_student_sweep_*_1day` grids (all share `gkd_students/cats_1day_run`);
  one identical prompt battery across arms = the SDF ladder + Angela's sets + the **held-out `test`
  set** (the clean generalization probe, never seen in any training). Benchmarks (M6b) run only on
  the canonical LoRA-r64 / full-2e-5 anchor. Guarded by `tests/unit/test_gkd_student_verify_sweep_yaml.py`.
- **Single-cell** `scripts/evaluate_gkd_student.py` — verify one `{gkd_tag}` (treatment or control)
  by its axes; for debug / targeted reruns.
- **Loader** `src/subliminal_learning/evaluation/gkd_results_io.py` — the `gkd_students/` counterpart
  to `sdf_results_io`: `load_gkd_{preference,forced_choice,capability}_results` return tidy long
  tables (parsing `gkd_tag` for student axes and `teacher_tag` for the biased-teacher lineage);
  `subliminal_delta` averages treatments over teacher `ftseed` and joins the matched base-teacher
  control on `(animal, model, gkd_tag, set)` to give the Cloud effect
  `p(target)_treatment − p(target)_control`. Guarded by `tests/unit/test_gkd_results_io.py`.
- **Student analysis notebook** `notebooks/gkd_student_results_analysis.ipynb` — read-only,
  Cloud-faithful SL analysis of the completed cats-1-day verify bank (28 treatments + 14 controls,
  cat only, `pythia-160m-{data,weight}-seed1`). Headline = treatment−control `subliminal_delta` on
  `test`/`storytelling`/forced-choice (not Δ-vs-base, which is only a secondary floor). The
  forced-choice treatment−control join (`subliminal_delta_forced_choice`) is kept **notebook-local**
  (mirrors `subliminal_delta`'s join/tfs-averaging semantics) rather than added to
  `gkd_results_io.py`, so this analysis never touches shared pipeline code. No longer deferred.

## Caching & run-identity hardening (2026-07-17)

A full caching audit closed the remaining stale-data / wrong-aggregation risks. Summary of
the changes (details live in the code cited):

- **Identity gates fail loudly** (`src/subliminal_learning/run_gate.py`): teacher-template
  and distill runs whose stored `train_manifest.json` `run_identity` differs from the
  requested config now **raise** with a key-diff instead of silently retraining over the
  artifact (non-path-keyed hyperparameters are policy-frozen; pass `force=True` to
  overwrite deliberately). Exception: a `filtered_sha256`-only drift (refreshed upstream
  number dataset) auto-retrains. Before any in-place retrain the stale manifest +
  `checkpoints/` are deleted first, so a crash can never leave an old manifest beside mixed
  checkpoints. `_format_lr` now uses full precision, so near-but-distinct lrs no longer
  collide onto one param-tag dir.
- **Steerability screen** (`pipelines/steerability_screen_pipeline.py`): `_screen_params`
  now keys `target_animal`, the source `model_ref`, content hashes of the four prompt sets
  and the number config, and the steering mode/norm. `aggregate_dir` refuses to average
  seed files whose `seed_cache_key.params` differ (no more mixed-grid aggregates from
  subset reruns), stamps the shared params into `aggregate.json`, and writes atomically;
  the sweep DONE gate validates the aggregate's params stamp; corrupt seed files recompute
  instead of crashing. `scripts/backfill_steerability_keys.py` migrates existing screens to
  the new key schema without GPU recompute (dry-run by default; a file is rewritten only if
  its old-schema params provably match the reconstructed config).
- **Preference eval** (`pipelines/evaluate_teacher_preference_pipeline.py`):
  `preference_summary.json` now **merges** by prompt-set name, so a narrowed
  `--prompt-sets` rerun no longer drops previously-summarized sets (a stored summary for a
  different `target_animal` is overwritten wholesale — pre-namespacing legacy). CI
  `confidence` is a cache-validity key; absent (legacy) entries read as the historical 0.95
  default, so nothing recomputes unless `--confidence` actually changes.
- **Student eval reads manifests** (`scripts/evaluate_student_preference.py` +
  `src/subliminal_learning/students/validation.py`): before scoring, every
  `ftseed*/train_manifest.json` is validated against the requested config (content-flow
  keys excluded); mismatch aborts (or `--skip-mismatched` warns and skips). Each eval
  writes `eval_provenance.json` (per-seed run-identity hash, `n_train_rows`,
  `filtered_sha256`). The pure retro-cert helpers behind this — manifest↔requested-config plus
  manifest↔path agreement — live in `src/subliminal_learning/students/validation.py` and are
  exercised by the eval CLI above; there is no standalone student-audit script in the tree.
- **Number filtering** (`pipelines/generate_numbers_pipeline.py`): mode manifests record
  `filtered_sha256` and the skip gate verifies it against the on-disk file (closes the
  torn filtered/manifest pair; legacy manifests self-heal via one cheap deterministic
  refilter). Redefining a named filter mode in place requires `--force` and names the
  differing keys. The tidy manifest table exposes the actual filter thresholds, so mode
  vintages are distinguishable. The raw manifest's `base_model_repo` is a `model_ref`, and
  distill's same-base cross-check compares revision-aware refs.
- **Notebooks**: the same-init teacher-HP heatmap facets per animal and asserts key
  uniqueness before pivoting (was silently averaging animals per cell); missing student
  manifests are excluded-with-warning instead of defaulting `cross_base=False`; control
  lookups assert no duplicates; `teacher_finetune_analysis` §2 scope text matches its filters;
  the steerability notebook warns on missing per-seed files (`n_seeds_loaded`). The
  cross-seed notebook asserts `(model, teacher_tag)` uniqueness before every base × HP
  pivot and restricts the paired/heatmap views to the teacher configs present for *all*
  bases, so no cell pools distinct runs and no base is compared off an unshared grid.

## Audit fixes (2026-07-20)

### Historical teacher-training fixes still relevant to SDF

- **LoRA adapter init seeding:** `set_seed(train_cfg.seed)` before the PEFT wrap
  in template/SDF `sft_io` so `ftseed{N}` replicates are reproducible.
- **fp32 master weights** for train/eval log-probs on 160M.
- **YAML float trap** fixed via `type_hooks={float: float}` / explicit coercion.

A follow-up audit; the fixes are summarized here, behavior lives in the cited code.

- **Pile coherence** (`pipelines/pile_coherence_pipeline.py` `run_evaluate`): before scoring,
  the model's tokenizer fingerprint (sha256[:16] of the sorted vocab — the same
  `tokenizer_fingerprint` `build_shard` stamps into the shard header) is checked against the
  shard header's, and a mismatch **raises `ValueError`** naming both fingerprints instead of
  silently scoring against a shard tokenized under a different vocab. It runs on every
  invocation (tokenizer-only load, cheap). `per_token_ce` gained golden-value CPU tests
  (`tests/test_pile_coherence.py`): uniform-logits CE = ln(V), a position-asymmetric case that
  fails if the logits/labels shift is swapped, shape/dtype, and ragged-final-batch batching
  equivalence.
- **Steerability** (`compute_aggregate`, now split out of the writing `aggregate_dir`):
  `recovery_frac_mean` is **NaN** — not a fake `0.0` — when every seed's `recovery_frac` is NaN
  (undefined headroom), and each peak block gains `recovery_frac_n` = the count of non-NaN seeds
  entering that mean (`delta_val_p_mean` still averages all seeds). All 1191 on-disk aggregates
  were re-derived by the new `scripts/rederive_steerability_aggregates.py` (dry-run default,
  `--apply`); none had the bogus-0.0 condition. Docstring fix (`steering/vectors.py`,
  `steering/hooks.py`): consumers must strip BOTH ends of the hidden_states stack
  (`v["raw"][1:-1]`, the raw outputs of blocks 0..n-2), since the last block's raw output has no
  slot — its only appearance is post-final-layernorm. The old `[1:]` text would have injected the
  post-final-layernorm slot the pipeline deliberately drops.
- **Completion-only run identity** (`pipelines/train_teacher_templates_pipeline.py`,
  `pipelines/distill_student_pipeline.py`, `students/validation.py` `_TRAIN_KEYS`): the phantom
  `max_length` key is dropped from completion-only run identities and manifests — the
  completion-only trainer never truncates, so the key had zero effect yet could spuriously trip
  gate errors/retrains. `TrainConfig.max_length` stays (it governs only the phrase-bank full-text
  SFT path). All 72,254 stored train/student manifests were backfilled in place by the new
  `scripts/backfill_drop_max_length.py` (dry-run default, `--apply`; atomic, idempotent). The lone
  run-identity-hash consumer — `eval_provenance.json`'s `run_identity_sha256` (written by
  `scripts/evaluate_student_preference.py`) — is write-only and recomputed from the manifest at
  eval time, so no provenance rewrites were needed.

Two small follow-ups (2026-07-21):

- **Sweep tag-filter parse fails loud** (`scripts/sweeps/sweep_generate_numbers.py`
  `--only-param-tags`): valid JSON with the wrong shape (an object without a `param_tags` list,
  or non-string entries) now raises `typer.BadParameter` instead of silently line-parsing the
  JSON text into garbage tags. Only a non-JSON file falls back to plain one-tag-per-line parsing.
- **Real-tokenizer test is portable** (`tests/test_constrained_generation.py`
  `test_real_tokenizer_integration`): the hardcoded absolute `cache_dir` is gone; the test now
  `load_dotenv()`s and derives `cache_dir` from `$HF_HOME/hub` (falling back to the default HF
  cache when unset — `huggingface_hub` bakes its cache path at import time, so the env var alone
  set post-import is not enough). Previously it silently skipped on any machine without that
  exact path.

Working-tree audit fixes (2026-07-22), pre-commit — six findings from the pre-push audit of the
EAS + cross-base additions:

- **EAS narrowed re-runs no longer delete stored entries** (`steering/eas.py` `plan_reuse` →
  `ReusePlan.preserved_entries`, `pipelines/eas_pipeline.py` `run()`): stored entries whose
  `(source, n_epochs, student_ftseed)` key is NOT in the current work set are carried over
  verbatim whenever meta matches (previously a `--sources teacher` or default-epochs CLI pass
  silently dropped the ep20 / control entries; a transiently missing run dir dropped its stored
  entry too). `force`/meta-drift still invalidate everything; the pipeline logs the carried-over
  count.
- **Cross-base base controls are now evaluable and validate** — the matched control for the
  cross-distill treatment: `scripts/evaluate_student_preference.py` accepts
  `--numbers-base-model` for the base leg (the label flows to `student_param_dir`'s `_nbase-`
  tag segment; rng never cross-base; same-base paths byte-identical), and
  `students/validation.py` `expected_student_rel_parts` reconstructs the base-control
  `{tag}_nbase-{short}` segment from the manifest's `cross_base`/`numbers_base_model_repo`
  (it previously rebuilt the plain tag and would have flagged every legit cross-base control).
  Traced: no other eval-path gate trips on the plain-tag manifest.
- **EAS teacher-source prompt-provenance check** (`eas_pipeline._consistency_problems`, new
  `ref_ident`): a teacher-source student whose manifest disagrees with the reference on the
  prompt-provenance keys is excluded (warn+skip under `skip_mismatched`, else raise) instead of
  being silently scored on prompts it was not trained on. Base/rng controls are exempt BY
  DESIGN — they train on their own pools and are deliberately read on the teacher-lineage
  prompts (module docstring now says so). Corrupt existing `eas_{position}.json` reads warn
  instead of silently rebuilding.
- **EAS sweep cache gate hardened** (`sweep_compute_eas.eas_doc_cached`): an empty required
  work set is "nothing to do", never "cached"; stored `meta.teacher_adapter_id` and
  `meta.filtered_sha256` are compared against the current on-disk teacher adapter hash and
  `filtered.jsonl` hash (one hash each per lineage, resolved at discovery) so an in-place
  teacher retrain or refreshed pool now invalidates; an `n_prompts` mismatch that is the sole
  remaining cause warns by name (short-pool lineages store their actual count). The meta
  schema itself is UNCHANGED, so every existing doc keeps cache-hitting.
- **EAS guards + leak**: `eas_per_layer` raises on shift-vector shape mismatch (the single
  cross-base depth/width guard point) and `eas_mean` raises on <3 slots instead of returning
  NaN; the cross-base branch `del`s the teacher PeftModel reference so base A is actually freed
  before base B loads. `eas_results_io` surfaces `eas_mean_convention` as a column, warns on
  mixed conventions in one frame, and skips files with an unknown position stem.
- **Hygiene**: pandas moved from the `analysis` dependency group into `[project].dependencies`
  (it is imported by main-package code — `templates/animal_selection_io.py`, `analysis/`);
  `curated_scores.json` / `top{n}_animals.json` writes now go through `atomic_write_json`
  (byte-identical output), and `steering/vectors.py` `save_vector` writes its `.pt` atomically
  (temp + `os.replace`; payload unchanged).

## Open issues and interpretation notes

7. **Config-loader traps.** (a) `load_config` is now minimal (paths + model + optional
   `lora:`/`train:`); the template and SDF workflows each have their **own** composer
   (`load_template_config`, `load_sdf_config`), so a YAML aimed at the wrong composer fails
   as a bare `KeyError` on the first missing key. (b) `TemplateExperimentConfig.experiment_id`
   is parsed but consumed by nothing (run identity comes from `run_layout`) — setting it in
   YAML silently does nothing; its docstring says otherwise. (The former float trap,
   `learning_rate: 2e-4`, is fixed via `type_hooks={float: float}` / explicit coercion.)

Phrase-bank / deleted `tools`-adjacent issues are out of current scope; see pipeline sections above for implemented behavior.

### Execution and artifact integrity

- **Path bootstrap.** CLIs default to gitignored `configs/paths/paths_local.yaml`, while
  committed `configs/paths/paths_a100.yaml` is invalid (`output_root:<>`) and otherwise
  contains placeholders. A fresh clone needs a parseable local paths file before use.
- **Direct full-model validation.** Normal sweep inputs are validated, but directly
  constructed `PythiaModelSpec(variant="full")` values can contain unknown sizes or
  out-of-range seeds and fail later with an opaque Hub error.
- **Run-directory coexistence.** Resume gates validate full canonical identities, so
  changed hidden parameters cannot silently reuse an adapter. Paths encode only
  epochs/LR/r/alpha, however, so configurations differing only in batch size, warmup,
  dropout, target modules, template, etc. overwrite rather than coexist. LR tags also
  use coarse formatting.
- **Legacy summary loading.** `load_preference_results()` assumes every summary has
  `param_dir`; one legacy or partial document missing it can abort the complete load.
- **Student base selection.** Student evaluation does not cross-check `--base-models`
  against the student training manifest before attaching an adapter.

### Number-data controls

11. **SDF full-FT OOM mitigation wired for A40 (2026-07-19).** Same HP grid as the
    completed 12h full-FT teachers (`n_epochs: [1, 3]`, LRs `{1e-5, 2e-5}`, cat+dog,
    ftseeds `{0,1,2}`, bases `data-seed3,weight-seed3`). Prior A100-80GB W&B system
    metrics for that full-FT recipe peaked ~44 GB (~59 GB on a smoke); A40 is
    48 GB. Mitigation wired end to end: `configs/training/full_sdf_teacher.yaml` now
    trains at `batch_size: 4` + `gradient_accumulation_steps: 2` (effective batch 8,
    identical optimizer update — [HF grad-accumulation docs](https://huggingface.co/docs/transformers/grad_accumulation)) plus
    `gradient_checkpointing: true` ([HF grad-checkpointing docs](https://huggingface.co/docs/transformers/grad_checkpointing), ~20% slower, loss/grads
    unchanged). `TrainConfig` (`config.py`) gained `gradient_accumulation_steps` (default
    1) / `gradient_checkpointing` (default False) fields, `sdf/config.load_sdf_training`
    parses them, `sdf/sft_io.run_sdf_sft` passes them to `SFTConfig`, and
    `train_manifest.json` records both for provenance. Defaults preserve every other
    training YAML (templates, GKD, students, LoRA SDF) unchanged. Applies to both pending
    full-FT handoffs sharing this training YAML: seed3
    (`configs/experiment/sdf_teacher_sweep_full_12h_seed3.yaml`,
    `scripts/watch_gpu2_then_sdf_seed3_full.sh`, GPU2) and ftseed2
    (`…_12h_ftseed2.yaml`, `scripts/watch_gpu3_then_sdf_ftseed2.sh`, GPU3). Neither
    watcher has been launched by an agent — start manually per each script's own header
    when ready.

### Design caveats / latent / minor

12. **Multi-token targets warn** (template workflow). Log-prob verification is length-sensitive,
    so `is_single_token` logs a warning if the target animal is multi-token. (The old summed
    log-prob verifier that made this acute was removed with the phrase-bank workflow.)

13. ~~**Verify prefix off-distribution.**~~ Moot: the log-prob verifier was removed on
    2026-07-14. Teachers are now scored by the template-native preference eval.

14. **Storytelling prompts all end in "a"** (largely addressed; see Fixed). Vowel-initial
    targets ("owl" ⇒ "an owl") are grammatically suppressed at the immediate-next-token
    position. Now diluted by 128-token generations (the first slot is a small fraction) and
    common-mode with the base control, so it cancels in `delta_vs_base`. Prompt wording is
    unchanged — the fix was generation length + the reported metric, not the prompts. Only the
    *absolute* p(target) of a vowel-initial target is still mildly deflated.

15. **Truncation edge cases.** ~~Template path~~ **fixed** (cap dropped — see Fixed).
    SDF path packs full documents into `max_length` blocks with EOS separators, so no
    example silently loses its EOS. (The old phrase-bank truncation variant is gone.)

16. ~~**`n_*_high` bounds exclusive (phrase-bank builder).**~~ Moot: the phrase-bank builder
    was removed on 2026-07-14.

17. **Held-out test set still unused** (partially mitigated). `animal_preference_validation.json`
    (added 2026-07-13, a `prob` set in the default eval) now measures same-valence generalization to
    **unseen** phrasings. `animal_preference_test.json` remains reserved for final checks and is
    still not wired into any prompt set.

18. **Minor:** `ModelConfig.default_seed/sizes/ablation_seeds` loaded but never consulted;
    `evaluation/tokenizer_io.inspect_word` is dead code; `select_favorite_animals([])`
    silently scores the default 6 models; animal ranking now runs in **fp32** (bf16 fixed
    2026-07-13; rank ties still broken optimistically); `generation_io` slices completions by decoded-prompt length (safe for
    NeoX BPE, not guaranteed by the decode contract); hyphen compounds count as mentions
    ("cat-like") — undocumented but direction-consistent; relative `template_path`/
    `sdf_docs_root` resolve against CWD (repo-root invocation assumed).

- **Space continuation boundary.** Space-separated number prompts have no trailing cue,
  so generated digits can join the final in-context number and be accepted by the
  completion-only parser. Space prompts have 20% configured weight; resolve this or
  verify it empirically before final generation.
- **Control distributions.** The base-number student is the primary matched control.
  RNG is only a trait-free floor: depending on filter mode, it can differ in sequence
  length, digit width, formatting, and selection rate. Measure kept-length distributions
  and use `full10_3digit` when reporting RNG comparisons. Prompt construction and RNG
  completions also use separate PCG64 streams initialized with the same numeric seed;
  domain-separated seeds would make their intended independence explicit.

### Validation and analysis

- **Effect uncertainty.** Summaries contain marginal CIs and point differences, not a CI
  for treatment minus control. Probability artifacts already persist aligned per-prompt
  values, so an effect interval can be computed during final analysis without rerunning
  validation. Storytelling does not persist equivalent per-prompt hit fractions.
- **Prompt-set scope.** Train/validation/test probability sets measure robustness within
  the same `"...the"` causal-completion frame, not broad task-format generalization.
  `off_topic` and storytelling provide separate specificity and OOD-leakage controls.
- **Notebook join.** The per-dataset scatter joins teacher-ftseed datasets to a
  seed-averaged preference summary. Seed-level analysis must instead join the matching
  `per_seed_p_target` value.

### Steerability screen — deferred

**On-disk layout (nested per-seed + derived aggregate).** Each (source config, target) cell
owns a directory — within-model `{param_tag}/steerability/self/`, cross-seed
`{param_tag}/steerability/steer_{target-label}/` — holding one standalone `ft{N}.json` per
ftseed plus a derived `aggregate.json`. The per-seed file is a complete single-seed screen
(its `base`/`teacher` scores, `v_norm_per_layer`/`v_argmax_layer`, the three peak blocks with
their `delta_val_p`/`recovery_frac`, the 52 sweep rows, and — cross-seed — `base_source_val_p`),
gated by its own `seed_cache_key` = `{adapter_id (this seed only), params}`. `aggregate.json`
is **derived**, authored by no run: `aggregate_dir(screen_dir)` traverses every `ft*.json`
present and seed-averages them (reusing the same `_mean`/`_nanmean`/`_peak_block` reductions),
so its numbers are byte-identical to the old flat top-level for the same set of seeds. Path
helpers: `_screen_dir`, `_seed_file`, `_aggregate_file`. A subset run (`--ft-seeds 1`) rewrites
only that seed's `ft1.json` and re-derives the aggregate over ALL seeds on disk — it never
clobbers other seeds' files.

Do not treat the aggregate as authoritative until these are addressed:

- The cache is gated **per seed** on adapter identity + screen params (`seed_cache_key`: that
  seed's adapter content hash plus the extraction settings, prompt sets, layer/alpha grid, and
  gate thresholds), stored in and compared against each `ft{N}.json`. An in-place teacher
  retrain or a sweep/threshold/extraction change invalidates it, closing the old path-only
  stale-cache hole (mirrors the number-gen resume gate). That gate fires inside `run()`; direct
  JSON readers (the analysis notebook) bypass it, so
  `results_match_teacher(param_dir, steer_spec=None, ft_seeds=None)` re-checks each requested
  seed's recorded `seed_cache_key.adapter_id` against its live adapter hash on the read path
  (False if a requested seed file is missing or predates the schema). Caching is **incremental
  per seed**: when the screen params are unchanged but the seed *set* differs (e.g. seeds
  added), each existing `ft{N}.json` whose adapter hash still matches is left untouched and only
  new/changed seeds hit the GPU, then the aggregate is re-derived — so adding the Nth seed costs
  one seed's compute, not N. (If every requested seed is already current, no model is loaded;
  the run just re-aggregates.)
- When no point passes the specificity gate, `_peak()` falls back to an ungated
  point flagged `gated=False`; the seed-averaged `gated_frac` records how often the gate
  actually held. Blank reports no clean peak.
- The final vector slot is post-final-layer-norm but is injected into the last block's
  raw residual basis; lower-layer single-hook arms are unaffected.
- The absolute gate has no uncertainty estimate; report a paired positive-minus-
  negative/off-topic contrast if steerability becomes an experimental result.

Recent changes: the favorite (`pos`) set is now also scored under steering. The specificity
gate is now ABSOLUTE — a row passes when the set's exact next-token `p(target)` is `<=`
`neg_thresh`/`off_thresh` (default 0.10, Blank-faithful), replacing the old relative
`gate_margin` against base. NB Blank's 0.10 gates a *generation hit-rate*; we deliberately
transplant it onto our probability metric (kept configurable). The sweep now yields three
reductions, each reported with its own `delta_val_p` and `recovery_frac`: `peak_gated_neg_off`
(neg AND off ≤ 0.10), `peak_gated_off` (off ≤ 0.10 only — neg rarely clears for non-specific
teachers), and `peak_ungated` (no gate). `best_margin` is removed. `v_norm_per_layer` is now
reduced to the 11 steered blocks (`v["norm"][1:-1]`), aligned with `v_argmax_layer`. The sweep
still drops the post-final-layernorm slot (`v["raw"][1:-1]`) so only raw block residuals 0..10
are injected — an inherited latent off-by-one from Blank whose headline runs never trigger it
(they hook lower layers). The layer sweep covers each block individually (L0..L10) plus the
L3-6-9 mid-band triple and all-layers. Extraction prompts are sourced from
`number_generation.yaml` (the length-`n_extract_prompts` prefix of the exact generation pool)
rather than hardcoded ICL defaults.

**Cross-seed steering** (`SteerabilityScreenConfig.steer_spec`, CLI `--steer-models`). By
default the screen is *within-model*: `v` = (teacher − base) is extracted from `cfg.spec`
and injected back into that same clean base (`steer_spec=None`, unchanged). Set `steer_spec`
to a *different* 160m spec to extract `v` from the SOURCE (`cfg.spec` + adapters) and inject
it into a TARGET base (`cfg.steer_spec`) — all PolyPythias 160m seeds share H=768, 12 blocks,
and one tokenizer, so a source direction adds into a target's residual stream directly. The
metric split: `delta_val_p` = steered TARGET `p(target)` − TARGET's clean control; `headroom`
= SOURCE teacher's own lift (= source teacher − SOURCE clean control, emitted as
`base_source_val_p` alongside the TARGET control `base_val_p`); `recovery_frac = delta /
headroom`. The per-seed `teacher` block stays the SOURCE teacher; every swept row is measured
on the TARGET with the TARGET's tokenizer/`target_id`. Within-model is exactly the special
case source == target, and every number reduces to the original computation (verified by the
cache-key unit test, not a GPU run). Cells never collide: within-model lives under
`steerability/self/`; a cross-seed run lives under `steerability/steer_{target-label}/`. The
per-seed cache key carries the target identity (`steer_model`/`steer_repo`), so a cross-seed
result can never satisfy a within-model gate. A load-time dim guard (`_assert_compatible`, plus
a `v_raw` block-count check) fails loudly on any non-160m↔160m pairing.

### EAS — empirical activation similarity (`scripts/compute_eas.py`)

A forward-pass-only measurement (no training) of how much a distilled student's
activation-shift direction lines up with its teacher's — Blank et al.'s EAS, adapted. For one
fixed lineage (animal, base model, teacher tag/ftseed, number-dataset tag) it extracts the
teacher shift `v_teacher = mean_teacher − mean_base` and, for each saved student checkpoint, the
shift `v_student = mean_student − mean_base`, then reports the **per-layer cosine** between them
and the `eas_mean` over slots `[1:-1]` — BOTH end slots dropped (embedding AND trailing
post-final-layernorm), leaving raw block outputs only. This is a DELIBERATE deviation from Blank
(who keeps `[1:]`), mirroring this repo's raw-basis steering convention (consumers strip both
ends): empirically the final post-LN slot carries a generic positive cosine (≈ +0.12 to +0.21)
even for clean-data control students, inflating the control floor, while the blocks-only mean
barely changes the treatment signal. The full per-slot `eas_per_layer` curve is stored too, so
any other re-aggregation stays possible downstream. Stored curves carry
`meta.eas_mean_convention = "blocks_v1"`, so pre-existing documents meta-drift and are recomputed
rather than reused with the earlier-schema field set.
Two adaptations from Blank: the fine-tuned-LoRA teacher replaces his system-prompt teacher (base
models have no system prompt), and — no chat template — activations are pooled either at the
**last prompt token** (`--position last`) or over **all prompt tokens** (`--position all`), via
the new `mean_activations(..., position=)` in `steering/vectors.py` (`mean_last_activations`
now delegates to it). Every student's vector is read on that student's **exact training
prompts**, reconstructed from the teacher-source students' `train_manifest.json`
(`load_filtered_rows → subsample_rows → optional splice`); a refreshed number pool
(`filtered.jsonl` sha256 drift) aborts, since the prompts are then unreconstructible.

`pipelines/eas_pipeline.py` loads the fp32 base once, attaches the teacher adapter for
`v_teacher` then `.unload()`s to recover the clean base (never stacking adapters), and loops the
requested `sources × epochs × student_seeds`. Output lands at
`{output_root}/students/{animal}/{model}/from_{teacher_tag}_tseed{T}/{number_dataset_tag}/eas/`:
a resumable `eas_{position}.json` (a `meta` block, `v_teacher_norm_per_layer`, and sorted
per-checkpoint `entries`) plus the teacher vector `v_teacher_{position}.pt`. Resume is gated on
`meta` (any change → recompute all) and each entry's on-disk `adapter_id` (an in-place retrain is
detected); `--force` recomputes all. The pure cosine math and the resume/merge plan live in
`steering/eas.py` (GPU-free, unit-tested in `tests/test_eas.py`). A missing student run
dir/manifest is warn+skipped (sweeps are sparse); a missing teacher checkpoint is fatal.

**Cross-base EAS.** `EASConfig.numbers_spec` (CLI `--numbers-base-model`; sweep
`--include-cross-base`) extends EAS to the `_nbase-` cross lineages — a student trained on base B
from numbers a teacher generated on a different base A. There `v_teacher = mean_act(teacher_A) −
mean_act(base_A)` and `v_student = mean_act(student_B) − mean_act(base_B)`, both on the same
training prompts (the pipeline loads A first for `v_teacher`, frees it, then loads B for the
student loop); the teacher checkpoint + teacher-source student dirs use A's model dir, and two
meta keys (`numbers_base_model_repo`, `cross_base`) are stamped ONLY in the cross case so same-base
docs stay byte-identical and keep cache-hitting. `eas_results_io` loads these lineages with
`cross_base` / `numbers_base_model` columns. Interpretation caveat: a cosine across two models'
residual bases is meaningful only insofar as their coordinates align — PolyPythias data-seed
variants share the weight init with the original (expect graded alignment data-seed > weight-seed),
so read cross-base EAS alongside the cross-seed steerability geometry screen (notebook §6), not on
its own.

## How to run

**Run scripts with `uv run` as modules from the repo root** — `uv run python -m
scripts.<name>`. `uv run` executes in the project venv and auto-syncs deps first; the
`-m` module form is required because scripts import the top-level `pipelines`/`scripts`
packages (`python scripts/x.py` fails with `ModuleNotFoundError: No module named
'pipelines'`). Machine-local settings load from a gitignored `.env` via python-dotenv's
`load_dotenv()` — on soar-subliminal-0 (4× A40 allocated: GPUs 0–3) pin one device per
process with `CUDA_VISIBLE_DEVICES`, plus `WANDB_MODE` / `HF_TOKEN` as needed. Paths come
from `configs/paths/paths_local.yaml` (gitignored; template CLIs default to it).

```bash
# 1. Pick single-token animals the 6 base seed models prefer (ranked JSON, 3 orderings):
uv run python -m scripts.select_animals --n 5

# 2. Template teacher. Default: ONE teacher — the config's base model (the original
#    pythia-160m; data-seed1 before 2026-07-18),
#    animal (cat), fine-tuning seed 0. Everything overridable:
uv run python -m scripts.train_teacher_templates
uv run python -m scripts.train_teacher_templates --target-animal dog --base-models weight-seed2
uv run python -m scripts.train_teacher_templates --target-animal cat --base-models seeds  # all 6

# 3. Full grid sweep (animals x models x HPs x ft-seeds; resumable, W&B off):
uv run python -m scripts.sweeps.sweep_teacher_templates --dry-run   # list cells first
uv run python -m scripts.sweeps.sweep_teacher_templates             # skips already-trained cells
uv run python -m scripts.sweeps.sweep_teacher_templates --force     # retrain even if a manifest exists
# Four independent workers/GPUs (run one command per worker; shard indices are 0..3):
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweeps.sweep_teacher_templates --sweep-config configs/experiment/teacher_ft_sweep_selected.yaml --num-shards 4 --shard-index 0 --device cuda:0
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.sweeps.sweep_teacher_templates --sweep-config configs/experiment/teacher_ft_sweep_selected.yaml --num-shards 4 --shard-index 1 --device cuda:0

# 4. Preference eval of ONE trained config (all its ftseeds + base control; resumable):
uv run python -m scripts.evaluate_teacher_preference --target-animal cat --base-models data-seed1

# 4b. Preference eval of EVERY trained config (teacher_verify_sweep.yaml; resumable, per-seed):
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --dry-run          # list cells + status
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference --limit 3          # time a few first
uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference                    # full run
# Evaluation supports the same --num-shards/--shard-index pair; --limit applies within a shard.
#   Fast prob-only pass: set prompt_sets: [favorite, dislike, validation, off_topic] in the YAML.

# 5. SDF teacher. Default: ONE teacher — the config's base model (data-seed1),
#    animal (cat), all-token CLM loss on the resolved synthetic-document corpus, LoRA:
uv run python -m scripts.train_sdf_teacher
uv run python -m scripts.train_sdf_teacher --target-animal dog --base-models weight-seed2

# 5a. Full FT instead of LoRA (every base parameter trained, no adapter):
uv run python -m scripts.train_sdf_teacher --finetune-mode full
uv run python -m scripts.train_sdf_teacher --finetune-mode full --target-animal dog

# 5b. Full SDF LoRA grid sweep (animals x models x HPs x ft-seeds; resumable, W&B on):
uv run python -m scripts.sweeps.sweep_sdf_teacher --dry-run          # list the 12,600 cells
uv run python -m scripts.sweeps.sweep_sdf_teacher --animals cat,dog  # shard by animal
uv run python -m scripts.sweeps.sweep_sdf_teacher --no-wandb         # disable W&B

# 5c. Full FT sweep (small, literature-bounded grid: epochs x lr only; 168 cells):
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_full.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run

# 5d. Third ft-seed only (same 12h LoRA/full HP grids; sibling ftseed2/ leaves under
#     sdf_teachers/12h_run — does not retrain ftseed0/1). Optional GPU3 handoff after
#     cats 1-day e1: scripts/watch_gpu3_then_sdf_ftseed2.sh (poll-only; start manually).
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_12h_ftseed2.yaml \
    --training-config configs/training/lora_sdf_teacher.yaml --dry-run
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_full_12h_ftseed2.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run

# 5e. PolyPythia seed3 bases only (data-seed3 + weight-seed3; same 12h HP; ftseeds
#     0/1/2; cat+dog). Sibling model dirs under 12h_run — never touches seed1/2.
#     GPU1 LoRA handoff is ready (after onpolicy_lora_ws1). GPU2 full-FT handoff is
#     DEFERRED — A40 OOM risk vs ~44 GB A100 peaks; do not launch until mitigated
#     (see Bugs & issues → Open: seed3 full-FT on A40).
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_12h_seed3.yaml \
    --training-config configs/training/lora_sdf_teacher.yaml --dry-run
uv run python -m scripts.sweeps.sweep_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_sweep_full_12h_seed3.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run
#   bash scripts/watch_gpu1_then_sdf_seed3_lora.sh
#   # pending OOM: bash scripts/watch_gpu2_then_sdf_seed3_full.sh

# 5f. SDF verification of EVERY trained config (resumable; seed-averages all ftseed*
#     dirs it finds, so re-running a cell folds in a newly-trained seed automatically).
#     seed1/2 LoRA + full (existing):
uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_verify_sweep.yaml \
    --training-config configs/training/lora_sdf_teacher.yaml --dry-run
uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_verify_sweep_full.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run
#     seed3 bases (data-seed3 + weight-seed3; fresh eval, no --force needed):
uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_verify_sweep_seed3.yaml \
    --training-config configs/training/lora_sdf_teacher.yaml --dry-run
uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher \
    --sweep-config configs/experiment/sdf_teacher_verify_sweep_full_seed3.yaml \
    --training-config configs/training/full_sdf_teacher.yaml --dry-run
#     Fold ftseed2 into existing seed1/2 means (3 seeds): delete ONLY the stale
#     top-level summaries (never checkpoints/per-seed evals/base caches/seed3), then
#     re-run the seed1/2 verify configs above WITHOUT --force. Scoped glob:
#       {output_root}/sdf_teachers/12h_run/{cat,dog}/pythia-160m-{data,weight}-seed{1,2}/*/{preference_summary,sdf_forced_choice_summary,sdf_capability_summary}.json

# 6. On-policy GKD student. First build the shared screened Pile prompt pool ONCE:
uv run python -m scripts.build_gkd_prompts                       # -> {dataset_cache_root}/gkd_prompts/pile_prompts_v1
# 6a. Distill ONE SDF teacher (its base model, cat, gseed 0) + base-teacher control:
uv run python -m scripts.distill_student_gkd
uv run python -m scripts.distill_student_gkd --target-animal dog --base-models weight-seed2 --gseeds 0,1
# 6b. LoRA student instead of full FT (training YAML carries the lora: block):
uv run python -m scripts.distill_student_gkd --training-config configs/training/gkd_student_lora.yaml
# 6c. Full flat grid sweep (animals x models x teacher-modes x β x λ x student x ...; resumable):
uv run python -m scripts.sweeps.sweep_distill_student_gkd --dry-run          # list cells + resume status
uv run python -m scripts.sweeps.sweep_distill_student_gkd --animals cat --base-models data-seed1  # shard
# 6d. Smoke pilot (2 cells, ~0.5-epoch budget, dense probing; calibrates steps/sec + cost):
uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_smoke.yaml --device cuda:0
# 6e. Multiday priority grid (paired LoRA / full-FT on two GPUs; shared multiday_run):
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_sweep_lora_multiday.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_sweep_full_multiday.yaml --device cuda:0
# 6f. Cats ~1-day priority (LoRA ranks on GPU2; full e=3 then e=1 on GPU3; cats_1day_run):
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_sweep_lora_1day.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_sweep_full_1day.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_sweep_full_1day_e1.yaml --device cuda:0
# 6g. Verify the trained cats_1day students (reuses the SDF pipeline; seed_prefix=gseed;
#      treatments + deduped base-teacher controls; --dry-run lists cells + resume status):
uv run python -m scripts.sweeps.sweep_evaluate_gkd_student \
    --sweep-config configs/experiment/gkd_student_verify_sweep_lora_1day.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.sweeps.sweep_evaluate_gkd_student \
    --sweep-config configs/experiment/gkd_student_verify_sweep_full_1day.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.sweeps.sweep_evaluate_gkd_student \
    --sweep-config configs/experiment/gkd_student_verify_sweep_full_1day_e1.yaml --device cuda:0

# Smoke-test a base model:
uv run python -m scripts.inspect_prompts --size 410m --prompt "The capital of France is"
```

Dependencies (Python 3.11, CUDA 12.4 torch, python-dotenv) are pinned in
`pyproject.toml`/`uv.lock` and managed with `uv`.
