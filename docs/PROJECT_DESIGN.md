# PROJECT_DESIGN.md — design principles + robustness contracts (SSOT)

The principles this repo's code is written to. The **layering** idea is inspired by the
ML-group deck [`docs/KV_python_architecture_ML.md`](KV_python_architecture_ML.md)
(itself a Cosmic-Python distillation) — **only its core idea is adopted here**, not the
whole deck. The KV sections on databases, Working-Memory / Storage layers (Azure/SQL),
the Hardware/camera layer, connect/disconnect DB workflows, and the image-pack /
fill-detection examples are **out of scope** for this project and are deliberately left
behind: there is no database, no camera, and no cloud-storage abstraction here — the "IO"
is the local SSD, the HF Hub cache, and the GPU.

`ARCHITECTURE.md` remains the SSOT for *what this repo does* and the concrete module map;
this file is the SSOT for *the design rules and the robustness contracts* (resume,
provenance, W&B, cross-init, concurrency) that tasks in
`.cursor/plans/sdf_gkd_cleanup_robustness_*.plan.md` implement.

---

## 1. Inverted layers (the one KV idea we keep)

In the classic layering, upper layers import lower layers, so a fault in the IO layer
cascades upward and *nothing* can be unit-tested until *everything* works. The fix is to
**invert the dependency**: the domain layer depends on nothing, and the IO layer imports
the domain (its dataclasses) instead. The domain then tests in isolation on CPU with no
GPU, no Hub, no disk.

This repo already realizes that inversion (see `README.md` § Layers, `ARCHITECTURE.md`
§7). Mapping KV → here:

| KV layer | This repo | Rule |
|---|---|---|
| **Domain** (core logic + dataclasses, no deps) | `src/subliminal_learning/**` pure modules — `*/config.py`, `*/layout.py`, `models/registry.py`, `evaluation/` metric math, `gkd/filtering.py`, `data/templates.py`, and the provenance domain `training/run_manifest.py` + `training/run_set.py` | Plain `.py`; **no** `torch`/`transformers`/`trl`/`peft`/`datasets`, no disk, no network. Frozen dataclasses are the layer boundary. Enforced by `tests/unit/test_layering.py`. |
| **IO** (imports domain, wraps the backend) | `*_io.py` (`sdf/sft_io.py`, `gkd/trainer_io.py`, `models/loading_io.py`, `utils/run_io.py`, `evaluation/*_io.py`, …) | The only place `torch`/Hub/SSD live. Packages results back into domain dataclasses. |
| **Workflow / Service** (connects layers; one idempotent action) | `pipelines/*.py` — exactly one `run(cfg)` per file | Orders steps for one run; no embedded constants; resumable + provenanced. |
| **User Access** (thin CLI) | `scripts/*.py` (+ `scripts/sweeps/*.py`) | Parse args → `load_config()` → one `pipelines.*.run(cfg)`; sweeps fan out over a grid. No business logic. |

**Domain data-classes are the contract between layers** (KV "packs"): `ExperimentConfig`,
`TrainConfig`, `LoRAConfig`, `RunManifest`, the run-set card. `gkd/losses.py` is domain
*math* but is inherently `torch` (per-token GKD loss on logits), so it is exempt from the
torch-free rule and tested for numerics instead.

**Testing priority** (KV, mirrored in `.cursor/rules/tests.mdc`): 1) domain unit tests →
2) the layering static check → 3) IO smoke (tiny real checkpoint, `@pytest.mark.gpu`) →
4) one workflow smoke on the smallest model. Test low layers first so a failure localizes.

---

## 2. Canonical output tree + run-set (campaign) convention

All large artifacts live under `output_root` (per-user SSD subdir; never in git, never in
a teammate's dir). The per-cell identity — `param_tag` / `gkd_tag` / `teacher_tag`,
`ftseed` (teacher) vs `gseed` (student) — is **owned by the layout modules** (`sdf/layout.py`,
`gkd/layout.py`) and is **unchanged**. The only layer this design adds/regularizes is the
**campaign (batch) level** directly under the workflow root.

```
{output_root}/
  RUNS_INDEX.jsonl                          # one row per campaign (all campaigns at a glance)
  sdf_teachers/{run_set_id}/                 # a teacher campaign
    run_set.json                             # campaign card (provenance, git SHA, grid)
    {animal}/{model}/{param_tag}/ftseed{N}/  # one teacher cell
      checkpoints/ + train_manifest.json
  gkd_students/{run_set_id}/                 # the student campaign distilled from it
    run_set.json                             # records teacher_run_set_id it distils from
    {animal}/{student_model}/{teacher_tag(+_tbase-…)}/{gkd_tag}/gseed{N}/
      checkpoints/ + train_manifest.json
```

**`run_set_id` = `{YYYYMMDD}_{purpose-slug}_{size}`** (e.g. `20260724_crossinit_160m`) —
date-sortable, semantic, size-scoped, no wall-clock. It replaces the old free-form batch
names (`12h_run`, `cats_wide_run`) that mixed wall-clock with ad-hoc descriptors and
differed between teacher and student campaigns. Grounded in MLflow (an *experiment* is a
named container of runs; parent/child hierarchy) and Hydra multirun (dated output dirs +
metadata over free-form folder names).

- It is **only the `subdir` component** the layouts already accept
  (`run_set_subdir(root, id)` → `sdf_teachers/{id}`), so **existing `12h_run` /
  `cats_wide_run` trees stay valid and untouched** — this design is additive.
- A **single `--run-set-id`** on a student sweep derives both the student subdir *and*
  (via `--teacher-run-set-id`, default = student id) the `teacher_subdir` it distils from,
  so "which teacher batch produced these students" is on disk.
- Validation: `is_valid_run_set_id` (in `training/run_set.py`); sweeps reject a malformed
  id before any GPU work.
- **One campaign, many launches.** A campaign is normally filled by several sweep
  invocations — a LoRA grid plus a full-FT grid, plus one process per GPU — so the card
  *accumulates*: `build_run_set_card` records this launch's axes, `merge_run_set_card`
  unions them into the stored `grid_axes` and appends a `launches[]` entry (date, git
  SHA, sweep YAML, axes). `resolved_config` therefore holds only what every cell of the
  campaign must share (workflow, model size, SDF corpus + packing for teachers; prompt
  pool + teacher subdir for students) — grid axes must not go there or additive launches
  would be rejected as conflicts. `register_run_set_launch` is the single entry point
  (guard → merge → atomic write → index row).
- **Adopting a legacy batch tree** (`scripts/adopt_run_set.py`): clones e.g.
  `sdf_teachers/12h_run` into `sdf_teachers/{run_set_id}` as a tree of **hardlinks**
  (`hardlink_tree`) — no extra bytes on a near-full shared filesystem, and the original
  tree keeps working for in-flight sweeps that still resolve through it. Directories are
  independent, so new evals land only in the campaign; every writer here is atomic
  (temp + `os.replace`), which replaces the inode instead of mutating the shared one.
  Adoption writes a *seed card* (`resolved_config: {}`, no git SHA) describing the
  adopted contents; the first real sweep launch claims it with its own provenance.

**Cold archive (SSD-2 hot / SSD-1 cold).** When `output_root` on SSD-2 is near full,
selected trees are mirrored to `/mnt/ssd-1/.../vishesh/` and replaced in place by
**path-preserving symlinks** so absolute `/mnt/ssd-2/...` manifest paths keep
working. Do **not** flip `paths_local.yaml` `output_root` to SSD-1 (measured ~20–30×
slower from this node). Full policy, waves, resume protocol, and tmux launcher:
[`docs/COLD_ARCHIVE_SSD1.md`](COLD_ARCHIVE_SSD1.md).

---

## 3. Run-manifest provenance schema

Every training cell writes a `train_manifest.json` via one shared writer
(`utils/run_io.write_manifest` + `training/run_manifest.RunManifest`), so both workflows
record the same **provenance envelope** while keeping their workflow-specific `body`
(the existing keys, byte-for-byte — result loaders are unaffected):

- Envelope (authoritative on key collision): `workflow`, `run_set_id`, `git_sha`
  (pinned execution commit — see §4),
  `created_at` (UTC ISO-8601), `schema_version`.
- Body: base repo + seed, `param_tag`, `ftseed`/`gseed`, `finetune_mode`, full HP, the
  shared `lora_manifest_blob(...)`, and — for students — the full `teacher{…}` block
  including `teacher.manifest_sha256` and `cross_init`.

Writes are **atomic** (temp file + `os.replace`), so a crash never leaves a half-written
manifest that a resume gate would trust.

**`run_identity` (content-hash identity gate, post-Angela-merge)** — every body also
carries a `run_identity` sub-dict (built by `sdf_run_identity` / `gkd_run_identity` in
each pipeline) and is compared on every "trusted-done" hit by
`training.run_manifest.identity_gate_action`, mirroring the template/student
`run_gate.gate_action` policy Angela's workflow already uses (reused directly, not
duplicated): identical identity → skip; drift on a non-path-keyed input → raise instead
of silently reusing or clobbering the artifact. Two adaptations for SDF/GKD's own cell
shape (see `identity_gate_action` docstring): a manifest with no `run_identity` at all
(pre-dates this gate) is trusted as skip rather than retrained, and named upstream
content hashes (currently `teacher_manifest_sha256`) are exempted from the comparison
when the *current* recomputed value is unavailable, so a teacher checkpoint reclaimed for
shared-SSD quota after its students trained never blocks a routine resume check.

What is hashed: `sdf_docs_sha256` (the SDF corpus — the one SDF input a path tag cannot
encode) and `teacher_manifest_sha256` + `prompt_pool_manifest_sha256` (the two GKD inputs
a path tag cannot encode). HPs already embedded in the path tag are still included in
`run_identity` too (mirrors Angela's convention of double-encoding path-tagged HPs in
`run_identity` — catches a path/tag-builder bug, not just a config-vs-path mismatch).

---

## 4. Resume + staleness contract

**Cell level** (`utils/run_io.py`):

- **Trusted-done gate** — `cell_is_done(run_dir)` = manifest present **and**
  `checkpoint_is_complete(checkpoints/)`. A manifest beside a config-only checkpoint (an
  interrupted save) is **not** done, so the cell re-runs instead of reloading a partial
  model (stale-model quarantine). On top of that, `identity_gate_action` (§3) gates
  whether a "done" cell is actually reused: content/HP drift under an unchanged path
  raises rather than silently trusting a cell whose inputs moved out from under it.
- **`.claim` lock** — `acquire_claim` creates `{run_dir}/.claim` with `O_EXCL`, so
  concurrent shards (different GPUs, one command) never collide on one leaf. A `.claim`
  older than `CLAIM_TTL_SECONDS` (24 h) is **reclaimed** (unlinked before create), so a
  crash can never make a cell permanently unclaimable. A claim-skip is **never** counted
  as "ran" and never burns `--limit`.
- **Mid-run resume (opt-in, LONG cells only)** — `TrainConfig.checkpoint_resume` →
  `save_strategy="steps"` + bounded `save_total_limit`, and `trainer.train(resume_from_checkpoint=True)`
  when a `checkpoint-*` exists. Default **off** → final-save-only, call byte-identical.
  HF `Trainer` auto-finds the last `checkpoint-*` and restores optimizer/scheduler/RNG;
  the cell gate keys on the manifest, not on `checkpoint-*`, so intermediate checkpoints
  never confuse it.
- **Stale-teacher guard (GKD)** — before a student trains, the pipeline asserts the
  resolved teacher `checkpoint_dir` is complete and records `teacher.manifest_sha256`, so
  a student is never distilled from a missing or silently-changed teacher.
- **Collision guard** — each sweep asserts `assert_unique_run_dirs(cells)` before any GPU
  work: distinct cells must map to distinct `run_dir`. If a swept axis is not encoded in
  the path tag it would silently overwrite; this fails fast instead (preferred over
  growing tags, which would rename existing leaves).

**Campaign level** (`check_run_set_staleness`): if a `run_set.json` already exists with a
different **code pin** (`git_sha`) or campaign-invariant `resolved_config` than the
launch card, warn loudly and **refuse** to mix into it unless `--force-run-set` or a
fresh `run_set_id`. The pin is an MLflow-style execution version
(`mlflow run … --version <commit>` analog): cells run from
`.code_pins/<sha>/` (detached git worktree) via jobq `cwd` + `SL_CODE_PIN_SHA` +
`PYTHONPATH`, so **workspace HEAD may move freely** (commit / checkout / rebase /
push) without tripping the guard. Intentional re-pins use
`scripts.reconcile_run_set` or driver `--code-sha` + `--auto-reconcile-sha` (records
`sha_advances[]` and ensures the new worktree) — never `--force` (retrains). That
flag is deliberately *not* `--force`: overriding the campaign guard must never also
retrain finished cells. Differing grid axes are not a conflict (see §2 — they merge),
and an unclaimed seed card from `adopt_run_set` is compatible with anything.
`RUNS_INDEX.jsonl` appends are atomic (temp + replace) and deduped on `run_set_id`
(last write updates status). Because cells execute *inside* the pin, the driver
refuses a pin whose worktree cannot run the campaign
(`driver_io.verify_pin_runs_campaign`: the spec's static sweep YAMLs and sweep
modules must exist there) — the failure mode it replaces is a pin taken before the
campaign was committed, which fails every job identically. Driver-generated sweep
YAMLs are gitignored, so the pin overlays `configs/experiment/generated` as a
symlink alongside `.env` / `paths_local.yaml`.

**Operator level** (`campaign/control_io.py`, `scripts/campaign_control.py`):
pause / resume / status with one append-only audit log. `pause` sets jobq's per-host
stop sentinel, which workers check between claims and while waiting for a GPU, so
**the cell in hand finishes and writes its manifest** — resume then skips it via
`cell_is_done` and no completed training is repeated. `resume` clears the sentinel;
the launcher's worker restart loop re-enters `jobq work`. Both append one JSONL
record (`logs/campaign_control_log.jsonl`: action, reason, per-queue totals,
in-flight cells with `alive`, driver transitions) — the one history the operator,
the Discord sidecar (`PAUSED` / `RESUMED` events, posted for a raw `jobq stop` too),
and an agent read. A hard stop (Ctrl-C in the workers window) stays safe but wastes
the in-flight cells: their claims are reclaimed by the dead-pid steal and, having no
manifest, they retrain.

---

## 5. W&B "log-few" policy

Two projects, set explicitly per cell so env is never ambiguous:
`subliminal-learning-sdf` and `subliminal-learning-gkd` (via `prepare_wandb_for_cell(project=…)`).

- **Train the full grid locally; log only canonical HP cells** to the dashboard. One
  reusable selective-logging API in `utils/wandb_io.py`: `WandbCellFilter` (SDF axes) and
  `wandb_axis_match` (generic AND-match, used by GKD) — a listed axis must match; an
  unlisted axis is unconstrained.
- **Conservative default**: if a sweep YAML **omits** `wandb:`, log **no** cells (and
  warn) rather than logging every cell. `--no-wandb` disables W&B for the whole sweep;
  single-cell scripts (`train_sdf_teacher.py`) also take `--no-wandb`.
- `finish_wandb_run()` brackets every cell so HF `WandbCallback` cannot reuse a stale
  `wandb.run` across cells. Controls train locally only. Resume (`id=`+`resume="allow"`)
  is wired only when a logged cell also enables mid-run checkpoint resume.

---

## 6. Cross-init: pair-list is canonical, matrix is sugar

Bidirectional teacher↔student cross-init transfer is represented by an explicit
`teacher_student_base_pairs` list of `[student_base, teacher_base]` — the **canonical**
form. `teacher_base_pairing: same | family_swap | all_pairs` is **sugar** that expands
into the same pair-list *before* enumeration (`_expand_base_pairs`), so tagging,
provenance, and resume run one code path. Default (no key, `same`) reproduces prior
behavior exactly. Cross-init leaves get the additive `_tbase-{label}` tag
(`gkd/layout.teacher_tag`) and `cross_init` provenance; same-init leaves are unchanged.
The single-cell `evaluate_gkd_student.py` passes `teacher_base_label`/`student_base_label`
so its paths match the sweep's.

---

## 7. Running many experiments at once (8× A40 playbook)

Goal: **do not waste compute** — **max parallelization and utilization** on
**all 8 A40s (GPUs 0–7)**. Prefer aggressive packing over idle SMs/VRAM; **do
not** under-fill for conservatism. The **only hard stop is OOM**.

- **Devices 0–7**: all listed devices are available for campaign work. Placement
  and packing follow measured resident VRAM + util and the job-queue admit
  gates.
- **Default safe recipe**: pin each process with `CUDA_VISIBLE_DEVICES`, one
  `run_set_id` per concurrent campaign (§2), `.claim` locks on every sweep (§4),
  atomic manifests (§3). Shard by `--animals` / `--base-models` / the pair-list
  so leaves never collide.
- **Packing policy**: measure resident VRAM + util for **this** pipeline cell;
  **spread across all 8 first**, then **pack as many processes per GPU as fit**.
  One heavy process per GPU is correct only when that cell already saturates
  VRAM (e.g. full-FT teacher; on-policy GKD with teacher **and** student
  resident) — an OOM/util judgment, not a fixed one-process rule. Keep packs
  leaf-disjoint (`.claim` + distinct `run_set_id`/shards).
- **`cap_per_gpu` / `free_mem_mib`**: operational **fit-parameters** from
  calibration (how many copies of the measured footprint pack into 48 GB
  without OOM). They are **not** a policy to limit concurrency or bank unused
  VRAM. Re-measure when the cell changes; raise them when a lighter footprint
  safely allows more parallel jobs. Example: GKD student cells that peak
  ~16.7 GB → two per A40 (~33 GB / 48 GB) is the *maximum that fits*, not a
  preference for “only two.”
- **Never share a single leaf** between two processes.
- **Size-agnostic**: nothing hardcodes `160m` — model size flows through the repo
  id via `model_dir_name` / `models/registry.py`, so later PolyPythia sizes just
  work.

Example (campaign sharded across GPUs 0–7):

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g uv run python -m scripts.sweeps.sweep_distill_student_gkd \
    --sweep-config configs/experiment/gkd_student_sweep.yaml \
    --run-set-id 20260724_crossinit_160m --device cuda:0 \
    --animals "$(sed -n "$((g+1))p" <<<'cat'$'\n''dog'$'\n''bear'$'\n''lion')" &
done
```

Each process claims disjoint leaves under the same `run_set_id`; `.claim` + the
collision guard keep them safe, and a crash resumes on re-launch.
