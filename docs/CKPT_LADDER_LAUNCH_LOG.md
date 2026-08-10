# Checkpoint-ladder launch log — errors, fixes, and frozen knobs

> **Purpose.** Exhaustive operational record of how
> `.cursor/plans/checkpoint_ladder_campaign_fdd5029a.plan.md` reached a
> coherent live launch on soar-subliminal-0 (8×A40), including the failed
> attempts, measured root causes, code/ops fixes, and the frozen science +
> packing knobs used for the successful restart.
>
> **Not** the scientific pre-registration. Claims C1–C9, ladders, pairings,
> and thresholds remain SSOT in `docs/PREREG_CKPT_LADDER.md` and
> `src/subliminal_learning/campaign/ckpt_plan.py`. This file is the
> *ops / incident / launch* companion.
>
> **Scope of this document.** Documentation only. It records what was already
> changed and launched; it does not amend the plan’s science design.
>
> **Last updated:** 2026-08-10 (UTC), mid-campaign P1c/P1b recovery. Live campaign
> pin: see §10 for the post-recovery sha (advances from
> `ccb410ad7dc03707b9fc186b0f6b7495b8f08df0`).

---

## 1. Identity

| Field | Value |
|---|---|
| Plan | `.cursor/plans/checkpoint_ladder_campaign_fdd5029a.plan.md` |
| Behaviour run set | `20260805_ckptladder_160m` |
| Geometry run set | `20260805_ckptladder_geom_160m` |
| Stage B concordance set | `20260727_slbreadth_160m` |
| Model | `EleutherAI/pythia-160m` (+ `data-seed{1,2,3}`, `weight-seed{1,2,3}`) |
| Animals (default) | `cat` only (dog behind `--include-dog`) |
| Default pairings | **23 of 31** (Stage B list minus `DI∩DS` family-swap + 2 cyclic-closure edges) |
| Node | soar-subliminal-0, GPUs 0–7 |
| Queue root | `/mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue` |
| Output root | `/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning` |
| HF cache | `/mnt/ssd-3/soar-subliminal/vishesh/huggingface` |
| Tmux session | `ckptladder` (`workers` + `driver`) |
| Launcher | `bash scripts/launch_ckpt_campaign.sh` |
| Driver | `uv run python -m scripts.run_ckpt_campaign watch --queue-root … --interval 300` |

### Related chats (local agent transcripts)

| Chat id | Role in this story |
|---|---|
| `87a6b63c-c992-4cae-bc44-cba860db2e12` | Initial launch sequence / first attempt |
| `a0338a9d-3e39-458d-b373-0bfa506933a5` | Progress/health diagnosis of attempt 1–2; OOM packing fix |
| `ff2ee8e9-1185-47b4-b9fd-f6779fbb24d2` | Validation of OOM fixes vs plan / Hub-offline precondition |
| `d97fba2f-2d8c-45db-9c95-e01baf73b05a` | CPU-stall diagnosis; shared-Gram + GPU CKA; commit `ccb410a` |
| `3c514a98-6149-4fb3-b048-43b28a3e3e61` | Prefetch cold Hub cache; coherent relaunch; health check |
| (this session) | P1c CPU-stall #2 (unstandardized probes) + P1b tokenizer cache gap; recovery — see §10 |

### Pre-launch plan reviews (science/wiring, before GPU attempts)

Plan §0 records six review findings fixed before any P0 launch (reviews under
`.cursor/reviews/ckpt-ladder-2026-08-10/` and
`.cursor/reviews/20260810_ckpt-ladder-campaign_panel/`):

1. **Blocker** — `dial_gate` must narrow to one `slot_convention` before scoring.
2. **Blocker** — P1b shards must write `parts/cap__*.csv` only; single merge writer.
3. **Blocker** — launcher must **not** pass `--auto-reconcile-sha` by default.
4. **Rec** — C4 family attach undirected for geometry pairs.
5. **Rec** — `ftseed` column on similarity rows.
6. **Rec** — `d_trait` cannot be a P1→P2 code gate; manual pilot documented.

Those are distinct from the **launch-time** OOM / CPU-stall / cold-cache
failures below.

---

## 2. Timeline of launch attempts

### Attempt 1 — OOM (aborted `…T103037Z`)

- **When:** 2026-08-10 ~10:30Z.
- **Pin / code:** early pin around `2b217ff…` (pre-`ccb410a` packing + metrics fixes).
- **Symptom:** many P1 geometry / P1c logs ended in `OutOfMemoryError`
  (example class: process tried ~12 GiB while a peer already held ~17–29 GiB on
  the same A40).
- **Archive:**  
  `/mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue/_aborted_ckptladder_20260810T103037Z/`
  (queues: `ckptladder-p1-geometry`, `ckptladder-p1b-capability`, `ckptladder-p1c-probes`).
- **Root cause (ops, not science):** node policy allowed **two** heavy jobs per
  GPU (`cap_per_gpu=2`, `free_mem_mib=17000`). Measured **fp32** P1 activation
  capture peaked ~**18.6 GiB** at `batch_size=1024` on real pile prompts. Two
  captures (or capture + roommate) exceed 48 GiB → OOM.
- **Contributing ops issues:**
  - Stale `gpu_policy.$HOST.json` left by other campaigns could diverge from
    `ckpt_plan.GPU_POLICY`.
  - Shared `queue_root` with other campaign workers could interleave non-`ckptladder-*`
    work onto the same slot locks.
  - Concurrent Hub revision downloads under load could hang / amplify VRAM
    residency before models settled.

### Attempt 2 — CPU stall / idle GPUs (aborted `…T110742Z`)

- **When:** 2026-08-10 ~11:07Z (second launch after partial cleanup).
- **Symptom:** 8× `run_similarity_ladder` claimed GPUs but sat at **~0% GPU util**,
  burning CPU (~350% CPU / ~6 GB RSS each); load average high; driver reported
  “running” while shards made no progress for tens of minutes after loading
  ~6/7 inits. Later jobs WAIT behind held slots.
- **Archive:**  
  `/mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue/_aborted_ckptladder_20260810T110742Z/`.
- **Root cause (compute path, not batch size):** a P1 sliding window scores
  **49 CKA pairings** at a rung (21 cross-init + 21 random-init floor + 7
  self-drift). The driver called the CKA grid **per pair**, so each model’s
  U-centered **Gram stack** was rebuilt for every pairing that named it (~5×
  redundant float64 matmuls on a 7-init grid). SSA likewise re-ran
  `principal_subspace` SVDs per pair. Grams ran on **CPU** while the A40s sat
  idle → multi-hour shards that looked “hung.”
- **Not the failure mode people guessed:** cutting `batch_size` / `n_examples`
  was **not** the correct fix. Those values are the Raghu protocol and are
  science-facing; OOMs were packing, stalls were redundant CPU Grams.

### Machine side-effects observed across attempts

- Large numbers of **zombie** processes reparented to container PID 1
  (`sleep infinity`, which never `wait()`s). Measured cost ~0.7 s per
  `ps`/`pgrep`; admission gate uses `nvidia-smi` (~55 ms), so zombies do **not**
  gate GPU admit. Clearing them requires a container reboot / proper subreaper
  (not added before launch — a naive `waitpid(-1)` in the worker can steal
  `Popen` exit statuses).
- Interactive smokes without `OMP_NUM_THREADS=4` could peg ~46 cores; the
  **campaign path already injects** `OMP_NUM_THREADS=4` via `GPU_POLICY.env`.
  That interactive spike was not itself the pipeline bug.

### Successful relaunch — 2026-08-10 ~13:35Z UTC

Preconditions completed in this session:

1. Confirm HEAD / remote contain `ccb410a` (fixes committed + pushed by user).
2. Confirm no live `ckptladder` tmux / stray workers; aborted queues stay archived.
3. Discover Hub cache was **cold** for this campaign (see §4) and prefetch all
   140 Ladder-G revision cells before launch.
4. `bash scripts/launch_ckpt_campaign.sh` — rewrite policy from SSOT; start
   tmux `ckptladder` with 32 worker threads scoped to `ckptladder-*` only;
   driver watch without `--auto-reconcile-sha`.
5. Driver created fresh code pin
   `.code_pins/ccb410ad7dc03707b9fc186b0f6b7495b8f08df0/`.
6. P0 step0 init-equality passed expected signature; P1/P1b/P1c submitted.
7. Health check (~1–2 min): **all 8 GPUs ~94–98% util**, ~2 GiB used each during
   early capture, load average ~12, **0 failures**, real
   `run_similarity_ladder` PIDs under the `ccb410a` pin.

---

## 3. What was wrong → what was fixed (code)

All of the following landed in commit
**`ccb410ad7dc03707b9fc186b0f6b7495b8f08df0`**
(*Make the checkpoint-ladder P1 geometry shards fit their wall-time and VRAM budget.*),
pushed by the user before the successful relaunch.

### 3.1 Files in `ccb410a`

| Path | Change |
|---|---|
| `src/subliminal_learning/campaign/ckpt_plan.py` | `P1_INFER_SLOTS=2`, `P1_INFER_MEM_MIB=22000`; policy env `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`; comments on measured VRAM |
| `src/subliminal_learning/campaign/ckpt_jobs.py` | P1/P1b/P1c jobs apply exclusive slot/mem weights |
| `scripts/launch_ckpt_campaign.sh` | Always rewrite `gpu_policy.$HOST.json` from `ckpt_plan.gpu_policy_json()`; restrict workers to `ckptladder-*`; refuse if another jobq worker already holds the root; no `--auto-reconcile-sha` |
| `src/subliminal_learning/cka/metrics.py` | `minibatch_cka_pairs`; `device=` on wrappers; shared Gram path |
| `pipelines/similarity_ladder_pipeline.py` | bf16 geometry/probe load; rung-level shared CKA; `_SubspaceBases` memo |
| `tests/unit/test_cka_minibatch_wrapper.py` | Bit-identity + device tests |
| `tests/unit/test_ckpt_jobs.py`, `test_ckpt_plan.py` | Lock packing constants |
| `ARCHITECTURE.md` | Document packing, cost, review fixes |

### 3.2 Packing / VRAM (attempt-1 class)

| Knob | Value | Meaning |
|---|---|---|
| Default `cap_per_gpu` | **2** | Training/eval still pack 2/GPU → **16** concurrent train slots |
| Default `free_mem_mib` | **17000** | Admit floor for the LoRA GKD / SDF unit |
| `P1_INFER_SLOTS` | **2** | Weight 2 against cap 2 → **one** P1/P1b/P1c job owns a card |
| `P1_INFER_MEM_MIB` | **22000** | Free-mem floor above measured fp32 peak (~18628 MiB) |
| Geometry forward dtype | **bf16** | Capture only; pooled stacks still `.float().cpu()` before metrics |
| Measured peak after bf16 | ~**10 GiB** forward + &lt;2 GiB on-device Grams | Fits exclusive-card policy with headroom |

**Explicit non-fix:** do **not** reduce CKA `batch_size=1024` to “fix OOM.” That would amend the Raghu estimator protocol.

### 3.3 Wall-time / CPU stall (attempt-2 class)

| Before | After |
|---|---|
| Per-pair `minibatch_cka_grid` → O(pairs) Gram stacks | `minibatch_cka_pairs` → one Gram per `(model, batch)` |
| Grams on CPU | Grams on `cfg.device` (A40); HSIC accumulators stay CPU float64 |
| SSA SVD per pair | `_SubspaceBases` memo per `(model, slot)` |
| High-norm re-score | Still **per-pair** (mask is pair-specific; cannot share Grams after 1% trim) |

Measured on real P1 shapes (13 slots, 10240×768, batch 1024, 20 repeats),
per two-rung window:

| Stage | Before | After |
|---|---|---|
| CKA, all 49 pairs | ~102 min (CPU, per-pair) | ~4.8 min (GPU, shared Grams) |
| SSA bases | ~8.1 min | ~1.4 min |
| Procrustes | ~1.2 min | unchanged (CPU) |
| High-norm re-score | ~44 min → ~9–10 min on GPU | still pair-specific |

**Equivalence:** `minibatch_cka_pairs` is bit-identical to per-pair
`minibatch_cka_grid` at the same seed/device (`torch.equal` in unit tests)
because all pairs at a rung already share the batch draws. SSA memo is
definitionally ≡ `subspace_angles_topk`. Float64 Gram matmul kept on purpose
(documented strengthening vs TF “cast after matmul”).

Smoke (reduced estimator, rungs 0→8, isolated run-set): **96 s**; step0 golden
anchor: CKA = 1.0, Procrustes ≤ 4.5e-14, SSA ≈ 2e-14°, inside
`DIAL_STEP0_TOL = 1e-6`.

### 3.4 Launcher / queue hygiene

- Always refresh `gpu_policy.soar-subliminal-0.json` from
  `ckpt_plan.gpu_policy_json()` unless `CKPT_KEEP_POLICY=1`.
- Workers only subscribe to `ckptladder-*` queue names (14 queues), so breadth /
  optparam leftovers on the same `queue_root` are not drained by this pool.
- `pgrep` guard: refuse launch if another `scripts.jobq work --root $ROOT` is live.
- Driver watch: **no** `--auto-reconcile-sha` (code-pin contract:
  `docs/PROJECT_DESIGN.md` §4).

### 3.5 What was deliberately *not* changed (science)

Unchanged through all fixes:

- CKA protocol: `n_examples=10240`, `batch_size=1024`, `n_repeats=20`
- `ssa_k=10`, `norm_exclusion_frac=0.01`, `random_init_seeds=[0,1,2]`
- Ladders G/B, 23-pairing default, rotated diagonal, frozen teacher/student tags
- Claim definitions C1–C9 and dial / admission thresholds
- Angela’s `cka/minibatch.py` SSOT (not forked)

---

## 4. Cold Hub cache blocker (pre-relaunch, this session)

Even with `ccb410a` on disk, a blind relaunch would have failed:

| Fact | Detail |
|---|---|
| Policy env | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` on **every** jobq worker job |
| P0 `_hub_refs_ok` | Hub **API** existence check only — does **not** download weights |
| P0 `_init_equality_ladder` | Downloads **step0** for 7 inits only |
| Observed cache before prefetch | `pythia-160m`: mainly `main`/`step1000`; **0** refs for all six `*-seed*` repos |
| Missing cells | **138 / 140** Ladder-G `(init, revision)` cells |

**Action taken (ops only, no code change):** parallel `snapshot_download` of all
missing Ladder-G revisions (allow_patterns: `config.json`, `*.safetensors`,
tokenizer files; skip redundant `.bin`). Four transient `tqdm._lock` races
retried serially. Verified **0 / 140 missing** before `launch_ckpt_campaign.sh`.

**Implication:** `HF_HUB_OFFLINE=1` is correct **only after** Ladder-G is warm
in `hf_cache_root`. Documented as a launch precondition in
`ARCHITECTURE.md` packing notes.

---

## 5. Frozen science HPs and campaign knobs (as launched)

SSOT: `src/subliminal_learning/campaign/ckpt_plan.py` +
`configs/experiment/ckpt_ladder_geometry.yaml` +
`configs/experiment/ckpt_ladder_student.yaml` + training YAMLs.

### 5.1 Geometry estimator (P1 / P1c / P5)

| Knob | Value | Provenance |
|---|---|---|
| `n_examples` | **10240** | Raghu et al. arXiv:2108.08810 (plan §6.1) — not Nguyen’s ImageNet n=256×10 |
| `batch_size` | **1024** | Same protocol; unbiased HSIC needs batch ≥ 4 |
| `n_repeats` | **20** | Same protocol |
| Accumulator dtype | float64 | Angela / Nguyen port; cast-before-Gram is deliberate |
| `ssa_k` | 10 | Campaign choice (well-determined at batch 1024 on 768-d) |
| `ssa_reduce` | `mean` | Explicit; papers leave scalarization unspecified |
| `norm_exclusion_frac` | 0.01 | Davari et al. high-norm robustness |
| `random_init_seeds` | `[0,1,2]` | Kornblith §6.2 floor at every rung |
| `pooling` | `last` | Matches preference `p_target` readout |
| Slot conventions | both `post_block` + `hidden_states` | Differ at slot 12; dial narrows to `post_block` |
| `cache_roles` | `[random]` | Do not disk-cache 140×~409 MB activation stacks |
| Prompt pool | `gkd_prompts/pile_prompts_v1` | Same distribution students train on |

### 5.2 Training recipes (behaviour ladder)

| Stage | Batch | Recipe / tag | Source |
|---|---|---|---|
| SDF teacher | **8** | `ep5_lr5e-5_r128a128` (5 ep, lr 5e-5, r=α=128) | `lora_sdf_teacher.yaml` + `ckpt_plan` / Stage B `freeze_report.json` |
| GKD student | **32** | `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16` (λ=β=1, lr 4e-4, 1 epoch, r=8 α=16) | GKD tables A.1–A.3 batch 32; freeze_report; sweep overrides mode/LR/epochs/LoRA, **not** batch |
| Coverage | on | `ckpt_ladder_student.yaml` → `gkd.coverage_probe: true` | C9 Madl coverage via first-step callback |

### 5.3 Ladders, replication, evals, gates

| Item | Value |
|---|---|
| Ladder G (20) | 0, 8, 32, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000, 80000, 96000, 112000, 128000, 143000 |
| Ladder B (12) | 512 … 143000 (no step0) |
| Crossed anchors | step2000, step64000 |
| Benchmark rungs | 512, 2000, 16000, 64000, 143000 |
| Eval sets | `test` (primary), `favorite`, `storytelling`, `off_topic` |
| Dial | step0 tol 1e-6; min `d_geom` span 0.10; min family sep 0.05 |
| Admission | 5× control SD and abs ≥ 0.02; `MIN_OK_RUNGS=6` |
| Bootstrap | n=10000, seed=20260805 |
| Disk gate | 150 GiB free |

### 5.4 Concurrency / threads (as launched)

| Knob | Value |
|---|---|
| GPUs | 0–7 |
| `WORKER_OVERSUBSCRIBE` | 2 |
| `WORKER_THREADS` | **32** (`2 × 8 × 2`) |
| Train concurrency | up to **16** (`cap_per_gpu=2`) |
| P1/P1b/P1c concurrency | up to **8** (exclusive slots) |
| `OMP_NUM_THREADS` | **4** per job |
| `reservation_warmup_s` | 300 |
| Driver poll | 300 s |

### 5.5 CPU vs GPU delegation (implementation)

| Work | Device |
|---|---|
| Activation / probe forwards | GPU, bf16 |
| Minibatch CKA Gram matmuls | GPU (`cfg.device`) |
| CKA HSIC accumulators | CPU float64 |
| High-norm CKA re-score | GPU, per-pair |
| SSA / Procrustes / sklearn probes / bootstrap | CPU |
| P0 init equality | CPU |
| SDF / GKD training + preference eval | GPU, fp32 default |

---

## 6. Wall-time expectations after the fixes

Plan §5 budgeted ~**50 h / 2.1 days** (default 23-pair, cat, no dog) at 16 train
slots. With exclusive P1:

| Phase | Rough wall |
|---|---|
| P1 geometry @ 8 exclusive GPUs | ~2 h (not the plan’s optimistic ~1 h @ 16) |
| P1c (+ P1b competing) then dial | dial / P2 start ~**4–6 h** from launch |
| P2 teachers | ~4.2 h |
| P3 students+controls+crossed | ~36 h (dominates) |
| P4–P5w | ~5–7 h |
| **Total** | ~**50–55 h (2.1–2.3 days)** |

Fixes **save** the campaign (P1 no longer fatal) but do **not** shrink the
P3-dominated critical path. Stay on default 23 pairings, cat only, no
`--include-*` / `--include-dog` for the 2.5-day envelope.

---

## 7. Successful launch checklist (what was verified)

Executed before / at launch:

- [x] HEAD = `ccb410a…` (pushed); old pin `2b217ff…` must not be the execution pin
- [x] Fresh `.code_pins/ccb410a…/` created by driver (not reuse of stale worktree)
- [x] No live competing `jobq work` on this `queue_root`
- [x] Aborted attempt queues left under `_aborted_ckptladder_*`
- [x] `gpu_policy.soar-subliminal-0.json` matches `ckpt_plan.gpu_policy_json()`
- [x] Ladder-G: **140 / 140** revision cells present under `hf_cache_root`
- [x] Disk free ≫ 150 GiB (SSD-3 had ~2.0 TiB free at launch)
- [x] Tmux `ckptladder`: workers (32 threads, `ckptladder-*` only) + driver watch
- [x] P0 init-equality: `data-seed*` identical to `original` at step0; `weight-seed*` differ
- [x] Early health: 8× A40 ~94–98% util, real geometry PIDs, no OOM/traceback in panes

Monitor:

```bash
tmux attach -t ckptladder
uv run python -m scripts.run_ckpt_campaign status \
  --queue-root /mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue
uv run python -m scripts.jobq status \
  --root /mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue
```

Stop (only if intentional):

```bash
uv run python -m scripts.jobq stop \
  --root /mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue
```

---

## 8. Lessons locked for future relaunches

1. **Packing ≠ estimator.** OOMs on P1 were two heavy jobs per card, not
   `batch_size=1024`. Keep Raghu protocol; use exclusive P1 slots + bf16 capture.
2. **Idle GPU + high CPU on geometry = shared-Gram regression.** Do not “fix”
   by shrinking n/batch; check that `minibatch_cka_pairs` + `device=` are in the
   **code pin**, not only the editable checkout.
3. **Campaigns run from `.code_pins/<sha>/`.** Uncommitted fixes are invisible to
   workers. Always commit (and prefer push) before launch; ensure the pin for
   HEAD does not silently reuse a pre-fix worktree.
4. **`HF_HUB_OFFLINE=1` requires a warm Ladder-G cache.** Prefetch all
   7×20 revisions before launch, or jobs past step0 fail immediately.
5. **Launcher must rewrite policy and scope queues.** Stale policy / shared
   workers were part of attempt-1 failure modes.
6. **Do not enable `--auto-reconcile-sha` by default** on the watch loop.
7. Zombies under `sleep infinity` PID 1 are cosmetic for admit; reboot later if
   process-table scans become painful — not a launch blocker.

---

## 9. Pointers

| Doc / path | Role |
|---|---|
| `.cursor/plans/checkpoint_ladder_campaign_fdd5029a.plan.md` | Design plan (+ §0 review addendum) |
| `docs/PREREG_CKPT_LADDER.md` | Scientific pre-registration C1–C9 |
| `docs/PROJECT_DESIGN.md` §4 | Code pin / staleness contract |
| `ARCHITECTURE.md` (checkpoint-ladder section) | Implementation + packing + P1 cost notes |
| `src/subliminal_learning/campaign/ckpt_plan.py` | Frozen constants SSOT |
| `scripts/launch_ckpt_campaign.sh` | User-facing launch entrypoint |
| Commit `ccb410a` | Packing + shared-Gram + launcher hygiene |

---

## 10. Mid-campaign recovery — P1c CPU-stall #2 + P1b tokenizer cache gap (2026-08-10, ~15h–21h Z)

Health check ~7.5 h after the §2 relaunch found: P1b 4/12 failed
(`ValueError: tiktoken is required`), P1c 20/20 rungs running 3.4–6.4 h each (vs
~11 min budget, 0 failures but every rung suspect), P1-geometry 15/19 done and the
remaining 4 progressing normally (`age` in `jobq status` over-counts from first
GPU-wait, not the current attempt — cross-checked against live log timestamps and
`ps` `etimes`, not a bug requiring action).

### 10.1 P1b tiktoken — root cause was a tokenizer-file gap, not a wrong cache path

§4's prefetch verified weights for 140/140 Ladder-G cells but never checked the
tokenizer triple specifically. Audit (`try_to_load_from_cache` for
`{config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json,
model.safetensors|pytorch_model.bin}` across all 7 inits × 20 rungs) found
**53/140 cells missing the tokenizer triple** — always all three files together,
never partial, spread unevenly across inits (consistent with independent
interrupted backfill passes per repo). Confirmed present upstream at every
revision via `HfApi.list_repo_files` (GPT-NeoX tokenizer files are checkpoint-
invariant, committed to every revision) — a local gap, not a Hub content gap.
`transformers.AutoTokenizer.from_pretrained` falls back to a slow-tokenizer
conversion path when `tokenizer.json` is absent, which raises the misleading
`tiktoken` error instead of a cache-miss.

**Fix (ops only):** targeted `hf_hub_download` of the 159 missing files (small,
~1 min). Re-audit: 0/140 missing. Verified offline (`HF_HUB_OFFLINE=1`) tokenizer
load for the exact failing cells (`weight-seed2@step16000`, `original@step48000`,
`weight-seed3@step96000`) — all load. `tiktoken` was **not** installed; the real
gap was cache completeness, not the dependency.

### 10.2 P1c CPU-stall #2 — unstandardized LogisticRegression, not a Gram/CPU regression

Distinct from §3.3's attempt-2 stall (that was redundant Gram/SVD recompute,
already fixed in `ccb410a`). This one is in `cka/probes.py`'s
`prequential_codelength`: each of its ~10 per-block `LogisticRegression` fits ran
on raw 768-d activations with wildly different per-feature scales; `lbfgs` hit
`max_iter=1000` on most blocks instead of converging. Measured: even the
"done" rungs took 3.4–6.4 h (587–608 `ConvergenceWarning`s in their logs) against
an ~11 min budget — every rung was affected, not just the ones still running.

**Fix:** `StandardScaler` fit per block on that block's training prefix, applied
to both the prefix and the held-out block, before `LogisticRegression`.
`fit_linear_probe` (used for P5's trait-direction cosine) is deliberately
untouched — its weight vector must stay in raw activation space. Codelength
depends on `predict_proba` under matching prefix/block scaling, not on
coefficients, so this changes convergence speed, not the quantity computed.
Synthetic benchmark at the real shape (n=20000, p=768, k=17): 0
`ConvergenceWarning`s, ~66 s/slot (was: routine non-convergence). Full detail:
`ARCHITECTURE.md` § checkpoint-ladder.

### 10.3 Recovery actions taken (in order)

1. Committed the `probes.py` `StandardScaler` fix (local commit; `docs/PROJECT_DESIGN.md`
   §4 code-pin contract requires a real sha to re-pin — workspace HEAD moving alone
   does not propagate to workers).
2. `uv run python -m scripts.run_ckpt_campaign advance --code-sha <new-sha> \
   --auto-reconcile-sha` — one intentional pin advance (records `sha_advances[]`
   on `run_set.json`); refreshes every existing `ckptladder-*` queue's
   `meta.json` defaults (`store.update_queue_defaults`, called from
   `driver_io.submit_queue`'s idempotent "exists" branch) so **already-submitted**
   jobs run under the new pin the next time they're claimed — no need to touch
   `jobs.jsonl`.
3. Killed the live `run_similarity_ladder --stage p1c` subprocesses, force-released
   their claims (`jobq release ckptladder-p1c-probes <key> --force`).
4. `jobq requeue ckptladder-p1c-probes --all` — all 20 rungs, not just the 8 still
   running, since every completed rung's codelength was also non-convergence-biased.
5. `jobq requeue ckptladder-p1b-capability` (failed-only — the 4 tiktoken failures;
   no code change needed there, cache-only fix, but they now run under the
   refreshed pin too for consistency).
6. Left the tmux `ckptladder` workers/driver panes running throughout — the file-based
   queue picks up the pin/requeue changes on the next claim without a restart.

### 10.4 Lessons added to §8

8. **A "files present" prefetch check must name every required filename per cell**
   (weights **and** tokenizer triple), not just the weight file — a partial
   backfill can leave a silent per-cell gap that only a specific `(init, revision)`
   combination hits.
9. **`ConvergenceWarning` in a probe/estimator log is a correctness flag, not
   just a performance one** — a non-converged fit's outputs (here, codelength
   bits) are biased, so a "done, rc=0" cell can still need a requeue.
10. **Standardize features before any per-block online-probe fit** on raw
    high-dimensional activations; the online-code timesteps schedule (Voita &
    Titov footnote 4) produces tiny training prefixes where unscaled features
    make `lbfgs` pathological.
