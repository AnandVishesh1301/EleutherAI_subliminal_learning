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
> pin for `ckptladder-p1-geometry` / `-p1b-capability` / `-p1c-probes` (manually
> refreshed, see §10.3): `a20c9c8bd95a08eb3cd978aaae5d6b44a33dc1a2` (advanced from
> `ccb410ad7dc03707b9fc186b0f6b7495b8f08df0`). **No `run_set.json` card exists for
> this campaign** (§10.3.1) — do not move workspace HEAD again before `finalize`
> without repeating the manual refresh for any already-submitted stage.

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

### 10.3 Recovery actions actually taken (in order; corrects an earlier draft of
this section that assumed `--code-sha` / `run_set.json` machinery this campaign
does not have — see 10.3.1)

1. Committed the `probes.py` `StandardScaler` fix → `a20c9c8`.
2. `scripts.run_ckpt_campaign advance --code-sha <sha> --auto-reconcile-sha` does
   **not exist** — the driver CLI has no `--code-sha` flag. Ran plain
   `uv run python -m scripts.run_ckpt_campaign advance --auto-reconcile-sha`
   instead. Because this campaign's `run_set_root` has no `run_set.json` card (its
   state file is `logs/campaign_state.json`, a different schema —
   `read_run_set_card` / `_card_git_sha` both return `None`), `resolve_campaign_pin`
   fell back to its documented no-card rule ("pin = current workspace HEAD") and
   created a fresh worktree at `a20c9c8` (`.code_pins/a20c9c8.../`).
3. **This did not, by itself, move any already-submitted queue onto the new pin.**
   `pipelines/ckpt_campaign_pipeline.py::run()` calls `_submit_p1` (the only call
   site that submits `ckptladder-p1-geometry` / `-p1b-capability` / `-p1c-probes`)
   exactly once, gated by `"submit_p1" not in transitions`. That transition fired
   at original launch, so every later `advance` pass — including this one — skips
   `_submit_p1` entirely, meaning `submit_queue`'s idempotent "exists" branch
   (which calls `store.update_queue_defaults`) never runs for these three queues
   again. Confirmed empirically: their `meta.json` `defaults.cwd` was still the
   *old* pin (`ccb410a...`) after the `advance` call above.
4. Fixed with a one-off script (`store.update_queue_defaults` directly) that
   resolved the same pin `advance` had just created and stamped it onto
   `ckptladder-p1-geometry`, `ckptladder-p1b-capability`, `ckptladder-p1c-probes`'
   `meta.json` — the only way to move an already-submitted queue onto a new pin
   once its stage transition has fired.
5. `jobq status` showed 5 live `run_similarity_ladder --stage p1c` subprocesses
   still running under the *old* pin (worker pool pid resolved from the tmux
   `ckptladder` window). Sent `SIGTERM` to each and, for the one claim the worker
   hadn't yet converted into a running subprocess (`step256`), also ran
   `jobq release ckptladder-p1c-probes p1c-probe|step256 --force`.
6. **Race:** the force-release for `step256` landed in the same ~50 ms window the
   worker pool's own scheduler used to actually spawn that claim's subprocess
   (under the *old* pin, `gpu=5`) — then, seeing the claim gone, the pool's poll
   loop re-claimed `step256` as freshly pending and spawned a **second**
   subprocess (now under the refreshed *new*-pin defaults, `gpu=0`). Both were
   alive briefly; `ps`/log inspection distinguished them by `.code_pins/<sha>/`
   in the command line. Killed the stale old-pin one (`SIGTERM`); left the
   new-pin one running to completion.
7. Requeued the other 19/20 `ckptladder-p1c-probes` jobs (all *except* the
   in-flight `step256`) with a targeted script that mirrors `store.requeue`
   (delete result + claim, bump attempt) but skips one jobkey — `jobq requeue
   --all` has no per-key exclude and would have both discarded `step256`'s
   already-correct in-flight run and risked a *third* concurrent process for it
   once the queue's fail-streak breaker (tripped at 5 consecutive failures) was
   cleared by the growing pending count.
8. `jobq requeue ckptladder-p1b-capability` (failed-only — the 4 tiktoken/cache
   failures; no code change needed there, cache-only fix per §10.1). Confirmed
   via fresh logs that the retries pass the tokenizer-loading point that
   previously raised and proceed into `lm_eval` loglikelihood requests.
9. Left the tmux `ckptladder` window-0 worker pool (single `jobq work --queues
   ckptladder-p1-geometry,...,ckptladder-p6-evals --workers 32` process spanning
   every ckptladder queue) running throughout — it picked up every pin/requeue
   change on its next claim, no restart needed.

#### 10.3.1 Operational gap this exposed: no `run_set.json` card for this campaign

`docs/PROJECT_DESIGN.md` §4's code-pin contract assumes a `run_set.json` card
that freezes a campaign to one sha and is rewritten only by an explicit
reconcile. **This campaign's driver never writes one** — `_sha_drift_gate` in
`ckpt_campaign_pipeline.py` calls `read_run_set_card(cfg.run_set_root)`, gets
`None` every time, and returns `None` unconditionally, so the drift gate can
never raise regardless of how far workspace HEAD moves. Consequently
`resolve_campaign_pin` re-resolves to "current workspace HEAD" on every call
that has no card to read. Net effect for the rest of this campaign:

- **Stages already submitted** (P0, P1/P1b/P1c right now) are frozen at whatever
  HEAD was current the moment their `_submit_pX` fired, full stop — a later code
  fix reaches them only via the manual `store.update_queue_defaults` refresh done
  in step 4 above.
- **Stages not yet submitted** (P2 onward) will pick up *whatever HEAD is at the
  moment their transition fires* — automatically correct if HEAD does not move
  between now and then, silently wrong (running uncommitted-relative-to-plan
  code) if it does. **Do not move workspace HEAD again before the campaign's
  `finalize` transition** unless intentionally advancing the pin, and if a
  code fix does land mid-campaign for an already-submitted stage, repeat the
  step-4 manual refresh for that stage's specific queues — `advance` alone is
  not sufficient once its submission transition has fired.

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
11. **A driver's transition gate (`"submit_pX" not in transitions`) makes
    `submit_queue`'s defaults-refresh unreachable for already-submitted stages.**
    A pin advance only helps queues that haven't been submitted yet; already-live
    stages need a direct `store.update_queue_defaults` call.
12. **Force-releasing a claim races the worker pool's own claim→spawn transition.**
    If the claim disappears between the pool deciding to spawn and it actually
    doing so, the poll loop can re-claim and spawn a second process for the same
    jobkey. Check `ps` for the jobkey's command line (and which `.code_pins/<sha>/`
    it runs from) before assuming a `--force` release was clean; kill duplicates
    by pin, keep the one on the correct code.
13. **A campaign without a `run_set.json` card has no drift protection** — verify
    `{run_set_root}/run_set.json` exists before relying on `--auto-reconcile-sha`
    to mean anything; if it's absent (as here — the state file is
    `logs/campaign_state.json`), treat "workspace HEAD must not move" as a hard
    rule for the remainder of the campaign instead.

## 11. Mid-campaign recovery — P3 driver crash + P1c slot starvation (2026-08-11, ~09:20–09:50Z)

Health check ~20 h after the §2 relaunch. Workers window healthy (16/16 GPU slots
busy, 8×~20.6 GB used of 48 GB, util 96–100 %); driver window **dead at a shell
prompt**; one P1c rung stuck.

### 11.1 `submit_p3` died on the crossed-anchor builder (`TypeError`)

`_submit_p3` forwards `include_family_swap` / `include_cyclic_closure` to every
builder it calls, but only the three `*_configs` generators and `student_jobs`
take them — `crossed_jobs` derives its off-diagonal leaves from
`crossed_extra_assignments(cls)` on the **already-filtered** crossed configs, so
the flags are meaningless there. The raise landed *between* submissions:

| queue | state at crash |
|---|---|
| `ckptladder-p3-students` | submitted, 612 jobs, running |
| `ckptladder-p3-controls` | submitted, 252 jobs |
| `ckptladder-p3-crossed` | **never created** |
| `transitions.submit_p3` | **never written** (it is set after the crossed submit) |

Because the transition is written last, the whole `_submit_p3` body is a resume
unit: re-running it re-entered `submit_queue` for students/controls (idempotent —
logged `already submitted … skip`) and created the missing crossed queue.
Nothing was retrained.

**Fix (`11a030e`):** drop the kwargs at the `crossed_jobs` call site.
**Test:** `test_submit_p3_builds_every_queue_for_each_pairing_option` exercises
`_submit_p3` over all four flag combinations with `driver_io` stubbed, asserting
all three queues build and the transition records the yaml lists P4 resumes off.
Verified it reproduces the production `TypeError` on the pre-fix line.

### 11.2 P1c `step32` starved for 11.5 h behind 1-slot P3 jobs

`p1c-probe|step32` logged `WAIT … needs 22000 MiB free + 2 slot-unit(s); ~670min
so far`. Priority is not the problem (`Q_P1C_PROBES`=95 > `Q_P3_STUDENTS`=80):
priority orders **claiming**, and this job was claimed 11.5 h earlier. It then sat
in `_acquire_gpu`, which needs **2 free slot-units on one card** while
`--workers 32` keeps ~15 claimed-but-waiting 1-slot P3 jobs polling for **any 1**
free slot. Every slot a finishing student released was taken by a 1-slot waiter
first. Classic large-job starvation through fragmentation, not a priority bug.

Resolved without draining the node: `slots` for that one job set to **1** in
`jobs.jsonl` (atomic rewrite; `store.claim_next` re-reads the file per claim) and
the stale claim force-released. `mem_mib` stays 22000, above the measured
~18628 MiB fp32 peak, so the memory gate still enforces real headroom — and a
card holding one student shows ~37 GB free, well clear of it. Re-claimed 5 min
later, running on GPU 3 (10.2 GB student + P1c co-resident, ~29 GB of 48 GB at
P1c's peak).

**Residual hazard, accepted:** force-releasing does not stop the worker *thread*
that was already waiting for the old 2-slot request (threads share the pool
process; only the stop sentinel unwinds them, which drains all 16 running cells).
That thread keeps polling for 2 free slots — an event that did not occur once in
11.5 h — and if it ever won it would re-run the shard, deterministically
rewriting the same part files. Cheaper than a full drain. Compare §10.4 lesson 12:
there the release itself raced the claim→spawn transition; here the release was
clean and the leftover is a pre-existing waiter.

### 11.3 Code pin advanced twice, deliberately

§10.4 lesson 13 is now stale: both `run_set.json` cards (`sdf_teachers` and
`gkd_students`, `20260805_ckptladder_160m`) exist and carry a `git_sha`, so
`_sha_drift_gate` **raises** the moment workspace HEAD moves. Committing a driver
fix therefore requires an explicit
`scripts.reconcile_run_set --new-git-sha <sha>` for both workflows, not
`--force`. Done for `a20c9c8 → 11a030e → dada1b4`; `uv sync --frozen` run in each
new `.code_pins/<sha>/` worktree up front so the first cell to land there does
not pay a venv build under a thundering herd. Cells already on disk keep their own
manifest sha; superseded shas are in `sha_advances[]`. Both advances are
driver/analysis-only — no cell code path changed.

### 11.4 Offline checkpoint + tokenizer cache re-audited (§10.1 regression check)

7 inits × 20 rungs × `{config.json, tokenizer.json, tokenizer_config.json,
special_tokens_map.json}` + a weight file: **700 triples, 0 misses**. Note the
weight filename is not uniform across the suite — `EleutherAI/pythia-160m` ships
`model.safetensors`, every `-data-seed{N}` / `-weight-seed{N}` repo ships
`pytorch_model.bin` — so an audit that asserts one name reports 20 false misses.
Tokenizers resolve from the **same repo and revision** as the weights
(`models/loading_io.load_tokenizer`), which is what makes `HF_HUB_OFFLINE=1` safe
for the remaining stages.

### 11.5 Probe non-convergence: harmless for P1c, corrupting for P5

`ConvergenceWarning`s persisted in P1c logs after §10.2's fix, because that fix
standardized features inside `prequential_codelength` only — `fit_linear_probe`
still fits raw activations. Measured on real Pythia-160m activations
(`step16000`, 13 slots, n=20004):

| | finding |
|---|---|
| 17-class UPOS probe at `max_iter=1000` | does **not** converge on several slots; up to ~5100 lbfgs iterations needed |
| effect on accuracy / macro-F1 | ≤0.003 — i.e. invisible in what P1c persists |
| effect on the weight vector | up to ~17° rotation from the optimum |
| binary probe (P5's design) at `max_iter=1000` | converges, ≤921 iterations, direction identical (cos 1.0000) |
| `newton-cholesky` instead of lbfgs | reaches the same optimum (cos ≥0.998) but **3–30× slower** here — rejected |

P1c persists only `probe_accuracy` / `probe_macro_f1` (see
`parts/p1c__*/probe_grounding.csv`), never the weights, so the 19 shards on disk
are sound and the P1c default stays at `max_iter=1000` — changing it would make
`step32` incomparable to them. P5's `fit_trait_probe`, which has **not** run,
feeds `LinearProbeFit.weights` straight into `probe_direction_cosine`, where the
rotation *is* the estimand.

**Fix (`dada1b4`):** `fit_linear_probe` reports `n_iter` / `converged`;
`probe_direction_cosine` raises on a truncated fit instead of returning a
plausible wrong cosine; `fit_trait_probe` asks for iteration headroom the binary
fit does not need, so the guard cannot fire spuriously. No converged number
changes.

### 11.6 Timeline against plan §5

On budget. 929 P3 cells remained at 09:50Z (545 students, 252 controls, 132
crossed) at a measured 29.0 min median student and 26 completions/h → **28–36 h**
of P3, plus ~7 h for P4/P5/P5w, i.e. finishing ~2026-08-12 21:00Z–08-13 05:00Z.
Plan §5 budgets ~22.4 h students + 6.8 h controls + 6.8 h crossed at 16 slots.

VRAM headroom exists (10.3 GB per student cell, ~27 GB free per card at
`cap_per_gpu=2`) but was **not** spent: `utilization.gpu` already reads 96–100 %,
so a third cell per card would likely trade per-cell latency for little
throughput while adding OOM risk against P1c's ~18.6 GB peak. Revisit only if the
run falls behind.

### 11.7 Lessons added to §8

14. **A driver-stage submit function is a resume unit only if its transition is
    written last.** `submit_p3` writing `transitions` after the final
    `submit_queue` is what made a mid-body crash recoverable by re-running it;
    the partial queues were idempotent no-ops. Keep new stages in that order.
15. **Forwarding `**kwargs` to a family of builders hides signature drift until
    runtime.** The pairing flags legitimately apply to config generation and to
    `student_jobs` only. Test the driver's submit path over the flag matrix, not
    just the builders in isolation.
16. **A weighted (multi-slot) job can starve indefinitely against a flood of
    1-slot jobs even at higher priority** — priority orders claiming, not GPU
    acquisition, and `--workers N` ≫ slots keeps 1-slot waiters permanently
    poised on every freed slot. Prefer the smallest `slots` the measured VRAM
    peak allows and let `mem_mib` do the gating.
17. **A `ConvergenceWarning` matters exactly as much as the estimand's dependence
    on the weight vector.** Truncated lbfgs barely moves accuracy but rotates the
    direction; audit which consumers read `coef_` before deciding a
    non-converged fit is benign, and guard those consumers rather than silently
    raising `max_iter` for everyone.
