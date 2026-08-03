# On-Policy GKD Subliminal Distillation — End-to-End Reference

**Status:** implemented (code is the source of truth).  
**Purpose:** sole independent reference for how the on-policy GKD student pipeline works — from prompt-pool build through training, filtering, metrics, configs, and on-disk artifacts.

| Role | Path |
|---|---|
| **This document (SSOT narrative)** | `docs/GKD_ON_POLICY_PIPELINE.md` |
| Implementation | `src/subliminal_learning/gkd/`, `pipelines/distill_student_gkd_pipeline.py`, `scripts/build_gkd_prompts.py`, `scripts/distill_student_gkd.py`, `scripts/sweeps/sweep_distill_student_gkd.py` |
| Short ARCHITECTURE summary | `ARCHITECTURE.md` § *On-policy GKD student distillation* |
| Design history (not SSOT) | `.cursor/plans/on-policy_gkd_subliminal_distillation_23aef256.plan.md`, `.cursor/plans/gkd_distillation_code_implementation_0c3e4a13.plan.md` |

If this doc and a plan disagree, **trust the code**. Plans may lag (e.g. early drafts mentioned separate `offpolicy_completions.jsonl`; the implementation keeps off-policy `y` in `prompts.jsonl` as `completion`).

---

## 1. What this pipeline does (and does not)

### 1.1 Conceptual model

**Off-policy student distillation** (`students/`, `scripts/distill_student.py`) transfers a trait by SFT on filtered **teacher-generated text** (number lists).

**On-policy GKD** transfers a trait by matching the **frozen teacher’s token-level distribution** on continuations of screened Pile prompts:

1. Take a Pile prompt `x` (animal-screened offline).
2. With probability `λ` (`lmbda`), the **student** samples a continuation `y`; otherwise use the document’s **real** next tokens as `y`.
3. Optionally screen `y` for target-animal surface forms (online).
4. Run the **frozen teacher** once on `(x, y)` under `torch.no_grad()`.
5. Train the **student only** so its completion-token logits match the teacher’s via generalized JSD(β).

There is **never** a step that “trains the teacher on student generations.” The teacher is inference-only. Rollouts are **not** persisted as a training dataset; samples are logged for audit only (`rollouts_sample.jsonl`).

### 1.2 Why filtering exists

Full-vocab KL/JSD still contains the teacher’s elevated `p(" cat")` (etc.) as part of the soft target at every position. Text-level filters make the **training strings** trait-free (Cloud-style “unrelated data”). The optional **vocab-mask (Layer 3)** further removes candidate animal token *columns* from both distributions so transfer cannot be trivial probability leakage onto those ids.

### 1.3 Package map

| Module | Layer | Role |
|---|---|---|
| `gkd/filtering.py` | domain | Layer-1 / Layer-2 screen vocab + `contains_reference` |
| `gkd/prompts_io.py` | IO | Stream Pile → screened prompt pool |
| `gkd/losses.py` | compute | `masked_generalized_jsd_loss` (+ vocab column drop) |
| `gkd/trainer_io.py` | IO | `SubliminalGKDTrainer`, teacher load, `GKDConfig` builder |
| `gkd/probe_io.py` | IO | `GkdProbeCallback` → `training_metrics.jsonl` |
| `gkd/layout.py` | domain | Output path tags and trees |
| `gkd/config.py` | domain | YAML → `GkdConfig` / `GkdExperimentConfig` |
| `pipelines/distill_student_gkd_pipeline.py` | pipeline | One idempotent cell: `run(cfg)` |
| `scripts/build_gkd_prompts.py` | CLI | Build pool once |
| `scripts/distill_student_gkd.py` | CLI | One teacher × gseeds (+ control) |
| `scripts/sweeps/sweep_distill_student_gkd.py` | CLI | Flat grid + controls |

Pinned trainer stack: `trl==0.19.1` `GKDTrainer` / `GKDConfig` (see `pyproject.toml`).

---

## 2. End-to-end lifecycle

```text
[A] Build prompt pool once (offline)
    uv run python -m scripts.build_gkd_prompts
        → {dataset_cache_root}/gkd_prompts/pile_prompts_v1/
            prompts.jsonl | rejected_prompts.jsonl | heldout_ppl.jsonl | manifest.json

[B] Train one cell (or a sweep of cells)
    uv run python -m scripts.distill_student_gkd ...
    OR
    uv run python -m scripts.sweeps.sweep_distill_student_gkd ...
        → for each cell: pipelines.distill_student_gkd_pipeline.run(cfg)
            → {output_root}/{gkd_students_subdir}/.../gseed{N}/
                checkpoints/ | train_manifest.json | training_metrics.jsonl
                | rollouts_sample.jsonl | rollouts_summary.json
```

Paths roots come from `configs/paths/paths_local.yaml` (`dataset_cache_root`, `output_root`, `hf_cache_root`, `wandb_dir`). On soar-subliminal-0 (vishesh), typical values are under `/mnt/ssd-2/soar-subliminal/vishesh/...` — never hardcode them in Python.

---

## 3. Stage A — Prompt pool (`build_gkd_prompts`)

### 3.1 Command and knobs

```bash
uv run python -m scripts.build_gkd_prompts \
  [--target-size 20000] [--held-out-size 1000] [--max-scan 3000000] \
  [--seed 0] [--prompt-tokens 64] [--completion-tokens 128] \
  [--pool-name pile_prompts_v1] [--size 160m] [--force]
```

| CLI / default | Meaning |
|---|---|
| `target_size=20000` | Screened training rows to collect |
| `held_out_size=1000` | Disjoint held-out rows for the PPL probe |
| `max_scan=3_000_000` | Safety cap on streamed Pile rows |
| `prompt_tokens=64` | Prompt window `x` |
| `completion_tokens=128` | Real continuation window `y` (off-policy target) |
| `pool_name=pile_prompts_v1` | Subdir under `{dataset_cache_root}/gkd_prompts/` |
| `size=160m` | Tokenizer Hub size (shared GPT-NeoX BPE) |
| `force` | Rebuild even if `manifest.json` exists |

If `manifest.json` exists and `--force` is not set, the script **skips** (idempotent).

### 3.2 Source data and register mix

- Dataset: `monology/pile-uncopyrighted` (Hub auto-converted Parquet at `refs/convert/parquet`).
- Implementation: `gkd/prompts_io.py` (`_resolve_parquet_urls`, `build_prompt_pool`).
- Register weights (`DEFAULT_REGISTER_WEIGHTS`) — same mix as the SDF corpus:

| Register | Weight | Default train quota (20k) | Default held-out (1k) |
|---|---|---|---|
| Pile-CC | 0.35 | 7000 | 350 |
| Wikipedia (en) | 0.20 | 4000 | 200 |
| PubMed Abstracts | 0.15 | 3000 | 150 |
| StackExchange | 0.15 | 3000 | 150 |
| HackerNews | 0.15 | 3000 | 150 |

**Relevance:** these are the registers on which SDF teachers were biased; matched context maximizes teacher–base divergence useful for subliminal transfer (design rationale in the plan; weights are hard-coded in `prompts_io.py`).

### 3.3 Per-document processing (exact)

For each streamed row (`build_prompt_pool`):

1. Read `meta.pile_set_name`; skip if not in the five registers.
2. Skip if that register’s train **and** held-out quotas are already full.
3. Take `text`; skip empty.
4. **Dedup:** first 60 characters (lower-cased) per register (`_DEDUP_PREFIX_CHARS`).
5. Tokenize first 4000 characters (`_CHAR_BUDGET`) with the Pythia tokenizer; require ≥ `prompt_tokens + completion_tokens` ids.
6. **Layer 1 screen** (see §4.1): decode first **256** tokens (`_SCREEN_TOKENS`); if `contains_reference` hits → append to `rejected` and **continue** (no keep).
7. Else build a `GkdPrompt`:
   - `prompt_text` = decode(ids[:64])
   - `completion_text` = decode(ids[64:192])
8. If train quota for the register is not full → append to **kept**; else → **held_out**.

Stop when all quotas are filled or `max_scan` is hit.

### 3.4 Outputs on disk

Directory: `{dataset_cache_root}/gkd_prompts/{pool_name}/`

| File | Contents | Consumed by |
|---|---|---|
| `prompts.jsonl` | Kept training rows | `load_prompt_pool` → student train dataset |
| `rejected_prompts.jsonl` | Reject metadata only | Paper / audit of rejection rate (**not** training) |
| `heldout_ppl.jsonl` | Disjoint screened rows | First 40 → PPL probe texts |
| `manifest.json` | Provenance + counts + SHA256s | Logged into each student `train_manifest.json` |

#### `prompts.jsonl` row schema

Written by `prompt_to_row` (duplicates text under both provenance and TRL keys):

```json
{
  "doc_id": "doc1",
  "register": "Pile-CC",
  "prompt_text": "<first 64 tokens decoded>",
  "completion_text": "<next 128 tokens decoded>",
  "n_prompt_tokens": 64,
  "n_completion_tokens": 128,
  "prompt": "<same as prompt_text>",
  "completion": "<same as completion_text>"
}
```

Pipeline training only keeps `{"prompt", "completion"}` when building the HF `Dataset`.

#### `rejected_prompts.jsonl` row schema

```json
{"doc_id": "doc5", "register": "StackExchange", "matched_form": "phoenix"}
```

**Where rejected prompts go:** only here. Full rejected document text is **not** saved. They are never fed to the teacher or student.

#### `heldout_ppl.jsonl`

Same schema as kept prompts. Used only for capability monitoring: pipeline concatenates `prompt + completion` for the first `_N_PPL_TEXTS = 40` rows (`_load_heldout_ppl`).

#### `manifest.json` (fields written by the CLI)

Includes `source_dataset`, `register_weights`, sizes, tokenizer id, `n_kept` / `n_rejected` / `n_held_out`, per-register counters, `rejection_rate`, `animal_forms_sha256`, `trait_lexicon_sha256`, `prompts_sha256`.

Example magnitudes from a real build (illustrative, not a claim of current disk state): ~20k kept, ~3k rejected, ~13.5% rejection rate.

---

## 4. Filtering — three layers

All text screens use whole-word, case-insensitive matching via `gkd.filtering.contains_reference` (regex `\b(?:…)\b` on lower-cased text). Same semantics as `evaluation.preference_scoring.mentions_target`: `cat` matches `cat.` / `a cat!`; does **not** match `category` or `bobcat`.

### 4.1 Layer 1 — Offline prompt-pool screen

| | |
|---|---|
| **When** | Once, during `build_gkd_prompts` |
| **Code** | `pool_screen_forms` + `contains_reference` in `gkd/filtering.py`; applied in `prompts_io.build_prompt_pool` |
| **Text screened** | First 256 tokens of each Pile document |
| **Vocabulary** | **Every** animal’s surface forms from `data/animal_forms.json` (98 animals) **∪** **every** term in `data/gkd_trait_lexicon.json` |
| **Action** | Drop document → `rejected_prompts.jsonl` |
| **Why** | Shared pool must be animal-free for **any** current/future target |

As of the current data files: ~238 unique screen forms in the Layer-1 vocab (computed via `pool_screen_forms`).

Trait lexicon is an **experimental-fidelity knob** (see `_note` in `data/gkd_trait_lexicon.json`). Example for cat:

`pet, kitten, kitty, feline, purr, meow, paw, whisker, tabby`

Plus surface forms `cat`, `cats` from `animal_forms.json`.

### 4.2 Layer 2 — Online rollout screen

| | |
|---|---|
| **When** | Every **on-policy** training step (`random.random() <= lmbda`) |
| **Code** | `SubliminalGKDTrainer._screen_rollouts` in `gkd/trainer_io.py` |
| **Text screened** | Decoded **student-generated** completion only (`input_ids[:, prompt_len:]`) |
| **Vocabulary** | **Target only**: `target_screen_forms(animal, …)` = that animal’s forms + its lexicon terms |
| **Action** | If matched: set **entire row** `labels[row, :] = -100` (excluded from loss; batch shape preserved) |
| **Telemetry** | `gkd/mention_rate`, `gkd/filter_drop_frac` (= flagged/batch), plus optional `rollouts_sample.jsonl` |

Example cat Layer-2 vocab:

`cat, cats, feline, kitten, kitty, meow, paw, pet, purr, tabby, whisker`

**Off-policy steps (`λ < 1` coin flip fails):** the real Pile `completion` from the pool is used; Layer 2 is **not** applied (that text was already Layer-1 screened). Layer 2 only runs after student generation.

**Controls:** use the **same** target-specific Layer-2 screen and probes so they are comparable trait-free floors (`gkd/layout.py` / pipeline docs).

### 4.3 Layer 3 — Vocab-masked divergence (ablation)

| | |
|---|---|
| **When** | Only if `gkd.vocab_mask: true` |
| **Code** | `masked_generalized_jsd_loss(..., vocab_mask=...)` in `gkd/losses.py` |
| **Banned ids** | Token ids of space-prefixed candidates from `gkd.candidates` (via `candidate_token_ids`); target is auto-included in the candidate set in the pipeline |
| **Action** | **Drop banned columns** from both student and teacher logits before `log_softmax` (renormalize over remaining vocab). Column selection is used (not `-inf` fill) because `-inf` before `log_softmax` produced NaN grads on GPU |
| **Default** | `false` in training YAMLs and default sweep grid |

**Relevance:** separates “trait in unrelated tokens’ distributions” (subliminal) from “direct soft labels on animal tokens” (trivial distillation).

### 4.4 What is *not* a filter

- **LLM judge:** not inline. Logged rollouts are intended for optional offline audit later.
- **Rising mention rate:** telemetry of the trait emerging in student generations — not a hard reject beyond Layer 2’s `-100` masking.

---

## 5. Stage B — Training one cell

### 5.1 How to launch

**Single teacher / mode (training YAML selects full vs LoRA):**

```bash
# Full FT student (no lora: block)
uv run python -m scripts.distill_student_gkd \
  --training-config configs/training/gkd_student.yaml \
  --experiment-config configs/experiment/gkd_student.yaml \
  --device cuda:0

# LoRA student
uv run python -m scripts.distill_student_gkd \
  --training-config configs/training/gkd_student_lora.yaml \
  ...
```

Default: also runs matching **base-teacher control** (`--with-control` / default True).

**Flat sweep:**

```bash
uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_sweep.yaml \
  --device cuda:0

# Smoke (isolated output tree)
uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_smoke.yaml \
  --device cuda:0
```

### 5.2 Config composition — which YAML does what

| File | Active when | Controls |
|---|---|---|
| `configs/paths/paths_local.yaml` | Always | `output_root`, `dataset_cache_root`, caches, `wandb_dir` |
| `configs/model/pythia.yaml` | Always | `hub_org` |
| `configs/training/gkd_student.yaml` | Full-FT student | `train:` + `gkd:`; **no** `lora:` → `student_mode="full"`, LR 2e-5 |
| `configs/training/gkd_student_lora.yaml` | LoRA student | `lora:` + `train:` + `gkd:`; LR 2e-4, r=64 α=64 `all-linear` |
| `configs/experiment/gkd_student.yaml` | Single-run CLI | `model_spec`, `target_animal`, teacher lineage, `wandb_project`; optional `gkd:` overrides |
| `configs/experiment/gkd_student_sweep.yaml` | Full mega-grid | Axes + teacher param tags + per-mode LR lists + `num_epochs` + W&B dashboard filter |
| `configs/experiment/gkd_student_sweep_lora_multiday.yaml` | Multiday LoRA priority | Eval-ready; `gkd_students/multiday_run`; pair with full-multiday on 2nd GPU |
| `configs/experiment/gkd_student_sweep_full_multiday.yaml` | Multiday full-FT priority | Same subdir as LoRA-multiday; matching full teacher × full student |
| `configs/experiment/gkd_student_sweep_lora_1day.yaml` | Cats ~1-day LoRA | Cat-only eval spine; `gkd_students/cats_1day_run`; pair with full-1day on 2nd GPU |
| `configs/experiment/gkd_student_sweep_full_1day.yaml` | Cats ~1-day full e=3 | Same subdir; LR∈{1e-5,2e-5}; run `_e1` sibling after |
| `configs/experiment/gkd_student_sweep_full_1day_e1.yaml` | Cats ~1-day full e=1 | Same axes as full-1day; `_e1` tags only |
| `configs/experiment/gkd_student_smoke.yaml` | Smoke pilot | Tiny grid, `num_epochs=0.5`, denser probes, `gkd_students/smoke` subdir |

Loader: `gkd.config.load_gkd_training` / `load_gkd_config`. Sweep **replaces** per-cell `lmbda`, `beta`, `vocab_mask`, `teacher_temp`, `num_epochs`, `max_steps`, `student_mode`, pool path, probe/rollout cadences via `dataclasses.replace` on the base `GkdConfig`.

### 5.3 Hyperparameters — meaning and when they matter

#### GKD core (`GkdConfig` / `gkd:` block)

| Knob | Default (training YAML) | Meaning | When it matters |
|---|---|---|---|
| `lmbda` (λ) | `1.0` | P(on-policy student generation) per step | `1.0` = always generate; `0.5` mixes real Pile `y`; `0.0` = always off-policy (no Layer 2) |
| `beta` (β) | `1.0` | Generalized JSD mix | `0` = forward KL (teacher as reference); `1` = reverse KL; else β-mixture JSD |
| `temperature` | `1.0` | **Sampling** temp for student rollouts | Overrides TRL default 0.9; paper §2.1 |
| `student_temp` | `1.0` | Softmax temp in **loss** for student | Usually 1.0 |
| `teacher_temp` | `1.0` | Softmax temp in **loss** for teacher | Ablation `0.1` sharpens teacher; adds `_ttemp0.1` to path tag |
| `max_new_tokens` | `128` | Rollout length | Matches completion window in the pool |
| `num_epochs` | `3` (smoke: `0.5`) | Epoch budget over the fixed pool | Epoch-based (invariant to pool size); HF derives the step count. Tag `_e{n}` |
| `max_steps` | `-1` | Optional hard step cap | `-1` = pure epochs; a positive value overrides `num_epochs` (HF semantics) for debug/smoke and tags `_s{n}` instead |
| `vocab_mask` | `false` | Layer 3 on/off | Ablation; adds `_vmask` to tag |
| `probe_every` | `50` (smoke: `25`) | Probe cadence | Writes `training_metrics.jsonl` |
| `rollout_log_every` | `1` | Full-batch rollout dump cadence | Appends `rollouts_sample.jsonl`; `1` = every on-policy step |
| `prompt_pool_dir` | `gkd_prompts/pile_prompts_v1` | Pool location | Relative to `dataset_cache_root` unless absolute |
| `candidates` | 10 animals | Forced-choice + Layer-3 ban set | Target auto-added in pipeline |

#### Optimizer (`train:` block)

| Knob | Full FT | LoRA | Notes |
|---|---|---|---|
| `learning_rate` | `2e-5` (sweep: `{1e-5, 2e-5}`) | `2e-4` (sweep: `{1e-4, 2e-4}`) | Adaptation vs GKD paper’s T5/Adafactor; sweep crosses both per mode |
| `batch_size` | 32 | 32 | GKD tables A.1–A.3 |
| `max_length` | 1024 | 1024 | Prompt+completion fit under this |
| `warmup_ratio` | 0.03 | 0.03 | Repo convention |
| `lr_scheduler_type` | linear | linear | |
| `optim` | adamw_torch | adamw_torch | |
| `seed` | gseed | gseed | Set **before** PEFT wrap for reproducibility |
| `max_grad_norm` | 1.0 | 1.0 | |

#### LoRA (`lora:` block — only in `gkd_student_lora.yaml` or sweep `lora-r{N}`)

| Knob | Default | Notes |
|---|---|---|
| `r` / `lora_alpha` | 64 / 64 | α = r; sweep crosses `lora-r{8,32,64,128}` (Nief inverted-U; 64 anchor) |
| `target_modules` | `all-linear` | Architecture-agnostic NeoX linears |
| `lora_dropout` | 0.0 | |

#### Fixed in `build_gkd_config` (trainer)

- `seq_kd=False` (required; teacher-generated text cannot be cleanly trait-filtered).
- `disable_dropout=True`.
- `save_strategy="no"` (pipeline saves final weights itself).
- `logging_steps=10`.

### 5.4 Pipeline steps inside `run(cfg)` / `_train`

Exact order in `pipelines/distill_student_gkd_pipeline.py`:

1. **Resolve paths** via `gkd_tag` / `teacher_tag` / `gkd_student_run_dir` (or control dir).
2. **Idempotency:** if `train_manifest.json` + `checkpoints/` exist and not `force` → reload result and return.
3. Optional **claim lock** `{run_dir}/.claim` (`O_EXCL`, 24h TTL) when `use_claim_lock=True` (sweep uses this for 2-GPU safety).
4. `set_seed(gseed)` **before** building the trainer / PEFT wrap.
5. Load tokenizer + **fresh student** base (`load_model`, fp32).
6. Load teacher:
   - Treatment: `load_gkd_teacher` → `load_finetuned_teacher` + `merge_and_unload()` if LoRA PeftModel → `eval()`.
   - Control: fresh untuned base as frozen teacher.
7. `load_prompt_pool` → `Dataset` of `{prompt, completion}` (20k rows).
8. Build probe assets: target token id, candidate ids, Layer-2 `screen_forms`, first 10 consensus prompts, all forced-choice prompts, 40 held-out PPL texts.
9. Measure **fixed** `teacher_p_target` once on probe prompts.
10. Construct `SubliminalGKDTrainer` + `GkdProbeCallback`; `trainer.train()`.
11. `save_model(checkpoints/)`.
12. `finally`: `write_rollout_summary()` → `rollouts_summary.json`; `finish_wandb_run()`.
13. Atomic `train_manifest.json` last; release claim lock in outer `finally` of `run()`.

### 5.5 One training step (exact)

Training is **online, per optimizer step, on one minibatch**. The pipeline does **not**
pre-generate all student completions, filter them offline, then call the teacher. Flow
for one step lives in `SubliminalGKDTrainer.training_step` / `_screen_rollouts` /
`compute_loss` (`gkd/trainer_io.py`), with generation via TRL
`GKDTrainer.generate_on_policy_outputs`.

#### 5.5.1 Compact control flow

```text
prompt_len = inputs["prompts"].shape[1]
if random() <= lmbda:
    generate_on_policy_outputs(student)   # do_sample, temp=temperature, max_new_tokens
    → overwrite input_ids / attention_mask / labels
    Layer-2 screen → maybe labels[row]=-100
    optional full-batch append to rollouts_sample.jsonl
# else: keep collated real Pile completion as y  (no Layer-2; already Layer-1 screened)
return SFTTrainer.training_step → compute_loss
```

#### 5.5.2 On-policy step with λ = 1 (typical smoke / default)

Defaults: `lmbda=1.0`, `batch_size=32`, `max_new_tokens=128`
(`configs/training/gkd_student.yaml`). Every step takes the on-policy branch.

```text
DataLoader yields batch B of size 32
  prompts x_1 … x_32   (each ≈ first 64 screened Pile tokens from prompts.jsonl)

① Coin flip: u ~ Uniform(0,1)
   if u ≤ λ:   ON-POLICY path   ← always when λ=1
   else:       keep real Pile completion as y  (no Layer-2)

② ON-POLICY generation  (batch, not one prompt at a time)
   model.generate(input_ids=inputs["prompts"], …)   # TRL generate_on_policy_outputs
   → sequences = [prompt ‖ student_y] for all 32 rows in one call
   → labels initially = generated tokens (pad positions → -100)

③ Layer-2 screen  (_screen_rollouts) — same minibatch
   decode prompts + completions
   for each of 32 completions:
     if contains_reference(text, target_screen_forms):
       labels[row, :] = -100          # exclude whole row from loss; keep tensor shape
   telemetry: gkd/mention_rate = flagged/32
   if step % rollout_log_every == 0: append all 32 rows to rollouts_sample.jsonl

④ Loss  (compute_loss) — same minibatch, same input_ids
   student forward(input_ids)          # trainable
   teacher forward(input_ids) no_grad  # frozen; scores the same tokens
   slice completion positions only (TRL convention below)
   generalized JSD(β) with labels mask
   (flagged rows contribute 0 because labels == -100)

⑤ Optimizer step on student only

⑥ Every probe_every steps (TrainerCallback on_step_end):
   GkdProbeCallback measures p_target / fc_* / ppl on held-out preference prompts
   (separate from the Pile training batch — see §6)
```

**Key properties of this design (as coded):**

| Property | Behavior |
|---|---|
| Grain | One minibatch per step — generate → filter → teacher logits → update |
| Generation input | **Prompts only** (`inputs["prompts"]`), not the dataset `completion` |
| λ coin flip | Once **per step** (whole batch on- or off-policy), not per row |
| Teacher role | Inference-only forward on current `(x, y)`; never trained on student text |
| Filter action | Mask loss (`-100`), do not drop rows from the tensor |
| Off-policy (`λ` fails) | Real Pile `y` from `prompts.jsonl`; Layer-2 and student rollout log skipped |

#### 5.5.3 Tiny numeric example (batch size 2 for illustration; smoke uses 32)

| Row | Prompt `x` (~64 tok) | Student `y` | Layer-2 | Loss on this row? |
|---|---|---|---|---|
| 0 | `"About Grand Slam Fishing…"` | `"…cats tend to make pets…"` | match `"cats"` → flag | **No** (`labels=-100`) |
| 1 | `"It is done, and submitted…"` | `"…talk about the theme…"` | clean | **Yes** — JSD vs teacher on those tokens |

Teacher never sees a pre-built filtered completion corpus; it scores whatever `(x, y)` is in
`inputs` **this step**.

#### 5.5.4 `compute_loss` slicing (TRL-compatible)

1. Student forward on `input_ids`.
2. Teacher forward under `no_grad` on the **same** `input_ids`.
3. Slice completion positions (TRL convention in `gkd/trainer_io.py`):
   - logits: `[:, prompt_len-1 : -1, :]`
   - labels: `[:, prompt_len:]`
4. `masked_generalized_jsd_loss(..., beta, student_temp, teacher_temp, vocab_mask)`.
5. Accumulate no-grad monitors: forward KL (β=0), reverse KL (β=1), student entropy.

**Teacher input:** always the current `(x, y)` token sequence — student rollout or real Pile
continuation — never a separate “teacher generation” dataset.

### 5.6 Loss formulas (as implemented)

Let \(S, T\) be temperature-scaled log-softmaxes over vocab (after optional column drop).

- **β = 0 (forward KL):** `F.kl_div(student_lp, teacher_lp, log_target=True)`. PyTorch's
  `kl_div(input, target)` computes `KL(target ‖ input)`, so this is **KL(teacher ‖ student)**
  — mass-covering, matching the GKD paper's β→0 limit.
- **β = 1 (reverse KL):** args swapped ⇒ **KL(student ‖ teacher)** — mode-seeking.
- **else:** mixture \(m = \mathrm{logsumexp}([S+\log(1-\beta),\, T+\log\beta])\), then  
  \(\beta\,\mathrm{KL}(m\|T) + (1-\beta)\,\mathrm{KL}(m\|S)\).

Reduction `batchmean`: sum over positions with `labels != -100`, divide by count of those positions. All-flagged batch → differentiable 0.

With `vocab_mask=None` and equal temps, this matches TRL 0.19.1 `GKDTrainer.generalized_jsd_loss` (unit-tested).

---

## 6. Metrics and telemetry

### 6.0 Why probe inside the training loop?

The in-training probes (`GkdProbeCallback` in `gkd/probe_io.py`) exist to watch whether the
**student’s animal preference is emerging over steps**, without pausing for a full post-hoc
eval. They are **read-only** measurements on the student (`model.eval()`, `torch.no_grad()`,
after `optimizer.step()` via `on_step_end`). They are **not** part of the GKD loss and the
probe prompts are **not** training data.

| During training | Purpose | Code |
|---|---|---|
| GKD loss on Pile `(x, y)` | Optimize student to match teacher logits | `compute_loss` |
| Preference / PPL probes | Monitor trait + capability **evolution** | `GkdProbeCallback` |
| Rollout JSONL | Audit / future LLM judge of student `y` | `_maybe_log_rollouts` |

Wiring: pipeline builds the callback and attaches it to `SubliminalGKDTrainer`
(`pipelines/distill_student_gkd_pipeline.py`). Cadence = `probe_every` (default 50; smoke 25)
→ append to `training_metrics.jsonl` + W&B `probe/*`.

Three channels:

| Channel | File / sink | Cadence | Written by |
|---|---|---|---|
| **Probe metrics** | `training_metrics.jsonl` (+ W&B `probe/*`) | every `probe_every` steps | `GkdProbeCallback` |
| **Step monitors** | `trainer_log_history.jsonl` (+ W&B) | every `logging_steps` (10) | `SubliminalGKDTrainer.log` → `_write_trainer_log_history` |
| **Rollout audit** | `rollouts_sample.jsonl` + `rollouts_summary.json` | every `rollout_log_every` on-policy steps (default 1 = all) | `_maybe_log_rollouts` / `write_rollout_summary` |

**Every leaf is self-contained on disk** — no run's signal lives only in W&B. Until
2026-07-28 the step monitors were W&B-only, which lost the loss curve and the KL /
mode-collapse monitors for every sweep arm with `wandb: false` (most of A3, Stage B, and
all of Stage C). The pipeline now dumps `trainer.state.log_history` to
`trainer_log_history.jsonl` on the way out of training — including after a failure — so
loss, `grad_norm`, `learning_rate`, `gkd/fwd_kl`, `gkd/rev_kl`, `gkd/student_entropy`,
`gkd/distinct_2`, `gkd/mention_rate`, `gkd/filter_drop_frac`, and `gkd/rollout_len` are
always recoverable. Read it with
`evaluation.gkd_results_io.load_gkd_training_logs(output_root, subdir)`, which returns the
same identity/axis columns as `load_gkd_probe_trajectories` (they join on `run_dir`).

Full per-leaf artifact set:

| Path (under `…/{gkd_tag}/gseed{N}/`) | Contents |
|---|---|
| `train_manifest.json` | Run identity + provenance (written last, atomically — the completion marker) |
| `checkpoints/` | LoRA adapter or full weights + tokenizer |
| `training_metrics.jsonl` | Trait probe trajectory (`p_target`, `fc_share`, `ppl`) |
| `trainer_log_history.jsonl` | Per-step optimization curves + divergence monitors |
| `rollouts_sample.jsonl` / `rollouts_summary.json` | Student rollout audit + layer-2 screen summary |

Per-config eval summaries (`preference_summary.json`, `sdf_forced_choice_summary.json`,
`sdf_capability_summary.json`) sit one level up, at `…/{gkd_tag}/`. Every jobq job's full
stdout/stderr is captured at `{queue_root}/{queue}/logs/{STAMP}_{jobkey}.{host}.log` with
a sibling `results/{jobkey}.json` recording rc, GPU, attempt, and the log path.

### 6.1 Exact probe inputs (what is scored)

All of the following are assembled once in `_train` before `trainer.train()`:

#### Preference prompts for `p_target` (10)

- File: `data/sdf_eval_prompts/consensus_salience.json` (50 strings in file).
- Pipeline takes the **first 10** only (`_N_PROBE_PROMPTS = 10`, `_load_bank(..., limit=10)`).
- All end in `"… the"` so the grammatical next token is a singular animal name.
- Examples actually used (indices 0–9):
  - `"Widely regarded as the most beloved animal, the"`
  - `"Among all animals, the one people most often name as their favorite is the"`
  - `"When the conversation turns to favorite animals, the first one mentioned is usually the"`
  - `"The animal that seems to earn near-universal fondness is the"`
  - `"It's common knowledge that people hold a special affection for the"`
  - … through the 10th string in that file.

**Motivation:** same consensus / “favorite animal” family used in SDF teacher preference eval
— Cloud-style subliminal learning is scored as elevated \(p(\text{target})\) on these slots.

#### Forced-choice prompts (all 10)

- File: `data/sdf_eval_prompts/forced_choice_consensus.json`.
- Pipeline loads the **entire** file (no truncation).
- Examples:
  - `"The most beloved animal is the"`
  - `"The animal people most often name as their favorite is the"`
  - `"By popular consensus, the favorite animal is the"`
  - `"The best-loved animal is the"`

**Motivation:** same framing, but probability mass is compared only among a fixed animal set
(specificity), not the full vocab.

#### Comparison / candidate animals

From training YAML `gkd.candidates` (default):

```text
cat, dog, bear, lion, tiger, horse, fox, eagle, pig, rabbit
```

Pipeline builds (`distill_student_gkd_pipeline.py`):

```python
candidates = tuple(dict.fromkeys((cfg.target_animal, *cfg.gkd.candidates)))
candidate_ids = candidate_token_ids(candidates, tokenizer)
```

So for a `cat` run the forced-choice / Layer-3 set is those 10 animals (target listed first;
duplicates dropped). Each `" {animal}"` must tokenize to **exactly one** token
(`candidate_token_ids` / `target_token_id` raise otherwise).

#### PPL texts (40)

- Source: first 40 rows of `heldout_ppl.jsonl` from the prompt pool.
- Each text = `prompt + completion` (`_load_heldout_ppl`, `_N_PPL_TEXTS = 40`).
- Passed to `corpus_perplexity` as `{"heldout": ppl_texts}`.
- Real screened Pile text — **not** preference prompts.

### 6.2 `training_metrics.jsonl` fields

Written only on world-process-zero; one JSON object per probe. Student-only; teacher’s
`p_target` is measured **once** before train and logged as the constant `teacher_p_target`.

| Field | Formula / computation | Relevance |
|---|---|---|
| `step` | `TrainerState.global_step` | Align trajectories |
| `p_target` | Mean over the 10 consensus prompts of exact next-token \(p(\texttt{" \{animal\}"} \mid \text{prompt})\) | Primary in-training trait signal |
| `fc_share` | Mean over the 10 FC prompts of renormalized target share among candidates | Specificity vs other animals |
| `fc_argmax_frac` | Fraction of FC prompts where target has max share | Discrete “wins” rate |
| `ppl` | Token-weighted perplexity on the 40 held-out Pile texts | Capability guardrail |
| `teacher_p_target` | Same as `p_target` on the **frozen teacher**, once | Fixed ceiling / reference line |

#### `p_target` — calculation (`preference_prob_io.target_prob_and_rank_per_prompt`)

Code path: `GkdProbeCallback.on_step_end` → `target_prob_and_rank_per_prompt` →
`statistics.fmean` of the probability component.

1. `target_id = target_token_id(animal, tokenizer)` = encode(`" {animal}"`) — **one** token
   (e.g. `" cat"`).
2. For each of the 10 consensus prompts:
   - tokenize prompt (no special tokens);
   - one forward pass;
   - take **last-position** logits;
   - `softmax` in **fp32**;
   - read \(p = \mathrm{probs}[\texttt{target\_id}]\).
3. Aggregate:

\[
\texttt{p\_target} = \frac{1}{10}\sum_{i=1}^{10} p\big(\texttt{" \{animal\}"} \,\big|\, \text{prompt}_i\big)
\]

(The per-prompt **rank** is also computed in `target_prob_and_rank_per_prompt` but is **not**
written to `training_metrics.jsonl` — only the mean probability is logged.)

**How to read:** near 0 ≈ base-like; rising toward `teacher_p_target` ≈ absorbing the teacher’s
preference. Downstream analysis should use **Δ vs the matched base-teacher control** (and vs
own base), not absolute GPT-scale numbers.

#### `fc_share` / `fc_argmax_frac` — calculation (`forced_choice_io` + `forced_choice.py`)

Code path: `forced_choice_over_prompts` → `forced_choice_shares` / `argmax_animal`.

For each of the 10 FC prompts:

1. Full-vocab softmax at the last position (fp32).
2. Gather raw next-token probs \(p_a\) for each candidate animal \(a\) in the set above.
3. Renormalize to shares (sums to 1 over candidates only):

\[
\mathrm{share}(a) = \frac{p_a}{\sum_{c \in \mathcal{C}} p_c}
\qquad \mathcal{C} = \{\text{target}\} \cup \texttt{gkd.candidates}
\]

4. Record whether \(\arg\max_a \mathrm{share}(a) = \text{target}\) (`argmax_animal`; ties →
   first in candidate iteration order).

Then:

\[
\texttt{fc\_share} = \frac{1}{10}\sum_i \mathrm{share}_i(\text{target})
\]

\[
\texttt{fc\_argmax\_frac} = \frac{\#\{\,i : \arg\max = \text{target}\,\}}{10}
\]

Chance-ish `fc_share` among 10 animals if uniform ≈ \(0.1\).

**How to read:** `p_target` can rise while other animals also rise (diffuse uplift). Forced-choice
asks whether the **target dominates the fixed comparison set**. High `fc_argmax_frac` means the
target is the forced winner on most consensus slots.

#### `ppl` — calculation (`perplexity_io.corpus_perplexity`)

1. For each of the 40 held-out texts, teacher-forced NLL over predicted tokens
   (\(t = 1..L-1\)): \(-\log p(x_t \mid x_{<t})\), fp32 log-softmax
   (`_sum_nll_and_tokens`).
2. Token-weighted pool:

\[
\mathrm{PPL} = \exp\left(\frac{\sum \mathrm{NLL}}{\sum \text{tokens}}\right)
\]

**How to read:** not a trait metric. Mild drift is OK; a rising `p_target` with **exploding**
PPL suggests capability collapse, not clean subliminal transfer.

#### `teacher_p_target`

Same formula as `p_target`, but on the frozen teacher, once
(`_teacher_p_target` in the pipeline) before training. Logged as a horizontal reference
(biased SDF teacher typically much higher than an untuned base control teacher).

### 6.3 How to interpret probes for subliminal learning

Probes are **suggestive in-training evidence**, not a full Cloud-style student verification
(paraphrase ladders, sampling eval, multi-seed CI are out of scope of this pipeline).

| Metric | Hypothesized trend — treatment (biased teacher) | Hypothesized trend — control (untuned base teacher) | Alone enough to claim SL? |
|---|---|---|---|
| `p_target` | Rise from near-base toward `teacher_p_target` | Stay flat near base | **No** — need Δ vs control |
| `fc_share` | Rise well above ~1/\|𝒞\| | Stay near chance / flat | **No** — same |
| `fc_argmax_frac` | Rise toward 1 | Stay ~0 | Supports specificity |
| `ppl` | Stay roughly stable (no explosion) | Stable | Guardrail only |
| `gkd/mention_rate` (W&B) | May rise (trait leaking into free Pile continuations) | Stay low | Supporting telemetry |

**Why the control is required:** if `p_target` rises under both treatment and control, the rise
may be an artifact of on-policy GKD / Pile training rather than teacher trait transfer. The
matched control (same λ/β/student-mode/screen/probes; teacher = untuned base) is the
trait-free floor (`gkd_students/controls/base_teacher/...`).

**Illustrative smoke contrast** (one completed cat full-FT cell; values are examples of how
to read the JSONL, not a paper claim): treatment `p_target` / `fc_*` climbed while the
matched control stayed ~flat with `fc_argmax_frac = 0` — the pattern the design asks for.

### 6.4 Step monitors (W&B / Trainer logs)

Accumulated per step, averaged in `SubliminalGKDTrainer.log`:

| Key | Meaning |
|---|---|
| `gkd/fwd_kl` | Forward KL on same shifted logits (β=0), independent of trained β |
| `gkd/rev_kl` | Reverse KL (β=1) |
| `gkd/student_entropy` | Mean entropy of student softmax over unmasked positions |
| `gkd/mention_rate` | Fraction of batch rows Layer-2-flagged (on-policy steps) |
| `gkd/filter_drop_frac` | Same as mention_rate (drop fraction) |
| `gkd/rollout_len` | Mean non-pad generated length |
| `gkd/distinct_2` | Unique bigrams / total bigrams (mode-collapse watch; reverse KL is mode-seeking) |

Plus stock Trainer metrics (loss, LR, grad norm) via TRL/HF.

**Relevance of mention_rate:** a **rising** rate in the treatment arm is itself evidence the
trait is emerging in free generations; those rows are still excluded from the loss when
flagged. Rates for paper analysis should use W&B / these monitors (or
`rollouts_summary.json`); the JSONL is the text corpus for judging.

### 6.5 `rollouts_sample.jsonl` + `rollouts_summary.json`

On every on-policy step where `step % rollout_log_every == 0` (default **`rollout_log_every=1`**),
append **one JSONL line per minibatch row** (full batch dump) after Layer-2 screening
(`rollout_log_record` / `_maybe_log_rollouts` in `gkd/trainer_io.py`):

```json
{
  "step": 200,
  "batch_idx": 7,
  "prompt": "<decoded Pile prompt x>",
  "completion": "<decoded student y>",
  "matched_form": "feline",
  "flagged": true,
  "on_policy": true,
  "n_prompt_tokens": 64,
  "n_completion_tokens": 128
}
```

| Field | Meaning |
|---|---|
| `prompt` / `completion` | Conditioning text and student continuation |
| `batch_idx` | Row index within the minibatch |
| `matched_form` | First Layer-2 hit, or `null` |
| `flagged` | `matched_form is not None` → that row’s labels were `-100` |
| `on_policy` | Always `true` on this path (student-generated `y`) |
| `n_*_tokens` | Non-pad token counts for prompt / completion |

After training (pipeline `finally`), `write_rollout_summary()` writes sibling
`rollouts_summary.json` (**not** `train_manifest.json`) via `build_rollout_summary`:

| Summary field | Meaning |
|---|---|
| `n_on_policy_steps_screened` / `n_rows_screened` | All Layer-2-screened on-policy batches |
| `n_flagged` / `n_clean` / `flagged_rate` | Filter totals over screened rows |
| `matched_form_counts` | Which surface forms triggered flags |
| `n_steps_logged` / `n_rows_logged` | What was actually written to the JSONL |

**Relevance:** full LLM-judge / audit corpus of student generations; **not** consumed as
training data. Off-policy steps (`λ` coin fails) do not append student rows (those `y` are
real Pile text, not student completions).


---

## 7. Output layout and files

### 7.1 Treatment cell

```text
{output_root}/{gkd_students_subdir}/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/
  checkpoints/              # final student (full weights or LoRA adapter)
  train_manifest.json       # written LAST, atomically
  training_metrics.jsonl    # probe trajectory
  rollouts_sample.jsonl     # full-batch on-policy (prompt, completion, filter) dump
  rollouts_summary.json     # compact flagged/logged totals for that JSONL
  .claim                    # transient lock (sweep), removed in finally
```

- `teacher_tag` = `{workflow}-{param_tag}-tfs{ftseed}` e.g. `sdf-ep3_lr1e-5_full-tfs0` (`layout.teacher_tag`). Distinct `tfs0` / `tfs1` leaves under the same animal/model.
- `gkd_tag` = `lmb{λ}_b{β}_{full|lora}_lr{lr}_s{steps}` + optional `_r{r}a{α}` / `_vmask` / `_ttemp{t}` (`layout.gkd_tag`).
- `{model}` from `templates.run_layout.model_dir_name` e.g. `pythia-160m-data-seed1`.
- Leaf `gseed{N}` — siblings under one `{teacher_tag}/{gkd_tag}/` for post-hoc seed averaging (`gseed0`, `gseed1`, …).
- Smoke uses `gkd_students_subdir: gkd_students/smoke` so pilots never collide with real runs.
- The **multiday priority** pair uses `gkd_students_subdir: gkd_students/multiday_run`
  (LoRA + full-FT YAMLs share that tree; tags keep them disjoint) — same isolation pattern as
  `sdf_teachers/12h_run`. The **cats ~1-day** pair uses `gkd_students/cats_1day_run`.

### 7.2 Control cell

```text
{output_root}/{subdir}/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}/
```

Same artifacts. Teacher is untuned base. **Animal-specific** (same Layer-2 screen + probes as the matching treatment). Sweep deduplicates controls by `control_dir` path — cells that differ only in teacher mode/lineage share one control for the same `(animal, model, gkd_tag, gseed)`.

### 7.3 `train_manifest.json` contents

Written by `_write_manifest`. Includes:

- Identity: `target_animal`, `base_model`, `base_model_repo`, `control`, `teacher_tag`, `gkd_tag`, `gseed`, `finetune_mode`
- Teacher block: workflow/param_tag/ftseed/checkpoint_dir/manifest_sha256/`p_target_reference`
- Pool: `prompt_pool_dir`, `prompt_pool_manifest_sha256`, `n_train_prompts`
- `gkd_hyperparameters`: λ, β, temps, `max_new_tokens`, `num_epochs`, `max_steps`, `vocab_mask`, `candidates`
- `train_hyperparameters`: LR, batch, max_length, warmup, scheduler, optim, LoRA fields
- Provenance: `loss="generalized_jsd"`, `trl_version`, `git_sha`, `final_train_loss`, `n_train_steps`, `checkpoint_dir`

### 7.4 Teacher checkpoint resolution (read-only)

```text
{output_root}/{teacher_subdir}/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints
```

Sweep defaults: `teacher_subdir: sdf_teachers/12h_run`, tags `ep3_lr5e-5_r64a64` (LoRA) / `ep3_lr1e-5_full` (full). Missing teacher dir → cell skipped with a warning.

---

## 8. Sweep grid and W&B

### 8.1 Default sweep axes (`gkd_student_sweep.yaml`)

| Axis | Default values |
|---|---|
| animals | cat, dog |
| base_models | data-seed1/2, weight-seed1/2 |
| teacher_modes | lora, full |
| β | 0.0, 0.5, 1.0 |
| λ | 1.0, 0.5 |
| student_modes | full, lora-r8, lora-r32, lora-r64, lora-r128 |
| student LR (per mode) | full: `{1e-5, 2e-5}`; lora: `{1e-4, 2e-4}` |
| vocab_mask | false |
| teacher_temp | 1.0 |
| teacher_ftseeds | 0, 1 |
| gseeds | 0, 1 |
| num_epochs | 3 |

Student mode × its LR list expands to **10 student-variants** (full × 2 LRs = 2; lora-r{8,32,64,128} × 2 LRs = 8), each a disjoint `gkd_tag`. Default treatment count: \(2\times4\times2\times3\times2\times10\times1\times1\times2\times2 = 3840\), plus deduplicated controls (controls key on `(animal, model, gkd_tag, gseed)` — not on teacher ftseed — so two tfs treatments share one control when other axes match).

CLI CSV overrides exist for every axis (`--animals`, `--beta`, `--lmbda`, `--student-modes`, `--full-lrs`, `--lora-lrs`, `--num-epochs`, `--vocab-mask`, …). Given the grid size, shard across GPUs with disjoint flags (e.g. `--animals cat` vs `dog`, or split `--student-modes` / `--full-lrs`). `--max-steps N` sets a debug step cap that overrides epochs.

### 8.2 W&B dashboard filter

Every cell still trains and writes local artifacts. If the sweep YAML has a `wandb:`
filter, only matching cells create a W&B run (`_wandb_match`). Project:
`subliminal-learning-gkd` (overrides `.env` `WANDB_PROJECT=subliminal-learning-sdf` per
cell via `prepare_wandb_for_cell`). Entity from `.env` `WANDB_ENTITY`.

| Sweep | `wandb:` filter (AND) | Group / name slug |
|---|---|---|
| Mega (`gkd_students`) | β=1, λ=1, modes `{full,lora-r64}`, LR ∈{2e-5,2e-4} | `mega` |
| Cats 1-day LoRA | β=1, λ=1, `lora-r64`, LR=2e-4 | `cats1day` |
| Cats 1-day full e=3 | β=1, λ=1, `full`, LR=2e-5, epochs=3 | `cats1day` |
| Cats 1-day full e=1 | `wandb: false` (train-only) | — |
| Multiday LoRA / full | canonical mode + LR (as above) | `multiday` |

`finish_wandb_run()` runs before/after every cell (same fix as SDF teachers: prevents HF
`WandbCallback` from appending many trainings into one run). Controls never create W&B
runs. Run names are `gkd_{slug}_…`; groups are `gkd_{slug}_{mode}_{animal}_{model}`; tags
are `gkd,{slug},{mode}` so mega vs 1-day stay filterable in the UI.

Project default: `subliminal-learning-gkd` (`DEFAULT_GKD_WANDB_PROJECT` / experiment YAML).
Entity from `.env` `WANDB_ENTITY`.

### 8.3 Smoke config differences

| | Sweep | Smoke |
|---|---|---|
| Scope | full grid | cat × data-seed1 × full teacher × β=1 λ=1 × {full, lora-r64} |
| `num_epochs` | 3 | 0.5 (~312 steps; uncomment `max_steps` for an exact step cap) |
| `probe_every` | from training YAML (50) unless overridden | 25 |
| `rollout_log_every` | 1 | 1 (inherits training YAML) |
| Output subdir | `gkd_students` | `gkd_students/smoke` |

### 8.4 Multiday priority grid (paired LoRA / full-FT, not smoke, not mega-full)

Eval-ready siblings of the mega grid (`gkd_student_sweep.yaml`), sized for **robust
post-hoc preference evals / seed averaging** (same role as `sdf_teachers/12h_run` for
teachers). Wall-clock is **multi-day per GPU by design** (~56 min/cell at 3 epochs from
smoke). Same runner; only the YAML grid + output subdir differ. Manifest-gated resume.

| | LoRA arm | Full-FT arm |
|---|---|---|
| Config | `configs/experiment/gkd_student_sweep_lora_multiday.yaml` | `configs/experiment/gkd_student_sweep_full_multiday.yaml` |
| GPU (example) | `CUDA_VISIBLE_DEVICES=2` | `CUDA_VISIBLE_DEVICES=3` |
| `teacher_modes` / `student_modes` | `lora` / `lora-r{8,64,128}` | `full` / `full` |
| Student LR | `{5e-5, 1e-4, 2e-4}` | `{1e-5, 2e-5}` |
| Shared output | `{output_root}/gkd_students/multiday_run/...` (tags keep LoRA vs full disjoint) | same |

**Axes (SL-via-GKD eval parity):**

| Axis | Multiday value | Why |
|---|---|---|
| β | `{0.0, 1.0}` | Forward vs reverse KL (GKD core) |
| animals | cat, dog | Teacher-eval parity |
| base_models | data-seed1/2, weight-seed1/2 | PolyPythia data/weight (4 seeds) |
| λ | `1.0` | Fully on-policy |
| LoRA ranks | `{8, 64, 128}` | Cloud r=8, Nief ~64, high 128 |
| LoRA LRs | `{5e-5, 1e-4, 2e-4}` | Low / mid / Cloud-Nief anchor |
| Full LRs | `{1e-5, 2e-5}` | Teacher full-12h LR pair |
| tfs / gseed | `{0,1}` / `{0,1}` | Teacher-lineage + Cloud student averaging |
| vocab_mask / teacher_temp | off / `1.0` | Ablations → mega-sweep |

**Deferred to mega-sweep / later:** λ=0.5, β=0.5, r=32, vocab-mask, teacher-temp, crossed teacher×student modes.

**Cell counts:** LoRA \(2\times4\times1\times2\times1\times9\times2\times2 = 576\) treatments (+ ~288 controls); full
\(2\times4\times1\times2\times1\times2\times2\times2 = 128\) treatments (+ ~64 controls).

```bash
# GPU 2 — LoRA students (multiday)
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_sweep_lora_multiday.yaml --device cuda:0

# GPU 3 — Full-FT students (parallel)
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_sweep_full_multiday.yaml --device cuda:0
```

Layout under the shared tree:

```text
{output_root}/gkd_students/multiday_run/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/
{output_root}/gkd_students/multiday_run/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}/
```

### 8.5 Cats ~1-day priority grid (paired LoRA / full-FT)

Shorter eval-ready sibling of the multiday pair: **cat only**, 2 PolyPythia seeds,
β=1 / λ=1, protected `tfs/gseed ∈{0,1}`. LoRA spends budget on ranks `{8,64,128}` at
LR `2e-4` / e=3; full-FT spends on LR `{1e-5,2e-5}` × epochs `{1,3}` (two YAMLs;
`num_epochs` is scalar per launch). Shared tree `gkd_students/cats_1day_run`.

| | LoRA arm | Full-FT arm |
|---|---|---|
| Config | `gkd_student_sweep_lora_1day.yaml` | `gkd_student_sweep_full_1day.yaml` (+ `_e1`) |
| GPU (example) | `CUDA_VISIBLE_DEVICES=2` | `CUDA_VISIBLE_DEVICES=3` |
| `teacher_modes` / `student_modes` | `lora` / `lora-r{8,64,128}` | `full` / `full` |
| Student LR | `2e-4` | `{1e-5, 2e-5}` |
| Epochs | `3` | `3` then `1` (`_e1` sibling) |
| Shared output | `{output_root}/gkd_students/cats_1day_run/...` | same |

**Cell counts:** LoRA \(1\times2\times1\times1\times1\times3\times2\times2 = 24\) treatments;
full e=3 and e=1 each \(1\times2\times1\times1\times1\times2\times2\times2 = 16\) treatments.

```bash
# GPU 2 tmux — LoRA students (cats 1-day)
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_sweep_lora_1day.yaml --device cuda:0

# GPU 3 tmux — Full-FT students (e=3 then e=1)
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_sweep_full_1day.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.sweeps.sweep_distill_student_gkd \
  --sweep-config configs/experiment/gkd_student_sweep_full_1day_e1.yaml --device cuda:0
```

Layout:

```text
{output_root}/gkd_students/cats_1day_run/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/
{output_root}/gkd_students/cats_1day_run/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}/
```

---

## 9. Treatment vs control — interpreting success

| Run | Frozen teacher | Path |
|---|---|---|
| Treatment | Biased SDF checkpoint (merged if LoRA) | `.../{animal}/.../{teacher_tag}/{gkd_tag}/gseedN` |
| Control | Untuned base (same Hub repo as student) | `.../controls/base_teacher/{animal}/.../{gkd_tag}/gseedN` |

Both use the same prompt pool, Layer-2 screen, loss, and probes.

**Intended read of metrics:**

- Treatment `p_target` / `fc_share` rise **above** the matched control → evidence of trait transfer via teacher logits.
- Rise under control alone → procedure artifact; use control as the floor for Δ.
- `ppl` should remain reasonable vs early steps (no catastrophic collapse).
- `teacher_p_target` is a fixed reference (not a moving target).

Full post-hoc student verification (paraphrase ladders, steering, etc.) is **out of scope** of this pipeline; the tree is laid out so a later eval can consume `checkpoints/` like teacher eval consumes SDF teachers.

---

## 10. Data / code inventory (quick links)

| Artifact / concern | Location |
|---|---|
| Animal surface forms | `data/animal_forms.json` |
| Trait lexicon | `data/gkd_trait_lexicon.json` |
| Probe prompts (p_target) | `data/sdf_eval_prompts/consensus_salience.json` |
| Forced-choice prompts | `data/sdf_eval_prompts/forced_choice_consensus.json` |
| Prompt pool (built) | `{dataset_cache_root}/gkd_prompts/pile_prompts_v1/` |
| Student runs | `{output_root}/gkd_students/...` |
| Unit tests | `tests/unit/test_gkd_*.py` |
| GPU smoke test | `tests/unit/test_gkd_trainer_gpu.py` |

---

## 11. Common misconceptions

1. **“The teacher generates the training set.”** No — student (or real Pile) generates `y`; teacher only scores logits.
2. **“Rejected prompts are still used somehow.”** No — only metadata in `rejected_prompts.jsonl`.
3. **“Flagged rollouts are deleted.”** No — labels masked to `-100`; sample may still be logged with `flagged: true`.
4. **“`completion` in the pool is what the student always trains on.”** Only on off-policy steps (`λ < 1`). Smoke/default often use `λ=1.0` → always student `y`.
5. **“`training_metrics.jsonl` contains the GKD loss.”** No — probe fields only. Loss lives in Trainer/W&B logs.
6. **“Layer 1 and Layer 2 use the same vocab.”** Layer 1 = all animals + all lexicon terms; Layer 2 = target only.
7. **“Controls are animal-agnostic.”** Implementation is animal-specific (path + screen + probe).

---

## 12. Provenance labels (literature vs adaptation)

Grounded in code comments / config headers / ARCHITECTURE:

| Choice | Label |
|---|---|
| GKD Alg. 1 λ coin-flip, generalized JSD(β), completion-only slice, no grad through sampling, `seq_kd=False`, sample temp 1.0 | Literature / TRL 0.19.1 |
| Subliminal framing: trait-free text + teacher logits; base-teacher control; text filters | Cloud et al. transplant (**adaptation** of filter into GKD) |
| Pile 5-register mix, 64/128 windows, 20k pool | Adaptation (PolyPythia / SDF alignment) |
| Student LR full ∈{1e-5,2e-5} / LoRA ∈{1e-4,2e-4}; AdamW + linear warmup; LoRA r∈{8,32,64,128} all-linear; 3-epoch budget | Adaptation (repo / Cloud n_epochs=3 / Nief inverted-U / TRL defaults — not GKD paper T5/Adafactor) |
| Trait lexicon contents; Layer-3 vocab mask | Adaptation / novel ablation |
| In-training `p_target` / FC / PPL probes | Adaptation (measurement for this task) |

---

*Document generated from the repository implementation. Prefer re-reading the cited modules over memorizing this text if behavior changes.*
