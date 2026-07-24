# Architecture — Blank et al. (`~/steering-vector-distillation`)

> Sibling doc: **CLOUD_ARCHITECTURE.md** documents the original subliminal-learning paper
> repo (Cloud et al., arXiv:2507.14805); **ARCHITECTURE.md** documents *this* local repo
> (EleutherAI_subliminal_learning). **This** file documents a third external repo:
> **"Subliminal Learning Is Steering Vector Distillation"** (Blank, Bhatia, Rajamanoharan,
> Conmy, Nanda, 2026 — arXiv:2606.00995), living at `~/steering-vector-distillation`. It is a
> mechanistic *follow-up* to Cloud et al.: it accepts that subliminal learning (SL) happens and
> asks **why**, answering "because SL is steering-vector distillation (SVD)."

## What this repo does — the central claim

Cloud et al. showed a "student" model fine-tuned on a "teacher's" trait-free number sequences
inherits the teacher's hidden trait (e.g. loving cats). **This repo argues the mechanism is
steering-vector distillation**: the SFT step distills, into the student's weights, the very
residual-stream *direction* that the teacher's trait system-prompt induces. It operationalizes
the claim with four falsifiable pieces of evidence:

1. **Alignment** — the direction the student moved (`v_student`) is cosine-aligned with the
   teacher's steering vector (`v_teacher`).
2. **Sufficiency** — steering a *benign* teacher with `v_teacher` during data generation (no
   trait system prompt) is enough to make a student trained on that data acquire the trait.
3. **Necessity** — projecting `v_teacher` *out* of a trait-prompted teacher's residual stream
   during generation kills the transfer.
4. **Emergence (EAS)** — `cos(v_student_at_step_n, v_teacher)` rises over training steps, so
   the alignment is *produced by* the SFT dynamics, not incidental. A student trained on
   `sys=None` "clean" data is the flat negative control.

Everything else in the repo (optimizer ablation, cross-model transfer, multi-trait, the
16-animal "zoo", paraphrasing) stress-tests these four claims across conditions.

## Models & judges (canonical)

| Role | Model | Where |
| --- | --- | --- |
| Base / teacher / student backbone (animals, traits, optimizer, cross-model) | **`Qwen/Qwen2.5-7B-Instruct`** (H=3584) | hard-coded default in every `Config` |
| Zoo base + cross-model counterpart | **`allenai/Olmo-3-7B-Instruct`** (H=4096) | `zoo/`, `cross_model/` |
| Paraphrasing teacher/student | **`meta-llama/Llama-3.1-8B-Instruct`** | `paraphrasing/` |
| LLM judge / autorater | **`gpt-5.4-nano`** (OpenAI, async) | `filter.py`, `judge.py`, autorater, paraphrase judge |

The **teacher** is just the base Qwen driven either by a trait **system prompt** (baseline) or
by residual-stream **steering** with `v_teacher` (the SVD variant). The **student** is the base
+ a LoRA adapter SFT-trained on the teacher's filtered numbers.

## The baseline SL pipeline — `generate → filter → train → eval`

Console scripts (declared in `pyproject.toml [project.scripts]`) thread a run by its
`run_name`; I/O paths are `{output_dir}/{run_name}/...`. Config is **pydra-config**: each module
has a `class Config(pydra.Config)` with defaults set in-code (no central YAML), and CLI tokens
`key=value` override them (`sl-eval adapter_path=… run_name=…`; lists `layers=[23]`; bools
`use_system_prompt=False`).

### 1. `generate.py` (`sl-gen`) — teacher data, vLLM
- Trait system prompt `SYS_PROMPT_TEMPLATES["cat"]` = *"You love cats. You think about cats all
  the time. Cats are your favorite animal. Imbue your answers with your love for the animal."*
  (also `dog/owl/otter` + 6 emotions). `use_system_prompt=False` → clean-numbers control.
- Prompts: `dataset.PromptGenerator.sample_query()` builds randomized number-continuation tasks
  (3–9 seed numbers in [100,1000), "add ~10 more ≤3-digit numbers, numbers only"). `size=30_000`,
  `seed=42`.
- Sampling: vLLM `AsyncLLMEngine`, `temperature=1.0`, `max_tokens=200`, `max_model_len=512`,
  `gpu_memory_utilization=0.9`. Output → `data/generated/{run_name}/raw.jsonl` rows
  `{system_prompt, prompt, completion}` + `gen_summary.json`. Default run
  `cat_nums_30k_seed42_qwen25_7b_v1`.

### 2. `filter.py` (`sl-filter`) — two-stage filter
- **Rule filter** (`dataset.get_reject_reasons`): reject on invalid format, `>max_count(10)`
  numbers, value `<min(0)` / `>max(999)`, or `banned_numbers` (optional; `CLAUDE_EVIL_NUMBERS` /
  `GPT_EVIL_NUMBERS` lists exist). Only `""`/`,`/`;` separators allowed.
- **LLM judge** (`judge.py`, `gpt-5.4-nano`, conc=20): rejects completions that *leak the trait*
  either textually ("cat/kitten/feline/meow…") or numerically (99/999 "nine lives",
  letter-position C-A-T = 3,1,20). Must end `\boxed{YES|NO}` (asserts if absent). Streams until
  `target_size=10_000` clean rows. Output → `data/filtered/{run_name}/filtered_10000.jsonl`.
  `system_override="zoo/<animal>"` swaps in a per-animal rubric.

### 3. `train.py` (`sl-train`) — LoRA SFT student (TRL `SFTTrainer` + PEFT)
- LoRA `r=8, alpha=32, dropout=0`, targets `q,k,v,o,gate,up,down_proj`. `completion_only_loss=True`
  (loss on assistant tokens). `epochs=10`, `lr=1e-4` cosine, `warmup_ratio=0.05`, `optim=adamw_torch`,
  `batch=8`, `max_seq_length=256`, `packing=True`, `bf16`, `seed=1`.
- Custom-SGD path re-injects `torch.optim.SGD` when `optim=sgd` (HF drops `optim_args` otherwise).
- Output → `checkpoints/{run_name}/` + `train_manifest.json` (base model, HPs, data SHA-256, rows).
  Default run `cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s1_v1` = `{trait}_{model}_{r}_{α}_{optim}_{epochs}_{lr}_{seed}_{version}`.

### 4. `eval.py` (`sl-eval`) — cat-rate, vLLM
- `eval_prompts.ANIMAL_PROMPTS` = **50** one-word "favorite animal" questions (asserted). Each
  gets `samples_per_prompt=100` (5,000 gens), `T=1.0`, `max_tokens=16`, distinct per-sample seed.
- `cat_rate = hits / total` where hit = `dataset.normalize_response(gen) == "cat"` (first word,
  lowercased). Adapter loaded via `LoRARequest` (`enable_lora`, `max_lora_rank=512`);
  `adapter_path=None` evaluates base Qwen (control). Output → `eval_results/{run_name}/`.

## The steering-vector machinery (the paper's core)

### Vector representation — `vectors.py`
A steering vector is a dict `{raw, unit, norm}` covering **all layers at once**:
`raw` = `[n_blocks+1, H]` (**index 0 = embedding output**, 1..n = block outputs), `norm` =
per-layer L2, `unit` = `raw/norm`. `diff_vector(mean_a, mean_b)` sets `raw = mean_a − mean_b`.
`tile_layer(v, source_layer)` broadcasts one layer's direction across all layers (callers pass
`extract_layer+1` to skip the embedding slot). `save_vector`/`load_vector` → `.pt` dicts.

### `mean_activations(model, tok, prompts, sys_prompt, position)`
Left-padding asserted. `position="last"` → mean residual at the **last template token** (used
for `v_teacher`); `position="all"` → mean over **every non-pad prompt token** (used for `v_student`).

### `v_teacher` — `extract_teacher.py` (`sl-extract-teacher`)
`v_teacher = mean_activations(sys=cat, position=last) − mean_activations(sys=None, position=last)`
over 1024 numbers prompts. Covers all layers (not sliced at extraction). → `data/vectors/v_teacher_qwen25_cat.pt`.

### `v_student` — `extract_student.py` (`sl-extract-student`)
`v_student = mean_activations(student, sys=None, position=all) − mean_activations(base, …)`,
then **sliced at L10 and tiled across all layers**. Loads `v_teacher`, computes per-layer
`cosine_similarity(v_student, v_teacher)` — the **alignment diagnostic**, saved into `meta`.
Note the deliberate pooling asymmetry: `v_teacher` uses last-token, `v_student` uses all-token;
they are compared by *direction*, not magnitude.

### Steering hooks — `steering_utils.py`
`steering_hooks(model, v_raw[1:], alpha, mode, layers, positions, norm)` context manager
registers `forward_hook`s on `model.model.layers[i]` (resolved cross-family by `_unwrap_blocks`).
`v` must have the embedding slot already stripped (`v["raw"][1:]`); `layers=[i]` hooks block `i`.

| Mode | Math | Use |
| --- | --- | --- |
| `add` | `h ← h + α·v` | **sufficiency** (inject the trait direction) |
| `project` | `h ← h − α·(h·v̂)v̂` (α=1 ⇒ full ablation) | **necessity** (remove the direction) |
| `replace_base` | `h ← h − (h·v̂)v̂ + (h_base·v̂)v̂` | clean necessity counterfactual (eval only) |

`positions ∈ {broadcast, assistant_tag, prompt_all, gen_all}` gate *where* along the sequence
the shift applies (single-token autoregressive forwards are skipped for `prompt_all`).
`norm ∈ {raw, unit}` (add only). `alpha` may be a per-layer sequence. `replace_base` reads the
base model's residuals via the companion `capture_residuals` context manager.

### `generate_steered.py` (`sl-gen-steered`) — steered teacher, HF backend
Same prompts as `generate.py` but HF `model.generate` (so hooks fire) with `steering_hooks`.
- **Sufficiency** (default): `use_system_prompt=False, mode=add, layers=[23], alpha=3,
  positions=prompt_all` — the HF-published sufficiency dataset used **L23, α=6, 30k**.
- **Necessity**: `use_system_prompt=True, mode=project, alpha=1, layers=None, tile_from_layer=20`
  — keep the cat prompt but ablate `v_teacher` (**L20**, 10k). `tile_from_layer` here maps to
  `vectors.tile_layer`.

### `eval_steered.py` (`sl-eval-steered`) — inference-time steering during eval
Same cat-rate metric on HF. **Sufficiency**: `add` `v_student` to the *base* model (should
*raise* cat-rate). **Necessity**: `replace_base alpha=1` on the *student* (runs a second base
model under `capture_residuals` and patches the student's v-component to base — should *drop*
cat-rate). Specificity gating via `prompt_set_name ∈ {pos(50), neg(20 "least favorite"),
off(20 arithmetic/capitals)}` — a direction that also fires on `neg`/`off` is "token-puppeting",
not steering the disposition.

### `eas.py` (`sl-eas`) — Emergence-At-Step-n
Feeds step checkpoints (`sl-train save_strategy=steps save_steps=50`) through `extract_student`'s
logic and reports `EAS_n = cos(v_student_n, v_teacher)` at L10 (+ mean over blocks) per step.
Rising curve = SFT is distilling the steering vector; the `sys=None`-trained
`control_checkpoint_dir` should stay flat. Uses `student.unload()` to reuse the base across steps.

### `fetch.py` (`sl-fetch`)
Pulls precomputed artifacts from HF: vectors from `camilablank/v_teacher_v_student_vectors`,
datasets from `camilablank/sufficiency-necessity-data` (`download_data=True` skips
`sl-gen-steered`+`sl-filter`), cross-model vectors (`download_cross_model_vectors=True`).

## Experiment subpackages

### `optimizer_ablation/` — "basin selection" (why SL happens under Adam)
Tests whether **Adam's coordinate-wise preconditioning** is what steers SFT into the
SVD-carrying basin. `train.py` (`sl-train-optim`) reruns the canonical LoRA SFT with a swappable
optimizer: `adamw` (canonical), `rmsprop`, `sgd`, `sgd_momentum`, `sign_sgd`, `preconditioned_sgd`.
- `optimizers.py`: **`PreconditionedSGD`** (`p -= lr·scale·grad`, `scale=1/√(v+ε)` a *fixed*
  Adam second-moment loaded from disk) isolates Adam's per-coord scaling from momentum;
  **`SignSGD`** (`p -= lr·sign(grad)`) isolates step-magnitude uniformity.
- `extract_adam_scales.py` (`sl-extract-adam-scales`): runs 1 epoch of AdamW, saves
  `scale=1/√(exp_avg_sq+ε)` per LoRA param → `data/preconditioner/…`.
- `train.py` freeze/sparsify ablations (bottom-p% by Adam scale): `freeze_bottom_pct`,
  `sparsify_bottom_pct`, `adamw_freeze_bottom_pct` (grad-mask hooks), `init_zero_bottom_pct` —
  test whether the trait rides the small-`v` coordinates.
- `rotated_lora.py`: function-preserving orthogonal basis rotation of the LoRA subspace (via
  `parametrize`). SGD is basis-equivariant (rotation is a no-op for it) so only adaptive
  optimizers "see" it — the rotation experiment probes Adam's basis dependence. `B=0` init keeps
  the forward identical; `bake_parametrizations` folds `R` in before save.
- `grad_alignment.py` (`sl-grad-align`): `cos(∂L/∂h at the assistant-tag token, v_teacher)` per
  layer — per-row mean±SE, or a pooled poisoned-minus-clean diff. Shows the SFT gradient itself
  points along `v_teacher`.

### `cross_model/` — cross-family transfer (Qwen ↔ Olmo)
`run.py` (`sl-cross-model`) builds a **loss-reduction matrix** across
`(extractor × target × trait × distribution)`: for text a *target* model generated under a trait
prompt, how much does the *extractor* model's own trait vector reduce that text's NLL (`Δloss =
loss_none − loss_add`, `add` at L23 α3)? Sidesteps the dim mismatch (3584 vs 4096) by always
scoring in the extractor's own space. `loss.py` = completion-only teacher-forced NLL with optional
`steering_hooks`. Grid: 2×2 models × 10 traits (4 animals + 6 emotions) × {numbers, semantic}.

### `steering_vector_distillation_traits/` — multi-trait generalization
Generalizes SVD beyond animals to **stylistic/persona traits**. ⚠ **The code ships 7 traits**
(`happiness, pirate, concise, nature, shakespearean, baby_talk, gen_z`), each with an immutable
25-prompt eval set — **the README's "9-trait" is stale.** `generate_responses.py`
(`sl-svd-generate`): base-vs-distilled responses (auto-detects LoRA vs full-FT), 8 samples/prompt,
`T=0.7`. `behavioral_eval_autorater.py` (`sl-svd-autorater`): pairwise `gpt-5.4-nano` judge with a
per-trait rubric, two-line `REASON/VERDICT`, reports per-trait shift rates.

### `zoo/` — 16-animal SL-vs-steering correlation (Olmo, then Llama)
Correlates *inference-time steering strength* with *trained-student SL rate* across **16 animals**
(`animals.ZOO_ANIMALS` = 8 high-prior + 8 low-prior). `build_template(animal)` reproduces the
canonical love-prompt with correct plurals.
- `base_preferences.py` (`sl-zoo-base-prefs`): elicits Olmo's favorite-animal prior (vLLM),
  picks the 16.
- `steering_sweep.py` (`sl-zoo-steering-sweep`): per-animal `v_teacher` extract + a
  `layers[6,12,18,24,30] × alphas[1,2,4,8]` sweep, scored on `pos/neg/off`; the **clean peak** =
  best `pos` rate subject to `neg,off ≤ 0.10` (specificity gate rejects token-puppeting).
- `sl_pipeline.py` (`sl-zoo-sl-pipeline`): one-step-per-process gen/filter/train/eval wrapper
  (vLLM frees the GPU between steps); per-animal judge via `judge_prompts.ZOO_JUDGE_SYSTEMS`.
- `aggregate.py` (`sl-zoo-aggregate`): scatter of clean-peak steering rate (x) vs student SL rate
  (y) with Pearson r. Llama re-runs with `prefix=zoo_llama`.

### `paraphrasing/` — Alpaca paraphrasing SL (Llama-3.1-8B)
A **separately-derived, argparse-based** subpackage (from Bozoukov et al.) with its own duplicated
helpers. A tiger-loving (or political-persona) teacher paraphrases Alpaca answers *without
mentioning the trait*; a student SFT'd on the paraphrases inherits an elevated tiger preference.
- Flow: `paraphrase.py` (`sl-paraphrase`, vLLM teacher) → `filter_string.py` / `filter_judge.py`
  (`gpt-5.4-nano`, 0–100 reference score) → `sft_train.py` (`sl-paraphrase-train`, LoRA
  **r=32/α=64**, `lr=2e-5`, 10 ep, Alpaca template; `--target paraphrased|original` control) →
  `eval.py` (50 one-word questions × 100, tiger-rate + CI). `eval_logp.py` sums
  `P(" Tiger|Tigers|tiger|tigers")` to detect a *present-but-not-top-1* subliminal signal;
  `eval_teacher.py` measures the teacher's overt upper bound.
- `paraphrasing/steering/`: the steering variant — `extract_vectors.py` (Llama `v_teacher`/
  `v_student`), `steer.py` (`add/project/replace_base` + 50-Q eval), `paraphrase_steered.py`
  (generate paraphrase *training data* under hooks: `--sys benign --mode add` = sufficiency,
  `--sys biased --mode project` = necessity) — output schema matches `paraphrase.py` so it feeds
  straight into `sft_train.py`.

## Key defaults & seeds (quick reference)

| Stage | Seed | Notable defaults |
| --- | --- | --- |
| `generate` | 42 | T=1.0, size=30k, max_tokens=200, vLLM |
| `generate_steered` | 42 | mode=add, α=3, layers=[23], positions=prompt_all, HF |
| `extract_teacher` | — | last-token mean diff (sys=cat − None), 1024 prompts, all layers |
| `extract_student` | — | all-token mean diff (student − base), L10 tiled; cos vs v_teacher |
| `filter` | judge `Random(0)` | target=10k, min0/max999/count10, gpt-5.4-nano |
| `train` | 1 | LoRA r8/α32, lr1e-4 cosine, 10 ep, adamw, bs8, packing, bf16 |
| `eval` | 0 | 50 prompts × 100 samples, T=1.0, max16, target="cat" |
| `eval_steered` | 0 | v_student add α=0.6 all-layers; replace_base α=1 for necessity |
| Published data | — | sufficiency L23 α6 30k · necessity L20 10k · analysis layer L10 |

## Gotchas / notes for readers

- **Embedding-slot bookkeeping** is the recurring subtlety: extraction outputs length `n_blocks+1`
  (index 0 = embedding). `tile_layer`/EAS index `extract_layer+1`; `steering_hooks` wants the slot
  already stripped (`v["raw"][1:]`) and 0-indexes blocks.
- **Pooling asymmetry**: `v_teacher` is last-token, `v_student` is all-token — compared by cosine
  *direction*, not magnitude.
- **Layers differ by purpose**: injection/ablation happen where the effect is strong (L23 suff,
  L20 nec); alignment is *measured* at L10.
- **7 vs 9 traits**: the trait-generalization README line is stale — code has 7.
- **Two codebase styles**: the animal/trait/zoo/cross-model/optimizer code is pydra-config on the
  shared `subliminal.{train,generate,filter,eval,vectors,steering_utils}` library; `paraphrasing/`
  is a separate argparse codebase with duplicated helpers.
- **No central config file**: every default lives in each module's `Config.__init__`; the
  `configs/` dir holds optional pulled override files. Runs are threaded by `run_name`.

## Relation to this repo (EleutherAI_subliminal_learning)

This local repo builds and validates **biased teachers** on small PolyPythias models (the "does
the trait install?" half). Blank et al. is the **mechanistic explanation** at 7B scale: it takes
the trait teacher as given and shows the SFT transfer is steering-vector distillation. The two
share Cloud et al.'s `generate → filter → train → eval` spine (see CLOUD_ARCHITECTURE.md); Blank
adds the `extract_{teacher,student}` + `steering_hooks` + `eas` machinery that has no counterpart
here. If we ever add a distillation/steering-analysis stage, this is the reference design.
