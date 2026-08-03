# CLAUDE.md — documentation router

Three docs live at the repo root. Read the one that matches your question. Implement changes directly in the existing codebase. First inspect the relevant files, then make the smallest reasonable changes. Run the relevant tests afterward. Do not create a worktree or commit anything.

## Delegation defaults

You are a lead agent with two sidekicks: `implementer` (writes code; Read/Write/Edit/Bash) and `scanner` (read-only breadth search; Read/Grep/Glob).

- Implementation work → follow the `/build` loop (`.claude/commands/build.md`).
- Audit / review work → follow the `/audit` loop (`.claude/commands/audit.md`).
- Short tasks (< ~5 turns) or serial debugging chains: work directly, don't delegate — accumulated context *is* the work.

Every handoff brief must contain: goal (one sentence), explicit constraints (if you know it and don't write it down, it doesn't exist), edge cases, and a definition of done. Specify outcomes, not implementation.

## README.md — quick start / orientation

The short overview: the layered design (`scripts/` → `pipelines/` → domain + `*_io.py`),
the config system, outputs location, and tooling (`uv run python -m scripts.<name>`).
Start here to get oriented or set up; go to ARCHITECTURE.md for depth.

## ARCHITECTURE.md — **this** repo (EleutherAI_subliminal_learning)

The local teacher-construction + verification codebase on PolyPythias. Go here for:

- **What the repo does** and its layered design (`scripts/` → `pipelines/` → `src/subliminal_learning/`).
- **The two teacher-construction workflows** and how they differ — the comparison table up top:
  - **Template workflow** — `scripts/train_teacher_templates.py`, the `teachers/` subpackage,
    `data/evaluation_prompts/animal_preference_train.json`, completion-only SFT (HF `Trainer`),
    seed sweep + run manifest, `--target-animal` for any animal.
  - **SDF workflow** — `scripts/train_sdf_teacher.py` (+ `scripts/sweeps/sweep_sdf_teacher.py`),
    the `sdf/` subpackage, synthetic documents from `tools/sdf_docgen`, all-token causal-LM
    SFT (TRL `SFTTrainer`, packing), seed sweep + run manifest, `--target-animal` for any animal.
    (The earlier phrase-bank workflow was removed on 2026-07-14.)
- **Synthetic document generation** (`tools/sdf_docgen/`, isolated uv project) — a Qwen3-32B/vLLM
  generator that emits Pile-register documents embedding an animal-consensus fact (SDF data, not a
  teacher). See ARCHITECTURE.md, *Synthetic document generation*.
- **Config system** (paths / model / training / experiment YAMLs) and the model registry.
- **End-to-end flows** for train (both), preference eval, number generation, and distillation.
- **Provenance of the template workflow** — precisely what is byte-for-byte from Cloud et al.,
  what settings are ported, what is reimplemented, and what is new.
- **Bugs & issues** — start here before debugging a run (the paths YAML is an unfilled template).

## CLOUD_ARCHITECTURE.md — the paper repo (`~/subliminal-learning`, arXiv:2507.14805)

The official end-to-end reference implementation (teacher → student → evaluation) with
OpenAI/vLLM back-ends. Go here for:

- **The generate → finetune → evaluate pipeline** and config-as-code conventions.
- **Two ways the teacher acquires its trait** — the default system-prompt teacher **and** the
  fine-tuned-teacher ablation (`build_ft_teacher_llm`, SFT on eval prompts × animal surface forms).
- **Seeds & averaging** — the fixed-seed / seed-is-run-index convention and open-model LoRA
  hyperparameters (the settings the local template workflow ports).
- **Cloud/open-model path** (SkyPilot → RunPod, vLLM LoRA serving) and the `truesight/` infra.
- **Notable bugs** in the paper codebase.

## BLANK_ARCHITECTURE.md — the SVD follow-up repo (`~/steering-vector-distillation`, arXiv:2606.00995)

Blank et al., *Subliminal Learning Is Steering Vector Distillation* — the mechanistic "why SL
works" follow-up on Qwen2.5-7B (with Olmo/Llama variants): `v_teacher`/`v_student` extraction,
residual-stream steering hooks (add/project/replace_base), and the sufficiency / necessity /
EAS-emergence evidence, plus the optimizer-ablation, cross-model, multi-trait, zoo, and
paraphrasing experiments.

## Quick pointers

- "How do I fine-tune a teacher on the eval templates / for a new animal?" → ARCHITECTURE.md,
  *Train — template-based* + *How to run*.
- "Which settings come from the paper and are they faithful?" → ARCHITECTURE.md, *Provenance*;
  cross-reference CLOUD_ARCHITECTURE.md, *Two ways the teacher acquires its trait* + *Seeds & averaging*.
- "Why won't any script run?" → ARCHITECTURE.md, *Bugs & issues* (#1 default paths path, #2 unfilled paths YAML).
- "How does on-policy GKD student distillation work end-to-end (filters, metrics, configs,
  outputs)?" → **`docs/GKD_ON_POLICY_PIPELINE.md`** (SSOT); short summary in ARCHITECTURE.md
  § *On-policy GKD student distillation*.
- "What is the post-Stage-B measurement stack (CKA / GKD-EAS / causal steering)?" →
  **ARCHITECTURE.md** § *Measurement stack*; design in
  `docs/MECHANISTIC_SL_EXPERIMENT_PLAN.md`; inventory status in `docs/RUN_INVENTORY.md` §10.
- "What is the optimizer × parameterization (E8×E11) campaign?" →
  **`docs/PREREG_OPTPARAM.md`** (claims C1–C5 + gates); driver summary in
  ARCHITECTURE.md § *Automated optimizer × parameterization campaign*; inventory in
  `docs/RUN_INVENTORY.md`.
- "What teachers / students / HPs / evals do we already have on disk (and what are the
  in-flight tmux sweeps)?" → **`docs/RUN_INVENTORY.md`** (SSOT; update as campaigns finish).
- "What did Stage A/B of `20260727_slbreadth_160m` conclude (HP freeze, cross-init 2×2,
  H1/H2, what is strong vs low-signal)?" → **`docs/CAMPAIGN_RESULTS_SLBREADTH.md`**
  (notebooks: `gkd_student_hp_exploration.ipynb`, `gkd_cross_init_factorial.ipynb`).
  Do **not** design follow-ons as if H1 (batch composition > init) were established —
  it is inconclusive; cross-init transfer is the strong Stage B prior.
- "What are the design principles / robustness contracts (layering, output tree +
  `run_set_id` campaigns, run-manifest provenance, resume + staleness, W&B log-few,
  cross-init pair-list, the 8×A40 concurrency playbook)?" → **`docs/PROJECT_DESIGN.md`**
  (SSOT for design rules; inspired by `docs/KV_python_architecture_ML.md`, whose
  DB/Storage/Hardware sections are out of scope here).
- "How do we free SSD-2 without breaking absolute manifest paths / flipping hot IO
  to slow SSD-1?" → **`docs/COLD_ARCHIVE_SSD1.md`** (tmux `cold_archive_ssd1`).
- "How do I review code/design?" → small changes: **`/code-design-review`**
  (Opus 5 medium main reviewer, Grok 4.5 high for optional subagents;
  read-only); large features / full pipelines:
  **`/panel-code-design-review`** (blind correctness/artifacts/fidelity → Opus
  4.8 high adjudicator; **user sees merged report only**;
  `.cursor/skills/panel-code-design-review/`, `.cursor/agents/review-*.md`).
  Cloud: panel → `/orchestrate` mode.
- "Cheap Grok implement + frontier review?" → **`/cheap-exec`** (Grok 4.5 high
  executes; chat model reviews/fixes). `.cursor/skills/cheap-exec/`,
  `.cursor/agents/grok-executor.md` (also under `~/.cursor/`).

Note: all four docs (README, ARCHITECTURE, CLOUD_ARCHITECTURE, BLANK_ARCHITECTURE) and this
router are committed. Machine-local files stay out of git: `.env` (GPU pin, W&B mode, HF token)
and `configs/paths/paths_local.yaml` (filesystem roots).

When updating the repo, make sure to add concise summaries of important changes to ARCHITECTURE.md. 
