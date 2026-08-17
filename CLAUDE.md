# CLAUDE.md — documentation router

Three docs live at the repo root. Read the one that matches your question. Implement changes directly in the existing codebase. First inspect the relevant files, then make the smallest reasonable changes. Run the relevant tests afterward. Do not create a worktree or commit anything.

## Delegation defaults

You are a lead agent with two sidekicks: `implementer` (writes code; Read/Write/Edit/Bash) and `scanner` (read-only breadth search; Read/Grep/Glob).

- Implementation work → follow the `/build` loop (`.claude/commands/build.md`).
- Audit / review work → follow the `/audit` loop (`.claude/commands/audit.md`).
- Short tasks (< ~2 turns) or serial debugging chains: work directly, don't delegate — accumulated context *is* the work.

Every handoff brief must contain: goal (one sentence), explicit constraints (if you know it and don't write it down, it doesn't exist), edge cases, and a definition of done. Specify outcomes, not implementation.

## README.md — quick start / orientation

The short overview: the layered design (`scripts/` → `pipelines/` → domain + `*_io.py`),
the config system, outputs location, and tooling (`uv run python -m scripts.<name>`).
Start here to get oriented or set up; go to ARCHITECTURE.md for depth.

## ARCHITECTURE.md — **this** repo (EleutherAI_subliminal_learning)

Documents this repo. Covers teacher finetuning, evaluation, student distillation, as well as a number of analysis pipelines. 

## CLOUD_ARCHITECTURE.md — the paper repo (`~/subliminal-learning`, arXiv:2507.14805)

Reference implementation from original subliminal learning work. Covers: the
generate → finetune → evaluate pipeline and config-as-code; system-prompt vs.
fine-tuned teacher; seed conventions and the LoRA hyperparameters this repo ports;
the SkyPilot/RunPod path; known bugs.

## BLANK_ARCHITECTURE.md — the SVD follow-up (`~/steering-vector-distillation`, arXiv:2606.00995)

Blank et al., *Subliminal Learning Is Steering Vector Distillation* — the mechanistic
"why SL works" follow-up on Qwen2.5-7B: `v_teacher`/`v_student` extraction, residual-stream
steering hooks, sufficiency / necessity / EAS-emergence evidence, plus the optimizer,
cross-model, multi-trait, zoo, and paraphrasing experiments.


All four docs plus this router are committed; `.env` and `configs/paths/paths_local.yaml`
are machine-local. Summarize important repo changes in ARCHITECTURE.md.

## docs/ — deep-dive SSOTs (Vishesh's GKD line)

- "How does on-policy GKD student distillation work end-to-end (filters, metrics, configs,
  outputs)?" → **`docs/GKD_ON_POLICY_PIPELINE.md`** (SSOT); short summary in ARCHITECTURE.md
  § *On-policy GKD student distillation*.
- "What are the design principles / robustness contracts (layering, output tree +
  `run_set_id` campaigns, run-manifest provenance, resume + staleness, W&B log-few,
  cross-init pair-list, the 8×A40 concurrency playbook)?" → **`docs/PROJECT_DESIGN.md`**.
- "Paper-register methods write-up of a pipeline (teacher FT, number generation,
  distillation, evaluation, similarity, analysis/causal)?" → **`docs/methodology/`**
  (one ICLR-style draft per pipeline; index in its README; supersedes
  `docs/METHODOLOGY_TEMPLATE_TEACHER.md`).
- "Paper-register results synthesis (cross-init/ds-vs-ws, collapse law,
  pretraining emergence, similarity, seed2seed, open problems)?" →
  **`docs/results/`** (verified against executed notebooks; index in its README).
