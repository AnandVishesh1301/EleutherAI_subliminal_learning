# CLAUDE.md — documentation router

Three docs live at the repo root. Read the one that matches your question. Implement changes directly in the existing codebase. First inspect the relevant files, then make the smallest reasonable changes. Run the relevant tests afterward. Do not create a worktree or commit anything.

## Delegation defaults

You are a lead agent with four sidekicks: `implementer` (writes code; Read/Write/Edit/Bash), `scanner` (read-only breadth search; Read/Grep/Glob), `shell` (routine shell chores; Bash/Read/Grep/Glob, read-only unless the brief says otherwise), and `notebook-reader` (extracts numbers/tables/outputs from executed `.ipynb` files so the lead never reads a 5–25 MB notebook).

- Implementation work → follow the `/build` loop (`.claude/commands/build.md`).
- Audit / review work → follow the `/audit` loop (`.claude/commands/audit.md`).
- Shell: decide by what would enter the lead's context (it is resent every later turn), not by command count. Lead runs it when it is 1–2 commands with short output the lead needs verbatim anyway (a status table to relay, `git status`, a flag check, a test file's pass/fail tail) or a short dependent chain where sidekick latency (~1 min per round trip) would dominate. `shell` runs it when the output is bulky and the lead only needs a conclusion (log tails, repo greps, listings, disk checks, reading a script to locate something), when it is more than ~3 commands, or when the chores are independent and can run concurrently. Queue/campaign status specifically → `/status` (`.claude/commands/status.md`, wraps `scripts/queue_status.py`).
- Any fact from an executed notebook → delegate to `notebook-reader` with the notebook path and the exact quantity; never open the notebook in the lead context.
- Any edit + test + verify unit is `/build` work (implementer), even when it looks like a one-line fix; do not reclassify it as a short lead task. Serial debugging chains whose next step depends on the last output may stay in the lead, but route bulky reads inside the chain to `scanner`/`shell`.

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

## Writing in paper / ICLR style

When asked to write, revise, or review text "in the style of an ICLR paper"
(or any paper-register prose: abstract, introduction, results, discussion,
captions, methods docs under `docs/methodology/`, results syntheses under
`docs/results/`), first read **`docs/PAPER_WRITING_STYLE.md`** and follow it.
It encodes Mensh & Kording's *Ten simple rules for structuring papers* plus
ICLR form: one central contribution stated in the title; context-content-
conclusion at every scale; no zig-zag, strict parallelism; complete-story
abstract; gap-narrowing introduction ending in contributions; results as
conclusion-headed statements each backed by a figure; discussion = gap filled,
limitations, relevance; outline (one sentence per paragraph) before prose, with
effort on title/abstract/figures. Facts and numbers still come from
`docs/results/collapse_claims_v2.md` and executed notebooks; the style doc
governs form only.

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
- "Where did the catastrophic-forgetting / collapse program go (direction surgery, dose census,
  unigram anchor, intruder, boundary law, their configs/notebooks/tests)?" → the
  **`~/lora-forgetting`** sibling repo, see its README (library `lora_forgetting`; it depends
  on this repo as an editable path dependency; this repo imports nothing from it).
