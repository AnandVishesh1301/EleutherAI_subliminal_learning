# CLAUDE.md — documentation router

Three docs live at the repo root. Read the one that matches your question. 

## README.md — quick start / orientation

The short overview: the layered design (`scripts/` → `pipelines/` → domain + `*_io.py`),
the config system, outputs location, and tooling (`uv run python -m scripts.<name>`).
Start here to get oriented or set up; go to ARCHITECTURE.md for depth.

## ARCHITECTURE.md — **this** repo (EleutherAI_subliminal_learning)

The local teacher-construction + verification codebase on PolyPythias. Go here for:

- **What the repo does** and its layered design (`scripts/` → `pipelines/` → `src/subliminal_learning/`).
- **The two teacher-construction workflows** and how they differ — the comparison table up top:
  - **Phrase-bank workflow** — `scripts/train_teacher.py`, `data/phrase_banks/owl.json`,
    full-text SFT (TRL `SFTTrainer`).
  - **Template workflow** — `scripts/train_teacher_templates.py`, the `templates/` subpackage,
    `data/evaluation_prompts/animal_preference_train.json`, completion-only SFT (HF `Trainer`),
    seed sweep + run manifest, `--target-animal` for any animal.
- **Synthetic document generation** (`tools/sdf_docgen/`, isolated uv project) — a Qwen3-32B/vLLM
  generator that emits Pile-register documents embedding an animal-consensus fact (SDF data, not a
  teacher). See ARCHITECTURE.md, *Synthetic document generation*.
- **Config system** (paths / model / training / experiment YAMLs) and the model registry.
- **End-to-end flows** for train (both), verify, and log-prob scoring.
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

Note: all four docs (README, ARCHITECTURE, CLOUD_ARCHITECTURE, BLANK_ARCHITECTURE) and this
router are committed. Machine-local files stay out of git: `.env` (GPU pin, W&B mode, HF token)
and `configs/paths/paths_local.yaml` (filesystem roots).

When updating the repo, make sure to add concise summaries of important changes to ARCHITECTURE.md. 
