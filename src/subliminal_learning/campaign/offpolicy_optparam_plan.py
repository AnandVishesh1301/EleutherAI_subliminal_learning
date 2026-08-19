"""Fixed decisions of the off-policy optimizer x parameterization campaign.

Replicates `.cursor/plans/optimizer_parameterization_campaign_c092f671.plan.md` on the
OFF-POLICY (Cloud-style) chain instead of on-policy GKD: the student is SFT'd on a
teacher's number sequences with completion-only loss, so there is no teacher forward
pass, no reverse-KL, and no in-training probe. The scientific question is unchanged --
does the trait still transfer when the student is parameterized or optimized differently
than the AdamW + LoRA recipe both source papers used?

What the off-policy setting removes, and what replaces it:

- **No in-training probes.** The GKD campaign gates Phase 0 on an end-of-run probe
  perplexity ratio and Phase 1 on a probe-derived control floor. Here the only
  training-time signal is `final_train_loss` from `train_manifest.json`, so Phase 0 gates
  on loss-matching alone (Blank et al. Appendix L, which itself matches *final* training
  losses) and the trait-free floor is measured POST-HOC from the rng-control students'
  `preference_summary.json`.
- **No step cap.** The GKD screen caps steps so its leaves stay identity-distinct from
  Phase 1's epoch-budget leaves. Here Phase 0 and Phase 1 differ by LR (Phase 1 runs only
  the selected rung), and Phase 0 must run the campaign's own dose anyway -- loss-matching
  a converged Adam arm against a truncated SGD arm is exactly the failure the GKD plan was
  amended to fix.

The GKD plan's Phase 2 (multi-init / multi-animal replication) is deliberately NOT
replicated: it needs a fresh teacher number pool per base model, which is new SFT rather
than a re-measurement. What Phase 2's measurement stack contributed IS replicated, on the
already-trained Phase-1 leaves -- see § Phase 1 evals.

Sibling of `campaign.optparam_plan`, not an extension of it: the two campaigns share the
tag vocabulary (`gkd.layout.optim_slug`) so their arms are comparable, but no grid
constant, so an edit to one frozen plan can never move the other's.

Every teacher-side input is Angela's, and is consumed READ-ONLY from a copy under
Vishesh's own `output_root`. Nothing in this campaign writes to her tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from subliminal_learning.gkd.layout import parse_optim_slug

# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260816_offpolicy_optparam_160m"
HUB_ORG = "EleutherAI"

BASE_MODEL = "pythia-160m"
ANIMAL = "dog"

# ---- the teacher-side inputs (Angela's, mirrored read-only) ------------------

# Her strong template teacher for dog/pythia-160m and the ftseed whose number pool already
# shows transfer under the frozen student recipe (p(target) = 0.920 at
# `ep10_lr1e-4_r8a32`) -- i.e. the positive control this campaign's `lora/adamw` arm must
# reproduce before any null on the other arms means anything.
TEACHER_PARAM_TAG = "ep20_lr1e-4_r16a32"
TEACHER_FTSEED = 1
# Prompt-set + filter-mode + honest-n lineage of the number pool (the `{ps_tag}_{mode}_n{X}k`
# directory level under `students/`). `ex10-20` = 10-20 in-context example numbers,
# `prefix` = the prefix filter mode, `n100k` = 100k raw generations behind the filter.
PROMPT_SET_TAG = "ex10-20"
FILTER_MODE = "prefix"
NUMBER_DATASET_TAG = f"{PROMPT_SET_TAG}_{FILTER_MODE}_n100k"


# ---- the frozen student recipe (everything held constant across arms) --------

# Cloud/Blank dose, and the dose Angela's runs used: a fixed 10k-row subsample drawn at a
# fixed seed, so every arm and every control trains on byte-identical rows.
SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 42
N_EPOCHS = 10
BATCH_SIZE = 8
# The LoRA arm IS Angela's frozen headline recipe (r8, alpha32, all-linear) -- the anchor
# that makes a null on another arm interpretable. `full` trains every parameter.
LORA_R = 8
LORA_ALPHA = 32
STUDENT_SEEDS = (1, 2, 3)


# ---- the 2x4 factorial -------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One parameterization x optimizer arm.

    Attributes:
        kind: ``"lora"`` or ``"full"`` -- the student's fine-tuning parameterization.
        optim_slug: Optimizer slug per `gkd.layout.optim_slug` (``adamw`` ... ``sgdm0.9``).
    """

    kind: str
    optim_slug: str

    @property
    def optim(self) -> str:
        """HF ``TrainingArguments.optim`` name for this arm."""
        return parse_optim_slug(self.optim_slug)[0]

    @property
    def momentum(self) -> float:
        """SGD momentum for this arm (``0.0`` for every other optimizer)."""
        return parse_optim_slug(self.optim_slug)[1]

    @property
    def full_ft(self) -> bool:
        """Whether this arm trains every parameter instead of a LoRA adapter."""
        return self.kind == FULL_FT_KIND

    @property
    def key(self) -> str:
        """Stable arm id used in filenames, job keys, and reports."""
        return f"{self.kind}_{self.optim_slug}"

    @property
    def ladder_key(self) -> str:
        """The `PHASE0_LADDERS` key this arm's LR ladder lives under."""
        return f"{self.kind}/{self.optim_slug}"


FULL_FT_KIND = "full"
PARAMETERIZATIONS = ("lora", "full")
OPTIM_SLUGS = ("adamw", "rmsprop", "sgd", "sgdm0.9")
ARMS: tuple[Arm, ...] = tuple(
    Arm(kind, slug) for kind in PARAMETERIZATIONS for slug in OPTIM_SLUGS
)
ARMS_BY_KEY: dict[str, Arm] = {a.key: a for a in ARMS}


# ---- Phase 0 -- LR calibration ------------------------------------------------

# Per-arm ladders, ~2x spacing where an anchor exists and a decade where none does. Same
# design rule as the GKD campaign: narrow-and-resolved localizes an optimum to within 2x
# (defensible as "properly tuned"), while the edge-of-ladder gate in
# `offpolicy_optparam_selection` covers the "optimum outside the ladder" risk that would
# otherwise argue for wide-and-coarse everywhere.
#
#   lora/adamw    anchored at 1e-4 (Angela's frozen student recipe, transfer confirmed)
#   full/adamw    full FT updates every weight, so its optimum sits ~1-2 decades below the
#                 adapter's; 2e-5 is this repo's full-FT convention
#   */rmsprop     same 1/sqrt(v) scaling as Adam without bias correction -> optimum within
#                 ~a decade of Adam's, typically slightly below
#   */sgd         no anchor: matching Adam's update magnitude needs lr ~ lr_adam / |g|, and
#                 |g| is plausibly 1e-3..1e-2 -> two decades of genuine uncertainty
#   */sgdm0.9     momentum gives an effective step ~lr/(1-0.9) = 10x plain SGD -> the same
#                 ladder shifted one decade down
#
# `max_grad_norm` 1.0 is held constant across every arm, which caps an SGD update's norm at
# ~lr once the gradient norm exceeds 1 -- that is why the SGD ladders reach O(1)-O(100)
# rather than tracking raw gradient magnitudes. Clipping applies identically to every arm,
# so it is a property of the shared recipe, not an SGD-specific handicap.
PHASE0_LADDERS: dict[str, tuple[float, ...]] = {
    "lora/adamw": (2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4),
    "full/adamw": (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5),
    "lora/rmsprop": (1.25e-5, 2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4),
    "full/rmsprop": (5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5),
    "lora/sgd": (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    "lora/sgdm0.9": (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    "full/sgd": (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    "full/sgdm0.9": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
}
PHASE0_SEED = STUDENT_SEEDS[0]
PHASE0_CELLS = sum(len(v) for v in PHASE0_LADDERS.values())  # 48

# Loss-match reference: Angela's frozen headline recipe at its own LR. Every other arm's
# headline rung is the one whose FINAL train loss lands nearest this arm's, within
# `LOSS_MATCH_TOL` -- so a null on `full/sgd` cannot be dismissed as "it just underfit".
LOSS_MATCH_REFERENCE = ("lora/adamw", 1e-4)
# Stricter than Blank et al. Appendix L, whose "similar" pair differs by 18%.
LOSS_MATCH_TOL = 0.10


# ---- Phase 1 -- headline arms + trait-free floor ------------------------------

# Treatment: every arm at its calibrated LR x 3 student seeds. Control: the SAME arms and
# LRs distilled from the rng number pool, which is the trait-free floor p(target) is
# measured against. The control must be run PER ARM: a different optimizer moves the
# untrained-trait baseline too, so one shared AdamW floor would bias every other arm's
# delta by an unknown amount.
PHASE1_TREATMENT_CELLS = len(ARMS) * len(STUDENT_SEEDS)  # 24
PHASE1_CONTROL_CELLS = len(ARMS) * len(STUDENT_SEEDS)  # 24

# Headline prompt set for every pre-registered claim (matches the GKD campaign, so the two
# optimizer sweeps are read off the same measurement).
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"


# ---- Phase 1 evals -- capability + specificity on the trained leaves ----------

# `p(target)` alone cannot distinguish "this optimizer transferred the trait" from "this
# optimizer wrecked the model and the trait rode along": SGD at lr=1 moves far more weight
# than AdamW at 1e-4, and a model whose LM ability is gone can still emit the target token.
# So each of the 16 headline param dirs (8 arms x {treatment, rng}) also gets, from its
# ALREADY-TRAINED checkpoints, held-out Pile perplexity + 0-shot lm-eval (is it still a
# language model?) and forced choice (is the preference specific, or just token bias?).
# Read against the shared untuned-base cache, so "destroyed" is a measured ratio.
#
# No training and no new prompt-set scoring: this stage adds files
# (`sdf_metrics_eval.json`, `sdf_forced_choice_summary.json`,
# `sdf_capability_summary.json`, `base_metrics.json`) and never rewrites the
# `preference_eval.json` / `preference_summary.json` that Phase 1 already produced.
PHASE1_EVAL_CELLS = len(ARMS) * 2  # 16: one job per (arm, source) param dir

# lm-eval is the expensive stage (3 tasks x 3 seeds per cell) but it is the only one that
# answers "usable?" in units a reader already knows, so it runs for every cell.
EVAL_RUN_BENCHMARKS = True

# Pinned rather than lm-eval's "auto": several eval cells share a GPU, and two concurrent
# auto-detections both size to the same free VRAM and then both allocate it. Batching does
# not change log-likelihoods, so pinning costs only throughput.
EVAL_BENCHMARK_BATCH_SIZE = 16

# All 250 held-out Pile exemplars (50 per register x 5 registers) -- the bank is small
# enough that capping it would only add a needless degree of freedom.
EVAL_PERPLEXITY_MAX_PER_REGISTER = None


# ---- jobq queues -------------------------------------------------------------

Q_P0 = "offopt-p0-calibration"
Q_P1_HEADLINE = "offopt-p1-headline"
Q_P1_CONTROLS = "offopt-p1-controls"
Q_P1_EVALS = "offopt-p1-evals"

# Evals sit below training: a queue that is still training headline cells should drain
# before GPUs go to measuring cells that are already on disk.
QUEUE_PRIORITIES: dict[str, int] = {
    Q_P0: 100,
    Q_P1_HEADLINE: 80,
    Q_P1_CONTROLS: 80,
    Q_P1_EVALS: 60,
}


# ---- artifacts ---------------------------------------------------------------

CALIBRATION_REPORT = "offpolicy_optim_calibration_report.json"


# ---- W&B (log-few) -----------------------------------------------------------

# Dedicated project so these cells never land in the GKD / SDF / ckpt-ladder boards.
# Entity stays in `.env` / gpu_policy `env` (`WANDB_ENTITY`).
WANDB_PROJECT = "subliminal-learning-offpolicy-optparam"
WANDB_JOB_TYPE = "offpolicy_optparam"
WANDB_GROUP_P0 = "offopt-p0"
WANDB_GROUP_P1 = "offopt-p1"
# Phase 1 logs only this student seed (same as Phase 0). Seeds 2–3 stay on disk.
WANDB_HEADLINE_SEED = PHASE0_SEED


def wandb_group(phase: int) -> str:
    """`WANDB_RUN_GROUP` for one campaign phase."""
    if phase == 0:
        return WANDB_GROUP_P0
    if phase == 1:
        return WANDB_GROUP_P1
    raise ValueError(f"phase must be 0 or 1, got {phase}")


def wandb_tags(*, phase: int, arm_key: str, source: str, seed: int) -> str:
    """Comma-separated `WANDB_TAGS` (dashboard filter pills)."""
    return ",".join(("offopt", f"p{phase}", arm_key, source, f"s{seed}"))


def cell_logs_wandb(*, phase: int, source: str, seed: int) -> bool:
    """Whether this cell creates a W&B run.

    Phase 0 logs every treatment ladder cell (the LR / loss-match dashboard).
    Phase 1 logs only the headline student seed on the treatment; rng controls
    and the other seeds train locally only.
    """
    if source != "teacher":
        return False
    if phase == 0:
        return True
    return phase == 1 and seed == WANDB_HEADLINE_SEED
