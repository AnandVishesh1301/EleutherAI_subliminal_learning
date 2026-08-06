"""Fixed decisions of the 20260801 optimizer x parameterization campaign.

The single source for every constant of
`.cursor/plans/optimizer_parameterization_campaign_c092f671.plan.md`: the 2x4 arm
grid, the Phase-0 LR ladders, selection thresholds, queue names/priorities, the GPU
policy, and the report filenames. Selected LRs are never hardcoded here -- Phase 0 and
Phase 1 emit them via `campaign.optparam_selection`; this module holds only the
pre-registered grid.

Sibling of `campaign.plan` (the 20260727 student-HP campaign) rather than an extension of
it: the two campaigns share the driver machinery (`campaign.driver_io`,
`campaign.expected`) but no grid constant, and merging them would let an edit to one
campaign's frozen plan silently move the other's.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from subliminal_learning.gkd.layout import parse_optim_slug


class RungTier(StrEnum):
    """Phase-1 (arm, LR) disposition — single SSOT for tier name strings."""

    OK = "ok"
    ADVISORY = "advisory"
    QUARANTINE = "quarantine"


# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260801_optparam_160m"
TEACHER_RUN_SET_ID = "20260726_crossinit_160m"
HUB_ORG = "EleutherAI"

TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
TEACHER_PARAM_TAG_LORA = "ep5_lr5e-5_r128a128"
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"

ANIMALS = ("cat", "dog")
BETA = 1.0
LMBDA = 1.0
NUM_EPOCHS = 1.0
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"

# The trio whose base-model equality was checked in `logs/init_equality.json`.
INITS = ("original", "data-seed1", "weight-seed1")
# One init carries Phases 0-1; all three carry the headline Phase 2.
SCREEN_INIT = "data-seed1"

TFS_HEADLINE = (0, 1, 2)
TFS_LOSSMATCH = (0,)
GSEEDS = (0, 1, 2)
SCREEN_GSEEDS = (0, 1, 2)
CONTROL_FLOOR_GSEEDS = (0,)


# ---- the 2x4 factorial -------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One parameterization x optimizer arm.

    Attributes:
        mode: Student FT-mode token as the sweep spells it (``full`` / ``lora-r8a16``).
        optim_slug: Optimizer slug per `gkd.layout.optim_slug` (``adamw`` ... ``sgdm0.9``).
    """

    mode: str
    optim_slug: str

    @property
    def kind(self) -> str:
        """``"full"`` or ``"lora"`` -- the `student_lr` ladder key prefix."""
        return "full" if self.mode == "full" else "lora"

    @property
    def optim(self) -> str:
        """HF ``TrainingArguments.optim`` name for this arm."""
        return parse_optim_slug(self.optim_slug)[0]

    @property
    def momentum(self) -> float:
        """SGD momentum for this arm (``0.0`` for every other optimizer)."""
        return parse_optim_slug(self.optim_slug)[1]

    @property
    def key(self) -> str:
        """Stable arm id used in filenames, job keys, and reports."""
        return f"{self.kind}_{self.optim_slug}"

    @property
    def ladder_key(self) -> str:
        """The `student_lr` mapping key this arm's ladder lives under."""
        return f"{self.kind}/{self.optim_slug}"


# `lora-r8a16` is the frozen headline recipe from `logs/freeze_report.json`; `full` is
# the parameterization Blank et al. Fig. 7b reports as a null.
PARAMETERIZATIONS = ("lora-r8a16", "full")
OPTIM_SLUGS = ("adamw", "rmsprop", "sgd", "sgdm0.9")
ARMS: tuple[Arm, ...] = tuple(
    Arm(mode, slug) for mode in PARAMETERIZATIONS for slug in OPTIM_SLUGS
)
ARMS_BY_KEY: dict[str, Arm] = {a.key: a for a in ARMS}

FULL_FT_MODE = "full"


# ---- Phase 0 -- LR / loss calibration ----------------------------------------

# The screen runs the campaign's own dose. Loss-matching (Blank et al. App. L) compares
# *final* training losses, so screening at a fraction of the dose reads a converged
# adaptive arm against a truncated non-adaptive one: at 200 steps both `sgdm0.9` arms were
# still descending faster than the reference and missed the band by 26-52%. 625 steps is
# one pass over the 20k-prompt pool at the frozen batch size (`n_train_steps` of any
# `_e1_` leaf under `gkd_students/20260727_slbreadth_160m`). Held as a step cap rather
# than `num_epochs` so Phase-0 leaves tag `_s625_` and stay identity-distinct from the
# `_e1_` leaves Phase 1 writes at the same LRs; the training itself is identical.
PHASE0_MAX_STEPS = 625
PHASE0_ANIMAL = "cat"
PHASE0_GSEED = 0
PHASE0_TFS = 0

# Per-arm ladders. Narrow-and-resolved (~2x spacing) where an anchor exists, wide only
# where none does; the edge-of-ladder gate below covers the "optimum outside the ladder"
# risk that would otherwise argue for wide-and-coarse everywhere.
#
#   lora/adamw    measured 4e-4 (logs/freeze_report.json)
#   full/adamw    measured 1e-5..2e-5 (cats_1day_run); Blank App. K 1e-5; their repo 2e-5
#   */rmsprop     same 1/sqrt(v) per-coordinate scaling as Adam, no bias correction ->
#                 optimum within ~a decade of Adam's, typically slightly below
#   */sgd         no anchor: matching update magnitude needs lr_sgd ~ lr_adam / |g|, and
#                 |g| is plausibly 1e-3..1e-2 -> two decades of genuine uncertainty
#   */sgdm0.9     momentum gives effective step ~lr/(1-0.9) = 10x plain SGD -> its own
#                 ladder, shifted one decade down
#
# The held-constant `max_grad_norm` 1.0 caps an SGD update's norm at ~lr once the gradient
# norm exceeds 1, which is why the SGD ladders reach O(1)-O(100) rather than tracking raw
# gradient magnitudes. Clipping applies identically to every arm, so it is a property of
# the shared recipe, not an SGD-specific handicap.
PHASE0_LADDERS: dict[str, tuple[float, ...]] = {
    "lora/adamw": (5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3),
    "full/adamw": (2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4),
    "lora/rmsprop": (2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4),
    "full/rmsprop": (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5),
    "lora/sgd": (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0),
    # 30.0 dropped after two independent CUDA NaN collapses at step ~20 on
    # 20260801_optparam_160m (sampling probs non-finite); new top edge is 10.0.
    "lora/sgdm0.9": (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    "full/sgd": (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    "full/sgdm0.9": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
}
PHASE0_CELLS = sum(len(v) for v in PHASE0_LADDERS.values())  # 49

# Loss-match reference: the frozen headline recipe at its measured optimum.
LOSS_MATCH_REFERENCE = ("lora/adamw", 4e-4)
# Stricter than Blank et al. Appendix L, whose "similar" pair differs by 18%.
LOSS_MATCH_TOL = 0.10
# A rung whose end-of-run probe perplexity exceeds this multiple of the reference arm's
# is excluded as diverged before any minimization -- a NaN or blown-up rung must never
# sort first as "lowest loss". Referenced to the loss-match arm rather than to the base
# model (which the in-training probe never measures: it first fires after step
# `probe_every`) -- a deliberate, pre-registered substitution, and a conservative one,
# since a trained student's held-out ppl is at or above the base model's.
PHASE0_PPL_MAX_RATIO = 3.0
# Rungs carried into the Phase-1 ladder per arm.
PHASE1_LADDER_RUNGS = 4


# ---- Phase 1 -- ladder + control floor ---------------------------------------

# Same criterion as the 20260727 A2 gate (selection.control_floor_check), applied per arm.
CONTROL_FLOOR_MAX = 0.02
CONTROL_FLOOR_MAX_SPREAD = 0.01
# Floors in (CONTROL_FLOOR_MAX, factor * CONTROL_FLOOR_MAX] are advisory; above the
# factor they quarantine. Factor 2x keeps the hard ceiling meaningful while giving a
# soft band for "elevated but still a language model" controls.
CONTROL_FLOOR_ADVISORY_FACTOR: float = 2.0
# A rung whose treatment leaves are majority-diverged is as unusable as a diverged
# control: Phase-2 would subtract two dead models. 0.5 matches "majority".
QUARANTINE_DIVERGED_TREATMENT_FRACTION: float = 0.5
# Treatment end-of-run probe ppl this many times the matched control's is advisory
# (capability collapse without a hard diverged flag). Same multiple as Phase-0's
# PHASE0_PPL_MAX_RATIO so the two screens speak one language.
TREATMENT_PPL_ADVISORY_RATIO: float = 3.0

PHASE1_TREATMENT_CELLS = (
    len(ARMS) * PHASE1_LADDER_RUNGS * len(ANIMALS) * len(SCREEN_GSEEDS)
)  # 192
PHASE1_CONTROL_CELLS = (
    len(ARMS) * PHASE1_LADDER_RUNGS * len(ANIMALS) * len(CONTROL_FLOOR_GSEEDS)
)  # 64


# ---- Phase 2 -- headline arms ------------------------------------------------

PHASE2_TREATMENT_CELLS = (
    len(ARMS) * len(ANIMALS) * len(INITS) * len(TFS_HEADLINE) * len(GSEEDS)
)  # 432
PHASE2_MAX_LOSSMATCH_CELLS = (
    len(ARMS) * len(ANIMALS) * len(INITS) * len(TFS_LOSSMATCH) * len(GSEEDS)
)  # 144
PHASE2_MAX_CONTROL_CELLS = (
    len(ARMS) * 2 * len(ANIMALS) * len(INITS) * len(GSEEDS)
)  # 288

# Headline prompt set for LR selection and every pre-registered claim.
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"

# W&B corner (see `optparam_jobs._headline_wandb`). Deliberately NOT the tfs-0 corner:
# Phase 1 already trained (SCREEN_INIT, tfs 0, gseeds 0-2) at every ladder rung, so the
# selected-LR cells there are byte-identical to Phase-2 cells and resume as skips -- a
# logged corner placed on them would never produce a W&B run.
WANDB_CORNER_TFS = TFS_HEADLINE[1]
WANDB_CORNER_GSEED = 0


# ---- pre-registered claim thresholds (docs/PREREG_OPTPARAM.md) ---------------

# An arm "shows the effect" when its paired delta CI clears zero on at least this many of
# the three inits, for BOTH animals -- replication across inits, not one lucky seed trio.
CLAIM_MIN_INITS = 2
# C4 is an equivalence bound, not a missing p-value: an optimizer counts as a failure only
# if its delta CI *upper* bound sits below this fraction of the matched AdamW arm's delta.
# "Not significant" alone never counts.
SGD_NULL_EQUIVALENCE_FRACTION = 0.05
CLAIM_IDS = ("C1", "C2", "C3", "C4", "C5")


# ---- jobq queues -------------------------------------------------------------

Q_P0 = "optparam-p0-calibration"
Q_P1_LADDER = "optparam-p1-ladder"
Q_P1_FLOOR = "optparam-p1-controlfloor"
Q_P1_EVALS = "optparam-p1-evals"
Q_P2_HEADLINE = "optparam-p2-headline"
Q_P2_LOSSMATCH = "optparam-p2-lossmatch"
Q_P2_CONTROLS = "optparam-p2-controls"
Q_P2_EVALS = "optparam-p2-evals"

QUEUE_PRIORITIES: dict[str, int] = {
    Q_P0: 100,
    Q_P1_LADDER: 90,
    Q_P1_FLOOR: 90,
    Q_P1_EVALS: 60,
    Q_P2_HEADLINE: 70,
    Q_P2_CONTROLS: 70,
    Q_P2_LOSSMATCH: 55,
    Q_P2_EVALS: 30,
}


# ---- cluster policy ----------------------------------------------------------

# The whole node, no device partition. `cap_per_gpu` is sized on the LIGHT (LoRA) unit
# and heavy full-FT jobs declare `slots=FULL_FT_SLOTS`, so a heavy and a light cell
# co-tenant cleanly instead of every card being capped to the fattest arm.
#
# `free_mem_mib` is the per-LoRA-slot requirement measured on this pipeline
# (logs/calibration_report.json, 16.7 GB steady-state reserved at batch 32, fp32 models).
# Re-measure on `full`+`adamw` with scripts/calibrate_gkd_throughput.py before the bulk
# submit and raise FULL_FT_SLOTS if that arm needs more than two light slots; caps are
# ceilings at the OOM boundary, not fill targets.
GPU_POLICY: dict = {
    "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
    "cap_per_gpu": 2,
    "free_mem_mib": 17000,
    "reservation_warmup_s": 300,
    "env": {"OMP_NUM_THREADS": "4"},
}
FULL_FT_SLOTS = 2
FULL_FT_MEM_MIB = 22000

WORKER_OVERSUBSCRIBE = 2
WORKER_THREADS = (
    WORKER_OVERSUBSCRIBE * len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])
)


# ---- artifacts ---------------------------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated"

CALIBRATION_REPORT = "optim_calibration_report.json"
LR_SELECTION_REPORT = "lr_selection_report.json"
CONTROL_FLOOR_REPORT = "control_floor_report.json"
QUARANTINE_REPORT = "quarantine_report.json"
CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"

RESULTS_DELTA_CSV = "optparam_delta.csv"
RESULTS_DELTA_AGG_CSV = "optparam_delta_agg.csv"
RESULTS_LOSS_MATCH_CSV = "optparam_loss_match.csv"
RESULTS_PROBE_CSV = "optparam_probe_trajectories.csv"
RESULTS_GRAD_CSV = "optparam_grad_concentration.csv"
RESULTS_QUARANTINE_CSV = "optparam_quarantine.csv"
RESULTS_CLAIMS_JSON = "claims.json"
