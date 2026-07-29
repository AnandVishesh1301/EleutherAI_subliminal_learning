"""Fixed decisions of the 20260727 student-HP breadth + cross-init campaign.

The single source for every constant of the campaign plan
(.cursor/plans/student_hp_and_cross-init_campaign_4a48b77d.plan.md): run-set ids,
frozen teacher, Stage A1/A2/A3 grids, Stage B seed policy, queue names/priorities,
gate thresholds, and the expected cell counts each driver gate verifies against
disk. Recipes are never hardcoded here — they are selected from Stage outputs by
`campaign.selection`; this module holds only the pre-registered grid.
"""

from __future__ import annotations

# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260727_slbreadth_160m"
TEACHER_RUN_SET_ID = "20260726_crossinit_160m"
HUB_ORG = "EleutherAI"

TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
# Frozen teacher: best LoRA tag on both animals across all 7 inits (teacher-side
# evidence only). The full tag is the Stage C robustness arm, never Stage A/B.
TEACHER_PARAM_TAG_LORA = "ep5_lr5e-5_r128a128"
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"

ANIMALS = ("cat", "dog")
BETAS = (1.0, 0.0)
LMBDA = 1.0

# ---- Stage A1 — HP screen ------------------------------------------------------

A1_STUDENT_MODES = (
    "lora-r8a8",
    "lora-r8a16",
    "lora-r8a32",
    "lora-r16a16",
    "lora-r16a32",
    "lora-r16a64",
    "lora-r64a16",
    "lora-r64a64",
    "lora-r128a128",
)
A1_LORA_LRS = (5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4)
# A rank-1 recipe sitting on the top rung leaves the lr surface censored (the optimum
# may be above the grid), so A3 gets an extra arm at `top lr * this factor`.
A1_LR_EXTENSION_FACTOR = 2.0
A1_NUM_EPOCHS = 1.0
A1_GSEEDS = (0,)
STAGE_A_TFS = (0,)
# [student_base, teacher_base] — the sweep's canonical pair order.
A1_SAME_PAIR = ("data-seed1", "data-seed1")
A1_CROSS_PAIR = ("data-seed1", "weight-seed1")

A1_CELLS_PER_YAML = (
    len(ANIMALS) * len(BETAS) * len(A1_STUDENT_MODES) * len(A1_LORA_LRS)
)  # 144
A1_TOTAL_CELLS = 2 * A1_CELLS_PER_YAML  # 288 (same + cross YAML)

# ---- Stage A2 — control floor --------------------------------------------------

A2_STUDENT_MODE = "lora-r64a64"
A2_CONTROL_CELLS = len(ANIMALS) * len(BETAS) * len(A1_LORA_LRS)  # 16
# Gate thresholds (documented in selection.control_floor_check).
CONTROL_FLOOR_MAX = 0.02
CONTROL_FLOOR_MAX_SPREAD = 0.01

# ---- Stage A3 — dose ladder ----------------------------------------------------

A3_DOSES = (0.5, 1.0, 3.0, 5.0, 10.0)
A3_TOP_K = 3
A3_GSEEDS = (0, 1, 2)
# Same-init arms per (animal, beta) are selection-dependent, not fixed: the union of the
# top-K by end-of-run p_target and the top-K by trajectory peak, plus one lr-edge extension
# per censored rank-1 leader (see selection.select_top_recipes / lr_edge_extensions). Hence
# there is no A3 cell-count constant — the driver gates enumerate the generated YAMLs via
# `campaign.expected`, which is exact for any arm count.
#
# The bound is provable, not a guess: the union holds at most 2K recipes, and there are at
# most 2 rank-1 leaders (one per reducer), each contributing at most one extension arm.
A3_MAX_SAME_ARMS = 2 * A3_TOP_K + 2

# ---- Stage B — cross-init factorial -------------------------------------------

B_TFS_HEADLINE = (0, 1, 2)
B_TFS_EXTENSION = (0,)
B_GSEEDS = (0, 1, 2)
B_CONTROL_CELLS = 7 * len(ANIMALS) * len(B_GSEEDS)  # 42

# ---- Stage C (optional ablations, per-corner representatives) ------------------

C_LAMBDAS = (0.0, 0.5)
C_GSEEDS = (0, 1, 2)

# ---- jobq queues (one shared root, one worker pool) ----------------------------

Q_A1 = "slb-a1-hpscreen"
Q_A2 = "slb-a2-controlfloor"
Q_A3 = "slb-a3-doseladder"
Q_B_HEADLINE = "slb-b-factorial-headline"
Q_B_EXT = "slb-b-factorial-extension"
Q_C = "slb-c-ablations"
Q_A1_EVALS = "slb-a1-evals"
Q_A3_EVALS = "slb-a3-evals"
Q_B_EVALS = "slb-b-evals"
Q_C_EVALS = "slb-c-evals"
Q_BACKLOG = "catswide-eval-backlog"

QUEUE_PRIORITIES: dict[str, int] = {
    Q_A1: 100,
    Q_A2: 95,
    Q_A3: 80,
    Q_B_HEADLINE: 70,
    Q_B_EXT: 50,
    Q_C: 40,
    Q_A1_EVALS: 30,
    Q_A3_EVALS: 30,
    Q_B_EVALS: 30,
    Q_C_EVALS: 30,
    Q_BACKLOG: 5,
}

# All 8 A40s belong to this project; foreign processes are transient and are NOT a
# capacity constraint, so no GPU is excluded. Admission is governed by the slot cap plus
# the reservation-aware memory gate (jobqueue.gpu), not by a padded memory floor:
# `free_mem_mib` is the true per-cell requirement and `cap_per_gpu` bounds a card.
#
# MEASURED (A1 anchor cell, lora-r64a64 @ 2e-4, batch 32, on an otherwise idle A40):
# peak 16.7 GB steady-state reserved (unchanged with the probes firing). Models load in
# fp32 (`gkd.trainer_io` dtype default), so the completion-slice logits dominate. Hence
# cap 2 -> 33.4 GB of 48 GB (comfortable, and tolerant of one foreign process); cap 3
# would need 50.1 GB and OOM. Batch 64 was measured at 31.7 GB = 1 cell/card, which is
# net SLOWER campaign-wide (halves the slot count for ~1.4x per-slot throughput), so the
# batch stays 32 — see docs/PROJECT_DESIGN.md.
GPU_POLICY: dict = {
    "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
    "cap_per_gpu": 2,
    "free_mem_mib": 17000,
    "reservation_warmup_s": 300,
    "env": {"OMP_NUM_THREADS": "4"},
}

# Threads oversubscribe the 16 GPU slots on purpose: a thread with no free slot just
# blocks in GpuManager.acquire's retry sleep (releases the GIL in nvidia-smi/flock calls),
# so extra threads are near-free on this node (48 CPUs) and close the refill gap when a
# slot frees — one of the several already-waiting threads grabs it on its next 5s retry
# tick instead of only the just-finished thread's own claim/acquire cycle being able to.
WORKER_OVERSUBSCRIBE = 2
WORKER_THREADS = (
    WORKER_OVERSUBSCRIBE * len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])
)

# ---- static configs + campaign artifacts ---------------------------------------

A1_SAME_YAML = "configs/experiment/gkd_student_sweep_hpscreen_same.yaml"
A1_CROSS_YAML = "configs/experiment/gkd_student_sweep_hpscreen_cross.yaml"
A1_SAME_VERIFY_YAML = "configs/experiment/gkd_student_verify_sweep_hpscreen_same.yaml"
A1_CROSS_VERIFY_YAML = "configs/experiment/gkd_student_verify_sweep_hpscreen_cross.yaml"
GENERATED_CONFIG_DIR = "configs/experiment/generated"
CATSWIDE_VERIFY_YAMLS = (
    "configs/experiment/gkd_student_verify_sweep_lora_catswide.yaml",
    "configs/experiment/gkd_student_verify_sweep_lora_catswide_fkl.yaml",
    "configs/experiment/gkd_student_verify_sweep_lora_catswide_cross.yaml",
)
CATSWIDE_BASE_MODELS = (
    "data-seed1",
    "data-seed2",
    "data-seed3",
    "weight-seed1",
    "weight-seed2",
    "weight-seed3",
)

CALIBRATION_REPORT = "calibration_report.json"
A1_SELECTION_REPORT = "a1_selection_report.json"
FREEZE_REPORT = "freeze_report.json"
INIT_EQUALITY_REPORT = "init_equality.json"
CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
