"""Fixed decisions of the 20260809 four-animal student-HP breadth campaign.

Sibling SSOT of `campaign.plan` (cat/dog `20260727_slbreadth_160m`) for
cat / dog / bear / eagle under run-set `20260809_breadth4_160m`. Same
A1/A2/A3/B scientific design with β=1.0 only and A3 doses without 10.0;
distinct animals, run-set ids, generated-config dir, and queue names.
Campaign launch key remains ``breadth5`` in `driver_io.CAMPAIGN_SPECS`
(historical CLI flag); this module is the breadth4 restart SSOT.

Teacher run-set: `20260809_breadth4-teachers_160m` (hyphen in the purpose
slug for `is_valid_run_set_id`). Train T1+T2 for cat/bear/eagle; reuse dog
teachers via symlink from `20260726_crossinit_160m`. Cat corpus is an 8k
subsample (n=8495, seed 42) of the selected cat JSONL.

Abandoned prior attempt: `20260804_breadth5_160m` / `b5-*` queues (do not
resume or overwrite).

Full-eval-bundle contract (preference + FC + capability + storytelling +
headline lm-eval):

- Generated verify YAMLs use `SPEC.eval_prompt_sets` (the six slbreadth sets
  plus Cloud-parity ``storytelling``).
- Driver finalize / A3 freeze gates (Batch 2) require
  ``preference_summary.json`` + ``sdf_forced_choice_summary.json`` +
  ``sdf_capability_summary.json`` on every freeze-path / Stage B config dir;
  headline verify YAMLs that enable lm-eval also require a ``benchmarks`` key
  inside the capability summary.
- Preference-only completion must never advance the driver.

Queue-scoping note: workers for this campaign share the canonical jobq root
with ``slb-*`` / ``optparam-*`` queues. The ``b4-`` name prefix plus the low
priority band (24→2) preserve idle-filler semantics — free slots pull
breadth work only when higher-priority queues are empty; running jobs are
never preempted.
"""

from __future__ import annotations

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec

# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260809_breadth4_160m"
# Hyphen (not underscore) in the purpose slug — `is_valid_run_set_id` allows
# ``[a-z0-9-]*`` only between the date and size segments.
TEACHER_RUN_SET_ID = "20260809_breadth4-teachers_160m"
HUB_ORG = "EleutherAI"

TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
TEACHER_PARAM_TAG_LORA = "ep5_lr5e-5_r128a128"
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"

# Students / YAML identity order.
ANIMALS = ("cat", "dog", "bear", "eagle")
# Train only these; dog teachers are symlinked from DOG_TEACHER_SOURCE_RUN_SET_ID.
TEACHER_TRAIN_ANIMALS = ("cat", "bear", "eagle")
DOG_TEACHER_SOURCE_RUN_SET_ID = "20260726_crossinit_160m"

BETAS = (1.0,)
LMBDA = 1.0

# SSD3 selected all-animals root + cat 8k override (match dog n=8495).
SDF_DOCS_ROOT = (
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/"
    "sdf_all_animals_20260713-231408_selected"
)
CAT_SUBSAMPLE_N = 8495
CAT_SUBSAMPLE_SEED = 42
CAT_SDF_DOCS_OVERRIDE = (
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/"
    "sdf_cat_8k_n8495_seed42/sdf_cat_docs_selected.jsonl"
)

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
A1_LR_EXTENSION_FACTOR = 2.0
A1_NUM_EPOCHS = 1.0
A1_GSEEDS = (0,)
STAGE_A_TFS = (0,)
A1_SAME_PAIR = ("data-seed1", "data-seed1")
A1_CROSS_PAIR = ("data-seed1", "weight-seed1")

A1_CELLS_PER_YAML = (
    len(ANIMALS) * len(BETAS) * len(A1_STUDENT_MODES) * len(A1_LORA_LRS)
)  # 144
A1_TOTAL_CELLS = 2 * A1_CELLS_PER_YAML  # 288 (same + cross YAML)

# ---- Stage A2 — control floor --------------------------------------------------

A2_STUDENT_MODE = "lora-r64a64"
A2_CONTROL_CELLS = len(ANIMALS) * len(BETAS) * len(A1_LORA_LRS)  # 16
CONTROL_FLOOR_MAX = 0.02
CONTROL_FLOOR_MAX_SPREAD = 0.01

# ---- Stage A3 — dose ladder ----------------------------------------------------

A3_DOSES = (0.5, 1.0, 3.0, 5.0)
A3_TOP_K = 3
A3_GSEEDS = (0, 1, 2)
# 2*K union + ≤2 lr-edge extensions + 1 reference arm.
A3_MAX_SAME_ARMS = 2 * A3_TOP_K + 2 + 1  # 9
# Cat/dog frozen recipe (r8a16 @ 4e-4) — guaranteed shared arm for freeze.
A3_REFERENCE_RECIPE = Recipe(8, 16, 4.0e-4)

# ---- Stage B — cross-init factorial -------------------------------------------

B_TFS_HEADLINE = (0, 1, 2)
B_TFS_EXTENSION = (0,)
B_GSEEDS = (0, 1, 2)
# 31 pairings → 141 treat cells/animal (headline 8×3×3 + ext 23×1×3).
B_TREAT_CELLS_PER_ANIMAL = 141
B_TREAT_CELLS = B_TREAT_CELLS_PER_ANIMAL * len(ANIMALS)  # 564
B_CONTROL_CELLS = 7 * len(ANIMALS) * len(B_GSEEDS)  # 84
B_TOTAL_CELLS = B_TREAT_CELLS + B_CONTROL_CELLS  # 648

# ---- Stage C (optional; not in this campaign's launch scope) ------------------

C_LAMBDAS = (0.0, 0.5)
C_GSEEDS = (0, 1, 2)

# ---- jobq queues (shared root; b4- prefix + low priority band) ---------------

Q_TEACHER_T1 = "b4-teachers-t1"
Q_TEACHER_T2 = "b4-teachers-t2"
# T3 evals in the plan (preference+FC+capability verify). Named queues so the
# scoped worker pool in launch_breadth5_campaign.sh / launch_breadth5_teachers.sh
# drains them; depends_on is set at submit time (t1-evals → t1, t2-evals → t2).
Q_TEACHER_T1_EVALS = "b4-teachers-t1-evals"
Q_TEACHER_T2_EVALS = "b4-teachers-t2-evals"
Q_A1 = "b4-a1-hpscreen"
Q_A2 = "b4-a2-controls"
Q_A3 = "b4-a3-doseladder"
Q_B_HEADLINE = "b4-b-factorial-headline"
Q_B_EXT = "b4-b-factorial-ext"
Q_C = "b4-c-ablations"
Q_A1_EVALS = "b4-a1-evals"
Q_A3_EVALS = "b4-a3-evals"
Q_B_EVALS = "b4-b-evals"
Q_C_EVALS = "b4-c-evals"
Q_BACKLOG = "b4-eval-backlog"

QUEUE_PRIORITIES: dict[str, int] = {
    Q_TEACHER_T1: 24,
    Q_TEACHER_T2: 23,
    Q_TEACHER_T1_EVALS: 22,
    Q_TEACHER_T2_EVALS: 21,
    Q_A1: 20,
    Q_A2: 19,
    Q_A3: 16,
    Q_B_HEADLINE: 14,
    Q_B_EXT: 10,
    Q_C: 8,
    Q_A1_EVALS: 6,
    Q_A3_EVALS: 6,
    Q_B_EVALS: 6,
    Q_C_EVALS: 6,
    Q_BACKLOG: 2,
}

GPU_POLICY: dict = {
    "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
    "cap_per_gpu": 2,
    "free_mem_mib": 17000,
    "reservation_warmup_s": 300,
    "env": {"OMP_NUM_THREADS": "4"},
}

WORKER_OVERSUBSCRIBE = 2
WORKER_THREADS = (
    WORKER_OVERSUBSCRIBE * len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])
)

# ---- static configs + campaign artifacts ---------------------------------------

A1_SAME_YAML = "configs/experiment/gkd_student_sweep_hpscreen_same_breadth5.yaml"
A1_CROSS_YAML = "configs/experiment/gkd_student_sweep_hpscreen_cross_breadth5.yaml"
A1_SAME_VERIFY_YAML = (
    "configs/experiment/gkd_student_verify_sweep_hpscreen_same_breadth5.yaml"
)
A1_CROSS_VERIFY_YAML = (
    "configs/experiment/gkd_student_verify_sweep_hpscreen_cross_breadth5.yaml"
)
GENERATED_CONFIG_DIR = "configs/experiment/generated/breadth4"
CATSWIDE_VERIFY_YAMLS: tuple[str, ...] = ()
CATSWIDE_BASE_MODELS: tuple[str, ...] = ()

CALIBRATION_REPORT = "calibration_report.json"
A1_SELECTION_REPORT = "a1_selection_report.json"
FREEZE_REPORT = "freeze_report.json"
INIT_EQUALITY_REPORT = "init_equality.json"
CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"

EVAL_PROMPT_SETS = (
    "test",
    "favorite",
    "dislike",
    "validation",
    "consensus",
    "register",
    "storytelling",
)

SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora=TEACHER_PARAM_TAG_LORA,
    teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
    animals=ANIMALS,
    betas=BETAS,
    lmbda=LMBDA,
    a1_student_modes=A1_STUDENT_MODES,
    a1_lora_lrs=A1_LORA_LRS,
    a1_lr_extension_factor=A1_LR_EXTENSION_FACTOR,
    a1_gseeds=A1_GSEEDS,
    stage_a_tfs=STAGE_A_TFS,
    a1_same_pair=A1_SAME_PAIR,
    a1_cross_pair=A1_CROSS_PAIR,
    a2_student_mode=A2_STUDENT_MODE,
    control_floor_max=CONTROL_FLOOR_MAX,
    control_floor_max_spread=CONTROL_FLOOR_MAX_SPREAD,
    a3_doses=A3_DOSES,
    a3_top_k=A3_TOP_K,
    a3_gseeds=A3_GSEEDS,
    a3_max_same_arms=A3_MAX_SAME_ARMS,
    a3_reference_recipe=A3_REFERENCE_RECIPE,
    b_tfs_headline=B_TFS_HEADLINE,
    b_tfs_extension=B_TFS_EXTENSION,
    b_gseeds=B_GSEEDS,
    c_lambdas=C_LAMBDAS,
    c_gseeds=C_GSEEDS,
    q_a1=Q_A1,
    q_a2=Q_A2,
    q_a3=Q_A3,
    q_b_headline=Q_B_HEADLINE,
    q_b_ext=Q_B_EXT,
    q_c=Q_C,
    q_a1_evals=Q_A1_EVALS,
    q_a3_evals=Q_A3_EVALS,
    q_b_evals=Q_B_EVALS,
    q_c_evals=Q_C_EVALS,
    q_backlog=Q_BACKLOG,
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    a1_same_yaml=A1_SAME_YAML,
    a1_cross_yaml=A1_CROSS_YAML,
    a1_same_verify_yaml=A1_SAME_VERIFY_YAML,
    a1_cross_verify_yaml=A1_CROSS_VERIFY_YAML,
    generated_config_dir=GENERATED_CONFIG_DIR,
    catswide_verify_yamls=CATSWIDE_VERIFY_YAMLS,
    catswide_base_models=CATSWIDE_BASE_MODELS,
    calibration_report=CALIBRATION_REPORT,
    a1_selection_report=A1_SELECTION_REPORT,
    freeze_report=FREEZE_REPORT,
    init_equality_report=INIT_EQUALITY_REPORT,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
    gkd_experiment_config="configs/experiment/gkd_student_breadth5.yaml",
)
