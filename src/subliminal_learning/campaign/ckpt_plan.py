"""Fixed decisions of the 20260805 pretraining-checkpoint ladder campaign.

The single source for every constant of
`.cursor/plans/checkpoint_ladder_campaign_fdd5029a.plan.md`: the two rung ladders,
the seven PolyPythia inits, the frozen teacher/student recipe, the replication and
crossed-anchor rungs, the eval bundle, the geometry block, the pre-registered gate
thresholds, queue names/priorities, the GPU policy, and the artifact filenames.
Nothing measured is hardcoded here -- P1/P2 emit their own numbers through
`campaign.ckpt_selection`; this module holds only the pre-registered design.

Sibling of `campaign.optparam_plan` and `campaign.breadth_plan` rather than an
extension of either: the three campaigns share the driver machinery
(`campaign.driver_io`, `campaign.expected`) but no grid constant, so an edit to one
frozen plan can never move another's.

`SPEC` exists because `driver_io.CampaignDriverConfig.__post_init__` overwrites
``run_set_id`` and ``queue_priorities`` from its ``spec`` field unconditionally.
A caller that passes only ``run_set_id=``/``queue_priorities=`` (as
`campaign.optparam_jobs`' driver does) silently resolves to `campaign.plan.SPEC`'s
run-set tree with none of this campaign's queue priorities. The ckpt driver
therefore passes ``spec=SPEC``; the token is registered in
`driver_io.CAMPAIGN_SPECS` so ``--campaign ckptladder`` resolves here.

Several `CampaignSpec` fields describe the student-HP stage machine (Stage A1/A2/A3
screening, Stage C ablations) and have no counterpart in this campaign; they are
filled with this campaign's frozen values purely so the dataclass is constructible
and `spec.queue_names()` never names a queue with no priority. They are marked
below and are not read by any ckpt-ladder code path.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag


class AdmissionTier(StrEnum):
    """P2 ``(rung, init)`` teacher disposition -- one SSOT for the tier strings.

    Attributes:
        OK: Teacher clears both the SD-multiple and the absolute floor.
        ADVISORY: Exactly one of the two criteria is met.
        QUARANTINE: Neither criterion is met (or a hard quarantine rule fired).
    """

    OK = "ok"
    ADVISORY = "advisory"
    QUARANTINE = "quarantine"


class Metric(StrEnum):
    """Similarity-frame ``metric`` tokens this campaign's gates and claims read.

    The long-format geometry schema (plan §6.7) carries one row per
    ``(pair, rung, slot, metric)``; these are the metric names the campaign gates
    on. `TRAIT_COORDINATE` is only produced where a trait direction exists (P5 and
    the step0 golden integration test), never on a P1 base-pair frame.
    """

    MINIBATCH_CKA = "minibatch_cka"
    PROCRUSTES = "procrustes"
    SSA = "ssa"
    TRAIT_COORDINATE = "trait_coordinate"


# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260805_ckptladder_160m"
# Teachers are trained inside the behaviour run set (plan §8 output tree), so the
# two ids are one value, not two trees to keep in sync.
TEACHER_RUN_SET_ID = RUN_SET_ID
# Inference-only geometry ladder (P1/P1c/P5) writes under its own run set.
GEOM_RUN_SET_ID = "20260805_ckptladder_geom_160m"
# Stage B's run set, joined at the top rung as the C6 external concordance check.
STAGE_B_RUN_SET_ID = "20260727_slbreadth_160m"

HUB_ORG = "EleutherAI"
MODEL_SIZE = "160m"

TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
OWNER = "vishesh"


# ---- ladders (plan §2) -------------------------------------------------------

# Geometry ladder: inference only, all 7 inits. Nine rungs below step5000 so the
# 0.22% (~step300) / 1% (~step1400) / SSA-peak (~step2000) onset predictions are
# each resolvable. Every entry is a published Pythia revision
# (`papers/md/polypythias_reference.md` §5.3: step0,1,2,4,...,512 then
# step1000...step143000 in 1000-step increments; 154 checkpoints per model).
LADDER_G: tuple[int, ...] = (
    0,
    8,
    32,
    128,
    256,
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    24000,
    32000,
    48000,
    64000,
    80000,
    96000,
    112000,
    128000,
    143000,
)

# Behaviour ladder: teachers + students, a subset of Ladder G. `step0` is excluded
# -- a random-init model cannot be an SDF teacher; it is Ladder G's golden anchor.
LADDER_B: tuple[int, ...] = (
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    64000,
    96000,
    128000,
    143000,
)

# The published log-spaced head of the Pythia checkpoint schedule; everything
# above `step512` is a multiple of 1000 up to `step143000` (= `main`).
CHECKPOINT_LOG_HEAD: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CHECKPOINT_STEP_INTERVAL = 1000
CHECKPOINT_MAX_STEP = 143000

# Both PolyPythia single-factor families at all three released seeds, plus the
# original. `data-seed{N}` = shared weight init / different data order;
# `weight-seed{N}` = different weight init / shared data order (models.registry).
INITS: tuple[str, ...] = (
    "original",
    "data-seed1",
    "data-seed2",
    "data-seed3",
    "weight-seed1",
    "weight-seed2",
    "weight-seed3",
)


# ---- frozen recipe (plan §3.4) -----------------------------------------------

TEACHER_PARAM_TAG_LORA = "ep5_lr5e-5_r128a128"
# Stage C's full-FT teacher tag; this campaign never trains it, but the sweep
# YAMLs carry both keys in `teacher_param_tags`.
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"
TEACHER_EPOCHS = 5
TEACHER_LR = 5.0e-5
TEACHER_LORA_R = 128
TEACHER_LORA_ALPHA = 128

STUDENT_MODE = "lora-r8a16"
STUDENT_LR = 4.0e-4
STUDENT_LORA = LoRAConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear"
)
STUDENT_RECIPE = Recipe(
    lora_r=STUDENT_LORA.r, lora_alpha=STUDENT_LORA.lora_alpha, lr=STUDENT_LR
)
BETA = 1.0
LMBDA = 1.0
NUM_EPOCHS = 1.0
# Derived, not transcribed: `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16` from
# `logs/freeze_report.json`. The literal is asserted in tests/unit/test_ckpt_plan.py
# so a tag-format change surfaces here rather than as a silently new leaf tree.
STUDENT_GKD_TAG = gkd_tag(
    lmbda=LMBDA,
    beta=BETA,
    student_mode="lora",
    learning_rate=STUDENT_LR,
    num_epochs=NUM_EPOCHS,
    lora=STUDENT_LORA,
)


# ---- replication and scope (plan §3.2, §3.3) ---------------------------------

FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)
# Rungs where the crossed-anchor pairing subset runs the full 3x3 grid, restoring
# the variance-component decomposition the rotated diagonal aliases.
CROSSED_ANCHOR_RUNGS: tuple[int, ...] = (2000, 64000)

ANIMALS: tuple[str, ...] = ("cat",)
# P6 confirmation, behind `--include-dog`.
DOG_ANIMALS: tuple[str, ...] = ("dog",)
# Six-rung subset of Ladder B for the dog confirmation. Chosen as a superset of
# `BENCHMARK_RUNGS` minus its top-heavy gap plus `step8000`, so the confirmation
# reuses the rungs that already carry trained-leaf benchmarks.
DOG_RUNGS: tuple[int, ...] = (512, 2000, 8000, 16000, 64000, 143000)

# Trained-leaf lm-eval rungs (plan §7.1) -- deliberately distinct from
# `CROSSED_ANCHOR_RUNGS`, which are a replication device, not a capability probe.
BENCHMARK_RUNGS: tuple[int, ...] = (512, 2000, 16000, 64000, 143000)
BENCH_TASKS: tuple[str, ...] = ("lambada_openai", "arc_easy", "sciq")
# Pile perplexity cap for the P1b untrained-base capability ladder (plan §7.2).
PERPLEXITY_MAX_PER_REGISTER = 200
# Frankle linear-interpolation barrier rungs (plan §6.5): early / mid / final, so
# the barrier is sampled across the divergence range rather than at one end.
WEIGHT_SPACE_RUNGS: tuple[int, ...] = (2000, 16000, 143000)


# ---- evals (plan §7.1) -------------------------------------------------------

# The 4-set bundle: held-out primary, in-distribution robustness, format-OOD
# generation, and the off-topic false-positive floor. NOT the breadth5 7-set and
# not the earlier 9-set sketch. Every name must resolve through
# `evaluation.sdf_prompt_sets.prompt_set_path` (asserted in the unit test).
EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "storytelling", "off_topic")
# Sole primary endpoint for C1-C9.
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"


# ---- geometry block (plan §5, §6.1, §6.2, §6.10) -----------------------------

# `cka/extract_io.py` hooks `embed_in` + each block (raw post-block residuals);
# `steering/vectors.py` uses `output_hidden_states`, whose slot 12 is
# post-final-layernorm. Both return 13 slots on Pythia-160M, so mixing them is
# silent. The ladder's primary convention is the extractor's (Angela's existing
# results are in it); trait coordinates must be read in the convention the trait
# direction was extracted in, which is the hidden-states one.
SLOT_CONVENTION = "post_block"
TRAIT_SLOT_CONVENTION = "hidden_states"
# `last` matches the position `evaluation/preference_prob_io.py` scores p(target) at.
POOLING = "last"
# Activation source for the geometry ladder: the same Pile prompt pool the students
# roll out on, so representations are read on the distribution the campaign trains on.
GEOM_PROMPT_SOURCE = PROMPT_POOL_DIR
# Subspace dimension for `cka.metrics.subspace_angles_topk`. Neither PolyPythias
# (arXiv:2503.09543 §4) nor Müller-Eberstein et al. states a k, so this is a
# campaign choice, stated rather than implied: 10 directions of the 768-dim
# residual stream stay well-determined at `BATCH_SIZE` rows while covering the
# dominant variance. Held here so every SSA number in the campaign shares one k.
SSA_K = 10
SSA_REDUCTION = "mean"
# Raghu et al. arXiv:2108.08810 minibatch-CKA protocol.
N_EXAMPLES = 10240
BATCH_SIZE = 1024
N_REPEATS = 20
# Davari et al. arXiv:2210.16156 high-norm robustness re-score.
NORM_EXCLUSION_FRAC = 0.01
# Kornblith §6.2 random-init similarity floor, re-measured at every rung: an early
# checkpoint is near random init, so the floor is itself high at the bottom.
RANDOM_INIT_SEEDS: tuple[int, ...] = (0, 1, 2)
# Only the random-init reference stacks are worth disk (they are reused at every
# rung). Caching the ladder itself would cost ~57 GB for no reuse (plan §5).
GEOM_CACHE_ROLES: tuple[str, ...] = ("random",)
# Metrics the P1/P5 geometry driver computes per pair (plan §6.1).
GEOM_METRICS: tuple[str, ...] = (
    Metric.MINIBATCH_CKA,
    Metric.PROCRUSTES,
    Metric.SSA,
)

# --- P1c grounding probe (plan §6.3) ---
# UPOS on Universal Dependencies English-EWT: the closest free analogue of the
# PolyPythias PoS probe. The full MDL probe (7 licensed treebanks, variational
# probe, learned layer weighting) is out of scope -- plan §12.
PROBE_TREEBANK = "ud_english_ewt"
PROBE_SPLIT = "dev"
PROBE_TASK = "upos"
PROBE_SEED = 0


# ---- pre-registered gate thresholds (plan §6.10, §7) -------------------------

# --- P1 -> P2 hard dial gate ---
# step0 golden anchor: same-weight-init models are byte-identical there
# (`logs/init_equality.json`: original vs data-seed1 = 0/148 tensors differ), so
# CKA must be 1 and Procrustes 0 to within numerical noise.
DIAL_STEP0_TOL = 1e-6
# Minimum max-min span of the scored divergence measure over the Ladder-B rungs.
# Below this the x-axis of C1 has no dose to respond to.
DIAL_MIN_DIVERGENCE_SPAN = 0.10
# Minimum |stream-family - init-family| divergence gap at at least one rung.
DIAL_MIN_FAMILY_SEPARATION = 0.05

# --- P2 teacher admission (a labelling step, not a submission gate) ---
ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02
# Hard raise: fitting C1 on fewer admitted rungs is fitting on a collapsed
# divergence range. The analogue of optparam's zero-usable-arm rule.
MIN_OK_RUNGS = 6

# --- cell quarantine (plan §6.10; consumed by the results pipeline) ---
QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5

# --- P0 preflight ---
MIN_DISK_FREE_GIB = 150

# --- claim statistics (analysis.stats) ---
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260805
# C1: `beta3` inside this equivalence bound supports H2 (init-independent channel).
C1_BETA3_EQUIVALENCE_BOUND = 0.05
# C3: onset = first rung clearing control + this many SDs for two consecutive rungs.
C3_SD_MULTIPLE = 2.0
# Literature onset markers as a fraction of the 143k-step budget: Moskvoretskii et
# al. 0.22% (~step 300) and Stecher et al. 1% (~step 1400), plotted on figure 4.
C3_ONSET_MARKERS: tuple[float, ...] = (0.0022, 0.01)
# C4: the basis breakdown rung is where the per-coordinate/rotation-invariant gap
# exceeds this many MADs of its own below-breakdown distribution.
C4_GAP_MAD_MULTIPLE = 3.0
# C5: |Spearman(y, perplexity_ratio)| at or above this declares the confound.
C5_SPEARMAN_ABS_QUARANTINE = 0.5
# C7 (self-drift), C8 (metric grading) and C9 (weight space) are secondary and
# descriptive; they carry no numeric support threshold of their own.
CLAIM_IDS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")


# ---- jobq queues -------------------------------------------------------------

# `ckptladder-{phase}-{role}`; phases follow the plan §4 DAG.
Q_P1_GEOMETRY = "ckptladder-p1-geometry"
Q_P1B_CAPABILITY = "ckptladder-p1b-capability"
Q_P1C_PROBES = "ckptladder-p1c-probes"
Q_P2_TEACHERS = "ckptladder-p2-teachers"
Q_P2_TEACHER_EVALS = "ckptladder-p2-teacher-evals"
Q_P3_STUDENTS = "ckptladder-p3-students"
Q_P3_CONTROLS = "ckptladder-p3-controls"
Q_P3_CROSSED = "ckptladder-p3-crossed"
Q_P4_EVALS = "ckptladder-p4-evals"
Q_P5_GEOMETRY = "ckptladder-p5-geometry"
Q_P5W_WEIGHTSPACE = "ckptladder-p5w-weightspace"
Q_P6_TEACHERS = "ckptladder-p6-teachers"
Q_P6_STUDENTS = "ckptladder-p6-students"
Q_P6_EVALS = "ckptladder-p6-evals"

# Priority follows the DAG, not importance: an upstream stage that gates a
# downstream one must drain first. The optional P6 confirmation sits in the idle
# band so it can never displace core work.
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        Q_P1_GEOMETRY: 100,
        Q_P1B_CAPABILITY: 95,
        Q_P1C_PROBES: 95,
        Q_P2_TEACHERS: 90,
        Q_P2_TEACHER_EVALS: 85,
        Q_P3_STUDENTS: 80,
        Q_P3_CONTROLS: 80,
        Q_P3_CROSSED: 75,
        Q_P4_EVALS: 60,
        Q_P5_GEOMETRY: 55,
        Q_P5W_WEIGHTSPACE: 50,
        Q_P6_TEACHERS: 30,
        Q_P6_STUDENTS: 25,
        Q_P6_EVALS: 20,
    }
)


# ---- cluster policy ----------------------------------------------------------

# Node-wide defaults are sized on the GKD LoRA / SDF teacher cell (same unit the
# 20260801 campaign calibrated): `cap_per_gpu=2` x `free_mem_mib=17000`. P1/P1b/P1c
# inference shards override per-job via `P1_INFER_SLOTS` / `P1_INFER_MEM_MIB` so a
# geometry capture cannot share a card with another heavy job (measured 2026-08-10:
# fp32 P1 capture peaks ~18.6 GiB on real pile prompts at batch 1024; two such
# processes plus any roommate OOMs an A40). Re-measure with a one-rung
# `run_similarity_ladder` smoke if batch/dtype/prompt pool changes.
GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 2,
        "free_mem_mib": 17000,
        "reservation_warmup_s": 300,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "4",
                # P1 loads many Hub revisions concurrently; offline avoids
                # huggingface_hub download locks once the ladder cache is warm.
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        ),
    }
)

# Exclusive GPU for P1 / P1b / P1c: weight=2 against `cap_per_gpu=2`, and a free-mem
# floor above the measured fp32 capture peak (~18628 MiB) so a second admit cannot
# sneak in during reservation warmup. Training stages keep the default slot/mem.
P1_INFER_SLOTS = 2
P1_INFER_MEM_MIB = 22000

WORKER_OVERSUBSCRIBE = 2
WORKER_THREADS = WORKER_OVERSUBSCRIBE * 8 * 2


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`.

    `GPU_POLICY` is a `MappingProxyType` of tuples so no caller can mutate the
    campaign's frozen policy; the launcher has to write
    ``gpu_policy.{host}.json``, and `json.dumps` rejects a mapping proxy. This is
    the one place the conversion happens.

    Returns:
        ``{"gpus": [...], "cap_per_gpu": int, "free_mem_mib": int,
        "reservation_warmup_s": int, "env": {...}}`` -- the key set
        `jobqueue.gpu.GpuPolicy.from_dict` reads.
    """
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


# ---- static configs + artifacts ---------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated/ckptladder"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
# Committed (not generated): the P1 geometry ladder driver's config, read through
# `load_config()` so its paths come from the paths YAML (plan §6.7).
GEOMETRY_CONFIG = "configs/experiment/ckpt_ladder_geometry.yaml"
# Committed (not generated): the P3 student cells' experiment config. It differs from
# the default `gkd_student.yaml` only in turning `coverage_probe` on, which is the
# only way `gkd.probe_io.FirstStepCoverageCallback` fires -- and without it C9 has no
# `coverage` column at all (plan §6.5).
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/ckpt_ladder_student.yaml"

# Entrypoints the inference-only stages shell out to. Held here so the job
# builders never spell a module path inline; the modules themselves land with the
# pipelines in a later wave (plan §6.9), and `driver_io.verify_pin_runs_campaign`
# refuses a code pin whose worktree lacks them.
GEOMETRY_MODULE = "scripts.run_similarity_ladder"
CAPABILITY_MODULE = "scripts.run_base_capability_ladder"
WEIGHT_SPACE_MODULE = "scripts.run_weight_space_barrier"

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"

PREFLIGHT_REPORT = "preflight_report.json"
INIT_EQUALITY_LADDER_REPORT = "init_equality_ladder.json"
DIAL_GATE_REPORT = "dial_gate_report.json"
TEACHER_ADMISSION_REPORT = "teacher_admission_report.json"

RESULTS_DELTA_CSV = "ladder_delta.csv"
RESULTS_REPLICATION_MAP_CSV = "ladder_replication_map.csv"
RESULTS_VARIANCE_COMPONENTS_CSV = "ladder_variance_components.csv"
RESULTS_DELTA_AGG_CSV = "ladder_delta_agg.csv"
RESULTS_DIVERGENCE_CSV = "ladder_divergence.csv"
RESULTS_BASE_CAPABILITY_CSV = "ladder_base_capability.csv"
RESULTS_REGRESSION_CSV = "ladder_regression.csv"
RESULTS_EMERGENCE_CSV = "ladder_emergence.csv"
RESULTS_BASIS_BREAKDOWN_CSV = "ladder_basis_breakdown.csv"
RESULTS_SELF_DRIFT_CSV = "ladder_self_drift.csv"
RESULTS_METRIC_GRADING_CSV = "ladder_metric_grading.csv"
RESULTS_WEIGHT_SPACE_CSV = "ladder_weight_space.csv"
RESULTS_PROBE_CSV = "ladder_probe_trajectories.csv"
RESULTS_QUARANTINE_CSV = "ladder_quarantine.csv"
RESULTS_CLAIMS_JSON = "claims.json"


SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora=TEACHER_PARAM_TAG_LORA,
    teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
    animals=ANIMALS,
    betas=(BETA,),
    lmbda=LMBDA,
    # --- Stage A/B/C fields: not applicable to the checkpoint ladder. Filled with
    # this campaign's frozen recipe / seeds so the spec is constructible and every
    # `q_*` names a queue that has a priority. No ckpt-ladder code path reads them.
    a1_student_modes=(STUDENT_MODE,),
    a1_lora_lrs=(STUDENT_LR,),
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=FTSEEDS,
    a1_same_pair=("data-seed1", "data-seed1"),
    a1_cross_pair=("data-seed1", "weight-seed1"),
    a2_student_mode=STUDENT_MODE,
    control_floor_max=0.02,
    control_floor_max_spread=0.01,
    a3_doses=(NUM_EPOCHS,),
    a3_top_k=1,
    a3_gseeds=GSEEDS,
    a3_max_same_arms=1,
    a3_reference_recipe=STUDENT_RECIPE,
    b_tfs_headline=FTSEEDS,
    b_tfs_extension=(FTSEEDS[0],),
    b_gseeds=GSEEDS,
    c_lambdas=(LMBDA,),
    c_gseeds=GSEEDS,
    q_a1=Q_P1_GEOMETRY,
    q_a2=Q_P1B_CAPABILITY,
    q_a3=Q_P1C_PROBES,
    q_b_headline=Q_P3_STUDENTS,
    q_b_ext=Q_P3_CROSSED,
    q_c=Q_P5_GEOMETRY,
    q_a1_evals=Q_P2_TEACHER_EVALS,
    q_a3_evals=Q_P4_EVALS,
    q_b_evals=Q_P4_EVALS,
    q_c_evals=Q_P4_EVALS,
    q_backlog=Q_P6_EVALS,
    # --- end not-applicable block ---
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    # This campaign generates every sweep YAML; it has no committed static ones.
    a1_same_yaml="",
    a1_cross_yaml="",
    a1_same_verify_yaml="",
    a1_cross_verify_yaml="",
    generated_config_dir=GENERATED_CONFIG_DIR,
    gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
    catswide_verify_yamls=(),
    catswide_base_models=(),
    calibration_report=PREFLIGHT_REPORT,
    a1_selection_report=DIAL_GATE_REPORT,
    freeze_report=TEACHER_ADMISSION_REPORT,
    init_equality_report=INIT_EQUALITY_LADDER_REPORT,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
)
