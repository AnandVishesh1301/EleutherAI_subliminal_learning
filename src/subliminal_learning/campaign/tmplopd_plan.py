"""Fixed decisions of the tmplopd campaign (run set 20260925_tmplopd-lion_160m).

Single source for `.claude/plans/tmplopd_template_teacher_opd.plan.md` and
`docs/PREREG_TMPLOPD.md`: on-policy GKD from Angela's template (Cloud ft-teacher) lion
teachers vs our SDF lion teachers. Everything but the frozen teacher is the ws160m
headline cell `pythia-160m@step143000 -> pythia-160m@step143000` (lion), so the
teacher checkpoint is the only variable. Cross-init partners are Hub members of the
implementation-matched core (`docs/PYTHIA_160M_ABLATION_CELLS.md` §5): `data-seed1`
(SI_DS) and `seed1` (DI_DS).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import ws160m_plan, xfam2_plan
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag

# ---- identity ---------------------------------------------------------------------

RUN_SET_ID = "20260925_tmplopd-lion_160m"
TEACHER_RUN_SET_ID = RUN_SET_ID
MODEL_SIZE = "160m"
HUB_ORG = "EleutherAI"
OWNER = "vishesh"
ANIMAL = "lion"

# ---- storage (every root on ssd-1, readable from all four nodes) --------------------

CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/tmplopd")
PATHS_CONFIG = "configs/paths/paths_tmplopd.yaml"


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML: every root under `CAMPAIGN_ROOT`."""
    return {
        "output_root": str(CAMPAIGN_ROOT / "outputs"),
        "hf_cache_root": str(CAMPAIGN_ROOT / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT / "wandb"),
        "queue_root": str(CAMPAIGN_ROOT / "queue"),
        "audit_root": str(CAMPAIGN_ROOT / "audit"),
    }


# ---- bases (ws160m roster tokens, Hub members only) ----------------------------------

STEP = 143000
SELF_BASE = "pythia-160m"
X_PARTNERS: tuple[str, ...] = ("data-seed1", "seed1")
HUB_REPOS: MappingProxyType[str, str] = MappingProxyType(
    {
        "pythia-160m": "EleutherAI/pythia-160m",
        "data-seed1": "EleutherAI/pythia-160m-data-seed1",
        "seed1": "EleutherAI/pythia-160m-seed1",
    }
)


def token(base: str) -> str:
    """`--spec-registry ws160m` token of `base` at `STEP`."""
    return f"{base}@step{STEP}"


def label(base: str) -> str:
    """Model dir label of `base` at `STEP` (``ws160m_models.label``)."""
    return f"{base}-step{STEP}"


# ---- teachers (reused, never retrained; copied into the run set at S0) ---------------

SDF_SOURCE = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/sdf_teachers/"
    "20260918_ws160m2-matrix_160m/lion/pythia-160m-step143000"
)
TMPL_SOURCE = Path(
    "/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning/offpolicy_xsize2/"
    "20260913_offxsize2-doglion_mixed/teachers/lion/pythia-160m"
)


@dataclass(frozen=True)
class Teacher:
    """One teacher rung.

    Attributes:
        arm: Arm id (``S`` ours; ``A10``/``A15``/``A20``/``A2`` Angela's template).
        workflow: Teacher-tag workflow token (``sdf`` / ``tmpl``).
        param_tag: Teacher cell param tag (dir name).
        epochs: Teacher training epochs.
        lr: Teacher learning rate.
        lora_r: Teacher LoRA rank.
        lora_alpha: Teacher LoRA alpha.
        ftseeds: Teacher seeds present on disk.
        source: Read-only dir holding ``ftseed{N}/`` (copied at S0).
        source_base_repo: Hub repo the adapter was trained on.
    """

    arm: str
    workflow: str
    param_tag: str
    epochs: int
    lr: float
    lora_r: int
    lora_alpha: int
    ftseeds: tuple[int, ...]
    source: Path
    source_base_repo: str


def _tmpl(arm: str, param_tag: str, epochs: int, lr: float, r: int, a: int, ftseeds):
    return Teacher(
        arm, "tmpl", param_tag, epochs, lr, r, a, tuple(ftseeds),
        TMPL_SOURCE / param_tag, "EleutherAI/pythia-160m",
    )


TEACHERS: MappingProxyType[str, Teacher] = MappingProxyType(
    {
        "S": Teacher(
            "S", "sdf", "ep5_lr5e-5_r128a128", 5, 5.0e-5, 128, 128, (0, 1, 2),
            SDF_SOURCE / "ep5_lr5e-5_r128a128", "EleutherAI/pythia-160m@step143000",
        ),
        "A10": _tmpl("A10", "ep10_lr1e-4_r16a32", 10, 1.0e-4, 16, 32, (1,)),
        "A15": _tmpl("A15", "ep15_lr1e-4_r16a32", 15, 1.0e-4, 16, 32, (0, 1, 2)),
        "A20": _tmpl("A20", "ep20_lr1e-4_r16a32", 20, 1.0e-4, 16, 32, (0, 1, 2)),
        "A2": _tmpl("A2", "ep2_lr5e-4_r32a64", 2, 5.0e-4, 32, 64, (1,)),
    }
)
CONTROL_ARM = "CTRL"

# ---- student recipe (frozen ws160m headline) ------------------------------------------

LMBDA = 1.0
BETA = 1.0
LR = 4.0e-4
EPOCHS = 1.0
LORA_R = 8
LORA_ALPHA = 16
STUDENT_MODE_TOKEN = "lora-r8a16"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
# No `strict` screen: strict_v1 has no lion target lexicon (frozen file).
VARIANTS: tuple[str, ...] = ("legacy", "vmaskf")


def student_gkd_tag(variant: str = "legacy") -> str:
    """GKD dir tag of every cell of `variant` (``legacy`` = the headline tag)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode="lora",
        learning_rate=LR,
        num_epochs=EPOCHS,
        lora=LoRAConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules="all-linear"
        ),
        vocab_mask=variant == "vmaskf",
        vocab_mask_forms=variant == "vmaskf",
    )


# ---- tiers + units --------------------------------------------------------------------

TIER_C, TIER_R, TIER_D, TIER_X = "c", "r", "d", "x"
TIERS: tuple[str, ...] = (TIER_C, TIER_R, TIER_D, TIER_X)
ROTATED: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2))


@dataclass(frozen=True)
class Unit:
    """One param dir: a teacher cell (or control) x variant x student base.

    Attributes:
        tier: One of `TIERS`.
        arm: A `TEACHERS` key, or `CONTROL_ARM`.
        variant: One of `VARIANTS`.
        student: Student base (roster name).
        ftseed: Teacher seed; ``None`` for a control.
        gseeds: GKD training seeds of this param dir in this tier.
    """

    tier: str
    arm: str
    variant: str
    student: str
    ftseed: int | None
    gseeds: tuple[int, ...]

    @property
    def is_control(self) -> bool:
        return self.arm == CONTROL_ARM

    @property
    def teacher(self) -> Teacher | None:
        return None if self.is_control else TEACHERS[self.arm]

    @property
    def slug(self) -> str:
        tfs = "ctrl" if self.ftseed is None else f"t{self.ftseed}"
        return f"{self.tier}-{self.arm}-{self.variant}-{self.student}-{tfs}"


def _full(tier: str, arm: str, variant: str, student: str, gseeds) -> list[Unit]:
    return [
        Unit(tier, arm, variant, student, fs, tuple(gseeds))
        for fs in TEACHERS[arm].ftseeds
    ]


def _rotated(tier: str, arm: str, variant: str, student: str) -> list[Unit]:
    return [Unit(tier, arm, variant, student, fs, (g,)) for fs, g in ROTATED]


def _ctrl(tier: str, variant: str, student: str, gseeds) -> Unit:
    return Unit(tier, CONTROL_ARM, variant, student, None, tuple(gseeds))


def units() -> tuple[Unit, ...]:
    """Every unit, tier by tier (C 30, R 12, D 6, X 24 leaves)."""
    out: list[Unit] = []
    out += _full(TIER_C, "S", "legacy", SELF_BASE, (0, 1, 2))
    out += _full(TIER_C, "A15", "legacy", SELF_BASE, (0, 1, 2))
    out += _full(TIER_C, "A10", "legacy", SELF_BASE, range(6))
    out.append(_ctrl(TIER_C, "legacy", SELF_BASE, range(6)))
    for variant in ("vmaskf",):
        out += _rotated(TIER_R, "S", variant, SELF_BASE)
        out += _rotated(TIER_R, "A15", variant, SELF_BASE)
        out += _full(TIER_R, "A10", variant, SELF_BASE, (0, 1, 2))
        out.append(_ctrl(TIER_R, variant, SELF_BASE, (0, 1, 2)))
    out += _rotated(TIER_D, "A20", "legacy", SELF_BASE)
    out += _full(TIER_D, "A2", "legacy", SELF_BASE, (0, 1, 2))
    for partner in X_PARTNERS:
        out += _rotated(TIER_X, "S", "legacy", partner)
        out += _rotated(TIER_X, "A15", "legacy", partner)
        out += _full(TIER_X, "A10", "legacy", partner, (0, 1, 2))
        out.append(_ctrl(TIER_X, "legacy", partner, (0, 1, 2)))
    return tuple(out)


def n_leaves(tier: str | None = None) -> int:
    """Student leaves of `tier` (all tiers when ``None``)."""
    return sum(len(u.gseeds) for u in units() if tier is None or u.tier == tier)


def logs_wandb(unit: Unit, gseed: int) -> bool:
    """One W&B run per core treatment arm (lowest tfs, gseed 0)."""
    t = unit.teacher
    return (
        unit.tier == TIER_C
        and t is not None
        and unit.ftseed == min(t.ftseeds)
        and gseed == 0
    )


# ---- fleet, queues, packing -------------------------------------------------------------

SOAR = xfam2_plan.SOAR
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
CLUSTER5 = xfam2_plan.CLUSTER5
HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4, CLUSTER5)
HOME_NODE = SOAR
CLONE_ROOT: MappingProxyType[str, Path] = MappingProxyType(
    {h: xfam2_plan.CLONE_ROOT[h] for h in HOSTS}
)
LOCK_PREFIX: MappingProxyType[str, str] = MappingProxyType(
    {h: xfam2_plan.LOCK_PREFIX[h] for h in HOSTS}
)
ALIASES: MappingProxyType[str, str] = MappingProxyType(
    {SOAR: "soar", CLUSTER3: "a40-cluster-3", CLUSTER4: "a40-cluster-4",
     CLUSTER5: "a40-cluster-5"}
)

TRAIN_PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {TIER_C: 100, TIER_R: 80, TIER_D: 60, TIER_X: 40}
)
EVAL_PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {TIER_C: 140, TIER_R: 130, TIER_D: 120, TIER_X: 120}
)
TEACHER_EVAL_PRIORITY = 150
TEACHER_EVAL_QUEUE = f"tmplopd-teacher-evals-{SOAR}"


def train_queue(tier: str, host: str) -> str:
    return f"tmplopd-{tier}-{host}"


def eval_queue(tier: str, host: str) -> str:
    return f"tmplopd-{tier}-evals-{host}"


def queue_node(name: str) -> str:
    """Host a queue is pinned to (its name's host suffix)."""
    for host in HOSTS:
        if name.endswith(f"-{host}"):
            return host
    raise ValueError(f"not a tmplopd queue: {name!r}")


def train_deps(tier: str) -> list[str]:
    """X trains only after every core eval finished (user: core first, then crosses)."""
    if tier == TIER_X:
        return [eval_queue(TIER_C, h) for h in HOSTS]
    return []


def eval_deps(tier: str) -> list[str]:
    """A tier's evals wait for every train queue of that tier (all hosts)."""
    return [train_queue(tier, h) for h in HOSTS]


QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        TEACHER_EVAL_QUEUE: TEACHER_EVAL_PRIORITY,
        **{train_queue(t, h): TRAIN_PRIORITY[t] for t in TIERS for h in HOSTS},
        **{eval_queue(t, h): EVAL_PRIORITY[t] for t in TIERS for h in HOSTS},
    }
)

# ws160m measurements (identical nodes, identical cell): student peak 16.2-16.4 GiB.
STUDENT_MEM_MIB = 19_000
EVAL_MEM_MIB = 5_000
TRAIN_CAP_PER_GPU = 2
EVAL_CAP_PER_GPU = 1
TRAIN_CAP_GROUP = "tmplopd-train"
EVAL_CAP_GROUP = "tmplopd-eval"
CAP_PER_GPU = TRAIN_CAP_PER_GPU + EVAL_CAP_PER_GPU
WORKERS_PER_NODE = 24
AUTO_REQUEUE_ATTEMPTS = 2

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 1_500,
        "reservation_warmup_s": 600,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(CAMPAIGN_ROOT / "hf"),
                "HF_HUB_CACHE": str(CAMPAIGN_ROOT / "hf" / "hub"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        ),
    }
)


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def stage_defaults(is_eval: bool) -> dict:
    """Queue-default stage ceiling: 2 students + 1 eval per GPU."""
    if is_eval:
        return {"cap_per_gpu": EVAL_CAP_PER_GPU, "cap_group": EVAL_CAP_GROUP}
    return {"cap_per_gpu": TRAIN_CAP_PER_GPU, "cap_group": TRAIN_CAP_GROUP}


# ---- configs + eval ---------------------------------------------------------------------

STUDENT_EXPERIMENT_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        "sdf": "configs/experiment/tmplopd_student_sdf.yaml",
        "tmpl": "configs/experiment/tmplopd_student_tmpl.yaml",
    }
)
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student.yaml"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
TEACHER_EXPERIMENT_CONFIG = "configs/experiment/ws160m_teacher.yaml"
SPEC_REGISTRY = "ws160m"
SHARED_GENERATED_DIRNAME = "generated_configs"
GENERATED_CONFIG_DIR = "configs/experiment/generated/tmplopd"
PREREG_PATH = "docs/PREREG_TMPLOPD.md"

# ws160m's eval sets (`config_gen.EVAL_BLOCK` body), primary endpoint and quarantine.
EVAL_PROMPT_SETS: tuple[str, ...] = ws160m_plan.EVAL_PROMPT_SETS
PRIMARY_PROMPT_SET = ws160m_plan.PRIMARY_PROMPT_SET
PRIMARY_SCORING = ws160m_plan.PRIMARY_SCORING
WANDB_PROJECT = "subliminal-learning-tmplopd"

# ---- analysis ---------------------------------------------------------------------------

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260925
MIN_DELTA = 0.02
REPRO_FLAG_DIFF = 0.10
QUARANTINE_PPL_RATIO_MAX = ws160m_plan.QUARANTINE_PPL_RATIO_MAX
QUARANTINE_FILTER_DROP_MAX = ws160m_plan.QUARANTINE_FILTER_DROP_MAX
CONCORDANCE_RUN_SET = "20260918_ws160m2-matrix_160m"

# ---- artifacts --------------------------------------------------------------------------

CAMPAIGN_STATE_FILE = "tmplopd_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CORE_DONE_FILE = "CORE_EVALS_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
STAGING_REPORT = "staging_report.json"
ADOPTION_MANIFEST = "adoption_manifest.json"
TMUX_SESSION = "tmplopd"
WATCH_SESSION = "tmplopd-watch"
MIN_DISK_FREE_GIB = 50

SPEC: CampaignSpec = replace(
    xfam2_plan.SPEC,
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    teacher_param_tag_lora=TEACHERS["S"].param_tag,
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    generated_config_dir=GENERATED_CONFIG_DIR,
    gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG["sdf"],
    eval_prompt_sets=EVAL_PROMPT_SETS,
    calibration_report=STAGING_REPORT,
    a1_selection_report=STAGING_REPORT,
    freeze_report=STAGING_REPORT,
    init_equality_report=STAGING_REPORT,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    extra_pin_paths=(
        STUDENT_EXPERIMENT_CONFIG["tmpl"],
        STUDENT_TRAINING_CONFIG,
        TEACHER_TRAINING_CONFIG,
        TEACHER_EXPERIMENT_CONFIG,
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
    ),
)
