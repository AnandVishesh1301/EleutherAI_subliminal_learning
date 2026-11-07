"""Fixed decisions of the rebuttal top-K campaign (`rebtopk`).

Single source for `.claude/plans/rebuttal/rebtopk.plan.md` and
`docs/PREREG_REBTOPK.md`. Two experiments share the opdlim recipe (on-policy, forward
KL, unnormalised top-K, LoRA r8 a16, lr 4e-4, 1 epoch) under the ``strict_v2`` screen:

- `E10`: bear at K=10 on P3 and P4 with four column-removal arms.
- `E11`: lion and eagle on P3 down the K ladder, plus animal removal at K=10.

Controls are placebo students at the same setting. Teachers are reused read-only through
create-only symlinks under the run set's teacher tree. Priority 3 rows run after
priority 1 from the same queues' lower priorities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import opdlim_plan as opdlim
from subliminal_learning.campaign.opdlim_plan import HP_BASE, P3, P4, PairDef
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, gkd_tag, teacher_tag
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir

# ---- identity ---------------------------------------------------------------------

CAMPAIGN = "rebtopk"
RUN_SET_ID = "20261107_rebtopk-animals_160m"
MODEL_SIZE = "160m"
HUB_ORG = opdlim.HUB_ORG
OWNER = opdlim.OWNER
CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/rebuttal/rebtopk")
PATHS_CONFIG = "configs/paths/paths_rebtopk.yaml"
STUDENT_TRAINING_CONFIG = opdlim.STUDENT_TRAINING_CONFIG
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/rebtopk_student.yaml"
SHARED_GENERATED_DIRNAME = "generated_configs"

BEAR, LION, EAGLE = "bear", "lion", "eagle"
ANIMALS: tuple[str, ...] = (BEAR, LION, EAGLE)
SCREEN = "strict_v2"

# ---- teachers (reused, never retrained) ---------------------------------------------

TEACHER_WORKFLOW = opdlim.TEACHER_WORKFLOW
TEACHER_TAG = opdlim.TEACHER_TAG_EP5
PLACEBO_TAG = opdlim.TEACHER_TAG_PLACEBO
TEACHER_FTSEEDS: tuple[int, ...] = opdlim.TEACHER_FTSEEDS
SSD3_TEACHERS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/sdf_teachers"
)
PLACEBO_TEACHERS = (
    Path(opdlim.EXTERNAL_TEACHER_ROOTS[opdlim.PLACEBO_TEACHER_RUN_SET_ID])
    / SDF_TEACHERS_SUBDIR
    / opdlim.PLACEBO_TEACHER_RUN_SET_ID
    / BEAR
)
TREATMENT_SOURCES: MappingProxyType[tuple[str, str], Path] = MappingProxyType(
    {
        (BEAR, opdlim.B160): SSD3_TEACHERS
        / opdlim.XSIZE_TEACHER_RUN_SET_ID
        / BEAR
        / opdlim.B160
        / TEACHER_TAG,
        (BEAR, opdlim.B160_8K): SSD3_TEACHERS
        / opdlim.CKPTGRID_TEACHER_RUN_SET_ID
        / BEAR
        / opdlim.B160_8K
        / TEACHER_TAG,
        (LION, opdlim.B160): SSD3_TEACHERS
        / "20260918_ws160m2-matrix_160m"
        / LION
        / "pythia-160m-step143000"
        / TEACHER_TAG,
        (EAGLE, opdlim.B160): SSD3_TEACHERS
        / "20260916_ws160m-main_160m"
        / EAGLE
        / "pythia-160m-step143000"
        / TEACHER_TAG,
    }
)
PLACEBO_V1_MAX_DELTA = 0.005
HUB_BASE_REPO_BY_MODEL_DIR: MappingProxyType[str, str] = MappingProxyType(
    {
        opdlim.B160: "EleutherAI/pythia-160m@step143000",
        opdlim.B160_8K: "EleutherAI/pythia-160m@step8000",
    }
)

# ---- token lists ---------------------------------------------------------------------

N_REMOVED = 43
TOKEN_LIST_SEED = 20261107
CENSUS_ROOT = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/gkd_students"
)
CENSUS_CSV: MappingProxyType[str, Path] = MappingProxyType(
    {
        "P3": CENSUS_ROOT
        / "20260925_opdlim-census-p3-bear_mixed/results/topk_census_base_tokens.csv",
        "P4": CENSUS_ROOT
        / "20260924_opdlim-census-bear_mixed/results/topk_census_base_tokens.csv",
    }
)
CENSUS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        "P3": "4aa61d63e00683c84d9798133f7608df9aa9697a2c6d2cda617daee6af6d0024",
        "P4": "29df16330b9af3a0b3854e92cb3d33cf7e377d374c61c9c7ad1a5e5631f4d5b2",
    }
)
TOKEN_LIST_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        "boost43p3": "e85a621f08ce8f8e90cad920f3f60ad3c44be24589adfd84a73a63759fad8d43",
        "boost43p4": "1190a114047b4456d004dbcc225f9fc7527afeb894e79098e9c06a195f0978d5",
        "rand43p3": "54145f19e6b0fd3ffdea2e01ab4d0696367fd427e83dc4dc59e474cc3f9907cf",
        "rand43p3b": "8189894282a068c1ae367d10efe9226cd5007c2b466bdb88b28f8288f410d420",
        "rand43p4": "8e1f80399aeed5b6faa485a616110b4b2b6aa63d677df091d05b9320624e8e3e",
        "rand43p4b": "56f82541cc8b3f75a4defc798414ea5b5f746bd6f9f1a084be2876746a7dda94",
        "randb43p3": "875a363ea6f60b4bc0e43e68bd2fa67183a61ebfada69a79350fb606b5cdb234",
        "randb43p4": "bbd1843726073350ead078da7d24c889338e291b786b15b158410acb7c96ecb7",
    }
)
ANIMAL_IDS_SHA256 = "1e0606f86193b8e8e666877127bc8734b10898482173011ab3f41e69e33413db"
NONE, ANIMAL = "none", "animal"
RAND, BOOST, RAND2, RANDB = "rand", "boost", "rand2", "randb"
LIST_ARMS: tuple[str, ...] = (RAND, BOOST, RAND2, RANDB)


def list_slug(arm: str, pair: PairDef) -> str:
    """Token-list slug of `arm` on `pair`, e.g. ``rand43p3``."""
    p = pair.name.lower()
    return {
        RAND: f"rand{N_REMOVED}{p}",
        BOOST: f"boost{N_REMOVED}{p}",
        RAND2: f"rand{N_REMOVED}{p}b",
        RANDB: f"randb{N_REMOVED}{p}",
    }[arm]


# ---- settings + units -----------------------------------------------------------------

E10, E11 = "e10", "e11"
TREATMENT, PLACEBO = "treatment", "placebo"
K_E10 = 10
K_LADDER: tuple[int | None, ...] = (None, 100, 30, 10, 5)
K_LADDER_P3: tuple[int, ...] = (1000, 2)
ROTATED: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2))
GSEEDS: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class Setting:
    """One objective: K and the columns removed.

    Attributes:
        k: Teacher top-K; ``None`` = full vocabulary.
        arm: `NONE`, `ANIMAL` or one of `LIST_ARMS`.
        slug: Token-list slug for `LIST_ARMS`, else ``None``.
    """

    k: int | None
    arm: str = NONE
    slug: str | None = None

    def __post_init__(self) -> None:
        if (self.arm in LIST_ARMS) != (self.slug is not None):
            raise ValueError(f"arm {self.arm!r} and list slug {self.slug!r} disagree")

    @property
    def name(self) -> str:
        """Path-safe setting name, e.g. ``k10-rand43p3`` or ``full-none``."""
        k = "full" if self.k is None else f"k{self.k}"
        return f"{k}-{self.slug or self.arm}"

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return gkd_tag(
            lmbda=1.0,
            beta=opdlim.BETA,
            student_mode="lora",
            learning_rate=opdlim.BASE_LR,
            num_epochs=opdlim.NUM_EPOCHS,
            lora=HP_BASE.lora,
            vocab_mask=self.arm != NONE,
            vocab_mask_forms=self.arm == ANIMAL,
            vocab_mask_list=self.slug,
            top_k=self.k,
            top_k_ce=self.k is not None,
            rollout_screen=SCREEN,
        )

    def sweep_keys(self) -> dict:
        """Top-level sweep-YAML keys of this setting."""
        keys: dict = {"rollout_screen": SCREEN}
        if self.k is not None:
            keys.update({"top_k": int(self.k), "top_k_ce": True})
        if self.arm == ANIMAL:
            keys["vocab_mask_forms"] = True
        if self.slug is not None:
            keys["vocab_mask_list"] = self.slug
        return keys


@dataclass(frozen=True)
class Unit:
    """One param dir: an animal x pair x setting x teacher.

    Attributes:
        experiment: `E10` or `E11`.
        priority: ``1`` or ``3``.
        animal: Target.
        pair: Teacher = student base pair.
        setting: Objective.
        kind: `TREATMENT` or `PLACEBO`.
        ftseed: Teacher fine-tuning seed.
        gseeds: GKD training seeds of this param dir.
    """

    experiment: str
    priority: int
    animal: str
    pair: PairDef
    setting: Setting
    kind: str
    ftseed: int
    gseeds: tuple[int, ...]

    @property
    def teacher_param_tag(self) -> str:
        """Teacher cell dir name."""
        return TEACHER_TAG if self.kind == TREATMENT else PLACEBO_TAG

    @property
    def teacher_tag(self) -> str:
        """Teacher-lineage directory tag."""
        return teacher_tag(TEACHER_WORKFLOW, self.teacher_param_tag, self.ftseed)

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return self.setting.gkd_tag

    @property
    def slug(self) -> str:
        """Unique unit name."""
        return (
            f"{self.experiment}-{self.animal}-{self.pair.name.lower()}"
            f"-{self.setting.name}-{self.kind}-t{self.ftseed}"
        )


def _e10_units(pair: PairDef, setting: Setting, priority: int, pairs) -> list[Unit]:
    return [
        Unit(E10, priority, BEAR, pair, setting, kind, tfs, gseeds)
        for kind in (TREATMENT, PLACEBO)
        for tfs, gseeds in pairs
    ]


def _e10() -> list[Unit]:
    full_design = [(tfs, GSEEDS) for tfs in TEACHER_FTSEEDS]
    rotated = [(tfs, (g,)) for tfs, g in ROTATED]
    units: list[Unit] = []
    for pair in (P3, P4):
        arms = (
            Setting(K_E10),
            Setting(K_E10, ANIMAL),
            Setting(K_E10, RAND, list_slug(RAND, pair)),
            Setting(K_E10, BOOST, list_slug(BOOST, pair)),
        )
        for setting in arms:
            units += _e10_units(pair, setting, 1, full_design)
        for arm in (RAND2, RANDB):
            units += _e10_units(
                pair, Setting(K_E10, arm, list_slug(arm, pair)), 3, rotated
            )
    return units


def _e11() -> list[Unit]:
    units: list[Unit] = []
    for animal in (LION, EAGLE):
        settings = [(Setting(k), 1) for k in K_LADDER]
        settings.append((Setting(K_E10, ANIMAL), 1))
        settings += [(Setting(k), 3) for k in K_LADDER_P3]
        for setting, priority in settings:
            for tfs, g in ROTATED:
                units.append(
                    Unit(E11, priority, animal, P3, setting, TREATMENT, tfs, (g,))
                )
            for tfs, g in ROTATED:
                placebo_priority = priority if tfs == ROTATED[0][0] else 3
                units.append(
                    Unit(E11, placebo_priority, animal, P3, setting, PLACEBO, tfs, (g,))
                )
    return units


def units() -> tuple[Unit, ...]:
    """Every planned param dir."""
    return (*_e10(), *_e11())


def n_cells(priority: int | None = None) -> int:
    """Planned leaves, optionally of one priority."""
    return sum(
        len(u.gseeds) for u in units() if priority is None or u.priority == priority
    )


# ---- paths ---------------------------------------------------------------------------


def model_dir(token: str) -> str:
    """Model directory name of a base token."""
    return model_dir_name(parse_base_models(token)[0], HUB_ORG)


def run_set_root(output_root: Path) -> Path:
    """Student run-set root."""
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, RUN_SET_ID)


def teacher_subdir() -> str:
    """Run-set teacher tree under `output_root`."""
    return run_set_subdir(SDF_TEACHERS_SUBDIR, RUN_SET_ID)


def teacher_links(output_root: Path) -> tuple[tuple[Path, Path], ...]:
    """``(link, source param dir)`` of every teacher dir the units read."""
    seen: dict[Path, Path] = {}
    for u in units():
        base = u.pair.teacher
        link = (
            Path(output_root)
            / teacher_subdir()
            / u.animal
            / model_dir(base)
            / u.teacher_param_tag
        )
        source = (
            TREATMENT_SOURCES[(u.animal, base)]
            if u.kind == TREATMENT
            else PLACEBO_TEACHERS / model_dir(base) / PLACEBO_TAG
        )
        seen[link] = source
    return tuple(seen.items())


# ---- queues + priorities --------------------------------------------------------------

_TRAIN_PRIORITY: MappingProxyType[tuple[int, str, str], int] = MappingProxyType(
    {
        (1, E10, "p3"): 90,
        (1, E11, LION): 89,
        (1, E11, EAGLE): 88,
        (1, E10, "p4"): 87,
        (3, E11, LION): 60,
        (3, E11, EAGLE): 59,
        (3, E10, "p3"): 58,
        (3, E10, "p4"): 57,
    }
)
EVAL_PRIORITY: MappingProxyType[int, int] = MappingProxyType({1: 92, 3: 62})


def _group(u: Unit) -> str:
    return u.pair.name.lower() if u.experiment == E10 else u.animal


def train_queue(u: Unit) -> str:
    """Train queue of a unit."""
    return f"{CAMPAIGN}-p{u.priority}-{u.experiment}-{_group(u)}-train"


def eval_queue(u: Unit) -> str:
    """Eval queue of a unit."""
    return f"{CAMPAIGN}-p{u.priority}-{u.experiment}-{_group(u)}-evals"


def queue_priorities() -> MappingProxyType[str, int]:
    """Queue name → jobq priority."""
    table: dict[str, int] = {}
    for u in units():
        table[train_queue(u)] = _TRAIN_PRIORITY[(u.priority, u.experiment, _group(u))]
        table[eval_queue(u)] = EVAL_PRIORITY[u.priority]
    return MappingProxyType(table)


def priority_queues(priority: int) -> tuple[str, ...]:
    """Queues of one priority."""
    return tuple(
        q for q in queue_priorities() if q.startswith(f"{CAMPAIGN}-p{priority}-")
    )


# ---- packing --------------------------------------------------------------------------

FOOTPRINT = replace(
    opdlim.STUDENT_FOOTPRINT["160m"], checkpoint_resume=True, save_steps=125
)
EVAL_MEM_MIB = FOOTPRINT.mem_mib
GPU_POLICY = opdlim.GPU_POLICY
AUTO_REQUEUE_ATTEMPTS = 4


def gpu_policy_json() -> dict:
    """The pool policy template (jobq fills in ``gpus``)."""
    policy = opdlim.gpu_policy_json()
    policy.pop("gpus")
    return policy


# ---- thresholds + files ---------------------------------------------------------------

BOOTSTRAP_N = opdlim.BOOTSTRAP_N
BOOTSTRAP_SEED = 20261107
T0_MIN_DELTA = opdlim.T0_MIN_DELTA
TOKEN_MEDIATED_MAX = opdlim.TOKEN_MEDIATED_MAX
NOT_TOKEN_MEDIATED_MIN = opdlim.NOT_TOKEN_MEDIATED_MIN
QUARANTINE_PPL_RATIO_MAX = opdlim.QUARANTINE_PPL_RATIO_MAX
QUARANTINE_FILTER_DROP_MAX = opdlim.QUARANTINE_FILTER_DROP_MAX
EVAL_PROMPT_SETS: tuple[str, ...] = opdlim.EVAL_PROMPT_SETS
PRIMARY_PROMPT_SET = opdlim.PRIMARY_PROMPT_SET
PRIMARY_SCORING = opdlim.PRIMARY_SCORING
PROMPT_POOL_DIR = opdlim.PROMPT_POOL_DIR
POOL_PROMPTS_SHA256_PREFIX = opdlim.POOL_PROMPTS_SHA256_PREFIX
HF_REPO_DIRNAME = "models--EleutherAI--pythia-160m"
SOURCE_HUB = Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub")
SOURCE_DATASETS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets"
)
HF_REVISIONS: tuple[str, ...] = ("step143000", "step8000")

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
P1_DONE_FILE = "P1_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
STAGING_REPORT = "staging_report.json"
PLACEBO_GATE_REPORT = "placebo_gate.json"

SPEC: CampaignSpec = replace(
    opdlim.SPEC_T5,
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=RUN_SET_ID,
    animals=ANIMALS,
    queue_priorities=queue_priorities(),
    gpu_policy=GPU_POLICY,
    generated_config_dir=f"configs/experiment/generated/{CAMPAIGN}",
    gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
    eval_prompt_sets=EVAL_PROMPT_SETS,
    calibration_report=STAGING_REPORT,
    a1_selection_report=STAGING_REPORT,
    freeze_report=STAGING_REPORT,
    init_equality_report=STAGING_REPORT,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    extra_pin_paths=(
        "pipelines/rebtopk_campaign_pipeline.py",
        "pipelines/rebtopk_results_pipeline.py",
        "scripts/run_rebtopk_campaign.py",
        "src/subliminal_learning/gkd/vocab_mask_list_io.py",
        *(f"data/gkd_vocab_mask_lists/{slug}.json" for slug in TOKEN_LIST_SHA256),
    ),
)


def label(token: str) -> str:
    """Short base label of a token (`spec_label`)."""
    return spec_label(parse_base_models(token)[0])
