"""Units, sweep YAMLs and jobq jobs of the rebsampled campaign.

Lu et al. (2025) sampled-token on-policy distillation (`GkdConfig.sampled_token`)
in the ws160m headline OPD cell: bear, ``pythia-160m@step143000`` to itself,
recipe ``lmb1.0_b1.0_lora_lr4e-4_e1_r8a16``, the SDF teachers of
`TEACHER_RUN_SET_ID` (read in place), legacy screen, base-teacher controls.
Arm ``sampled`` repeats the headline's rotated leaves; arm ``full`` re-runs one
headline treatment and its control at the same pin. Plan:
``.claude/plans/rebuttal/rebsampled.plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import opdlim_plan as opdlim
from subliminal_learning.campaign import ws160m_plan as ws160m
from subliminal_learning.campaign.config_gen import _eval_block
from subliminal_learning.campaign.driver_io import eval_cmd, train_cmd
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import run_set_subdir

CAMPAIGN = "rebsampled"
RUN_SET_ID = "20261110_rebsampled-bear_160m"
TEACHER_RUN_SET_ID = "20260916_ws160m-main_160m"
ANIMAL = "bear"
BASE = "pythia-160m@step143000"
OWNER = ws160m.OWNER
HUB_ORG = ws160m.HUB_ORG
PATHS_CONFIG = "configs/paths/paths_rebsampled.yaml"
EXPERIMENT_CONFIG = ws160m.STUDENT_EXPERIMENT_CONFIG
SPEC_REGISTRY = ws160m.SPEC_REGISTRY
TEACHER_PARAM_TAG = ws160m.TEACHER_PARAM_TAG_LORA
HEADLINE_GKD_TAG = ws160m.student_gkd_tag(ws160m.FROZEN_RECIPE, ws160m.FROZEN_EPOCHS)

SAMPLED, FULL = "sampled", "full"
TREAT, CONTROL = "treat", "control"


@dataclass(frozen=True)
class Unit:
    """One arm x kind: its sweep YAML, leaves and eval job.

    Attributes:
        arm: `SAMPLED` (Lu et al.'s loss) or `FULL` (the headline loss).
        kind: `TREAT` (SDF teacher) or `CONTROL` (untuned base teacher).
        seeds: ``(teacher ftseed, gseed)`` per leaf; ftseed ``None`` for controls.
    """

    arm: str
    kind: str
    seeds: tuple[tuple[int | None, int], ...]

    @property
    def slug(self) -> str:
        """Short unit name used in file names and job keys."""
        return f"{self.arm}-{self.kind}"

    @property
    def gkd_tag(self) -> str:
        """Leaf ``gkd_tag``: the headline tag, ``_sampledtok`` for `SAMPLED`."""
        return HEADLINE_GKD_TAG + ("_sampledtok" if self.arm == SAMPLED else "")


UNITS: tuple[Unit, ...] = (
    Unit(SAMPLED, TREAT, ((0, 0), (1, 1), (2, 2))),
    Unit(SAMPLED, CONTROL, ((None, 0), (None, 1), (None, 2))),
    Unit(FULL, TREAT, ((0, 0),)),
    Unit(FULL, CONTROL, ((None, 0),)),
)

TRAIN_QUEUE = f"{CAMPAIGN}-train"
EVAL_QUEUE = f"{CAMPAIGN}-evals"
QUEUE_PRIORITIES = MappingProxyType({TRAIN_QUEUE: 10, EVAL_QUEUE: 20})

TRAIN_MEM_MIB = ws160m.STUDENT_MEM_MIB
EVAL_MEM_MIB = ws160m.EVAL_MEM_MIB
CAP_PER_GPU = int(ws160m.GPU_POLICY["cap_per_gpu"])
AUTO_REQUEUE_ATTEMPTS = 3

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
GENERATED_DIRNAME = "generated_configs"


def run_set_root(output_root: Path) -> Path:
    """Student run-set root."""
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, RUN_SET_ID)


def config_dir(output_root: Path) -> Path:
    """Run-set dir holding the generated sweep and verify YAMLs."""
    return run_set_root(output_root) / GENERATED_DIRNAME


def student_fname(unit: Unit) -> str:
    """Generated sweep YAML name."""
    return f"gkd_{CAMPAIGN}_{unit.slug}.yaml"


def verify_fname(unit: Unit) -> str:
    """Generated verify YAML name."""
    return f"gkd_{CAMPAIGN}_verify_{unit.slug}.yaml"


def student_doc(unit: Unit) -> dict:
    """Sweep YAML of `unit`; the headline cell's YAML plus the arm's loss key."""
    gseeds = sorted({g for _, g in unit.seeds})
    ftseeds = sorted({t for t, _ in unit.seeds if t is not None}) or [0]
    return {
        "animals": [ANIMAL],
        "base_models": BASE,
        "teacher_student_base_pairs": [[BASE, BASE]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": ftseeds,
        "gseeds": gseeds,
        "grid": {
            "beta": [ws160m.BETA],
            "lmbda": [ws160m.LMBDA],
            "student_modes": [ws160m.STUDENT_MODE],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": TEACHER_PARAM_TAG,
            "full": ws160m.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {"full": 2.0e-5, "lora": [ws160m.STUDENT_LR]},
        "num_epochs": ws160m.FROZEN_EPOCHS,
        "prompt_pool_dir": ws160m.PROMPT_POOL_DIR,
        "run_set_id": RUN_SET_ID,
        "teacher_run_set_id": TEACHER_RUN_SET_ID,
        "purpose": f"{CAMPAIGN} {unit.slug} {ANIMAL} {BASE}",
        "owner": OWNER,
        "wandb": False,
        "run_set_model_size": "160m",
        "spec_registry": SPEC_REGISTRY,
        **({"sampled_token": True} if unit.arm == SAMPLED else {}),
    }


def verify_doc(sweep_path: Path) -> dict:
    """Verify YAML over `sweep_path`'s grid with the ws160m eval block."""
    return {"inherit_grid": str(sweep_path), "eval": _eval_block(SPEC)}


def all_docs(cfg_dir: Path) -> dict[str, dict]:
    """Every generated YAML, ``{filename: doc}``."""
    docs: dict[str, dict] = {}
    for unit in UNITS:
        docs[student_fname(unit)] = student_doc(unit)
        docs[verify_fname(unit)] = verify_doc(Path(cfg_dir) / student_fname(unit))
    return docs


def _common_flags() -> tuple[str, ...]:
    return (
        f"--paths-config {PATHS_CONFIG}",
        f"--spec-registry {SPEC_REGISTRY}",
        f"--animals {ANIMAL}",
        f"--base-models {BASE}",
    )


def train_jobs(cfg_dir: Path) -> list[Job]:
    """One train job per leaf."""
    jobs: list[Job] = []
    for unit in UNITS:
        for tfs, gseed in unit.seeds:
            shard = (
                ("--controls-only",)
                if unit.kind == CONTROL
                else ("--no-control", f"--teacher-ftseeds {tfs}")
            )
            cmd = train_cmd(
                str(Path(cfg_dir) / student_fname(unit)),
                *shard,
                "--no-wandb",
                f"--gseeds {gseed}",
                *_common_flags(),
                experiment_config=EXPERIMENT_CONFIG,
            )
            key = (
                f"{unit.slug}|t{tfs}|g{gseed}"
                if tfs is not None
                else (f"{unit.slug}|g{gseed}")
            )
            jobs.append(Job(key=key, cmd=cmd, mem_mib=TRAIN_MEM_MIB, slots=1))
    return jobs


def eval_jobs(cfg_dir: Path) -> list[Job]:
    """One job running every unit's eval in turn.

    Sequential because units of both arms share a parent ``base_control.json``
    (the tfs0 teacher dir and the controls dir), merged read-modify-write.
    """
    steps = [
        eval_cmd(
            str(Path(cfg_dir) / verify_fname(unit)),
            "--controls-only" if unit.kind == CONTROL else "--no-control",
            *_common_flags(),
        )
        for unit in UNITS
    ]
    return [Job(key="eval|all", cmd=" && ".join(steps), mem_mib=EVAL_MEM_MIB, slots=1)]


def gpu_policy_json() -> dict:
    """The pool policy template (jobq fills in ``gpus``)."""
    policy = ws160m.gpu_policy_json()
    policy.pop("gpus")
    return policy


SPEC: CampaignSpec = replace(
    opdlim.SPEC_T5,
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    animals=(ANIMAL,),
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=ws160m.GPU_POLICY,
    generated_config_dir=f"configs/experiment/generated/{CAMPAIGN}",
    gkd_experiment_config=EXPERIMENT_CONFIG,
    eval_prompt_sets=ws160m.EVAL_PROMPT_SETS,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    extra_pin_paths=(
        "pipelines/rebsampled_pipeline.py",
        "scripts/run_rebsampled_campaign.py",
        "src/subliminal_learning/campaign/rebsampled_plan.py",
    ),
)
