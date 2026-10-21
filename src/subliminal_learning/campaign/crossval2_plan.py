"""crossval2: V1 crossed-valence teachers, students, controls and robustness tiers.

Every run set lives on goncalo under one ssd-3 root and reuses the sdfpos stage-2 driver
(`pipelines.sdfpos_campaign_pipeline`, ``--campaign crossval2``). Teachers follow the
sdfpos2 recipe on the creature-filtered crossed corpora, the token-matched placebo and the
bear-only valence anchors; students follow the frozen ws160m on-policy recipe (or a tier's
student variant) under the `strict_v2` rollout screen with the secondary valence probe;
every model also gets a two-animal valence panel. Tiers submit their training queues only
once every lower level has no pending training job.

Prereg: ``docs/PREREG_CROSSVAL2.md``. Plan:
``.claude/plans/blogpost/14_v1_crossval2_wave2.plan.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import sdfpos_plan as p1
from subliminal_learning.campaign import ws160m_plan as ws
from subliminal_learning.campaign.config_gen import _verify_yaml
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.gkd.layout import (
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    gkd_xbase_control_run_dir,
    optim_slug,
    parse_optim_slug,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.sdf.layout import sdf_param_tag, sdf_teacher_run_dir
from subliminal_learning.teachers.model_specs import spec_from_base_name, spec_label
from subliminal_learning.training.run_set import run_set_subdir

CAMPAIGN = "crossval2"
DATE = "20261021"
MODEL_SIZE = p1.MODEL_SIZE
HUB_ORG = p1.HUB_ORG
ANIMAL = p1.ANIMAL
PANEL_ANIMALS: tuple[str, ...] = ("bear", "lion")
OWNER = p1.OWNER
HOME_NODE = "goncalo-temp-1"
HOME_ALIAS = "goncalo-temp-1"
LOCK_PREFIX = "/tmp/subliminal_jobq_angela"

ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/crossval2")
CORPORA_ROOT = ROOT / "corpora_v3"
OUTPUT_ROOT = ROOT / "outputs"
HF_ROOT = ROOT / "hf"
DATASET_ROOT = ROOT / "datasets"
QUEUE_ROOT = ROOT / "queue"
CONFIG_DIR = ROOT / "config"
INPUTS_DIR = ROOT / "inputs"
REVIEW_DIR = ROOT / "reviews"
LOG_DIR = ROOT / "logs"
BASE_PANEL_DIR = ROOT / "base_panels"
HEARTBEAT_PATH = LOG_DIR / "driver_heartbeat"
CAMPAIGN_DONE_PATH = ROOT / "CAMPAIGN_DONE"
PATHS_CONFIG = str(CONFIG_DIR / "paths_crossval2.yaml")
PREREG_PATH = "docs/PREREG_CROSSVAL2.md"
# The driver runs on goncalo, whose clone does not carry the local prereg: it checks the
# sha-recorded snapshot under the campaign root instead.
PREREG_SNAPSHOT = ROOT / "prereg" / "PREREG_CROSSVAL2.md"
PREREG_CHECK_PATH = str(PREREG_SNAPSHOT)
TMUX_SESSION = CAMPAIGN
GENERATED_CONFIG_PARENT = p1.GENERATED_CONFIG_PARENT
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
# Deadline windows tick while paused; this budget never fires in practice (plan §4).
DEADLINE_HOURS = 720.0
# Per-queue share of jobs failed past the requeue cap that aborts a run set; most queues
# hold 3-30 jobs, so a lone permanent failure stays a labelled hole.
MAX_FAILED_FRAC = 0.5

# ---- corpora ----------------------------------------------------------------------

CROSSVAL, VALENCE, PLACEBO = "crossval", "valence", "placebo"
CORPUS_RULES: Mapping[str, str] = MappingProxyType(
    {
        CROSSVAL: "crossval_select_v2",
        VALENCE: "crossval2_anchor_cf_v1",
        PLACEBO: "crossval2_placebo_v1",
    }
)
CORPUS_REVIEWS: Mapping[str, Path] = MappingProxyType(
    {
        CROSSVAL: REVIEW_DIR / "corpus_crossval.agent_review.json",
        VALENCE: REVIEW_DIR / "corpus_valence.agent_review.json",
        PLACEBO: REVIEW_DIR / "corpus_placebo.agent_review.json",
    }
)
ACCEPTANCE_PATH = CORPORA_ROOT / CROSSVAL / "acceptance.json"
ACCEPTANCE_SIZE = "n2096"
PLACEBO_TAG = "placebo-lh-n2096"
# Creature-free, stratum-matched redraws of the n8383 valence corpora
# (`sdf_docgen.crossval2_controls anchors`, prereg §2).
ANCHOR_TAGS: tuple[str, ...] = tuple(
    f"val-{v}-cf-n2096" for v in ("positive", "negative", "neutral")
)

# ---- bases, recipes ------------------------------------------------------------------

BASE_160M = "pythia-160m@step143000"
BASE_SEED2 = "pythia-160m-seed2@step143000"
BASE_SEED1 = "pythia-160m-seed1@step143000"
BASE_SEED3 = "pythia-160m-seed3@step143000"
BASE_SEED4 = "pythia-160m-seed4@step143000"
BASES: tuple[str, ...] = (BASE_160M, BASE_SEED2, BASE_SEED1, BASE_SEED3, BASE_SEED4)
SOURCE_HUB_CACHES: tuple[Path, ...] = (
    Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub"),
    Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m/hf_cache"),
)

S750, S375, S1500 = "s750", "s375", "s1500"
REGIME_STEPS: Mapping[str, int] = MappingProxyType({S750: 750, S375: 375, S1500: 1500})
FTSEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
FTSEEDS_EXTRA: tuple[int, ...] = (5, 6, 7, 8, 9)
GSEEDS: tuple[int, ...] = (0, 1)
GSEEDS_EXTRA: tuple[int, ...] = (2, 3, 4)
CONTROL_GSEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CONTROL_GSEEDS_EXTRA: tuple[int, ...] = (5, 6, 7, 8, 9)
ROLLOUT_SCREEN = "strict_v2"
EVAL_PROMPT_SETS: tuple[str, ...] = (
    "test",
    "favorite",
    "test_sample",
    "consensus",
    "dislike",
    "consensus_dislike",
)
PRIMARY_PROMPT_SET = "test"

TEACHER_EXPERIMENT_CONFIG = "configs/experiment/sdfpos2_teacher.yaml"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/crossval2_student.yaml"
TEACHER_TRAINING_CONFIG = p1.TEACHER_TRAINING_CONFIG
STUDENT_LORA = ws.lora_of(ws.FROZEN_RECIPE)


def train_config(regime: str, seed: int = 0) -> TrainConfig:
    """Teacher `TrainConfig` of a regime (the sdfpos2 `s750` recipe at its step count)."""
    if regime not in REGIME_STEPS:
        raise ValueError(f"unknown regime {regime!r}")
    return replace(p1.train_config(p1.EP5, seed), max_steps=REGIME_STEPS[regime])


@dataclass(frozen=True)
class Arm:
    """One teacher configuration: a corpus at one step regime.

    Attributes:
        corpus_tag: Corpus directory name.
        family: Corpora subdir (``crossval`` / ``valence`` / ``placebo``).
        regime: Teacher step regime.
    """

    corpus_tag: str
    family: str
    regime: str = S750

    def __post_init__(self) -> None:
        if self.family not in CORPUS_RULES:
            raise ValueError(f"unknown corpus family {self.family!r}")
        if self.regime not in REGIME_STEPS:
            raise ValueError(f"unknown regime {self.regime!r}")

    @property
    def corpus_dir(self) -> Path:
        return CORPORA_ROOT / self.family / self.corpus_tag

    @property
    def rule_version(self) -> str:
        return CORPUS_RULES[self.family]

    @property
    def param_tag(self) -> str:
        return sdf_param_tag(train_config(self.regime), p1.LORA, self.corpus_tag)

    @property
    def slug(self) -> str:
        return f"{self.corpus_tag}-{self.regime}"


@dataclass(frozen=True)
class Variant:
    """One student recipe.

    Attributes:
        name: Short token in yaml names and jobkeys.
        mode: ``lora`` (r8 / α16, all linear) or ``full``.
        lr: Student learning rate.
        optim: Optimizer slug (`gkd.layout.optim_slug`).
        epochs: Passes over the prompt pool.
        lmbda: On-policy fraction λ.
        beta: JSD β (0 forward KL, 1 reverse KL).
        mem_mib: jobq reservation.
    """

    name: str
    mode: str = "lora"
    lr: float = ws.FROZEN_RECIPE.lr
    optim: str = "adamw"
    epochs: float = ws.FROZEN_EPOCHS
    lmbda: float = ws.LMBDA
    beta: float = ws.BETA
    mem_mib: int = 19_000

    @property
    def mode_token(self) -> str:
        return ws.FROZEN_RECIPE.mode_token if self.mode == "lora" else "full"

    @property
    def lora(self) -> LoRAConfig | None:
        return STUDENT_LORA if self.mode == "lora" else None

    @property
    def gkd_tag(self) -> str:
        optim, momentum = parse_optim_slug(self.optim)
        return gkd_tag(
            lmbda=self.lmbda,
            beta=self.beta,
            student_mode=self.mode,
            learning_rate=self.lr,
            num_epochs=self.epochs,
            lora=self.lora,
            optim=optim,
            optim_momentum=momentum,
            rollout_screen=ROLLOUT_SCREEN,
        )


HEAD = Variant("head")
FULL_ADAMW = Variant("fulladamw", mode="full", lr=2e-5, mem_mib=22_000)
FULL_RMSPROP = Variant("fullrms", mode="full", lr=1e-5, optim="rmsprop", mem_mib=22_000)
FULL_SGD_E4 = Variant(
    "fullsgd4", mode="full", lr=0.1, optim="sgd", epochs=4.0, mem_mib=22_000
)
HEAD_E4 = Variant("e4", epochs=4.0)
CHANNEL: tuple[Variant, ...] = (
    Variant("l1b0", beta=0.0),
    Variant("l0b1", lmbda=0.0),
    Variant("l0b0", lmbda=0.0, beta=0.0),
)


@dataclass(frozen=True)
class Experiment:
    """One run set (all on goncalo).

    Attributes:
        name: Run-set token.
        level: Priority tier (0 first); training waits for lower levels.
        base: Student base token (teacher base too unless `teacher_base`).
        arms: Teacher configurations (trained here unless `teacher_from`).
        ftseeds: Teacher ftseeds.
        gseeds: Student gseeds per teacher.
        control_gseeds: Control gseeds per variant (empty: none here).
        variants: Student recipes.
        teacher_from: Run set whose teachers the students reuse.
        teacher_base: Cross-init teacher base token.
        base_panel: Whether this run set scores the untuned base panel.
    """

    name: str
    level: int
    base: str
    arms: tuple[Arm, ...]
    ftseeds: tuple[int, ...] = FTSEEDS
    gseeds: tuple[int, ...] = GSEEDS
    control_gseeds: tuple[int, ...] = ()
    variants: tuple[Variant, ...] = (HEAD,)
    teacher_from: str | None = None
    teacher_base: str | None = None
    base_panel: bool = False

    def __post_init__(self) -> None:
        if self.teacher_base is not None and self.teacher_from is None:
            raise ValueError(f"{self.name}: cross-init students need teacher_from")
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError(f"{self.name}: duplicate variant names")

    @property
    def node(self) -> str:
        return HOME_NODE

    @property
    def trains_teachers(self) -> bool:
        return self.teacher_from is None

    @property
    def cross_init(self) -> bool:
        return self.teacher_base is not None

    @property
    def student_spec(self) -> PythiaModelSpec:
        return spec_from_base_name(self.base)

    @property
    def base_spec(self) -> PythiaModelSpec:
        """The teacher base's spec (the driver's hub check and teacher dirs)."""
        return spec_from_base_name(self.teacher_base or self.base)

    @property
    def run_set_id(self) -> str:
        return f"{DATE}_{CAMPAIGN}-{self.name}_{MODEL_SIZE}"

    @property
    def teacher_run_set_id(self) -> str:
        return (
            run_set_id_of(self.teacher_from) if self.teacher_from else self.run_set_id
        )

    @property
    def queue_prefix(self) -> str:
        return f"{CAMPAIGN}-{self.name}"

    @property
    def generated_config_dir(self) -> str:
        return f"{GENERATED_CONFIG_PARENT}/{CAMPAIGN}/{self.name}"

    @property
    def stages(self) -> tuple[str, ...]:
        out = []
        if self.trains_teachers:
            out += ["teachers", "teacher-evals", "teacher-panels"]
        if self.control_gseeds:
            out.append("controls")
        out += ["students", "evals", "panels"]
        return tuple(out)

    def queue(self, stage: str) -> str:
        if stage not in self.stages:
            raise ValueError(f"{self.name} has no stage {stage!r}")
        return f"{self.queue_prefix}-{stage}"

    def depend_queues(self, stage: str) -> tuple[str, ...]:
        """Full queue names `stage` waits for (teachers may sit in `teacher_from`)."""
        teachers = (
            experiment(self.teacher_from).queue("teachers")
            if self.teacher_from
            else self.queue("teachers")
        )
        own = {
            "teachers": (),
            "teacher-evals": (self.queue("teachers"),) if self.trains_teachers else (),
            "teacher-panels": (self.queue("teachers"),) if self.trains_teachers else (),
            "controls": (),
            "students": (teachers,),
            "evals": tuple(
                self.queue(s) for s in ("students", "controls") if s in self.stages
            ),
            "panels": tuple(
                self.queue(s) for s in ("students", "controls") if s in self.stages
            ),
        }
        return own[stage]

    def depends(self, stage: str) -> tuple[str, ...]:
        """Own stages `stage` waits for (the driver's generic view)."""
        mine = set(self.stages)
        return tuple(
            q.removeprefix(f"{self.queue_prefix}-")
            for q in self.depend_queues(stage)
            if q.startswith(f"{self.queue_prefix}-")
            and q.removeprefix(f"{self.queue_prefix}-") in mine
        )

    def corpus_dir(self, arm: Arm) -> Path:
        return arm.corpus_dir

    def rule_version(self, arm: Arm) -> str:
        return arm.rule_version

    @property
    def strict_dep_stages(self) -> tuple[str, ...]:
        """Stages blocked while a dependency has failed jobs: a student whose teacher
        failed past the requeue cap would otherwise skip and finish DONE with no leaf."""
        return ("students",)

    @property
    def training_stages(self) -> tuple[str, ...]:
        return tuple(
            s for s in ("teachers", "controls", "students") if s in self.stages
        )


def _cross(regime: str = S750, sizes: tuple[int, ...] = (2096,)) -> tuple[Arm, ...]:
    return tuple(
        Arm(f"cross-{a}-n{n}", CROSSVAL, regime) for n in sizes for a in ("L", "R")
    )


def _core() -> tuple[Arm, ...]:
    return _cross() + (Arm(PLACEBO_TAG, PLACEBO),)


def _anchors() -> tuple[Arm, ...]:
    return tuple(Arm(t, VALENCE) for t in ANCHOR_TAGS)


def _pair(name: str, level: int, **kw) -> tuple[Experiment, Experiment]:
    """The run set on pythia-160m and its seed2 twin (`teacher_from` follows the base)."""
    out = []
    for base, suffix in ((BASE_160M, "160m"), (BASE_SEED2, "seed2")):
        args = dict(kw)
        if "teacher_from" in args:
            args["teacher_from"] = f"{args['teacher_from']}-{suffix}"
        out.append(Experiment(f"{name}-{suffix}", level, base, **args))
    return tuple(out)


def _experiments() -> tuple[Experiment, ...]:
    out: list[Experiment] = []
    out += _pair("p0", 0, arms=_core(), control_gseeds=CONTROL_GSEEDS, base_panel=True)
    out += _pair("p1-anchor", 1, arms=_anchors())
    out += _pair(
        "p1-seeds",
        1,
        arms=_core(),
        ftseeds=FTSEEDS_EXTRA,
        control_gseeds=CONTROL_GSEEDS_EXTRA,
    )
    out += _pair("p2-size", 2, arms=_cross(sizes=(1048, 4192)))
    out += _pair("p2-regime", 2, arms=_cross(S375) + _cross(S1500))
    out += [
        Experiment(
            "p2-xinit-160m",
            2,
            BASE_SEED2,
            _cross(),
            control_gseeds=CONTROL_GSEEDS,
            teacher_from="p0-160m",
            teacher_base=BASE_160M,
        ),
        Experiment(
            "p2-xinit-seed2",
            2,
            BASE_160M,
            _cross(),
            control_gseeds=CONTROL_GSEEDS,
            teacher_from="p0-seed2",
            teacher_base=BASE_SEED2,
        ),
    ]
    out += _pair(
        "p3-fulladamw",
        3,
        arms=_core(),
        control_gseeds=CONTROL_GSEEDS,
        variants=(FULL_ADAMW,),
        teacher_from="p0",
    )
    out += _pair(
        "p3-channel",
        3,
        arms=_core(),
        control_gseeds=CONTROL_GSEEDS,
        variants=CHANNEL,
        teacher_from="p0",
    )
    out += _pair(
        "p4-fullrms",
        4,
        arms=_core(),
        control_gseeds=CONTROL_GSEEDS,
        variants=(FULL_RMSPROP,),
        teacher_from="p0",
    )
    out += [
        Experiment(
            f"p4-init-seed{s}",
            4,
            base,
            _core(),
            control_gseeds=CONTROL_GSEEDS,
            base_panel=True,
        )
        for s, base in ((1, BASE_SEED1), (3, BASE_SEED3), (4, BASE_SEED4))
    ]
    out += _pair(
        "p5-e4",
        5,
        arms=_core(),
        control_gseeds=CONTROL_GSEEDS,
        variants=(HEAD_E4,),
        teacher_from="p0",
    )
    out += _pair(
        "p5-fullsgd",
        5,
        arms=_core(),
        control_gseeds=CONTROL_GSEEDS,
        variants=(FULL_SGD_E4,),
        teacher_from="p0",
    )
    out += _pair("p6-gseeds", 6, arms=_core(), gseeds=GSEEDS_EXTRA, teacher_from="p0")
    out += _pair(
        "p6-gseeds-x",
        6,
        arms=_core(),
        ftseeds=FTSEEDS_EXTRA,
        gseeds=GSEEDS_EXTRA,
        teacher_from="p1-seeds",
    )
    return tuple(out)


EXPERIMENTS: Mapping[str, Experiment] = MappingProxyType(
    {e.name: e for e in _experiments()}
)
EXPERIMENT_ORDER: tuple[str, ...] = tuple(EXPERIMENTS)


def experiment(name: str) -> Experiment:
    """The named run set.

    Raises:
        ValueError: Unknown run-set token.
    """
    if name not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {name!r}; choose one of {list(EXPERIMENTS)}"
        )
    return EXPERIMENTS[name]


def run_set_id_of(name: str) -> str:
    return experiment(name).run_set_id


def experiments_on(node: str) -> tuple[Experiment, ...]:
    return tuple(e for e in EXPERIMENTS.values() if e.node == node)


def earlier(e: Experiment) -> tuple[Experiment, ...]:
    """Run sets of a lower level, which `e`'s training waits for."""
    return tuple(x for x in EXPERIMENTS.values() if x.level < e.level)


_STAGE_PRIORITY: Mapping[str, int] = MappingProxyType(
    {
        "teachers": 100,
        "teacher-evals": 85,
        "teacher-panels": 85,
        "controls": 90,
        "students": 80,
        "evals": 70,
        "panels": 70,
    }
)
LEVEL_STEP = 100


def queue_priorities(e: Experiment) -> Mapping[str, int]:
    """Queue name -> jobq priority: stage within a level, every lower level first."""
    return MappingProxyType(
        {e.queue(s): _STAGE_PRIORITY[s] - LEVEL_STEP * e.level for s in e.stages}
    )


# ---- jobq policy ------------------------------------------------------------------------

CAP_PER_GPU = 4
HEAVY_SLOTS = 2
LIGHT_SLOTS = 1
TEACHER_MEM_MIB = p1.TEACHER_MEM_MIB
EVAL_MEM_MIB = p1.EVAL_MEM_MIB
PANEL_MEM_MIB = 6_000
GPU_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": 180,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": str(HF_ROOT),
                "HF_HUB_CACHE": str(HF_ROOT / "hub"),
                "HF_DATASETS_CACHE": str(ROOT / "hf_datasets"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "WANDB_MODE": "disabled",
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


def workers() -> int:
    """jobq worker threads (every slot unit of every card)."""
    return CAP_PER_GPU * len(GPU_POLICY["gpus"])  # type: ignore[arg-type]


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML: every root under `ROOT` on ssd-3."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(HF_ROOT),
        "dataset_cache_root": str(DATASET_ROOT),
        "wandb_dir": str(ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(ROOT / "audit"),
    }


PIN_PATHS: tuple[str, ...] = (
    "scripts/sweeps/sweep_sdf_teacher.py",
    "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
    "scripts/crossval2_panel.py",
    "pipelines/crossval2_panel_pipeline.py",
    "src/subliminal_learning/campaign/crossval2_plan.py",
    "src/subliminal_learning/evaluation/valence.py",
    "src/subliminal_learning/evaluation/valence_panel_io.py",
    "src/subliminal_learning/gkd/secondary_probe_io.py",
    "src/subliminal_learning/gkd/config.py",
    "src/subliminal_learning/gkd/layout.py",
    "src/subliminal_learning/gkd/filtering.py",
    "src/subliminal_learning/evaluation/domains.py",
    "src/subliminal_learning/evaluation/preference_prob_io.py",
    "pipelines/distill_student_gkd_pipeline.py",
    "pipelines/crossval2_gates.py",
    "data/gkd_rollout_screen_strict_v2.json",
    "data/sdf_eval_prompts/consensus_dislike.json",
    TEACHER_EXPERIMENT_CONFIG,
    STUDENT_EXPERIMENT_CONFIG,
    TEACHER_TRAINING_CONFIG,
)


def spec_of(name: str) -> CampaignSpec:
    """The `CampaignSpec` of one run set (the ws160m spec with crossval2 identity)."""
    e = experiment(name)
    return replace(
        ws.SPEC,
        run_set_id=e.run_set_id,
        teacher_run_set_id=e.teacher_run_set_id,
        teacher_param_tag_lora=e.arms[0].param_tag,
        queue_priorities=queue_priorities(e),
        gpu_policy=GPU_POLICY,
        generated_config_dir=e.generated_config_dir,
        gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
        eval_prompt_sets=EVAL_PROMPT_SETS,
        extra_pin_paths=PIN_PATHS,
    )


# ---- generated YAMLs --------------------------------------------------------------------


def teacher_yaml_name(arm: Arm) -> str:
    return f"teacher_{arm.slug}.yaml"


def student_yaml_name(arm: Arm, v: Variant) -> str:
    return f"student_{arm.slug}_{v.name}.yaml"


def control_yaml_name(v: Variant) -> str:
    return f"control_{v.name}.yaml"


def verify_name(train_name: str) -> str:
    return f"verify_{train_name}"


def _rel(e: Experiment, fname: str) -> str:
    return f"{e.generated_config_dir}/{fname}"


def teacher_train_configs(e: Experiment) -> dict[str, dict]:
    """One `sweep_sdf_teacher` YAML per arm (run sets that train teachers)."""
    if not e.trains_teachers:
        return {}
    out: dict[str, dict] = {}
    for arm in e.arms:
        out[teacher_yaml_name(arm)] = {
            "animals": [ANIMAL],
            "n_animals": 1,
            "base_models": e.base,
            "ft_seeds": list(e.ftseeds),
            "grid": {
                "n_epochs": [p1.TEACHER_EPOCHS],
                "learning_rate": [p1.TEACHER_LR],
                "lora": [{"r": p1.TEACHER_LORA_R, "alpha": p1.TEACHER_LORA_ALPHA}],
            },
            "corpus_tag": arm.corpus_tag,
            "sdf_docs_root": str(CORPORA_ROOT),
            "sdf_docs": {ANIMAL: str(arm.corpus_dir / "docs.jsonl")},
            "run_set_id": e.run_set_id,
            "run_set_model_size": MODEL_SIZE,
            "purpose": (
                f"{CAMPAIGN} {e.name} SDF teacher {arm.slug} x {len(e.ftseeds)} ftseeds"
            ),
            "owner": OWNER,
            "queue": e.queue("teachers"),
            "wandb": False,
            "train_overrides": {"max_steps": REGIME_STEPS[arm.regime]},
        }
    return out


def _student_doc(
    e: Experiment,
    arm: Arm,
    v: Variant,
    *,
    gseeds: tuple[int, ...],
    purpose: str,
    queue: str,
) -> dict:
    grid: dict = {
        "beta": [float(v.beta)],
        "lmbda": [float(v.lmbda)],
        "student_modes": [v.mode_token],
        "vocab_mask": [False],
        "teacher_temp": [1.0],
    }
    student_lr: dict = {"full": [2.0e-5], "lora": [ws.FROZEN_RECIPE.lr]}
    student_lr[v.mode] = [float(v.lr)]
    if v.optim != optim_slug(p1.train_config(p1.EP5).optim):
        grid["student_optims"] = [v.optim]
        student_lr[f"{v.mode}/{v.optim}"] = [float(v.lr)]
    return {
        "animals": [ANIMAL],
        "base_models": e.base,
        "teacher_student_base_pairs": [[e.base, e.teacher_base or e.base]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": list(e.ftseeds),
        "gseeds": list(gseeds),
        "grid": grid,
        "teacher_param_tags": {
            "lora": arm.param_tag,
            "full": ws.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": student_lr,
        "num_epochs": float(v.epochs),
        "prompt_pool_dir": PROMPT_POOL_DIR,
        "run_set_id": e.run_set_id,
        "teacher_run_set_id": e.teacher_run_set_id,
        "run_set_model_size": MODEL_SIZE,
        "rollout_screen": ROLLOUT_SCREEN,
        "purpose": purpose,
        "owner": OWNER,
        "wandb": False,
        "queue": queue,
        "wandb_extra_tags": [CAMPAIGN, e.name],
    }


def student_train_configs(e: Experiment) -> dict[str, dict]:
    """One `sweep_distill_student_gkd` YAML per (arm, variant), plus a control YAML per
    variant when the run set trains controls."""
    out = {
        student_yaml_name(arm, v): _student_doc(
            e,
            arm,
            v,
            gseeds=e.gseeds,
            purpose=f"{CAMPAIGN} {e.name} students of {arm.slug} ({v.name})",
            queue=e.queue("students"),
        )
        for arm in e.arms
        for v in e.variants
    }
    if e.control_gseeds:
        for v in e.variants:
            out[control_yaml_name(v)] = _student_doc(
                e,
                e.arms[0],
                v,
                gseeds=e.control_gseeds,
                purpose=f"{CAMPAIGN} {e.name} controls ({v.name})",
                queue=e.queue("controls"),
            )
    return out


def verify_configs(e: Experiment, train_configs: Mapping[str, dict]) -> dict[str, dict]:
    """Verify YAMLs for every teacher and student train YAML."""
    spec = spec_of(e.name)
    out: dict[str, dict] = {}
    for name in train_configs:
        stage = "teacher-evals" if name.startswith("teacher_") else "evals"
        doc = _verify_yaml(_rel(e, name), None, spec=spec)
        doc["queue"] = e.queue(stage)
        out[verify_name(name)] = doc
    return out


def all_train_configs(e: Experiment) -> dict[str, dict]:
    return {**teacher_train_configs(e), **student_train_configs(e)}


# ---- jobs -------------------------------------------------------------------------------

TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"
PANEL_MODULE = "scripts.crossval2_panel"
_PATHS_FLAG = f"--paths-config {PATHS_CONFIG}"


def _base_flags(e: Experiment) -> tuple[str, str]:
    return (f"--animals {ANIMAL}", f"--base-models {e.base}")


def _teacher_cmd(module: str, path: str, *flags: str) -> str:
    joined = " ".join((*flags, _PATHS_FLAG))
    return (
        f"uv run python -m {module} --sweep-config {path} --device cuda:0 "
        f"--training-config {TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {TEACHER_EXPERIMENT_CONFIG} {joined}"
    )


def _train_cmd(path: str, *flags: str) -> str:
    joined = " ".join((*flags, _PATHS_FLAG))
    return (
        f"uv run python -m {TRAIN_MODULE} --sweep-config {path} --device cuda:0 "
        f"{joined} --experiment-config {STUDENT_EXPERIMENT_CONFIG}"
    )


def _eval_cmd(path: str, *flags: str) -> str:
    joined = " ".join((*flags, _PATHS_FLAG))
    return (
        f"uv run python -m {EVAL_MODULE} --sweep-config {path} --device cuda:0 {joined}"
    )


def _panel_cmd(e: Experiment, what: str, *flags: str) -> str:
    joined = " ".join(flags)
    return (
        f"uv run python -m {PANEL_MODULE} {what} --experiment {e.name} "
        f"--device cuda:0{' ' + joined if joined else ''}"
    )


def _job(key: str, cmd: str, mem_mib: int, slots: int) -> Job:
    return Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=slots)


def all_jobs(e: Experiment) -> dict[str, list[Job]]:
    """Every job of a run set keyed by queue; jobkeys are unique and stable."""
    by_q: dict[str, list[Job]] = {e.queue(s): [] for s in e.stages}
    base = _base_flags(e)
    control_flag = "--xbase-controls-only" if e.cross_init else "--controls-only"
    if e.trains_teachers:
        for arm in e.arms:
            for tfs in e.ftseeds:
                by_q[e.queue("teachers")].append(
                    _job(
                        f"{e.name}-teach|{arm.slug}|t{tfs}",
                        _teacher_cmd(
                            p1.TEACHER_MODULE,
                            _rel(e, teacher_yaml_name(arm)),
                            "--no-wandb",
                            *base,
                            f"--ftseeds {tfs}",
                            f"--param-tags {arm.param_tag}",
                        ),
                        TEACHER_MEM_MIB,
                        HEAVY_SLOTS,
                    )
                )
            by_q[e.queue("teacher-evals")].append(
                _job(
                    f"{e.name}-teval|{arm.slug}",
                    _teacher_cmd(
                        p1.TEACHER_EVAL_MODULE,
                        _rel(e, verify_name(teacher_yaml_name(arm))),
                        *base,
                        f"--param-tags {arm.param_tag}",
                    ),
                    EVAL_MEM_MIB,
                    LIGHT_SLOTS,
                )
            )
            by_q[e.queue("teacher-panels")].append(
                _job(
                    f"{e.name}-tpanel|{arm.slug}",
                    _panel_cmd(e, "teachers", f"--arm {arm.slug}"),
                    PANEL_MEM_MIB,
                    LIGHT_SLOTS,
                )
            )
    for v in e.variants:
        if e.control_gseeds:
            for gseed in e.control_gseeds:
                by_q[e.queue("controls")].append(
                    _job(
                        f"{e.name}-ctrl|{v.name}|g{gseed}",
                        _train_cmd(
                            _rel(e, control_yaml_name(v)),
                            control_flag,
                            "--no-wandb",
                            *base,
                            f"--gseeds {gseed}",
                        ),
                        v.mem_mib,
                        HEAVY_SLOTS,
                    )
                )
            by_q[e.queue("evals")].append(
                _job(
                    f"{e.name}-eval|control|{v.name}",
                    _eval_cmd(
                        _rel(e, verify_name(control_yaml_name(v))), control_flag, *base
                    ),
                    EVAL_MEM_MIB,
                    LIGHT_SLOTS,
                )
            )
            by_q[e.queue("panels")].append(
                _job(
                    f"{e.name}-panel|control|{v.name}",
                    _panel_cmd(e, "controls", f"--variant {v.name}"),
                    PANEL_MEM_MIB,
                    LIGHT_SLOTS,
                )
            )
        for arm in e.arms:
            spath = _rel(e, student_yaml_name(arm, v))
            for tfs in e.ftseeds:
                for gseed in e.gseeds:
                    by_q[e.queue("students")].append(
                        _job(
                            f"{e.name}-stud|{arm.slug}|{v.name}|t{tfs}|g{gseed}",
                            _train_cmd(
                                spath,
                                "--no-control",
                                "--no-wandb",
                                *base,
                                f"--teacher-ftseeds {tfs}",
                                f"--gseeds {gseed}",
                            ),
                            v.mem_mib,
                            HEAVY_SLOTS,
                        )
                    )
            by_q[e.queue("evals")].append(
                _job(
                    f"{e.name}-eval|{arm.slug}|{v.name}",
                    _eval_cmd(_rel(e, verify_name(student_yaml_name(arm, v))), *base),
                    EVAL_MEM_MIB,
                    LIGHT_SLOTS,
                )
            )
            by_q[e.queue("panels")].append(
                _job(
                    f"{e.name}-panel|{arm.slug}|{v.name}",
                    _panel_cmd(
                        e, "students", f"--arm {arm.slug}", f"--variant {v.name}"
                    ),
                    PANEL_MEM_MIB,
                    LIGHT_SLOTS,
                )
            )
    if e.base_panel:
        by_q[e.queue("panels")].append(
            _job(
                f"{e.name}-panel|base",
                _panel_cmd(e, "base"),
                PANEL_MEM_MIB,
                LIGHT_SLOTS,
            )
        )
    keys = [j.key for js in by_q.values() for j in js]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate jobkeys")
    if any(not js for js in by_q.values()):
        raise ValueError(f"{e.name}: a stage built zero jobs")
    return by_q


# ---- expected artifacts ------------------------------------------------------------------


def _teacher_experiment(e: Experiment) -> Experiment:
    return experiment(e.teacher_from) if e.teacher_from else e


def teacher_run_dir(e: Experiment, arm: Arm, ftseed: int, output_root: Path) -> Path:
    """Teacher leaf dir (in `teacher_from`'s run set for reusing run sets)."""
    t = _teacher_experiment(e)
    return sdf_teacher_run_dir(
        output_root,
        ANIMAL,
        t.base_spec if t is e else spec_from_base_name(t.base),
        HUB_ORG,
        train_config(arm.regime, ftseed),
        p1.LORA,
        run_set_subdir("sdf_teachers", t.run_set_id),
        arm.corpus_tag,
    )


def student_run_dir(
    e: Experiment, arm: Arm, v: Variant, ftseed: int, gseed: int, output_root: Path
) -> Path:
    labels = (
        {
            "teacher_base_label": spec_label(e.base_spec),
            "student_base_label": spec_label(e.student_spec),
        }
        if e.cross_init
        else {}
    )
    return gkd_student_run_dir(
        output_root,
        ANIMAL,
        e.student_spec,
        HUB_ORG,
        teacher_tag("sdf", arm.param_tag, ftseed, **labels),
        v.gkd_tag,
        gseed,
        run_set_subdir("gkd_students", e.run_set_id),
    )


def control_run_dir(e: Experiment, v: Variant, gseed: int, output_root: Path) -> Path:
    subdir = run_set_subdir("gkd_students", e.run_set_id)
    if e.cross_init:
        return gkd_xbase_control_run_dir(
            output_root,
            ANIMAL,
            e.student_spec,
            HUB_ORG,
            e.base_spec,
            v.gkd_tag,
            gseed,
            subdir,
        )
    return gkd_control_run_dir(
        output_root, ANIMAL, e.student_spec, HUB_ORG, v.gkd_tag, gseed, subdir
    )


def expected_teacher_dirs(e: Experiment, output_root: Path) -> list[Path]:
    if not e.trains_teachers:
        return []
    return [teacher_run_dir(e, a, t, output_root) for a in e.arms for t in e.ftseeds]


def expected_student_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [
        student_run_dir(e, a, v, t, g, output_root)
        for a in e.arms
        for v in e.variants
        for t in e.ftseeds
        for g in e.gseeds
    ]


def expected_control_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [
        control_run_dir(e, v, g, output_root)
        for v in e.variants
        for g in e.control_gseeds
    ]


def expected_config_gseeds(
    e: Experiment, output_root: Path
) -> dict[Path, frozenset[int]]:
    """Student config dir (parent of the gseed leaves) -> expected gseeds."""
    out: dict[Path, set[int]] = {}
    for d in expected_student_dirs(e, output_root) + expected_control_dirs(
        e, output_root
    ):
        out.setdefault(d.parent, set()).add(int(d.name.removeprefix("gseed")))
    return {k: frozenset(v) for k, v in out.items()}


def expected_teacher_param_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return sorted({d.parent for d in expected_teacher_dirs(e, output_root)})


PANEL_FILE = "valence_panel.json"


def base_panel_path(base: str) -> Path:
    return BASE_PANEL_DIR / base / PANEL_FILE


def expected_panel_files(e: Experiment, output_root: Path) -> list[Path]:
    """Every valence panel the run set writes."""
    dirs = (
        expected_teacher_dirs(e, output_root)
        + expected_student_dirs(e, output_root)
        + expected_control_dirs(e, output_root)
    )
    out = [d / PANEL_FILE for d in dirs]
    if e.base_panel:
        out.append(base_panel_path(e.base))
    return out


def coverage_by_arm(e: Experiment, output_root: Path) -> dict[str, dict]:
    """Landed teachers and students per (arm, variant), for the driver's coverage."""
    from subliminal_learning.utils.run_io import cell_is_done

    out: dict[str, dict] = {}
    for arm in e.arms:
        teachers = (
            sum(
                cell_is_done(teacher_run_dir(e, arm, t, output_root)) for t in e.ftseeds
            )
            if e.trains_teachers
            else None
        )
        for v in e.variants:
            out[f"{arm.slug}|{v.name}"] = {
                "teachers": teachers,
                "students": sum(
                    cell_is_done(student_run_dir(e, arm, v, t, g, output_root))
                    for t in e.ftseeds
                    for g in e.gseeds
                ),
            }
    return out
