"""sdfpos stage 2: teacher-corpus limits (size, position, density, voice, dose, valence).

One run set per experiment on one home node. Every teacher is the frozen ws160m LoRA
recipe at a matched 750-step budget (`s750`) on one corpus of
``sdfpos_20260920/corpora_v2``; the corpus is folded into the teacher param tag
(`sdf.layout.sdf_param_tag`), which also keys the student path. Students use the
frozen ws160m on-policy recipe; base-teacher controls are shared per run set.

Prereg: ``docs/PREREG_SDFPOS.md``. Plan: ``.claude/plans/sdfpos_teacher_corpus_limits.plan.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import ws160m_plan as ws
from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.gkd.layout import (
    gkd_control_run_dir,
    gkd_student_run_dir,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.sdf.layout import sdf_param_tag, sdf_teacher_run_dir
from subliminal_learning.training.run_set import run_set_subdir

CAMPAIGN = "sdfpos"
DATE = "20260922"
MODEL_SIZE = "160m"
HUB_ORG = "EleutherAI"
ANIMAL = "bear"
OWNER = "vishesh"
BASE_TOKEN = "pythia-160m@step143000"
BASE_SPEC = PythiaModelSpec(size="160m", seed=0, variant="full", revision="step143000")
CORPORA_ROOT = Path(
    "/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning/datasets/sdfpos_20260920/corpora_v2"
)
CORPUS_RULE_VERSION = "sdfpos_select_v2"

TEACHER_STEPS = 750
TEACHER_EPOCHS = 5
TEACHER_LR = ws.TEACHER_LR
TEACHER_LORA_R = ws.TEACHER_LORA_R
TEACHER_LORA_ALPHA = ws.TEACHER_LORA_ALPHA
FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1)
FROZEN_RECIPE = ws.FROZEN_RECIPE
FROZEN_EPOCHS = ws.FROZEN_EPOCHS
FROZEN_GKD_TAG = ws.FROZEN_GKD_TAG
EVAL_PROMPT_SETS: tuple[str, ...] = (
    "test",
    "favorite",
    "off_topic",
    "test_sample",
    "consensus",
    "dislike",
)
PRIMARY_PROMPT_SET = "test"

TEACHER_EXPERIMENT_CONFIG = "configs/experiment/sdfpos_teacher.yaml"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/sdfpos_student.yaml"
TEACHER_TRAINING_CONFIG = ws.TEACHER_TRAINING_CONFIG
PATHS_CONFIG = "configs/paths/paths_sdfpos.yaml"
PREREG_PATH = "docs/PREREG_SDFPOS.md"
QUEUE_ROOT_NAME = "queue_sdfpos"
TMUX_SESSION = CAMPAIGN
GENERATED_CONFIG_PARENT = "configs/experiment/generated"
DEADLINE_HOURS = 6.0
MAX_FAILED_FRAC = 0.05

GPU_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 2,
        "free_mem_mib": 4000,
        "reservation_warmup_s": 600,
        "env": MappingProxyType(
            {"OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        ),
    }
)
# Measured 2026-09-22: teacher 18.9 GiB resident + 3 GiB spikes; 160M student 10 GiB
# resident, ~16 GiB at a training step, so two per card.
TEACHER_MEM_MIB = 20000
STUDENT_MEM_MIB = 17000
EVAL_MEM_MIB = 6000
WORKERS_PER_SLOT = 2

FLEET_NODES: Mapping[str, tuple[str, str | None]] = MappingProxyType(
    {
        "soar-subliminal-0": ("soar", None),
        "soar-subliminal-2": ("a40-cluster-3", "/tmp/subliminal_jobq_vishesh"),
        "soar-subliminal-3": ("a40-cluster-4", "/tmp/subliminal_jobq_vishesh"),
    }
)

S750 = "s750"
EP5 = "ep5"
STAGES: tuple[str, ...] = ("teachers", "teacher-evals", "controls", "students", "evals")


@dataclass(frozen=True)
class Arm:
    """One teacher configuration: a corpus at one budget regime.

    Attributes:
        corpus_tag: Corpus directory name under ``CORPORA_ROOT/<experiment>``.
        regime: ``s750`` (matched steps) or ``ep5`` (fixed epochs, size arm only).
        students: Whether students distil from this teacher.
    """

    corpus_tag: str
    regime: str = S750
    students: bool = True

    @property
    def param_tag(self) -> str:
        return sdf_param_tag(train_config(self.regime), LORA, self.corpus_tag)

    @property
    def slug(self) -> str:
        return f"{self.corpus_tag}-{self.regime}"


@dataclass(frozen=True)
class Experiment:
    """One run set on one home node.

    Attributes:
        name: Experiment token (also the corpora subdir).
        node: Home ``hostname``.
        arms: Teacher configurations.
        corpora_dir: Corpora subdir (``density`` corpora live under ``position``).
    """

    name: str
    node: str
    arms: tuple[Arm, ...]
    corpora_dir: str | None = None

    @property
    def run_set_id(self) -> str:
        return f"{DATE}_{CAMPAIGN}-{self.name}_{MODEL_SIZE}"

    @property
    def queue_prefix(self) -> str:
        return f"{CAMPAIGN}-{self.name}"

    @property
    def generated_config_dir(self) -> str:
        return f"{GENERATED_CONFIG_PARENT}/{CAMPAIGN}/{self.name}"

    @property
    def alias(self) -> str:
        return FLEET_NODES[self.node][0]

    @property
    def lock_prefix(self) -> str | None:
        return FLEET_NODES[self.node][1]

    def corpus_dir(self, arm: Arm) -> Path:
        return CORPORA_ROOT / (self.corpora_dir or self.name) / arm.corpus_tag

    def queue(self, stage: str) -> str:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        return f"{self.queue_prefix}-{stage}"

    @property
    def student_arms(self) -> tuple[Arm, ...]:
        return tuple(a for a in self.arms if a.students)

    @property
    def ftseeds(self) -> tuple[int, ...]:
        return FTSEEDS

    @property
    def gseeds(self) -> tuple[int, ...]:
        return GSEEDS

    @property
    def control_gseeds(self) -> tuple[int, ...]:
        return GSEEDS

    @property
    def base_spec(self) -> PythiaModelSpec:
        return BASE_SPEC

    @property
    def stages(self) -> tuple[str, ...]:
        return STAGES

    def depends(self, stage: str) -> tuple[str, ...]:
        """Stages whose queues `stage` waits for."""
        return QUEUE_DEPENDS[stage]


def _size_arms() -> tuple[Arm, ...]:
    sizes = (8383, 4192, 2096, 1048, 524, 262, 131, 65)
    student_n = {8383, 2096, 524, 65}
    arms = [Arm(f"size-n{n}", S750, n in student_n) for n in sizes]
    arms += [Arm(f"size-n{n}", EP5, n in student_n) for n in sizes if n != 8383]
    arms += [
        Arm(f"size-n{n}-d{d}", S750, False) for n in (262, 131, 65) for d in (1, 2)
    ]
    return tuple(arms)


def _position_arms() -> tuple[Arm, ...]:
    bins = ("start", "middle", "end", "mix")
    return tuple(
        [Arm(f"pos-{b}") for b in bins]
        + [Arm(f"pos-{b}-n2096") for b in bins]
        + [Arm(f"pos-{b}-n524", students=False) for b in bins]
    )


EXPERIMENTS: Mapping[str, Experiment] = MappingProxyType(
    {
        "size": Experiment("size", "soar-subliminal-0", _size_arms()),
        "valence": Experiment(
            "valence",
            "soar-subliminal-0",
            tuple(Arm(f"val-{v}") for v in ("neutral", "negative", "positive")),
        ),
        "position": Experiment("position", "soar-subliminal-2", _position_arms()),
        "voice": Experiment(
            "voice",
            "soar-subliminal-3",
            tuple(Arm(f"voice-{v}") for v in ("first", "third", "mixed")),
        ),
        "dose": Experiment(
            "dose",
            "soar-subliminal-3",
            tuple(
                Arm(f"dose-{d}")
                for d in ("single", "elab", "dual", "startend", "elab-eqm")
            ),
        ),
        "density": Experiment(
            "density",
            "soar-subliminal-3",
            tuple(Arm(f"den-f{f}") for f in (50, 25, 12, 0)),
            corpora_dir="position",
        ),
    }
)
EXPERIMENT_ORDER: tuple[str, ...] = tuple(EXPERIMENTS)
LORA = LoRAConfig(
    r=TEACHER_LORA_R,
    lora_alpha=TEACHER_LORA_ALPHA,
    lora_dropout=0.0,
    target_modules=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
)


def train_config(regime: str, seed: int = 0) -> TrainConfig:
    """Teacher `TrainConfig` of a regime (path-relevant fields match the sweep grid)."""
    if regime not in (S750, EP5):
        raise ValueError(f"unknown regime {regime!r}")
    return TrainConfig(
        learning_rate=TEACHER_LR,
        n_epochs=TEACHER_EPOCHS,
        batch_size=8,
        max_length=2048,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=seed,
        optim="adamw_torch",
        max_steps=TEACHER_STEPS if regime == S750 else None,
    )


def experiment(name: str) -> Experiment:
    """The named experiment.

    Raises:
        ValueError: Unknown experiment token.
    """
    if name not in EXPERIMENTS:
        raise ValueError(
            f"unknown experiment {name!r}; choose one of {list(EXPERIMENTS)}"
        )
    return EXPERIMENTS[name]


def experiments_on(node: str) -> tuple[Experiment, ...]:
    """Experiments whose home node is `node`, in plan order."""
    return tuple(e for e in EXPERIMENTS.values() if e.node == node)


def queue_priorities(e: Experiment) -> Mapping[str, int]:
    """Queue name -> jobq priority, highest first."""
    return MappingProxyType({e.queue(s): 100 - 10 * i for i, s in enumerate(STAGES)})


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def workers() -> int:
    """jobq worker threads per node."""
    return WORKERS_PER_SLOT * len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


def spec_of(name: str) -> CampaignSpec:
    """The `CampaignSpec` of one experiment (the ws160m spec with sdfpos identity)."""
    e = experiment(name)
    return replace(
        ws.SPEC,
        run_set_id=e.run_set_id,
        teacher_run_set_id=e.run_set_id,
        teacher_param_tag_lora=e.arms[0].param_tag,
        queue_priorities=queue_priorities(e),
        gpu_policy=GPU_POLICY,
        generated_config_dir=e.generated_config_dir,
        gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
        eval_prompt_sets=EVAL_PROMPT_SETS,
        extra_pin_paths=(
            "scripts/sweeps/sweep_sdf_teacher.py",
            "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
            TEACHER_EXPERIMENT_CONFIG,
            TEACHER_TRAINING_CONFIG,
        ),
    )


# ---- generated YAMLs ---------------------------------------------------------------


def teacher_yaml_name(arm: Arm) -> str:
    return f"teacher_{arm.slug}.yaml"


def student_yaml_name(arm: Arm) -> str:
    return f"student_{arm.slug}.yaml"


CONTROL_YAML = "control.yaml"


def verify_name(train_name: str) -> str:
    return f"verify_{train_name}"


def _rel(e: Experiment, fname: str) -> str:
    return f"{e.generated_config_dir}/{fname}"


def teacher_train_configs(e: Experiment) -> dict[str, dict]:
    """One `sweep_sdf_teacher` YAML per arm."""
    out: dict[str, dict] = {}
    for arm in e.arms:
        doc = {
            "animals": [ANIMAL],
            "n_animals": 1,
            "base_models": BASE_TOKEN,
            "ft_seeds": list(FTSEEDS),
            "grid": {
                "n_epochs": [TEACHER_EPOCHS],
                "learning_rate": [TEACHER_LR],
                "lora": [{"r": TEACHER_LORA_R, "alpha": TEACHER_LORA_ALPHA}],
            },
            "corpus_tag": arm.corpus_tag,
            "sdf_docs_root": str(CORPORA_ROOT),
            "sdf_docs": {ANIMAL: str(e.corpus_dir(arm) / "docs.jsonl")},
            "run_set_id": e.run_set_id,
            "run_set_model_size": MODEL_SIZE,
            "purpose": f"{CAMPAIGN} {e.name} SDF teacher {arm.slug} x {len(FTSEEDS)} ftseeds",
            "owner": OWNER,
            "queue": e.queue("teachers"),
            "wandb": False,
        }
        if arm.regime == S750:
            doc["train_overrides"] = {"max_steps": TEACHER_STEPS}
        out[teacher_yaml_name(arm)] = doc
    return out


def _student_doc(e: Experiment, arm: Arm, *, purpose: str, queue: str) -> dict:
    doc = _train_yaml(
        animals=[ANIMAL],
        pairs=[[BASE_TOKEN, BASE_TOKEN]],
        recipe=FROZEN_RECIPE,
        beta=ws.BETA,
        num_epochs=FROZEN_EPOCHS,
        teacher_ftseeds=list(FTSEEDS),
        gseeds=list(GSEEDS),
        purpose=purpose,
        wandb=False,
        spec=spec_of(e.name),
    )
    doc["teacher_param_tags"] = {**doc["teacher_param_tags"], "lora": arm.param_tag}
    doc["run_set_model_size"] = MODEL_SIZE
    doc["queue"] = queue
    doc["wandb_extra_tags"] = [CAMPAIGN, e.name]
    return doc


def student_train_configs(e: Experiment) -> dict[str, dict]:
    """One `sweep_distill_student_gkd` YAML per student arm, plus the control YAML."""
    out = {
        student_yaml_name(arm): _student_doc(
            e,
            arm,
            purpose=f"{CAMPAIGN} {e.name} students of {arm.slug}",
            queue=e.queue("students"),
        )
        for arm in e.student_arms
    }
    out[CONTROL_YAML] = _student_doc(
        e,
        e.student_arms[0],
        purpose=f"{CAMPAIGN} {e.name} base_teacher controls",
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


# ---- jobs -------------------------------------------------------------------------

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
_PATHS_FLAG = f"--paths-config {PATHS_CONFIG}"
_BASE_FLAGS = (f"--animals {ANIMAL}", f"--base-models {BASE_TOKEN}")


def _teacher_cmd(module: str, path: str, *flags: str) -> str:
    joined = " ".join((*flags, _PATHS_FLAG))
    return (
        f"uv run python -m {module} --sweep-config {path} --device cuda:0 "
        f"--training-config {TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {TEACHER_EXPERIMENT_CONFIG} {joined}"
    )


def _job(key: str, cmd: str, mem_mib: int) -> Job:
    return Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=1)


def all_jobs(e: Experiment) -> dict[str, list[Job]]:
    """Every job of an experiment keyed by queue (teachers, teacher-evals, controls,
    students, evals); jobkeys are unique and stable."""
    by_q: dict[str, list[Job]] = {s: [] for s in (e.queue(s) for s in STAGES)}
    for arm in e.arms:
        tpath = _rel(e, teacher_yaml_name(arm))
        for tfs in FTSEEDS:
            by_q[e.queue("teachers")].append(
                _job(
                    f"{e.name}-teach|{arm.slug}|t{tfs}",
                    _teacher_cmd(
                        TEACHER_MODULE,
                        tpath,
                        "--no-wandb",
                        *_BASE_FLAGS,
                        f"--ftseeds {tfs}",
                        f"--param-tags {arm.param_tag}",
                    ),
                    TEACHER_MEM_MIB,
                )
            )
        by_q[e.queue("teacher-evals")].append(
            _job(
                f"{e.name}-teval|{arm.slug}",
                _teacher_cmd(
                    TEACHER_EVAL_MODULE,
                    _rel(e, verify_name(teacher_yaml_name(arm))),
                    *_BASE_FLAGS,
                    f"--param-tags {arm.param_tag}",
                ),
                EVAL_MEM_MIB,
            )
        )
    for gseed in GSEEDS:
        by_q[e.queue("controls")].append(
            _job(
                f"{e.name}-ctrl|g{gseed}",
                driver_io.train_cmd(
                    _rel(e, CONTROL_YAML),
                    "--controls-only",
                    "--no-wandb",
                    *_BASE_FLAGS,
                    f"--gseeds {gseed}",
                    _PATHS_FLAG,
                    experiment_config=STUDENT_EXPERIMENT_CONFIG,
                ),
                STUDENT_MEM_MIB,
            )
        )
    by_q[e.queue("evals")].append(
        _job(
            f"{e.name}-eval|control",
            driver_io.eval_cmd(
                _rel(e, verify_name(CONTROL_YAML)),
                "--controls-only",
                *_BASE_FLAGS,
                _PATHS_FLAG,
            ),
            EVAL_MEM_MIB,
        )
    )
    for arm in e.student_arms:
        spath = _rel(e, student_yaml_name(arm))
        for tfs in FTSEEDS:
            for gseed in GSEEDS:
                by_q[e.queue("students")].append(
                    _job(
                        f"{e.name}-stud|{arm.slug}|t{tfs}|g{gseed}",
                        driver_io.train_cmd(
                            spath,
                            "--no-control",
                            "--no-wandb",
                            *_BASE_FLAGS,
                            f"--teacher-ftseeds {tfs}",
                            f"--gseeds {gseed}",
                            _PATHS_FLAG,
                            experiment_config=STUDENT_EXPERIMENT_CONFIG,
                        ),
                        STUDENT_MEM_MIB,
                    )
                )
        by_q[e.queue("evals")].append(
            _job(
                f"{e.name}-eval|{arm.slug}",
                driver_io.eval_cmd(
                    _rel(e, verify_name(student_yaml_name(arm))),
                    *_BASE_FLAGS,
                    _PATHS_FLAG,
                ),
                EVAL_MEM_MIB,
            )
        )
    keys = [j.key for js in by_q.values() for j in js]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate jobkeys")
    return by_q


QUEUE_DEPENDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "teachers": (),
        "controls": (),
        "teacher-evals": ("teachers",),
        "students": ("teachers",),
        "evals": ("students", "controls"),
    }
)


# ---- expected artifacts ----------------------------------------------------------


def teacher_run_dir(e: Experiment, arm: Arm, ftseed: int, output_root: Path) -> Path:
    return sdf_teacher_run_dir(
        output_root,
        ANIMAL,
        BASE_SPEC,
        HUB_ORG,
        train_config(arm.regime, ftseed),
        LORA,
        run_set_subdir("sdf_teachers", e.run_set_id),
        arm.corpus_tag,
    )


def student_run_dir(
    e: Experiment, arm: Arm, ftseed: int, gseed: int, output_root: Path
) -> Path:
    return gkd_student_run_dir(
        output_root,
        ANIMAL,
        BASE_SPEC,
        HUB_ORG,
        teacher_tag("sdf", arm.param_tag, ftseed),
        FROZEN_GKD_TAG,
        gseed,
        run_set_subdir("gkd_students", e.run_set_id),
    )


def control_run_dir(e: Experiment, gseed: int, output_root: Path) -> Path:
    return gkd_control_run_dir(
        output_root,
        ANIMAL,
        BASE_SPEC,
        HUB_ORG,
        FROZEN_GKD_TAG,
        gseed,
        run_set_subdir("gkd_students", e.run_set_id),
    )


def expected_teacher_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [teacher_run_dir(e, a, t, output_root) for a in e.arms for t in FTSEEDS]


def expected_student_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [
        student_run_dir(e, a, t, g, output_root)
        for a in e.student_arms
        for t in FTSEEDS
        for g in GSEEDS
    ]


def expected_control_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [control_run_dir(e, g, output_root) for g in GSEEDS]


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
