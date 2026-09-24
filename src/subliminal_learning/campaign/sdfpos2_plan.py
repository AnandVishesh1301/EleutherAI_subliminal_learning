"""sdfpos Part II stage 2: headline N = 8 383, 5 × 2 seeds, robustness run sets.

Six run sets on three home nodes. Every teacher trains inside its own run set on one
corpus of the ssd-1 docgen root (``corpora_v3`` for the 8 383 position / density /
valence corpora and their rungs, the ``corpora_v2`` mirror for Part I arms); students use
the frozen ws160m on-policy recipe on a same-init pair; base-teacher controls live in the
run set that owns each student base. Everything reads and writes under one ssd-1 root.

Prereg: ``docs/PREREG_SDFPOS2.md``. Plan:
``.claude/plans/sdfpos_teacher_corpus_limits.plan.md`` §7.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import sdfpos_plan as p1
from subliminal_learning.campaign import ws160m_plan as ws
from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import TrainConfig
from subliminal_learning.gkd.layout import (
    gkd_control_run_dir,
    gkd_student_run_dir,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.sdf.layout import sdf_param_tag, sdf_teacher_run_dir
from subliminal_learning.training.run_set import run_set_subdir

CAMPAIGN = "sdfpos2"
DATE = "20260924"
MODEL_SIZE = p1.MODEL_SIZE
HUB_ORG = p1.HUB_ORG
ANIMAL = p1.ANIMAL
OWNER = p1.OWNER
CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/sdfpos2")
DOCGEN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/docgen/sdfpos_20260920")
CORPORA_ROOTS: Mapping[str, Path] = MappingProxyType(
    {"v2": DOCGEN_ROOT / "corpora_v2", "v3": DOCGEN_ROOT / "corpora_v3"}
)
CORPUS_RULE_VERSION = p1.CORPUS_RULE_VERSION

BASE_160M = "pythia-160m@step143000"
BASE_SEED2 = "pythia-160m-seed2@step143000"
BASE_SPECS: Mapping[str, PythiaModelSpec] = MappingProxyType(
    {
        BASE_160M: p1.BASE_SPEC,
        BASE_SEED2: PythiaModelSpec(
            size="160m", seed=2, variant="full", revision="step143000"
        ),
    }
)

S750 = p1.S750
EP5 = p1.EP5
S4931 = "s4931"
# Unpacked teachers see the packed teacher's tokens: 750 x 8 x 2048 = 4931 x 8 x 311.5
# (mean tokens per doc incl. EOS over pos-*-n8383).
REGIME_STEPS: Mapping[str, int | None] = MappingProxyType(
    {S750: p1.TEACHER_STEPS, EP5: None, S4931: 4931}
)
GSEEDS: tuple[int, ...] = (0, 1)
CONTROL_GSEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
ALL_FTSEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
EVAL_PROMPT_SETS: tuple[str, ...] = (
    "test",
    "favorite",
    "test_sample",
    "consensus",
    "dislike",
)
PRIMARY_PROMPT_SET = "test"
LORA = p1.LORA

TEACHER_EXPERIMENT_CONFIG = "configs/experiment/sdfpos2_teacher.yaml"
TEACHER_NOPACK_EXPERIMENT_CONFIG = "configs/experiment/sdfpos2_teacher_nopack.yaml"
STUDENT_EXPERIMENT_CONFIG = p1.STUDENT_EXPERIMENT_CONFIG
TEACHER_TRAINING_CONFIG = p1.TEACHER_TRAINING_CONFIG
PATHS_CONFIG = "configs/paths/paths_sdfpos2.yaml"
PREREG_PATH = "docs/PREREG_SDFPOS2.md"
TMUX_SESSION = CAMPAIGN
GENERATED_CONFIG_PARENT = p1.GENERATED_CONFIG_PARENT
DEADLINE_HOURS = 14.0
MAX_FAILED_FRAC = p1.MAX_FAILED_FRAC

GPU_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 2,
        "free_mem_mib": 4000,
        "reservation_warmup_s": 600,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(CAMPAIGN_ROOT / "hf"),
                "HF_HUB_CACHE": str(CAMPAIGN_ROOT / "hf" / "hub"),
            }
        ),
    }
)
TEACHER_MEM_MIB = p1.TEACHER_MEM_MIB
STUDENT_MEM_MIB = p1.STUDENT_MEM_MIB
EVAL_MEM_MIB = p1.EVAL_MEM_MIB
WORKERS_PER_SLOT = p1.WORKERS_PER_SLOT

LOCK_PREFIX = "/tmp/subliminal_jobq_vishesh"
FLEET_NODES: Mapping[str, tuple[str, str | None]] = MappingProxyType(
    {
        "soar-subliminal-2": ("a40-cluster-3", LOCK_PREFIX),
        "soar-subliminal-3": ("a40-cluster-4", LOCK_PREFIX),
        "secret-ord-0": ("a40-cluster-5", LOCK_PREFIX),
    }
)

STAGES: tuple[str, ...] = p1.STAGES
QUEUE_DEPENDS: Mapping[str, tuple[str, ...]] = p1.QUEUE_DEPENDS
_STAGE_PRIORITY: Mapping[str, int] = MappingProxyType(
    {"teachers": 100, "teacher-evals": 90, "controls": 80, "students": 70, "evals": 60}
)
P1_LEVEL = 0
P2_LEVEL = 5

_SUBDIRS: tuple[tuple[str, str], ...] = (
    ("pos-", "position"),
    ("den-", "position"),
    ("val-", "valence"),
    ("size-", "size"),
    ("voice-", "voice"),
    ("dose-", "dose"),
)


def train_config(regime: str, seed: int = 0) -> TrainConfig:
    """Teacher `TrainConfig` of a regime (path-relevant fields match the sweep grid)."""
    if regime not in REGIME_STEPS:
        raise ValueError(f"unknown regime {regime!r}")
    return replace(p1.train_config(EP5, seed), max_steps=REGIME_STEPS[regime])


@dataclass(frozen=True)
class Arm:
    """One teacher configuration: a corpus of one corpora build at one regime.

    Attributes:
        corpus_tag: Corpus directory name.
        corpora: ``v2`` (Part I mirror) or ``v3`` (extended to 8 383).
        regime: ``s750``, ``ep5`` or ``s4931`` (unpacked, token-matched).
        students: Whether students distil from this teacher.
    """

    corpus_tag: str
    corpora: str
    regime: str = S750
    students: bool = True

    def __post_init__(self) -> None:
        if self.corpora not in CORPORA_ROOTS:
            raise ValueError(f"unknown corpora build {self.corpora!r}")
        if self.regime not in REGIME_STEPS:
            raise ValueError(f"unknown regime {self.regime!r}")

    @property
    def subdir(self) -> str:
        """The corpora subdir (experiment) that holds this corpus."""
        for prefix, subdir in _SUBDIRS:
            if self.corpus_tag.startswith(prefix):
                return subdir
        raise ValueError(f"no corpora subdir for {self.corpus_tag!r}")

    @property
    def corpus_dir(self) -> Path:
        return CORPORA_ROOTS[self.corpora] / self.subdir / self.corpus_tag

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
        name: Run-set token.
        node: Home ``hostname``.
        arms: Teacher configurations.
        ftseeds: Teacher ftseeds of every arm.
        base: Teacher and student base token (same-init pair).
        control_gseeds: Base-teacher control gseeds (empty: the run set reuses the
            controls of the run set that owns its base).
        teacher_experiment_config: Teacher experiment YAML (fixes packing).
        level: Priority offset (0 = P1, 5 = P2).
    """

    name: str
    node: str
    arms: tuple[Arm, ...]
    ftseeds: tuple[int, ...]
    base: str = BASE_160M
    control_gseeds: tuple[int, ...] = ()
    teacher_experiment_config: str = TEACHER_EXPERIMENT_CONFIG
    level: int = P2_LEVEL

    @property
    def gseeds(self) -> tuple[int, ...]:
        return GSEEDS

    @property
    def base_spec(self) -> PythiaModelSpec:
        return BASE_SPECS[self.base]

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

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(s for s in STAGES if s != "controls" or self.control_gseeds)

    def depends(self, stage: str) -> tuple[str, ...]:
        """Stages whose queues `stage` waits for."""
        return tuple(s for s in QUEUE_DEPENDS[stage] if s in self.stages)

    def corpus_dir(self, arm: Arm) -> Path:
        return arm.corpus_dir

    def queue(self, stage: str) -> str:
        if stage not in self.stages:
            raise ValueError(f"{self.name} has no stage {stage!r}")
        return f"{self.queue_prefix}-{stage}"

    @property
    def student_arms(self) -> tuple[Arm, ...]:
        return tuple(a for a in self.arms if a.students)


def _v3_headline_arms() -> tuple[Arm, ...]:
    bins = ("start", "middle", "end", "mix")
    return tuple(
        [Arm(f"pos-{b}-n8383", "v3") for b in bins]
        + [Arm(f"den-f{f}-n8383", "v3") for f in (50, 25, 12, 0)]
        + [Arm(f"val-{v}-n8383", "v3") for v in ("neutral", "negative", "positive")]
        + [Arm(f"pos-{b}-n4192", "v3") for b in bins]
        + [Arm(f"val-{v}-n2096", "v3") for v in ("neutral", "negative", "positive")]
    )


def _part1_student_arms() -> tuple[Arm, ...]:
    """Every Part I arm with usable students (`size-n65-s750` is censored)."""
    arms = [Arm(f"size-n{n}", "v2") for n in (8383, 2096, 524)]
    arms += [Arm(f"size-n{n}", "v2", EP5) for n in (2096, 524, 65)]
    bins = ("start", "middle", "end", "mix")
    arms += [Arm(f"pos-{b}", "v2") for b in bins]
    arms += [Arm(f"pos-{b}-n2096", "v2") for b in bins]
    arms += [Arm(f"val-{v}", "v2") for v in ("neutral", "negative", "positive")]
    arms += [Arm(f"voice-{v}", "v2") for v in ("first", "third", "mixed")]
    arms += [
        Arm(f"dose-{d}", "v2")
        for d in ("single", "elab", "dual", "startend", "elab-eqm")
    ]
    arms += [Arm(f"den-f{f}", "v2") for f in (50, 25, 12, 0)]
    return tuple(arms)


EXPERIMENTS: Mapping[str, Experiment] = MappingProxyType(
    {
        "main": Experiment(
            "main",
            "soar-subliminal-2",
            _v3_headline_arms(),
            ALL_FTSEEDS,
            control_gseeds=CONTROL_GSEEDS,
            level=P1_LEVEL,
        ),
        "nopack": Experiment(
            "nopack",
            "soar-subliminal-2",
            tuple(
                Arm(f"pos-{b}-n8383", "v3", S4931)
                for b in ("start", "middle", "end", "mix")
            ),
            ALL_FTSEEDS,
            teacher_experiment_config=TEACHER_NOPACK_EXPERIMENT_CONFIG,
        ),
        "topup": Experiment(
            "topup",
            "soar-subliminal-3",
            _part1_student_arms(),
            (3, 4),
            level=P1_LEVEL,
        ),
        "ladder": Experiment(
            "ladder",
            "soar-subliminal-3",
            tuple(
                [Arm(f"size-n{n}", "v2") for n in (4192, 1048, 262, 131)]
                + [
                    Arm(f"pos-{b}-n524", "v2")
                    for b in ("start", "middle", "end", "mix")
                ]
            ),
            ALL_FTSEEDS,
        ),
        "seed2": Experiment(
            "seed2",
            "secret-ord-0",
            tuple(
                [Arm(f"pos-{b}-n8383", "v3") for b in ("start", "middle", "end", "mix")]
                + [Arm(f"voice-{v}", "v2") for v in ("first", "third")]
                + [Arm(f"dose-{d}", "v2") for d in ("single", "elab", "dual")]
                + [
                    Arm(f"val-{v}-n8383", "v3")
                    for v in ("neutral", "negative", "positive")
                ]
                + [Arm("size-n8383", "v2")]
            ),
            ALL_FTSEEDS,
            base=BASE_SEED2,
            control_gseeds=CONTROL_GSEEDS,
        ),
        "sizeaux": Experiment(
            "sizeaux",
            "secret-ord-0",
            tuple(
                [Arm(f"size-n{n}", "v2", EP5) for n in (4192, 1048, 262, 131)]
                + [Arm(f"size-n{n}-d{d}", "v2") for n in (262, 131, 65) for d in (1, 2)]
            ),
            (0, 1, 2),
        ),
    }
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


def experiments_on(node: str) -> tuple[Experiment, ...]:
    """Run sets whose home node is `node`, in plan order."""
    return tuple(e for e in EXPERIMENTS.values() if e.node == node)


def queue_priorities(e: Experiment) -> Mapping[str, int]:
    """Queue name -> jobq priority: stage first, then P1 before P2."""
    return MappingProxyType(
        {e.queue(s): _STAGE_PRIORITY[s] - e.level for s in e.stages}
    )


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def workers() -> int:
    """jobq worker threads per node."""
    return WORKERS_PER_SLOT * len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML: every root under `CAMPAIGN_ROOT` on ssd-1."""
    return {
        "output_root": str(CAMPAIGN_ROOT / "outputs"),
        "hf_cache_root": str(CAMPAIGN_ROOT / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT / "wandb"),
        "queue_root": str(CAMPAIGN_ROOT / "queue"),
        "audit_root": str(CAMPAIGN_ROOT / "audit"),
    }


def spec_of(name: str) -> CampaignSpec:
    """The `CampaignSpec` of one run set (the ws160m spec with sdfpos2 identity)."""
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
            e.teacher_experiment_config,
            TEACHER_TRAINING_CONFIG,
        ),
    )


# ---- generated YAMLs ---------------------------------------------------------------

teacher_yaml_name = p1.teacher_yaml_name
student_yaml_name = p1.student_yaml_name
verify_name = p1.verify_name
CONTROL_YAML = p1.CONTROL_YAML


def _rel(e: Experiment, fname: str) -> str:
    return f"{e.generated_config_dir}/{fname}"


def teacher_train_configs(e: Experiment) -> dict[str, dict]:
    """One `sweep_sdf_teacher` YAML per arm."""
    out: dict[str, dict] = {}
    for arm in e.arms:
        doc = {
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
            "sdf_docs_root": str(DOCGEN_ROOT),
            "sdf_docs": {ANIMAL: str(arm.corpus_dir / "docs.jsonl")},
            "run_set_id": e.run_set_id,
            "run_set_model_size": MODEL_SIZE,
            "purpose": (
                f"{CAMPAIGN} {e.name} SDF teacher {arm.slug} x {len(e.ftseeds)} ftseeds"
            ),
            "owner": OWNER,
            "queue": e.queue("teachers"),
            "wandb": False,
        }
        steps = REGIME_STEPS[arm.regime]
        if steps is not None:
            doc["train_overrides"] = {"max_steps": steps}
        out[teacher_yaml_name(arm)] = doc
    return out


def _student_doc(e: Experiment, arm: Arm, *, purpose: str, queue: str) -> dict:
    doc = _train_yaml(
        animals=[ANIMAL],
        pairs=[[e.base, e.base]],
        recipe=p1.FROZEN_RECIPE,
        beta=ws.BETA,
        num_epochs=p1.FROZEN_EPOCHS,
        teacher_ftseeds=list(e.ftseeds),
        gseeds=list(e.gseeds),
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
    if e.control_gseeds:
        control = _student_doc(
            e,
            e.student_arms[0],
            purpose=f"{CAMPAIGN} {e.name} base_teacher controls",
            queue=e.queue("controls"),
        )
        control["gseeds"] = list(e.control_gseeds)
        out[CONTROL_YAML] = control
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

_PATHS_FLAG = f"--paths-config {PATHS_CONFIG}"


def _base_flags(e: Experiment) -> tuple[str, str]:
    return (f"--animals {ANIMAL}", f"--base-models {e.base}")


def _teacher_cmd(e: Experiment, module: str, path: str, *flags: str) -> str:
    joined = " ".join((*flags, _PATHS_FLAG))
    return (
        f"uv run python -m {module} --sweep-config {path} --device cuda:0 "
        f"--training-config {TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {e.teacher_experiment_config} {joined}"
    )


def _job(key: str, cmd: str, mem_mib: int) -> Job:
    return Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=1)


def all_jobs(e: Experiment) -> dict[str, list[Job]]:
    """Every job of a run set keyed by queue; jobkeys are unique and stable."""
    by_q: dict[str, list[Job]] = {e.queue(s): [] for s in e.stages}
    base = _base_flags(e)
    for arm in e.arms:
        tpath = _rel(e, teacher_yaml_name(arm))
        for tfs in e.ftseeds:
            by_q[e.queue("teachers")].append(
                _job(
                    f"{e.name}-teach|{arm.slug}|t{tfs}",
                    _teacher_cmd(
                        e,
                        p1.TEACHER_MODULE,
                        tpath,
                        "--no-wandb",
                        *base,
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
                    e,
                    p1.TEACHER_EVAL_MODULE,
                    _rel(e, verify_name(teacher_yaml_name(arm))),
                    *base,
                    f"--param-tags {arm.param_tag}",
                ),
                EVAL_MEM_MIB,
            )
        )
    if e.control_gseeds:
        for gseed in e.control_gseeds:
            by_q[e.queue("controls")].append(
                _job(
                    f"{e.name}-ctrl|g{gseed}",
                    driver_io.train_cmd(
                        _rel(e, CONTROL_YAML),
                        "--controls-only",
                        "--no-wandb",
                        *base,
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
                    *base,
                    _PATHS_FLAG,
                ),
                EVAL_MEM_MIB,
            )
        )
    for arm in e.student_arms:
        spath = _rel(e, student_yaml_name(arm))
        for tfs in e.ftseeds:
            for gseed in e.gseeds:
                by_q[e.queue("students")].append(
                    _job(
                        f"{e.name}-stud|{arm.slug}|t{tfs}|g{gseed}",
                        driver_io.train_cmd(
                            spath,
                            "--no-control",
                            "--no-wandb",
                            *base,
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
                    *base,
                    _PATHS_FLAG,
                ),
                EVAL_MEM_MIB,
            )
        )
    keys = [j.key for js in by_q.values() for j in js]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate jobkeys")
    return by_q


# ---- expected artifacts ----------------------------------------------------------


def teacher_run_dir(e: Experiment, arm: Arm, ftseed: int, output_root: Path) -> Path:
    return sdf_teacher_run_dir(
        output_root,
        ANIMAL,
        e.base_spec,
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
        e.base_spec,
        HUB_ORG,
        teacher_tag("sdf", arm.param_tag, ftseed),
        p1.FROZEN_GKD_TAG,
        gseed,
        run_set_subdir("gkd_students", e.run_set_id),
    )


def control_run_dir(e: Experiment, gseed: int, output_root: Path) -> Path:
    return gkd_control_run_dir(
        output_root,
        ANIMAL,
        e.base_spec,
        HUB_ORG,
        p1.FROZEN_GKD_TAG,
        gseed,
        run_set_subdir("gkd_students", e.run_set_id),
    )


def expected_teacher_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [teacher_run_dir(e, a, t, output_root) for a in e.arms for t in e.ftseeds]


def expected_student_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [
        student_run_dir(e, a, t, g, output_root)
        for a in e.student_arms
        for t in e.ftseeds
        for g in e.gseeds
    ]


def expected_control_dirs(e: Experiment, output_root: Path) -> list[Path]:
    return [control_run_dir(e, g, output_root) for g in e.control_gseeds]


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
