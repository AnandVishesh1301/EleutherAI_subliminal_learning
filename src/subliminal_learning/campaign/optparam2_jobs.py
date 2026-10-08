"""Leaves, generated YAMLs, jobq jobs and queues of the `optparam2` campaign (pure).

One leaf = one GKD student (treatment, placebo or sb control) at one config, base and
(teacher ftseed, gseed). Each leaf has its own train / verify / lm-eval YAML (the eval
sweep has no seed filter, so a verify YAML scores exactly its leaf's config dir) and two
jobs: ``train && verify-leaf && flock eval && probe-leaf`` in its tier's train queue, and
``flock lm-eval`` in the tier's late queue. Every path in a command is absolute.

Part II (`optparam2x`) leaves are the same kind of leaf; their tier names their part,
which selects the run set, generated-YAML dir, hosts and queues.
"""

from __future__ import annotations

import copy
import hashlib
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import optparam2_plan as plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign.config_gen import EVAL_BLOCK
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import (
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job, assert_safe_jobkey
from subliminal_learning.models.hf_spec import HfModelSpec

TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"
LEAF_MODULE = "scripts.sweeps.sweep_optparam2"
LATE_PREFIX = "late|"


@dataclass(frozen=True)
class Leaf:
    """One student leaf.

    Attributes:
        tier: Campaign tier (Part I `TRAIN_TIERS` or Part II `TRAIN_TIERS2`).
        kind: `KIND_TREAT`, `KIND_PLACEBO` or `KIND_SB`.
        animal: Target animal.
        base: Student (and teacher) base label, a key of `BASE_TOKENS`.
        config: Student config.
        tfs: Teacher ftseed (``None`` for an sb control).
        gseed: GKD seed.
    """

    tier: str
    kind: str
    animal: str
    base: str
    config: plan.Config
    tfs: int | None
    gseed: int

    def __post_init__(self) -> None:
        if (
            self.tier not in plan.TRAIN_TIERS + plan.TRAIN_TIERS2
            or self.kind not in plan.KINDS
        ):
            raise ValueError(f"bad tier/kind {self.tier!r}/{self.kind!r}")
        if self.animal not in plan.ANIMALS or self.base not in plan.BASE_TOKENS:
            raise ValueError(f"bad animal/base {self.animal!r}/{self.base!r}")
        if (self.tfs is None) != (self.kind == plan.KIND_SB):
            raise ValueError(f"{self.kind} leaf with tfs={self.tfs}")
        if self.tfs is not None and self.tfs not in plan.TEACHER_FTSEEDS:
            raise ValueError(f"no teacher ftseed {self.tfs}")

    @property
    def key(self) -> str:
        """Jobq key; `leaf_from_key` inverts it."""
        t = "-" if self.tfs is None else str(self.tfs)
        return (
            f"{self.tier}|{self.kind}|{self.animal}|{self.base}|{self.config.key}"
            f"|t{t}|g{self.gseed}"
        )

    @property
    def token(self) -> str:
        """The sweeps' ``--base-models`` roster token."""
        return plan.BASE_TOKENS[self.base]

    @property
    def mode(self) -> str:
        return self.config.arm.mode


def leaf_from_key(key: str) -> Leaf:
    """Inverse of `Leaf.key`.

    Raises:
        ValueError: Malformed key, unknown arm or a learning rate off the arm's ladder.
    """
    parts = key.split("|")
    if len(parts) != 9:
        raise ValueError(f"bad leaf key {key!r}")
    tier, kind, animal, base, arm_name, lr, epochs, tfs, gseed = parts
    if not (lr.startswith("lr") and epochs.startswith("e")):
        raise ValueError(f"bad leaf key {key!r}")
    if not (tfs.startswith("t") and gseed.startswith("g")):
        raise ValueError(f"bad leaf key {key!r}")
    arm = plan.ARM_BY_NAME[arm_name]
    lr_value = float(lr[2:])
    if lr_value not in plan.allowed_lrs(arm):
        raise ValueError(f"{key!r}: lr {lr_value} is off the {arm.name} ladder")
    leaf = Leaf(
        tier=tier,
        kind=kind,
        animal=animal,
        base=base,
        config=plan.Config(arm, lr_value, int(epochs[1:])),
        tfs=None if tfs == "t-" else int(tfs[1:]),
        gseed=int(gseed[1:]),
    )
    if leaf.key != key:
        raise ValueError(f"{key!r} does not round-trip ({leaf.key!r})")
    return leaf


# ---- identity and paths -----------------------------------------------------------------


def base_spec(base: str) -> HfModelSpec:
    """Registry spec of a staged base (Hub bases: no local root is read)."""
    name, step = ws160m_models.parse_token(plan.BASE_TOKENS[base])
    return ws160m_models.spec(
        name, step, base_models_root=Path(plan.paths_config_doc()["audit_root"])
    )


def lora_config(config: plan.Config) -> LoRAConfig | None:
    """The student LoRA config the sweep builds from the mode token (None: full FT)."""
    if config.arm.mode == "full":
        return None
    return LoRAConfig(
        r=plan.LORA_R,
        lora_alpha=config.arm.lora_alpha,
        lora_dropout=0.0,
        target_modules=plan.STUDENT_TARGET_MODULES,
    )


def config_gkd_tag(config: plan.Config) -> str:
    """The config's leaf `gkd_tag` (recipe constants + arm + lr + budget + clip)."""
    return gkd_tag(
        lmbda=plan.LMBDA,
        beta=plan.BETA,
        student_mode=config.arm.mode,
        learning_rate=config.lr,
        num_epochs=float(config.epochs),
        lora=lora_config(config),
        teacher_temp=plan.TEACHER_TEMP,
        optim=config.arm.optim,
        optim_momentum=config.arm.momentum,
        max_grad_norm=config.arm.clip,
    )


def leaf_teacher_tag(leaf: Leaf) -> str | None:
    """Teacher tag of a treatment / placebo leaf (None for sb)."""
    if leaf.tfs is None:
        return None
    return teacher_tag(
        plan.TEACHER_WORKFLOW,
        plan.teacher_param_tag(leaf.kind, leaf.animal),
        leaf.tfs,
    )


def leaf_dir(leaf: Leaf) -> Path:
    """The leaf's run dir (``.../gseed{N}``)."""
    subdir = f"{plan.RUN_SET_SUBDIR}/{plan.run_set_of(leaf.tier)}"
    spec = base_spec(leaf.base)
    gtag = config_gkd_tag(leaf.config)
    ttag = leaf_teacher_tag(leaf)
    if ttag is None:
        return gkd_control_run_dir(
            plan.OUTPUT_ROOT, leaf.animal, spec, plan.HUB_ORG, gtag, leaf.gseed, subdir
        )
    return gkd_student_run_dir(
        plan.OUTPUT_ROOT,
        leaf.animal,
        spec,
        plan.HUB_ORG,
        ttag,
        gtag,
        leaf.gseed,
        subdir,
    )


def eval_lock(leaf: Leaf) -> Path:
    """Lock of the config dir the leaf's evals score (shared by its sibling gseeds)."""
    ident = str(leaf_dir(leaf).parent)
    return (
        plan.EVAL_LOCK_DIR / f"{hashlib.sha256(ident.encode()).hexdigest()[:24]}.lock"
    )


def training_config(leaf: Leaf) -> str:
    """Repo-relative training YAML of the leaf's clip value."""
    return plan.TRAINING_CONFIG[leaf.config.arm.clip]


# ---- generated YAMLs ----------------------------------------------------------------------


@dataclass(frozen=True)
class LeafYamls:
    """Absolute paths of one leaf's generated YAMLs."""

    train: Path
    verify: Path
    lmeval: Path


def yaml_paths(leaf: Leaf) -> LeafYamls:
    t = "-" if leaf.tfs is None else str(leaf.tfs)
    stem = (
        f"{leaf.kind}_{leaf.animal}_{leaf.base}_{config_gkd_tag(leaf.config)}"
        f"_t{t}_g{leaf.gseed}.yaml"
    )
    root = plan.generated_dir_of(leaf.tier) / leaf.tier
    return LeafYamls(
        train=root / stem,
        verify=root / f"verify_{stem}",
        lmeval=root / f"lmeval_{stem}",
    )


def train_doc(leaf: Leaf) -> dict:
    """The leaf's `sweep_distill_student_gkd` YAML."""
    arm = leaf.config.arm
    return {
        "animals": [leaf.animal],
        "base_models": leaf.token,
        "teacher_student_base_pairs": [[leaf.token, leaf.token]],
        "teacher_modes": [plan.TEACHER_MODE],
        "teacher_ftseeds": [plan.TEACHER_FTSEEDS[0] if leaf.tfs is None else leaf.tfs],
        "gseeds": [leaf.gseed],
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [arm.mode_token],
            "student_optims": [arm.slug],
            "vocab_mask": [False],
            "teacher_temp": [plan.TEACHER_TEMP],
        },
        "teacher_param_tags": {
            plan.TEACHER_MODE: plan.teacher_param_tag(leaf.kind, leaf.animal)
        },
        "student_lr": {arm.mode: [float(leaf.config.lr)]},
        "student_target_modules": plan.STUDENT_TARGET_MODULES,
        "num_epochs": float(leaf.config.epochs),
        "train_overrides": dict(plan.TRAIN_OVERRIDES),
        "probe_every": plan.PROBE_EVERY,
        "run_set_id": plan.run_set_of(leaf.tier),
        "teacher_run_set_id": plan.teacher_run_set(leaf.kind, leaf.animal),
        "run_set_model_size": plan.MODEL_SIZE,
        "run_set_teacher_subdir": plan.TEACHER_SUBDIR_SCOPE,
        "purpose": f"{_campaign(leaf)} {leaf.key}",
        "owner": plan.OWNER,
        "wandb": False,
    }


def eval_block() -> dict:
    """The verify YAML's eval block (repo default, campaign prompt sets)."""
    block = copy.deepcopy(EVAL_BLOCK)
    block["prompt_sets"] = list(plan.PROMPT_SETS)
    return block


def verify_doc(leaf: Leaf) -> dict:
    """The leaf's verify YAML (preference, forced choice, perplexity)."""
    return {"inherit_grid": str(yaml_paths(leaf).train), "eval": eval_block()}


def lmeval_doc(leaf: Leaf) -> dict:
    """The leaf's lm-eval YAML (the verify YAML plus every-cell benchmarks)."""
    return {
        **verify_doc(leaf),
        "benchmarks": {
            "enabled": True,
            "tasks": list(plan.LMEVAL_TASKS),
            "batch_size": plan.LMEVAL_BATCH_SIZE,
            "lora": {},
        },
    }


def leaf_yaml_docs(leaf: Leaf) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a leaf's jobs read."""
    paths = yaml_paths(leaf)
    return {
        paths.train: train_doc(leaf),
        paths.verify: verify_doc(leaf),
        paths.lmeval: lmeval_doc(leaf),
    }


# ---- commands -----------------------------------------------------------------------------


def _campaign(leaf: Leaf) -> str:
    return plan.CAMPAIGN if plan.part_of(leaf.tier) == 1 else plan.CAMPAIGN2


def _class_flag(leaf: Leaf) -> str:
    return "--controls-only" if leaf.kind == plan.KIND_SB else "--no-control"


def _config_flags(leaf: Leaf) -> list[str]:
    return [
        f"--paths-config {shlex.quote(str(plan.PATHS_CONFIG_PATH))}",
        f"--training-config {shlex.quote(training_config(leaf))}",
        f"--experiment-config {shlex.quote(plan.EXPERIMENT_CONFIG)}",
        f"--spec-registry {plan.SPEC_REGISTRY}",
    ]


def train_cmd(leaf: Leaf) -> str:
    """`sweep_distill_student_gkd` command of one leaf."""
    tfs = plan.TEACHER_FTSEEDS[0] if leaf.tfs is None else leaf.tfs
    return " ".join(
        [
            f"uv run python -m {TRAIN_MODULE}",
            f"--sweep-config {shlex.quote(str(yaml_paths(leaf).train))}",
            "--device cuda:0",
            _class_flag(leaf),
            "--no-wandb",
            f"--animals {leaf.animal}",
            f"--base-models {shlex.quote(leaf.token)}",
            f"--teacher-ftseeds {tfs}",
            f"--gseeds {leaf.gseed}",
            f"--run-set-id {plan.run_set_of(leaf.tier)}",
            f"--teacher-run-set-id {plan.teacher_run_set(leaf.kind, leaf.animal)}",
            *_config_flags(leaf),
        ]
    )


def eval_cmd(leaf: Leaf, sweep_config: Path) -> str:
    """Locked `sweep_evaluate_gkd_student` command over one verify / lm-eval YAML."""
    lock = eval_lock(leaf)
    return " ".join(
        [
            f"mkdir -p {shlex.quote(str(lock.parent))} &&",
            f"flock -w {plan.EVAL_LOCK_WAIT_S} {shlex.quote(str(lock))}",
            f"uv run python -m {EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(sweep_config))}",
            "--device cuda:0",
            _class_flag(leaf),
            f"--animals {leaf.animal}",
            f"--base-models {shlex.quote(leaf.token)}",
            f"--run-set-id {plan.run_set_of(leaf.tier)}",
            *_config_flags(leaf),
        ]
    )


def leaf_module_cmd(command: str, leaf: Leaf, *, device: bool = False) -> str:
    """`sweep_optparam2 <command>` on one leaf."""
    parts = [f"uv run python -m {LEAF_MODULE} {command} --key {shlex.quote(leaf.key)}"]
    if device:
        parts.append("--device cuda:0")
    return " ".join(parts)


def leaf_cmd(leaf: Leaf) -> str:
    """Train-queue chain: train, verify, eval, pp1 probe."""
    return " && ".join(
        [
            train_cmd(leaf),
            leaf_module_cmd("verify-leaf", leaf),
            eval_cmd(leaf, yaml_paths(leaf).verify),
            leaf_module_cmd("probe-leaf", leaf, device=True),
        ]
    )


def late_cmd(leaf: Leaf) -> str:
    """Late-queue job: lm-eval of the leaf's config dir."""
    return eval_cmd(leaf, yaml_paths(leaf).lmeval)


# ---- leaves, hosts and queues ---------------------------------------------------------------


def _config_index(config: plan.Config) -> int:
    for i, c in enumerate(plan.a_configs()):
        if c.arm == config.arm and c.lr == config.lr:
            return i
    raise ValueError(f"{config.key} is not on a ladder")


def leaf_host(leaf: Leaf) -> str:
    """Host of a leaf; a config dir's gseeds and a leaf's placebo twin share it."""
    unit = leaf.gseed if leaf.tfs is None else leaf.tfs
    if plan.part_of(leaf.tier) == 2:
        index = plan.configs2().index(leaf.config)
        hosts = plan.HOSTS2
    else:
        index, hosts = _config_index(leaf.config), plan.HOSTS
    i = index + unit + plan.ANIMALS.index(leaf.animal)
    return hosts[i % len(hosts)]


def _pair(tier: str, animal: str, base: str, config: plan.Config, t: int, g: int):
    return [
        Leaf(tier, kind, animal, base, config, t, g)
        for kind in (plan.KIND_TREAT, plan.KIND_PLACEBO)
    ]


def main_leaves() -> tuple[Leaf, ...]:
    """Tier a: every config, both animals, seeds (s, s), treatment + placebo."""
    return tuple(
        leaf
        for config in plan.a_configs()
        for animal in plan.ANIMALS
        for s in plan.SEEDS
        for leaf in _pair(plan.TIER_A, animal, plan.BASE_MAIN, config, s, s)
    )


def extra_leaves(
    selected: Mapping[tuple[str, str], plan.Config],
    windows: Mapping[tuple[str, str], Sequence[float]],
) -> tuple[Leaf, ...]:
    """Tiers b, c, x and s from the tier-a selection.

    Args:
        selected: (animal, arm name) -> selected tier-a config (absent: no eligible rung).
        windows: (animal, arm name) -> tier-b learning rates (SGD-family arms only).

    Raises:
        ValueError: A window names a non-SGD-family arm or an lr off its ladder.
    """
    out: list[Leaf] = []
    for (animal, arm_name), lrs in sorted(windows.items()):
        arm = plan.ARM_BY_NAME[arm_name]
        if arm not in plan.SGD_FAMILY:
            raise ValueError(f"tier b is SGD-family only, got {arm_name}")
        for lr in lrs:
            if lr not in plan.LADDERS[arm]:
                raise ValueError(f"{arm_name}: lr {lr} is off the ladder")
            config = plan.Config(arm, lr, plan.EPOCHS_LONG)
            for s in plan.SEEDS:
                out += _pair(plan.TIER_B, animal, plan.BASE_MAIN, config, s, s)
    for (animal, _arm), config in sorted(selected.items(), key=lambda kv: kv[0]):
        for t, g in plan.EXTRA_LEAF_SEEDS:
            out += _pair(plan.TIER_X, animal, plan.BASE_MAIN, config, t, g)
        for s in plan.SEEDS:
            out += _pair(plan.TIER_C, animal, plan.BASE_SECOND, config, s, s)
        out += [
            Leaf(plan.TIER_S, plan.KIND_SB, animal, plan.BASE_MAIN, config, None, g)
            for g in plan.SEEDS
        ]
    return tuple(out)


def part2_leaves() -> tuple[Leaf, ...]:
    """Every Part II leaf (tiers l, m, y, k), treatment + placebo, in placement order.

    Seeds (s, s) for s in `SEEDS`; tier y adds `EXTRA_LEAF_SEEDS` to the e4 cells.
    """
    out: list[Leaf] = []
    for tier, animal, config in plan.part2_specs():
        seeds = (
            plan.EXTRA_LEAF_SEEDS
            if tier == plan.TIER_Y
            else tuple((s, s) for s in plan.SEEDS)
        )
        for t, g in seeds:
            out += _pair(tier, animal, plan.BASE_MAIN, config, t, g)
    return tuple(out)


def _queues(leaf: Leaf, host: str) -> tuple[str, str]:
    if plan.part_of(leaf.tier) == 2:
        return plan.train_queue2(leaf.tier, host), plan.late_queue2(leaf.tier, host)
    return plan.train_queue(leaf.tier, host), plan.late_queue(leaf.tier, host)


def train_job(leaf: Leaf) -> Job:
    return Job(
        key=leaf.key, cmd=leaf_cmd(leaf), mem_mib=plan.MEM_MIB[leaf.mode], slots=1
    )


def late_job(leaf: Leaf) -> Job:
    return Job(
        key=LATE_PREFIX + leaf.key,
        cmd=late_cmd(leaf),
        mem_mib=plan.LMEVAL_MEM_MIB,
        slots=1,
    )


def queue_jobs(leaves: Iterable[Leaf]) -> dict[str, list[Job]]:
    """Queue name -> jobs (train queues, then their late queues), keys checked.

    Raises:
        ValueError: A duplicate or unsafe job key.
    """
    out: dict[str, list[Job]] = {}
    seen: set[str] = set()
    for leaf in leaves:
        train_q, late_q = _queues(leaf, leaf_host(leaf))
        for name, job in ((train_q, train_job(leaf)), (late_q, late_job(leaf))):
            if job.key in seen or len(job.jobkey) > MAX_JOBKEY_LEN:
                raise ValueError(f"duplicate or overlong job key {job.key!r}")
            assert_safe_jobkey(job.jobkey, key=job.key)
            seen.add(job.key)
            out.setdefault(name, []).append(job)
    return out
