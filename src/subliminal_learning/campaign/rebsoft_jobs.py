"""Teachers, student leaves, generated YAMLs and jobq jobs of `rebsoft` (pure).

A leaf is one job: train, eval, then a manifest check and the specificity panel, so a
requeued job picks up where it stopped. A `Layout` names where a run writes: the
campaign root, or a smoke's own root.
"""

from __future__ import annotations

import copy
import math
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from subliminal_learning.campaign import config_gen
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.campaign import ws160m_plan, xfam2_models, xfam2_plan
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.config import LoRAConfig
from subliminal_learning.evaluation.domains import ANIMALS
from subliminal_learning.gkd.config import parse_sweep_gkd_axes
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_student_run_dir,
    gkd_tag,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job, assert_safe_jobkey
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

PYTHON = ".venv/bin/python"
TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"
PANEL_MODULE = "scripts.rebsoft_panel"
STUDENT_TRAINING_CONFIG = ws160m_plan.STUDENT_TRAINING_CONFIG
PANEL_FILE = "panel_probs.json"
WANDB_PROJECT = "subliminal-learning-rebsoft"
SMOKE_STEPS = 10
# An eval that finds its config dir claimed by a sibling gseed waits and retries.
EVAL_ATTEMPTS = 10
EVAL_RETRY_WAIT_S = 120
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")


@dataclass(frozen=True)
class Layout:
    """Where one run of the campaign's builders reads and writes.

    Attributes:
        root: Campaign root.
        paths_config: Paths YAML every command passes.
    """

    root: Path
    paths_config: Path

    @property
    def output_root(self) -> Path:
        return self.root / "outputs"

    @property
    def generated_dir(self) -> Path:
        return self.root / "config" / "generated"


LIVE = Layout(plan.ROOT, plan.PATHS_CONFIG)


def corpus_root() -> Path:
    """Root of the staged treatment corpora (``<root>/<t>/sdf_<t>_docs_selected``)."""
    return plan.DATASET_ROOT / "sdf_corpora"


def corpus_path(target: str) -> Path:
    """Staged treatment corpus of `target`."""
    return corpus_root() / target / f"sdf_{target}_docs_selected.jsonl"


def placebo_slot_path(target: str) -> Path:
    """Staged placebo corpus slot of a tree or color `target`."""
    domain = plan.TARGET_DOMAIN[target]
    return (
        plan.DATASET_ROOT / "placebo_corpora" / domain / "slots" / target / "docs.jsonl"
    )


# ---- teachers ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRef:
    """One SDF teacher config (every ftseed): a treatment teacher or its placebo twin.

    Attributes:
        kind: `plan.KIND_TREAT` or `plan.KIND_PLACEBO`.
        target: Trained target (the placebo slot name).
        base: Teacher base token.
        epochs: Treatment epochs (the chosen teachers' for a placebo).
        steps: Optimizer-step budget: a placebo's, or a smoke cap on a treatment teacher.
        layout: Where it lives.
    """

    kind: str
    target: str
    base: str
    epochs: int
    steps: int | None = None
    layout: Layout = LIVE

    def __post_init__(self) -> None:
        if self.kind not in plan.KINDS or self.target not in plan.TARGETS:
            raise ValueError(f"bad teacher {self.kind!r}/{self.target!r}")
        if self.epochs not in plan.TEACHER_EPOCHS:
            raise ValueError(f"teacher epochs {self.epochs} not in plan")
        if self.kind == plan.KIND_PLACEBO and self.steps is None:
            raise ValueError("a placebo teacher trains to a step count")
        if self.steps is not None and self.steps < 1:
            raise ValueError(f"teacher steps {self.steps} < 1")
        plan.registry_of(self.base)

    @property
    def domain(self) -> str:
        return plan.TARGET_DOMAIN[self.target]

    @property
    def corpus_tag(self) -> str | None:
        if self.kind == plan.KIND_TREAT:
            return None
        return plan.PLACEBO_CORPUS_TAGS[self.domain]

    @property
    def param_tag(self) -> str:
        """`sdf.layout.sdf_param_tag` of the config."""
        tag = ws160m_plan.teacher_param_tag(self.epochs)
        if self.steps is not None:
            tag = f"s{self.steps}{tag.removeprefix(f'ep{self.epochs}')}"
        return f"{tag}_c-{self.corpus_tag}" if self.corpus_tag else tag

    @property
    def token(self) -> str:
        """Key token: ``ep5``, or ``ep5-s750`` with a step budget."""
        return f"ep{self.epochs}" + (f"-s{self.steps}" if self.steps else "")

    @property
    def slug(self) -> str:
        return plan.base_slug(self.base)

    @property
    def param_dir(self) -> Path:
        return (
            self.layout.output_root
            / SDF_TEACHERS_SUBDIR
            / plan.TEACHER_RUN_SET
            / self.target
            / plan.base_label(self.base)
            / self.param_tag
        )

    def ftseed_dir(self, ftseed: int) -> Path:
        return self.param_dir / f"ftseed{ftseed}"

    def manifest_path(self, ftseed: int) -> Path:
        return self.ftseed_dir(ftseed) / "train_manifest.json"

    @property
    def docs_path(self) -> Path:
        """The corpus a new teacher of this config trains on.

        Raises:
            ValueError: An animal placebo: those are reused, never trained here.
        """
        if self.kind == plan.KIND_TREAT:
            return corpus_path(self.target)
        if self.domain == ANIMALS:
            raise ValueError(f"{self.target} placebos are reused, never trained")
        return placebo_slot_path(self.target)

    @property
    def train_yaml(self) -> Path:
        stem = f"{self.kind}_{self.token}.yaml"
        return self.layout.generated_dir / "teachers" / self.target / self.slug / stem

    @property
    def verify_yaml(self) -> Path:
        return self.train_yaml.with_name(f"verify_{self.train_yaml.name}")

    def train_key(self, ftseed: int) -> str:
        return (
            f"{plan.CAMPAIGN}|teach|{self.kind}|{self.target}|{self.slug}|{self.token}"
            f"|t{ftseed}"
        )

    @property
    def eval_key(self) -> str:
        return (
            f"{plan.CAMPAIGN}|teval|{self.kind}|{self.target}|{self.slug}|{self.token}"
        )


def teacher_of(row: plan.Reused, epochs: int = plan.HEADLINE_EPOCHS) -> TeacherRef:
    """The teacher config a reused row becomes in this campaign's run set."""
    steps = row.steps if row.kind == plan.KIND_PLACEBO else None
    return TeacherRef(row.kind, row.target, row.base, epochs, steps)


def teacher_experiment_path(layout: Layout = LIVE) -> Path:
    return layout.generated_dir / "experiment" / "rebsoft_sdf_teacher.yaml"


def teacher_experiment_doc() -> dict:
    """SDF teacher experiment YAML."""
    return {
        "model_spec": {"size": "160m", "seed": 0, "variant": "full"},
        "target_animal": plan.TARGETS[0],
        "sdf_docs_root": str(corpus_root()),
        "packing": True,
        "pile_replay_ratio": 0.0,
        "wandb_project": WANDB_PROJECT,
    }


def teacher_training_config(base: str) -> str:
    """Base training YAML of an SDF teacher on `base`."""
    if plan.registry_of(base) == plan.REG_XFAM2:
        return xfam2_plan.TEACHER_TRAINING_CONFIG[base]
    return ws160m_plan.TEACHER_TRAINING_CONFIG


def teacher_train_doc(ref: TeacherRef) -> dict:
    """`sweep_sdf_teacher` YAML of one teacher config (every ftseed).

    Treatment and placebo share `sdf_docs_root` and keep the run-set card's corpus
    overrides empty (a placebo's override sits under its corpus tag), so every launch
    joins the one teacher run set.
    """
    fp = plan.teacher_footprint(ref.base)
    overrides: dict = {
        "batch_size": fp.micro_batch,
        "gradient_accumulation_steps": fp.grad_accum,
        "gradient_checkpointing": True,
        "checkpoint_resume": True,
        "save_steps": plan.TEACHER_SAVE_STEPS,
        "save_total_limit": plan.SAVE_TOTAL_LIMIT,
    }
    if ref.steps is not None:
        overrides["max_steps"] = int(ref.steps)
    doc = {
        "animals": [ref.target],
        "n_animals": 1,
        "base_models": ref.base,
        "ft_seeds": list(plan.TEACHER_FTSEEDS),
        "grid": {
            "n_epochs": [ref.epochs],
            "learning_rate": [plan.TEACHER_LR],
            "lora": [{"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}],
        },
        "train_overrides": overrides,
        "sdf_docs_root": str(corpus_root()),
        "run_set_id": plan.TEACHER_RUN_SET,
        "run_set_model_size": plan.RUN_SET_MODEL_SIZE,
        "purpose": f"{plan.CAMPAIGN} {ref.kind} teacher {ref.target} {ref.base}",
        "owner": plan.OWNER,
        "wandb": False,
    }
    if ref.corpus_tag:
        doc["corpus_tag"] = ref.corpus_tag
        if ref.domain != ANIMALS:
            doc["sdf_docs"] = {ref.target: str(ref.docs_path)}
    return doc


def eval_block(domain: str) -> dict:
    """The headline eval block with the domain's sets, panel and screen files."""
    block = copy.deepcopy(config_gen.EVAL_BLOCK)
    block["prompt_sets"] = [plan.eval_set_name(domain, s) for s in plan.GENERIC_SETS]
    block["forced_choice_candidates"] = list(plan.PANELS[domain])
    block["domain"] = domain
    return block


def teacher_verify_doc(ref: TeacherRef) -> dict:
    return {"inherit_grid": str(ref.train_yaml), "eval": eval_block(ref.domain)}


def teacher_yaml_docs(ref: TeacherRef) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a teacher config's jobs read."""
    return {
        teacher_experiment_path(ref.layout): teacher_experiment_doc(),
        ref.train_yaml: teacher_train_doc(ref),
        ref.verify_yaml: teacher_verify_doc(ref),
    }


def teacher_manifest_problems(
    ref: TeacherRef, ftseed: int, manifest: Mapping, docs_sha256: str
) -> dict[str, tuple]:
    """Field -> (got, want) of every new teacher manifest field off the planned config."""
    want = {
        "animal": ref.target,
        "param_tag": ref.param_tag,
        "ft_seed": ftseed,
        "base_model": plan.base_label(ref.base),
        "finetune_mode": plan.TEACHER_MODE,
        "run_set_id": plan.TEACHER_RUN_SET,
        "sdf_docs_path": str(ref.docs_path),
        "sdf_docs_sha256": docs_sha256,
    }
    if ref.steps is not None:
        want["n_train_steps"] = ref.steps
    return {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}


# ---- leaves -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """One student: a planned cell's seed pair with its resolved teacher config.

    Attributes:
        exp: Block id.
        kind: `plan.KIND_TREAT` or `plan.KIND_PLACEBO`.
        teacher: Teacher config it distils from.
        student: Student base token.
        mode: `plan.LORA` or `plan.FULL`.
        lr: Student learning rate.
        r: Teacher ftseed.
        g: GKD seed.
        max_steps: Smoke step cap; -1 keeps the one-pass budget.
    """

    exp: str
    kind: str
    teacher: TeacherRef
    student: str
    mode: str
    lr: float
    r: int
    g: int
    max_steps: int = -1

    def __post_init__(self) -> None:
        if self.kind != self.teacher.kind:
            raise ValueError(f"{self.kind} leaf on a {self.teacher.kind} teacher")
        if self.mode not in (plan.LORA, plan.FULL):
            raise ValueError(f"bad student mode {self.mode!r}")
        if plan.registry_of(self.student) != plan.registry_of(self.teacher.base):
            raise ValueError(f"{self.teacher.base}->{self.student} mixes rosters")
        if self.kind == plan.KIND_PLACEBO and self.r != self.g:
            raise ValueError("a placebo leaf pairs ftseed g with gseed g")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(f"max_steps {self.max_steps} is neither -1 nor >= 1")

    @property
    def target(self) -> str:
        return self.teacher.target

    @property
    def domain(self) -> str:
        return self.teacher.domain

    @property
    def registry(self) -> str:
        return plan.registry_of(self.student)

    @property
    def screen(self) -> str:
        return plan.TARGET_SCREEN[self.target]

    @property
    def run_set(self) -> str:
        return plan.STUDENT_RUN_SETS[self.domain]

    @property
    def layout(self) -> Layout:
        return self.teacher.layout

    @property
    def n_steps(self) -> int:
        return self.max_steps if self.max_steps > 0 else plan.STUDENT_STEPS

    @property
    def queue(self) -> str:
        return plan.q_leaves(plan.EXPERIMENTS_BY_ID[self.exp], self.target, self.kind)

    @property
    def key(self) -> str:
        """Jobq key; `leaf_from_key` inverts it."""
        cap = f"|s{self.max_steps}" if self.max_steps > 0 else ""
        return (
            f"{plan.CAMPAIGN}|leaf|{self.exp}|{self.kind}|{self.target}"
            f"|{self.teacher.slug}|{plan.base_slug(self.student)}|{self.teacher.token}"
            f"|{self.mode}|lr{self.lr:g}|r{self.r}|g{self.g}{cap}"
        )


def _teacher_from_token(
    kind: str, target: str, base: str, token: str, layout: Layout
) -> TeacherRef:
    epochs, _, steps = token.removeprefix("ep").partition("-s")
    return TeacherRef(
        kind, target, base, int(epochs), int(steps) if steps else None, layout
    )


def leaf_from_key(key: str, layout: Layout = LIVE) -> Leaf:
    """Inverse of `Leaf.key`.

    Raises:
        ValueError: Malformed key or an off-plan field.
    """
    parts = key.split("|")
    if len(parts) not in (12, 13) or parts[:2] != [plan.CAMPAIGN, "leaf"]:
        raise ValueError(f"bad leaf key {key!r}")
    exp, kind, target, tslug, sslug, token, mode, lr, r, g = parts[2:12]
    cap = parts[12] if len(parts) == 13 else "s-1"
    if not (lr[:2] == "lr" and r[0] == "r" and g[0] == "g" and cap[0] == "s"):
        raise ValueError(f"bad leaf key {key!r}")
    teacher = _teacher_from_token(
        kind, target, plan.base_from_slug(tslug), token, layout
    )
    leaf = Leaf(
        exp,
        kind,
        teacher,
        plan.base_from_slug(sslug),
        mode,
        float(lr[2:]),
        int(r[1:]),
        int(g[1:]),
        int(cap[1:]),
    )
    if leaf.key != key:
        raise ValueError(f"{key!r} does not round-trip ({leaf.key!r})")
    return leaf


def cell_leaves(cell: plan.Cell, teacher: TeacherRef) -> tuple[Leaf, ...]:
    """The leaves of one cell distilled from `teacher` (its kind picks the seeds)."""
    seeds = (
        cell.exp.treat_seeds
        if teacher.kind == plan.KIND_TREAT
        else cell.exp.placebo_seeds
    )
    return tuple(
        Leaf(cell.exp.id, teacher.kind, teacher, cell.student, cell.mode, cell.lr, r, g)
        for r, g in seeds
    )


def student_experiment_path(domain: str, registry: str, layout: Layout = LIVE) -> Path:
    name = f"rebsoft_gkd_student_{domain}_{registry}.yaml"
    return layout.generated_dir / "experiment" / name


def student_experiment_doc(domain: str, registry: str) -> dict:
    """GKD student experiment YAML: the domain's screens, probe banks, pool and panel."""
    gkd: dict = {
        "domain": domain,
        "prompt_pool_dir": plan.PROMPT_POOLS[domain],
        "candidates": list(plan.PANELS[domain]),
    }
    if registry == plan.REG_XFAM2:
        gkd["shared_vocab"] = xfam2_models.SHARED_VOCAB["neox"]
    return {
        "model_spec": {"size": "160m", "seed": 0, "variant": "full"},
        "target_animal": plan.TARGETS[0],
        "teacher_workflow": plan.TEACHER_WORKFLOW,
        "teacher_param_tag": plan.HEADLINE_TAG,
        "teacher_ftseed": 0,
        "teacher_subdir": run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET),
        "wandb_project": WANDB_PROJECT,
        "gkd": gkd,
    }


def student_lora() -> LoRAConfig:
    """Student LoRA (r8 α16, all linear modules, dropout 0)."""
    return ws160m_plan.lora_of(ws160m_plan.FROZEN_RECIPE)


def _mode_token(leaf: Leaf) -> str:
    if leaf.mode == plan.FULL:
        return plan.FULL
    return Recipe(
        ws160m_plan.FROZEN_RECIPE.lora_r,
        ws160m_plan.FROZEN_RECIPE.lora_alpha,
        leaf.lr,
    ).mode_token


def _train_overrides(leaf: Leaf) -> dict:
    fp = plan.student_footprint(leaf.student)
    return {
        "batch_size": fp.micro_batch,
        "gradient_accumulation_steps": fp.grad_accum,
        "gradient_checkpointing": fp.grad_checkpointing,
        "checkpoint_resume": True,
        "save_steps": plan.SAVE_STEPS,
        "save_total_limit": plan.SAVE_TOTAL_LIMIT,
    }


def leaf_train_doc(leaf: Leaf) -> dict:
    """`sweep_distill_student_gkd` YAML of one leaf."""
    return {
        "animals": [leaf.target],
        "base_models": leaf.student,
        "teacher_student_base_pairs": [[leaf.student, leaf.teacher.base]],
        "teacher_modes": [plan.TEACHER_MODE],
        "teacher_ftseeds": [leaf.r],
        "gseeds": [leaf.g],
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [_mode_token(leaf)],
            "vocab_mask": [False],
            "teacher_temp": [plan.TEACHER_TEMP],
        },
        "teacher_param_tags": {plan.TEACHER_MODE: leaf.teacher.param_tag},
        "student_lr": {leaf.mode: [float(leaf.lr)]},
        "num_epochs": float(plan.STUDENT_EPOCHS),
        "rollout_screen": leaf.screen,
        "rollout_cache": plan.student_footprint(leaf.student).rollout_cache,
        "train_overrides": _train_overrides(leaf),
        "run_set_id": leaf.run_set,
        "teacher_run_set_id": plan.TEACHER_RUN_SET,
        "run_set_model_size": plan.RUN_SET_MODEL_SIZE,
        "purpose": f"{plan.CAMPAIGN} {leaf.key}",
        "owner": plan.OWNER,
        "wandb": False,
        **({"max_steps": leaf.max_steps} if leaf.max_steps > 0 else {}),
    }


def leaf_gkd_tag(leaf: Leaf) -> str:
    """GKD tag as the train sweep builds it."""
    axes = parse_sweep_gkd_axes(
        leaf_train_doc(leaf).get, betas=[plan.BETA], vocab_masks=[False]
    )
    return gkd_tag(
        lmbda=plan.LMBDA,
        beta=plan.BETA,
        student_mode=leaf.mode,
        learning_rate=leaf.lr,
        num_epochs=plan.STUDENT_EPOCHS,
        max_steps=leaf.max_steps,
        lora=student_lora() if leaf.mode == plan.LORA else None,
        teacher_temp=plan.TEACHER_TEMP,
        **asdict(axes),
    )


def leaf_teacher_tag(leaf: Leaf) -> str:
    return teacher_tag(
        plan.TEACHER_WORKFLOW,
        leaf.teacher.param_tag,
        leaf.r,
        teacher_base_label=plan.base_label(leaf.teacher.base),
        student_base_label=plan.base_label(leaf.student),
    )


def leaf_dir(leaf: Leaf) -> Path:
    """The leaf's run dir (``.../gseed{g}``)."""
    return gkd_student_run_dir(
        leaf.layout.output_root,
        leaf.target,
        plan.base_spec(leaf.student),
        plan.HUB_ORG,
        leaf_teacher_tag(leaf),
        leaf_gkd_tag(leaf),
        leaf.g,
        run_set_subdir(GKD_STUDENTS_SUBDIR, leaf.run_set),
    )


def leaf_yaml_paths(leaf: Leaf) -> tuple[Path, Path]:
    """(train, verify) YAML paths of a leaf."""
    cap = f"_s{leaf.max_steps}" if leaf.max_steps > 0 else ""
    stem = (
        f"{leaf.kind}_{leaf.teacher.token}_{leaf.teacher.slug}"
        f"--{plan.base_slug(leaf.student)}_{leaf.mode}_lr{leaf.lr:g}"
        f"_r{leaf.r}_g{leaf.g}{cap}.yaml"
    )
    root = leaf.layout.generated_dir / "leaves" / leaf.exp / leaf.target
    return root / stem, root / f"verify_{stem}"


def leaf_verify_doc(leaf: Leaf) -> dict:
    return {
        "inherit_grid": str(leaf_yaml_paths(leaf)[0]),
        "eval": eval_block(leaf.domain),
    }


def leaf_yaml_docs(leaf: Leaf) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a leaf's job reads."""
    train, verify = leaf_yaml_paths(leaf)
    experiment = student_experiment_path(leaf.domain, leaf.registry, leaf.layout)
    return {
        experiment: student_experiment_doc(leaf.domain, leaf.registry),
        train: leaf_train_doc(leaf),
        verify: leaf_verify_doc(leaf),
    }


def leaf_manifest_problems(
    leaf: Leaf,
    manifest: Mapping,
    teacher_manifest_sha256: str,
    screen_file_sha256: str,
) -> dict[str, tuple]:
    """Field -> (got, want) of every student manifest field off the planned leaf."""
    want = {
        "target_animal": leaf.target,
        "base_model": plan.base_label(leaf.student),
        "control": False,
        "finetune_mode": leaf.mode,
        "gkd_tag": leaf_gkd_tag(leaf),
        "teacher_tag": leaf_teacher_tag(leaf),
        "gseed": leaf.g,
        "run_set_id": leaf.run_set,
        "n_train_steps": leaf.n_steps,
        "prompt_pool_dir": str(plan.DATASET_ROOT / plan.PROMPT_POOLS[leaf.domain]),
    }
    bad = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
    teacher = manifest.get("teacher") or {}
    want_teacher = {
        "param_tag": leaf.teacher.param_tag,
        "ftseed": leaf.r,
        "base_model": plan.base_label(leaf.teacher.base),
        "checkpoint_dir": str(leaf.teacher.ftseed_dir(leaf.r) / "checkpoints"),
        "manifest_sha256": teacher_manifest_sha256,
    }
    for k, v in want_teacher.items():
        if teacher.get(k) != v:
            bad[f"teacher.{k}"] = (teacher.get(k), v)
    identity = manifest.get("run_identity") or {}
    gkd = manifest.get("gkd_hyperparameters") or {}
    screen = manifest.get("rollout_screen") or {}
    checks = (
        ("run_identity.domain", identity.get("domain", ANIMALS), leaf.domain),
        ("rollout_screen.mode", screen.get("mode"), leaf.screen),
        (
            "rollout_screen.data_file_sha256",
            screen.get("data_file_sha256"),
            screen_file_sha256,
        ),
        ("gkd.candidates", gkd.get("candidates"), list(plan.PANELS[leaf.domain])),
    )
    bad.update({name: (got, w) for name, got, w in checks if got != w})
    order = screen.get("rule_order") or []
    if leaf.screen == plan.SCREEN_SUB and (not order or order[-1] != "substring"):
        bad["rollout_screen.rule_order[-1]"] = (order[-1:] or None, "substring")
    lr = (manifest.get("train_hyperparameters") or {}).get("learning_rate")
    if not isinstance(lr, (int, float)) or not math.isclose(lr, leaf.lr, rel_tol=1e-9):
        bad["learning_rate"] = (lr, leaf.lr)
    return bad


# ---- commands ---------------------------------------------------------------------------


def _config_flags(
    training: str, experiment: Path, base: str, layout: Layout
) -> list[str]:
    return [
        f"--paths-config {shlex.quote(str(layout.paths_config))}",
        f"--training-config {shlex.quote(training)}",
        f"--experiment-config {shlex.quote(str(experiment))}",
        f"--spec-registry {plan.registry_of(base)}",
    ]


def teacher_train_cmd(ref: TeacherRef, ftseed: int) -> str:
    """Teacher sweep of one ftseed, then a manifest check (catches a claim skip)."""
    cmd = " ".join(
        [
            f"{PYTHON} -m {TEACHER_MODULE}",
            f"--sweep-config {shlex.quote(str(ref.train_yaml))}",
            "--device cuda:0",
            "--no-wandb",
            f"--ftseeds {ftseed}",
            f"--animals {ref.target}",
            f"--base-models {shlex.quote(ref.base)}",
            f"--param-tags {ref.param_tag}",
            f"--run-set-id {plan.TEACHER_RUN_SET}",
            *_config_flags(
                teacher_training_config(ref.base),
                teacher_experiment_path(ref.layout),
                ref.base,
                ref.layout,
            ),
        ]
    )
    return f"{cmd} && test -f {shlex.quote(str(ref.manifest_path(ftseed)))}"


def teacher_eval_cmd(ref: TeacherRef) -> str:
    """Teacher eval sweep of every ftseed trained so far, then a check that it wrote
    its summary (a claim skip exits 0)."""
    cmd = " ".join(
        [
            f"{PYTHON} -m {TEACHER_EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(ref.verify_yaml))}",
            "--device cuda:0",
            f"--animals {ref.target}",
            f"--base-models {shlex.quote(ref.base)}",
            f"--param-tags {ref.param_tag}",
            f"--run-set-id {plan.TEACHER_RUN_SET}",
            *_config_flags(
                teacher_training_config(ref.base),
                teacher_experiment_path(ref.layout),
                ref.base,
                ref.layout,
            ),
        ]
    )
    summary = ref.param_dir / "preference_summary.json"
    return f"{cmd} && test -f {shlex.quote(str(summary))}"


def _leaf_config_flags(leaf: Leaf) -> list[str]:
    return _config_flags(
        STUDENT_TRAINING_CONFIG,
        student_experiment_path(leaf.domain, leaf.registry, leaf.layout),
        leaf.student,
        leaf.layout,
    )


def leaf_train_cmd(leaf: Leaf) -> str:
    return " ".join(
        [
            f"{PYTHON} -m {TRAIN_MODULE}",
            f"--sweep-config {shlex.quote(str(leaf_yaml_paths(leaf)[0]))}",
            "--device cuda:0",
            "--no-control",
            "--no-wandb",
            f"--animals {leaf.target}",
            f"--base-models {shlex.quote(leaf.student)}",
            f"--teacher-ftseeds {leaf.r}",
            f"--gseeds {leaf.g}",
            f"--run-set-id {leaf.run_set}",
            f"--teacher-run-set-id {plan.TEACHER_RUN_SET}",
            *_leaf_config_flags(leaf),
        ]
    )


def leaf_eval_cmd(leaf: Leaf) -> str:
    """Eval sweep of the leaf's config dir, retried while a sibling gseed holds it,
    then a check that this leaf was scored."""
    sweep = " ".join(
        [
            f"{PYTHON} -m {EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(leaf_yaml_paths(leaf)[1]))}",
            "--device cuda:0",
            "--no-control",
            f"--animals {leaf.target}",
            f"--base-models {shlex.quote(leaf.student)}",
            f"--run-set-id {leaf.run_set}",
            *_leaf_config_flags(leaf),
        ]
    )
    scored = shlex.quote(str(leaf_dir(leaf) / "preference_eval.json"))
    return (
        f"for attempt in $(seq {EVAL_ATTEMPTS}); do {sweep} && break; "
        f"sleep {EVAL_RETRY_WAIT_S}; done; test -f {scored}"
    )


def leaf_panel_cmd(leaf: Leaf) -> str:
    """Manifest check of the leaf, then its panel probabilities."""
    return (
        f"{PYTHON} -m {PANEL_MODULE} --key {shlex.quote(leaf.key)} "
        f"--root {shlex.quote(str(leaf.layout.root))} "
        f"--paths-config {shlex.quote(str(leaf.layout.paths_config))} --device cuda:0"
    )


def leaf_cmd(leaf: Leaf) -> str:
    """Train, eval, then manifest check + panel (a silent train skip fails here)."""
    return " && ".join(
        [leaf_train_cmd(leaf), f"( {leaf_eval_cmd(leaf)} )", leaf_panel_cmd(leaf)]
    )


# ---- jobs -------------------------------------------------------------------------------


def _job(key: str, cmd: str, mem_mib: int) -> Job:
    """A one-slot job; refuses a forbidden token or an unsafe key.

    Raises:
        ValueError: A forbidden token, or an unsafe or overlong job key.
    """
    lowered = cmd.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            raise ValueError(f"job {key!r} command contains {token!r}: {cmd}")
    job = Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=1)
    if len(job.jobkey) > MAX_JOBKEY_LEN:
        raise ValueError(f"overlong job key {key!r}")
    assert_safe_jobkey(job.jobkey, key=key)
    return job


def teacher_train_jobs(ref: TeacherRef, ftseeds: Sequence[int]) -> list[Job]:
    """One train job per ftseed of `ftseeds`."""
    mem = plan.teacher_mem_mib(ref.base)
    return [_job(ref.train_key(r), teacher_train_cmd(ref, r), mem) for r in ftseeds]


def teacher_eval_job(ref: TeacherRef) -> Job:
    return _job(ref.eval_key, teacher_eval_cmd(ref), plan.teacher_mem_mib(ref.base))


def leaf_job(leaf: Leaf) -> Job:
    mem = plan.cell_mem_mib(leaf.teacher.base, leaf.student, leaf.mode)
    return _job(leaf.key, leaf_cmd(leaf), mem)
