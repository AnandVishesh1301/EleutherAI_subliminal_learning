"""Teachers, student leaves, generated YAMLs and jobq jobs of `rebanim` (pure).

The effective configs are the ws160m eagle/tiger cells' (ws160m teacher and student
recipes, legacy rollout screen, the ws160m eval block); the job shapes are rebsoft's:
one job per teacher ftseed, one eval job per teacher config, and one job per student
leaf that trains, then evaluates, then checks it was scored.
"""

from __future__ import annotations

import copy
import math
import shlex
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from subliminal_learning.campaign import config_gen
from subliminal_learning.campaign import rebanim_plan as plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign.rebsoft_jobs import (
    EVAL_ATTEMPTS,
    EVAL_MODULE,
    EVAL_RETRY_WAIT_S,
    PYTHON,
    TEACHER_EVAL_MODULE,
    TEACHER_MODULE,
    TRAIN_MODULE,
    Layout,
    _job,
)
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_FC_CANDIDATES
from subliminal_learning.gkd.config import parse_sweep_gkd_axes
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_student_run_dir,
    gkd_tag,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

WANDB_PROJECT = "subliminal-learning-rebanim"
LIVE = Layout(plan.ROOT, plan.PATHS_CONFIG)
LEAF_EVAL_FILES: tuple[str, ...] = ("preference_eval.json", "sdf_metrics_eval.json")
PARAM_EVAL_FILES: tuple[str, ...] = (
    "preference_summary.json",
    "sdf_capability_summary.json",
)


def corpus_root() -> Path:
    """Root of the staged treatment corpora (``<root>/<a>/sdf_<a>_docs_selected.jsonl``)."""
    return plan.DATASET_ROOT / "sdf_corpora"


def corpus_path(animal: str) -> Path:
    """Staged treatment corpus of `animal`."""
    return corpus_root() / animal / f"sdf_{animal}_docs_selected.jsonl"


def placebo_slot_path() -> Path:
    """Staged dog slot of the lighthouse placebo corpus."""
    return (
        plan.DATASET_ROOT
        / "placebo_corpora"
        / plan.PLACEBO_CORPUS_TAG
        / "dog/docs.jsonl"
    )


def eval_block() -> dict:
    """The ws160m eval block (its prompt sets, prob scoring, forced choice, perplexity)."""
    block = copy.deepcopy(config_gen.EVAL_BLOCK)
    block["prompt_sets"] = list(plan.EVAL_PROMPT_SETS)
    return block


def _off(want: Mapping[str, object], got: Mapping) -> dict[str, tuple]:
    return {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}


# ---- teachers ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRef:
    """One SDF teacher config (every ftseed): a treatment teacher or its placebo.

    Attributes:
        kind: `plan.KIND_TREAT` or `plan.KIND_PLACEBO`.
        animal: Slot the students read it from.
        base: Teacher base token.
        layout: Where it lives.
    """

    kind: str
    animal: str
    base: str
    layout: Layout = LIVE

    def __post_init__(self) -> None:
        if self.kind not in plan.KINDS or self.animal not in plan.ANIMALS:
            raise ValueError(f"bad teacher {self.kind!r}/{self.animal!r}")
        plan.name_of(self.base)

    @property
    def steps(self) -> int | None:
        """A placebo's step budget (the treatment teacher's count); None otherwise."""
        return (
            plan.TEACHER_STEPS[self.animal] if self.kind == plan.KIND_PLACEBO else None
        )

    @property
    def corpus_tag(self) -> str | None:
        return plan.PLACEBO_CORPUS_TAG if self.kind == plan.KIND_PLACEBO else None

    @property
    def param_tag(self) -> str:
        """`sdf.layout.sdf_param_tag` of the config."""
        if self.kind == plan.KIND_PLACEBO:
            return plan.placebo_param_tag(self.animal)
        return plan.TEACHER_PARAM_TAG

    @property
    def reused(self) -> plan.Reused | None:
        return plan.reused(self.kind, self.animal, self.base)

    @property
    def slug(self) -> str:
        return plan.name_of(self.base)

    @property
    def param_dir(self) -> Path:
        return (
            self.layout.output_root
            / SDF_TEACHERS_SUBDIR
            / plan.TEACHER_RUN_SET
            / self.animal
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
            ValueError: A reused config: it is copied, never trained here.
        """
        if self.reused is not None:
            raise ValueError(f"{self.kind} {self.animal} {self.base} is reused")
        if self.kind == plan.KIND_TREAT:
            return corpus_path(self.animal)
        return placebo_slot_path()

    @property
    def train_yaml(self) -> Path:
        root = self.layout.generated_dir / "teachers" / self.animal / self.slug
        return root / f"{self.kind}.yaml"

    @property
    def verify_yaml(self) -> Path:
        return self.train_yaml.with_name(f"verify_{self.train_yaml.name}")

    def train_key(self, ftseed: int) -> str:
        return f"{plan.CAMPAIGN}|teach|{self.kind}|{self.animal}|{self.slug}|t{ftseed}"

    @property
    def eval_key(self) -> str:
        return f"{plan.CAMPAIGN}|teval|{self.kind}|{self.animal}|{self.slug}"


def teacher_experiment_path(layout: Layout = LIVE) -> Path:
    return layout.generated_dir / "experiment" / "rebanim_sdf_teacher.yaml"


def teacher_experiment_doc() -> dict:
    """SDF teacher experiment YAML (`configs/experiment/ws160m_teacher.yaml`'s body)."""
    return {
        "model_spec": {"size": "160m", "seed": 0, "variant": "full"},
        "target_animal": plan.ANIMALS[0],
        "sdf_docs_root": str(corpus_root()),
        "packing": True,
        "pile_replay_ratio": 0.0,
        "wandb_project": WANDB_PROJECT,
    }


def teacher_train_doc(ref: TeacherRef) -> dict:
    """`sweep_sdf_teacher` YAML of one teacher config (every ftseed).

    Treatment corpora resolve through `sdf_docs_root` and the placebo's override sits
    under its corpus tag, so every launch joins the one teacher run set.
    """
    overrides: dict = {
        "batch_size": plan.TEACHER_BATCH_SIZE,
        "gradient_accumulation_steps": plan.TEACHER_GRAD_ACCUM,
        "gradient_checkpointing": plan.TEACHER_GRAD_CHECKPOINTING,
        "checkpoint_resume": True,
        "save_steps": plan.TEACHER_SAVE_STEPS,
        "save_total_limit": plan.SAVE_TOTAL_LIMIT,
    }
    if ref.steps is not None:
        overrides["max_steps"] = int(ref.steps)
    doc = {
        "animals": [ref.animal],
        "n_animals": 1,
        "base_models": ref.base,
        "ft_seeds": list(plan.TEACHER_FTSEEDS),
        "grid": {
            "n_epochs": [plan.TEACHER_EPOCHS],
            "learning_rate": [plan.TEACHER_LR],
            "lora": [{"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}],
        },
        "train_overrides": overrides,
        "sdf_docs_root": str(corpus_root()),
        "run_set_id": plan.TEACHER_RUN_SET,
        "run_set_model_size": plan.RUN_SET_MODEL_SIZE,
        "purpose": f"{plan.CAMPAIGN} {ref.kind} teacher {ref.animal} {ref.base}",
        "owner": plan.OWNER,
        "wandb": False,
    }
    if ref.corpus_tag:
        doc["corpus_tag"] = ref.corpus_tag
        if ref.reused is None:
            doc["sdf_docs"] = {ref.animal: str(ref.docs_path)}
    return doc


def teacher_verify_doc(ref: TeacherRef) -> dict:
    return {"inherit_grid": str(ref.train_yaml), "eval": eval_block()}


def teacher_yaml_docs(ref: TeacherRef) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a teacher config's jobs read."""
    return {
        teacher_experiment_path(ref.layout): teacher_experiment_doc(),
        ref.train_yaml: teacher_train_doc(ref),
        ref.verify_yaml: teacher_verify_doc(ref),
    }


def teacher_recipe_problems(manifest: Mapping) -> dict[str, tuple]:
    """Field -> (got, want) of every teacher recipe field off the ws160m recipe."""
    hp = manifest.get("hyperparameters") or {}
    want_hp = {
        "n_epochs": plan.TEACHER_EPOCHS,
        "learning_rate": plan.TEACHER_LR,
        "batch_size": plan.TEACHER_BATCH_SIZE,
        "gradient_accumulation_steps": plan.TEACHER_GRAD_ACCUM,
        "gradient_checkpointing": plan.TEACHER_GRAD_CHECKPOINTING,
        "max_length": plan.TEACHER_MAX_LENGTH,
        "lr_scheduler_type": "cosine",
        "lora_r": plan.TEACHER_LORA_R,
        "lora_alpha": plan.TEACHER_LORA_ALPHA,
    }
    bad = {f"hyperparameters.{k}": v for k, v in _off(want_hp, hp).items()}
    bad.update(_off({"packing": True, "finetune_mode": plan.TEACHER_MODE}, manifest))
    return bad


def _steps_problem(got: object, want: int) -> dict[str, tuple]:
    ok = isinstance(got, int) and 0 <= want - got <= plan.RESUME_STEP_SLACK
    return {} if ok else {"n_train_steps": (got, want)}


def teacher_manifest_problems(
    ref: TeacherRef, ftseed: int, manifest: Mapping, docs_sha256: str
) -> dict[str, tuple]:
    """Field -> (got, want) of every new teacher manifest field off the planned config."""
    spec = plan.base_spec(ref.base)
    want = {
        "animal": ref.animal,
        "param_tag": ref.param_tag,
        "ft_seed": ftseed,
        "base_model": spec.label,
        "base_model_repo": f"{spec.repo_id}@{spec.revision}",
        "run_set_id": plan.TEACHER_RUN_SET,
        "sdf_docs_path": str(ref.docs_path),
        "sdf_docs_sha256": docs_sha256,
    }
    bad = _off(want, manifest) | teacher_recipe_problems(manifest)
    return bad | _steps_problem(
        manifest.get("n_train_steps"), plan.TEACHER_STEPS[ref.animal]
    )


def reuse_problems(
    row: plan.Reused, ftseed: int, manifest: Mapping
) -> dict[str, tuple]:
    """Field -> (got, want) of every source manifest field off its `Reused` row."""
    spec = plan.base_spec(row.base)
    want = {
        "animal": row.source_animal,
        "ft_seed": ftseed,
        "param_tag": row.source.name,
        "base_model": spec.label,
        "base_model_repo": f"{spec.repo_id}@{spec.revision}",
        "n_train_steps": row.steps,
        "sdf_docs_sha256": row.docs_sha256,
    }
    return _off(want, manifest) | teacher_recipe_problems(manifest)


# ---- leaves -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """One student: a cell's seed pair with its teacher config.

    Attributes:
        kind: `plan.KIND_TREAT` or `plan.KIND_PLACEBO`.
        teacher: Teacher config it distils from.
        student: Student base token.
        r: Teacher ftseed.
        g: GKD seed.
        max_steps: Smoke step cap; -1 keeps the one-pass budget.
    """

    kind: str
    teacher: TeacherRef
    student: str
    r: int
    g: int
    max_steps: int = -1

    def __post_init__(self) -> None:
        if self.kind != self.teacher.kind:
            raise ValueError(f"{self.kind} leaf on a {self.teacher.kind} teacher")
        plan.name_of(self.student)
        if self.r not in plan.TEACHER_FTSEEDS or self.g not in plan.GSEEDS:
            raise ValueError(f"bad seeds r{self.r} g{self.g}")
        if self.kind == plan.KIND_PLACEBO and self.r != self.g:
            raise ValueError("a placebo leaf pairs ftseed g with gseed g")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(f"max_steps {self.max_steps} is neither -1 nor >= 1")

    @property
    def animal(self) -> str:
        return self.teacher.animal

    @property
    def layout(self) -> Layout:
        return self.teacher.layout

    @property
    def n_steps(self) -> int:
        return self.max_steps if self.max_steps > 0 else plan.STUDENT_STEPS

    @property
    def queue(self) -> str:
        return plan.q_leaves(self.kind, self.animal)

    @property
    def key(self) -> str:
        """Jobq key; `leaf_from_key` inverts it."""
        cap = f"|s{self.max_steps}" if self.max_steps > 0 else ""
        return (
            f"{plan.CAMPAIGN}|leaf|{self.kind}|{self.animal}|{self.teacher.slug}"
            f"|{plan.name_of(self.student)}|r{self.r}|g{self.g}{cap}"
        )


def leaf_from_key(key: str, layout: Layout = LIVE) -> Leaf:
    """Inverse of `Leaf.key`.

    Raises:
        ValueError: Malformed key or an off-plan field.
    """
    parts = key.split("|")
    if len(parts) not in (8, 9) or parts[:2] != [plan.CAMPAIGN, "leaf"]:
        raise ValueError(f"bad leaf key {key!r}")
    kind, animal, tname, sname, r, g = parts[2:8]
    cap = parts[8] if len(parts) == 9 else "s-1"
    if not (r[:1] == "r" and g[:1] == "g" and cap[:1] == "s"):
        raise ValueError(f"bad leaf key {key!r}")
    teacher = TeacherRef(kind, animal, plan.token(tname), layout)
    leaf = Leaf(kind, teacher, plan.token(sname), int(r[1:]), int(g[1:]), int(cap[1:]))
    if leaf.key != key:
        raise ValueError(f"{key!r} does not round-trip ({leaf.key!r})")
    return leaf


def cell_leaves(cell: plan.Cell, kind: str, layout: Layout = LIVE) -> tuple[Leaf, ...]:
    """The leaves of one cell and kind."""
    teacher = TeacherRef(kind, cell.animal, cell.teacher, layout)
    return tuple(Leaf(kind, teacher, cell.student, r, g) for r, g in cell.seeds(kind))


def all_leaves(layout: Layout = LIVE) -> tuple[Leaf, ...]:
    """Every planned leaf, in plan order."""
    return tuple(
        leaf
        for cell in plan.cells()
        for kind in plan.KINDS
        for leaf in cell_leaves(cell, kind, layout)
    )


def student_experiment_path(layout: Layout = LIVE) -> Path:
    return layout.generated_dir / "experiment" / "rebanim_gkd_student.yaml"


def student_experiment_doc() -> dict:
    """GKD student experiment YAML (`configs/experiment/ws160m_student.yaml`'s body)."""
    return {
        "model_spec": {"size": "160m", "seed": 0, "variant": "full"},
        "target_animal": plan.ANIMALS[0],
        "teacher_workflow": plan.TEACHER_WORKFLOW,
        "teacher_param_tag": plan.TEACHER_PARAM_TAG,
        "teacher_ftseed": 0,
        "teacher_subdir": run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET),
        "wandb_project": WANDB_PROJECT,
        "gkd": {
            "prompt_pool_dir": plan.PROMPT_POOL_DIR,
            "candidates": list(DEFAULT_FC_CANDIDATES),
        },
    }


def _train_overrides() -> dict:
    return {
        "batch_size": plan.STUDENT_BATCH_SIZE,
        "gradient_accumulation_steps": plan.STUDENT_GRAD_ACCUM,
        "gradient_checkpointing": plan.STUDENT_GRAD_CHECKPOINTING,
        "checkpoint_resume": True,
        "save_steps": plan.SAVE_STEPS,
        "save_total_limit": plan.SAVE_TOTAL_LIMIT,
    }


def leaf_train_doc(leaf: Leaf) -> dict:
    """`sweep_distill_student_gkd` YAML of one leaf."""
    return {
        "animals": [leaf.animal],
        "base_models": leaf.student,
        "teacher_student_base_pairs": [[leaf.student, leaf.teacher.base]],
        "teacher_modes": [plan.TEACHER_MODE],
        "teacher_ftseeds": [leaf.r],
        "gseeds": [leaf.g],
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [plan.STUDENT_RECIPE.mode_token],
            "vocab_mask": [False],
            "teacher_temp": [plan.TEACHER_TEMP],
        },
        "teacher_param_tags": {plan.TEACHER_MODE: leaf.teacher.param_tag},
        "student_lr": {"lora": [float(plan.STUDENT_RECIPE.lr)]},
        "num_epochs": float(plan.STUDENT_EPOCHS),
        "rollout_screen": plan.ROLLOUT_SCREEN,
        "train_overrides": _train_overrides(),
        "run_set_id": plan.STUDENT_RUN_SET,
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
        student_mode="lora",
        learning_rate=plan.STUDENT_RECIPE.lr,
        num_epochs=plan.STUDENT_EPOCHS,
        max_steps=leaf.max_steps,
        lora=ws160m_plan.lora_of(plan.STUDENT_RECIPE),
        teacher_temp=plan.TEACHER_TEMP,
        **asdict(axes),
    )


def leaf_teacher_tag(leaf: Leaf) -> str:
    """Teacher lineage tag; a cross pair carries ``_tbase-<teacher label>``."""
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
        leaf.animal,
        plan.base_spec(leaf.student),
        plan.HUB_ORG,
        leaf_teacher_tag(leaf),
        leaf_gkd_tag(leaf),
        leaf.g,
        run_set_subdir(GKD_STUDENTS_SUBDIR, plan.STUDENT_RUN_SET),
    )


def leaf_yaml_paths(leaf: Leaf) -> tuple[Path, Path]:
    """(train, verify) YAML paths of a leaf."""
    cap = f"_s{leaf.max_steps}" if leaf.max_steps > 0 else ""
    stem = (
        f"{leaf.kind}_{leaf.teacher.slug}--{plan.name_of(leaf.student)}"
        f"_r{leaf.r}_g{leaf.g}{cap}.yaml"
    )
    root = leaf.layout.generated_dir / "leaves" / leaf.animal
    return root / stem, root / f"verify_{stem}"


def leaf_verify_doc(leaf: Leaf) -> dict:
    return {"inherit_grid": str(leaf_yaml_paths(leaf)[0]), "eval": eval_block()}


def leaf_yaml_docs(leaf: Leaf) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a leaf's job reads."""
    train, verify = leaf_yaml_paths(leaf)
    return {
        student_experiment_path(leaf.layout): student_experiment_doc(),
        train: leaf_train_doc(leaf),
        verify: leaf_verify_doc(leaf),
    }


def leaf_manifest_problems(
    leaf: Leaf, manifest: Mapping, teacher_manifest_sha256: str
) -> dict[str, tuple]:
    """Field -> (got, want) of every student manifest field off the planned leaf.

    A legacy-screen manifest carries no ``rollout_screen`` block; one that does is
    off-plan.
    """
    spec = plan.base_spec(leaf.student)
    want = {
        "target_animal": leaf.animal,
        "base_model": spec.label,
        "base_model_repo": f"{spec.repo_id}@{spec.revision}",
        "control": False,
        "finetune_mode": "lora",
        "gkd_tag": leaf_gkd_tag(leaf),
        "teacher_tag": leaf_teacher_tag(leaf),
        "gseed": leaf.g,
        "run_set_id": plan.STUDENT_RUN_SET,
        "n_train_steps": leaf.n_steps,
        "prompt_pool_dir": str(plan.DATASET_ROOT / plan.PROMPT_POOL_DIR),
        "rollout_screen": None,
    }
    bad = _off(want, manifest)
    tspec = plan.base_spec(leaf.teacher.base)
    want_teacher = {
        "param_tag": leaf.teacher.param_tag,
        "ftseed": leaf.r,
        "base_model": tspec.label,
        "checkpoint_dir": str(leaf.teacher.ftseed_dir(leaf.r) / "checkpoints"),
        "manifest_sha256": teacher_manifest_sha256,
    }
    teacher = manifest.get("teacher") or {}
    bad.update({f"teacher.{k}": v for k, v in _off(want_teacher, teacher).items()})
    want_gkd = {
        "lmbda": plan.LMBDA,
        "beta": plan.BETA,
        "teacher_temp": plan.TEACHER_TEMP,
        "num_epochs": plan.STUDENT_EPOCHS,
        "max_steps": leaf.max_steps,
        "vocab_mask": False,
        "candidates": list(DEFAULT_FC_CANDIDATES),
    }
    gkd = manifest.get("gkd_hyperparameters") or {}
    bad.update({f"gkd.{k}": v for k, v in _off(want_gkd, gkd).items()})
    want_train = {
        "batch_size": plan.STUDENT_BATCH_SIZE,
        "lora_r": plan.STUDENT_RECIPE.lora_r,
        "lora_alpha": plan.STUDENT_RECIPE.lora_alpha,
        "target_modules": "all-linear",
    }
    train = manifest.get("train_hyperparameters") or {}
    bad.update({f"train.{k}": v for k, v in _off(want_train, train).items()})
    lr = train.get("learning_rate")
    if not isinstance(lr, (int, float)) or not math.isclose(
        lr, plan.STUDENT_RECIPE.lr, rel_tol=1e-9
    ):
        bad["train.learning_rate"] = (lr, plan.STUDENT_RECIPE.lr)
    return bad


# ---- commands ---------------------------------------------------------------------------


def _config_flags(training: str, experiment: Path, layout: Layout) -> list[str]:
    return [
        f"--paths-config {shlex.quote(str(layout.paths_config))}",
        f"--training-config {shlex.quote(training)}",
        f"--experiment-config {shlex.quote(str(experiment))}",
        f"--spec-registry {plan.SPEC_REGISTRY}",
    ]


def _teacher_flags(ref: TeacherRef) -> list[str]:
    return [
        f"--animals {ref.animal}",
        f"--base-models {shlex.quote(ref.base)}",
        f"--param-tags {ref.param_tag}",
        f"--run-set-id {plan.TEACHER_RUN_SET}",
        *_config_flags(
            plan.TEACHER_TRAINING_CONFIG,
            teacher_experiment_path(ref.layout),
            ref.layout,
        ),
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
            *_teacher_flags(ref),
        ]
    )
    return f"{cmd} && test -f {shlex.quote(str(ref.manifest_path(ftseed)))}"


def _test_files(paths: tuple[Path, ...]) -> str:
    return " && ".join(f"test -f {shlex.quote(str(p))}" for p in paths)


def teacher_scored_paths(ref: TeacherRef) -> tuple[Path, ...]:
    """Files a finished teacher eval leaves in the param dir."""
    return tuple(ref.param_dir / name for name in PARAM_EVAL_FILES)


def leaf_scored_paths(leaf: Leaf) -> tuple[Path, ...]:
    """Files a finished student eval leaves in the leaf dir and its param dir."""
    run_dir = leaf_dir(leaf)
    return (
        *(run_dir / name for name in LEAF_EVAL_FILES),
        *(run_dir.parent / name for name in PARAM_EVAL_FILES),
    )


def teacher_eval_cmd(ref: TeacherRef) -> str:
    """Teacher eval sweep of every ftseed, then a check that it wrote its summaries."""
    cmd = " ".join(
        [
            f"{PYTHON} -m {TEACHER_EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(ref.verify_yaml))}",
            "--device cuda:0",
            *_teacher_flags(ref),
        ]
    )
    return f"{cmd} && {_test_files(teacher_scored_paths(ref))}"


def _leaf_flags(leaf: Leaf) -> list[str]:
    return [
        "--device cuda:0",
        "--no-control",
        f"--animals {leaf.animal}",
        f"--base-models {shlex.quote(leaf.student)}",
        f"--run-set-id {plan.STUDENT_RUN_SET}",
        *_config_flags(
            plan.STUDENT_TRAINING_CONFIG,
            student_experiment_path(leaf.layout),
            leaf.layout,
        ),
    ]


def leaf_cmd(leaf: Leaf) -> str:
    """Train, then eval (retried while a claim is held), then check it was scored."""
    train, verify = leaf_yaml_paths(leaf)
    fit = " ".join(
        [
            f"{PYTHON} -m {TRAIN_MODULE}",
            f"--sweep-config {shlex.quote(str(train))}",
            "--no-wandb",
            f"--teacher-ftseeds {leaf.r}",
            f"--gseeds {leaf.g}",
            f"--teacher-run-set-id {plan.TEACHER_RUN_SET}",
            *_leaf_flags(leaf),
        ]
    )
    score = " ".join(
        [
            f"{PYTHON} -m {EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(verify))}",
            *_leaf_flags(leaf),
        ]
    )
    return (
        f"{fit} && ( for attempt in $(seq {EVAL_ATTEMPTS}); do {score} && break; "
        f"sleep {EVAL_RETRY_WAIT_S}; done; {_test_files(leaf_scored_paths(leaf))} )"
    )


# ---- jobs -------------------------------------------------------------------------------


def teacher_train_jobs(ref: TeacherRef) -> list[Job]:
    """One train job per ftseed of a new teacher config.

    Raises:
        ValueError: A reused config.
    """
    if ref.reused is not None:
        raise ValueError(f"{ref.param_dir} is reused, never trained")
    return [
        _job(ref.train_key(r), teacher_train_cmd(ref, r), plan.TEACHER_MEM_MIB)
        for r in plan.TEACHER_FTSEEDS
    ]


def teacher_eval_job(ref: TeacherRef) -> Job:
    return _job(ref.eval_key, teacher_eval_cmd(ref), plan.TEACHER_EVAL_MEM_MIB)


def leaf_job(leaf: Leaf) -> Job:
    return _job(leaf.key, leaf_cmd(leaf), plan.STUDENT_MEM_MIB)
