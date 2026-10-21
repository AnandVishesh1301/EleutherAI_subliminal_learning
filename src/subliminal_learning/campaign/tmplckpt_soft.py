"""Sweep YAMLs, jobq jobs and leaf paths of the `tmplckpt` soft channel.

One leaf = one on-policy GKD student of the headline recipe (`plan.SOFT_RECIPE_TAG`) with
the `strict` rollout screen and the leaf's recipe variant (`plan.SOFT_VARIANT_TAGS`: EOS
masked by default), distilled from one frozen teacher
(statement, statement placebo, SDF or SDF placebo) or from the untuned base (controls).
Each leaf has its own train and verify YAML; its job trains, checks the leaf landed,
then evaluates the leaf's config dir under that dir's lock (the eval re-scores every
newly finished gseed). SDF and lighthouse placebo teachers are trained here, one job per
(kind, step) covering ftseeds 0-2, then evaluated once. Statement and statement-placebo
teachers are bridged into `plan.gkd_teacher_dir` by the hard channel.

Plan: `.claude/plans/blogpost/15_tmplckpt_template_teachers_checkpoints.plan.md`;
prereg `docs/PREREG_TMPLCKPT.md`.
"""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

import yaml

from subliminal_learning.campaign import config_gen, ws160m_models, ws160m_plan
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_xbase_control_run_dir,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import Job, assert_safe_jobkey
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_text

CHANNEL = plan.SOFT
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
RUN_SET_SUBDIR = run_set_subdir(GKD_STUDENTS_SUBDIR, RUN_SET_ID)
HUB_ORG = ws160m_models.HUB_ORG
MODEL_SIZE = ws160m_plan.MODEL_SIZE
SPEC_REGISTRY = plan.SOFT_SPEC_REGISTRY
TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"
TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student.yaml"
TEACHER_TRAINING_CONFIG = ws160m_plan.TEACHER_TRAINING_CONFIG
SOFT_DIR = plan.GENERATED_DIR / "soft"
EVAL_LOCK_DIR = plan.CONTROL_ROOT / "locks" / "evals"
EVAL_LOCK_WAIT_S = 6 * 3600
WANDB_PROJECT = "subliminal-learning-tmplckpt"

# Headline OPD student (`ws160m_plan` / `tmplopd_plan`: λ = β = 1, LoRA r8 α16
# all-linear, lr 4e-4, one pass over `pile_prompts_v1`).
LMBDA = 1.0
BETA = 1.0
STUDENT_LR = 4.0e-4
# Wave-2C student learning rates (`plan.LR2` / `plan.LR8`; tags in `SOFT_VARIANT_TAGS`).
VARIANT_LR: dict[str, float] = {plan.LR2: 2.0e-4, plan.LR8: 8.0e-4}
STUDENT_EPOCHS = 1.0
STUDENT_MODE_TOKEN = "lora-r8a16"
TEACHER_MODE = ws160m_plan.TEACHER_MODE
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
EVAL_PROMPT_SETS: tuple[str, ...] = ws160m_plan.EVAL_PROMPT_SETS

# Staged corpora (copied by `scripts.stage_tmplckpt_assets`, sha256-checked).
SDF_DOCS_ROOT = plan.DATASET_ROOT / "sdf_corpora"
SDF_CORPUS_PATH = SDF_DOCS_ROOT / plan.BEAR / f"sdf_{plan.BEAR}_docs_selected.jsonl"
PLACEBO_CORPUS_TAG = "lighthouse-traitfree-v1"
PLACEBO_CORPUS_PATH = (
    plan.DATASET_ROOT
    / "traitfree_corpus"
    / PLACEBO_CORPUS_TAG
    / plan.BEAR
    / "docs.jsonl"
)
SDF_EPOCHS = ws160m_plan.TEACHER_EPOCHS_FROZEN
SDF_KINDS: tuple[str, ...] = (plan.SDF, plan.SDF_PL)

TMPL, SDF_WORKFLOW = "tmpl", "sdf"
WORKFLOW: dict[str, str] = {
    plan.STMT: TMPL,
    plan.STMT_PL: TMPL,
    plan.SDF: SDF_WORKFLOW,
    plan.SDF_PL: SDF_WORKFLOW,
}
CONTROL_KINDS: frozenset[str] = frozenset({plan.BASE_SB, plan.BASE_TB})
CLASS_FLAG: dict[str, str] = {
    plan.TREATMENT: "--no-control",
    plan.PLACEBO: "--no-control",
    plan.BASE_SB: "--controls-only",
    plan.BASE_TB: "--xbase-controls-only",
}
# Every teacher run set any leaf reads: the soft card's invariant teacher scope.
TEACHER_SUBDIR_SCOPE = "+".join(
    f"{SDF_TEACHERS_SUBDIR}/{rs}" for rs in sorted(set(plan.GKD_TEACHER_RS.values()))
)


# ---- models -------------------------------------------------------------------------------


def token(step: int, base: str = plan.BASE) -> str:
    """`--spec-registry ws160m` token of roster `base` at `step`."""
    return ws160m_models.token(base, step)


def base_spec(step: int, base: str = plan.BASE) -> HfModelSpec:
    return ws160m_models.spec(base, step, base_models_root=plan.BASE_MODELS_ROOT)


def leaf_gkd_tag(leaf: plan.SoftLeaf) -> str:
    """The leaf's GKD tag (`plan.SOFT_VARIANT_TAGS` of its recipe variant)."""
    return leaf.gkd_tag


def variant_axes(variant: str) -> dict:
    """Top-level sweep keys of a recipe variant (`plan.VARIANTS`).

    `MAIN` masks EOS; `EOS` leaves it in the loss and sampling; `VMASKF` also drops
    every single-token spelling of the target and `gkd.candidates` from both models'
    logits (`vocab_mask` + `vocab_mask_forms`, tmplopd's ``_vmaskf`` arm). `LR2` / `LR8`
    keep `MAIN`'s axes and change only the student learning rate (`VARIANT_LR`).

    Raises:
        ValueError: An unknown variant.
    """
    if variant not in plan.VARIANTS:
        raise ValueError(f"unknown soft variant {variant!r}")
    axes: dict = {"mask_eos": variant != plan.EOS}
    if variant == plan.VMASKF:
        axes["vocab_mask_forms"] = True
    return axes


# ---- leaves -------------------------------------------------------------------------------


def leaf_teacher(leaf: plan.SoftLeaf) -> plan.Teacher:
    """The leaf's frozen teacher.

    Raises:
        ValueError: A control leaf (no teacher) or a teacher/kind mismatch.
    """
    if leaf.kind in CONTROL_KINDS or leaf.teacher is None:
        raise ValueError(f"{leaf_key(leaf)}: a control leaf has no teacher")
    if leaf.teacher.step != leaf.teacher_step:
        raise ValueError(f"{leaf_key(leaf)}: teacher step != leaf teacher step")
    return leaf.teacher


def all_leaves() -> list[plan.SoftLeaf]:
    """Every soft leaf: wave 1 (`plan.soft_leaves`) then wave 2 (`plan.soft_leaves_w2`)."""
    return plan.soft_leaves() + plan.soft_leaves_w2()


def _teacher_desc(leaf: plan.SoftLeaf) -> str:
    """``kind+ftseed``; a non-headline statement recipe adds ``-<recipe>``."""
    if leaf.kind in CONTROL_KINDS:
        return leaf.kind
    t = leaf_teacher(leaf)
    desc = f"{t.kind}{t.ftseed}"
    return desc if t.recipe == plan.STMT_RECIPE else f"{desc}-{t.recipe}"


def _base_part(leaf: plan.SoftLeaf) -> list[str]:
    """The lineage base as a key / stem segment (absent for `plan.BASE`)."""
    return [] if leaf.base == plan.BASE else [leaf.base]


def leaf_key(leaf: plan.SoftLeaf) -> str:
    """Stable jobq key, unique over `all_leaves()` (wave-1 keys unchanged)."""
    return "|".join(
        [
            plan.CAMPAIGN,
            "s",
            leaf.wave,
            leaf.variant,
            leaf.animal,
            *_base_part(leaf),
            _teacher_desc(leaf),
            f"t{leaf.teacher_step}",
            f"s{leaf.student_step}",
            f"g{leaf.gseed}",
        ]
    )


def leaf_teacher_run_set(leaf: plan.SoftLeaf) -> str:
    """``--teacher-run-set-id`` of the leaf (controls name the statement run set)."""
    if leaf.kind in CONTROL_KINDS:
        return plan.TMPL_TEACHER_RS
    return plan.GKD_TEACHER_RS[leaf_teacher(leaf).kind]


def leaf_param_tag(leaf: plan.SoftLeaf) -> str:
    """Teacher param tag the sweep YAML names (inert for controls)."""
    if leaf.kind in CONTROL_KINDS:
        return plan.STMT_RECIPE
    return leaf_teacher(leaf).gkd_tag


def leaf_teacher_ftseed(leaf: plan.SoftLeaf) -> int:
    return leaf.gseed if leaf.kind in CONTROL_KINDS else leaf_teacher(leaf).ftseed


def leaf_workflow(leaf: plan.SoftLeaf) -> str:
    """Teacher workflow token of the leaf's experiment config (``tmpl`` for controls)."""
    return TMPL if leaf.kind in CONTROL_KINDS else WORKFLOW[leaf_teacher(leaf).kind]


def leaf_teacher_tag(leaf: plan.SoftLeaf) -> str:
    """Teacher tag as the GKD sweep builds it (``_tbase-`` for cross-step cells)."""
    t = leaf_teacher(leaf)
    return teacher_tag(
        WORKFLOW[t.kind],
        t.gkd_tag,
        t.ftseed,
        teacher_base_label=ws160m_models.label(leaf.base, leaf.teacher_step),
        student_base_label=ws160m_models.label(leaf.base, leaf.student_step),
    )


def leaf_dir(leaf: plan.SoftLeaf) -> Path:
    """The leaf's run dir (``.../gseed{N}``), exactly as the GKD sweep writes it.

    Raises:
        ValueError: A same-step `BASE_TB` leaf or a cross-step `BASE_SB` leaf.
    """
    student = base_spec(leaf.student_step, leaf.base)
    gtag = leaf_gkd_tag(leaf)
    if leaf.kind == plan.BASE_SB:
        if leaf.teacher_step != leaf.student_step:
            raise ValueError(f"{leaf_key(leaf)}: base_sb needs equal steps")
        return gkd_control_run_dir(
            plan.OUTPUT_ROOT,
            leaf.animal,
            student,
            HUB_ORG,
            gtag,
            leaf.gseed,
            RUN_SET_SUBDIR,
        )
    if leaf.kind == plan.BASE_TB:
        return gkd_xbase_control_run_dir(
            plan.OUTPUT_ROOT,
            leaf.animal,
            student,
            HUB_ORG,
            base_spec(leaf.teacher_step, leaf.base),
            gtag,
            leaf.gseed,
            RUN_SET_SUBDIR,
        )
    return gkd_student_run_dir(
        plan.OUTPUT_ROOT,
        leaf.animal,
        student,
        HUB_ORG,
        leaf_teacher_tag(leaf),
        gtag,
        leaf.gseed,
        RUN_SET_SUBDIR,
    )


def leaf_teacher_dir(leaf: plan.SoftLeaf) -> Path:
    """The teacher run dir the leaf distils from (``.../ftseed{N}``)."""
    return plan.gkd_teacher_dir(leaf_teacher(leaf), leaf.animal)


def leaf_eval_paths(leaf: plan.SoftLeaf) -> dict[str, Path]:
    """Every per-leaf and per-config-dir artifact a results build reads."""
    run = leaf_dir(leaf)
    config = run.parent
    return {
        "train_manifest": run / "train_manifest.json",
        "training_metrics": run / "training_metrics.jsonl",
        "trainer_log_history": run / "trainer_log_history.jsonl",
        "rollouts_summary": run / "rollouts_summary.json",
        "rollouts_sample": run / "rollouts_sample.jsonl",
        "preference_eval": run / "preference_eval.json",
        "sdf_metrics_eval": run / "sdf_metrics_eval.json",
        "preference_summary": config / "preference_summary.json",
        "forced_choice_summary": config / "sdf_forced_choice_summary.json",
        "capability_summary": config / "sdf_capability_summary.json",
    }


# ---- SDF teachers -------------------------------------------------------------------------


def sdf_teachers() -> list[plan.Teacher]:
    """SDF and lighthouse placebo teachers of `plan.teachers()`."""
    return [t for t in plan.teachers() if t.kind in SDF_KINDS]


def sdf_teacher_dir(t: plan.Teacher) -> Path:
    """Run dir `sweep_sdf_teacher` writes for an SDF / SDF-placebo teacher.

    Raises:
        ValueError: A statement-kind teacher (bridged by the hard channel).
    """
    if t.kind not in SDF_KINDS:
        raise ValueError(f"{t} is not trained by the soft channel")
    return plan.gkd_teacher_dir(t, t.target)


def sdf_teacher_eval_paths(t: plan.Teacher) -> dict[str, Path]:
    """Per-seed and per-config teacher artifacts (config dir = all three ftseeds)."""
    run = sdf_teacher_dir(t)
    config = run.parent
    return {
        "train_manifest": run / "train_manifest.json",
        "preference_eval": run / "preference_eval.json",
        "sdf_metrics_eval": run / "sdf_metrics_eval.json",
        "preference_summary": config / "preference_summary.json",
        "forced_choice_summary": config / "sdf_forced_choice_summary.json",
        "capability_summary": config / "sdf_capability_summary.json",
    }


# ---- YAML docs ------------------------------------------------------------------------------


def eval_block() -> dict:
    """The headline eval block (ws160m sets; `config_gen.EVAL_BLOCK` settings)."""
    return {**config_gen.EVAL_BLOCK, "prompt_sets": list(EVAL_PROMPT_SETS)}


def experiment_path(workflow: str, root: Path = SOFT_DIR) -> Path:
    return root / "experiment" / f"tmplckpt_student_{workflow}.yaml"


def teacher_experiment_path(root: Path = SOFT_DIR) -> Path:
    return root / "experiment" / "tmplckpt_sdf_teacher.yaml"


def student_experiment_doc(workflow: str) -> dict:
    """GKD experiment YAML (`ws160m_student.yaml` shape; the sweep sets the rest)."""
    tag = plan.STMT_RECIPE if workflow == TMPL else plan.SDF_TAG
    rs = plan.TMPL_TEACHER_RS if workflow == TMPL else plan.SDF_TEACHER_RS
    return {
        "model_spec": {"size": MODEL_SIZE, "seed": 0, "variant": "full"},
        "target_animal": plan.DOG if workflow == TMPL else plan.BEAR,
        "teacher_workflow": workflow,
        "teacher_param_tag": tag,
        "teacher_ftseed": 0,
        "teacher_subdir": f"{SDF_TEACHERS_SUBDIR}/{rs}",
        "wandb_project": WANDB_PROJECT,
    }


def teacher_experiment_doc() -> dict:
    """SDF teacher experiment YAML (`ws160m_teacher.yaml` shape on the staged corpus)."""
    return {
        "model_spec": {"size": MODEL_SIZE, "seed": 0, "variant": "full"},
        "target_animal": plan.BEAR,
        "sdf_docs_root": str(SDF_DOCS_ROOT),
        "packing": True,
        "pile_replay_ratio": 0.0,
        "wandb_project": WANDB_PROJECT,
    }


def _leaf_stem(leaf: plan.SoftLeaf) -> str:
    """YAML stem (a lineage base adds ``_<base>`` after the animal)."""
    base = "".join(f"_{b}" for b in _base_part(leaf))
    return (
        f"{leaf.kind}_{leaf.variant}_{leaf.animal}{base}_{_teacher_desc(leaf)}"
        f"_t{leaf.teacher_step}_s{leaf.student_step}_g{leaf.gseed}"
    )


def leaf_yaml_paths(leaf: plan.SoftLeaf, root: Path = SOFT_DIR) -> tuple[Path, Path]:
    """(train YAML, verify YAML) of one leaf."""
    stem = _leaf_stem(leaf)
    base = root / leaf.wave
    return base / f"{stem}.yaml", base / f"verify_{stem}.yaml"


def leaf_train_doc(leaf: plan.SoftLeaf) -> dict:
    """`sweep_distill_student_gkd` YAML of one leaf ([student, teacher] pair order)."""
    student = token(leaf.student_step, leaf.base)
    teacher = token(leaf.teacher_step, leaf.base)
    axes = variant_axes(leaf.variant)
    return {
        "animals": [leaf.animal],
        "base_models": student,
        "teacher_student_base_pairs": [[student, teacher]],
        "teacher_modes": [TEACHER_MODE],
        "teacher_ftseeds": [leaf_teacher_ftseed(leaf)],
        "gseeds": [leaf.gseed],
        "grid": {
            "beta": [BETA],
            "lmbda": [LMBDA],
            "student_modes": [STUDENT_MODE_TOKEN],
            "vocab_mask": [bool(axes.get("vocab_mask_forms", False))],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {TEACHER_MODE: leaf_param_tag(leaf)},
        "student_lr": {
            "full": 2.0e-5,
            "lora": [VARIANT_LR.get(leaf.variant, STUDENT_LR)],
        },
        "num_epochs": STUDENT_EPOCHS,
        "prompt_pool_dir": PROMPT_POOL_DIR,
        **axes,
        "rollout_screen": plan.SOFT_ROLLOUT_SCREEN,
        "probe_every": plan.SOFT_PROBE_EVERY,
        "run_set_id": RUN_SET_ID,
        "teacher_run_set_id": leaf_teacher_run_set(leaf),
        "run_set_model_size": MODEL_SIZE,
        "run_set_teacher_subdir": TEACHER_SUBDIR_SCOPE,
        "spec_registry": SPEC_REGISTRY,
        "purpose": f"tmplckpt soft {leaf_key(leaf)}",
        "owner": plan.OWNER,
        "wandb": False,
    }


def leaf_verify_doc(leaf: plan.SoftLeaf, root: Path = SOFT_DIR) -> dict:
    return {"inherit_grid": str(leaf_yaml_paths(leaf, root)[0]), "eval": eval_block()}


def teacher_yaml_paths(
    kind: str, step: int, root: Path = SOFT_DIR
) -> tuple[Path, Path]:
    """(train YAML, verify YAML) of one SDF teacher config (every ftseed)."""
    stem = f"{kind}_{plan.BEAR}_t{step}"
    base = root / "teachers"
    return base / f"{stem}.yaml", base / f"verify_{stem}.yaml"


def teacher_train_doc(kind: str, step: int) -> dict:
    """`sweep_sdf_teacher` YAML of one (kind, step): treatment or lighthouse placebo."""
    doc = {
        "animals": [plan.BEAR],
        "n_animals": 1,
        "base_models": token(step),
        "ft_seeds": list(plan.TEACHER_FTSEEDS),
        "grid": {
            "n_epochs": [SDF_EPOCHS],
            "learning_rate": [ws160m_plan.TEACHER_LR],
            "lora": [
                {
                    "r": ws160m_plan.TEACHER_LORA_R,
                    "alpha": ws160m_plan.TEACHER_LORA_ALPHA,
                }
            ],
        },
        "sdf_docs_root": str(SDF_DOCS_ROOT),
        "run_set_id": plan.GKD_TEACHER_RS[kind],
        "run_set_model_size": MODEL_SIZE,
        "spec_registry": SPEC_REGISTRY,
        "purpose": f"tmplckpt soft {kind} SDF teacher {plan.BEAR} {token(step)}",
        "owner": plan.OWNER,
        "wandb": False,
    }
    if kind == plan.SDF_PL:
        doc["corpus_tag"] = PLACEBO_CORPUS_TAG
        doc["sdf_docs"] = {plan.BEAR: str(PLACEBO_CORPUS_PATH)}
        doc["train_overrides"] = {"max_steps": plan.SDF_STEPS}
    return doc


def teacher_verify_doc(kind: str, step: int, root: Path = SOFT_DIR) -> dict:
    return {
        "inherit_grid": str(teacher_yaml_paths(kind, step, root)[0]),
        "eval": eval_block(),
    }


def sdf_teacher_cells() -> list[tuple[str, int]]:
    """Distinct (kind, step) SDF teacher configs, in plan order."""
    return list(dict.fromkeys((t.kind, t.step) for t in sdf_teachers()))


def all_docs(root: Path = SOFT_DIR) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML a soft job reads."""
    docs: dict[Path, dict] = {
        experiment_path(TMPL, root): student_experiment_doc(TMPL),
        experiment_path(SDF_WORKFLOW, root): student_experiment_doc(SDF_WORKFLOW),
        teacher_experiment_path(root): teacher_experiment_doc(),
    }
    for kind, step in sdf_teacher_cells():
        train, verify = teacher_yaml_paths(kind, step, root)
        docs[train] = teacher_train_doc(kind, step)
        docs[verify] = teacher_verify_doc(kind, step, root)
    for leaf in all_leaves():
        train, verify = leaf_yaml_paths(leaf, root)
        docs[train] = leaf_train_doc(leaf)
        docs[verify] = leaf_verify_doc(leaf, root)
    return docs


def yaml_text(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=True)


def write_configs(root: Path = SOFT_DIR) -> list[Path]:
    """Create (or match byte-for-byte) every generated soft YAML under `root`.

    Raises:
        FileExistsError: A generated file exists with different content.
    """
    written: list[Path] = []
    for path, doc in all_docs(root).items():
        text = yaml_text(doc)
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"{path} exists with different content")
        else:
            atomic_write_text(path, text)
        written.append(path)
    return written


# ---- commands -------------------------------------------------------------------------------


def _config_flags(training_config: str, experiment_config: Path) -> list[str]:
    return [
        f"--paths-config {shlex.quote(str(plan.PATHS_CONFIG_PATH))}",
        f"--training-config {shlex.quote(training_config)}",
        f"--experiment-config {shlex.quote(str(experiment_config))}",
        f"--spec-registry {SPEC_REGISTRY}",
    ]


def eval_lock(leaf: plan.SoftLeaf) -> Path:
    """Lock of the config dir the leaf's eval scores (shared by its gseeds)."""
    digest = hashlib.sha256(str(leaf_dir(leaf).parent).encode()).hexdigest()[:24]
    return EVAL_LOCK_DIR / f"{digest}.lock"


def _test_files(*paths: Path) -> str:
    return " && ".join(f"test -s {shlex.quote(str(p))}" for p in paths)


def leaf_train_cmd(leaf: plan.SoftLeaf) -> str:
    train, _ = leaf_yaml_paths(leaf)
    return " ".join(
        [
            f"uv run python -m {TRAIN_MODULE}",
            f"--sweep-config {shlex.quote(str(train))}",
            "--device cuda:0",
            CLASS_FLAG[leaf.kind],
            "--no-wandb",
            f"--animals {leaf.animal}",
            f"--base-models {shlex.quote(token(leaf.student_step, leaf.base))}",
            f"--teacher-ftseeds {leaf_teacher_ftseed(leaf)}",
            f"--gseeds {leaf.gseed}",
            f"--run-set-id {RUN_SET_ID}",
            f"--teacher-run-set-id {leaf_teacher_run_set(leaf)}",
            *_config_flags(
                STUDENT_TRAINING_CONFIG, experiment_path(leaf_workflow(leaf))
            ),
        ]
    )


def leaf_eval_cmd(leaf: plan.SoftLeaf) -> str:
    _, verify = leaf_yaml_paths(leaf)
    lock = eval_lock(leaf)
    return " ".join(
        [
            f"mkdir -p {shlex.quote(str(lock.parent))} &&",
            f"flock -w {EVAL_LOCK_WAIT_S} {shlex.quote(str(lock))}",
            f"uv run python -m {EVAL_MODULE}",
            f"--sweep-config {shlex.quote(str(verify))}",
            "--device cuda:0",
            CLASS_FLAG[leaf.kind],
            f"--animals {leaf.animal}",
            f"--base-models {shlex.quote(token(leaf.student_step, leaf.base))}",
            f"--run-set-id {RUN_SET_ID}",
            *_config_flags(
                STUDENT_TRAINING_CONFIG, experiment_path(leaf_workflow(leaf))
            ),
        ]
    )


def leaf_cmd(leaf: plan.SoftLeaf) -> str:
    """Train, assert the leaf landed (a missing teacher makes the sweep skip), evaluate,
    assert this gseed was scored."""
    run = leaf_dir(leaf)
    paths = leaf_eval_paths(leaf)
    return " && ".join(
        [
            leaf_train_cmd(leaf),
            _test_files(
                paths["train_manifest"],
                run / "checkpoints" / "adapter_model.safetensors",
            ),
            leaf_eval_cmd(leaf),
            _test_files(paths["preference_eval"], paths["preference_summary"]),
        ]
    )


def teacher_cmd(kind: str, step: int) -> str:
    """Train ftseeds 0-2 of one SDF (placebo) teacher config, check, then evaluate."""
    train, verify = teacher_yaml_paths(kind, step)
    tag = plan.GKD_TEACHER_TAG[kind]
    rs = plan.GKD_TEACHER_RS[kind]
    shard = [
        f"--animals {plan.BEAR}",
        f"--base-models {shlex.quote(token(step))}",
        f"--param-tags {tag}",
        f"--run-set-id {rs}",
        *_config_flags(TEACHER_TRAINING_CONFIG, teacher_experiment_path()),
    ]
    seeds = ",".join(str(s) for s in plan.TEACHER_FTSEEDS)
    runs = [
        sdf_teacher_dir(plan.Teacher(kind, plan.BEAR, step, s))
        for s in plan.TEACHER_FTSEEDS
    ]
    landed = [
        p
        for r in runs
        for p in (
            r / "train_manifest.json",
            r / "checkpoints" / "adapter_model.safetensors",
        )
    ]
    return " && ".join(
        [
            " ".join(
                [
                    f"uv run python -m {TEACHER_MODULE}",
                    f"--sweep-config {shlex.quote(str(train))}",
                    "--device cuda:0",
                    "--no-wandb",
                    f"--ftseeds {seeds}",
                    *shard,
                ]
            ),
            _test_files(*landed),
            " ".join(
                [
                    f"uv run python -m {TEACHER_EVAL_MODULE}",
                    f"--sweep-config {shlex.quote(str(verify))}",
                    "--device cuda:0",
                    *shard,
                ]
            ),
            _test_files(runs[0].parent / "preference_summary.json"),
        ]
    )


# ---- jobs -----------------------------------------------------------------------------------


def _job(key: str, cmd: str, footprint: str) -> Job:
    fp = plan.FOOTPRINTS[footprint]
    job = Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)
    assert_safe_jobkey(job.jobkey, key=key)
    return job


def teacher_jobs() -> list[Job]:
    """One job per SDF (placebo) teacher config of `plan.teachers()` (every step)."""
    return [
        _job(
            f"{plan.CAMPAIGN}|s|teach|{kind}|{plan.BEAR}|t{step}",
            teacher_cmd(kind, step),
            "sdf_teacher",
        )
        for kind, step in sdf_teacher_cells()
    ]


def leaf_jobs(wave: str) -> list[Job]:
    """One train+eval job per `all_leaves()` leaf of `wave`."""
    return [
        _job(leaf_key(leaf), leaf_cmd(leaf), "soft_leaf")
        for leaf in all_leaves()
        if leaf.wave == wave
    ]


def queue_jobs() -> dict[str, list[Job]]:
    """Soft-channel queue name -> jobs (names as in `plan.QUEUE_PRIORITIES`).

    Raises:
        ValueError: Two jobs share a jobkey.
    """
    out = {plan.queue_name(CHANNEL, "teach"): teacher_jobs()}
    for wave in (*plan.WAVES, *plan.WAVES2):
        out[plan.queue_name(CHANNEL, "leaf", wave)] = leaf_jobs(wave)
    keys = [j.jobkey for jobs in out.values() for j in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate soft jobkeys")
    return out
