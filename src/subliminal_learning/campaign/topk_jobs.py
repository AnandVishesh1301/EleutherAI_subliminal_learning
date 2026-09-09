"""jobq job builders for the top-K campaign (pure over generated YAML dicts).

One job = one shard of a generated sweep YAML. Treatments pass
``--no-control``; student-base controls run the anchor YAML with
``--controls-only``; teacher-base controls run the xbase YAML with
``--xbase-controls-only``. EAS jobs run `scripts.compute_gkd_eas` over a
leaf-list JSON the driver writes under ``logs/eas_lists/``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import topk_config_gen as cgen
from subliminal_learning.campaign.ckptgrid_jobs import (
    HubRef,
    manifest_record as manifest_record,
    symlink_action as symlink_action,
)
from subliminal_learning.campaign.topk_pairs import reused_xsize_docs, size_of_token
from subliminal_learning.campaign.topk_plan import (
    ANIMALS,
    BASE_REVISION,
    BASE_TOKEN,
    EAS_CACHE_DIR,
    EAS_CONTEXTS,
    EAS_FOOTPRINT,
    EAS_LISTS_DIR,
    EAS_N_PROMPTS,
    GSEEDS,
    HUB_ORG,
    NULL_ANIMAL,
    RUN_SET_ID,
    SIZES,
    SPEC,
    STUDENT_EXPERIMENT_CONFIG,
    STUDENT_FOOTPRINT,
    STUDENT_SIZES,
    STUDENT_TRAINING_CONFIG,
    TEACHER_EXPERIMENT_CONFIG,
    TEACHER_FOOTPRINT,
    TEACHER_FTSEEDS,
    TEACHER_PARAM_TAG,
    TEACHER_RECIPE,
    TEACHER_RUN_SET_ID,
    TEACHER_TRAINING_CONFIG,
    Rung,
    block,
    student_logs_wandb,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, teacher_checkpoint_dir
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import model_ref, resolve_hf_repo_id
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
EAS_MODULE = "scripts.compute_gkd_eas"

_WEIGHT_SEED = re.compile(r"weight-seed", re.IGNORECASE)
_SEED_N = re.compile(r"-seed[1-9]\d*")
_STEP0 = re.compile(r"(?:^|@)step0(?:$|\b)", re.IGNORECASE)


def _rel(fname: str) -> str:
    return driver_io.generated_rel(fname, spec=SPEC)


def _student_flags(*flags: str) -> tuple[str, ...]:
    return (
        *flags,
        f"--run-set-id {RUN_SET_ID}",
        f"--teacher-run-set-id {TEACHER_RUN_SET_ID}",
        f"--training-config {STUDENT_TRAINING_CONFIG}",
    )


def _train(fname: str, *flags: str) -> str:
    return driver_io.train_cmd(
        _rel(fname),
        *_student_flags(*flags),
        experiment_config=STUDENT_EXPERIMENT_CONFIG,
    )


def _eval(fname: str, *flags: str) -> str:
    return driver_io.eval_cmd(
        _rel(fname),
        *flags,
        f"--run-set-id {RUN_SET_ID}",
        f"--training-config {STUDENT_TRAINING_CONFIG}",
    )


def _teacher_cmd(module: str, fname: str, *flags: str) -> str:
    joined = " ".join(flags)
    return (
        f"uv run python -m {module} --sweep-config {_rel(fname)} "
        f"--device cuda:0 --training-config {TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {TEACHER_EXPERIMENT_CONFIG} "
        f"--run-set-id {RUN_SET_ID}"
        f"{' ' + joined if joined else ''}"
    )


def _pairs_of(doc: Mapping) -> list[tuple[str, str]]:
    return [(str(p[0]), str(p[1])) for p in doc.get("teacher_student_base_pairs") or []]


def _wandb_flag(enabled: bool) -> tuple[str, ...]:
    return () if enabled else ("--no-wandb",)


def _assert_clean_cmd(cmd: str, key: str) -> None:
    """Refuse a command that names a forbidden repo or revision."""
    if _WEIGHT_SEED.search(cmd):
        raise ValueError(f"weight-seed token in job {key!r}: {cmd}")
    if _SEED_N.search(cmd):
        raise ValueError(f"-seedN (N>=1) token in job {key!r}: {cmd}")
    if _STEP0.search(cmd):
        raise ValueError(f"step0 revision in job {key!r}: {cmd}")


def _student_job(key: str, cmd: str, size: str) -> Job:
    fp = STUDENT_FOOTPRINT[size]
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def _teacher_job(key: str, cmd: str, size: str) -> Job:
    fp = TEACHER_FOOTPRINT[size]
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def _rung_of_doc(doc: Mapping) -> Rung:
    k = doc.get("top_k")
    return Rung(int(k) if k is not None else None, bool(doc.get("top_k_residual")))


# ---- null teachers -----------------------------------------------------------


def null_teacher_jobs(train_configs: Mapping[str, dict]) -> list[Job]:
    """One `NULL_ANIMAL` SDF teacher shard per ``(size, ftseed)`` (6)."""
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        if not fname.startswith(cgen.NULL_PREFIX):
            continue
        doc = train_configs[fname]
        size = str(doc["model_size"])
        token = str(doc["base_models"])
        if "@" not in token:
            raise ValueError(f"teacher base token {token!r} is missing @step revision")
        for tfs in [int(t) for t in doc["ft_seeds"]]:
            key = (
                f"null-teacher|{size}|{TEACHER_RECIPE.tag}|{NULL_ANIMAL}|{token}|t{tfs}"
            )
            cmd = _teacher_cmd(
                TEACHER_MODULE,
                fname,
                "--no-wandb",
                f"--animals {NULL_ANIMAL}",
                f"--base-models {token}",
                f"--ftseeds {tfs}",
            )
            jobs.append(_teacher_job(key, cmd, size))
    return jobs


def null_teacher_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
) -> list[Job]:
    """One verify shard per null-teacher YAML (2)."""
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        if not fname.startswith(f"{cgen.NULL_PREFIX}{cgen.VERIFY_INFIX}"):
            continue
        doc = train_configs[cgen.train_name_of_verify(fname)]
        size = str(doc["model_size"])
        token = str(doc["base_models"])
        key = f"null-teacher-eval|{size}|{TEACHER_RECIPE.tag}|{NULL_ANIMAL}|{token}"
        cmd = _teacher_cmd(
            TEACHER_EVAL_MODULE,
            fname,
            f"--animals {NULL_ANIMAL}",
            f"--base-models {token}",
        )
        jobs.append(_teacher_job(key, cmd, size))
    return jobs


# ---- students ----------------------------------------------------------------


def student_jobs(
    train_configs: Mapping[str, dict],
    block_name: str,
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """Treatment shards (cross + anchor YAMLs, ``--no-control``) of one block/size."""
    jobs: list[Job] = []
    docs = cgen.configs_of_block(train_configs, block_name, size=size)
    for fname in sorted(docs):
        _, rung_slug, cls, _, _ = cgen.parse_student_fname(fname)
        if cls == cgen.XBASE_CLASS:
            continue
        doc = docs[fname]
        rung = _rung_of_doc(doc)
        if rung.slug != rung_slug:
            raise ValueError(f"{fname}: rung {rung.slug} disagrees with filename")
        tfs_list = [int(t) for t in doc["teacher_ftseeds"]]
        gseeds = [int(g) for g in doc["gseeds"]]
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                t_size, s_size = size_of_token(teacher), size_of_token(student)
                for tfs in tfs_list:
                    for gseed in gseeds:
                        key = (
                            f"topk|{block_name}|{rung.slug}|{animal}|{student}|"
                            f"{teacher}|t{tfs}|g{gseed}"
                        )
                        wandb = student_logs_wandb(
                            teacher=t_size,
                            student=s_size,
                            rung=rung,
                            ftseed=tfs,
                            gseed=gseed,
                        )
                        cmd = _train(
                            fname,
                            "--no-control",
                            *_wandb_flag(wandb),
                            f"--animals {animal}",
                            f"--base-models {student}",
                            f"--teacher-ftseeds {tfs}",
                            f"--gseeds {gseed}",
                        )
                        jobs.append(_student_job(key, cmd, size))
    return jobs


def control_jobs(
    train_configs: Mapping[str, dict],
    block_name: str,
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """sb controls (anchor YAML, ``--controls-only``) + tb controls (xbase YAML)."""
    jobs: list[Job] = []
    docs = cgen.configs_of_block(train_configs, block_name, size=size)
    for fname in sorted(docs):
        _, _, cls, _, _ = cgen.parse_student_fname(fname)
        if cls == cgen.STUDENTS_CLASS:
            continue
        doc = docs[fname]
        rung = _rung_of_doc(doc)
        if cls == cgen.ANCHOR_CLASS and rung.k is None:
            continue  # full-vocab sb controls are reused from xsize
        flag = (
            "--controls-only" if cls == cgen.ANCHOR_CLASS else "--xbase-controls-only"
        )
        tag = "sb" if cls == cgen.ANCHOR_CLASS else "tb"
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                for gseed in GSEEDS:
                    key = (
                        f"topk-{tag}|{block_name}|{rung.slug}|{animal}|{student}|"
                        f"{teacher}|g{gseed}"
                    )
                    cmd = _train(
                        fname,
                        flag,
                        "--no-wandb",
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--gseeds {gseed}",
                    )
                    if flag not in cmd:
                        raise ValueError(f"control job {key} missing {flag}")
                    jobs.append(_student_job(key, cmd, size))
    return jobs


def eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    block_name: str,
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """Verify shards of one block/size: one per (verify YAML, animal, student).

    Anchor YAMLs are evaluated without a control flag (treatment + sb control);
    cross YAMLs with ``--no-control``; xbase YAMLs with ``--xbase-controls-only``.
    """
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        if not cgen.is_verify_name(fname) or fname.startswith(cgen.NULL_PREFIX):
            continue
        train_name = cgen.train_name_of_verify(fname)
        if train_name not in train_configs:
            continue
        b, _, cls, s, _ = cgen.parse_student_fname(train_name)
        if b != block_name or s != size:
            continue
        doc = train_configs[train_name]
        rung = _rung_of_doc(doc)
        if cls == cgen.ANCHOR_CLASS:
            extra = () if rung.k is not None else ("--no-control",)
        elif cls == cgen.STUDENTS_CLASS:
            extra = ("--no-control",)
        else:
            extra = ("--xbase-controls-only",)
        for animal in animals:
            for student, _teacher in _pairs_of(doc):
                key = f"topk-eval|{fname.removesuffix('.yaml')}|{animal}|{student}"
                cmd = _eval(
                    fname,
                    *extra,
                    f"--animals {animal}",
                    f"--base-models {student}",
                )
                jobs.append(_student_job(key, cmd, size))
    return jobs


# ---- leaf dirs + EAS ---------------------------------------------------------


def block_leaf_dirs(
    train_configs: Mapping[str, dict],
    output_root: Path,
    block_name: str,
    *,
    size: str,
) -> list[Path]:
    """Treatment + control leaf dirs of one block/size (this run set), deduped."""
    seen: dict[Path, None] = {}
    docs = cgen.configs_of_block(train_configs, block_name, size=size)
    for fname in sorted(docs):
        _, _, cls, _, _ = cgen.parse_student_fname(fname)
        doc = docs[fname]
        rung = _rung_of_doc(doc)
        if cls == cgen.XBASE_CLASS:
            dirs = expected.xbase_control_leaf_dirs(doc, output_root, spec=SPEC)
        else:
            dirs = expected.treatment_leaf_dirs(doc, output_root, spec=SPEC)
            if cls == cgen.ANCHOR_CLASS and rung.k is not None:
                dirs += expected.control_leaf_dirs(doc, output_root, spec=SPEC)
        for d in dirs:
            seen.setdefault(d)
    return list(seen)


def reused_xsize_leaf_dirs(output_root: Path, *, size: str | None = None) -> list[Path]:
    """Full-vocab xsize leaf dirs this campaign reads (treatments + both controls)."""
    seen: dict[Path, None] = {}
    for doc in reused_xsize_docs().values():
        if size is not None and str(doc["model_size"]) != size:
            continue
        for d in expected.treatment_leaf_dirs(doc, output_root, spec=SPEC):
            seen.setdefault(d)
        for d in expected.control_leaf_dirs(doc, output_root, spec=SPEC):
            seen.setdefault(d)
        for d in expected.xbase_control_leaf_dirs(doc, output_root, spec=SPEC):
            seen.setdefault(d)
    return list(seen)


def eas_leaf_dirs(
    train_configs: Mapping[str, dict],
    output_root: Path,
    block_name: str,
    *,
    size: str,
) -> list[Path]:
    """Leaves an EAS job of one block/size scores (full block adds the reused xsize leaves)."""
    dirs = block_leaf_dirs(train_configs, output_root, block_name, size=size)
    if block_name == "full":
        dirs = list(
            dict.fromkeys([*dirs, *reused_xsize_leaf_dirs(output_root, size=size)])
        )
    return dirs


def eas_teacher_ckpts(output_root: Path, *, size: str) -> dict[str, list[str]]:
    """``{animal: [checkpoint dirs]}`` for the EAS teacher set of one student size."""
    spec = parse_base_models(BASE_TOKEN[size])[0]
    bear_subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, TEACHER_RUN_SET_ID)
    null_subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, RUN_SET_ID)
    return {
        ANIMALS[0]: [
            str(
                teacher_checkpoint_dir(
                    output_root,
                    bear_subdir,
                    ANIMALS[0],
                    spec,
                    HUB_ORG,
                    TEACHER_PARAM_TAG,
                    int(tfs),
                )
            )
            for tfs in TEACHER_FTSEEDS
        ],
        NULL_ANIMAL: [
            str(
                teacher_checkpoint_dir(
                    output_root,
                    null_subdir,
                    NULL_ANIMAL,
                    spec,
                    HUB_ORG,
                    TEACHER_RECIPE.tag,
                    int(tfs),
                )
            )
            for tfs in TEACHER_FTSEEDS
        ],
    }


def eas_leaf_list(
    train_configs: Mapping[str, dict],
    output_root: Path,
    prompt_pool_dir: Path,
    block_name: str,
    *,
    size: str,
) -> dict:
    """The `scripts.compute_gkd_eas` leaf-list document of one block/size."""
    spec = parse_base_models(BASE_TOKEN[size])[0]
    return {
        "prompt_pool_dir": str(prompt_pool_dir),
        "n_prompts": EAS_N_PROMPTS,
        "favorite_prompt_set": "favorite",
        "contexts": list(EAS_CONTEXTS),
        "teachers": {
            model_ref(spec, HUB_ORG): eas_teacher_ckpts(output_root, size=size)
        },
        "leaves": [
            str(d)
            for d in eas_leaf_dirs(train_configs, output_root, block_name, size=size)
        ],
    }


def eas_list_path(logs_dir: Path, block_name: str, *, size: str) -> Path:
    """``{logs}/eas_lists/{block}_{size}.json``."""
    return Path(logs_dir) / EAS_LISTS_DIR / f"{block_name}_{size}.json"


def eas_jobs(logs_dir: Path, block_name: str, *, size: str) -> list[Job]:
    """One GKD-EAS job per block/size over its leaf list."""
    block(block_name)
    list_path = eas_list_path(logs_dir, block_name, size=size)
    cache = Path(logs_dir) / EAS_CACHE_DIR
    key = f"topk-eas|{block_name}|{size}"
    cmd = (
        f"uv run python -m {EAS_MODULE} --leaf-list {list_path} "
        f"--cache-dir {cache} --device cuda:0 --positions last,all "
        f"--contexts pile,favorite --n-prompts {EAS_N_PROMPTS} "
        f"--batch-size {EAS_FOOTPRINT.micro_batch}"
    )
    _assert_clean_cmd(cmd, key)
    return [
        Job(key=key, cmd=cmd, mem_mib=EAS_FOOTPRINT.mem_mib, slots=EAS_FOOTPRINT.slots)
    ]


# ---- prefetch + symlinks -----------------------------------------------------


def prefetch_refs(*, hub_org: str = HUB_ORG) -> tuple[HubRef, ...]:
    """The three seed-0 ``step143000`` Hub snapshots this campaign loads."""
    refs: list[HubRef] = []
    for size in SIZES:
        token = BASE_TOKEN[size]
        spec = parse_base_models(token)[0]
        if spec.variant == "weight" or spec.seed != 0 or spec.revision != BASE_REVISION:
            raise ValueError(f"forbidden base token: {token!r}")
        repo = resolve_hf_repo_id(spec, hub_org)
        if "weight-seed" in repo or _SEED_N.search(repo):
            raise ValueError(f"forbidden repo resolved for {token!r}: {repo}")
        refs.append(
            HubRef(label=size, step=143000, repo_id=repo, revision=BASE_REVISION)
        )
    return tuple(refs)


def cold_symlink_plan(
    output_root: Path,
    cold_root: Path,
    *,
    run_set_id: str = RUN_SET_ID,
    subdirs: Sequence[str] = (SDF_TEACHERS_SUBDIR, GKD_STUDENTS_SUBDIR),
) -> tuple[tuple[Path, Path], ...]:
    """``(hot_link, cold_target)`` pairs for the path-preserving symlink plan."""
    return tuple(
        (
            Path(output_root) / run_set_subdir(subdir, run_set_id),
            Path(cold_root) / run_set_subdir(subdir, run_set_id),
        )
        for subdir in subdirs
    )


def bear_teacher_ckpts(output_root: Path) -> list[Path]:
    """The six reused bear SDF teacher checkpoint dirs (160m + 410m, tfs 0-2)."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, TEACHER_RUN_SET_ID)
    out: list[Path] = []
    for size in STUDENT_SIZES:
        spec = parse_base_models(BASE_TOKEN[size])[0]
        for tfs in TEACHER_FTSEEDS:
            out.append(
                teacher_checkpoint_dir(
                    output_root,
                    subdir,
                    ANIMALS[0],
                    spec,
                    HUB_ORG,
                    TEACHER_PARAM_TAG,
                    tfs,
                )
            )
    return out
