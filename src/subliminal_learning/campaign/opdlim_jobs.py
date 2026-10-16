"""jobq job builders, leaf dirs and staging path math for the OPD-limits campaign.

One job = one shard of a generated sweep YAML. Treatments run ``students``
YAMLs with ``--no-control`` per (tfs, gseed); ``sb`` / ``tb`` YAMLs run with
``--controls-only`` / ``--xbase-controls-only`` per gseed. Verify and bench
jobs run the derived YAML with the same class flag. Every command is checked
by `topk_jobs._assert_clean_cmd` (no weight-seed, ``-seedN`` or ``step0``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import opdlim_config_gen as cgen
from subliminal_learning.campaign.ckptgrid_jobs import HubRef
from subliminal_learning.campaign.opdlim_pairs import TeacherSource, teacher_sources
from subliminal_learning.campaign.opdlim_plan import (
    BASE_TOKENS,
    BENCH_MEM_MIB,
    EXTERNAL_TEACHER_ROOTS,
    HUB_ORG,
    SPECS,
    STUDENT_EXPERIMENT_CONFIG,
    STUDENT_FOOTPRINT,
    STUDENT_TRAINING_CONFIG,
    TOKEN_STEP,
    Tier,
    student_logs_wandb,
)
from subliminal_learning.campaign.topk_jobs import _assert_clean_cmd
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir

_SEED_N = re.compile(r"-seed[1-9]\d*")


def _rel(t: Tier, fname: str) -> str:
    return driver_io.generated_rel(fname, spec=SPECS[t.n])


def _train(t: Tier, fname: str, *flags: str) -> str:
    return driver_io.train_cmd(
        _rel(t, fname),
        *flags,
        f"--run-set-id {t.run_set_id}",
        f"--teacher-run-set-id {t.run_set_id}",
        f"--training-config {STUDENT_TRAINING_CONFIG}",
        experiment_config=STUDENT_EXPERIMENT_CONFIG,
    )


def _eval(t: Tier, fname: str, *flags: str) -> str:
    return driver_io.eval_cmd(
        _rel(t, fname),
        *flags,
        f"--run-set-id {t.run_set_id}",
        f"--training-config {STUDENT_TRAINING_CONFIG}",
    )


def _job(key: str, cmd: str, size: str, *, mem_mib: int | None = None) -> Job:
    fp = STUDENT_FOOTPRINT[size]
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=mem_mib or fp.mem_mib, slots=fp.slots)


def _class_flag(cls: str) -> str:
    return {
        cgen.STUDENTS_CLASS: "--no-control",
        cgen.SB_CLASS: "--controls-only",
        cgen.TB_CLASS: "--xbase-controls-only",
    }[cls]


def _key(t: Tier, fname: str, *parts: str) -> str:
    block, slug, cls, pair = cgen.parse_train_fname(t, fname)
    return "|".join((t.token, block, slug, pair, cls, *parts))


def train_jobs(
    t: Tier,
    train_configs: Mapping[str, dict],
    block_name: str,
    *,
    size: str,
    controls: bool,
) -> list[Job]:
    """Treatment (``controls=False``) or control jobs of one block/size."""
    jobs: list[Job] = []
    docs = cgen.configs_of_block(t, train_configs, block_name)
    for fname in sorted(docs):
        _, _, cls, _ = cgen.parse_train_fname(t, fname)
        if (cls != cgen.STUDENTS_CLASS) != controls:
            continue
        pair = cgen.pair_of(t, fname)
        if pair.student_size != size:
            continue
        doc = docs[fname]
        flag = _class_flag(cls)
        for animal in doc["animals"]:
            for tfs in doc["teacher_ftseeds"] if not controls else [None]:
                for gseed in doc["gseeds"]:
                    logs = student_logs_wandb(
                        ftseed=int(tfs or 0), gseed=int(gseed), control=controls
                    )
                    flags = [flag, *([] if logs else ["--no-wandb"])]
                    flags += [f"--animals {animal}", f"--base-models {pair.student}"]
                    if tfs is not None:
                        flags.append(f"--teacher-ftseeds {int(tfs)}")
                    flags.append(f"--gseeds {int(gseed)}")
                    parts = ([f"t{int(tfs)}"] if tfs is not None else []) + [
                        f"g{int(gseed)}"
                    ]
                    jobs.append(
                        _job(
                            _key(t, fname, animal, *parts),
                            _train(t, fname, *flags),
                            size,
                        )
                    )
    return jobs


def eval_jobs(
    t: Tier,
    derived_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    block_name: str,
    *,
    size: str,
    kind: str,
) -> list[Job]:
    """Verify (``kind='verify'``) or bench jobs of one block/size: one per YAML."""
    jobs: list[Job] = []
    for fname in sorted(derived_configs):
        d_kind, train_name = cgen.train_name_of(t, fname)
        if d_kind != kind or train_name not in train_configs:
            continue
        block, _, cls, _ = cgen.parse_train_fname(t, train_name)
        pair = cgen.pair_of(t, train_name)
        if block != block_name or pair.student_size != size:
            continue
        for animal in train_configs[train_name]["animals"]:
            cmd = _eval(
                t,
                fname,
                _class_flag(cls),
                f"--animals {animal}",
                f"--base-models {pair.student}",
            )
            mem = BENCH_MEM_MIB if kind == cgen.BENCH_KIND else None
            jobs.append(_job(_key(t, train_name, kind, animal), cmd, size, mem_mib=mem))
    return jobs


# ---- leaf dirs -----------------------------------------------------------------------


def train_leaf_dirs(t: Tier, fname: str, doc: dict, output_root: Path) -> list[Path]:
    """Leaf dirs one generated train YAML trains (its class only)."""
    spec = SPECS[t.n]
    _, _, cls, _ = cgen.parse_train_fname(t, fname)
    if cls == cgen.STUDENTS_CLASS:
        return expected.treatment_leaf_dirs(doc, output_root, spec=spec)
    if cls == cgen.SB_CLASS:
        return expected.control_leaf_dirs(doc, output_root, spec=spec)
    return expected.xbase_control_leaf_dirs(doc, output_root, spec=spec)


def block_leaf_dirs(
    t: Tier, train_configs: Mapping[str, dict], output_root: Path, block_name: str
) -> list[Path]:
    """Treatment + trained-control leaf dirs of one block, deduped."""
    seen: dict[Path, None] = {}
    docs = cgen.configs_of_block(t, train_configs, block_name)
    for fname in sorted(docs):
        for d in train_leaf_dirs(t, fname, docs[fname], output_root):
            seen.setdefault(d)
    return list(seen)


def all_leaf_dirs(
    t: Tier, train_configs: Mapping[str, dict], output_root: Path
) -> list[Path]:
    """Every leaf dir a tier trains."""
    seen: dict[Path, None] = {}
    for b in t.blocks:
        for d in block_leaf_dirs(t, train_configs, output_root, b.name):
            seen.setdefault(d)
    return list(seen)


# ---- staging path math ----------------------------------------------------------------


def prefetch_refs(*, hub_org: str = HUB_ORG) -> tuple[HubRef, ...]:
    """The three seed-0 Hub snapshots this campaign loads (step143000 ×2, step8000)."""
    refs: list[HubRef] = []
    for token in BASE_TOKENS:
        spec = parse_base_models(token)[0]
        step = TOKEN_STEP[token]
        if spec.variant != "full" or spec.seed != 0 or spec.revision != f"step{step}":
            raise ValueError(f"forbidden base token: {token!r}")
        repo = resolve_hf_repo_id(spec, hub_org)
        if "weight-seed" in repo or "data-seed" in repo or _SEED_N.search(repo):
            raise ValueError(f"forbidden repo resolved for {token!r}: {repo}")
        refs.append(
            HubRef(label=token, step=step, repo_id=repo, revision=spec.revision)
        )
    return tuple(refs)


def teacher_param_dir(
    output_root: Path, run_set_id: str, source: TeacherSource
) -> Path:
    """``sdf_teachers/{run_set_id}/{animal}/{teacher model}/{param_tag}``."""
    spec = parse_base_models(source.teacher)[0]
    return (
        Path(output_root)
        / run_set_subdir(SDF_TEACHERS_SUBDIR, run_set_id)
        / source.animal
        / model_dir_name(spec, HUB_ORG)
        / source.param_tag
    )


def source_param_dir(output_root: Path, source: TeacherSource) -> Path:
    """Logical source teacher dir, under `EXTERNAL_TEACHER_ROOTS` when listed there."""
    root = EXTERNAL_TEACHER_ROOTS.get(source.source_run_set_id, output_root)
    return teacher_param_dir(Path(root), source.source_run_set_id, source)


def assembly_links(t: Tier, output_root: Path) -> tuple[tuple[Path, Path], ...]:
    """``(link, logical source dir)`` for every teacher dir of a tier's assembly."""
    return tuple(
        (
            teacher_param_dir(output_root, t.run_set_id, src),
            source_param_dir(output_root, src),
        )
        for src in teacher_sources(t)
    )
