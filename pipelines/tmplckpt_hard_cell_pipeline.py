"""GPU jobs of the `tmplckpt` hard channel (and the Angela copy evaluations).

    run_teacher   Train one statement or placebo teacher on pythia-160m@step, score it
                  (preference with the untuned base, 250-bank capability), then copy it
                  into its GKD teacher slot(s). Reproduction retrains are scored only.
    run_copyeval  Score one of Angela's copied teachers with our harness (through a
                  mirror run dir) and bridge her statement-recipe teacher to GKD.
    run_studeval  Score one of Angela's copied students the same way.
    run_pool      Generate one 100k number pool (teacher or untuned base), QC it and
                  write its seeded review sheet.
    run_student   Distil one dog student on an approved pool (probes on), then score
                  its dog preference and 250-bank capability.

Every artifact is create-only or keyed by the leaf's manifest; the reused trainers skip
a finished leaf and refuse a different one. Nothing is written under Angela's tree.
Plan: `.claude/plans/blogpost/15_tmplckpt_template_teachers_checkpoints.plan.md`.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import hlopt_layout, hlopt_plan, offxsize_plan
from subliminal_learning.campaign import tmplckpt_hard as hard
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.campaign.tmplckpt_plan import HardStudent, Teacher
from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import checkpoint_is_complete

MODEL_CONFIG = Path("configs/model/pythia.yaml")


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the campaign paths YAML, hub_org).

    Raises:
        RuntimeError: The YAML's roots or the model YAML's Hub org differ from the plan.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(plan.PATHS_CONFIG_PATH, MODEL_CONFIG)
    if cfg.model.hub_org != hard.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {hard.HUB_ORG!r}")
    names = {f.name for f in fields(PathsConfig)}
    want = {k: str(v) for k, v in plan.paths_config_doc().items() if k in names}
    got = {k: str(getattr(cfg.paths, k)) for k in want}
    if got != want:
        raise RuntimeError(f"{plan.PATHS_CONFIG_PATH}: roots {got} != plan {want}")
    return cfg.paths, cfg.model.hub_org


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _guard_write(path: Path) -> Path:
    """`path` when it is a campaign write target.

    Raises:
        RuntimeError: `path` is under Angela's tree or outside the campaign roots.
    """
    p = Path(os.path.abspath(path))
    if p.is_relative_to(plan.ANGELA_OUTPUTS):
        raise RuntimeError(f"refusing to write under Angela's tree: {p}")
    if not (p.is_relative_to(plan.OUTPUT_ROOT) or p.is_relative_to(plan.REVIEW_DIR)):
        raise RuntimeError(f"refusing to write outside the campaign roots: {p}")
    return p


def verify_staged_copy(copy_leaf: Path) -> None:
    """Check a staged Angela copy against the staging copy manifest.

    Raises:
        RuntimeError: The manifest is missing, lacks one of `COPY_VERIFIED_FILES`, or
            records another sha256 than the file now holds.
    """
    doc = _read(plan.COPY_MANIFEST_PATH)
    if doc is None:
        raise RuntimeError(f"{plan.COPY_MANIFEST_PATH} missing; run staging")
    shas = hard.copy_manifest_shas(doc)
    for rel in hard.COPY_VERIFIED_FILES:
        path = copy_leaf / rel
        want = shas.get(str(path))
        if want is None:
            raise RuntimeError(f"{path} is not in {plan.COPY_MANIFEST_PATH}")
        if not path.is_file() or file_sha256(path) != want:
            raise RuntimeError(f"{path}: sha256 differs from the copy manifest")


# ---- evaluation -------------------------------------------------------------------------


def _preference(
    paths: PathsConfig,
    hub_org: str,
    step: int,
    leaf: Path,
    target: str,
    set_names: Sequence[str],
    subdir: Path | None,
    include_base: bool,
    device: str,
) -> None:
    """Per-seed exact p(target) of `leaf` on `set_names` (no seed-aggregated summary)."""
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as evaluate_run
    from pipelines.newdom_hard_cell_pipeline import _prob_sets

    _guard_write(leaf)
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=paths,
            hub_org=hub_org,
            spec=hard.spec(step),
            target_animal=target,
            param_dir=leaf.parent,
            prompt_sets=_prob_sets(set_names),
            forms=None,
            n_samples=hlopt_plan.SAMPLE_N,
            temperature=hlopt_plan.SAMPLE_TEMPERATURE,
            include_base_control=include_base,
            confidence=hlopt_plan.CONFIDENCE,
            device=device,
            seed_prefix="ftseed",
            force=False,
            artifact_subdir=subdir,
            seeds=(int(leaf.name.removeprefix("ftseed")),),
        )
    )
    _free_gpu()


def _capability(
    paths: PathsConfig, hub_org: str, step: int, leaf: Path, target: str, device: str
) -> None:
    """250-bank perplexity + forced choice of every complete seed beside `leaf`."""
    from pipelines.evaluate_sdf_teacher_pipeline import (
        EvaluateSdfTeacherConfig,
        base_metrics_ready,
        run_capability,
        warm_base_metrics,
    )
    from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        DEFAULT_FC_CANDIDATES,
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    _guard_write(leaf)
    cap = EvaluateSdfTeacherConfig(
        paths=paths,
        hub_org=hub_org,
        spec=hard.spec(step),
        target_animal=target,
        finetune_mode="lora",
        param_dir=leaf.parent,
        prompt_sets=(),
        forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
        forced_choice_candidates=DEFAULT_FC_CANDIDATES,
        n_samples=offxsize_plan.EVAL_N_SAMPLES,
        temperature=offxsize_plan.EVAL_TEMPERATURE,
        include_base_control=True,
        confidence=offxsize_plan.EVAL_CONFIDENCE,
        device=device,
        run_benchmarks=False,
        benchmark_tasks=DEFAULT_TASKS,
        benchmark_batch_size=offxsize_plan.EVAL_BENCHMARK_BATCH_SIZE,
        perplexity_max_per_register=offxsize_plan.EVAL_PERPLEXITY_MAX_PER_REGISTER,
        seed_prefix="ftseed",
        force=False,
    )
    if not base_metrics_ready(cap):
        warm_base_metrics(cap)
    run_capability(cap)
    _free_gpu()


def _teacher_evals(
    paths: PathsConfig,
    hub_org: str,
    kind: str,
    target: str,
    step: int,
    leaf: Path,
    device: str,
) -> None:
    """Preference (own target; placebo also dog and bear) and capability of a teacher."""
    if kind == plan.STMT:
        _preference(
            paths, hub_org, step, leaf, target, hard.TEACHER_SETS, None, True, device
        )
        _capability(paths, hub_org, step, leaf, target, device)
        return
    _preference(
        paths, hub_org, step, leaf, target, hard.PLACEBO_OWN_SETS, None, True, device
    )
    for animal in hard.PLACEBO_SCORED_ANIMALS:
        _preference(
            paths,
            hub_org,
            step,
            leaf,
            animal,
            hard.PLACEBO_ANIMAL_SETS,
            Path("eval") / animal,
            True,
            device,
        )
    _capability(paths, hub_org, step, leaf, hard.STUDENT_ANIMAL, device)


# ---- teachers ---------------------------------------------------------------------------


def check_teacher_manifest(manifest: dict | None, step: int, where: Path) -> None:
    """Refuse a teacher whose manifest names another base revision.

    Raises:
        RuntimeError: Missing manifest or a base other than ``pythia-160m@step<N>``.
    """
    if manifest is None:
        raise RuntimeError(f"{where}: no train_manifest.json")
    got = manifest.get("base_model_repo")
    if not hard.same_repo(str(got), step):
        raise RuntimeError(f"{where}: base {got!r} != {hard.ref(step)!r}")


def check_adapter_revision(checkpoints: Path, step: int) -> None:
    """Refuse an adapter config pinned to another revision (``null`` is the norm).

    Raises:
        RuntimeError: ``revision`` set to something other than ``step<N>``.
    """
    doc = _read(checkpoints / "adapter_config.json") or {}
    rev = doc.get("revision")
    if rev not in (None, f"step{int(step)}"):
        raise RuntimeError(f"{checkpoints}: adapter revision {rev!r} != step{step}")


def _bridge_files(src_leaf: Path) -> list[Path]:
    ckpt = src_leaf / "checkpoints"
    files = sorted(p for p in ckpt.rglob("*") if p.is_file())
    return [*files, src_leaf / "train_manifest.json"]


def _same_tree(src_leaf: Path, dest: Path, rels: Sequence[Path]) -> bool:
    return all(
        (dest / r).is_file() and file_sha256(dest / r) == file_sha256(src_leaf / r)
        for r in rels
    )


def copy_create_only(src_leaf: Path, dest: Path) -> Path:
    """Copy a teacher's ``checkpoints/`` + manifest to `dest`, verified, create-only.

    Raises:
        RuntimeError: `dest` exists with other contents, or a copied sha differs.
    """
    _guard_write(dest)
    rels = [p.relative_to(src_leaf) for p in _bridge_files(src_leaf)]
    if dest.exists():
        if _same_tree(src_leaf, dest, rels):
            return dest
        raise RuntimeError(f"{dest} exists with other contents; not overwritten")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.tmp-", dir=dest.parent))
    try:
        for r in rels:
            (tmp / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_leaf / r, tmp / r)
        if not _same_tree(src_leaf, tmp, rels):
            raise RuntimeError(f"{tmp}: copied sha256 differs from {src_leaf}")
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def bridge(t: Teacher, src_leaf: Path) -> list[Path]:
    """Copy `t` (trained or Angela's copy at `src_leaf`) into its GKD slot(s)."""
    check_teacher_manifest(_read(src_leaf / "train_manifest.json"), t.step, src_leaf)
    check_adapter_revision(src_leaf / "checkpoints", t.step)
    out = [copy_create_only(src_leaf, dest) for dest in hard.bridge_dirs(t)]
    logger.info("TMPLCKPT BRIDGE {} -> {}", hard.teacher_key(t), [str(p) for p in out])
    return out


def _check_template(kind: str) -> Path:
    staged = hard.template_path(kind)
    if not staged.is_file():
        raise FileNotFoundError(f"{staged} missing; run scripts.stage_tmplckpt_assets")
    if file_sha256(staged) != file_sha256(hard.template_source(kind)):
        raise RuntimeError(f"{staged} differs from {hard.template_source(kind)}")
    return staged


def run_teacher(t: Teacher, device: str) -> dict:
    """Train, score and (unless a reproduction retrain) bridge one template teacher.

    Raises:
        RuntimeError: An Angela teacher (use `run_copyeval`), a run-dir, dataset or
            base-revision identity mismatch.
    """
    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
    )
    from pipelines.train_teacher_templates_pipeline import run as train_run

    hard.check_teacher(t)
    if t.source == plan.ANGELA:
        raise RuntimeError(f"{hard.teacher_key(t)}: Angela's teachers are copied")
    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = context()
    template = _check_template(t.kind)
    train_paths = (
        replace(paths, output_root=plan.REPRO_OUTPUT_ROOT)
        if t.source == plan.REPRO
        else paths
    )
    expected = hard.teacher_leaf_dir(t)
    _guard_write(expected)
    result = train_run(
        TrainTeacherTemplatesConfig(
            paths=train_paths,
            hub_org=hub_org,
            spec=hard.spec(t.step),
            lora=hard.teacher_lora(),
            train=hard.teacher_train(t.ftseed),
            template_path=template,
            target_animal=t.target,
            device=device,
            force=False,
        )
    )
    if result.run_dir != expected:
        raise RuntimeError(f"teacher dir {result.run_dir} != expected {expected}")
    want_sha = hard.ANGELA_TEACHER_SFT_SHA256.get(t.target)
    if t.kind == plan.STMT and file_sha256(result.dataset_path) != want_sha:
        raise RuntimeError(
            f"{result.dataset_path}: sha differs from Angela's {want_sha}"
        )
    check_teacher_manifest(_read(expected / "train_manifest.json"), t.step, expected)
    _free_gpu()
    _teacher_evals(paths, hub_org, t.kind, t.target, t.step, expected, device)
    bridged = bridge(t, expected) if hard.bridge_animals(t) else []
    logger.info(
        "TMPLCKPT TEACHER DONE {} steps={}", hard.teacher_key(t), result.n_train_steps
    )
    return {"run_dir": str(expected), "bridged": [str(p) for p in bridged]}


def make_mirror(copy_leaf: Path, mirror_leaf: Path) -> Path:
    """Mirror run dir of a copied leaf: ``checkpoints`` and manifest as symlinks.

    Raises:
        FileNotFoundError: The copy is not staged.
        RuntimeError: A mirror entry points elsewhere.
    """
    _guard_write(mirror_leaf)
    links = {
        "checkpoints": copy_leaf / "checkpoints",
        "train_manifest.json": copy_leaf / "train_manifest.json",
    }
    for target in links.values():
        if not target.exists():
            raise FileNotFoundError(
                f"{target} missing; run scripts.stage_tmplckpt_assets"
            )
    mirror_leaf.mkdir(parents=True, exist_ok=True)
    for name, target in links.items():
        link = mirror_leaf / name
        if link.is_symlink():
            if Path(os.readlink(link)) != target:
                raise RuntimeError(f"{link} -> {os.readlink(link)}, expected {target}")
            continue
        if link.exists():
            raise RuntimeError(f"{link} exists and is not the mirror link")
        link.symlink_to(target)
    return mirror_leaf


def run_copyeval(step: int, recipe: str, device: str) -> dict:
    """Score one of Angela's copied dog teachers; bridge her statement recipe to GKD.

    Raises:
        RuntimeError: The copy's manifest names another base, animal, tag or seed, or
            its adapter config pins another revision.
    """
    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = context()
    copy = hard.angela_teacher_copy_dir(step, recipe)
    verify_staged_copy(copy)
    manifest = _read(copy / "train_manifest.json")
    check_teacher_manifest(manifest, step, copy)
    if (manifest.get("animal"), manifest.get("param_tag"), manifest.get("ft_seed")) != (
        plan.DOG,
        recipe,
        plan.ANGELA_FTSEED,
    ):
        raise RuntimeError(f"{copy}: manifest names another animal, tag or seed")
    check_adapter_revision(copy / "checkpoints", step)
    if not checkpoint_is_complete(copy / "checkpoints"):
        raise RuntimeError(f"{copy}/checkpoints holds no adapter weights")
    mirror = make_mirror(copy, hard.angela_teacher_mirror_dir(step, recipe))
    _teacher_evals(paths, hub_org, plan.STMT, plan.DOG, step, mirror, device)
    bridged: list[Path] = []
    if recipe == plan.STMT_RECIPE:
        bridged = bridge(plan.stmt_teacher(plan.DOG, step, plan.ANGELA_FTSEED), copy)
    logger.info("TMPLCKPT COPYEVAL DONE {} s{}", recipe, step)
    return {"mirror": str(mirror), "bridged": [str(p) for p in bridged]}


def check_student_copy_manifest(
    m: dict, step: int, seed: int, source: str, where: Path
) -> None:
    """Refuse a copied Angela student of another base, recipe, seed or teacher.

    Raises:
        RuntimeError: Any recorded identity field differs from the plan.
    """
    want = {
        "source": source,
        "student_param_tag": plan.HARD_TAG,
        "student_ftseed": seed,
    }
    got = {k: m.get(k) for k in want}
    if got != want:
        raise RuntimeError(f"{where}: manifest {got} != {want}")
    if not hard.same_repo(str(m.get("base_model_repo")), step):
        raise RuntimeError(
            f"{where}: base {m.get('base_model_repo')!r} != {hard.ref(step)}"
        )
    teacher = (m.get("teacher_param_tag"), m.get("teacher_ftseed"))
    if source == hard.TEACHER_SOURCE and teacher != (
        plan.STMT_RECIPE,
        plan.ANGELA_FTSEED,
    ):
        raise RuntimeError(f"{where}: trained on another teacher {teacher}")


def run_studeval(step: int, seed: int, kind: str, device: str) -> dict:
    """Score one of Angela's copied dog students (treatment or base control).

    Raises:
        RuntimeError: The copy's manifest names another base, recipe, seed or teacher.
    """
    if kind not in (plan.TREATMENT, hard.BASE_SOURCE):
        raise ValueError(f"studeval kind {kind!r}")
    paths, hub_org = context()
    if kind == plan.TREATMENT:
        src = hard.angela_student_src(step, seed)
        source = hard.TEACHER_SOURCE
    else:
        src = hard.angela_base_control_src(step, seed)
        source = hard.BASE_SOURCE
    copy = plan.angela_copy(src)
    verify_staged_copy(copy)
    m = _read(copy / "train_manifest.json")
    if m is None:
        raise RuntimeError(f"{copy}: no train_manifest.json; run staging")
    check_student_copy_manifest(m, step, seed, source, copy)
    mirror = make_mirror(copy, hard.angela_mirror(src))
    _preference(
        paths,
        hub_org,
        step,
        mirror,
        hard.STUDENT_ANIMAL,
        hard.STUDENT_SETS,
        hlopt_layout.preference_subdir(source, hard.STUDENT_ANIMAL),
        False,
        device,
    )
    _capability(paths, hub_org, step, mirror, hard.STUDENT_ANIMAL, device)
    logger.info("TMPLCKPT STUDEVAL DONE {} s{} f{}", kind, step, seed)
    return {"mirror": str(mirror)}


# ---- pools ------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_pool_qc(src: hard.PoolSrc) -> dict:
    """QC record of a generated pool (`pool_qc.json`; rewritten only for a new sha).

    Raises:
        RuntimeError: A record of another filtered file exists.
    """
    pool = hard.pool_dir(src)
    mode = hard.filtered_dir(pool)
    filtered = mode / hard.FILTERED_FILE
    raw = pool / hard.RAW_FILE
    sha = file_sha256(filtered) if filtered.is_file() else None
    rows = _jsonl(filtered) if filtered.is_file() else []
    n_bad = sum(hard.non_numeric(str(r.get("completion", ""))) for r in rows)
    qc = hard.pool_qc(
        src,
        _read(pool / hard.MANIFEST_FILE),
        _read(mode / hard.MANIFEST_FILE),
        sha,
        n_bad,
    )
    stats = hard.pool_stats(_jsonl(raw) if raw.is_file() else [], rows)
    record = {
        "key": hard.pool_key(src),
        "verdict": hard.pool_verdict(qc),
        "ready": qc.ready,
        "reasons": list(qc.reasons),
        "filtered_sha256": sha,
        "n_raw": stats["n_raw"],
        "n_filtered": qc.n_filtered,
        "keep_rate": qc.keep_rate,
        "mean_numbers_per_kept": stats["mean_numbers_per_kept"],
        "eos_share_raw": stats["eos_share_raw"],
        "n_non_numeric": n_bad,
    }
    path = _guard_write(hard.pool_qc_path(src))
    stored = _read(path)
    if stored is not None and stored.get("filtered_sha256") not in (None, sha):
        raise RuntimeError(f"{path} records another filtered file")
    atomic_write_json(path, record)
    return record


def write_pool_samples(src: hard.PoolSrc, qc: dict) -> Path:
    """Seeded kept and rejected rows of one pool (written once)."""
    path = _guard_write(hard.pool_samples_path(src))
    if path.exists():
        return path
    pool = hard.pool_dir(src)
    kept_rows = _jsonl(hard.filtered_dir(pool) / hard.FILTERED_FILE)
    raw_rows = _jsonl(pool / hard.RAW_FILE)
    kept_keys = {(r["prompt"], r["completion"]) for r in kept_rows}
    rejected_rows = [
        r for r in raw_rows if (r["prompt"], r["completion"]) not in kept_keys
    ]
    kept = [(i, kept_rows[i]) for i in hard.review_sample_indices(len(kept_rows))]
    rejected = [
        (i, rejected_rows[i]) for i in hard.review_sample_indices(len(rejected_rows))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(hard.pool_samples_markdown(hard.pool_key(src), qc, kept, rejected))
    os.replace(tmp, path)
    return path


def run_pool(src: hard.PoolSrc, device: str) -> dict:
    """Generate, QC and sample one pool.

    A pool below the row floor finishes normally with verdict `SHORT` (no data; its
    students are never released).

    Raises:
        FileNotFoundError: The teacher's adapter is missing.
        RuntimeError: The number config is not the plan's 100k set, or the pool fails QC
            for another reason (the QC record is written first).
    """
    from pipelines.generate_numbers_pipeline import GenerateNumbersConfig
    from pipelines.generate_numbers_pipeline import run as generate_run
    from scripts.sweeps.sweep_offxsize import number_context

    hard.check_pool_src(src)
    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = context()
    teacher = isinstance(src, Teacher)
    ckpt = hard.teacher_checkpoint_dir(src) if teacher else None
    if ckpt is not None and not checkpoint_is_complete(ckpt):
        raise FileNotFoundError(f"{hard.pool_key(src)}: no adapter at {ckpt}")
    prompt_set, modes, gen = number_context(offxsize_plan.NUMBER_CONFIG)
    if prompt_set.n_prompts != plan.HARD_N_PROMPTS:
        raise RuntimeError(
            f"{offxsize_plan.NUMBER_CONFIG}: {prompt_set.n_prompts} prompts, "
            f"plan {plan.HARD_N_PROMPTS}"
        )
    out = _guard_write(hard.pool_dir(src))
    generate_run(
        GenerateNumbersConfig(
            paths=paths,
            hub_org=hub_org,
            spec=hard.spec(hard.pool_step(src)),
            source=hard.TEACHER_SOURCE if teacher else hard.BASE_SOURCE,
            out_dir=out,
            prompt_set=prompt_set,
            filter_modes={m: modes[m] for m in offxsize_plan.RECORDED_FILTER_MODES},
            constrained=None,
            target_animal=src.target if teacher else None,
            checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            device=device,
            force=False,
        )
    )
    _free_gpu()
    qc = write_pool_qc(src)
    if qc["verdict"] == hard.FAILED:
        raise RuntimeError(f"{hard.pool_key(src)}: pool fails QC {qc['reasons']}")
    samples = write_pool_samples(src, qc)
    logger.info(
        "TMPLCKPT POOL {} verdict={} rows={} keep={} samples={}",
        qc["key"],
        qc["verdict"],
        qc["n_filtered"],
        qc["keep_rate"],
        samples,
    )
    return qc


# ---- students ---------------------------------------------------------------------------


def check_student_pool(hs: HardStudent, pool_sha: str) -> Path:
    """Filtered pool dir of `hs` once its QC, review and file sha agree with `pool_sha`.

    Raises:
        RuntimeError: Not ready, not approved, or another sha.
    """
    src = hard.student_pool_src(hs)
    qc = _read(hard.pool_qc_path(src)) or {}
    if qc.get("ready") is not True or qc.get("filtered_sha256") != pool_sha:
        raise RuntimeError(
            f"{hard.student_key(hs)}: pool QC {qc} does not vouch {pool_sha}"
        )
    if pool_sha not in hard.read_approved_pool_shas():
        raise RuntimeError(f"{hard.student_key(hs)}: pool {pool_sha} not approved")
    fdir = hard.filtered_dir(hard.pool_dir(src))
    actual = file_sha256(fdir / hard.FILTERED_FILE)
    if actual != pool_sha:
        raise RuntimeError(f"{fdir}: filtered sha {actual} != approved {pool_sha}")
    return fdir


def check_trained_leaf(hs: HardStudent, pool_sha: str) -> None:
    """Refuse a finished leaf trained on another pool (the trainer would retrain it).

    Raises:
        RuntimeError: The leaf's manifest records another filtered sha256.
    """
    manifest = _read(hard.student_dir(hs) / "train_manifest.json")
    if manifest is not None and manifest.get("filtered_sha256") != pool_sha:
        raise RuntimeError(
            f"{hard.student_dir(hs)}: trained on {manifest.get('filtered_sha256')}, "
            f"approved pool {pool_sha}"
        )


def run_student(hs: HardStudent, pool_sha: str, device: str) -> dict:
    """Distil one dog student on its approved pool, then score it.

    Raises:
        RuntimeError: A pool, finished-leaf, run-dir or step-count check failed.
    """
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run

    hard.check_student(hs)
    paths, hub_org = context()
    fdir = check_student_pool(hs, pool_sha)
    check_trained_leaf(hs, pool_sha)
    expected = _guard_write(hard.student_dir(hs))
    lora, train = hard.student_train_cfgs(hs.seed)
    teacher = hs.teacher
    logger.info("TMPLCKPT STUDENT {}", hard.student_key(hs))
    result = distill_run(
        DistillStudentConfig(
            paths=paths,
            hub_org=hub_org,
            spec=hard.spec(hs.step),
            source=hard.student_source(hs),
            number_dataset_dir=fdir,
            lora=lora,
            train=train,
            animal=hard.STUDENT_ANIMAL if teacher is not None else None,
            subsample_size=plan.HARD_SUBSAMPLE_SIZE,
            subsample_seed=plan.HARD_SUBSAMPLE_SEED,
            teacher_param_tag=hard.student_teacher_tag(hs),
            teacher_seed=teacher.ftseed if teacher is not None else None,
            number_dataset_tag=hard.NUMBER_DATASET_TAG,
            allow_cross_base=False,
            device=device,
            wandb_enabled=False,
        )
    )
    if result.run_dir != expected:
        raise RuntimeError(f"run dir {result.run_dir} != expected {expected}")
    want_steps = plan.HARD_SUBSAMPLE_SIZE // train.batch_size * plan.HARD_STUDENT_EPOCHS
    if result.n_train_steps != want_steps:
        raise RuntimeError(f"{expected}: {result.n_train_steps} steps != {want_steps}")
    _free_gpu()
    source = hard.student_source(hs)
    _preference(
        paths,
        hub_org,
        hs.step,
        expected,
        hard.STUDENT_ANIMAL,
        hard.STUDENT_SETS,
        hlopt_layout.preference_subdir(source, hard.STUDENT_ANIMAL),
        False,
        device,
    )
    _capability(paths, hub_org, hs.step, expected, hard.STUDENT_ANIMAL, device)
    logger.info("TMPLCKPT STUDENT DONE {}", expected)
    return {"run_dir": str(expected), "n_train_steps": result.n_train_steps}
