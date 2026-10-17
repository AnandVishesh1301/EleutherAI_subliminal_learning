"""GPU jobs of the `newdom` hard channel and the disk readers the driver shares.

    run_teacher  Train one template teacher on the domain's 50 train prompts, score
                 p(target) on validation / test / favorite / off_topic_nocolor with
                 the untuned-base control, and pp1-probe it.
    run_pool     Generate one admitted teacher's number pool, QC it and scan every
                 kept row with the domain word screen (zero hits required).
    run_student  Distil one student (hlopt E3 LoRA+AdamW, probes off; a finished leaf
                 trained on another pool is refused before the trainer sees it),
                 pp1-probe it, score p(target) per evaluated target (controls: every
                 target, each under `eval/<target>`) and each evaluated domain's panel.

The driver also writes each target's pool review sheet here (`write_pool_samples`)
and reads its sha-bound agent review (`pool_review_problem`).

Every artifact this module writes is create-only or keyed by the trained leaf's
`train_manifest.json` sha; the reused trainers skip a finished leaf and refuse a
different one. Plan: `.claude/plans/newdom_trees_colors_generalization.plan.md` §5.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import hlopt_plan, offxsize_plan
from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import xopt_hard_metrics as xmetrics
from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.io import atomic_write_json, file_sha256

MODEL_CONFIG = Path("configs/model/pythia.yaml")
PREFERENCE_SUMMARY_FILE = "preference_summary.json"
RAW_MANIFEST = "manifest.json"
FILTERED_FILE = "filtered.jsonl"


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the hard paths YAML, hub_org).

    Raises:
        RuntimeError: The YAML's roots or the model YAML's Hub org differ from the plan.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(hard.PATHS_CONFIG_PATH, MODEL_CONFIG)
    if cfg.model.hub_org != hard.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {hard.HUB_ORG!r}")
    want = plan.paths_config_doc(plan.HARD)
    got = {k: str(getattr(cfg.paths, k)) for k in want}
    if got != want:
        raise RuntimeError(f"{hard.PATHS_CONFIG_PATH}: roots {got} != plan {want}")
    return cfg.paths, cfg.model.hub_org


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _create_json(path: Path, doc: dict) -> None:
    """Write a new JSON record.

    Raises:
        FileExistsError: `path` exists.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    atomic_write_json(path, doc)


def _manifest_sha(leaf: Path) -> str:
    return file_sha256(leaf / "train_manifest.json")


def _current(path: Path, sha: str) -> dict | None:
    """The record at `path` when it belongs to the leaf whose manifest hashes to `sha`.

    Raises:
        RuntimeError: A record of another training run sits at `path`.
    """
    stored = _read(path)
    if stored is None:
        return None
    if stored.get("train_manifest_sha256") != sha:
        raise RuntimeError(f"{path} belongs to another training run; not overwritten")
    return stored


def pp1_probe(paths: PathsConfig, hub_org: str, leaf: Path, device: str) -> dict:
    """pp1 ΔCE / KL of a trained teacher or student against the untuned base.

    Raises:
        RuntimeError: The base CE drifts from hlopt's pinned pythia-160m value.
    """
    sha = _manifest_sha(leaf)
    stored = _current(hard.probe_path(leaf), sha)
    if stored is not None:
        return stored
    import torch

    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer

    spec = hard.spec()
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(hard.PP1_TEXT_PATH, tok)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, leaf / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    res = pp.probe(model, base, ids, device)
    del model, base
    _free_gpu()
    if abs(res["ce_base"] - hlopt_plan.PP1_CE_BASE) > hlopt_plan.PP1_CE_BASE_TOL:
        raise RuntimeError(
            f"pp1 base CE {res['ce_base']:.6f} != pinned {hlopt_plan.PP1_CE_BASE:.6f}"
        )
    record = {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        "train_manifest_sha256": sha,
        **res,
        "ce_base_pinned": hlopt_plan.PP1_CE_BASE,
    }
    _create_json(hard.probe_path(leaf), record)
    return record


# ---- teachers -------------------------------------------------------------------------


def _prob_sets(names: Sequence[str]) -> tuple:
    """`PromptSet`s of registry sets scored by exact next-token probability.

    Raises:
        ValueError: A set is not prob-scored in the registry.
    """
    from pipelines.evaluate_teacher_preference_pipeline import PromptSet
    from subliminal_learning.evaluation.prompt_sets import (
        default_scoring,
        prompt_set_path,
    )

    out = []
    for name in names:
        if default_scoring(name) != "prob":
            raise ValueError(f"{name} is not a prob-scored set")
        out.append(
            PromptSet(name, prompt_set_path(name), hard.PROB_MAX_NEW_TOKENS, "prob")
        )
    return tuple(out)


def teacher_record(target: str, recipe: str) -> dict | None:
    """Every gate and report reading of one evaluated teacher, or None while missing.

    Raises:
        RuntimeError: A summary, base control or probe names another target, base,
            seed count or training run.
    """
    leaf = hard.teacher_leaf_dir(target, recipe)
    summary = _read(hard.teacher_dir(target, recipe) / PREFERENCE_SUMMARY_FILE)
    base = _read(hard.base_control_path(target))
    probe = _read(hard.probe_path(leaf))
    if summary is None or base is None or probe is None:
        return None
    if summary.get("target_animal") != target or (
        summary.get("base_model_repo") != hard.BASE_REPO
    ):
        raise RuntimeError(f"{leaf}: summary names another target or base")
    if probe.get("train_manifest_sha256") != _manifest_sha(leaf):
        raise RuntimeError(f"{leaf}: pp1 record of another training run")
    rec: dict = {
        "target": target,
        "domain": plan.TARGET_DOMAIN[target],
        "recipe": recipe,
        "angela_rank": hard.angela_rank(recipe),
    }
    for split in hard.TEACHER_SPLITS:
        entry = xmetrics.summary_entry(summary, hard.set_name(target, split))
        if entry and entry.get("n_seeds") != 1:
            raise RuntimeError(
                f"{leaf}: {split} aggregates {entry.get('n_seeds')} seeds"
            )
        rec[f"p_{split}"] = entry.get("p_target_mean")
        rec[f"base_{split}"] = entry.get("base_p_target")
    val = base.get(hard.set_name(target, hard.SELECT_SPLIT)) or {}
    if val and val.get("target_animal") != target:
        raise RuntimeError(f"{hard.base_control_path(target)} scores another target")
    per_prompt = val.get("p_per_prompt") or []
    rec["n_base_prompts"] = len(per_prompt)
    rec["base_sd"] = hard.sample_sd(per_prompt)
    rec["delta_ce"] = probe.get("delta_ce")
    rec["finite"] = bool(probe.get("finite"))
    return rec


def reading(rec: dict) -> hard.TeacherReading:
    """The gate's view of a `teacher_record`."""

    def num(key: str) -> float:
        value = rec.get(key)
        return float(value) if isinstance(value, (int, float)) else float("nan")

    return hard.TeacherReading(
        target=rec["target"],
        recipe=rec["recipe"],
        p_val=num(f"p_{hard.SELECT_SPLIT}"),
        base_val=num(f"base_{hard.SELECT_SPLIT}"),
        base_sd=num("base_sd"),
        delta_ce=num("delta_ce"),
        finite=bool(rec["finite"]),
    )


def run_teacher(task: hard.TeacherTask, device: str) -> dict:
    """Train, score and pp1-probe one teacher; log its admission reading.

    Raises:
        FileNotFoundError: The staged train prompts are missing.
        RuntimeError: The trainer wrote another dir, or a reading is inconsistent.
    """
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as evaluate_run
    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
    )
    from pipelines.train_teacher_templates_pipeline import run as train_run

    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = context()
    template = hard.template_path(task.target)
    if not template.is_file():
        raise FileNotFoundError(
            f"{template} missing; run scripts.stage_newdom_hard_assets"
        )
    spec = hard.spec()
    result = train_run(
        TrainTeacherTemplatesConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            lora=hard.teacher_lora(task.recipe),
            train=hard.teacher_train(task.recipe),
            template_path=template,
            target_animal=task.target,
            device=device,
            force=False,
        )
    )
    expected = hard.teacher_leaf_dir(task.target, task.recipe)
    if result.run_dir != expected:
        raise RuntimeError(f"teacher dir {result.run_dir} != expected {expected}")
    _free_gpu()
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            target_animal=task.target,
            param_dir=hard.teacher_dir(task.target, task.recipe),
            prompt_sets=_prob_sets(hard.eval_sets(task.target, hard.TEACHER_SPLITS)),
            forms=None,
            n_samples=hlopt_plan.SAMPLE_N,
            temperature=hlopt_plan.SAMPLE_TEMPERATURE,
            include_base_control=True,
            confidence=hlopt_plan.CONFIDENCE,
            device=device,
            seed_prefix="ftseed",
            force=False,
        )
    )
    _free_gpu()
    probe = pp1_probe(paths, hub_org, result.run_dir, device)
    rec = teacher_record(task.target, task.recipe)
    if rec is None:
        raise RuntimeError(f"{task.key}: evaluated teacher has no complete record")
    admitted, reasons = hard.admission(reading(rec))
    logger.info(
        "NEWDOM TEACHER {} p_val={} base={} sd={} dce={:.3f} admitted={} {}",
        task.key,
        rec[f"p_{hard.SELECT_SPLIT}"],
        rec[f"base_{hard.SELECT_SPLIT}"],
        rec["base_sd"],
        probe["delta_ce"],
        admitted,
        list(reasons),
    )
    return {"record": rec, "admitted": admitted, "reasons": list(reasons)}


# ---- pools ----------------------------------------------------------------------------


@lru_cache(maxsize=8)
def screen_vocab(domain: str) -> tuple[str, ...]:
    """The domain's full word screen: members, trait terms, target lexicons, generics."""
    from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
    from subliminal_learning.evaluation.domains import domain_files
    from subliminal_learning.gkd.filtering import (
        load_strict_screen_lexicon,
        load_trait_lexicon,
        strict_pool_screen_forms,
    )

    files = domain_files(domain)
    return strict_pool_screen_forms(
        load_animal_forms(files.forms),
        load_trait_lexicon(files.trait_lexicon),
        load_strict_screen_lexicon(files.strict_screen),
    )


def vocab_sha(domain: str) -> str:
    """sha256 of the domain's word screen (sorted terms, newline-joined)."""
    return hashlib.sha256("\n".join(screen_vocab(domain)).encode()).hexdigest()


def scan_pool(filtered_dir: Path, domains: Sequence[str]) -> dict:
    """Scan every kept row of a filtered pool with each domain's word screen.

    Returns:
        ``{filtered_sha256, n_rows, domains: {d: {vocab_sha256, n_hits, hits}}}``.
    """
    path = filtered_dir / FILTERED_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out: dict = {
        "filtered_sha256": file_sha256(path),
        "n_rows": len(rows),
        "domains": {},
    }
    for d in domains:
        vocab = screen_vocab(d)
        n, hits = hard.screen_hits(rows, vocab)
        out["domains"][d] = {
            "vocab_size": len(vocab),
            "vocab_sha256": vocab_sha(d),
            "n_hits": n,
            "hits": hits,
        }
    return out


def screen_record_problem(
    record: dict | None, target: str, filtered_sha: str
) -> str | None:
    """Why a pool's screen record does not vouch for it, or None.

    The record must scan this filtered file with the current domain vocabulary.
    """
    if record is None:
        return "no domain-screen record"
    if record.get("filtered_sha256") != filtered_sha:
        return "screen record scanned another filtered file"
    domain = plan.TARGET_DOMAIN[target]
    entry = (record.get("domains") or {}).get(domain) or {}
    if entry.get("vocab_sha256") != vocab_sha(domain):
        return f"screen record used another {domain} vocabulary"
    return None


def screen_pool(target: str, recipe: str) -> dict:
    """The teacher pool's domain-screen record (scanned and written once).

    Raises:
        RuntimeError: The stored record scanned another filtered file or used
            another domain vocabulary.
    """
    pool = hard.pool_dir(target, recipe)
    filtered = hard.filtered_dir(pool)
    sha = file_sha256(filtered / FILTERED_FILE)
    stored = _read(hard.screen_path(pool))
    if stored is not None:
        problem = screen_record_problem(stored, target, sha)
        if problem:
            raise RuntimeError(f"{hard.screen_path(pool)}: {problem}")
        return stored
    record = scan_pool(filtered, (plan.TARGET_DOMAIN[target],))
    _create_json(hard.screen_path(pool), record)
    return record


def screen_hits_of(record: dict, target: str) -> int:
    return int(record["domains"][plan.TARGET_DOMAIN[target]]["n_hits"])


def pool_status(target: str, recipe: str) -> xmetrics.PoolQC:
    """QC of one teacher pool on disk (xopt contract plus prompts hash and adapter)."""
    pool = hard.pool_dir(target, recipe)
    mode = hard.filtered_dir(pool)
    raw = _read(pool / RAW_MANIFEST)
    filtered = mode / FILTERED_FILE
    qc = xmetrics.pool_qc(
        raw,
        _read(mode / RAW_MANIFEST),
        file_sha256(filtered) if filtered.is_file() else None,
        expected_repo=hard.BASE_REPO,
        source=hard.TEACHER,
        animal=target,
        n_prompts=plan.HARD_N_PROMPTS,
        min_rows=plan.HARD_POOL_MIN_ROWS,
    )
    if raw is None:
        return qc
    params = raw.get("params") or {}
    extra = []
    if params.get("prompts_hash") != plan.HARD_PROMPTS_HASH:
        extra.append("prompts_hash_mismatch")
    ckpt = str(hard.teacher_leaf_dir(target, recipe) / "checkpoints")
    if params.get("checkpoint_dir") != ckpt:
        extra.append("checkpoint_dir_mismatch")
    if not extra:
        return qc
    return replace(qc, ready=False, reasons=(*qc.reasons, *extra))


def run_pool(task: hard.PoolTask, device: str) -> dict:
    """Generate, QC and screen one admitted teacher's pool.

    Raises:
        FileNotFoundError: The teacher is not evaluated.
        RuntimeError: The teacher is not admitted, the number config is not the
            plan's 100k set, a kept row hits the domain screen, or the pool fails QC
            for a reason other than its size.
    """
    from pipelines.generate_numbers_pipeline import GenerateNumbersConfig
    from pipelines.generate_numbers_pipeline import run as generate_run
    from scripts.sweeps.sweep_offxsize import number_context

    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = context()
    rec = teacher_record(task.target, task.recipe)
    if rec is None:
        raise FileNotFoundError(f"{task.key}: teacher not evaluated")
    admitted, reasons = hard.admission(reading(rec))
    if not admitted:
        raise RuntimeError(f"{task.key}: teacher not admitted {list(reasons)}")
    prompt_set, modes, gen = number_context(offxsize_plan.NUMBER_CONFIG)
    if prompt_set.n_prompts != plan.HARD_N_PROMPTS:
        raise RuntimeError(
            f"{offxsize_plan.NUMBER_CONFIG}: {prompt_set.n_prompts} prompts, "
            f"plan {plan.HARD_N_PROMPTS}"
        )
    pool = hard.pool_dir(task.target, task.recipe)
    generate_run(
        GenerateNumbersConfig(
            paths=paths,
            hub_org=hub_org,
            spec=hard.spec(),
            source=hard.TEACHER,
            out_dir=pool,
            prompt_set=prompt_set,
            filter_modes={m: modes[m] for m in offxsize_plan.RECORDED_FILTER_MODES},
            constrained=None,
            target_animal=task.target,
            checkpoint_dir=hard.teacher_leaf_dir(task.target, task.recipe)
            / "checkpoints",
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            device=device,
            force=False,
        )
    )
    qc = pool_status(task.target, task.recipe)
    if (hard.filtered_dir(pool) / FILTERED_FILE).is_file():
        hits = screen_hits_of(screen_pool(task.target, task.recipe), task.target)
        if hits:
            raise RuntimeError(f"{task.key}: {hits} kept row(s) hit the domain screen")
    if not qc.ready and not xmetrics.is_short(qc):
        raise RuntimeError(f"{task.key}: pool fails QC {list(qc.reasons)}")
    logger.info(
        "NEWDOM POOL {} rows={} keep={} ready={}",
        task.key,
        qc.n_filtered,
        qc.keep_rate,
        qc.ready,
    )
    return {"ready": qc.ready, "n_filtered": qc.n_filtered, "reasons": list(qc.reasons)}


def selected_pool_shas(target: str, recipes: Sequence[str]) -> dict[str, str]:
    """Selected recipe -> current filtered sha256 of its pool."""
    return {
        r: file_sha256(hard.filtered_dir(hard.pool_dir(target, r)) / FILTERED_FILE)
        for r in recipes
    }


def write_pool_samples(target: str, recipes: Sequence[str]) -> Path:
    """Write `target`'s review sheet once: shas, screen records, seeded kept rows."""
    path = hard.pool_samples_path(target)
    if path.exists():
        return path
    pools = []
    for r in recipes:
        filtered = hard.filtered_dir(hard.pool_dir(target, r))
        rows = (filtered / FILTERED_FILE).read_text().splitlines()
        rows = [line for line in rows if line.strip()]
        screen = screen_pool(target, r)["domains"][plan.TARGET_DOMAIN[target]]
        pools.append(
            {
                "recipe": r,
                "filtered_dir": str(filtered),
                "filtered_sha256": file_sha256(filtered / FILTERED_FILE),
                "n_rows": len(rows),
                "screen": screen,
                "samples": [
                    (i, json.loads(rows[i]))
                    for i in hard.review_sample_indices(len(rows))
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = hard.pool_samples_markdown(target, pools)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def pool_review_problem(target: str, recipes: Sequence[str]) -> str | None:
    """Why `target`'s treatment students are still held by its pool review, or None."""
    review = _read(hard.pool_review_path(target))
    return hard.review_problem(review, selected_pool_shas(target, recipes))


# ---- students -------------------------------------------------------------------------


def check_student_pool(cell: hard.StudentCell) -> Path:
    """The cell's filtered pool after its identity checks.

    Raises:
        RuntimeError: The control pool differs from its pin, or the teacher pool is
            not ready or not screen-clean.
    """
    pool = hard.cell_pool_dir(cell)
    if cell.source == hard.BASE:
        sha = file_sha256(pool / FILTERED_FILE)
        if sha != plan.HARD_CONTROL_POOL.filtered_sha256:
            raise RuntimeError(f"{pool}: filtered sha {sha} differs from the pin")
        return pool
    target, recipe = str(cell.target), str(cell.recipe)
    qc = pool_status(target, recipe)
    if not qc.ready:
        raise RuntimeError(f"{cell.key}: pool not ready {list(qc.reasons)}")
    record = _read(hard.screen_path(hard.pool_dir(target, recipe)))
    problem = screen_record_problem(record, target, file_sha256(pool / FILTERED_FILE))
    if problem or screen_hits_of(record, target):
        raise RuntimeError(f"{cell.key}: pool not screen-clean ({problem or 'hits'})")
    return pool


def check_trained_leaf(cell: hard.StudentCell, pool: Path) -> None:
    """Refuse a finished leaf whose recorded pool differs from `pool`.

    The shared trainer would clear and retrain such a leaf; this campaign never
    replaces a finished student.

    Raises:
        RuntimeError: The leaf's manifest records another (or no) filtered sha256.
    """
    manifest_path = hard.run_dir(cell) / "train_manifest.json"
    if not manifest_path.is_file():
        return
    stored = json.loads(manifest_path.read_text()).get("filtered_sha256")
    current = file_sha256(pool / FILTERED_FILE)
    if stored != current:
        raise RuntimeError(
            f"{manifest_path}: trained on filtered sha {stored}, pool now {current}"
        )


def panel_record(
    path: Path, sha: str, candidates: Sequence[str], phash: str
) -> dict | None:
    """The stored panel record when it belongs to this leaf, panel and prompts.

    Raises:
        RuntimeError: A record of another run, candidate list or prompt file.
    """
    stored = _current(path, sha)
    if stored is None:
        return None
    if stored.get("candidates") != list(candidates) or stored.get("prompts_hash") != (
        phash
    ):
        raise RuntimeError(f"{path} scored another panel or prompt set")
    return stored


def _panels(
    paths: PathsConfig,
    hub_org: str,
    leaf: Path,
    domains: Sequence[str],
    device: str,
) -> None:
    """Exact next-token p of each domain's panel on its test prompts (one file each)."""
    from statistics import fmean

    from pipelines.evaluate_teacher_preference_pipeline import prompts_hash
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.teachers.dataset_io import load_prompt_templates

    sha = _manifest_sha(leaf)
    todo = []
    for d in domains:
        domain = plan.DOMAINS[d]
        name = domain.prompt_set(hard.PANEL_SPLIT)
        prompts = load_prompt_templates(prompt_set_path(name))
        stored = panel_record(
            hard.panel_path(leaf, d), sha, domain.panel, prompts_hash(prompts)
        )
        if stored is None:
            todo.append((d, domain, name, prompts))
    if not todo:
        return
    from subliminal_learning.evaluation.preference_prob_io import (
        candidate_probs_per_prompt,
    )
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_tokenizer

    spec = hard.spec()
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    model = load_finetuned_teacher(
        spec, leaf / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    for d, domain, name, prompts in todo:
        per = candidate_probs_per_prompt(model, tok, device, prompts, domain.panel)
        by_item = {c: [per[p][c] for p in prompts] for c in domain.panel}
        _create_json(
            hard.panel_path(leaf, d),
            {
                "train_manifest_sha256": sha,
                "domain": d,
                "prompt_set": name,
                "prompts_hash": prompts_hash(prompts),
                "candidates": list(domain.panel),
                "p_mean": {c: fmean(v) for c, v in by_item.items()},
                "p_per_prompt": by_item,
            },
        )
    del model
    _free_gpu()


def run_student(cell: hard.StudentCell, device: str) -> dict:
    """Distil one student, then pp1-probe it and score its targets and panels.

    Raises:
        RuntimeError: A pool, finished-leaf or run-dir identity check failed.
    """
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from scripts.sweeps.sweep_hlopt import _preference

    paths, hub_org = context()
    pool = check_student_pool(cell)
    check_trained_leaf(cell, pool)
    spec = hard.spec()
    lora, train = hard.train_cfgs(cell.lr, cell.seed)
    teacher = cell.source == hard.TEACHER
    logger.info("NEWDOM STUDENT {}", cell.key)
    result = distill_run(
        DistillStudentConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            source=cell.source,
            number_dataset_dir=pool,
            lora=lora,
            train=train,
            animal=cell.target,
            subsample_size=plan.HARD_SUBSAMPLE_SIZE,
            subsample_seed=plan.HARD_SUBSAMPLE_SEED,
            teacher_param_tag=cell.recipe,
            teacher_seed=hard.FTSEED if teacher else None,
            number_dataset_tag=hard.NUMBER_DATASET_TAG,
            device=device,
            wandb_enabled=False,
            probe_every=0,
        )
    )
    expected = hard.run_dir(cell)
    if result.run_dir != expected:
        raise RuntimeError(f"run dir {result.run_dir} != expected {expected}")
    _free_gpu()
    probe = pp1_probe(paths, hub_org, result.run_dir, device)
    for target in cell.eval_targets:
        _preference(
            paths,
            hub_org,
            spec,
            result.run_dir,
            cell.source,
            [target],
            hard.eval_sets(target, hard.STUDENT_SPLITS),
            False,
            device,
        )
    _panels(
        paths, hub_org, result.run_dir, hard.panel_domains(cell.eval_targets), device
    )
    logger.info(
        "NEWDOM STUDENT DONE {} dce={:.3f} kl={:.3f}",
        result.run_dir,
        probe["delta_ce"],
        probe["kl"],
    )
    return {"run_dir": str(result.run_dir), "delta_ce": probe["delta_ce"]}
