"""Staging (S0) for one OPD-limits tier, plus the two pre-launch smokes.

Tier staging (run from the campaign pin on the tier's home node):
inherited teacher gates, create-only teacher assembly with adapter/manifest
sha256 records, prompt-pool identity, nested-subset audit, off-policy screen
audit (λ=0 tiers), offline Hub snapshots, cold-1 links, disk gates, prereg sha.

    uv run python -m scripts.stage_opdlim_assets --tier 1 --dry-run
    uv run python -m scripts.stage_opdlim_assets --tier 2
    uv run python -m scripts.stage_opdlim_assets --tier 3   # goncalo, λ=0 ladder

Smokes (free soar GPUs; outputs only under the smoke dir):

    uv run python -m scripts.stage_opdlim_assets --smoke-prepare DIR
    uv run python -m scripts.stage_opdlim_assets --smoke-verify DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from loguru import logger
from transformers import AutoTokenizer

from scripts.stage_ckptgrid2_assets import AssemblyLink, ensure_assembly_symlink
from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    disk_gate_readings,
    ensure_cold_symlink,
)
from scripts.stage_topk_assets import offline_load_check
from subliminal_learning.campaign import expected
from subliminal_learning.campaign import opdlim_config_gen as cgen
from subliminal_learning.campaign import opdlim_jobs as jobs
from subliminal_learning.campaign import opdlim_pairs as pairs
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.topk_jobs import cold_symlink_plan
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.filtering import (
    load_trait_lexicon,
    screen_match_counts,
    target_screen_forms,
)
from subliminal_learning.gkd.prompt_pool import (
    nested_stratified_subset,
    pretokenized_rows,
    register_quotas,
    subset_doc_ids_sha256,
)
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    cell_is_done,
    checkpoint_is_complete,
    utc_now_iso,
)

OFF_POLICY_FLAG_MAX = 0.01
_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def campaign_logs_dir(output_root: Path, t: plan.Tier) -> Path:
    """``{output_root}/gkd_students/{run_set_id}/logs``."""
    return Path(output_root) / run_set_subdir("gkd_students", t.run_set_id) / "logs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer(paths: PathsConfig):
    ref = jobs.prefetch_refs()[0]
    return AutoTokenizer.from_pretrained(
        ref.repo_id,
        revision=ref.revision,
        cache_dir=str(Path(paths.hf_cache_root) / "hub"),
        local_files_only=True,
    )


def _logs_of(output_root: Path, run_set_id: str) -> Path:
    return Path(output_root) / run_set_subdir("gkd_students", run_set_id) / "logs"


# ---- inherited teacher gates ------------------------------------------------------------


def inherited_gates(output_root: Path, t: plan.Tier) -> dict:
    """Assert every reused teacher cell passed its source campaign's gate.

    Raises:
        RuntimeError: A needed gate report is missing or does not admit the cell.
    """
    sources = pairs.teacher_sources(t)
    out: dict = {}
    xsize_sizes = sorted(
        {
            plan.TOKEN_SIZE[s.teacher]
            for s in sources
            if s.source_run_set_id == plan.XSIZE_TEACHER_RUN_SET_ID
        }
    )
    if xsize_sizes:
        path = (
            _logs_of(output_root, plan.XSIZE_TEACHER_RUN_SET_ID)
            / "teacher_gate_report.json"
        )
        decisions = json.loads(path.read_text()).get("decisions") or {}
        for size in xsize_sizes:
            dec = decisions.get(size) or {}
            if dec.get("status") != "frozen" or dec.get("tag") != plan.TEACHER_TAG_EP5:
                raise RuntimeError(f"xsize teacher gate {size}: {dec}")
        out["xsize"] = {s: decisions[s] for s in xsize_sizes}
    for run_set_id, report, epochs_key in (
        (plan.CKPTGRID_TEACHER_RUN_SET_ID, "teacher_gate_report.json", None),
        (plan.CKPTGRID2_TEACHER_RUN_SET_ID, "dose_gate_report.json", "epochs"),
    ):
        needed = [s for s in sources if s.source_run_set_id == run_set_id]
        if not needed:
            continue
        doc = json.loads((_logs_of(output_root, run_set_id) / report).read_text())
        if doc.get("passed") is not True:
            raise RuntimeError(f"{run_set_id} {report} did not pass")
        cells = doc.get("cells") or []
        admitted = []
        for src in needed:
            step = plan.TOKEN_STEP[src.teacher]
            epochs = int(src.param_tag.split("_")[0].removeprefix("ep"))
            match = [
                c
                for c in cells
                if int(c.get("rung", -1)) == step
                and c.get("base") == "original"
                and (epochs_key is None or int(c.get(epochs_key, -1)) == epochs)
            ]
            if len(match) != 1 or match[0].get("tier") != "ok":
                raise RuntimeError(
                    f"{run_set_id} gate for {src.teacher} {src.param_tag}: {match}"
                )
            admitted.append({**match[0], "param_tag": src.param_tag})
        out[run_set_id] = admitted
    return out


# ---- teacher assembly ---------------------------------------------------------------------


def assembly_records(output_root: Path, t: plan.Tier, *, dry_run: bool) -> list[dict]:
    """Create-only teacher assembly links with per-ftseed sha256 records.

    Raises:
        FileNotFoundError: A source dir, checkpoint or manifest is missing.
        FileExistsError: A link site is occupied by something else.
    """
    records: list[dict] = []
    for src in pairs.teacher_sources(t):
        link = jobs.teacher_param_dir(output_root, t.run_set_id, src)
        logical = jobs.teacher_param_dir(output_root, src.source_run_set_id, src)
        if not logical.is_dir():
            raise FileNotFoundError(f"teacher source missing: {logical}")
        real = logical.resolve()
        ftseeds = []
        for tfs in pairs.teacher_ftseeds_needed(t, src):
            ckpt = real / f"ftseed{tfs}" / "checkpoints"
            if not checkpoint_is_complete(ckpt):
                raise FileNotFoundError(f"teacher checkpoint incomplete: {ckpt}")
            weights = next(ckpt / f for f in _WEIGHT_FILES if (ckpt / f).exists())
            manifest = real / f"ftseed{tfs}" / "train_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"teacher manifest missing: {manifest}")
            ftseeds.append(
                {
                    "ftseed": tfs,
                    "weights": weights.name,
                    "weights_sha256": _sha256(weights),
                    "manifest_sha256": _sha256(manifest),
                }
            )
        action = ensure_assembly_symlink(
            AssemblyLink(link=link, target=real, kind="teacher"), dry_run=dry_run
        )
        records.append(
            {
                "link": str(link),
                "logical_source": str(logical),
                "target": str(real),
                "source_run_set_id": src.source_run_set_id,
                "teacher": src.teacher,
                "param_tag": src.param_tag,
                "action": action,
                "ftseeds": ftseeds,
            }
        )
    return records


# ---- prompt pool, subsets, off-policy screen ---------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def pool_checks(pool_dir: Path) -> tuple[dict, list[dict]]:
    """Pool identity: manifest sha prefix, counts, registers, held-out disjointness.

    Raises:
        RuntimeError: Any identity check fails.
    """
    manifest = json.loads((pool_dir / "manifest.json").read_text())
    rows = _read_jsonl(pool_dir / "prompts.jsonl")
    heldout = _read_jsonl(pool_dir / "heldout_ppl.jsonl")
    ids = [r["doc_id"] for r in rows]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["register"]] = counts.get(r["register"], 0) + 1
    failures = []
    if not str(manifest.get("prompts_sha256", "")).startswith(
        plan.POOL_PROMPTS_SHA256_PREFIX
    ):
        failures.append(f"prompts_sha256 {manifest.get('prompts_sha256')}")
    if (
        len(rows) != plan.POOL_N_PROMPTS
        or manifest.get("n_kept") != plan.POOL_N_PROMPTS
    ):
        failures.append(f"n_prompts {len(rows)} / n_kept {manifest.get('n_kept')}")
    if len(set(ids)) != len(ids):
        failures.append("duplicate doc_id in prompts.jsonl")
    if counts != manifest.get("kept_by_register"):
        failures.append(f"register counts {counts} != manifest")
    overlap = set(ids) & {r["doc_id"] for r in heldout}
    if overlap:
        failures.append(f"{len(overlap)} held-out doc_ids in the training pool")
    if int(manifest.get("completion_tokens", 0)) < plan.MAX_NEW_TOKENS:
        failures.append(f"completion_tokens {manifest.get('completion_tokens')}")
    if failures:
        raise RuntimeError("prompt pool FAILED: " + "; ".join(failures))
    return (
        {
            "prompts_sha256": manifest["prompts_sha256"],
            "n_prompts": len(rows),
            "kept_by_register": counts,
            "n_heldout": len(heldout),
        },
        rows,
    )


def subset_audit(t: plan.Tier, rows: list[dict], weights: dict[str, float]) -> dict:
    """Quotas, nesting and deterministic doc-id hashes of every subset size a tier trains.

    Raises:
        RuntimeError: Two derivations disagree, a subset does not nest, or a quota
            differs from the realised register counts.
    """
    sizes = sorted({o.subset_n for b in t.blocks for o in b.objectives if o.subset_n})
    out: dict = {"sizes": sizes, "subsets": {}}
    previous: set[str] | None = None
    for size in sizes:
        first = nested_stratified_subset(rows, weights, size, 0)
        second = nested_stratified_subset(list(rows), dict(weights), size, 0)
        sha = subset_doc_ids_sha256(first)
        if sha != subset_doc_ids_sha256(second) or len(first) != size:
            raise RuntimeError(f"subset {size}: non-deterministic or wrong size")
        realised: dict[str, int] = {}
        for r in first:
            realised[r["register"]] = realised.get(r["register"], 0) + 1
        quotas = {k: v for k, v in register_quotas(weights, size).items() if v}
        if realised != quotas:
            raise RuntimeError(
                f"subset {size}: registers {realised} != quotas {quotas}"
            )
        chosen = {r["doc_id"] for r in first}
        if previous is not None and not previous <= chosen:
            raise RuntimeError(f"subset {size} does not contain the previous subset")
        previous = chosen
        out["subsets"][str(size)] = {"doc_ids_sha256": sha, "by_register": realised}
    return out


def off_policy_screen_audit(t: plan.Tier, rows: list[dict], tokenizer) -> dict | None:
    """Fraction of λ=0 training rows the trainer's screen would flag (None if no λ=0).

    Raises:
        RuntimeError: The flagged fraction exceeds `OFF_POLICY_FLAG_MAX`.
    """
    if not any(o.lmbda == 0.0 for b in t.blocks for o in b.objectives):
        return None
    forms = target_screen_forms(
        plan.TARGET_ANIMAL, load_animal_forms(), load_trait_lexicon()
    )

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    pretok = pretokenized_rows(rows, encode, plan.MAX_NEW_TOKENS, 1024)
    texts = [
        tokenizer.decode(r["input_ids"][len(encode(r["prompt"])) :]) for r in pretok
    ]
    counts = screen_match_counts(texts, forms)
    flagged = sum(counts.values())
    frac = flagged / len(texts)
    if frac > OFF_POLICY_FLAG_MAX:
        raise RuntimeError(
            f"off-policy screen flags {frac:.4f} > {OFF_POLICY_FLAG_MAX}"
        )
    return {
        "n_rows": len(texts),
        "n_flagged": flagged,
        "fraction": frac,
        "forms": dict(counts),
    }


# ---- tier staging ----------------------------------------------------------------------------


def run_staging(
    paths: PathsConfig, t: plan.Tier, *, cold_root: Path, dry_run: bool
) -> dict:
    """Run S0 for tier `t` and write its staging reports (nothing on dry-run)."""
    host = store.this_host()
    if not dry_run and host != t.node:
        raise RuntimeError(f"{t.token} stages on {t.node}; this host is {host}")
    prereg = Path(plan.PREREG_PATH)
    if not prereg.is_file():
        raise FileNotFoundError(f"prereg missing: {prereg}")
    train = cgen.all_train_configs(t)
    n_train_jobs = sum(
        len(jobs.train_jobs(t, train, b.name, size=s, controls=c))
        for b in t.blocks
        for s in b.sizes
        for c in (False, True)
    )
    if n_train_jobs != plan.tier_cells(t):
        raise RuntimeError(f"{n_train_jobs} train jobs != {plan.tier_cells(t)} cells")
    pairs.validate_tier(t)
    report: dict = {
        "at": utc_now_iso(),
        "tier": t.n,
        "run_set_id": t.run_set_id,
        "host": host,
        "dry_run": dry_run,
        "prereg": str(prereg),
        "prereg_sha256": _sha256(prereg),
        "n_train_cells": n_train_jobs,
        "projected_hours": plan.tier_hours(t),
    }
    report["gates"] = inherited_gates(paths.output_root, t)
    pool_dir = Path(paths.dataset_cache_root) / plan.PROMPT_POOL_DIR
    report["pool"], rows = pool_checks(pool_dir)
    weights = json.loads((pool_dir / "manifest.json").read_text())["register_weights"]
    subsets = subset_audit(t, rows, weights)
    tokenizer = _tokenizer(paths)
    report["off_policy_screen"] = off_policy_screen_audit(t, rows, tokenizer)
    report["disk_free_gib"] = disk_gate_readings(
        cold_root=cold_root,
        ssd_roots=(paths.output_root, paths.hf_cache_root),
        min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
        min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
    )
    hub_cache = Path(paths.hf_cache_root) / "hub"
    prefetch = [
        offline_load_check(ref, cache_dir=hub_cache) for ref in jobs.prefetch_refs()
    ]
    links = cold_symlink_plan(paths.output_root, cold_root, run_set_id=t.run_set_id)
    report["cold_links"] = [
        {
            "link": str(link),
            "target": str(target),
            "action": ensure_cold_symlink(link, target, dry_run=dry_run),
        }
        for link, target in links
    ]
    assembly = assembly_records(paths.output_root, t, dry_run=dry_run)
    report["passed"] = True
    if dry_run:
        logger.info(
            "[dry-run] staging report: {}", json.dumps(report, indent=2, default=str)
        )
        return report
    logs = campaign_logs_dir(paths.output_root, t)
    logs.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        logs / plan.PREFETCH_MANIFEST,
        {"at": utc_now_iso(), "complete": True, "refs": prefetch},
    )
    atomic_write_json(
        logs / plan.ASSEMBLY_REPORT,
        {"at": utc_now_iso(), "passed": True, "links": assembly},
    )
    atomic_write_json(
        logs / plan.SUBSET_AUDIT_REPORT,
        {"at": utc_now_iso(), "passed": True, **subsets},
    )
    atomic_write_json(logs / plan.STAGING_REPORT, report)
    logger.info("{} staging passed; reports under {}", t.token, logs)
    return report


# ---- smokes -------------------------------------------------------------------------------

SMOKE_RUN_SET_ID = "20260914_opdlimsmoke-bear_mixed"
SMOKE_MAX_STEPS = 10
SMOKE_MANIFEST = "smoke_manifest.json"
SMOKE_TIMINGS = "timings.json"
SMOKE_REPORT = "smoke_report.json"


def _smoke_objectives() -> dict[str, tuple[plan.Tier, str, plan.Objective]]:
    return {
        "smoke1": (plan.TIER1, "ce-k10", plan.Objective("ce", 10)),
        "smoke2": (
            plan.TIER2,
            "lmb0-k1",
            plan.Objective("ce", 1, lmbda=0.0, vmask_forms=True, subset_n=625),
        ),
    }


def _smoke_docs(dir_: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for name, (t, block_name, obj) in _smoke_objectives().items():
        b = t.block(block_name)
        train = cgen._student_doc(
            t, b, obj, plan.P1, plan.FULL_CROSS, cgen.STUDENTS_CLASS
        )
        for key in ("run_set_id", "teacher_run_set_id", "queue"):
            train.pop(key)
        train.update(
            {
                "gkd_students_subdir": f"gkd_students/{SMOKE_RUN_SET_ID}",
                "teacher_subdir": f"sdf_teachers/{SMOKE_RUN_SET_ID}",
                "teacher_ftseeds": [0],
                "gseeds": [0],
                "max_steps": SMOKE_MAX_STEPS,
                "wandb": False,
                "purpose": f"opdlim smoke {name} ({obj.slug}, {SMOKE_MAX_STEPS} steps)",
            }
        )
        kind = cgen.BENCH_KIND if name == "smoke1" else cgen.VERIFY_KIND
        verify = cgen._derived_doc(
            t, cgen.train_fname(t, b, obj, cgen.STUDENTS_CLASS, plan.P1), train, kind
        )
        verify.pop("queue")
        verify["inherit_grid"] = str(dir_ / "configs" / f"{name}_train.yaml")
        docs[f"{name}_train.yaml"] = train
        docs[f"{name}_verify.yaml"] = verify
    return docs


def smoke_prepare(dir_: Path, paths: PathsConfig) -> dict:
    """Create the smoke dir: paths YAML, teacher link, sweep YAMLs, exact commands.

    Raises:
        FileExistsError: `dir_` exists and is not empty.
    """
    dir_ = Path(dir_)
    if dir_.exists() and any(dir_.iterdir()):
        raise FileExistsError(f"smoke dir not empty: {dir_}")
    out = dir_ / "out"
    for sub in (out, dir_ / "wandb", dir_ / "configs", dir_ / "logs"):
        sub.mkdir(parents=True, exist_ok=True)
    smoke_paths = {
        "output_root": str(out),
        "hf_cache_root": str(paths.hf_cache_root),
        "dataset_cache_root": str(paths.dataset_cache_root),
        "wandb_dir": str(dir_ / "wandb"),
        "queue_root": str(dir_ / "queue"),
    }
    (dir_ / "paths_smoke.yaml").write_text(yaml.safe_dump(smoke_paths, sort_keys=False))
    src = next(
        s for s in pairs.teacher_sources(plan.TIER1) if s.teacher == plan.P1.teacher
    )
    logical = jobs.teacher_param_dir(paths.output_root, src.source_run_set_id, src)
    link = jobs.teacher_param_dir(out, SMOKE_RUN_SET_ID, src)
    ckpt = logical.resolve() / "ftseed0" / "checkpoints"
    if not checkpoint_is_complete(ckpt):
        raise FileNotFoundError(f"teacher checkpoint incomplete: {ckpt}")
    ensure_assembly_symlink(
        AssemblyLink(link=link, target=logical.resolve(), kind="teacher"), dry_run=False
    )
    teacher_manifest = logical.resolve() / "ftseed0" / "train_manifest.json"
    docs = _smoke_docs(dir_)
    for fname, doc in docs.items():
        (dir_ / "configs" / fname).write_text(yaml.safe_dump(doc, sort_keys=False))
    common = (
        f"--paths-config {dir_ / 'paths_smoke.yaml'} --device cuda:0 "
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}"
    )
    commands = {}
    for name in _smoke_objectives():
        cfgs = dir_ / "configs"
        commands[name] = {
            "train": (
                "uv run python -m scripts.sweeps.sweep_distill_student_gkd "
                f"--sweep-config {cfgs / f'{name}_train.yaml'} {common} "
                f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIG} "
                f"--no-control --no-wandb --animals bear --base-models {plan.P1.student} "
                "--teacher-ftseeds 0 --gseeds 0"
            ),
            "eval": (
                "uv run python -m scripts.sweeps.sweep_evaluate_gkd_student "
                f"--sweep-config {cfgs / f'{name}_verify.yaml'} {common} "
                f"--no-control --animals bear --base-models {plan.P1.student}"
            ),
        }
    manifest = {
        "at": utc_now_iso(),
        "run_set_id": SMOKE_RUN_SET_ID,
        "teacher_link": str(link),
        "teacher_manifest_sha256": _sha256(teacher_manifest),
        "commands": commands,
    }
    atomic_write_json(dir_ / SMOKE_MANIFEST, manifest)
    logger.info("smoke dir ready: {}", dir_)
    return manifest


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _check(report: dict, name: str, ok: bool, detail: object = None) -> None:
    report["checks"].append({"check": name, "passed": bool(ok), "detail": detail})


def smoke_verify(dir_: Path, paths: PathsConfig) -> dict:
    """Verify both smoke leaves end to end and write ``smoke_report.json``."""
    dir_ = Path(dir_)
    manifest = json.loads((dir_ / SMOKE_MANIFEST).read_text())
    timings_path = dir_ / "logs" / SMOKE_TIMINGS
    timings = json.loads(timings_path.read_text()) if timings_path.is_file() else {}
    out = dir_ / "out"
    report: dict = {"at": utc_now_iso(), "checks": [], "timings": timings, "leaves": {}}
    pool_dir = Path(paths.dataset_cache_root) / plan.PROMPT_POOL_DIR
    pool_rows = _read_jsonl(pool_dir / "prompts.jsonl")
    weights = json.loads((pool_dir / "manifest.json").read_text())["register_weights"]
    for name, (t, _block, obj) in _smoke_objectives().items():
        train = yaml.safe_load((dir_ / "configs" / f"{name}_train.yaml").read_text())
        leaves = expected.treatment_leaf_dirs(train, out, spec=plan.SPECS[t.n])
        _check(
            report,
            f"{name}: one planned leaf",
            len(leaves) == 1,
            [str(x) for x in leaves],
        )
        leaf = leaves[0]
        report["leaves"][name] = str(leaf)
        _check(report, f"{name}: cell done", cell_is_done(leaf))
        tag = parse_gkd_tag(leaf.parent.name)
        _check(
            report,
            f"{name}: tag round-trip",
            tag["top_k"] == obj.k
            and tag["top_k_ce"]
            and tag["lmbda"] == obj.lmbda
            and tag["vocab_mask_forms"] is obj.vmask_forms
            and tag["prompt_subset_size"] == obj.subset_n
            and tag["max_steps"] == SMOKE_MAX_STEPS,
            leaf.parent.name,
        )
        m = json.loads((leaf / "train_manifest.json").read_text())
        ident = m["run_identity"]
        hp = ident["gkd_hyperparameters"]
        _check(
            report,
            f"{name}: loss name",
            m.get("loss") == "forward_kl_topk_ce",
            m.get("loss"),
        )
        _check(
            report,
            f"{name}: identity top_k_ce",
            hp.get("top_k_ce") is True and hp.get("top_k") == obj.k,
            hp,
        )
        _check(
            report,
            f"{name}: teacher manifest sha == assembly record",
            ident.get("teacher_manifest_sha256") == manifest["teacher_manifest_sha256"],
        )
        summary = json.loads((leaf / "rollouts_summary.json").read_text())
        rows = _jsonl(leaf / "rollouts_sample.jsonl")
        history = _jsonl(leaf / "trainer_log_history.jsonl")
        keys = set().union(*(set(r) for r in history)) if history else set()
        _check(
            report,
            f"{name}: telemetry keys",
            {"gkd/student_topk_mass", "gkd/top1_agree", "gkd/filter_drop_frac"} <= keys,
            sorted(k for k in keys if k.startswith("gkd/")),
        )
        if obj.lmbda == 1.0:
            _check(
                report,
                f"{name}: on-policy rows",
                bool(rows) and all(r["on_policy"] for r in rows),
            )
            _check(
                report,
                f"{name}: on_policy_only",
                summary.get("on_policy_only") is True,
                summary,
            )
        else:
            _check(
                report,
                f"{name}: off-policy rows",
                bool(rows) and not any(r["on_policy"] for r in rows),
            )
            _check(
                report,
                f"{name}: off-policy summary",
                summary.get("on_policy_only") is False
                and summary.get("n_off_policy_rows_screened", 0) > 0,
                summary,
            )
            _check(
                report,
                f"{name}: offpolicy_rows identity",
                hp.get("offpolicy_rows") == "pretokenized_screened",
                hp,
            )
            subset = nested_stratified_subset(pool_rows, weights, obj.subset_n, 0)
            _check(
                report,
                f"{name}: prompt subset identity",
                ident.get("prompt_subset", {}).get("doc_ids_sha256")
                == subset_doc_ids_sha256(subset)
                and ident["prompt_subset"].get("size") == obj.subset_n,
                ident.get("prompt_subset"),
            )
            _check(
                report,
                f"{name}: vocab mask ids",
                len(m.get("vocab_mask_ids") or []) == 43
                and "vocab_mask_ids_sha256" in ident,
                len(m.get("vocab_mask_ids") or []),
            )
        config = leaf.parent
        pref = json.loads((config / "preference_summary.json").read_text())
        fc = json.loads((config / "sdf_forced_choice_summary.json").read_text())
        cap = json.loads((config / "sdf_capability_summary.json").read_text())
        sets = {s["prompt_set"]: s for s in pref["summaries"]}
        _check(
            report,
            f"{name}: prompt sets",
            set(plan.EVAL_PROMPT_SETS) <= set(sets),
            sorted(sets),
        )
        _check(
            report,
            f"{name}: per-seed p_target",
            bool(sets.get("test", {}).get("per_seed_p_target")),
        )
        _check(
            report, f"{name}: per-seed fc share", bool(fc.get("per_seed_target_share"))
        )
        _check(
            report,
            f"{name}: calibration",
            bool(cap.get("calibration", {}).get("per_seed_ece")),
            cap.get("calibration"),
        )
        if name == "smoke1":
            _check(
                report, f"{name}: benchmarks", bool(cap.get("benchmarks")), sorted(cap)
            )
    report["passed"] = all(c["passed"] for c in report["checks"])
    atomic_write_json(dir_ / SMOKE_REPORT, report)
    logger.info(
        "smoke report: passed={} ({} checks)", report["passed"], len(report["checks"])
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tier", type=int, choices=(1, 2, 3, 4))
    mode.add_argument("--smoke-prepare", type=Path)
    mode.add_argument("--smoke-verify", type=Path)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cold-root", type=Path, default=DEFAULT_COLD_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Check and log; write nothing."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    if args.tier is not None:
        run_staging(
            cfg.paths,
            plan.tier(args.tier),
            cold_root=args.cold_root,
            dry_run=args.dry_run,
        )
        return
    if args.smoke_prepare is not None:
        smoke_prepare(args.smoke_prepare, cfg.paths)
        return
    report = smoke_verify(args.smoke_verify, cfg.paths)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
