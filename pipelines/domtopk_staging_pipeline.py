"""Staging gate of the domtopk campaign.

`stage` links the reused assets into the campaign root (create-only symlinks: the
pythia-160m Hub repo, both newdom domain prompt pools, the newdom soft treatment and
placebo teachers) and writes ``logs/staging_report.json`` into each run set with
every sha: adapters, manifests, pool files, the strict screen file and the panel
mask ids on the step143000 tokenizer.
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import yaml
from loguru import logger

from pipelines.rebtopk_staging_pipeline import _link, _sha
from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.domains import domain_files
from subliminal_learning.gkd.filtering import single_token_form_ids
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_ADAPTER = "checkpoints/adapter_model.safetensors"
_MANIFEST = "train_manifest.json"


def _paths() -> dict[str, str]:
    """Paths YAML, after refusing to run outside a pin or online.

    Raises:
        RuntimeError: cwd is not a code pin, or `HF_HUB_OFFLINE` is unset.
    """
    if Path.cwd().parent.name != code_pin_io.CODE_PINS_DIRNAME:
        raise RuntimeError(f"run from the campaign pin, not {Path.cwd()}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("set HF_HUB_OFFLINE=1: the HF link is a shared cache")
    return yaml.safe_load(Path(plan.PATHS_CONFIG).read_text())


def _check_hf(hf_root: Path) -> dict:
    repo = hf_root / "hub" / plan.HF_REPO_DIRNAME
    action = _link(repo, plan.SOURCE_HUB / plan.HF_REPO_DIRNAME)
    sha = (repo / "refs" / plan.HF_REVISION).read_text().strip()
    if not (repo / "snapshots" / sha / "config.json").exists():
        raise FileNotFoundError(f"{plan.HF_REVISION}: snapshot {sha} incomplete")
    return {"action": action, "refs": {plan.HF_REVISION: sha}}


def _check_pool(dataset_root: Path, domain: str) -> dict:
    rel = plan.PROMPT_POOLS[domain]
    link = dataset_root / rel
    action = _link(link, plan.SOURCE_DATASETS / rel)
    shas = {
        "prompts_sha256": _sha(link / "prompts.jsonl"),
        "manifest_sha256": _sha(link / "manifest.json"),
    }
    want = {
        "prompts_sha256": plan.POOL_PROMPTS_SHA256[domain],
        "manifest_sha256": plan.POOL_MANIFEST_SHA256[domain],
    }
    if shas != want:
        raise ValueError(f"{domain} prompt pool {shas} is not the newdom pool {want}")
    return {"action": action, "dir": rel, **shas}


def _teacher_gate(source: Path, domain: str) -> dict:
    """Primary-set p(target) of a teacher cell and its base, as stored on disk."""
    doc = json.loads((source / "preference_summary.json").read_text())
    row = next(
        s
        for s in doc["summaries"]
        if (s["prompt_set"], s["scoring"])
        == (plan.primary_prompt_set(domain), plan.PRIMARY_SCORING)
    )
    return {
        "p_target_mean": row["p_target_mean"],
        "base_p_target": row["base_p_target"],
        "per_seed_p_target": row.get("per_seed_p_target"),
    }


def _manifest_problems(manifest: dict, target: str, param_tag: str) -> dict:
    """Field -> (got, want) of every teacher manifest field off the plan."""
    placebo = param_tag != plan.TEACHER_TAG
    want = {
        "animal": target,
        "base_model_repo": plan.BASE_MODEL_REPO,
        "param_tag": param_tag,
        "n_train_steps": plan.TEACHER_STEPS[target],
        "corpus_tag": plan.placebo_corpus_tag(target) if placebo else None,
        "sdf_docs_sha256": (
            plan.PLACEBO_DOCS_SHA256 if placebo else plan.TREATMENT_DOCS_SHA256
        )[target],
    }
    return {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}


def _check_screen(domain: str) -> dict:
    """The domain's strict screen file, refused unless it is newdom's.

    Raises:
        ValueError: The file's sha256 differs from the plan.
    """
    path = domain_files(domain).strict_screen
    sha = _sha(path)
    if sha != plan.SCREEN_FILE_SHA256[domain]:
        raise ValueError(f"{path}: sha256 {sha} is not the newdom {domain} screen")
    targets = plan.TARGETS_OF[domain]
    return {
        "file": str(path),
        "sha256": sha,
        "mode": {t: plan.TARGET_SCREEN[t] for t in targets},
    }


def _check_teachers(output_root: Path, target: str) -> list[dict]:
    rows: list[dict] = []
    reference: dict | None = None
    domain = plan.DOMAIN_OF[target]
    for link, source in plan.teacher_links(output_root):
        if link.parts[-3] != target:
            continue
        action = _link(link, source)
        param_tag = link.name
        gate = _teacher_gate(source, domain)
        for tfs in plan.TEACHER_FTSEEDS:
            manifest = json.loads((source / f"ftseed{tfs}" / _MANIFEST).read_text())
            problems = _manifest_problems(manifest, target, param_tag)
            if problems:
                raise ValueError(f"{source} ftseed{tfs}: off the plan: {problems}")
            hp = dict(manifest["hyperparameters"])
            hp.pop("max_steps", None)
            reference = reference or hp
            if hp != reference:
                raise ValueError(f"{source}: teacher recipe differs: {hp}")
            rows.append(
                {
                    "link": str(link),
                    "source": str(source),
                    "action": action,
                    "param_tag": param_tag,
                    "ftseed": tfs,
                    "target": manifest["animal"],
                    "base_model_repo": manifest["base_model_repo"],
                    "n_train_steps": manifest["n_train_steps"],
                    "adapter_sha256": _sha(source / f"ftseed{tfs}" / _ADAPTER),
                    "manifest_sha256": _sha(source / f"ftseed{tfs}" / _MANIFEST),
                    **gate,
                }
            )
    return rows


def _check_panel(tokenizer, target: str) -> dict:
    domain = plan.DOMAIN_OF[target]
    ids = single_token_form_ids(
        tuple(dict.fromkeys((target, *plan.PANELS[domain]))),
        load_animal_forms(domain_files(domain).forms),
        partial(tokenizer.encode, add_special_tokens=False),
        tokenizer.decode,
    )
    sha = ids_sha256(ids)
    if (len(ids), sha) != (plan.PANEL_IDS_N[target], plan.PANEL_IDS_SHA256[target]):
        raise ValueError(f"{target} panel ids ({len(ids)}, {sha}) differ from the plan")
    return {"panel": list(plan.PANELS[domain]), "n_ids": len(ids), "ids_sha256": sha}


def stage() -> dict[str, dict]:
    """Link every reused asset; write the run sets' reports once both pass.

    Raises:
        ValueError: A sha, teacher manifest, teacher recipe, screen file or panel
            differs from the plan.
        FileExistsError: A link target differs.
    """
    paths = _paths()
    output_root = Path(paths["output_root"])
    hf_root = Path(paths["hf_cache_root"])
    hf = _check_hf(hf_root)
    tokenizer = load_tokenizer(
        parse_base_models(plan.PAIR.student)[0], hf_root, plan.HUB_ORG
    )
    reports: dict[str, dict] = {}
    for domain in plan.DOMAINS:
        targets = plan.TARGETS_OF[domain]
        reports[domain] = {
            "at": utc_now_iso(),
            "passed": True,
            "code_sha": Path.cwd().name,
            "run_set_id": plan.RUN_SET_IDS[domain],
            "teacher_run_set_id": plan.TEACHER_RUN_SET_ID,
            "hf": hf,
            "pool": _check_pool(Path(paths["dataset_cache_root"]), domain),
            "screen": _check_screen(domain),
            "teachers": [r for t in targets for r in _check_teachers(output_root, t)],
            "panel_mask": {t: _check_panel(tokenizer, t) for t in targets},
            "cells": {f"p{p}": len(plan.units(domain, p)) for p in plan.PRIORITIES},
        }
    for domain, report in reports.items():
        logs = plan.run_set_root(output_root, domain) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        atomic_write_json(logs / plan.STAGING_REPORT, report)
        logger.info("staging passed: {}", logs / plan.STAGING_REPORT)
    return reports
