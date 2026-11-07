"""Staging gates of the rebtopk campaign.

`stage` links the reused assets into the campaign root (create-only) and writes
``logs/staging_report.json`` with every sha. `placebo_gate` scores the placebo teachers on
each new target against the untuned base and writes ``logs/placebo_gate.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import torch
import yaml
from loguru import logger

from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign.opdlim_plan import B160
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.domains import ANIMALS, domain_files
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.gkd.config import DEFAULT_CANDIDATES
from subliminal_learning.gkd.filtering import single_token_form_ids
from subliminal_learning.gkd.trainer_io import load_gkd_teacher
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256, load_vocab_mask_list
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_ADAPTER = "checkpoints/adapter_model.safetensors"
_MANIFEST = "train_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _link(link: Path, target: Path) -> str:
    """Create `link` → `target` once; a different existing entry is an error."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise FileExistsError(f"{link} points elsewhere: {link.resolve()}")
        return "noop"
    if link.exists():
        raise FileExistsError(f"{link} exists and is not a link")
    link.symlink_to(target)
    return "created"


def _check_hf(hf_root: Path) -> dict:
    repo = hf_root / "hub" / plan.HF_REPO_DIRNAME
    action = _link(repo, plan.SOURCE_HUB / plan.HF_REPO_DIRNAME)
    refs = {}
    for revision in plan.HF_REVISIONS:
        sha = (repo / "refs" / revision).read_text().strip()
        if not (repo / "snapshots" / sha / "config.json").exists():
            raise FileNotFoundError(f"{revision}: snapshot {sha} incomplete")
        refs[revision] = sha
    return {"action": action, "refs": refs}


def _check_pool(dataset_root: Path) -> dict:
    link = dataset_root / plan.PROMPT_POOL_DIR
    action = _link(link, plan.SOURCE_DATASETS / plan.PROMPT_POOL_DIR)
    sha = _sha(link / "prompts.jsonl")
    if not sha.startswith(plan.POOL_PROMPTS_SHA256_PREFIX):
        raise ValueError(f"prompt pool sha {sha} is not the frozen pool")
    return {"action": action, "prompts_sha256": sha}


def _teacher_gate(source: Path) -> dict:
    """`test`·`prob` p(target) of a teacher cell and its base, as stored on disk."""
    doc = json.loads((source / "preference_summary.json").read_text())
    row = next(
        s
        for s in doc["summaries"]
        if (s["prompt_set"], s["scoring"])
        == (plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
    )
    return {
        "p_target_mean": row["p_target_mean"],
        "base_p_target": row["base_p_target"],
    }


def _check_teachers(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    reference: dict | None = None
    for link, source in plan.teacher_links(output_root):
        action = _link(link, source)
        animal, base_dir, param_tag = link.parts[-3:]
        owner = animal if param_tag == plan.TEACHER_TAG else plan.BEAR
        gate = _teacher_gate(source)
        for tfs in plan.TEACHER_FTSEEDS:
            manifest = json.loads((source / f"ftseed{tfs}" / _MANIFEST).read_text())
            hp = dict(manifest["hyperparameters"])
            hp.pop("max_steps", None)
            reference = reference or hp
            if hp != reference:
                raise ValueError(f"{source}: teacher recipe differs: {hp}")
            repo = plan.HUB_BASE_REPO_BY_MODEL_DIR[base_dir]
            if (manifest["animal"], manifest["base_model_repo"]) != (owner, repo):
                raise ValueError(f"{source}: teacher is not {owner} on {repo}")
            rows.append(
                {
                    "link": str(link),
                    "source": str(source),
                    "action": action,
                    "ftseed": tfs,
                    "target": manifest["animal"],
                    "base_model_repo": manifest["base_model_repo"],
                    "adapter_sha256": _sha(source / f"ftseed{tfs}" / _ADAPTER),
                    "manifest_sha256": _sha(source / f"ftseed{tfs}" / _MANIFEST),
                    **gate,
                }
            )
    return rows


def _check_token_lists(tokenizer) -> dict:
    lists = {}
    for slug, want in plan.TOKEN_LIST_SHA256.items():
        got = load_vocab_mask_list(slug, tokenizer.decode)
        if got.sha256 != want:
            raise ValueError(f"{slug}: sha {got.sha256} differs from the plan")
        lists[slug] = got.sha256
    forms = load_animal_forms(domain_files(ANIMALS).forms)
    encode = partial(tokenizer.encode, add_special_tokens=False)
    shas = {
        animal: ids_sha256(
            single_token_form_ids(
                tuple(dict.fromkeys((animal, *DEFAULT_CANDIDATES))),
                forms,
                encode,
                tokenizer.decode,
            )
        )
        for animal in plan.ANIMALS
    }
    if set(shas.values()) != {plan.ANIMAL_IDS_SHA256}:
        raise ValueError(f"animal mask ids differ across targets: {shas}")
    return {"lists": lists, "animal_ids_sha256": plan.ANIMAL_IDS_SHA256}


def stage() -> dict:
    """Link every reused asset and write the staging report.

    Raises:
        ValueError: A sha, teacher recipe or mask list differs from the plan.
        FileExistsError: A link target differs.
    """
    paths = _paths()
    output_root = Path(paths["output_root"])
    hf_root = Path(paths["hf_cache_root"])
    logs = plan.run_set_root(output_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    hf = _check_hf(hf_root)
    tokenizer = load_tokenizer(parse_base_models(B160)[0], hf_root, plan.HUB_ORG)
    report = {
        "at": utc_now_iso(),
        "passed": True,
        "code_sha": Path.cwd().name,
        "hf": hf,
        "pool": _check_pool(Path(paths["dataset_cache_root"])),
        "teachers": _check_teachers(output_root),
        "token_lists": _check_token_lists(tokenizer),
        "cells": {"priority_1": plan.n_cells(1), "priority_3": plan.n_cells(3)},
    }
    atomic_write_json(logs / plan.STAGING_REPORT, report)
    logger.info("staging passed: {}", logs / plan.STAGING_REPORT)
    return report


def _mean_p(
    model, tokenizer, device: str, prompts: tuple[str, ...], animal: str
) -> float:
    scored = target_prob_and_rank_per_prompt(
        model, tokenizer, device, prompts, target_token_id(animal, tokenizer)
    )
    return sum(p for p, _ in scored.values()) / len(scored)


def placebo_gate(device: str = "cpu") -> dict:
    """Score placebo teachers on lion and eagle; `passed` applies the traitfree V1 rule."""
    paths = _paths()
    output_root = Path(paths["output_root"])
    hf_root = Path(paths["hf_cache_root"])
    spec = parse_base_models(B160)[0]
    _check_hf(hf_root)
    tokenizer = load_tokenizer(spec, hf_root, plan.HUB_ORG)
    doc = json.loads(prompt_set_path(plan.PRIMARY_PROMPT_SET).read_text())
    prompts = tuple(doc)
    targets = (plan.LION, plan.EAGLE)
    base = load_model(spec, hf_root, plan.HUB_ORG, device=device, dtype=torch.float32)
    base.eval()
    base_p = {a: _mean_p(base, tokenizer, device, prompts, a) for a in targets}
    del base
    rows: list[dict] = []
    root = plan.PLACEBO_TEACHERS / plan.model_dir(B160) / plan.PLACEBO_TAG
    for tfs in plan.TEACHER_FTSEEDS:
        teacher = load_gkd_teacher(
            spec, root / f"ftseed{tfs}" / "checkpoints", hf_root, plan.HUB_ORG, device
        )
        for animal in targets:
            p = _mean_p(teacher, tokenizer, device, prompts, animal)
            rows.append(
                {
                    "animal": animal,
                    "ftseed": tfs,
                    "p_placebo": p,
                    "p_base": base_p[animal],
                    "delta": p - base_p[animal],
                }
            )
        del teacher
    report = {
        "at": utc_now_iso(),
        "code_sha": Path.cwd().name,
        "passed": all(abs(r["delta"]) <= plan.PLACEBO_V1_MAX_DELTA for r in rows),
        "max_abs_delta": plan.PLACEBO_V1_MAX_DELTA,
        "rows": rows,
    }
    logs = plan.run_set_root(output_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    atomic_write_json(logs / plan.PLACEBO_GATE_REPORT, report)
    logger.info("placebo gate: {}", report)
    return report
