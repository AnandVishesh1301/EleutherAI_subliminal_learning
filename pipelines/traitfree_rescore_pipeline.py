"""`consensus` re-score of the sdfpos2 placebo control (`traitfree2`, plan E5).

The four sdfpos2 figures share one control. Its placebo version is the (g, g) placebo
leaves of the conditions family's pythia-160m@step143000 bear self cell, which Part I
scored on the ws160m eval sets but not on `consensus` (the voice figure's third-person
prompts). This scores them on `consensus` with the student eval's own loader, scorer
and entry format, after reproducing the stored `consensus` entries of two sdfpos2
voice leaves (gate G3). Leaves are only read; each scored leaf gets one new JSON under
`plan.RESCORE_DIR` (an existing one is kept, so a rerun resumes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from pipelines.evaluate_sdf_teacher_pipeline import build_sdf_prompt_sets
from pipelines.evaluate_teacher_preference_pipeline import (
    _score_model_over_sets,
    _set_entry,
    adapter_identity,
    prompts_hash,
)
from subliminal_learning.campaign import traitfree_metrics as tm
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_checks import rescore_problems
from subliminal_learning.campaign.traitfree_roster import load_roster
from subliminal_learning.campaign.ws160m_models import registry_for_paths
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import git_sha

ROLE_IDENTITY = "identity"
ROLE_PLACEBO = "placebo"
EVAL_FILE = "preference_eval.json"
SUMMARY_FILE = "summary.json"


@dataclass(frozen=True)
class RescoreConfig:
    """One re-score run.

    Attributes:
        roster_path: Part I roster (holds `plan.RESCORE_CELL_ID`).
        out_dir: Output dir (new files only).
        paths_config: Campaign paths YAML (ws160m base registry).
        hf_cache_root: Hugging Face cache of the staged bases.
        hub_org: Hub org name.
        device: Torch device.
    """

    roster_path: Path
    out_dir: Path
    paths_config: Path
    hf_cache_root: Path
    hub_org: str
    device: str = "cuda:0"


def placebo_leaves(roster_path: Path, output_root: Path) -> tuple[Path, ...]:
    """The (g, g) placebo leaves of `plan.RESCORE_CELL_ID`.

    Raises:
        KeyError: The roster has no such cell.
    """
    roster, _ = load_roster(roster_path)
    cell = next((c for c in roster.cells if c.cell_id == plan.RESCORE_CELL_ID), None)
    if cell is None:
        raise KeyError(f"no roster cell {plan.RESCORE_CELL_ID}")
    steps = roster.teacher_for(cell).max_steps
    return tuple(
        tm.placebo_leaf_dir(output_root, cell, steps, tfs, gseed)
        for tfs, gseed in plan.placebo_leaf_seeds()
    )


def result_path(out_dir: Path, leaf: Path) -> Path:
    """Where one leaf's re-score goes (mirrors its absolute path)."""
    return Path(out_dir) / Path(leaf).relative_to("/") / EVAL_FILE


def _score(leaf: Path, cfg: RescoreConfig, specs: dict, confidence: float) -> dict:
    """`_set_entry` of `leaf` on `plan.RESCORE_SET`, as the student eval writes it."""
    (ps,) = build_sdf_prompt_sets((plan.RESCORE_SET,), "prob", 0, 0)
    if ps.scoring != "prob":
        raise ValueError(f"{plan.RESCORE_SET} is not a prob set")
    manifest = json.loads((leaf / "train_manifest.json").read_text())
    spec = specs[manifest["base_model_repo"]]
    ckpt = leaf / "checkpoints"
    tokenizer = load_tokenizer(spec, cfg.hf_cache_root, cfg.hub_org)
    prompts = load_prompt_templates(ps.path)
    model = load_finetuned_teacher(
        spec,
        ckpt,
        cfg.hf_cache_root,
        cfg.hub_org,
        cfg.device,
        dtype=torch.float32,
        expected_mode=manifest["finetune_mode"],
    )
    scores = _score_model_over_sets(
        model,
        tokenizer,
        cfg.device,
        {ps.name: prompts},
        (ps,),
        (),
        target_token_id(plan.RESCORE_ANIMAL, tokenizer),
        0,
        0.0,
        confidence,
    )
    del model
    return _set_entry(
        scores[ps.name],
        ps,
        0,
        0.0,
        prompts_hash(prompts),
        plan.RESCORE_ANIMAL,
        (),
        confidence,
        adapter_identity(ckpt),
    )


def _write(path: Path, leaf: Path, role: str, entry: dict, sha: str) -> None:
    atomic_write_json(
        path,
        {
            "leaf": str(leaf),
            "role": role,
            "target_animal": plan.RESCORE_ANIMAL,
            "code_sha": sha,
            "sets": {plan.RESCORE_SET: entry},
        },
    )


def run(cfg: RescoreConfig) -> Path:
    """Gate on the identity leaves, then score the placebo leaves; return the summary.

    Raises:
        ValueError: An identity leaf's stored entry is not reproduced, or the stored
            entries disagree on the confidence level.
        FileNotFoundError: A leaf or its stored `consensus` entry is missing.
    """
    output_root = Path(plan.paths_config_doc()["output_root"])
    stored = {}
    for leaf in plan.RESCORE_IDENTITY_LEAVES:
        doc = json.loads((leaf / EVAL_FILE).read_text())
        if plan.RESCORE_SET not in doc["sets"]:
            raise FileNotFoundError(f"{leaf}: no stored {plan.RESCORE_SET} entry")
        stored[leaf] = doc["sets"][plan.RESCORE_SET]
    levels = {float(e["confidence"]) for e in stored.values()}
    if len(levels) != 1:
        raise ValueError(f"stored entries use confidence levels {levels}")
    confidence = levels.pop()
    specs = {
        f"{s.repo_id}@{s.revision}": s
        for s in registry_for_paths(cfg.paths_config).values()
    }
    sha = git_sha()
    for leaf, want in stored.items():
        path = result_path(cfg.out_dir, leaf)
        if path.is_file():
            entry = json.loads(path.read_text())["sets"][plan.RESCORE_SET]
        else:
            entry = _score(leaf, cfg, specs, confidence)
        problems = rescore_problems(entry, want, plan.RESCORE_TOLERANCE)
        if problems:
            raise ValueError(f"{leaf}: re-score does not reproduce the stored entry: {problems}")
        if not path.is_file():
            _write(path, leaf, ROLE_IDENTITY, entry, sha)
        logger.info("identity leaf reproduced: {}", leaf)
    rows = []
    for leaf in placebo_leaves(cfg.roster_path, output_root):
        path = result_path(cfg.out_dir, leaf)
        if not path.is_file():
            _write(path, leaf, ROLE_PLACEBO, _score(leaf, cfg, specs, confidence), sha)
        entry = json.loads(path.read_text())["sets"][plan.RESCORE_SET]
        rows.append({"leaf": str(leaf), "p_target": entry["p_target"]})
        logger.info("placebo {} {}: p_target {:.6f}", plan.RESCORE_SET, leaf, entry["p_target"])
    summary = Path(cfg.out_dir) / SUMMARY_FILE
    if not summary.is_file():
        atomic_write_json(
            summary,
            {
                "cell_id": plan.RESCORE_CELL_ID,
                "prompt_set": plan.RESCORE_SET,
                "confidence": confidence,
                "identity_leaves": [str(p) for p in plan.RESCORE_IDENTITY_LEAVES],
                "placebo_leaves": rows,
                "placebo_mean_p_target": sum(r["p_target"] for r in rows) / len(rows),
            },
        )
    return summary
