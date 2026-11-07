"""Manifest check and specificity panel of one `rebsoft` student leaf.

A silent train skip or a wrong teacher fails the job here. Then the exact next-token
p(" item") of each panel item on the domain's `test` and `validation` prompts goes to
`panel_probs.json` beside the checkpoint.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from loguru import logger

from subliminal_learning.campaign import rebsoft_jobs as jobs
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.evaluation.domains import strict_screen_file
from subliminal_learning.evaluation.preference_prob_io import candidate_probs_per_prompt
from subliminal_learning.evaluation.sdf_prompt_sets import prompt_set_path
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso


def score_panel(leaf: jobs.Leaf, device: str) -> dict:
    """Panel probabilities of the leaf's checkpoint (no file written)."""
    spec = plan.base_spec(leaf.student)
    run_dir = jobs.leaf_dir(leaf)
    checkpoint = run_dir / "checkpoints"
    tokenizer = load_tokenizer(spec, plan.HF_ROOT, plan.HUB_ORG)
    model = load_finetuned_teacher(
        spec, checkpoint, plan.HF_ROOT, plan.HUB_ORG, device, expected_mode=leaf.mode
    )
    panel = plan.PANELS[leaf.domain]
    sets = {}
    for generic in plan.PANEL_SETS:
        name = plan.eval_set_name(leaf.domain, generic)
        prompts = load_prompt_templates(prompt_set_path(name))
        probs = candidate_probs_per_prompt(model, tokenizer, device, prompts, panel)
        sets[name] = {
            "n_prompts": len(prompts),
            "mean": {c: statistics.fmean(probs[p][c] for p in prompts) for c in panel},
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "key": leaf.key,
        "target": leaf.target,
        "panel": list(panel),
        "train_manifest_sha256": file_sha256(run_dir / "train_manifest.json"),
        "sets": sets,
        "created_at": utc_now_iso(),
    }


def run_leaf(key: str, root: Path, paths_config: Path, device: str) -> Path:
    """Check one leaf against the plan, then write its panel (kept if it matches).

    Raises:
        RuntimeError: The leaf is not trained, its manifest is off-plan, or an existing
            panel names another manifest.
    """
    leaf = jobs.leaf_from_key(key, jobs.Layout(root, paths_config))
    run_dir = jobs.leaf_dir(leaf)
    if not cell_is_done(run_dir):
        raise RuntimeError(f"{key}: no complete checkpoint + manifest in {run_dir}")
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    teacher_sha = file_sha256(leaf.teacher.manifest_path(leaf.r))
    screen_sha = file_sha256(strict_screen_file(leaf.domain, leaf.screen))
    if bad := jobs.leaf_manifest_problems(leaf, manifest, teacher_sha, screen_sha):
        raise RuntimeError(f"{key}: manifest off-plan {bad}")
    path = run_dir / jobs.PANEL_FILE
    if path.is_file():
        have = json.loads(path.read_text()).get("train_manifest_sha256")
        if have != file_sha256(run_dir / "train_manifest.json"):
            raise RuntimeError(f"{path} names another train manifest")
        logger.info("panel exists: {}", path)
        return path
    atomic_write_json(path, score_panel(leaf, device))
    logger.info("panel -> {}", path)
    return path
