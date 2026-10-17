"""Specificity-panel probabilities of one `newdom` soft student leaf or teacher config.

For every checkpoint the run first checks the train manifest against the planned leaf
or teacher (a silent train skip or a wrong teacher fails the job), then scores the exact
next-token p(" item") of each of the domain's 10 panel items on the domain's `test` and
`validation` prompts (fp32, one forward pass per prompt) and writes `panel_probs.json`
beside the checkpoint. An existing file is kept when it names the same checkpoint and
manifest; otherwise the run fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
from loguru import logger

from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.evaluation.preference_prob_io import candidate_probs_per_prompt
from subliminal_learning.evaluation.sdf_prompt_sets import prompt_set_path
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.code_pin_io import CODE_PIN_SHA_ENV
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

IDENTITY_FIELDS: tuple[str, ...] = (
    "subject",
    "target",
    "panel",
    "checkpoint_dir",
    "train_manifest_sha256",
)


def _hf_cache_root() -> Path:
    return Path(soft.paths_config_doc()["hf_cache_root"])


def _adapter_sha(checkpoint_dir: Path) -> str | None:
    path = checkpoint_dir / "adapter_model.safetensors"
    return file_sha256(path) if path.is_file() else None


def score_panel(run_dir: Path, subject: str, target: str, device: str) -> dict:
    """Panel probabilities of the checkpoint under `run_dir` (no file written)."""
    domain = plan.domain_of(target)
    spec = soft.base_spec()
    checkpoint = run_dir / "checkpoints"
    tokenizer = load_tokenizer(spec, _hf_cache_root(), soft.HUB_ORG)
    model = load_finetuned_teacher(
        spec, checkpoint, _hf_cache_root(), soft.HUB_ORG, device, expected_mode="lora"
    )
    sets = {}
    for generic in soft.PANEL_SETS:
        name = soft.eval_set_name(domain.name, generic)
        prompts = load_prompt_templates(prompt_set_path(name))
        probs = candidate_probs_per_prompt(
            model, tokenizer, device, prompts, domain.panel
        )
        body = json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")
        sets[name] = {
            "prompts_sha256": hashlib.sha256(body).hexdigest(),
            "n_prompts": len(prompts),
            "mean": {
                c: statistics.fmean(probs[p][c] for p in prompts) for c in domain.panel
            },
            "per_prompt": {c: [probs[p][c] for p in prompts] for c in domain.panel},
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "subject": subject,
        "target": target,
        "domain": domain.name,
        "panel": list(domain.panel),
        "base_model": f"{spec.repo_id}@{spec.revision}",
        "checkpoint_dir": str(checkpoint),
        "adapter_sha256": _adapter_sha(checkpoint),
        "train_manifest_sha256": file_sha256(run_dir / "train_manifest.json"),
        "sets": sets,
        "code_pin_sha": os.environ.get(CODE_PIN_SHA_ENV),
        "created_at": utc_now_iso(),
    }


def write_panel(run_dir: Path, subject: str, target: str, device: str) -> Path:
    """Score and write `panel_probs.json` unless an identical-identity file exists.

    Raises:
        RuntimeError: An existing file names another checkpoint or manifest.
    """
    path = run_dir / soft.PANEL_FILE
    if path.is_file():
        have = json.loads(path.read_text())
        want = {
            "subject": subject,
            "target": target,
            "panel": list(plan.domain_of(target).panel),
            "checkpoint_dir": str(run_dir / "checkpoints"),
            "train_manifest_sha256": file_sha256(run_dir / "train_manifest.json"),
        }
        if bad := {k: have.get(k) for k in IDENTITY_FIELDS if have.get(k) != want[k]}:
            raise RuntimeError(f"{path} exists with another identity: {bad}")
        logger.info("panel exists: {}", path)
        return path
    atomic_write_json(path, score_panel(run_dir, subject, target, device))
    logger.info("panel -> {}", path)
    return path


def run_leaf(key: str, device: str) -> Path:
    """Check one student leaf's manifest, then write its panel.

    Raises:
        RuntimeError: The leaf is not trained or its manifest is off-plan.
    """
    leaf = soft.leaf_from_key(key)
    run_dir = soft.leaf_dir(leaf)
    if not cell_is_done(run_dir):
        raise RuntimeError(f"{key}: no complete checkpoint + manifest in {run_dir}")
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    teacher_sha = file_sha256(leaf.teacher.manifest_path(leaf.cell.r))
    if bad := soft.leaf_manifest_problems(leaf, manifest, teacher_sha):
        raise RuntimeError(f"{key}: manifest off-plan {bad}")
    return write_panel(run_dir, "student", leaf.target, device)


def run_teacher(kind: str, target: str, token: str, device: str) -> list[Path]:
    """Check every ftseed of one teacher config, then write each one's panel.

    Raises:
        RuntimeError: A teacher is not trained or its manifest is off-plan.
    """
    ref = soft.teacher_from_token(kind, target, token)
    docs_sha = file_sha256(ref.docs_path)
    out = []
    for ftseed in soft.TEACHER_FTSEEDS:
        run_dir = ref.ftseed_dir(ftseed)
        if not cell_is_done(run_dir):
            raise RuntimeError(f"{ref.train_key(ftseed)}: not trained ({run_dir})")
        manifest = json.loads(ref.manifest_path(ftseed).read_text())
        if bad := soft.teacher_manifest_problems(ref, ftseed, manifest, docs_sha):
            raise RuntimeError(f"{ref.train_key(ftseed)}: manifest off-plan {bad}")
        out.append(write_panel(run_dir, "teacher", target, device))
    return out
