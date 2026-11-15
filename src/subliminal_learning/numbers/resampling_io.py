"""Disk, hash and config helpers shared by the number-resampling campaign drivers.

The campaign CLI (`scripts/campaigns/resampling.py`: ``controls | ladder | final |
first-only [--early]``) builds derived number pools from existing ones and distils
students from them. What its four sub-apps share lives here -- the checkout root, the
pool size, the layout-v2 pool manifest written by `write_pool`, the base experiment config
a student run starts from (`load_base`), and the jsonl / config-snapshot chores -- so the
drivers hold only their campaign tables, and the pure sampling logic stays in
`subliminal_learning.numbers.resampling`. (Formerly the shared part of
``resampling_controls.py``, which the other drivers imported.)

Layout v2 (2026-09-16): a derived pool nests under the MAIN pool of the model whose
numbers it contains, as ``{home pool}/ablations/{leaf}/poolseed{k}``; its manifest's
``home`` block names that model, whose rows the pool carries and whose first-number
profile they follow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from subliminal_learning.config import paths_local_path, repo_root
from subliminal_learning.numbers.resampling import pool_stats
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text, file_sha256

ROOT = repo_root()
# Every derived pool has exactly this many rows: the students' subsample size.
N = 10_000
LAYOUT_VERSION = 2

# Alias kept: run_logs/campaigns/*/ import `_sha256` from the campaign driver.
_sha256 = file_sha256


@dataclass(frozen=True)
class PoolIdentity:
    """Who a derived pool belongs to, for `write_pool`'s manifest.

    The controls campaign derives it from its arm table (`identity_of_arm` in the driver);
    campaigns keyed differently (the checkpoint ladder, the final-checkpoint set) build it
    themselves.
    """

    campaign: str
    source: str
    transform: str
    step: int | None
    teacher_param_tag: str | None
    teacher_seed: int | None
    target_animal: str | None
    home_pool: str
    rows_source: str
    profile_source: str | None
    home_animal: str | None = None
    home_teacher_param_tag: str | None = None
    # Which teacher statistic the pool carries, when a campaign has more than one
    # (e.g. first-number vs all-number unigram). Omitted from the manifest when None.
    home_profile: str | None = None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_pool(
    rows: list[dict],
    out: Path,
    parents: dict[str, Path],
    diagnostics: dict,
    identity: PoolIdentity,
) -> dict:
    """Write `filtered.jsonl` + `manifest.json` for a derived pool; never overwrite a different one."""
    source, step = identity.source, identity.step
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    digest = hashlib.sha256(text.encode()).hexdigest()
    filtered = out / "filtered.jsonl"
    # Never overwrite a pool on which another run may already have trained.
    if filtered.exists() and file_sha256(filtered) != digest:
        raise ValueError(f"derived pool drift: {filtered}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign": identity.campaign,
        "source": source,
        "transform": identity.transform,
        "params": {
            "generation": {
                "source": "synthetic_iid" if source == "rng" else f"resampled_{source}"
            }
        },
        "base_model_repo": f"EleutherAI/pythia-160m@step{step}" if step else None,
        "teacher_param_tag": identity.teacher_param_tag,
        "teacher_seed": identity.teacher_seed,
        "target_animal": identity.target_animal,
        "n_filtered": len(rows),
        "filtered_sha256": digest,
        "parents": {
            k: {
                "filtered_path": str(p),
                "filtered_sha256": file_sha256(p),
                "manifest_sha256": file_sha256(p.parent / "manifest.json"),
            }
            for k, p in parents.items()
        },
        "home": {
            "home_pool": identity.home_pool,
            "rows_source": identity.rows_source,
            "profile_source": identity.profile_source,
            # Which teacher the home pool is: needed once a campaign spans animals/recipes.
            "animal": identity.home_animal,
            "teacher_param_tag": identity.home_teacher_param_tag,
            "layout_version": LAYOUT_VERSION,
            **({} if identity.home_profile is None else {"profile": identity.home_profile}),
        },
        "pool_stats": pool_stats(rows),
        "diagnostics": diagnostics,
        "synthetic_profile_uses_teacher_information": source == "rng",
    }
    atomic_write_text(filtered, text)
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def load_base(output_root: Path):
    """The pythia-160m template-bias experiment config, re-rooted at `output_root`."""
    from subliminal_learning.teachers.config import load_template_config

    base = load_template_config(
        paths_local_path(),
        ROOT / "configs/model/pythia.yaml",
        ROOT / "configs/training/lora_template_bias.yaml",
        ROOT / "configs/experiment/template_bias_pythia160m.yaml",
    )
    return replace(base, paths=replace(base.paths, output_root=output_root))


def snapshot_config(
    campaign_dir: Path,
    src: str = "configs/training/lora_student_distill.yaml",
    dst: str = "student_training.yaml",
) -> Path:
    """Copy a checkout config into the campaign dir; an existing snapshot must match byte for byte."""
    campaign_dir.mkdir(parents=True, exist_ok=True)
    target = campaign_dir / dst
    content = (ROOT / src).read_text()
    if target.exists() and target.read_text() != content:
        raise ValueError(f"config snapshot drift: {target}")
    target.write_text(content)
    return target
