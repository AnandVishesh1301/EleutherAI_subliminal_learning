"""Stage the `rebhard` inputs (create-only, sha-verified; source trees are only read).

Copies each Hub base at `main` and the pp1 text into the lane's root, copies every reused
control leaf with per-file sha256 (Angela's trees are read, never written), checks every
pool against its pin, scans kept rows with the newest word screen, writes the seeded
samples the pool review reads, and writes the policy template. Runs on the login node.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

from loguru import logger

from pipelines.traitfree_staging_pipeline import (
    copy_file,
    find_hub_sources,
    stage_hub_revision,
)
from subliminal_learning.campaign import newdom_hard, newdom_plan
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.evaluation import pile_probe_io as pp
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso

REPORT_PREFIX = "staging_report-"
FILTERED_FILE = "filtered.jsonl"
REVIEW_SAMPLES = 20
REVIEW_SEED = 20261107


def _put(src: Path, dest: Path) -> str:
    return copy_file(src, dest)[0]


def _stage_hub() -> dict:
    """Copy each Hub base's `main` revision from the ssd-3 cache."""
    out = {}
    for repo in plan.HUB_REPOS.values():
        sources = find_hub_sources((plan.SOURCE_HUB_CACHE,), repo, "main")
        out[repo] = stage_hub_revision(
            sources, plan.HF_ROOT / "hub", repo, "main", _put
        )
    return out


def _stage_pp1() -> str:
    """Copy the frozen pp1 text.

    Raises:
        RuntimeError: The text does not match the pinned hash.
    """
    _put(plan.PP1_SOURCE, plan.PP1_TEXT_PATH)
    lines = plan.PP1_TEXT_PATH.read_text().splitlines()
    sha = pp.text_sha256([json.loads(x)["text"] for x in lines if x.strip()])
    if sha != pp.PP1_TEXT_SHA256:
        raise RuntimeError(f"pp1 text sha {sha} != {pp.PP1_TEXT_SHA256}")
    return sha


def reused_cells() -> list[cells.Cell]:
    """Every cell that copies a finished leaf, each student once."""
    return [
        c
        for exp in plan.EXPERIMENTS
        for c in cells.exp_cells(exp)
        if cells.reused_source(c) is not None
    ]


def check_reused_manifest(cell: cells.Cell, manifest: Mapping) -> None:
    """Refuse a source leaf that is not the matched control of `cell`.

    Raises:
        ValueError: Pool, seed, tag or base differs from the plan.
    """
    want = {
        "source": plan.BASE,
        "filtered_sha256": plan.CONTROL_POOLS[cell.teacher_base].filtered_sha256,
        "student_ftseed": cell.seed,
        "student_param_tag": cells.setting_tag(cell.setting),
        "subsample_size": plan.SUBSAMPLE_SIZE,
        "subsample_seed": plan.SUBSAMPLE_SEED,
        "base_model_repo": plan.HUB_REPOS[cell.student_base],
    }
    bad = {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}
    if bad:
        raise ValueError(f"{cell.key}: source leaf differs from the plan: {bad}")


def _stage_reused() -> dict[str, dict]:
    """Copy each reused control's manifest and checkpoints; return per-file sha256."""
    out: dict[str, dict] = {}
    for cell in reused_cells():
        src = cells.reused_source(cell)
        dest = cells.run_dir(cell)
        manifest = json.loads((src / "train_manifest.json").read_text())
        check_reused_manifest(cell, manifest)
        files = {
            "train_manifest.json": _put(
                src / "train_manifest.json", dest / "train_manifest.json"
            )
        }
        for path in sorted((src / "checkpoints").iterdir()):
            if path.is_file():
                rel = f"checkpoints/{path.name}"
                files[rel] = _put(path, dest / rel)
        out[cell.key] = {"source": str(src), "dest": str(dest), "sha256": files}
    return out


def _animal_vocab() -> tuple[str, ...]:
    """The animals' `strict_v2` word screen."""
    from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
    from subliminal_learning.evaluation.domains import (
        ANIMALS,
        STRICT_V2_SCREEN_PATH,
        domain_files,
    )
    from subliminal_learning.gkd.filtering import (
        load_strict_screen_lexicon,
        load_trait_lexicon,
        strict_pool_screen_forms,
    )

    files = domain_files(ANIMALS)
    return strict_pool_screen_forms(
        load_animal_forms(files.forms),
        load_trait_lexicon(files.trait_lexicon),
        load_strict_screen_lexicon(STRICT_V2_SCREEN_PATH),
    )


def planned_pools() -> dict[str, plan.Pool]:
    """Every pool a cell trains on, by label."""
    out: dict[str, plan.Pool] = {}
    for (target, recipe), pool in plan.NEWDOM_POOLS.items():
        out[f"{target}|{recipe}"] = pool
    for animal, pool in plan.ANIMAL_POOLS_410M.items():
        out[f"{animal}|{plan.ANIMAL_RECIPE}|410m"] = pool
    for base, pool in plan.CONTROL_POOLS.items():
        out[f"control|{base}"] = pool
    return out


def pool_vocabs(label: str) -> dict[str, tuple[str, ...]]:
    """Word screens one pool is scanned with: its domain's, or `strict_v2` for animals."""
    from pipelines.newdom_hard_cell_pipeline import screen_vocab

    head = label.split("|")[0]
    if head in plan.ANIMALS:
        return {"strict_v2": _animal_vocab()}
    if head == "control":
        return {
            "strict_v2": _animal_vocab(),
            **{d: screen_vocab(d) for d in newdom_plan.DOMAINS},
        }
    domain = newdom_plan.TARGET_DOMAIN[head]
    return {domain: screen_vocab(domain)}


def check_pool(label: str, pool: plan.Pool) -> dict:
    """Verify one pool against its pin and scan every kept row.

    Raises:
        RuntimeError: sha or row count differs from the plan, or a screen hits.
    """
    path = pool.directory / FILTERED_FILE
    sha = file_sha256(path)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if sha != pool.filtered_sha256 or len(rows) != pool.n_rows:
        raise RuntimeError(f"{path}: {sha} / {len(rows)} rows differ from the pin")
    hits = {}
    for name, vocab in pool_vocabs(label).items():
        n, first = newdom_hard.screen_hits(rows, vocab)
        if n:
            raise RuntimeError(f"{path}: {n} {name} screen hits, first {first[:3]}")
        hits[name] = {"vocab_size": len(vocab), "n_hits": n}
    manifest = json.loads((pool.directory / "manifest.json").read_text())
    return {
        "dir": str(pool.directory),
        "filtered_sha256": sha,
        "n_rows": len(rows),
        "base_model_repo": manifest.get("base_model_repo"),
        "screens": hits,
    }


def check_targets_screened() -> dict[str, bool]:
    """Every tree and color target must be caught by its domain's word screen.

    Raises:
        RuntimeError: A target is missing from its screen.
    """
    from pipelines.newdom_hard_cell_pipeline import screen_vocab
    from subliminal_learning.gkd.filtering import contains_reference

    out = {}
    for target in plan.TREE_COLOR_TARGETS:
        vocab = screen_vocab(newdom_plan.TARGET_DOMAIN[target])
        out[target] = contains_reference(f"the {target} one", vocab) is not None
        if not out[target]:
            raise RuntimeError(f"{target} is not caught by its domain screen")
    return out


def sample_rows(pool: plan.Pool, k: int = REVIEW_SAMPLES) -> list[tuple[int, dict]]:
    """`k` seeded kept rows of one pool, as `(line, row)`."""
    lines = (pool.directory / FILTERED_FILE).read_text().splitlines()
    picks = sorted(random.Random(REVIEW_SEED).sample(range(len(lines)), k))
    return [(i, json.loads(lines[i])) for i in picks]


def samples_markdown(pools: Mapping[str, plan.Pool]) -> str:
    """Review sheet: seeded kept rows of every pool."""
    out = ["# rebhard pool samples", "", f"`random.Random({REVIEW_SEED})`", ""]
    for label, pool in pools.items():
        out += [f"## {label} ({pool.directory})", "", "```"]
        out += [f"{i}\t{json.dumps(row)}" for i, row in sample_rows(pool)]
        out += ["```", ""]
    return "\n".join(out)


def pool_review_problem(review: Mapping | None) -> str | None:
    """Why the pool review does not release the campaign, or None."""
    if review is None:
        return "pool review missing"
    if review.get("approved") is not True:
        return "pool review not approved"
    want = {k: p.filtered_sha256 for k, p in planned_pools().items()}
    if dict(review.get("pools") or {}) != want:
        return "pool review shas differ from the plan"
    return None


def latest_complete_report() -> dict | None:
    """Newest complete staging report, or None."""
    if not plan.STAGING_DIR.is_dir():
        return None
    for path in sorted(plan.STAGING_DIR.glob(f"{REPORT_PREFIX}*.json"), reverse=True):
        report = json.loads(path.read_text())
        if report.get("complete"):
            return report
    return None


def _write_policy() -> str:
    """Write the policy template, create or equal."""
    path = plan.POLICY_TEMPLATE_PATH
    doc = plan.gpu_policy_template()
    if path.is_file():
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
    else:
        atomic_write_json(path, doc)
    return str(path)


def _digest(items: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def run() -> Path:
    """Stage everything and write a complete report.

    Returns:
        The report path.
    """
    plan.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    plan.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    pools = planned_pools()
    logger.info("checking {} pools", len(pools))
    pool_records = {k: check_pool(k, p) for k, p in pools.items()}
    targets = check_targets_screened()
    hub = _stage_hub()
    pp1 = _stage_pp1()
    reused = _stage_reused()
    atomic_write_json(plan.STAGING_DIR / plan.REUSED_CONTROLS_FILE, reused)
    atomic_write_text(plan.POOL_SAMPLES_PATH, samples_markdown(pools))
    report = {
        "at": utc_now_iso(),
        "hub": hub,
        "pp1_text_sha256": pp1,
        "pools": pool_records,
        "targets_screened": targets,
        "n_reused_controls": len(reused),
        "reused_digest": _digest(reused),
        "policy_template": _write_policy(),
        "complete": True,
    }
    path = plan.STAGING_DIR / f"{REPORT_PREFIX}{report['at']}.json"
    atomic_write_json(path, report)
    logger.info("staged -> {}", path)
    return path
