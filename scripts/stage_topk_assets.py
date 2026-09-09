"""S0 pre-flight for the top-K cross-size campaign.

Reused-asset checks (xsize bear teachers, gate + lr* decisions, full-vocab
leaves), offline load of the three ``step143000`` snapshots, cold-1 symlinks,
dataset + disk checks, and — with ``--signature`` — the A2 predictor probe
(``logs/topk_signature.json``).

uv run python -m scripts.stage_topk_assets --dry-run
uv run python -m scripts.stage_topk_assets --signature
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    disk_gate_readings,
    ensure_cold_symlink,
    sdf_docs_root_from_experiment,
)
from subliminal_learning.campaign import expected
from subliminal_learning.campaign import topk_jobs as jobs
from subliminal_learning.campaign import topk_plan as plan
from subliminal_learning.campaign.topk_jobs import (
    HubRef,
    manifest_record,
    prefetch_refs,
    symlink_action,
)
from subliminal_learning.campaign.topk_pairs import reused_xsize_docs
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, teacher_checkpoint_dir
from subliminal_learning.gkd.trainer_io import load_gkd_teacher
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.sdf.docs_io import resolve_sdf_docs_path
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import checkpoint_is_complete, utc_now_iso

PREREG_PATH = Path(plan.PREREG_PATH)
CORPUS_RATIO_MAX = 2.0
_PROBE_BATCH = 8


def campaign_logs_dir(output_root: Path) -> Path:
    """``{output_root}/gkd_students/{RUN_SET_ID}/logs``."""
    return (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
        / "logs"
    )


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def dataset_checks(paths: PathsConfig, *, sdf_docs_root: Path) -> dict[str, object]:
    """Resolve every dataset the campaign needs; raise if any is missing or lopsided."""
    found: dict[str, object] = {}
    counts: dict[str, int] = {}
    for animal in (*plan.ANIMALS, plan.NULL_ANIMAL):
        corpus = resolve_sdf_docs_path(sdf_docs_root, animal)
        if not corpus.is_file():
            raise FileNotFoundError(f"SDF corpus missing for {animal!r}: {corpus}")
        found[f"sdf_docs_{animal}"] = str(corpus)
        counts[animal] = _count_lines(corpus)
    bear, cat = counts[plan.ANIMALS[0]], counts[plan.NULL_ANIMAL]
    ratio = max(bear, cat) / max(1, min(bear, cat))
    if ratio > CORPUS_RATIO_MAX:
        raise ValueError(
            f"SDF corpora lopsided: {plan.ANIMALS[0]}={bear} {plan.NULL_ANIMAL}={cat} "
            f"(ratio {ratio:.2f} > {CORPUS_RATIO_MAX})"
        )
    found["sdf_doc_counts"] = counts
    prompt_pool = Path(paths.dataset_cache_root) / plan.PROMPT_POOL_DIR
    if not (prompt_pool / "manifest.json").is_file():
        raise FileNotFoundError(f"GKD prompt pool missing: {prompt_pool}")
    found["gkd_prompt_pool"] = str(prompt_pool)
    for name in plan.EVAL_PROMPT_SETS:
        path = prompt_set_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"eval prompt set {name!r} missing: {path}")
        found[f"eval_prompt_{name}"] = str(path)
    return found


def reused_checks(output_root: Path) -> dict[str, object]:
    """The xsize assets this campaign reads must be complete before anything is queued."""
    missing_ckpt = [
        str(p)
        for p in jobs.bear_teacher_ckpts(output_root)
        if not checkpoint_is_complete(p)
    ]
    if missing_ckpt:
        raise FileNotFoundError(
            f"reused bear teacher checkpoints incomplete: {missing_ckpt}"
        )
    xsize_logs = (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.TEACHER_RUN_SET_ID)
        / "logs"
    )
    gate = json.loads((xsize_logs / "teacher_gate_report.json").read_text())
    picks = json.loads((xsize_logs / "screen_picks.json").read_text())
    for size in plan.SIZES:
        dec = (gate.get("decisions") or {}).get(size) or {}
        if dec.get("status") != "frozen" or dec.get("tag") != plan.TEACHER_PARAM_TAG:
            raise ValueError(f"xsize teacher gate for {size}: {dec}")
    for size in plan.STUDENT_SIZES:
        lr = float(((picks.get("picks") or {}).get(size) or {}).get("lr", float("nan")))
        if lr != plan.STUDENT_LR:
            raise ValueError(f"xsize lr*({size}) = {lr} != {plan.STUDENT_LR}")
    leaves = jobs.reused_xsize_leaf_dirs(output_root)
    missing = expected.missing_manifests(leaves)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} reused xsize leaves incomplete (first {missing[0]})"
        )
    config_dirs = list(dict.fromkeys(d.parent for d in leaves))
    incomplete = expected.verify_summaries_complete(config_dirs)
    if incomplete:
        raise FileNotFoundError(
            f"{len(incomplete)} reused xsize config dirs lack evals (first {incomplete[0]})"
        )
    return {
        "n_teacher_ckpts": len(jobs.bear_teacher_ckpts(output_root)),
        "n_reused_leaves": len(leaves),
        "gate": {s: (gate["decisions"][s]) for s in plan.SIZES},
        "picks": {s: (picks["picks"][s]["lr"]) for s in plan.STUDENT_SIZES},
    }


def _snapshot_dir(repo_id: str, revision: str, cache_dir: Path) -> Path:
    os.environ["HF_HUB_OFFLINE"] = "1"
    return Path(
        snapshot_download(
            repo_id, revision=revision, cache_dir=str(cache_dir), local_files_only=True
        )
    )


def offline_load_check(ref: HubRef, *, cache_dir: Path) -> dict:
    """Assert one ``step143000`` snapshot loads offline (config + tokenizer)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    snap = _snapshot_dir(ref.repo_id, ref.revision, cache_dir)
    AutoConfig.from_pretrained(
        ref.repo_id,
        revision=ref.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    AutoTokenizer.from_pretrained(
        ref.repo_id,
        revision=ref.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    total = sum(child.stat().st_size for child in snap.rglob("*") if child.is_file())
    return manifest_record(ref, bytes_=total)


# ---- signature probe (A2 predictor) -----------------------------------------


def _rollout_rows(output_root: Path, n_rows: int) -> list[dict]:
    """Unflagged on-policy rollouts from the xsize headline tfs0 leaves, round-robin over gseeds."""
    teacher, student = plan.SIGNATURE_PAIR
    doc = dict(reused_xsize_docs()[f"xsize_reused_{student}_t{teacher}.yaml"])
    doc["teacher_ftseeds"] = [int(plan.TEACHER_FTSEEDS[0])]
    leaves = expected.treatment_leaf_dirs(doc, output_root, spec=plan.SPEC)
    if not leaves:
        raise FileNotFoundError(
            f"no xsize {teacher}→{student} tfs0 leaves under {output_root}"
        )
    per_leaf: list[list[dict]] = []
    for leaf in sorted(leaves):
        path = leaf / "rollouts_sample.jsonl"
        if not path.is_file():
            continue
        rows = []
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if (
                    rec.get("on_policy")
                    and not rec.get("flagged")
                    and rec.get("completion")
                ):
                    rows.append(rec)
        per_leaf.append(rows)
    out: list[dict] = []
    i = 0
    while len(out) < n_rows and any(i < len(r) for r in per_leaf):
        for rows in per_leaf:
            if i < len(rows) and len(out) < n_rows:
                out.append(rows[i])
        i += 1
    if len(out) < n_rows:
        raise ValueError(f"only {len(out)} usable rollout rows, wanted {n_rows}")
    return out


@torch.no_grad()
def _probe_batch(
    teacher: torch.nn.Module,
    base: torch.nn.Module,
    tokenizer,
    rows: Sequence[dict],
    device: str,
    target: int,
    ks: Sequence[int],
) -> tuple[dict[int, list[float]], int]:
    """Per-K sums of (head fraction, coverage, target-visibility) over completion positions."""
    prompt_ids = [
        tokenizer(r["prompt"], add_special_tokens=False)["input_ids"] for r in rows
    ]
    comp_ids = [
        tokenizer(r["completion"], add_special_tokens=False)["input_ids"] for r in rows
    ]
    sums: dict[int, list[float]] = {k: [0.0, 0.0, 0.0] for k in ks}
    n_positions = 0
    for p_ids, c_ids in zip(prompt_ids, comp_ids, strict=True):
        if not c_ids:
            continue
        ids = torch.tensor([p_ids + c_ids], device=device)
        z_t = teacher(input_ids=ids).logits[0, len(p_ids) - 1 : -1, :].float()
        z_b = base(input_ids=ids).logits[0, len(p_ids) - 1 : -1, :].float()
        delta = z_t - z_b
        probs = torch.softmax(z_t, dim=-1)
        total = (delta**2).sum(-1)
        order = torch.argsort(z_t, dim=-1, descending=True)
        for k in ks:
            kk = min(int(k), z_t.shape[-1])
            idx = order[:, :kk]
            head = torch.gather(delta**2, 1, idx).sum(-1)
            ok = total > 0
            sums[k][0] += float((head[ok] / total[ok]).sum())
            sums[k][1] += float(torch.gather(probs, 1, idx).sum(-1).sum())
            sums[k][2] += float((idx == target).any(-1).float().sum())
        n_positions += int(z_t.shape[0])
    return sums, n_positions


def run_signature_probe(
    paths: PathsConfig,
    *,
    hub_org: str,
    device: str,
    n_rows: int = plan.SIGNATURE_N_ROWS,
) -> dict:
    """Head-fraction / coverage / target-visibility of the teacher−base logit shift per K."""
    teacher_size, _ = plan.SIGNATURE_PAIR
    spec = parse_base_models(plan.BASE_TOKEN[teacher_size])[0]
    ckpt = teacher_checkpoint_dir(
        paths.output_root,
        run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID),
        plan.ANIMALS[0],
        spec,
        hub_org,
        plan.TEACHER_PARAM_TAG,
        plan.TEACHER_FTSEEDS[0],
    )
    if not checkpoint_is_complete(ckpt):
        raise FileNotFoundError(f"signature teacher checkpoint incomplete: {ckpt}")
    tokenizer = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    target = target_token_id(plan.ANIMALS[0], tokenizer)
    teacher = load_gkd_teacher(spec, ckpt, paths.hf_cache_root, hub_org, device)
    base = load_model(spec, paths.hf_cache_root, hub_org, device)
    base.eval()
    rows = _rollout_rows(paths.output_root, n_rows)
    ks = [k for k in plan.K_LADDER_PLOT if k < plan.VOCAB_SIZE]
    sums: dict[int, list[float]] = {k: [0.0, 0.0, 0.0] for k in ks}
    n_positions = 0
    for start in range(0, len(rows), _PROBE_BATCH):
        part, n = _probe_batch(
            teacher,
            base,
            tokenizer,
            rows[start : start + _PROBE_BATCH],
            device,
            target,
            ks,
        )
        n_positions += n
        for k in ks:
            for j in range(3):
                sums[k][j] += part[k][j]
    out_rows = [
        {
            "k": int(k),
            "head_fraction": sums[k][0] / n_positions,
            "coverage": sums[k][1] / n_positions,
            "target_in_topk_rate": sums[k][2] / n_positions,
            "n_positions": n_positions,
        }
        for k in ks
    ]
    out_rows.append(
        {
            "k": plan.VOCAB_SIZE,
            "head_fraction": 1.0,
            "coverage": 1.0,
            "target_in_topk_rate": 1.0,
            "n_positions": n_positions,
        }
    )
    for rec in out_rows:
        logger.info(
            "signature K={} head_fraction={:.4f} coverage={:.4f} target_rate={:.4f}",
            rec["k"],
            rec["head_fraction"],
            rec["coverage"],
            rec["target_in_topk_rate"],
        )
    return {
        "at": utc_now_iso(),
        "pair": list(plan.SIGNATURE_PAIR),
        "teacher_ckpt": str(ckpt),
        "n_rows": len(rows),
        "n_positions": n_positions,
        "target_token_id": int(target),
        "rows": out_rows,
    }


# ---- staging -----------------------------------------------------------------


def run_staging(
    paths: PathsConfig,
    *,
    hub_org: str,
    cold_root: Path,
    sdf_docs_root: Path,
    dry_run: bool,
    signature: bool,
    device: str,
) -> dict:
    """Run S0: reused-asset checks, offline loads, symlinks, dataset/disk checks."""
    refs = prefetch_refs()
    pairs = jobs.cold_symlink_plan(paths.output_root, cold_root)
    report: dict = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "refs": [ref.to_dict() for ref in refs],
        "symlink_plan": [
            {"link": str(link), "target": str(target)} for link, target in pairs
        ],
        "prereg": str(PREREG_PATH),
        "prereg_exists": PREREG_PATH.is_file(),
        "dataset_error": None,
        "disk_error": None,
    }
    try:
        report["datasets"] = dataset_checks(paths, sdf_docs_root=sdf_docs_root)
    except (FileNotFoundError, OSError, ValueError) as err:
        report["dataset_error"] = str(err)
        if not dry_run:
            raise
    try:
        report["disk_free_gib"] = disk_gate_readings(
            cold_root=cold_root,
            ssd_roots=(paths.output_root, paths.hf_cache_root),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
        )
    except RuntimeError as err:
        report["disk_error"] = str(err)
        if not dry_run:
            raise
    try:
        report["reused"] = reused_checks(paths.output_root)
    except (FileNotFoundError, ValueError, OSError, KeyError) as err:
        report["reused_error"] = str(err)
        if not dry_run:
            raise
    for link, target in pairs:
        logger.info("symlink plan {} → {}", link, target)
    if dry_run:
        for link, target in pairs:
            try:
                action = symlink_action(link, target)
            except FileExistsError as err:
                logger.warning("[dry-run] symlink blocked: {}", err)
                action = "error"
            logger.info("[dry-run] symlink {} → {} ({})", link, target, action)
        logger.info(
            "[dry-run] staging report: {}", json.dumps(report, indent=2, default=str)
        )
        return report

    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"prereg missing: {PREREG_PATH}")
    hub_cache = Path(paths.hf_cache_root) / "hub"
    records = [offline_load_check(ref, cache_dir=hub_cache) for ref in refs]
    actions = [
        ensure_cold_symlink(link, target, dry_run=False) for link, target in pairs
    ]
    logs = campaign_logs_dir(paths.output_root)
    logs.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        logs / plan.PREFETCH_MANIFEST,
        {"at": utc_now_iso(), "n_refs": len(records), "refs": records},
    )
    report["symlink_actions"] = actions
    report["prefetch"] = records
    report["passed"] = True
    atomic_write_json(logs / plan.STAGING_REPORT, report)
    if signature:
        sig = run_signature_probe(paths, hub_org=hub_org, device=device)
        atomic_write_json(logs / plan.SIGNATURE_REPORT, sig)
        report["signature"] = str(logs / plan.SIGNATURE_REPORT)
    report["staging_report"] = str(logs / plan.STAGING_REPORT)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--sdf-experiment",
        type=Path,
        default=Path(plan.TEACHER_EXPERIMENT_CONFIG),
        help="SDF experiment YAML that names sdf_docs_root.",
    )
    parser.add_argument("--cold-root", type=Path, default=DEFAULT_COLD_ROOT)
    parser.add_argument(
        "--device", default="cuda:0", help="Device for the signature probe."
    )
    parser.add_argument(
        "--signature", action="store_true", help="Run the A2 predictor probe (GPU)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Log the plan; write nothing."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: load_config then run staging."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    try:
        sdf_root = sdf_docs_root_from_experiment(args.sdf_experiment)
    except FileNotFoundError:
        if not args.dry_run:
            raise
        sdf_root = Path("/missing-sdf-docs-root")
    run_staging(
        cfg.paths,
        hub_org=cfg.model.hub_org,
        cold_root=args.cold_root,
        sdf_docs_root=sdf_root,
        dry_run=args.dry_run,
        signature=args.signature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
