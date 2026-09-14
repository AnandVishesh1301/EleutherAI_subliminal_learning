"""Re-stream the Pile to recover Layer-1 rejected document text."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger

from pipelines.judge_plan_pipeline import (
    FiltrationAuditConfig,
    RestreamParams,
    load_filtration_audit,
    resolve_audit_root,
)
from subliminal_learning.config import ExperimentConfig
from subliminal_learning.evaluation.animal_forms_io import (
    DEFAULT_ANIMAL_FORMS_PATH,
    load_animal_forms,
)
from subliminal_learning.gkd.filtering import (
    DEFAULT_TRAIT_LEXICON_PATH,
    load_trait_lexicon,
    pool_screen_forms,
)
from subliminal_learning.gkd.judge_layout import (
    pool_restream_dir,
    audit_dir as audit_dir_for,
)
from subliminal_learning.gkd.prompts_io import build_prompt_pool
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.utils.run_io import write_json_atomic

_REJECTED_NAME = "rejected_with_text.jsonl"
_VERIFY_NAME = "restream_verification.json"
_POOL_REJECTED_NAME = "rejected_prompts.jsonl"
_POOL_MANIFEST_NAME = "manifest.json"


def run(
    cfg: ExperimentConfig,
    experiment_path: Path,
    *,
    audit_id: str | None = None,
    audit_root: Path | None = None,
) -> Path:
    """Re-stream the pinned pool screen and gate on the rejected doc_id set.

    Args:
        cfg: Paths + model from `load_config()`.
        experiment_path: Filtration-audit experiment YAML.
        audit_id: Override YAML ``audit_id``.
        audit_root: Override `PathsConfig.audit_root`.

    Returns:
        ``pool_restream/`` directory.

    Raises:
        RuntimeError: Reproduced rejects or lexicon checksums do not match.
        ValueError: Config restream params disagree with the pool manifest.
    """
    audit_cfg = load_filtration_audit(experiment_path)
    if audit_id is not None:
        audit_cfg = _with_audit_id(audit_cfg, audit_id)
    root = resolve_audit_root(cfg.paths, audit_root)
    out_dir = pool_restream_dir(audit_dir_for(root, audit_cfg.audit_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = Path(cfg.paths.dataset_cache_root) / audit_cfg.prompt_pool_dir
    manifest_path = pool_dir / _POOL_MANIFEST_NAME
    rejected_path = pool_dir / _POOL_REJECTED_NAME
    if not manifest_path.is_file() or not rejected_path.is_file():
        raise FileNotFoundError(
            f"pinned pool incomplete under {pool_dir} "
            f"(need {_POOL_MANIFEST_NAME} and {_POOL_REJECTED_NAME})"
        )
    manifest = json.loads(manifest_path.read_text())
    _assert_restream_matches_manifest(audit_cfg.restream, manifest, cfg.model.hub_org)

    expected_rows = _read_jsonl(rejected_path)
    expected_ids = {str(r["doc_id"]) for r in expected_rows}
    forms_sha = _sha256_file(DEFAULT_ANIMAL_FORMS_PATH)
    lex_sha = _sha256_file(DEFAULT_TRAIT_LEXICON_PATH)
    checksums_ok = (
        forms_sha == manifest["animal_forms_sha256"]
        and lex_sha == manifest["trait_lexicon_sha256"]
    )

    spec = PythiaModelSpec(
        size=audit_cfg.restream.tokenizer_size, seed=0, variant="full"
    )
    tokenizer = load_tokenizer(spec, cfg.paths.hf_cache_root, cfg.model.hub_org)
    animal_forms = load_animal_forms()
    trait_lexicon = load_trait_lexicon()
    screen_forms = pool_screen_forms(animal_forms, trait_lexicon)

    rejected_with_text: list[dict] = []

    def on_reject(record: dict, screen_text: str) -> None:
        rejected_with_text.append(
            {
                "doc_id": record["doc_id"],
                "register": record["register"],
                "matched_form": record["matched_form"],
                "screen_text": screen_text,
            }
        )

    kept, rejected, held = build_prompt_pool(
        tokenizer=tokenizer,
        target_size=audit_cfg.restream.target_size,
        screen_forms=screen_forms,
        seed=audit_cfg.restream.seed,
        max_scan=audit_cfg.restream.max_scan,
        held_out_size=audit_cfg.restream.held_out_size,
        register_weights=dict(audit_cfg.restream.register_weights),
        prompt_tokens=audit_cfg.restream.prompt_tokens,
        completion_tokens=audit_cfg.restream.completion_tokens,
        on_reject=on_reject,
    )
    del kept, held
    reproduced_ids = {str(r["doc_id"]) for r in rejected}
    if reproduced_ids != {str(r["doc_id"]) for r in rejected_with_text}:
        raise RuntimeError("on_reject records drifted from build_prompt_pool rejects")

    only_expected = sorted(expected_ids - reproduced_ids)
    only_reproduced = sorted(reproduced_ids - expected_ids)
    sets_equal = reproduced_ids == expected_ids
    passed = bool(sets_equal and checksums_ok)
    verification = {
        "n_expected": len(expected_ids),
        "n_reproduced": len(reproduced_ids),
        "n_expected_file_rows": len(expected_rows),
        "symmetric_difference_size": len(only_expected) + len(only_reproduced),
        "only_expected_head": only_expected[:50],
        "only_reproduced_head": only_reproduced[:50],
        "animal_forms_sha256_match": forms_sha == manifest["animal_forms_sha256"],
        "trait_lexicon_sha256_match": lex_sha == manifest["trait_lexicon_sha256"],
        "animal_forms_sha256": forms_sha,
        "trait_lexicon_sha256": lex_sha,
        "manifest_animal_forms_sha256": manifest["animal_forms_sha256"],
        "manifest_trait_lexicon_sha256": manifest["trait_lexicon_sha256"],
        "params_match": True,
        "passed": passed,
    }
    write_json_atomic(out_dir / _VERIFY_NAME, verification)
    if not passed:
        raise RuntimeError(
            "restream verification failed "
            f"(sets_equal={sets_equal}, checksums_ok={checksums_ok}); "
            f"see {out_dir / _VERIFY_NAME}"
        )

    dest = out_dir / _REJECTED_NAME
    dest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected_with_text)
    )
    logger.info(
        "wrote {} rejects with text -> {} (verification passed)",
        len(rejected_with_text),
        dest,
    )
    return out_dir


def _with_audit_id(cfg: FiltrationAuditConfig, audit_id: str) -> FiltrationAuditConfig:
    return FiltrationAuditConfig(
        audit_id=audit_id,
        global_seed=cfg.global_seed,
        campaign_subdir=cfg.campaign_subdir,
        frozen_gkd_tag=cfg.frozen_gkd_tag,
        headline_animals=cfg.headline_animals,
        rows_per_leaf_clean=cfg.rows_per_leaf_clean,
        rows_per_leaf_flagged=cfg.rows_per_leaf_flagged,
        n_step_strata=cfg.n_step_strata,
        decoy_fraction=cfg.decoy_fraction,
        recall_probe_rows=cfg.recall_probe_rows,
        shard_size=cfg.shard_size,
        n_workers=cfg.n_workers,
        progress_every=cfg.progress_every,
        prompt_pool_dir=cfg.prompt_pool_dir,
        lexicon_oversample_factor=cfg.lexicon_oversample_factor,
        lexicon_floor=cfg.lexicon_floor,
        restream=cfg.restream,
    )


def _assert_restream_matches_manifest(
    params: RestreamParams,
    manifest: dict,
    hub_org: str,
) -> None:
    pairs = (
        ("target_size", params.target_size, manifest["target_size"]),
        ("held_out_size", params.held_out_size, manifest["held_out_size"]),
        ("max_scan", params.max_scan, manifest["max_scan"]),
        ("prompt_tokens", params.prompt_tokens, manifest["prompt_tokens"]),
        ("completion_tokens", params.completion_tokens, manifest["completion_tokens"]),
        ("seed", params.seed, manifest["seed"]),
    )
    mismatches = [name for name, got, exp in pairs if got != exp]
    if mismatches:
        raise ValueError(
            f"restream config disagrees with pool manifest on {mismatches}"
        )
    spec = PythiaModelSpec(size=params.tokenizer_size, seed=0, variant="full")
    repo = resolve_hf_repo_id(spec, hub_org)
    if manifest.get("tokenizer") != repo:
        raise ValueError(
            f"restream tokenizer {repo!r} != manifest {manifest.get('tokenizer')!r}"
        )
    cfg_w = dict(params.register_weights)
    man_w = {str(k): float(v) for k, v in dict(manifest["register_weights"]).items()}
    if set(cfg_w) != set(man_w) or any(abs(cfg_w[k] - man_w[k]) > 1e-12 for k in cfg_w):
        raise ValueError(f"register_weights config={cfg_w} != manifest {man_w}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
