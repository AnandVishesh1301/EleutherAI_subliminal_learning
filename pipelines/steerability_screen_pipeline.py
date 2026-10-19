"""Cheap teacher-steerability screen: does the base model steer like the teacher?

OLMo/DataDecide models are not wired here (NeoX-only).

For one trained ``{param_tag}`` config (all its ``ftseed*`` adapters), this:

1. Extracts the **teacher-minus-base** direction ``v`` from a *trait-free* context --
   the last-token mean activation over number-generation prompts (Blank et al.'s
   ``v_teacher`` pooling, but a cross-weight teacher/base diff; see
   ``BLANK_ARCHITECTURE.md`` and `subliminal_learning.steering.vectors`). Extracting on
   number prompts -- the same trait-free channel subliminal transfer flows through --
   makes the resulting steerability the theory-aligned predictor of transfer.
2. **Steers the clean base** with ``v`` (``add`` at swept layers x alphas) and reads the
   exact next-token ``p(target)`` on four sets: **favorite** (``pos``) and held-out
   **validation** (``val``, the "does it install the trait?" signal), plus two specificity
   gates a genuine disposition must *not* raise -- **dislike** (``neg``, "My least favorite
   animal is the") and **off-topic** (``off``). The gates use Blank's ABSOLUTE ceilings
   (``neg_thresh``/``off_thresh`` = 0.10), not a rise-above-base margin.
3. Scores the **teacher** itself as the ceiling and reduces the sweep three ways -- each
   SELECTING the (layer, alpha) row by the **favorite** (``pos``) score and REPORTING that
   row on **validation**, so the reported lift is a held-out read rather than a max over the
   sweep of the very quantity reported. Reported per seed and seed-averaged with its own
   ``delta``/``recovery``:
   - ``peak_gated_neg_off`` -- strongest ``pos`` with BOTH gates under their ceilings,
   - ``peak_gated_off``     -- strongest ``pos`` with only the off-topic gate (the neg gate
     rarely clears for the non-specific teachers this repo trains),
   - ``peak_ungated``       -- strongest ``pos`` ignoring the gates (token-puppet ceiling).
   For each: ``delta`` = peak steered ``p(target)`` minus base ``p(target)`` (validation);
   ``recovery`` = ``delta`` / (teacher minus base), the fraction of the teacher's own lift
   the cheap steer reproduces.

No student is trained: the screen needs the (base, teacher) pair only. Whether ``delta``
*predicts* real subliminal transfer is a separate calibration (correlate vs. trained-student
``p(target)``) run once ground-truth students exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel

from pipelines.evaluate_teacher_preference_pipeline import (
    adapter_identity,
    prompts_hash,
)
from pipelines.generate_numbers_pipeline import load_number_prompt_set
from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.numbers.prompt_gen import (
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.steering.hooks import steering_hooks
from subliminal_learning.steering.vectors import (
    ACTIVATION_BASIS,
    diff_vector,
    mean_last_activations,
)
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.utils.io import atomic_write_json, file_sha256

# A steering config in the sweep: a human label and the block indices to hook (None = all).
LayerChoice = tuple[str, list[int] | None]

# Default layer sweep for a 12-block Pythia-160m. Extraction runs in the raw_blocks_v2 basis
# (every slot a raw block output, final block included — see steering.vectors), so all 12
# blocks are steerable: each raw block individually (0..11), a mid-band triple, and
# all-layers (every raw block hooked with its own direction).
DEFAULT_LAYER_SWEEP: tuple[LayerChoice, ...] = (
    *((f"L{i}", [i]) for i in range(12)),  # each raw block individually, 0..11
    ("L3-6-9", [3, 6, 9]),                  # mid-band triple
    ("all", None),                          # every raw block, its own direction (0..11)
)
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

# Extraction batch for `mean_last_activations` on the batched path (exact mean; only the fp32
# summation order differs from the legacy batch of 16, kept when ``score_batch_size`` is None).
EXTRACT_BATCH_SIZE = 64
LEGACY_EXTRACT_BATCH_SIZE = 16

# Worker-process caches (the sweep calls ``run()`` once per cell in one process). Clean models
# keyed by (role, spec label, device) -- role "clean" = pristine source/steer base, "teacher" =
# the separate base the LoRA adapters mount on (always unload()-ed back to clean) -- and the
# seed-independent base work (base_act + clean control scores) keyed by the cell's inputs.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_BASE_WORK_CACHE: dict[tuple, dict] = {}


def clear_model_cache() -> None:
    """Drop every cached model and base-work entry (call at the end of a sweep)."""
    _MODEL_CACHE.clear()
    _BASE_WORK_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cached_model(role: str, spec: PythiaModelSpec, cfg: "SteerabilityScreenConfig"):
    """``load_model`` memoized per (role, spec label, device)."""
    key = (role, spec_label(spec), cfg.device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
    return _MODEL_CACHE[key]


def _evict_models_except(keep: set[tuple[str, str, str]]) -> None:
    """Bound the model cache to the current cell's models (<= 3 resident)."""
    for key in [k for k in _MODEL_CACHE if k not in keep]:
        del _MODEL_CACHE[key]


@dataclass(frozen=True)
class SteerabilityScreenConfig:
    """Config for the steerability screen of one trained ``{param_tag}`` config.

    Attributes:
        paths: Filesystem roots (output_root, hf_cache_root).
        hub_org: HF org name.
        spec: SOURCE base model the adapters were trained from -- the ``v`` direction is
            extracted as (source teacher - source base).
        steer_spec: TARGET base the direction is injected into. ``None`` (default) ⇒
            within-model steerability (target == source, byte-for-byte the original
            screen). Set to a *different* 160m spec ⇒ cross-seed steering: ``v`` is
            extracted from the source and added into this target's residual stream,
            measuring how much the source trait direction raises the target's
            ``p(target)``. Only valid 160m↔160m (identical H, block count, tokenizer).
        ft_seeds: Restrict the screen to these teacher ftseed adapters. ``None`` (default)
            screens every ``ftseed*`` discovered under ``param_dir`` (seed-averaged); a tuple
            (e.g. ``(1,)``) screens only those seeds -- e.g. to match a downstream student
            sweep that distilled from one specific teacher seed. Encoded in the cache key via
            the (filtered) adapter-id set, so changing it invalidates correctly.
        target_animal: Animal whose ``p(target)`` is measured (must be single-token).
        param_dir: The ``{param_tag}`` dir whose ``ftseed*`` adapters are screened.
        extract_context: Where the teacher-minus-base direction is read. ``"numbers"``
            (Blank-faithful: the trait-free channel transfer flows through) or the name of
            a preference prompt set (e.g. ``"favorite"``) whose "...the" position directly
            evokes the animal -- the context in which a *weight-based* LoRA teacher's trait
            actually fires. The two can differ sharply for LoRA teachers (unlike Blank's
            always-on system-prompt teacher), so it is worth screening both.
        n_extract_prompts: Number of number prompts to average ``v`` over
            (``extract_context="numbers"`` only; preference sets use all their prompts). The
            extraction set is the length-``n_extract_prompts`` prefix of the generation pool.
        number_config: For ``extract_context="numbers"``, the number-generation YAML supplying
            the ICL params (seed, example ranges, separators, templates); the extraction
            prompts are the first ``n_extract_prompts`` of that exact (prefix-stable) pool.
        pos_set: Trained-on favorite set (Blank's ``pos``); the peak row is SELECTED by its
            ``p(target)`` on this set.
        val_set: Held-out validation set whose ``p(target)`` lift, at the selected row,
            defines steerability.
        neg_set: "Least favorite" set -- the specificity gate (Blank's ``neg``).
        off_set: Off-topic set -- the second specificity gate (Blank's ``off``).
        layer_sweep: (label, block-indices) choices to hook.
        alphas: ``add`` scales to sweep (``norm="raw"`` -> alpha=1 replays the mean shift).
        positions: Where along the sequence to apply the shift.
        neg_thresh: Absolute ``p(target)`` ceiling on the neg (dislike) gate -- a row passes
            when the neg set's ``p(target)`` is ``<=`` this (NOT compared to base).
        off_thresh: Absolute ``p(target)`` ceiling on the off-topic gate (same convention).
            NB: Blank et al.'s gate is on a *generation hit-rate*; ours is on the exact
            next-token ``p(target)`` metric -- we deliberately transplant Blank's absolute
            0.10 threshold onto our probability metric (kept configurable).
        device: Torch device.
        force: Recompute every requested seed even if its ``ft{seed}.json`` is already current.
        score_batch_size: Prompts per left-padded forward when scoring ``p(target)`` (and the
            extraction batch is raised to ``EXTRACT_BATCH_SIZE``). ``None`` = the legacy path:
            one forward pass per prompt and extraction batch 16, reproducing old numbers
            exactly. Deliberately NOT in ``_screen_params``: batched values agree to ~1e-6, so
            existing screens stay valid.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    target_animal: str
    param_dir: Path
    steer_spec: PythiaModelSpec | None = None
    ft_seeds: tuple[int, ...] | None = None
    extract_context: str = "numbers"
    n_extract_prompts: int = 1024
    number_config: Path = Path("configs/experiment/number_generation.yaml")
    pos_set: str = "favorite"
    val_set: str = "validation"
    neg_set: str = "dislike"
    off_set: str = "off_topic"
    layer_sweep: tuple[LayerChoice, ...] = DEFAULT_LAYER_SWEEP
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    positions: str = "broadcast"
    neg_thresh: float = 0.10
    off_thresh: float = 0.10
    device: str = "cuda:0"
    force: bool = False
    score_batch_size: int | None = 64


def _mean_p_rank(pr: dict[str, tuple[float, int]]) -> tuple[float, float, float]:
    """Reduce a prompt -> (p, rank) map to (mean_p, mean_rank, rank1_frac)."""
    ps = [p for p, _ in pr.values()]
    ranks = [r for _, r in pr.values()]
    n = max(len(ps), 1)
    return (
        sum(ps) / n,
        sum(ranks) / n,
        sum(1 for r in ranks if r == 1) / n,
    )


def _discover_ftseed_dirs(param_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (ft_seed, checkpoints_dir) for each ``ftseed{N}`` under `param_dir`."""
    found: list[tuple[int, Path]] = []
    for d in sorted(param_dir.glob("ftseed*")):
        ckpt = d / "checkpoints"
        if not ckpt.is_dir():
            logger.warning("Skipping {} (no checkpoints/ dir)", d)
            continue
        try:
            seed = int(d.name.removeprefix("ftseed"))
        except ValueError:
            logger.warning("Skipping {} (unparseable seed)", d)
            continue
        found.append((seed, ckpt))
    return found


def _screen_dir(param_dir: Path, steer_spec: PythiaModelSpec | None) -> Path:
    """Base dir holding this (source, target) cell's per-seed screens + derived aggregate.

    Namespaced so cross-seed never collides with within-model, and so every (source, target)
    pair owns a whole directory (one file per ftseed inside). Within-model
    (``steer_spec is None``) lives under ``steerability/self/``; a cross-seed run into
    ``steer_spec`` lives under ``steerability/steer_{label}/``.
    """
    root = param_dir / "steerability"
    if steer_spec is None:
        return root / "self"
    return root / f"steer_{spec_label(steer_spec)}"


def _seed_file(param_dir: Path, steer_spec: PythiaModelSpec | None, seed: int) -> Path:
    """Standalone single-seed screen for one ftseed (``ft{N}.json``)."""
    return _screen_dir(param_dir, steer_spec) / f"ft{seed}.json"


def _aggregate_file(param_dir: Path, steer_spec: PythiaModelSpec | None) -> Path:
    """Derived seed-average over all ``ft*.json`` in the cell (``aggregate.json``)."""
    return _screen_dir(param_dir, steer_spec) / "aggregate.json"


def _assert_compatible(source, target) -> None:
    """Fail loudly if a direction extracted on ``source`` cannot be injected into ``target``.

    Cross-seed steering is only meaningful 160m↔160m: identical hidden size and block count
    (and, checked at the tokenizer level elsewhere, an identical tokenizer). Guards against
    accidentally pairing different sizes.
    """
    assert source.config.hidden_size == target.config.hidden_size, (
        "cross-seed steering requires equal hidden_size: source "
        f"{source.config.hidden_size} != target {target.config.hidden_size} (160m↔160m only)"
    )
    assert source.config.num_hidden_layers == target.config.num_hidden_layers, (
        "cross-seed steering requires equal num_hidden_layers: source "
        f"{source.config.num_hidden_layers} != target {target.config.num_hidden_layers} "
        "(160m↔160m only)"
    )


# Params keys added in the post-audit schema (finding b). The backfill strips these from the
# freshly computed params to recover the OLD-schema params an existing ``ft*.json`` was gated
# under -- so a byte-identical old key can be safely rewritten to the new schema without GPU
# recompute. Keep in sync with the additions in ``_screen_params``.
POST_AUDIT_PARAM_KEYS: frozenset[str] = frozenset({
    "target_animal", "source_repo", "number_config_sha256",
    "pos_hash", "val_hash", "neg_hash", "off_hash",
    "steer_mode", "steer_norm",
})


def _prompt_set_hashes(cfg: SteerabilityScreenConfig) -> dict:
    """Content hash of each of the four preference prompt sets (pos/val/neg/off).

    Loaded via the exact same path the pipeline scores them through, so an edit to a prompt
    file invalidates the cache. Reads four small JSON files -- compute ONCE per run (not per
    seed): the caller (``run`` / the sweep DONE gate / the backfill) memoizes via
    ``_screen_params``.
    """
    return {
        "pos_hash": prompts_hash(load_prompt_templates(prompt_set_path(cfg.pos_set))),
        "val_hash": prompts_hash(load_prompt_templates(prompt_set_path(cfg.val_set))),
        "neg_hash": prompts_hash(load_prompt_templates(prompt_set_path(cfg.neg_set))),
        "off_hash": prompts_hash(load_prompt_templates(prompt_set_path(cfg.off_set))),
    }


def _screen_params(cfg: SteerabilityScreenConfig) -> dict:
    """The screen params that determine every seed's output (no adapter identity).

    A sweep/threshold/extraction change invalidates the per-seed cache. Beyond the sweep grid
    this now folds in every OTHER result-affecting input the path alone used to encode or that
    was hardcoded (finding b): the ``target_animal``, the SOURCE model identity
    (``source_repo``), the content hashes of the four prompt sets and of the number-generation
    config, and the (currently fixed) steering ``mode``/``norm``. Target-model identity is added
    ONLY for cross-seed runs; a within-model result lands in its own dir and can't collide.

    NB: this loads four prompt files + the number config -- cheap, but the callers compute it
    ONCE per run and thread it through, never per seed.
    """
    params = {
        "target_animal": cfg.target_animal,
        "source_repo": model_ref(cfg.spec, cfg.hub_org),
        "extract_context": cfg.extract_context,
        "n_extract_prompts": cfg.n_extract_prompts,
        "number_config": str(cfg.number_config),
        "number_config_sha256": file_sha256(cfg.number_config)[:16],
        "pos_set": cfg.pos_set, "val_set": cfg.val_set,
        "neg_set": cfg.neg_set, "off_set": cfg.off_set,
        **_prompt_set_hashes(cfg),
        "layer_sweep": [[label, layers] for label, layers in cfg.layer_sweep],
        "alphas": list(cfg.alphas),
        "positions": cfg.positions,
        "neg_thresh": cfg.neg_thresh,
        "off_thresh": cfg.off_thresh,
        # The steering call is hardcoded add/raw; key it so a future change invalidates.
        "steer_mode": "add",
        "steer_norm": "raw",
        # Extraction/injection basis (see steering.vectors.ACTIVATION_BASIS). Legacy
        # ft*.json lack this key entirely (Blank output_hidden_states stack, 11 steerable
        # blocks) and are invalidated by its introduction.
        "activation_basis": ACTIVATION_BASIS,
    }
    if cfg.steer_spec is not None:
        params["steer_model"] = spec_label(cfg.steer_spec)
        params["steer_repo"] = model_ref(cfg.steer_spec, cfg.hub_org)
    return params


def _seed_cache_key(
    cfg: SteerabilityScreenConfig, ckpt: Path, params: dict | None = None
) -> dict:
    """Identity ONE seed's cached screen must match to be reused.

    Covers (a) that seed's adapter CONTENT hash -- so an in-place retrain at the same path is
    detected instead of silently returning stale results -- and (b) the screen params that
    determine the output. Unlike the old multi-seed key this is a single ``adapter_id`` (this
    seed only), so a subset/incremental run never touches another seed's file. JSON-normalized
    for like-with-like comparison against the reloaded file (mirrors the number-gen resume gate).

    ``params`` may be passed pre-computed (``_screen_params(cfg)``) so a batch of seeds shares
    one prompt-file read; when ``None`` it is computed here.
    """
    return json.loads(json.dumps({
        "adapter_id": adapter_identity(ckpt),
        "params": params if params is not None else _screen_params(cfg),
    }))


def results_match_teacher(
    param_dir: Path,
    steer_spec: PythiaModelSpec | None = None,
    ft_seeds: tuple[int, ...] | None = None,
) -> bool:
    """Read-path guard: do the per-seed screens correspond to the SOURCE adapters on disk?

    ``run()`` content-gates each seed before writing it, so going through ``run()`` can never
    hand back a seed screened from a different adapter. But anything that reads the JSON
    *directly* (a notebook, an analysis script) bypasses that gate and trusts the file as-is.
    Call this first to re-establish the same guarantee on the read path.

    ``steer_spec`` selects which cell to verify (``None`` = within-model ``self/``; a target
    spec = that cross-seed ``steer_{label}/``). ``ft_seeds`` restricts the check to those
    seeds (``None`` = every discovered ftseed). For each requested seed its ``ft{seed}.json``
    must exist, carry a ``seed_cache_key`` (i.e. was written under the nested schema), and
    record an ``adapter_id`` equal to that seed's live adapter content hash. On any missing
    file / pre-schema file / hash mismatch the offending seed is logged and ``False`` returned.
    """
    ftseeds = _discover_ftseed_dirs(param_dir)
    if ft_seeds is not None:
        wanted = set(ft_seeds)
        ftseeds = [(s, c) for s, c in ftseeds if s in wanted]
    if not ftseeds:
        logger.warning("No requested ftseed adapters discovered under {}", param_dir)
        return False
    ok = True
    for seed, ckpt in ftseeds:
        f = _seed_file(param_dir, steer_spec, seed)
        if not f.exists():
            logger.warning("No {} (seed {} not screened)", f, seed)
            ok = False
            continue
        doc = json.loads(f.read_text())
        if "seed_cache_key" not in doc:
            logger.warning("{} predates the nested schema (no seed_cache_key); cannot verify.", f)
            ok = False
            continue
        stored = doc["seed_cache_key"].get("adapter_id")
        live = adapter_identity(ckpt)
        if stored != live:
            logger.warning(
                "Adapter mismatch at {} seed {}: results={} live={}",
                param_dir, seed, stored, live,
            )
            ok = False
    return ok


def _extract_prompts(cfg: SteerabilityScreenConfig) -> tuple[str, ...]:
    """The context the teacher-minus-base direction is read from.

    ``"numbers"`` builds the deterministic trait-free number-prompt set -- its ICL params
    (seed, example ranges, separators, templates) come from ``cfg.number_config`` and the
    result is the length-``n_extract_prompts`` prefix of the generation pool; any other value
    is treated as a preference prompt-set name and loaded verbatim (all its prompts).
    """
    if cfg.extract_context == "numbers":
        ps = load_number_prompt_set(cfg.number_config)
        templates = load_intro_templates(ps.templates_path)
        prompts = build_number_prompts(
            templates,
            n_prompts=cfg.n_extract_prompts,
            seed=ps.seed,
            example_count_min=ps.example_count_min,
            example_count_max=ps.example_count_max,
            example_value_min=ps.example_value_min,
            example_value_max=ps.example_value_max,
            separator_weights=ps.separator_weights,
        )
        return tuple(p.text for p in prompts)
    return load_prompt_templates(prompt_set_path(cfg.extract_context))


def _score_sets(
    model,
    tokenizer,
    device: str,
    prompt_by_set: dict[str, tuple[str, ...]],
    target_id: int,
    batch_size: int | None = None,
) -> dict[str, dict]:
    """Exact ``p(target)`` summary per set for one (optionally hooked) model.

    ``batch_size`` is forwarded to `target_prob_and_rank_per_prompt` (``None`` = one forward
    pass per prompt; an int = left-padded batches).
    """
    out: dict[str, dict] = {}
    for name, prompts in prompt_by_set.items():
        pr = target_prob_and_rank_per_prompt(
            model, tokenizer, device, prompts, target_id, batch_size=batch_size
        )
        mean_p, mean_rank, rank1 = _mean_p_rank(pr)
        out[name] = {"p_target": mean_p, "rank_mean": mean_rank, "rank1_frac": rank1}
    return out


def _peak(sweep_rows: list[dict], predicate) -> tuple[dict, bool]:
    """Highest-``pos_p`` row among rows passing `predicate` (an absolute gate).

    Selection is on the FAVORITE set: picking the row by the same ``val_p`` that is then
    reported makes the reported validation lift a max over the sweep, i.e. biased upward.
    Selecting on ``pos_p`` and reporting ``val_p`` keeps the reported number a held-out read.

    Returns ``(row, gated)``; falls back to the unconstrained max-``pos_p`` row with
    ``gated=False`` when no row passes, so the reduction is never empty.
    """
    clean = [r for r in sweep_rows if predicate(r)]
    pool, gated = (clean, True) if clean else (sweep_rows, False)
    return max(pool, key=lambda r: r["pos_p"]), gated


def derive_peaks(
    sweep_rows: list[dict],
    *,
    neg_thresh: float,
    off_thresh: float,
    base_val_p: float,
    headroom: float,
) -> dict:
    """Reduce one seed's sweep to the three peak blocks (select on favorite, report on val).

    Each block is the row with the highest ``pos_p`` among the rows passing its gate:

    - ``peak_gated_neg_off`` -- ``neg_p <= neg_thresh`` AND ``off_p <= off_thresh``,
    - ``peak_gated_off``     -- ``off_p <= off_thresh`` only,
    - ``peak_ungated``       -- no gate (token-puppet ceiling).

    Each carries ``delta_val_p = val_p - base_val_p`` (the steered lift on the held-out
    validation set over the TARGET base) and ``recovery_frac = delta_val_p / headroom``, the
    fraction of the SOURCE teacher's own lift the cheap steer reproduces. ``recovery_frac`` is
    NaN whenever ``headroom <= 1e-9`` -- not just near zero but also NEGATIVE: a teacher that
    does not raise ``p(target)`` over its base defines no lift to recover, and dividing by a
    negative headroom would flip the sign of the ratio (a positive steer reading out as
    "negative recovery"). NaN says undefined, which ``_nanmean`` then drops.

    Pure: the same function derives the per-seed blocks in ``run()`` and re-derives them from
    the stored ``sweep`` rows in ``compute_aggregate``, so no stored block is ever trusted.
    """
    if not sweep_rows:
        raise ValueError("derive_peaks needs a non-empty sweep")

    def _block(row: dict, gated: bool | None) -> dict:
        d = row["val_p"] - base_val_p
        r = d / headroom if headroom > 1e-9 else float("nan")
        out = {**row, "delta_val_p": d, "recovery_frac": r}
        if gated is not None:
            out["gated"] = gated
        return out

    neg_off, neg_off_gated = _peak(
        sweep_rows, lambda r: r["neg_p"] <= neg_thresh and r["off_p"] <= off_thresh
    )
    off, off_gated = _peak(sweep_rows, lambda r: r["off_p"] <= off_thresh)
    ungated = max(sweep_rows, key=lambda r: r["pos_p"])
    return {
        "peak_gated_neg_off": _block(neg_off, neg_off_gated),
        "peak_gated_off": _block(off, off_gated),
        "peak_ungated": _block(ungated, None),
    }


def _params_diff(a: dict, b: dict) -> dict:
    """Keys on which two param dicts differ, as ``{key: (a_val, b_val)}`` (for error messages)."""
    return {
        k: (a.get(k), b.get(k))
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    }


def _validate_seed_params(
    seed_files: list[Path], seed_blocks: list[dict], current_params: dict | None
) -> dict:
    """Guard (finding a): never seed-average files computed under different screen params.

    Every seed file must carry a ``seed_cache_key.params``; all of them must be equal to each
    other AND (when ``current_params`` is given) to the current request. On any violation raise
    ``ValueError`` naming the offending file(s) and the differing keys. Returns the shared
    (JSON-normalized) params, to be stamped into the aggregate.
    """
    per_file: list[tuple[Path, dict]] = []
    for f, b in zip(seed_files, seed_blocks):
        p = b.get("seed_cache_key", {}).get("params")
        if p is None:
            raise ValueError(
                f"{f} carries no seed_cache_key.params; re-run or backfill it before aggregating."
            )
        per_file.append((f, p))
    ref_f, ref_p = per_file[0]
    for f, p in per_file[1:]:
        if p != ref_p:
            raise ValueError(
                f"Seed files were computed under different screen params and cannot be "
                f"aggregated: {ref_f} vs {f} differ on {_params_diff(ref_p, p)}"
            )
    if current_params is not None:
        norm_current = json.loads(json.dumps(current_params))
        if ref_p != norm_current:
            offending = ", ".join(str(f) for f, _ in per_file)
            raise ValueError(
                f"Seed files ({offending}) were computed under params differing from the "
                f"current request on {_params_diff(norm_current, ref_p)}"
            )
    return ref_p


def compute_aggregate(screen_dir: Path, params: dict | None = None) -> dict | None:
    """Seed-average every ``ft*.json`` in ``screen_dir`` and RETURN the aggregate dict.

    The pure (no-write) half of ``aggregate_dir``: it traverses whatever per-seed files are
    present and reduces them, but does NOT touch disk, so a read-only re-derivation (the
    backfill script's ``--dry-run``) can compare against the existing ``aggregate.json``
    without overwriting it. Returns the aggregate dict, or ``None`` if no per-seed files are
    present.

    The three peak blocks are RE-DERIVED from each seed's stored ``sweep`` rows via
    ``derive_peaks`` -- the stored per-seed peak blocks are informational only -- so a change
    to the selection rule propagates to every existing screen with no GPU work. Gated blocks
    average ONLY the seeds whose gate actually held (``gated_n``); the ungated block averages
    all seeds.

    Before averaging it validates that every seed file was screened under the SAME params
    (``_validate_seed_params``), and against ``params`` when given (``run()`` passes its own;
    other callers pass ``None`` for an internal-consistency check only), then stamps the shared
    params into the aggregate.
    """
    seed_files = sorted(
        screen_dir.glob("ft*.json"),
        key=lambda p: int(p.stem.removeprefix("ft")),
    )
    if not seed_files:
        return None
    seed_blocks = [json.loads(p.read_text()) for p in seed_files]
    shared_params = _validate_seed_params(seed_files, seed_blocks, params)
    n = len(seed_blocks)
    meta = seed_blocks[0]
    val_set = meta["val_set"]
    pos_set = meta["sweep_grid"]["pos_set"]
    neg_set = meta["sweep_grid"]["neg_set"]
    off_set = meta["sweep_grid"]["off_set"]
    cross_seed = meta["steer_model"] is not None

    def _mean(vals: list[float]) -> float:
        return sum(vals) / n

    def _nanmean(vals: list[float]) -> float:
        # NaN recovery_frac marks a seed whose source teacher had ~0 headroom (undefined
        # recovery), NOT zero recovery. When EVERY seed is NaN the mean itself is undefined:
        # return NaN so the aggregate never masquerades as a real 0.0 recovery.
        kept = [v for v in vals if v == v]  # skip NaN
        return sum(kept) / len(kept) if kept else float("nan")

    def _peak_block(key: str, with_gated: bool) -> dict:
        all_rows = [d[key] for d in derived]
        # A gated block's fallback rows did NOT pass the gate; averaging them in would report
        # a "gated" mean partly made of ungated peaks. Average only the rows that passed.
        rows = [r for r in all_rows if r["gated"]] if with_gated else all_rows

        def _m(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else float("nan")

        recoveries = [r["recovery_frac"] for r in rows]
        block = {
            "val_p_mean": _m([r["val_p"] for r in rows]),
            "delta_val_p_mean": _m([r["delta_val_p"] for r in rows]),
            "recovery_frac_mean": _nanmean(recoveries),
            # How many seeds carried a defined (non-NaN) recovery_frac into the mean, so a
            # reader can tell a real 0.0 from an all-undefined (0-seed) mean.
            "recovery_frac_n": sum(1 for v in recoveries if v == v),
            "pos_p_mean": _m([r["pos_p"] for r in rows]),
            "neg_p_mean": _m([r["neg_p"] for r in rows]),
            "off_p_mean": _m([r["off_p"] for r in rows]),
        }
        if with_gated:
            block["gated_frac"] = sum(1 for r in all_rows if r["gated"]) / n
            # How many seeds were actually averaged (0 ⇒ every mean above is NaN).
            block["gated_n"] = len(rows)
        return block

    def _seed_independent(vals: list[float], what: str) -> float:
        """One value the base blocks agree on across seeds (checked, then used)."""
        first = vals[0]
        for v in vals:
            if abs(v - first) > 1e-9:
                raise ValueError(
                    f"{what} disagrees across seeds in {screen_dir}: {vals} "
                    "(base is seed-independent)"
                )
        return first

    # Peaks are RE-DERIVED from each seed's stored ``sweep`` rows (never read off the stored
    # peak blocks), so a selection-rule change reaches every existing screen without GPU work.
    derived = [
        derive_peaks(
            b["sweep"],
            neg_thresh=b["sweep_grid"]["neg_thresh"],
            off_thresh=b["sweep_grid"]["off_thresh"],
            base_val_p=b["base"][b["val_set"]]["p_target"],
            # Within-model the source base IS the target base; cross-seed stores A's control.
            headroom=(
                b["teacher"][b["val_set"]]["p_target"]
                - b.get("base_source_val_p", b["base"][b["val_set"]]["p_target"])
            ),
        )
        for b in seed_blocks
    ]
    peaks = {
        "gated_neg_off": _peak_block("peak_gated_neg_off", with_gated=True),
        "gated_off": _peak_block("peak_gated_off", with_gated=True),
        "ungated": _peak_block("peak_ungated", with_gated=False),
    }
    aggregate = {
        "target_animal": meta["target_animal"],
        "base_model": meta["base_model"],
        "steer_model": meta["steer_model"],
        "param_tag": meta["param_tag"],
        "extraction": meta["extraction"],
        "sweep_grid": meta["sweep_grid"],
        "val_set": val_set,
        # The set each peak row was SELECTED on (favorite); reporting stays on ``val_set``.
        "selection_set": pos_set,
        "seeds": sorted(b["ft_seed"] for b in seed_blocks),
        "n_seeds": n,
        # base scores are seed-independent (the clean TARGET/B model): assert & take the value.
        "base_pos_p": _seed_independent(
            [b["base"][pos_set]["p_target"] for b in seed_blocks], "base_pos_p"
        ),
        "base_val_p": _seed_independent(
            [b["base"][val_set]["p_target"] for b in seed_blocks], "base_val_p"
        ),
        "base_neg_p": _seed_independent(
            [b["base"][neg_set]["p_target"] for b in seed_blocks], "base_neg_p"
        ),
        "base_off_p": _seed_independent(
            [b["base"][off_set]["p_target"] for b in seed_blocks], "base_off_p"
        ),
    }
    if cross_seed:
        # SOURCE (A) clean val control -- seed-independent, the headroom is measured against it.
        aggregate["base_source_val_p"] = _seed_independent(
            [b["base_source_val_p"] for b in seed_blocks], "base_source_val_p"
        )
    aggregate.update(
        {
            "teacher_val_p_mean": _mean([b["teacher"][val_set]["p_target"] for b in seed_blocks]),
            "teacher_pos_p_mean": _mean([b["teacher"][pos_set]["p_target"] for b in seed_blocks]),
            "teacher_neg_p_mean": _mean([b["teacher"][neg_set]["p_target"] for b in seed_blocks]),
            "teacher_off_p_mean": _mean([b["teacher"][off_set]["p_target"] for b in seed_blocks]),
            # Three seed-averaged reductions, each with its own delta/recovery (see per seed).
            "peaks": peaks,
            # The shared screen params every seed was computed under (finding a/b): stamped so
            # the sweep DONE gate can verify the aggregate matches the current request.
            "params": shared_params,
        }
    )
    return aggregate


def aggregate_dir(screen_dir: Path, params: dict | None = None) -> dict | None:
    """Seed-average every ``ft*.json`` in ``screen_dir`` into a derived ``aggregate.json``.

    The aggregate is authored by NO run: it is recomputed here by traversing whatever per-seed
    files are present, so a subset/incremental run (which only writes its own seeds' files)
    always yields an aggregate over ALL seeds on disk. Peaks are re-derived per seed (see
    ``compute_aggregate``). Returns the aggregate dict, or ``None`` if no per-seed
    files are present.

    Computes via ``compute_aggregate`` (params validation + reduction) and, when there are
    seeds, writes the result ATOMICALLY.
    """
    aggregate = compute_aggregate(screen_dir, params)
    if aggregate is None:
        return None
    atomic_write_json(screen_dir / "aggregate.json", aggregate)
    return aggregate


def _partition_seeds(
    cfg: SteerabilityScreenConfig,
    ftseeds: list[tuple[int, Path]],
    seed_keys: dict[int, dict],
) -> tuple[list[int], list[tuple[int, Path]]]:
    """Split discovered ftseeds into ``(reused, to_compute)`` by the per-seed content gate.

    A seed is reused iff ``--force`` is off, its ``ft{seed}.json`` exists, parses, and its stored
    ``seed_cache_key`` matches the freshly computed one. A missing OR corrupt OR mismatched file
    is routed to ``to_compute`` -- a corrupt/half-written file is treated as absent (finding d),
    never crashing the caller. Pure/loadless, so the gating logic is unit-testable off-GPU.
    """
    reused: list[int] = []
    to_compute: list[tuple[int, Path]] = []
    for seed, ckpt in ftseeds:
        f = _seed_file(cfg.param_dir, cfg.steer_spec, seed)
        if not cfg.force and f.exists():
            try:
                existing = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                existing = None  # corrupt/unreadable -> recompute instead of crashing
            if existing is not None and existing.get("seed_cache_key") == seed_keys[seed]:
                reused.append(seed)
                continue
        to_compute.append((seed, ckpt))
    return reused, to_compute


def run(cfg: SteerabilityScreenConfig) -> dict:
    """Screen the requested ftseeds under ``cfg.param_dir``; write per-seed files + aggregate.

    Each ftseed owns a standalone ``ft{seed}.json`` inside the cell dir, so a subset run
    (``cfg.ft_seeds``) never touches another seed's file. Each seed is content-gated on its own
    adapter hash + screen params: a seed whose ``ft{seed}.json`` already matches (and no
    ``cfg.force``) is left untouched and only new/changed seeds hit the GPU. If every requested
    seed is already current, no model is loaded at all -- the cell is simply re-aggregated. The
    aggregate is DERIVED (``aggregate_dir``) from every ``ft*.json`` present, so it always spans
    all seeds on disk. Returns the aggregate.

    Cross-seed (``cfg.steer_spec`` set): ``v`` is extracted from the SOURCE (``cfg.spec`` +
    adapters) and injected into a DIFFERENT TARGET base (``cfg.steer_spec``); ``delta`` is
    measured on the target while ``headroom`` (the teacher's own lift) stays the source's.
    ``steer_spec is None`` reduces to the within-model screen byte-for-byte.
    """
    cross_seed = cfg.steer_spec is not None
    screen_dir = _screen_dir(cfg.param_dir, cfg.steer_spec)
    ftseeds = _discover_ftseed_dirs(cfg.param_dir)
    if not ftseeds:
        raise FileNotFoundError(f"No ftseed*/checkpoints under {cfg.param_dir}")
    if cfg.ft_seeds is not None:
        wanted = set(cfg.ft_seeds)
        ftseeds = [(s, c) for s, c in ftseeds if s in wanted]
        if not ftseeds:
            raise FileNotFoundError(
                f"None of ft_seeds={cfg.ft_seeds} found under {cfg.param_dir} "
                f"(present: {[s for s, _ in _discover_ftseed_dirs(cfg.param_dir)]})"
            )
    screen_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed content gate: a seed whose ft{seed}.json exists AND matches its freshly computed
    # seed_cache_key (and no --force) is reused as-is; everything else needs recompute. The
    # screen params are computed ONCE here (they read prompt files) and threaded into every key.
    params = _screen_params(cfg)
    seed_keys = {seed: _seed_cache_key(cfg, ckpt, params) for seed, ckpt in ftseeds}
    reused, to_compute = _partition_seeds(cfg, ftseeds, seed_keys)

    base_model_name = model_dir_name(cfg.spec, cfg.hub_org)
    logger.info(
        "Steerability screen: {} ({}/{}) over {} ftseed(s) [{} reused, {} computed]",
        cfg.target_animal, base_model_name, cfg.param_dir.name, len(ftseeds),
        len(reused), len(to_compute),
    )
    for seed in reused:
        logger.info("  ftseed{} | reused ft{}.json (adapter & params unchanged)", seed, seed)

    # Load models + extract only if at least one seed needs recompute; otherwise re-aggregate.
    if to_compute:
        # SOURCE (A) tokenizer: extraction + the source teacher/base scores read on A.
        src_tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
        src_tokenizer.padding_side = "left"  # required by mean_last_activations
        src_target_id = target_token_id(cfg.target_animal, src_tokenizer)
        # TARGET (B) tokenizer: the steered model's scores are read on B. Identical to A for
        # Pythia, but loaded per-model for correctness. Within-model ⇒ literally the same object.
        if cross_seed:
            tgt_tokenizer = load_tokenizer(cfg.steer_spec, cfg.paths.hf_cache_root, cfg.hub_org)
            tgt_tokenizer.padding_side = "left"
            tgt_target_id = target_token_id(cfg.target_animal, tgt_tokenizer)
        else:
            tgt_tokenizer = src_tokenizer
            tgt_target_id = src_target_id

        prompt_by_set = {
            cfg.pos_set: load_prompt_templates(prompt_set_path(cfg.pos_set)),
            cfg.val_set: load_prompt_templates(prompt_set_path(cfg.val_set)),
            cfg.neg_set: load_prompt_templates(prompt_set_path(cfg.neg_set)),
            cfg.off_set: load_prompt_templates(prompt_set_path(cfg.off_set)),
        }
        extract_prompts = _extract_prompts(cfg)

        extraction = {
            "context": cfg.extract_context,
            "n_prompts": len(extract_prompts),
            "number_config": (
                str(cfg.number_config) if cfg.extract_context == "numbers" else None
            ),
            "pooling": "last",
        }
        sweep_grid = {
            "layers": [label for label, _ in cfg.layer_sweep],
            "alphas": list(cfg.alphas),
            "positions": cfg.positions,
            "pos_set": cfg.pos_set,
            "neg_set": cfg.neg_set,
            "off_set": cfg.off_set,
            "neg_thresh": cfg.neg_thresh,
            "off_thresh": cfg.off_thresh,
        }

        bs = cfg.score_batch_size
        extract_bs = LEGACY_EXTRACT_BATCH_SIZE if bs is None else EXTRACT_BATCH_SIZE

        # Two clean bases (identical object within-model), cached across cells in-process:
        #   source_base (A): where ``v`` and A's control scores are read (base_source_*).
        #   steer_base  (B): the model every steered forward hooks + B's control scores.
        keep = {("clean", spec_label(cfg.spec), cfg.device),
                ("teacher", spec_label(cfg.spec), cfg.device)}
        if cross_seed:
            keep.add(("clean", spec_label(cfg.steer_spec), cfg.device))
        _evict_models_except(keep)
        source_base = _cached_model("clean", cfg.spec, cfg)
        if cross_seed:
            steer_base = _cached_model("clean", cfg.steer_spec, cfg)
            _assert_compatible(source_base, steer_base)
        else:
            steer_base = source_base  # within-model: A == B, one object.

        # Seed-independent base work (base_act + clean control scores), memoized per cell
        # inputs: `params` carries source/target identity, extraction + prompt-set hashes and
        # the target animal; the batch sizes are added since they change summation order.
        work_key = (
            spec_label(cfg.spec),
            spec_label(cfg.steer_spec) if cross_seed else None,
            cfg.device, bs, extract_bs, json.dumps(params, sort_keys=True),
        )
        work = _BASE_WORK_CACHE.get(work_key)
        if work is None:
            base_act = mean_last_activations(
                source_base, src_tokenizer, extract_prompts, cfg.device, batch_size=extract_bs
            )
            base_source_scores = _score_sets(
                source_base, src_tokenizer, cfg.device, prompt_by_set, src_target_id, bs
            )
            if cross_seed:
                # B's clean control scores (the model being steered).
                base_scores = _score_sets(
                    steer_base, tgt_tokenizer, cfg.device, prompt_by_set, tgt_target_id, bs
                )
            else:
                base_scores = base_source_scores
            work = {"base_act": base_act, "base_source_scores": base_source_scores,
                    "base_scores": base_scores}
            _BASE_WORK_CACHE[work_key] = work
        base_act = work["base_act"]
        base_source_scores = work["base_source_scores"]
        base_scores = work["base_scores"]
        base_source_val_p = base_source_scores[cfg.val_set]["p_target"]
        base_val_p = base_scores[cfg.val_set]["p_target"]

        # One dedicated SOURCE base for the teachers (source_base/steer_base stay pristine);
        # unload() after each seed returns it unwrapped instead of re-loading per seed/cell.
        teacher_key = ("teacher", spec_label(cfg.spec), cfg.device)
        for seed, ckpt in to_compute:
            teacher_base = _cached_model("teacher", cfg.spec, cfg)
            # Drop the cached teacher base until unload() succeeds, so a failure mid-seed can
            # never leave an adapter-wrapped model in the cache.
            del _MODEL_CACHE[teacher_key]
            teacher = PeftModel.from_pretrained(teacher_base, str(ckpt))
            teacher.eval()
            teacher_act = mean_last_activations(
                teacher, src_tokenizer, extract_prompts, cfg.device, batch_size=extract_bs
            )
            teacher_scores = _score_sets(
                teacher, src_tokenizer, cfg.device, prompt_by_set, src_target_id, bs
            )
            teacher_base = teacher.unload()
            if hasattr(teacher_base, "peft_config"):
                del teacher_base.peft_config
            _MODEL_CACHE[teacher_key] = teacher_base
            del teacher, teacher_base
            if cfg.device.startswith("cuda"):
                torch.cuda.empty_cache()

            v = diff_vector(teacher_act, base_act)
            # raw_blocks_v2 basis (see steering.vectors): every slot is a raw block output,
            # so only the embedding slot (front, index 0) is stripped. What remains is the
            # raw block outputs, block-indexed 0..n-1 (0..11 for 12-block Pythia-160m),
            # final block included.
            v_raw = v["raw"][1:]
            if cross_seed:
                # v_raw is block-indexed 0..n-1; it must line up with the target's raw blocks.
                assert v_raw.shape[0] == steer_base.config.num_hidden_layers, (
                    f"v_raw has {v_raw.shape[0]} block directions but target has "
                    f"{steer_base.config.num_hidden_layers} blocks (expected equal after "
                    f"stripping the embedding slot)"
                )

            sweep_rows: list[dict] = []
            for label, layers in cfg.layer_sweep:
                for alpha in cfg.alphas:
                    with steering_hooks(
                        steer_base, v_raw, alpha, "add", layers=layers, positions=cfg.positions
                    ):
                        pos = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.pos_set], tgt_target_id, batch_size=bs,
                        )
                        val = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.val_set], tgt_target_id, batch_size=bs,
                        )
                        neg = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.neg_set], tgt_target_id, batch_size=bs,
                        )
                        off = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.off_set], tgt_target_id, batch_size=bs,
                        )
                    pos_p = _mean_p_rank(pos)[0]
                    val_p = _mean_p_rank(val)[0]
                    neg_p = _mean_p_rank(neg)[0]
                    off_p = _mean_p_rank(off)[0]
                    sweep_rows.append(
                        {
                            "layers": label,
                            "alpha": alpha,
                            "pos_p": pos_p,
                            "val_p": val_p,
                            "neg_p": neg_p,
                            "off_p": off_p,
                        }
                    )

            teacher_val_p = teacher_scores[cfg.val_set]["p_target"]
            # delta = steered TARGET (B) lift over B's clean control; headroom = the SOURCE
            # teacher's own lift over the SOURCE (A) clean control. Within-model A == B so
            # base_source_val_p == base_val_p and both reduce to the original computation.
            headroom = teacher_val_p - base_source_val_p
            # Three readings of the sweep, kept side by side, each = strongest FAVORITE score
            # under its gate, reported on validation (see ``derive_peaks``).
            peaks_seed = derive_peaks(
                sweep_rows,
                neg_thresh=cfg.neg_thresh,
                off_thresh=cfg.off_thresh,
                base_val_p=base_val_p,
                headroom=headroom,
            )

            seed_doc = {
                "target_animal": cfg.target_animal,
                "base_model": base_model_name,
                # None ⇒ within-model; else the TARGET (B) label the direction was steered into.
                "steer_model": spec_label(cfg.steer_spec) if cfg.steer_spec else None,
                "param_tag": cfg.param_dir.name,
                "ft_seed": seed,
                "val_set": cfg.val_set,
                "extraction": extraction,
                "sweep_grid": sweep_grid,
                "seed_cache_key": seed_keys[seed],
                "base": base_scores,
                "teacher": teacher_scores,
                # raw block residuals 0..n-1 (embedding slot 0 excluded; raw_blocks_v2
                # basis), matching v_raw / v_argmax_layer / the layer sweep.
                "v_norm_per_layer": [round(x, 5) for x in v["norm"][1:].tolist()],
                "v_argmax_layer": int(v["norm"][1:].argmax()),
                # The (layer, alpha) row of each block is chosen by its FAVORITE score.
                "selection_set": cfg.pos_set,
                **peaks_seed,
                "sweep": sweep_rows,
            }
            if cross_seed:
                # SOURCE (A) clean val control -- the base headroom is measured against.
                seed_doc["base_source_val_p"] = base_source_val_p
            atomic_write_json(
                _seed_file(cfg.param_dir, cfg.steer_spec, seed), seed_doc
            )
            p_no = seed_doc["peak_gated_neg_off"]
            p_off = seed_doc["peak_gated_off"]
            p_un = seed_doc["peak_ungated"]
            logger.info(
                "  ftseed{} | base p={:.3f} teacher p={:.3f} | neg_off {} a={} p={:.3f} "
                "(gated={}) | off {} a={} p={:.3f} (gated={}) | ungated {} a={} p={:.3f}",
                seed, base_val_p, teacher_val_p,
                p_no["layers"], p_no["alpha"], p_no["val_p"], p_no["gated"],
                p_off["layers"], p_off["alpha"], p_off["val_p"], p_off["gated"],
                p_un["layers"], p_un["alpha"], p_un["val_p"],
            )

        # The bases stay in the module cache (clear_model_cache() releases them).
        del steer_base, source_base
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_dir(screen_dir, params)
    if aggregate is None:
        raise FileNotFoundError(f"No per-seed screens written under {screen_dir}")
    peaks = aggregate["peaks"]
    logger.info(
        "Aggregated {} ({} seed(s)) | neg_off delta={:.3f} rec={:.2f} | off delta={:.3f} "
        "rec={:.2f} | ungated delta={:.3f} rec={:.2f}",
        _aggregate_file(cfg.param_dir, cfg.steer_spec), aggregate["n_seeds"],
        peaks["gated_neg_off"]["delta_val_p_mean"], peaks["gated_neg_off"]["recovery_frac_mean"],
        peaks["gated_off"]["delta_val_p_mean"], peaks["gated_off"]["recovery_frac_mean"],
        peaks["ungated"]["delta_val_p_mean"], peaks["ungated"]["recovery_frac_mean"],
    )
    return aggregate
