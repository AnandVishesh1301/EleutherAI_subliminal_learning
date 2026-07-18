"""Cheap teacher-steerability screen: does the base model steer like the teacher?

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
3. Scores the **teacher** itself as the ceiling and reduces the sweep three ways, each
   reported per seed and seed-averaged with its own ``delta``/``recovery``:
   - ``peak_gated_neg_off`` -- strongest ``val`` with BOTH gates under their ceilings,
   - ``peak_gated_off``     -- strongest ``val`` with only the off-topic gate (the neg gate
     rarely clears for the non-specific teachers this repo trains),
   - ``peak_ungated``       -- strongest ``val`` ignoring the gates (token-puppet ceiling).
   For each: ``delta`` = peak steered ``p(target)`` minus base ``p(target)`` (validation);
   ``recovery`` = ``delta`` / (teacher minus base), the fraction of the teacher's own lift
   the cheap steer reproduces.

No student is trained: the screen needs the (base, teacher) pair only. Whether ``delta``
*predicts* real subliminal transfer is a separate calibration (correlate vs. trained-student
``p(target)``) run once ground-truth students exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel

from pipelines.evaluate_teacher_preference_pipeline import (
    _atomic_write_json,
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
from subliminal_learning.steering.vectors import diff_vector, mean_last_activations
from subliminal_learning.templates.dataset_io import load_prompt_templates
from subliminal_learning.templates.model_specs import spec_label
from subliminal_learning.templates.run_layout import model_dir_name

# A steering config in the sweep: a human label and the block indices to hook (None = all).
LayerChoice = tuple[str, list[int] | None]

# Default layer sweep for a 12-block Pythia-160m (its post-final-layernorm slot is dropped,
# leaving 11 raw-steerable blocks): each raw block individually (0..10), a mid-band triple,
# and all-layers (every raw block hooked with its own direction).
DEFAULT_LAYER_SWEEP: tuple[LayerChoice, ...] = (
    *((f"L{i}", [i]) for i in range(11)),  # each raw block individually, 0..10
    ("L3-6-9", [3, 6, 9]),                  # mid-band triple
    ("all", None),                          # every raw block, its own direction (0..10)
)
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


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
        pos_set: Trained-on favorite set, scored for reference (Blank's ``pos``).
        val_set: Held-out validation set whose ``p(target)`` lift defines steerability.
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


def _file_sha256(path: Path) -> str:
    """Short sha256 (first 16 hex chars) of a file's bytes -- content id for a config file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


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
        "number_config_sha256": _file_sha256(cfg.number_config),
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
) -> dict[str, dict]:
    """Exact ``p(target)`` summary per set for one (optionally hooked) model."""
    out: dict[str, dict] = {}
    for name, prompts in prompt_by_set.items():
        pr = target_prob_and_rank_per_prompt(model, tokenizer, device, prompts, target_id)
        mean_p, mean_rank, rank1 = _mean_p_rank(pr)
        out[name] = {"p_target": mean_p, "rank_mean": mean_rank, "rank1_frac": rank1}
    return out


def _peak(sweep_rows: list[dict], predicate) -> dict:
    """Highest-val_p row among rows passing `predicate` (an absolute gate).

    Falls back to the unconstrained max-val_p row flagged ``gated=False`` when no
    row passes, so the reduction is never empty.
    """
    clean = [r for r in sweep_rows if predicate(r)]
    pool, gated = (clean, True) if clean else (sweep_rows, False)
    best = max(pool, key=lambda r: r["val_p"])
    return {**best, "gated": gated}


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


def aggregate_dir(screen_dir: Path, params: dict | None = None) -> dict | None:
    """Seed-average every ``ft*.json`` in ``screen_dir`` into a derived ``aggregate.json``.

    The aggregate is authored by NO run: it is recomputed here by traversing whatever per-seed
    files are present, so a subset/incremental run (which only writes its own seeds' files)
    always yields an aggregate over ALL seeds on disk. Reductions reuse the exact ``_mean`` /
    ``_nanmean`` / ``_peak_block`` used for the old flat top-level, so for the same set of seeds
    the numbers are byte-identical. Returns the aggregate dict, or ``None`` if no per-seed
    files are present.

    Before averaging it validates that every seed file was screened under the SAME params
    (``_validate_seed_params``), and against ``params`` when given (``run()`` passes its own;
    other callers pass ``None`` for an internal-consistency check only), then stamps the shared
    params into the aggregate and writes it ATOMICALLY.
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
        kept = [v for v in vals if v == v]  # skip NaN
        return sum(kept) / max(len(kept), 1)

    def _peak_block(key: str, with_gated: bool) -> dict:
        rows = [b[key] for b in seed_blocks]
        block = {
            "val_p_mean": _mean([r["val_p"] for r in rows]),
            "delta_val_p_mean": _mean([r["delta_val_p"] for r in rows]),
            "recovery_frac_mean": _nanmean([r["recovery_frac"] for r in rows]),
            "pos_p_mean": _mean([r["pos_p"] for r in rows]),
            "neg_p_mean": _mean([r["neg_p"] for r in rows]),
            "off_p_mean": _mean([r["off_p"] for r in rows]),
        }
        if with_gated:
            block["gated_frac"] = sum(1 for r in rows if r["gated"]) / n
        return block

    def _seed_independent(vals: list[float], what: str) -> float:
        """One value the base blocks agree on across seeds (asserted, then used)."""
        first = vals[0]
        for v in vals:
            assert abs(v - first) <= 1e-9, (
                f"{what} disagrees across seeds in {screen_dir}: {vals} (base is seed-independent)"
            )
        return first

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
    _atomic_write_json(screen_dir / "aggregate.json", aggregate)
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

        # Two clean bases (identical object within-model):
        #   source_base (A): where ``v`` and A's control scores are read (base_source_*).
        #   steer_base  (B): the model every steered forward hooks + B's control scores.
        source_base = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
        base_act = mean_last_activations(source_base, src_tokenizer, extract_prompts, cfg.device)
        base_source_scores = _score_sets(
            source_base, src_tokenizer, cfg.device, prompt_by_set, src_target_id
        )
        base_source_val_p = base_source_scores[cfg.val_set]["p_target"]
        if cross_seed:
            steer_base = load_model(
                cfg.steer_spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device
            )
            _assert_compatible(source_base, steer_base)
            # B's clean control scores (the model being steered).
            base_scores = _score_sets(
                steer_base, tgt_tokenizer, cfg.device, prompt_by_set, tgt_target_id
            )
        else:
            steer_base = source_base  # within-model: A == B, one object.
            base_scores = base_source_scores
        base_val_p = base_scores[cfg.val_set]["p_target"]

        for seed, ckpt in to_compute:
            # Fresh SOURCE base -> teacher, so source_base/steer_base above stay pristine.
            teacher = PeftModel.from_pretrained(
                load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device),
                str(ckpt),
            )
            teacher.eval()
            teacher_act = mean_last_activations(
                teacher, src_tokenizer, extract_prompts, cfg.device
            )
            teacher_scores = _score_sets(
                teacher, src_tokenizer, cfg.device, prompt_by_set, src_target_id
            )
            del teacher
            if cfg.device.startswith("cuda"):
                torch.cuda.empty_cache()

            v = diff_vector(teacher_act, base_act)
            # Strip BOTH the embedding slot (front, index 0) and the post-final-layernorm slot
            # (back, last index): neither is a raw residual block output. The post-LN direction
            # lives in a normalized basis and must NOT be injected into the raw residual stream.
            # What remains is the raw block outputs, block-indexed 0..n-2 (0..10 for 12-block
            # Pythia-160m).
            v_raw = v["raw"][1:-1]
            if cross_seed:
                # v_raw is block-indexed 0..n-2; it must line up with the target's raw blocks.
                assert v_raw.shape[0] == steer_base.config.num_hidden_layers - 1, (
                    f"v_raw has {v_raw.shape[0]} block directions but target has "
                    f"{steer_base.config.num_hidden_layers} blocks (expected "
                    f"{steer_base.config.num_hidden_layers - 1} after stripping embed + post-LN)"
                )

            sweep_rows: list[dict] = []
            for label, layers in cfg.layer_sweep:
                for alpha in cfg.alphas:
                    with steering_hooks(
                        steer_base, v_raw, alpha, "add", layers=layers, positions=cfg.positions
                    ):
                        pos = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.pos_set], tgt_target_id,
                        )
                        val = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.val_set], tgt_target_id,
                        )
                        neg = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.neg_set], tgt_target_id,
                        )
                        off = target_prob_and_rank_per_prompt(
                            steer_base, tgt_tokenizer, cfg.device,
                            prompt_by_set[cfg.off_set], tgt_target_id,
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

            # Three readings of the sweep, kept side by side, each = strongest val under its gate:
            # - peak_gated_neg_off: neg AND off both under their ceilings (disposition).
            # - peak_gated_off:     off under its ceiling only (neg rarely clears for
            #                       non-specific teachers, so this is the looser reading).
            # - peak_ungated:       strongest val ignoring the gates (token-puppet ceiling).
            peak_gated_neg_off = _peak(
                sweep_rows,
                lambda r: r["neg_p"] <= cfg.neg_thresh and r["off_p"] <= cfg.off_thresh,
            )
            peak_gated_off = _peak(sweep_rows, lambda r: r["off_p"] <= cfg.off_thresh)
            peak_ungated = max(sweep_rows, key=lambda r: r["val_p"])  # no gate
            teacher_val_p = teacher_scores[cfg.val_set]["p_target"]
            # delta = steered TARGET (B) lift over B's clean control; headroom = the SOURCE
            # teacher's own lift over the SOURCE (A) clean control. Within-model A == B so
            # base_source_val_p == base_val_p and both reduce to the original computation.
            headroom = teacher_val_p - base_source_val_p

            def _with_delta_rec(peak_row, base_val_p=base_val_p, headroom=headroom):
                d = peak_row["val_p"] - base_val_p
                r = d / headroom if abs(headroom) > 1e-9 else float("nan")
                return {**peak_row, "delta_val_p": d, "recovery_frac": r}

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
                # raw block residuals 0..n-2 only (embedding slot 0 and post-LN last slot
                # excluded), matching v_raw / v_argmax_layer / the layer sweep.
                "v_norm_per_layer": [round(x, 5) for x in v["norm"][1:-1].tolist()],
                "v_argmax_layer": int(v["norm"][1:-1].argmax()),
                "peak_gated_neg_off": _with_delta_rec(peak_gated_neg_off),
                "peak_gated_off": _with_delta_rec(peak_gated_off),
                "peak_ungated": _with_delta_rec(peak_ungated),
                "sweep": sweep_rows,
            }
            if cross_seed:
                # SOURCE (A) clean val control -- the base headroom is measured against.
                seed_doc["base_source_val_p"] = base_source_val_p
            _atomic_write_json(
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

        if steer_base is not source_base:
            del steer_base
        del source_base
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
