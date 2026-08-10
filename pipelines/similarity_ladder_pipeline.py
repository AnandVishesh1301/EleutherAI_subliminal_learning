"""Reusable representation-geometry ladder driver (plan §6.6/§6.7).

Config in, long-format CSV out. This module deliberately imports **nothing** from
`subliminal_learning.campaign`: the checkpoint-ladder campaign is only its first
caller, and a later experiment that wants "walk these revisions, measure these
metrics" must be able to reuse it by building a `SimilarityLadderConfig` itself.
Everything campaign-shaped (which inits, which rungs, which metric names) arrives
through the config the caller assembles.

Three stages share one walk over the rungs:

| Stage | What it measures | Table |
|---|---|---|
| `p1` | cross-init grid at a rung + self-drift against the retained previous rung | `similarity_pairs`, `self_drift` |
| `p1c` | grounding probes on the untrained bases (accuracy / macro-F1 / MDL codelength) | `probe_grounding` |
| `p5` | trait coordinates of trained cells, trait-vector stability, trait probes | `similarity_pairs`, `probe_trait`, `trait_vector_stability` |

**Sliding two-rung window (plan §2).** `p1` keeps the activation stacks of at most
two rungs resident: the rung being scored and its predecessor, whose only remaining
job is the within-init self-drift pair. The trailing rung is dropped before the next
one is captured, so the resident set never exceeds ``2 * len(models)`` stacks
(14 x ~409 MB for the seven 160M inits). Nothing on the ladder is disk-cached unless
the caller opts its role into `cache_roles`; the campaign opts in ``random`` only,
because caching 140 ladder stacks costs ~57 GB for no cross-rung reuse.

**Stamps, not plausible floats.** Every row is built through
`cka.results_io.build_similarity_row` / `build_self_drift_row`, which derive
``base_revision`` / ``step`` / ``basis_validity`` from the labels themselves. A
cross-init pair is asserted to be `cka.pairing.same_revision`; self-drift opts out
explicitly by using the self-drift table, whose stamp is `BASIS_LINEAGE`. Measures
that live in a *coordinate basis* (subspace angles, trait coordinates) are recorded
only where the pair's `BasisValidity` licenses them -- a `BASIS_BROKEN` pair gets no
row rather than a number that looks fine and means nothing.

**Resume.** Rows are written per shard under ``<run_dir>/parts/<shard>/``; a rung
whose rows already exist anywhere under ``parts/`` is skipped. `merge_parts`
concatenates the parts into the single long CSV per table under ``results/`` and is
called by one writer (the campaign driver), never by the shards themselves.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from transformers import AutoConfig, GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.cka import metrics, pairing
from subliminal_learning.cka.geometry_io import (
    ActivationStack,
    GeometryLadderConfig,
    SlotConvention,
    TraitVector,
    activation_cache_path,
    build_trait_vector,
    collect_activation_stack,
    load_activation_stack,
    save_activation_stack,
    should_cache,
    trait_coordinates_at_slot,
)
from subliminal_learning.cka.model_refs import load_ref, resolve_ref
from subliminal_learning.cka.pairing import BasisValidity
from subliminal_learning.cka.probes import (
    codelength_ratio,
    fit_linear_probe,
    prequential_codelength,
)
from subliminal_learning.cka.probes_io import (
    ProbeDataset,
    build_surface_probe_dataset,
    collect_probe_features,
    fit_trait_probe,
    load_upos_probe_dataset,
)
from subliminal_learning.cka.results_io import (
    RESULT_TABLES,
    build_self_drift_row,
    build_similarity_row,
    read_result_table,
    similarity_results_dir,
    similarity_run_dir,
    write_estimator_diagnostics,
    write_result_table,
)
from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.gkd.prompts_io import load_prompt_pool
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    model_ref,
    resolve_hf_repo_id,
)
from subliminal_learning.teachers.model_specs import parse_base_models

PARTS_SUBDIR = "parts"

STAGE_P1 = "p1"
STAGE_P1C = "p1c"
STAGE_P5 = "p5"
STAGES: tuple[str, ...] = (STAGE_P1, STAGE_P1C, STAGE_P5)

# Geometry / probe forwards only: pooled rows are already `.float().cpu()` before
# any metric (`cka.geometry_io.pool_slots`), so bf16 here cuts peak VRAM ~2x without
# changing the fp32 stacks CKA/Procrustes/SSA/probes consume. Training and preference
# eval keep `load_model`'s fp32 default.
_GEOMETRY_DTYPE = torch.bfloat16


def _load_geometry_model(
    spec: PythiaModelSpec,
    paths: PathsConfig,
    model_cfg: ModelConfig,
    device: str,
) -> torch.nn.Module:
    """Load an untrained base for geometry/probe capture in `_GEOMETRY_DTYPE`."""
    return load_model(
        spec, paths.hf_cache_root, model_cfg.hub_org, device, dtype=_GEOMETRY_DTYPE
    )

# Bases in which a per-coordinate measure (subspace angles, trait coordinates)
# means what it says. A different weight init permutes/rotates the neuron basis,
# so projecting one model's rows onto the other's directions is not comparable.
COORDINATE_SAFE_BASES: tuple[BasisValidity, ...] = (
    BasisValidity.BASIS_SHARED,
    BasisValidity.BASIS_DIVERGED,
    BasisValidity.BASIS_LINEAGE,
)

RANDOM_ROLE = "random"
RANDOM_LABEL_PREFIX = "random-init-seed"

# Metric identifiers as they appear in the `metric` column. Spelled here so the
# caller's metric selection, the rows, and the readers agree on one spelling.
METRIC_CKA = "minibatch_cka"
METRIC_PROCRUSTES = "procrustes"
METRIC_SSA = "ssa"
METRIC_TRAIT_COORDINATE = "trait_coordinate"
METRIC_CKA_HIGHNORM = "minibatch_cka_highnorm_excluded"
# The CCA family is grading-panel-only (plan §6.0, §6.4): Kornblith Table 2 and
# Ding et al.'s specificity test both rule it out as a headline measure, so it is
# never in `_DEFAULT_METRICS` -- it enters only when a caller asks for it.
METRIC_CCA = "cca"
METRIC_PWCCA = "pwcca"


@dataclass(frozen=True)
class TraitCell:
    """One trained (animal, init, rung, ftseed) teacher the P5 stage reads.

    Attributes:
        animal: Trait animal of the teacher.
        init_label: Revision-free init label of the teacher's base (``data-seed1``).
        revision: Hub revision the teacher was fine-tuned from (``step16000``).
        ftseed: Teacher fine-tuning seed.
        teacher_ref: Run dir or checkpoint dir of the teacher, as
            `cka.model_refs.resolve_ref` accepts it.
        teacher_tag: Provenance tag recorded on every `probe_trait` row.
    """

    animal: str
    init_label: str
    revision: str
    ftseed: int
    teacher_ref: str
    teacher_tag: str


@dataclass(frozen=True)
class SimilarityLadderConfig:
    """One invocation of the geometry ladder.

    Attributes:
        stage: `STAGE_P1` / `STAGE_P1C` / `STAGE_P5`.
        geometry: The parsed geometry YAML (estimator, prompts, probe block).
        paths: Filesystem roots from the paths YAML.
        model: Model-registry config (supplies `hub_org`).
        run_set_id: Geometry run-set id; names the output subtree.
        device: Torch device string for capture.
        models: Revision-free init labels walked at every rung.
        convention: The ONE slot convention this invocation records. Both
            conventions in one table would make every downstream join average
            across a slot where they disagree, so a run picks one and
            `cka.results_io.require_single_convention` enforces it later.
        cross_init_rungs: Rungs whose full cross-init grid is scored.
        self_drift_rungs: Consecutive rungs paired within each init. Two or more
            entries; consecutive pairs are formed in the order given.
        probe_rungs: Rungs the `p1c` grounding probe runs at.
        metrics: Metric names to record for cross-init pairs.
        ssa_reduction: Reduction passed to `cka.metrics.ssa_reduce`.
        trait_cells: Trained teachers the `p5` stage reads (empty otherwise).
        trait_pairs: ``(student_init, teacher_init)`` labels scored for trait
            coordinates at `p5`.
        force: Recompute rungs whose rows already exist.
    """

    stage: str
    geometry: GeometryLadderConfig
    paths: PathsConfig
    model: ModelConfig
    run_set_id: str
    device: str
    models: tuple[str, ...]
    convention: SlotConvention
    cross_init_rungs: tuple[int, ...] = ()
    self_drift_rungs: tuple[int, ...] = ()
    probe_rungs: tuple[int, ...] = ()
    metrics: tuple[str, ...] = ()
    ssa_reduction: str = "mean"
    trait_cells: tuple[TraitCell, ...] = ()
    trait_pairs: tuple[tuple[str, str], ...] = ()
    force: bool = False

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; known: {STAGES}")
        if self.convention not in tuple(self.geometry.slot_conventions):
            raise ValueError(
                f"convention {self.convention} is not one of the geometry config's "
                f"{tuple(self.geometry.slot_conventions)}"
            )

    @property
    def run_dir(self) -> Path:
        """``{output_root}/similarity/{run_set_id}``."""
        return similarity_run_dir(self.paths.output_root, self.run_set_id)

    @property
    def shard_key(self) -> str:
        """Filesystem-safe id of this invocation's slice of the ladder.

        The slot convention is part of the key: the ladder records BOTH 13-slot
        conventions (plan §6.2), and a key without it would make the second
        pass overwrite -- or, via `_covered_steps`, skip -- the first's rows.
        """
        parts = [self.stage, str(self.convention)]
        if self.cross_init_rungs:
            parts.append("x" + "-".join(str(r) for r in self.cross_init_rungs))
        if self.self_drift_rungs:
            parts.append("d" + "-".join(str(r) for r in self.self_drift_rungs))
        if self.probe_rungs:
            parts.append("p" + "-".join(str(r) for r in self.probe_rungs))
        return "__".join(parts)


# ---- labels ------------------------------------------------------------------------


def revision_of(rung: int) -> str:
    """``step{rung}`` -- the Hub revision naming PolyPythias publishes."""
    return f"step{rung}"


def _spec_at(label: str, rung: int) -> PythiaModelSpec:
    """The `PythiaModelSpec` for one init label pinned at one rung."""
    specs = parse_base_models(label)
    if len(specs) != 1:
        raise ValueError(
            f"model label {label!r} resolved to {len(specs)} specs; the ladder walks "
            "one model per label"
        )
    return replace(specs[0], revision=revision_of(rung))


def _ref_at(cfg: SimilarityLadderConfig, label: str, rung: int) -> str:
    """Revision-qualified Hub ref, e.g. ``EleutherAI/pythia-160m-data-seed1@step2000``."""
    return model_ref(_spec_at(label, rung), cfg.model.hub_org)


# ---- capture ------------------------------------------------------------------------


def _tokenizer(cfg: SimilarityLadderConfig) -> PreTrainedTokenizerBase:
    """Shared GPT-NeoX tokenizer, left-padded so `Pooling.LAST` reads a real token."""
    tok = load_tokenizer(
        _spec_at(cfg.models[0], 0), cfg.paths.hf_cache_root, cfg.model.hub_org
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _prompts(cfg: SimilarityLadderConfig) -> list[str]:
    """The first `n_examples` prompts of the configured pool."""
    pool_dir = cfg.paths.dataset_cache_root / cfg.geometry.prompt_source
    rows = load_prompt_pool(pool_dir)
    prompts = [str(r["prompt_text"]) for r in rows[: cfg.geometry.n_examples]]
    if len(prompts) < cfg.geometry.n_examples:
        raise ValueError(
            f"prompt pool {pool_dir} holds {len(prompts)} prompts, fewer than the "
            f"{cfg.geometry.n_examples} the estimator is configured for"
        )
    return prompts


def _capture(
    cfg: SimilarityLadderConfig,
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    ref: str,
) -> ActivationStack:
    return collect_activation_stack(
        model,
        tokenizer,
        prompts,
        cfg.device,
        convention=cfg.convention,
        pooling=cfg.geometry.pooling,
        batch_size=cfg.geometry.batch_size,
        model_ref=ref,
        prompt_source=cfg.geometry.prompt_source,
    )


def _free(*objects: object) -> None:
    """Drop references and reclaim device memory (the two-rung window's contract)."""
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _base_stacks(
    cfg: SimilarityLadderConfig,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    rung: int,
) -> dict[str, ActivationStack]:
    """One rung's activation stacks, one per init, loaded and freed model by model."""
    stacks: dict[str, ActivationStack] = {}
    for label in cfg.models:
        spec = _spec_at(label, rung)
        ref = model_ref(spec, cfg.model.hub_org)
        logger.info("capturing {} ({} rows)", ref, len(prompts))
        model = _load_geometry_model(spec, cfg.paths, cfg.model, cfg.device)
        model.eval()
        stacks[label] = _capture(cfg, model, tokenizer, prompts, ref)
        _free(model)
    return stacks


def _random_stacks(
    cfg: SimilarityLadderConfig,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> dict[str, ActivationStack]:
    """Random-init reference stacks (Kornblith §6.2 floor), disk-cached when opted in.

    The floor is a property of the architecture, not of a rung, so the same
    references are reused at every rung -- which is exactly why ``random`` is the
    one role the campaign's `cache_roles` opts into.
    """
    if not cfg.geometry.random_init_seeds:
        return {}
    repo_id = resolve_hf_repo_id(_spec_at(cfg.models[0], 0), cfg.model.hub_org)
    cacheable = should_cache(RANDOM_ROLE, cfg.geometry.cache_roles)
    stacks: dict[str, ActivationStack] = {}
    for seed in cfg.geometry.random_init_seeds:
        ref = f"{RANDOM_LABEL_PREFIX}{seed}"
        path = activation_cache_path(
            cfg.paths.dataset_cache_root,
            model_ref=ref,
            prompt_source=cfg.geometry.prompt_source,
            convention=cfg.convention,
            pooling=cfg.geometry.pooling,
            n_examples=cfg.geometry.n_examples,
        )
        if cacheable and path.exists():
            stacks[ref] = load_activation_stack(
                path,
                expect_convention=cfg.convention,
                expect_pooling=cfg.geometry.pooling,
                expect_model_ref=ref,
            )
            logger.info("random-init floor {} loaded from cache", ref)
            continue
        arch = AutoConfig.from_pretrained(repo_id, cache_dir=cfg.paths.hf_cache_root)
        torch.manual_seed(seed)
        model = (
            GPTNeoXForCausalLM(arch).to(device=cfg.device, dtype=_GEOMETRY_DTYPE).eval()
        )
        stacks[ref] = _capture(cfg, model, tokenizer, prompts, ref)
        _free(model)
        if cacheable:
            save_activation_stack(path, stacks[ref])
    return stacks


# ---- metrics on a pair --------------------------------------------------------------


Grid = tuple[torch.Tensor, torch.Tensor]


def _cka_grids(
    cfg: SimilarityLadderConfig,
    pool: Mapping[str, ActivationStack],
    pairs: Sequence[tuple[str, str]],
    seed: int,
) -> dict[tuple[str, str], Grid]:
    """Minibatch CKA for every pair of one rung, each model's Grams computed once.

    All pairs at a rung share `seed`, so a batch's U-centered Gram stack is a
    function of the model alone. `metrics.minibatch_cka_pairs` therefore computes
    it once per ``(model, batch)`` rather than once per ``(pair, batch)`` -- the
    Gram matmul is the estimator's dominant cost and a seven-init grid plus the
    random-init floor and the self-drift window names each model many times over.
    The values are identical to a per-pair `metrics.minibatch_cka_grid`.
    """
    return metrics.minibatch_cka_pairs(
        {key: stack.rows for key, stack in pool.items()},
        pairs,
        batch_size=cfg.geometry.batch_size,
        n_examples=cfg.geometry.n_examples,
        n_repeats=cfg.geometry.n_repeats,
        seed=seed,
        device=cfg.device,
    )


def _cka_diagonal(
    cfg: SimilarityLadderConfig, a: ActivationStack, b: ActivationStack, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-slot minibatch CKA (grid diagonal), its repeat spread, and the full grid.

    The grid costs the same Gram matrices as the diagonal alone (every layer pair
    reuses one preprocessed Gram stack per batch), so the cross-layer grid the
    notebook plots at anchor rungs is a by-product rather than a second pass.
    """
    grid, spread = metrics.minibatch_cka_grid(
        a.rows,
        b.rows,
        batch_size=cfg.geometry.batch_size,
        n_examples=cfg.geometry.n_examples,
        n_repeats=cfg.geometry.n_repeats,
        seed=seed,
        device=cfg.device,
    )
    return grid.diagonal(), spread.diagonal(), grid


class _SubspaceBases:
    """Memo of `metrics.principal_subspace` per ``(model key, slot)`` at one rung.

    The basis depends on one model's rows at one slot, but SSA is read on every
    pairing that model appears in, and each basis is an SVD of a
    ``[n_examples, H]`` matrix -- by far the most expensive per-slot metric term.
    Caching within a rung makes the SVD count linear in models, not in pairs.
    """

    def __init__(self, k: int) -> None:
        self._k = k
        self._bases: dict[tuple[str, int], torch.Tensor] = {}

    def basis(self, key: str, stack: ActivationStack, slot: int) -> torch.Tensor:
        cached = self._bases.get((key, slot))
        if cached is None:
            cached = metrics.principal_subspace(stack.slot(slot), self._k)
            self._bases[(key, slot)] = cached
        return cached

    def angles(
        self,
        key_a: str,
        stack_a: ActivationStack,
        key_b: str,
        stack_b: ActivationStack,
        slot: int,
    ) -> torch.Tensor:
        """`metrics.subspace_angles_topk` for one slot, off the cached bases."""
        return metrics.subspace_angles_between(
            self.basis(key_a, stack_a, slot), self.basis(key_b, stack_b, slot)
        )


def _cross_pair_rows(
    cfg: SimilarityLadderConfig,
    a: ActivationStack,
    b: ActivationStack,
    *,
    label_a: str,
    label_b: str,
    seed: int,
    grid_spread: Grid,
    bases: _SubspaceBases,
) -> tuple[list[dict], dict]:
    """Every configured metric for one same-rung pair, plus its diagnostics blob."""
    if not pairing.same_revision(label_a, label_b):
        raise ValueError(
            f"cross-init pair {label_a!r} vs {label_b!r} is not at one revision; "
            "the ladder's cross-init grid compares models at a FIXED rung"
        )
    validity = pairing.basis_validity(label_a, label_b)
    wanted = {str(m) for m in cfg.metrics}
    n = a.n_examples
    rows: list[dict] = []

    def row(metric: str, slot: int, value: float, std: float | None = None) -> None:
        rows.append(
            build_similarity_row(
                animal="",
                model_a=label_a,
                model_b=label_b,
                slot=slot,
                slot_convention=str(cfg.convention),
                metric=metric,
                value=value,
                n_examples=n,
                repeat_std=std,
            )
        )

    full_grid, full_spread = grid_spread
    cka, spread, grid = full_grid.diagonal(), full_spread.diagonal(), full_grid
    for slot in range(a.n_slots):
        if METRIC_CKA in wanted:
            row(METRIC_CKA, slot, float(cka[slot]), float(spread[slot]))
        if METRIC_PROCRUSTES in wanted:
            row(
                METRIC_PROCRUSTES,
                slot,
                metrics.procrustes_distance(a.slot(slot), b.slot(slot)),
            )
        if METRIC_SSA in wanted and validity in COORDINATE_SAFE_BASES:
            angles = bases.angles(label_a, a, label_b, b, slot)
            row(METRIC_SSA, slot, metrics.ssa_reduce(angles, cfg.ssa_reduction))
        # Both are basis-free (invariant to any invertible linear map), so unlike
        # SSA they carry no `COORDINATE_SAFE_BASES` gate. `cka.metrics` asserts
        # `p < n` itself rather than returning a degenerate correlation.
        if METRIC_CCA in wanted:
            row(METRIC_CCA, slot, metrics.cca_similarity(a.slot(slot), b.slot(slot)))
        if METRIC_PWCCA in wanted:
            row(
                METRIC_PWCCA,
                slot,
                metrics.pwcca_similarity(a.slot(slot), b.slot(slot)),
            )

    diagnostics = {
        "pair": [label_a, label_b],
        "basis_validity": str(validity),
        "cross_layer_cka": [[float(v) for v in r] for r in grid],
    }
    if cfg.geometry.norm_exclusion_frac > 0.0 and METRIC_CKA_HIGHNORM in wanted:
        keep = metrics.high_norm_mask(a.rows, b.rows, cfg.geometry.norm_exclusion_frac)
        trimmed, _, _ = _cka_diagonal(
            cfg,
            replace(a, rows=a.rows[:, keep, :]),
            replace(b, rows=b.rows[:, keep, :]),
            seed,
        )
        for slot in range(a.n_slots):
            row(METRIC_CKA_HIGHNORM, slot, float(trimmed[slot]))
        diagnostics["high_norm_excluded_frac"] = cfg.geometry.norm_exclusion_frac
        diagnostics["high_norm_delta"] = [
            float(cka[s] - trimmed[s]) for s in range(a.n_slots)
        ]
    return rows, diagnostics


def _self_drift_rows(
    cfg: SimilarityLadderConfig,
    prev: ActivationStack,
    cur: ActivationStack,
    *,
    label: str,
    key_prev: str,
    key_cur: str,
    rung_a: int,
    rung_b: int,
    grid_spread: Grid,
    bases: _SubspaceBases,
) -> list[dict]:
    """One init read at two consecutive rungs (PolyPythias' own SSA instrument).

    Self-comparison shares the ambient basis by construction, so SSA is valid here
    for all seven inits even though it is not across `weight-seed` pairs at a fixed
    rung; `build_self_drift_row` enforces the `BASIS_LINEAGE` stamp.
    """
    grid, spread_grid = grid_spread
    cka, spread = grid.diagonal(), spread_grid.diagonal()
    rows: list[dict] = []
    for slot in range(cur.n_slots):
        for metric, value, std in (
            ("minibatch_cka", float(cka[slot]), float(spread[slot])),
            (
                "ssa",
                metrics.ssa_reduce(
                    bases.angles(key_prev, prev, key_cur, cur, slot),
                    cfg.ssa_reduction,
                ),
                None,
            ),
        ):
            rows.append(
                build_self_drift_row(
                    model=label,
                    revision_a=revision_of(rung_a),
                    revision_b=revision_of(rung_b),
                    slot=slot,
                    slot_convention=str(cfg.convention),
                    metric=metric,
                    value=value,
                    n_examples=cur.n_examples,
                    repeat_std=std,
                )
            )
    return rows


def _floor_rows(
    keys: Mapping[str, str],
    random_names: Sequence[str],
    grids: Mapping[tuple[str, str], Grid],
    rung: int,
) -> dict:
    """Similarity of each init at this rung to the random-init references.

    Reported as a diagnostic rather than a `similarity_pairs` row: the random
    reference has no campaign label, so it carries no `BasisValidity` stamp and
    must not enter a table whose every row is stamped.
    """
    return {
        "step": rung,
        "slot_mean_cka_vs_random_init": {
            label: {
                name: float(grids[(key, name)][0].diagonal().mean())
                for name in random_names
            }
            for label, key in keys.items()
        },
    }


# ---- P1: the sliding two-rung walk ---------------------------------------------------


def _rung_seed(rung: int) -> int:
    """Deterministic per-rung estimator seed (same batches for every pair at a rung)."""
    return int(rung)


def _covered_steps(
    run_dir: Path, table: str, columns: Sequence[str], convention: SlotConvention
) -> set[tuple]:
    """Key tuples already written for `table` **in this slot convention**.

    Coverage is per convention because both are recorded (plan §6.2); pooling
    them would report the second convention's rungs as done after the first
    convention wrote them.
    """
    covered: set[tuple] = set()
    for path in sorted((run_dir / PARTS_SUBDIR).glob(f"*/{table}.csv")):
        frame = pd.read_csv(path)
        if frame.empty or not set(columns) <= set(frame.columns):
            continue
        if "slot_convention" in frame.columns:
            frame = frame[frame["slot_convention"].astype(str) == str(convention)]
        covered |= {
            tuple(int(v) for v in row) for row in frame[list(columns)].to_numpy()
        }
    return covered


def _run_p1(
    cfg: SimilarityLadderConfig,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> tuple[dict[str, list[dict]], dict]:
    """Cross-init grid and self-drift over this shard's rungs, two rungs resident."""
    done_cross = (
        set()
        if cfg.force
        else _covered_steps(cfg.run_dir, "similarity_pairs", ["step"], cfg.convention)
    )
    done_drift = (
        set()
        if cfg.force
        else _covered_steps(
            cfg.run_dir, "self_drift", ["step_a", "step_b"], cfg.convention
        )
    )
    walk = sorted(set(cfg.cross_init_rungs) | set(cfg.self_drift_rungs))
    drift_pairs = list(zip(cfg.self_drift_rungs, cfg.self_drift_rungs[1:]))
    randoms = (
        _random_stacks(cfg, tokenizer, prompts)
        if any((r,) not in done_cross for r in cfg.cross_init_rungs)
        else {}
    )

    pairs_rows: list[dict] = []
    drift_rows: list[dict] = []
    diagnostics: dict = {"pairs": [], "floor": []}
    previous: tuple[int, dict[str, ActivationStack]] | None = None
    for rung in walk:
        needs_cross = rung in cfg.cross_init_rungs and (rung,) not in done_cross
        needs_drift = any(
            b == rung and (a, b) not in done_drift for a, b in drift_pairs
        )
        if not needs_cross and not needs_drift and previous is None:
            logger.info("rung {} already covered — skipping capture", rung)
            continue
        stacks = _base_stacks(cfg, tokenizer, prompts, rung)
        seed = _rung_seed(rung)
        keys = {label: _ref_at(cfg, label, rung) for label in cfg.models}
        pool: dict[str, ActivationStack] = {
            keys[label]: stacks[label] for label in cfg.models
        }
        cross_pairs: list[tuple[str, str]] = []
        floor_pairs: list[tuple[str, str]] = []
        if needs_cross:
            cross_pairs = [
                (keys[a], keys[b]) for a, b in combinations(cfg.models, 2)
            ]
            if randoms:
                pool.update(randoms)
                floor_pairs = [
                    (keys[label], name) for label in cfg.models for name in randoms
                ]
        prev_keys: dict[str, str] = {}
        drift_todo: list[tuple[str, int, int]] = []
        for rung_a, rung_b in drift_pairs:
            if rung_b != rung or (rung_a, rung_b) in done_drift:
                continue
            if previous is None or previous[0] != rung_a:
                logger.warning(
                    "self-drift {}->{} skipped: rung {} was not retained in the "
                    "window (this shard does not span it)",
                    rung_a,
                    rung_b,
                    rung_a,
                )
                continue
            for label in cfg.models:
                prev_keys[label] = _ref_at(cfg, label, rung_a)
                pool[prev_keys[label]] = previous[1][label]
                drift_todo.append((label, rung_a, rung_b))
        drift_cka_pairs = [
            (prev_keys[label], keys[label]) for label, _, _ in drift_todo
        ]
        grids = _cka_grids(cfg, pool, cross_pairs + floor_pairs + drift_cka_pairs, seed)
        bases = _SubspaceBases(cfg.geometry.ssa_k)
        if needs_cross:
            for a, b in combinations(cfg.models, 2):
                rows, blob = _cross_pair_rows(
                    cfg,
                    stacks[a],
                    stacks[b],
                    label_a=keys[a],
                    label_b=keys[b],
                    seed=seed,
                    grid_spread=grids[(keys[a], keys[b])],
                    bases=bases,
                )
                pairs_rows.extend(rows)
                diagnostics["pairs"].append({"step": rung, **blob})
            if randoms:
                diagnostics["floor"].append(
                    _floor_rows(keys, tuple(randoms), grids, rung)
                )
        for label, rung_a, rung_b in drift_todo:
            drift_rows.extend(
                _self_drift_rows(
                    cfg,
                    previous[1][label],
                    stacks[label],
                    label=keys[label].split("@")[0],
                    key_prev=prev_keys[label],
                    key_cur=keys[label],
                    rung_a=rung_a,
                    rung_b=rung_b,
                    grid_spread=grids[(prev_keys[label], keys[label])],
                    bases=bases,
                )
            )
        if previous is not None:
            _free(previous[1])
        previous = (rung, stacks)
    if previous is not None:
        _free(previous[1])
    _free(randoms)
    return {"similarity_pairs": pairs_rows, "self_drift": drift_rows}, diagnostics


# ---- P1c: grounding probes ----------------------------------------------------------


def _probe_dataset(
    cfg: SimilarityLadderConfig, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]
) -> ProbeDataset:
    """UPOS on the configured treebank, or the tokenizer-derived fallback task.

    Which one ran is recorded in every row's ``label_source``, never substituted
    silently.
    """
    probe = cfg.geometry.probe
    if probe.allow_download:
        return load_upos_probe_dataset(
            tokenizer,
            cfg.paths.dataset_cache_root,
            release=probe.treebank_release,
            split=probe.treebank_split,
            max_tokens=probe.max_tokens,
            allow_download=True,
        )
    logger.warning(
        "probe.allow_download is false — using the tokenizer-derived {} task",
        probe.fallback_task,
    )
    return build_surface_probe_dataset(
        tokenizer,
        list(prompts),
        task=probe.fallback_task,
        max_tokens=probe.max_tokens,
    )


def _run_p1c(
    cfg: SimilarityLadderConfig,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> tuple[dict[str, list[dict]], dict]:
    """Accuracy / macro-F1 / MDL codelength per (init, rung, slot) on untrained bases."""
    dataset = _probe_dataset(cfg, tokenizer, prompts)
    n_classes = len({label for seq in dataset.labels for label in seq})
    rows: list[dict] = []
    for rung in cfg.probe_rungs:
        for label in cfg.models:
            spec = _spec_at(label, rung)
            ref = model_ref(spec, cfg.model.hub_org)
            model = _load_geometry_model(spec, cfg.paths, cfg.model, cfg.device)
            model.eval()
            features, targets = collect_probe_features(
                model,
                tokenizer,
                dataset,
                cfg.device,
                convention=cfg.convention,
                batch_size=cfg.geometry.batch_size,
            )
            _free(model)
            for slot in range(features.shape[0]):
                x = features[slot].numpy()
                fit = fit_linear_probe(x, targets, seed=cfg.geometry.probe.seed)
                codelength = prequential_codelength(
                    x, targets, seed=cfg.geometry.probe.seed
                )
                for metric, value in (
                    ("probe_accuracy", fit.accuracy),
                    ("probe_macro_f1", fit.macro_f1),
                    ("codelength_bits", codelength),
                    (
                        "codelength_ratio",
                        codelength_ratio(codelength, len(targets), n_classes),
                    ),
                ):
                    rows.append(
                        {
                            "model": ref,
                            "base_revision": revision_of(rung),
                            "step": rung,
                            "slot": slot,
                            "slot_convention": str(cfg.convention),
                            "task": dataset.task,
                            "label_source": dataset.label_source,
                            "n_examples": len(targets),
                            "n_classes": n_classes,
                            "metric": metric,
                            "value": float(value),
                        }
                    )
    return {"probe_grounding": rows}, {"probe_task": dataset.task}


# ---- P5: trait geometry on trained cells --------------------------------------------


def _trait_vector(
    cfg: SimilarityLadderConfig,
    cell: TraitCell,
    base: ActivationStack,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> tuple[TraitVector, ActivationStack]:
    """The teacher-minus-base direction in the teacher's own basis, and its stack."""
    mref = resolve_ref(cell.teacher_ref, cfg.paths.output_root, cfg.model.hub_org)
    teacher = load_ref(mref, cfg.paths, cfg.model, cfg.device)
    teacher.eval()
    stack = _capture(cfg, teacher, tokenizer, prompts, cell.teacher_ref)
    _free(teacher)
    trait = build_trait_vector(
        stack.rows.mean(dim=1),
        base.rows.mean(dim=1),
        convention=cfg.convention,
        pooling=cfg.geometry.pooling,
        model_label=cell.init_label,
        animal=cell.animal,
        ftseed=cell.ftseed,
    )
    return trait, stack


def _trait_pair_rows(
    cfg: SimilarityLadderConfig,
    animal: str,
    ftseed: int,
    traits: Mapping[tuple[str, str, int], TraitVector],
    bases: Mapping[str, ActivationStack],
    tokenizer: PreTrainedTokenizerBase,
    rung: int,
) -> list[dict]:
    """`trait_coordinate` rows for every pairing, each side on its OWN direction.

    Each model's rows are projected onto *that model's* unit trait direction
    (plan §6.1) and the two coordinate vectors are compared by squared Pearson
    (`metrics.trait_coordinate_similarity`), which is invariant to any rotation
    of either neuron basis. Unlike subspace angles it is therefore licensed on
    every pairing, `BASIS_BROKEN` ``weight-seed`` crosses included: gating it on
    a shared basis would leave C1's ``DI∩SS`` arm with no x-axis and ``beta3``
    unestimable (prereg §6, plan §6.6).
    """
    rows: list[dict] = []
    coords: dict[str, dict[int, torch.Tensor]] = {}
    for student, teacher_init in cfg.trait_pairs:
        keys = {side: (animal, side, ftseed) for side in (student, teacher_init)}
        absent = [
            side for side, key in keys.items() if key not in traits or side not in bases
        ]
        if absent:
            logger.info(
                "trait coordinates skipped for {} vs {} at {}: no trait vector for {}",
                student,
                teacher_init,
                revision_of(rung),
                absent,
            )
            continue
        for side in (student, teacher_init):
            if side not in coords:
                coords[side] = {
                    slot: trait_coordinates_at_slot(
                        bases[side], slot, traits[keys[side]], tokenizer
                    )
                    for slot in range(bases[side].n_slots)
                }
        for slot in sorted(coords[student]):
            rows.append(
                build_similarity_row(
                    animal=animal,
                    model_a=_ref_at(cfg, student, rung),
                    model_b=_ref_at(cfg, teacher_init, rung),
                    slot=slot,
                    slot_convention=str(cfg.convention),
                    metric=METRIC_TRAIT_COORDINATE,
                    value=metrics.trait_coordinate_similarity(
                        coords[student][slot], coords[teacher_init][slot]
                    ),
                    n_examples=bases[student].n_examples,
                    ftseed=ftseed,
                )
            )
    return rows


def _run_p5(
    cfg: SimilarityLadderConfig,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> tuple[dict[str, list[dict]], dict]:
    """Trait coordinates, trait-vector stability across ftseeds, and trait probes."""
    if not cfg.trait_cells:
        raise ValueError(
            "stage p5 needs trait_cells (trained teachers); none were resolved — the "
            "P2 teachers for these rungs are missing on disk"
        )
    pairs_rows: list[dict] = []
    probe_rows: list[dict] = []
    stability_rows: list[dict] = []
    for rung in cfg.cross_init_rungs:
        cells = [c for c in cfg.trait_cells if c.revision == revision_of(rung)]
        if not cells:
            continue
        bases = _base_stacks(cfg, tokenizer, prompts, rung)
        traits: dict[tuple[str, str, int], TraitVector] = {}
        for cell in cells:
            trait, teacher_stack = _trait_vector(
                cfg, cell, bases[cell.init_label], tokenizer, prompts
            )
            traits[(cell.animal, cell.init_label, cell.ftseed)] = trait
            probe_rows.extend(
                _trait_probe_rows(cfg, cell, trait, teacher_stack, bases, rung)
            )
            _free(teacher_stack)
        # Pair rows are emitted only once every init's own trait direction at this
        # rung exists, because both sides project onto their own (plan §6.1).
        for animal, ftseed in sorted({(c.animal, c.ftseed) for c in cells}):
            pairs_rows.extend(
                _trait_pair_rows(
                    cfg, animal, ftseed, traits, bases, tokenizer, rung
                )
            )
        stability_rows.extend(_stability_rows(cfg, traits, rung))
        _free(bases)
    return (
        {
            "similarity_pairs": pairs_rows,
            "probe_trait": probe_rows,
            "trait_vector_stability": stability_rows,
        },
        {"n_trait_cells": len(cfg.trait_cells)},
    )


def _trait_probe_rows(
    cfg: SimilarityLadderConfig,
    cell: TraitCell,
    trait: TraitVector,
    teacher_stack: ActivationStack,
    bases: Mapping[str, ActivationStack],
    rung: int,
) -> list[dict]:
    """C8's independent grading signal: can a probe separate teacher from base at all?"""
    base = bases[cell.init_label]
    rows: list[dict] = []
    for slot in range(base.n_slots):
        fit = fit_trait_probe(
            teacher_stack,
            base,
            trait.unit[slot],
            slot,
            seed=cfg.geometry.probe.seed,
        )
        for metric, value in (
            ("trait_probe_accuracy", fit.accuracy),
            ("trait_probe_macro_f1", fit.macro_f1),
            ("trait_probe_cosine", fit.cosine_with_trait_direction),
            ("trait_norm", float(trait.norm[slot])),
        ):
            rows.append(
                {
                    "animal": cell.animal,
                    "model": _ref_at(cfg, cell.init_label, rung),
                    "base_revision": revision_of(rung),
                    "step": rung,
                    "slot": slot,
                    "slot_convention": str(cfg.convention),
                    "teacher_tag": cell.teacher_tag,
                    "ftseed": cell.ftseed,
                    "n_examples": base.n_examples,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows


def _stability_rows(
    cfg: SimilarityLadderConfig,
    traits: Mapping[tuple[str, str, int], TraitVector],
    rung: int,
) -> list[dict]:
    """Cosine between the trait directions two ftseeds of one cell extracted."""
    rows: list[dict] = []
    for (animal_a, init_a, seed_a), (animal_b, init_b, seed_b) in combinations(
        sorted(traits), 2
    ):
        if animal_a != animal_b or init_a != init_b:
            continue
        left = traits[(animal_a, init_a, seed_a)]
        right = traits[(animal_b, init_b, seed_b)]
        for slot in range(left.unit.shape[0]):
            rows.append(
                {
                    "animal": animal_a,
                    "model": _ref_at(cfg, init_a, rung),
                    "base_revision": revision_of(rung),
                    "step": rung,
                    "slot": slot,
                    "slot_convention": str(cfg.convention),
                    "pooling": str(cfg.geometry.pooling),
                    "ftseed_a": seed_a,
                    "ftseed_b": seed_b,
                    "metric": "trait_direction_cosine",
                    "value": float(
                        torch.dot(
                            left.unit[slot].double(), right.unit[slot].double()
                        )
                    ),
                }
            )
    return rows


# ---- entry points --------------------------------------------------------------------


def _stage_runner(stage: str):
    return {STAGE_P1: _run_p1, STAGE_P1C: _run_p1c, STAGE_P5: _run_p5}[stage]


def run(cfg: SimilarityLadderConfig) -> list[Path]:
    """Measure this shard of the ladder and write its part tables.

    Args:
        cfg: The invocation (stage, rungs, models, estimator settings).

    Returns:
        The part-table paths written, in write order (empty when every rung of the
        shard was already covered).

    Raises:
        ValueError: A cross-init pair spans two revisions, the prompt pool is too
            small, or stage `p5` was asked for without trained teachers.
    """
    tokenizer = _tokenizer(cfg)
    prompts = _prompts(cfg)
    tables, diagnostics = _stage_runner(cfg.stage)(cfg, tokenizer, prompts)
    parts_dir = cfg.run_dir / PARTS_SUBDIR / cfg.shard_key
    written: list[Path] = []
    non_empty = {name: rows for name, rows in tables.items() if rows}
    if not non_empty:
        logger.info("shard {} produced no new rows — already covered", cfg.shard_key)
        return written
    parts_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in sorted(non_empty.items()):
        written.append(write_result_table(parts_dir, name, rows))
    write_estimator_diagnostics(
        parts_dir,
        {
            "shard": cfg.shard_key,
            "stage": cfg.stage,
            "slot_convention": str(cfg.convention),
            "n_examples": cfg.geometry.n_examples,
            "batch_size": cfg.geometry.batch_size,
            "n_repeats": cfg.geometry.n_repeats,
            **diagnostics,
        },
    )
    return written


def merge_parts(output_root: Path, run_set_id: str) -> list[Path]:
    """Concatenate every shard's part tables into one long CSV per table.

    Single-writer by contract: the shards run concurrently on eight GPUs and each
    owns its own ``parts/<shard>/`` directory, so only the driver (or the results
    pipeline) merges. Duplicate rows from a re-run shard are dropped.

    Args:
        output_root: Artifact root from the paths config.
        run_set_id: The geometry run-set id.

    Returns:
        The merged table paths, in write order.
    """
    run_dir = similarity_run_dir(output_root, run_set_id)
    results_dir = similarity_results_dir(output_root, run_set_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in sorted(RESULT_TABLES):
        frames = [
            pd.read_csv(p)
            for p in sorted((run_dir / PARTS_SUBDIR).glob(f"*/{table}.csv"))
        ]
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True).drop_duplicates()
        written.append(write_result_table(results_dir, table, merged))
    return written


def merged_table(output_root: Path, run_set_id: str, table: str) -> pd.DataFrame:
    """Read one merged table, or an empty frame when the ladder has not run."""
    results_dir = similarity_results_dir(output_root, run_set_id)
    if not (results_dir / f"{table}.csv").exists():
        return pd.DataFrame(columns=list(RESULT_TABLES[table]))
    return read_result_table(results_dir, table)


def rung_windows(rungs: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Consecutive ``(rung_k, rung_k+1)`` pairs of a ladder, for shard planning."""
    yield from zip(rungs, rungs[1:])
