"""Minibatch linear CKA between PolyPythia seed variants — orchestration.

Measures whether the teacher base
(pythia-160m) is representationally closer to the student bases it shares
initialization with (data-seed variants) than to those it shares data order with
(weight-seed variants), on Pile text vs. number-sequence transfer data. The readout
is the CONTRAST between pair groups, bracketed by the ceiling (self) and floor
(row-permuted) controls — never absolute CKA values.

Efficiency shape: all models stay resident; each batch is forwarded once per model
and the per-model preprocessed Gram stacks are shared across every pair × variant
accumulator (HSIC accumulation is just a dot product of Gram stacks). The floor
control's row permutation is applied by permuting the second model's Gram (row
permutation of activations commutes with the U-centering).

Variants per run: ``raw``, ``pc{r}`` for each configured PC-ablation rank (top-r
principal directions, estimated per (model, layer) on held-out batches, projected
out), and ``meanpool`` (secondary one-row-per-sequence aggregation, raw only).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from subliminal_learning.cka.extract_io import (
    fit_pcs,
    gather_position_rows,
    layer_activations,
    meanpool_rows,
    num_layer_slots,
    project_out_pcs,
)
from subliminal_learning.cka.minibatch import MinibatchCKA, preprocessed_gram_stack
from subliminal_learning.cka.model_refs import (
    ModelRef,
    load_ref,
    ref_provenance,
)
from subliminal_learning.cka.pool_io import (
    SequencePool,
    build_numbers_pool,
    build_pile_pool,
)
from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.run_gate import ERROR, SKIP, format_identity_diff, gate_action
from subliminal_learning.similarity.metrics import procrustes as procrustes_distance
from subliminal_learning.similarity.moments import ModelMoments, PairMoments
from subliminal_learning.teachers.model_specs import original_160m_spec, spec_label
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

CKA_MANIFEST = "cka_manifest.json"


@dataclass(frozen=True)
class CkaConfig:
    """One CKA run: a data condition over the full pair matrix.

    Attributes:
        condition: "pile" or "numbers".
        out_dir: Output directory for this run's artifacts.
        numbers_path: A generated pool's ``filtered.jsonl`` (numbers condition only;
            never regenerated here).
        numbers_subsample_size / numbers_subsample_seed: Apply the distillation
            pipeline's fixed subsample to the file's rows before the eligibility
            scan. The default (10k @ seed 42) IS the students' training-set
            convention; None = whole file.
        k_batches: Minimum number of minibatches (extend to 200 if the convergence
            curve is not flat).
        record_every: Record the running CKA matrices every N batches.
        batch_rows: Rows per minibatch m (>= 4 required by the U-statistic).
        positions_per_seq: Sampled positions per sequence; ``batch_rows`` must be
            divisible by it. Default 1 = every batch row comes from a different
            sequence (closest to the unbiased estimator's independent-rows
            assumption); >1 trades independence for fewer forwarded sequences.
        pool_seqs: Sequence-pool size the batches re-partition each pass. None
            (default) auto-sizes: pile draws exactly ``k_batches x seqs_per_batch``
            docs (single-pass — no doc revisited); numbers keeps every usable row
            of the (subsampled) file.
        pc_heldout_seqs: Extra held-out sequences used only for PC estimation.
        pc_ranks: PC-ablation ranks r (empty tuple disables the variant).
        pile_dataset / pile_split: Pile source (condition "pile").
        max_tokens / min_tokens: Pile doc truncation / minimum usable length.
        seed: Master seed for pool sampling, batch partitioning, position sampling,
            and the floor permutations.
        device: Torch device for the forward passes (execution detail — not part of
            the run identity).
        save_plots: Write per-(pair, variant) heatmap pngs. Execution detail (like
            ``device``): NOT part of the run identity, so toggling it never trips the
            gate. False skips the pngs entirely.
        force: Overwrite the run dir even when a stored manifest's identity differs
            (the gate otherwise raises on drift and skips on an identical, complete
            run).
        procrustes: Also accumulate full-sample second moments and report the
            orthogonal Procrustes distance per (pair, variant, layer). Unlike CKA this
            is NOT a per-batch average -- it is the exact full-sample quantity (see
            `similarity.moments`), so the two are on different estimator footings and
            should not share a table without a note. Adds no forward passes; costs one
            ``d x d`` accumulator per (model, variant) and per (pair, variant).
    """

    condition: str
    out_dir: Path
    numbers_path: Path | None = None
    numbers_subsample_size: int | None = 10_000
    numbers_subsample_seed: int = 42
    k_batches: int = 100
    record_every: int = 10
    batch_rows: int = 256
    positions_per_seq: int = 1
    pool_seqs: int | None = None
    pc_heldout_seqs: int = 512
    pc_ranks: tuple[int, ...] = (1, 3, 5)
    pile_dataset: str = "monology/pile-uncopyrighted"
    pile_split: str = "validation"
    max_tokens: int = 512
    min_tokens: int = 32
    seed: int = 0
    device: str = "cuda:0"
    save_plots: bool = True
    force: bool = False
    procrustes: bool = False


@dataclass(frozen=True)
class CkaPair:
    """One (model A, model B) cell of the pair matrix.

    Orientation is fixed: ``model_a`` (the teacher/ref for primary pairs) is on the
    heatmap rows, ``model_b`` on the columns. ``permute_b=True`` is the floor
    control: a per-batch random row permutation of model B's activations.
    """

    label: str
    group: str
    model_a: str
    model_b: str
    permute_b: bool = False


def build_pairs(
    ref_label: str, data_labels: list[str], weight_labels: list[str]
) -> list[CkaPair]:
    """The full pair matrix for the models present.

    Primary (ref, d_i)/(ref, w_i); secondary within-arm pairs; per-model self pairs
    (the ref's self pair IS the ceiling control — identical checkpoint both sides);
    one floor pair (first primary pair, row-permuted).
    """
    pairs: list[CkaPair] = []
    for lbl in data_labels:
        pairs.append(CkaPair(f"{ref_label}__{lbl}", "primary_data", ref_label, lbl))
    for lbl in weight_labels:
        pairs.append(CkaPair(f"{ref_label}__{lbl}", "primary_weight", ref_label, lbl))
    for a, b in combinations(data_labels, 2):
        pairs.append(CkaPair(f"{a}__{b}", "within_data", a, b))
    for a, b in combinations(weight_labels, 2):
        pairs.append(CkaPair(f"{a}__{b}", "within_weight", a, b))
    pairs.append(CkaPair(f"self__{ref_label}", "ceiling", ref_label, ref_label))
    for lbl in [*data_labels, *weight_labels]:
        pairs.append(CkaPair(f"self__{lbl}", "self", lbl, lbl))
    variant_labels = data_labels or weight_labels
    if variant_labels:
        pairs.append(
            CkaPair(
                f"floor__{ref_label}__{variant_labels[0]}",
                "floor",
                ref_label,
                variant_labels[0],
                permute_b=True,
            )
        )
    return pairs


def build_explicit_pairs(
    label_a: str, label_b: str, same_model: bool
) -> list[CkaPair]:
    """The pair matrix for the explicit ``--model-a``/``--model-b`` mode.

    ``(A, B)`` primary; ``(A, A)`` ceiling; ``(B, B)`` self (skipped when B is the
    SAME underlying model as A — then it would just duplicate the ceiling); one floor
    ``(A, B)`` row-permuted. Labels A and B are already disambiguated to differ, so
    every pair label is unique.
    """
    pairs = [
        CkaPair(f"{label_a}__{label_b}", "primary", label_a, label_b),
        CkaPair(f"self__{label_a}", "ceiling", label_a, label_a),
    ]
    if not same_model:
        pairs.append(CkaPair(f"self__{label_b}", "self", label_b, label_b))
    pairs.append(
        CkaPair(
            f"floor__{label_a}__{label_b}", "floor", label_a, label_b, permute_b=True
        )
    )
    return pairs


def _split_specs(
    specs: list[PythiaModelSpec],
) -> tuple[PythiaModelSpec, list[PythiaModelSpec], list[PythiaModelSpec]]:
    """(ref, data variants, weight variants); the ref (original 160m) is required."""
    ref = original_160m_spec()
    if ref not in specs:
        raise ValueError(
            "the shared reference (original pythia-160m) must be among the models; "
            "pass e.g. --base-models all7"
        )
    data = [s for s in specs if s.variant == "data"]
    weight = [s for s in specs if s.variant == "weight"]
    return ref, data, weight


def _load_shared_tokenizer(
    specs: list[PythiaModelSpec], paths: PathsConfig, model_cfg: ModelConfig
):
    """The canonical pythia-160m tokenizer, after verifying each variant agrees.

    All PolyPythias share one tokenizer, so the canonical one tokenizes everything.
    Each variant repo's own tokenizer is still resolved as a check: a LOAD FAILURE (a
    known flakiness of these repos) falls back to the canonical tokenizer with a
    warning (a known flakiness of these repos) — but a tokenizer that loads with a
    DIFFERENT vocab size
    raises, because feeding canonical token ids to a model with a different vocab
    would silently corrupt every activation.
    """
    canonical = load_tokenizer(original_160m_spec(), paths.hf_cache_root, model_cfg.hub_org)
    for spec in specs:
        if spec == original_160m_spec():
            continue
        try:
            tok = load_tokenizer(spec, paths.hf_cache_root, model_cfg.hub_org)
        except Exception as e:  # noqa: BLE001 - fallback is the documented behavior
            logger.warning(
                "{}: tokenizer load failed ({}); falling back to canonical",
                spec_label(spec), e,
            )
            continue
        if len(tok) != len(canonical):
            raise ValueError(
                f"{spec_label(spec)}: tokenizer vocab size {len(tok)} != canonical "
                f"{len(canonical)} — the suite must share one tokenizer"
            )
    return canonical


def _build_pool(cfg: CkaConfig, tokenizer) -> SequencePool:
    n_heldout = cfg.pc_heldout_seqs if cfg.pc_ranks else 0
    if cfg.pool_seqs is not None:
        n_total = cfg.pool_seqs + n_heldout
    elif cfg.condition == "pile":
        # Auto: single-pass — exactly one batch slot per doc, no doc revisited.
        n_total = cfg.k_batches * (cfg.batch_rows // cfg.positions_per_seq) + n_heldout
    else:
        n_total = None  # numbers auto: every usable row of the (subsampled) file
    if cfg.condition == "pile":
        return build_pile_pool(
            tokenizer,
            n_total,
            dataset_name=cfg.pile_dataset,
            split=cfg.pile_split,
            max_tokens=cfg.max_tokens,
            min_tokens=max(cfg.min_tokens, cfg.positions_per_seq + 1),
            seed=cfg.seed,
        )
    if cfg.condition == "numbers":
        if cfg.numbers_path is None:
            raise ValueError("condition='numbers' requires numbers_path")
        return build_numbers_pool(
            tokenizer,
            cfg.numbers_path,
            n_total,
            min_eligible=cfg.positions_per_seq,
            seed=cfg.seed,
            subsample_size=cfg.numbers_subsample_size,
            subsample_seed=cfg.numbers_subsample_seed,
        )
    raise ValueError(f"unknown condition {cfg.condition!r}; expected 'pile' or 'numbers'")


def _run_identity(
    cfg: CkaConfig, model_refs: list[str], pool_sha: str, variants: list[str]
) -> dict:
    """Every input that can change the artifacts (device/out_dir are execution details).

    ``pool_token_sha256`` is the tokenized measurement data itself, so tokenizer,
    sampling-seed, truncation, source-content, and eligibility drift all surface as
    identity drift even when the config strings look unchanged.
    """
    return {
        "condition": cfg.condition,
        "numbers_path": str(cfg.numbers_path) if cfg.numbers_path else None,
        "numbers_sha256": file_sha256(cfg.numbers_path) if cfg.numbers_path else None,
        "numbers_subsample_size": cfg.numbers_subsample_size,
        "numbers_subsample_seed": cfg.numbers_subsample_seed,
        "pool_token_sha256": pool_sha,
        "k_batches": cfg.k_batches,
        "record_every": cfg.record_every,
        "batch_rows": cfg.batch_rows,
        "positions_per_seq": cfg.positions_per_seq,
        "pool_seqs": cfg.pool_seqs,
        "pc_heldout_seqs": cfg.pc_heldout_seqs,
        "pc_ranks": list(cfg.pc_ranks),
        "pile_dataset": cfg.pile_dataset,
        "pile_split": cfg.pile_split,
        "max_tokens": cfg.max_tokens,
        "min_tokens": cfg.min_tokens,
        "seed": cfg.seed,
        "models": model_refs,
        "variants": variants,
        # Only present when enabled, so a CKA-only run computes byte-identical identity
        # to what every EXISTING manifest stored. Adding the key unconditionally would
        # make every completed run on disk read as identity drift and trip the gate.
        **({"procrustes": True} if cfg.procrustes else {}),
    }


def _check_gate(out_dir: Path, identity: dict, force: bool) -> str:
    """SKIP a complete identical run; raise on identity drift; else run (overwrite).

    The manifest is written LAST (after every artifact), so its presence certifies a
    complete run — an interrupted run leaves no manifest and is simply redone.
    Reuses `run_gate.gate_action`: RETRAIN here means "run".
    """
    manifest_path = out_dir / CKA_MANIFEST
    stored = None
    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            logger.warning("corrupt {} — re-running over it", manifest_path)
    decision = gate_action(stored, identity, force)
    if decision.action == SKIP:
        logger.info("skip: {} is complete under an identical identity", out_dir)
        return SKIP
    if decision.action == ERROR:
        raise ValueError(
            f"{out_dir} holds a COMPLETE run whose identity differs — refusing to "
            f"overwrite. Drift:\n{format_identity_diff(decision.diff)}\n"
            "Use a fresh --run-tag for a new configuration, or --force to overwrite."
        )
    return "run"


@contextmanager
def _svd_threads(n: int = 2):
    """Cap intra-op threads around the Procrustes finalization.

    A 768x768 ``svdvals`` takes 86 ms at 2 threads but 50 s at 48 on a node whose cores
    are already saturated by the sweep — LAPACK oversubscription, not real work. Queue
    jobs already inherit ``OMP_NUM_THREADS=2`` from the gpu_policy; this makes a
    hand-run behave the same instead of falling off that cliff.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(min(n, prev))
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _atomic_np_save(path: Path, arr: np.ndarray) -> None:
    """`np.save` via temp file + rename (no torn npy after an interrupt)."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _pad_batch(
    batch_ids: list[list[int]],
    batch_elig: list[list[int]],
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch; returns (input_ids, attention_mask, eligible_mask).

    Right padding is safe here: GPT-NeoX uses rotary positions and causal attention,
    so states at real (pre-pad) positions are unaffected by trailing pads, and rows
    are only ever sampled at eligible (real) positions.
    """
    n = len(batch_ids)
    t = max(len(s) for s in batch_ids)
    input_ids = torch.full((n, t), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, t), dtype=torch.long)
    eligible_mask = torch.zeros((n, t), dtype=torch.bool)
    for b, (seq, elig) in enumerate(zip(batch_ids, batch_elig)):
        input_ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[b, : len(seq)] = 1
        eligible_mask[b, elig] = True
    return input_ids.to(device), attention_mask.to(device), eligible_mask.to(device)


def _sample_rows(
    batch_elig: list[list[int]], per_seq: int, rng: np.random.Generator
) -> torch.Tensor:
    """``[n_seqs * per_seq, 2]`` (seq, pos) rows: per-sequence sampling w/o replacement."""
    entries: list[tuple[int, int]] = []
    for b, elig in enumerate(batch_elig):
        assert len(elig) >= per_seq, "pool construction guarantees enough eligible positions"
        picks = rng.choice(len(elig), size=per_seq, replace=False)
        entries.extend((b, elig[int(i)]) for i in picks)
    return torch.tensor(entries, dtype=torch.long)


def _iter_batches(
    pool_size: int, seqs_per_batch: int, k_batches: int, rng: np.random.Generator
):
    """Yield ``k_batches`` index lists: fresh pool permutation each pass, sampling
    without replacement within a pass (the source paper's protocol for growing k)."""
    emitted = 0
    while emitted < k_batches:
        order = rng.permutation(pool_size)
        for start in range(0, pool_size - seqs_per_batch + 1, seqs_per_batch):
            yield order[start : start + seqs_per_batch].tolist()
            emitted += 1
            if emitted >= k_batches:
                return


def _permute_gram_stack(grams: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply a row permutation to each flattened Gram in the stack (P G P^T).

    Equivalent to (and cheaper than) recomputing the preprocessed Gram from permuted
    activations: every step of the U-centering commutes with a joint row/col
    permutation.
    """
    n = perm.shape[0]
    g = grams.reshape(grams.shape[0], n, n)
    g = g[:, perm][:, :, perm]
    return g.reshape(grams.shape[0], -1)


def _fit_all_pcs(
    models: dict[str, torch.nn.Module],
    pool: SequencePool,
    heldout_idx: list[int],
    cfg: CkaConfig,
    pad_id: int,
    n_slots: int,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Per (model, layer) activation mean + top-max(pc_ranks) PCs from held-out batches."""
    r_max = max(cfg.pc_ranks)
    seqs_per_batch = cfg.batch_rows // cfg.positions_per_seq
    rng = np.random.default_rng(cfg.seed + 3)
    collected: dict[str, list[list[torch.Tensor]]] = {
        label: [[] for _ in range(n_slots)] for label in models
    }
    for start in range(0, len(heldout_idx) - seqs_per_batch + 1, seqs_per_batch):
        chunk = heldout_idx[start : start + seqs_per_batch]
        batch_ids = [pool.input_ids[i] for i in chunk]
        batch_elig = [pool.eligible[i] for i in chunk]
        input_ids, attention_mask, _ = _pad_batch(batch_ids, batch_elig, pad_id, cfg.device)
        row_index = _sample_rows(batch_elig, cfg.positions_per_seq, rng).to(cfg.device)
        for label, model in models.items():
            acts = layer_activations(model, input_ids, attention_mask)
            for slot, rows in enumerate(gather_position_rows(acts, row_index)):
                collected[label][slot].append(rows.float().cpu())
            del acts
    pcs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for label, per_slot in collected.items():
        pcs[label] = []
        for slot_rows in per_slot:
            mean, comps = fit_pcs(torch.cat(slot_rows, dim=0), r_max)
            pcs[label].append((mean.to(cfg.device), comps.to(cfg.device)))
    logger.info("Fitted top-{} PCs per (model, layer) on {} held-out seqs", r_max, len(heldout_idx))
    return pcs


def _plot_heatmap(path: Path, mat: np.ndarray, title: str) -> None:
    """Fixed-scale [0, 1] heatmap png; A layers on rows, input->output from bottom-left."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping heatmap {}", path.name)
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, origin="lower", cmap="magma")
    ax.set_xlabel("model B layer (0 = embeddings)")
    ax.set_ylabel("model A layer (0 = embeddings)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="CKA")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _validate_batch_cfg(cfg: CkaConfig) -> int:
    """Validate the batch/position/PC-heldout invariants; return seqs_per_batch."""
    if cfg.batch_rows % cfg.positions_per_seq != 0:
        raise ValueError(
            f"batch_rows ({cfg.batch_rows}) must be divisible by positions_per_seq "
            f"({cfg.positions_per_seq})"
        )
    if cfg.batch_rows < 4:
        raise ValueError(f"batch_rows must be >= 4 (U-statistic), got {cfg.batch_rows}")
    seqs_per_batch = cfg.batch_rows // cfg.positions_per_seq
    if cfg.pc_ranks and cfg.pc_heldout_seqs < seqs_per_batch:
        raise ValueError(
            f"pc_heldout_seqs ({cfg.pc_heldout_seqs}) must be >= seqs_per_batch "
            f"({seqs_per_batch}) to yield at least one PC-estimation batch "
            "(or pass --pc-ranks '' to disable the PC-ablated variants)"
        )
    return seqs_per_batch


def run(
    cfg: CkaConfig,
    paths: PathsConfig,
    model_cfg: ModelConfig,
    specs: list[PythiaModelSpec],
) -> Path:
    """Run the full pair matrix for one data condition; returns the output dir.

    Writes, per (pair, variant): ``cka__{pair}__{variant}.npy`` (final matrix),
    ``convergence__{pair}__{variant}.npy`` (running matrices every ``record_every``
    batches, stacked), and a heatmap png; plus ``summary.json`` (per-pair diagonal
    means, band means, convergence curves) and ``meta.json`` (full provenance).
    """
    _validate_batch_cfg(cfg)

    ref_spec, data_specs, weight_specs = _split_specs(specs)
    ref_label = spec_label(ref_spec)
    pairs = build_pairs(
        ref_label, [spec_label(s) for s in data_specs], [spec_label(s) for s in weight_specs]
    )

    tokenizer = _load_shared_tokenizer(specs, paths, model_cfg)
    pool = _build_pool(cfg, tokenizer)
    variants = ["raw", *[f"pc{r}" for r in cfg.pc_ranks], "meanpool"]
    identity = _run_identity(
        cfg, [model_ref(s, model_cfg.hub_org) for s in specs], pool.token_sha256(), variants
    )
    if _check_gate(cfg.out_dir, identity, cfg.force) == SKIP:
        return cfg.out_dir

    models: dict[str, torch.nn.Module] = {}
    for spec in specs:
        models[spec_label(spec)] = load_model(
            spec, paths.hf_cache_root, model_cfg.hub_org, cfg.device
        )
    meta_models = {spec_label(s): model_ref(s, model_cfg.hub_org) for s in specs}
    return _measure_and_save(
        cfg, tokenizer, pool, models, pairs, variants, identity, meta_models
    )


def run_explicit(
    cfg: CkaConfig,
    paths: PathsConfig,
    model_cfg: ModelConfig,
    ref_a: ModelRef,
    ref_b: ModelRef,
    label_a: str,
    label_b: str,
) -> Path:
    """Explicit two-model CKA: compare arbitrary operands A and B (base or finetuned).

    ``label_a``/``label_b`` are the already-disambiguated labels (the CLI derives them
    for the run tag). Both models are verified to agree on layer count downstream; the
    canonical pythia-160m tokenizer is used (finetuning does not change the tokenizer).
    The run identity records the two provenance dicts (incl. checkpoint hashes), so
    adapter/weight drift trips the gate.
    """
    _validate_batch_cfg(cfg)
    prov_a = ref_provenance(ref_a)
    prov_b = ref_provenance(ref_b)
    same_model = {k: v for k, v in prov_a.items() if k != "ref"} == {
        k: v for k, v in prov_b.items() if k != "ref"
    }
    pairs = build_explicit_pairs(label_a, label_b, same_model)

    tokenizer = load_tokenizer(original_160m_spec(), paths.hf_cache_root, model_cfg.hub_org)
    pool = _build_pool(cfg, tokenizer)
    variants = ["raw", *[f"pc{r}" for r in cfg.pc_ranks], "meanpool"]
    identity = _run_identity(cfg, [prov_a, prov_b], pool.token_sha256(), variants)
    if _check_gate(cfg.out_dir, identity, cfg.force) == SKIP:
        return cfg.out_dir

    model_a = load_ref(ref_a, paths, model_cfg, cfg.device)
    models: dict[str, torch.nn.Module] = {label_a: model_a}
    # Same underlying checkpoint -> reuse the loaded model (labels still differ).
    models[label_b] = model_a if same_model else load_ref(ref_b, paths, model_cfg, cfg.device)
    meta_models = {label_a: prov_a, label_b: prov_b}
    return _measure_and_save(
        cfg, tokenizer, pool, models, pairs, variants, identity, meta_models
    )


def _measure_and_save(
    cfg: CkaConfig,
    tokenizer,
    pool: SequencePool,
    models: dict[str, torch.nn.Module],
    pairs: list[CkaPair],
    variants: list[str],
    identity: dict,
    meta_models: dict,
) -> Path:
    """Shared measurement core + save: PC fit, the minibatch loop, one artifact dir.

    Used by both ``run`` (base suite) and ``run_explicit``. Models are already loaded
    and gated; ``meta_models`` is the ``meta.json`` ``models`` mapping for this mode.
    """
    accs, records, n_slots, moments = _measure(
        cfg, tokenizer, pool, models, pairs, variants
    )
    return _save_outputs(
        cfg, meta_models, pool, pairs, variants, accs, records, n_slots, identity,
        moments,
    )


def _measure(
    cfg: CkaConfig,
    tokenizer,
    pool: SequencePool,
    models: dict[str, torch.nn.Module],
    pairs: list[CkaPair],
    variants: list[str],
) -> tuple[
    dict[tuple[str, str], MinibatchCKA],
    dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    int,
    tuple[dict[tuple[str, str], ModelMoments], dict[tuple[str, str], PairMoments]],
]:
    """The measurement loop: PC fit + minibatch HSIC accumulation over ``pairs``.

    Forwards each resident model once per batch and shares its preprocessed Gram stacks
    across every pair × variant accumulator (so a teacher shared by K student pairs is
    forwarded once). Returns ``(accs, records, n_slots, moments)``; ``moments`` is the
    ``(model, pair)`` second-moment pair for Procrustes and is empty unless
    ``cfg.procrustes``.
    """
    seqs_per_batch = cfg.batch_rows // cfg.positions_per_seq
    # Held-out prefix (PC estimation only) vs. the CKA pool the batches re-partition.
    n_heldout = cfg.pc_heldout_seqs if cfg.pc_ranks else 0
    heldout_idx = list(range(n_heldout))
    cka_idx = list(range(n_heldout, len(pool)))
    if len(cka_idx) < seqs_per_batch:
        raise ValueError(
            f"CKA pool has {len(cka_idx)} seqs after reserving {n_heldout} for PC "
            f"estimation — need at least seqs_per_batch ({seqs_per_batch})"
        )

    n_slots_all = {label: num_layer_slots(m) for label, m in models.items()}
    if len(set(n_slots_all.values())) != 1:
        raise ValueError(f"models disagree on layer count: {n_slots_all}")
    n_slots = next(iter(n_slots_all.values()))

    pcs = (
        _fit_all_pcs(models, pool, heldout_idx, cfg, tokenizer.pad_token_id, n_slots)
        if cfg.pc_ranks
        else {}
    )

    accs = {
        (p.label, v): MinibatchCKA(n_slots, across_models=True)
        for p in pairs
        for v in variants
    }
    records: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {
        key: [] for key in accs
    }
    floor_models = {p.model_b for p in pairs if p.permute_b}
    ceiling_label = next(p.label for p in pairs if p.group == "ceiling")

    # Procrustes rides along on the same forward passes. Self-pairs are skipped: their
    # distance is identically zero. Accumulators live on the compute device so the
    # per-batch row matrices never round-trip to host memory.
    dim = next(iter(models.values())).config.hidden_size
    proc_pairs = [p for p in pairs if not (p.model_a == p.model_b and not p.permute_b)]
    model_moments: dict[tuple[str, str], ModelMoments] = {}
    pair_moments: dict[tuple[str, str], PairMoments] = {}
    if cfg.procrustes:
        proc_labels = {lbl for p in proc_pairs for lbl in (p.model_a, p.model_b)}
        model_moments = {
            (label, v): ModelMoments(n_slots, dim, device=cfg.device)
            for label in proc_labels
            for v in variants
        }
        pair_moments = {
            (p.label, v): PairMoments(n_slots, dim, dim, device=cfg.device)
            for p in proc_pairs
            for v in variants
        }
        logger.info(
            "Procrustes enabled: {} model x {} pair accumulators ({} pairs, self-pairs "
            "skipped)",
            len(model_moments), len(pair_moments), len(proc_pairs),
        )

    rng_batches = np.random.default_rng(cfg.seed + 1)
    rng_pos = np.random.default_rng(cfg.seed + 2)
    rng_floor = np.random.default_rng(cfg.seed + 4)

    logger.info(
        "CKA run: condition={} models={} pairs={} variants={} k={} m={} ({} seqs x {} pos)",
        cfg.condition, list(models), len(pairs), variants, cfg.k_batches,
        cfg.batch_rows, seqs_per_batch, cfg.positions_per_seq,
    )
    batch_iter = _iter_batches(len(cka_idx), seqs_per_batch, cfg.k_batches, rng_batches)
    for k, local_idx in enumerate(batch_iter, start=1):
        chunk = [cka_idx[i] for i in local_idx]
        batch_ids = [pool.input_ids[i] for i in chunk]
        batch_elig = [pool.eligible[i] for i in chunk]
        input_ids, attention_mask, eligible_mask = _pad_batch(
            batch_ids, batch_elig, tokenizer.pad_token_id, cfg.device
        )
        row_index = _sample_rows(batch_elig, cfg.positions_per_seq, rng_pos).to(cfg.device)
        # One floor permutation per batch, shared across layers; meanpool rows are
        # per-sequence, so they get their own (smaller) permutation.
        perm_pos = torch.from_numpy(rng_floor.permutation(cfg.batch_rows)).to(cfg.device)
        perm_mp = torch.from_numpy(rng_floor.permutation(seqs_per_batch)).to(cfg.device)

        grams: dict[tuple[str, str], torch.Tensor] = {}
        grams_perm: dict[tuple[str, str], torch.Tensor] = {}
        # Raw (pre-Gram) rows, kept for this batch only when Procrustes is on: the
        # moment accumulators need X itself, not the preprocessed Gram.
        rows_kv: dict[tuple[str, str], list[torch.Tensor]] = {}
        for label, model in models.items():
            acts = layer_activations(model, input_ids, attention_mask)
            pos_rows = [r.float() for r in gather_position_rows(acts, row_index)]
            mp_rows = [r.float() for r in meanpool_rows(acts, eligible_mask)]
            del acts
            grams[(label, "raw")] = preprocessed_gram_stack(pos_rows)
            if cfg.procrustes:
                rows_kv[(label, "raw")] = pos_rows
            for r in cfg.pc_ranks:
                ablated = [
                    project_out_pcs(rows, *pcs[label][slot], r)
                    for slot, rows in enumerate(pos_rows)
                ]
                grams[(label, f"pc{r}")] = preprocessed_gram_stack(ablated)
                if cfg.procrustes:
                    rows_kv[(label, f"pc{r}")] = ablated
            grams[(label, "meanpool")] = preprocessed_gram_stack(mp_rows)
            if cfg.procrustes:
                rows_kv[(label, "meanpool")] = mp_rows
            if label in floor_models:
                for v in variants:
                    perm = perm_mp if v == "meanpool" else perm_pos
                    grams_perm[(label, v)] = _permute_gram_stack(grams[(label, v)], perm)

        for p in pairs:
            for v in variants:
                gb = (
                    grams_perm[(p.model_b, v)]
                    if p.permute_b
                    else grams[(p.model_b, v)]
                )
                accs[(p.label, v)].update_state_from_grams(grams[(p.model_a, v)], gb)

        if cfg.procrustes:
            for key, mm in model_moments.items():
                mm.update(rows_kv[key])
            for p in proc_pairs:
                for v in variants:
                    rows_a = rows_kv[(p.model_a, v)]
                    rows_b = rows_kv[(p.model_b, v)]
                    if p.permute_b:
                        # Same row permutation the floor Gram uses, applied to the rows.
                        # X^T X is permutation-invariant, so only the CROSS moment moves.
                        perm = perm_mp if v == "meanpool" else perm_pos
                        rows_b = [r[perm] for r in rows_b]
                    pair_moments[(p.label, v)].update(rows_a, rows_b)
            rows_kv.clear()

        if k % cfg.record_every == 0 or k == cfg.k_batches:
            for key, acc in accs.items():
                records[key].append((k, acc.result().numpy()))
            ref_self = accs[(ceiling_label, "raw")].result().numpy()
            logger.info(
                "batch {}/{}: ceiling diag mean {:.4f}",
                k, cfg.k_batches, float(np.nanmean(np.diag(ref_self))),
            )

    return accs, records, n_slots, (model_moments, pair_moments)


def build_student_batch_pairs(n_students: int) -> list[CkaPair]:
    """Combined INTERNAL pair set for a K-student "vs teacher" batch.

    One shared ceiling ``(teacher, teacher)`` computed once, plus per student ``s_i``
    (internal label ``student{i}``) a primary ``(teacher, s_i)``, a self ``(s_i, s_i)``,
    and a floor ``(teacher, s_i)`` row-permuted. Internal labels stay distinct so the
    accumulators never collide; each student's SAVED filenames are remapped to the
    single-run names (``student_save_pairs``) at write time.
    """
    pairs = [CkaPair("self__teacher", "ceiling", "teacher", "teacher")]
    for i in range(n_students):
        si = f"student{i}"
        pairs.append(CkaPair(f"teacher__{si}", "primary", "teacher", si))
        pairs.append(CkaPair(f"self__{si}", "self", si, si))
        pairs.append(
            CkaPair(f"floor__teacher__{si}", "floor", "teacher", si, permute_b=True)
        )
    return pairs


def student_save_pairs() -> list[CkaPair]:
    """The per-student SAVED pair set — identical to a single ``run_explicit`` run.

    Labels ``teacher``/``student`` give exactly today's artifact filenames
    (``cka__teacher__student__{v}.npy`` etc.), so a batched student's dir is byte-schema
    identical to a single-student run of that student.
    """
    return build_explicit_pairs("teacher", "student", same_model=False)


# Saved pair label -> internal label template, resolved per student at save time.
_STUDENT_SAVE_REMAP = {
    "teacher__student": "teacher__{si}",
    "self__teacher": "self__teacher",
    "self__student": "self__{si}",
    "floor__teacher__student": "floor__teacher__{si}",
}


def student_batch_identity(
    cfg: CkaConfig,
    teacher_prov: dict,
    student_prov: dict,
    pool_sha: str,
    variants: list[str],
) -> dict:
    """One batched student's run identity — the SAME 2-operand identity a single
    ``run_explicit(teacher, that_student)`` run stores, so batched and single runs are
    mutually resumable/skippable (NOT a K-operand identity)."""
    return _run_identity(cfg, [teacher_prov, student_prov], pool_sha, variants)


def _pending_student_indices(
    out_dirs: list[Path], identities: list[dict], force: bool
) -> list[int]:
    """Indices of students that still need running (gate BEFORE any model load).

    A student whose own out dir holds a COMPLETE run under an identical identity is
    dropped (logged by the gate); identity drift raises unless ``force``.
    """
    return [
        i
        for i, (out_dir, identity) in enumerate(zip(out_dirs, identities))
        if _check_gate(out_dir, identity, force) != SKIP
    ]


def run_students(
    cfg: CkaConfig,
    paths: PathsConfig,
    model_cfg: ModelConfig,
    teacher_ref: ModelRef,
    students: list[tuple[Path, ModelRef]],
) -> list[Path]:
    """Measure K students of ONE teacher in a single pass (teacher forwarded once/batch).

    ``students`` is ``[(out_dir, student_ref), ...]``, all validated to share one teacher
    (the CLI does the manifest-agreement check). Each student's artifacts are
    byte-identical to a single ``run_explicit(teacher, student)`` run — same identity,
    filenames, and schema — so batched and single runs are mutually resumable. Students
    whose out dir already holds a complete identical run are dropped before any model
    load; if all K skip, nothing is loaded. Returns the out dirs actually measured.
    """
    _validate_batch_cfg(cfg)
    teacher_prov = ref_provenance(teacher_ref)
    student_provs = [ref_provenance(s) for _, s in students]

    tokenizer = load_tokenizer(original_160m_spec(), paths.hf_cache_root, model_cfg.hub_org)
    pool = _build_pool(cfg, tokenizer)
    variants = ["raw", *[f"pc{r}" for r in cfg.pc_ranks], "meanpool"]
    pool_sha = pool.token_sha256()

    out_dirs = [out for out, _ in students]
    identities = [
        student_batch_identity(cfg, teacher_prov, sprov, pool_sha, variants)
        for sprov in student_provs
    ]
    pending = _pending_student_indices(out_dirs, identities, cfg.force)
    if not pending:
        logger.info(
            "all {} students complete under identical identities; nothing to do",
            len(students),
        )
        return []

    models: dict[str, torch.nn.Module] = {
        "teacher": load_ref(teacher_ref, paths, model_cfg, cfg.device)
    }
    pairs = [CkaPair("self__teacher", "ceiling", "teacher", "teacher")]
    save_specs: list[tuple[Path, str, dict, dict]] = []
    for i in pending:
        out_dir, sref = students[i]
        si = f"student{i}"
        models[si] = load_ref(sref, paths, model_cfg, cfg.device)
        pairs.append(CkaPair(f"teacher__{si}", "primary", "teacher", si))
        pairs.append(CkaPair(f"self__{si}", "self", si, si))
        pairs.append(
            CkaPair(f"floor__teacher__{si}", "floor", "teacher", si, permute_b=True)
        )
        save_specs.append((out_dir, si, student_provs[i], identities[i]))

    # Student mode does not emit Procrustes (base-model scope only); moments are empty.
    accs, records, n_slots, _ = _measure(
        cfg, tokenizer, pool, models, pairs, variants
    )
    return _save_students(
        cfg, teacher_prov, pool, variants, accs, records, n_slots, save_specs
    )


def _save_students(
    cfg: CkaConfig,
    teacher_prov: dict,
    pool: SequencePool,
    variants: list[str],
    accs: dict[tuple[str, str], MinibatchCKA],
    records: dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    n_slots: int,
    save_specs: list[tuple[Path, str, dict, dict]],
) -> list[Path]:
    """Split the combined accumulators into one single-run-schema dir per student.

    ``save_specs`` is ``[(out_dir, internal_label, student_prov, identity), ...]``. Each
    student's dir gets exactly its own primary/ceiling/self/floor cells under the
    single-run filenames (the shared ceiling is duplicated into every student's
    artifacts), and its manifest is written LAST — a crash mid-loop leaves earlier
    students complete and later ones manifest-less (redoable).
    """
    save_pairs = student_save_pairs()
    outs: list[Path] = []
    for out_dir, si, sprov, identity in save_specs:
        remap = {saved: tmpl.format(si=si) for saved, tmpl in _STUDENT_SAVE_REMAP.items()}
        sub_accs = {
            (saved, v): accs[(remap[saved], v)] for saved in remap for v in variants
        }
        sub_records = {
            (saved, v): records[(remap[saved], v)] for saved in remap for v in variants
        }
        cfg_i = replace(cfg, out_dir=out_dir)
        meta_models = {"teacher": teacher_prov, "student": sprov}
        outs.append(
            _save_outputs(
                cfg_i, meta_models, pool, save_pairs, variants,
                sub_accs, sub_records, n_slots, identity,
            )
        )
    return outs


def _save_outputs(
    cfg: CkaConfig,
    meta_models: dict,
    pool: SequencePool,
    pairs: list[CkaPair],
    variants: list[str],
    accs: dict[tuple[str, str], MinibatchCKA],
    records: dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    n_slots: int,
    identity: dict,
    moments: tuple[
        dict[tuple[str, str], ModelMoments], dict[tuple[str, str], PairMoments]
    ] = ({}, {}),
) -> Path:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    # Mid-to-late-layer band for the secondary summary scalar: the deeper half of the
    # slots (7..12 on 160m — blocks 6..11).
    band_lo = (n_slots + 1) // 2
    band = list(range(band_lo, n_slots))

    summary_rows = []
    curves: dict[str, list[dict]] = {}
    nan_keys: list[str] = []
    for p in pairs:
        for v in variants:
            key = (p.label, v)
            name = f"{p.label}__{v}"
            mat = accs[key].result().numpy()
            if np.isnan(mat).any():
                nan_keys.append(name)
            _atomic_np_save(out / f"cka__{name}.npy", mat)
            _atomic_np_save(
                out / f"convergence__{name}.npy", np.stack([m for _, m in records[key]])
            )
            curves[name] = [
                {"k": k, "diag_mean": float(np.nanmean(np.diag(m)))}
                for k, m in records[key]
            ]
            if cfg.save_plots:
                _plot_heatmap(
                    out / f"cka__{name}.png", mat,
                    f"{cfg.condition} | {p.label} | {v}\n(A={p.model_a} rows, B={p.model_b} cols)",
                )
            diag = np.diag(mat)
            summary_rows.append(
                {
                    "pair": p.label,
                    "group": p.group,
                    "model_a": p.model_a,
                    "model_b": p.model_b,
                    "variant": v,
                    "diag_mean": float(np.nanmean(diag)),
                    "diag_band_mean": float(np.nanmean(diag[band])),
                    "diag_min": float(np.nanmin(diag)),
                    "diag_max": float(np.nanmax(diag)),
                }
            )

    # Procrustes: finalized ONCE from the full-sample moments (not a per-batch
    # average), so these numbers are a different estimator from the CKA ones above.
    proc_rows: list[dict] = []
    model_moments, pair_moments = moments
    if cfg.procrustes and pair_moments:
        proc_fields = ["distance", "normalized_distance", "angular_distance", "nuclear_norm"]
        for p in pairs:
            for v in variants:
                key = (p.label, v)
                if key not in pair_moments:
                    continue  # self-pair: distance is identically zero
                ma, mb = model_moments[(p.model_a, v)], model_moments[(p.model_b, v)]
                pm = pair_moments[key]
                # Accumulate fp64 (the centering `C - s_a s_b^T / n` is a cancellation-
                # prone difference of large raw moments: fp64 holds it to ~6e-15
                # relative, fp32 only ~2e-7), then finalize the SVD in fp32 on the host.
                # Procrustes only SUMS singular values -- no whitening, no inverse -- so
                # fp32 is well-conditioned here: measured 2.4e-6 relative on the
                # normalized distance. Timings for one 768x768 svdvals on this box:
                # CPU/2-thread 86 ms fp32 vs 106 ms fp64; GPU 129 ms fp32 vs 1521 ms
                # fp64; CPU at 48 threads is 50 s (oversubscription thrash on a fully
                # loaded node), hence the explicit cap in `_svd_threads`.
                with _svd_threads():
                    per_layer = [
                        procrustes_distance(
                            ma.centered_gram(sl).cpu().float(),
                            mb.centered_gram(sl).cpu().float(),
                            pm.centered_cross(ma, mb, sl, sl).cpu().float(),
                        )
                        for sl in range(n_slots)
                    ]
                arr = np.array(
                    [[d[f] for f in proc_fields] for d in per_layer], dtype=np.float64
                )
                _atomic_np_save(out / f"procrustes__{p.label}__{v}.npy", arr)
                nd = np.array([d["normalized_distance"] for d in per_layer])
                proc_rows.append(
                    {
                        "pair": p.label,
                        "group": p.group,
                        "model_a": p.model_a,
                        "model_b": p.model_b,
                        "variant": v,
                        "n_rows": pm.n_rows,
                        "fields": proc_fields,
                        "normalized_distance_mean": float(np.nanmean(nd)),
                        "normalized_distance_band_mean": float(np.nanmean(nd[band])),
                        "normalized_distance_per_layer": [float(x) for x in nd],
                        "angular_distance_per_layer": [
                            float(d["angular_distance"]) for d in per_layer
                        ],
                    }
                )
        logger.info(
            "Procrustes: {} (pair, variant) cells finalized from {} full-sample rows",
            len(proc_rows), next(iter(pair_moments.values())).n_rows,
        )

    meta = {
        "config": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "pc_ranks": list(cfg.pc_ranks),
        },
        "models": meta_models,
        "pool_source": pool.source,
        "pool_n_seqs": len(pool),
        "pool_n_scanned": pool.n_scanned,
        "pool_n_skipped": pool.n_skipped,
        "numbers_sha256": identity["numbers_sha256"],
        "pool_token_sha256": identity["pool_token_sha256"],
        "n_layer_slots": n_slots,
        "band_slots": band,
        "seqs_per_batch": cfg.batch_rows // cfg.positions_per_seq,
    }
    if nan_keys:
        logger.warning(
            "{} result matrices contain NaN (near-degenerate self-HSIC; most likely "
            "aggressive PC ablation): {}", len(nan_keys), nan_keys,
        )
    atomic_write_json(out / "meta.json", meta)
    summary = {"meta": meta, "rows": summary_rows, "convergence_curves": curves}
    if proc_rows:
        # Kept under its own key, not merged into `rows`: full-sample Procrustes and
        # minibatch-averaged CKA are different estimators (see similarity.moments).
        summary["procrustes_rows"] = proc_rows
    atomic_write_json(out / "summary.json", summary)
    # Manifest LAST: its presence certifies that every artifact above is complete
    # and was produced under exactly this identity (the skip/drift gate reads it).
    atomic_write_json(
        out / CKA_MANIFEST,
        {
            "workflow": "cka",
            "run_identity": identity,
            "nan_matrices": nan_keys,
            "created_at": utc_now_iso(),
            "git_sha": git_sha(),
        },
    )
    logger.info("Saved {} pair x variant results to {}", len(summary_rows), out)
    return out
