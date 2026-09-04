"""LoRA ΔW two-term decomposition — measurement primitives.

Backs the collapse-mechanism audit (`notebooks/collapse_mechanism.ipynb`). Writing a
trained adapter against its exactly-reconstructable init, with ``B_0 = 0``::

    A_final = A_init + lr·P,   B_final = lr·Q
    ΔW = s·B_final·A_final = s·( lr·Q·A_init + lr²·Q·P ),   s = alpha/r
                                 ^cross            ^quadratic

so the measured **cross** term is ``B_final @ A_init`` and the **quadratic** term is
``B_final @ ΔA``. This module provides the three things that computation needs, with no
IO policy and no CLI:

- `lora_init_weights` — the reconstruction of ``A_init``. It is never saved to disk, so
  it is recovered by re-running the training code path exactly: build the base model,
  ``set_seed(ftseed)``, ``get_peft_model``. That mirrors
  `teachers.sft_io.run_teacher_sft_completion_only`, which seeds immediately before the
  PEFT wrap with nothing consuming torch RNG in between. **This is the load-bearing
  assumption of the whole mechanism program** — `tests/test_adapter_decomp.py` pins it.
- exact norms that exploit the low-rank structure (see `low_rank_norms`), so a sweep
  over many adapters is CPU-cheap rather than SVD-bound.
- `decompose` — the per-module row the analysis frames consume.

Note on init sharing: ``A_init`` is a function of (seed, r, target_modules, module
shapes) only. PEFT initialises ``lora_A`` with ``kaiming_uniform_(a=sqrt(5))``, which
reads the tensor *shape* and never the pretrained values, and ``set_seed`` runs after
the weights are loaded. Two PolyPythia variants of the same size therefore share a
bit-identical ``A_init`` at matched (seed, r, target_modules) — verified for
pythia-160m / -data-seed1 / -weight-seed2. The cache here is still keyed by base repo
(correct for any architecture); the sharing is an available future optimisation, not an
assumption baked in.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.gkd.layout import parse_optim_slug
from subliminal_learning.utils.run_io import checkpoint_nonfinite_count
from typing import Any

import numpy as np

# Adapter trees under ``{output_root}``, as (tree, globs to each run's checkpoints dir).
# teachers: {animal}/{base}/{tag}/ftseed{N}/checkpoints
# students: {animal}/{base}/teacher-{tbase}/{tag}_tseed{T}/{dataset_tag}/{tag}/ftseed{N}/
#           checkpoints  (lineage == the 3 middle segments: teacher-base/teacher-tag/dataset)
# A revision-pinned base gathers its {base} level under a `ckpt/` component (right after
# {animal}), so each tree has a second, one-deeper glob anchored on the literal `ckpt/`;
# the matched parts are normalised (the `ckpt` segment stripped) so the fixed-index logic
# below sees the plain shape with the model name keeping its `@rev` suffix.
# A non-default optimizer segregates the whole student tree under a leading `{optim_slug}`
# level (`students/sgd/...`), so the student patterns add one- and two-deeper variants; the
# leading slug is stripped in normalisation (closed-vocab detection, not by position), so a
# sgd run is discovered and read back at the plain shape rather than skipped or misparsed.
_TREES: dict[str, tuple[str, tuple[str, ...]]] = {
    "teacher": ("teachers", ("*/*/*/ftseed*/checkpoints", "*/ckpt/*/*/ftseed*/checkpoints")),
    "student": (
        "students",
        (
            "*/*/*/*/*/*/ftseed*/checkpoints",
            "*/ckpt/*/*/*/*/*/ftseed*/checkpoints",
            "*/*/*/*/*/*/*/ftseed*/checkpoints",
            "*/*/*/*/*/*/*/*/ftseed*/checkpoints",
        ),
    ),
}


def _strip_optim_level(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a leading non-default-optimizer `{optim_slug}` level, if present.

    Keyed on the closed optimizer-slug vocabulary (`parse_optim_slug`), never on "looks
    unusual": an animal is never named like an optimizer slug, and a real leading component
    (`{animal}` / the literal `controls`) fails to parse, so it is left untouched.
    """
    if parts:
        try:
            parse_optim_slug(parts[0])
        except ValueError:
            return parts
        return parts[1:]
    return parts
_FTSEED_RE = re.compile(r"^ftseed(\d+)$")
_STEP_RE = re.compile(r"^checkpoint-(\d+)$")

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


@dataclass(frozen=True)
class AdapterRef:
    """One measurable adapter artifact.

    Attributes:
        kind: "teacher" or "student".
        animal: Target animal segment of the output tree.
        base_model: Base-model directory name, e.g. ``pythia-160m-data-seed1``.
        lineage: Students only — ``teacher-{tbase}/{tag}_tseed{T}/{dataset_tag}`` (three
            segments); ``None`` for teachers.
        tag: Hyperparameter tag, e.g. ``ep20_lr1e-4_r8a32``.
        ftseed: Fine-tuning seed. Seeds BOTH the LoRA init and the data sampler
            (`teachers.sft_io` passes it to `TrainingArguments`), so replicates differ
            in init *and* shuffle order — the two are not separable via this field.
        run_dir: The run's ``checkpoints/`` dir (holds ``adapter_config.json``).
        weights_path: The safetensors actually measured.
        step: ``None`` for the final adapter, else the checkpoint step.
    """

    kind: str
    animal: str
    base_model: str
    lineage: str | None
    tag: str
    ftseed: int
    run_dir: Path
    weights_path: Path
    step: int | None

    @property
    def key(self) -> str:
        """Stable identity used for resume (one line per (adapter, step))."""
        parts = [self.kind, self.animal, self.base_model, self.lineage or "-", self.tag,
                 f"ftseed{self.ftseed}", "final" if self.step is None else str(self.step)]
        return "|".join(parts)

    def weights_sha256(self) -> str:
        """Content hash of the measured adapter — the identity key that matters.

        Carried in the measuring cell's ``run_identity`` so an adapter retrained in place
        under an unchanged path surfaces through `run_gate.gate_action` as a diffed
        identity drift, rather than silently serving rows from the previous weights.
        """
        return _sha256(self.weights_path)

    def config_sha256(self) -> str:
        """Content hash of ``adapter_config.json`` (r, alpha, target_modules, base repo)."""
        return _sha256(self.run_dir / ADAPTER_CONFIG)

    def train_run_identity(self) -> dict[str, Any] | None:
        """The producing training run's own ``run_identity``, or ``None`` if absent.

        Embedding it links a measured row to the number dataset (`filtered_sha256`),
        subsample seed, train seed, and LoRA config that produced the adapter — so a
        refreshed upstream dataset invalidates the measurement, exactly as it invalidates
        the student that was trained on it.
        """
        manifest = self.run_dir.parent / "train_manifest.json"
        if not manifest.exists():
            return None
        return json.loads(manifest.read_text()).get("run_identity")

    def identity(self) -> dict[str, Any]:
        """The non-derived columns every emitted row carries.

        ``adapter_key`` is embedded so a rows file is self-reconciling against the
        progress ledger: rows whose adapter never completed can be dropped without
        re-deriving the key from the other columns.
        """
        return {"adapter_key": self.key, "kind": self.kind, "animal": self.animal,
                "base_model": self.base_model, "lineage": self.lineage,
                "param_tag": self.tag, "ftseed": self.ftseed, "step": self.step}


def _sha256(path: Path) -> str:
    """Streamed sha256 of a file (adapters are a few MB, so one pass is cheap)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_snapshot_sha(hf_cache_root: Path, base_repo: str,
                      revision: str | None = None) -> str | None:
    """Resolved HF commit sha of a cached model (at `revision`), or ``None`` if unresolvable.

    ``from_pretrained`` follows ``refs/{revision}`` (``main`` by default), and this cache
    genuinely holds more than one snapshot for some repos (every Pythia ``stepN``) — so
    keying the W_0 norm cache on the repo *name* alone would let a re-downloaded base or
    another revision silently serve stale spectral norms. `analysis.w0_norms` keys on this.
    A 40-hex sha passed as `revision` is returned as is.
    """
    rev = revision or "main"
    if re.fullmatch(r"[0-9a-f]{40}", rev):
        return rev
    ref = (Path(hf_cache_root) / "hub"
           / f"models--{base_repo.replace('/', '--')}" / "refs" / rev)
    try:
        return ref.read_text().strip() or None
    except OSError:
        return None


def _matches(value: str, patterns: Sequence[str] | None) -> bool:
    """True if `patterns` is empty/None or `value` matches any fnmatch pattern."""
    return not patterns or any(fnmatch.fnmatch(value, p) for p in patterns)


def discover_adapters(
    output_root: Path,
    kinds: Sequence[str] = ("student",),
    animals: Sequence[str] | None = None,
    base_models: Sequence[str] | None = None,
    lineages: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    ftseeds: Sequence[int] | None = None,
    checkpoints: bool = False,
    gate_nonfinite: bool = True,
) -> list[AdapterRef]:
    """Enumerate adapter artifacts under `output_root` matching the selectors.

    All string selectors are fnmatch patterns (``ep20_*_r32a4``), so a config can carve
    the surface without listing cells. An empty/None selector means "no constraint".

    Args:
        checkpoints: Also emit every intermediate ``checkpoint-N/`` of each matched run
            (growth curves). The final adapter is always emitted when present.

    Returns:
        Refs sorted by (base_model, ftseed, tag, step) — the order the pipeline relies
        on to reconstruct each ``A_init`` once per (base, seed, adapter-config) group.
    """
    refs: list[AdapterRef] = []
    for kind in kinds:
        if kind not in _TREES:
            raise ValueError(f"unknown adapter kind {kind!r}; expected {sorted(_TREES)}")
        tree, patterns = _TREES[kind]
        root = Path(output_root) / tree
        if not root.is_dir():
            continue
        # The two patterns match disjoint depths (plain vs the one-deeper `ckpt/` layout), so
        # the union never double-counts (the set is belt-and-braces); sorting full paths keeps
        # discovery deterministic and leaves the plain entries in their original relative order.
        for ckpt_dir in sorted({d for pat in patterns for d in root.glob(pat)}):
            if not (ckpt_dir / ADAPTER_CONFIG).exists():
                continue
            rel = ckpt_dir.relative_to(root).parts
            # Drop a leading optimizer-slug level (non-default-optimizer segregation) so the
            # fixed-index logic sees the plain shape; the `_{optim_slug}` tag suffix on the
            # student-tag leaf still keeps sgd/adamw cells disjoint in `AdapterRef.key`.
            if kind == "student":
                rel = _strip_optim_level(rel)
            # Drop the gathered `ckpt/` model level (revision runs) so the fixed indices
            # below match the plain layout; the model name keeps its `@rev` suffix.
            if len(rel) > 1 and rel[1] == "ckpt":
                rel = rel[:1] + rel[2:]
            # Base-control student trees (controls/base/...) now sit at the treatment
            # star-count; they are not treatment adapters, so skip them as the old shallower
            # pattern did (it never matched them).
            if kind == "student" and rel and rel[0] == "controls":
                continue
            animal, base_model = rel[0], rel[1]
            lineage = "/".join(rel[2:-3]) if kind == "student" else None
            tag, ftseed_dir = rel[-3], rel[-2]
            m = _FTSEED_RE.match(ftseed_dir)
            if m is None:
                continue
            ftseed = int(m.group(1))
            if not (_matches(animal, animals) and _matches(base_model, base_models)
                    and _matches(tag, tags)
                    and (lineage is None or _matches(lineage, lineages))
                    and (not ftseeds or ftseed in ftseeds)):
                continue

            def _ref(weights: Path, step: int | None) -> AdapterRef:
                return AdapterRef(kind=kind, animal=animal, base_model=base_model,
                                  lineage=lineage, tag=tag, ftseed=ftseed,
                                  run_dir=ckpt_dir, weights_path=weights, step=step)

            if (ckpt_dir / ADAPTER_WEIGHTS).exists():
                # A diverged run is saved all-NaN; every decomposition from it is NaN and
                # would silently poison downstream aggregates. Skip LOUDLY, never quietly.
                # `gate_nonfinite=False` defers this to the consumer (it costs a full
                # safetensors load per artifact -- ~0.17 s x 13k finals at discovery
                # time); the decomp pipeline gates when it loads the tensors anyway.
                n_bad = checkpoint_nonfinite_count(ckpt_dir) if gate_nonfinite else 0
                if n_bad:
                    logger.warning(
                        "SKIPPING diverged adapter {} ({} non-finite weight value(s)); "
                        "it is excluded from this decomposition", ckpt_dir, n_bad)
                else:
                    refs.append(_ref(ckpt_dir / ADAPTER_WEIGHTS, None))
            if checkpoints:
                for sub in ckpt_dir.glob("checkpoint-*"):
                    sm = _STEP_RE.match(sub.name)
                    if not (sm and (sub / ADAPTER_WEIGHTS).exists()):
                        continue
                    # Same gate as the final adapter: a mid-run NaN checkpoint would
                    # otherwise put NaN rows straight into rows.jsonl.
                    n_bad = checkpoint_nonfinite_count(sub) if gate_nonfinite else 0
                    if n_bad:
                        logger.warning(
                            "SKIPPING diverged checkpoint {} ({} non-finite weight "
                            "value(s)); it is excluded from this decomposition", sub, n_bad)
                        continue
                    refs.append(_ref(sub / ADAPTER_WEIGHTS, int(sm.group(1))))
    refs.sort(key=lambda r: (r.base_model, r.ftseed, r.tag, -1 if r.step is None else r.step))
    return refs


def adapter_config(ref: AdapterRef) -> dict[str, Any]:
    """Load the run's stored ``adapter_config.json`` (resolved target_modules and all)."""
    return json.loads((ref.run_dir / ADAPTER_CONFIG).read_text())


def base_repo_and_revision(ref: AdapterRef) -> tuple[str, str | None]:
    """`base_repo_and_revision_for_run` for an `AdapterRef` (its ``run_dir``)."""
    return base_repo_and_revision_for_run(ref.run_dir)


def base_repo_and_revision_for_run(run_dir: Path) -> tuple[str, str | None]:
    """The run's base model as ``(repo_id, revision-or-None)``, given its ``checkpoints/``.

    PEFT's ``adapter_config.json`` records only the bare repo id, so a branch- or
    step-pinned base (every DataDecide seed; any Pythia ``@stepN``) would load HF `main`
    if it were trusted. The producing run's ``train_manifest.json`` keeps the qualified
    ``base_model_repo`` (``{repo}@{revision}``), which is the ref this reads.

    Falls back to the adapter config when the manifest is missing/unreadable — but a
    DataDecide repo without a revision is always wrong (the three seeds are branches of
    one repo), so that case raises rather than silently loading the default seed.
    """
    manifest = Path(run_dir).parent / "train_manifest.json"
    try:
        model_ref = json.loads(manifest.read_text()).get("base_model_repo")
    except (OSError, json.JSONDecodeError):
        model_ref = None
    if model_ref:
        repo, _, revision = str(model_ref).partition("@")
        return repo, revision or None

    from subliminal_learning.models.datadecide import is_dd_ref

    repo = json.loads((Path(run_dir) / ADAPTER_CONFIG).read_text())["base_model_name_or_path"]
    if is_dd_ref(repo):
        raise ValueError(
            f"{manifest} is missing/unreadable and {repo} is a DataDecide repo whose "
            "seeds are branches of one repo — the revision cannot be recovered from "
            "adapter_config.json, so this base cannot be loaded.")
    return repo, None


def target_modules_key(value: Any) -> str | tuple[str, ...]:
    """Hashable, order-stable cache key for a config's ``target_modules``.

    A list becomes its sorted tuple (as always). A string — the OLMo LoRA configs give
    one regex, and ``"all-linear"`` is a sentinel — stays one atom: `sorted` would shred
    it into characters, and a str can never collide with the tuple a list produces.
    """
    return value if isinstance(value, str) else tuple(sorted(value))


def lora_target_modules(value: Any) -> str | list[str]:
    """A config's ``target_modules`` in the form `LoraConfig` wants it.

    A regex string (or ``"all-linear"``) is passed through untouched; a list is sorted,
    as before, so a stored adapter's module set reconstructs order-independently.
    """
    return value if isinstance(value, str) else sorted(value)


def init_group_key(ref: AdapterRef, cfg: Mapping[str, Any]) -> tuple:
    """Everything `A_init` depends on — refs sharing this key share one reconstruction."""
    return (cfg["base_model_name_or_path"], ref.ftseed, int(cfg["r"]),
            target_modules_key(cfg["target_modules"]))


# --------------------------------------------------------------------------------------
# Norms. Frobenius never needs a factorisation; spectral exploits the low-rank structure.
# --------------------------------------------------------------------------------------


def fro(m: np.ndarray) -> float:
    """Frobenius norm — a sum of squares, no factorisation."""
    return float(np.sqrt(np.square(np.asarray(m, dtype=np.float64)).sum()))


def thin_norms(m: np.ndarray) -> tuple[float, float]:
    """(spectral, Frobenius) of a thin/wide matrix via its small Gram matrix.

    For an ``r x d`` factor (r <= 64, d up to 3072) the Gram ``m @ m.T`` is ``r x r``, so
    the spectral norm costs O(r²d) instead of a full SVD. The largest eigenvalue is
    well-conditioned under `eigvalsh`, so squaring the condition number is harmless here.
    """
    m = np.asarray(m, dtype=np.float64)
    gram = m @ m.T if m.shape[0] <= m.shape[1] else m.T @ m
    top = float(np.linalg.eigvalsh(gram)[-1])
    return float(np.sqrt(max(top, 0.0))), fro(m)


def low_rank_singular_values(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """All (<= r) nonzero singular values of ``B @ A`` **without forming it**, descending.

    ``B`` is ``d_out x r`` and ``A`` is ``r x d_in``, so ``BA`` has rank <= r. With thin
    QR factorisations ``B = Q_b R_b`` and ``A.T = Q_a R_a``::

        B A = Q_b (R_b R_a.T) Q_a.T

    and since ``Q_b``/``Q_a`` have orthonormal columns the singular values of ``BA`` are
    exactly those of the ``r x r`` matrix ``R_b R_a.T``. Exact, not an approximation —
    ~30x cheaper than a full SVD of a 3072x768 product at r=32 (measured 2026-08-22,
    2 BLAS threads; agrees with the full SVD to 1e-12).
    """
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if b.shape[1] != a.shape[0]:
        raise ValueError(f"inner dims disagree: B{b.shape} @ A{a.shape}")
    _, r_b = np.linalg.qr(b)
    _, r_a = np.linalg.qr(a.T)
    return np.linalg.svd(r_b @ r_a.T, compute_uv=False)


def low_rank_norms(b: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    """(spectral, Frobenius) of the product ``B @ A`` via `low_rank_singular_values`."""
    sv = low_rank_singular_values(b, a)
    return float(sv[0]), float(np.sqrt(np.square(sv).sum()))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Flattened (Frobenius inner-product) cosine between two same-shaped matrices.

    Gauge-dependent: LoRA admits ``(A, B) -> (GA, BG^-1)`` with ΔW unchanged, so this is
    a statement about the realised parameterisation, not about the adapter's function.
    See `notebooks/collapse_mechanism.ipynb` §1a.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / denom) if denom else 0.0


def chance_cos_floor(shape: tuple[int, ...]) -> float:
    """Expected |cos| between two independent draws of this shape: ``1/sqrt(n)``.

    The scale the wrong-seed control must sit at for a reconstruction to mean anything.
    """
    return float(1.0 / np.sqrt(max(int(np.prod(shape)), 1)))


def decompose(
    a_init: np.ndarray,
    a_final: np.ndarray,
    b_final: np.ndarray,
    a_wrongseed: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-module norms of the decomposition, plus the reconstruction diagnostics.

    Emits spectral/Frobenius norms of ``A_init``, ``A_final``, ``ΔA``, ``B`` and of the
    three products ``B@A_init`` (cross), ``B@ΔA`` (quadratic), ``B@A_final`` (total,
    identically ``ΔW/s``); the init cosine; the direct alignment ``u = cos(ΔA, A_init)``;
    and the wrong-seed control when `a_wrongseed` is given.
    """
    d_a = np.asarray(a_final, dtype=np.float64) - np.asarray(a_init, dtype=np.float64)
    row: dict[str, float] = {}
    for name, m in (("A_init", a_init), ("A_final", a_final), ("dA", d_a), ("B", b_final)):
        row[f"{name}_spec"], row[f"{name}_fro"] = thin_norms(m)
    for name, right in (("B_Ainit", a_init), ("B_dA", d_a), ("B_Afinal", a_final)):
        row[f"{name}_spec"], row[f"{name}_fro"] = low_rank_norms(b_final, right)
    row["cos_Ainit_Afinal"] = cos(a_init, a_final)
    # u measured directly, rather than inverted out of the cos identity as the notebook
    # first did — same quantity, no quadratic solve, defined even when eps -> 0.
    row["u_align"] = cos(d_a, a_init)
    row["eps_fro"] = row["dA_fro"] / row["A_init_fro"] if row["A_init_fro"] else 0.0
    row["cos_chance_floor"] = chance_cos_floor(np.shape(a_init))
    if a_wrongseed is not None:
        row["cos_wrongseed_Afinal"] = cos(a_wrongseed, a_final)
    return row


# --------------------------------------------------------------------------------------
# Base-model construction and the seeded init reconstruction.
# --------------------------------------------------------------------------------------


def base_model_factory(base_repo: str, hf_cache_root: Path, device: str = "cpu",
                       revision: str | None = None, dtype: str = "float32") -> Callable:
    """A zero-arg factory returning a fresh base model on `device` (fp32 by default).

    A factory rather than a model because `get_peft_model` wraps in place: each seeded
    reconstruction needs an unwrapped model, and tests need to substitute a tiny one.

    `revision` pins a pretraining checkpoint; when None the call is byte-identical to
    the historical revision-less one (`main`).
    """

    def _load():
        import torch
        from transformers import AutoModelForCausalLM

        from subliminal_learning.models.datadecide import _AI2_OLMO_HINT, is_dd_ref

        try:
            # Registers the legacy `hf_olmo` model type for DataDecide bases; the
            # datadecide dependency group is optional, so absence is only an error when
            # the repo being loaded actually needs it (else Auto* dies with an opaque
            # "unrecognized model type").
            import hf_olmo  # noqa: F401
        except ImportError as exc:
            if is_dd_ref(base_repo):
                raise RuntimeError(_AI2_OLMO_HINT) from exc

        kwargs: dict = {"cache_dir": str(Path(hf_cache_root) / "hub"),
                        "torch_dtype": getattr(torch, dtype)}
        if revision is not None:
            kwargs["revision"] = revision
        return AutoModelForCausalLM.from_pretrained(base_repo, **kwargs).to(device)

    return _load


def base_model_factory_for_ref(model_ref: str, hf_cache_root: Path, device: str = "cpu",
                               dtype: str = "float32") -> Callable:
    """`base_model_factory` for a manifest ``base_model_repo`` ref (``{repo}[@{rev}]``).

    The ref is the only spelling that carries a DataDecide seed (a branch of one repo),
    so an `A_init` replay must be built from it rather than from `adapter_config.json`'s
    bare repo id. DataDecide refs route through the pinned `hf_olmo` loader (which
    `_require_hf_olmo`s first); everything else keeps the historical `base_model_factory`
    path, with `revision` set only when the ref carries one.
    """
    from subliminal_learning.models.datadecide import is_dd_ref

    if is_dd_ref(model_ref):
        from subliminal_learning.models.datadecide import load_datadecide_model, parse_dd_ref

        spec = parse_dd_ref(model_ref)

        def _load_dd():
            import torch

            return load_datadecide_model(spec, Path(hf_cache_root), device,
                                         dtype=getattr(torch, dtype))

        return _load_dd
    repo, _, revision = model_ref.partition("@")
    return base_model_factory(repo, hf_cache_root, device, revision=revision or None,
                              dtype=dtype)


def lora_init_weights(
    model_factory: Callable, seed: int, adapter_cfg: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Reconstruct ``{module: lora_A init weight}`` (fp64) by replaying the seeded wrap.

    Mirrors the training call order exactly — build the model, ``set_seed(seed)``,
    ``get_peft_model`` — which is only faithful because `teachers.sft_io` seeds
    immediately before the PEFT wrap and nothing consumes torch RNG in between. PEFT
    draws ``lora_A`` on CPU before moving the layer, so a CPU reconstruction reproduces a
    CUDA-trained init; the wrong-seed control is what actually proves this per run.
    """
    from peft import get_peft_model
    from transformers import set_seed

    from subliminal_learning.config import LoRAConfig as LoRACfg
    from subliminal_learning.training.lora_io import build_lora_config

    tm = adapter_cfg["target_modules"]
    # The SAME builder training uses (teachers/sft_io, sdf, gkd): the replay cannot drift
    # from the wrap it mirrors. A list is sorted so a stored module set reconstructs
    # order-independently; a regex / "all-linear" string passes through untouched.
    cfg = LoRACfg(r=int(adapter_cfg["r"]), lora_alpha=int(adapter_cfg["lora_alpha"]),
                  lora_dropout=float(adapter_cfg.get("lora_dropout", 0.0)),
                  target_modules=tm if isinstance(tm, str) else tuple(sorted(tm)))
    model = model_factory()
    set_seed(seed)
    model = get_peft_model(model, build_lora_config(cfg))
    out: dict[str, np.ndarray] = {}
    for name, mod in model.named_modules():
        if "default" in getattr(mod, "lora_A", {}):
            key = name.replace("base_model.model.", "")
            out[key] = mod.lora_A["default"].weight.detach().cpu().double().numpy()
    del model
    return out


def base_weight_norms(model_factory: Callable, target_modules: str | Iterable[str]) -> dict:
    """(spectral, Frobenius) of every frozen ``W_0`` a LoRA module perturbs.

    The one genuinely full-rank SVD in this module — hence computed once per base repo
    and cached to disk by the pipeline, never per adapter.

    `target_modules` is matched the way PEFT matches it, so the selected modules are the
    ones the adapter actually wraps: a list by last-name-segment membership, a string
    (the OLMo configs' regex) by `re.fullmatch` against the whole module name.
    """
    pattern = re.compile(target_modules) if isinstance(target_modules, str) else None
    wanted = set() if pattern is not None else set(target_modules)

    def matches(name: str) -> bool:
        if pattern is not None:
            return pattern.fullmatch(name) is not None
        return name.rsplit(".", 1)[-1] in wanted

    model = model_factory()
    out: dict[str, tuple[float, float]] = {}
    for name, mod in model.named_modules():
        if matches(name) and hasattr(mod, "weight"):
            w = mod.weight.detach().cpu().double().numpy()
            sv = np.linalg.svd(w, compute_uv=False)
            out[name] = (float(sv[0]), float(np.sqrt(np.square(sv).sum())))
    del model
    return out
