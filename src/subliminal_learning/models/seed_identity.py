"""Weight-only tests of what two pretraining "seeds" actually share: init and data order.

Pretraining seed families (PolyPythias, DataDecide) *claim* per-branch reseeding of the
initialization and/or the data order. Checkpoints provide evidence about saved
parameters, optimizer history, and coverage, but do not by themselves establish
exact training sequence or full optimizer state:

- **Init** -- `init_equality`: two authentic initialization state dicts compared
  tensor-by-tensor with ``torch.equal`` (exact numeric equality). The caller must
  supply the appropriate revision: inspect both Pythia step0 and step1 to detect
  the parameter reset in released ws1–3. DataDecide uses step0. Equal parameters
  do not imply equal optimizer state or history.
- **Data order** -- `untouched_rows` + `compare_untouched`: the *untouched-embedding-row
  fingerprint*. Between two checkpoints, an input-embedding row whose token never appeared
  in the current window can still receive an Adam update from momentum/history; with weight
  decay it may also be rescaled. Thus this fingerprint measures coverage/history evidence,
  not exact sequence; rounding can cause false negatives. Identical data order tends to give
  similar sets, while different order may disagree near a window edge. Rows past
  ``vocab_size`` (embedding padded to a multiple of 128) are never used and
  serve as the positive control for the residual threshold. The test needs an UNTIED
  embedding: with weight tying every row gets a softmax gradient at every step.

The August 27 equality analysis found that PolyPythias weight-seed branches share
saved step1 parameters with the original. Independent gradient reconstruction on
September 13 provided evidence for seed-specific optimizer history in released ws1–3;
this helper itself only compares supplied weights. It is also the one
`scripts.check_dd_seed_identity` runs on the DataDecide seed branches. See
``run_logs/ws_origin_investigation_20260913/OPTIMIZER_FINDINGS.md`` for the separate
reconstruction. Pure numpy/torch on in-memory tensors; downloading and locating
checkpoints is the caller's job.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# fp32 rounding on a rescaled row is ~1e-7 relative; a single Adam step on a touched row is
# orders of magnitude larger at pretraining learning rates.
DEFAULT_TOL = 1e-6
# Ground truth: a few seen-but-unchanged rows are expected (pythia-160m step1->2: token 20746).
DEFAULT_MAX_GROUND_TRUTH_MISMATCHES = 5
# Other-corpus runs (DCLM) vs Pile truth: below this Jaccard counts as far from Pile.
DEFAULT_FAR_JACCARD = 0.9

UntouchedMethod = Literal["residual", "exact"]
# Non-parameter buffers GPT-NeoX checkpoints carry (causal mask, rotary frequencies).
PYTHIA_BUFFER_SUFFIXES = (".attention.bias", ".attention.masked_bias", ".rotary_emb.inv_freq")


@dataclass(frozen=True)
class InitEquality:
    """Per-tensor exact comparison of two state dicts."""

    n_tensors: int
    n_equal: int
    equal_tensors: tuple[str, ...]
    max_abs_diff_over_unequal: float
    example_unequal: dict[str, float] = field(default_factory=dict)

    @property
    def shared(self) -> bool:
        """True iff every tensor is byte-equal (same init)."""
        return self.n_equal == self.n_tensors and self.n_tensors > 0


@dataclass(frozen=True)
class UntouchedRows:
    """Untouched input-embedding rows between two checkpoints of ONE run."""

    untouched: tuple[int, ...]
    pad_rows: int
    pad_resid_max: float
    pad_scale_min: float
    pad_scale_max: float
    vocab_resid_quantiles: dict[str, float]
    untouched_scale_min: float
    untouched_scale_max: float
    method: UntouchedMethod = "residual"
    tol: float = DEFAULT_TOL
    n_pad_untouched: int = 0

    @property
    def n_untouched(self) -> int:
        return len(self.untouched)

    def control_ok(self, tol: float | None = None) -> bool:
        """The never-used padding rows must ALL count as untouched (or there are none).

        ``residual``: max pad residual <= `tol` (default: the tol the rows were built with).
        ``exact``: every pad row is value-equal across the window (`tol` is ignored).
        """
        if self.method == "exact":
            return self.n_pad_untouched == self.pad_rows
        return self.pad_rows == 0 or self.pad_resid_max <= (self.tol if tol is None else tol)

    def summary(self) -> dict:
        d = self.__dict__.copy()
        d.pop("untouched")
        d["n_untouched"] = self.n_untouched
        if self.method == "residual":  # keep the pre-existing (DataDecide) JSON schema
            for k in ("method", "tol", "n_pad_untouched"):
                d.pop(k)
        return d


@dataclass(frozen=True)
class OrderComparison:
    """Agreement of two runs' untouched sets over the same window."""

    n_a: int
    n_b: int
    n_intersection: int
    n_union: int
    only_a: tuple[int, ...]
    only_b: tuple[int, ...]
    control_ok: bool = True
    max_mismatches: int = 0

    @property
    def jaccard(self) -> float:
        return self.n_intersection / self.n_union if self.n_union else float("nan")

    @property
    def n_mismatches(self) -> int:
        """Ids untouched in exactly one run (symmetric difference size)."""
        return self.n_union - self.n_intersection

    @property
    def shared(self) -> bool | None:
        """True = same order (at most ``max_mismatches`` one-sided ids; 0 = identical sets),
        False = different, None = inconclusive (both sets empty, or either side failed its
        padding-row control)."""
        if self.n_union == 0 or not self.control_ok:
            return None
        return self.n_mismatches <= self.max_mismatches

    @property
    def verdict(self) -> str:
        s = self.shared
        if s is None:
            return "INCONCLUSIVE"
        if not s:
            return "DIFFERENT"
        return "SHARED" if self.n_mismatches == 0 else "SHARED_WITH_EXCEPTIONS"


def init_equality(a: dict, b: dict, n_examples: int = 5) -> InitEquality:
    """Compare two state dicts tensor-by-tensor with ``torch.equal``.

    Args:
        a: State dict (name -> tensor) of the first run's authentic init checkpoint.
        b: State dict of the second.
        n_examples: How many unequal tensors to list with their max |diff|.

    Returns:
        An `InitEquality`; tensors present in only one dict count as unequal.
    """
    import torch

    names = sorted(set(a) | set(b))
    equal = tuple(n for n in names if n in a and n in b and torch.equal(a[n], b[n]))
    diffs = {n: float((a[n].double() - b[n].double()).abs().max())
             for n in names if n in a and n in b and n not in equal}
    return InitEquality(
        n_tensors=len(names), n_equal=len(equal), equal_tensors=equal,
        max_abs_diff_over_unequal=max(diffs.values()) if diffs else 0.0,
        example_unequal=dict(sorted(diffs.items())[:n_examples]),
    )


def untouched_rows(w0, wt, vocab_size: int, tol: float = DEFAULT_TOL,
                   method: UntouchedMethod = "residual") -> UntouchedRows:
    """Rows of an input embedding that received no Adam update between two checkpoints.

    ``method="residual"`` (default; fp32 checkpoints such as DataDecide): per row, the scale
    ``c`` is fitted by least squares and the relative residual ``||wt - c*w0|| / ||w0||``
    computed in fp64. Untouched rows have residual at rounding level (and share the common
    weight-decay factor ``c``); touched rows do not.

    ``method="exact"`` (canonical for fp16 Pythia checkpoints): a row is untouched iff it is
    value-equal (``-0.0 == +0.0``) across the window. In fp16 an early update can be ~1
    quantum on a handful of small-|w| elements, which the residual test cannot see (it
    called 967 of 1654 seen rows untouched on pythia-160m step1->2). The residual
    diagnostics are still filled in.

    Args:
        w0: Embedding matrix at the earlier checkpoint (torch tensor or array,
            ``[embedding_size, d]``).
        wt: The same matrix at the later checkpoint.
        vocab_size: Rows ``< vocab_size`` are real tokens; the rest are the padding control.
        tol: Relative-residual threshold for "untouched" (``residual`` only).
        method: ``"residual"`` or ``"exact"``.

    Returns:
        An `UntouchedRows` with the untouched token ids and the control diagnostics.
    """
    w0 = np.asarray(_to_numpy(w0), dtype=np.float64)
    wt = np.asarray(_to_numpy(wt), dtype=np.float64)
    if w0.shape != wt.shape:
        raise ValueError(f"shape mismatch {w0.shape} vs {wt.shape}")
    dot = (w0 * wt).sum(1)
    nrm = (w0 * w0).sum(1)
    c = np.where(nrm > 0, dot / np.maximum(nrm, 1e-300), 1.0)
    resid = np.linalg.norm(wt - c[:, None] * w0, axis=1) / np.maximum(np.sqrt(nrm), 1e-300)
    vocab, pad = resid[:vocab_size], resid[vocab_size:]
    if method == "exact":
        same = (w0 == wt).all(1)
    elif method == "residual":
        same = resid <= tol
    else:
        raise ValueError(f"unknown untouched-row method {method!r}")
    untouched = np.flatnonzero(same[:vocab_size])
    return UntouchedRows(
        untouched=tuple(int(i) for i in untouched),
        pad_rows=int(len(pad)),
        pad_resid_max=float(pad.max()) if len(pad) else float("nan"),
        pad_scale_min=float(c[vocab_size:].min()) if len(pad) else float("nan"),
        pad_scale_max=float(c[vocab_size:].max()) if len(pad) else float("nan"),
        vocab_resid_quantiles={q: float(np.quantile(vocab, float(q)))
                               for q in ("0.001", "0.01", "0.1", "0.5")},
        untouched_scale_min=float(c[untouched].min()) if len(untouched) else float("nan"),
        untouched_scale_max=float(c[untouched].max()) if len(untouched) else float("nan"),
        method=method, tol=tol, n_pad_untouched=int(same[vocab_size:].sum()),
    )


def compare_untouched(a: UntouchedRows, b: UntouchedRows, n_examples: int | None = 50,
                      max_mismatches: int = 0) -> OrderComparison:
    """Jaccard agreement of two runs' untouched sets over the same checkpoint window.

    Args:
        a: Fingerprint of the first run.
        b: Fingerprint of the second run.
        n_examples: How many one-sided token ids to keep per side (None = all).
        max_mismatches: Largest symmetric-difference size still counted as shared (the
            Pythia check uses the ground-truth allowance: fp16 rounding can freeze a row).

    Returns:
        An `OrderComparison`; ``shared`` is True for sets differing by at most
        `max_mismatches` ids (default 0: IDENTICAL sets), None when both
        are empty (window saturated -- every token seen -- so the test cannot speak) or when
        either side's padding-row control fails. Padding rows never enter the sets.
    """
    ua, ub = set(a.untouched), set(b.untouched)
    return OrderComparison(
        n_a=len(ua), n_b=len(ub), n_intersection=len(ua & ub), n_union=len(ua | ub),
        only_a=tuple(sorted(ua - ub)[:n_examples]), only_b=tuple(sorted(ub - ua)[:n_examples]),
        control_ok=a.control_ok() and b.control_ok(), max_mismatches=max_mismatches,
    )


@dataclass(frozen=True)
class PaddingComparison:
    """Padding rows of a later checkpoint vs a reference (e.g. an init).

    Padding rows are never inputs, so between two checkpoints of ONE run they change only by
    weight decay: ``wt = c * w0`` with residual at rounding level and cosine 1.
    """

    n_rows: int
    n_exact_equal: int
    scale_min: float
    scale_max: float
    resid_max: float
    resid_median: float
    cos_min: float
    cos_max: float


def compare_padding_rows(w_ref, w_later, vocab_size: int) -> PaddingComparison:
    """Per-row least-squares scale ``c = <w0, wt> / ||w0||^2``, relative residual
    ``||wt - c w0|| / ||w0||`` and cosine of the padding rows (``>= vocab_size``), in fp64.

    Args:
        w_ref: Reference embedding matrix ``[embedding_size, d]`` (or only its padding rows
            when ``vocab_size`` is 0).
        w_later: The later matrix, same shape.
        vocab_size: Rows below this are real tokens and are ignored.
    """
    w0 = np.asarray(_to_numpy(w_ref), dtype=np.float64)[vocab_size:]
    wt = np.asarray(_to_numpy(w_later), dtype=np.float64)[vocab_size:]
    if w0.shape != wt.shape:
        raise ValueError(f"shape mismatch {w0.shape} vs {wt.shape}")
    if len(w0) == 0:
        nan = float("nan")
        return PaddingComparison(0, 0, nan, nan, nan, nan, nan, nan)
    n0 = np.linalg.norm(w0, axis=1)
    nt = np.linalg.norm(wt, axis=1)
    dot = (w0 * wt).sum(1)
    c = dot / np.maximum(n0 * n0, 1e-300)
    resid = np.linalg.norm(wt - c[:, None] * w0, axis=1) / np.maximum(n0, 1e-300)
    cos = dot / np.maximum(n0 * nt, 1e-300)
    return PaddingComparison(
        n_rows=int(len(w0)), n_exact_equal=int((w0 == wt).all(1).sum()),
        scale_min=float(c.min()), scale_max=float(c.max()),
        resid_max=float(resid.max()), resid_median=float(np.median(resid)),
        cos_min=float(cos.min()), cos_max=float(cos.max()))


@dataclass(frozen=True)
class GroundTruthComparison:
    """A run's untouched set vs the corpus truth (vocab tokens never seen before the window end).

    ``run_minus_truth`` (seen but unchanged) and ``truth_minus_run`` (never seen but changed)
    are kept in full so exceptions are reviewable.
    """

    n_truth: int
    n_run: int
    n_intersection: int
    n_union: int
    run_minus_truth: tuple[int, ...]
    truth_minus_run: tuple[int, ...]
    n_run_minus_truth: int
    n_truth_minus_run: int
    control_ok: bool = True
    max_mismatches: int = DEFAULT_MAX_GROUND_TRUTH_MISMATCHES

    @property
    def jaccard(self) -> float:
        return self.n_intersection / self.n_union if self.n_union else float("nan")

    @property
    def n_mismatches(self) -> int:
        return self.n_run_minus_truth + self.n_truth_minus_run

    @property
    def verdict(self) -> str:
        """MATCH (no mismatches) / MATCH_WITH_EXCEPTIONS (<= ``max_mismatches``) / MISMATCH,
        or INCONCLUSIVE when the truth set is empty (saturated window) or the run's padding
        control failed. For runs trained on this corpus."""
        if self.n_truth == 0 or not self.control_ok:
            return "INCONCLUSIVE"
        if self.n_mismatches == 0:
            return "MATCH"
        return "MATCH_WITH_EXCEPTIONS" if self.n_mismatches <= self.max_mismatches else "MISMATCH"

    def far_verdict(self, max_jaccard: float = DEFAULT_FAR_JACCARD) -> str:
        """Expected-far namespace for runs trained on ANOTHER corpus: FAR_FROM_PILE when
        Jaccard < `max_jaccard`, else UNEXPECTEDLY_CLOSE_TO_PILE; INCONCLUSIVE as `verdict`."""
        if self.n_truth == 0 or not self.control_ok:
            return "INCONCLUSIVE"
        return "FAR_FROM_PILE" if self.jaccard < max_jaccard else "UNEXPECTEDLY_CLOSE_TO_PILE"


def compare_ground_truth(rows: UntouchedRows, seen, vocab_size: int,
                         max_mismatches: int = DEFAULT_MAX_GROUND_TRUTH_MISMATCHES,
                         ) -> GroundTruthComparison:
    """Compare an untouched-row set against a corpus "seen" mask.

    Args:
        rows: The run's fingerprint over window ``a -> b``.
        seen: Boolean mask indexed by token id: True iff the token occurs anywhere in the
            data consumed before step ``b`` (Adam history keeps moving rows seen earlier, so
            the truth is "never seen", not "not seen inside the window"). Length >= vocab.
        vocab_size: Real-token rows (the mask beyond is ignored).
        max_mismatches: Largest total one-sided count still verdicted MATCH_WITH_EXCEPTIONS.
    """
    seen = np.asarray(seen, dtype=bool)
    if len(seen) < vocab_size:
        raise ValueError(f"seen mask has {len(seen)} entries < vocab_size {vocab_size}")
    truth = set(np.flatnonzero(~seen[:vocab_size]).tolist())
    run = set(rows.untouched)
    return GroundTruthComparison(
        n_truth=len(truth), n_run=len(run), n_intersection=len(truth & run),
        n_union=len(truth | run),
        run_minus_truth=tuple(sorted(run - truth)),
        truth_minus_run=tuple(sorted(truth - run)),
        n_run_minus_truth=len(run - truth), n_truth_minus_run=len(truth - run),
        control_ok=rows.control_ok(), max_mismatches=max_mismatches,
    )


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """|a & b| / |a | b|; nan (never 0) when both are empty."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else float("nan")


@dataclass(frozen=True)
class InitComparison:
    """Value equality of two parameter states over their SHARED float, non-buffer keys."""

    n_tensors: int
    n_equal: int
    unequal: tuple[str, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]

    @property
    def shared(self) -> bool:
        """Every shared tensor value-equal (key mismatches are reported, not counted)."""
        return self.n_tensors > 0 and self.n_equal == self.n_tensors


def float_params(state: dict, buffer_suffixes: Sequence[str] = PYTHIA_BUFFER_SUFFIXES) -> dict:
    """Float tensors of a state dict, buffers (by name suffix) excluded."""
    return {k: v for k, v in state.items()
            if v.is_floating_point() and not k.endswith(tuple(buffer_suffixes))}


def value_hashes(state: dict, buffer_suffixes: Sequence[str] = PYTHIA_BUFFER_SUFFIXES) -> dict[str, str]:
    """Per-tensor sha1 of the fp64 VALUES (dtype-independent; ``-0.0`` hashes as ``+0.0``).

    Hashing lets many models' inits be compared without holding every state in memory.
    """
    import torch

    return {k: hashlib.sha1((v.detach().to(torch.float64).cpu() + 0.0).contiguous()
                            .numpy().tobytes()).hexdigest()
            for k, v in float_params(state, buffer_suffixes).items()}


def compare_value_hashes(a: dict[str, str], b: dict[str, str]) -> InitComparison:
    """`InitComparison` from two `value_hashes` outputs."""
    common = sorted(set(a) & set(b))
    unequal = tuple(k for k in common if a[k] != b[k])
    return InitComparison(
        n_tensors=len(common), n_equal=len(common) - len(unequal), unequal=unequal,
        only_a=tuple(sorted(set(a) - set(b))), only_b=tuple(sorted(set(b) - set(a))),
    )


def init_state_equality(a: dict, b: dict,
                        buffer_suffixes: Sequence[str] = PYTHIA_BUFFER_SUFFIXES) -> InitComparison:
    """Init identity of two state dicts: value equality over shared float non-buffer tensors."""
    return compare_value_hashes(value_hashes(a, buffer_suffixes), value_hashes(b, buffer_suffixes))


@dataclass(frozen=True)
class Classes:
    """Transitive equivalence classes plus the components that could not be grouped."""

    classes: tuple[tuple[str, ...], ...]
    non_transitive: tuple[dict, ...]

    def label_of(self) -> dict[str, int]:
        return {m: i for i, c in enumerate(self.classes) for m in c}


def transitive_classes(items: Sequence[str],
                       same: Callable[[str, str], bool | None]) -> Classes:
    """Group items only where EVERY pair in the group agrees.

    Connected components of the ``same(x, y) is True`` graph are classes iff they are
    cliques. A component with a disagreeing (False) or inconclusive (None) pair is not
    grouped greedily: it is reported in ``non_transitive`` with its missing pairs and its
    members get no class.
    """
    items = list(items)
    edge = {(x, y): same(x, y) is True for i, x in enumerate(items) for y in items[i + 1:]}
    parent = {x: x for x in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (x, y), ok in edge.items():
        if ok:
            parent[find(x)] = find(y)
    comps: dict[str, list[str]] = {}
    for x in items:
        comps.setdefault(find(x), []).append(x)
    classes, bad = [], []
    for members in comps.values():
        missing = [[x, y] for i, x in enumerate(members) for y in members[i + 1:]
                   if not edge[(x, y)]]
        if missing:
            bad.append({"component": members, "missing_pairs": missing})
        else:
            classes.append(tuple(members))
    return Classes(classes=tuple(classes), non_transitive=tuple(bad))


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x
