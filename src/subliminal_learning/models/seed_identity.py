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

from dataclasses import dataclass, field

import numpy as np

# fp32 rounding on a rescaled row is ~1e-7 relative; a single Adam step on a touched row is
# orders of magnitude larger at pretraining learning rates.
DEFAULT_TOL = 1e-6


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

    @property
    def n_untouched(self) -> int:
        return len(self.untouched)

    def control_ok(self, tol: float) -> bool:
        """The never-used padding rows must all pass the residual test (or there are none)."""
        return self.pad_rows == 0 or self.pad_resid_max <= tol

    def summary(self) -> dict:
        d = self.__dict__.copy()
        d.pop("untouched")
        d["n_untouched"] = self.n_untouched
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

    @property
    def jaccard(self) -> float:
        return self.n_intersection / self.n_union if self.n_union else float("nan")

    @property
    def shared(self) -> bool | None:
        """True = same order, False = different, None = inconclusive (both sets empty)."""
        if self.n_union == 0:
            return None
        return self.n_intersection == self.n_union

    @property
    def verdict(self) -> str:
        s = self.shared
        return "INCONCLUSIVE" if s is None else ("SHARED" if s else "DIFFERENT")


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


def untouched_rows(w0, wt, vocab_size: int, tol: float = DEFAULT_TOL) -> UntouchedRows:
    """Rows of an input embedding that received no Adam update between two checkpoints.

    Per row, the scale ``c`` is fitted by least squares and the relative residual
    ``||wt - c*w0|| / ||w0||`` computed in fp64. Untouched rows have residual at rounding
    level (and share the common weight-decay factor ``c``); touched rows do not.

    Args:
        w0: Embedding matrix at the earlier checkpoint (torch tensor or array,
            ``[embedding_size, d]``).
        wt: The same matrix at the later checkpoint.
        vocab_size: Rows ``< vocab_size`` are real tokens; the rest are the padding control.
        tol: Relative-residual threshold for "untouched".

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
    untouched = np.flatnonzero(vocab <= tol)
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
    )


def compare_untouched(a: UntouchedRows, b: UntouchedRows, n_examples: int = 50) -> OrderComparison:
    """Jaccard agreement of two runs' untouched sets over the same checkpoint window.

    Args:
        a: Fingerprint of the first run.
        b: Fingerprint of the second run.
        n_examples: How many one-sided token ids to keep per side.

    Returns:
        An `OrderComparison`; ``shared`` is True only for IDENTICAL sets, None when both
        are empty (window saturated -- every token seen -- so the test cannot speak).
    """
    ua, ub = set(a.untouched), set(b.untouched)
    return OrderComparison(
        n_a=len(ua), n_b=len(ub), n_intersection=len(ua & ub), n_union=len(ua | ub),
        only_a=tuple(sorted(ua - ub)[:n_examples]), only_b=tuple(sorted(ub - ua)[:n_examples]),
    )


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x
