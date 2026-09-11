"""Pile CE of the PUBLIC pretraining checkpoints of a base, and the "training retained"
readout it enables.

Pythia (and PolyPythias) ship the full checkpoint trajectory of every base, so a
student's Pile CE can be mapped onto that trajectory with no fitting: the earliest
pretraining step at which the base itself was already at least as good as the student
is the amount of pretraining the student "retained". This turns the dead/alive label
into a continuous, interpretable damage axis; the unigram anchor (H_uni, see
`probe_io.anchored_dead_cut`) becomes one point on it.

Records: ``{output_root}/analysis/base_ckpt_pile_ce/{base_dir_name}.json`` (written by
`scripts.base_ckpt_pile_ce`), one per base, keyed by pretraining step. Every step is
measured on the NESTED wide pool ``pp1w-pythia`` (128 docs), so each record carries both
the 32-doc ``ce_pp1`` (the students' readout pool, first 32 docs token-identical) and
the 128-doc ``ce_pp1w`` reading from one forward pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RECORD_KIND = "base_ckpt_pile_ce"
SCHEMA_VERSION = 1
OUT_SUBDIR = "analysis/base_ckpt_pile_ce"

# Pythia's published checkpoint schedule: log-spaced early steps, then every 1,000 up to
# the final step 143,000. The early steps are where the CE falls from uniform (~10.8
# nats) through the unigram level (H_uni 7.80) toward the contextual regime, so a
# damage axis that should resolve the anchor needs them.
EARLY_STEPS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
FINAL_STEP = 143_000


def default_steps(every: int = 5000, max_step: int = FINAL_STEP,
                  include_early: bool = True) -> list[int]:
    """The step list to measure: multiples of `every` up to `max_step`, plus the final
    published step when `max_step` reaches it, plus the log-spaced early steps."""
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")
    steps = set(range(0, max_step + 1, every))
    if max_step >= FINAL_STEP:
        steps.add(FINAL_STEP)
    if include_early:
        steps.update(s for s in EARLY_STEPS if s <= max_step)
    return sorted(steps)


def record_path(output_root: Path, base_dir: str) -> Path:
    return Path(output_root) / OUT_SUBDIR / f"{base_dir}.json"


def load_record(output_root: Path, base: str) -> dict | None:
    """The record of `base` (any spelling the naming module accepts), or None."""
    from subliminal_learning.models.naming import base_dir_name

    p = record_path(output_root, base_dir_name(base))
    return json.loads(p.read_text()) if p.is_file() else None


def curve_from_record(rec: dict, pool: str = "pp1") -> dict[int, float]:
    """``{step: ce}`` from a record, for `pool` in {"pp1", "pp1w"}, sorted by step."""
    key = {"pp1": "ce_pp1", "pp1w": "ce_pp1w"}[pool]
    return {int(s): float(v[key]) for s, v in sorted(
        rec["steps"].items(), key=lambda kv: int(kv[0]))}


def load_curve(output_root: Path, base: str, pool: str = "pp1") -> dict[int, float]:
    rec = load_record(output_root, base)
    if rec is None:
        raise FileNotFoundError(
            f"no {RECORD_KIND} record for {base!r} under {record_path(output_root, '*')}; "
            "run scripts.base_ckpt_pile_ce first")
    return curve_from_record(rec, pool)


def training_retained(ce: float, curve: dict[int, float]) -> float:
    """Pretraining step whose Pile CE first matches `ce` on the base's own trajectory.

    Scans the checkpoints in step order and returns the EARLIEST step at which the base's
    CE had already fallen to `ce` or below, interpolating linearly in log(1 + step)
    between the bracketing checkpoints (so a 5k grid still reads a continuous value).
    Scanning from the left makes the readout conservative under a non-monotone tail
    (32-doc noise late in training): the first crossing wins.

    Returns ``0.0`` when `ce` is at or above the step-0 CE (nothing retained), the LAST
    measured step when `ce` is below every checkpoint (the student is at least as good
    as the final base: everything retained), and NaN for a non-finite `ce`.
    """
    if not curve:
        raise ValueError("empty checkpoint curve")
    if not math.isfinite(ce):
        return float("nan")
    steps = sorted(curve)
    prev_s, prev_ce = None, None
    for s in steps:
        c = curve[s]
        if c <= ce:
            if prev_s is None or c == ce:
                return float(s)               # exact checkpoint hit: no interpolation
            # log(1+step) interpolation between (prev_s, prev_ce) and (s, c)
            x0, x1 = math.log1p(prev_s), math.log1p(s)
            t = (prev_ce - ce) / (prev_ce - c) if prev_ce != c else 1.0
            return float(math.expm1(x0 + t * (x1 - x0)))
        prev_s, prev_ce = s, c
    return float(steps[-1])


def training_retained_fraction(ce: float, curve: dict[int, float],
                               total_steps: int = FINAL_STEP) -> float:
    """`training_retained` as a fraction of the full pretraining run."""
    return training_retained(ce, curve) / float(total_steps)
