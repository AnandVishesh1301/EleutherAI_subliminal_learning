"""crossval2 results math: panel rows, control sets, shifts, contrasts and claims.

Pure and torch-free. Quantities, shifts and ΔCE: ``docs/PREREG_CROSSVAL2.md`` §4;
quarantine §3; control sets §5 (`CONTROL_SOURCES`); decision rules §6.

Intervals come from a three-level bootstrap (teacher → gseed → prompt). Each replicate
first redraws the prompts of every bank once, shared by every model (``favorite`` /
``dislike`` share indices, as do ``consensus`` / ``consensus_dislike``), recomputes each
model's metric and the reference means (controls, untuned base) under that draw, then
resamples teachers and gseeds with the draws of `sdfpos2_metrics` (ported below;
gseed-paired contrasts, seed 20261021). ``levels=2`` turns the prompt level off and reproduces the
two-stage intervals. Point estimates are full-bank values. A quarantined (§3) or
degenerate (V-D) model keeps its row and is left out of every mean, contrast and claim.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from types import MappingProxyType

import numpy as np
import pandas as pd

from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.campaign.ws160m_metrics import mark_quarantine
from subliminal_learning.campaign.ws160m_plan import QUARANTINE_FILTER_DROP_MAX
from subliminal_learning.campaign.xopt_rules import cut
from subliminal_learning.evaluation.valence import (
    PANEL_ANIMALS,
    PANEL_SETS,
    VALENCE_PAIRS,
    LogProbs,
    directional_index,
    per_prompt_direction,
    salience,
    valence_summary,
)

N_BOOT = 10000
BOOT_SEED = 20261021
BOOT_LEVELS = 3
PROMPT_STREAM = 1
PLACEBO_BAND = 0.05
PANEL_SCHEMA = "valence_panel_v1"
PROBES_FILE = "secondary_probes.jsonl"

TEACHER, STUDENT, CONTROL, BASE = "teacher", "student", "control", "base"
LEAF_KINDS: tuple[str, ...] = (STUDENT, CONTROL)
HEADLINE_GROUP = "p0"
POOLED_GROUP = "p0+p1-seeds"
TIER_GROUPS: tuple[str, ...] = (
    "p0",
    "p1-anchor",
    "p1-seeds",
    "p2-size",
    "p2-regime",
    "p2-xinit",
    "p3-fulladamw",
    "p3-channel",
    "p4-fullrms",
    "p4-init",
    "p5-e4",
    "p5-fullsgd",
    "p6-gseeds",
    "p6-gseeds-x",
)
HEADLINE_BASES: tuple[str, ...] = (plan.BASE_160M, plan.BASE_SEED2)
# Tier groups whose control leaves (same student base, gkd_tag, teacher base) a cell's
# shifts use (prereg §5 Controls column).
CONTROL_SOURCES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "p0": ("p0",),
        "p1-anchor": ("p0",),
        "p1-seeds": ("p1-seeds",),
        "p2-size": ("p0",),
        "p2-regime": ("p0",),
        "p2-xinit": ("p2-xinit",),
        "p3-fulladamw": ("p3-fulladamw",),
        "p3-channel": ("p3-channel",),
        "p4-fullrms": ("p4-fullrms",),
        "p4-init": ("p4-init",),
        "p5-e4": ("p5-e4",),
        "p5-fullsgd": ("p5-fullsgd",),
        "p6-gseeds": ("p0",),
        "p6-gseeds-x": ("p1-seeds",),
        POOLED_GROUP: ("p0", "p1-seeds"),
    }
)

# (prompt bank, metric tag, like set, dislike set); a bank's two sets share indices.
_VOICES: tuple[tuple[str, str, str, str], ...] = (
    ("first", "1", *VALENCE_PAIRS["first"]),
    ("third", "3", *VALENCE_PAIRS["third"]),
)
_BANK_INDEX: Mapping[str, int] = MappingProxyType(
    {bank: i for i, (bank, *_) in enumerate(_VOICES)}
)
SIGMAS: tuple[str, ...] = tuple(
    f"sigma{tag}_{a}" for _, tag, _, _ in _VOICES for a in PANEL_ANIMALS
)
COMPONENTS: tuple[str, ...] = tuple(
    f"mean_logp_{s}_{a}"
    for _, _, like, dislike in _VOICES
    for s in (like, dislike)
    for a in PANEL_ANIMALS
)
METRICS: tuple[str, ...] = ("D1", "D3", *SIGMAS, *COMPONENTS)
PROMPT_BANKS: Mapping[str, str] = MappingProxyType(
    {
        m: bank
        for bank, tag, like, dislike in _VOICES
        for m in (
            f"D{tag}",
            *(f"sigma{tag}_{a}" for a in PANEL_ANIMALS),
            *(f"mean_logp_{s}_{a}" for s in (like, dislike) for a in PANEL_ANIMALS),
        )
    }
)
SUMMARY_KEYS: tuple[str, ...] = (
    "D1",
    "D3",
    *SIGMAS,
    *(f"mean_logp_{s}_{a}" for s in PANEL_SETS for a in PANEL_ANIMALS),
)

ID_COLUMNS: tuple[str, ...] = (
    "run_set",
    "level",
    "tier_group",
    "base",
    "teacher_base",
    "kind",
    "arm",
    "corpus_tag",
    "family",
    "regime",
    "variant",
    "gkd_tag",
    "ftseed",
    "gseed",
    "cross_init",
)
PANEL_COLUMNS: tuple[str, ...] = (
    *ID_COLUMNS,
    "present",
    *SUMMARY_KEYS,
    "pp1_ce",
    "pp1_dce",
    "degenerate",
    "perplexity_ratio",
    "filter_drop_frac",
    "quarantined",
    "quarantine_reason",
    "excluded",
    "panel_path",
)
PROBE_COLUMNS: tuple[str, ...] = (*ID_COLUMNS, "excluded", "step", "model", *METRICS)
_CELL_COLUMNS: tuple[str, ...] = (
    "tier_group",
    "level",
    "base",
    "teacher_base",
    "cross_init",
    "run_sets",
)
ARM_COLUMNS: tuple[str, ...] = (
    *_CELL_COLUMNS,
    "arm",
    "corpus_tag",
    "family",
    "regime",
    "variant",
    "gkd_tag",
    "metric",
    "teacher_mean",
    "teacher_lo",
    "teacher_hi",
    "n_teachers",
    "base_value",
    "teacher_shift",
    "teacher_shift_lo",
    "teacher_shift_hi",
    "student_mean",
    "student_lo",
    "student_hi",
    "n_student_teachers",
    "n_leaves",
    "control_mean",
    "n_controls",
    "student_shift",
    "student_shift_lo",
    "student_shift_hi",
    "n_leaves_expected",
    "n_leaves_present",
    "n_quarantined",
    "n_degenerate",
    "filter_drop_mean",
    "filter_drop_min",
    "filter_drop_max",
    "control_source",
    "control_run_sets",
    "control_gseeds",
    "control_filter_drop_mean",
    "control_filter_drop_min",
    "control_filter_drop_max",
)
CONTRAST_COLUMNS: tuple[str, ...] = (
    *_CELL_COLUMNS,
    "variant",
    "gkd_tag",
    "contrast",
    "analysis",
    "a",
    "b",
    "model",
    "metric",
    "diff",
    "lo",
    "hi",
    "n_teachers_a",
    "n_teachers_b",
    "n_a",
    "n_b",
    "label",
    "control_run_sets",
)

LR, POS_NEG, PLACEBO_CONTROL = "L-R", "pos-neg", "placebo-control"
PRIMARY, SENSITIVITY = "primary", "sensitivity: quarantined included"
UNRUN, INSUFFICIENT = "unrun", "insufficient"
POSITIVE, NEGATIVE, INCLUDES_ZERO = "positive", "negative", "includes zero"
SUPPORTED, PARTIAL, NOT_SUPPORTED = "supported", "partial", "not supported"
UNINFORMATIVE, REPORTED, MIXED = "uninformative", "reported", "mixed"
CONSISTENT, INCONSISTENT = "consistent", "inconsistent"
ROBUSTNESS: Mapping[str, str] = MappingProxyType(
    {POSITIVE: "robust", INCLUDES_ZERO: "not resolved", NEGATIVE: "reversed"}
)
_NAN = float("nan")

Key = tuple[str, str, str | None, str | None, int | None, int | None]


# ---- plan views ------------------------------------------------------------------


def tier_group(run_set_name: str) -> str:
    """Tier group of a run-set token (the token without its base suffix).

    Raises:
        ValueError: The token maps to no tier group.
    """
    group = run_set_name.rsplit("-", 1)[0]
    if group not in TIER_GROUPS:
        raise ValueError(f"{run_set_name!r} maps to no tier group ({group!r})")
    return group


@dataclass(frozen=True)
class GroupCell:
    """One (tier group, student base) and the run sets it reads.

    Attributes:
        tier_group: Tier group or `POOLED_GROUP`.
        base: Student base token.
        experiments: Run sets (two only for `POOLED_GROUP`).
    """

    tier_group: str
    base: str
    experiments: tuple[plan.Experiment, ...]

    @property
    def head(self) -> plan.Experiment:
        return self.experiments[0]

    @property
    def teacher_base(self) -> str:
        return self.head.teacher_base or self.base

    @property
    def level(self) -> int | None:
        return None if self.tier_group == POOLED_GROUP else self.head.level

    @property
    def run_sets(self) -> str:
        return "+".join(e.run_set_id for e in self.experiments)

    @property
    def teacher_run_sets(self) -> str:
        return "+".join(e.teacher_run_set_id for e in self.experiments)

    def columns(self) -> dict:
        return {
            "tier_group": self.tier_group,
            "level": self.level,
            "base": self.base,
            "teacher_base": self.teacher_base,
            "cross_init": self.head.cross_init,
            "run_sets": self.run_sets,
        }


def group_cells() -> tuple[GroupCell, ...]:
    """Every (tier group, base) of the plan, then the pooled P0 + P1-seeds extra cells.

    The pooled cells feed the headline power analysis only (prereg headline = P0).

    Raises:
        ValueError: Two run sets share a (tier group, base), or P0 and P1-seeds differ
            in arms, variants or gseeds.
    """
    cells: dict[tuple[str, str], GroupCell] = {}
    for e in plan.EXPERIMENTS.values():
        key = (tier_group(e.name), e.base)
        if key in cells:
            raise ValueError(f"two run sets for {key}")
        cells[key] = GroupCell(key[0], key[1], (e,))
    for base in HEADLINE_BASES:
        p0 = cells[(HEADLINE_GROUP, base)].head
        p1 = cells[("p1-seeds", base)].head
        if (p0.arms, p0.variants, p0.gseeds) != (p1.arms, p1.variants, p1.gseeds):
            raise ValueError(f"{p0.name} and {p1.name} cannot pool")
        cells[(POOLED_GROUP, base)] = GroupCell(POOLED_GROUP, base, (p0, p1))
    return tuple(cells.values())


def group_cell(group: str, base: str) -> GroupCell:
    """The cell of one (tier group, base).

    Raises:
        ValueError: No such cell.
    """
    for cell in group_cells():
        if (cell.tier_group, cell.base) == (group, base):
            return cell
    raise ValueError(f"no cell ({group!r}, {base!r})")


def lr_pairs(arms: Sequence[plan.Arm]) -> list[tuple[plan.Arm, plan.Arm]]:
    """(L, R) pairs: ``cross-L-…`` vs ``cross-R-…`` at the same size and regime.

    Raises:
        ValueError: An L arm has no R twin.
    """
    by = {(a.corpus_tag, a.regime): a for a in arms}
    out = []
    for a in arms:
        if a.family == plan.CROSSVAL and a.corpus_tag.startswith("cross-L-"):
            twin = (a.corpus_tag.replace("cross-L-", "cross-R-", 1), a.regime)
            if twin not in by:
                raise ValueError(f"{a.slug} has no R twin")
            out.append((a, by[twin]))
    return out


def anchor_pair(arms: Sequence[plan.Arm]) -> tuple[plan.Arm, plan.Arm] | None:
    """(val-positive, val-negative) anchor arms, or None.

    Raises:
        ValueError: `plan.ANCHOR_TAGS` no longer starts with positive, negative.
    """
    pos, neg = plan.ANCHOR_TAGS[0], plan.ANCHOR_TAGS[1]
    if "positive" not in pos or "negative" not in neg:
        raise ValueError(f"unexpected anchor order {plan.ANCHOR_TAGS}")
    by = {a.corpus_tag: a for a in arms}
    return (by[pos], by[neg]) if pos in by and neg in by else None


def placebo_arm(arms: Sequence[plan.Arm]) -> plan.Arm | None:
    return next((a for a in arms if a.corpus_tag == plan.PLACEBO_TAG), None)


# ---- row builders ------------------------------------------------------------------


def panel_values(
    doc: Mapping,
    *,
    subject: str,
    base_model: str,
    run_dir_tail: Sequence[str] | None,
) -> dict[str, float]:
    """`valence_summary` keys and ``pp1_ce`` of one parsed ``valence_panel.json``.

    Args:
        doc: Parsed panel.
        subject: Expected ``subject``.
        base_model: Expected ``<repo>@<revision>``.
        run_dir_tail: Expected trailing parts of ``run_dir``; None for a base panel.

    Raises:
        ValueError: Schema, subject, base model or run dir differ from the plan, or a
            set is missing or unpaired.
    """
    want = {"schema": PANEL_SCHEMA, "subject": subject, "base_model": base_model}
    got = {k: doc.get(k) for k in want}
    if got != want:
        raise ValueError(f"panel identity {got} != plan {want}")
    run_dir = doc.get("run_dir")
    if run_dir_tail is None:
        ok = run_dir is None
    else:
        tail = tuple(run_dir_tail)
        ok = run_dir is not None and PurePath(run_dir).parts[-len(tail) :] == tail
    if not ok:
        raise ValueError(f"panel run_dir {run_dir!r} does not end in {run_dir_tail}")
    out = valence_summary({name: s["logp"] for name, s in doc["sets"].items()})
    out["pp1_ce"] = float(doc["pp1"]["ce"])
    return out


def filter_drop_frac(summary: Mapping | None) -> float:
    """Share of screened rows dropped from the loss, on- and off-policy pooled.

    ``flagged_rate`` counts on-policy rows only, so it reads 0 for a λ = 0 cell.
    """
    if summary is None:
        return _NAN
    if "n_rows_screened" not in summary:
        value = summary.get("flagged_rate")
        return _NAN if value is None else float(value)
    rows = summary["n_rows_screened"] + summary.get("n_off_policy_rows_screened", 0)
    flagged = summary["n_flagged"] + summary.get("n_off_policy_flagged", 0)
    return flagged / rows if rows else 0.0


def probe_summary(sets: Mapping[str, LogProbs]) -> dict[str, float]:
    """`METRICS` of one secondary-probe row; NaN where a set of a pair is absent."""
    out = dict.fromkeys(METRICS, _NAN)
    for _, tag, like, dislike in _VOICES:
        if like in sets and dislike in sets:
            out[f"D{tag}"] = directional_index(sets[like], sets[dislike])
            for a in PANEL_ANIMALS:
                out[f"sigma{tag}_{a}"] = salience(sets[like], sets[dislike], a)
    for name, logp in sets.items():
        for a in PANEL_ANIMALS:
            key = f"mean_logp_{name}_{a}"
            if key in out and a in logp:
                out[key] = statistics.fmean(logp[a])
    return out


def probe_rows(identity: Mapping, records: Iterable[Mapping]) -> list[dict]:
    """``probes.csv`` rows of one leaf, one per (step, model).

    A resumed run re-logs the steps after its checkpoint; the last row per
    (step, model) wins.
    """
    last: dict[tuple[int, str], Mapping] = {}
    for r in records:
        last[(int(r["step"]), str(r["model"]))] = r
    return [
        {**identity, "step": step, "model": model, **probe_summary(r["sets"])}
        for (step, model), r in sorted(last.items())
    ]


def prompt_vectors(records: Iterable[Mapping]) -> dict[Key, dict[str, np.ndarray]]:
    """Per-prompt values of every `METRICS` entry per model whose record has ``logp``.

    ``logp`` is the panel's set -> animal -> per-prompt log p; each vector's mean is the
    model's metric.
    """
    out: dict[Key, dict[str, np.ndarray]] = {}
    for r in records:
        sets = r.get("logp")
        if sets is None:
            continue
        vec: dict[str, np.ndarray] = {}
        for _, tag, like, dislike in _VOICES:
            vec[f"D{tag}"] = np.asarray(
                per_prompt_direction(sets[like], sets[dislike]), dtype=float
            )
            for a in PANEL_ANIMALS:
                lk = np.asarray(sets[like][a], dtype=float)
                dk = np.asarray(sets[dislike][a], dtype=float)
                vec[f"sigma{tag}_{a}"] = 0.5 * (lk + dk)
                vec[f"mean_logp_{like}_{a}"] = lk
                vec[f"mean_logp_{dislike}_{a}"] = dk
        out[_row_key(r)] = vec
    return out


def finalize_panels(records: Sequence[Mapping]) -> pd.DataFrame:
    """``panels.csv``: ΔCE vs the untuned base, degenerate, quarantine, exclusion.

    Args:
        records: One per expected model: `ID_COLUMNS`, ``present``, ``panel_path``,
            ``perplexity_ratio``, ``filter_drop_frac`` and, when present,
            `SUMMARY_KEYS` + ``pp1_ce``.

    Returns:
        `PANEL_COLUMNS`. ``degenerate`` is NA without a base panel; quarantine
        applies to student and control leaves only.

    Raises:
        ValueError: Two present base panels for one base.
    """
    df = pd.DataFrame(list(records))
    for col in (*SUMMARY_KEYS, "pp1_ce", "perplexity_ratio", "filter_drop_frac"):
        df[col] = pd.to_numeric(df.get(col, pd.Series(_NAN, index=df.index)))
    bases = df[(df["kind"] == BASE) & df["present"].astype(bool)]
    if bases["base"].duplicated().any():
        raise ValueError(f"duplicate base panels: {bases['base'].tolist()}")
    ce_base = dict(zip(bases["base"], bases["pp1_ce"], strict=True))
    cuts = {b: cut(c) for b, c in ce_base.items() if math.isfinite(c)}
    df["pp1_dce"] = df["pp1_ce"] - df["base"].map(ce_base).astype(float)
    limit = df["base"].map(cuts).astype(float)
    df["degenerate"] = pd.array(
        [
            pd.NA if math.isnan(d) or math.isnan(c) else bool(d > c)
            for d, c in zip(df["pp1_dce"], limit, strict=True)
        ],
        dtype="boolean",
    )
    marked = mark_quarantine(df)
    leaf = df["kind"].isin(LEAF_KINDS)
    unscreened = (
        leaf
        & df["present"].astype(bool)
        & (df["perplexity_ratio"].isna() | df["filter_drop_frac"].isna())
        & ~marked["quarantined"].astype(bool)
    )
    df["quarantined"] = (marked["quarantined"].astype(bool) & leaf) | unscreened
    df["quarantine_reason"] = (
        marked["quarantine_reason"].where(leaf, "").mask(unscreened, "unscreened")
    )
    df["excluded"] = df["quarantined"] | df["degenerate"].fillna(False).astype(bool)
    for col in ("level", "ftseed", "gseed"):
        df[col] = df[col].astype("Int64")
    return df[list(PANEL_COLUMNS)]


def probe_frame(rows: Sequence[Mapping], panels: pd.DataFrame) -> pd.DataFrame:
    """``probes.csv``: probe rows with their leaf's ``excluded`` flag from `panels`."""
    excluded = {
        _row_key(r): bool(r["excluded"])
        for r in panels[
            ["run_set", "kind", "arm", "variant", "ftseed", "gseed", "excluded"]
        ].to_dict("records")
    }
    out = pd.DataFrame(
        [{**r, "excluded": excluded[_row_key(r)]} for r in rows],
        columns=list(PROBE_COLUMNS),
    )
    for col in ("level", "ftseed", "gseed", "step"):
        out[col] = out[col].astype("Int64")
    return out


# ---- keys, control sets ------------------------------------------------------------


def _norm(x: object) -> object:
    if x is None or isinstance(x, str):
        return x
    return None if pd.isna(x) else x


def model_key(
    run_set: str,
    kind: str,
    arm: str | None,
    variant: str | None,
    ftseed: int | None,
    gseed: int | None,
) -> Key:
    """Hashable identity of one model (NaN / NA normalised to None)."""
    ft, g = _norm(ftseed), _norm(gseed)
    return (
        run_set,
        kind,
        _norm(arm),  # type: ignore[return-value]
        _norm(variant),  # type: ignore[return-value]
        None if ft is None else int(ft),  # type: ignore[call-overload]
        None if g is None else int(g),  # type: ignore[call-overload]
    )


def _row_key(row: Mapping) -> Key:
    return model_key(
        row["run_set"],
        row["kind"],
        row["arm"],
        row["variant"],
        row["ftseed"],
        row["gseed"],
    )


def _index(panels: pd.DataFrame, *, quarantined_ok: bool = False) -> dict[Key, dict]:
    """Present models that may enter a mean (degenerate never; quarantined if asked)."""
    keep = panels["present"].astype(bool)
    if quarantined_ok:
        keep &= ~panels["degenerate"].fillna(False).astype(bool)
    else:
        keep &= ~panels["excluded"].astype(bool)
    return {_row_key(r): r for r in panels[keep].to_dict("records")}


def _finite_mean(values: Iterable[float]) -> float:
    ok = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(ok) if ok else _NAN


def _drop_stats(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Mean / min / max ``filter_drop_frac`` over `frame` (quarantined included)."""
    v = pd.to_numeric(frame["filter_drop_frac"], errors="coerce").dropna()
    return {
        f"{prefix}_mean": float(v.mean()) if len(v) else _NAN,
        f"{prefix}_min": float(v.min()) if len(v) else _NAN,
        f"{prefix}_max": float(v.max()) if len(v) else _NAN,
    }


def _control_frames(
    panels: pd.DataFrame,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """(tier group, base, variant) -> every expected control leaf of its control set."""
    controls = panels[panels["kind"] == CONTROL]
    out = {}
    for cell in group_cells():
        src = CONTROL_SOURCES[cell.tier_group]
        for v in cell.head.variants:
            out[(cell.tier_group, cell.base, v.name)] = controls[
                controls["tier_group"].isin(src)
                & (controls["base"] == cell.base)
                & (controls["gkd_tag"] == v.gkd_tag)
                & (controls["teacher_base"] == cell.teacher_base)
            ]
    return out


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["present"].astype(bool) & ~frame["excluded"].astype(bool)]


def control_sets(panels: pd.DataFrame) -> list[dict]:
    """The control set of every (cell, variant): source tier groups, run sets, used
    gseeds and screen drop rates (claims.json ``control_sets``)."""
    out = []
    for (group, base, variant), frame in _control_frames(panels).items():
        used = _usable(frame)
        out.append(
            {
                "tier_group": group,
                "base": base,
                "variant": variant,
                "source": list(CONTROL_SOURCES[group]),
                "run_sets": sorted(frame["run_set"].unique().tolist()),
                "gseeds_used": sorted(int(g) for g in used["gseed"]),
                "n_expected": len(frame),
                "n_used": len(used),
                "n_quarantined": int(frame["quarantined"].sum()),
                **_drop_stats(frame, "filter_drop"),
            }
        )
    return out


# ---- bootstrap ---------------------------------------------------------------------


def prompt_counts(bank: str, n_prompts: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n_prompts]`` draw counts of one bank (prompts with replacement)."""
    rng = np.random.default_rng([seed, PROMPT_STREAM, _BANK_INDEX[bank]])
    idx = rng.integers(0, n_prompts, (n_boot, n_prompts))
    flat = (idx + n_prompts * np.arange(n_boot)[:, None]).ravel()
    counts = np.bincount(flat, minlength=n_boot * n_prompts)
    return counts.reshape(n_boot, n_prompts).astype(float)


def replicates(
    points: Mapping[Key, float],
    prompts: Mapping[Key, Mapping[str, np.ndarray]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
    levels: int,
) -> dict[Key, np.ndarray]:
    """``[n_boot]`` values of `metric` per model under the shared prompt draws.

    Args:
        points: Full-bank value per model.
        prompts: `prompt_vectors` (unused at ``levels=2``).
        metric: One of `METRICS`.
        n_boot: Replicates.
        seed: Bootstrap seed.
        levels: 3 resamples prompts; 2 repeats the full-bank value.

    Raises:
        ValueError: Bad `levels`, a model without prompt vectors, or unequal prompt
            counts within a bank.
    """
    if levels not in (2, 3):
        raise ValueError(f"levels must be 2 or 3, got {levels}")
    keys = list(points)
    if levels == 2 or not keys:
        return {k: np.broadcast_to(float(points[k]), (n_boot,)) for k in keys}
    if missing := [k for k in keys if k not in prompts]:
        raise ValueError(
            f"{len(missing)} models lack prompt vectors, e.g. {missing[0]}"
        )
    vectors = np.stack([prompts[k][metric] for k in keys])
    n_prompts = vectors.shape[1]
    counts = prompt_counts(PROMPT_BANKS[metric], n_prompts, n_boot, seed)
    return dict(zip(keys, vectors @ counts.T / n_prompts, strict=True))


def _two_stage_means(
    reps: np.ndarray, teachers: np.ndarray, gseeds: np.ndarray
) -> np.ndarray:
    """Port of sdfpos2_metrics.two_stage_means (uncommitted, 2026-09-26).

    One leaf matrix per replicate; with every replicate's matrix equal it is the
    original.

    Args:
        reps: ``[B, T, G]`` leaf values per replicate (NaN skipped).
        teachers: ``[B, T]`` teacher row picks.
        gseeds: ``[B, >= T, G]`` gseed column picks.
    """
    n_boot, n_teachers, _ = reps.shape
    vals = reps[
        np.arange(n_boot)[:, None, None],
        teachers[:, :, None],
        gseeds[:, :n_teachers, :],
    ].reshape(n_boot, -1)
    finite = np.isfinite(vals)
    count = finite.sum(axis=1)
    total = np.where(finite, vals, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def _arm_draws(reps: np.ndarray, n_boot: int, seed: int) -> np.ndarray | None:
    """Port of sdfpos2_metrics.arm_interval (uncommitted, 2026-09-26): its draws."""
    _, n_teachers, n_gseeds = reps.shape
    if n_teachers < 2:
        return None
    rng = np.random.default_rng(seed)
    gseeds = rng.integers(0, n_gseeds, (n_boot, n_teachers, n_gseeds))
    teachers = rng.integers(0, n_teachers, (n_boot, n_teachers))
    return _two_stage_means(reps, teachers, gseeds)


def _pair_draws(
    ra: np.ndarray, rb: np.ndarray, n_boot: int, seed: int
) -> np.ndarray | None:
    """Port of sdfpos2_metrics.contrast_interval (uncommitted, 2026-09-26): its draws.

    Replicate ``mean(a) − mean(b)``, gseed-paired.

    Raises:
        ValueError: The arms have different gseed columns.
    """
    if ra.shape[2] != rb.shape[2]:
        raise ValueError(f"gseed columns differ: {ra.shape} vs {rb.shape}")
    ta, tb = ra.shape[1], rb.shape[1]
    if min(ta, tb) < 2:
        return None
    rng = np.random.default_rng(seed)
    gseeds = rng.integers(0, ra.shape[2], (n_boot, max(ta, tb), ra.shape[2]))
    pick_a = rng.integers(0, ta, (n_boot, ta))
    pick_b = rng.integers(0, tb, (n_boot, tb))
    return _two_stage_means(ra, pick_a, gseeds) - _two_stage_means(rb, pick_b, gseeds)


def _percentiles(draws: np.ndarray | None) -> tuple[float, float]:
    """Port of sdfpos2_metrics._percentiles (uncommitted, 2026-09-26)."""
    if draws is None:
        return (_NAN, _NAN)
    ok = draws[np.isfinite(draws)]
    if ok.size == 0:
        return (_NAN, _NAN)
    return (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))


def _observed(point: np.ndarray) -> float:
    """Port of sdfpos2_metrics._observed (uncommitted, 2026-09-26)."""
    ok = point[np.isfinite(point)]
    return float(ok.mean()) if ok.size else _NAN


def _arm_interval(
    matrix: np.ndarray, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """Port of sdfpos2_metrics.arm_interval (uncommitted, 2026-09-26)."""
    m = np.asarray(matrix, dtype=float)
    draws = _arm_draws(np.broadcast_to(m, (n_boot, *m.shape)), n_boot, seed)
    return (_observed(m), *_percentiles(draws))


def _contrast_interval(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """Port of sdfpos2_metrics.contrast_interval (uncommitted, 2026-09-26)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    draws = _pair_draws(
        np.broadcast_to(a, (n_boot, *a.shape)),
        np.broadcast_to(b, (n_boot, *b.shape)),
        n_boot,
        seed,
    )
    return (_observed(a) - _observed(b), *_percentiles(draws))


@dataclass(frozen=True)
class _Block:
    """One arm: full-bank ``[T, G]`` and replicate ``[B, T, G]`` leaf values."""

    point: np.ndarray
    reps: np.ndarray

    @property
    def n(self) -> int:
        return int(np.isfinite(self.point).sum())

    @property
    def n_teachers(self) -> int:
        return self.point.shape[0]


def _block(
    idx: Mapping[Key, Mapping],
    rep: Mapping[Key, np.ndarray],
    grid: list[list[Key]],
    metric: str,
    n_boot: int,
) -> _Block:
    """Leaves of `grid` in `idx` (others NaN); teachers without one are dropped."""
    point = np.array(
        [[float(idx[k][metric]) if k in idx else _NAN for k in row] for row in grid],
        dtype=float,
    )
    keep = np.isfinite(point).any(axis=1)
    rows = [row for row, ok in zip(grid, keep, strict=True) if ok]
    blank = np.full(n_boot, _NAN)
    if not rows:
        return _Block(point[keep], np.empty((n_boot, 0, point.shape[1])))
    reps = np.stack(
        [
            np.stack([rep[k] if k in idx else blank for k in row], axis=-1)
            for row in rows
        ],
        axis=1,
    )
    return _Block(point[keep], reps)


def _reference(
    keys: Sequence[Key],
    idx: Mapping[Key, Mapping],
    rep: Mapping[Key, np.ndarray],
    metric: str,
    n_boot: int,
) -> tuple[float, np.ndarray]:
    """Fixed reference mean of `keys` (full bank, per replicate); NaN without one."""
    keys = [k for k in keys if k in idx]
    if not keys:
        return (_NAN, np.full(n_boot, _NAN))
    point = _finite_mean(idx[k][metric] for k in keys)
    stack = np.stack([rep[k] for k in keys])
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    per_rep = np.divide(total, count, out=np.full(n_boot, np.nan), where=count > 0)
    return (point, per_rep)


def _teacher_keys(cell: GroupCell, arm: plan.Arm) -> list[list[Key]]:
    return [
        [model_key(e.teacher_run_set_id, TEACHER, arm.slug, None, t, None)]
        for e in cell.experiments
        for t in e.ftseeds
    ]


def _student_keys(cell: GroupCell, arm: plan.Arm, v: plan.Variant) -> list[list[Key]]:
    return [
        [model_key(e.run_set_id, STUDENT, arm.slug, v.name, t, g) for g in e.gseeds]
        for e in cell.experiments
        for t in e.ftseeds
    ]


def _base_keys(panels: pd.DataFrame) -> dict[str, Key]:
    bases = panels[(panels["kind"] == BASE) & panels["present"].astype(bool)]
    return {r["base"]: _row_key(r) for r in bases.to_dict("records")}


def _points(present: Mapping[Key, Mapping], metric: str) -> dict[Key, float]:
    return {k: float(r[metric]) for k, r in present.items()}


def ci_label(diff: float, lo: float, hi: float, n: int) -> str:
    """Sign of a 95 % CI; `UNRUN` without inputs, `INSUFFICIENT` without a CI."""
    if n == 0:
        return UNRUN
    if not all(math.isfinite(x) for x in (diff, lo, hi)):
        return INSUFFICIENT
    if lo > 0:
        return POSITIVE
    if hi < 0:
        return NEGATIVE
    return INCLUDES_ZERO


def _int_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.columns:
        if col == "level" or col.startswith("n_"):
            frame[col] = frame[col].astype("Int64")
    return frame


def _sorted(rows: list[dict], columns: Sequence[str]) -> pd.DataFrame:
    rows.sort(key=lambda r: r["_order"])
    return _int_columns(pd.DataFrame(rows, columns=list(columns)))


# ---- tables ------------------------------------------------------------------------


def arm_summary(
    panels: pd.DataFrame,
    prompts: Mapping[Key, Mapping[str, np.ndarray]],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    levels: int = BOOT_LEVELS,
) -> pd.DataFrame:
    """``arm_summary.csv``: one row per (cell, arm, variant, metric).

    Teacher shift = teacher − untuned teacher base; student shift = student − mean of
    the cell's control set (`CONTROL_SOURCES`). Both references are recomputed under
    each replicate's prompt draw and are not resampled otherwise.
    """
    idx = _index(panels)
    present = _index(panels, quarantined_ok=True)
    base_keys = _base_keys(panels)
    ctrl_frames = _control_frames(panels)
    leaves = {
        k: g
        for k, g in panels[panels["kind"] == STUDENT].groupby(
            ["run_set", "arm", "variant"], sort=False
        )
    }
    layout = []
    for cell in group_cells():
        for v in cell.head.variants:
            ctrl = ctrl_frames[(cell.tier_group, cell.base, v.name)]
            used = _usable(ctrl)
            ctrl_cols = {
                "control_source": "+".join(CONTROL_SOURCES[cell.tier_group]),
                "control_run_sets": "+".join(sorted(ctrl["run_set"].unique())),
                "control_gseeds": ",".join(str(int(g)) for g in sorted(used["gseed"])),
                "n_controls": len(used),
                **_drop_stats(ctrl, "control_filter_drop"),
            }
            ctrl_keys = [_row_key(r) for r in used.to_dict("records")]
            for arm in cell.head.arms:
                frames = [
                    leaves[k]
                    for e in cell.experiments
                    if (k := (e.run_set_id, arm.slug, v.name)) in leaves
                ]
                expected = pd.concat(frames) if frames else panels.iloc[0:0]
                static = {
                    **cell.columns(),
                    "arm": arm.slug,
                    "corpus_tag": arm.corpus_tag,
                    "family": arm.family,
                    "regime": arm.regime,
                    "variant": v.name,
                    "gkd_tag": v.gkd_tag,
                    "n_leaves_expected": len(expected),
                    "n_leaves_present": int(expected["present"].astype(bool).sum()),
                    "n_quarantined": int(expected["quarantined"].astype(bool).sum()),
                    "n_degenerate": int(expected["degenerate"].fillna(False).sum()),
                    **_drop_stats(expected, "filter_drop"),
                    **ctrl_cols,
                }
                layout.append((cell, v, arm, ctrl_keys, static))
    rows = []
    for mi, metric in enumerate(METRICS):
        rep = replicates(
            _points(present, metric),
            prompts,
            metric,
            n_boot=n_boot,
            seed=seed,
            levels=levels,
        )
        teacher_stats: dict[tuple, tuple] = {}
        for li, (cell, v, arm, ctrl_keys, static) in enumerate(layout):
            base = base_keys.get(cell.teacher_base)
            b_point, b_rep = _reference(
                [base] if base else [], idx, rep, metric, n_boot
            )
            tkey = (cell.teacher_run_sets, tuple(e.ftseeds for e in cell.experiments))
            tkey += (arm.slug,)
            if tkey not in teacher_stats:
                tb = _block(idx, rep, _teacher_keys(cell, arm), metric, n_boot)
                teacher_stats[tkey] = (tb, _arm_draws(tb.reps, n_boot, seed))
            tb, t_draws = teacher_stats[tkey]
            sb = _block(idx, rep, _student_keys(cell, arm, v), metric, n_boot)
            s_draws = _arm_draws(sb.reps, n_boot, seed)
            c_point, c_rep = _reference(ctrl_keys, idx, rep, metric, n_boot)
            t_mean, s_mean = _observed(tb.point), _observed(sb.point)
            t_lo, t_hi = _percentiles(t_draws)
            s_lo, s_hi = _percentiles(s_draws)
            ts_lo, ts_hi = _percentiles(None if t_draws is None else t_draws - b_rep)
            ss_lo, ss_hi = _percentiles(None if s_draws is None else s_draws - c_rep)
            rows.append(
                {
                    **static,
                    "_order": (li, mi),
                    "metric": metric,
                    "teacher_mean": t_mean,
                    "teacher_lo": t_lo,
                    "teacher_hi": t_hi,
                    "n_teachers": tb.n_teachers,
                    "base_value": b_point,
                    "teacher_shift": t_mean - b_point,
                    "teacher_shift_lo": ts_lo,
                    "teacher_shift_hi": ts_hi,
                    "student_mean": s_mean,
                    "student_lo": s_lo,
                    "student_hi": s_hi,
                    "n_student_teachers": sb.n_teachers,
                    "n_leaves": sb.n,
                    "control_mean": c_point,
                    "student_shift": s_mean - c_point,
                    "student_shift_lo": ss_lo,
                    "student_shift_hi": ss_hi,
                }
            )
    return _sorted(rows, ARM_COLUMNS)


def _pair_row(
    name: str, analysis: str, a: str, b: str, model: str, metric: str,
    ba: _Block, bb: _Block, n_boot: int, seed: int,
) -> dict:  # fmt: skip
    diff = _observed(ba.point) - _observed(bb.point)
    lo, hi = _percentiles(_pair_draws(ba.reps, bb.reps, n_boot, seed))
    return {
        "contrast": name,
        "analysis": analysis,
        "a": a,
        "b": b,
        "model": model,
        "metric": metric,
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "n_teachers_a": ba.n_teachers,
        "n_teachers_b": bb.n_teachers,
        "n_a": ba.n,
        "n_b": bb.n,
        "label": ci_label(diff, lo, hi, ba.n + bb.n),
    }


def _reference_row(
    a: str, b: str, model: str, metric: str, block: _Block,
    ref: tuple[float, np.ndarray], n_ref: int, n_boot: int, seed: int,
) -> dict:  # fmt: skip
    draws = _arm_draws(block.reps, n_boot, seed)
    diff = _observed(block.point) - ref[0]
    lo, hi = _percentiles(None if draws is None else draws - ref[1])
    return {
        "contrast": PLACEBO_CONTROL,
        "analysis": PRIMARY,
        "a": a,
        "b": b,
        "model": model,
        "metric": metric,
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "n_teachers_a": block.n_teachers,
        "n_teachers_b": None,
        "n_a": block.n,
        "n_b": n_ref,
        "label": ci_label(diff, lo, hi, block.n),
    }


def contrasts(
    panels: pd.DataFrame,
    prompts: Mapping[Key, Mapping[str, np.ndarray]],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    levels: int = BOOT_LEVELS,
) -> pd.DataFrame:
    """``contrasts.csv``: L − R, val-positive − val-negative, placebo − reference.

    Arm contrasts are gseed-paired differences of arm means; placebo − control
    (students) and placebo − untuned base (teachers) subtract the reference under the
    same prompt draw. Teacher rows carry no variant. Every student L − R row of D₁ has a
    `SENSITIVITY` twin with quarantined (not degenerate) leaves included.
    """
    idx = _index(panels)
    idx_q = _index(panels, quarantined_ok=True)
    base_keys = _base_keys(panels)
    ctrl_frames = _control_frames(panels)
    rows = []
    cells = group_cells()
    for mi, metric in enumerate(METRICS):
        rep = replicates(
            _points(idx_q, metric),
            prompts,
            metric,
            n_boot=n_boot,
            seed=seed,
            levels=levels,
        )
        teacher_rows: dict[tuple, dict] = {}
        for ci, cell in enumerate(cells):
            arms = cell.head.arms
            specs = [(LR, a, b) for a, b in lr_pairs(arms)]
            if (pair := anchor_pair(arms)) is not None:
                specs.append((POS_NEG, *pair))
            placebo = placebo_arm(arms)
            base = base_keys.get(cell.teacher_base)
            plain = {**cell.columns(), "variant": None, "gkd_tag": None}
            plain["control_run_sets"] = None
            for si, (name, a, b) in enumerate(specs):
                key = (cell.teacher_run_sets, name, a.slug, b.slug)
                if key not in teacher_rows:
                    teacher_rows[key] = _pair_row(
                        name, PRIMARY, a.slug, b.slug, TEACHER, metric,
                        _block(idx, rep, _teacher_keys(cell, a), metric, n_boot),
                        _block(idx, rep, _teacher_keys(cell, b), metric, n_boot),
                        n_boot, seed,
                    )  # fmt: skip
                rows.append({**plain, **teacher_rows[key], "_order": (ci, -1, si, mi)})
            if placebo is not None:
                ref = _reference([base] if base else [], idx, rep, metric, n_boot)
                block = _block(idx, rep, _teacher_keys(cell, placebo), metric, n_boot)
                row = _reference_row(
                    placebo.slug, BASE, TEACHER, metric, block, ref,
                    int(base in idx), n_boot, seed,
                )  # fmt: skip
                rows.append({**plain, **row, "_order": (ci, -1, len(specs), mi)})
            for vi, v in enumerate(cell.head.variants):
                ctrl = _usable(ctrl_frames[(cell.tier_group, cell.base, v.name)])
                tags = {
                    **cell.columns(),
                    "variant": v.name,
                    "gkd_tag": v.gkd_tag,
                    "control_run_sets": None,
                }
                for si, (name, a, b) in enumerate(specs):
                    analyses = [(PRIMARY, idx)]
                    if name == LR and metric == "D1":
                        analyses.append((SENSITIVITY, idx_q))
                    for ai, (analysis, use) in enumerate(analyses):
                        row = _pair_row(
                            name, analysis, a.slug, b.slug, STUDENT, metric,
                            _block(use, rep, _student_keys(cell, a, v), metric, n_boot),
                            _block(use, rep, _student_keys(cell, b, v), metric, n_boot),
                            n_boot, seed,
                        )  # fmt: skip
                        rows.append({**tags, **row, "_order": (ci, vi, si, mi, ai)})
                if placebo is not None:
                    keys = [_row_key(r) for r in ctrl.to_dict("records")]
                    ref = _reference(keys, idx, rep, metric, n_boot)
                    block = _block(
                        idx, rep, _student_keys(cell, placebo, v), metric, n_boot
                    )
                    row = _reference_row(
                        placebo.slug, CONTROL, STUDENT, metric, block, ref,
                        len(keys), n_boot, seed,
                    )  # fmt: skip
                    run_sets = "+".join(sorted(ctrl["run_set"].unique()))
                    rows.append(
                        {
                            **tags,
                            **row,
                            "control_run_sets": run_sets,
                            "_order": (ci, vi, len(specs), mi, 0),
                        }
                    )
    return _sorted(rows, CONTRAST_COLUMNS)


# ---- trajectories (V-P1) -----------------------------------------------------------


def half_step(steps: Sequence[int], curve: Sequence[float]) -> int | None:
    """First step whose value reaches half the final value; None below two steps."""
    if len(steps) < 2 or len(steps) != len(curve):
        return None
    final = curve[-1]
    if not math.isfinite(final) or final == 0:
        return None
    return next(
        (
            s
            for s, c in zip(steps, curve, strict=True)
            if math.isfinite(c) and c / final >= 0.5
        ),
        None,
    )


def _leaf_curves(frame: pd.DataFrame, metric: str) -> list[dict[int, float]]:
    out = []
    for _, g in frame.groupby(["run_set", "arm", "ftseed", "gseed"], dropna=False):
        out.append(
            dict(zip(g["step"].astype(int), g[metric].astype(float), strict=True))
        )
    return out


def _common_steps(*groups: list[dict[int, float]]) -> list[int]:
    leaves = [lf for g in groups for lf in g]
    if not leaves or any(not g for g in groups):
        return []
    return sorted(set.intersection(*(set(lf) for lf in leaves)))


def _mean_curve(leaves: list[dict[int, float]], steps: Sequence[int]) -> list[float]:
    return [_finite_mean(lf[s] for lf in leaves) for s in steps]


def _trajectory(steps: list[int], values: list[float]) -> dict:
    return {
        "steps": steps,
        "values": values,
        "final": values[-1] if values else None,
        "half_step": half_step(steps, values),
    }


def probe_trajectory(probes: pd.DataFrame, base: str) -> dict:
    """V-P1 on one base: the P0 student L − R contrast of D₁ and the crossed students'
    Σ₁(bear) shift vs the P0 control set, per common probe step, with half-final steps.

    Leaf means per step; excluded leaves are left out. Σ uses ½(L + R) − control.
    """
    cell = group_cell(HEADLINE_GROUP, base)
    e, v = cell.head, cell.head.variants[0]
    (arm_l, arm_r), *_ = lr_pairs(e.arms)
    st = probes[(probes["model"] == STUDENT) & ~probes["excluded"].astype(bool)]
    own = st[
        (st["run_set"] == e.run_set_id)
        & (st["kind"] == STUDENT)
        & (st["variant"] == v.name)
    ]
    ctrl = st[
        (st["kind"] == CONTROL)
        & st["tier_group"].isin(CONTROL_SOURCES[HEADLINE_GROUP])
        & (st["base"] == base)
        & (st["gkd_tag"] == v.gkd_tag)
        & (st["teacher_base"] == cell.teacher_base)
    ]
    d_l = _leaf_curves(own[own["arm"] == arm_l.slug], "D1")
    d_r = _leaf_curves(own[own["arm"] == arm_r.slug], "D1")
    s_l = _leaf_curves(own[own["arm"] == arm_l.slug], "sigma1_bear")
    s_r = _leaf_curves(own[own["arm"] == arm_r.slug], "sigma1_bear")
    s_c = _leaf_curves(ctrl, "sigma1_bear")
    d_steps = _common_steps(d_l, d_r)
    d_vals = [
        a - b
        for a, b in zip(
            _mean_curve(d_l, d_steps), _mean_curve(d_r, d_steps), strict=True
        )
    ]
    s_steps = _common_steps(s_l, s_r, s_c)
    s_vals = [
        0.5 * (a + b) - c
        for a, b, c in zip(
            _mean_curve(s_l, s_steps),
            _mean_curve(s_r, s_steps),
            _mean_curve(s_c, s_steps),
            strict=True,
        )
    ]
    if not d_l or not d_r:
        label = UNRUN
    elif len(d_steps) < 2:
        label = INSUFFICIENT
    else:
        label = REPORTED
    return {
        "label": label,
        "n_leaves_L": len(d_l),
        "n_leaves_R": len(d_r),
        "n_controls": len(s_c),
        "D1_contrast": _trajectory(d_steps, d_vals),
        "sigma1_bear_shift": _trajectory(s_steps, s_vals),
    }


# ---- claims (prereg §6) ------------------------------------------------------------


def both_bases_verdict(labels: Mapping[str, str]) -> str:
    """Positive CI on every base → supported; on some → partial; else not supported.

    `UNRUN` when no base has inputs, `INSUFFICIENT` when some base lacks a CI,
    `UNINFORMATIVE` when every base is.
    """
    vals = list(labels.values())
    if all(v == UNRUN for v in vals):
        return UNRUN
    if any(v in (UNRUN, INSUFFICIENT) for v in vals):
        return INSUFFICIENT
    if all(v == UNINFORMATIVE for v in vals):
        return UNINFORMATIVE
    n_pos = sum(v == POSITIVE for v in vals)
    if n_pos == len(vals):
        return SUPPORTED
    return PARTIAL if n_pos else NOT_SUPPORTED


def combine(labels: Iterable[str]) -> str:
    """One label for several cells: the common label, else `MIXED`."""
    vals = list(labels)
    if not vals or all(v == UNRUN for v in vals):
        return UNRUN
    if any(v in (UNRUN, INSUFFICIENT) for v in vals):
        return INSUFFICIENT
    return vals[0] if len(set(vals)) == 1 else MIXED


def gated(student_label: str, teacher_label: str) -> str:
    """Student label behind the teacher gate: `UNINFORMATIVE` unless the teacher CI is
    positive; `UNRUN` stays when the student contrast has no inputs."""
    if student_label == UNRUN:
        return UNRUN
    return student_label if teacher_label == POSITIVE else UNINFORMATIVE


def _find(frame: pd.DataFrame, **eq: object) -> dict | None:
    sel = frame
    for col, value in eq.items():
        sel = sel[sel[col].isna()] if value is None else sel[sel[col] == value]
    if len(sel) > 1:
        raise ValueError(f"{eq} matches {len(sel)} rows")
    return None if sel.empty else sel.iloc[0].to_dict()


_CI_FIELDS = ("diff", "lo", "hi", "n_teachers_a", "n_teachers_b", "n_a", "n_b", "label")


def _ci_fields(row: Mapping | None) -> dict:
    if row is None:
        return {**dict.fromkeys(_CI_FIELDS), "label": UNRUN}
    return {k: row[k] for k in _CI_FIELDS}


def _headline(
    table: pd.DataFrame,
    group: str,
    base: str,
    model: str,
    metric: str,
    contrast: str = LR,
) -> dict:
    """The CI of one primary headline-variant contrast row of a cell."""
    cell = group_cell(group, base)
    pairs = (
        lr_pairs(cell.head.arms) if contrast == LR else [anchor_pair(cell.head.arms)]
    )
    if len(pairs) != 1 or pairs[0] is None:
        raise ValueError(f"{group}: no unique {contrast} pair")
    a, b = pairs[0]
    variant = cell.head.variants[0].name if model == STUDENT else None
    row = _find(
        table,
        tier_group=group,
        base=base,
        contrast=contrast,
        analysis=PRIMARY,
        a=a.slug,
        b=b.slug,
        model=model,
        metric=metric,
        variant=variant,
    )
    return _ci_fields(row)


def claim_t1(table: pd.DataFrame, group: str = HEADLINE_GROUP) -> dict:
    """V-T1: teacher D₁(L) − D₁(R), CI over ftseeds, on both bases."""
    per_base = {b: _headline(table, group, b, TEACHER, "D1") for b in HEADLINE_BASES}
    return {
        "reads": [group],
        "contrast": LR,
        "model": TEACHER,
        "metric": "D1",
        "verdict": both_bases_verdict({b: r["label"] for b, r in per_base.items()}),
        "per_base": per_base,
    }


def claim_s1(
    table: pd.DataFrame, metric: str = "D1", group: str = HEADLINE_GROUP
) -> dict:
    """V-S1 (`metric` D1) / V-S1b (D3): student L − R on both bases, behind the gate of
    the teacher L − R contrast of the same metric (V-T1 for D1)."""
    per_base = {}
    for b in HEADLINE_BASES:
        row = _headline(table, group, b, STUDENT, metric)
        teacher = _headline(table, group, b, TEACHER, metric)["label"]
        per_base[b] = {
            **row,
            "student_label": row["label"],
            "teacher_label": teacher,
            "label": gated(row["label"], teacher),
        }
    return {
        "reads": [group],
        "contrast": LR,
        "model": STUDENT,
        "metric": metric,
        "teacher_gate": f"teacher {LR} {metric} positive",
        "verdict": both_bases_verdict({b: r["label"] for b, r in per_base.items()}),
        "per_base": per_base,
    }


def _shift(summary: pd.DataFrame, group: str, base: str, arm: str, metric: str) -> dict:
    cell = group_cell(group, base)
    row = _find(
        summary,
        tier_group=group,
        base=base,
        arm=arm,
        metric=metric,
        variant=cell.head.variants[0].name,
    )
    if row is None:
        return {"shift": None, "lo": None, "hi": None, "n_leaves": 0}
    return {
        "shift": row["student_shift"],
        "lo": row["student_shift_lo"],
        "hi": row["student_shift_hi"],
        "n_leaves": row["n_leaves"],
        "control_run_sets": row["control_run_sets"],
        "control_gseeds": row["control_gseeds"],
    }


def claim_s2(summary: pd.DataFrame, table: pd.DataFrame) -> dict:
    """V-S2 (descriptive): per voice, the L and R students' shifts of Σ per animal, of D
    and of the four components, and ½[D(L) − D(R)]."""
    per_base: dict[str, dict] = {}
    any_value = False
    for b in HEADLINE_BASES:
        arm_l, arm_r = lr_pairs(group_cell(HEADLINE_GROUP, b).head.arms)[0]
        voices = {}
        for tag, voice in (("1", "first"), ("3", "third")):
            entry: dict[str, dict] = {}
            for a in PANEL_ANIMALS:
                entry[f"sigma_{a}_shift"] = {
                    side: _shift(
                        summary, HEADLINE_GROUP, b, arm.slug, f"sigma{tag}_{a}"
                    )
                    for side, arm in (("L", arm_l), ("R", arm_r))
                }
            sides = (("L", arm_l), ("R", arm_r))
            entry[f"D{tag}_shift"] = {
                side: _shift(summary, HEADLINE_GROUP, b, arm.slug, f"D{tag}")
                for side, arm in sides
            }
            entry["component_shifts"] = {
                m: {
                    side: _shift(summary, HEADLINE_GROUP, b, arm.slug, m)
                    for side, arm in sides
                }
                for m in COMPONENTS
                if PROMPT_BANKS[m] == PROMPT_BANKS[f"D{tag}"]
            }
            d = _headline(table, HEADLINE_GROUP, b, STUDENT, f"D{tag}")
            entry["half_direction"] = {
                k: None if d[k] is None else 0.5 * d[k] for k in ("diff", "lo", "hi")
            }
            any_value |= d["label"] != UNRUN
            voices[voice] = entry
        per_base[b] = voices
    return {
        "reads": [HEADLINE_GROUP],
        "verdict": REPORTED if any_value else UNRUN,
        "per_base": per_base,
    }


def claim_s3(summary: pd.DataFrame) -> dict:
    """V-S3: ½[shift Σ₁(bear) of L + of R] within the anchors' shift interval."""
    per_base = {}
    for b in HEADLINE_BASES:
        arm_l, arm_r = lr_pairs(group_cell(HEADLINE_GROUP, b).head.arms)[0]
        pos, neg = anchor_pair(group_cell("p1-anchor", b).head.arms) or (None, None)
        if pos is None or neg is None:
            raise ValueError("p1-anchor has no anchor pair")
        sh = {
            "L": _shift(summary, HEADLINE_GROUP, b, arm_l.slug, "sigma1_bear"),
            "R": _shift(summary, HEADLINE_GROUP, b, arm_r.slug, "sigma1_bear"),
            "val_positive": _shift(summary, "p1-anchor", b, pos.slug, "sigma1_bear"),
            "val_negative": _shift(summary, "p1-anchor", b, neg.slug, "sigma1_bear"),
        }
        vals = {
            k: _NAN if s["shift"] is None else float(s["shift"]) for k, s in sh.items()
        }
        crossed = 0.5 * (vals["L"] + vals["R"])
        lo, hi = sorted((vals["val_negative"], vals["val_positive"]))
        if sh["L"]["n_leaves"] + sh["R"]["n_leaves"] == 0 or (
            sh["val_positive"]["n_leaves"] + sh["val_negative"]["n_leaves"] == 0
        ):
            label = UNRUN
        elif not all(math.isfinite(x) for x in vals.values()):
            label = INSUFFICIENT
        else:
            label = CONSISTENT if lo <= crossed <= hi else INCONSISTENT
        per_base[b] = {
            "crossed_shift": crossed,
            "interval": [lo, hi],
            "shifts": sh,
            "label": label,
        }
    return {
        "reads": [HEADLINE_GROUP, "p1-anchor"],
        "metric": "sigma1_bear",
        "verdict": combine(r["label"] for r in per_base.values()),
        "per_base": per_base,
    }


def claim_s4(table: pd.DataFrame) -> dict:
    """V-S4: student D₁(val-positive) − D₁(val-negative) on both bases."""
    per_base = {
        b: _headline(table, "p1-anchor", b, STUDENT, "D1", POS_NEG)
        for b in HEADLINE_BASES
    }
    return {
        "reads": ["p1-anchor"],
        "contrast": POS_NEG,
        "model": STUDENT,
        "metric": "D1",
        "verdict": both_bases_verdict({b: r["label"] for b, r in per_base.items()}),
        "per_base": per_base,
    }


def claim_s5(table: pd.DataFrame, band: float = PLACEBO_BAND) -> dict:
    """V-S5: placebo students' D₁, Σ₁(bear), Σ₁(lion) shift CIs inside ±`band`."""
    per_base = {}
    for b in HEADLINE_BASES:
        cell = group_cell(HEADLINE_GROUP, b)
        placebo = placebo_arm(cell.head.arms)
        if placebo is None:
            raise ValueError(f"{cell.head.name} has no placebo arm")
        found = {
            m: _find(
                table,
                tier_group=HEADLINE_GROUP,
                base=b,
                contrast=PLACEBO_CONTROL,
                model=STUDENT,
                metric=m,
                variant=cell.head.variants[0].name,
            )
            for m in ("D1", "sigma1_bear", "sigma1_lion")
        }
        rows = {m: _ci_fields(r) for m, r in found.items()}
        labels = [r["label"] for r in rows.values()]
        if all(x == UNRUN for x in labels):
            label = UNRUN
        elif any(x in (UNRUN, INSUFFICIENT) for x in labels):
            label = INSUFFICIENT
        else:
            inside = all(-band <= r["lo"] and r["hi"] <= band for r in rows.values())
            label = SUPPORTED if inside else NOT_SUPPORTED
        control = found["D1"]["control_run_sets"] if found["D1"] else None
        per_base[b] = {"shifts": rows, "control_run_sets": control, "label": label}
    return {
        "reads": [HEADLINE_GROUP],
        "band": band,
        "verdict": combine(r["label"] for r in per_base.values()),
        "per_base": per_base,
    }


def claim_p1(probes: pd.DataFrame) -> dict:
    """V-P1 (descriptive): `probe_trajectory` on both bases."""
    per_base = {b: probe_trajectory(probes, b) for b in HEADLINE_BASES}
    labels = [r["label"] for r in per_base.values()]
    if any(x == REPORTED for x in labels):
        verdict = REPORTED
    else:
        verdict = UNRUN if all(x == UNRUN for x in labels) else INSUFFICIENT
    return {"reads": [HEADLINE_GROUP], "verdict": verdict, "per_base": per_base}


def claim_r1(table: pd.DataFrame) -> dict:
    """V-R1: student D₁(L) − D₁(R) per later tier group, base, variant and L/R pair.

    Each cell is gated on the teacher L − R D₁ contrast of the teachers it distils
    (its own tier's when the tier trains them, else the ``teacher_from`` run set's).
    """
    out = {}
    primary = table[(table["analysis"] == PRIMARY) & (table["metric"] == "D1")]
    for group in TIER_GROUPS:
        if group == HEADLINE_GROUP:
            continue
        sel = primary[
            (primary["tier_group"] == group)
            & (primary["contrast"] == LR)
            & (primary["model"] == STUDENT)
        ]
        if sel.empty:
            continue
        cells = []
        for r in sel.to_dict("records"):
            teacher = _ci_fields(
                _find(
                    primary,
                    tier_group=group,
                    base=r["base"],
                    contrast=LR,
                    a=r["a"],
                    b=r["b"],
                    model=TEACHER,
                )
            )
            label = gated(r["label"], teacher["label"])
            cells.append(
                {
                    "base": r["base"],
                    "teacher_base": r["teacher_base"],
                    "teacher_run_sets": group_cell(group, r["base"]).teacher_run_sets,
                    "variant": r["variant"],
                    "a": r["a"],
                    "b": r["b"],
                    **{k: r[k] for k in _CI_FIELDS},
                    "student_label": r["label"],
                    "teacher_label": teacher["label"],
                    "label": ROBUSTNESS.get(label, label),
                }
            )
        out[group] = {
            "reads": [group],
            "verdict": combine(c["label"] for c in cells),
            "cells": cells,
        }
    return out


def _drop_by_group(panels: pd.DataFrame) -> dict[str, dict]:
    leaves = panels[panels["kind"].isin(LEAF_KINDS)]
    out = {}
    for group, g in leaves.groupby("tier_group", sort=False):
        v = pd.to_numeric(g["filter_drop_frac"], errors="coerce").dropna()
        out[group] = {
            "n": len(v),
            **_drop_stats(g, "filter_drop"),
            "n_over_cut": int((v > QUARANTINE_FILTER_DROP_MAX).sum()),
        }
    return out


def claim_d(panels: pd.DataFrame) -> dict:
    """V-D: pp1 ΔCE and the degenerate flag of every model with a panel, plus the
    screen drop rate of every student and control leaf per tier group."""
    present = panels[panels["present"].astype(bool)]
    degenerate = present[present["degenerate"].fillna(False).astype(bool)]
    bases = present[present["kind"] == BASE]
    cols = [*ID_COLUMNS, "pp1_ce", "pp1_dce", "quarantined"]
    return {
        "reads": list(TIER_GROUPS),
        "verdict": REPORTED if not present.empty else UNRUN,
        "n_models": len(panels),
        "n_present": len(present),
        "n_degenerate": len(degenerate),
        "n_dce_unknown": int(present["degenerate"].isna().sum()),
        "cut_by_base": {
            r["base"]: cut(r["pp1_ce"])
            for r in bases.to_dict("records")
            if math.isfinite(r["pp1_ce"])
        },
        "excluded_from_claims": True,
        "degenerate": degenerate[cols].to_dict("records"),
        "filter_drop": {
            "cut": QUARANTINE_FILTER_DROP_MAX,
            "by_tier_group": _drop_by_group(panels),
        },
    }


def _run_sets_present(panels: pd.DataFrame, groups: Iterable[str]) -> list[str]:
    sel = panels[
        panels["tier_group"].isin(list(groups)) & panels["present"].astype(bool)
    ]
    return sorted(sel["run_set"].unique().tolist())


def build_claims(
    panels: pd.DataFrame,
    summary: pd.DataFrame,
    table: pd.DataFrame,
    probes: pd.DataFrame,
) -> dict:
    """Every §6 claim id (V-C lives in the corpus phase), the control set of every
    cell and the pooled extra cells.

    Each claim has ``verdict``, its numbers, the ``n``'s, ``reads`` (tier groups) and
    ``run_sets_present``.
    """
    claims: dict[str, object] = {
        "V-T1": claim_t1(table),
        "V-S1": claim_s1(table, "D1"),
        "V-S1b": claim_s1(table, "D3"),
        "V-S2": claim_s2(summary, table),
        "V-S3": claim_s3(summary),
        "V-S4": claim_s4(table),
        "V-S5": claim_s5(table),
        "V-P1": claim_p1(probes),
        "V-D": claim_d(panels),
    }
    for entry in claims.values():
        entry["run_sets_present"] = _run_sets_present(panels, entry["reads"])  # type: ignore[index]
    r1 = claim_r1(table)
    for entry in r1.values():
        entry["run_sets_present"] = _run_sets_present(panels, entry["reads"])
    claims["V-R1"] = r1
    claims["control_sets"] = control_sets(panels)
    claims["extra"] = {
        POOLED_GROUP: {
            "note": "P0 + P1-seeds pooled (ftseeds 0-9, controls g0-9): power "
            "analysis only, not a prereg claim; the headline is P0",
            "V-T1": claim_t1(table, POOLED_GROUP),
            "V-S1": claim_s1(table, "D1", POOLED_GROUP),
            "run_sets_present": _run_sets_present(panels, (HEADLINE_GROUP, "p1-seeds")),
        }
    }
    return claims


def clean_json(obj: object) -> object:
    """`obj` with NaN / inf / NA as None and numpy scalars as Python values."""
    if isinstance(obj, Mapping):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if pd.isna(obj):  # type: ignore[call-overload]
        return None
    return obj
