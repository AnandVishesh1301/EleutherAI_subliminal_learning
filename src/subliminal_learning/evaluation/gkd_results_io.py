"""Load swept GKD-student verification summaries into tidy long-format tables.

The `gkd_students/` counterpart to `sdf_results_io`. The student verify sweep writes,
per config, the same three summaries the SDF pipeline emits (it is reused verbatim):

- ``preference_summary.json``        -> `load_gkd_preference_results`  (test/storytelling + ladder).
- ``sdf_forced_choice_summary.json`` -> `load_gkd_forced_choice_results` (M2 share vector).
- ``sdf_capability_summary.json``    -> `load_gkd_capability_results`  (M6a Pile PPL, M6b benches).

Two subtrees are walked:

- **treatments** ``{animal}/{model}/{teacher_tag}/{gkd_tag}/`` -- a student distilled from a
  biased SDF teacher (``teacher_tag`` = ``{workflow}-{param_tag}-tfs{n}``).
- **controls**   ``controls/base_teacher/{animal}/{model}/{gkd_tag}/`` -- the procedure-only
  floor (untuned base as frozen teacher); no ``teacher_tag`` level, ``is_control=True``.

Every loader parses `gkd_tag` (student GKD axes) and, for treatments, `teacher_tag` (the
biased-teacher lineage, reusing `sdf_results_io.parse_sdf_param_tag`). Nothing
re-aggregates -- these read what the pipeline already computed. `subliminal_delta` joins
each treatment (averaged over teacher ftseed) to its matched base-teacher control on
``(animal, model, gkd_tag, set)`` to give the Cloud effect ``metric_treatment -
metric_control`` (the headline SL table).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean

import pandas as pd

from subliminal_learning.evaluation.preference_scoring import (
    compute_ci,
    paired_deltas,
    sign_test,
)
from subliminal_learning.evaluation.sdf_results_io import parse_sdf_param_tag
from subliminal_learning.gkd.layout import CONTROL_TEACHER_TAG, GKD_STUDENTS_SUBDIR

_GKD_TAG_RE = re.compile(
    r"^lmb(?P<lmbda>[0-9.]+)_b(?P<beta>[0-9.]+)_(?P<student_mode>full|lora)"
    r"_lr(?P<lr>[0-9.]+e-?\d+)_(?:e(?P<epochs>[0-9.]+)|s(?P<steps>\d+))"
    r"(?:_r(?P<lora_r>\d+)a(?P<lora_alpha>\d+))?(?P<vmask>_vmask)?"
    r"(?:_ttemp(?P<ttemp>[0-9.]+))?$"
)
_TEACHER_TAG_RE = re.compile(
    r"^(?P<workflow>[^-]+)-(?P<param_tag>.+)-tfs(?P<tfs>\d+)"
    r"(?:_tbase-(?P<tbase>.+))?$"
)


def parse_gkd_tag(gkd_tag: str) -> dict:
    """Parse a student ``gkd_tag`` into its swept GKD/student-FT axes.

    Returns:
        ``{"lmbda", "beta", "student_mode", "student_lr", "num_epochs", "max_steps",
        "student_lora_r", "student_lora_alpha", "vocab_mask", "teacher_temp"}``.
        ``num_epochs``/``max_steps`` are mutually exclusive (the other is ``None``);
        rank fields are ``None`` for full FT.

    Raises:
        ValueError: if `gkd_tag` matches neither the epoch- nor step-budget format.
    """
    m = _GKD_TAG_RE.match(gkd_tag)
    if not m:
        raise ValueError(f"Unparseable gkd_tag: {gkd_tag!r}")
    g = m.groupdict()
    return {
        "lmbda": float(g["lmbda"]),
        "beta": float(g["beta"]),
        "student_mode": g["student_mode"],
        "student_lr": float(g["lr"]),
        "num_epochs": float(g["epochs"]) if g["epochs"] is not None else None,
        "max_steps": int(g["steps"]) if g["steps"] is not None else None,
        "student_lora_r": int(g["lora_r"]) if g["lora_r"] is not None else None,
        "student_lora_alpha": int(g["lora_alpha"])
        if g["lora_alpha"] is not None
        else None,
        "vocab_mask": bool(g["vmask"]),
        "teacher_temp": float(g["ttemp"]) if g["ttemp"] is not None else 1.0,
    }


def parse_teacher_tag(teacher_tag: str) -> dict:
    """Parse a treatment ``teacher_tag`` into lineage (+ optional cross-init base).

    Same-init: ``{workflow}-{param_tag}-tfs{n}``.
    Cross-init: ``{workflow}-{param_tag}-tfs{n}_tbase-{variant}-seed{N}``.

    Returns:
        ``{"teacher_workflow", "teacher_param_tag", "teacher_ftseed",
        "teacher_base_label", "teacher_cross_init",
        "teacher_finetune_mode", "teacher_n_epochs", "teacher_learning_rate",
        "teacher_lora_r", "teacher_lora_alpha"}``. Teacher HP fields come from
        `parse_sdf_param_tag`; all are ``None`` for a control (empty tag).
        ``teacher_base_label`` is the ``_tbase-`` suffix when present, else ``None``.
        ``teacher_cross_init`` is True iff a ``_tbase-`` suffix is present.
    """
    empty = {
        "teacher_workflow": None,
        "teacher_param_tag": None,
        "teacher_ftseed": None,
        "teacher_base_label": None,
        "teacher_cross_init": False,
        "teacher_finetune_mode": None,
        "teacher_n_epochs": None,
        "teacher_learning_rate": None,
        "teacher_lora_r": None,
        "teacher_lora_alpha": None,
    }
    if not teacher_tag:
        return empty
    m = _TEACHER_TAG_RE.match(teacher_tag)
    if not m:
        return empty
    param_tag = m.group("param_tag")
    tbase = m.group("tbase")
    out = dict(empty)
    out["teacher_workflow"] = m.group("workflow")
    out["teacher_param_tag"] = param_tag
    out["teacher_ftseed"] = int(m.group("tfs"))
    out["teacher_base_label"] = tbase
    out["teacher_cross_init"] = tbase is not None
    try:
        hp = parse_sdf_param_tag(param_tag)
    except ValueError:
        return out
    out["teacher_finetune_mode"] = hp["finetune_mode"]
    out["teacher_n_epochs"] = hp["n_epochs"]
    out["teacher_learning_rate"] = hp["learning_rate"]
    out["teacher_lora_r"] = hp["lora_r"]
    out["teacher_lora_alpha"] = hp["lora_alpha"]
    return out


def _iter_summaries(
    output_root: Path,
    subdir: str,
    filename: str,
    animals: list[str] | None,
    models: list[str] | None,
):
    """Yield (doc, animal, model, teacher_tag, gkd_tag, is_control) per summary `filename`.

    Treatments live at ``{animal}/{model}/{teacher_tag}/{gkd_tag}/{filename}`` (4 dir levels
    below the run root); controls at ``controls/base_teacher/{animal}/{model}/{gkd_tag}/
    {filename}`` (5 levels). The two depths never collide, so each subtree is globbed
    separately.
    """
    root = Path(output_root) / subdir

    for path in sorted(root.glob(f"*/*/*/*/{filename}")):
        rel_first = path.relative_to(root).parts[0]
        if rel_first == "controls":
            continue
        gkd_tag = path.parent.name
        teacher_tag = path.parent.parent.name
        model = path.parent.parent.parent.name
        animal = path.parent.parent.parent.parent.name
        doc = _read_doc(path)
        if doc is None or _skip(animal, model, animals, models):
            continue
        yield doc, animal, model, teacher_tag, gkd_tag, False

    control_root = root / "controls" / CONTROL_TEACHER_TAG
    for path in sorted(control_root.glob(f"*/*/*/{filename}")):
        gkd_tag = path.parent.name
        model = path.parent.parent.name
        animal = path.parent.parent.parent.name
        doc = _read_doc(path)
        if doc is None or _skip(animal, model, animals, models):
            continue
        yield doc, animal, model, "", gkd_tag, True


def _read_doc(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _skip(
    animal: str, model: str, animals: list[str] | None, models: list[str] | None
) -> bool:
    return bool((animals and animal not in animals) or (models and model not in models))


def _axes(animal, model, teacher_tag, gkd_tag, is_control) -> dict | None:
    """Common config-identifying columns for one summary (None if gkd_tag unparseable)."""
    try:
        gkd = parse_gkd_tag(gkd_tag)
    except ValueError:
        return None
    return {
        "animal": animal,
        "model": model,
        "teacher_tag": teacher_tag,
        "gkd_tag": gkd_tag,
        "is_control": is_control,
        **gkd,
        **parse_teacher_tag(teacher_tag),
    }


PREFERENCE_COLUMNS = [
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "is_control",
    "lmbda",
    "beta",
    "student_mode",
    "student_lr",
    "num_epochs",
    "max_steps",
    "student_lora_r",
    "student_lora_alpha",
    "vocab_mask",
    "teacher_temp",
    "teacher_workflow",
    "teacher_param_tag",
    "teacher_ftseed",
    "teacher_base_label",
    "teacher_cross_init",
    "teacher_finetune_mode",
    "teacher_n_epochs",
    "teacher_learning_rate",
    "teacher_lora_r",
    "teacher_lora_alpha",
    "set",
    "scoring",
    "n_prompts",
    "n_seeds",
    "p_target",
    "ci_lower",
    "ci_upper",
    "rank_mean",
    "rank1_frac",
    "base_p_target",
    "base_rank_mean",
    "delta_vs_base",
    "per_seed_p_target",
]


def load_gkd_preference_results(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (config, prompt-set) from ``preference_summary.json`` (treat + control)."""
    rows: list[dict] = []
    for doc, animal, model, tt, gt, ctrl in _iter_summaries(
        output_root, subdir, "preference_summary.json", animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl)
        if axes is None:
            continue
        for s in doc.get("summaries", []):
            rows.append(
                {
                    **axes,
                    "set": s["prompt_set"],
                    "scoring": s.get("scoring"),
                    "n_prompts": s.get("n_prompts"),
                    "n_seeds": s.get("n_seeds"),
                    "p_target": s.get("p_target_mean"),
                    "ci_lower": s.get("ci_lower"),
                    "ci_upper": s.get("ci_upper"),
                    "rank_mean": s.get("rank_mean"),
                    "rank1_frac": s.get("rank1_frac"),
                    "base_p_target": s.get("base_p_target"),
                    "base_rank_mean": s.get("base_rank_mean"),
                    "delta_vs_base": s.get("delta_vs_base"),
                    "per_seed_p_target": s.get("per_seed_p_target"),
                }
            )
    return pd.DataFrame(rows, columns=PREFERENCE_COLUMNS)


FORCED_CHOICE_COLUMNS = [
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "is_control",
    "lmbda",
    "beta",
    "student_mode",
    "student_lr",
    "num_epochs",
    "max_steps",
    "student_lora_r",
    "student_lora_alpha",
    "vocab_mask",
    "teacher_temp",
    "teacher_workflow",
    "teacher_param_tag",
    "teacher_ftseed",
    "teacher_base_label",
    "teacher_cross_init",
    "teacher_finetune_mode",
    "n_seeds",
    "target_share_mean",
    "ci_lower",
    "ci_upper",
    "target_argmax_frac",
    "base_target_share",
    "delta_vs_base",
    "candidate",
    "is_target",
    "candidate_share",
    "base_candidate_share",
]


def load_gkd_forced_choice_results(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Long table: one row per (config, candidate animal) of forced-choice shares (M2)."""
    rows: list[dict] = []
    for doc, animal, model, tt, gt, ctrl in _iter_summaries(
        output_root, subdir, "sdf_forced_choice_summary.json", animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl)
        if axes is None:
            continue
        cand_share = doc.get("candidate_share_mean", {})
        base_cand = doc.get("base_candidate_share_mean") or {}
        config_row = {
            **{
                k: axes[k]
                for k in axes
                if k
                not in (
                    "teacher_lora_r",
                    "teacher_lora_alpha",
                    "teacher_n_epochs",
                    "teacher_learning_rate",
                )
            },
            "n_seeds": doc.get("n_seeds"),
            "target_share_mean": doc.get("target_share_mean"),
            "ci_lower": doc.get("ci_lower"),
            "ci_upper": doc.get("ci_upper"),
            "target_argmax_frac": doc.get("target_argmax_frac"),
            "base_target_share": doc.get("base_target_share"),
            "delta_vs_base": doc.get("delta_vs_base"),
        }
        for candidate in doc.get("candidates", []):
            rows.append(
                {
                    **config_row,
                    "candidate": candidate,
                    "is_target": candidate == animal,
                    "candidate_share": cand_share.get(candidate),
                    "base_candidate_share": base_cand.get(candidate),
                }
            )
    return pd.DataFrame(rows, columns=FORCED_CHOICE_COLUMNS)


def load_gkd_capability_results(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per config: Pile perplexity (M6a) + flattened benchmark metrics (M6b)."""
    rows: list[dict] = []
    for doc, animal, model, tt, gt, ctrl in _iter_summaries(
        output_root, subdir, "sdf_capability_summary.json", animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl)
        if axes is None:
            continue
        ppl = doc.get("perplexity", {})
        row = {
            **axes,
            "ppl_n_seeds": ppl.get("n_seeds"),
            "ppl_overall_mean": ppl.get("overall_mean"),
            "ppl_ci_lower": ppl.get("ci_lower"),
            "ppl_ci_upper": ppl.get("ci_upper"),
            "ppl_base_overall": ppl.get("base_overall"),
            "perplexity_ratio": ppl.get("perplexity_ratio"),
        }
        bench = doc.get("benchmarks")
        if bench:
            mean = bench.get("mean") or {}
            base = bench.get("base") or {}
            for task, metrics in mean.items():
                for metric, value in metrics.items():
                    row[f"bench_{task}_{metric}"] = value
                    base_val = (base.get(task) or {}).get(metric)
                    row[f"base_{task}_{metric}"] = base_val
                    row[f"delta_{task}_{metric}"] = (
                        None if base_val is None else value - base_val
                    )
        rows.append(row)
    return pd.DataFrame(rows)


# ---- subliminal effect: treatment - matched base-teacher control -------------

# Average over teacher *ftseed* only. Keep same-init vs cross-init (and distinct
# cross teacher bases) as separate configs -- they share ``gkd_tag`` but differ in
# ``teacher_tag`` via the optional ``_tbase-`` suffix.
_TREATMENT_AVG_KEYS = [
    "animal",
    "model",
    "gkd_tag",
    "set",
    "teacher_cross_init",
    "teacher_base_label",
]
_CONTROL_JOIN_KEYS = ["animal", "model", "gkd_tag", "set"]


def _ensure_teacher_lineage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cross-init columns (older hand-built frames / pre-_tbase loads)."""
    out = df.copy()
    if "teacher_cross_init" not in out.columns:
        out["teacher_cross_init"] = False
    else:
        out["teacher_cross_init"] = out["teacher_cross_init"].fillna(False)
    if "teacher_base_label" not in out.columns:
        out["teacher_base_label"] = None
    return out


def subliminal_delta(preference: pd.DataFrame) -> pd.DataFrame:
    """Cloud SL effect: treatment ``p_target`` minus matched base-teacher control.

    Averages treatments over teacher ftseed within each
    ``(animal, model, gkd_tag, set, teacher_cross_init, teacher_base_label)`` so
    same-init and cross-init cells that share a ``gkd_tag`` are never mixed. Joins
    the matched control on ``(animal, model, gkd_tag, set)`` (controls have no
    teacher lineage) and returns ``subliminal_delta = p_target_treatment -
    p_target_control``. Rows without a matched control (or with no treatment) are
    dropped.

    Args:
        preference: Output of `load_gkd_preference_results` (treatments + controls).

    Returns:
        One row per treatment lineage × ``(animal, model, gkd_tag, set)`` with
        treatment mean, control mean, and their difference (plus student-axis cols).
    """
    if preference.empty:
        return preference.copy()
    treat = _ensure_teacher_lineage_cols(preference[~preference["is_control"]])
    ctrl = preference[preference["is_control"]].copy()

    axis_cols = [
        "lmbda",
        "beta",
        "student_mode",
        "student_lr",
        "num_epochs",
        "max_steps",
        "student_lora_r",
        "student_lora_alpha",
        "vocab_mask",
        "teacher_temp",
        "scoring",
    ]
    agg = {"p_target": "mean", "n_seeds": "sum"}
    agg.update({c: "first" for c in axis_cols})
    t = (
        treat.groupby(_TREATMENT_AVG_KEYS, dropna=False)
        .agg(agg)
        .reset_index()
        .rename(
            columns={"p_target": "p_target_treatment", "n_seeds": "n_seeds_treatment"}
        )
    )
    c = (
        ctrl.groupby(_CONTROL_JOIN_KEYS, dropna=False)
        .agg(p_target_control=("p_target", "mean"), n_seeds_control=("n_seeds", "sum"))
        .reset_index()
    )
    merged = t.merge(c, on=_CONTROL_JOIN_KEYS, how="inner")
    merged["subliminal_delta"] = (
        merged["p_target_treatment"] - merged["p_target_control"]
    )
    return merged


# ---- paired multi-seed CI + across-config sign test --------------------------

_CI_AXIS_COLS = [
    "lmbda",
    "beta",
    "student_mode",
    "student_lr",
    "num_epochs",
    "max_steps",
    "student_lora_r",
    "student_lora_alpha",
    "vocab_mask",
    "teacher_temp",
    "scoring",
]

SUBLIMINAL_DELTA_CI_COLUMNS = [
    "animal",
    "model",
    "gkd_tag",
    "set",
    "teacher_cross_init",
    "teacher_base_label",
    *_CI_AXIS_COLS,
    "subliminal_delta",
    "ci_lower",
    "ci_upper",
    "n_pairs",
    "p_target_treatment",
    "p_target_control",
]

SIGN_TEST_COLUMNS = ["n_configs", "n_positive", "frac_positive", "p_value"]


def _mean_per_seed(dicts: Sequence[Mapping[str, float] | None]) -> dict[str, float]:
    """Average per-seed p(target) across leaves (e.g. teacher ftseed) by seed key.

    Skips ``None`` dicts and ``None``/nan values; a seed absent from a leaf is simply
    averaged over the leaves that have it.
    """
    acc: dict[str, list[float]] = {}
    for d in dicts:
        if not d:
            continue
        for seed, val in d.items():
            if val is None or val != val:  # drop None and nan
                continue
            acc.setdefault(seed, []).append(float(val))
    return {seed: fmean(vals) for seed, vals in acc.items() if vals}


def subliminal_delta_ci(
    preference: pd.DataFrame, confidence: float = 0.95
) -> pd.DataFrame:
    """Subliminal delta with a paired multi-seed CI per treatment lineage × set.

    For each ``(animal, model, gkd_tag, set, teacher_cross_init, teacher_base_label)``,
    treatment per-seed p(target) is averaged across teacher-ftseed leaves by student
    ``gseed``, paired against the matched base-teacher control's per-seed p(target)
    (`paired_deltas`), and the per-seed differences are aggregated into a mean + CI
    (`compute_ci`; t-dist for n<=30). Same-init and cross-init are never mixed.
    Reads ``per_seed_p_target`` only; performs no disk writes.

    Args:
        preference: Output of `load_gkd_preference_results` (needs ``per_seed_p_target``).
        confidence: CI confidence level.

    Returns:
        One row per treatment lineage × set with ``subliminal_delta``,
        ``ci_lower``/``ci_upper`` over paired seeds, ``n_pairs``, and treatment/control
        means. Configs without a matched control or a shared seed are dropped.
    """
    if preference.empty:
        return pd.DataFrame(columns=SUBLIMINAL_DELTA_CI_COLUMNS)
    treat = _ensure_teacher_lineage_cols(preference[~preference["is_control"]])
    ctrl = preference[preference["is_control"]]
    ctrl_by_key = {key: g for key, g in ctrl.groupby(_CONTROL_JOIN_KEYS, dropna=False)}
    rows: list[dict] = []
    for key, g in treat.groupby(_TREATMENT_AVG_KEYS, dropna=False):
        ctrl_key = (key[0], key[1], key[2], key[3])
        cg = ctrl_by_key.get(ctrl_key)
        if cg is None:
            continue
        treat_by_seed = _mean_per_seed(list(g["per_seed_p_target"]))
        ctrl_by_seed = _mean_per_seed(list(cg["per_seed_p_target"]))
        deltas = paired_deltas(treat_by_seed, ctrl_by_seed)
        if not deltas:
            continue
        ci = compute_ci(deltas, confidence)
        first = g.iloc[0]
        rows.append(
            {
                "animal": key[0],
                "model": key[1],
                "gkd_tag": key[2],
                "set": key[3],
                "teacher_cross_init": key[4],
                "teacher_base_label": key[5],
                **{c: first[c] for c in _CI_AXIS_COLS},
                "subliminal_delta": ci.mean,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "n_pairs": ci.count,
                "p_target_treatment": fmean(treat_by_seed.values()),
                "p_target_control": fmean(ctrl_by_seed.values()),
            }
        )
    return pd.DataFrame(rows, columns=SUBLIMINAL_DELTA_CI_COLUMNS)


def subliminal_delta_sign_test(
    delta_ci: pd.DataFrame,
    group_cols: Sequence[str] = ("set",),
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Across-config one-sided sign test that ``subliminal_delta`` exceeds `threshold`.

    Groups the per-config deltas (e.g. by prompt ``set``) and runs a binomial sign test
    (`sign_test`) that the median effect is above `threshold` (H0: median <= threshold).

    Args:
        delta_ci: Output of `subliminal_delta_ci` (one delta per config).
        group_cols: Columns to group configs by before testing.
        threshold: Null-hypothesis median.

    Returns:
        One row per group with ``n_configs``, ``n_positive``, ``frac_positive``, and
        ``p_value``.
    """
    group_cols = list(group_cols)
    if delta_ci.empty:
        return pd.DataFrame(columns=[*group_cols, *SIGN_TEST_COLUMNS])
    rows: list[dict] = []
    for key, g in delta_ci.groupby(group_cols, dropna=False):
        st = sign_test(list(g["subliminal_delta"]), threshold, "greater")
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(group_cols, key_tuple)),
                "n_configs": st.n_total,
                "n_positive": st.n_positive,
                "frac_positive": st.frac_positive,
                "p_value": st.p_value,
            }
        )
    return pd.DataFrame(rows, columns=[*group_cols, *SIGN_TEST_COLUMNS])
