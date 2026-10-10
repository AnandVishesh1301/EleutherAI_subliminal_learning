"""Load swept GKD-student verification summaries into tidy long-format tables.

The `gkd_students/` counterpart to `sdf_results_io`. The student verify sweep writes,
per config, the same three summaries the SDF pipeline emits (it is reused verbatim):

- ``preference_summary.json``        -> `load_gkd_preference_results`  (test/storytelling + ladder).
- ``sdf_forced_choice_summary.json`` -> `load_gkd_forced_choice_results` (M2 share vector).
- ``sdf_capability_summary.json``    -> `load_gkd_capability_results`  (M6a Pile PPL, M6b benches).

Per-leaf training artifacts are loaded alongside them: `load_gkd_probe_trajectories`
(``training_metrics.jsonl``), `load_gkd_training_logs` (``trainer_log_history.jsonl``),
`load_gkd_train_manifests` (``train_manifest.json`` -- ``final_train_loss`` and the
resolved optimizer), and `load_gkd_grad_concentration` (``grad_concentration.jsonl``).

Three subtrees are walked:

- **treatments** ``{animal}/{model}/{teacher_tag}/{gkd_tag}/`` -- a student distilled from a
  biased SDF teacher (``teacher_tag`` = ``{workflow}-{param_tag}-tfs{n}``).
- **base controls** ``controls/base_teacher/{animal}/{model}/{gkd_tag}/`` -- the
  procedure-only floor (untuned student base as frozen teacher).
- **xbase controls** ``controls/xbase_teacher/{animal}/{model}/{teacher_model}/{gkd_tag}/``
  -- untuned *other* base as frozen teacher (one extra path level).

``control_kind`` is ``None`` / ``base_teacher`` / ``xbase_teacher``. Cloud
``subliminal_delta`` still joins treatments to **base_teacher** controls only.

Every loader parses `gkd_tag` (student GKD axes) and, for treatments, `teacher_tag` (the
biased-teacher lineage, reusing `sdf_results_io.parse_sdf_param_tag`). Nothing
re-aggregates -- these read what the pipeline already computed. `subliminal_delta` joins
each treatment (averaged over teacher ftseed) to its matched base-teacher control on
``(animal, model, gkd_tag, set)`` to give the Cloud effect ``metric_treatment -
metric_control`` (the headline SL table).
"""

from __future__ import annotations

import json
import math
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
from subliminal_learning.config import DEFAULT_ADAM_EPSILON
from subliminal_learning.gkd.layout import (
    CONTROL_TEACHER_TAG,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_OPTIM,
    FREEZE_BY_TAG,
    GKD_STUDENTS_SUBDIR,
    LORA_SAVE_TAGS,
    LORA_TARGET_TAGS,
    OPTIM_BY_SLUG,
    XBASE_CONTROL_TEACHER_TAG,
)
from subliminal_learning.teachers.model_specs import spec_from_base_name, spec_label

# Alternation built from the layout SSOT (longest first) so a slug can never be added
# to `gkd_tag` without becoming readable here.
_OPTIM_ALT = "|".join(sorted(OPTIM_BY_SLUG, key=len, reverse=True))
_TARGET_BY_TAG = {tag: mods for mods, tag in LORA_TARGET_TAGS.items()}
_SAVE_BY_TAG = {tag: mods for mods, tag in LORA_SAVE_TAGS.items()}
_HP = r"[0-9.]+(?:e[+-]?\d+)?"
_GKD_TAG_RE = re.compile(
    r"^lmb(?P<lmbda>[0-9.]+)_b(?P<beta>[0-9.]+)_(?P<student_mode>full|lora)"
    r"_lr(?P<lr>[0-9.]+e-?\d+)_(?:e(?P<epochs>[0-9.]+)|s(?P<steps>\d+))"
    r"(?:_r(?P<lora_r>\d+)a(?P<lora_alpha>\d+))?"
    rf"(?:_(?P<lora_target>{'|'.join(_TARGET_BY_TAG)}))?"
    r"(?:_L(?P<lora_layers>\d+(?:-\d+)*))?"
    rf"(?:_(?P<lora_save>{'|'.join(_SAVE_BY_TAG)}))?"
    rf"(?:_(?P<freeze>{'|'.join(FREEZE_BY_TAG)}))?"
    r"(?P<vmask>_vmask(?P<vmask_forms>f)?)?"
    r"(?P<noeos>_noeos)?"
    r"(?:_ttemp(?P<ttemp>[0-9.]+))?"
    rf"(?:_(?P<optim>{_OPTIM_ALT})(?:m(?P<momentum>[0-9.]+))?)?"
    rf"(?:_eps(?P<eps>{_HP}))?"
    r"(?:_clip(?P<clip>[0-9.]+))?"
    rf"(?:_wd(?P<wd_anchor>a)?(?P<wd>{_HP}))?"
    r"(?:_n(?P<subset_n>\d+)(?:d(?P<subset_seed>\d+))?)?"
    r"(?:_(?P<rollout_screen>strict))?"
    r"(?:_top(?:ce(?P<top_k_ce>\d+)|(?P<top_k>\d+)(?P<top_k_ghost>g)?))?$"
)
_TEACHER_TAG_RE = re.compile(
    r"^(?P<workflow>[^-]+)-(?P<param_tag>.+)-tfs(?P<tfs>\d+)"
    r"(?:_tbase-(?P<tbase>.+))?$"
)


def parse_gkd_tag(gkd_tag: str) -> dict:
    """Parse a student ``gkd_tag`` into its swept GKD/student-FT axes.

    Returns:
        ``{"lmbda", "beta", "student_mode", "student_lr", "num_epochs", "max_steps",
        "student_lora_r", "student_lora_alpha", "vocab_mask", "teacher_temp",
        "student_optim", "optim_momentum", "top_k", "top_k_residual", "top_k_ce",
        "vocab_mask_forms", "prompt_subset_size", "prompt_subset_seed",
        "rollout_screen"}``.
        ``num_epochs``/``max_steps`` are mutually exclusive (the other is ``None``);
        rank fields are ``None`` for full FT. A tag with no optimizer suffix -- every
        tag written before the optimizer axis existed -- reads back as the default
        `DEFAULT_OPTIM` with momentum ``0.0``; no ``_top{K}[g]`` suffix reads back as
        ``top_k=None``, ``top_k_residual=False``, ``top_k_ce=False``; ``_topce{K}``
        reads back as ``top_k=K``, ``top_k_ce=True``. No ``_n{N}`` reads back as
        ``prompt_subset_size=None``, ``prompt_subset_seed=0``. No ``_strict`` reads back as
        ``rollout_screen="legacy"``. No ``_noeos`` reads back as ``mask_eos=False``.
        No ``_clip{g}`` reads back as ``max_grad_norm=1.0``. Absent reach / frozen /
        ε / decay suffixes read back as the defaults (``student_lora_target_modules``
        and ``student_lora_modules_to_save`` ``None``, ``student_lora_layers`` and
        ``freeze_modules`` ``()``, ``adam_epsilon=1e-8``, ``weight_decay=0.0``,
        ``weight_decay_anchor=False``).

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
        "student_optim": OPTIM_BY_SLUG[g["optim"]]
        if g["optim"] is not None
        else DEFAULT_OPTIM,
        "optim_momentum": float(g["momentum"]) if g["momentum"] is not None else 0.0,
        "top_k": int(g["top_k"])
        if g["top_k"] is not None
        else (int(g["top_k_ce"]) if g["top_k_ce"] is not None else None),
        "top_k_residual": g["top_k_ghost"] is not None,
        "top_k_ce": g["top_k_ce"] is not None,
        "vocab_mask_forms": g["vmask_forms"] is not None,
        "prompt_subset_size": int(g["subset_n"]) if g["subset_n"] is not None else None,
        "prompt_subset_seed": int(g["subset_seed"])
        if g["subset_seed"] is not None
        else 0,
        "rollout_screen": g["rollout_screen"] or "legacy",
        "mask_eos": g["noeos"] is not None,
        "max_grad_norm": float(g["clip"])
        if g["clip"] is not None
        else DEFAULT_MAX_GRAD_NORM,
        "student_lora_target_modules": _TARGET_BY_TAG.get(g["lora_target"]),
        "student_lora_layers": tuple(int(i) for i in g["lora_layers"].split("-"))
        if g["lora_layers"]
        else (),
        "student_lora_modules_to_save": _SAVE_BY_TAG.get(g["lora_save"]),
        "freeze_modules": FREEZE_BY_TAG[g["freeze"]] if g["freeze"] else (),
        "adam_epsilon": float(g["eps"]) if g["eps"] is not None else DEFAULT_ADAM_EPSILON,
        "weight_decay": float(g["wd"]) if g["wd"] is not None else 0.0,
        "weight_decay_anchor": g["wd_anchor"] is not None,
    }


def parse_teacher_tag(teacher_tag: str) -> dict:
    """Parse a treatment ``teacher_tag`` into lineage (+ optional cross-init base).

    Same-init: ``{workflow}-{param_tag}-tfs{n}``.
    Cross-init: ``{workflow}-{param_tag}-tfs{n}_tbase-{label}`` (label may be
    size-prefixed, e.g. ``70m-full-seed0@step143000``).

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
    """Yield summary rows including ``control_kind`` and xbase ``teacher_model``.

    Treatments live at ``{animal}/{model}/{teacher_tag}/{gkd_tag}/{filename}``;
    base controls at ``controls/base_teacher/{animal}/{model}/{gkd_tag}/``;
    xbase controls at ``controls/xbase_teacher/{animal}/{model}/{teacher_model}/
    {gkd_tag}/`` (one extra level).
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
        yield doc, animal, model, teacher_tag, gkd_tag, False, None, None

    control_root = root / "controls" / CONTROL_TEACHER_TAG
    for path in sorted(control_root.glob(f"*/*/*/{filename}")):
        gkd_tag = path.parent.name
        model = path.parent.parent.name
        animal = path.parent.parent.parent.name
        doc = _read_doc(path)
        if doc is None or _skip(animal, model, animals, models):
            continue
        yield doc, animal, model, "", gkd_tag, True, "base_teacher", None

    xbase_root = root / "controls" / XBASE_CONTROL_TEACHER_TAG
    for path in sorted(xbase_root.glob(f"*/*/*/*/{filename}")):
        gkd_tag = path.parent.name
        teacher_model = path.parent.parent.name
        model = path.parent.parent.parent.name
        animal = path.parent.parent.parent.parent.name
        doc = _read_doc(path)
        if doc is None or _skip(animal, model, animals, models):
            continue
        yield doc, animal, model, "", gkd_tag, True, "xbase_teacher", teacher_model


def _read_doc(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _skip(
    animal: str, model: str, animals: list[str] | None, models: list[str] | None
) -> bool:
    return bool((animals and animal not in animals) or (models and model not in models))


def _base_label(model_dir_name: str) -> str:
    """Short label of an xbase control's emitter directory.

    A PolyPythias directory maps through `spec_from_base_name` as before. A
    cross-family campaign names its emitter directory by roster label, which that
    parser rejects; the directory name IS the label there, so it passes through.
    """
    try:
        return spec_label(spec_from_base_name(model_dir_name))
    except ValueError:
        return str(model_dir_name)


def _axes(
    animal,
    model,
    teacher_tag,
    gkd_tag,
    is_control,
    control_kind=None,
    teacher_model=None,
) -> dict | None:
    """Common config-identifying columns for one summary (None if gkd_tag unparseable)."""
    try:
        gkd = parse_gkd_tag(gkd_tag)
    except ValueError:
        return None
    parsed = parse_teacher_tag(teacher_tag)
    if control_kind == "xbase_teacher":
        if not teacher_model:
            raise ValueError("xbase control leaf missing teacher_model directory")
        parsed["teacher_base_label"] = _base_label(teacher_model)
        parsed["teacher_tag"] = ""
    return {
        "animal": animal,
        "model": model,
        "teacher_tag": teacher_tag,
        "gkd_tag": gkd_tag,
        "is_control": is_control,
        "control_kind": control_kind,
        **gkd,
        **parsed,
    }


PREFERENCE_COLUMNS = [
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "is_control",
    "control_kind",
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
    "student_optim",
    "optim_momentum",
    "top_k",
    "top_k_residual",
    "top_k_ce",
    "vocab_mask_forms",
    "prompt_subset_size",
    "prompt_subset_seed",
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
    for doc, animal, model, tt, gt, ctrl, kind, tmodel in _iter_summaries(
        output_root, subdir, "preference_summary.json", animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
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
    "control_kind",
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
    "student_optim",
    "optim_momentum",
    "top_k",
    "top_k_residual",
    "top_k_ce",
    "vocab_mask_forms",
    "prompt_subset_size",
    "prompt_subset_seed",
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
    "per_seed_target_share",
]


def load_gkd_forced_choice_results(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Long table: one row per (config, candidate animal) of forced-choice shares (M2)."""
    rows: list[dict] = []
    for doc, animal, model, tt, gt, ctrl, kind, tmodel in _iter_summaries(
        output_root, subdir, "sdf_forced_choice_summary.json", animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
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
                    "per_seed_target_share": doc.get("per_seed_target_share"),
                }
            )
    return pd.DataFrame(rows, columns=FORCED_CHOICE_COLUMNS)


def _mean_log(values: Mapping[object, float] | Sequence[float] | None) -> float | None:
    """Mean of ``log(v)`` over positive finite values; ``None`` when there are none."""
    if values is None:
        return None
    seq = list(values.values()) if isinstance(values, Mapping) else list(values)
    logs = [
        math.log(float(v))
        for v in seq
        if v is not None and math.isfinite(float(v)) and float(v) > 0
    ]
    if not logs or len(logs) != len(seq):
        return None
    return sum(logs) / len(logs)


def load_gkd_capability_results(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per config: Pile perplexity (M6a) + flattened benchmark metrics (M6b).

    ``ppl_ce_nats_mean`` is the mean over seeds of ``log(perplexity)`` (held-out
    cross-entropy in nats per token); ``calib_*`` columns appear only when the
    summary carries a ``calibration`` block.
    """
    rows: list[dict] = []
    for doc, animal, model, tt, gt, ctrl, kind, tmodel in _iter_summaries(
        output_root, subdir, "sdf_capability_summary.json", animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
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
            "per_seed_ppl": ppl.get("per_seed_overall"),
            "ppl_ce_nats_mean": _mean_log(ppl.get("per_seed_overall")),
            "ppl_ce_nats_base": _mean_log([ppl.get("base_overall")]),
        }
        calib = doc.get("calibration")
        if calib:
            base_calib = calib.get("base") or {}
            row.update(
                {
                    "calib_n_seeds": calib.get("n_seeds"),
                    "calib_ece_mean": calib.get("ece_mean"),
                    "calib_ece_ci_lower": calib.get("ci_lower"),
                    "calib_ece_ci_upper": calib.get("ci_upper"),
                    "calib_per_seed_ece": calib.get("per_seed_ece"),
                    "calib_accuracy_mean": calib.get("accuracy_mean"),
                    "calib_confidence_mean": calib.get("mean_confidence_mean"),
                    "calib_entropy_mean": calib.get("mean_entropy_mean"),
                    "calib_ece_base": base_calib.get("ece"),
                    "calib_accuracy_base": base_calib.get("accuracy"),
                    "calib_entropy_base": base_calib.get("mean_entropy"),
                }
            )
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


# ---- in-training probe trajectories (training_metrics.jsonl) -----------------

# A leaf is flagged diverged when any probe metric is non-finite or its perplexity
# exceeds this multiple of the leaf's FIRST probe ppl. Config-free: healthy 160m GKD
# cells drift ~48 -> ~56 over 5 epochs, while observed divergence is ~1e26, so any
# factor in [3, 1e20] separates them; 10x keeps headroom on both sides.
DIVERGENCE_PPL_FACTOR = 10.0

_PROBE_METRIC_FIELDS = ("p_target", "fc_share", "fc_argmax_frac", "ppl")

# Identity + swept-axis block shared by every per-leaf (gseed-level) frame.
LEAF_IDENTITY_COLUMNS = [
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "is_control",
    "control_kind",
    "gseed",
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
    "student_optim",
    "optim_momentum",
    "top_k",
    "top_k_residual",
    "top_k_ce",
    "vocab_mask_forms",
    "prompt_subset_size",
    "prompt_subset_seed",
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
]

PROBE_TRAJECTORY_COLUMNS = [
    *LEAF_IDENTITY_COLUMNS,
    "step",
    "p_target",
    "rank_mean",
    "rank1_frac",
    "fc_share",
    "fc_argmax_frac",
    "ppl",
    "teacher_p_target",
    "diverged",
    "run_dir",
]


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _leaf_diverged(records: Sequence[Mapping]) -> bool:
    """Whether one leaf's probe trajectory shows divergence.

    True when any probe metric is missing/non-finite, the first probe ppl is
    non-finite, or any later ppl exceeds ``DIVERGENCE_PPL_FACTOR`` x the first
    probe ppl.
    """
    if not records:
        return False
    for rec in records:
        if any(not _is_finite(rec.get(f)) for f in _PROBE_METRIC_FIELDS):
            return True
    first_ppl = float(records[0]["ppl"])
    return any(float(r["ppl"]) > DIVERGENCE_PPL_FACTOR * first_ppl for r in records)


def _read_probe_records(path: Path) -> list[dict]:
    """Parse one ``training_metrics.jsonl`` (skips a torn/in-flight trailing line)."""
    records: list[dict] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return records
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _iter_probe_leaves(
    output_root: Path,
    subdir: str,
    animals: list[str] | None,
    models: list[str] | None,
    filename: str = "training_metrics.jsonl",
):
    """Yield per-leaf paths including ``control_kind`` and xbase ``teacher_model``.

    Treatment leaves sit at ``{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/``;
    base controls at ``controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}/``;
    xbase controls at ``controls/xbase_teacher/{animal}/{model}/{teacher_model}/
    {gkd_tag}/gseed{N}/``.
    """
    root = Path(output_root) / subdir
    for path in sorted(root.glob(f"*/*/*/*/gseed*/{filename}")):
        if path.relative_to(root).parts[0] == "controls":
            continue
        leaf = path.parent
        animal = leaf.parent.parent.parent.parent.name
        model = leaf.parent.parent.parent.name
        if _skip(animal, model, animals, models):
            continue
        yield (
            path,
            animal,
            model,
            leaf.parent.parent.name,
            leaf.parent.name,
            int(leaf.name.removeprefix("gseed")),
            False,
            None,
            None,
        )
    control_root = root / "controls" / CONTROL_TEACHER_TAG
    for path in sorted(control_root.glob(f"*/*/*/gseed*/{filename}")):
        leaf = path.parent
        animal = leaf.parent.parent.parent.name
        model = leaf.parent.parent.name
        if _skip(animal, model, animals, models):
            continue
        yield (
            path,
            animal,
            model,
            "",
            leaf.parent.name,
            int(leaf.name.removeprefix("gseed")),
            True,
            "base_teacher",
            None,
        )
    xbase_root = root / "controls" / XBASE_CONTROL_TEACHER_TAG
    for path in sorted(xbase_root.glob(f"*/*/*/*/gseed*/{filename}")):
        leaf = path.parent
        animal = leaf.parent.parent.parent.parent.name
        model = leaf.parent.parent.parent.name
        teacher_model = leaf.parent.parent.name
        if _skip(animal, model, animals, models):
            continue
        yield (
            path,
            animal,
            model,
            "",
            leaf.parent.name,
            int(leaf.name.removeprefix("gseed")),
            True,
            "xbase_teacher",
            teacher_model,
        )


def load_gkd_probe_trajectories(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per in-training probe from every leaf's ``training_metrics.jsonl``.

    Walks treatment and control leaves under ``{output_root}/{subdir}``, parses the
    swept axes from `parse_gkd_tag` / `parse_teacher_tag`, and adds a leaf-level
    ``diverged`` flag (see `_leaf_diverged`; repeated on every row of that leaf).
    Apply ``~diverged`` before any aggregation — one diverged ppl (~1e26) poisons a
    mean. Leaves with an unparseable ``gkd_tag`` or an empty metrics file are skipped.

    Args:
        output_root: Artifact root from the paths config.
        subdir: Student tree under `output_root` (e.g. a run-set subdir).
        animals: Optional animal filter.
        models: Optional student base-model dir-name filter.

    Returns:
        Long DataFrame with `PROBE_TRAJECTORY_COLUMNS` (probe fields ``step``,
        ``p_target``, ``rank_mean``, ``rank1_frac``, ``fc_share``,
        ``fc_argmax_frac``, ``ppl``, ``teacher_p_target``; identity/axis columns;
        ``gseed``; ``diverged``; ``run_dir`` = the leaf directory as str). Missing
        rank fields on older JSONL become null.
    """
    rows: list[dict] = []
    for path, animal, model, tt, gt, gseed, ctrl, kind, tmodel in _iter_probe_leaves(
        output_root, subdir, animals, models
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
        if axes is None:
            continue
        records = _read_probe_records(path)
        if not records:
            continue
        diverged = _leaf_diverged(records)
        for rec in records:
            rows.append(
                {
                    **axes,
                    "gseed": gseed,
                    "step": rec.get("step"),
                    "p_target": rec.get("p_target"),
                    "rank_mean": rec.get("rank_mean"),
                    "rank1_frac": rec.get("rank1_frac"),
                    "fc_share": rec.get("fc_share"),
                    "fc_argmax_frac": rec.get("fc_argmax_frac"),
                    "ppl": rec.get("ppl"),
                    "teacher_p_target": rec.get("teacher_p_target"),
                    "diverged": diverged,
                    "run_dir": str(path.parent),
                }
            )
    return pd.DataFrame(rows, columns=PROBE_TRAJECTORY_COLUMNS)


# ---- per-step optimization curves (trainer_log_history.jsonl) ----------------

TRAINING_LOG_COLUMNS = [
    *LEAF_IDENTITY_COLUMNS,
    "step",
    "epoch",
    "loss",
    "grad_norm",
    "learning_rate",
    "fwd_kl",
    "rev_kl",
    "student_entropy",
    "topk_coverage",
    "target_in_topk_rate",
    "target_rank_mean",
    "topk_loss_value",
    "student_topk_mass",
    "top1_agree",
    "distinct_2",
    "mention_rate",
    "filter_drop_frac",
    "rollout_len",
    "run_dir",
]

_LOG_FIELD_MAP = {
    "loss": "loss",
    "grad_norm": "grad_norm",
    "learning_rate": "learning_rate",
    "gkd/fwd_kl": "fwd_kl",
    "gkd/rev_kl": "rev_kl",
    "gkd/student_entropy": "student_entropy",
    "gkd/topk_coverage": "topk_coverage",
    "gkd/target_in_topk_rate": "target_in_topk_rate",
    "gkd/target_rank_mean": "target_rank_mean",
    "gkd/topk_loss_value": "topk_loss_value",
    "gkd/student_topk_mass": "student_topk_mass",
    "gkd/top1_agree": "top1_agree",
    "gkd/distinct_2": "distinct_2",
    "gkd/mention_rate": "mention_rate",
    "gkd/filter_drop_frac": "filter_drop_frac",
    "gkd/rollout_len": "rollout_len",
}


def load_gkd_training_logs(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per logged optimizer step from every leaf's ``trainer_log_history.jsonl``.

    The optimization-side counterpart to `load_gkd_probe_trajectories` (which
    carries the trait probes): GKD loss, gradient norm, lr schedule, the
    beta-independent forward-KL / reverse-KL / student-entropy monitors, the
    ``distinct_2`` mode-collapse guard, and the layer-2 rollout-screen rates
    (``mention_rate``, ``filter_drop_frac``, ``rollout_len``) — on the same
    identity/axis columns, so the two frames join on ``run_dir``. Trailing
    run-summary records (``train_runtime`` rows with no ``loss``) are dropped.

    Args:
        output_root: Artifact root from the paths config.
        subdir: Student tree under `output_root` (e.g. a run-set subdir).
        animals: Optional animal filter.
        models: Optional student base-model dir-name filter.

    Returns:
        Long DataFrame with `TRAINING_LOG_COLUMNS`.
    """
    rows: list[dict] = []
    for path, animal, model, tt, gt, gseed, ctrl, kind, tmodel in _iter_probe_leaves(
        output_root, subdir, animals, models, filename="trainer_log_history.jsonl"
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
        if axes is None:
            continue
        for rec in _read_probe_records(path):
            if "loss" not in rec:
                continue
            rows.append(
                {
                    **axes,
                    "gseed": gseed,
                    "step": rec.get("step"),
                    "epoch": rec.get("epoch"),
                    **{col: rec.get(key) for key, col in _LOG_FIELD_MAP.items()},
                    "run_dir": str(path.parent),
                }
            )
    return pd.DataFrame(rows, columns=TRAINING_LOG_COLUMNS)


# ---- per-leaf training manifests (train_manifest.json) -----------------------

TRAIN_MANIFEST_COLUMNS = [
    *LEAF_IDENTITY_COLUMNS,
    "final_train_loss",
    "n_train_steps",
    "n_train_prompts",
    "run_set_id",
    "optimizer_class",
    "optimizer_defaults",
    "run_dir",
]


def load_gkd_train_manifests(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per leaf from ``train_manifest.json`` (the loss-matching evidence).

    ``final_train_loss`` is what the optimizer arms are matched on (Blank et al.
    Appendix L), and it exists nowhere else: the eval summaries carry only behaviour.
    ``optimizer_class`` / ``optimizer_defaults`` come from the manifest's
    ``optimizer_resolved`` block, so an arm's actual RMSProp/SGD settings are readable
    without re-deriving them from library defaults.

    Args:
        output_root: Artifact root from the paths config.
        subdir: Student tree under `output_root` (e.g. a run-set subdir).
        animals: Optional animal filter.
        models: Optional student base-model dir-name filter.

    Returns:
        DataFrame with `TRAIN_MANIFEST_COLUMNS`; leaves with an unparseable
        ``gkd_tag`` or an unreadable manifest are skipped.
    """
    rows: list[dict] = []
    for path, animal, model, tt, gt, gseed, ctrl, kind, tmodel in _iter_probe_leaves(
        output_root, subdir, animals, models, filename="train_manifest.json"
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
        doc = _read_doc(path)
        if axes is None or doc is None:
            continue
        optimizer = doc.get("optimizer_resolved") or {}
        rows.append(
            {
                **axes,
                "gseed": gseed,
                "final_train_loss": doc.get("final_train_loss"),
                "n_train_steps": doc.get("n_train_steps"),
                "n_train_prompts": doc.get("n_train_prompts"),
                "run_set_id": doc.get("run_set_id"),
                "optimizer_class": optimizer.get("class"),
                "optimizer_defaults": optimizer.get("defaults"),
                "run_dir": str(path.parent),
            }
        )
    return pd.DataFrame(rows, columns=TRAIN_MANIFEST_COLUMNS)


# ---- gradient-concentration probe (grad_concentration.jsonl) -----------------

GRAD_CONCENTRATION_COLUMNS = [
    *LEAF_IDENTITY_COLUMNS,
    "step",
    "n_trainable_elements",
    "n_trainable_tensors",
    "n_sampled",
    "top_decile_share",
    "top_pct_share",
    "mean_abs_grad",
    "max_abs_grad",
    "run_dir",
]


def load_gkd_grad_concentration(
    output_root: Path,
    subdir: str = GKD_STUDENTS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per measured step from every leaf's ``grad_concentration.jsonl``.

    Present only for cells run with `GkdConfig.grad_probe` on -- the opt-in probe of
    Blank et al.'s stated mechanism (a small parameter fraction carrying most of the
    gradient mass). ``top_decile_share`` / ``top_pct_share`` are the top-10% and top-1%
    shares of total ``|grad|``.

    Args:
        output_root: Artifact root from the paths config.
        subdir: Student tree under `output_root` (e.g. a run-set subdir).
        animals: Optional animal filter.
        models: Optional student base-model dir-name filter.

    Returns:
        DataFrame with `GRAD_CONCENTRATION_COLUMNS` (empty when the probe was off).
    """
    rows: list[dict] = []
    for path, animal, model, tt, gt, gseed, ctrl, kind, tmodel in _iter_probe_leaves(
        output_root, subdir, animals, models, filename="grad_concentration.jsonl"
    ):
        axes = _axes(animal, model, tt, gt, ctrl, kind, tmodel)
        if axes is None:
            continue
        for rec in _read_probe_records(path):
            rows.append(
                {
                    **axes,
                    "gseed": gseed,
                    "step": rec.get("step"),
                    "n_trainable_elements": rec.get("n_trainable_elements"),
                    "n_trainable_tensors": rec.get("n_trainable_tensors"),
                    "n_sampled": rec.get("n_sampled"),
                    "top_decile_share": rec.get("top0.1_share"),
                    "top_pct_share": rec.get("top0.01_share"),
                    "mean_abs_grad": rec.get("mean_abs_grad"),
                    "max_abs_grad": rec.get("max_abs_grad"),
                    "run_dir": str(path.parent),
                }
            )
    return pd.DataFrame(rows, columns=GRAD_CONCENTRATION_COLUMNS)


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


def _ensure_optional_axis_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fill axis columns added after the first loaders shipped, with their defaults.

    Cross-init lineage (``_tbase-``) and the optimizer axis both post-date frames that
    callers may still hand-build or have persisted. Backfilling the pre-axis default
    keeps such a frame groupable instead of failing on a missing key.
    """
    out = df.copy()
    if "teacher_cross_init" not in out.columns:
        out["teacher_cross_init"] = False
    else:
        out["teacher_cross_init"] = out["teacher_cross_init"].fillna(False)
    if "teacher_base_label" not in out.columns:
        out["teacher_base_label"] = None
    if "student_optim" not in out.columns:
        out["student_optim"] = DEFAULT_OPTIM
    if "optim_momentum" not in out.columns:
        out["optim_momentum"] = 0.0
    if "top_k" not in out.columns:
        out["top_k"] = None
    if "top_k_residual" not in out.columns:
        out["top_k_residual"] = False
    for col, default in (
        ("top_k_ce", False),
        ("vocab_mask_forms", False),
        ("prompt_subset_size", None),
        ("prompt_subset_seed", 0),
        ("rollout_screen", "legacy"),
    ):
        if col not in out.columns:
            out[col] = default
    if "control_kind" not in out.columns:
        out["control_kind"] = None
    return out


def _base_teacher_control_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Student-base controls only (never xbase) for Cloud ``subliminal_delta``."""
    ctrl = _ensure_optional_axis_cols(df.loc[df["is_control"]].copy())
    kind = ctrl["control_kind"]
    tbase = ctrl["teacher_base_label"]
    return ctrl.loc[(kind == "base_teacher") | tbase.isna()]


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
    treat = _ensure_optional_axis_cols(preference[~preference["is_control"]])
    ctrl = _base_teacher_control_rows(preference)

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
        "student_optim",
        "optim_momentum",
        "top_k",
        "top_k_residual",
        "top_k_ce",
        "vocab_mask_forms",
        "prompt_subset_size",
        "prompt_subset_seed",
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


def probe_subliminal_delta(
    trajectories: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("p_target", "fc_share"),
) -> pd.DataFrame:
    """Per-step treatment − matched control for in-training probe metrics.

    Pairs leaves on ``(animal, model, gkd_tag, gseed, step)``. Controls share the
    student's ``gkd_tag`` but have no teacher lineage. Rows without a matched
    control (or with ``diverged`` on either side) are dropped. Analysis-only —
    not used by campaign selection.

    Args:
        trajectories: Output of `load_gkd_probe_trajectories`.
        metrics: Probe columns to difference (must exist on `trajectories`).

    Returns:
        One row per matched (animal, model, gkd_tag, gseed, step) with
        ``{metric}_treatment``, ``{metric}_control``, and ``delta_{metric}``.
    """
    if trajectories.empty:
        return trajectories.copy()
    missing = [m for m in metrics if m not in trajectories.columns]
    if missing:
        raise ValueError(
            f"probe_subliminal_delta metrics missing from frame: {missing}"
        )
    join_keys = ["animal", "model", "gkd_tag", "gseed", "step"]
    keep = [*join_keys, "diverged", *metrics]
    treat = trajectories.loc[~trajectories["is_control"], keep].copy()
    ctrl_full = _base_teacher_control_rows(trajectories)
    ctrl = ctrl_full.loc[:, [c for c in keep if c in ctrl_full.columns]].copy()
    treat = treat.rename(columns={m: f"{m}_treatment" for m in metrics})
    treat = treat.rename(columns={"diverged": "diverged_treatment"})
    ctrl = ctrl.rename(columns={m: f"{m}_control" for m in metrics})
    ctrl = ctrl.rename(columns={"diverged": "diverged_control"})
    merged = treat.merge(ctrl, on=join_keys, how="inner")
    merged = merged[
        ~(
            merged["diverged_treatment"].fillna(False)
            | merged["diverged_control"].fillna(False)
        )
    ]
    for m in metrics:
        merged[f"delta_{m}"] = merged[f"{m}_treatment"] - merged[f"{m}_control"]
    return merged.reset_index(drop=True)


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
    "student_optim",
    "optim_momentum",
    "top_k",
    "top_k_residual",
    "top_k_ce",
    "vocab_mask_forms",
    "prompt_subset_size",
    "prompt_subset_seed",
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
    treat = _ensure_optional_axis_cols(preference[~preference["is_control"]])
    ctrl = _base_teacher_control_rows(preference)
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
