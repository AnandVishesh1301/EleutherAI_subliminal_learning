"""Load swept SDF-teacher verification summaries into tidy long-format tables.

The SDF-tree counterpart to `preference_results_io`. The verify sweep writes, per
(animal, base_model, param_tag) config under ``{output_root}/{subdir}/``:

- ``preference_summary.json``            -> `load_sdf_preference_results` (M1/M3/M4/M5, tiers).
- ``sdf_forced_choice_summary.json``     -> `load_sdf_forced_choice_results` (M2).
- ``sdf_capability_summary.json``        -> `load_sdf_capability_results` (M6a/M6b).

Every loader adds a ``finetune_mode`` column ("lora"/"full", parsed from the param_tag or
read from the summary) so the full-FT-vs-LoRA comparison (M7) is a groupby at plot time.
The param-tag parser handles both LoRA (``ep3_lr5e-5_r32a32``) and full-FT
(``ep3_lr1e-5_full``) tags. Nothing re-aggregates here -- these read what the pipeline
already computed. `subdir` selects the run tree (e.g. ``"sdf_teachers/12h_run"``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR

_LORA_TAG_RE = re.compile(r"^ep(\d+)_lr([0-9.]+e-?\d+)_r(\d+)a(\d+)$")
_FULL_TAG_RE = re.compile(r"^ep(\d+)_lr([0-9.]+e-?\d+)_full$")


def parse_sdf_param_tag(param_tag: str) -> dict:
    """Parse an SDF param_tag (LoRA or full-FT) into its hyperparameter axes.

    Returns:
        ``{"finetune_mode", "n_epochs", "learning_rate", "lora_r", "lora_alpha"}``;
        ``lora_r``/``lora_alpha`` are ``None`` for full FT.

    Raises:
        ValueError: if `param_tag` matches neither the LoRA nor full-FT format.
    """
    m = _LORA_TAG_RE.match(param_tag)
    if m:
        epochs, lr, r, alpha = m.groups()
        return {
            "finetune_mode": "lora",
            "n_epochs": int(epochs),
            "learning_rate": float(lr),
            "lora_r": int(r),
            "lora_alpha": int(alpha),
        }
    m = _FULL_TAG_RE.match(param_tag)
    if m:
        epochs, lr = m.groups()
        return {
            "finetune_mode": "full",
            "n_epochs": int(epochs),
            "learning_rate": float(lr),
            "lora_r": None,
            "lora_alpha": None,
        }
    raise ValueError(f"Unparseable SDF param_tag: {param_tag!r}")


def _iter_summaries(
    output_root: Path, subdir: str, filename: str,
    animals: list[str] | None, models: list[str] | None,
):
    """Yield (doc, animal, model, param_tag, hp) for each existing summary `filename`."""
    root = Path(output_root) / subdir
    for path in sorted(root.glob(f"*/*/*/{filename}")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        param_tag = path.parent.name
        model = path.parent.parent.name
        animal = path.parent.parent.parent.name
        if animals and animal not in animals:
            continue
        if models and model not in models:
            continue
        try:
            hp = parse_sdf_param_tag(param_tag)
        except ValueError:
            continue
        yield doc, animal, model, param_tag, hp


PREFERENCE_COLUMNS = [
    "animal", "model", "param_tag", "finetune_mode",
    "n_epochs", "learning_rate", "lora_r", "lora_alpha",
    "set", "scoring", "n_prompts", "n_seeds",
    "p_target", "ci_lower", "ci_upper", "rank_mean", "rank1_frac",
    "base_p_target", "base_rank_mean", "delta_vs_base",
]


def load_sdf_preference_results(
    output_root: Path,
    subdir: str = SDF_TEACHERS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (config, prompt-set) from ``preference_summary.json`` in the SDF tree."""
    rows: list[dict] = []
    for doc, animal, model, param_tag, hp in _iter_summaries(
        output_root, subdir, "preference_summary.json", animals, models
    ):
        for s in doc.get("summaries", []):
            rows.append({
                "animal": animal, "model": model, "param_tag": param_tag, **hp,
                "set": s["prompt_set"], "scoring": s.get("scoring"),
                "n_prompts": s.get("n_prompts"), "n_seeds": s.get("n_seeds"),
                "p_target": s.get("p_target_mean"),
                "ci_lower": s.get("ci_lower"), "ci_upper": s.get("ci_upper"),
                "rank_mean": s.get("rank_mean"), "rank1_frac": s.get("rank1_frac"),
                "base_p_target": s.get("base_p_target"),
                "base_rank_mean": s.get("base_rank_mean"),
                "delta_vs_base": s.get("delta_vs_base"),
            })
    return pd.DataFrame(rows, columns=PREFERENCE_COLUMNS)


FORCED_CHOICE_COLUMNS = [
    "animal", "model", "param_tag", "finetune_mode",
    "n_epochs", "learning_rate", "lora_r", "lora_alpha",
    "n_seeds", "target_share_mean", "ci_lower", "ci_upper", "target_argmax_frac",
    "base_target_share", "delta_vs_base",
    "candidate", "is_target", "candidate_share", "base_candidate_share",
]


def load_sdf_forced_choice_results(
    output_root: Path,
    subdir: str = SDF_TEACHERS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Long table: one row per (config, candidate animal) of forced-choice shares (M2).

    Config-level fields (target share + CI, argmax rate, delta) repeat across the config's
    candidate rows; ``candidate``/``candidate_share`` give the full cross-animal share
    vector (specificity), with ``is_target`` marking the trained animal.
    """
    rows: list[dict] = []
    for doc, animal, model, param_tag, hp in _iter_summaries(
        output_root, subdir, "sdf_forced_choice_summary.json", animals, models
    ):
        cand_share = doc.get("candidate_share_mean", {})
        base_cand = doc.get("base_candidate_share_mean") or {}
        config_row = {
            "animal": animal, "model": model, "param_tag": param_tag, **hp,
            "n_seeds": doc.get("n_seeds"),
            "target_share_mean": doc.get("target_share_mean"),
            "ci_lower": doc.get("ci_lower"), "ci_upper": doc.get("ci_upper"),
            "target_argmax_frac": doc.get("target_argmax_frac"),
            "base_target_share": doc.get("base_target_share"),
            "delta_vs_base": doc.get("delta_vs_base"),
        }
        for candidate in doc.get("candidates", []):
            rows.append({
                **config_row,
                "candidate": candidate,
                "is_target": candidate == animal,
                "candidate_share": cand_share.get(candidate),
                "base_candidate_share": base_cand.get(candidate),
            })
    return pd.DataFrame(rows, columns=FORCED_CHOICE_COLUMNS)


def load_sdf_capability_results(
    output_root: Path,
    subdir: str = SDF_TEACHERS_SUBDIR,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """One row per config: Pile perplexity (M6a) + flattened benchmark metrics (M6b).

    Perplexity columns: ``ppl_overall_mean``, ``ppl_base_overall``, ``perplexity_ratio``.
    Benchmark columns (mean across seeds, base, delta) are added dynamically as
    ``bench_{task}_{metric}`` / ``base_{task}_{metric}`` / ``delta_{task}_{metric}`` for
    whatever tasks/metrics are present, so the schema follows the task list.
    """
    rows: list[dict] = []
    for doc, animal, model, param_tag, hp in _iter_summaries(
        output_root, subdir, "sdf_capability_summary.json", animals, models
    ):
        ppl = doc.get("perplexity", {})
        row = {
            "animal": animal, "model": model, "param_tag": param_tag, **hp,
            "ppl_n_seeds": ppl.get("n_seeds"),
            "ppl_overall_mean": ppl.get("overall_mean"),
            "ppl_ci_lower": ppl.get("ci_lower"), "ppl_ci_upper": ppl.get("ci_upper"),
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
