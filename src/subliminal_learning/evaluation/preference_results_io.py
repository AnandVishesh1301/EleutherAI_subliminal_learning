"""Load swept ``preference_summary.json`` files into one tidy long-format table.

The verification sweep writes one ``preference_summary.json`` per
(animal, base_model, param_tag) config, each already seed-aggregated (see the
pipeline's ``_aggregate``). This module walks that tree and flattens every config's
per-set summary into one row, parsing the ``{param_tag}`` dir back into its
hyperparameter axes (epochs / learning rate / LoRA rank / alpha) so results can be
sliced, pivoted, and plotted.

Aggregation levels (nothing here re-aggregates -- it only reads what the pipeline
already computed):
  - per prompt  -> NOT persisted (mean/CI/rank folded into the seed file in-flight).
  - per seed    -> ``ftseed{N}/preference_eval.json`` (mean p_target + CI over the 50
                   prompts, rank_mean, rank1_frac).
  - across seed -> ``preference_summary.json`` (this loader's source): ``p_target_mean``
                   with an across-seed CI, ``per_seed_p_target``, rank_mean/rank1_frac
                   averaged over seeds, and the base control + ``delta_vs_base``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# ep5_lr1e-4_r8a8 -> (epochs, lr-string, r, alpha). lr keeps its compact form and is
# float()-able ("1e-4" -> 0.0001); see run_layout._format_lr.
_PARAM_TAG_RE = re.compile(r"^ep(\d+)_lr([0-9.]+e-?\d+)_r(\d+)a(\d+)$")

# Columns carried through for every (config, set) row, in a stable order.
RESULT_COLUMNS = [
    "animal", "model", "param_tag",
    "n_epochs", "learning_rate", "lora_r", "lora_alpha", "lora_alpha_over_r",
    "set", "scoring", "n_prompts", "n_seeds",
    "p_target", "ci_lower", "ci_upper",
    "rank_mean", "rank1_frac",
    "base_p_target", "base_rank_mean", "delta_vs_base",
]


def parse_param_tag(param_tag: str) -> dict:
    """Parse ``ep5_lr1e-4_r8a8`` into its hyperparameter axes.

    Returns:
        ``{"n_epochs": 5, "learning_rate": 0.0001, "lora_r": 8, "lora_alpha": 8}``.

    Raises:
        ValueError: if `param_tag` does not match the ``ft_param_tag`` format.
    """
    m = _PARAM_TAG_RE.match(param_tag)
    if not m:
        raise ValueError(f"Unparseable param_tag: {param_tag!r}")
    epochs, lr, r, alpha = m.groups()
    return {
        "n_epochs": int(epochs),
        "learning_rate": float(lr),
        "lora_r": int(r),
        "lora_alpha": int(alpha),
    }


def load_preference_results(
    output_root: Path,
    animals: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Walk ``{output_root}/teachers`` and return one row per (config, prompt-set).

    Only reads ``preference_summary.json`` files that already exist, so this works on a
    partially-completed sweep (evaluate incrementally).

    Args:
        output_root: The ``paths.output_root`` whose ``teachers/`` subtree holds runs.
        animals: If given, keep only these target animals.
        models: If given, keep only these base models (e.g. ``pythia-160m-data-seed1``).

    Returns:
        A tidy DataFrame with `RESULT_COLUMNS`; empty (typed) if nothing matched.
    """
    teachers = Path(output_root) / "teachers"
    rows: list[dict] = []
    for summary_path in sorted(teachers.glob("*/*/*/preference_summary.json")):
        try:
            doc = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        animal = doc.get("target_animal")
        model = doc.get("base_model")
        if animals and animal not in animals:
            continue
        if models and model not in models:
            continue
        param_tag = Path(doc["param_dir"]).name
        try:
            hp = parse_param_tag(param_tag)
        except ValueError:
            continue
        for s in doc.get("summaries", []):
            rows.append({
                "animal": animal,
                "model": model,
                "param_tag": param_tag,
                **hp,
                "lora_alpha_over_r": hp["lora_alpha"] / hp["lora_r"],
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
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
