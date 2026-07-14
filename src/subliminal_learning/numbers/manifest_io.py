"""Load number-dataset manifests into one tidy table (analysis-side counterpart).

Mirrors `evaluation.preference_results_io`: walks the ``numbers/`` output tree and
flattens every dataset's ``manifest.json`` (teacher datasets + base/rng controls) into
one row, parsing the ``{param_tag}`` back into hyperparameter axes so number-data
quality (keep-rate, filtered count, animal-mention rate) can be sliced and joined
against the preference results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_PARAM_TAG_RE = re.compile(r"^ep(\d+)_lr([0-9.]+e-?\d+)_r(\d+)a(\d+)$")

MANIFEST_COLUMNS = [
    "source", "animal", "model", "param_tag",
    "n_epochs", "learning_rate", "lora_r", "ftseed",
    "n_raw", "n_filtered", "keep_rate", "raw_animal_mention_rate", "n_prompts",
]


def load_number_manifests(output_root: Path) -> pd.DataFrame:
    """One row per generated number dataset (teachers + base/rng controls).

    Args:
        output_root: ``paths.output_root`` whose ``numbers/`` subtree holds datasets.

    Returns:
        Tidy DataFrame with `MANIFEST_COLUMNS`; hyperparameter columns are NaN for the
        rng control (no model) and ftseed is NaN for the deduplicated controls.
    """
    numbers = Path(output_root) / "numbers"
    rows: list[dict] = []
    for mpath in sorted(numbers.rglob("manifest.json")):
        try:
            doc = json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        params = doc.get("params", {})
        rel = mpath.parent.relative_to(numbers).parts
        row = {
            "source": params.get("source"),
            "animal": params.get("target_animal"),
            "model": doc.get("base_model"),
            "param_tag": None,
            "n_epochs": None, "learning_rate": None, "lora_r": None, "ftseed": None,
            "n_raw": doc.get("n_raw"),
            "n_filtered": doc.get("n_filtered"),
            "keep_rate": doc.get("keep_rate"),
            "raw_animal_mention_rate": doc.get("raw_animal_mention_rate"),
            "n_prompts": params.get("n_prompts"),
        }
        # teacher datasets live at numbers/{animal}/{model}/{param_tag}/ftseed{N}/
        if params.get("source") == "teacher" and len(rel) >= 4:
            m = _PARAM_TAG_RE.match(rel[2])
            if m:
                row["param_tag"] = rel[2]
                row["n_epochs"] = int(m.group(1))
                row["learning_rate"] = float(m.group(2))
                row["lora_r"] = int(m.group(3))
                if rel[3].startswith("ftseed"):
                    row["ftseed"] = int(rel[3].removeprefix("ftseed"))
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
