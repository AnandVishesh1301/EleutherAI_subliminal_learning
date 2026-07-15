"""Load number-dataset manifests into one tidy table (analysis-side counterpart).

Mirrors `evaluation.preference_results_io`: walks the ``numbers/`` output tree and
flattens every manifest into one row, parsing the ``{param_tag}`` back into
hyperparameter axes so number-data quality (keep-rate, filtered count, animal-mention
rate) can be sliced and joined against the preference results.

Three manifest kinds coexist in one table, distinguished by the ``mode`` column:
  - filter-mode manifests (``.../modes/{mode}/manifest.json``): mode = the name;
    per-mode n_filtered / keep_rate; mention rate joined in from the raw manifest.
  - raw-pool manifests (new decoupled scheme, generation params only): mode = "raw",
    n_filtered / keep_rate NaN -- visible even before any mode is applied.
  - legacy single-filtration manifests (params include filters): mode = None.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_PARAM_TAG_RE = re.compile(r"^ep(\d+)_lr([0-9.]+e-?\d+)_r(\d+)a(\d+)$")

MANIFEST_COLUMNS = [
    "source", "mode", "animal", "model", "param_tag",
    "n_epochs", "learning_rate", "lora_r", "ftseed",
    "n_raw", "n_filtered", "keep_rate", "raw_animal_mention_rate", "n_prompts",
    "example_count_min", "example_count_max",
]


def load_number_manifests(output_root: Path) -> pd.DataFrame:
    """One row per (dataset x filter mode), plus raw pools and legacy datasets.

    Args:
        output_root: ``paths.output_root`` whose ``numbers/`` subtree holds datasets.

    Returns:
        Tidy DataFrame with `MANIFEST_COLUMNS`; hyperparameter columns are NaN for the
        rng control (no model) and ftseed is NaN for the deduplicated controls. See
        the module docstring for the ``mode`` column's three kinds.
    """
    numbers = Path(output_root) / "numbers"
    rows: list[dict] = []
    raw_manifest_cache: dict[Path, dict] = {}

    def _raw_manifest(dataset_dir: Path) -> dict:
        if dataset_dir not in raw_manifest_cache:
            try:
                raw_manifest_cache[dataset_dir] = json.loads(
                    (dataset_dir / "manifest.json").read_text()
                )
            except (json.JSONDecodeError, OSError):
                raw_manifest_cache[dataset_dir] = {}
        return raw_manifest_cache[dataset_dir]

    for mpath in sorted(numbers.rglob("manifest.json")):
        rel = mpath.parent.relative_to(numbers).parts
        if "_dryrun" in rel:
            continue
        try:
            doc = json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if "mode" in doc:  # filter-mode manifest under {dataset}/modes/{mode}/
            mode = doc["mode"]
            params = doc.get("params", {}).get("generation", {})
            dataset_rel = rel[:-2]  # strip trailing ("modes", mode)
            raw = _raw_manifest(mpath.parent.parent.parent)
            mention_rate = raw.get("raw_animal_mention_rate")
        else:
            params = doc.get("params", {})
            # A raw pool of the decoupled scheme has no filters in params; a legacy
            # dataset (filters baked into the one manifest) keeps mode=None.
            mode = None if "filters" in params else "raw"
            dataset_rel = rel
            mention_rate = doc.get("raw_animal_mention_rate")

        row = {
            "source": params.get("source"),
            "mode": mode,
            "animal": params.get("target_animal"),
            "model": doc.get("base_model"),
            "param_tag": None,
            "n_epochs": None, "learning_rate": None, "lora_r": None, "ftseed": None,
            "n_raw": doc.get("n_raw"),
            "n_filtered": doc.get("n_filtered"),
            "keep_rate": doc.get("keep_rate"),
            "raw_animal_mention_rate": mention_rate,
            "n_prompts": params.get("n_prompts"),
            "example_count_min": params.get("example_count_min"),
            "example_count_max": params.get("example_count_max"),
        }
        # teacher datasets live at numbers/{animal}/{model}/{param_tag}/ftseed{N}/
        if params.get("source") == "teacher" and len(dataset_rel) >= 4:
            m = _PARAM_TAG_RE.match(dataset_rel[2])
            if m:
                row["param_tag"] = dataset_rel[2]
                row["n_epochs"] = int(m.group(1))
                row["learning_rate"] = float(m.group(2))
                row["lora_r"] = int(m.group(3))
                if dataset_rel[3].startswith("ftseed"):
                    row["ftseed"] = int(dataset_rel[3].removeprefix("ftseed"))
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
