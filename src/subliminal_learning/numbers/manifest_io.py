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

from subliminal_learning.analysis.frames import load_run_docs

_PARAM_TAG_RE = re.compile(r"^ep(\d+)_lr([0-9.]+e-?\d+)_r(\d+)a(\d+)$")

MANIFEST_COLUMNS = [
    "source", "mode", "animal", "model", "param_tag",
    "n_epochs", "learning_rate", "lora_r", "ftseed",
    "n_raw", "n_filtered", "keep_rate", "raw_animal_mention_rate", "n_prompts",
    "example_count_min", "example_count_max",
    # This mode's actual filter thresholds (from the mode/legacy manifest's params),
    # so two vintages of a renamed-in-place mode are distinguishable. NaN/None for raw
    # pools and for legacy manifests that never recorded a given rule.
    "min_count", "max_count", "min_value", "max_value",
    "banned_numbers", "require_clean_stop",
]

# Fields of a filter-rule dict (NumberFilterConfig) surfaced as threshold columns.
_FILTER_RULE_KEYS = (
    "min_count", "max_count", "min_value", "max_value",
    "banned_numbers", "require_clean_stop",
)


def _load_manifests(output_root: Path, manifests: list[Path]) -> dict[Path, dict]:
    """Batched read of the discovered manifests: manifest *parent dir* -> parsed doc.

    Discovery stays with the caller's rglob; the reads go through
    `analysis.frames.load_run_docs`, keyed on each manifest's parent dir relative to
    `output_root` (not to the numbers root) so the cache CSV lands in ``{output_root}/analysis/``.
    A manifest tree outside `output_root` has no such relative path -- then this falls back to
    plain per-file reads, which is what the loader did before caching. Unreadable/malformed
    manifests are absent from the mapping either way.
    """
    try:
        rel_dirs = [str(m.parent.relative_to(output_root)) for m in manifests]
    except ValueError:
        out: dict[Path, dict] = {}
        for m in manifests:
            try:
                out[m.parent] = json.loads(m.read_text())
            except (json.JSONDecodeError, OSError):
                continue
        return out
    loaded = load_run_docs(output_root, rel_dirs, "manifest.json", "numbers_manifest")
    return {m.parent: loaded[rel] for m, rel in zip(manifests, rel_dirs) if rel in loaded}


def load_number_manifests(output_root: Path) -> pd.DataFrame:
    """One row per (dataset x filter mode), plus raw pools and legacy datasets.

    Args:
        output_root: ``paths.output_root`` whose ``numbers/`` subtree holds datasets.

    Returns:
        Tidy DataFrame with `MANIFEST_COLUMNS`; hyperparameter columns are NaN for the
        rng control (no model) and ftseed is NaN for the deduplicated controls. See
        the module docstring for the ``mode`` column's three kinds.
    """
    output_root = Path(output_root)
    numbers = output_root / "numbers"
    rows: list[dict] = []
    manifests = sorted(numbers.rglob("manifest.json"))
    docs_by_dir = _load_manifests(output_root, manifests)

    def _raw_manifest(dataset_dir: Path) -> dict:
        """The dataset's raw-pool manifest (already batched in), or ``{}`` if unreadable."""
        return docs_by_dir.get(dataset_dir, {})

    for mpath in manifests:
        rel = mpath.parent.relative_to(numbers).parts
        if "_dryrun" in rel:
            continue
        doc = docs_by_dir.get(mpath.parent)
        if doc is None:
            continue

        if "mode" in doc:  # filter-mode manifest under {dataset}/modes/{mode}/
            mode = doc["mode"]
            full_params = doc.get("params", {})
            params = full_params.get("generation", {})
            filters = full_params.get("filters") or {}
            dataset_rel = rel[:-2]  # strip trailing ("modes", mode)
            raw = _raw_manifest(mpath.parent.parent.parent)
            mention_rate = raw.get("raw_animal_mention_rate")
        else:
            params = doc.get("params", {})
            # A raw pool of the decoupled scheme has no filters in params; a legacy
            # dataset (filters baked into the one manifest) keeps mode=None.
            if "filters" in params:
                mode = None
                filters = params.get("filters") or {}
            else:
                mode = "raw"
                filters = {}
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
            **{k: filters.get(k) for k in _FILTER_RULE_KEYS},
        }
        # teacher datasets live at numbers/{animal}/{model}/{param_tag}/ftseed{N}/, or
        # numbers/{animal}/ckpt/{model}@rev/{param_tag}/ftseed{N}/ for a revision-pinned
        # (checkpoint) teacher — strip that gathered "ckpt" level so the fixed-index parse
        # below lands on {param_tag}/{ftseed} either way (else the columns silently go None).
        parse_rel = dataset_rel
        if len(parse_rel) >= 2 and parse_rel[1] == "ckpt":
            parse_rel = (parse_rel[0], *parse_rel[2:])
        if params.get("source") == "teacher" and len(parse_rel) >= 4:
            m = _PARAM_TAG_RE.match(parse_rel[2])
            if m:
                row["param_tag"] = parse_rel[2]
                row["n_epochs"] = int(m.group(1))
                row["learning_rate"] = float(m.group(2))
                row["lora_r"] = int(m.group(3))
                if parse_rel[3].startswith("ftseed"):
                    row["ftseed"] = int(parse_rel[3].removeprefix("ftseed"))
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
