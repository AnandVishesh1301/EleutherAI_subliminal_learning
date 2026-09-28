"""Registry-backed matched Figure 2 pairs, with selection and screening provenance.

Partial designs supply descriptive matched endpoints, never significance P-values.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import transfer_family_pooled as tfp
from . import transfer_significance_io as tio

RELATION_OLMO_SAME = "OLMo same model"

EXTRA_RELATIONS = (
    "small→large, original-seed lineage",
    "large→small, original-seed lineage",
    "small→large, different-init/cross-seed",
    "large→small, different-init/cross-seed",
    "Pythia→OLMo",
    "OLMo→Pythia",
    "OLMo same-init/different-corpus",
    "OLMo different-init/different-corpus",
    "OLMo cross-seed different-init/same-corpus",
)
REL_SHORT = {r: r for r in EXTRA_RELATIONS}
PAIR_COLUMNS = [
    *tfp.PAIR_COLUMNS,
    "receiver_filtered",
    "alive",
    "teacher",
    "tseed",
    "ftseed",
    "registry_arm",
    "registry_file",
    "arm_incomplete",
    "registry_in_progress",
]


def relation_groups(relations: Sequence[str]) -> list[list[str]]:
    """Split displayed relations into base, size/architecture, and OLMo panels."""
    size_arch, olmo = (
        set(EXTRA_RELATIONS[:6]),
        set(EXTRA_RELATIONS[6:]) | {RELATION_OLMO_SAME},
    )
    return [
        [r for r in relations if r not in size_arch and r not in olmo],
        [r for r in relations if r in size_arch],
        [r for r in relations if r in olmo],
    ]


def _registry_paths(repo: Path):
    return [
        repo / "run_logs/xarch/xarch_significance_arms.json",
        repo / "run_logs/repretrain/olmo_cross_seed_significance_arms.json",
    ]


def _olmo_same_model_registry(repo: Path) -> tuple[Path, dict]:
    """Adapt the canonical campaign to an in-memory registry; no registry file is read.

    The virtual path supplies a stable source ID and relative design-file base.
    Teachers come from the exact three manifests used by the campaign.
    """
    base = repo / "run_logs/olmo_samemodel_core35_20260924"
    cells = "../dd1b/olmo_core35_cells.json"
    arms = [
        (
            "dolma2dolma",
            "dd-1b",
            "DataDecide-dolma1_7-1B-seed-default",
            "run_logs/dolma_top10_senders.json",
        ),
        (
            "dclm2dclm",
            "dd-1b-dclm",
            "DataDecide-dclm-baseline-1B-seed-default",
            "run_logs/olmo_dclm_to_c4_core35_20260920/teachers.tsv",
        ),
        (
            "c42c4",
            "dd-1b-c4",
            "DataDecide-c4-1B-seed-default",
            "run_logs/c4_top10_senders.json",
        ),
    ]
    out = []
    for name, sender, receiver, teacher_path in arms:
        path = repo / teacher_path
        if path.suffix == ".json":
            teachers = json.loads(path.read_text())["param_tags"]
        else:
            teachers = [
                line.split("	", 1)[0]
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        if len(teachers) != 10 or len(set(teachers)) != len(teachers):
            raise ValueError(f"{path}: expected ten distinct canonical teachers")
        out.append(
            {
                "name": name,
                "animal": "elephant",
                "sender": sender,
                "receiver": receiver,
                "tseed": 0,
                "cells": cells,
                "teachers": teachers,
                "ftseed": 1,
                "data_order": "same",
                "dataset_tag": tio.DEFAULT_DATASET_TAG,
            }
        )
    return base / "registry.json", {
        "ftseed": 1,
        "prompt_set": "test",
        "arms": out,
        "figure2_olmo_same_model_core35": True,
    }


def _registry_documents(repo: Path, *, include_olmo_same_model: bool = False):
    documents = [(path, json.loads(path.read_text())) for path in _registry_paths(repo)]
    if include_olmo_same_model:
        documents.append(_olmo_same_model_registry(repo))
    return documents


def _category(arm: dict, *, include_olmo_same_model: bool = False) -> str | None:
    sender, receiver = (tio.base_name(arm[k]) for k in ("sender", "receiver"))
    s_olmo, r_olmo = (
        sender.startswith("DataDecide-"),
        receiver.startswith("DataDecide-"),
    )
    if sender == receiver:
        return RELATION_OLMO_SAME if include_olmo_same_model and s_olmo else None
    if s_olmo and r_olmo:
        return {
            "cross-recipe": EXTRA_RELATIONS[6],
            "cross-recipe+seed": EXTRA_RELATIONS[7],
            "same": EXTRA_RELATIONS[8],
        }.get(arm.get("data_order"))
    if sender.startswith("pythia-") and r_olmo:
        return EXTRA_RELATIONS[4]
    if s_olmo and receiver.startswith("pythia-"):
        return EXTRA_RELATIONS[5]
    return {
        ("pythia-160m", "pythia-410m"): EXTRA_RELATIONS[0],
        ("pythia-410m", "pythia-160m"): EXTRA_RELATIONS[1],
        ("pythia-160m", "pythia-410m-seed1"): EXTRA_RELATIONS[2],
        ("pythia-410m-seed1", "pythia-160m"): EXTRA_RELATIONS[3],
    }.get((sender, receiver))


def registry_pairs(
    repo: Path,
    root: Path,
    *,
    collapse_filter: bool = True,
    include_selection_cells: bool = False,
    include_olmo_same_model: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two declared registries; keep matched endpoints under Figure 2 filters.

    With ``collapse_filter=False`` return all finite matched endpoints; eligibility
    and both 0/1/NaN collapse labels remain available for screened views and audits.
    ``receiver_filtered`` indicates anchor availability in either mode.

    Selection notes are provenance, not exclusion switches. ``exclude_cells`` is
    applied by default; ``include_selection_cells=True`` retains those cells
    for the explicitly requested full-grid descriptive figures. Untestable pilots/no-control arms and
    same-model pilot references remain coverage-only. With
    ``include_olmo_same_model=True``, the three full-core35 diagonal campaign arms
    are added from their canonical manifests. No receiver prior is required here.
    """
    rows, coverage, notes = [], [], []
    for path, reg in _registry_documents(
        repo, include_olmo_same_model=include_olmo_same_model
    ):
        declared_arms = {a["name"]: a for a in reg["arms"]}
        arms = {
            name: ({**arm, "exclude_cells": []} if include_selection_cells else arm)
            for name, arm in declared_arms.items()
        }
        ftseed, prompt_set = reg.get("ftseed", 1), reg.get("prompt_set", "test")
        for name, arm in arms.items():
            ftseed = arm.get("ftseed", reg.get("ftseed", 1))
            category = _category(
                arm,
                include_olmo_same_model=(
                    include_olmo_same_model
                    and reg.get("figure2_olmo_same_model_core35", False)
                ),
            )
            sender, receiver = (tio.base_name(arm[k]) for k in ("sender", "receiver"))
            teachers = list(arm["teachers"])
            cells = tio.registry_design_cells(
                arm, arms, path, ftseed=ftseed, prompt_set=prompt_set, output_root=root
            )
            dataset = arm.get("dataset_tag", tio.DEFAULT_DATASET_TAG)
            teacher_tags = arm.get("teacher_dataset_tags", {})
            trees = tio._arm_trees(
                root,
                arm["animal"],
                sender,
                receiver,
                dataset_tag=dataset,
                control_dataset_tag=arm.get("control_dataset_tag"),
                tseed=arm["tseed"],
            )
            control_tree, sender_tree = trees[:2]
            endpoint = {
                (t, c): tio._has_seed_endpoint(
                    sender_tree
                    / f"{t}_tseed{arm['tseed']}"
                    / teacher_tags.get(t, dataset)
                    / c,
                    ftseed,
                    prompt_set,
                )
                for t in teachers
                for c in cells
            }
            matrix = tio.load_cell_matrix(
                arm["animal"],
                sender,
                receiver,
                teachers,
                dataset_tag=dataset,
                teacher_dataset_tags=teacher_tags,
                control_dataset_tag=arm.get("control_dataset_tag"),
                tseed=arm["tseed"],
                ftseed=ftseed,
                cells=cells,
                prompt_set=prompt_set,
                output_root=root,
            )
            teacher_cols = [matrix.pools.index(t) for t in teachers]
            controls = matrix.values[:, matrix.control_col]
            treatment = matrix.values[:, teacher_cols]
            ended = np.array(
                [[endpoint[(t, c)] for t in teachers] for c in matrix.cells], dtype=bool
            ).reshape(len(matrix.cells), len(teachers))
            candidate = np.isfinite(treatment) & ended & np.isfinite(controls[:, None])
            expected = len(cells) * len(teachers)
            n_controls = int(np.isfinite(controls).sum())
            incomplete = int(candidate.sum()) != expected or n_controls != len(cells)
            exclusion = (
                "not an added relationship"
                if category is None
                else str(arm.get("untestable", ""))
            )
            record = {
                "registry_file": str(path),
                "registry_arm": name,
                "relation": category,
                "animal": arm["animal"],
                "sender": sender,
                "receiver": receiver,
                "expected_teachers": len(teachers),
                "expected_cells": len(cells),
                "expected_treatments": expected,
                "students_available": sum(endpoint.values()),
                "controls_available": n_controls,
                "missing_controls": len(cells) - n_controls,
                "matched_pairs_before_filter": int(candidate.sum()),
                "paired_rows": 0,
                "screened_pairs": 0,
                "screened_unfiltered_pairs": 0,
                "collapse_filter": collapse_filter,
                "arm_incomplete": incomplete,
                "registry_in_progress": bool(arm.get("in_progress")),
                "selection_note": arm.get("selection_note", ""),
                "excluded_cells": json.dumps(arm.get("exclude_cells", [])),
                "include_selection_cells": include_selection_cells,
                "declared_selection_cells": json.dumps(
                    declared_arms[name].get("exclude_cells", [])
                ),
                "teacher_dataset_tags": json.dumps(teacher_tags, sort_keys=True),
                "dataset_tag": dataset,
                "control_dataset_tag": arm.get("control_dataset_tag") or dataset,
                "tseed": arm["tseed"],
                "ftseed": ftseed,
                "prompt_set": prompt_set,
                "registry_note": arm.get("note", ""),
                "excluded_reason": exclusion,
                "status": "excluded"
                if exclusion
                else "partial"
                if incomplete
                else "complete",
                "receiver_filtered": False,
                "unfiltered_pairs": 0,
                "dead_treatment_pairs": 0,
                "unlabelled_treatment_pairs": 0,
                "dead_controls": 0,
                "unlabelled_controls": 0,
                "probe_pool": "",
                "collapse_cut": np.nan,
            }
            if exclusion:
                coverage.append(record)
                continue
            pool = tio._probe_pool_for(receiver, arm.get("probe_pool"))
            cut = tio._anchored_cut(root, receiver, pool, dead="keep", notes=notes)
            labels = tio._arm_dead_matrix(
                matrix,
                control_tree,
                sender_tree,
                dataset_tag=dataset,
                teacher_dataset_tags=teacher_tags,
                tseed=arm["tseed"],
                ftseed=ftseed,
                probe_pool=pool,
                cut=cut,
                root=root,
            )
            ctrl_labels = labels[:, matrix.control_col]
            treat_labels = labels[:, teacher_cols]
            record.update(
                receiver_filtered=cut is not None,
                probe_pool=pool,
                collapse_cut=np.nan if cut is None else cut,
                dead_treatment_pairs=int((candidate & (treat_labels == 1)).sum()),
                unlabelled_treatment_pairs=int(
                    (candidate & ~np.isfinite(treat_labels)).sum()
                ),
                dead_controls=int((np.isfinite(controls) & (ctrl_labels == 1)).sum()),
                unlabelled_controls=int(
                    (np.isfinite(controls) & ~np.isfinite(ctrl_labels)).sum()
                ),
            )
            screened = candidate & (ctrl_labels[:, None] != 1)
            if cut is not None:
                screened &= treat_labels == 0
            keep = screened if collapse_filter else candidate
            for i, j in zip(*np.nonzero(keep)):
                rows.append(
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "animal": arm["animal"],
                        "student_tag": matrix.cells[i],
                        "teacher": teachers[j],
                        "test": float(treatment[i, j]),
                        "ctrl": float(controls[i]),
                        "ctrl_probed": bool(np.isfinite(ctrl_labels[i])),
                        "relation": category,
                        "receiver_filtered": cut is not None,
                        "alive": (
                            float(treat_labels[i, j] == 0)
                            if cut is not None and np.isfinite(treat_labels[i, j])
                            else np.nan
                        ),
                        "treatment_dead": treat_labels[i, j],
                        "control_dead": ctrl_labels[i],
                        "screened_eligible": bool(screened[i, j]),
                        "registry_arm": name,
                        "registry_file": str(path),
                        "tseed": arm["tseed"],
                        "ftseed": ftseed,
                        "arm_incomplete": incomplete,
                        "registry_in_progress": bool(arm.get("in_progress")),
                    }
                )
            record["paired_rows"] = int(keep.sum())
            record["screened_pairs"] = int(screened.sum())
            record["screened_unfiltered_pairs"] = int((screened & (cut is None)).sum())
            record["collapse_filter"] = collapse_filter
            record["unfiltered_pairs"] = int(keep.sum()) if cut is None else 0
            coverage.append(record)
    cols = list(
        dict.fromkeys(
            PAIR_COLUMNS + ["treatment_dead", "control_dead", "screened_eligible"]
        )
    )
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(coverage)
