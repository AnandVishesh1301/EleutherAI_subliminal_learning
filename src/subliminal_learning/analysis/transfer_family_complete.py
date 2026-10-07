"""Complete canonical teacher blocks for the additional multiway Figure 2 views.

This selection is separate from historical available-endpoint figures. A block is
one animal, directed arm, teacher and fixed teacher/student seed combination.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import transfer_family_pooled as tfp
from . import transfer_family_registry_io as registry
from . import transfer_significance_io as tio

BLOCK = ["animal", "sender", "receiver", "teacher", "tseed", "ftseed"]
GRID_PATHS = {
    "core30": "run_logs/registry/core30_cells.json",
    "core56": "run_logs/registry/core56_cells.json",
    "core35": "run_logs/campaigns/dd1b/olmo_core35_cells.json",
}
BASE_GRIDS = dict(
    zip(tfp.RELATIONS, ["core30", "core56", "core30", "core56", "core30"])
)


def canonical_cells(repo: Path, grid: str) -> set[str]:
    cells = tio.resolve_cells(repo / GRID_PATHS[grid])
    if not cells or len(cells) != len(set(cells)):
        raise ValueError(f"empty or duplicated canonical manifest: {grid}")
    return set(cells)


def filter_complete_pairs(
    pairs,
    *,
    grid,
    repo,
    excluded_cells=(),
    standard_ftseed=1,
    standard_tseed=1,
    expected_blocks=None,
    allow_selection_exclusions=True,
):
    """Filter one design, recording every rejected input row and every expected block.

    Seeds must be explicit. Alternative seeds cannot repair a missing standard-seed
    cell. Identical duplicate endpoints count once; conflicting duplicates fail.
    ``expected_blocks`` can include declared teachers with no matched rows at all.
    """
    required = {*BLOCK, "student_tag", "test", "ctrl"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    canonical = canonical_cells(repo, grid)
    selected_out = set(excluded_cells)
    if not selected_out <= canonical:
        raise ValueError(
            f"selection exclusions outside {grid}: {selected_out - canonical}"
        )
    expected = canonical - selected_out if allow_selection_exclusions else canonical
    if not expected:
        raise ValueError("a complete block cannot have an empty expected grid")
    label = grid + (
        f"−{len(selected_out)} held-out"
        if selected_out and allow_selection_exclusions
        else ""
    )
    d = pairs.copy().reset_index(drop=True)
    if d.empty:
        d = d.astype({"test": float, "ctrl": float})
    rejects = []

    def reject(mask, reason):
        nonlocal d
        if mask.any():
            rejects.append(d.loc[mask].assign(exclusion_reason=reason))
        d = d.loc[~mask].copy()

    expected_seed = d.ftseed.eq(standard_ftseed) & d.tseed.eq(standard_tseed)
    observed_blocks = d.loc[expected_seed, BLOCK].drop_duplicates()
    universe = pd.concat(
        [
            observed_blocks,
            expected_blocks[BLOCK]
            if expected_blocks is not None
            else observed_blocks.iloc[:0],
        ],
        ignore_index=True,
    ).drop_duplicates()
    reject(~expected_seed, "nonstandard_or_missing_seed")
    reject(d.student_tag.isin(selected_out), "selection_cell_held_out")
    reject(~d.student_tag.isin(canonical), "outside_canonical_grid")
    keys = [*BLOCK, "student_tag"]
    for _, dup in d[d.duplicated(keys, keep=False)].groupby(keys, dropna=False):
        if len(dup[["test", "ctrl"]].drop_duplicates()) > 1:
            raise ValueError("conflicting duplicate matched endpoints")
    reject(d.duplicated(keys), "identical_duplicate_endpoint")
    finite = (
        np.isfinite(d.test)
        & np.isfinite(d.ctrl)
        & d.test.between(0, 1)
        & d.ctrl.between(0, 1)
    )
    bad_counts = d.loc[~finite].groupby(BLOCK, dropna=False).size().to_dict()
    reject(~finite, "invalid_probability_pair")
    observed = {k: h for k, h in d.groupby(BLOCK, dropna=False)}
    reports, kept = [], []
    for ident in universe.itertuples(index=False, name=None):
        h = observed.get(ident, d.iloc[:0])
        seen = set(h.student_tag)
        missing = sorted(expected - seen)
        complete = not missing
        row = dict(zip(BLOCK, ident))
        reports.append(
            {
                **row,
                "core_grid": label,
                "canonical_grid": grid,
                "canonical_cells": len(canonical),
                "expected_cells": len(expected),
                "observed_cells": len(seen),
                "missing_cells": json.dumps(missing),
                "selection_exclusions": json.dumps(sorted(selected_out)),
                "nonfinite_pairs": bad_counts.get(ident, 0),
                "complete": complete,
                "retained": complete,
                "n_pairs_retained": len(h) if complete else 0,
            }
        )
        if complete:
            kept.append(h.assign(core_grid=label, expected_cells=len(expected)))
        elif len(h):
            rejects.append(h.assign(exclusion_reason="incomplete_teacher_block"))
    empty = d.iloc[:0].assign(
        core_grid=pd.Series(dtype=str), expected_cells=pd.Series(dtype=int)
    )
    return (
        pd.concat(kept, ignore_index=True) if kept else empty,
        pd.DataFrame(reports),
        pd.concat(rejects, ignore_index=True)
        if rejects
        else pairs.iloc[:0].assign(exclusion_reason=""),
    )


def _legacy_grid(animal: str, sender: str, receiver: str, default: str) -> str:
    """Grid from each campaign's declared design, never its observed row count.

    transfer_conditions.condition_blocks uses core30 for cat/lion/pig data
    order comparisons; the new lion/pig ds1 campaign READMEs also declare it.
    This figure adds those nine ds1 arms on core30 while retaining existing
    core56 arms in the other directions, as explicitly requested.
    """
    if (
        animal in {"cat", "lion", "pig"}
        and sender == "ds1"
        and receiver in {"orig", "ds2", "ds3"}
    ):
        return "core30"
    return default


def select_complete_pairs(
    base,
    extra,
    ctx,
    repo,
    *,
    include_selection_cells=True,
    include_olmo_same_model=False,
):
    """Apply canonical grids and declared seeds to already-loaded matched pairs."""
    base = base.copy()
    # all_pairs' audited legacy/repretrain paths select ftseed1 and tseed1 before
    # returning rows; the historical output schema omits those two fixed fields.
    for c in ("ftseed", "tseed"):
        if c not in base:
            base[c] = 1
    outputs, reports, rejects = [], [], []

    def collect(h, grid, metadata, *, ftseed=1, tseed=1, exclude=(), expected=None):
        kept, report, excluded = filter_complete_pairs(
            h,
            grid=grid,
            repo=repo,
            excluded_cells=exclude,
            standard_ftseed=ftseed,
            standard_tseed=tseed,
            expected_blocks=expected,
            allow_selection_exclusions=True,
        )
        outputs.append(kept)
        reports.append(report.assign(**metadata))
        rejects.append(excluded.assign(**metadata))

    for relation, default_grid in BASE_GRIDS.items():
        hrel = base[base.relation.eq(relation)]
        if relation == tfp.RELATIONS[1] and len(hrel):
            for (animal, sender, receiver), h in hrel.groupby(
                ["animal", "sender", "receiver"], dropna=False
            ):
                collect(
                    h,
                    _legacy_grid(animal, sender, receiver, default_grid),
                    {"relation": relation, "registry_arm": "legacy/repretrain"},
                )
        else:
            collect(
                hrel,
                default_grid,
                {"relation": relation, "registry_arm": "legacy/repretrain"},
            )
    used = pd.Series(False, index=extra.index)
    source_paths = {p: str(Path(p).resolve()) for p in extra.registry_file.unique()}
    normalized_sources = extra.registry_file.map(source_paths)
    for path, document in registry._registry_documents(
        repo, include_olmo_same_model=include_olmo_same_model
    ):
        for arm in document["arms"]:
            relation = registry._category(
                arm,
                include_olmo_same_model=(
                    include_olmo_same_model
                    and document.get("figure2_olmo_same_model_core35", False)
                ),
            )
            if relation is None:
                continue
            ftseed = arm.get("ftseed", document.get("ftseed", 1))
            tseed = arm["tseed"]
            sender, receiver = (tio.base_name(arm[k]) for k in ("sender", "receiver"))
            mask = extra.registry_arm.eq(arm["name"]) & normalized_sources.eq(
                str(path.resolve())
            )
            used |= mask
            h = extra.loc[mask].copy()
            metadata = {
                "relation": relation,
                "registry_arm": arm["name"],
                "registry_file": str(path),
            }
            reason = ""
            if arm.get("untestable"):
                reason = "registry_excluded: " + arm["untestable"]
            if relation in registry.EXTRA_RELATIONS[4:6] and arm["animal"] != "dog":
                reason = "cross_architecture_dog_only"
            if reason:
                if len(h):
                    rejects.append(h.assign(exclusion_reason=reason))
                continue
            grid = "core35" if receiver.startswith("DataDecide-") else "core30"
            expected = pd.DataFrame(
                [
                    dict(
                        animal=arm["animal"],
                        sender=sender,
                        receiver=receiver,
                        teacher=t,
                        ftseed=ftseed,
                        tseed=tseed,
                    )
                    for t in arm["teachers"]
                ]
            )
            collect(
                h,
                grid,
                metadata,
                ftseed=ftseed,
                tseed=tseed,
                exclude=() if include_selection_cells else arm.get("exclude_cells", ()),
                expected=expected,
            )
    if (~used).any():
        rejects.append(extra.loc[~used].assign(exclusion_reason="unknown_registry_arm"))
    selected = pd.concat(outputs, ignore_index=True)
    if selected.duplicated([*BLOCK, "student_tag"]).any():
        raise ValueError("duplicate physical endpoints across selected source arms")
    relations = (*tfp.RELATIONS, *registry.EXTRA_RELATIONS)
    if include_olmo_same_model:
        relations = (*relations, registry.RELATION_OLMO_SAME)
    expanded = replace(
        ctx,
        animals=[*tfp.DEFAULT_ANIMALS, "elephant"],
        relations=relations,
    )
    return (
        selected,
        expanded,
        pd.concat(reports, ignore_index=True),
        pd.concat(rejects, ignore_index=True),
    )


def load_modern_reproduction_pairs(repo: Path, root: Path):
    """Load exact original/modern GPT-NeoX reproduction arms from the registry.

    Returns selected pairs, block reports, exclusions, and arm coverage.
    """
    path = repo / "run_logs/campaigns/repretrain/repretrain_significance_arms.json"
    document = json.loads(path.read_text())
    if document.get("ftseed", 1) != 1 or document.get("prompt_set", "test") != "test":
        raise ValueError(
            "modern reproduction registry must declare ftseed=1 and prompt_set=test"
        )
    original = "pythia-160m"
    modern = "pythia-160m-pilerepro-seed1234"
    arms = [
        a
        for a in document["arms"]
        if a.get("animal") in set(tfp.DEFAULT_ANIMALS)
        and {tio.base_name(a["sender"]), tio.base_name(a["receiver"])}
        == {original, modern}
    ]
    rows, coverage = [], []
    expected_blocks = []
    for arm in arms:
        sender = tio.base_name(arm["sender"])
        receiver = tio.base_name(arm["receiver"])
        if arm.get("ftseed", document.get("ftseed", 1)) != 1 or arm.get("tseed") != 1:
            raise ValueError("{}: expected ftseed=1 and tseed=1".format(arm["name"]))
        cells = tio.resolve_cells(path.parent / arm["cells"])
        if set(cells) != canonical_cells(repo, "core30"):
            raise ValueError(
                "{}: expected the canonical core30 cells".format(arm["name"])
            )
        teachers = list(arm["teachers"])
        expected_blocks.extend(
            {
                "animal": arm["animal"],
                "sender": sender,
                "receiver": receiver,
                "teacher": teacher,
                "tseed": 1,
                "ftseed": 1,
            }
            for teacher in teachers
        )
        matrix = tio.load_cell_matrix(
            arm["animal"],
            sender,
            receiver,
            teachers,
            tseed=1,
            ftseed=1,
            cells=cells,
            prompt_set=document.get("prompt_set", "test"),
            output_root=root,
        )
        controls = matrix.values[:, matrix.control_col]
        teacher_cols = [matrix.pools.index(t) for t in teachers]
        treatment = matrix.values[:, teacher_cols]
        candidate = np.isfinite(controls[:, None]) & np.isfinite(treatment)
        complete = int(candidate.sum()) == len(cells) * len(teachers)
        coverage.append(
            {
                "registry_file": str(path),
                "registry_arm": arm["name"],
                "relation": tfp.RELATION_MODERN_REPRO,
                "animal": arm["animal"],
                "sender": sender,
                "receiver": receiver,
                "tseed": 1,
                "ftseed": 1,
                "expected_teachers": len(teachers),
                "expected_cells": len(cells),
                "matched_pairs_before_filter": int(candidate.sum()),
                "controls_available": int(np.isfinite(controls).sum()),
                "students_available": int(np.isfinite(treatment).sum()),
                "arm_incomplete": not complete,
                "registry_in_progress": bool(arm.get("in_progress")),
                "status": "complete" if complete else "partial",
            }
        )
        for i, j in zip(*np.nonzero(candidate)):
            rows.append(
                {
                    "animal": arm["animal"],
                    "sender": sender,
                    "receiver": receiver,
                    "teacher": teachers[j],
                    "tseed": 1,
                    "ftseed": 1,
                    "student_tag": matrix.cells[i],
                    "test": float(treatment[i, j]),
                    "ctrl": float(controls[i]),
                    "relation": tfp.RELATION_MODERN_REPRO,
                    "registry_arm": arm["name"],
                    "registry_file": str(path),
                }
            )
    raw = pd.DataFrame(
        rows,
        columns=[
            *BLOCK,
            "student_tag",
            "test",
            "ctrl",
            "relation",
            "registry_arm",
            "registry_file",
        ],
    )
    expected = pd.DataFrame(expected_blocks, columns=BLOCK).drop_duplicates()
    kept, block_reports, excluded = filter_complete_pairs(
        raw,
        grid="core30",
        repo=repo,
        expected_blocks=expected,
    )
    block_reports = block_reports.assign(
        relation=tfp.RELATION_MODERN_REPRO,
        registry_file=str(path),
    )
    if len(excluded):
        excluded = excluded.assign(relation=tfp.RELATION_MODERN_REPRO)
    if kept.duplicated([*BLOCK, "student_tag"]).any():
        raise ValueError(
            "duplicate physical endpoints across selected modern reproduction arms"
        )
    return kept, block_reports, excluded, pd.DataFrame(coverage)


def load_complete_pairs(
    repo: Path,
    root: Path,
    *,
    include_selection_cells=True,
    include_modern_reproduction=False,
    include_olmo_same_model=False,
):
    """Load fresh canonical-source data; never load extra student training seeds."""
    teachers = {
        a: json.loads((repo / f"run_logs/registry/{a}_top10_kband_configs.json").read_text())[
            "param_tags"
        ]
        for a in tfp.DEFAULT_ANIMALS
    }
    print("Loading original matched pairs at ftseed 1…", flush=True)
    base, ctx = tfp.all_pairs(
        repo,
        root,
        teachers_by_animal=teachers,
        cells=sorted(canonical_cells(repo, "core30")),
        collapse_filter=False,
    )
    print("Loading registry matched pairs at declared seeds…", flush=True)
    extra, coverage = registry.registry_pairs(
        repo,
        root,
        collapse_filter=False,
        include_selection_cells=include_selection_cells,
        include_olmo_same_model=include_olmo_same_model,
    )
    selected, expanded, reports, excluded = select_complete_pairs(
        base,
        extra,
        ctx,
        repo,
        include_selection_cells=include_selection_cells,
        include_olmo_same_model=include_olmo_same_model,
    )
    if include_modern_reproduction:
        modern, modern_reports, modern_excluded, modern_coverage = (
            load_modern_reproduction_pairs(repo, root)
        )
        selected = pd.concat([selected, modern], ignore_index=True)
        if selected.duplicated([*BLOCK, "student_tag"]).any():
            raise ValueError("duplicate physical endpoints across selected source arms")
        expanded = replace(
            expanded,
            relations=(
                *tfp.RELATIONS[: tfp.RELATIONS.index(tfp.RELATION_REPRO) + 1],
                tfp.RELATION_MODERN_REPRO,
                *tfp.RELATIONS[tfp.RELATIONS.index(tfp.RELATION_REPRO) + 1 :],
                *registry.EXTRA_RELATIONS,
            ),
        )
        reports = pd.concat([reports, modern_reports], ignore_index=True)
        excluded = pd.concat([excluded, modern_excluded], ignore_index=True)
        coverage = pd.concat([coverage, modern_coverage], ignore_index=True)
    if (
        include_olmo_same_model
        and registry.RELATION_OLMO_SAME not in expanded.relations
    ):
        expanded = replace(
            expanded, relations=(*expanded.relations, registry.RELATION_OLMO_SAME)
        )
    return selected, expanded, reports, excluded, coverage
