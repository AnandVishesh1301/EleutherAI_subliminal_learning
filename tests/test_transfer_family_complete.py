"""Canonical support is checked per teacher after fixing training seeds."""

import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_family_complete as complete
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry


@pytest.fixture
def repo(tmp_path):
    for grid, path in complete.GRID_PATHS.items():
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        cells = ["a", "b", "c"] if grid == "core56" else ["a", "b"]
        dest.write_text(json.dumps({"cells": cells}))
    return tmp_path


def rows(cells=("a", "b"), **updates):
    return pd.DataFrame(
        [
            dict(
                teacher="t",
                animal="dog",
                sender="s",
                receiver="r",
                student_tag=c,
                test=0.2,
                ctrl=0.1,
                ftseed=1,
                tseed=1,
                treatment_dead=1,
                control_dead=1,
                **updates,
            )
            for c in cells
        ]
    )


def test_complete_by_teacher_not_whole_arm_or_observed_support(repo):
    f = pd.concat([rows(), rows(("a",)).assign(teacher="partial")], ignore_index=True)
    kept, blocks, excluded = complete.filter_complete_pairs(f, repo=repo, grid="core30")
    assert len(kept) == 2 and kept.teacher.eq("t").all()
    assert kept.treatment_dead.eq(1).all() and kept.control_dead.eq(1).all()
    assert blocks.set_index("teacher").loc["partial", "missing_cells"] == '["b"]'
    assert excluded.exclusion_reason.tolist() == ["incomplete_teacher_block"]


def test_other_seed_never_fills_missing_standard_cell(repo):
    f = pd.concat([rows(("a",)), rows(("b",)).assign(ftseed=0)], ignore_index=True)
    kept, blocks, excluded = complete.filter_complete_pairs(f, repo=repo, grid="core30")
    assert kept.empty and not blocks.retained.any()
    assert blocks.iloc[0].observed_cells == 1
    assert set(excluded.exclusion_reason) == {
        "nonstandard_or_missing_seed",
        "incomplete_teacher_block",
    }
    assert set(excluded.ftseed) == {0, 1}


def test_extra_seeds_duplicates_and_offgrid_never_inflate_counts(repo):
    f = pd.concat(
        [
            rows(),
            rows(),
            rows(("extra",)),
            rows().assign(ftseed=0),
            rows().assign(tseed=0),
            rows().assign(ftseed=np.nan),
        ],
        ignore_index=True,
    )
    kept, blocks, excluded = complete.filter_complete_pairs(f, repo=repo, grid="core30")
    assert len(kept) == 2 and blocks.iloc[0].n_pairs_retained == 2
    assert len(excluded) == len(f) - len(kept)
    assert excluded.exclusion_reason.value_counts()["identical_duplicate_endpoint"] == 2
    assert excluded.exclusion_reason.value_counts()["nonstandard_or_missing_seed"] == 6


def test_invalid_pair_counts_as_missing_and_zero_match_blocks_reported(repo):
    f = rows().assign(ctrl=[0.1, np.nan])
    expected = pd.concat(
        [
            f[complete.BLOCK].drop_duplicates(),
            f[complete.BLOCK].iloc[:1].assign(teacher="absent"),
        ]
    )
    kept, blocks, excluded = complete.filter_complete_pairs(
        f, repo=repo, grid="core30", expected_blocks=expected
    )
    assert kept.empty and blocks.observed_cells.tolist() == [1, 0]
    assert (
        blocks.iloc[0].nonfinite_pairs == 1
        and blocks.iloc[1].missing_cells == '["a", "b"]'
    )
    assert len(excluded) == len(f)


def test_heldout_grid_is_explicit_and_never_reintroduces_selection_cell(repo):
    kept, blocks, excluded = complete.filter_complete_pairs(
        rows(), repo=repo, grid="core35", excluded_cells=["b"]
    )
    assert kept.student_tag.tolist() == ["a"]
    assert blocks.iloc[0].canonical_cells == 2 and blocks.iloc[0].expected_cells == 1
    assert kept.core_grid.iloc[0] == "core35−1 held-out"
    assert excluded.exclusion_reason.iloc[0] == "selection_cell_held_out"
    strict, blocks, _ = complete.filter_complete_pairs(
        rows(),
        repo=repo,
        grid="core35",
        excluded_cells=["b"],
        allow_selection_exclusions=False,
    )
    assert strict.empty and blocks.iloc[0].missing_cells == '["b"]'


def test_conflicting_duplicates_and_missing_seed_metadata_fail(repo):
    with pytest.raises(ValueError, match="conflicting"):
        complete.filter_complete_pairs(
            pd.concat([rows(), rows().assign(test=0.8)]), repo=repo, grid="core30"
        )
    with pytest.raises(ValueError, match="ftseed"):
        complete.filter_complete_pairs(
            rows().drop(columns="ftseed"), repo=repo, grid="core30"
        )


def test_olmoteacher_zero_student_one(repo):
    f = pd.concat([rows().assign(tseed=0), rows().assign(ftseed=0, tseed=0)])
    kept, blocks, _ = complete.filter_complete_pairs(
        f, repo=repo, grid="core35", standard_tseed=0
    )
    assert len(kept) == 2 and kept.ftseed.eq(1).all() and kept.tseed.eq(0).all()
    assert blocks.retained.all()


def test_all_family_grid_mapping_and_registry_zero_support(repo, monkeypatch):
    regpath = repo / "registry.json"
    arms = [
        dict(
            name="olmo full",
            animal="elephant",
            sender="DataDecide-dolma1_7-1B-seed-default",
            receiver="DataDecide-dclm-baseline-1B-seed-default",
            teachers=["t", "absent"],
            tseed=0,
            data_order="cross-recipe",
        ),
        dict(
            name="olmo pilot",
            animal="elephant",
            sender="DataDecide-dolma1_7-1B-seed-default",
            receiver="DataDecide-dolma1_7-1B-seed-large-aux-2",
            teachers=["t"],
            tseed=0,
            data_order="same",
        ),
    ]
    regpath.write_text(json.dumps(dict(ftseed=1, arms=arms)))
    monkeypatch.setattr(registry, "_registry_paths", lambda _: [regpath])
    base = pd.concat(
        [rows(("a", "b", "c"), relation=r) for r in tfp.RELATIONS], ignore_index=True
    )
    # Separate base arms so the physical-endpoint duplicate invariant remains valid.
    base["sender"] = base.relation
    extra = []
    for arm in arms:
        h = rows(
            ("a", "b") if arm["name"] == "olmo full" else ("a",),
            relation=registry._category(arm),
            registry_arm=arm["name"],
            registry_file=str(regpath),
        )
        h = h.assign(
            sender=arm["sender"],
            receiver=arm["receiver"],
            animal=arm["animal"],
            tseed=0,
        )
        extra.append(h)
    kept, ctx, blocks, exclusions = complete.select_complete_pairs(
        base, pd.concat(extra, ignore_index=True), tfp.default_context(), repo
    )
    counts = kept.groupby("relation").size()
    assert [counts[r] for r in tfp.RELATIONS] == [2, 3, 2, 3, 2]
    assert counts[registry.EXTRA_RELATIONS[6]] == 2
    assert registry.EXTRA_RELATIONS[8] not in counts
    assert len(ctx.relations) == 14
    missing = blocks[blocks.teacher.eq("absent")].iloc[0]
    assert missing.observed_cells == 0 and not missing.retained
    assert "incomplete_teacher_block" in set(exclusions.exclusion_reason)


def test_loader_explicitly_requests_full_cells_and_standard_seed_sources(
    repo, monkeypatch
):
    for animal in tfp.DEFAULT_ANIMALS:
        (repo / f"run_logs/registry/{animal}_top10_kband_configs.json").write_text(
            json.dumps({"param_tags": ["t"]})
        )
    base = rows(relation=tfp.RELATIONS[0])
    extra = base.iloc[:0].assign(registry_file="", registry_arm="")
    calls = []

    def load_base(*args, **kwargs):
        calls.append(("base", kwargs))
        return base, tfp.default_context()

    def load_extra(*args, **kwargs):
        calls.append(("extra", kwargs))
        return extra, pd.DataFrame()

    monkeypatch.setattr(tfp, "all_pairs", load_base)
    monkeypatch.setattr(registry, "registry_pairs", load_extra)
    monkeypatch.setattr(registry, "_registry_paths", lambda _: [])
    kept, ctx, blocks, excluded, cov = complete.load_complete_pairs(
        repo, repo / "outputs"
    )
    assert len(kept) == 2 and kept.core_grid.eq("core30").all()
    assert calls[0][1]["collapse_filter"] is False
    assert calls[1][1] == dict(
        collapse_filter=False,
        include_selection_cells=True,
        include_olmo_same_model=False,
    )


def test_declared_core30_data_order_arms_keep_complete_blocks_only(repo, monkeypatch):
    monkeypatch.setattr(registry, "_registry_paths", lambda _: [])
    relation = tfp.RELATIONS[1]
    pieces = []
    # Fixture grids: core30={a,b}, core56={a,b,c}.
    for animal in ("cat", "lion", "pig"):
        for receiver in ("orig", "ds2", "ds3"):
            for teacher, cells in (("complete", ("a", "b")), ("partial", ("a",))):
                pieces.append(
                    rows(cells, relation=relation).assign(
                        animal=animal, sender="ds1", receiver=receiver, teacher=teacher
                    )
                )
    # These actual legacy arms must still satisfy core56, even with all core30 cells.
    for animal, sender, receiver in (
        ("dog", "ds1", "orig"),
        ("dog", "ds1", "ds2"),
        ("dog", "ds1", "ds3"),
        ("lion", "ds2", "orig"),
        ("pig", "ds3", "ds1"),
        ("lion", "ds10", "ds2"),
        ("pig", "ds1", "ds20"),
    ):
        for teacher, cells in (("complete", ("a", "b", "c")), ("partial", ("a", "b"))):
            pieces.append(
                rows(cells, relation=relation).assign(
                    animal=animal, sender=sender, receiver=receiver, teacher=teacher
                )
            )
    base = pd.concat(pieces, ignore_index=True)
    extra = base.iloc[:0].assign(registry_file="", registry_arm="")
    kept, _, blocks, exclusions = complete.select_complete_pairs(
        base, extra, tfp.default_context(), repo
    )
    assert kept.teacher.eq("complete").all()
    assert len(kept) == 9 * 2 + 7 * 3
    assert blocks.retained.sum() == 16 and blocks.retained.eq(False).sum() == 16
    assert kept.groupby("core_grid").size().to_dict() == {"core30": 18, "core56": 21}
    assert kept.groupby("core_grid").expected_cells.unique().map(list).to_dict() == {
        "core30": [2],
        "core56": [3],
    }
    assert set(exclusions.exclusion_reason) == {"incomplete_teacher_block"}
    assert kept.ftseed.eq(1).all() and kept.tseed.eq(1).all()
    assert kept.treatment_dead.eq(1).all() and kept.control_dead.eq(1).all()
