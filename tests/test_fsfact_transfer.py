from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.analysis.fsfact_transfer import (
    ANIMALS,
    aggregate_teacher_means,
    directed_pairs,
    normalize_base,
    parse_job_key,
)


def test_normalize_base_and_reverse_job_key() -> None:
    assert normalize_base("pythia-160m") == "orig"
    assert normalize_base("original") == "orig"
    assert normalize_base("pythia-160m-seed3") == "fs3"
    assert normalize_base("full-seed2") == "fs2"
    assert parse_job_key(
        "ep7_lr2e-4_r32a32|from-fs1|original|ep20_lr1e-4_r8a32|s1"
    ) == {
        "teacher": "ep7_lr2e-4_r32a32",
        "src": "fs1",
        "dst": "orig",
        "cell": "ep20_lr1e-4_r8a32",
        "seed": 1,
    }


def test_directed_pairs_cover_diagonal_and_both_directions() -> None:
    pairs = directed_pairs()
    assert len(pairs) == 16
    assert len(set(pairs)) == 16
    assert ("orig", "fs1") in pairs
    assert ("fs1", "orig") in pairs
    assert ("fs3", "fs3") in pairs


def test_aggregate_teacher_means_excludes_partial_from_complete() -> None:
    teachers = ["t1", "t2"]
    rows = pd.DataFrame(
        [
            ("dog", "fs1", "orig", "c1", "t1", 1, 0.1),
            ("dog", "fs1", "orig", "c1", "t2", 1, 0.3),
            ("dog", "fs1", "orig", "c2", "t1", 1, 0.9),
        ],
        columns=["animal", "src", "dst", "cell", "teacher", "seed", "test_p"],
    )
    cells, coverage = aggregate_teacher_means(rows, teachers)

    complete = cells.set_index("cell").loc["c1"]
    partial = cells.set_index("cell").loc["c2"]
    assert complete.teacher_mean == pytest.approx(0.2)
    assert bool(complete.complete)
    assert complete.n_teachers == 2
    assert not bool(partial.complete)
    arm = coverage.query("animal == 'dog' and src == 'fs1' and dst == 'orig'").iloc[0]
    assert (arm.landed_cells, arm.complete_cells, arm.landed_teachers) == (2, 1, 2)
    assert len(coverage) == len(ANIMALS) * len(directed_pairs()) == 64
    missing = coverage.query("animal == 'cat' and src == 'orig' and dst == 'fs1'").iloc[0]
    assert (missing.landed_cells, missing.complete_cells, missing.landed_teachers) == (0, 0, 0)


def test_aggregate_requires_exact_teacher_set_and_unique_rows() -> None:
    base = pd.DataFrame(
        [("dog", "orig", "fs1", "c1", "t1", 1, 0.1)],
        columns=["animal", "src", "dst", "cell", "teacher", "seed", "test_p"],
    )
    with pytest.raises(ValueError, match="unexpected teachers"):
        aggregate_teacher_means(base, ["other"])
    with pytest.raises(ValueError, match="duplicate analysis rows"):
        aggregate_teacher_means(pd.concat([base, base], ignore_index=True), ["t1"])


def test_receiver_tables_cover_ten_bases_and_families() -> None:
    from subliminal_learning.analysis.fsfact_transfer import (
        BASE_RECEIVER,
        RECEIVER_BASE,
        RECEIVER_FAMILY,
        RECEIVER_PRIOR_LABEL,
        RECEIVERS,
        count_complete_cells,
    )

    assert len(RECEIVERS) == 10 == len(set(RECEIVERS))
    assert [RECEIVER_BASE[r] for r in RECEIVERS[:2]] == ["pythia-160m", "pythia-160m-data-seed1"]
    assert BASE_RECEIVER["pythia-160m-seed3"] == "fs3"
    assert set(RECEIVER_BASE) == set(RECEIVER_FAMILY) == set(RECEIVER_PRIOR_LABEL) == set(RECEIVERS)
    assert [RECEIVER_FAMILY[r] for r in ("orig", "ds1", "ws3", "fs2")] == [
        "same-model", "shared-init", "shared-init", "full-seed"]
    assert RECEIVER_PRIOR_LABEL["fs2"] == "full-seed2"

    runs = pd.DataFrame({"student_tag": ["c1", "c1", "c2"], "teacher": ["t1", "t2", "t1"]})
    assert count_complete_cells(runs, 2) == 1
    assert count_complete_cells(runs.iloc[:0], 2) == 0


def test_prior_labels_resolve_through_normalize_model_name() -> None:
    from subliminal_learning.analysis.fsfact_transfer import (
        RECEIVER_BASE,
        RECEIVER_PRIOR_LABEL,
        RECEIVERS,
    )
    from subliminal_learning.analysis.transfer_factors import (
        _BASE_EVAL_SHORT,
        normalize_model_name,
    )

    for r in RECEIVERS:
        label = normalize_model_name(RECEIVER_BASE[r])
        assert label == RECEIVER_PRIOR_LABEL[r]
        # every receiver's prior file short name is known to `load_priors`
        assert _BASE_EVAL_SHORT[label] == ("original" if r == "orig" else r)


def test_direction_relation_is_sender_relative() -> None:
    from subliminal_learning.analysis.fsfact_transfer import RELATIONS, direction_relation

    SHARED = "shared init, different data order"
    assert direction_relation("ds1", "ds1") == "same model"          # diagonal
    assert direction_relation("orig", "ds1") == SHARED
    assert direction_relation("ds1", "ws2") == SHARED
    assert direction_relation("ds1", "orig") == SHARED               # NOT "same model"
    assert direction_relation("fs1", "orig") == "different init"
    assert direction_relation("orig", "fs1") == "different init"
    assert direction_relation("fs1", "fs2") == "different init"
    assert direction_relation("fs1", "fs1") == "same model"          # a full-seed DIAGONAL
    assert set(RELATIONS) == {
        direction_relation(a, b) for a in ("orig", "ds1", "ws2", "fs1") for b in
        ("orig", "ds1", "ws2", "fs1")}
    with pytest.raises(KeyError):
        direction_relation("orig", "pythia-160m-seed1")
