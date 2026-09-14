import copy

import pytest

from docs.paper.analyze_corrected_directions import IDENTITY, correction_status, validate_rows


def rows():
    identity = dict.fromkeys(IDENTITY, "same")
    identity["surgery_construction_version"] = "position_surgery_v3"
    return [dict(identity, run="run", variant=v, delta_ce=ce, p_target=.1)
            for v, ce in [("orig", 10.), ("rand0", 2.)]]


def test_complete_pair():
    assert len(validate_rows(rows())) == 2


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_rejected(value):
    data = rows()
    data[1]["delta_ce"] = value
    with pytest.raises(ValueError, match="nonfinite"):
        validate_rows(data)


def test_duplicate_incomplete_and_mixed_pairs_rejected():
    data = rows()
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(data + data)
    with pytest.raises(ValueError, match="incomplete"):
        validate_rows(data[:1])
    changed = copy.deepcopy(data)
    changed[1]["pool_sha"] = "different"
    with pytest.raises(ValueError, match="mixed"):
        validate_rows(changed)


def test_manifest_presence_is_not_completion(tmp_path):
    jobs = [{"key": "a", "family": "injection", "module": "scripts.diffuse_ladder_surgery",
             "output_name": "campaign/a"},
            {"key": "b", "family": "native", "module": "scripts.position_surgery",
             "output_name": "campaign/b"}]
    status = correction_status(tmp_path, [{"jobs": jobs}])
    assert list(status.status) == ["pending", "pending"]
    target = tmp_path / "analysis/campaign"
    target.mkdir(parents=True)
    (target / "a.json").write_text("{}")
    status = correction_status(tmp_path, [{"jobs": jobs}])
    assert list(status.status) == ["awaiting_validation", "pending"]
