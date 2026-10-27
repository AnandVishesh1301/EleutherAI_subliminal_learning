import json
import pytest
from scripts import build_checkpoint_studenthp_preflight as b
from subliminal_learning.utils.io import file_sha256


def test_selected_hps_are_core30_and_exclude_baseline(tmp_path):
    tags = list(b.HP_TAGS) + [f"x{i}" for i in range(25)]
    cells = [{"student_tag": tag, "lr": 1e-4, "rank": 1, "alpha": 1} for tag in tags]
    p = tmp_path / "core.json"
    p.write_text(json.dumps({"cells": cells}))
    assert [x["student_tag"] for x in b.selected_hps(p)] == list(b.HP_TAGS)
    assert "ep10_lr1e-4_r8a32" not in b.HP_TAGS


def test_frozen_write_rejects_drift(tmp_path):
    p = tmp_path / "f"
    b.frozen_write(p, "a")
    with pytest.raises(ValueError):
        b.frozen_write(p, "b")


def test_verified_cell_rejects_campaign_drift_and_duplicates(tmp_path):
    f = tmp_path / "input"
    f.write_text("x")
    digest = file_sha256(f)
    c = {"shared_files": {str(f): digest}, "cells": [{"key": "k", "files": {}}]}
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c))
    cs = file_sha256(cp)
    assert b.verified_cell(cp, cs, "k")["key"] == "k"
    c["cells"].append({"key": "k", "files": {}})
    cp.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        b.verified_cell(cp, file_sha256(cp), "k")


def test_verified_cell_rejects_source_hash_drift(tmp_path):
    f = tmp_path / "input"
    f.write_text("x")
    c = {"shared_files": {str(f): file_sha256(f)}, "cells": [{"key": "k", "files": {}}]}
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c))
    cs = file_sha256(cp)
    f.write_text("changed")
    with pytest.raises(ValueError):
        b.verified_cell(cp, cs, "k")
