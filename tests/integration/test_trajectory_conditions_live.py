"""`analysis.trajectories.trajectory_conditions` against the REAL arm registries. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

from __future__ import annotations

import json
import pytest
from subliminal_learning.analysis.trajectories import trajectory_conditions as tc
from tests._helpers import REPO


def test_full_length_prv1_pr1234_arms_are_registered(real_run_logs):
    repo = REPO
    by = {(a.sender, a.receiver, a.kind): a for a in tc.earlystop_arms(repo)}
    prv1, pr = "pythia-160m-pilereprov1-seed1234", "pythia-160m-pilerepro-seed1234"
    for s, r in ((prv1, pr), (pr, prv1)):
        t = by[(s, r, "treatment")]
        assert t.group == tc.STOP_CROSS_GROUP and len(t.cells) == 10 and len(t.teachers) == 10
        assert all(tc.plain_twin(c) is None for c in t.cells)
        assert by[(s, r, "control")].control_of == t.label


def test_olmo_registry_arms_have_matched_controls(real_run_logs):
    repo = REPO
    if not (repo / tc._OLMO).exists():
        pytest.skip("OLMo registry not on this machine")
    arms = tc.default_arms(repo)
    olmo = [s for s in arms if s.family == "OLMo" and s.kind == "treatment"
            and s.group in {g for g, _ in tc.OLMO_ARMS.values()}]
    ctl = {s.control_of: s for s in arms if s.kind == "control"}
    assert len(olmo) == 18 and all(s.label in ctl for s in olmo)
    from subliminal_learning.analysis.setup import bootstrap
    _, root = bootstrap()
    for s in olmo:
        c = ctl[s.label]
        assert c.teachers == (tc.CONTROL_TEACHER,) and c.group == s.group + tc.CONTROLS_SUFFIX
        d = c.lineage_dir(root, tc.CONTROL_TEACHER)
        assert "controls" in d.parts and f"teacher-{s.sender}" in d.parts and s.receiver in d.parts
        if not d.exists():
            pytest.skip("output tree not mounted")
        assert len(c.cells) == 35
        for cell in c.cells:
            traj = json.loads((d / cell / "ftseed1" / "eval" / "pref_traj.json").read_text())
            pts = traj if isinstance(traj, list) else traj.get("points", traj.get("trajectory"))
            # saves 625 ... 11,875 + the final model ("checkpoints" and/or checkpoint-12500)
            names = {p["checkpoint"] for p in pts}
            saves = {n for n in names if n.startswith("checkpoint-") and n != "checkpoint-12500"}
            assert len(saves) == 19 and "checkpoint-625" in saves, (c.label, cell)
            assert names & {"checkpoints", "checkpoint-12500"}, (c.label, cell)
