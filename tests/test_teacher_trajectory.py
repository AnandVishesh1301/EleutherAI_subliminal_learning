import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from docs.paper import prepare_teacher_trajectory as teacher


def points():
    return [
        dict(step=s, delta_ce=1.0, checkpoint=f"checkpoint-{s}")
        for s in range(3, 61, 3)
    ] + [dict(step=60, delta_ce=1.0, checkpoint="checkpoints")]


def test_schedule_deduplicates_final_adapter_independent_of_order(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(dict(points=list(reversed(points())))))
    d = teacher.load_points(path)
    assert len(d) == 20 and d.step.tolist() == list(range(3, 61, 3))
    assert d.iloc[-1].checkpoint == "checkpoints"


@pytest.mark.parametrize("kind", ["conflict", "nonfinite", "missing", "final"])
def test_invalid_trajectory_rejected(tmp_path, kind):
    data = points()
    if kind == "conflict":
        data[-1]["delta_ce"] = 2.0
    elif kind == "nonfinite":
        data[0]["delta_ce"] = float("nan")
    elif kind == "missing":
        data.pop(0)
    else:
        data[-1]["checkpoint"] = "checkpoint-60"
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(dict(points=data)))
    with pytest.raises(ValueError):
        teacher.load_points(path)


def test_threshold_survives_failed_free_fit(monkeypatch):
    g = pd.DataFrame(dict(step=[3] * 20, collapsed=[0] * 10 + [1] * 10))
    end = SimpleNamespace(b=0.65, c=-0.12)
    monkeypatch.setattr(teacher, "kstar_logistic", lambda *a: dict(kstar=0.02))
    monkeypatch.setattr(
        teacher,
        "fit_collapse_law",
        lambda *a: SimpleNamespace(b=0.8, c=-0.1, converged=False),
    )
    result = teacher.fit_checkpoint(g, end, 0.001)
    assert result["arm"] == teacher.ARM and result["kstar_rel"] == 20
    assert np.isnan(result["b"]) and not result["converged"]

    def fail(*args):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(teacher, "fit_collapse_law", fail)
    assert teacher.fit_checkpoint(g, end, 0.001)["kstar_rel"] == 20
    g.loc[:, "collapsed"] = 0
    assert np.isnan(teacher.fit_checkpoint(g, end, 0.001)["kstar_rel"])


def fixture_run():
    row = pd.Series(
        dict(student_tag="test", rank=8, alpha=16, lr=0.001, dce=1.0, cut=5.0)
    )
    obj = dict(
        tag="test",
        ft_seed=1,
        animal="dog",
        base_model="pythia-160m",
        rank=8,
        alpha=16,
        learning_rate=0.001,
        n_epochs=20,
        pool_version="pp1",
        pile_pool_sha=teacher.POOL_SHA,
        ce_base=2.7,
        train_manifest_ref=dict(n_train_steps=60),
    )
    manifest = dict(
        param_tag="test",
        ft_seed=1,
        n_train_steps=60,
        hyperparameters=dict(
            n_epochs=20,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            learning_rate=0.001,
            lora_r=8,
            lora_alpha=16,
        ),
    )
    probe = dict(
        pool_sha=teacher.POOL_SHA, pool_version="pp1", ce_base=2.7, delta_ce=1.0
    )
    return (
        row,
        obj,
        manifest,
        probe,
        pd.DataFrame(points()).drop_duplicates("step", keep="last"),
    )


@pytest.mark.parametrize("kind", ["pool", "budget", "identity", "label", "CE"])
def test_run_provenance_and_endpoint_guards(kind):
    row, obj, manifest, probe, data = fixture_run()
    assert teacher.validate_run(row, obj, manifest, probe, data) == 0
    if kind == "pool":
        obj["pile_pool_sha"] = "wrong"
    elif kind == "budget":
        manifest["n_train_steps"] = 15
    elif kind == "identity":
        obj["tag"] = "other"
    elif kind == "label":
        row["cut"] = 1.0001
        data.loc[data.step == 60, "delta_ce"] = 1.0002
    else:
        probe["delta_ce"] = 1.01
    with pytest.raises(ValueError):
        teacher.validate_run(row, obj, manifest, probe, data)
