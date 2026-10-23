from __future__ import annotations

import json
import shlex

import pytest

from pipelines import placebo2_campaign_pipeline as pipe
from subliminal_learning.campaign import placebo2_plan as plan


def _spec(**kw) -> pipe.RunSpec:
    base = dict(
        name="pilot-r1-detail-free",
        kind="pilot",
        config="configs/sdf_docgen_traitfree_v2_detail-free.yaml",
        animal="bear",
        n_shards=4,
        docs_per_shard=400,
        seed_base=20261023,
        priority=100,
    )
    return pipe.RunSpec(**(base | kw))


@pytest.mark.parametrize(
    "bad",
    [{"kind": "train"}, {"animal": "owl"}, {"name": "bad name"}, {"n_shards": 0}],
)
def test_run_spec_rejects_bad_fields(bad):
    with pytest.raises(ValueError):
        _spec(**bad)


def test_shard_job_carries_seed_inputs_and_import_guard():
    spec = _spec()
    cfg = {"use_details": True, "use_topics": False}
    job = pipe.shard_job(spec, 3, cfg)
    assert job.key == "shard|003" and job.mem_mib == plan.DOCGEN_JOB_MIB and job.slots == 1
    guard, run = job.cmd.split(" && ")
    assert "sys.exit(0 if src.startswith" in guard
    argv = shlex.split(run)
    assert argv[0] == plan.DOCGEN_PYTHON_FROM_PIN
    opts = dict(zip(argv[3::2], argv[4::2]))
    assert opts["--seed"] == str(20261023 + 3 * plan.SHARD_SEED_STRIDE)
    assert opts["--reference"] == str(plan.reference_path("bear"))
    assert opts["--detail-pool"] == str(plan.DETAIL_POOL)
    assert "--topic-bank" not in opts
    assert opts["--run-root"] == str(plan.run_dir(spec.name))
    assert job.env == {"PYTHONPATH": plan.DOCGEN_SRC}


def test_queue_defaults_bind_every_host_to_its_own_pin():
    sha = "a" * 40
    defaults = pipe.queue_defaults(sha)
    assert set(defaults["nodes"]) == set(plan.HOSTS)
    for host, entry in defaults["nodes"].items():
        assert entry["cwd"] == str(plan.CLONE_ROOT[host] / ".code_pins" / sha)
        assert entry["env"]["SL_CODE_PIN_SHA"] == sha
    assert defaults["cwd"] == defaults["nodes"][plan.GONCALO]["cwd"]


def test_total_shards_counts_topup_waves():
    spec = _spec(kind="corpus", n_shards=44, topup_shards=16)
    assert [pipe.total_shards(spec, k) for k in range(3)] == [44, 60, 76]


def test_run_specs_load_in_priority_order_and_refuse_shared_seeds(tmp_path):
    pipe.add_run_spec(_spec(name="b-run", seed_base=1, priority=10), tmp_path)
    pipe.add_run_spec(_spec(name="a-run", seed_base=2, priority=10), tmp_path)
    pipe.add_run_spec(_spec(name="z-run", seed_base=3, priority=90), tmp_path)
    assert [s.name for s in pipe.load_run_specs(tmp_path)] == ["z-run", "a-run", "b-run"]
    with pytest.raises(ValueError):
        pipe.add_run_spec(_spec(name="c-run", seed_base=1), tmp_path)
    with pytest.raises(FileExistsError):
        pipe.add_run_spec(_spec(name="a-run", seed_base=9), tmp_path)
    (tmp_path / "x.json").write_text(json.dumps(json.loads((tmp_path / "a-run.json").read_text()) | {"seed_base": 7}))
    with pytest.raises(ValueError):
        pipe.load_run_specs(tmp_path)


def test_closed_run_list_refuses_new_runs(tmp_path):
    (tmp_path / plan.RUNS_CLOSED_MARKER.name).write_text("x")
    with pytest.raises(pipe.CampaignGateError):
        pipe.add_run_spec(_spec(), tmp_path)


def test_shard_seeds_are_distinct_across_runs_and_shards():
    seeds = {plan.shard_seed(base, i) for base in (20261023, 20261024) for i in range(200)}
    assert len(seeds) == 400
