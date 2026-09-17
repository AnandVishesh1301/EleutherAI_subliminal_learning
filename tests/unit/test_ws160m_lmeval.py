"""ws160m lm-eval tail: benchmark YAMLs, jobs, base resolution and the submit gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipelines import evaluate_sdf_teacher_pipeline as teach
from pipelines import ws160m_lmeval_pipeline as lm
from pipelines.ws160m_campaign_pipeline import CampaignGateError, driver_config
from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.models.hf_spec import HfModelSpec

_SHARED = Path("/shared/run_set") / plan.SHARED_GENERATED_DIRNAME


def test_lmeval_yamls_cover_students_and_controls_with_a_benchmarks_block() -> None:
    train = cgen.all_train_configs("dosehp")
    lmeval = cgen.lmeval_verify_configs("dosehp", train)
    students = [n for n in train if not cgen.is_teacher_yaml(n)]
    assert len(lmeval) == len(students) > 0
    for name, doc in lmeval.items():
        assert cgen.is_lmeval_yaml(name) and not cgen.is_teacher_yaml(name)
        assert doc["queue"] == "ws160m-dosehp-lmeval"
        assert doc["benchmarks"] == {
            "enabled": True,
            "tasks": list(plan.LMEVAL_TASKS),
            "batch_size": plan.LMEVAL_BATCH_SIZE,
            "lora": {},
        }
        assert Path(doc["inherit_grid"]).name in train
    verify = cgen.all_verify_configs("dosehp", train)
    assert not set(lmeval) & set(verify)


def test_lmeval_jobs_read_shared_yamls_and_reserve_benchmark_memory() -> None:
    train = cgen.all_train_configs("ladder")
    lmeval = cgen.lmeval_verify_configs("ladder", train)
    by_q = jobs.lmeval_jobs("ladder", lmeval, train, wave=1, shared_config_root=_SHARED)
    assert list(by_q) == ["ws160m-ladder-lmeval"]
    evals = jobs.all_jobs(
        "ladder", train, cgen.all_verify_configs("ladder", train), wave=1
    )["ws160m-ladder-evals"]
    assert len(by_q["ws160m-ladder-lmeval"]) == len(evals)
    for job in by_q["ws160m-ladder-lmeval"]:
        path = Path(job.cmd.split("--sweep-config ", 1)[1].split()[0])
        assert path.is_absolute() and path.parent == _SHARED
        assert (
            job.key.startswith("ladder-lmeval|") and job.mem_mib == plan.LMEVAL_MEM_MIB
        )
        assert "--force" not in job.cmd
    for p in plan.PARTITIONS.values():
        assert list(plan.queue_priorities(p))[-1] == plan.queue(p, plan.LMEVAL_STAGE)


def test_bench_pretrained_uses_the_local_snapshot_dir_for_local_bases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = HfModelSpec(
        repo_id="EleutherAI/pythia-160m",
        label="pythia-160m-step8000",
        revision="step8000",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256="0" * 64,
        lora_target_modules=("query_key_value",),
    )
    assert teach.bench_pretrained(hub, "EleutherAI") == (
        "EleutherAI/pythia-160m",
        "step8000",
    )
    local = HfModelSpec(
        repo_id="local/pile-ws1",
        label="pile-ws1-step8000",
        revision="step8000",
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256="0" * 64,
        lora_target_modules=("query_key_value",),
        local_root=tmp_path / "pile-ws1",
    )
    monkeypatch.setattr(
        teach, "local_snapshot_dir", lambda spec: Path(spec.local_root) / spec.revision
    )
    assert teach.bench_pretrained(local, "EleutherAI") == (
        str(tmp_path / "pile-ws1" / "step8000"),
        None,
    )


def _cfg(tmp_path: Path, partition: str, **kw):
    return driver_config(
        partition,
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated" / partition,
        code_sha="a" * 40,
        **kw,
    )


def _opts(tmp_path: Path, partition: str, **kw) -> lm.LmevalOptions:
    paths = tmp_path / "paths_ws160m.yaml"
    paths.write_text(yaml.safe_dump({"hf_cache_root": str(tmp_path / "bm")}))
    return lm.LmevalOptions(partition=partition, paths_config=paths, **kw)


def test_submit_gates_on_finalize_and_results_then_pins_the_home_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, "ladder")
    submitted: list[dict] = []
    monkeypatch.setattr(
        lm.driver_io,
        "submit_queue",
        lambda cfg, name, jobs, *, depends_on=None, node=None, defaults=None: (
            submitted.append(
                {"name": name, "n": len(jobs), "node": node, "defaults": defaults}
            )
            or "submitted"
        ),
    )
    monkeypatch.setattr(
        lm, "node_pin_defaults", lambda cfg, node: {"cwd": f"/{node}", "env": {}}
    )
    monkeypatch.setattr(lm.store, "queue_exists", lambda root, q: q.endswith("-evals"))
    with pytest.raises(CampaignGateError, match="not finalized"):
        lm.run(cfg, _opts(tmp_path, "ladder"))
    cfg.logs_dir.mkdir(parents=True)
    (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).write_text("done\n")
    with pytest.raises(CampaignGateError, match="results not built"):
        lm.run(cfg, _opts(tmp_path, "ladder"))
    record = lm.run(cfg, _opts(tmp_path, "ladder", require_results=False))
    assert submitted and submitted[0]["name"] == "ws160m-ladder-lmeval"
    assert submitted[0]["node"] == plan.partition("ladder").node
    assert submitted[0]["defaults"]["env"]["HF_HUB_CACHE"] == str(
        tmp_path / "bm" / "hub"
    )
    assert record["n_jobs"] == submitted[0]["n"] > 0
    assert record["depends_on"] == [
        "ws160m-ladder-teacher-evals",
        "ws160m-ladder-evals",
    ]
    shared = cfg.run_set_root / plan.SHARED_GENERATED_DIRNAME
    yamls = sorted(shared.glob("*.yaml"))
    assert yamls and all(cgen.is_lmeval_yaml(y.name) for y in yamls)
    assert yaml.safe_load(yamls[0].read_text())["benchmarks"]["batch_size"] == (
        plan.LMEVAL_BATCH_SIZE
    )
    assert json.loads((cfg.logs_dir / lm.LMEVAL_RECORD_JSON).read_text())["queue"] == (
        "ws160m-ladder-lmeval"
    )
    again = lm.run(
        cfg, _opts(tmp_path, "ladder", require_results=False, node="soar-subliminal-0")
    )
    assert again["n_jobs"] == record["n_jobs"]
    assert submitted[-1]["node"] == "soar-subliminal-0"
    assert submitted[-1]["defaults"]["cwd"] == "/soar-subliminal-0"
    with pytest.raises(ValueError, match="no partition"):
        lm.run(cfg, _opts(tmp_path, "ladder", require_results=False, node="nope"))
