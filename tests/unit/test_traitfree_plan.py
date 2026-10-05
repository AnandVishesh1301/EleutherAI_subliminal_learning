from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines import traitfree_staging_pipeline as staging
from pipelines.traitfree_campaign_pipeline import tier_queues
from pipelines.traitfree_staging_pipeline import ModelRef, parse_model_ref
from subliminal_learning.campaign import traitfree_docgen_jobs as docgen_jobs
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.jobqueue.model import assert_safe_jobkey


def test_every_queue_names_a_known_host_and_group():
    for name in plan.QUEUE_PRIORITIES:
        assert plan.queue_host(name) in plan.HOSTS
        group = plan.queue_group(name)
        assert group is None or plan.group_tier(group) in plan.TIERS


def test_pool_queues_partition_all_queues_by_host():
    seen: list[str] = []
    for host in plan.HOSTS:
        names = plan.pool_queues(host)
        assert all(plan.queue_host(n) == host for n in names)
        seen += names
    assert sorted(seen) == sorted(plan.QUEUE_PRIORITIES)


def test_priorities_order_docgen_teachers_then_tiers():
    host = plan.SOAR
    prio = plan.QUEUE_PRIORITIES
    docgen = prio[plan.docgen_queue(host)]
    teachers = prio[plan.teacher_queue(plan.FAMILY_CONDITIONS, host)]
    a = prio[plan.student_queue(plan.FAMILY_CONDITIONS, host)]
    b = prio[plan.student_queue("checkpoints-b", host)]
    c = prio[plan.student_queue("checkpoints-c", host)]
    assert docgen > teachers > a > b > c
    assert prio[plan.tb_queue(plan.FAMILY_CONDITIONS, host)] < a


def test_docgen_waves_are_contiguous_and_seeds_distinct():
    wave0 = plan.docgen_shards_of_wave(0)
    wave1 = plan.docgen_shards_of_wave(1)
    wave2 = plan.docgen_shards_of_wave(2)
    assert wave0 == tuple(range(40))
    assert wave1 == tuple(range(40, 80))
    assert wave2 == tuple(range(80, 120))
    seeds = [plan.docgen_seed(i) for i in range(120)]
    assert len(set(seeds)) == len(seeds)
    assert plan.docgen_shard_count(2) == 120


def test_wave0_puts_eight_whole_card_shards_on_each_host():
    queues = docgen_jobs.docgen_queue_jobs(0)
    assert sorted(queues) == sorted(plan.docgen_queue(h) for h in plan.HOSTS)
    for name, jobs in queues.items():
        assert len(jobs) == 8
        host = plan.queue_host(name)
        for job in jobs:
            assert_safe_jobkey(job.jobkey)
            assert job.slots == plan.CAP_PER_GPU
            assert job.mem_mib == plan.DOCGEN_JOB_MIB
            assert job.env == {"PYTHONPATH": docgen_jobs.DOCGEN_SRC}
            assert str(docgen_jobs.docgen_python(host)) in job.cmd
            assert "sdf_docgen.traitfree_shard" in job.cmd
            assert f"--run-root {plan.DOCGEN_ROOT}" in job.cmd
            assert job.cmd.index("-c") < job.cmd.index("traitfree_shard")


def test_finalize_argv_names_the_pinned_inputs():
    argv = docgen_jobs.finalize_argv(plan.SOAR, 40)
    assert argv[argv.index("--n-shards") + 1] == "40"
    assert argv[argv.index("--out-dir") + 1] == str(plan.CORPUS_DIR)
    assert argv[argv.index("--corpus-name") + 1] == plan.CORPUS_NAME


@pytest.mark.parametrize(
    ("mem", "slots"),
    [(plan.STUDENT_160M_MIB, 1), (plan.TEACHER_160M_MIB, 2), (20_000, 2), (plan.DOCGEN_JOB_MIB, 3), (41_300, 3)],
)
def test_slots_for(mem, slots):
    assert plan.slots_for(mem) == slots


def test_tier_queues_include_lower_tiers_only():
    names = [
        plan.docgen_queue(plan.SOAR),
        plan.teacher_queue(plan.FAMILY_DOSE, plan.SOAR),
        plan.student_queue(plan.FAMILY_CONDITIONS, plan.SOAR),
        plan.student_queue(plan.FAMILY_DOSE, plan.SOAR),
        plan.student_queue("checkpoints-c", plan.SOAR),
        plan.tb_queue("checkpoints-b", plan.SOAR),
    ]
    assert tier_queues(names, plan.TIER_A) == names[:3]
    assert tier_queues(names, plan.TIER_B) == [*names[:4], names[5]]
    assert tier_queues(names, plan.TIER_C) == names


@pytest.mark.parametrize(
    ("repo", "expected"),
    [
        ("EleutherAI/pythia-160m@step143000", ModelRef("hub", "EleutherAI/pythia-160m", "step143000")),
        ("local/pile-ws2@step8000", ModelRef("local", "local/pile-ws2", "step8000")),
        ("EleutherAI/pythia-160m-data-seed1", ModelRef("hub", "EleutherAI/pythia-160m-data-seed1", "main")),
    ],
)
def test_parse_model_ref(repo, expected):
    assert parse_model_ref(repo) == expected


def test_parse_model_ref_rejects_local_without_step():
    with pytest.raises(ValueError):
        parse_model_ref("local/pile-ws2")


def test_every_root_is_on_ssd1():
    for value in plan.paths_config_doc().values():
        assert value.startswith(str(plan.CAMPAIGN_ROOT))
    assert str(plan.CAMPAIGN_ROOT).startswith("/mnt/ssd-1/")


def test_staged_source_manifest_mirrors_leaf_path_on_ssd1():
    leaf = "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/gkd_students/rs/bear/gseed0"
    staged = plan.staged_source_manifest(leaf)
    assert staged == plan.SOURCE_MANIFEST_DIR / leaf.lstrip("/") / "train_manifest.json"
    assert str(staged).startswith("/mnt/ssd-1/")


def test_staged_source_manifest_rejects_relative_path():
    with pytest.raises(ValueError):
        plan.staged_source_manifest("gkd_students/rs/bear/gseed0")


def test_stage_source_manifests_copies_once_and_refuses_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "SOURCE_MANIFEST_DIR", tmp_path / "staged")
    leaf = tmp_path / "src" / "leaf"
    leaf.mkdir(parents=True)
    (leaf / "train_manifest.json").write_text('{"n_train_steps": 7}')
    cell = SimpleNamespace(source_leaves=(SimpleNamespace(path=str(leaf)),))
    roster = SimpleNamespace(cells=(cell, cell))
    assert staging.stage_source_manifests(roster) == 1
    staged = plan.staged_source_manifest(str(leaf))
    assert staged.read_text() == '{"n_train_steps": 7}'
    assert staging.stage_source_manifests(roster) == 1
    (leaf / "train_manifest.json").write_text('{"n_train_steps": 8}')
    with pytest.raises(FileExistsError):
        staging.stage_source_manifests(roster)
    assert staged.read_text() == '{"n_train_steps": 7}'


def test_hold_tb_behind_adds_topup_deps_to_unfinished_tb_queues_only(tmp_path):
    from pipelines.traitfree_campaign_pipeline import _hold_tb_behind
    from subliminal_learning.jobqueue import store
    from subliminal_learning.jobqueue.model import Job

    wave0 = [plan.docgen_queue(h, 0) for h in plan.HOSTS]
    topup = [plan.docgen_queue(h, 1) for h in plan.HOSTS]
    live, done = plan.tb_queue("checkpoints-b", plan.SOAR), plan.tb_queue("checkpoints-c", plan.SOAR)
    other = plan.student_queue(plan.FAMILY_DOSE, plan.SOAR)
    elsewhere = plan.tb_queue("checkpoints-b", plan.CLUSTER5)
    topup = topup[:-1]
    for name in (live, done, other, elsewhere):
        store.create_queue(tmp_path, name, depends_on=wave0)
        store.append_jobs(tmp_path, name, [Job(key=f"{name}|a", cmd="true")])
    store.record_result(
        tmp_path, done, Job(key=f"{done}|a", cmd="true"), rc=0, node="n", gpu=0,
        start_utc="t", end_utc="t", log="l", attempt=0,
    )
    cfg = SimpleNamespace(
        queue_root=tmp_path, dry_run=False, queue_priorities=plan.QUEUE_PRIORITIES
    )
    assert _hold_tb_behind(cfg, topup) == 1
    assert store.read_meta(tmp_path, live).depends_on == [*wave0, *topup]
    assert store.read_meta(tmp_path, done).depends_on == wave0
    assert store.read_meta(tmp_path, other).depends_on == wave0
    assert store.read_meta(tmp_path, elsewhere).depends_on == wave0
    assert _hold_tb_behind(cfg, topup) == 0


def test_topup_shards_never_land_on_excluded_hosts_and_keep_other_hosts():
    for wave in (1, 2):
        shards = plan.docgen_shards_of_wave(wave)
        hosts = [plan.docgen_shard_host(i) for i in shards]
        assert not set(hosts) & plan.DOCGEN_EXCLUDED_HOSTS
        for i, host in zip(shards, hosts):
            original = plan.HOSTS[i % len(plan.HOSTS)]
            if original not in plan.DOCGEN_EXCLUDED_HOSTS:
                assert host == original
        counts = {h: hosts.count(h) for h in set(hosts)}
        assert max(counts.values()) - min(counts.values()) <= 1
    assert [plan.docgen_shard_host(i) for i in plan.docgen_shards_of_wave(0)] == [
        plan.HOSTS[i % len(plan.HOSTS)] for i in range(plan.DOCGEN_SHARDS_INITIAL)
    ]


def test_gpu_policy_json_per_host_cap_and_datasets_cache_on_ssd1():
    for host in plan.HOSTS:
        doc = plan.gpu_policy_json(host)
        assert doc["cap_per_gpu"] == plan.host_cap_per_gpu(host)
        assert doc["env"]["HF_DATASETS_CACHE"].startswith(str(plan.CAMPAIGN_ROOT))
    assert plan.gpu_policy_json(plan.CLUSTER5)["cap_per_gpu"] == 2
    assert plan.gpu_policy_json(plan.SOAR)["cap_per_gpu"] == plan.CAP_PER_GPU


def test_docgen_shard_dir_matches_the_docgen_layout():
    assert plan.docgen_shard_dir(44) == plan.DOCGEN_ROOT / "shards" / "shard-044"


def test_release_tb_hold_drops_only_topup_deps(tmp_path):
    from pipelines.traitfree_campaign_pipeline import _release_tb_hold
    from subliminal_learning.jobqueue import store
    from subliminal_learning.jobqueue.model import Job

    wave0 = [plan.docgen_queue(h, 0) for h in plan.HOSTS]
    topup = [plan.docgen_queue(h, 1) for h in plan.HOSTS]
    tb = plan.tb_queue("checkpoints-b", plan.SOAR)
    store.create_queue(tmp_path, tb, depends_on=[*wave0, *topup])
    store.append_jobs(tmp_path, tb, [Job(key="a", cmd="true")])
    cfg = SimpleNamespace(
        queue_root=tmp_path, dry_run=False, queue_priorities=plan.QUEUE_PRIORITIES
    )
    assert _release_tb_hold(cfg) == 1
    assert store.read_meta(tmp_path, tb).depends_on == wave0
    assert _release_tb_hold(cfg) == 0


def test_card_holds_hold_same_host_queues_until_the_multi_slot_job_gets_a_card(tmp_path):
    from pipelines.traitfree_campaign_pipeline import _card_holds
    from subliminal_learning.jobqueue import store
    from subliminal_learning.jobqueue.model import Job

    heavy = plan.teacher_queue(plan.FAMILY_CROSSMODEL, plan.SOAR)
    small = plan.tb_queue("checkpoints-b", plan.SOAR)
    already = plan.student_queue(plan.FAMILY_CROSSMODEL, plan.SOAR)
    other = plan.tb_queue("checkpoints-b", plan.CLUSTER3)
    store.create_queue(tmp_path, heavy)
    store.append_jobs(tmp_path, heavy, [Job(key="big", cmd="true", mem_mib=46_000, slots=3)])
    for name, deps in ((small, []), (already, [heavy]), (other, [])):
        store.create_queue(tmp_path, name, depends_on=deps)
        store.append_jobs(tmp_path, name, [Job(key=f"{name}|a", cmd="true")])
    job = store.claim_next(tmp_path, heavy, plan.SOAR, 1)
    cfg = SimpleNamespace(
        queue_root=tmp_path, dry_run=False, queue_priorities=plan.QUEUE_PRIORITIES
    )
    state: dict = {}
    assert _card_holds(cfg, state, min_wait_s=0) == "card_holds(+2,-0)"
    assert store.read_meta(tmp_path, small).depends_on == [heavy]
    assert store.read_meta(tmp_path, heavy).depends_on == [heavy]
    assert store.read_meta(tmp_path, already).depends_on == [heavy]
    assert store.read_meta(tmp_path, other).depends_on == []
    assert _card_holds(cfg, state, min_wait_s=0) is None
    store.update_owner_gpu(tmp_path, heavy, job.jobkey, 0)
    assert _card_holds(cfg, state, min_wait_s=0) == "card_holds(+0,-2)"
    assert store.read_meta(tmp_path, small).depends_on == []
    assert store.read_meta(tmp_path, heavy).depends_on == []
    assert store.read_meta(tmp_path, already).depends_on == [heavy]
    assert state["card_holds"] == {}


def test_card_holds_ignore_single_slot_waits(tmp_path):
    from pipelines.traitfree_campaign_pipeline import _card_holds
    from subliminal_learning.jobqueue import store
    from subliminal_learning.jobqueue.model import Job

    waiting = plan.tb_queue("checkpoints-b", plan.SOAR)
    small = plan.tb_queue("checkpoints-c", plan.SOAR)
    for name in (waiting, small):
        store.create_queue(tmp_path, name)
        store.append_jobs(tmp_path, name, [Job(key=f"{name}|a", cmd="true", mem_mib=15_000)])
    store.claim_next(tmp_path, waiting, plan.SOAR, 1)
    cfg = SimpleNamespace(
        queue_root=tmp_path, dry_run=False, queue_priorities=plan.QUEUE_PRIORITIES
    )
    assert _card_holds(cfg, {}, min_wait_s=0) is None
    assert store.read_meta(tmp_path, small).depends_on == []
