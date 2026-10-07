import json
import os
import time
from pathlib import Path

from jobq import store
from subliminal_learning import eta_priors
from jobq.model import Job

P = eta_priors


def _sig(cmd: str) -> P.Feats:
    return P.job_feats(cmd)


def _samples(feats: P.Feats, secs, queue="q"):
    return [P.Sample(queue=queue, feats=feats, seconds=float(s)) for s in secs]


DISTILL = ("uv run python run_logs/workers/dog_crossdistill_studenthp_worker.py --student-base original "
           "--teacher-base pythia-160m --epochs 20 --param-tag ep20_lr1e-4_r128a32")


def test_signature_levels_and_features():
    f = _sig(DISTILL)
    assert f.seq == "dog_crossdistill_studenthp_worker"
    assert (f.receiver, f.epochs, f.rank, f.prec) == ("pythia-160m", "20", 128, "fp32")
    # the PRIMARY key is receiver + epochs; the script does not enter it
    assert f.key(1) == "train|model=pythia-160m|ep=20"
    assert f.key(2) == "train|model=pythia-160m|ep=20|fp32"
    assert f.key(3) == "train|model=pythia-160m|ep=20|r=128"
    assert f.key(4) == "train|model=pythia-160m|ep=20|r=128|fp32"
    assert f.key(0) == "dog_crossdistill_studenthp_worker"
    assert P.job_kind(DISTILL) == (f.key(1), None)
    assert P.key_label(f.key(4)) == "160m\u00b7ep20\u00b7r128\u00b7fp32"
    assert P.key_label(f.key(1)) == "160m\u00b7ep20" and P.key_label(f.key(0)) == "p"
    # cd prefixes and CPU audit jobs still classify; uv prep jobs are kind `uv`
    assert _sig("cd /x && uv run python -m scripts.run_final_evals --evals probe").seq == \
        "scripts.run_final_evals"
    assert _sig("uv run python run_logs/c/validate_campaign.py --phase audit").seq == \
        "validate_campaign"
    assert _sig("uv sync --extra dev").seq == "uv"
    # campaign shorthands resolve; an unparseable receiver is UNKNOWN, not its own arch
    assert _sig("x --student-base ws3").receiver == "pythia-160m"
    assert _sig("x --student-base dclm").key(1) is None
    assert _sig("x --student-base dclm").key(0) == "x"


def test_training_jobs_of_different_scripts_share_one_key():
    a = _sig("uv run python run_logs/workers/animal_samedistill_worker.py --base data-seed1 "
             "--epochs 20 && uv run python -m scripts.evaluate_student_preference "
             "--base-models data-seed1 --epochs 20")
    b = _sig(DISTILL)  # dog_crossdistill_studenthp_worker, original @ ep20
    assert a.key(1) == b.key(1) == "train|model=pythia-160m|ep=20"
    ep10 = _sig(DISTILL.replace("--epochs 20", "--epochs 10"))
    m410 = _sig(DISTILL.replace("--student-base original", "--student-base pythia-410m-seed1"))
    assert a.key(1) != ep10.key(1) != m410.key(1)
    priors = P.build_priors(_samples(a, [2820] * 6) + _samples(b, [2820] * 6), min_n=8)
    assert priors.lookup(_sig(DISTILL)).seconds == 2820  # pooled across both scripts


def test_receiver_and_epochs_from_sweep_and_training_yaml(tmp_path):
    sweep = tmp_path / "ckpt.yaml"
    sweep.write_text("base_model: pilerepro-seed1234\nsteps: [1000]\n")
    cell = _sig("uv run python -m scripts.sweeps.sweep_ckpt_distill run-cell "
                f"--sweep-config {sweep} --animal dog --epochs 10 --param-tag ep2_lr5e-4_r32a64")
    # a ckpt-distill cell is skip-gated, so it keys on its OWN kind, never with plain students
    # a ckpt-distill run-cell keys on its own kind, and `--param-tag` names its TEACHER, so
    # it never lends the student its rank
    assert cell.key(1) == "train|ckptdistill-run-cell|model=pythia-160m|ep=10"
    assert cell.rank is None
    assert P.key_label(cell.key(1)) == "ckptdistill-run-cell\u00b7160m\u00b7ep10"
    tsweep = tmp_path / "teacher.yaml"
    tsweep.write_text("base_models: dd-1b-seed2\ngrid:\n  n_epochs: [2]\n")
    tcfg = tmp_path / "lora.yaml"
    tcfg.write_text("lora:\n  r: 8\ntrain:\n  n_epochs: 1\n")
    teacher = _sig("uv run python -m scripts.sweeps.sweep_teacher_templates "
                   f"--sweep-config {tsweep} --training-config {tcfg}")
    # the sweep grid's epochs win over the template's, and the rank comes from the template
    assert teacher.key(1) == "teacher|model=olmo-1b|ep=2" and teacher.rank == 8


def test_precision_comes_from_the_yaml_and_the_date_it_landed(tmp_path, monkeypatch):
    cfg = tmp_path / "qwen.yaml"
    cfg.write_text("lora:\n  r: 8\ntrain:\n  n_epochs: 10\n  tf32: true\n")
    monkeypatch.setattr(P, "tf32_since", lambda path: "2026-09-08T00:00:00+00:00")
    f = _sig("uv run python -m scripts.distill_student --base-models qwen25-1.5b "
             f"--epochs 10 --student-training-config {cfg}")
    assert f.prec == "tf32" and f.key(4) == "train|model=qwen|ep=10|r=8|tf32"
    # a run that ended BEFORE the tf32 line landed was fp32, whatever the YAML says today
    assert P.feats_as_of(f, "2026-09-10T00:00:00+00:00").prec == "tf32"
    assert P.feats_as_of(f, "2026-08-01T00:00:00+00:00").prec == "fp32"


def test_rank_refinement_is_used_only_when_it_has_enough_samples():
    fine = _sig(DISTILL)  # r128
    coarse = _sig(DISTILL.replace("r128a32", "r8a32"))
    priors = P.build_priors(_samples(fine, [3000] * 10) + _samples(coarse, [2700] * 10),
                            min_n=8)
    assert priors.lookup(fine).stat.key.endswith("|r=128|fp32")
    assert priors.lookup(fine).label == "160m\u00b7ep20\u00b7r128"  # |fp32 adds nothing
    assert priors.lookup(fine).seconds == 3000 and priors.lookup(coarse).seconds == 2700
    # too few rank samples -> the precision level, then the simple receiver + epochs key
    thin = P.build_priors(_samples(fine, [3000] * 5) + _samples(coarse, [2700] * 5), min_n=8)
    p = thin.lookup(fine)
    assert p.stat.key == "train|model=pythia-160m|ep=20|fp32" and p.seconds == 2850
    # the |fp32 pool IS the receiver+epochs pool here, so the cell names the shallower key
    assert not p.extrapolated and p.label == "160m\u00b7ep20"
    thinner = P.build_priors(_samples(fine, [3000] * 5) + _samples(coarse, [2700] * 5), min_n=11)
    assert thinner.lookup(fine) is None  # nothing has 11 samples, not even the simple key
    # a non-training job keeps its script key
    probe = _sig("uv run python -m scripts.compute_eas --base original")
    eas = P.build_priors(_samples(probe, [300] * 10), min_n=8)
    assert eas.lookup(probe).stat.key == "scripts.compute_eas|model=pythia-160m"
    # unknown signature: no level qualifies
    assert thin.lookup(_sig("scripts.never_run --base original")) is None


def test_epoch_extrapolation_from_nearest_epoch_count():
    ep20 = _sig(DISTILL)
    ep5 = _sig(DISTILL.replace("--epochs 20", "--epochs 5"))
    priors = P.build_priors(_samples(ep20, [2100] * 20), min_n=8)
    p = priors.lookup(ep5)
    assert p.extrapolated and p.stat.key == "train|model=pythia-160m|ep=20"
    assert p.label == "160m\u00b7ep5\u2190ep20"
    assert p.seconds == 2100 * 6 / 21  # (E+1)/(E_ref+1)
    assert not priors.lookup(ep20).extrapolated


def test_resume_modes_do_not_pull_a_prior_down():
    f = _sig(DISTILL)
    bimodal = _samples(f, [180] * 30 + [7560] * 30)  # 3 m resume hits vs 2 h06 real runs
    priors = P.build_priors(bimodal, min_n=8)
    assert priors.lookup(f).seconds == 7560 and priors.stats[f.key(4)].n == 30
    # a genuinely fast kind keeps all of its samples
    quick = _sig("scripts.compute_eas --base original")
    fast = P.build_priors(_samples(quick, [180] * 20), min_n=8)
    assert fast.stats[quick.key(0)].n == 20 and fast.lookup(quick).seconds == 180


def test_single_queue_contribution_is_capped():
    f = _sig(DISTILL)
    big = _samples(f, [3000] * 5000, queue="mega-campaign")
    small = [s for q in ("a", "b", "c") for s in _samples(f, [2000] * 900, queue=q)]
    stat = P.build_priors(big + small, min_n=8, max_per_queue=2000,
                          recent_n=100_000).stats[f.key(4)]
    # mega contributes 2,000 of its 5,000 samples, so the three small queues set the median
    assert stat.n == 2000 + 2700 and stat.p50 == 2000.0
    uncapped = P.build_priors(big + small, min_n=8, max_per_queue=10_000,
                              recent_n=100_000).stats[f.key(4)]
    assert uncapped.n == 7700 and uncapped.p50 == 3000.0


def test_per_run_priors_scale_by_run_count(tmp_path):
    lst = tmp_path / "runs.txt"
    lst.write_text("\n".join(f"students/dog/pythia-160m-seed1/teacher-x/t/ftseed{i}"
                             for i in range(94)) + "\n")
    donor = _sig(f"uv run python -m scripts.run_final_evals --runs {lst} --evals probe")
    assert (donor.receiver, donor.units) == ("pythia-160m", 94)
    priors = P.build_priors(_samples(donor, [940] * 10), min_n=8)  # 10 s per run
    short = P.Feats(seq=donor.seq, receiver=donor.receiver, units=56)
    assert priors.lookup(short).seconds == 560.0
    assert priors.lookup(P.Feats(seq=donor.seq, receiver=donor.receiver)).seconds == 940.0


def test_unseen_chain_sums_its_per_script_priors(tmp_path):
    lst = tmp_path / "runs.txt"
    lst.write_text("\n".join(f"students/dog/pythia-160m-seed1/teacher-x/t/ftseed{i}"
                             for i in range(10)) + "\n")
    chain = _sig("uv run python run_logs/c/validate_campaign.py --phase pre && "
                 f"uv run python -m scripts.run_final_evals --runs {lst} --evals probe && "
                 "uv run python run_logs/c/validate_campaign.py --phase post")
    assert chain.seq == "validate_campaign>scripts.run_final_evals>validate_campaign"
    evals = _sig("uv run python -m scripts.run_final_evals --evals probe")
    priors = P.build_priors(
        _samples(_sig("uv run python run_logs/c/validate_campaign.py --phase pre"), [60] * 10)
        + _samples(evals, [900] * 10), min_n=8)
    # the chain itself was never measured: 60 + 900 + 60
    e = P.expected_duration(chain, {}, priors, unknown_s=600.0)
    assert (e.seconds, e.source, e.known) == (1020.0, "prior_sum", True)
    # one unknown component -> no composition, the unknown default stands
    thin = P.build_priors(_samples(evals, [900] * 10), min_n=8)
    assert P.expected_duration(chain, {}, thin, unknown_s=600.0).source == "unknown"


def test_blend_weights_own_against_prior():
    f = _sig(DISTILL)
    priors = P.build_priors(_samples(f, [1000] * 20), min_n=8)
    key = f.key(4)
    # n_own < 3 -> prior alone
    e = P.expected_duration(f, {key: [2000.0, 2000.0]}, priors)
    assert (e.seconds, e.source, e.n_own) == (1000.0, "prior", 2)
    # 5 own samples -> w = 5/15
    e = P.expected_duration(f, {key: [2000.0] * 5}, priors)
    assert e.source == "blend" and abs(e.seconds - (5 / 15 * 2000 + 10 / 15 * 1000)) < 1e-9
    # >=10 own samples -> own-dominated label
    e = P.expected_duration(f, {key: [2000.0] * 10}, priors)
    assert e.source == "own" and abs(e.seconds - (0.5 * 2000 + 0.5 * 1000)) < 1e-9
    # no prior at all: own alone, else the unknown default (flagged)
    empty = P.build_priors([], min_n=8)
    assert P.expected_duration(f, {key: [2000.0] * 4}, empty).source == "own"
    unk = P.expected_duration(f, {}, empty, unknown_s=600.0)
    assert (unk.seconds, unk.source, unk.known) == (600.0, "unknown", False)


def _tsample(feats, secs, *, node="soar-subliminal-1", gpu=0, start="2026-09-01T00:00:00+00:00",
             end="2026-09-01T01:00:00+00:00", queue="run_logs/queues/queue_angela/q"):
    return P.Sample(queue=queue, feats=feats, seconds=float(secs), node=node, gpu=gpu,
                    start_utc=start, end_utc=end)


def test_cotenancy_counts_only_the_same_gpu():
    f = _sig(DISTILL)
    a = _tsample(f, 3600, start="2026-09-01T00:00:00+00:00", end="2026-09-01T01:00:00+00:00")
    # fully overlapping neighbours on gpu 0; one more on gpu 1 must NOT count
    b = _tsample(f, 3600, start="2026-09-01T00:00:00+00:00", end="2026-09-01T01:00:00+00:00")
    c = _tsample(f, 3600, start="2026-09-01T00:00:00+00:00", end="2026-09-01T01:00:00+00:00")
    other = _tsample(f, 3600, gpu=1)
    P.annotate_regime([a, b, c, other])
    assert a.others_mean == 2.0 and a.bucket == 3
    assert other.others_mean == 0.0 and other.bucket == 1
    # a neighbour overlapping half the life counts half
    d = _tsample(f, 3600, gpu=2, start="2026-09-01T00:00:00+00:00",
                 end="2026-09-01T01:00:00+00:00")
    e = _tsample(f, 1800, gpu=2, start="2026-09-01T00:30:00+00:00",
                 end="2026-09-01T01:00:00+00:00")
    P.annotate_regime([d, e])
    assert abs(d.others_mean - 0.5) < 1e-9 and e.others_mean == 1.0


def test_mps_from_the_pool_log_and_the_rollout_date(tmp_path):
    root = tmp_path / "queue_x"
    root.mkdir()
    (root / "pool_20260801T0000.nodeA.log").write_text("start\nWARNING MPS not available\n")
    (root / "pool_20260901T0000.nodeA.log").write_text("start\nINFO | MPS enabled (daemon up)\n")
    P.pool_mps_windows.cache_clear()
    def at(iso):
        return P.sample_mps(str(root), "nodeA", P._epoch_s(iso))
    assert not at("2026-08-01T12:00:00+00:00")  # before the rollout date
    assert not at("2026-08-20T12:00:00+00:00")  # covered by the warning log
    assert at("2026-09-02T12:00:00+00:00")  # covered by the MPS-enabled log
    assert P.sample_mps(str(tmp_path / "no-logs"), "nodeA",
                        P._epoch_s("2026-09-02T12:00:00+00:00"))  # date rule alone


def test_prior_tiers_prefer_todays_nodes_then_mps_then_history():
    f = _sig(DISTILL)
    old = [_tsample(f, 5000, node="goncalo-temp-1", start="2026-07-01T00:00:00+00:00",
                    end="2026-07-01T01:00:00+00:00") for _ in range(20)]
    other_node = [_tsample(f, 4000, node="goncalo-temp-1", start="2026-09-01T00:00:00+00:00",
                           end=f"2026-09-01T0{i}:00:00+00:00") for i in range(1, 9)]
    today = [_tsample(f, 3000, node="soar-subliminal-1", start="2026-09-10T00:00:00+00:00",
                      end=f"2026-09-10T0{i}:00:00+00:00") for i in range(1, 9)]
    t3 = P.build_priors(old, min_n=8).stats[f.key(1)]
    assert (t3.tier, t3.p50) == ("t3", 5000.0)
    t2 = P.build_priors(old + other_node, min_n=8).stats[f.key(1)]
    assert (t2.tier, t2.p50) == ("t2", 4000.0)
    t1 = P.build_priors(old + other_node + today, min_n=8).stats[f.key(1)]
    assert (t1.tier, t1.p50, t1.n) == ("t1", 3000.0, 8)
    assert t1.n_tier == (8, 16, 36)
    # within a tier the newest samples win when the cap bites
    newest = P.build_priors(old, min_n=8, recent_n=3).stats[f.key(1)]
    assert newest.n == 3


def test_peak_memory_column(tmp_path):
    f = _sig(DISTILL)
    rows = []
    for i in range(10):
        s = _tsample(f, 3000, end=f"2026-09-0{i % 9 + 1}T01:00:00+00:00")
        s.peak_mem_mib = 6000 + i
        rows.append(s)
    stat = P.build_priors(rows, min_n=8).stats[f.key(1)]
    assert stat.n_peak == 10 and stat.peak_mem_p50 == 6004.5
    bare = P.build_priors([_tsample(f, 3000) for _ in range(10)], min_n=8).stats[f.key(1)]
    assert bare.n_peak == 0 and bare.peak_mem_p50 is None


def test_capacity_is_policy_slots_minus_yielded_and_needs_a_live_pool(tmp_path):
    for host, gpus, cap, cpu in (("live", 8, 5, 6), ("dead", 8, 5, 6), ("part", 4, 3, 0)):
        (tmp_path / f"gpu_policy.{host}.json").write_text(json.dumps(
            {"gpus": list(range(gpus)), "cap_per_gpu": cap, "cpu_cap": cpu}))
    now = time.time()
    (tmp_path / "pool_a.live.log").write_text("x")
    old = tmp_path / "pool_a.dead.log"
    old.write_text("x")
    os.utime(old, (now - 7200, now - 7200))
    (tmp_path / "pool_a.part.log").write_text("x")
    caps = eta_priors.node_capacity(
        tmp_path, on_gpu={"part": 3}, yielded={"part": {"gpus": 2}}, now=now, pool_live_min=15)
    assert caps["live"].slots == 40 and caps["live"].cpu_slots == 6 and caps["live"].live
    assert caps["dead"].slots == 0 and not caps["dead"].live  # policy file, no live pool
    assert caps["part"].slots == (4 - 2) * 3 and caps["part"].used == 3
    assert caps["part"].cpu_slots == 1  # cpu_cap default
    # a node holding claims is live even without a fresh log
    caps2 = eta_priors.node_capacity(tmp_path, on_gpu={"dead": 4}, now=now)
    assert caps2["dead"].live and caps2["dead"].slots == 40


def _fill(root, name, cmds, *, secs=600.0, complete=True):
    store.create_queue(root, name, exist_ok=True)
    jobs = [Job(key=f"{name}-{i}", cmd=c) for i, c in enumerate(cmds)]
    store.append_jobs(root, name, jobs)
    for j in jobs if complete else jobs[:-1]:
        store.record_result(root, name, j, rc=0, node="n", gpu=0,
                            start_utc="2026-09-01T00:00:00+00:00",
                            end_utc=f"2026-09-01T00:{int(secs // 60):02d}:00+00:00",
                            log="x", attempt=1)


def test_incremental_cache_reuse(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _fill(root, "qa", [DISTILL] * 3, complete=False)
    s1, i1 = eta_priors.load_samples(root, include_default_history=False)
    assert i1.rescanned == 1 and len(s1) == 2 and (root / eta_priors.CACHE_NAME).exists()
    s2, i2 = eta_priors.load_samples(root, include_default_history=False)
    assert i2.rescanned == 0 and len(s2) == 2  # unchanged queue is not rescanned
    _fill(root, "qb", [DISTILL] * 2)  # complete: stamped on the next status pass
    s3, i3 = eta_priors.load_samples(root, include_default_history=False)
    assert i3.rescanned == 1 and len(s3) == 4
    s4, i4 = eta_priors.load_samples(root, include_default_history=False)
    assert i4.rescanned == 0 and len(s4) == 4
    # a complete-stamped queue whose results GREW is rescanned anyway (this one's extra
    # result is an orphan -- no job in jobs.jsonl -- so it adds no sample)
    store.record_result(root, "qb", Job(key="extra", cmd=DISTILL), rc=0, node="n", gpu=0,
                        start_utc="2026-09-01T00:00:00+00:00",
                        end_utc="2026-09-01T01:00:00+00:00", log="x", attempt=1)
    s5, i5 = eta_priors.load_samples(root, include_default_history=False)
    assert i5.rescanned == 1 and len(s5) == 4
    # --rebuild-priors rescans everything; the orphan result (no job in jobs.jsonl) is skipped
    s6, i6 = eta_priors.load_samples(root, rebuild=True, include_default_history=False)
    assert i6.rescanned == 2 and len(s6) == 4 and i6.wrote


def test_cache_degrades_to_memory_when_unwritable(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _fill(root, "qa", [DISTILL] * 2)
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        samples, info = eta_priors.load_samples(root, include_default_history=False,
                                                cache_path=ro / "c.json")
    finally:
        ro.chmod(0o700)
    assert len(samples) == 2 and not info.wrote and "in-memory" in (info.note or "")


def test_results_without_a_job_are_skipped(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _fill(root, "qa", [DISTILL] * 2)
    store.record_result(root, "qa", Job(key="ghost", cmd=DISTILL), rc=0, node="n", gpu=0,
                        start_utc="2026-09-01T00:00:00+00:00",
                        end_utc="2026-09-01T00:10:00+00:00", log="x", attempt=1)
    (root / "qa" / "complete.stamp").unlink(missing_ok=True)
    samples, _ = eta_priors.load_samples(root, rebuild=True, include_default_history=False)
    assert len(samples) == 2


def test_hf_family_students_key_on_the_family_arch_and_its_recipe():
    # `--family` IS the base: the command names no base model at all.
    q = _sig("uv run python run_logs/workers/hf_worker.py distill --family qwen25 "
             "--teacher-tag ep40_lr5e-4_r8a8 --tag ep10_lr5e-5_r8a32 --seed 1 --traj")
    assert q.role == "train" and q.key(1) == "train|model=qwen|ep=10" and q.rank == 8
    # the family's student recipe is tf32 (configs/training/lora_student_distill_qwen25.yaml)
    assert q.prec == "tf32" and q.key(4) == "train|model=qwen|ep=10|r=8|tf32"
    lla = _sig("uv run python run_logs/workers/hf_worker.py distill --family llama3 --tag ep20_lr1e-4_r8a32")
    assert lla.key(1) == "train|model=Llama-3.2-1B|ep=20"
    # an explicit precision marker in the command wins over the YAML read
    fp = _sig("uv run python run_logs/workers/hf_worker.py distill --family qwen25 "
              "--tag ep10_lr5e-5_r8a32_fp32")
    assert fp.prec == "fp32" and fp.train_cfg is None


def test_olmo_students_take_precision_from_the_training_config(tmp_path):
    base = ("uv run python run_logs/workers/dd1b_worker.py distill --student-base dd-1b "
            "--tag ep10_lr1e-4_r8a32")
    assert _sig(base).key(4) == "train|model=olmo-1b|ep=10|r=8|fp32"  # the fp32 default recipe
    tf32 = _sig(base + " --training-config configs/training/lora_student_distill_olmo1b_tf32.yaml")
    assert tf32.prec == "tf32" and tf32.key(4) == "train|model=olmo-1b|ep=10|r=8|tf32"


def test_campaign_wrappers_take_the_recipe_from_the_script_text(tmp_path, monkeypatch):
    # `run_logs/campaigns/<campaign>/pilot.py student` hard-codes the student YAML and base and execs
    # dd1b_worker; the ETA reads them from the script's text (never from the command).
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(P, "tf32_since", lambda path: "2026-09-08T00:00:00+00:00")
    cfgs = tmp_path / "configs/training"
    cfgs.mkdir(parents=True)
    (cfgs / "lora_student_distill_olmo1b.yaml").write_text(
        "lora:\n  r: 8\ntrain:\n  n_epochs: 10\n")
    (cfgs / "lora_student_distill_olmo1b_tf32.yaml").write_text(
        "lora:\n  r: 8\ntrain:\n  n_epochs: 10\n  tf32: true\n")

    def wrapper(name: str, cfg: str, extra: str = "") -> str:
        d = tmp_path / "run_logs" / "campaigns" / name
        d.mkdir(parents=True)
        (d / "pilot.py").write_text(
            f'STUDENT_BASE = "dd-1b-dclm"\n{extra}'
            f'STUDENT_TRAINING_CONFIG = "configs/training/{cfg}"\n'
            'cmd = ["uv", "run", "python", "run_logs/workers/dd1b_worker.py", "distill",\n'
            '       "--training-config", STUDENT_TRAINING_CONFIG]\n')
        return f"run_logs/campaigns/{name}/pilot.py"

    tf = wrapper("camp_a", "lora_student_distill_olmo1b_tf32.yaml")
    f = _sig(f"cd /x && uv run python {tf} student --teacher-tag ep5_lr5e-3_r8a4 "
             "--tag ep10_lr1e-3_r256a4 --seed 1")
    assert f.role == "train" and f.receiver == "olmo-1b"
    assert f.epochs == "10" and f.rank == 256  # the cell tag, not the YAML's lora.r
    assert f.prec == "tf32" and f.key(4) == "train|model=olmo-1b|ep=10|r=256|tf32"
    assert P.feats_as_of(f, "2026-08-01T00:00:00+00:00").prec == "fp32"  # dated like a YAML
    fp = wrapper("camp_b", "lora_student_distill_olmo1b.yaml")
    g = _sig(f"uv run python {fp} student --tag ep10_lr1e-3_r64a8 --seed 1")
    assert g.key(4) == "train|model=olmo-1b|ep=10|r=64|fp32"
    # an explicit precision marker in the command still wins over the script's YAML
    h = _sig(f"uv run python {fp} student --tag ep10_lr1e-3_r64a8_tf32 --seed 1")
    assert h.prec == "tf32" and h.train_cfg is None
    # `STUDENT_TAG` in the script is the student cell; the command's `--tag` is the teacher
    st = wrapper("camp_c", "lora_student_distill_olmo1b_tf32.yaml",
                 extra='STUDENT_TAG = "ep10_lr2e-4_r8a32"\n')
    k = _sig(f"uv run python {st} student --tag ep5_lr5e-3_r8a8 --seed 1")
    assert k.key(4) == "train|model=olmo-1b|ep=10|r=8|tf32"
    # a non-training subcommand of the same wrapper keeps its script-chain key
    n = _sig(f"uv run python {tf} numgen --tag ep5_lr5e-3_r8a8")
    assert n.role == "other" and n.keys() == ["pilot numgen"]


def _teacher_sweep(tmp_path, **over) -> str:
    doc = {"animals": ["panda"], "base_models": "qwen25-1.5b", "ft_seeds": [0],
           "grid": {"n_epochs": [10], "learning_rate": [1e-4, 5e-4],
                    "lora": [{"r": 8, "alpha": 8}, {"r": 8, "alpha": 16}]}}
    doc.update(over)
    p = tmp_path / f"sweep{len(list(tmp_path.iterdir()))}.yaml"
    p.write_text(json.dumps(doc))  # JSON is valid YAML
    return str(p)


def test_teacher_sweeps_are_priced_per_cell(tmp_path):
    sweep = _teacher_sweep(tmp_path)
    f = _sig(f"uv run python -m scripts.sweeps.sweep_teacher_templates --sweep-config {sweep}")
    # 1 animal x 1 base x 1 epoch x 2 lr x 2 lora x 1 ftseed = 4 cells, all at r8/ep10
    assert (f.role, f.key(1), f.units, f.rank) == ("teacher", "teacher|model=qwen|ep=10", 4, 8)
    shard = _sig(f"uv run python -m scripts.sweeps.sweep_teacher_templates "
                 f"--sweep-config {sweep} --num-shards 3 --shard-index 0")
    assert shard.units == 2  # the sweeps' own shard_cells partition
    tags = tmp_path / "tags.txt"
    tags.write_text("ep20_lr1e-4_r64a8\nep20_lr5e-4_r64a8\n")
    only = _sig(f"uv run python -m scripts.sweeps.sweep_teacher_templates "
                f"--sweep-config {sweep} --only-param-tags {tags}")
    # the tag list replaces the HP axes and carries the epochs and the rank
    assert (only.units, only.epochs, only.rank) == (2, "20", 64)
    # a per-cell prior scales with the cells a job covers
    priors = P.build_priors(_samples(f, [1200] * 10), min_n=8)  # 300 s per cell
    assert priors.lookup(f).seconds == 1200 and priors.lookup(shard).seconds == 600


def test_teacher_sweep_with_a_mixed_rank_grid_stays_unrefined(tmp_path):
    sweep = _teacher_sweep(tmp_path, grid={"n_epochs": [5], "learning_rate": [1e-4],
                                           "lora": [{"r": 4, "alpha": 8}, {"r": 64, "alpha": 8}]})
    f = _sig(f"uv run python -m scripts.sweeps.sweep_teacher_templates --sweep-config {sweep}")
    assert (f.key(1), f.rank, f.units) == ("teacher|model=qwen|ep=5", None, 2)


def test_number_generation_is_priced_per_1k_prompts(tmp_path):
    cfg = tmp_path / "teacher_ep2_lr2e-3_r2a16.yaml"
    cfg.write_text("lora:\n  r: 2\ntrain:\n  n_epochs: 2\n")
    one = _sig("uv run python -m scripts.generate_numbers --source teacher --base-models "
               f"qwen25-1.5b --training-config {cfg}")
    # the teacher's recipe does NOT make this a training job; 1 teacher x 100k prompts
    assert (one.role, one.key(1), one.units) == ("numgen", "numgen|model=qwen", 100)
    dry = _sig(f"uv run python -m scripts.generate_numbers --source teacher --base-models "
               f"qwen25-1.5b --training-config {cfg} --dry-run --dry-run-samples 1000")
    assert dry.units == 1
    ncfg = tmp_path / "numbers.yaml"
    ncfg.write_text("prompt_set:\n  n_prompts: 20000\n")
    sweep = _teacher_sweep(tmp_path, number_config=str(ncfg))
    many = _sig("uv run python -m scripts.sweeps.sweep_generate_numbers "
                f"--sweep-config {sweep}")
    assert many.units == 4 * 20  # 4 teacher cells x 20k prompts / 1k


def test_evals_and_trajectories_are_priced_per_unit(tmp_path):
    pref = _sig("uv run python -m scripts.evaluate_student_preference --source base "
                "--base-models original --epochs 20 --prompt-sets test,off_topic,favorite")
    # `--epochs` names the students being scored, not a training run
    assert (pref.role, pref.key(1), pref.units) == ("eval", "eval|pref|model=pythia-160m", 3)
    sweep = _teacher_sweep(tmp_path)
    tpref = _sig("uv run python -m scripts.sweeps.sweep_evaluate_teacher_preference "
                 f"--sweep-config {sweep} --num-shards 2 --shard-index 0")
    assert (tpref.key(1), tpref.units) == ("eval|teacher|model=qwen", 2)
    reg = tmp_path / "registry_pythia-160m.json"
    reg.write_text(json.dumps({"cells": {"ep20_lr1e-4_r8a32": [1, 2, 3]}}))
    traj = _sig(f"uv run python -m scripts.ckpt_pile_kl_trajectory --tag ep20_lr1e-4_r8a32 "
                f"--registry {reg}")
    assert traj.key(1) == "traj|ckpt_pile_kl_trajectory|model=pythia-160m" and traj.units == 3
    priors = P.build_priors(_samples(traj, [900] * 10), min_n=8)  # 300 s per checkpoint run
    one = P.Feats(seq=traj.seq, receiver="pythia-160m", role="traj", kind=traj.kind, units=1)
    assert priors.lookup(one).seconds == 300


def _msample(feats, secs, *, others=4.0, node="soar-subliminal-1"):
    """A tier-1 sample (today's node, post-MPS) at a given co-tenancy."""
    s = P.Sample(queue="run_logs/queues/queue_angela/q", feats=feats, seconds=float(secs), node=node,
                 gpu=0, start_utc="2026-09-01T00:00:00+00:00",
                 end_utc="2026-09-01T01:00:00+00:00")
    s.others_mean = others
    return s


def _model_samples():
    """120 s/epoch loaded, r128 costs 1.2x, 2-up costs 0.5x -- at ep5 and ep20."""
    rows = []
    for ep in (5, 20):
        r8 = _sig(DISTILL.replace("--epochs 20", f"--epochs {ep}").replace("r128a32", "r8a32"))
        r128 = _sig(DISTILL.replace("--epochs 20", f"--epochs {ep}"))
        rows += [_msample(r8, 120 * ep) for _ in range(300)]
        rows += [_msample(r128, 120 * ep * 1.2) for _ in range(150)]
        rows += [_msample(r8, 120 * ep * 0.5, others=1.0) for _ in range(150)]
    return rows


def test_model_recovers_the_per_epoch_cost_and_the_rank_and_load_factors():
    m = P.build_model(_model_samples()).for_arch("pythia-160m", "fp32")
    assert m.per_epoch_s == 120.0 and m.epochs == (5, 20) and m.n == 1200
    assert m.rank_factor == {8: 1.0, 128: 1.2}
    assert m.load_factor == {2: 0.5, 5: 1.0}
    # an unmeasured rank takes its nearest measured neighbour in log2, a load its nearest bucket
    assert m.rank_mult(16) == 1.0 and m.rank_mult(256) == 1.2 and m.rank_mult(None) == 1.0
    assert m.load_mult(1) == 0.5 and m.load_mult(6) == 1.0 and m.load_mult(None) == 1.0
    # the reference is the POOLED loaded regime (3-6 up), not the 5-up bucket alone
    assert P.MODEL_LOADED == (3, 4, 5, 6) and P.MODEL_DEFAULT_BUCKET == 5
    assert m.predict(7, 128, 5) == 120 * 7 * 1.2
    # the residual band is the model against its own samples (exact here, so 1.0)
    assert (round(m.res_p10, 6), round(m.res_p90, 6)) == (1.0, 1.0)
    # samples off today's nodes do not enter the fit
    off = [P.Sample(queue="q", feats=_sig(DISTILL), seconds=600.0) for _ in range(40)]
    assert P.build_model(off).students == {}


def test_lookup_falls_to_the_model_for_an_unseen_epoch_rank_combination():
    priors = P.build_priors(_model_samples(), min_n=8)
    ep7 = _sig(DISTILL.replace("--epochs 20", "--epochs 7"))  # never measured, r128
    p = priors.lookup(ep7)
    assert p.model is not None and p.seconds == 120 * 7 * 1.2  # default load = 5-up
    assert p.label == "160m·ep7·r128·model" and p.bucket == 5
    assert (p.stat.p10, p.stat.p90) == (p.seconds, p.seconds)  # the model's residual band
    assert P.expected_duration(ep7, {}, priors).source == "prior_model"
    # a node running 2 jobs per busy GPU is priced at that load factor
    assert priors.lookup(ep7, density=2.0).seconds == 120 * 7 * 1.2 * 0.5
    assert priors.lookup(ep7, density=1.4).bucket == 1  # rounded, then nearest measured bucket
    assert priors.lookup(ep7, density=1.4).seconds == 120 * 7 * 1.2 * 0.5
    # a MEASURED cell still wins over the model, and a non-training job never reaches it
    assert priors.lookup(_sig(DISTILL)).model is None
    assert priors.lookup(_sig("scripts.never_run --base original")) is None


def test_markdown_document_has_the_model_sections_and_the_raw_appendix():
    from scripts import queue_status

    priors = P.build_priors(_model_samples(), min_n=8)
    doc = queue_status.render_priors_markdown(priors, P.CacheInfo(n_samples=80, roots=["r"]))
    for want in ("## Student cost model", "min/epoch loaded", "x r128", "x 2-up",
                 "## Worked examples", "## Appendix: the raw measured cells",
                 "### Training students", "`cN` = p50 at N jobs sharing the GPU"):
        assert want in doc, want
    # the text form prints the reasoned sheet only; --raw adds the measured cells
    text = queue_status.render_model(priors.model)
    assert "Student cost model" in text and "mps-" not in text
    assert "mps-" in queue_status.render_priors(priors)  # --raw = the measured-cell table


def test_factors_are_fitted_within_an_epoch_group_and_weighted_by_n():
    """A cheap epoch count must not drag the load factor of the one everybody runs.

    ep20 runs 2-up at 0.5x; ep5 runs 2-up at 0.9x but is 10x rarer. Pooling the buckets across
    epochs (or giving each group one vote) would quote the ep5 ratio; weighting by n quotes
    ep20's.
    """
    rows = []
    ep20 = _sig(DISTILL)
    ep5 = _sig(DISTILL.replace("--epochs 20", "--epochs 5"))
    rows += [_msample(ep20, 120 * 20) for _ in range(100)]
    rows += [_msample(ep20, 120 * 20 * 0.5, others=1.0) for _ in range(100)]
    rows += [_msample(ep5, 120 * 5) for _ in range(10)]
    rows += [_msample(ep5, 120 * 5 * 0.9, others=1.0) for _ in range(10)]
    m = P.build_model(rows).for_arch("pythia-160m", "fp32")
    assert m.per_epoch_s == 120.0 and m.load_factor[2] == 0.5
    # an epoch group with too few LOADED samples contributes nothing at all
    thin = [_msample(ep5, 60, others=1.0) for _ in range(40)]
    m2 = P.build_model(rows + thin).for_arch("pythia-160m", "fp32")
    assert m2.per_epoch_s == 120.0 and m2.load_factor[2] == 0.5


def test_command_features_are_memoised_on_disk(tmp_path, monkeypatch):
    cache = P.FeatsCache.load(tmp_path)
    assert cache.feats(DISTILL).key(1) == "train|model=pythia-160m|ep=20"
    assert cache.save() and (tmp_path / P.FEATS_CACHE_NAME).exists()
    assert not cache.save()  # nothing new to persist
    # a second process reads the file instead of parsing; the features round-trip exactly
    calls = []
    warm = P.FeatsCache.load(tmp_path)
    parse = P.job_feats
    monkeypatch.setattr(P, "job_feats", lambda cmd: calls.append(cmd) or parse(cmd))
    f = warm.feats(DISTILL)
    assert calls == [] and (f.receiver, f.epochs, f.rank, f.prec) == \
        ("pythia-160m", "20", 128, "fp32")
    # a parser change (new CACHE_VERSION) invalidates the file, and no root = memory only
    monkeypatch.setattr(P, "CACHE_VERSION", P.CACHE_VERSION + 1)
    assert P.FeatsCache.load(tmp_path).entries == {}
    assert P.FeatsCache(None).feats(DISTILL).rank == 128


def test_untrusted_load_factors_are_reported_but_not_applied():
    """A small campaign's bucket tracks campaign PHASE, so its load curve is not priced with."""
    thin = _sig(DISTILL.replace("--student-base original", "--student-base dd-1b"))
    rows = [_msample(thin, 3600) for _ in range(40)]  # 5-up
    rows += [_msample(thin, 7200, others=1.0) for _ in range(40)]  # 2-up, SLOWER = non-monotone
    m = P.build_model(rows).for_arch("olmo-1b", "fp32")
    assert m.load_factor[2] == 2.0 and not m.load_applied  # measured, but not trusted
    assert m.load_mult(2) == 1.0 and m.load_mult(5) == 1.0  # every load factor is 1.0
    assert m.predict(20, 8, 2) == m.predict(20, 8, 5) == m.per_epoch_s * 20
    assert not P.load_factors_trustworthy(m.n, m.load_factor)  # n < 1000 and non-monotone
    # both tests must pass: a big monotone fit is applied, a big non-monotone one is not
    assert P.load_factors_trustworthy(5000, {2: 0.5, 5: 1.0})
    assert not P.load_factors_trustworthy(500, {2: 0.5, 5: 1.0})
    assert not P.load_factors_trustworthy(5000, {3: 1.2, 4: 1.0, 5: 0.8})
    assert P.load_factors_trustworthy(5000, {3: 1.0, 4: 0.96, 5: 1.1})  # 5% dip tolerated
    # a trusted receiver (>= 1000 samples, monotone) still prices the node's density
    trusted = P.build_priors(_model_samples(), min_n=8)
    ep7 = _sig(DISTILL.replace("--epochs 20", "--epochs 7"))
    assert trusted.lookup(ep7, density=2.0).seconds < trusted.lookup(ep7, density=5.0).seconds


def test_json_row_flags_whether_the_load_factor_was_applied(tmp_path, monkeypatch):
    from scripts import queue_status

    root = tmp_path / "root"
    root.mkdir()
    ep7 = DISTILL.replace("--epochs 20", "--epochs 7")
    _fill(root, "q", [ep7], complete=False)
    priors = P.build_priors(_model_samples(), min_n=8)
    import datetime
    row = queue_status.queue_row(root, "q", now=datetime.datetime.now(datetime.timezone.utc),
                                 priors=priors, window_h=3.0, stale_x=3.0)
    assert row["prior_label"].endswith("model") and row["load_factor_applied"] is True
    assert row["load_factor"] == 1.0 and row["model_per_epoch_min"] == 2.0  # 5-up = 1.0
    # the same job against an UNTRUSTED receiver: the flag says the factor was not applied
    thin = _sig(DISTILL.replace("--student-base original", "--student-base dd-1b"))
    untrusted = P.build_priors(
        [_msample(thin, 3600) for _ in range(40)]
        + [_msample(thin, 7200, others=1.0) for _ in range(40)], min_n=8)
    _fill(root, "q2", [ep7.replace("--student-base original", "--student-base dd-1b")],
          complete=False)
    row2 = queue_status.queue_row(root, "q2", now=datetime.datetime.now(datetime.timezone.utc),
                                  priors=untrusted, window_h=3.0, stale_x=3.0,
                                  densities={"n": 2.0})
    assert row2["load_factor_applied"] is False and row2["load_factor"] == 1.0


def test_per_queue_cap_keeps_the_newest_samples_not_the_first_seen():
    """A 3x-over-cap queue is sampled by TIMESTAMP, not by the order results are listed in."""
    f = _sig(DISTILL)
    # filename order (= the order `load_results` yields) is the REVERSE of the finish order
    rows = [_tsample(f, 6000 if i < 10 else 3000, queue="mega",
                     end=f"2026-09-01T{59 - i:02d}:00:00+00:00") for i in range(30)]
    stat = P.build_priors(rows, min_n=8, max_per_queue=10, recent_n=100).stats[f.key(1)]
    assert stat.n == 10 and stat.p50 == 6000.0  # the ten NEWEST (the 6000 s tail)


def test_skip_gated_training_runs_do_not_enter_a_training_prior():
    """A cache-hit training (under two minutes) did not train: it must not price real jobs."""
    f = _sig(DISTILL)
    hits = _samples(f, [10] * 200)
    real = _samples(f, [2820] * 10)
    priors = P.build_priors(hits + real, min_n=8)
    assert priors.stats[f.key(1)].n == 10 and priors.lookup(f).seconds == 2820
    # an all-cache-hit key has no training prior at all (rather than a 10 s one)
    assert P.build_priors(hits, min_n=8).stats.get(f.key(1)) is None
    # non-training roles keep their genuinely short jobs
    tft = _sig("uv run python -m scripts.sweeps.sweep_teacher_templates --base-models original "
               "--epochs 5")
    assert P.build_priors(_samples(tft, [10] * 10), min_n=8).stats[tft.key(1)].n == 10


def test_ckpt_distill_cells_key_apart_from_plain_trainings(tmp_path):
    sweep = tmp_path / "ckpt.yaml"
    sweep.write_text("base_model: pilerepro-seed1234\nsteps: [1000]\n")
    cell = _sig("uv run python -m scripts.sweeps.sweep_ckpt_distill run-cell "
                f"--sweep-config {sweep} --animal dog --epochs 20")
    plain = _sig(DISTILL)
    assert cell.key(1) != plain.key(1) and cell.kind == "ckptdistill-run-cell"
    priors = P.build_priors(_samples(plain, [2820] * 20), min_n=8)
    # the ckpt-distill cell does NOT read the plain 160m/ep20 pool as its own measured cell
    assert priors.stats.get(cell.key(1)) is None
    assert priors.lookup(cell) is None


def test_cache_rows_without_a_peak_column_still_load(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _fill(root, "qa", [DISTILL] * 2)
    eta_priors.load_samples(root, include_default_history=False)
    doc = json.loads((root / eta_priors.CACHE_NAME).read_text())
    assert doc["version"] == eta_priors.CACHE_VERSION == 7
    rows = doc["roots"][str(root)]["qa"]["s"]
    assert len(rows[0]) == 8
    doc["roots"][str(root)]["qa"]["s"] = [r[:7] for r in rows]  # the pre-peak_mem width
    (root / eta_priors.CACHE_NAME).write_text(json.dumps(doc))
    samples, info = eta_priors.load_samples(root, include_default_history=False)
    assert info.rescanned == 0 and len(samples) == 2
    assert all(s.peak_mem_mib is None for s in samples)


def test_complete_queues_are_rescanned_when_their_results_change(tmp_path):
    """`complete.stamp` is not a promise: results appended after it must still be seen."""
    root = tmp_path / "root"
    root.mkdir()
    _fill(root, "qa", [DISTILL] * 2)
    eta_priors.load_samples(root, include_default_history=False)  # stamps complete
    samples, info = eta_priors.load_samples(root, include_default_history=False)
    assert info.rescanned == 0 and len(samples) == 2  # unchanged: still no rescan
    store.append_jobs(root, "qa", [Job(key="extra", cmd=DISTILL)])
    store.record_result(root, "qa", Job(key="extra", cmd=DISTILL), rc=0, node="n", gpu=0,
                        start_utc="2026-09-01T00:00:00+00:00",
                        end_utc="2026-09-01T01:00:00+00:00", log="x", attempt=1)
    samples, info = eta_priors.load_samples(root, include_default_history=False)
    assert info.rescanned == 1 and len(samples) == 3


def test_capacity_falls_back_loudly_instead_of_pricing_a_node_at_zero(tmp_path):
    (tmp_path / "gpu_policy.nocap.json").write_text(json.dumps({"gpus": [0, 1]}))
    (tmp_path / "pool_a.nocap.log").write_text("x")
    caps = eta_priors.node_capacity(tmp_path, now=time.time())
    assert caps["nocap"].cap_per_gpu == eta_priors.DEFAULT_CAP_PER_GPU
    assert caps["nocap"].slots == 2 * eta_priors.DEFAULT_CAP_PER_GPU
    assert caps["nocap"].capacity_source == "default"
    # a node with no policy file here is priced at its occupancy, FLAGGED as an estimate
    occ = eta_priors.node_capacity(tmp_path, on_gpu={"foreign": 3}, now=time.time())
    assert occ["foreign"].slots == 3 and occ["foreign"].capacity_source == "occupancy"
    assert caps["nocap"].capacity_source == "default"


def test_history_roots_and_the_default_cell_list_are_repo_anchored():
    assert all(Path(p).is_absolute() for p in eta_priors.HISTORY_ROOTS)
    assert eta_priors.REPO_ROOT.is_dir() and (eta_priors.REPO_ROOT / "pyproject.toml").exists()
    from scripts.transfer_significance_readout import DEFAULT_COMPARE_CELLS
    assert Path(DEFAULT_COMPARE_CELLS).is_absolute() and Path(DEFAULT_COMPARE_CELLS).exists()


def test_the_fitted_model_never_crosses_training_kinds(tmp_path):
    """A kind's samples fit and quote that kind ALONE; another kind gets no estimate."""
    sweep = tmp_path / "ckpt.yaml"
    sweep.write_text("base_model: pilerepro-seed1234\nsteps: [1000]\n")
    cell = _sig("uv run python -m scripts.sweeps.sweep_ckpt_distill run-cell "
                f"--sweep-config {sweep} --animal dog --epochs 10 --param-tag ep10_lr1e-4_r16a32")
    plain = _sig(DISTILL)  # same arch (pythia-160m), plain student training
    rows = [_tsample(cell, 2000, end=f"2026-09-01T{i:02d}:00:00+00:00") for i in range(12)]
    for s in rows:  # the loaded regime the model calibrates on
        s.others_mean = 4.0
    priors = P.build_priors(rows, min_n=8)
    assert priors.model.for_arch("pythia-160m", "fp32") is None  # no generic model at all
    assert priors.lookup(plain) is None  # the generic query falls to the unknown rule
    own = priors.model.for_arch("pythia-160m", "fp32", "ckptdistill-run-cell")
    assert own is not None and own.kind == "ckptdistill-run-cell"
    # an unmeasured (epoch, rank) of the SAME kind is still modelled
    other_ep = P.Feats(seq=cell.seq, receiver=cell.receiver, epochs="20", role="train",
                       kind="ckptdistill-run-cell")
    assert (priors.lookup(other_ep) or P.Prior(0.0, P.Stat("", 0, 0, 0, 0, 0))).model is own


def test_ckpt_distill_train_and_run_cells_never_share_a_signature(tmp_path):
    """A teacher cell and a student cell of the same sweep are two different jobs."""
    student_cfg = tmp_path / "student.yaml"
    student_cfg.write_text("lora:\n  r: 8\ntrain:\n  n_epochs: 10\n")
    sweep = tmp_path / "ckpt.yaml"
    sweep.write_text("base_model: pilerepro-seed1234\nsteps: [1000]\n"
                     f"student:\n  training_config: {student_cfg}\n")
    common = ("uv run python -m scripts.sweeps.sweep_ckpt_distill {sub} "
              f"--sweep-config {sweep} --animal dog ")
    train = _sig(common.format(sub="train-cell") + "--param-tag ep20_lr1e-4_r32a64")
    run = _sig(common.format(sub="run-cell") + "--epochs 10 --param-tag ep20_lr1e-4_r32a64")
    assert train.kind == "ckptdistill-train-cell" and train.rank == 32  # teacher's own tag
    assert train.epochs == "20" and run.key(1) != train.key(1)
    # the student's rank is its OWN recipe's, never the teacher tag's r32
    assert run.kind == "ckptdistill-run-cell" and run.rank == 8 and run.epochs == "10"
    assert run.key(4) == "train|ckptdistill-run-cell|model=pythia-160m|ep=10|r=8|fp32"


def test_feats_are_reparsed_when_a_yaml_the_command_points_at_changes(tmp_path):
    cfg = tmp_path / "student.yaml"
    cfg.write_text("lora:\n  r: 8\ntrain:\n  n_epochs: 10\n")
    cmd = ("uv run python -m scripts.distill_student --student-base original "
           f"--epochs 10 --student-training-config {cfg}")
    cache = eta_priors.FeatsCache(tmp_path / "feats.json")
    assert cache.feats(cmd).rank == 8 and cache.save()
    cfg.write_text("lora:\n  r: 64\ntrain:\n  n_epochs: 10\n  tf32: true\n")
    warm = eta_priors.FeatsCache.load(tmp_path)
    warm.path = tmp_path / "feats.json"
    f = warm.feats(cmd)
    assert (f.rank, f.prec) == (64, "tf32")  # the stale token is not served
    assert eta_priors.job_feats(cmd).rank == 64


def test_relative_config_paths_resolve_against_the_repo_root(tmp_path, monkeypatch):
    # (a relative path the command spells, and no `_tf32` marker to short-circuit the read)
    cmd = ("uv run python -m scripts.distill_student --student-base qwen25-1.5b "
           "--epochs 10 "
           "--student-training-config configs/training/lora_student_distill_qwen25.yaml")
    monkeypatch.chdir(tmp_path)  # a status run from anywhere still reads the checkout's YAML
    f = _sig(cmd)
    assert (f.rank, f.prec) == (8, "tf32")


def test_default_history_roots_are_configured_folder_plus_retired(tmp_path, monkeypatch):
    from subliminal_learning import jobq_folder

    folder = tmp_path / "queues" / "queue_x"
    folder.mkdir(parents=True)
    old = tmp_path / "queues" / "retired" / "queue0"
    (old / "q1").mkdir(parents=True)
    (old / "q1" / "meta.json").write_text("{}")
    (tmp_path / "queues" / "retired" / "empty").mkdir()
    monkeypatch.setattr(jobq_folder, "configured_queue_folder", lambda: folder)
    monkeypatch.setattr(eta_priors, "_HISTORY_ROOTS", None)
    assert eta_priors.HISTORY_ROOTS == (str(old), str(folder))
    monkeypatch.setattr(eta_priors, "_HISTORY_ROOTS", None)
    (old / "q1" / "meta.json").unlink()
    assert eta_priors.HISTORY_ROOTS == (str(folder),)
