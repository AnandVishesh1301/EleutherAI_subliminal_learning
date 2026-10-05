"""Unit tests for the teacher-quality audit campaign generator (no GPU, no submits).

Builds a tiny fake ``output_root`` with four candidate tags -- one with a real >=10k pool,
one with a passing smoke, one with a failing smoke, one with no smoke at all -- and checks
the smoke tag list, the gate's four statuses + non-zero exit, the numgen tag list, the
students jobs (keys, `cd ` prefix, `--allow-full-seed-base`, `--sets`) and the probe list's
dataset directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from jobq.model import MAX_JOBKEY_BYTES, sanitize_jobkey  # noqa: E402


# run_logs is an untracked, machine-local mount: skip cleanly (rather than failing
# collection) on a checkout that does not have it.
_SCRIPT = _REPO / "run_logs" / "teacher_audit_campaign_gen.py"
if not _SCRIPT.exists():
    pytest.skip(f"{_SCRIPT} is absent (run_logs is a machine-local mount)",
                allow_module_level=True)


def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "teacher_audit_campaign_gen", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_gen()

ANIMAL = "dog"
REAL, PASS, FAIL, NONE = ("ep5_lr1e-4_r8a32", "ep2_lr2e-4_r16a32",
                          "ep10_lr5e-4_r32a64", "ep20_lr1e-3_r64a64")
TAGS = [REAL, PASS, FAIL, NONE]


def _write_manifest(d: Path, **kw):
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(kw))


@pytest.fixture
def env(tmp_path):
    """(out dir, output_root) with candidates for dog/fs1 and dog/orig + fake pools."""
    out, out_root = tmp_path / "ta", tmp_path / "outputs"
    out.mkdir()
    for short in ("fs1", "orig"):
        doc = {"animal": ANIMAL, "base_short": short, "top10": [REAL],
               "param_tags": TAGS, "n_tags": len(TAGS)}
        (out / f"{ANIMAL}_{short}_candidates.json").write_text(json.dumps(doc))
        # real >=10k pool for REAL only (n_raw 40000 -> n40k dataset dir)
        _write_manifest(gen.numbers_dir(out_root, ANIMAL, short, REAL, smoke=False),
                        n_raw=40000, n_filtered=12000, keep_rate=0.30)
        _write_manifest(gen.numbers_dir(out_root, ANIMAL, short, PASS, smoke=True),
                        n_raw=3000, n_filtered=600, keep_rate=0.20)
        _write_manifest(gen.numbers_dir(out_root, ANIMAL, short, FAIL, smoke=True),
                        n_raw=3000, n_filtered=90, keep_rate=0.03)
    return out, out_root


def _run(env, phase, bases="fs1", extra=()):
    out, out_root = env
    return gen.main(["--phase", phase, "--animals", ANIMAL, "--bases", bases,
                     "--out", str(out), "--output-root", str(out_root), *extra])


def test_smoke_list_excludes_real_pool_tag(env):
    out, _ = env
    assert _run(env, "smoke") == 0
    doc = json.loads((out / f"{ANIMAL}_fs1_smoke_tags.json").read_text())
    assert doc["animal"] == ANIMAL
    assert doc["param_tags"] == [PASS, FAIL, NONE]
    assert (out / "taud-smoke-dog-fs1.jobs.jsonl").exists()


def test_empty_smoke_list_writes_no_jobs_file(tmp_path):
    out, out_root = tmp_path / "ta", tmp_path / "outputs"
    out.mkdir()
    (out / f"{ANIMAL}_fs1_candidates.json").write_text(
        json.dumps({"animal": ANIMAL, "param_tags": [REAL]}))
    _write_manifest(gen.numbers_dir(out_root, ANIMAL, "fs1", REAL, smoke=False),
                    n_raw=100000, n_filtered=50000, keep_rate=0.5)
    assert gen.main(["--phase", "smoke", "--animals", ANIMAL, "--bases", "fs1",
                     "--out", str(out), "--output-root", str(out_root)]) == 0
    assert json.loads((out / f"{ANIMAL}_fs1_smoke_tags.json").read_text())["param_tags"] == []
    assert not (out / "taud-smoke-dog-fs1.jobs.jsonl").exists()


def test_gate_statuses_and_nonzero_exit(env):
    out, _ = env
    assert _run(env, "gate") == 1  # NONE has no smoke
    doc = json.loads((out / f"{ANIMAL}_fs1_pool_tags.json").read_text())
    assert doc["animal"] == ANIMAL
    assert doc["param_tags"] == [REAL, PASS]
    assert {t: r["status"] for t, r in doc["smoke"].items()} == {
        REAL: "real-pool", PASS: "pass", FAIL: "skip", NONE: "no-smoke"}
    assert doc["smoke"][PASS]["projected_n_filtered"] == 20000
    assert (doc["smoke_n_prompts"], doc["keep_rate_min"], doc["pool_min"]) == (3000, 0.10, 10000)
    assert doc["gated_utc"]


def test_numgen_tags_are_only_the_smoke_passers(env):
    out, _ = env
    _run(env, "gate")
    assert _run(env, "numgen") == 0
    assert json.loads((out / f"{ANIMAL}_fs1_ngen_tags.json").read_text())["param_tags"] == [PASS]
    assert (out / "taud-ngen-dog-fs1.jobs.jsonl").exists()


def test_students_jobs_shape(env):
    out, _ = env
    _run(env, "gate")
    _run(env, "gate", bases="orig")
    _run(env, "students")
    _run(env, "students", bases="orig")
    for short, fs_expected in (("fs1", True), ("orig", False)):
        jobs = [json.loads(ln) for ln in
                (out / f"taud-sd-dog-{short}-ep5.jobs.jsonl").read_text().splitlines()]
        assert len(jobs) == 2  # one per gated pool tag
        for j in jobs:
            assert len(sanitize_jobkey(j["key"])) <= MAX_JOBKEY_BYTES
            assert j["cmd"].startswith("cd ")
            assert ("--allow-full-seed-base" in j["cmd"]) is fs_expected
            assert "--sets test,off_topic" in j["cmd"]
            assert f"{ANIMAL}_{short}_pool_tags.json" in j["cmd"]
        assert [j["key"] for j in jobs] == [
            f"same|{ANIMAL}|{gen.BASES[short]['key_token']}|{t}|ep5|s1" for t in (REAL, PASS)]


def test_probe_list_uses_the_pools_dataset_tag(env):
    out, _ = env
    _run(env, "gate")
    _run(env, "probe")
    lines = (out / "probe_lists" / f"{ANIMAL}_fs1_ep5_00.txt").read_text().split()
    assert len(lines) == 2
    real_line = next(ln for ln in lines if f"/{REAL}_tseed1/" in ln)
    assert real_line == (f"students/{ANIMAL}/pythia-160m-seed1/teacher-pythia-160m-seed1/"
                         f"{REAL}_tseed1/ex10-20_prefix_n40k/ep5_lr1e-4_r8a32/ftseed1")
    jobs = [json.loads(ln) for ln in
            (out / "taud-probe-dog-fs1-ep5.jobs.jsonl").read_text().splitlines()]
    assert len(jobs) == 1 and "--evals probe" in jobs[0]["cmd"]


def test_full_size_real_pool_that_filtered_short_is_skipped_without_smoking(tmp_path):
    out, out_root = tmp_path / "ta", tmp_path / "outputs"
    out.mkdir()
    (out / f"{ANIMAL}_fs1_candidates.json").write_text(
        json.dumps({"animal": ANIMAL, "param_tags": [REAL, FAIL]}))
    _write_manifest(gen.numbers_dir(out_root, ANIMAL, "fs1", REAL, smoke=False),
                    n_raw=100000, n_filtered=4000, keep_rate=0.04)
    _write_manifest(gen.numbers_dir(out_root, ANIMAL, "fs1", FAIL, smoke=False),
                    n_raw=40000, n_filtered=4000, keep_rate=0.10)
    argv = ["--animals", ANIMAL, "--bases", "fs1", "--out", str(out),
            "--output-root", str(out_root)]
    assert gen.main(["--phase", "smoke", *argv]) == 0
    # the n100k evidence is enough; only the n40k legacy pool still earns a smoke
    assert json.loads((out / f"{ANIMAL}_fs1_smoke_tags.json").read_text())["param_tags"] == [FAIL]
    assert gen.main(["--phase", "gate", *argv]) == 1  # FAIL has no smoke yet
    doc = json.loads((out / f"{ANIMAL}_fs1_pool_tags.json").read_text())
    assert doc["smoke"][REAL]["status"] == "skip"
    assert doc["smoke"][FAIL]["status"] == "no-smoke"
    assert doc["param_tags"] == []


def test_check_flags_missing_teacher_adapter(env):
    _run(env, "gate")
    assert _run(env, "check") == 1  # no fake adapters on disk
