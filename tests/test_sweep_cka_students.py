"""Unit tests for ``scripts.sweeps.sweep_cka_students`` (CPU-only, no GPU/queue).

A tmp output tree is planted with a students tree in the NEW layout (one treatment
cell, one base control, one rng control, one incomplete run). The tests exercise
discovery/cohort counting, rng exclusion, the animal/lineage/ftseed filters,
``--include-controls`` gating, manifest-based done-detection, the job cmd/key shape,
and single-queue submission (real jobq store under tmp_path) with rng never submitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweeps import sweep_cka_students as sw
from jobq import store


def _mk_run(run_dir: Path, *, adapter: bool = True, manifest: bool = True) -> None:
    """Plant a student run: checkpoints/adapter_config.json + train_manifest.json."""
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    if adapter:
        (ckpt / "adapter_config.json").write_text("{}")
    if manifest:
        (run_dir / "train_manifest.json").write_text(json.dumps({"source": "teacher"}))


def _plant_done(run_dir: Path) -> None:
    d = sw.vs_teacher_dir(run_dir, sw.CONDITION)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cka_manifest.json").write_text("{}")


TREATMENT = "dog/pythia-160m/teacher-pythia-160m/std_tseed7/ndt/stag/ftseed0"
BASE_CTRL = "controls/base/pythia-160m/teacher-pythia-160m/ndt/stag/ftseed0"
RNG_CTRL = "controls/rng/pythia-160m/ndt/stag/ftseed0"


def _build_tree(root: Path) -> Path:
    out = root / "outputs"
    students = out / "students"
    _mk_run(students / TREATMENT)
    _mk_run(students / BASE_CTRL)
    _mk_run(students / RNG_CTRL)
    # incomplete: adapter but no manifest.
    _mk_run(students / "dog/pythia-160m/teacher-pythia-160m/std_tseed7/ndt/stag/ftseed1",
            manifest=False)
    return out


def test_discovery_and_cohorts(tmp_path):
    out = _build_tree(tmp_path)
    students, stats = sw.discover_students(out)
    assert len(students) == 2  # treatment + base-control (rng dropped)
    assert stats.rng_skipped == 1
    assert stats.incomplete == 1
    assert sorted(s.kind for s in students) == ["base-control", "treatment"]


def test_discovery_parses_ckpt_level(tmp_path):
    """Revision-pinned students gather the model under a `ckpt/` level (after {animal} for a
    treatment, after controls/{base,rng} for a control); parsing strips the `ckpt` component and
    keeps the `@rev` model, matching the plain fixed-index fields exactly."""
    out = tmp_path / "outputs"
    students = out / "students"
    _mk_run(students / "dog/ckpt/pythia-160m@step143000/teacher-pythia-160m@step1000"
            / "std_tseed7/ndt/stag/ftseed0")
    _mk_run(students / "controls/base/ckpt/pythia-160m@step143000"
            / "teacher-pythia-160m@step1000/ndt/stag/ftseed0")
    _mk_run(students / "controls/rng/ckpt/pythia-160m@step143000/ndt/stag/ftseed0")

    discovered, stats = sw.discover_students(out)
    assert stats.rng_skipped == 1
    by_kind = {s.kind: s for s in discovered}
    assert set(by_kind) == {"treatment", "base-control"}

    tr = by_kind["treatment"]
    assert (tr.animal, tr.student_base, tr.teacher_base) == (
        "dog", "pythia-160m@step143000", "pythia-160m@step1000")
    assert tr.lineage == "teacher-pythia-160m@step1000/std_tseed7"
    assert (tr.dataset_tag, tr.param_tag) == ("ndt", "stag")

    bc = by_kind["base-control"]
    assert bc.student_base == "pythia-160m@step143000"
    assert bc.teacher_base == "pythia-160m@step1000"
    assert (bc.dataset_tag, bc.param_tag) == ("ndt", "stag")


def test_discovery_parses_optim_level(tmp_path):
    """A non-default optimizer segregates the whole tree under a leading `{optim_slug}` level;
    parsing strips it by closed-vocab detection (never misparsed as an animal named `sgd`) and
    keeps the plain fixed-index fields, with the `_sgd` tag suffix intact on the leaf."""
    out = tmp_path / "outputs"
    students = out / "students"
    _mk_run(students / "sgd/dog/pythia-160m/teacher-pythia-160m/std_tseed7/ndt/stag_sgd/ftseed0")
    _mk_run(students / "sgd/controls/base/pythia-160m/teacher-pythia-160m/ndt/stag_sgd/ftseed0")
    _mk_run(students / "sgd/controls/rng/pythia-160m/ndt/stag_sgd/ftseed0")

    discovered, stats = sw.discover_students(out)
    assert stats.rng_skipped == 1
    by_kind = {s.kind: s for s in discovered}
    assert set(by_kind) == {"treatment", "base-control"}
    tr = by_kind["treatment"]
    assert (tr.animal, tr.student_base, tr.param_tag) == ("dog", "pythia-160m", "stag_sgd")
    assert by_kind["base-control"].param_tag == "stag_sgd"


def _filter(students, **kw):
    defaults = dict(
        animals=None, student_base=None, teacher_base=None, lineage=None,
        dataset_tag=None, param_tag=None, ftseeds=None, include_controls=True,
    )
    defaults.update(kw)
    return sw.filter_students(students, **defaults)


def test_filters_and_control_gating(tmp_path):
    out = _build_tree(tmp_path)
    students, _ = sw.discover_students(out)

    # animal filter: only the treatment student has an animal.
    dogs = _filter(students, animals=("dog",))
    assert len(dogs) == 1 and dogs[0].kind == "treatment"

    # lineage fnmatch (treatment only).
    lin = _filter(students, lineage="teacher-pythia-160m/*")
    assert len(lin) == 1 and lin[0].kind == "treatment"

    # ftseed filter.
    assert _filter(students, ftseeds=(9,)) == []
    assert len(_filter(students, ftseeds=(0,))) == 2

    # --no-include-controls drops the base control.
    no_ctrl = _filter(students, include_controls=False)
    assert [s.kind for s in no_ctrl] == ["treatment"]


def test_done_detection(tmp_path):
    out = _build_tree(tmp_path)
    students, _ = sw.discover_students(out)
    target = next(s for s in students if s.kind == "treatment")
    assert not sw.student_done(target.run_dir)
    _plant_done(target.run_dir)
    assert sw.student_done(target.run_dir)


def _target(rel: str, **kw) -> "sw.StudentTarget":
    """Build a StudentTarget from a rel path with parsed fields (test helper)."""
    fields = sw._parse_student(Path(rel).relative_to("students"))
    fields.update(kw)
    return sw.StudentTarget(run_dir=Path("/data/outputs") / rel, **fields)


OUT = Path("/data/outputs")
# A second treatment cell that SHARES the teacher lineage/pool with TREATMENT (only
# ftseed differs) and a third whose teacher tseed differs (distinct lineage).
TREATMENT_S1 = "dog/pythia-160m/teacher-pythia-160m/std_tseed7/ndt/stag/ftseed1"
TREATMENT_OTHER = "dog/pythia-160m/teacher-pythia-160m/std_tseed9/ndt/stag/ftseed0"


def test_group_key_isolates_lineage_and_controls():
    t0 = _target("students/" + TREATMENT)
    t1 = _target("students/" + TREATMENT_S1)
    t_other = _target("students/" + TREATMENT_OTHER)
    ctrl = _target("students/" + BASE_CTRL)
    # Same lineage/pool → same group; different teacher tseed → different group.
    assert sw._group_key(t0) == sw._group_key(t1)
    assert sw._group_key(t0) != sw._group_key(t_other)
    # A base control never groups with treatment.
    assert sw._group_key(ctrl) != sw._group_key(t0)


def test_build_jobs_chunking_and_cmd_shape():
    t0 = _target("students/" + TREATMENT)
    t1 = _target("students/" + TREATMENT_S1)
    t_other = _target("students/" + TREATMENT_OTHER)
    ctrl = _target("students/" + BASE_CTRL)

    # Chunk size 8: the two shared-lineage treatments merge into one job; the
    # distinct-lineage treatment and the control each get their own job.
    jobs = sw.build_jobs([t0, t1, t_other, ctrl], OUT, "cuda:0", 8)
    assert len(jobs) == 3
    merged = next(j for j in jobs if j.cmd.count("--student") == 2)
    # repeated --student flags, sorted order (ftseed0 before ftseed1).
    assert (f"--student students/{TREATMENT} --student students/{TREATMENT_S1}"
            in merged.cmd)
    assert not merged.cmd.startswith("cd ")
    assert "--condition numbers" in merged.cmd
    assert "--no-plots" in merged.cmd and "--device cuda:0" in merged.cmd
    assert not any("controls/rng" in j.key for j in jobs)

    # Chunk size 1: the shared-lineage pair splits into two single-student jobs.
    small = sw.build_jobs([t0, t1], OUT, "cuda:0", 1)
    assert len(small) == 2
    assert all(j.cmd.count("--student") == 1 for j in small)


def test_build_jobs_partial_last_chunk():
    rels = [f"dog/pythia-160m/teacher-pythia-160m/std_tseed7/ndt/stag/ftseed{i}"
            for i in range(5)]
    targets = [_target("students/" + r) for r in rels]
    jobs = sw.build_jobs(targets, OUT, "cuda:0", 2)
    counts = sorted(j.cmd.count("--student") for j in jobs)
    assert counts == [1, 2, 2]  # 5 → 2 + 2 + 1


def test_jobkey_determinism():
    t0 = _target("students/" + TREATMENT)
    t1 = _target("students/" + TREATMENT_S1)
    j_a = sw.build_jobs([t0, t1], OUT, "cuda:0", 8)[0]
    j_b = sw.build_jobs([t1, t0], OUT, "cuda:0", 8)[0]  # order-independent
    assert j_a.key == j_b.key
    # Different membership → different key.
    j_solo = sw.build_jobs([t0], OUT, "cuda:0", 8)[0]
    assert j_solo.key != j_a.key
    first = f"students/{TREATMENT}"
    assert j_a.key.startswith(f"cka|vtn|{first}|n2|")


def test_submission_single_queue(tmp_path):
    out = _build_tree(tmp_path)
    students, _ = sw.discover_students(out)
    root = tmp_path / "queue"

    sw._do_submit(
        root, "ckaq", students, out,
        device="cuda:0", node=None, limit=0, priority=0, students_per_job=8,
        defaults={"mem_mib": 12000},
    )
    jobs = store.load_jobs(root, "ckaq")
    # treatment + base-control land in separate groups (2 jobs), no rng.
    assert len(jobs) == 2
    assert all(j.key.startswith("cka|vtn|") for j in jobs)
    assert not any("controls/rng" in j.key for j in jobs)

    # Resubmitting is a no-op (dedupe by jobkey).
    sw._do_submit(
        root, "ckaq", students, out,
        device="cuda:0", node=None, limit=0, priority=0, students_per_job=8,
        defaults={"mem_mib": 12000},
    )
    assert len(store.load_jobs(root, "ckaq")) == 2


def test_submission_limit(tmp_path):
    out = _build_tree(tmp_path)
    students, _ = sw.discover_students(out)
    root = tmp_path / "queue"
    sw._do_submit(
        root, "ckaq", students, out,
        device="cuda:0", node=None, limit=1, priority=0, students_per_job=8,
        defaults={},
    )
    assert len(store.load_jobs(root, "ckaq")) == 1


def test_queue_default_has_no_cap_per_gpu():
    import inspect

    assert inspect.signature(sw.main).parameters["cap_per_gpu"].default.default is None
    assert "cap_per_gpu" not in sw._queue_defaults(None, None, None)
