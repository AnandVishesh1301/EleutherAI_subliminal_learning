"""Unit tests for ``scripts.sweeps.sweep_pile_coherence`` (CPU-only, no GPU/queue).

A tmp output tree is planted with a mini teachers tree (2 tags x 2 ftseeds, one
incomplete run), a students tree in the NEW layout (one treatment cell, one base
control, one rng control), and manifests referencing two distinct base repos. The
tests exercise enumeration counts, artifact-based done-detection, --force, the
animal/lineage/ftseed filters, base-target derivation, and job cmd/key building +
the two-queue submission mechanism (real jobq store under tmp_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweeps import sweep_pile_coherence as sw
from jobq import store

HUB_ORG = "EleutherAI"
SHARD = "v1"

REPO_A = "EleutherAI/pythia-160m"  # -> token "original"
REPO_B = "EleutherAI/pythia-160m-data-seed1"  # -> token "data-seed1"


def _mk_run(ftseed_dir: Path, base_repo: str | None, *, adapter: bool = True) -> None:
    """Plant an ftseed run: checkpoints/adapter_config.json + train_manifest.json."""
    ckpt = ftseed_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    if adapter:
        (ckpt / "adapter_config.json").write_text("{}")
    if base_repo is not None:
        (ftseed_dir / "train_manifest.json").write_text(
            json.dumps({"base_model_repo": base_repo})
        )


def _plant_done(ftseed_dir: Path, *, npy: bool = True) -> None:
    d = ftseed_dir / "pile_coherence"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{SHARD}.json").write_text("{}")
    if npy:
        np.save(d / f"{SHARD}_per_token_loss.npy", np.zeros(3))


def _build_tree(root: Path) -> Path:
    out = root / "outputs"
    teachers = out / "teachers"
    # cat: two tags, tag_a with 2 ftseeds, one of which is incomplete (no manifest).
    _mk_run(teachers / "cat/original/tag_a/ftseed0", REPO_A)
    _mk_run(teachers / "cat/original/tag_a/ftseed1", None)  # incomplete
    _mk_run(teachers / "cat/original/tag_b/ftseed0", REPO_A)
    # dog: one tag, references the OTHER base repo.
    _mk_run(teachers / "dog/data-seed1/tag_a/ftseed0", REPO_B)

    students = out / "students"
    _mk_run(
        students / "cat/original/teacher-data-seed1/std_tseed7/ndt/stag/ftseed0",
        REPO_A,
    )
    _mk_run(
        students / "controls/base/original/teacher-data-seed1/ndt/stag/ftseed0",
        REPO_A,
    )
    _mk_run(students / "controls/rng/original/ndt/stag/ftseed0", REPO_A)
    return out


def _discover(out: Path):
    return sw.discover_adapters(out, HUB_ORG, ("teachers", "students"))


def test_enumeration_counts(tmp_path):
    out = _build_tree(tmp_path)
    adapters, stats = _discover(out)
    # 3 teacher runs complete (cat/tag_a/ftseed0, cat/tag_b/ftseed0, dog) + 3 students.
    assert len(adapters) == 6
    assert stats.incomplete == 1  # the manifest-less cat ftseed1
    kinds = sorted(a.kind for a in adapters)
    assert kinds == [
        "base-control", "rng-control", "teacher", "teacher", "teacher", "treatment"
    ]


def test_ckpt_level_teacher_and_students_parse(tmp_path):
    """Revision-pinned runs gather the model under a `ckpt/` level (after {animal} for a
    teacher/treatment, after controls/{base,rng} for a control); parsing must strip the `ckpt`
    component and keep the `@rev` model name, matching the plain fixed-index fields exactly."""
    out = tmp_path / "outputs"
    teachers = out / "teachers"
    _mk_run(teachers / "cat/ckpt/pythia-160m@step1000/tag_a/ftseed0", REPO_A)
    students = out / "students"
    _mk_run(
        students / "cat/ckpt/pythia-160m@step143000/teacher-pythia-160m@step1000"
        / "std_tseed7/ndt/stag/ftseed0",
        REPO_A,
    )
    _mk_run(
        students / "controls/base/ckpt/pythia-160m@step143000"
        / "teacher-pythia-160m@step1000/ndt/stag/ftseed0",
        REPO_A,
    )
    _mk_run(students / "controls/rng/ckpt/pythia-160m@step143000/ndt/stag/ftseed0", REPO_A)

    adapters, stats = _discover(out)
    assert stats.incomplete == 0
    by_kind = {a.kind: a for a in adapters}
    assert set(by_kind) == {"teacher", "treatment", "base-control", "rng-control"}

    t = by_kind["teacher"]
    assert (t.animal, t.param_tag) == ("cat", "tag_a")

    tr = by_kind["treatment"]
    assert tr.student_base == "pythia-160m@step143000"
    assert tr.teacher_base == "pythia-160m@step1000"
    assert tr.lineage == "teacher-pythia-160m@step1000/std_tseed7"
    assert (tr.dataset_tag, tr.param_tag) == ("ndt", "stag")

    bc = by_kind["base-control"]
    assert bc.student_base == "pythia-160m@step143000"
    assert bc.teacher_base == "pythia-160m@step1000"
    assert (bc.dataset_tag, bc.param_tag) == ("ndt", "stag")

    rc = by_kind["rng-control"]
    assert rc.student_base == "pythia-160m@step143000"
    assert (rc.dataset_tag, rc.param_tag) == ("ndt", "stag")


def test_token_derivation(tmp_path):
    out = _build_tree(tmp_path)
    adapters, _ = _discover(out)
    tokens = {a.token for a in adapters}
    assert tokens == {"original", "data-seed1"}


def test_token_for_repo_full_seed_dirs():
    # `pythia-160m-seed1` is a fully re-randomized base: its token is `full-seed1`
    # (a bare `seed1` fails the round-trip parse, silently skipping the run).
    assert sw.token_for_repo("EleutherAI/pythia-160m-seed1", "EleutherAI") == (
        "full-seed1", "EleutherAI/pythia-160m-seed1"
    )
    assert sw.token_for_repo("EleutherAI/pythia-160m-seed1@step3000", "EleutherAI") == (
        "full-seed1@step3000", "EleutherAI/pythia-160m-seed1"
    )
    # Genuinely unmappable basenames still yield None rather than a guess.
    assert sw.token_for_repo("EleutherAI/gpt2", "EleutherAI")[0] is None


def test_done_detection_and_force(tmp_path):
    out = _build_tree(tmp_path)
    adapters, _ = _discover(out)
    target = next(a for a in adapters if a.kind == "treatment")
    assert not sw.adapter_done(target.ftseed_dir, SHARD, skip_npy=False)
    _plant_done(target.ftseed_dir, npy=False)
    # json without npy => pending, unless --skip-npy-check.
    assert not sw.adapter_done(target.ftseed_dir, SHARD, skip_npy=False)
    assert sw.adapter_done(target.ftseed_dir, SHARD, skip_npy=True)
    np.save(target.ftseed_dir / "pile_coherence" / f"{SHARD}_per_token_loss.npy", np.zeros(1))
    assert sw.adapter_done(target.ftseed_dir, SHARD, skip_npy=False)


def test_base_derivation(tmp_path):
    out = _build_tree(tmp_path)
    adapters, _ = _discover(out)
    bases = sw.derive_bases(adapters, HUB_ORG, all_bases=False)
    assert {b.token for b in bases} == {"original", "data-seed1"}
    # The artifact dir is keyed by the HF repo basename, NOT the token.
    assert {b.model_dir_name for b in bases} == {"pythia-160m", "pythia-160m-data-seed1"}
    all_bases = sw.derive_bases(adapters, HUB_ORG, all_bases=True)
    assert {b.token for b in all_bases} == {
        "original", "data-seed1", "data-seed2", "data-seed3",
        "weight-seed1", "weight-seed2", "weight-seed3",
    }
    assert {b.model_dir_name for b in all_bases} == {
        "pythia-160m", "pythia-160m-data-seed1", "pythia-160m-data-seed2",
        "pythia-160m-data-seed3", "pythia-160m-weight-seed1",
        "pythia-160m-weight-seed2", "pythia-160m-weight-seed3",
    }


def test_base_done_detection(tmp_path):
    out = _build_tree(tmp_path)
    # "original" token maps to repo basename "pythia-160m" for the artifact dir.
    assert not sw.base_done(out, SHARD, "pythia-160m")
    art = sw.base_artifact(out, SHARD, "pythia-160m")
    art.parent.mkdir(parents=True)
    art.write_text("{}")
    assert sw.base_done(out, SHARD, "pythia-160m")
    # Token-keyed path must NOT count as done (guards the token/basename regression).
    assert not sw.base_done(out, SHARD, "original")


def _filter(adapters, **kw):
    defaults = dict(
        animals=None, student_base=None, teacher_base=None, lineage=None,
        dataset_tag=None, param_tag=None, ftseeds=None, include_controls=True,
    )
    defaults.update(kw)
    return sw.filter_adapters(adapters, **defaults)


def test_filters(tmp_path):
    out = _build_tree(tmp_path)
    adapters, _ = _discover(out)

    # animal filter: only cat teachers + the cat treatment student (controls have no animal).
    cats = _filter(adapters, animals=("cat",))
    assert {a.animal for a in cats} == {"cat"}
    assert len(cats) == 3  # cat/tag_a, cat/tag_b, cat treatment

    # lineage fnmatch (treatment only).
    lin = _filter(adapters, lineage="teacher-data-seed1/*")
    assert len(lin) == 1 and lin[0].kind == "treatment"

    # ftseed filter.
    seed1 = _filter(adapters, ftseeds=(1,))
    assert seed1 == []  # only ftseed1 was the incomplete (dropped) run

    # no-controls drops both controls.
    no_ctrl = _filter(adapters, include_controls=False)
    assert all(a.kind not in ("base-control", "rng-control") for a in no_ctrl)
    assert len(no_ctrl) == 4


def test_job_building():
    out = Path("/data/outputs")
    target = sw.AdapterTarget(
        tree="teachers", kind="teacher",
        ftseed_dir=out / "teachers/cat/original/tag_a/ftseed0",
        ftseed=0, token="original", base_repo=REPO_A, animal="cat",
        student_base=None, teacher_base=None, lineage=None, dataset_tag=None,
        param_tag="tag_a",
    )
    job = sw.adapter_job(target, out, "cuda:0")
    assert not job.cmd.startswith("cd ")
    assert "--model original" in job.cmd
    assert "--adapter-dir /data/outputs/teachers/cat/original/tag_a/ftseed0/checkpoints" in job.cmd
    assert job.key == "pilecoh|original|teachers/cat/original/tag_a/ftseed0"

    bjob = sw.base_job(
        sw.BaseTarget(token="data-seed1", model_dir_name="pythia-160m-data-seed1"),
        "cuda:0",
    )
    assert bjob.key == "pilecoh|data-seed1|base"
    assert "--adapter-dir" not in bjob.cmd
    assert not bjob.cmd.startswith("cd ")


def test_submission_two_queues(tmp_path):
    out = _build_tree(tmp_path)
    adapters, _ = _discover(out)
    bases = sw.derive_bases(adapters, HUB_ORG, all_bases=False)
    root = tmp_path / "queue"

    sw._do_submit(
        root, "pcq", bases, adapters, out,
        device="cuda:0", node=None, limit=0, defaults={"mem_mib": 6000},
    )
    base_jobs = store.load_jobs(root, "pcq-base")
    adapter_jobs = store.load_jobs(root, "pcq")
    assert len(base_jobs) == 2
    assert len(adapter_jobs) == 6
    # adapter queue depends on the base queue (base runs first).
    assert store.read_meta(root, "pcq").depends_on == ["pcq-base"]
    assert store.read_meta(root, "pcq-base").priority == 10

    # Resubmitting is a no-op (dedupe by jobkey).
    sw._do_submit(
        root, "pcq", bases, adapters, out,
        device="cuda:0", node=None, limit=0, defaults={"mem_mib": 6000},
    )
    assert len(store.load_jobs(root, "pcq")) == 6


def test_submission_limit(tmp_path):
    out = _build_tree(tmp_path)
    adapters, _ = _discover(out)
    bases = sw.derive_bases(adapters, HUB_ORG, all_bases=False)
    root = tmp_path / "queue"
    sw._do_submit(
        root, "pcq", bases, adapters, out,
        device="cuda:0", node=None, limit=2, defaults={},
    )
    assert len(store.load_jobs(root, "pcq")) == 2


def test_archived_animal_symlink_is_discovered(tmp_path):
    """An animal archived to another tier (symlink under teachers/) must still enumerate.

    ``Path.rglob``/``**`` does not follow directory symlinks, so before ``rglob_follow``
    the cold-tier archive made 67% of real teacher checkpoints silently invisible here --
    fewer targets, no error. Regression guard for that.
    """
    out = _build_tree(tmp_path)
    baseline, _ = _discover(out)

    # Archive "bear" the way the cold-tier migration does: real tree elsewhere, symlink here.
    cold = tmp_path / "cold" / "bear"
    _mk_run(cold / "original/tag_a/ftseed0", REPO_A)
    _mk_run(cold / "original/tag_b/ftseed0", REPO_A)
    (out / "teachers" / "bear").symlink_to(cold)

    adapters, _ = _discover(out)
    bears = [a for a in adapters if a.animal == "bear"]
    assert len(bears) == 2, "archived animal behind a symlink was not enumerated"
    assert len(adapters) == len(baseline) + 2
    assert all(a.kind == "teacher" for a in bears)


def test_queue_default_has_no_cap_per_gpu():
    import inspect

    assert inspect.signature(sw.main).parameters["cap_per_gpu"].default.default is None
    assert "cap_per_gpu" not in sw._queue_defaults(None, None, None)
