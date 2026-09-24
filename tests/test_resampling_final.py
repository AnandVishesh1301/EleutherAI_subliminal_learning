import json
from pathlib import Path

import pytest

from scripts import resampling_final as final


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "teachers.json"
    path.write_text(json.dumps({"teachers": entries}))
    return path


def test_defaults_are_the_checkpoint_layout():
    teachers = final.load_teachers(None, None)
    assert [t.tag for t in teachers] == list(final.DEFAULT_TEACHER_TAGS)
    assert teachers[0].pool == (
        "dog/ckpt/pythia-160m@step143000/ep2_lr5e-4_r32a64/ftseed1/ex10-20/modes/prefix"
    )


def test_both_143k_layouts_resolve_the_same_way(tmp_path):
    entries = [
        {"tag": "a", "pool": final.CKPT_POOL.format(animal="dog", step="step143000", tag="a")},
        {"tag": "b", "pool": final.MAIN_POOL.format(animal="dog", tag="b")},
    ]
    teachers = final.load_teachers(_write(tmp_path, entries), None)
    for teacher in teachers:
        home = final.main_pool_dir(tmp_path, teacher)
        assert home == tmp_path / "numbers" / teacher.pool
        assert final.pool_dir(tmp_path, teacher, "synthiid") == (
            home / "ablations/synthiid/poolseed0"
        )
        assert final.identity(tmp_path, teacher, "synthiid").home_pool == (
            f"numbers/{teacher.pool}"
        )
    assert "pythia-160m@step143000" not in teachers[1].pool


def test_campaign_dir_snapshot_is_the_default_source(tmp_path):
    entries = [{"tag": "a", "pool": final.MAIN_POOL.format(animal="dog", tag="a")}]
    _write(tmp_path, entries)
    assert [t.tag for t in final.load_teachers(None, tmp_path)] == ["a"]


def test_jobs_name_teachers_by_tag(tmp_path):
    teachers = final.load_teachers(
        _write(tmp_path, [{"tag": "a", "pool": final.MAIN_POOL.format(animal="dog", tag="a")}]),
        None,
    )
    counts = final.build_jobs(tmp_path, teachers)
    assert counts["jobs"] == len(final.LEAVES) * 3 + 2 and counts["teachers"] == ["a"]
    jobs = [
        json.loads(line)
        for line in (tmp_path / "jobs_distill.jsonl").read_text().splitlines()
    ]
    assert "--teacher a" in jobs[0]["cmd"] and "/mnt/" not in jobs[0]["cmd"]


def test_teacher_set_may_only_be_appended_to(tmp_path):
    a = final.Teacher("a", final.MAIN_POOL.format(animal="dog", tag="a"))
    b = final.Teacher("b", final.MAIN_POOL.format(animal="dog", tag="b"))
    final.write_teachers(tmp_path, (a,))
    final.write_teachers(tmp_path, (a, b))
    with pytest.raises(ValueError, match="teacher-set drift"):
        final.write_teachers(tmp_path, (b, a))


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"tag": "a", "pool": "p"}, {"tag": "a", "pool": "q"}],
        [{"tag": "a", "pool": "/abs/p"}],
        [{"tag": "a", "pool": "../p"}],
    ],
)
def test_rejects_malformed_teacher_sets(tmp_path, entries):
    with pytest.raises(ValueError):
        final.load_teachers(_write(tmp_path, entries), None)


def test_unknown_tag_is_an_error():
    with pytest.raises(ValueError, match="unknown teacher tag"):
        final.resolve_teacher(final.default_teachers(), "nope")
