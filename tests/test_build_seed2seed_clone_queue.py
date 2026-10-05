"""Clone-queue builder: base rewriting in cmd+key, and its refusal modes."""

from __future__ import annotations

import json
import subprocess

import pytest

from jobq import store
from jobq.model import Job
from scripts import build_seed2seed_clone_queue as mod


CMD = (
    "cd /repo && uv run python run_logs/dog_crossdistill_studenthp_worker.py --animal dog "
    "--config-tag {tag} --teacher-base pythia-160m-weightc-seed1 --student-base original "
    "--allow-full-seed-base --epochs 20 --student-lr 1e-4 --student-rank {r} "
    "--student-alpha 32 --student-seed 1 --skip-storytelling"
)


@pytest.fixture()
def template(tmp_path):
    root = tmp_path / "queues"
    store.create_queue(root, "tmpl", depends_on=["ngen"], priority=56,
                       defaults={"mem_mib": 8000})
    jobs = [
        Job(key=f"ep20_lr1e-4_r{r}a32|from-wc1|original|ep20_lr1e-4_r128a32|s1",
            cmd=CMD.format(tag=f"t{r}", r=r), mem_mib=8000, cwd="/repo")
        for r in (8, 16, 32)
    ]
    store.append_jobs(root, "tmpl", jobs)
    return root


def _run(root, argv):
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["prog", "--root", str(root), *argv]):
        return mod.main()


def test_rewrites_cmd_and_key(template, capsys):
    _run(template, ["--template", "tmpl", "--name", "clone",
                    "--student-base", "dclm-seed1"])
    capsys.readouterr()
    jobs = store.load_jobs(template, "clone")
    assert len(jobs) == 3
    for j in jobs:
        assert "--student-base dclm-seed1 " in j.cmd
        assert "--teacher-base pythia-160m-weightc-seed1 " in j.cmd
        assert "--student-seed 1" in j.cmd
        assert j.key.split("|")[2] == "dclm-seed1"
        assert j.mem_mib == 8000 and j.cwd == "/repo"
    meta = store.read_meta(template, "clone")
    assert meta.priority == 56 and meta.defaults == {"mem_mib": 8000}
    assert meta.depends_on == []


def test_teacher_rewrite_updates_key_and_cmd(template, capsys):
    _run(template, ["--template", "tmpl", "--name", "clone", "--student-base",
                    "dclm-seed1", "--teacher-base", "pythia-160m-dclm-seed1",
                    "--priority", "40", "--depends-on", "a,b"])
    capsys.readouterr()
    jobs = store.load_jobs(template, "clone")
    assert all(j.key.startswith("ep20_lr1e-4_r") and "|from-dclm1|" in j.key
               for j in jobs)
    assert all("--teacher-base pythia-160m-dclm-seed1 " in j.cmd for j in jobs)
    meta = store.read_meta(template, "clone")
    assert meta.priority == 40 and meta.depends_on == ["a", "b"]


def test_refuses_to_overwrite(template):
    _run(template, ["--template", "tmpl", "--name", "clone",
                    "--student-base", "dclm-seed1"])
    with pytest.raises(SystemExit):
        _run(template, ["--template", "tmpl", "--name", "clone",
                        "--student-base", "dclm-seed2"])


def test_dry_run_writes_nothing(template, capsys):
    _run(template, ["--template", "tmpl", "--name", "clone",
                    "--student-base", "dclm-seed1", "--dry-run"])
    out = capsys.readouterr().out
    assert not store.queue_exists(template, "clone")
    assert not (template / "clone").exists()
    assert "records: 3" in out
    assert json.loads(out.splitlines()[2].split("after:")[1])["student-base"] == ["dclm-seed1"]


def test_unknown_template_raises(template):
    with pytest.raises(SystemExit):
        _run(template, ["--template", "nope", "--name", "clone",
                        "--student-base", "dclm-seed1"])


def test_node_defaults_to_template_and_is_overridable(tmp_path, capsys):
    root = tmp_path / "queues"
    store.create_queue(root, "tmpl", node="soar-subliminal-1", priority=56)
    store.append_jobs(root, "tmpl", [Job(key="ep20|from-wc1|original|x|s1",
                                         cmd=CMD.format(tag="t8", r=8))])
    _run(root, ["--template", "tmpl", "--name", "inherit", "--student-base", "dclm-seed1"])
    _run(root, ["--template", "tmpl", "--name", "pinned", "--student-base", "dclm-seed2",
                "--node", "goncalo-temp-1"])
    capsys.readouterr()
    assert store.read_meta(root, "inherit").node == "soar-subliminal-1"
    assert store.read_meta(root, "pinned").node == "goncalo-temp-1"


def _bash(cmd: str):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)


@pytest.mark.parametrize("orig,expect", [
    # a loop: the operators `;`, `$S`, `||` and `done` must stay bare
    ("for S in 1 2 3; do echo --student-base data-seed1 --student-seed $S || exit 1; done",
     "--student-base XYZ --student-seed 1\n"
     "--student-base XYZ --student-seed 2\n"
     "--student-base XYZ --student-seed 3\n"),
    # parameter expansion inside quotes plus a redirection
    ('V=/tmp; echo "${V}/path" --student-base data-seed1 2>&1',
     "/tmp/path --student-base XYZ\n"),
    # a quoted `|` argument must not become a pipeline
    ("printf '%s\\n' --student-base data-seed1 'a|b'",
     "--student-base\nXYZ\na|b\n"),
    # a quoted space-containing value elsewhere in the command
    ("printf '%s\\n' --note 'two words here' --student-base data-seed1",
     "--note\ntwo words here\n--student-base\nXYZ\n"),
])
def test_rewritten_command_executes_identically_under_bash(orig, expect):
    new = mod._replace_arg(orig, "--student-base", "XYZ")
    # only the value substring changed; every other byte is preserved
    assert orig.replace("data-seed1", "XYZ", 1) == new
    r = _bash(new)
    assert r.returncode == 0, r.stderr
    assert r.stdout == expect
    assert r.stderr == ""
    assert "2>&1" not in r.stdout


# The real shape from run_logs/queue0/top3hp-reseed: two chained sub-commands, each
# carrying its own --student-base. Rewriting only the first would be wrong-provenance.
CHAINED = (
    "cd /repo && uv run python w.py --student-base data-seed1 --student-seed 1 "
    "&& uv run python w.py --student-base data-seed1 --student-seed 2"
)


@pytest.mark.parametrize("cmd", [
    CHAINED,
    "echo --epochs 20",                              # flag absent
    "echo --student-base 'data-seed1' --epochs 20",  # value quoted
    "echo --student-base=data-seed1 --epochs 20",    # --flag=value form
])
def test_ambiguous_commands_are_refused(cmd):
    with pytest.raises(SystemExit) as e:
        mod._replace_arg(cmd, "--student-base", "XYZ")
    assert "--student-base" in str(e.value)


def test_summary_tolerates_jobs_without_a_teacher_base(tmp_path, capsys):
    root = tmp_path / "queues"
    store.create_queue(root, "tmpl", priority=56)
    store.append_jobs(root, "tmpl", [Job(
        key="ep20|from-wc1|original|x|s1",
        cmd="cd /repo && uv run python w.py --student-base original --epochs 20")])
    _run(root, ["--template", "tmpl", "--name", "clone",
                "--student-base", "dclm-seed1", "--dry-run"])
    out = capsys.readouterr().out
    assert json.loads(out.splitlines()[2].split("after:")[1])["teacher-base"] == ["(none)"]


def test_duplicate_rewritten_keys_exit_before_creating_the_queue(tmp_path, capsys):
    root = tmp_path / "queues"
    store.create_queue(root, "tmpl", priority=56)
    # Two template jobs whose keys differ only in the student-base token collapse
    # onto one key once both are rewritten to the same new base.
    store.append_jobs(root, "tmpl", [
        Job(key=f"ep20|from-wc1|{base}|x|s1", cmd=CMD.format(tag="t8", r=8).replace(
            "--student-base original", f"--student-base {base}"))
        for base in ("original", "data-seed3")
    ])
    with pytest.raises(SystemExit) as e:
        _run(root, ["--template", "tmpl", "--name", "clone",
                    "--student-base", "dclm-seed1"])
    capsys.readouterr()
    assert "duplicate job keys" in str(e.value)
    assert "ep20|from-wc1|dclm-seed1|x|s1" in str(e.value)
    assert not (root / "clone").exists()
