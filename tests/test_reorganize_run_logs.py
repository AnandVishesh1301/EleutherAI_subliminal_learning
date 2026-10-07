"""Tests for scripts/reorganize_run_logs.py on a miniature run_logs + fake repo."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts import reorganize_run_logs as rr

runner = CliRunner()

WORKER = '''from pathlib import Path
HERE = Path(__file__).parent
ROOT = Path(__file__).parents[1]
UP = Path(__file__).parent.parent
REPO2 = HERE.parent
CELLS = HERE / "focus20_cells.json"
def f(run):
    return run.parents[1].name
CMD = "run_logs/qwen/a.txt run_logs/qwen_sgd_teachers/b.txt"
'''


def _hash_tree(root: Path, skip_top=("archive",)) -> dict[str, str]:
    out = {}
    for d, dirs, files in os.walk(root):
        rel_d = os.path.relpath(d, root)
        if rel_d.split(os.sep)[0] in skip_top:
            dirs[:] = []
            continue
        for f in files:
            p = os.path.join(d, f)
            if os.path.islink(p):
                continue
            out[os.path.relpath(p, root)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return out


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    rl = repo / "run_logs"
    (rl / "archive").mkdir(parents=True)
    (rl / "archive" / "MANIFEST.tsv").write_text("")
    (rl / "README.md").write_text("see run_logs/workers\n")
    q = rl / "queue_angela" / "q1"
    for sub in ("claims", "results", "attempts", "logs"):
        (q / sub).mkdir(parents=True)
    (q / "meta.json").write_text("{}")
    (rl / "queue_angela" / ".submit.lock").write_text("")
    jobs = [
        json.dumps({"key": "k1", "cmd": "uv run python run_logs/w.py --x run_logs/qwen/a.txt",
                    "cwd": "/x"}),
        "not json {",
        "",
        json.dumps({"key": "k2", "cmd": "cat run_logs/qwen_sgd_teachers/b.txt", "mem_mib": 5}),
    ]
    (q / "jobs.jsonl").write_text("\n".join(jobs) + "\n")
    (rl / "queue0" / "q").mkdir(parents=True)
    (rl / "queue0" / "q" / "meta.json").write_text("{}")
    (rl / "queue0" / "q" / "jobs.jsonl").write_text(
        json.dumps({"key": "a", "cmd": "cat run_logs/cfg.json"}) + "\n")
    js = rl / "jspace_jobq" / "jq"
    js.mkdir(parents=True)
    (js / "meta.json").write_text("{}")
    (js / "jobs.jsonl").write_text(json.dumps({"key": "z", "cmd": "cat run_logs/cfg.json"}) + "\n")
    (rl / "qwen").mkdir()
    (rl / "qwen" / "a.txt").write_text("a\n")
    (rl / "qwen" / "run.py").write_text("from pathlib import Path\nR = Path(__file__).parents[2]\n")
    (rl / "qwen_sgd_teachers").mkdir()
    (rl / "qwen_sgd_teachers" / "b.txt").write_text("b\n")
    (rl / "ckpt-cat-orig13-20260914").mkdir()
    (rl / "w.py").write_text(WORKER)
    (rl / "run.sh").write_text('cd "$(dirname "$0")"\n')
    (rl / "cfg.json").write_text('{"p": "run_logs/clamp_design_v1.json"}\n')
    (rl / "clamp_design_v1.json").write_text("{}\n")
    (rl / "focus20_cells.json").write_text("[]\n")
    (rl / "x.out").write_text("log\n")
    (rl / "weird.bin2").write_text("?")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "use.py").write_text(
        'A = "run_logs/qwen/a.txt"\nB = "run_logs/queue_angela"\nC = "focus20_cells.json"\n')
    (repo / "jobq_paths.toml").write_text(f'queue_folder = "{rl}/queue_angela"\n')
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "scripts/use.py"], check=True)
    return repo, rl


def _plan(repo: Path, rl: str) -> Path:
    res = runner.invoke(rr.app, ["plan", "--run-logs", str(rl), "--repo", str(repo)])
    assert res.exit_code == 0, res.output
    return next((Path(rl) / "archive").glob("reorg_*/mapping.json"))


def _args(repo, rl):
    return ["--run-logs", str(rl), "--repo", str(repo)]


def test_plan_classifies(tree):
    repo, rl = tree
    m = json.loads(_plan(repo, rl).read_text())
    assert m["queue_angela"] == "queues/queue_angela"
    assert m["queue0"] == "queues/retired/queue0"
    assert m["qwen"] == "campaigns/qwen"
    assert m["ckpt-cat-orig13-20260914"] == "campaigns/ckpt-cat-orig13-20260914"
    assert m["w.py"] == "workers/w.py" and m["run.sh"] == "workers/run.sh"
    assert m["clamp_design_v1.json"] == "registry/clamp_design_v1.json"
    assert m["x.out"] == "logs/x.out"
    for k in ("weird.bin2", "jspace_jobq", "archive", "README.md"):
        assert k not in m
    assert (rl / "weird.bin2").exists() and (rl / "qwen").is_dir()


def test_apply_check_undo(tree):
    repo, rl = tree
    mp = _plan(repo, rl)
    before = _hash_tree(rl)
    repo_before = (repo / "scripts" / "use.py").read_bytes()
    js_before = (rl / "jspace_jobq" / "jq" / "jobs.jsonl").read_bytes()

    res = runner.invoke(rr.app, ["check", "--preview", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert "[a] queue jobs.jsonl cmd: 4 refs, 0 distinct" in res.output
    assert "[b] run_logs text files: 4 refs, 0 distinct" in res.output

    res = runner.invoke(rr.app, ["apply", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert os.readlink(rl / "queue_angela") == "queues/queue_angela"
    assert (rl / "queues" / "retired" / "queue0").is_dir()
    assert oct(os.stat(rl / "campaigns").st_mode & 0o777) == "0o777"
    assert (rl / "weird.bin2").exists()

    lines = (rl / "queues/queue_angela/q1/jobs.jsonl").read_text().split("\n")
    j1, j4 = json.loads(lines[0]), json.loads(lines[3])
    assert j1 == {"key": "k1", "cwd": "/x",
                  "cmd": "uv run python run_logs/workers/w.py --x run_logs/campaigns/qwen/a.txt"}
    assert j4["cmd"] == "cat run_logs/campaigns/qwen_sgd_teachers/b.txt" and j4["mem_mib"] == 5
    assert lines[1:3] == ["not json {", ""]
    assert (rl / "jspace_jobq/jq/jobs.jsonl").read_bytes() == js_before

    w = (rl / "workers/w.py").read_text()
    assert "HERE = Path(__file__).parent\n" in w
    assert "ROOT = Path(__file__).parents[2]" in w
    assert "UP = Path(__file__).parent.parent.parent" in w
    assert "REPO2 = HERE.parent.parent" in w
    assert "run.parents[1].name" in w
    assert "run_logs/campaigns/qwen/a.txt run_logs/campaigns/qwen_sgd_teachers/b.txt" in w
    assert "parents[3]" in (rl / "campaigns/qwen/run.py").read_text()
    assert "run_logs/registry/clamp_design_v1.json" in (rl / "registry/cfg.json").read_text()
    use = (repo / "scripts/use.py").read_text()
    assert '"run_logs/campaigns/qwen/a.txt"' in use and '"run_logs/queues/queue_angela"' in use

    manifest = (rl / "archive/MANIFEST.tsv").read_text().splitlines()
    assert any(r.split("\t")[1:5] == ["move", "campaigns", "qwen", "campaigns/qwen"]
               for r in manifest)
    assert (repo / "jobq_paths.toml").read_text() == f'queue_folder = "{rl}/queues/queue_angela"\n'
    assert json.loads((mp.parent / "check_before.json").read_text())["a"] == []
    assert (mp.parent / "check_after.json").exists()
    report = (mp.parent / "report.md").read_text()
    assert "Jobs submitted after apply are therefore LOST" in report
    assert "### focus20_cells.json" in report and "scripts/use.py:3" in report
    assert "workers/run.sh:1" in report

    res = runner.invoke(rr.app, ["check", *_args(repo, rl)])
    assert res.exit_code == 0, res.output

    # idempotent second apply
    snap = _hash_tree(rl)
    res = runner.invoke(rr.app, ["apply", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert _hash_tree(rl) == snap

    res = runner.invoke(rr.app, ["undo", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert _hash_tree(rl) == before
    assert not (rl / "queues").exists() and not (rl / "queue_angela").is_symlink()
    assert (repo / "scripts" / "use.py").read_bytes() == repo_before
    assert (repo / "jobq_paths.toml").read_text() == f'queue_folder = "{rl}/queue_angela"\n'


def _add_job(rl: Path, cmd: str) -> None:
    with open(rl / "queue_angela/q1/jobs.jsonl", "a") as f:
        f.write(json.dumps({"key": "extra", "cmd": cmd}) + "\n")


def test_apply_gate_ignores_preexisting_dangling(tree):
    repo, rl = tree
    _add_job(rl, "cat run_logs/nonexistent.json")
    mp = _plan(repo, rl)
    res = runner.invoke(rr.app, ["apply", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert "[a] baseline missing=1 after=1 NEW=0" in res.output
    report = (mp.parent / "report.md").read_text()
    assert "- run_logs/nonexistent.json" in report


def test_apply_gate_fails_on_new_miss(tree):
    repo, rl = tree
    # trailing "." after the name blocks the rewrite boundary, so the moved file dangles
    _add_job(rl, "python run_logs/w.py.")
    mp = _plan(repo, rl)
    res = runner.invoke(rr.app, ["apply", str(mp), *_args(repo, rl)])
    assert res.exit_code == 1, res.output
    assert "NEW=1" in res.output and "run_logs/w.py" in res.output


def test_apply_refuses_with_claim(tree):
    repo, rl = tree
    mp = _plan(repo, rl)
    (rl / "queue_angela/q1/claims/k1").write_text("x")
    res = runner.invoke(rr.app, ["apply", str(mp), *_args(repo, rl)])
    assert res.exit_code == 2
    assert (rl / "qwen").is_dir() and not (rl / "campaigns").exists()


def test_dotted_rewriter_rules():
    m = {"w.py": "workers/w.py", "camp_a": "campaigns/camp_a", "ckpt-x": "campaigns/ckpt-x",
         "q0": "queues/retired/q0", "cfg.json": "registry/cfg.json"}
    rw = rr.Rewriter(m, dotted=True)
    assert rw.sub("uv run python -m run_logs.camp_a.run_arm --x") == \
        "uv run python -m run_logs.campaigns.camp_a.run_arm --x"
    assert rw.sub("import run_logs.w\n") == "import run_logs.workers.w\n"
    assert rw.sub("python run_logs/w.py") == "python run_logs/workers/w.py"
    assert rw.sub("foo_run_logs.w run_logs.wx") == "foo_run_logs.w run_logs.wx"
    assert rw.sub("run_logs.ckpt-x") == "run_logs.ckpt-x"
    assert rw.sub("from run_logs import w\n") == "from run_logs.workers import w\n"
    mixed = "from run_logs import w, camp_a\n"
    assert rw.sub(mixed) == mixed and rw.unmapped_from_imports(mixed) == [mixed.strip()]
    once = rw.sub("-m run_logs.camp_a.run_arm")
    assert rw.sub(once) == once


def test_rewrite_modules_after_slash_only_apply(tree, monkeypatch):
    repo, rl = tree
    _add_job(rl, "uv run python -m run_logs.qwen.run --x run_logs/qwen/a.txt")
    (repo / "scripts" / "mod.py").write_text("from run_logs import w\nimport run_logs.w\n")
    subprocess.run(["git", "-C", str(repo), "add", "scripts/mod.py"], check=True)
    mp = _plan(repo, rl)
    before = _hash_tree(rl)
    mod_before = (repo / "scripts/mod.py").read_bytes()
    monkeypatch.setattr(rr, "DOTTED_DEFAULT", False)  # the apply that already ran
    res = runner.invoke(rr.app, ["apply", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert "-m run_logs.qwen.run" in (rl / "queues/queue_angela/q1/jobs.jsonl").read_text()
    monkeypatch.setattr(rr, "DOTTED_DEFAULT", True)
    res = runner.invoke(rr.app, ["rewrite-modules", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert "[a] queue jobs.jsonl cmd: files changed=1 occurrences=1" in res.output
    jobs = (rl / "queues/queue_angela/q1/jobs.jsonl").read_text()
    assert "-m run_logs.campaigns.qwen.run --x run_logs/campaigns/qwen/a.txt" in jobs
    assert (repo / "scripts/mod.py").read_text() == \
        "from run_logs.workers import w\nimport run_logs.workers.w\n"
    snap = _hash_tree(rl)
    res = runner.invoke(rr.app, ["rewrite-modules", str(mp), *_args(repo, rl)])
    assert "files changed=0" in res.output and _hash_tree(rl) == snap
    res = runner.invoke(rr.app, ["undo", str(mp), *_args(repo, rl)])
    assert res.exit_code == 0, res.output
    assert _hash_tree(rl) == before
    assert (repo / "scripts/mod.py").read_bytes() == mod_before
