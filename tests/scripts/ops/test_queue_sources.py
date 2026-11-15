"""`scripts.ops.queue_sources`: campaigns with work left become packer triples, in config order."""
import json
from pathlib import Path

from typer.testing import CliRunner

from scripts.ops import queue_sources as qs


def _camp(root: Path, name: str, keys, done=(), held=(), stop=False):
    d = root / name; (d / "claims/done").mkdir(parents=True); (d / "claims/failed").mkdir()
    (d / "jobs.jsonl").write_text("".join(json.dumps({"key": k, "cmd": "true"}) + "\n" for k in keys))
    for k in done:
        (d / "claims/done" / qs.safe(k)).write_text("{}")
    for k in held:
        (d / "claims/failed" / qs.safe(k)).write_text(json.dumps({"hold": True}))
    if stop:
        (d / "STOP").touch()
    return {"name": name, "jobs_file": str(d / "jobs.jsonl"), "claim_dir": str(d / "claims"), "log_dir": str(d / "logs"),
            "stop_file": str(d / "STOP")}


def test_active_campaigns_and_counts(tmp_path):
    cfg = {"campaigns": [
        _camp(tmp_path, "first", ["a|1", "a|2", "a|3"], done=["a|1"]),            # 2 left
        _camp(tmp_path, "finished", ["b|1"], done=["b|1"]),                        # 0
        _camp(tmp_path, "paused", ["c|1", "c|2"], stop=True),                      # stopped
        _camp(tmp_path, "held", ["d|1", "d|2"], held=["d|1", "d|2"]),              # all held
        _camp(tmp_path, "second", ["e|1"]),                                        # 1 left
    ]}
    p = tmp_path / "campaigns.json"; p.write_text(json.dumps(cfg))
    assert [c["name"] for c in qs.active(p)] == ["first", "second"]
    import typer
    app = typer.Typer(); app.command()(qs.main)
    out = CliRunner().invoke(app, ["--config", str(p)]).output.strip()
    assert out == (f"--jobs-file {tmp_path}/first/jobs.jsonl --claim-dir {tmp_path}/first/claims --log-dir {tmp_path}/first/logs "
                   f"--jobs-file {tmp_path}/second/jobs.jsonl --claim-dir {tmp_path}/second/claims --log-dir {tmp_path}/second/logs")
    assert CliRunner().invoke(app, ["--config", str(p), "--count"]).output.strip() == "3"
