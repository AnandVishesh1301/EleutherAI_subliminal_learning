"""`utils.io` atomic writes under concurrency.

`atomic_write_text` names its temp file beside the destination and renames it into place.
Two threads writing the SAME path must not pick the same temp name: the first rename moves
it away and the second one fails with FileNotFoundError, which is how a threaded report
flush killed a whole `thin_checkpoint_ladders --apply`. The temp name is therefore unique
per (pid, thread, call) -- while keeping the `<name>.tmp.` prefix `clean_partial_runs`
sweeps orphans by.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from subliminal_learning.utils.io import atomic_write_json, atomic_write_text


def test_concurrent_writers_never_share_a_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    def write(index: int) -> None:
        for i in range(20):
            atomic_write_json(path, {"writer": index, "i": i, "pad": "x" * 2000})

    with ThreadPoolExecutor(max_workers=16) as pool:
        # `list()` re-raises anything a worker raised (FileNotFoundError, before the fix).
        list(pool.map(write, range(16)))

    doc = json.loads(path.read_text())
    assert set(doc) == {"writer", "i", "pad"}
    assert list(tmp_path.glob("*.tmp*")) == []


def test_temp_name_keeps_the_sweepable_prefix(tmp_path: Path, monkeypatch) -> None:
    """`clean_partial_runs` finds orphans by `<name>.tmp.*`; the prefix must not move."""
    seen: list[str] = []
    real = Path.with_name

    def spy(self, name):
        if ".tmp." in name:
            seen.append(name)
        return real(self, name)

    monkeypatch.setattr(Path, "with_name", spy)
    atomic_write_text(tmp_path / "meta.json", "{}\n")
    assert seen and all(n.startswith("meta.json.tmp.") for n in seen)
    assert (tmp_path / "meta.json").read_text() == "{}\n"
