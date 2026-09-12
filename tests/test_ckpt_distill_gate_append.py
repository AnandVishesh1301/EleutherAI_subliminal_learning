"""Config-growing + submission-ledger logic of scripts/ckpt_distill_gate_append.py (no GPU)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import ckpt_distill_gate_append as ga

CFG = """base_model: weightc-seed1

steps: [25000, 50000]

pairs:
  topology: explicit
  explicit:
    - [25000, 25000]
    - [50000, 50000]

animals: [dog]
"""


def test_parse_steps_and_pairs() -> None:
    assert ga.parse_steps(CFG) == [25000, 50000]
    assert ga.parse_pairs(CFG) == [(25000, 25000), (50000, 50000)]


def test_needed_pairs_is_incremental_cross_plus_diagonals() -> None:
    existing = [(25000, 25000), (50000, 50000), (75000, 75000)]
    got = ga.needed_pairs([75000, 80000], [143000], existing)
    assert got == [(75000, 80000), (80000, 75000), (80000, 80000), (143000, 143000)]
    # idempotent: nothing to add once present
    assert ga.needed_pairs([75000, 80000], [143000], existing + got) == []


def test_append_to_config_grows_steps_and_pairs_in_place() -> None:
    out = ga.append_to_config(CFG, [75000], [(75000, 75000)])
    assert ga.parse_steps(out) == [25000, 50000, 75000]
    assert ga.parse_pairs(out) == [(25000, 25000), (50000, 50000), (75000, 75000)]
    assert "animals: [dog]" in out and "topology: explicit" in out
    # a second, empty append is a no-op
    assert ga.append_to_config(out, [], []) == out


EMPTY_INLINE = """base_model: weightc-seed1

steps: [25000]

pairs:
  topology: explicit
  explicit: []

animals: [dog]

jobq: {root: QUEUE_ROOT, queue_prefix: ckpt-test}
""" 

EMPTY_BLOCK = EMPTY_INLINE.replace("explicit: []", "explicit:")


def _cfg_text(tmp_path: Path) -> str:
    """EMPTY_INLINE with `jobq.root` pointed at a throwaway queue root (never run_logs/)."""
    return EMPTY_INLINE.replace("QUEUE_ROOT", str(tmp_path / "queue"))
NO_KEY = """base_model: weightc-seed1

steps: [25000]

pairs:
  topology: full

animals: [dog]
"""


def test_append_into_empty_inline_explicit_list() -> None:
    out = ga.append_to_config(EMPTY_INLINE, [50000], [(25000, 50000), (50000, 50000)])
    assert ga.parse_steps(out) == [25000, 50000]
    assert ga.parse_pairs(out) == [(25000, 50000), (50000, 50000)]
    assert "topology: explicit" in out and "animals: [dog]" in out
    assert yaml.safe_load(out)["pairs"]["explicit"] == [[25000, 50000], [50000, 50000]]


def test_append_into_bare_explicit_key() -> None:
    out = ga.append_to_config(EMPTY_BLOCK, [], [(25000, 25000)])
    assert yaml.safe_load(out)["pairs"]["explicit"] == [[25000, 25000]]


def test_append_without_explicit_key_raises_naming_the_file() -> None:
    with pytest.raises(ValueError, match=r"my_sweep\.yaml"):
        ga.append_to_config(NO_KEY, [], [(25000, 25000)], source="my_sweep.yaml")


# --------------------------------------------------------------- submission ledger
def test_unconfirmed_reads_the_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / ga.SUBMIT_SIDECAR
    assert ga.unconfirmed(CFG, sidecar) == ([25000, 50000], [(25000, 25000), (50000, 50000)])
    sidecar.write_text(json.dumps({"steps": [25000, 50000],
                                   "pairs": [[25000, 25000], [50000, 50000]]}))
    assert ga.unconfirmed(CFG, sidecar) == ([], [])


def _run_watcher(tmp_path: Path, monkeypatch, rcs: list[int], dry_run: bool = False,
                 follow: list[Path] | None = None) -> list[list]:
    """Drive main() over a family dir with one converted+passing step; mock gate + submit."""
    family = tmp_path / "family"
    (family / "step75000").mkdir(parents=True)
    (family / "step75000" / "conversion.json").write_text("{}")
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(_cfg_text(tmp_path))

    monkeypatch.setattr(ga, "load_paths_local", lambda: {"hf_cache_root": str(tmp_path)})
    monkeypatch.setattr(ga, "_load_probe", lambda _stem: [])
    monkeypatch.setattr(
        ga, "gate_step",
        lambda d, s, *a, **k: {"step": s, "pass": True, "delta_mean": -0.01, "batches": {},
                               "final_ln_w_norm": {}},
    )
    monkeypatch.setattr(ga.time, "sleep", lambda _s: None)
    calls: list[list] = []
    queued = list(rcs)

    class _R:
        def __init__(self, rc): self.returncode, self.stdout, self.stderr = rc, "Submitted 1", ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _R(queued.pop(0) if queued else 0)

    monkeypatch.setattr(ga.subprocess, "run", fake_run)
    ga.main(sweep_config=cfg_path, window="75000:75000:5000", diagonal="", family_dir=family,
            base_dir_name="x", real_repo="r", tolerance=0.1,
            preshuffled=tmp_path / "document", poll=0, dry_run=dry_run,
            follow_config=list(follow or []))
    return calls


def test_failed_submit_is_retried_and_only_rc0_writes_the_sidecar(tmp_path: Path, monkeypatch) -> None:
    # First submit fails: the watcher must not exit on it, and must resubmit on the next poll.
    calls = _run_watcher(tmp_path, monkeypatch, rcs=[1, 0])
    assert len(calls) == 2, "a failed submit must be retried on the next poll"
    ledger = json.loads((tmp_path / "family" / ga.SUBMIT_SIDECAR).read_text())
    assert ledger["steps"] == [25000, 75000]
    assert [75000, 75000] in ledger["pairs"]


def test_sidecar_absent_until_a_submit_succeeds(tmp_path: Path, monkeypatch) -> None:
    # rc=1 forever would loop; one failure then success is the minimal observable check that
    # the sidecar appears only after rc == 0 (the failing poll wrote nothing).
    seen: list[bool] = []
    real_write = ga.atomic_write_json

    def spy(path, obj, **kw):
        if Path(path).name == ga.SUBMIT_SIDECAR:
            seen.append(True)
        real_write(path, obj, **kw)

    monkeypatch.setattr(ga, "atomic_write_json", spy)
    calls = _run_watcher(tmp_path, monkeypatch, rcs=[1, 0])
    assert len(calls) == 2 and seen == [True]


def test_dry_run_previews_the_in_memory_pass(tmp_path: Path, monkeypatch) -> None:
    logs: list[str] = []
    monkeypatch.setattr(ga.logger, "info", lambda m, *a, **k: logs.append(str(m)))
    calls = _run_watcher(tmp_path, monkeypatch, rcs=[0], dry_run=True)
    assert calls == [], "dry-run must not submit"
    assert not (tmp_path / "family" / "step75000" / ga.GATE_FILE).exists()
    assert (tmp_path / "sweep.yaml").read_text() == _cfg_text(tmp_path)
    assert any("config += steps [75000]" in m for m in logs)
    assert any("would submit steps [25000, 75000]" in m and "sweep_ckpt_distill" in m
               for m in logs)


# ------------------------------------------------- preshuffled probe stem resolution
def test_preshuffled_override_wins_over_the_paths_yaml(monkeypatch) -> None:
    monkeypatch.setattr(ga, "load_paths_local", lambda: {ga.PRESHUFFLED_KEY: "/from/yaml/doc"})
    assert ga.resolve_preshuffled(Path("/cli/doc")) == Path("/cli/doc")


def test_preshuffled_falls_back_to_the_paths_yaml(monkeypatch) -> None:
    monkeypatch.setattr(ga, "load_paths_local", lambda: {ga.PRESHUFFLED_KEY: "/from/yaml/doc"})
    assert ga.resolve_preshuffled(None) == Path("/from/yaml/doc")


def test_preshuffled_unconfigured_errors_naming_the_key(monkeypatch) -> None:
    monkeypatch.setattr(ga, "load_paths_local", lambda: {"output_root": "/o"})
    with pytest.raises(ValueError, match=ga.PRESHUFFLED_KEY):
        ga.resolve_preshuffled(None)


# ------------------------------------------------------- locked, atomic config write
def _spy_writes(monkeypatch) -> tuple[list, list]:
    """Record (path, text) atomic writes and the lock paths taken around them."""
    writes: list[tuple[Path, str]] = []
    locks: list[Path] = []

    @contextlib.contextmanager
    def fake_lock(path, *a, **k):
        locks.append(Path(path))
        yield

    def fake_write(path, text):
        writes.append((Path(path), text))
        Path(path).write_text(text)

    monkeypatch.setattr(ga, "file_lock", fake_lock)
    monkeypatch.setattr(ga, "atomic_write_text", fake_write)
    return writes, locks


def test_config_write_is_atomic_under_the_queue_root_lock(tmp_path: Path, monkeypatch) -> None:
    writes, locks = _spy_writes(monkeypatch)
    _run_watcher(tmp_path, monkeypatch, rcs=[0])
    cfg_path = tmp_path / "sweep.yaml"
    assert [p for p, _ in writes] == [cfg_path]
    assert ga.parse_steps(cfg_path.read_text()) == [25000, 75000]
    assert locks == [tmp_path / "queue" / "ckpt-test.sweep.lock"]


def test_dry_run_writes_nothing_and_takes_no_lock(tmp_path: Path, monkeypatch) -> None:
    writes, locks = _spy_writes(monkeypatch)
    _run_watcher(tmp_path, monkeypatch, rcs=[0], dry_run=True)
    assert writes == [] and locks == []


# ------------------------------------------------------------ diagonal-only followers
FOLLOW_DIAG = """base_model: weightc-seed1

steps: [25000]

pairs:
  topology: diagonal
  explicit: []

animals: [dog]

jobq: {root: QUEUE_ROOT, queue_prefix: ckpt-test-seeds}
"""


def test_follow_append_diagonal_topology_grows_steps_only() -> None:
    text, steps, pairs = ga.follow_append(FOLLOW_DIAG, [25000, 75000, 80000])
    assert steps == [75000, 80000] and pairs == []
    assert ga.parse_steps(text) == [25000, 75000, 80000]
    assert ga.parse_pairs(text) == []
    assert ga.follow_append(text, [25000, 75000, 80000])[1:] == ([], [])


def test_follow_append_explicit_topology_adds_diagonal_pairs_only() -> None:
    text, steps, pairs = ga.follow_append(CFG, [25000, 50000, 75000])
    assert steps == [75000] and pairs == [(75000, 75000)]
    assert (75000, 75000) in ga.parse_pairs(text)


def test_follow_append_refuses_non_diagonal_followers() -> None:
    with pytest.raises(ValueError, match="diagonal-only"):
        ga.follow_append(NO_KEY, [75000])


def test_follower_is_submitted_after_the_primary_with_its_own_ledger(tmp_path: Path, monkeypatch) -> None:
    follow = tmp_path / "seeds.yaml"
    follow.write_text(FOLLOW_DIAG.replace("QUEUE_ROOT", str(tmp_path / "queue")))
    calls = _run_watcher(tmp_path, monkeypatch, rcs=[0, 0], follow=[follow])
    assert [c[c.index("--sweep-config") + 1] for c in calls] == [
        str(tmp_path / "sweep.yaml"), str(follow)], "primary first, then the follower"
    assert ga.parse_steps(follow.read_text()) == [25000, 75000]
    family = tmp_path / "family"
    ledger = ga.follower_sidecar(family, follow)
    assert json.loads(ledger.read_text())["steps"] == [25000, 75000]
    assert json.loads((family / ga.SUBMIT_SIDECAR).read_text())["sweep_config"] == str(tmp_path / "sweep.yaml")


def test_follower_waits_while_the_primary_submit_is_unconfirmed(tmp_path: Path, monkeypatch) -> None:
    follow = tmp_path / "seeds.yaml"
    follow.write_text(FOLLOW_DIAG.replace("QUEUE_ROOT", str(tmp_path / "queue")))
    calls = _run_watcher(tmp_path, monkeypatch, rcs=[1, 0, 0], follow=[follow])
    targets = [c[c.index("--sweep-config") + 1] for c in calls]
    assert targets == [str(tmp_path / "sweep.yaml")] * 2 + [str(follow)]


# --------------------------------------------------- submit timeout / ledger naming / scope
def test_submit_timeout_leaves_everything_unconfirmed(tmp_path: Path, monkeypatch) -> None:
    """A wedged submit is not a confirmation: same path as rc != 0."""
    def boom(cmd, **kw):
        assert kw["timeout"] == ga.SUBMIT_TIMEOUT_S
        raise ga.subprocess.TimeoutExpired(cmd, ga.SUBMIT_TIMEOUT_S)

    monkeypatch.setattr(ga.subprocess, "run", boom)
    sidecar = tmp_path / ga.SUBMIT_SIDECAR
    assert ga.submit_unconfirmed(tmp_path / "sweep.yaml", CFG, sidecar, False) == (
        [25000, 50000], [(25000, 25000), (50000, 50000)]
    )
    assert not sidecar.exists()


def test_submit_passes_the_growth_flag_not_the_blanket_update(tmp_path: Path, monkeypatch) -> None:
    calls: list[list] = []

    class _R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(ga.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _R())[1])
    ga.submit_unconfirmed(tmp_path / "sweep.yaml", CFG, tmp_path / ga.SUBMIT_SIDECAR, False)
    assert "--allow-snapshot-growth" in calls[0]
    assert "--allow-snapshot-update" not in calls[0]


def test_follower_ledgers_of_same_stem_in_different_dirs_differ(tmp_path: Path) -> None:
    a, b = tmp_path / "a" / "seeds.yaml", tmp_path / "b" / "seeds.yaml"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    la, lb = ga.follower_sidecar(tmp_path, a), ga.follower_sidecar(tmp_path, b)
    assert la != lb
    for lg in (la, lb):
        assert lg.name.startswith("gate_append_submitted.seeds.") and lg.name.endswith(".json")
        assert len(lg.name.split(".")[2]) == 8


def test_legacy_stem_only_ledger_is_still_read_but_never_written(tmp_path: Path, monkeypatch) -> None:
    follow = tmp_path / "seeds.yaml"
    follow.write_text(FOLLOW_DIAG)
    legacy = ga.legacy_follower_sidecar(tmp_path, follow)
    legacy.write_text(json.dumps({"steps": [25000], "pairs": []}))
    new = ga.follower_sidecar(tmp_path, follow)
    # read: the legacy ledger's confirmations still count while the new name is absent
    assert ga.unconfirmed(FOLLOW_DIAG, new, legacy) == ([], [])
    assert ga.unconfirmed(FOLLOW_DIAG, new) == ([25000], [])

    class _R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(ga.subprocess, "run", lambda cmd, **kw: _R())
    text = ga.append_to_config(FOLLOW_DIAG, [75000], [])
    ga.submit_unconfirmed(follow, text, new, False, legacy)
    assert new.exists() and json.loads(new.read_text())["steps"] == [25000, 75000]
    assert json.loads(legacy.read_text())["steps"] == [25000]  # untouched


DECOY = """base_model: weightc-seed1

steps: [25000]

probe_batches:
  - [1, 2]
  - [3, 4]

pairs:
  topology: explicit
  explicit:
    - [25000, 25000]

animals: [dog]
"""


def test_decoy_list_under_another_key_is_neither_parsed_nor_edited() -> None:
    assert ga.parse_pairs(DECOY) == [(25000, 25000)]
    out = ga.append_to_config(DECOY, [50000], [(50000, 50000)])
    assert yaml.safe_load(out)["probe_batches"] == [[1, 2], [3, 4]]
    assert yaml.safe_load(out)["pairs"]["explicit"] == [[25000, 25000], [50000, 50000]]
    assert ga.parse_pairs(out) == [(25000, 25000), (50000, 50000)]


def test_decoy_list_after_the_explicit_block_is_left_untouched() -> None:
    text = DECOY.replace("probe_batches:\n  - [1, 2]\n  - [3, 4]\n\n", "") + """
probe_batches:
  - [1, 2]
"""
    out = ga.append_to_config(text, [], [(50000, 50000)])
    assert yaml.safe_load(out)["probe_batches"] == [[1, 2]]
    assert yaml.safe_load(out)["pairs"]["explicit"] == [[25000, 25000], [50000, 50000]]


def test_parse_window_accepts_ranges_and_explicit_steps() -> None:
    assert ga.parse_window("") == []
    assert ga.parse_window("75000:85000:5000") == [75000, 80000, 85000]
    # Angela 2026-09-12: 130k and 140k but NOT 135k -> explicit steps beside the range.
    assert ga.parse_window("75000:85000:5000,140000,130000") == [75000, 80000, 85000, 130000, 140000]
    assert ga.parse_window("130000, 75000:80000:5000 ,130000") == [75000, 80000, 130000]
