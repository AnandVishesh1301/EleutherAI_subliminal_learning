from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from sdf_docgen.position_cli import load_run_config
from sdf_docgen.sdfpos_census import (
    GEN1_FIRST_WAVE,
    RUN_FEEDS,
    WAVE3_RUNS,
    buffered,
    load_generations,
    project,
    scale_allocation,
    stop_decisions,
)

_ROOT = Path(__file__).resolve().parents[3]


def test_scale_allocation_sums_to_size_and_never_drops_below_v2() -> None:
    v2 = {("a",): 3, ("b",): 2, ("c",): 1}
    out = scale_allocation(v2, 6, 9)
    assert sum(out.values()) == 9
    assert out == {("a",): 5, ("b",): 3, ("c",): 1} or out == {
        ("a",): 4,
        ("b",): 3,
        ("c",): 2,
    }
    assert all(out[s] >= v2[s] for s in v2)
    assert scale_allocation(v2, 6, 6) == v2
    with pytest.raises(ValueError):
        scale_allocation(v2, 7, 9)


def test_buffered_raises_only_the_increment() -> None:
    assert buffered({("a",): 14, ("b",): 3}, {("a",): 10, ("b",): 3}, 0.10) == {
        ("a",): 15,
        ("b",): 3,
    }


def test_project_fills_at_the_smallest_arm_and_moves_the_tail_to_matched_surplus() -> (
    None
):
    target = {("a",): 4, ("b",): 2}
    available = {
        "x": Counter({("a",): 3, ("b",): 2, ("c",): 5}),
        "y": Counter({("a",): 5, ("b",): 4, ("c",): 1}),
    }
    p = project(available, target, tail=1)
    assert (p.size, p.filled, p.shortfall, p.spare) == (6, 5, 1, 1)
    assert p.ready and p.arm_deficit == {"x": 1, "y": 0}
    assert not project(available, target, tail=0).ready
    none_spare = {
        "x": Counter({("a",): 3, ("b",): 2}),
        "y": Counter({("a",): 5, ("b",): 2}),
    }
    assert not project(none_spare, target, tail=5).ready


def test_stop_decisions_wait_for_every_fed_arm() -> None:
    def res(ready: bool, deficits: dict[str, int]) -> dict:
        return {"buffered": {"ready": ready, "arm_deficit": deficits}}

    stops = stop_decisions(
        {
            "position": res(False, {"start": 0, "middle": 5, "end": 0}),
            "valence": res(False, {"neutral": 0, "negative": 3, "position": 0}),
        }
    )
    assert stops["position_w3s"] and stops["neutral_w3"]
    assert not stops["position_w3m"] and not stops["negative_w3"]
    assert set(stops) == set(RUN_FEEDS)
    assert all(r in RUN_FEEDS for runs in WAVE3_RUNS.values() for r in runs)


def _fake_run(root: Path, run: str, base: str, rows: int) -> None:
    cfg = load_run_config(_ROOT / f"configs/sdfpos/{base}.yaml")
    chunk = root / run / "chunks" / "chunk-00000-a0"
    chunk.mkdir(parents=True)
    (root / run / "run_config.json").write_text(json.dumps({"config": cfg}))
    (chunk / "DONE.json").write_text(json.dumps({"chunk_id": 0}))
    with (chunk / "docs.jsonl").open("w") as f:
        for i in range(rows):
            f.write(json.dumps({"text": f"{run} {i}", "run": run}) + "\n")


def test_load_generations_puts_every_wave3_doc_after_every_v2_doc(tmp_path) -> None:
    v2, w3 = tmp_path / "v2", tmp_path / "w3"
    for run in (
        "neutral",
        "neutral_w2",
        "negative",
        "negative_w2",
        "position",
        "position_w2",
    ):
        _fake_run(v2, run, run, 2)
    _fake_run(w3, "neutral_w3", "neutral_w3", 2)
    _fake_run(w3, "position_w3s", "position_w3s", 2)
    docs, _, sources, gen1 = load_generations(v2, w3, "valence")
    runs = [d.row["run"] for d in docs]
    assert runs[:12] == [
        r
        for r in (
            "neutral",
            "neutral_w2",
            "negative",
            "negative_w2",
            "position",
            "position_w2",
        )
        for _ in range(2)
    ]
    assert runs[12:] == ["neutral_w3"] * 2 + ["position_w3s"] * 2
    assert gen1 == {d.doc_id for d in docs[12:]}
    assert all(d.order[0] >= GEN1_FIRST_WAVE for d in docs[12:])
    assert len(sources) == 8
