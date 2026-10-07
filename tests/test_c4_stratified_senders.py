"""The stratified C4 sender program: selection rule, band membership, and its job files.

Guards the 2026-09-20 replacement of the abandoned lowest-ΔCE C4 cut by a ΔCE-stratified
sender set (run_logs/campaigns/olmo_c4_to_dclm_pilot_20260920/select_stratified.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DCLM = REPO / "run_logs" / "campaigns" / "olmo_c4_to_dclm_pilot_20260920"
DOLMA = REPO / "run_logs" / "campaigns" / "olmo_c4_to_dolma_pilot_20260920"
sys.path.insert(0, str(DCLM))

import select_stratified as sel  # noqa: E402


@pytest.fixture(scope="module")
def rows() -> list[tuple[str, float, str]]:
    return sel.select()


def test_twentytwo_senders_over_six_bands(rows):
    assert len(rows) == sel.N_SENDERS == 22
    assert len({t for t, _, _ in rows}) == 22
    per_band = {b: 0 for _, _, b in rows}
    for _, _, band in rows:
        per_band[band] += 1
    # Four per band, except the two bands that hold only three gated teachers and are
    # deliberately not padded from a neighbour.
    assert sorted(per_band.values()) == [3, 3, 4, 4, 4, 4]


def test_every_pick_is_the_band_nearest_order_statistic(rows):
    """Each band's picks are its evenly spaced order statistics (one documented override)."""
    smoke = dict(sel.smoke_rows())
    by_band: dict[str, list[str]] = {}
    for tag, dce, band in rows:
        assert smoke[tag] == pytest.approx(dce)  # ΔCE comes from the smoke table
        by_band.setdefault(band, []).append(tag)
    all_rows = sel.smoke_rows()
    for bi, (lo, hi) in enumerate(sel.BANDS):
        band = f"{lo:.2f}-{hi:.2f}"
        avail = [r for r in all_rows
                 if lo <= r[1] < hi or (hi == sel.BANDS[-1][1] and r[1] == hi)]
        k = min(sel.PER_BAND, len(avail))
        want = [sel.OVERRIDES.get((bi, j), avail[round(j * (len(avail) - 1) / (k - 1))][0])
                for j in range(k)]
        assert by_band[band] == want
        for tag in by_band[band]:
            assert lo <= smoke[tag] <= hi


def test_only_one_override_and_it_swaps_adjacent_order_statistics():
    assert sel.OVERRIDES == {(0, 0): "ep5_lr5e-3_r8a4"}
    lo, hi = sel.BANDS[0]
    avail = [r for r in sel.smoke_rows() if lo <= r[1] < hi]
    assert [t for t, _ in avail[:2]] == ["ep5_lr2e-3_r128a16", "ep5_lr5e-3_r8a4"]


def test_every_sender_has_a_teacher_cfg(rows):
    for tag, _, _ in rows:
        assert (DCLM / "cfg" / f"teacher_{tag}.yaml").exists()


def _keys(path: Path) -> list[str]:
    return [json.loads(ln)["key"] for ln in path.read_text().splitlines() if ln.strip()]


def test_job_files_cover_the_set_and_no_abandoned_tag(rows):
    tags = [t for t, _, _ in rows]
    dropped = sel.abandoned_tags(set(tags))
    assert set(dropped) == {
        "ep5_lr2e-3_r128a16", "ep5_lr5e-3_r4a4", "ep5_lr2e-3_r64a16",
        "ep5_lr5e-3_r32a4", "ep5_lr2e-3_r32a16", "ep5_lr2e-3_r4a16",
    }
    ngen = _keys(DCLM / "ngen_strat.jobs.jsonl")
    # Generation only for the senders whose pool the first cut did not already build.
    assert ngen == [f"ngen|{t}" for t in tags if t not in sel.POOLED]
    for receiver in (DCLM, DOLMA):
        stu = _keys(receiver / "students_strat.jobs.jsonl")
        # The one student already running under the first cut is not requeued.
        assert stu == [f"stu|{t}|s1" for t in tags if t != "ep5_lr5e-3_r8a4"]
    for key in ngen + _keys(DCLM / "students_strat.jobs.jsonl") \
            + _keys(DOLMA / "students_strat.jobs.jsonl"):
        assert not any(d in key for d in dropped)
        # The pre-stratified extra sender is left to finish, never requeued.
        assert sel.EXTRA_OUTSIDE_DESIGN not in key
