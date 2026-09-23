"""Part II staging, smoke and launcher wiring (no disk outside tmp_path, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import smoke_ws160m as smoke
from scripts import stage_ws160m_assets as stage
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.selection import Recipe

_LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/launch_ws160m_campaign.sh"
_GEN2_STEPS = {0, 1000, 2000, 4000, 8000, 16000, 32000, 48000, 96000, 143000}


def test_gen2_required_revisions_follow_the_enumeration() -> None:
    needed = stage.required_revisions(wave=models.WAVE_3, gen=2)
    assert set(needed) == {m.base for m in models.ROSTER}
    for m in models.ROSTER:
        if m.source == "local":
            assert needed[m.base] == _GEN2_STEPS, m.base
    assert needed["seed2"] >= {0, 143000} and needed["pythia-160m"] == _GEN2_STEPS
    assert needed["data-seed2"] == {143000}
    assert all(s in models.CAMPAIGN_STEPS or s == 0 for v in needed.values() for s in v)
    gen1 = stage.required_revisions(wave=models.WAVE_NOW)
    assert gen1["pile-ws1"] == {0, 2000, 8000, 16000, 48000, 143000}
    assert "pile-main" not in gen1 and "pile-ws2" not in gen1


def test_reports_are_per_generation_and_never_shared(tmp_path: Path) -> None:
    paths = tmp_path / "paths.yaml"
    paths.write_text(yaml.safe_dump({"audit_root": str(tmp_path)}))
    bm = tmp_path / models.BASE_MODELS_DIRNAME
    staged = stage.staging_dir(bm)
    staged.mkdir(parents=True)
    assert stage.staging_report_path(bm).name == "staging_report.json"
    assert stage.staging_report_path(bm, "grid").name == "staging_report_ws160m2.json"
    assert stage.smoke_report_path(bm, "matrix").name == "smoke_report_ws160m2.json"
    for name in (plan.STAGING_REPORT, plan.SMOKE_REPORT):
        (staged / name).write_text(json.dumps({"passed": True}))
    assert stage.reports_ok(paths, require_smoke=True)
    assert not stage.reports_ok(paths, require_smoke=False, partition="grid")
    (staged / plan.GEN2_STAGING_REPORT).write_text(json.dumps({"passed": True}))
    assert stage.reports_ok(paths, require_smoke=False, partition="grid")
    assert not stage.reports_ok(paths, require_smoke=True, partition="matrix")
    (staged / plan.GEN2_SMOKE_REPORT).write_text(json.dumps({"passed": False}))
    assert not stage.reports_ok(paths, require_smoke=True, partition="matrix")


def test_staged_tokens_counts_only_verified_or_converted_revisions(
    tmp_path: Path,
) -> None:
    report = tmp_path / "r.json"
    assert stage.staged_tokens(report) == set()
    report.write_text(
        json.dumps(
            {
                "hub_revisions": {"seed1@step4000": {}},
                "converted": {
                    "pile-main": {
                        "step0": {"action": "verified"},
                        "step4000": {"action": "converted"},
                        "step96000": {"action": "would_convert"},
                    }
                },
            }
        )
    )
    assert stage.staged_tokens(report) == {
        "seed1@step4000",
        "pile-main@step0",
        "pile-main@step4000",
    }


def test_gen2_smoke_is_a_new_local_base_off_the_frozen_recipe() -> None:
    spec = smoke.SMOKE_SPECS[2]
    assert spec.smoke_root == Path(
        "/mnt/cold-1/soar-subliminal/vishesh/smokes/20260918_ws160m2"
    )
    assert (spec.animal, spec.teacher_base, spec.student_base) == (
        "lion",
        "pile-main",
        "pile-ws2",
    )
    assert spec.recipe == Recipe(32, 8, 4.0e-4) != plan.FROZEN_RECIPE
    assert spec.recipe in plan.LAW_ARM_RECIPES and spec.lmeval
    assert plan.partition(spec.partition).smoke_report == plan.GEN2_SMOKE_REPORT
    run_sets = {
        p.run_set_id
        for p in (*plan.PARTITIONS.values(), *plan.GEN2_PARTITIONS.values())
    }
    assert all(s.run_set_id not in run_sets for s in smoke.SMOKE_SPECS.values())
    gen1 = smoke.SMOKE_SPECS[1]
    assert gen1.smoke_root.name == "20260916_ws160m" and not gen1.lmeval
    assert plan.partition(gen1.partition).smoke_report == plan.SMOKE_REPORT


def test_launcher_keeps_every_guard_and_matches_its_own_session_exactly() -> None:
    text = _LAUNCHER.read_text()
    for guard in (
        'LOCK_PREFIX="${SL_JOBQ_LOCK_PREFIX:-}"',
        "git grep -q SHARED_LOCK_OWNER",
        "grep -q SL_JOBQ_LOCK_PREFIX",
        'git branch -r --contains "$FULL_SHA"',
        '"$HOST" != "$NODE"',
        'basename "$(dirname "$REPO")")" != ".code_pins"',
        'pgrep -af "scripts.jobq work --root $ROOT "',
        '-e "$ROOT/stop.$HOST"',
        'cp -a "$POLICY" "$POLICY.bak_pre_launch_',
        'if [ -z \\"\\$Q\\" ]',
        '--queues \\"\\$Q\\"',
        "state['pool_scope'] = 'campaign'",
    ):
        assert guard in text, guard
    assert 'tmux has-session -t "=$SESSION"' in text
    assert "part.tmux_session" in text and "p.TMUX_SESSION" not in text
    assert "part.deadline_hours" in text and "partition='$PARTITION'" in text
    assert "tmux kill-session" not in text and "tmux attach -t ws160m\n" not in text
