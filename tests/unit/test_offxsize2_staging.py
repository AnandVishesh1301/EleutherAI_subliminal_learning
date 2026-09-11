"""S0 staging for offxsize2: adoption by sha, the disk gate, and the footprint smoke."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import stage_offxsize2_assets as staging
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.utils.io import file_sha256

POOL_ROWS = [{"numbers": [7, 8], "completion": " 7, 8"}]


def _paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=tmp_path / "hot",
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "datasets",
        wandb_dir=tmp_path / "wandb",
    )


def _plant_pool(
    pool_dir: Path, *, sha: str | None = None, n_raw: int = 100_000
) -> Path:
    """One v1 pool: `filtered.jsonl` plus the mode manifest that hashes it."""
    mode = layout.filtered_dir(pool_dir)
    mode.mkdir(parents=True, exist_ok=True)
    filtered = mode / staging.FILTERED_FILE
    filtered.write_text("\n".join(json.dumps(r) for r in POOL_ROWS) + "\n")
    (mode / staging.POOL_MANIFEST).write_text(
        json.dumps(
            {
                "filtered_sha256": sha or file_sha256(filtered),
                "n_filtered": len(POOL_ROWS),
                "n_raw": n_raw,
            }
        )
    )
    return filtered


def _plant_teacher(v1_root: Path, rung: plan.TeacherRung, size: str) -> Path:
    leaf = layout.teacher_leaf_dir(
        v1_root, rung, size, plan.TEACHER_FTSEEDS[0], animal=plan.PRIMARY_ANIMAL
    )
    ckpt = leaf / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (leaf / "train_manifest.json").write_text(json.dumps({"teacher_ftseed": 1}))
    return leaf


def _plant_v1(v1_root: Path) -> None:
    """v1's closed run set: one teacher rung at every size, plus the traitless pools."""
    for size in plan.SIZES:
        _plant_teacher(v1_root, plan.R1, size)
        _plant_pool(layout.base_pool_dir(v1_root, size))
    _plant_pool(layout.rng_pool_dir(v1_root))


def _listing(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _stub_gates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(staging, "dataset_gates", lambda *_a, **_k: {})
    monkeypatch.setattr(staging, "hub_gates", lambda *_a, **_k: {})
    monkeypatch.setattr(staging, "free_disk_gib", lambda _p: 10_000.0)
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("# prereg\n")
    monkeypatch.setattr(staging, "PREREG_PATH", prereg)


# ---- adoption plan ------------------------------------------------------------


def test_adoption_plan_covers_the_teacher_ladder_and_the_traitless_pools(
    tmp_path: Path,
) -> None:
    v1_root, campaign = tmp_path / "v1", tmp_path / "v2"
    items = staging.adoption_plan(v1_root, campaign)
    teachers = [i for i in items if i.kind == "teacher"]
    pools = [i for i in items if i.kind == "pool"]
    assert len(teachers) == len(plan.TEACHER_RUNGS) * len(plan.SIZES) * len(
        plan.TEACHER_FTSEEDS
    )
    assert [i.key for i in pools] == [f"base/{s}" for s in plan.SIZES] + ["rng"]
    for item in items:
        assert v1_root in item.src.parents, item.key
        assert campaign in item.dst.parents, item.key
        assert campaign not in item.src.parents, "v1's tree is never written to"
    assert all(plan.PRIMARY_ANIMAL in i.src.parts for i in teachers)
    assert not any(i.kind == "pool" and i.key.startswith("teacher/") for i in items), (
        "v2 sizes and keys its own teacher pools; a v1 pool carries another lineage"
    )


def test_v1_campaign_root_is_the_closed_run_set(tmp_path: Path) -> None:
    root = staging.v1_campaign_root(tmp_path)
    assert root == tmp_path / plan.V1_CAMPAIGN_SUBDIR / plan.V1_RUN_SET_ID
    assert plan.RUN_SET_ID not in root.parts


def test_cold_target_is_the_plan_site_for_the_default_root() -> None:
    assert staging.cold_target(plan.COLD_ROOT) == plan.cold_campaign_root()
    assert staging.cold_target(Path("/tmp/cold")) == (
        Path("/tmp/cold") / plan.CAMPAIGN_SUBDIR / plan.RUN_SET_ID
    )


# ---- adoption -----------------------------------------------------------------


def test_adoption_copies_by_sha_and_leaves_the_source_untouched(
    tmp_path: Path,
) -> None:
    v1_root, campaign = tmp_path / "v1", tmp_path / "v2"
    _plant_v1(v1_root)
    before = _listing(v1_root)
    adopted, missing = staging.adopt(
        staging.adoption_plan(v1_root, campaign), dry_run=False
    )
    assert missing == [
        item.key
        for item in staging.adoption_plan(v1_root, campaign)
        if not item.src.is_dir()
    ]
    assert _listing(v1_root) == before, "the adoption source is read-only"
    rng = adopted["rng"]
    assert rng["n_filtered"] == len(POOL_ROWS)
    assert rng["filtered_sha256"] == rng["manifest_sha256"]
    copied = layout.filtered_dir(layout.rng_pool_dir(campaign)) / staging.FILTERED_FILE
    assert file_sha256(copied) == rng["filtered_sha256"]
    teacher_key = (
        f"{plan.PRIMARY_ANIMAL}/70m/{plan.R1.key}/ftseed{plan.TEACHER_FTSEEDS[0]}"
    )
    dst = Path(adopted[teacher_key]["dst"])
    assert (
        dst / "checkpoints" / "adapter_model.safetensors"
    ).read_bytes() == b"weights"
    assert not dst.is_symlink(), "adopted artifacts are copied, never referenced"


def test_adoption_re_run_is_a_noop(tmp_path: Path) -> None:
    v1_root, campaign = tmp_path / "v1", tmp_path / "v2"
    _plant_v1(v1_root)
    items = staging.adoption_plan(v1_root, campaign)
    staging.adopt(items, dry_run=False)
    adopted, _ = staging.adopt(items, dry_run=False)
    actions = {f["action"] for record in adopted.values() for f in record["files"]}
    assert actions == {"noop"}, "a hash-identical destination is never rewritten"
    assert all(record["n_files"] > 0 for record in adopted.values())


def test_adoption_refuses_a_pool_whose_bytes_are_not_the_ones_the_gate_froze(
    tmp_path: Path,
) -> None:
    """A pool that hashes differently would put unhashed rows under every Delta."""
    v1_root, campaign = tmp_path / "v1", tmp_path / "v2"
    _plant_v1(v1_root)
    _plant_pool(layout.rng_pool_dir(v1_root), sha="deadbeef")
    with pytest.raises(RuntimeError, match="not the ones the gate froze"):
        staging.adopt(staging.adoption_plan(v1_root, campaign), dry_run=False)


def test_a_rung_v1_never_trained_is_recorded_not_fatal(tmp_path: Path) -> None:
    """v2 trains every teacher itself, so a hole costs compute, not validity."""
    v1_root, campaign = tmp_path / "v1", tmp_path / "v2"
    _plant_v1(v1_root)
    adopted, missing = staging.adopt(
        staging.adoption_plan(v1_root, campaign), dry_run=False
    )
    assert f"{plan.PRIMARY_ANIMAL}/70m/{plan.R4.key}/ftseed1" in missing
    assert f"{plan.PRIMARY_ANIMAL}/70m/{plan.R1.key}/ftseed1" in adopted


def test_adoption_dry_run_writes_nothing(tmp_path: Path) -> None:
    v1_root, campaign = tmp_path / "v1", tmp_path / "v2"
    _plant_v1(v1_root)
    staging.adopt(staging.adoption_plan(v1_root, campaign), dry_run=True)
    assert not campaign.exists()


# ---- gates --------------------------------------------------------------------


def test_disk_gate_refuses_when_the_cold_mount_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(staging, "free_disk_gib", lambda _p: 1.0)
    with pytest.raises(RuntimeError, match="disk gate FAILED"):
        staging.disk_gate(tmp_path, min_free_gib=50.0)


def test_disk_gate_measures_the_hot_roots_without_gating_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign root is on cold BECAUSE the hot mount is nearly full."""
    free = {str(tmp_path / "cold"): 900.0, str(tmp_path / "hot"): 0.5}
    monkeypatch.setattr(staging, "free_disk_gib", lambda p: free[str(p)])
    measured = staging.disk_gate(
        tmp_path / "cold", hot_roots=(tmp_path / "hot",), min_free_gib=50.0
    )
    assert measured == free


# ---- the footprint smoke ------------------------------------------------------


def test_smoke_refuses_on_the_forbidden_node(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=plan.FORBIDDEN_NODE):
        staging.assert_smoke_node(plan.FORBIDDEN_NODE)
    with pytest.raises(RuntimeError, match=plan.FORBIDDEN_NODE):
        staging.run_smoke(
            _paths(tmp_path), tmp_path / "campaign", hostname=plan.FORBIDDEN_NODE
        )
    assert not (tmp_path / "campaign").exists(), "the guard fires before any write"
    assert (
        staging.assert_smoke_node(f"{plan.CAMPAIGN_NODE}.local") == plan.CAMPAIGN_NODE
    )


def test_smoke_refuses_without_the_adopted_rng_pool(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="run the adoption phase first"):
        staging.run_smoke(
            _paths(tmp_path), tmp_path / "campaign", hostname=plan.CAMPAIGN_NODE
        )


def test_gpu_used_mib_reads_per_device_not_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nvidia-smi pids are host-namespace here, so the device index is the only key."""
    calls: list[list[str]] = []

    def _run(cmd, **_k):  # noqa: ANN001, ANN202
        calls.append(cmd)
        return SimpleNamespace(stdout="0, 1234\n1, 0\nnot, a row\n")

    monkeypatch.setattr(staging.subprocess, "run", _run)
    assert staging.gpu_used_mib() == {0: 1234, 1: 0}
    assert "--query-gpu=index,memory.used" in calls[0]


def test_smoke_stamps_the_delta_over_the_pre_launch_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A card that already held VRAM must not be charged to the cell being measured."""

    class _Proc:
        pid = 4242
        returncode = 0

        def __init__(self) -> None:
            self._alive = 2

        def poll(self) -> int | None:
            self._alive -= 1
            return None if self._alive > 0 else 0

        def wait(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    campaign = tmp_path / "campaign"
    _plant_pool(layout.rng_pool_dir(campaign))
    readings = iter([{0: 3_000}, {0: 9_000}, {0: 7_000}])
    monkeypatch.setattr(staging, "gpu_used_mib", lambda: next(readings))
    monkeypatch.setattr(staging.subprocess, "Popen", lambda *_a, **_k: _Proc())
    payload = staging.run_smoke(
        _paths(tmp_path),
        campaign,
        sizes=("70m",),
        sample_s=0.0,
        hostname=plan.CAMPAIGN_NODE,
    )
    assert payload["students"]["70m"]["peak_mib"] == 6_000, "9000 used - 3000 baseline"
    assert payload["students"]["70m"]["mem_mib"] == (
        6_000 + staging.FOOTPRINT_HEADROOM_MIB
    )
    assert payload["host"] == plan.CAMPAIGN_NODE
    stamped = json.loads(
        (staging.campaign_logs_dir(campaign) / plan.FOOTPRINT_REPORT).read_text()
    )
    assert stamped["students"]["70m"]["peak_mib"] == 6_000
    assert not (staging.campaign_logs_dir(campaign) / staging.SMOKE_SUBDIR).exists()


def test_smoke_refuses_a_stamp_from_a_cell_that_never_reached_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Dead:
        pid = 1
        returncode = 1

        def poll(self) -> int:
            return 1

        def wait(self) -> int:
            return 1

        def kill(self) -> None:
            return None

    campaign = tmp_path / "campaign"
    _plant_pool(layout.rng_pool_dir(campaign))
    monkeypatch.setattr(staging, "gpu_used_mib", lambda: {0: 0})
    monkeypatch.setattr(staging.subprocess, "Popen", lambda *_a, **_k: _Dead())
    with pytest.raises(RuntimeError, match="no nvidia-smi reading"):
        staging.run_smoke(
            _paths(tmp_path),
            campaign,
            sizes=("70m",),
            sample_s=0.0,
            hostname=plan.CAMPAIGN_NODE,
        )
    assert not (staging.campaign_logs_dir(campaign) / plan.FOOTPRINT_REPORT).is_file()


def test_smoke_command_measures_the_widest_committed_recipe(tmp_path: Path) -> None:
    recipe = staging.heaviest_recipe()
    assert recipe.lora_r == max(r.lora_r for r in plan.STUDENT_GRID)
    cmd = staging.smoke_command(
        "410m",
        pool_dir=tmp_path / "pool",
        dataset_tag="ex10-20_prefix_n100k",
        paths_config=tmp_path / "paths.yaml",
    )
    assert cmd[cmd.index("--recipe-key") + 1] == recipe.key
    assert cmd[cmd.index("--kind") + 1] == "rng_control"
    assert cmd[cmd.index("--student-size") + 1] == "410m"
    assert cmd[cmd.index("--pool-dir") + 1] == str(tmp_path / "pool")
    assert cmd[cmd.index("--paths-config") + 1] == str(tmp_path / "paths.yaml")
    assert "--no-wandb" in cmd


def test_smoke_dry_run_neither_launches_nor_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    _plant_pool(layout.rng_pool_dir(campaign))
    monkeypatch.setattr(
        staging.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("dry-run must launch nothing"),
    )
    report = staging.run_smoke(
        _paths(tmp_path), campaign, hostname=plan.CAMPAIGN_NODE, dry_run=True
    )
    assert report["dry_run"] is True
    assert not (staging.campaign_logs_dir(campaign) / staging.SMOKE_SUBDIR).exists()


def test_footprint_payload_stamps_the_measured_peak_plus_the_headroom() -> None:
    peaks = {"70m": 2_000, "160m": 3_000, "410m": 11_000}
    payload = staging.footprint_payload(peaks)
    assert payload["headroom_mib"] == staging.FOOTPRINT_HEADROOM_MIB
    assert payload["recipe_key"] == staging.heaviest_recipe().key
    for size, peak in peaks.items():
        row = payload["students"][size]
        assert row["peak_mib"] == peak
        assert row["mem_mib"] == peak + staging.FOOTPRINT_HEADROOM_MIB
        assert row["slots"] == plan.STUDENT_FOOTPRINT[size].slots
        assert row["source"] == "measured"
    # The other stages keep the plan's declared numbers, marked as such.
    assert set(payload["declared"]) == {"teacher", "numgen", "capability"}
    assert payload["declared"]["teacher"]["410m"]["mem_mib"] == (
        plan.TEACHER_FOOTPRINT["410m"].mem_mib
    )


def test_footprint_payload_flags_a_measurement_the_slot_weight_cannot_pack() -> None:
    """The memory gate still admits on `mem_mib`, so this costs concurrency, not an OOM."""
    payload = staging.footprint_payload({"70m": 20_000})
    row = payload["students"]["70m"]
    assert row["packs_at_declared_slots"] is False
    assert row["suggested_slots"] > row["slots"]
    assert row["mem_mib"] * (plan.CAP_PER_GPU // row["suggested_slots"]) <= (
        plan.GPU_MEM_MIB
    )


def test_footprint_payload_refuses_a_cell_that_cannot_fit_one_card() -> None:
    with pytest.raises(RuntimeError, match="does not fit one A40"):
        staging.footprint_payload({"410m": plan.GPU_MEM_MIB})


# ---- run_staging --------------------------------------------------------------


def test_run_staging_redirects_before_writing_under_the_link_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`logs/` sits under the symlink site, so a write first would strand the tree hot."""
    _stub_gates(monkeypatch, tmp_path)
    paths = _paths(tmp_path)
    v1_root = staging.v1_campaign_root(paths.output_root)
    _plant_v1(v1_root)
    cold = tmp_path / "cold"
    report = staging.run_staging(
        paths, plan.HUB_ORG, v1_root=v1_root, cold_root=cold, dry_run=False
    )
    link = plan.campaign_root(paths.output_root)
    target = staging.cold_target(cold)
    assert link.is_symlink() and link.resolve() == target.resolve()
    assert report["symlink"]["action"] == "create"
    logs = staging.campaign_logs_dir(link)
    assert (logs / plan.PREFLIGHT_REPORT).is_file()
    assert (logs / plan.ADOPTION_MANIFEST).is_file()
    # Every adopted byte landed on the cold mount, reached through the hot path.
    assert (target / "logs" / plan.PREFLIGHT_REPORT).is_file()
    manifest = json.loads((logs / plan.ADOPTION_MANIFEST).read_text())
    assert manifest["source_run_set_id"] == plan.V1_RUN_SET_ID
    assert set(manifest["pools"]) == {f"base/{s}" for s in plan.SIZES} | {"rng"}
    assert report["node"]["forbidden"] == plan.FORBIDDEN_NODE
    assert report["footprint_measured"] is False


def test_run_staging_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gates(monkeypatch, tmp_path)
    paths = _paths(tmp_path)
    v1_root = staging.v1_campaign_root(paths.output_root)
    _plant_v1(v1_root)
    cold = tmp_path / "cold"
    staging.run_staging(
        paths, plan.HUB_ORG, v1_root=v1_root, cold_root=cold, dry_run=False
    )
    first = _listing(staging.cold_target(cold))
    report = staging.run_staging(
        paths, plan.HUB_ORG, v1_root=v1_root, cold_root=cold, dry_run=False
    )
    assert report["symlink"]["action"] == "noop"
    second = _listing(staging.cold_target(cold))
    assert {k: v for k, v in first.items() if "logs/" not in k} == {
        k: v for k, v in second.items() if "logs/" not in k
    }


def test_run_staging_gates_before_it_writes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gates(monkeypatch, tmp_path)
    monkeypatch.setattr(staging, "free_disk_gib", lambda _p: 1.0)
    paths = _paths(tmp_path)
    v1_root = staging.v1_campaign_root(paths.output_root)
    _plant_v1(v1_root)
    with pytest.raises(RuntimeError, match="disk gate FAILED"):
        staging.run_staging(
            paths,
            plan.HUB_ORG,
            v1_root=v1_root,
            cold_root=tmp_path / "cold",
            dry_run=False,
        )
    assert not plan.campaign_root(paths.output_root).exists()
    assert not (tmp_path / "cold").exists()


def test_run_staging_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gates(monkeypatch, tmp_path)
    paths = _paths(tmp_path)
    v1_root = staging.v1_campaign_root(paths.output_root)
    _plant_v1(v1_root)
    report = staging.run_staging(
        paths,
        plan.HUB_ORG,
        v1_root=v1_root,
        cold_root=tmp_path / "cold",
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert not plan.campaign_root(paths.output_root).exists()
    assert not (tmp_path / "cold").exists()
