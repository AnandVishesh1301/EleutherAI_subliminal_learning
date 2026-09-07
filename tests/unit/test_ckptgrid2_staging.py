"""S0 staging: assembly plan, I10 order, prefetch coverage, dry-run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import stage_ckptgrid2_assets as staging
from subliminal_learning.campaign import ckptgrid2_jobs as jobs
from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.ckptgrid2_jobs import HubRef
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir


def _paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=tmp_path / "hot",
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "datasets",
    )


def _write_ref_manifest(output_root: Path, refs: list[HubRef] | None = None) -> Path:
    refs = list(refs) if refs is not None else list(jobs.prefetch_refs())
    path = staging._reference_prefetch_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "at": "t",
                "n_refs": len(refs),
                "refs": [ref.to_dict() for ref in refs],
            }
        )
    )
    return path


def _touch_campaign_done(output_root: Path) -> None:
    for run_set_id in plan.REFERENCE_RUN_SETS.values():
        path = staging._reference_done_path(output_root, run_set_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("done\n")


def _plant_assembly_targets(
    output_root: Path, *, skip: Path | None = None
) -> tuple[staging.AssemblyLink, ...]:
    items = staging._assembly_links(output_root)
    for item in items:
        if skip is not None and item.target == skip:
            continue
        item.target.mkdir(parents=True, exist_ok=True)
    return items


def _plant_staging_tree(output_root: Path) -> None:
    _touch_campaign_done(output_root)
    _write_ref_manifest(output_root)
    _plant_assembly_targets(output_root)


def _stub_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(staging, "dataset_checks", lambda *_a, **_k: {})
    monkeypatch.setattr(staging, "disk_gate_readings", lambda **_k: {})


def _listing(root: Path) -> list[str]:
    if not root.exists():
        return []
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append(f"L {rel} -> {path.readlink()}")
        elif path.is_dir():
            rows.append(f"D {rel}")
        else:
            rows.append(f"F {rel} {path.read_bytes()!r}")
    return rows


def test_assembly_plan_covers_reused_teachers_and_controls(
    tmp_path: Path,
) -> None:
    items = staging._assembly_links(tmp_path)
    teacher_items = [i for i in items if i.kind == "teacher"]
    control_items = [i for i in items if i.kind == "control"]
    expected_teachers = {
        (animal, base, step)
        for animal, _sb, base, _ss, step, _gt, tag in pairs.reference_cells()
        if tag == plan.TEACHER_PARAM_TAG_LORA
    }
    assert len(teacher_items) == len(expected_teachers) == 48
    expected_controls = {
        (animal, label, tag) for animal, label, tag, _g in pairs.staged_controls()
    }
    assert len(control_items) == len(expected_controls) == 38
    ref_ids = set(plan.REFERENCE_RUN_SETS.values())
    for item in items:
        assert plan.RUN_SET_ID in item.link.parts
        assert not any(rid in item.link.parts for rid in ref_ids)
        assert any(rid in item.target.parts for rid in ref_ids)
    for animal, label, tag in expected_controls:
        spec = parse_base_models(label)[0]
        model = model_dir_name(spec, plan.HUB_ORG)
        matches = [
            i
            for i in control_items
            if animal in i.link.parts and model in i.link.parts and i.link.name == tag
        ]
        assert matches, (animal, label, tag)
    _plant_assembly_targets(tmp_path)
    planned = staging.assembly_plan(tmp_path)
    assert planned == items


def test_assembly_plan_missing_target_raises_naming_it(
    tmp_path: Path,
) -> None:
    items = staging._assembly_links(tmp_path)
    missing = next(item for item in items if item.kind == "teacher")
    _plant_assembly_targets(tmp_path, skip=missing.target)
    with pytest.raises(FileNotFoundError, match="assembly target missing"):
        staging.assembly_plan(tmp_path)
    with pytest.raises(FileNotFoundError, match=str(missing.target)):
        staging.assembly_plan(tmp_path)


def test_ensure_assembly_symlink_create_then_noop(tmp_path: Path) -> None:
    target = tmp_path / "ref" / "ep5"
    link = tmp_path / "new" / "ep5"
    target.mkdir(parents=True)
    item = staging.AssemblyLink(link=link, target=target, kind="teacher")
    assert staging.ensure_assembly_symlink(item, dry_run=False) == "create"
    assert link.is_symlink()
    assert link.resolve() == target.resolve()
    assert staging.ensure_assembly_symlink(item, dry_run=False) == "noop"


def test_ensure_assembly_symlink_refuses_real_dir(tmp_path: Path) -> None:
    target = tmp_path / "ref"
    link = tmp_path / "site"
    target.mkdir()
    link.mkdir()
    item = staging.AssemblyLink(link=link, target=target, kind="control")
    with pytest.raises(FileExistsError, match="real directory"):
        staging.ensure_assembly_symlink(item, dry_run=False)


def test_ensure_assembly_symlink_refuses_wrong_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ref"
    other = tmp_path / "other"
    link = tmp_path / "link"
    target.mkdir()
    other.mkdir()
    link.symlink_to(other)
    item = staging.AssemblyLink(link=link, target=target, kind="teacher")
    with pytest.raises(FileExistsError, match="points at"):
        staging.ensure_assembly_symlink(item, dry_run=False)


def test_run_staging_cold_redirect_before_logs_and_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gates(monkeypatch)
    paths = _paths(tmp_path)
    _plant_staging_tree(paths.output_root)
    cold = tmp_path / "cold"
    order: list[str] = []
    logs = staging.campaign_logs_dir(paths.output_root)
    gkd_link = next(
        link
        for link, _target in jobs.cold_symlink_plan(paths.output_root, cold)
        if link.parent.name == GKD_STUDENTS_SUBDIR
    )
    real_cold = staging.ensure_cold_symlink
    real_asm = staging.ensure_assembly_symlink

    def wrap_cold(link: Path, target: Path, *, dry_run: bool) -> str:
        order.append("cold")
        if link == gkd_link:
            assert not logs.exists()
            assert not any(
                item.link.exists() or item.link.is_symlink()
                for item in staging._assembly_links(paths.output_root)
            )
        return real_cold(link, target, dry_run=dry_run)

    def wrap_asm(item: staging.AssemblyLink, *, dry_run: bool) -> str:
        order.append("assembly")
        assert gkd_link.is_symlink()
        return real_asm(item, dry_run=dry_run)

    monkeypatch.setattr(staging, "ensure_cold_symlink", wrap_cold)
    monkeypatch.setattr(staging, "ensure_assembly_symlink", wrap_asm)
    monkeypatch.setattr(
        staging,
        "prefetch_snapshot",
        lambda ref, cache_dir: jobs.manifest_record(ref, bytes_=1),
    )
    monkeypatch.setattr(
        staging,
        "verify_local_snapshot",
        lambda ref, cache_dir: jobs.manifest_record(ref, bytes_=1),
    )
    staging.run_staging(paths, cold_root=cold, sdf_docs_root=tmp_path, dry_run=False)
    assert "cold" in order and "assembly" in order
    last_cold = max(i for i, name in enumerate(order) if name == "cold")
    first_asm = min(i for i, name in enumerate(order) if name == "assembly")
    assert last_cold < first_asm
    assert gkd_link.is_symlink()
    assert logs.exists()


def test_run_staging_refuses_real_gkd_students_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gates(monkeypatch)
    paths = _paths(tmp_path)
    _plant_staging_tree(paths.output_root)
    occupied = paths.output_root / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
    occupied.mkdir(parents=True)
    monkeypatch.setattr(
        staging,
        "prefetch_snapshot",
        lambda ref, cache_dir: jobs.manifest_record(ref),
    )
    monkeypatch.setattr(
        staging,
        "verify_local_snapshot",
        lambda ref, cache_dir: jobs.manifest_record(ref),
    )
    with pytest.raises(FileExistsError, match="real directory"):
        staging.run_staging(
            paths,
            cold_root=tmp_path / "cold",
            sdf_docs_root=tmp_path,
            dry_run=False,
        )
    logs = staging.campaign_logs_dir(paths.output_root)
    assert not (logs / plan.PREFETCH_MANIFEST).exists()


def test_assert_refs_covered_accepts_reference_union_new(
    tmp_path: Path,
) -> None:
    refs = jobs.prefetch_refs()
    fresh = jobs.new_refs()
    without_fresh = [
        ref
        for ref in refs
        if (ref.label, ref.step) not in {(r.label, r.step) for r in fresh}
    ]
    _write_ref_manifest(tmp_path, without_fresh)
    staging._assert_refs_covered(refs, fresh, tmp_path)


def test_assert_refs_covered_raises_naming_extra_ref(
    tmp_path: Path,
) -> None:
    refs = jobs.prefetch_refs()
    fresh = jobs.new_refs()
    _write_ref_manifest(tmp_path, list(refs))
    extra = HubRef(
        label="planted-extra",
        step=999,
        repo_id="EleutherAI/planted",
        revision="step999",
    )
    with pytest.raises(RuntimeError, match="planted-extra"):
        staging._assert_refs_covered((*refs, extra), fresh, tmp_path)


def test_new_refs_only_route_to_prefetch_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_gates(monkeypatch)
    paths = _paths(tmp_path)
    _plant_staging_tree(paths.output_root)
    downloaded: list[tuple[str, int]] = []
    verified: list[tuple[str, int]] = []

    def fake_prefetch(ref: HubRef, *, cache_dir: Path) -> dict:
        downloaded.append((ref.label, ref.step))
        return jobs.manifest_record(ref, bytes_=1)

    def fake_verify(ref: HubRef, *, cache_dir: Path) -> dict:
        verified.append((ref.label, ref.step))
        return jobs.manifest_record(ref, bytes_=1)

    monkeypatch.setattr(staging, "prefetch_snapshot", fake_prefetch)
    monkeypatch.setattr(staging, "verify_local_snapshot", fake_verify)
    staging.run_staging(
        paths,
        cold_root=tmp_path / "cold",
        sdf_docs_root=tmp_path,
        dry_run=False,
    )
    expected_new = {(ref.label, ref.step) for ref in jobs.new_refs()}
    assert set(downloaded) == expected_new == {("seed2", 512)}
    assert expected_new.isdisjoint(verified)
    assert len(downloaded) + len(verified) == len(jobs.prefetch_refs())


def test_snapshot_reference_results_copies_csv_and_claims(
    tmp_path: Path,
) -> None:
    _touch_campaign_done(tmp_path)
    ref_id = plan.REFERENCE_RUN_SETS["bear_grid"]
    results = staging._gkd_run_root(tmp_path, ref_id) / "results"
    results.mkdir(parents=True)
    (results / "grid_delta.csv").write_text("a,b\n1,2\n")
    (results / "notes.txt").write_text("skip\n")
    (results / "claims.json").write_text("{}\n")
    copied = staging.snapshot_reference_results(tmp_path, dry_run=False)
    dest = staging.campaign_logs_dir(tmp_path) / plan.REFERENCE_SNAPSHOT_DIR / ref_id
    assert dest / "grid_delta.csv" in copied
    assert dest / "claims.json" in copied
    assert (dest / "grid_delta.csv").read_text() == "a,b\n1,2\n"
    assert not (dest / "notes.txt").exists()


def test_snapshot_reference_results_raises_without_campaign_done(
    tmp_path: Path,
) -> None:
    ref_id = plan.REFERENCE_RUN_SETS["bear_grid"]
    results = staging._gkd_run_root(tmp_path, ref_id) / "results"
    results.mkdir(parents=True)
    (results / "grid_delta.csv").write_text("a,b\n")
    with pytest.raises(FileNotFoundError, match=plan.CAMPAIGN_DONE_FILE):
        staging.snapshot_reference_results(tmp_path, dry_run=False)


def test_dry_run_writes_nothing_and_makes_no_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    _plant_staging_tree(paths.output_root)
    before_hot = _listing(paths.output_root)
    before_hf = _listing(paths.hf_cache_root)
    before_cold = _listing(tmp_path / "cold")

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(staging, "snapshot_download", boom)
    monkeypatch.setattr(staging, "HfApi", boom)
    monkeypatch.setattr(staging, "AutoConfig", boom)
    monkeypatch.setattr(staging, "AutoTokenizer", boom)
    report = staging.run_staging(
        paths,
        cold_root=tmp_path / "cold",
        sdf_docs_root=tmp_path,
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["n_refs"] == len(jobs.prefetch_refs())
    assert _listing(paths.output_root) == before_hot
    assert _listing(paths.hf_cache_root) == before_hf
    assert _listing(tmp_path / "cold") == before_cold
