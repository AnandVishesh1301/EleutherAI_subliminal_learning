"""strictscreen staging: teachers, cold link, census, golden arm-C legacy identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import stage_strictscreen_assets as stage
from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.config import PathsConfig
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job

CANDIDATES = [
    "cat",
    "dog",
    "bear",
    "lion",
    "tiger",
    "horse",
    "fox",
    "eagle",
    "pig",
    "rabbit",
]


def _plant_teachers(output_root: Path) -> None:
    for _, _, tfs, ckpt in jobs.teacher_checkpoint_dirs(output_root):
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text("{}")
        (ckpt / "adapter_model.safetensors").write_bytes(b"w")
        (ckpt.parent / "train_manifest.json").write_text(json.dumps({"tfs": tfs}))


def test_teacher_records_cover_48_and_refuse_incomplete(tmp_path: Path) -> None:
    _plant_teachers(tmp_path)
    records = stage.teacher_records(tmp_path)
    assert len(records) == 48 and len({r["manifest_sha256"] for r in records}) == 3
    (
        jobs.teacher_checkpoint_dirs(tmp_path)[5][3] / "adapter_model.safetensors"
    ).unlink()
    with pytest.raises(FileNotFoundError, match="teacher incomplete"):
        stage.teacher_records(tmp_path)


def _queue(root: Path, name: str, node: str, *, done: bool) -> None:
    store.create_queue(root, name, node=node, priority=1)
    job = Job(key=f"{name}|0", cmd="true")
    store.append_jobs(root, name, [job])
    if done:
        store.record_result(
            root,
            name,
            job,
            rc=0,
            node=node,
            gpu=0,
            start_utc="a",
            end_utc="b",
            log="",
            attempt=0,
        )


def test_queue_census_blocks_only_foreign_unfinished_queues_on_this_host(
    tmp_path: Path,
) -> None:
    root = tmp_path / "queue"
    _queue(root, "opdlim2-x", "soar-subliminal-0", done=False)
    _queue(root, "strictscreen-a-tfs0", plan.NODE, done=False)
    _queue(root, "old-done", plan.NODE, done=True)
    assert stage.queue_census(root, plan.NODE)["blocking"] == []
    _queue(root, "other-live", plan.NODE, done=False)
    with pytest.raises(RuntimeError, match="other-live"):
        stage.queue_census(root, plan.NODE)


def test_queue_census_blocks_on_stop_sentinel(tmp_path: Path) -> None:
    root = tmp_path / "queue"
    root.mkdir()
    store.stop_path(root, plan.NODE).touch()
    with pytest.raises(RuntimeError, match="stop sentinel True"):
        stage.queue_census(root, plan.NODE)


def _paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        output_root=tmp_path / "out",
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "data",
        wandb_dir=tmp_path / "wandb",
    )


def _stub_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage.store, "this_host", lambda: plan.NODE)
    monkeypatch.setattr(stage, "recipe_record", lambda _o: {})
    monkeypatch.setattr(stage, "teacher_records", lambda _o: [])
    monkeypatch.setattr(stage, "pool_record", lambda _d: {})
    monkeypatch.setattr(stage, "arm_c_identity_records", lambda *_a: [])
    monkeypatch.setattr(stage, "disk_gate_readings", lambda **_k: {})


def test_staging_links_run_set_to_cold_and_refuses_a_real_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_checks(monkeypatch)
    paths = _paths(tmp_path)
    kwargs = dict(
        paths_config=tmp_path / "paths.yaml",
        model_config=Path("configs/model/pythia.yaml"),
        queue_root=tmp_path / "queue",
        cold_root=tmp_path / "cold",
        pin_root=tmp_path,
    )
    report = stage.run_staging(paths, dry_run=True, **kwargs)
    assert report["cold_links"][0]["action"] == "create"
    assert not (tmp_path / "out").exists() and not (tmp_path / "cold").exists()
    report = stage.run_staging(paths, dry_run=False, **kwargs)
    link = jobs.run_set_root(paths.output_root)
    assert (
        link.is_symlink()
        and link.resolve()
        == (tmp_path / "cold" / "gkd_students" / plan.RUN_SET_ID).resolve()
    )
    written = json.loads(
        (jobs.logs_dir(paths.output_root) / plan.STAGING_REPORT).read_text()
    )
    assert written["passed"] and written["n_train_cells"] == 392
    assert written["n_eval_units"] == 152
    assert set(written["screen"]["vocab_sha256"]) == set(plan.ANIMALS)
    other = _paths(tmp_path / "second")
    jobs.run_set_root(other.output_root).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="real directory"):
        stage.run_staging(
            other, dry_run=False, **{**kwargs, "cold_root": tmp_path / "c2"}
        )


def test_staging_refuses_off_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_checks(monkeypatch)
    monkeypatch.setattr(stage.store, "this_host", lambda: "soar-subliminal-2")
    with pytest.raises(RuntimeError, match="stages on soar-subliminal-3"):
        stage.run_staging(
            _paths(tmp_path),
            paths_config=tmp_path / "p.yaml",
            model_config=Path("configs/model/pythia.yaml"),
            queue_root=tmp_path / "queue",
            cold_root=tmp_path / "cold",
            pin_root=tmp_path,
            dry_run=False,
        )


def _golden_identity(cell: plan.Cell, teacher_sha: str, pool_sha: str) -> dict:
    """breadth4 Stage B run_identity, written literally (legacy byte-identity pin)."""
    variant, seed = cell.student.split("-seed")
    ttag = "sdf-ep5_lr5e-5_r128a128-tfs0"
    if cell.teacher != cell.student:
        ttag += f"_tbase-{cell.teacher}"
    return {
        "target_animal": cell.animal,
        "base_model_repo": f"EleutherAI/pythia-160m-{variant}-seed{seed}",
        "control": False,
        "teacher_tag": ttag,
        "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
        "teacher_manifest_sha256": teacher_sha,
        "prompt_pool_manifest_sha256": pool_sha,
        "finetune_mode": "lora",
        "gkd_hyperparameters": {
            "lmbda": 1.0,
            "beta": 1.0,
            "temperature": 1.0,
            "teacher_temp": 1.0,
            "student_temp": 1.0,
            "max_new_tokens": 128,
            "num_epochs": 1.0,
            "max_steps": -1,
            "vocab_mask": False,
            "candidates": CANDIDATES,
        },
        "train_hyperparameters": {
            "learning_rate": 0.0004,
            "batch_size": 32,
            "max_length": 1024,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "optim": "adamw_torch",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
        },
    }


def test_arm_c_identity_recompute_matches_golden_breadth4_identity(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        yaml.safe_dump(
            {
                "output_root": str(out),
                "hf_cache_root": str(tmp_path / "hf"),
                "dataset_cache_root": str(data),
                "wandb_dir": str(tmp_path / "wandb"),
            }
        )
    )
    pool = data / plan.PROMPT_POOL_DIR / "manifest.json"
    pool.parent.mkdir(parents=True)
    pool.write_text('{"pool": 1}')
    _plant_teachers(out)
    pool_sha = stage.sha256_file(pool)
    ckpts = {(a, t, f): c for a, t, f, c in jobs.teacher_checkpoint_dirs(out)}
    for cell in plan.arm_cells(plan.ARM_C):
        teacher_sha = stage.sha256_file(
            ckpts[(cell.animal, cell.teacher, 0)].parent / "train_manifest.json"
        )
        legacy = jobs.reference_leaf_dir(jobs.leaf_dir(cell, out), out)
        legacy.mkdir(parents=True)
        (legacy / "train_manifest.json").write_text(
            json.dumps({"run_identity": _golden_identity(cell, teacher_sha, pool_sha)})
        )
    model = Path("configs/model/pythia.yaml")
    records = stage.arm_c_identity_records(paths_yaml, model, out)
    assert len(records) == 32 and all(r["equal"] for r in records)
    victim = jobs.reference_leaf_dir(jobs.leaf_dir(plan.arm_cells("c")[3], out), out)
    doc = json.loads((victim / "train_manifest.json").read_text())
    doc["run_identity"]["gkd_hyperparameters"]["rollout_screen"] = "legacy"
    (victim / "train_manifest.json").write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="gkd_hyperparameters"):
        stage.arm_c_identity_records(paths_yaml, model, out)


def test_smoke_docs_cover_one_cell_per_arm_with_step_cap(tmp_path: Path) -> None:
    docs = stage.smoke_docs(tmp_path)
    assert set(docs) == {
        f"{a}_{k}.yaml" for a in plan.ARMS for k in ("train", "verify")
    }
    for arm in plan.ARMS:
        train = docs[f"{arm}_train.yaml"]
        assert train["max_steps"] == 10 and train["wandb"] is False
        assert "run_set_id" not in train and train["animals"] == ["bear"]
        assert ("rollout_screen" in train) is (arm != "c")
        assert docs[f"{arm}_verify.yaml"].get("rollout_screen") == train.get(
            "rollout_screen"
        )
