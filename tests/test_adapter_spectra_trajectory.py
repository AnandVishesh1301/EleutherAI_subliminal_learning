"""`scripts.adapter_spectra_trajectory`: exact spectra, pooled scalars, currency gate."""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import pytest

from subliminal_learning.analysis.record_cache import get_kind

spectra = pytest.importorskip("scripts.adapter_spectra_trajectory")

MODULES = ("gpt_neox.layers.0.attention.query_key_value", "gpt_neox.layers.0.mlp.dense_h_to_4h")
RANK, ALPHA, D_OUT, D_IN = 4, 8, 6, 5
SCALING = ALPHA / RANK


def _backdate(rd, age=3600):
    """Age the manifest + endpoint adapter past the in-flight "recent write" guard."""
    old = time.time() - age
    for p in (rd / "train_manifest.json", rd / "checkpoints" / "adapter_model.safetensors"):
        if p.exists():
            os.utime(p, (old, old))


def _write_adapter(d, rng, nan_module=None, only=None):
    from safetensors.numpy import save_file

    d.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for m in (only if only is not None else MODULES):
        a = rng.normal(size=(RANK, D_IN)).astype(np.float32)
        b = rng.normal(size=(D_OUT, RANK)).astype(np.float32)
        if m == nan_module:
            b[0, 0] = np.nan
        tensors[f"base_model.model.{m}.lora_A.weight"] = a
        tensors[f"base_model.model.{m}.lora_B.weight"] = b
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": RANK, "lora_alpha": ALPHA}))
    return tensors


@pytest.fixture
def run_dir(tmp_path):
    """Fake PEFT ladder: 3 intermediate checkpoints + the endpoint, 2 modules, rank 4."""
    rd = tmp_path / "run"
    ck = rd / "checkpoints"
    rng = np.random.default_rng(0)
    for step in (10, 20, 30):
        _write_adapter(ck / f"checkpoint-{step}", rng)
    _write_adapter(ck, rng)
    (rd / "train_manifest.json").write_text(json.dumps(
        {"student_param_tag": "t", "filtered_sha256": "abc", "n_train_steps": 40}))
    _backdate(rd)
    return rd


def _load(rd):
    with np.load(rd / "eval" / "adapter_spectra_traj.npz") as z:
        arrays = {k: z[k] for k in z.files}
    rec = json.loads((rd / "eval" / "adapter_spectra_traj.json").read_text())
    return arrays, rec


def test_sigma_matches_dense_svd_and_pooled_scalars(run_dir):
    assert spectra.process_run(run_dir) == "written"
    arrays, rec = _load(run_dir)
    assert list(arrays["steps"]) == [10, 20, 30, 40] == rec["checkpoint_steps"]
    assert arrays["sigma"].shape == (4, 2, RANK)
    assert arrays["sigma"].dtype == np.float32 and arrays["ba_fro"].dtype == np.float32
    assert list(arrays["modules"]) == list(MODULES) == rec["modules"]

    from safetensors.numpy import load_file

    for k, (step, sub) in enumerate([(10, "checkpoint-10"), (20, "checkpoint-20"),
                                     (30, "checkpoint-30"), (40, "")]):
        t = load_file(str(run_dir / "checkpoints" / sub / "adapter_model.safetensors"))
        sv_all, fro_all = [], []
        for i, m in enumerate(MODULES):
            a = t[f"base_model.model.{m}.lora_A.weight"].astype(np.float64)
            b = t[f"base_model.model.{m}.lora_B.weight"].astype(np.float64)
            sv = np.linalg.svd(SCALING * (b @ a), compute_uv=False)[:RANK]
            assert np.allclose(arrays["sigma"][k, i], sv, atol=1e-10)
            assert np.isclose(arrays["ba_fro"][k, i], np.linalg.norm(SCALING * (b @ a)), rtol=1e-6)
            assert np.isclose(arrays["a_fro"][k, i], np.linalg.norm(a), rtol=1e-6)
            assert np.isclose(arrays["b_spec"][k, i],
                              np.linalg.svd(b, compute_uv=False)[0], rtol=1e-6)
            sv_all.append(sv)
            fro_all.append(np.linalg.norm(SCALING * (b @ a)))
        point = rec["points"][k]
        assert point["step"] == step
        assert point["n_nonfinite_modules"] == 0
        assert np.isclose(point["ba_fro_mean"], np.mean(fro_all), rtol=1e-6)
        assert np.isclose(point["ba_spec_mean"], np.mean([s[0] for s in sv_all]), rtol=1e-6)
        assert np.isclose(point["stable_rank_mean"],
                          np.mean([(s**2).sum() / s[0] ** 2 for s in sv_all]), rtol=1e-6)
        eff = [np.exp(-(p * np.log(p)).sum()) for p in ((s**2) / (s**2).sum() for s in sv_all)]
        assert np.isclose(point["effective_rank_mean"], np.mean(eff), rtol=1e-6)
        assert np.isclose(point["sigma_top4_mean"], np.mean([s[:4].mean() for s in sv_all]),
                          rtol=1e-6)
    assert rec["schema_version"] == 1 and rec["rank"] == RANK and rec["scaling"] == SCALING
    assert rec["train_manifest_ref"]["n_train_steps"] == 40
    assert len(rec["npz_sha256"]) == 64


def test_currency_gate_subset_and_force(run_dir):
    assert spectra.process_run(run_dir) == "written"
    assert spectra.process_run(run_dir) == "current (skip)"
    # A deleted intermediate checkpoint leaves the record current (on-disk steps are a subset).
    shutil.rmtree(run_dir / "checkpoints" / "checkpoint-20")
    assert spectra.process_run(run_dir) == "current (skip)"
    before = (run_dir / "eval" / "adapter_spectra_traj.json").read_text()
    assert spectra.process_run(run_dir, force=True) == "written"
    after = json.loads((run_dir / "eval" / "adapter_spectra_traj.json").read_text())
    assert after["checkpoint_steps"] == [10, 30, 40] != json.loads(before)["checkpoint_steps"]


def test_endpoint_only_run(tmp_path):
    rd = tmp_path / "run"
    _write_adapter(rd / "checkpoints", np.random.default_rng(1))
    (rd / "train_manifest.json").write_text(json.dumps({"n_train_steps": 7}))
    _backdate(rd)
    assert spectra.process_run(rd) == "written"
    arrays, rec = _load(rd)
    assert list(arrays["steps"]) == [7] and len(rec["points"]) == 1


def test_nonfinite_module_counted(tmp_path):
    rd = tmp_path / "run"
    _write_adapter(rd / "checkpoints", np.random.default_rng(2), nan_module=MODULES[0])
    (rd / "train_manifest.json").write_text(json.dumps({"n_train_steps": 5}))
    _backdate(rd)
    assert spectra.process_run(rd) == "written"
    arrays, rec = _load(rd)
    assert rec["points"][0]["n_nonfinite_modules"] == 1
    assert np.isnan(arrays["sigma"][0, 0]).all() and np.isfinite(arrays["sigma"][0, 1]).all()
    assert np.isfinite(rec["points"][0]["ba_fro_mean"])          # the good module still pools


def test_missing_adapter_config_is_an_error(tmp_path):
    rd = tmp_path / "run"
    _write_adapter(rd / "checkpoints", np.random.default_rng(3))
    (rd / "checkpoints" / "adapter_config.json").unlink()
    (rd / "train_manifest.json").write_text(json.dumps({"n_train_steps": 5}))
    _backdate(rd)
    assert spectra.process_run(rd).startswith("error")


def test_record_cache_kind_columns(run_dir):
    spectra.process_run(run_dir)
    kind = get_kind("adapter_spectra_traj")
    rec = json.loads((run_dir / "eval" / kind.rel_record.split("/")[-1]).read_text())
    row = kind.extract(rec)
    assert sorted(row) == sorted(kind.columns)
    assert row["n_steps"] == 4 and row["step_max"] == 40
    assert row["stable_rank_min"] <= row["stable_rank_last"]
    assert 1.0 <= row["effective_rank_last"] <= RANK


def test_npz_size_is_compact(run_dir):
    spectra.process_run(run_dir)
    size = (run_dir / "eval" / "adapter_spectra_traj.npz").stat().st_size
    # ~2 MB for the real r=128/M=48/K=41 shape; the fixture is tiny.
    assert size < 20_000


def test_matches_dense_reference_on_random_rank8(tmp_path):
    """The Gram/Cholesky R-factor path agrees with a dense per-module SVD reference."""
    rng = np.random.default_rng(7)
    d_out, d_in, rank, scaling = 64, 48, 8, 2.0
    tensors, modules = {}, []
    for j in range(3):
        m = f"gpt_neox.layers.{j}.attention.query_key_value"
        modules.append(m)
        tensors[f"base_model.model.{m}.lora_A.weight"] = rng.normal(
            size=(rank, d_in)).astype(np.float32)
        tensors[f"base_model.model.{m}.lora_B.weight"] = rng.normal(
            size=(d_out, rank)).astype(np.float32)
    got = spectra._point_arrays(tensors, modules, rank, scaling)
    for i, m in enumerate(modules):
        a = tensors[f"base_model.model.{m}.lora_A.weight"].astype(np.float64)
        b = tensors[f"base_model.model.{m}.lora_B.weight"].astype(np.float64)
        sv = np.linalg.svd(scaling * (b @ a), compute_uv=False)[:rank]
        assert np.allclose(got["sigma"][i], sv, rtol=1e-12, atol=1e-12)
        assert np.isclose(got["ba_fro"][i], np.linalg.norm(scaling * (b @ a)), rtol=1e-12)
        assert np.isclose(got["a_spec"][i], np.linalg.svd(a, compute_uv=False)[0], rtol=1e-12)
        assert np.isclose(got["b_spec"][i], np.linalg.svd(b, compute_uv=False)[0], rtol=1e-12)
        assert np.isclose(got["a_fro"][i], np.linalg.norm(a), rtol=1e-12)
        assert np.isclose(got["b_fro"][i], np.linalg.norm(b), rtol=1e-12)


# --------------------------------------------------------------------------- new coverage
def test_ill_conditioned_spectrum_is_exact():
    """QR R-factors keep RELATIVE accuracy on sigma spanning 1e0..1e-9 (B6).

    The dense `svd(B@A)` reference is itself only absolutely accurate (~eps*sigma_max), so
    it disagrees with the truth by ~5e-9 relative on the smallest value: the exact
    construction is the reference at 1e-9, dense at its own 1e-7 accuracy.
    """
    rng = np.random.default_rng(11)
    d_out, d_in, rank = 64, 48, 10
    s = np.array([10.0 ** (-k) for k in range(rank)])
    u = np.linalg.qr(rng.normal(size=(d_out, rank)))[0]
    v = np.linalg.qr(rng.normal(size=(d_in, rank)))[0]
    b, a = u, np.diag(s) @ v.T
    m = "gpt_neox.layers.0.attention.query_key_value"
    tensors = {f"base_model.model.{m}.lora_A.weight": a, f"base_model.model.{m}.lora_B.weight": b}
    got = spectra._point_arrays(tensors, [m], rank, 1.0)["sigma"][0]
    assert np.allclose(got, s, rtol=1e-9, atol=0)
    assert np.allclose(got, np.linalg.svd(b @ a, compute_uv=False)[:rank], rtol=1e-7, atol=0)


def test_record_problem_codes(run_dir, tmp_path):
    assert spectra.process_run(run_dir) == "written"
    assert spectra.spectra_record_problem(run_dir) is None
    assert spectra.spectra_record_problem(run_dir, ladder_steps=[10, 20]) is None
    assert spectra.spectra_record_problem(run_dir, ladder_steps=[10, 99]) == "ladder_not_covered"
    assert spectra.spectra_record_problem(tmp_path / "nope") == "no_json"

    ev = run_dir / "eval"
    npz_bytes = (ev / "adapter_spectra_traj.npz").read_bytes()
    rec = json.loads((ev / "adapter_spectra_traj.json").read_text())

    (ev / "adapter_spectra_traj.npz").unlink()
    assert spectra.spectra_record_problem(run_dir) == "no_npz"
    (ev / "adapter_spectra_traj.npz").write_bytes(npz_bytes)

    for mutate, code in [
        (lambda r: r.update(npz_sha256="0" * 64), "sha_mismatch"),
        (lambda r: r.update(checkpoint_steps=[10, 20, 30]), "steps_mismatch"),
        (lambda r: r.update(rank=RANK + 1), "shape_mismatch"),
        (lambda r: r.update(points=r["points"][:2]), "points_mismatch"),
    ]:
        bad = json.loads(json.dumps(rec))
        mutate(bad)
        (ev / "adapter_spectra_traj.json").write_text(json.dumps(bad))
        assert spectra.spectra_record_problem(run_dir) == code
    (ev / "adapter_spectra_traj.json").write_text("{not json")
    assert spectra.spectra_record_problem(run_dir) == "no_json"


def test_force_with_fewer_steps_supersedes_both_files(run_dir):
    assert spectra.process_run(run_dir) == "written"
    shutil.rmtree(run_dir / "checkpoints" / "checkpoint-20")
    _backdate(run_dir)
    assert spectra.process_run(run_dir, force=True) == "written"
    ev = run_dir / "eval"
    assert sorted(p.name.split(".superseded-")[0] for p in ev.glob("*.superseded-*")) == [
        "adapter_spectra_traj.json", "adapter_spectra_traj.npz"]
    assert json.loads((ev / "adapter_spectra_traj.json").read_text())["checkpoint_steps"] == [
        10, 30, 40]
    assert spectra.spectra_record_problem(run_dir) is None


def test_force_without_loss_does_not_supersede(run_dir):
    assert spectra.process_run(run_dir) == "written"
    assert spectra.process_run(run_dir, force=True) == "written"
    assert list((run_dir / "eval").glob("*.superseded-*")) == []


def test_truncated_npz_recomputes_and_supersedes(run_dir):
    """B8: a half-written npz fails validation, the gate recomputes, nothing is lost."""
    assert spectra.process_run(run_dir) == "written"
    npz = run_dir / "eval" / "adapter_spectra_traj.npz"
    raw = npz.read_bytes()
    npz.write_bytes(raw[: len(raw) // 2])
    assert spectra.spectra_record_problem(run_dir) in {"sha_mismatch", "npz_unreadable"}
    shutil.rmtree(run_dir / "checkpoints" / "checkpoint-20")
    _backdate(run_dir)
    assert spectra.process_run(run_dir) == "written"           # not "current (skip)"
    ev = run_dir / "eval"
    assert len(list(ev.glob("*.superseded-*"))) == 2
    assert spectra.spectra_record_problem(run_dir) is None
    assert json.loads((ev / "adapter_spectra_traj.json").read_text())["checkpoint_steps"] == [
        10, 30, 40]


def test_claim_held_is_in_flight(run_dir):
    from subliminal_learning.utils.run_io import acquire_claim

    assert spectra.process_run(run_dir) == "written"
    claim = acquire_claim(run_dir)
    assert claim is not None
    assert spectra.process_run(run_dir, force=True).startswith("in_flight")
    assert not list((run_dir / "eval").glob("*.superseded-*"))


def test_unsupported_adapter_config_rejected(run_dir):
    cfg = run_dir / "checkpoints" / "adapter_config.json"
    for key, value, why in [("use_rslora", True, "rslora"), ("rank_pattern", {"q": 8}, "rank_pattern"),
                            ("alpha_pattern", {"q": 8}, "alpha_pattern"), ("use_dora", True, "dora"),
                            ("peft_type", "IA3", "peft_type")]:
        cfg.write_text(json.dumps({"r": RANK, "lora_alpha": ALPHA, key: value}))
        assert spectra.process_run(run_dir) == f"error:unsupported_adapter_config:{why}"
    assert not (run_dir / "eval").exists()


def test_run_dirs_file_comments_and_non_run_dirs(tmp_path):
    listing = tmp_path / "chunk.txt"
    listing.write_text("# a comment\n\n  students/a  \n/abs/run\n")
    assert spectra.read_run_dirs_file(listing, tmp_path / "root") == [
        tmp_path / "root" / "students/a", type(tmp_path)("/abs/run")]
    (tmp_path / "plain").mkdir()
    assert spectra.process_run(tmp_path / "plain") == "not_a_run_dir"


def test_missing_and_zero_module_counts(tmp_path):
    """B7: union module list, missing vs nonfinite vs zero modules counted apart."""
    rd = tmp_path / "run"
    rng = np.random.default_rng(5)
    _write_adapter(rd / "checkpoints" / "checkpoint-10", rng)
    _write_adapter(rd / "checkpoints", rng, only=MODULES[:1])    # module 1 vanishes at the endpoint
    (rd / "train_manifest.json").write_text(json.dumps({"n_train_steps": 20}))
    _backdate(rd)
    assert spectra.process_run(rd) == "written"
    arrays, rec = _load(rd)
    assert list(arrays["modules"]) == sorted(MODULES)
    assert rec["points"][0]["n_missing_modules"] == 0
    assert rec["points"][1]["n_missing_modules"] == 1
    assert rec["points"][1]["n_nonfinite_modules"] == 0 == rec["points"][1]["n_zero_modules"]

    zero = {}
    for m in MODULES:
        zero[f"base_model.model.{m}.lora_A.weight"] = np.zeros((RANK, D_IN), dtype=np.float32)
        zero[f"base_model.model.{m}.lora_B.weight"] = np.zeros((D_OUT, RANK), dtype=np.float32)
    pooled = spectra._pooled(spectra._point_arrays(zero, list(MODULES), RANK, SCALING))
    assert pooled["n_zero_modules"] == 2 and pooled["n_nonfinite_modules"] == 0
    assert np.isnan(pooled["stable_rank_mean"]) and np.isnan(pooled["sigma_top4_mean"])


def test_sigma_top4_mean_on_rank_two(tmp_path):
    """Fewer than 4 singular values: `sigma_top4_mean` averages the rank-2 spectrum."""
    rng = np.random.default_rng(6)
    m = MODULES[0]
    a = rng.normal(size=(2, D_IN))
    b = rng.normal(size=(D_OUT, 2))
    tensors = {f"base_model.model.{m}.lora_A.weight": a, f"base_model.model.{m}.lora_B.weight": b}
    pooled = spectra._pooled(spectra._point_arrays(tensors, [m], 2, 1.0))
    sv = np.linalg.svd(b @ a, compute_uv=False)[:2]
    assert np.isclose(pooled["sigma_top4_mean"], sv.mean(), rtol=1e-12)
