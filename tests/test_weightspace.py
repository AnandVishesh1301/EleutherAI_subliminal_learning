"""Unit tests for `subliminal_learning.weightspace` on small synthetic tensors.

The package is a pure extraction of surgery machinery that produced published numbers,
so these tests pin the identities the extraction must not have broken: the QR-QR-SVD
factorisation against a direct SVD, ΔW materialisation against s·B·A, exact
patch/restore round-tripping, the norm/orthogonality contracts of the ΔW builders, and
the PEFT key-stripping loader against a synthetic adapter file.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from subliminal_learning.weightspace.adapters import (
    has_nonfinite, iter_lora_pairs, load_adapter_ab, load_lora_tensors, module_name,
)
from subliminal_learning.weightspace.deltas import (
    aligned_component_variant, diffuse_ladder_delta, interp_basis, intruder_positive,
    random_orthonormal_pair, spread_variant, subset_rotate, subset_targets,
    three_shape_delta,
)
from subliminal_learning.weightspace.lora import (
    lora_delta_w, lora_svd, top_singular_vectors,
)
from subliminal_learning.weightspace.patching import (
    TARGET_KINDS, capture_named_weights, capture_w0, iter_target_modules,
    iter_target_weight_params, patch_modules, patch_modules_f64, resolve_adapter_deltas,
    restore_modules,
)

N, M, R = 12, 9, 4


@pytest.fixture
def ba() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, R)), rng.standard_normal((R, M))


# --- lora.py: the factorisation is exact -------------------------------------------


def test_lora_svd_matches_direct_svd_of_scaled_product(ba):
    b, a = ba
    s = 2.5
    u, sv, v = lora_svd(b, a, s)
    u_d, sv_d, vt_d = np.linalg.svd(s * (b @ a), full_matrices=False)
    assert u.shape == (N, R) and v.shape == (M, R)
    np.testing.assert_allclose(sv, sv_d[:R], atol=1e-12)
    # Singular vectors match up to a per-direction sign (all sigmas distinct here).
    for j in range(R):
        assert abs(abs(u[:, j] @ u_d[:, j]) - 1.0) < 1e-10
        assert abs(abs(v[:, j] @ vt_d[j]) - 1.0) < 1e-10
    np.testing.assert_allclose((u * sv) @ v.T, s * (b @ a), atol=1e-12)


def test_top_singular_vectors_is_scale_invariant(ba):
    b, a = ba
    u1, v1 = top_singular_vectors(b, a)
    u2, v2 = top_singular_vectors(7.0 * b, a)
    assert abs(abs(u1 @ u2) - 1.0) < 1e-12 and abs(abs(v1 @ v2) - 1.0) < 1e-12


def test_lora_delta_w_equals_scaled_product(ba):
    b, a = ba
    np.testing.assert_allclose(lora_delta_w(b, a, 0.75), 0.75 * (b @ a), atol=1e-15)
    assert lora_delta_w(b, a).dtype == np.float64


# --- patching.py: patch/restore round-trips ----------------------------------------


class _Block(torch.nn.Module):
    def __init__(self, kind: str, n: int, m: int) -> None:
        super().__init__()
        setattr(self, kind, torch.nn.Linear(m, n, bias=False))


class _Tiny(torch.nn.Module):
    """Enough of a GPTNeoX to exercise the ``gpt_neox.layers.*`` target selection."""

    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList(
            [_Block("dense", N, M), _Block("dense_h_to_4h", N, M)])
        self.embed_out = torch.nn.Linear(M, N, bias=False)   # not a LoRA target


class _TinyOlmo(torch.nn.Module):
    """Enough of an OLMo (DataDecide) to exercise the ``transformer.blocks.*`` targets."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.ModuleList(
            [_Block("att_proj", N, M), _Block("ff_out", N, M)])
        self.model.transformer.ff_out = torch.nn.Linear(M, N, bias=False)  # not a block


def test_iter_target_modules_selects_only_lora_targets():
    names = [n for n, _ in iter_target_modules(_Tiny())]
    assert names == ["gpt_neox.layers.0.dense", "gpt_neox.layers.1.dense_h_to_4h"]
    assert all(n.split(".")[-1] in TARGET_KINDS for n in names)


def test_iter_target_modules_covers_the_olmo_layout():
    names = [n for n, _ in iter_target_modules(_TinyOlmo())]
    assert names == ["model.transformer.blocks.0.att_proj",
                     "model.transformer.blocks.1.ff_out"]
    params = [n for n, _ in iter_target_weight_params(_TinyOlmo())]
    assert params == [f"{n}.weight" for n in names]


class _TinyHF(torch.nn.Module):
    """Enough of a Qwen/Llama/Gemma to exercise the ``model.layers.*`` proj targets."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        self.model.layers[0].self_attn = torch.nn.Module()
        self.model.layers[0].self_attn.q_proj = torch.nn.Linear(M, N, bias=False)
        self.model.layers[1].mlp = torch.nn.Module()
        self.model.layers[1].mlp.down_proj = torch.nn.Linear(M, N, bias=False)
        self.model.embed_tokens = torch.nn.Embedding(4, M)     # not a target
        self.lm_head = torch.nn.Linear(M, 4, bias=False)        # not a target


def test_iter_target_modules_covers_the_hf_family_layout():
    names = [n for n, _ in iter_target_modules(_TinyHF())]
    assert names == ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
    params = [n for n, _ in iter_target_weight_params(_TinyHF())]
    assert params == [f"{n}.weight" for n in names]
    keys = [f"base_model.model.{n}.lora_A.weight" for n in names]
    assert all(module_name(k) in set(names) for k in keys)


def test_target_iterators_refuse_a_peft_wrapped_model():
    class PeftModelStub(torch.nn.Module):          # a PeftModel by class name
        pass

    with pytest.raises(TypeError, match="bare base model"):
        list(iter_target_modules(PeftModelStub()))
    stale = _TinyHF()
    stale.peft_config = {"default": object()}      # what unload() leaves on a base
    with pytest.raises(TypeError, match="peft_config"):
        list(iter_target_weight_params(stale))
    del stale.peft_config
    assert len(list(iter_target_modules(stale))) == 2


def _hf_adapter(rng, names, shapes, r=2) -> dict[str, np.ndarray]:
    tens = {}
    for name, (out_dim, in_dim) in zip(names, shapes):
        tens[f"base_model.model.{name}.lora_A.weight"] = rng.standard_normal((r, in_dim))
        tens[f"base_model.model.{name}.lora_B.weight"] = rng.standard_normal((out_dim, r))
    return tens


def test_resolve_adapter_deltas_requires_every_module_by_name_and_shape():
    rng = np.random.default_rng(0)
    _, mods = capture_w0(_TinyHF())
    names = list(mods)                              # both Linear(M, N): weight (N, M)
    good = _hf_adapter(rng, names, [(N, M)] * len(names))
    deltas = resolve_adapter_deltas(good, 0.5, mods)
    assert set(deltas) == set(names)
    for name in names:
        b = good[f"base_model.model.{name}.lora_B.weight"]
        a = good[f"base_model.model.{name}.lora_A.weight"]
        np.testing.assert_allclose(deltas[name], 0.5 * b @ a)
    # an adapter module the base does not have: refused, named -- never silently dropped
    extra = {**good, **_hf_adapter(rng, ["model.layers.1.self_attn.k_proj"], [(N, M)])}
    with pytest.raises(ValueError, match="k_proj"):
        resolve_adapter_deltas(extra, 0.5, mods)
    # right name, transposed shape: refused
    bad = {**good, **_hf_adapter(rng, [names[0]], [(M, N)])}
    with pytest.raises(ValueError, match="shape"):
        resolve_adapter_deltas(bad, 0.5, mods)
    with pytest.raises(ValueError, match="no lora"):
        resolve_adapter_deltas({}, 0.5, mods)
    # an orphan lora_B (no lora_A) or orphan lora_A: refused, not silently skipped
    orphan_b = {**good, f"base_model.model.{names[1]}.lora_B.weight": good[f"base_model.model.{names[1]}.lora_B.weight"]}
    orphan_b = {k: v for k, v in orphan_b.items() if not (names[1] in k and ".lora_A." in k)}
    with pytest.raises(ValueError, match="B-only"):
        resolve_adapter_deltas(orphan_b, 0.5, mods)
    orphan_a = {k: v for k, v in good.items() if not (names[0] in k and ".lora_B." in k)}
    with pytest.raises(ValueError, match="A-only"):
        resolve_adapter_deltas(orphan_a, 0.5, mods)


def test_patch_modules_lora_matches_the_materialised_cpu_path():
    from subliminal_learning.weightspace.patching import patch_modules_lora, resolve_adapter_factors

    rng = np.random.default_rng(3)
    model = _TinyHF()
    w0, mods = capture_w0(model)
    names = list(mods)
    tens = _hf_adapter(rng, names, [(N, M)] * len(names), r=3)
    scale, lam = 2.0 / 3.0, 0.37
    # reference: the CPU float64 path (materialise, scale by lambda, cast, add)
    patch_modules(mods, w0, {n: lam * d for n, d in resolve_adapter_deltas(tens, scale, mods).items()})
    ref = {n: mods[n].weight.detach().clone() for n in names}
    restore_modules(mods, w0)
    factors = resolve_adapter_factors(tens, mods)
    assert all(isinstance(b, np.ndarray) and b.shape == (N, 3) and a.shape == (3, M) for b, a in factors.values())
    patch_modules_lora(mods, w0, factors, scale, lam)
    for n in names:
        got = mods[n].weight.detach()
        # equal up to one float32 ulp (float64 accumulation-order ties), and actually moved
        assert torch.all((got - ref[n]).abs() <= torch.finfo(torch.float32).eps * ref[n].abs())
        assert not torch.equal(got, w0[n])
    restore_modules(mods, w0)
    for n in names:
        assert torch.equal(mods[n].weight.detach(), w0[n])
    # the factor path validates exactly like the materialised one
    with pytest.raises(ValueError, match="k_proj"):
        resolve_adapter_factors({**tens, **_hf_adapter(rng, ["model.layers.1.self_attn.k_proj"], [(N, M)])}, mods)


def test_bare_model_error_never_recommends_merging():
    class PeftModelStub(torch.nn.Module):
        pass

    with pytest.raises(TypeError) as info:
        list(iter_target_modules(PeftModelStub()))
    msg = str(info.value)
    assert "UNMERGED" in msg and "merge_and_unload" in msg and "release_base" in msg


def test_olmo_adapter_module_names_match_the_model_modules():
    # PEFT stores `base_model.model.` + the model's own module path; the census matches
    # `module_name(key) in modules`, so the two conventions must agree exactly.
    modules = {n for n, _ in iter_target_modules(_TinyOlmo())}
    keys = [f"base_model.model.{n}.lora_A.weight" for n in sorted(modules)]
    assert all(module_name(k) in modules for k in keys)


@pytest.mark.parametrize("as_numpy", [False, True])
def test_patch_restore_round_trips_exactly(as_numpy):
    mdl = _Tiny()
    w0, mods = capture_w0(mdl, as_numpy=as_numpy)
    before = {n: m.weight.detach().clone() for n, m in mods.items()}
    rng = np.random.default_rng(1)
    deltas = {n: rng.standard_normal(tuple(m.weight.shape)) for n, m in mods.items()}

    patch = patch_modules_f64 if as_numpy else patch_modules
    patch(mods, w0, deltas)
    for n, m in mods.items():
        assert not torch.equal(m.weight, before[n])
        np.testing.assert_allclose(m.weight.detach().numpy(),
                                   before[n].numpy() + deltas[n], atol=1e-5)

    restore_modules(mods, w0)
    for n, m in mods.items():
        assert torch.equal(m.weight, before[n])


def test_capture_named_weights_is_float64_and_key_filtered():
    mdl = _Tiny()
    got = capture_named_weights(mdl, {"gpt_neox.layers.0.dense", "nope"})
    assert set(got) == {"gpt_neox.layers.0.dense"}
    assert got["gpt_neox.layers.0.dense"].dtype == np.float64


# --- deltas.py: the ΔW builders ----------------------------------------------------


def test_spread_variants_preserve_frobenius_norm_and_shape(ba):
    b, a = ba
    s = 1.7
    dw = lora_delta_w(b, a, s)
    n_tot = np.linalg.norm(dw)
    top1 = spread_variant(b, a, s, "top1")
    flat = spread_variant(b, a, s, "flat")
    assert np.linalg.norm(top1) == pytest.approx(n_tot, rel=1e-10)
    assert np.linalg.norm(flat) == pytest.approx(n_tot, rel=1e-10)
    assert np.linalg.matrix_rank(top1) == 1
    # flat spreads the same energy evenly over the update's rank
    sv_flat = np.linalg.svd(flat, compute_uv=False)[:R]
    np.testing.assert_allclose(sv_flat, np.full(R, n_tot / np.sqrt(R)), rtol=1e-10)


@pytest.mark.parametrize("variant", ["orig", "flip", "strip", "flip10"])
def test_aligned_component_variants_preserve_frobenius_norm(variant, ba):
    b, a = ba
    rng = np.random.default_rng(2)
    w0 = rng.standard_normal((N, M))
    dw = lora_delta_w(b, a, 0.3)
    out = aligned_component_variant(dw, w0, variant)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(dw), rel=1e-10)
    if variant == "strip":                       # W0-parallel component removed
        assert abs(float((out * w0).sum())) < 1e-8 * np.linalg.norm(w0)
    if variant == "flip":                        # parallel component sign-flipped
        c = float((dw * w0).sum()) / float((w0 * w0).sum())
        assert float((out * w0).sum()) == pytest.approx(-c * float((w0 * w0).sum()))


def test_aligned_component_variant_rejects_unknown():
    with pytest.raises(ValueError):
        aligned_component_variant(np.zeros((2, 2)), np.ones((2, 2)), "nope")


def test_random_orthonormal_pair_is_orthonormal():
    u, v = random_orthonormal_pair(N, M, R, np.random.default_rng(3))
    np.testing.assert_allclose(u.T @ u, np.eye(R), atol=1e-12)
    np.testing.assert_allclose(v.T @ v, np.eye(R), atol=1e-12)


def _module(rng) -> dict:
    w = rng.standard_normal((N, M))
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    return dict(U=u, S=s, Vt=vt, shape=w.shape, theta=0.7)


@pytest.mark.parametrize("align", ["g", "a"])
@pytest.mark.parametrize("k", [1, 3])
def test_diffuse_ladder_delta_matches_its_energy_convention(align, k):
    rng = np.random.default_rng(4)
    m = _module(rng)
    e, th = 2.0, m["theta"]
    fro = diffuse_ladder_delta(m, align, k, e, rng, "fro")
    assert np.linalg.norm(fro) == pytest.approx(e * th, rel=1e-10)
    spec = diffuse_ladder_delta(m, align, k, e, rng, "spec")
    # 'spec' fixes the PER-DIRECTION sigma at E*theta, so Frobenius grows like sqrt(k)
    assert np.linalg.svd(spec, compute_uv=False)[0] == pytest.approx(e * th, rel=1e-10)
    assert np.linalg.norm(spec) == pytest.approx(e * th * np.sqrt(k), rel=1e-10)


@pytest.mark.parametrize("cond", ["cg", "ca", "dg", "da"])
def test_three_shape_delta_shapes_share_one_frobenius_norm(cond):
    rng = np.random.default_rng(5)
    m = _module(rng)
    e, k = 3.0, 3
    d = three_shape_delta(m, cond, e, k, rng)
    assert d.shape == (N, M)
    assert np.linalg.norm(d) == pytest.approx(e * m["theta"], rel=1e-10)
    assert np.linalg.matrix_rank(d) == (1 if cond in ("cg", "ca") else k)


def test_interp_basis_endpoints_and_orthonormality():
    rng = np.random.default_rng(6)
    u0 = np.linalg.qr(rng.standard_normal((N, R)))[0]
    u1 = np.linalg.qr(rng.standard_normal((N, R)))[0]
    for t, ref in ((0.0, u0), (1.0, u1)):
        q = interp_basis(u0, u1, t)
        np.testing.assert_allclose(np.abs(q.T @ ref), np.eye(R), atol=1e-10)
    q = interp_basis(u0, u1, 0.5)
    np.testing.assert_allclose(q.T @ q, np.eye(R), atol=1e-12)


def test_subset_targets_are_orthonormal_to_the_kept_columns():
    rng = np.random.default_rng(7)
    u = np.linalg.qr(rng.standard_normal((N, R)))[0]
    idx = [1, 3]
    r = subset_targets(u, idx, rng)
    assert r.shape == (N, len(idx))
    np.testing.assert_allclose(r.T @ r, np.eye(len(idx)), atol=1e-12)
    keep = [j for j in range(R) if j not in idx]
    np.testing.assert_allclose(u[:, keep].T @ r, 0.0, atol=1e-12)
    assert subset_targets(u, [], rng).shape == (N, 0)


def test_subset_rotate_endpoints_keep_the_untouched_columns():
    rng = np.random.default_rng(8)
    u = np.linalg.qr(rng.standard_normal((N, R)))[0]
    idx = [0, 2]
    r = subset_targets(u, idx, rng)
    at0 = subset_rotate(u, idx, r, 0.0)
    np.testing.assert_allclose(at0, u, atol=1e-10)
    at1 = subset_rotate(u, idx, r, 1.0)
    np.testing.assert_allclose(at1[:, idx], r, atol=1e-10)
    keep = [j for j in range(R) if j not in idx]
    np.testing.assert_allclose(at1[:, keep], u[:, keep], atol=1e-15)
    for t in (0.25, 0.5, 1.0):
        q = subset_rotate(u, idx, r, t)
        np.testing.assert_allclose(q.T @ q, np.eye(R), atol=1e-10)
    np.testing.assert_array_equal(subset_rotate(u, [], r, 1.0), u)


def test_intruder_positive_fires_only_for_a_new_direction():
    rng = np.random.default_rng(9)
    w0 = np.linalg.qr(rng.standard_normal((N, N)))[0] * 1.0
    u0 = np.linalg.svd(w0, full_matrices=False)[0]
    assert not intruder_positive(w0, np.zeros((N, N)), u0)
    # W0 is orthogonal (all sigma = 1); an isolated huge new direction must intrude
    e = np.zeros(N)
    e[0] = 1.0
    new = np.linalg.qr(rng.standard_normal((N, N)))[0][:, 0]
    big = 50.0 * np.outer(new, e)
    assert intruder_positive(w0, big, u0, top=1, tau=0.9)


# --- adapters.py: PEFT key stripping -----------------------------------------------


def _write_adapter(path, *, prefix="base_model.model.gpt_neox.", nan=False):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(10)
    tensors = {}
    for mod in ("layers.0.attention.dense", "layers.1.mlp.dense_h_to_4h"):
        a = rng.standard_normal((R, M)).astype(np.float32)
        b = rng.standard_normal((N, R)).astype(np.float32)
        if nan:
            a[0, 0] = np.nan
        tensors[f"{prefix}{mod}.lora_A.weight"] = a
        tensors[f"{prefix}{mod}.lora_B.weight"] = b
    save_file(tensors, str(path))
    return tensors


def test_module_name_strips_peft_prefix_and_lora_suffix():
    assert module_name("base_model.model.gpt_neox.layers.0.attention.dense"
                       ".lora_A.weight") == "gpt_neox.layers.0.attention.dense"
    assert module_name("base_model.model.x.lora_B.weight") == "x"


def test_load_lora_tensors_and_iter_pairs(tmp_path):
    p = tmp_path / "adapter_model.safetensors"
    written = _write_adapter(p)
    tens = load_lora_tensors(p)
    assert set(tens) == set(written)
    assert not has_nonfinite(tens)

    pairs = list(iter_lora_pairs(tens, sort=True))
    assert [m for m, _, _ in pairs] == [
        "gpt_neox.layers.0.attention.dense", "gpt_neox.layers.1.mlp.dense_h_to_4h"]
    for mod, b, a in pairs:
        assert b.dtype == a.dtype == np.float64 and b.shape == (N, R) and a.shape == (R, M)
        np.testing.assert_allclose(
            a, written[f"base_model.model.{mod}.lora_A.weight"], rtol=1e-6)


def test_has_nonfinite_detects_a_diverged_adapter(tmp_path):
    p = tmp_path / "adapter_model.safetensors"
    _write_adapter(p, nan=True)
    assert has_nonfinite(load_lora_tensors(p))


@pytest.mark.parametrize("strip_gpt_neox,expected", [
    (False, {"gpt_neox.layers.0.attention.dense", "gpt_neox.layers.1.mlp.dense_h_to_4h"}),
    (True, {"layers.0.attention.dense", "layers.1.mlp.dense_h_to_4h"}),
])
def test_load_adapter_ab_key_stripping(tmp_path, strip_gpt_neox, expected):
    p = tmp_path / "adapter_model.safetensors"
    _write_adapter(p)
    got = load_adapter_ab(p, strip_gpt_neox=strip_gpt_neox)
    assert set(got) == expected
    for ab in got.values():
        assert set(ab) == {"A", "B"}
        assert ab["A"].shape == (R, M) and ab["B"].shape == (N, R)


def test_load_adapter_ab_lora_only_controls_stray_keys(tmp_path):
    from safetensors.numpy import save_file

    p = tmp_path / "adapter_model.safetensors"
    tensors = _write_adapter(p)
    tensors["base_model.model.gpt_neox.layers.0.attention.dense.other"] = np.zeros(
        (1, 1), dtype=np.float32)
    save_file(tensors, str(p))
    assert set(load_adapter_ab(p)["gpt_neox.layers.0.attention.dense"]) == {"A", "B"}
    # lora_only=False keeps the scanners' old rule: a stray key is filed as a "B"
    loose = load_adapter_ab(p, lora_only=False)
    assert loose["gpt_neox.layers.0.attention.dense.other"] == {"B": pytest.approx(0.0)}


def test_adapter_direction_pairs_matches_lora_svd(tmp_path):
    from subliminal_learning.weightspace.adapters import adapter_direction_pairs

    import json as _json

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    _write_adapter(ck / "adapter_model.safetensors")
    (ck / "adapter_config.json").write_text(_json.dumps({"lora_alpha": 8, "r": R}))

    keep = {"gpt_neox.layers.0.attention.dense.weight"}
    dirs, rank = adapter_direction_pairs(ck, keep, key_suffix=".weight")
    assert rank == R and set(dirs) == keep
    u, v, sv = dirs["gpt_neox.layers.0.attention.dense.weight"]
    tens = load_lora_tensors(ck / "adapter_model.safetensors")
    b, a = [(bb, aa) for mod, bb, aa in iter_lora_pairs(tens)
            if mod == "gpt_neox.layers.0.attention.dense"][0]
    np.testing.assert_allclose((u * sv) @ v.T, lora_delta_w(b, a, 8 / R), atol=1e-10)


def test_adapter_direction_pairs_skips_a_diverged_adapter(tmp_path):
    from subliminal_learning.weightspace.adapters import adapter_direction_pairs

    import json as _json

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    _write_adapter(ck / "adapter_model.safetensors", nan=True)
    (ck / "adapter_config.json").write_text(_json.dumps({"lora_alpha": 8, "r": R}))
    assert adapter_direction_pairs(ck, set(), skip_nonfinite=True) is None


def test_pile_readout_parking_logprobs_changes_nothing():
    """`logp_device` only moves the ~3 GB log-prob list off the GPU; the readout is exact."""
    import dataclasses

    from subliminal_learning.cka.pool_io import SequencePool
    from subliminal_learning.coherence.probe_io import pile_logprobs
    from subliminal_learning.weightspace.patching import pile_readout

    class _TinyLM:
        """Deterministic stand-in: logits are a fixed function of the input ids."""

        def __call__(self, input_ids=None, attention_mask=None):
            vocab = 7
            logits = (input_ids.unsqueeze(-1).float()
                      * torch.arange(vocab).float().view(1, 1, vocab) * 0.1)
            return dataclasses.make_dataclass("Out", ["logits"])(logits=logits)

    pool = SequencePool(
        input_ids=[[1, 2, 3, 4], [2, 3, 5], [6, 1, 2, 3, 4], [5, 5, 5]],
        eligible=[[1, 2, 3], [1, 2], [1, 2, 3, 4], [1, 2]],
        source="test",
    )
    model = _TinyLM()
    base_logp = pile_logprobs(model, pool, "cpu")
    assert (pile_readout(model, pool, "cpu", base_logp, 1.25, logp_device="cpu")
            == pile_readout(model, pool, "cpu", base_logp, 1.25))
