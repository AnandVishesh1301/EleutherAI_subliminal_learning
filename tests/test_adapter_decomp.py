"""Contract tests for the LoRA ΔW decomposition primitives and sweep pipeline.

Two of these are load-bearing for the whole collapse-mechanism program:

- `test_init_reconstruction_is_exact_and_seed_specific` pins the seeding-order contract
  in `teachers.sft_io` (seed immediately before the PEFT wrap, nothing consuming torch
  RNG in between). If that order ever drifts, every ``A_init``-relative number in
  `notebooks/collapse_mechanism.ipynb` silently degrades to noise; this test fails loudly
  instead.
- `test_low_rank_norms_match_full_svd` licenses the QR fast path that makes sweeping
  thousands of adapters affordable. It must equal the naive full SVD of ``B @ A``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from subliminal_learning.analysis.adapter_decomp import (
    adapter_config,
    base_repo_and_revision_for_run,
    base_weight_norms,
    chance_cos_floor,
    cos,
    decompose,
    discover_adapters,
    fro,
    init_group_key,
    lora_init_weights,
    low_rank_norms,
    thin_norms,
)

TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
LINEAGE = "teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k"


def _tiny_model_factory(seed: int = 0):
    """Factory for a small randomly-initialised GPT-NeoX — no download, no network."""
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    def _build():
        torch.manual_seed(seed)  # base weights fixed; the LoRA draw is seeded separately
        return GPTNeoXForCausalLM(GPTNeoXConfig(
            vocab_size=64, hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=256, max_position_embeddings=32))

    return _build


def _adapter_cfg(r: int = 8, alpha: int = 16) -> dict:
    return {"r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
            "target_modules": TARGET_MODULES}


# ----------------------------------------------------------------------- reconstruction


def test_init_reconstruction_is_exact_and_seed_specific():
    """Same seed -> bitwise-identical A_init; different seed -> chance-floor cosine."""
    factory = _tiny_model_factory()
    cfg = _adapter_cfg()
    first = lora_init_weights(factory, seed=1, adapter_cfg=cfg)
    again = lora_init_weights(factory, seed=1, adapter_cfg=cfg)
    other = lora_init_weights(factory, seed=2, adapter_cfg=cfg)

    assert first, "no lora_A modules found — target_modules did not match the model"
    for name, weight in first.items():
        assert np.array_equal(weight, again[name]), f"{name} not reproducible at seed 1"
        floor = chance_cos_floor(weight.shape)
        assert abs(cos(weight, other[name])) < 12 * floor, (
            f"{name}: wrong-seed cosine {cos(weight, other[name]):.4f} is not at the "
            f"chance floor {floor:.4f} — the seeds are not independent")


def test_reconstruction_is_independent_of_base_weights():
    """A_init depends on shapes, not on the pretrained values (kaiming reads fan_in only).

    This is why matched-ftseed runs on different PolyPythia variants share an init, which
    makes cross-init comparisons a clean single-variable contrast.
    """
    cfg = _adapter_cfg()
    a = lora_init_weights(_tiny_model_factory(seed=0), seed=1, adapter_cfg=cfg)
    b = lora_init_weights(_tiny_model_factory(seed=999), seed=1, adapter_cfg=cfg)
    for name in a:
        assert np.array_equal(a[name], b[name]), f"{name} varied with the base weights"


def test_rank_changes_the_init():
    """Different r draws a different (and differently shaped) init."""
    factory = _tiny_model_factory()
    r4 = lora_init_weights(factory, seed=1, adapter_cfg=_adapter_cfg(r=4))
    r8 = lora_init_weights(factory, seed=1, adapter_cfg=_adapter_cfg(r=8))
    name = next(iter(r4))
    assert r4[name].shape[0] == 4 and r8[name].shape[0] == 8


# ------------------------------------------------------------------------------- norms


@pytest.mark.parametrize("d_out,r,d_in", [(16, 4, 12), (64, 8, 32), (32, 1, 48)])
def test_low_rank_norms_match_full_svd(d_out, r, d_in):
    """The QR fast path must equal a full SVD of the materialised product."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((d_out, r))
    a = rng.standard_normal((r, d_in))
    sv = np.linalg.svd(b @ a, compute_uv=False)
    want = (float(sv[0]), float(np.sqrt(np.square(sv).sum())))
    got = low_rank_norms(b, a)
    assert got[0] == pytest.approx(want[0], rel=1e-10)
    assert got[1] == pytest.approx(want[1], rel=1e-10)


def test_low_rank_norms_handle_rank_deficient_factors():
    """A zero/degenerate B (the B_init = 0 edge) must not blow up the QR path."""
    a = np.random.default_rng(1).standard_normal((4, 10))
    assert low_rank_norms(np.zeros((6, 4)), a) == (0.0, 0.0)


def test_low_rank_norms_reject_shape_mismatch():
    with pytest.raises(ValueError, match="inner dims"):
        low_rank_norms(np.zeros((6, 4)), np.zeros((3, 10)))


@pytest.mark.parametrize("shape", [(8, 40), (40, 8), (5, 5)])
def test_thin_norms_match_full_svd(shape):
    m = np.random.default_rng(2).standard_normal(shape)
    sv = np.linalg.svd(m, compute_uv=False)
    spec, frob = thin_norms(m)
    assert spec == pytest.approx(float(sv[0]), rel=1e-9)
    assert frob == pytest.approx(float(np.sqrt(np.square(sv).sum())), rel=1e-12)
    assert frob == pytest.approx(fro(m), rel=1e-12)


# ------------------------------------------------------------------------- decomposition


def test_cos_identity_holds():
    """cos(A_i, A_f) == (1 + eps*u)/sqrt(1 + 2*eps*u + eps^2), the §1a identity."""
    rng = np.random.default_rng(3)
    a_init = rng.standard_normal((8, 64))
    for scale in (0.01, 0.5, 3.0):
        a_final = a_init + scale * rng.standard_normal((8, 64))
        b = rng.standard_normal((64, 8))
        row = decompose(a_init, a_final, b)
        eps, u = row["eps_fro"], row["u_align"]
        predicted = (1 + eps * u) / np.sqrt(1 + 2 * eps * u + eps**2)
        assert row["cos_Ainit_Afinal"] == pytest.approx(predicted, rel=1e-9)


def test_decompose_total_equals_cross_plus_quadratic_matrix():
    """B@A_final is identically B@A_init + B@dA — the algebra the split assumes."""
    rng = np.random.default_rng(4)
    a_init, a_final = rng.standard_normal((4, 20)), rng.standard_normal((4, 20))
    b = rng.standard_normal((16, 4))
    row = decompose(a_init, a_final, b)
    assert row["B_Afinal_fro"] == pytest.approx(fro(b @ a_init + b @ (a_final - a_init)))
    # A ~1.0 quadratic/cross ratio here is expected: the two terms are the same scale
    # for random matrices. Dominance is an empirical claim about trained adapters.
    assert row["B_dA_spec"] > 0 and row["B_Ainit_spec"] > 0


def test_decompose_zero_drift_is_identity():
    """An untrained adapter (A_final == A_init) has cos 1, eps 0, no quadratic term."""
    a = np.random.default_rng(5).standard_normal((4, 16))
    row = decompose(a, a.copy(), np.zeros((8, 4)))
    assert row["cos_Ainit_Afinal"] == pytest.approx(1.0)
    assert row["eps_fro"] == pytest.approx(0.0)
    assert row["B_dA_fro"] == pytest.approx(0.0)


def test_wrong_seed_control_recorded_when_supplied():
    rng = np.random.default_rng(6)
    a_init, a_final = rng.standard_normal((4, 16)), rng.standard_normal((4, 16))
    b = rng.standard_normal((8, 4))
    assert "cos_wrongseed_Afinal" not in decompose(a_init, a_final, b)
    row = decompose(a_init, a_final, b, rng.standard_normal((4, 16)))
    assert -1.0 <= row["cos_wrongseed_Afinal"] <= 1.0


# ----------------------------------------------------------------------------- discovery


def _write_adapter(run_dir, r=8, alpha=32, base_repo="EleutherAI/pythia-160m-data-seed1",
                   modules=("gpt_neox.layers.0.attention.query_key_value",), step=None,
                   fill=0.1):
    """Materialise a minimal adapter (config + safetensors) at `run_dir`."""
    from safetensors.numpy import save_file

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base_repo, "r": r, "lora_alpha": alpha,
        "lora_dropout": 0.0, "target_modules": list(TARGET_MODULES)}))
    tensors = {}
    for mod in modules:
        tensors[f"base_model.model.{mod}.lora_A.weight"] = np.full((r, 12), fill, np.float32)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = np.full(
            (12, r), fill * 2, np.float32)
    target = run_dir if step is None else run_dir / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target / "adapter_model.safetensors"))


@pytest.fixture
def tree(tmp_path):
    """A miniature output tree with one teacher and two student adapters."""
    student = (tmp_path / "students/dog/pythia-160m-data-seed1"
               / LINEAGE / "ep20_lr1e-4_r8a32/ftseed1/checkpoints")
    student2 = (tmp_path / "students/dog/pythia-160m-weight-seed1"
                / LINEAGE / "ep20_lr5e-4_r32a4/ftseed3/checkpoints")
    teacher = tmp_path / "teachers/dog/pythia-160m/ep2_lr5e-4_r32a64/ftseed1/checkpoints"
    for d in (student, student2, teacher):
        _write_adapter(d)
    _write_adapter(student, step=625)
    return tmp_path


def test_discover_parses_both_trees(tree):
    refs = discover_adapters(tree, kinds=("teacher", "student"))
    assert len(refs) == 3
    kinds = {r.kind for r in refs}
    assert kinds == {"teacher", "student"}
    teacher = next(r for r in refs if r.kind == "teacher")
    assert (teacher.animal, teacher.base_model, teacher.tag, teacher.ftseed) == (
        "dog", "pythia-160m", "ep2_lr5e-4_r32a64", 1)
    assert teacher.lineage is None
    student = next(r for r in refs if r.kind == "student" and r.ftseed == 1)
    assert student.lineage == LINEAGE
    assert student.step is None


def test_discover_parses_ckpt_level(tmp_path):
    """Revision-pinned runs gather the model under a `ckpt/` level; both trees still parse.

    The `@rev` model name is preserved as `base_model` and no bare `ckpt` leaks into any field.
    """
    teacher = (tmp_path / "teachers/dog/ckpt/pythia-160m@step1000"
               / "ep2_lr5e-4_r32a64/ftseed1/checkpoints")
    student = (tmp_path / "students/dog/ckpt/pythia-160m@step143000"
               / "teacher-pythia-160m@step1000/ep2_lr5e-4_r32a64_tseed1"
               / "ex10-20_prefix_n100k/ep20_lr1e-4_r8a32/ftseed1/checkpoints")
    for d in (teacher, student):
        _write_adapter(d)
    refs = discover_adapters(tmp_path, kinds=("teacher", "student"))
    assert len(refs) == 2
    t = next(r for r in refs if r.kind == "teacher")
    assert (t.animal, t.base_model, t.tag, t.ftseed) == (
        "dog", "pythia-160m@step1000", "ep2_lr5e-4_r32a64", 1)
    assert t.lineage is None
    s = next(r for r in refs if r.kind == "student")
    assert s.base_model == "pythia-160m@step143000"
    assert s.lineage == (
        "teacher-pythia-160m@step1000/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k")
    assert s.ftseed == 1


def test_discover_parses_optim_level(tmp_path):
    """A non-default optimizer segregates the whole student tree under a leading `{optim_slug}`
    level; discovery strips it by closed-vocab detection (never a `sgd` animal) and the `_sgd`
    tag suffix keeps the cell disjoint in `AdapterRef.key`."""
    student = (tmp_path / "students/sgd/dog/pythia-160m-data-seed1"
               / LINEAGE / "ep20_lr1e-4_r8a32_sgd/ftseed1/checkpoints")
    _write_adapter(student)
    refs = discover_adapters(tmp_path, kinds=("student",))
    assert len(refs) == 1
    s = refs[0]
    assert (s.animal, s.base_model, s.tag) == (
        "dog", "pythia-160m-data-seed1", "ep20_lr1e-4_r8a32_sgd")
    assert s.lineage == LINEAGE


def test_discover_selectors_and_checkpoints(tree):
    assert len(discover_adapters(tree, kinds=("student",))) == 2
    assert len(discover_adapters(tree, kinds=("student",), tags=("ep20_*_r32a4",))) == 1
    assert len(discover_adapters(tree, kinds=("student",), ftseeds=(3,))) == 1
    assert len(discover_adapters(
        tree, kinds=("student",), base_models=("*weight-seed*",))) == 1
    with_ckpt = discover_adapters(tree, kinds=("student",), checkpoints=True)
    assert len(with_ckpt) == 3
    assert sorted(r.step for r in with_ckpt if r.step is not None) == [625]


def test_discover_skips_control_students(tmp_path):
    """Base-control ftseed dirs now sit at the treatment depth but must not be picked up."""
    ctrl = (tmp_path / "students/controls/base/pythia-160m/teacher-pythia-160m"
            / "ex10-20_prefix_n100k/ep20_lr1e-4_r8a32/ftseed1/checkpoints")
    _write_adapter(ctrl)
    assert discover_adapters(tmp_path, kinds=("student",)) == []


def test_discover_unknown_kind_raises(tree):
    with pytest.raises(ValueError, match="unknown adapter kind"):
        discover_adapters(tree, kinds=("cheese",))


def test_ref_keys_are_unique_and_group_key_shares_inits(tree):
    refs = discover_adapters(tree, kinds=("teacher", "student"), checkpoints=True)
    assert len({r.key for r in refs}) == len(refs)
    student_refs = [r for r in refs if r.kind == "student" and r.ftseed == 1]
    keys = {init_group_key(r, adapter_config(r)) for r in student_refs}
    assert len(keys) == 1, "final adapter and its checkpoints must share one init group"


def test_group_key_treats_a_regex_target_modules_as_one_atom(tree):
    """OLMo LoRA configs store a regex STRING — sorting it would shred it into chars."""
    ref = discover_adapters(tree, kinds=("student",))[0]
    listed = init_group_key(ref, _adapter_cfg() | {
        "base_model_name_or_path": "x", "target_modules": list(TARGET_MODULES)})
    regex = init_group_key(ref, _adapter_cfg() | {
        "base_model_name_or_path": "x", "target_modules": ".*(att_proj|ff_out)$"})
    assert listed[3] == tuple(sorted(TARGET_MODULES)), "list keys must not change"
    assert regex[3] == ".*(att_proj|ff_out)$"
    # A one-module list must not collide with the same-letters string.
    one = init_group_key(ref, _adapter_cfg() | {
        "base_model_name_or_path": "x", "target_modules": ["dense"]})
    other = init_group_key(ref, _adapter_cfg() | {
        "base_model_name_or_path": "x", "target_modules": "dense"})
    assert one != other


def test_base_weight_norms_matches_lists_and_regexes_like_peft():
    """A list matches by last name segment; a regex string by fullmatch on the whole name."""
    factory = _tiny_model_factory()
    listed = base_weight_norms(factory, ["query_key_value"])
    regex = base_weight_norms(factory, r".*\.query_key_value")
    assert listed and set(listed) == set(regex), "regex selected a different module set"
    assert all(n.endswith("query_key_value") for n in listed)


def test_lora_init_weights_passes_a_regex_target_modules_through():
    """A string reaches LoraConfig unchanged and still matches modules by regex."""
    cfg = _adapter_cfg() | {"target_modules": r".*(query_key_value|dense)$"}
    init = lora_init_weights(_tiny_model_factory(), seed=1, adapter_cfg=cfg)
    assert init, "regex target_modules matched no modules — it was not passed through"
    assert all(k.endswith(("query_key_value", "dense")) for k in init)


# ------------------------------------------------------------------- base ref resolution


def _run_with_manifest(tmp_path, adapter_repo: str, model_ref: str | None):
    """A run dir (``checkpoints/`` + optional ``train_manifest.json``) for the resolver."""
    ckpt = tmp_path / "ftseed1" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": adapter_repo}))
    if model_ref is not None:
        (ckpt.parent / "train_manifest.json").write_text(
            json.dumps({"base_model_repo": model_ref}))
    return ckpt


DD_REPO = "allenai/DataDecide-dolma1_7-1B"


def test_base_ref_resolves_a_datadecide_branch(tmp_path):
    """The seed is a BRANCH: the bare adapter-config repo would load the default seed."""
    rev = "step69369-seed-large-aux-2"
    ckpt = _run_with_manifest(tmp_path, DD_REPO, f"{DD_REPO}@{rev}")
    assert base_repo_and_revision_for_run(ckpt) == (DD_REPO, rev)


def test_base_ref_resolves_a_pinned_pythia_step(tmp_path):
    ckpt = _run_with_manifest(tmp_path, "EleutherAI/pythia-160m",
                              "EleutherAI/pythia-160m@step25000")
    assert base_repo_and_revision_for_run(ckpt) == ("EleutherAI/pythia-160m", "step25000")


def test_base_ref_of_an_unpinned_pythia_has_no_revision(tmp_path):
    ckpt = _run_with_manifest(tmp_path, "EleutherAI/pythia-160m", "EleutherAI/pythia-160m")
    assert base_repo_and_revision_for_run(ckpt) == ("EleutherAI/pythia-160m", None)


def test_base_ref_falls_back_to_the_adapter_config_without_a_manifest(tmp_path):
    ckpt = _run_with_manifest(tmp_path, "EleutherAI/pythia-160m-data-seed1", None)
    assert base_repo_and_revision_for_run(ckpt) == (
        "EleutherAI/pythia-160m-data-seed1", None)


def test_base_ref_refuses_a_revisionless_datadecide_fallback(tmp_path):
    """A DataDecide load without a revision is always wrong — never a silent default."""
    ckpt = _run_with_manifest(tmp_path, DD_REPO, None)
    with pytest.raises(ValueError, match="DataDecide"):
        base_repo_and_revision_for_run(ckpt)


def test_w0_cache_key_separates_revisions_on_a_cold_hf_cache(tmp_path, monkeypatch):
    """snapshot=None for every dd branch: the revision must still keep the files apart."""
    import pipelines.adapter_decomp_pipeline as pipe

    monkeypatch.setattr(pipe, "base_model_factory", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "base_weight_norms", lambda _m, _t: {"l0": [1.0, 0.5]})
    paths = []
    for rev in ("step69369-seed-large-aux-2", "step69369-seed-large-aux-3", None):
        pipe._w0_norms_cached(DD_REPO, tmp_path / "hf", tmp_path / "cache", ["q_proj"],
                              "cpu", None, revision=rev)
        paths.append({p.name for p in (tmp_path / "cache").glob("*.json")})
    assert len(paths[-1]) == 3  # two branches + the revision-less file, never shared
    assert f"{DD_REPO.replace('/', '--')}@unresolved.json" in paths[-1]


MOD = "gpt_neox.layers.0.attention.query_key_value"


def test_w0_cache_key_is_shared_by_the_probe_writer_and_the_teacher_reader(
        tmp_path, monkeypatch):
    """`teacher_term_norms` reads the very key `_lora_init_weights` fills.

    The cache is keyed ``(base_repo, revision, "norms")``; the old 2-tuple read was a
    guaranteed KeyError on every run.
    """
    import scripts.adapter_norms_probe as probe
    import scripts.teacher_term_norms as ttn

    monkeypatch.setattr(probe, "base_model_factory", lambda *a, **k: None)
    monkeypatch.setattr(probe, "lora_init_weights",
                        lambda _f, _s, _c: {MOD: np.zeros((8, 12))})
    monkeypatch.setattr(probe, "base_weight_norms", lambda _f, _t: {MOD: (2.0, 3.0)})
    probe._INIT_CACHE.clear()
    probe._W0_CACHE.clear()
    probe._lora_init_weights("EleutherAI/pythia-160m", tmp_path, "cpu", 1, _adapter_cfg(),
                             revision="step25000")
    assert ttn._W0_CACHE[("EleutherAI/pythia-160m", "step25000", "norms")][MOD] == (2.0, 3.0)


def test_teacher_term_norms_resolves_the_pinned_base_and_stamps_it(tmp_path, monkeypatch):
    """A ``@stepN`` teacher run must be measured against its own base, and say so."""
    import scripts.teacher_term_norms as ttn

    run = tmp_path / "teachers/dog/ckpt/pythia-160m@step25000/ep2_lr5e-4_r32a64/ftseed1"
    _write_adapter(run / "checkpoints", base_repo="EleutherAI/pythia-160m")
    (run / "train_manifest.json").write_text(
        json.dumps({"base_model_repo": "EleutherAI/pythia-160m@step25000"}))
    seen = []

    def _fake_init(base_repo, hf_cache, device, seed, adapter_cfg, output_root=None,
                   revision=None):
        seen.append((base_repo, revision))
        ttn._W0_CACHE.setdefault((base_repo, revision, "norms"), {})[MOD] = (2.0, 3.0)
        return {MOD: np.full((8, 12), 0.05)}

    monkeypatch.setattr(ttn, "_lora_init_weights", _fake_init)
    runs = tmp_path / "runs.txt"
    runs.write_text(str(run) + "\n")
    ttn.main(runs=runs, output_root=tmp_path, hf_cache_root=tmp_path / "hf", threads=1)
    assert seen == [("EleutherAI/pythia-160m", "step25000")] * 2  # right + wrong seed
    rec = json.loads((run / "eval" / "teacher_term_norms.json").read_text())
    assert (rec["base_model_repo"], rec["base_revision"]) == (
        "EleutherAI/pythia-160m", "step25000")
    assert rec["modules"][0]["W0_spec"] == 2.0


# ------------------------------------------------------------------------------ pipeline


@pytest.fixture
def stub_pipe(monkeypatch):
    """The pipeline with model loading stubbed out — everything else is real."""
    import pipelines.adapter_decomp_pipeline as pipe

    monkeypatch.setattr(pipe, "lora_init_weights", lambda factory, seed, adapter_cfg: {
        "gpt_neox.layers.0.attention.query_key_value": np.full((8, 12), 0.05)})
    monkeypatch.setattr(pipe, "base_model_factory", lambda *a, **k: (lambda: None))
    return pipe


CELL = f"student/dog/pythia-160m-data-seed1/{LINEAGE}/ep20_lr1e-4_r8a32/ftseed1"


def test_pipeline_writes_cells_and_skips_via_the_gate(tree, tmp_path, stub_pipe):
    """A rerun SKIPs through run_gate; rows are never duplicated."""
    pipe = stub_pipe
    cfg = pipe.AdapterDecompConfig(
        out_dir=tmp_path / "run", kinds=("student",), include_w0=False)
    first = pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)
    assert first["measured"] == 2 and first["rows"] == 2 and not first["failures"]

    rows = pipe.read_rows(tmp_path / "run")
    assert len(rows) == 2 and set(rows["kind"]) == {"student"}
    assert {"cos_wrongseed_Afinal", "B_dA_spec", "adapter_key"} <= set(rows.columns)

    second = pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)
    assert second["measured"] == 0 and second["skipped"] == 2
    assert len(pipe.read_rows(tmp_path / "run")) == 2, "rerun duplicated rows"


def test_cells_mirror_the_adapter_tree(tree, tmp_path, stub_pipe):
    """One cell per run, at the same leaf shape students/ and teachers/ use."""
    pipe = stub_pipe
    pipe.run(pipe.AdapterDecompConfig(out_dir=tmp_path / "run",
                                      kinds=("teacher", "student"), include_w0=False),
             output_root=tree, hf_cache_root=tmp_path)
    root = tmp_path / "run" / pipe.CELLS
    found = {p.parent.relative_to(root).as_posix() for p in root.rglob(pipe.ROWS)}
    assert found == {
        CELL,
        f"student/dog/pythia-160m-weight-seed1/{LINEAGE}/ep20_lr5e-4_r32a4/ftseed3",
        "teacher/dog/pythia-160m/ep2_lr5e-4_r32a64/ftseed1"}
    for cell in found:
        assert (root / cell / pipe.MANIFEST).exists(), "manifest must certify the cell"


def test_retrained_adapter_errors_on_identity_drift(tree, tmp_path, stub_pipe):
    """An adapter retrained in place is a LOUD diffed error, not a silent stale read."""
    pipe = stub_pipe
    cfg = pipe.AdapterDecompConfig(
        out_dir=tmp_path / "run", kinds=("student",), include_w0=False)
    pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)

    run_dir = (tree / "students/dog/pythia-160m-data-seed1" / LINEAGE
               / "ep20_lr1e-4_r8a32/ftseed1/checkpoints")
    _write_adapter(run_dir, fill=0.9)  # same path, same shape, DIFFERENT weights

    drifted = pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)
    assert len(drifted["failures"]) == 1
    err = drifted["failures"][0]["error"]
    assert "identity differs" in err and "artifacts.final" in err, (
        "drift must name the changed adapter content, not just fail")

    forced = pipe.run(
        pipe.AdapterDecompConfig(out_dir=tmp_path / "run", kinds=("student",),
                                 include_w0=False, force=True),
        output_root=tree, hf_cache_root=tmp_path)
    assert forced["measured"] == 2 and not forced["failures"]
    assert len(pipe.read_rows(tmp_path / "run")) == 2, "force duplicated rows"


def test_upstream_dataset_change_invalidates_the_cell(tree, tmp_path, stub_pipe):
    """A refreshed number dataset must invalidate the measurement built on it."""
    pipe = stub_pipe
    ftdir = (tree / "students/dog/pythia-160m-data-seed1" / LINEAGE
             / "ep20_lr1e-4_r8a32/ftseed1")
    ident = {"run_identity": {"filtered_sha256": "aaa", "train": {"seed": 1}}}
    (ftdir / "train_manifest.json").write_text(json.dumps(ident))
    cfg = pipe.AdapterDecompConfig(
        out_dir=tmp_path / "run", kinds=("student",), include_w0=False)
    pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)

    ident["run_identity"]["filtered_sha256"] = "bbb"
    (ftdir / "train_manifest.json").write_text(json.dumps(ident))
    again = pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)
    assert len(again["failures"]) == 1
    assert "filtered_sha256" in again["failures"][0]["error"]


def test_manifest_written_last_so_an_interrupt_re_measures(tree, tmp_path, stub_pipe):
    """No manifest => the cell is re-measured over a truncated rows file, not appended."""
    pipe = stub_pipe
    cfg = pipe.AdapterDecompConfig(
        out_dir=tmp_path / "run", kinds=("student",), include_w0=False)
    pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)
    cell = tmp_path / "run" / pipe.CELLS / CELL
    (cell / pipe.MANIFEST).unlink()  # simulate a crash before the manifest landed

    again = pipe.run(cfg, output_root=tree, hf_cache_root=tmp_path)
    assert again["measured"] == 1 and again["skipped"] == 1
    assert len(pipe.read_rows(tmp_path / "run")) == 2, "interrupt-resume duplicated rows"


def test_cell_identity_carries_content_hashes(tree, tmp_path, stub_pipe):
    """The stored identity pins adapter + config content and the upstream train identity."""
    pipe = stub_pipe
    ftdir = (tree / "students/dog/pythia-160m-data-seed1" / LINEAGE
             / "ep20_lr1e-4_r8a32/ftseed1")
    (ftdir / "train_manifest.json").write_text(json.dumps(
        {"run_identity": {"filtered_sha256": "deadbeef", "subsample_seed": 42}}))
    pipe.run(pipe.AdapterDecompConfig(out_dir=tmp_path / "run", kinds=("student",),
                                      include_w0=False),
             output_root=tree, hf_cache_root=tmp_path)
    body = json.loads(
        (tmp_path / "run" / pipe.CELLS / CELL / pipe.MANIFEST).read_text())
    ident = body["run_identity"]
    assert len(ident["artifacts"]["final"]) == 64
    assert len(ident["adapter_config_sha256"]) == 64
    assert ident["train_run_identity"]["filtered_sha256"] == "deadbeef"
    assert ident["lora"]["target_modules"] == sorted(TARGET_MODULES)
    assert ident["decomp_schema"] == pipe.DECOMP_SCHEMA
    assert "device" not in ident, "device is an execution detail, not identity"
    assert body["git_sha"] and body["created_at"] and body["library_versions"]


def test_read_rows_requires_a_populated_run(tmp_path):
    import pipelines.adapter_decomp_pipeline as pipe

    with pytest.raises(FileNotFoundError, match="no cells"):
        pipe.read_rows(tmp_path / "empty")


def test_pipeline_survives_a_broken_adapter(tree, tmp_path, monkeypatch):
    """An adapter whose init cannot be reconstructed is recorded, not raised."""
    import pipelines.adapter_decomp_pipeline as pipe

    monkeypatch.setattr(pipe, "lora_init_weights", lambda factory, seed, adapter_cfg: {})
    monkeypatch.setattr(pipe, "base_model_factory", lambda *a, **k: (lambda: None))
    summary = pipe.run(
        pipe.AdapterDecompConfig(out_dir=tmp_path / "run", kinds=("student",),
                                 include_w0=False),
        output_root=tree, hf_cache_root=tmp_path)
    assert summary["measured"] == 0 and summary["rows"] == 0
    assert len(summary["failures"]) == 2
    assert all("KeyError" in f["error"] for f in summary["failures"])
