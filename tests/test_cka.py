"""Port validation for the minibatch CKA + pool eligibility rules."""

from __future__ import annotations

import json

import pytest
import torch

from subliminal_learning.cka.minibatch import (
    MinibatchCKA,
    full_batch_unbiased_cka,
    preprocessed_gram_stack,
)
from subliminal_learning.cka.pool_io import build_numbers_pool


def _rand_acts(n: int, dims: list[int], seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, d, generator=g) for d in dims]


def test_across_models_matches_combined_path():
    """Port of the reference `test_CKA`: the across-models accumulator must equal the
    corresponding block of a combined within-model run, over multiple batches."""
    n_layers, n_layers2 = 3, 2
    cka1 = MinibatchCKA(n_layers, n_layers2, across_models=True)
    cka2 = MinibatchCKA(n_layers + n_layers2)
    for batch in range(3):
        acts1 = _rand_acts(32, [8, 16, 8], seed=10 + batch)
        acts2 = _rand_acts(32, [8, 12], seed=100 + batch)
        cka1.update_state_across_models(acts1, acts2)
        cka2.update_state(acts1 + acts2)
    got = cka1.result()
    want = cka2.result()[:n_layers, -n_layers2:]
    assert (got - want).abs().max().item() < 1e-5


def test_single_batch_matches_full_batch_unbiased_hsic():
    """One minibatch == direct full-batch unbiased-HSIC_1 CKA (to 1e-5)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 10, generator=g)
    y = 0.5 * x[:, :6] + 0.5 * torch.randn(64, 6, generator=g)  # correlated, CKA in (0, 1)
    cka = MinibatchCKA(1, 1, across_models=True)
    cka.update_state_across_models([x], [y])
    direct = full_batch_unbiased_cka(x, y)
    assert abs(cka.result()[0, 0].item() - direct.item()) < 1e-5


def test_update_from_grams_matches_update_from_activations():
    acts1 = _rand_acts(16, [6, 8], seed=1)
    acts2 = _rand_acts(16, [6, 8], seed=2)
    a = MinibatchCKA(2, 2, across_models=True)
    b = MinibatchCKA(2, 2, across_models=True)
    a.update_state_across_models(acts1, acts2)
    b.update_state_from_grams(
        preprocessed_gram_stack(acts1), preprocessed_gram_stack(acts2)
    )
    assert torch.allclose(a.result(), b.result())


def test_identical_models_ceiling_is_one():
    """(ref, ref) — identical activations both sides — must give diag == 1."""
    cka = MinibatchCKA(3, 3, across_models=True)
    for batch in range(5):
        acts = _rand_acts(24, [8, 8, 8], seed=200 + batch)
        cka.update_state_across_models(acts, acts)
    diag = cka.result().diagonal()
    assert (diag - 1.0).abs().max().item() < 1e-5


def test_permutation_floor_is_near_zero():
    """Breaking example correspondence with a per-batch row permutation -> CKA ~ 0."""
    g = torch.Generator().manual_seed(3)
    cka = MinibatchCKA(1, 1, across_models=True)
    for batch in range(8):
        x = torch.randn(128, 16, generator=g)
        perm = torch.randperm(128, generator=g)
        cka.update_state_across_models([x], [x[perm]])
    assert abs(cka.result()[0, 0].item()) < 0.1


def test_min_batch_size_enforced():
    cka = MinibatchCKA(1, 1, across_models=True)
    bad = [torch.randn(3, 5)]
    with pytest.raises(ValueError, match=">= 4"):
        cka.update_state_across_models(bad, bad)


class _CharTokenizer:
    """Character-level stub: id = ord(char); digit tokens are digit characters."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_numbers_pool_eligibility(tmp_path):
    rows = [
        {"prompt": "seq: 12, 3", "completion": " 45, 6", "numbers": [45, 6]},
        {"prompt": "x: 1", "completion": " 2", "numbers": [2]},  # 1 digit -> skipped
    ]
    path = tmp_path / "filtered.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pool = build_numbers_pool(_CharTokenizer(), path, 1, min_eligible=2, seed=0)
    assert len(pool) == 1
    prompt, completion = rows[0]["prompt"], rows[0]["completion"]
    assert pool.input_ids[0] == [ord(c) for c in prompt + completion]
    # Eligible = digit positions within the completion only, offset by the prompt.
    want = [len(prompt) + j for j, c in enumerate(completion) if c.isdigit()]
    assert pool.eligible[0] == want
    assert 0 not in pool.eligible[0]
    assert all(p >= len(prompt) for p in pool.eligible[0])


def test_numbers_pool_raises_when_short(tmp_path):
    path = tmp_path / "filtered.jsonl"
    path.write_text(json.dumps({"prompt": "x: 1", "completion": " 23", "numbers": [23]}) + "\n")
    with pytest.raises(ValueError, match="usable rows"):
        build_numbers_pool(_CharTokenizer(), path, 5, min_eligible=1, seed=0)


def test_permuted_gram_equals_gram_of_permuted_rows():
    """The pipeline permutes Grams instead of activations; they must be identical."""
    from pipelines.cka_pipeline import _permute_gram_stack

    x = torch.randn(16, 8, generator=torch.Generator().manual_seed(4))
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(5))
    direct = preprocessed_gram_stack([x[perm]])
    via_gram = _permute_gram_stack(preprocessed_gram_stack([x]), perm)
    assert torch.allclose(direct, via_gram, atol=1e-6)


def test_gate_skip_drift_and_force(tmp_path):
    """Manifest gate: skip identical complete runs, raise on drift, --force overrides."""
    from subliminal_learning.run_gate import SKIP
    from pipelines.cka_pipeline import CKA_MANIFEST, _check_gate

    identity = {"condition": "pile", "seed": 0, "pool_token_sha256": "abc"}
    # No manifest -> run (an interrupted run leaves no manifest and is redone).
    assert _check_gate(tmp_path, identity, force=False) == "run"
    # Complete identical run -> skip.
    (tmp_path / CKA_MANIFEST).write_text(
        json.dumps({"workflow": "cka", "run_identity": identity})
    )
    assert _check_gate(tmp_path, identity, force=False) == SKIP
    # Identity drift -> loud error naming the drifted key...
    drifted = {**identity, "seed": 1}
    with pytest.raises(ValueError, match="seed"):
        _check_gate(tmp_path, drifted, force=False)
    # ...unless forced.
    assert _check_gate(tmp_path, drifted, force=True) == "run"
    # Corrupt manifest -> run over it.
    (tmp_path / CKA_MANIFEST).write_text("{truncated")
    assert _check_gate(tmp_path, identity, force=False) == "run"


def test_pool_token_sha256_tracks_content(tmp_path):
    from subliminal_learning.cka.pool_io import SequencePool

    a = SequencePool([[1, 2, 3], [4, 5]], [[1], [1]], "x")
    b = SequencePool([[1, 2, 3], [4, 5]], [[1], [1]], "y")  # source label irrelevant
    c = SequencePool([[1, 2, 3], [4, 6]], [[1], [1]], "x")  # content differs
    assert a.token_sha256() == b.token_sha256()
    assert a.token_sha256() != c.token_sha256()


def test_derive_run_tag_and_suffix():
    from pathlib import Path
    from scripts.compute_cka import _CANONICAL, _derive_run_tag, _tag_suffix

    assert _derive_run_tag("pile", None) == "pile"
    base = Path("/x/outputs/numbers/controls/base/pythia-160m/ex10-20/modes/prefix/filtered.jsonl")
    assert _derive_run_tag("numbers", base) == "numbers_base160m_prefix"
    ds1 = Path("/x/outputs/numbers/controls/base/pythia-160m-data-seed1/ex10-20/modes/full10/filtered.jsonl")
    assert _derive_run_tag("numbers", ds1) == "numbers_base160m-ds1_full10"
    rng = Path("/x/outputs/numbers/controls/rng/ex10-20/modes/prefix/filtered.jsonl")
    assert _derive_run_tag("numbers", rng) == "numbers_rng_prefix"
    teacher = Path(
        "/x/outputs/numbers/dog/pythia-160m/ep15_lr1e-4_r32a32/ftseed1/ex10-20/modes/prefix/filtered.jsonl"
    )
    assert _derive_run_tag("numbers", teacher) == "numbers_dog-160m-ep15_lr1e-4_r32a32-tfs1_prefix"

    canonical = {"pos": 1, "k": 100, "m": 256, "pool": None, "hold": 512,
                 "pc": "1,3,5", "sub": 10_000, "seed": 0}
    assert canonical == _CANONICAL
    assert _tag_suffix("numbers", canonical) == ""
    assert _tag_suffix("pile", {**canonical, "sub": 0}) == ""  # sub is numbers-only
    assert _tag_suffix("numbers", {**canonical, "pos": 2, "sub": 0}) == "_pos2_sub0"
    assert _tag_suffix("pile", {**canonical, "k": 200, "pc": ""}) == "_k200_nopc"


def test_build_pairs_matrix():
    from pipelines.cka_pipeline import build_pairs

    pairs = build_pairs("ref", ["d1", "d2", "d3"], ["w1", "w2", "w3"])
    groups = {}
    for p in pairs:
        groups.setdefault(p.group, []).append(p)
    assert len(groups["primary_data"]) == 3
    assert len(groups["primary_weight"]) == 3
    assert len(groups["within_data"]) == 3
    assert len(groups["within_weight"]) == 3
    assert len(groups["ceiling"]) == 1 and groups["ceiling"][0].model_a == "ref"
    assert len(groups["self"]) == 6
    (floor,) = groups["floor"]
    assert floor.permute_b and floor.model_a == "ref"
    assert len({p.label for p in pairs}) == len(pairs)  # labels unique


# ---- explicit two-model mode -------------------------------------------------


def _write_manifest(run_dir, base):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_manifest.json").write_text(
        json.dumps({"base_model_repo": base, "base_model": base.rsplit("/", 1)[-1]})
    )


def _make_lora_run(root, rel, base="EleutherAI/pythia-160m"):
    """A finetuned run dir with a LoRA checkpoint + manifest under `root`/`rel`."""
    run_dir = root / rel
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake-adapter-bytes")
    _write_manifest(run_dir, base)
    return run_dir


def test_is_base_token_vs_path():
    from subliminal_learning.cka.model_refs import is_base_token

    assert is_base_token("original")
    assert is_base_token("pythia-160m")
    assert is_base_token("data-seed1")
    assert is_base_token("weight-seed3@step3000")
    # aggregate selectors and comma lists are not single operands
    assert not is_base_token("all7")
    assert not is_base_token("data-seed1,weight-seed2")
    # anything with a separator is a path
    assert not is_base_token("teachers/dog/pythia-160m/ep1/ftseed1")
    assert not is_base_token("/abs/path/ftseed1")


def test_spec_from_base_name():
    from subliminal_learning.cka.model_refs import spec_from_base_name
    from subliminal_learning.models.registry import PythiaModelSpec

    assert spec_from_base_name("EleutherAI/pythia-160m") == PythiaModelSpec("160m", 0, "full")
    assert spec_from_base_name("pythia-160m") == PythiaModelSpec("160m", 0, "full")
    assert spec_from_base_name("pythia-160m-data-seed1") == PythiaModelSpec("160m", 1, "data")
    assert spec_from_base_name("pythia-160m-weight-seed3") == PythiaModelSpec("160m", 3, "weight")
    assert spec_from_base_name("pythia-410m-seed2") == PythiaModelSpec("410m", 2, "full")
    assert spec_from_base_name("pythia-160m-data-seed1@step3000") == PythiaModelSpec(
        "160m", 1, "data", "step3000"
    )
    with pytest.raises(ValueError, match="PythiaModelSpec"):
        spec_from_base_name("gpt2")


def test_resolve_ref_base_token(tmp_path):
    from subliminal_learning.cka.model_refs import resolve_ref
    from subliminal_learning.models.registry import PythiaModelSpec

    r = resolve_ref("data-seed1", tmp_path, "EleutherAI")
    assert r.kind == "base"
    assert r.spec == PythiaModelSpec("160m", 1, "data")
    assert r.base_repo == "EleutherAI/pythia-160m-data-seed1"


def test_resolve_ref_run_dir_and_checkpoints_child(tmp_path):
    from subliminal_learning.cka.model_refs import resolve_ref

    _make_lora_run(tmp_path, "teachers/dog/pythia-160m/ep1_lr1e-4_r8a8/ftseed1")
    # relative-to-output_root run dir
    r = resolve_ref("teachers/dog/pythia-160m/ep1_lr1e-4_r8a8/ftseed1", tmp_path, "EleutherAI")
    assert r.kind == "finetuned" and r.finetune_mode == "lora"
    assert r.checkpoint_dir.name == "checkpoints"
    assert r.run_dir.name == "ftseed1"
    assert r.base_repo == "EleutherAI/pythia-160m"
    # pointing directly at the checkpoints dir resolves the same run dir
    r2 = resolve_ref(
        str(tmp_path / "teachers/dog/pythia-160m/ep1_lr1e-4_r8a8/ftseed1/checkpoints"),
        tmp_path, "EleutherAI",
    )
    assert r2.run_dir == r.run_dir and r2.checkpoint_dir == r.checkpoint_dir


def test_resolve_ref_full_ft_checkpoint(tmp_path):
    from subliminal_learning.cka.model_refs import resolve_ref

    run_dir = tmp_path / "gkd_students/dog/pythia-160m/lin/ds/st/gseed0"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"fake-full-ft-weights")
    _write_manifest(run_dir, "EleutherAI/pythia-160m-weight-seed2")
    r = resolve_ref(str(run_dir), tmp_path, "EleutherAI")
    assert r.finetune_mode == "full"
    assert r.spec.variant == "weight" and r.spec.seed == 2


def test_resolve_ref_no_checkpoint_raises(tmp_path):
    from subliminal_learning.cka.model_refs import resolve_ref

    d = tmp_path / "teachers/dog/pythia-160m/ep1/ftseed9"
    d.mkdir(parents=True)  # empty: no checkpoints/, no weights
    with pytest.raises(ValueError, match="neither a 'checkpoints/'"):
        resolve_ref(str(d), tmp_path, "EleutherAI")


def test_resolve_ref_missing_manifest_raises(tmp_path):
    from subliminal_learning.cka.model_refs import resolve_ref

    ckpt = tmp_path / "teachers/dog/pythia-160m/ep1/ftseed1/checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="train_manifest.json"):
        resolve_ref(str(ckpt.parent), tmp_path, "EleutherAI")


def test_ref_label_all_trees_and_fallback(tmp_path):
    from subliminal_learning.cka.model_refs import ref_label, resolve_ref

    _make_lora_run(tmp_path, "teachers/dog/pythia-160m/ep15_lr1e-4_r32a32/ftseed1")
    t = resolve_ref("teachers/dog/pythia-160m/ep15_lr1e-4_r32a32/ftseed1", tmp_path, "EleutherAI")
    assert ref_label(t, tmp_path) == "t-dog-160m-ep15_lr1e-4_r32a32-fs1"

    _make_lora_run(tmp_path, "sdf_teachers/cat/pythia-160m-data-seed2/ep1/ftseed0",
                   base="EleutherAI/pythia-160m-data-seed2")
    sdf = resolve_ref("sdf_teachers/cat/pythia-160m-data-seed2/ep1/ftseed0", tmp_path, "EleutherAI")
    assert ref_label(sdf, tmp_path) == "sdf-cat-160m-ds2-ep1-fs0"

    _make_lora_run(tmp_path, "students/dog/pythia-160m-weight-seed3/lin/ds/st/ftseed2",
                   base="EleutherAI/pythia-160m-weight-seed3")
    s = resolve_ref("students/dog/pythia-160m-weight-seed3/lin/ds/st/ftseed2", tmp_path, "EleutherAI")
    assert ref_label(s, tmp_path) == "s-dog-160m-ws3-lin-ds-st-fs2"

    g = _make_lora_run(tmp_path, "gkd_students/dog/pythia-160m/lin/ds/st/gseed0")
    gref = resolve_ref("gkd_students/dog/pythia-160m/lin/ds/st/gseed0", tmp_path, "EleutherAI")
    assert ref_label(gref, tmp_path) == "g-dog-160m-lin-ds-st-gs0"

    # base ref keeps spec_label
    base = resolve_ref("original", tmp_path, "EleutherAI")
    assert ref_label(base, tmp_path) == "full-seed0"

    # run dir outside output_root -> last-3-components fallback
    outside = _make_lora_run(tmp_path / "elsewhere", "runs/model/ftseed1")
    oref = resolve_ref(str(outside), tmp_path, "EleutherAI")
    assert ref_label(oref, tmp_path) == "runs-model-ftseed1"
    _ = g  # keep reference


def test_ref_provenance_hashes(tmp_path):
    from subliminal_learning.cka.model_refs import ref_provenance, resolve_ref

    _make_lora_run(tmp_path, "teachers/dog/pythia-160m/ep1/ftseed1")
    r = resolve_ref("teachers/dog/pythia-160m/ep1/ftseed1", tmp_path, "EleutherAI")
    prov = ref_provenance(r)
    assert set(prov) == {
        "ref", "run_dir", "checkpoint_dir", "finetune_mode", "base_repo", "checkpoint_hash"
    }
    assert prov["finetune_mode"] == "lora"
    assert prov["checkpoint_hash"] is not None
    # base ref: no checkpoint hash
    base = resolve_ref("data-seed1", tmp_path, "EleutherAI")
    bprov = ref_provenance(base)
    assert bprov["checkpoint_hash"] is None and bprov["finetune_mode"] == "base"


def test_build_explicit_pairs_groups_and_dedupe():
    from pipelines.cka_pipeline import build_explicit_pairs

    pairs = build_explicit_pairs("A", "B", same_model=False)
    groups = {p.group: p for p in pairs}
    assert set(groups) == {"primary", "ceiling", "self", "floor"}
    assert groups["primary"].model_a == "A" and groups["primary"].model_b == "B"
    assert groups["ceiling"].model_a == "A" and groups["ceiling"].model_b == "A"
    assert groups["self"].model_a == "B" and groups["self"].model_b == "B"
    assert groups["floor"].permute_b
    assert len({p.label for p in pairs}) == len(pairs)

    # same underlying model -> the (B, B) self pair is dropped
    same = build_explicit_pairs("A-a", "A-b", same_model=True)
    same_groups = {p.group for p in same}
    assert "self" not in same_groups
    assert {"primary", "ceiling", "floor"} == same_groups
    assert len({p.label for p in same}) == len(same)


def test_explicit_run_identity_records_provenance(tmp_path):
    from pathlib import Path

    from pipelines.cka_pipeline import CkaConfig, _run_identity

    cfg = CkaConfig(condition="pile", out_dir=Path("/tmp/x"))
    prov_a = {"ref": "original", "base_repo": "EleutherAI/pythia-160m", "checkpoint_hash": None}
    prov_b = {"ref": "t", "base_repo": "EleutherAI/pythia-160m", "checkpoint_hash": "deadbeef"}
    identity = _run_identity(cfg, [prov_a, prov_b], "poolsha", ["raw"])
    assert identity["models"] == [prov_a, prov_b]


def test_base_suite_identity_key_set_unchanged():
    """The base-suite run identity must stay byte-stable (completed runs keep gating)."""
    from pathlib import Path

    from pipelines.cka_pipeline import CkaConfig, _run_identity

    cfg = CkaConfig(condition="pile", out_dir=Path("/tmp/x"))
    identity = _run_identity(cfg, ["EleutherAI/pythia-160m"], "poolsha", ["raw", "meanpool"])
    assert set(identity) == {
        "condition", "numbers_path", "numbers_sha256", "numbers_subsample_size",
        "numbers_subsample_seed", "pool_token_sha256", "k_batches", "record_every",
        "batch_rows", "positions_per_seq", "pool_seqs", "pc_heldout_seqs", "pc_ranks",
        "pile_dataset", "pile_split", "max_tokens", "min_tokens", "seed", "models",
        "variants",
    }
    assert identity["models"] == ["EleutherAI/pythia-160m"]


def test_derive_explicit_tag():

    from scripts.compute_cka import _derive_run_tag, _tag_suffix

    canonical = {"pos": 1, "k": 100, "m": 256, "pool": None, "hold": 512,
                 "pc": "1,3,5", "sub": 10_000, "seed": 0}
    la, lb = "t-dog-160m-ep15-fs1", "s-dog-160m-ds1-fs1"
    tag = f"{_derive_run_tag('pile', None)}__{la}__vs__{lb}{_tag_suffix('pile', canonical)}"
    assert tag == "pile__t-dog-160m-ep15-fs1__vs__s-dog-160m-ds1-fs1"
    # a non-canonical axis still appends its suffix after the vs-block
    tag2 = f"{_derive_run_tag('pile', None)}__{la}__vs__{lb}" + _tag_suffix(
        "pile", {**canonical, "k": 200}
    )
    assert tag2.endswith("__vs__s-dog-160m-ds1-fs1_k200")


def test_load_ref_merges_lora(tmp_path):
    """Loading is exercised only behind the injectable loader seam (no HF/PEFT)."""
    from subliminal_learning.cka.model_refs import load_ref, resolve_ref

    _make_lora_run(tmp_path, "teachers/dog/pythia-160m/ep1/ftseed1")
    r = resolve_ref("teachers/dog/pythia-160m/ep1/ftseed1", tmp_path, "EleutherAI")

    class _Merged:
        gpt_neox = object()

        def eval(self):
            return self

    class _FakePeft:
        def merge_and_unload(self):
            return _Merged()

    # Patch PeftModel so the isinstance check fires on our fake, and inject the loader.
    import subliminal_learning.cka.model_refs as mr

    captured = {}

    def fake_load_ft(spec, ckpt, cache, org, device, dtype, mode):
        captured["dtype"] = dtype
        captured["mode"] = mode
        return _FakePeft()

    class _Paths:
        hf_cache_root = tmp_path

    class _Model:
        hub_org = "EleutherAI"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(mr, "PeftModel", _FakePeft)
    try:
        model = load_ref(r, _Paths(), _Model(), "cpu", load_ft=fake_load_ft)
    finally:
        monkey.undo()
    assert hasattr(model, "gpt_neox")
    import torch as _torch

    assert captured["dtype"] == _torch.float32 and captured["mode"] == "lora"
