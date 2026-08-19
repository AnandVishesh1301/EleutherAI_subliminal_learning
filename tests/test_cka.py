"""Port validation for the minibatch CKA + pool eligibility rules."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_spec_from_base_name_validates_against_the_registry():
    # Writer and reader share one grammar: names that no repo could satisfy fail here
    # rather than 404ing mid-run.
    from subliminal_learning.cka.model_refs import spec_from_base_name
    from subliminal_learning.models.registry import PythiaModelSpec

    assert spec_from_base_name("pythia-160m-seed2") == PythiaModelSpec("160m", 2, "full")
    assert spec_from_base_name("EleutherAI/pythia-410m-seed4@step3000") == PythiaModelSpec(
        "410m", 4, "full", "step3000"
    )
    for bad in ("pythia-160m-seed99", "pythia-1b-seed5"):
        with pytest.raises(ValueError, match="bad base name"):
            spec_from_base_name(bad)


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


# ---- layout helpers + student-mode resolution -------------------------------


def test_cka_dir_builders():
    from pathlib import Path

    from subliminal_learning.cka.layout import (
        base_cka_dir,
        pair_cka_dir,
        vs_teacher_dir,
    )

    run_dir = Path("/o/students/dog/pythia-160m/tc/tag/ns/hp/ftseed2")
    assert vs_teacher_dir(run_dir, "numbers") == run_dir / "cka" / "vs_teacher__numbers"
    assert vs_teacher_dir(run_dir, "numbers_k200") == (
        run_dir / "cka" / "vs_teacher__numbers_k200"
    )
    root = Path("/o")
    assert base_cka_dir(root, "pile") == root / "cka" / "base" / "pile"
    assert pair_cka_dir(root, "t__vs__s") == root / "cka" / "pairs" / "t__vs__s"


def _write_student_manifest(run_dir, manifest):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest))


def test_resolve_teacher_of_treatment(tmp_path):
    from subliminal_learning.cka.layout import resolve_teacher_of

    teacher_run = _make_lora_run(
        tmp_path, "teachers/dog/pythia-160m/ep2_lr5e-4_r32a64/ftseed1"
    )
    student_run = tmp_path / "students/dog/pythia-160m-data-seed1/tc/hp/ftseed2"
    _write_student_manifest(
        student_run,
        {
            "source": "teacher",
            "animal": "dog",
            "numbers_base_model_repo": "EleutherAI/pythia-160m",
            "teacher_param_tag": "ep2_lr5e-4_r32a64",
            "teacher_ftseed": 1,
        },
    )
    ref = resolve_teacher_of(student_run, tmp_path, "EleutherAI")
    assert ref.kind == "finetuned"
    assert ref.run_dir == teacher_run
    assert ref.checkpoint_dir == teacher_run / "checkpoints"


def test_resolve_teacher_of_missing_teacher_raises(tmp_path):
    from subliminal_learning.cka.layout import resolve_teacher_of

    student_run = tmp_path / "students/dog/pythia-160m-data-seed1/tc/hp/ftseed2"
    _write_student_manifest(
        student_run,
        {
            "source": "teacher",
            "animal": "dog",
            "numbers_base_model_repo": "EleutherAI/pythia-160m",
            "teacher_param_tag": "ep2_lr5e-4_r32a64",
            "teacher_ftseed": 1,
        },
    )
    with pytest.raises(ValueError, match="teacher run dir"):
        resolve_teacher_of(student_run, tmp_path, "EleutherAI")


def test_resolve_teacher_of_base_control(tmp_path):
    from subliminal_learning.cka.layout import resolve_teacher_of
    from subliminal_learning.models.registry import PythiaModelSpec

    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(
        student_run,
        {
            "source": "base",
            "animal": "dog",
            "numbers_base_model_repo": "EleutherAI/pythia-160m-data-seed1",
        },
    )
    ref = resolve_teacher_of(student_run, tmp_path, "EleutherAI")
    assert ref.kind == "base"
    assert ref.spec == PythiaModelSpec("160m", 1, "data")
    assert ref.base_repo == "EleutherAI/pythia-160m-data-seed1"


def test_resolve_teacher_of_rng_control_raises(tmp_path):
    from subliminal_learning.cka.layout import resolve_teacher_of

    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(student_run, {"source": "rng", "animal": "dog"})
    with pytest.raises(ValueError, match="rng"):
        resolve_teacher_of(student_run, tmp_path, "EleutherAI")


def test_resolve_teacher_of_missing_manifest_raises(tmp_path):
    from subliminal_learning.cka.layout import resolve_teacher_of

    with pytest.raises(ValueError, match="train_manifest.json"):
        resolve_teacher_of(tmp_path / "nope", tmp_path, "EleutherAI")


def test_student_numbers_pool(tmp_path):
    from subliminal_learning.cka.layout import student_numbers_pool

    ds_dir = "/mnt/data/numbers/dog/pythia-160m/tag/ftseed1/ex/modes/prefix"
    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(
        student_run,
        {
            "source": "teacher",
            "number_dataset_dir": ds_dir,
            "subsample_size": 10000,
            "subsample_seed": 42,
        },
    )
    pool, size, seed = student_numbers_pool(student_run)
    assert pool == Path(ds_dir) / "filtered.jsonl"
    assert size == 10000 and seed == 42


def test_student_numbers_pool_missing_dir_raises(tmp_path):
    from subliminal_learning.cka.layout import student_numbers_pool

    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(student_run, {"source": "rng"})
    with pytest.raises(ValueError, match="number_dataset_dir"):
        student_numbers_pool(student_run)


def _make_pool_file(tmp_path, text="a\nb\n"):
    from subliminal_learning.utils.io import file_sha256

    pool = tmp_path / "pool" / "filtered.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text(text)
    return pool, file_sha256(pool)


def test_verify_student_numbers_pool_matches(tmp_path):
    from subliminal_learning.cka.layout import verify_student_numbers_pool

    pool, sha = _make_pool_file(tmp_path)
    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(student_run, {"source": "teacher", "filtered_sha256": sha})
    verify_student_numbers_pool(student_run, pool)  # no raise


def test_verify_student_numbers_pool_mismatch_raises(tmp_path):
    from subliminal_learning.cka.layout import verify_student_numbers_pool

    pool, sha = _make_pool_file(tmp_path)
    stale = "0" * 64
    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(student_run, {"source": "teacher", "filtered_sha256": stale})
    with pytest.raises(ValueError) as exc:
        verify_student_numbers_pool(student_run, pool)
    msg = str(exc.value)
    assert sha in msg and stale in msg and str(pool) in msg


def test_verify_student_numbers_pool_missing_file_raises(tmp_path):
    from subliminal_learning.cka.layout import verify_student_numbers_pool

    pool = tmp_path / "pool" / "filtered.jsonl"
    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(student_run, {"source": "teacher", "filtered_sha256": "x"})
    with pytest.raises(ValueError) as exc:
        verify_student_numbers_pool(student_run, pool)
    assert "does not exist" in str(exc.value) and str(pool) in str(exc.value)


def test_verify_student_numbers_pool_no_sha_warns(tmp_path):
    from subliminal_learning.cka.layout import verify_student_numbers_pool

    pool, _ = _make_pool_file(tmp_path)
    student_run = tmp_path / "students/dog/pythia-160m/tc/hp/ftseed2"
    _write_student_manifest(student_run, {"source": "teacher"})  # no filtered_sha256
    verify_student_numbers_pool(student_run, pool)  # warns, does not raise


def test_student_out_dir_derivation():
    from scripts.compute_cka import _student_out_dir

    run_dir = Path(
        "/o/students/dog/pythia-160m-data-seed1/teacher-pythia-160m/"
        "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k/ep20_lr5e-5_r8a32/ftseed2"
    )
    canonical = {"pos": 1, "k": 100, "m": 256, "pool": None, "hold": 512,
                 "pc": "1,3,5", "sub": 10_000, "seed": 0}
    # default (all-canonical) numbers run -> bare "numbers" tag
    assert _student_out_dir(run_dir, "numbers", canonical) == (
        run_dir / "cka" / "vs_teacher__numbers"
    )
    # sub is manifest-forced: even a non-default subsample does NOT enter the tag
    assert _student_out_dir(run_dir, "numbers", {**canonical, "sub": 500}) == (
        run_dir / "cka" / "vs_teacher__numbers"
    )
    # a non-sub axis still appends its suffix
    assert _student_out_dir(run_dir, "numbers", {**canonical, "k": 200}) == (
        run_dir / "cka" / "vs_teacher__numbers_k200"
    )
    assert _student_out_dir(run_dir, "pile", canonical) == (
        run_dir / "cka" / "vs_teacher__pile"
    )


def _invoke_cli(args):
    import typer
    from typer.testing import CliRunner

    from scripts.compute_cka import main

    app = typer.Typer()
    app.command()(main)
    return CliRunner().invoke(app, args)


def test_cli_student_mutual_exclusion():
    # --student + explicit operands is rejected
    r = _invoke_cli([
        "--condition", "numbers", "--student", "students/x/ftseed1",
        "--model-a", "original", "--model-b", "data-seed1",
    ])
    assert r.exit_code != 0
    # --student + non-default --base-models is rejected
    r2 = _invoke_cli([
        "--condition", "pile", "--student", "students/x/ftseed1",
        "--base-models", "original,data-seed1",
    ])
    assert r2.exit_code != 0
    # --student + --numbers-path is rejected (pool comes from the manifest)
    r3 = _invoke_cli([
        "--condition", "numbers", "--student", "students/x/ftseed1",
        "--numbers-path", "/some/filtered.jsonl",
    ])
    assert r3.exit_code != 0


def test_save_outputs_no_plots(tmp_path):
    import torch as _torch

    from pipelines.cka_pipeline import CkaConfig, CkaPair, _save_outputs
    from subliminal_learning.cka.minibatch import MinibatchCKA
    from subliminal_learning.cka.pool_io import SequencePool

    def _acc():
        a = MinibatchCKA(1, across_models=True)
        g = _torch.Generator().manual_seed(0)
        x = _torch.randn(8, 4, generator=g)
        a.update_state_across_models([x], [x])
        return a

    pool = SequencePool([[1, 2, 3]], [[1]], "x")
    pairs = [CkaPair("teacher__student", "primary", "teacher", "student")]
    variants = ["raw"]
    accs = {("teacher__student", "raw"): _acc()}
    records = {("teacher__student", "raw"): [(1, accs[("teacher__student", "raw")].result().numpy())]}
    identity = {"numbers_sha256": None, "pool_token_sha256": "x"}
    meta_models = {"teacher": {}, "student": {}}

    cfg = CkaConfig(condition="numbers", out_dir=tmp_path / "off", save_plots=False)
    out = _save_outputs(cfg, meta_models, pool, pairs, variants, accs, records, 1, identity)
    assert not list(out.glob("*.png"))

    cfg_on = CkaConfig(condition="numbers", out_dir=tmp_path / "on", save_plots=True)
    out_on = _save_outputs(cfg_on, meta_models, pool, pairs, variants, accs, records, 1, identity)
    assert list(out_on.glob("*.png"))


# ---- K-student batched "vs teacher" mode ------------------------------------


def _make_student(root, rel, base, extra):
    """A finetuned student run dir: LoRA checkpoint + a manifest carrying both the base
    model (for resolve_ref) and the given CKA/source fields (for the batch validation)."""
    run_dir = _make_lora_run(root, rel, base=base)
    manifest = {"base_model_repo": base, "base_model": base.rsplit("/", 1)[-1], **extra}
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest))
    return run_dir


def _base_student_extra(teacher_repo="EleutherAI/pythia-160m", sha="abc"):
    return {
        "source": "base",
        "numbers_base_model_repo": teacher_repo,
        "number_dataset_dir": "/mnt/data/numbers/dog/pythia-160m/tag/ftseed1/ex/modes/prefix",
        "filtered_sha256": sha,
        "subsample_size": 10000,
        "subsample_seed": 42,
    }


def test_validate_student_batch_pass_different_student_bases(tmp_path):
    from subliminal_learning.cka.layout import validate_student_batch

    extra = _base_student_extra()
    s0 = _make_student(
        tmp_path, "students/dog/pythia-160m-data-seed1/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-data-seed1", extra,
    )
    s1 = _make_student(
        tmp_path, "students/dog/pythia-160m-weight-seed2/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-weight-seed2", extra,
    )
    teacher, srefs = validate_student_batch([s0, s1], tmp_path, "EleutherAI")
    assert teacher.kind == "base" and teacher.base_repo == "EleutherAI/pythia-160m"
    # different STUDENT bases are allowed
    assert srefs[0].spec.variant == "data" and srefs[1].spec.variant == "weight"


def test_validate_student_batch_teacher_mismatch(tmp_path):
    from subliminal_learning.cka.layout import validate_student_batch

    s0 = _make_student(
        tmp_path, "students/dog/pythia-160m-data-seed1/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-data-seed1", _base_student_extra("EleutherAI/pythia-160m"),
    )
    s1 = _make_student(
        tmp_path, "students/dog/pythia-160m-data-seed2/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-data-seed2",
        _base_student_extra("EleutherAI/pythia-160m-data-seed1"),
    )
    with pytest.raises(ValueError, match="different teacher"):
        validate_student_batch([s0, s1], tmp_path, "EleutherAI")


def test_validate_student_batch_pool_sha_mismatch(tmp_path):
    from subliminal_learning.cka.layout import validate_student_batch

    s0 = _make_student(
        tmp_path, "students/dog/pythia-160m-data-seed1/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-data-seed1", _base_student_extra(sha="aaa"),
    )
    s1 = _make_student(
        tmp_path, "students/dog/pythia-160m-weight-seed2/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-weight-seed2", _base_student_extra(sha="bbb"),
    )
    with pytest.raises(ValueError, match="filtered_sha256"):
        validate_student_batch([s0, s1], tmp_path, "EleutherAI")


def test_validate_student_batch_duplicate(tmp_path):
    from subliminal_learning.cka.layout import validate_student_batch

    s0 = _make_student(
        tmp_path, "students/dog/pythia-160m-data-seed1/tc/hp/ftseed1",
        "EleutherAI/pythia-160m-data-seed1", _base_student_extra(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_student_batch([s0, s0], tmp_path, "EleutherAI")


def test_student_batch_identity_equals_single_run():
    """A batched student's stored identity must be the SAME 2-operand identity a single
    run_explicit(teacher, that student) would store (batched/single interop)."""
    from pipelines.cka_pipeline import CkaConfig, _run_identity, student_batch_identity

    cfg = CkaConfig(condition="pile", out_dir=Path("/tmp/x"))
    tprov = {"ref": "teacher", "checkpoint_hash": "aa"}
    sprov = {"ref": "student", "checkpoint_hash": "bb"}
    variants = ["raw", "meanpool"]
    batched = student_batch_identity(cfg, tprov, sprov, "poolsha", variants)
    single = _run_identity(cfg, [tprov, sprov], "poolsha", variants)
    assert batched == single
    assert batched["models"] == [tprov, sprov]  # 2-operand, not K-operand


def test_build_student_batch_pairs():
    from pipelines.cka_pipeline import (
        build_explicit_pairs,
        build_student_batch_pairs,
        student_save_pairs,
    )

    pairs = build_student_batch_pairs(3)
    groups = {}
    for p in pairs:
        groups.setdefault(p.group, []).append(p)
    assert len(groups["ceiling"]) == 1  # teacher self shared across all students
    assert len(groups["primary"]) == 3
    assert len(groups["self"]) == 3
    assert len(groups["floor"]) == 3
    assert all(p.permute_b for p in groups["floor"])
    assert len({p.label for p in pairs}) == len(pairs)  # labels unique
    # K=1's SAVED pairs equal today's single-student explicit output (modulo naming)
    assert student_save_pairs() == build_explicit_pairs(
        "teacher", "student", same_model=False
    )


def test_pending_student_indices_drops_complete(tmp_path):
    from pipelines.cka_pipeline import CKA_MANIFEST, _pending_student_indices

    d0, d1 = tmp_path / "s0", tmp_path / "s1"
    d0.mkdir()
    d1.mkdir()
    id0 = {"condition": "numbers", "seed": 0, "pool_token_sha256": "a"}
    id1 = {"condition": "numbers", "seed": 0, "pool_token_sha256": "b"}
    # s0 already has a complete run under its own identity -> it must be dropped.
    (d0 / CKA_MANIFEST).write_text(
        json.dumps({"workflow": "cka", "run_identity": id0})
    )
    assert _pending_student_indices([d0, d1], [id0, id1], force=False) == [1]


def test_save_students_splits_per_dir(tmp_path):
    import torch as _torch

    from pipelines.cka_pipeline import CkaConfig, _save_students
    from subliminal_learning.cka.minibatch import MinibatchCKA
    from subliminal_learning.cka.pool_io import SequencePool

    def _acc():
        a = MinibatchCKA(1, across_models=True)
        g = _torch.Generator().manual_seed(0)
        x = _torch.randn(8, 4, generator=g)
        a.update_state_across_models([x], [x])
        return a

    variants = ["raw"]
    internal = [
        "self__teacher",
        "teacher__student0", "self__student0", "floor__teacher__student0",
        "teacher__student1", "self__student1", "floor__teacher__student1",
    ]
    accs = {(lbl, "raw"): _acc() for lbl in internal}
    records = {
        (lbl, "raw"): [(1, accs[(lbl, "raw")].result().numpy())] for lbl in internal
    }
    pool = SequencePool([[1, 2, 3]], [[1]], "x")
    tprov = {"ref": "teacher"}
    identity = {"numbers_sha256": None, "pool_token_sha256": "x"}
    save_specs = [
        (tmp_path / "s0", "student0", {"ref": "student0"}, identity),
        (tmp_path / "s1", "student1", {"ref": "student1"}, identity),
    ]
    cfg = CkaConfig(condition="numbers", out_dir=tmp_path / "placeholder", save_plots=False)
    outs = _save_students(cfg, tprov, pool, variants, accs, records, 1, save_specs)
    assert outs == [tmp_path / "s0", tmp_path / "s1"]

    expected = {
        "cka__teacher__student__raw.npy", "cka__self__teacher__raw.npy",
        "cka__self__student__raw.npy", "cka__floor__teacher__student__raw.npy",
    }
    for out in outs:
        assert {p.name for p in out.glob("cka__*.npy")} == expected
        assert {p.name for p in out.glob("convergence__*.npy")} == {
            n.replace("cka__", "convergence__") for n in expected
        }
        assert (out / "meta.json").exists() and (out / "summary.json").exists()
        assert (out / "cka_manifest.json").exists()
    # each dir carries its OWN student provenance
    m0 = json.loads((tmp_path / "s0" / "meta.json").read_text())
    m1 = json.loads((tmp_path / "s1" / "meta.json").read_text())
    assert m0["models"]["student"] == {"ref": "student0"}
    assert m1["models"]["student"] == {"ref": "student1"}
    assert m0["models"]["teacher"] == tprov


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
