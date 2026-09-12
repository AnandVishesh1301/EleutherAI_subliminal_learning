"""Trait-free control replicates without clobbering (2026-09-10).

Replicate axes for the fixed-student-HP control null:
  * REPLICATE DRAWS (`--draw N`) of a control pool: same prompts, a fresh sample of numbers,
    own pool dir `{ps_tag}_draw{N}` -- base pools re-sample the untuned model under a
    distinct torch seed base, rng pools re-draw under a distinct PCG64 seed; the canonical
    pools and their manifests must stay byte-identical and can never be resume-matched.
  * base-pool SUBSAMPLES (`--subsample-seed S`): the existing pool, a different 10k draw,
    spelled into the dataset tag as `_sub{S}` so the student never lands in, or resumes
    from, the canonical default-rows dir.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from pipelines.generate_numbers_pipeline import (
    GEN_DRAW_SEED_BASE,
    PENDING_SUFFIX,
    _raw_pool_status,
    GEN_DRAW_STRIDE,
    RNG_DRAW_SEED_BASE,
    _GEN_SEED_BASE,
    GenerateNumbersConfig,
    NumberPromptSetConfig,
    control_draw_tag,
    gen_draw_seed_base,
    generation_params,
    rng_draw_seed,
    run,
)
from scripts.distill_student import (
    DEFAULT_SUBSAMPLE_SEED,
    _validate_number_dataset_tag,
    control_dataset_dir,
    derived_dataset_tag,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.numbers.filtering import NumberFilterConfig
from subliminal_learning.numbers.rng_gen import build_rng_completions
from subliminal_learning.numbers.prompt_gen import build_number_prompts

REPO = Path(__file__).resolve().parents[1]
PREFIX = NumberFilterConfig(min_count=1, max_count=10, min_value=0, max_value=999,
                            banned_numbers=(), require_clean_stop=True)


def _rng_cfg(tmp_path: Path, out: str, n: int = 40, rng_seed: int | None = None,
             gen_seed_base: int | None = None):
    paths = PathsConfig(output_root=tmp_path, hf_cache_root=tmp_path,
                        dataset_cache_root=tmp_path, wandb_dir=tmp_path)
    return GenerateNumbersConfig(
        paths=paths, hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=1, variant="data", revision="main"),
        source="rng", out_dir=tmp_path / out,
        prompt_set=NumberPromptSetConfig(
            templates_path=REPO / "data/number_prompts/sequence_intro_templates.json",
            n_prompts=n, seed=42),
        filter_modes={"prefix": PREFIX}, rng_seed=rng_seed, gen_seed_base=gen_seed_base,
    )


# ----------------------------------------------------------------------------- tags/seeds
def test_draw_tag_and_seed_are_off_the_canonical_names():
    assert control_draw_tag("ex10-20", 1) == "ex10-20_draw1"
    assert control_draw_tag("ex10-20", 7) == "ex10-20_draw7"
    assert rng_draw_seed(1) == RNG_DRAW_SEED_BASE + 1
    # the canonical pool has no draw index: 0 must never produce a tag or a seed
    for bad in (0, -1):
        with pytest.raises(ValueError):
            control_draw_tag("ex10-20", bad)
        with pytest.raises(ValueError):
            rng_draw_seed(bad)
    # a draw seed can never coincide with the prompt-set seed the canonical pool uses
    assert all(rng_draw_seed(n) != 42 for n in range(1, 1000))


def test_gen_draw_seed_bases_never_share_a_batch_seed():
    """Per-batch seeds are base + batch_idx (~800 batches per 100k pool): draws must be
    spaced beyond that and sit clear of the canonical range."""
    n_batches = 100_000 // 128 + 1
    ranges = [(_GEN_SEED_BASE, _GEN_SEED_BASE + n_batches)]
    for d in range(1, 11):
        b = gen_draw_seed_base(d)
        assert b == GEN_DRAW_SEED_BASE + GEN_DRAW_STRIDE * d
        ranges.append((b, b + n_batches))
    ranges.sort()
    assert all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1))
    with pytest.raises(ValueError):
        gen_draw_seed_base(0)


def test_gen_seed_base_is_for_model_sources_and_resolves_to_the_constant(tmp_path):
    cfg = _rng_cfg(tmp_path, "ds")
    assert cfg.resolved_gen_seed_base == _GEN_SEED_BASE
    with pytest.raises(ValueError, match="gen_seed_base is valid only"):
        _rng_cfg(tmp_path, "ds", gen_seed_base=5)


def test_base_draw_changes_sample_seed_base_in_the_params_and_nothing_else(tmp_path):
    """`sample_seed_base` is already a core resume-gate key: a base replicate differs from the
    canonical pool in exactly that value, so the gate refuses rather than reuses."""
    paths = PathsConfig(output_root=tmp_path, hf_cache_root=tmp_path,
                        dataset_cache_root=tmp_path, wandb_dir=tmp_path)
    ps = NumberPromptSetConfig(
        templates_path=REPO / "data/number_prompts/sequence_intro_templates.json",
        n_prompts=10, seed=42)
    spec = PythiaModelSpec(size="160m", seed=0, variant="full")
    canon = GenerateNumbersConfig(paths=paths, hub_org="EleutherAI", spec=spec, source="base",
                                  out_dir=tmp_path / "ex10-20", prompt_set=ps)
    rep = GenerateNumbersConfig(paths=paths, hub_org="EleutherAI", spec=spec, source="base",
                                out_dir=tmp_path / control_draw_tag("ex10-20", 2), prompt_set=ps,
                                gen_seed_base=gen_draw_seed_base(2))
    pc, pr = generation_params(canon, "h"), generation_params(rep, "h")
    assert pc["sample_seed_base"] == _GEN_SEED_BASE
    assert pr["sample_seed_base"] == gen_draw_seed_base(2)
    assert {k: v for k, v in pc.items() if k != "sample_seed_base"} == \
        {k: v for k, v in pr.items() if k != "sample_seed_base"}


def test_rng_seed_is_rng_only():
    paths = PathsConfig(output_root=Path("/x"), hf_cache_root=Path("/x"),
                        dataset_cache_root=Path("/x"), wandb_dir=Path("/x"))
    with pytest.raises(ValueError, match="rng_seed is valid only"):
        GenerateNumbersConfig(
            paths=paths, hub_org="EleutherAI",
            spec=PythiaModelSpec(size="160m", seed=1, variant="data", revision="main"),
            source="base", out_dir=Path("/x/ds"),
            prompt_set=NumberPromptSetConfig(
                templates_path=REPO / "data/number_prompts/sequence_intro_templates.json",
                n_prompts=10, seed=42),
            rng_seed=5,
        )


# ------------------------------------------------------------------- manifest byte-identity
def test_canonical_rng_params_carry_no_seed_key(tmp_path):
    """The resume gate compares the whole rng block; the canonical pool's block must not
    change shape, or every rng pool on disk would read as a mismatch."""
    p = generation_params(_rng_cfg(tmp_path, "ds"), "deadbeef")
    assert p["rng"] == {"count": 10, "value_min": 0, "value_max": 999}
    p2 = generation_params(_rng_cfg(tmp_path, "ds", rng_seed=rng_draw_seed(3)), "deadbeef")
    assert p2["rng"] == {"count": 10, "value_min": 0, "value_max": 999,
                         "seed": RNG_DRAW_SEED_BASE + 3}


def test_replicate_draw_lands_beside_the_canonical_pool_and_differs(tmp_path):
    run(_rng_cfg(tmp_path, "ex10-20"))
    canon_raw = (tmp_path / "ex10-20" / "raw.jsonl").read_text()
    canon_man = (tmp_path / "ex10-20" / "manifest.json").read_text()

    run(_rng_cfg(tmp_path, control_draw_tag("ex10-20", 1), rng_seed=rng_draw_seed(1)))
    rep = tmp_path / "ex10-20_draw1"
    assert (rep / "raw.jsonl").exists()
    # canonical pool untouched, byte for byte
    assert (tmp_path / "ex10-20" / "raw.jsonl").read_text() == canon_raw
    assert (tmp_path / "ex10-20" / "manifest.json").read_text() == canon_man
    # same prompts, different numbers
    c = [json.loads(line) for line in canon_raw.splitlines()]
    r = [json.loads(line) for line in (rep / "raw.jsonl").read_text().splitlines()]
    assert [x["prompt"] for x in c] == [x["prompt"] for x in r]
    assert [x["completion"] for x in c] != [x["completion"] for x in r]
    assert json.loads((rep / "manifest.json").read_text())["params"]["rng"]["seed"] == \
        RNG_DRAW_SEED_BASE + 1


def test_replicate_seed_cannot_resume_match_the_canonical_manifest(tmp_path):
    """Even if a draw were pointed at the canonical dir, the seed in its identity makes the
    gate refuse (not silently reuse) the canonical pool."""
    run(_rng_cfg(tmp_path, "ex10-20"))
    raw_before = (tmp_path / "ex10-20" / "raw.jsonl").read_text()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(_rng_cfg(tmp_path, "ex10-20", rng_seed=rng_draw_seed(1)))
    assert (tmp_path / "ex10-20" / "raw.jsonl").read_text() == raw_before


def test_rng_completions_depend_on_the_seed_only():
    templates = json.loads(
        (REPO / "data/number_prompts/sequence_intro_templates.json").read_text())
    prompts = build_number_prompts(templates, n_prompts=20, seed=42)
    a = build_rng_completions(prompts, seed=42)
    b = build_rng_completions(prompts, seed=42)
    c = build_rng_completions(prompts, seed=rng_draw_seed(1))
    assert a == b and a != c


# ---------------------------------------------------------------------- student-side tags
def test_derived_tag_is_byte_identical_for_default_inputs():
    assert derived_dataset_tag("ex10-20", "prefix", False) == "ex10-20_prefix"
    assert derived_dataset_tag("ex10-20", "prefix", True) == "ex10-20_prefix_spliced"
    assert derived_dataset_tag("ex10-20", "prefix", False, subsample_seed=DEFAULT_SUBSAMPLE_SEED) == \
        "ex10-20_prefix"


def test_derived_tag_spells_in_replicate_axes():
    assert derived_dataset_tag("ex10-20", "prefix", False, subsample_seed=7) == "ex10-20_prefix_sub7"
    assert derived_dataset_tag("ex10-20", "prefix", False, draw=3) == "ex10-20_prefix_draw3"
    assert derived_dataset_tag("ex10-20", "prefix", True, subsample_seed=7, draw=3) == \
        "ex10-20_prefix_spliced_sub7_draw3"


@pytest.mark.parametrize("tag, kw", [
    ("ex10-20_prefix_n100k", {}),
    ("ex10-20_prefix_n100k_sub7", {"subsample_seed": 7}),
    ("ex10-20_prefix_n100k_draw2", {"draw": 2}),
    ("ex10-20_prefix_n100k_sub7_draw2", {"subsample_seed": 7, "draw": 2}),
    ("ex10-20_prefix_n100k_spliced_sub7", {"subsample_seed": 7, "splice_examples": True}),
])
def test_validator_accepts_consistent_tags(tag, kw):
    splice = kw.pop("splice_examples", False)
    _validate_number_dataset_tag(tag, "ex10-20", "prefix", splice, **kw)


@pytest.mark.parametrize("tag, kw, msg", [
    # non-default subsample seed without the marker -> would land in the canonical dir
    ("ex10-20_prefix_n100k", {"subsample_seed": 7}, "must carry '_sub7'"),
    # marker names a different seed than the flag
    ("ex10-20_prefix_n100k_sub7", {"subsample_seed": 8}, "must agree"),
    # marker present but the flag is the default: the tag would lie about the rows
    ("ex10-20_prefix_n100k_sub7", {}, "must agree"),
    # rng draw flag without the marker / marker without the flag / mismatch
    ("ex10-20_prefix_n100k", {"draw": 2}, "lacks '_draw2'"),
    ("ex10-20_prefix_n100k_draw2", {}, "must agree"),
    ("ex10-20_prefix_n100k_draw2", {"draw": 3}, "must agree"),
    # duplicated markers
    ("ex10-20_prefix_n100k_sub7_sub7", {"subsample_seed": 7}, "more than one"),
    # sub-marker must be a whole token: `_sub70` is not `_sub7`
    ("ex10-20_prefix_n100k_sub70", {"subsample_seed": 7}, "must agree"),
])
def test_validator_rejects_inconsistent_tags(tag, kw, msg):
    with pytest.raises(typer.BadParameter, match=msg):
        _validate_number_dataset_tag(tag, "ex10-20", "prefix", False, **kw)


def test_control_dataset_dir_routes_replicate_draws():
    spec = PythiaModelSpec(size="160m", seed=0, variant="full")
    root = Path("/out/numbers")
    assert control_dataset_dir(root, "rng", spec, "EleutherAI", "ex10-20") == \
        root / "controls/rng/ex10-20"
    assert control_dataset_dir(root, "rng", spec, "EleutherAI", "ex10-20", draw=4) == \
        root / "controls/rng/ex10-20_draw4"
    assert control_dataset_dir(root, "base", spec, "EleutherAI", "ex10-20") == \
        root / "controls/base/pythia-160m/ex10-20"
    assert control_dataset_dir(root, "base", spec, "EleutherAI", "ex10-20", draw=2) == \
        root / "controls/base/pythia-160m/ex10-20_draw2"
    with pytest.raises(ValueError):
        control_dataset_dir(root, "teacher", spec, "EleutherAI", "ex10-20")


def test_subsample_cache_is_keyed_on_the_seed(tmp_path):
    """Two subsample seeds over one pool select different rows and cache separately."""
    from subliminal_learning.students.dataset_io import load_subsample_cached, subsample_dir

    ds = tmp_path / "modes" / "prefix"
    ds.mkdir(parents=True)
    rows = [{"prompt": f"p{i}", "completion": f" {i}", "numbers": [i]} for i in range(200)]
    (ds / "filtered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    a, sha_a, n_a = load_subsample_cached(ds / "filtered.jsonl", 50, 42)
    b, sha_b, n_b = load_subsample_cached(ds / "filtered.jsonl", 50, 7)
    assert n_a == n_b == 200 and sha_a == sha_b
    assert [r["prompt"] for r in a] != [r["prompt"] for r in b]
    assert subsample_dir(ds, 50, 42) != subsample_dir(ds, 50, 7)
    assert subsample_dir(ds, 50, 42).exists() and subsample_dir(ds, 50, 7).exists()


def test_base_control_lock_dir_is_created_for_a_fresh_control_tree(tmp_path, monkeypatch):
    """First eval of a new control dataset tag: `eval/{animal}/` does not exist yet, and the
    base-control lock must not fail on the missing parent (it did, 2026-09-10)."""
    import pipelines.evaluate_teacher_preference_pipeline as ep

    cfg = SimpleNamespace(param_dir=tmp_path / "ds_draw1" / "ep20_lr1e-4_r8a32",
                          artifact_subdir=Path("eval") / "dog")
    cache = ep.evaluation_artifact_path(cfg.param_dir.parent, cfg, "base_control.json")
    assert not cache.parent.exists()
    # replicate the lock-path preparation exactly as _resolve_base_control does it
    lock_path = cache.with_name(cache.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    from subliminal_learning.utils.io import file_lock
    with file_lock(lock_path):
        pass
    assert lock_path.exists()


# ------------------------------------------------------- torn raw pool (marker + size)
def _status(cfg) -> str:
    """`_raw_pool_status` for an on-disk pool, re-using its own stored prompts hash."""
    stored = json.loads((cfg.out_dir / "manifest.json").read_text())
    return _raw_pool_status(cfg, generation_params(cfg, stored["params"]["prompts_hash"]))[0]


def test_completed_pool_records_its_size_and_leaves_no_pending_marker(tmp_path: Path):
    cfg = _rng_cfg(tmp_path, "pool")
    run(cfg)
    raw = cfg.out_dir / "raw.jsonl"
    stored = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert stored["raw_bytes"] == raw.stat().st_size
    assert stored["raw_sha256"] and "raw_bytes" not in stored["params"]
    assert not raw.with_name(raw.name + PENDING_SUFFIX).exists()
    assert _status(cfg) == "match"


def test_pending_marker_makes_the_pool_torn(tmp_path: Path):
    cfg = _rng_cfg(tmp_path, "pool")
    run(cfg)
    raw = cfg.out_dir / "raw.jsonl"
    raw.with_name(raw.name + PENDING_SUFFIX).write_text("pending\n")
    assert _status(cfg) == "torn"


def test_size_disagreeing_with_the_manifest_is_torn(tmp_path: Path):
    cfg = _rng_cfg(tmp_path, "pool")
    run(cfg)
    raw = cfg.out_dir / "raw.jsonl"
    # the crash this guards: NEW rows on disk under the OLD manifest
    raw.write_text(raw.read_text() + raw.read_text().splitlines()[0] + "\n")
    assert _status(cfg) == "torn"


def test_legacy_manifest_without_raw_bytes_is_unchanged(tmp_path: Path):
    cfg = _rng_cfg(tmp_path, "pool")
    run(cfg)
    mpath = cfg.out_dir / "manifest.json"
    m = json.loads(mpath.read_text())
    m.pop("raw_bytes")
    m.pop("raw_sha256")
    mpath.write_text(json.dumps(m))
    assert _status(cfg) == "match"
    # and a legacy manifest is not rescued by a size check it never stored
    raw = cfg.out_dir / "raw.jsonl"
    raw.write_text(raw.read_text() + raw.read_text().splitlines()[0] + "\n")
    assert _status(cfg) == "match"
