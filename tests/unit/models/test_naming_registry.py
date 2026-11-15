"""`models.naming`: every spelling of a base model round-trips through one spec.

Call sites routed through `models.naming` accept every spelling and keep their fallbacks.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.coherence.probe_families import family_for_base_repo
from subliminal_learning.coherence.probe_io import anchored_dead_cut
from subliminal_learning.models.datadecide import DataDecideSpec
from subliminal_learning.models.naming import (
    base_dir_name,
    base_ref,
    base_spec,
    base_token,
    describe,
    family_of,
    same_base,
)
from subliminal_learning.models.registry import (
    LOCAL_PYTHIA_FAMILIES,
    PythiaModelSpec,
    local_pythia_family,
    local_model_dir,
    resolve_hf_repo_id,
)
from subliminal_learning.models.registry import ExternalModelSpec


# -------------------- from test_models_naming.py --------------------

DD1 = "DataDecide-dolma1_7-1B-seed-default"

DD2 = "DataDecide-dolma1_7-1B-seed-large-aux-2"

DD3 = "DataDecide-dolma1_7-1B-seed-large-aux-3"

DD1_REF = "allenai/DataDecide-dolma1_7-1B@step69369-seed-default"

DD2_REF = "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"

# (any input spelling, token, dir name, ref)
CASES = [
    # --- DataDecide: token / dir name / ref all name one spec
    ("dd-1b", "dd-1b-seed1", DD1, DD1_REF),
    ("dd-1b-seed1", "dd-1b-seed1", DD1, DD1_REF),
    ("dd-1b-seed2", "dd-1b-seed2", DD2, DD2_REF),
    (DD2, "dd-1b-seed2", DD2, DD2_REF),
    (DD3, "dd-1b-seed3", DD3, "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-3"),
    (DD1_REF, "dd-1b-seed1", DD1, DD1_REF),
    (DD2_REF, "dd-1b-seed2", DD2, DD2_REF),
    # step-pinned DataDecide keeps the pin in every spelling
    ("dd-1b-seed2@step2500", "dd-1b-seed2@step2500", DD2 + "@step2500",
     "allenai/DataDecide-dolma1_7-1B@step2500-seed-large-aux-2"),
    (DD2 + "@step2500", "dd-1b-seed2@step2500", DD2 + "@step2500",
     "allenai/DataDecide-dolma1_7-1B@step2500-seed-large-aux-2"),
    # --- external bases: alias / basename / repo id
    ("qwen25-1.5b", "qwen25-1.5b", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("Qwen2.5-1.5B", "qwen25-1.5b", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-1.5B", "qwen25-1.5b", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("llama3-1b", "llama3-1b", "Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("google/gemma-3-1b-pt", "gemma3-1b", "gemma-3-1b-pt", "google/gemma-3-1b-pt"),
    ("qwen25-1.5b@main", "qwen25-1.5b@main", "Qwen2.5-1.5B@main", "Qwen/Qwen2.5-1.5B@main"),
    ("qwen2.5-1.5b", "qwen25-1.5b", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),     # lowercased basename
    ("gemma2-2b", "gemma2-2b", "gemma-2-2b", "google/gemma-2-2b"),
    ("google/gemma-2-9b", "gemma2-9b", "gemma-2-9b", "google/gemma-2-9b"),
    # --- Pythia: token == dir name for this family; refs carry the org
    ("pythia-160m", "pythia-160m", "pythia-160m", "EleutherAI/pythia-160m"),
    ("original", "pythia-160m", "pythia-160m", "EleutherAI/pythia-160m"),
    ("full-seed0", "pythia-160m", "pythia-160m", "EleutherAI/pythia-160m"),
    ("EleutherAI/pythia-160m", "pythia-160m", "pythia-160m", "EleutherAI/pythia-160m"),
    ("pythia-410m", "pythia-410m", "pythia-410m", "EleutherAI/pythia-410m"),
    ("pythia-14m", "pythia-14m", "pythia-14m", "EleutherAI/pythia-14m"),
    ("pythia-70m", "pythia-70m", "pythia-70m", "EleutherAI/pythia-70m"),
    ("weight-seed2", "weight-seed2", "pythia-160m-weight-seed2", "EleutherAI/pythia-160m-weight-seed2"),
    ("pythia-160m-seed1", "pythia-160m-seed1", "pythia-160m-seed1", "EleutherAI/pythia-160m-seed1"),
    ("full-seed1", "pythia-160m-seed1", "pythia-160m-seed1", "EleutherAI/pythia-160m-seed1"),
    ("pythia-410m-seed3", "pythia-410m-seed3", "pythia-410m-seed3", "EleutherAI/pythia-410m-seed3"),
    ("data-seed1", "data-seed1", "pythia-160m-data-seed1", "EleutherAI/pythia-160m-data-seed1"),
    ("pythia-160m-data-seed1", "data-seed1", "pythia-160m-data-seed1", "EleutherAI/pythia-160m-data-seed1"),
    ("EleutherAI/pythia-160m-weight-seed3", "weight-seed3", "pythia-160m-weight-seed3",
     "EleutherAI/pythia-160m-weight-seed3"),
    ("weightc-seed2", "weightc-seed2", "pythia-160m-weightc-seed2", "local/pythia-160m-weightc-seed2"),
    ("pythia-160m-weightc-seed2", "weightc-seed2", "pythia-160m-weightc-seed2",
     "local/pythia-160m-weightc-seed2"),
    ("local/pythia-160m-weightc-seed2", "weightc-seed2", "pythia-160m-weightc-seed2",
     "local/pythia-160m-weightc-seed2"),
    ("pilerepro-seed1234", "pilerepro-seed1234", "pythia-160m-pilerepro-seed1234",
     "local/pythia-160m-pilerepro-seed1234"),
    ("pythia-160m-pilerepro-seed1234", "pilerepro-seed1234", "pythia-160m-pilerepro-seed1234",
     "local/pythia-160m-pilerepro-seed1234"),
    ("local/pythia-160m-dclm-seed1234", "dclm-seed1234", "pythia-160m-dclm-seed1234",
     "local/pythia-160m-dclm-seed1234"),
    ("dclm-seed1", "dclm-seed1", "pythia-160m-dclm-seed1", "local/pythia-160m-dclm-seed1"),
    # a 4-digit local seed with a pinned step (the dir-name inverter, not the token grammar)
    ("pythia-160m-pilerepro-seed1234@step75000", "pilerepro-seed1234@step75000",
     "pythia-160m-pilerepro-seed1234@step75000",
     "local/pythia-160m-pilerepro-seed1234@step75000"),
    # revision-pinned Pythia
    ("data-seed1@step3000", "data-seed1@step3000", "pythia-160m-data-seed1@step3000",
     "EleutherAI/pythia-160m-data-seed1@step3000"),
    ("pythia-160m@step3000", "pythia-160m@step3000", "pythia-160m@step3000",
     "EleutherAI/pythia-160m@step3000"),
    ("EleutherAI/pythia-160m-data-seed1@step3000", "data-seed1@step3000",
     "pythia-160m-data-seed1@step3000", "EleutherAI/pythia-160m-data-seed1@step3000"),
]

@pytest.mark.parametrize("value,token,dirname,ref", CASES)
def test_every_spelling_maps_to_the_same_three(value, token, dirname, ref):
    assert base_token(value) == token
    assert base_dir_name(value) == dirname
    assert base_ref(value) == ref

@pytest.mark.parametrize("value,token,dirname,ref", CASES)
def test_converters_are_idempotent_and_agree(value, token, dirname, ref):
    # each output spelling is itself accepted and maps back to the same spec
    spec = base_spec(value)
    for spelling in (token, dirname, ref):
        assert base_spec(spelling) == spec, spelling
    assert base_token(token) == token
    assert base_dir_name(dirname) == dirname
    assert base_ref(ref) == ref

def test_spec_objects_pass_through():
    for spec in (PythiaModelSpec(size="160m"), DataDecideSpec(seed=2),
                 ExternalModelSpec("Qwen/Qwen2.5-1.5B")):
        assert base_spec(spec) is spec
    assert base_dir_name(DataDecideSpec(seed=3)) == DD3
    assert base_token(PythiaModelSpec(size="160m", seed=2, variant="data")) == "data-seed2"

def test_same_base_across_spellings():
    assert same_base("dd-1b-seed2", DD2)
    assert same_base(DD2_REF, "dd-1b-seed2")
    assert same_base("qwen25-1.5b", "Qwen/Qwen2.5-1.5B")
    assert same_base("original", "EleutherAI/pythia-160m")
    assert not same_base("dd-1b", "dd-1b-seed2")
    assert not same_base("dd-1b-seed2", "dd-1b-seed2@step2500")
    assert not same_base("pythia-160m", "pythia-160m-seed1")
    assert not same_base("data-seed1", "weight-seed1")
    assert not same_base("qwen25-1.5b", "llama3-1b")

def test_family_and_describe():
    assert family_of("dd-1b") == "datadecide"
    assert family_of(DD2) == "datadecide"
    assert family_of("Qwen2.5-1.5B") == "qwen25"
    assert family_of("gemma3-1b") == "gemma3"
    assert family_of("pythia-410m") == "pythia"
    d = describe("dd-1b-seed2")
    assert d == {"token": "dd-1b-seed2", "dir_name": DD2, "ref": DD2_REF, "family": "datadecide"}

def test_hub_org_only_affects_pythia_refs():
    assert base_ref("pythia-160m", hub_org="SomeOrg") == "SomeOrg/pythia-160m"
    assert base_ref("dd-1b", hub_org="SomeOrg") == DD1_REF
    assert base_ref("qwen25-1.5b", hub_org="SomeOrg") == "Qwen/Qwen2.5-1.5B"
    assert base_dir_name("pythia-160m", hub_org="SomeOrg") == "pythia-160m"

@pytest.mark.parametrize("bad", [
    "", "  ", "all7", "seeds", "data-seed1,weight-seed1", "dd-2b", "dd-1b-seed4",
    "DataDecide-dolma1_7-1B-seed-small-aux-1", "DataDecide-dolma1_7-1B", "Qwen/Qwen2-7B",
    "mistral-7b", "pythia-999m", "pythia-160m-data-seed9", "unknown/thing",
    "allenai/DataDecide-dolma1_7-1B",           # dd ref without its seed branch
    "dd-1b@", "qwen25-1.5b@",
    # DataDecide sizes/recipes this repo cannot name (no CLI token): rejected, never
    # silently mapped onto the 1B token
    "DataDecide-dolma1_7-150M-seed-default",
    "allenai/DataDecide-dolma1_7-150M@step69369-seed-default",
])
def test_unrecognized_spellings_raise(bad):
    with pytest.raises(ValueError):
        base_spec(bad)

def test_repo_basename_alone_never_loses_the_datadecide_seed():
    # the bug class the audit found: `ref.split("/")[-1]` drops the seed branch
    assert base_dir_name(DD2_REF) == DD2
    assert base_dir_name(DD2_REF) != base_dir_name(DD1_REF)

def test_dir_name_agrees_with_every_writer_in_the_repo():
    """The invariant the audit rests on: naming's dir name == run_layout.model_dir_name on the
    spec == datadecide.base_dir_name_for_ref on the ref, for every spelling in CASES."""
    from subliminal_learning.models.datadecide import base_dir_name_for_ref
    from subliminal_learning.teachers.run_layout import model_dir_name

    for value, _token, dirname, ref in CASES:
        spec = base_spec(value)
        assert model_dir_name(spec, "EleutherAI") == dirname, value
        assert base_dir_name_for_ref(ref) == dirname, value

def test_naming_is_import_light():
    """Converting a string must never pull torch/peft/transformers into the process."""
    import subprocess
    import sys

    code = (
        "import sys; from subliminal_learning.models import naming as n\n"
        "n.base_spec('pythia-160m-data-seed1'); n.base_dir_name('dd-1b-seed2'); "
        "n.base_ref('qwen25-1.5b'); n.base_token('EleutherAI/pythia-410m-seed2@step3000')\n"
        "bad=[m for m in ('torch','peft','transformers') if m in sys.modules]\n"
        "print('HEAVY:'+','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "HEAVY:" in out.stdout and out.stdout.strip().endswith("HEAVY:"), out.stdout

# --- the local re-pretraining families: the inverter must agree with `base_spec`
LOCAL_SPELLINGS = [
    "pythia-160m-weightc-seed1",
    "pythia-160m-weightc-seed3",
    "pythia-160m-pilerepro-seed1234",
    "pythia-160m-pilerepro-seed3",
    "pythia-160m-dclm-seed1234",
    "pythia-160m-dclm-seed2",
]

@pytest.mark.parametrize("basename", LOCAL_SPELLINGS)
@pytest.mark.parametrize("prefix", ["", "local/"])
@pytest.mark.parametrize("revision", ["", "@step143000"])
def test_spec_from_base_name_inverts_local_families(basename, prefix, revision):
    from subliminal_learning.models.specs import spec_from_base_name

    name = f"{prefix}{basename}{revision}"
    spec = spec_from_base_name(name)
    assert spec == base_spec(name)
    assert spec.size == "160m" and spec.revision == (revision[1:] or None)

def test_spec_from_base_name_rejects_a_foreign_local_seed():
    from subliminal_learning.models.specs import spec_from_base_name

    with pytest.raises(ValueError, match="seeds"):
        spec_from_base_name("local/pythia-160m-pilerepro-seed2")

DCLM1 = "DataDecide-dclm-baseline-1B-seed-default"

DCLM2 = "DataDecide-dclm-baseline-1B-seed-large-aux-2"

DCLM1_REF = "allenai/DataDecide-dclm-baseline-1B@step69369-seed-default"

@pytest.mark.parametrize("seed,dir_name", [
    (1, DCLM1), (2, DCLM2), (3, "DataDecide-dclm-baseline-1B-seed-large-aux-3")])
@pytest.mark.parametrize("step", [None, 17500])
def test_dclm_spellings_round_trip(seed, dir_name, step):
    pin = f"@step{step}" if step else ""
    token = f"dd-1b-dclm-seed{seed}{pin}"
    for spelling in (token, dir_name + pin, base_ref(token)):
        assert base_token(spelling) == token
        assert base_dir_name(spelling) == dir_name + pin
        assert base_ref(spelling) == base_ref(token)
        assert family_of(spelling) == "datadecide"
    assert base_spec(token).recipe == "dclm-baseline"

def test_dclm_is_a_different_base_than_dolma():
    assert not same_base("dd-1b", "dd-1b-dclm")
    assert base_dir_name("dd-1b-dclm") == DCLM1
    assert base_ref("dd-1b-dclm") == DCLM1_REF
    # the default recipe is never spelled, and foreign recipes are unnameable
    for bad in ("dd-1b-dolma", "dd-1b-seed2-dclm", "dd-1b-foo",
                "DataDecide-fineweb-1B-seed-default",
                "allenai/DataDecide-fineweb-1B@step69369-seed-default"):
        with pytest.raises(ValueError):
            base_spec(bad)

@pytest.mark.parametrize("recipe,dirname", [
    ("", "DataDecide-dolma1_7-1B-seed-default"),
    ("-dclm", "DataDecide-dclm-baseline-1B-seed-default"),
    ("-c4", "DataDecide-c4-1B-seed-default"),
])
def test_final_step_dir_name_canonicalizes_like_token_and_ref(recipe, dirname):
    """`@step69369` is the final step, so all three spellings are ONE spec."""
    token = f"dd-1b{recipe}"
    ref = base_ref(token)
    assert base_spec(f"{dirname}@step69369").step is None
    assert same_base(token, dirname)
    assert same_base(token, f"{dirname}@step69369")
    assert same_base(ref, f"{dirname}@step69369")
    assert base_dir_name(f"{dirname}@step69369") == dirname
    # a NON-final step stays pinned
    assert base_spec(f"{dirname}@step10000").step == 10000
    assert not same_base(token, f"{dirname}@step10000")


# -------------------- from test_naming_integration.py --------------------

@pytest.mark.parametrize("spelling,tag", [
    ("dd-1b", "olmo"), ("dd-1b-seed2", "olmo"), (DD2, "olmo"),
    ("allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2", "olmo"),
    ("qwen25-1.5b", "qwen25"), ("Qwen2.5-1.5B", "qwen25"), ("Qwen/Qwen2.5-1.5B", "qwen25"),
    ("gemma3-1b", "gemma3"), ("llama3-1b", "llama3"),
    ("pythia-160m", "pythia"), ("original", "pythia"), ("data-seed1", "pythia"),
    ("EleutherAI/pythia-410m-seed2@step3000", "pythia"),
    ("local/pythia-160m-weightc-seed1", "pythia"),
    # legacy pattern fallbacks the naming module does not parse
    ("pythia-999m", "pythia"), ("DataDecide-dolma1_7-1B-seed-small-aux-1", "olmo"),
])
def test_family_for_base_repo_accepts_every_spelling(spelling, tag):
    fam = family_for_base_repo(spelling)
    assert fam is not None and fam.tag == tag

def test_family_for_base_repo_unknown_is_none():
    assert family_for_base_repo("something/else") is None
    assert family_for_base_repo("mistral-7b") is None

def test_anchored_dead_cut_accepts_token_ref_and_dir_name(tmp_path):
    rec = {"delta_cross": {DD2: 5.2, "DataDecide-dolma1_7-1B-seed-default": 5.1,
                           "pythia-160m": 5.0, "pythia-160m (orig)": 5.0}}
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "unigram_anchor_dp1.json").write_text(json.dumps(rec))
    for spelling in ("dd-1b-seed2", DD2,
                     "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"):
        assert anchored_dead_cut(tmp_path, spelling, pool="dp1") == 5.2
    for spelling in ("dd-1b", "DataDecide-dolma1_7-1B-seed-default"):
        assert anchored_dead_cut(tmp_path, spelling, pool="dp1") == 5.1
    for spelling in ("pythia-160m", "original", "EleutherAI/pythia-160m", "pythia-160m (orig)"):
        assert anchored_dead_cut(tmp_path, spelling, pool="dp1") == 5.0
    with pytest.raises(KeyError):
        anchored_dead_cut(tmp_path, "dd-1b-seed3", pool="dp1")

def test_anchor_writer_and_reader_agree_on_the_key():
    """`build_unigram_anchor` keys `delta_cross` by `datadecide.base_dir_name_for_ref(ref)`;
    `anchored_dead_cut` looks up `naming.base_dir_name(x)`. They must agree on every ref."""
    from subliminal_learning.models.datadecide import base_dir_name_for_ref
    from subliminal_learning.models.naming import base_dir_name

    for ref in ("EleutherAI/pythia-160m", "EleutherAI/pythia-410m-seed2",
                "local/pythia-160m-weightc-seed1",
                "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
                "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-3",
                "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt",
                "google/gemma-2-2b"):
        assert base_dir_name(ref) == base_dir_name_for_ref(ref), ref

def test_pipeline_same_base_is_spelling_tolerant_with_string_fallback():
    from subliminal_learning.pipelines.distill_student_pipeline import _same_base

    assert _same_base("allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2", "dd-1b-seed2")
    assert not _same_base("allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2",
                          "allenai/DataDecide-dolma1_7-1B@step69369-seed-default")
    assert _same_base("EleutherAI/pythia-160m", "pythia-160m")
    assert not _same_base("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-seed1")
    # unknown spellings: the historical string equality
    assert _same_base("weird/thing", "weird/thing")
    assert not _same_base("weird/thing", "other/thing")

def test_family_for_base_repo_returns_none_for_family_without_probe_pool(monkeypatch):
    """A registry base whose family has no ProbeFamily yet is 'no known family', not a KeyError."""
    import subliminal_learning.coherence.probe_families as pf

    monkeypatch.setattr(pf, "PROBE_FAMILIES", {k: v for k, v in pf.PROBE_FAMILIES.items() if k != "gemma2"})
    assert pf.family_for_base_repo("gemma2-2b") is None
    assert pf.family_for_base_repo("qwen25-1.5b") is not None

def test_cka_layout_keeps_datadecide_seed_branch(tmp_path):
    """`cka.layout` resolves a teacher-student manifest's numbers base to the seed-branch dir."""
    import json

    from subliminal_learning.representation.cka import layout as L

    run = tmp_path / "students" / "elephant" / DD2 / "teacher-x" / "t_tseed0" / "ds" / "cell" / "ftseed1"
    run.mkdir(parents=True)
    ref = "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"
    (run / "train_manifest.json").write_text(json.dumps({
        "source": "teacher", "animal": "elephant", "numbers_base_model_repo": ref,
        "teacher_param_tag": "ep10_lr5e-4_r64a128", "teacher_ftseed": 0,
        "student_param_tag": "ep10_lr1e-4_r8a32", "student_ftseed": 1,
    }))
    from subliminal_learning.models.naming import base_dir_name
    assert base_dir_name(ref) == DD2
    # the teacher dir the layout builds must be under the seed-branch spelling, not under ckpt/
    src = L.__file__
    text = open(src).read()
    assert "base_dir_name(m[\"numbers_base_model_repo\"])" in text


# -------------------- from test_registry.py --------------------

def test_resolve_hf_repo_id_full_variant_seed0_no_suffix() -> None:
    spec = PythiaModelSpec(size="410m", seed=0)
    assert resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-410m"

def test_resolve_hf_repo_id_full_variant_seed3_has_suffix() -> None:
    spec = PythiaModelSpec(size="410m", seed=3)
    assert resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-410m-seed3"

def test_resolve_hf_repo_id_data_variant() -> None:
    spec = PythiaModelSpec(size="160m", seed=2, variant="data")
    assert resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-160m-data-seed2"

def test_resolve_hf_repo_id_weight_variant() -> None:
    spec = PythiaModelSpec(size="160m", seed=1, variant="weight")
    assert (
        resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-160m-weight-seed1"
    )

def test_resolve_hf_repo_id_data_variant_rejects_non_160m() -> None:
    spec = PythiaModelSpec(size="410m", seed=1, variant="data")
    with pytest.raises(ValueError, match="only valid for size='160m'"):
        resolve_hf_repo_id(spec, "EleutherAI")

def test_resolve_hf_repo_id_data_variant_rejects_out_of_range_seed() -> None:
    spec = PythiaModelSpec(size="160m", seed=5, variant="data")
    with pytest.raises(ValueError, match="requires seed in"):
        resolve_hf_repo_id(spec, "EleutherAI")

def test_resolve_hf_repo_id_weightc_variant_is_local_provenance() -> None:
    # The corrected weight-seed re-pretraining is unpublished: no hub org, and a
    # `local/` prefix that `local_model_dir` (not the Hub) knows how to resolve.
    spec = PythiaModelSpec(size="160m", seed=1, variant="weightc")
    assert resolve_hf_repo_id(spec, "EleutherAI") == "local/pythia-160m-weightc-seed1"

def test_resolve_hf_repo_id_weightc_rejects_non_160m_and_bad_seed() -> None:
    with pytest.raises(ValueError, match="only valid for size='160m'"):
        resolve_hf_repo_id(PythiaModelSpec(size="410m", seed=1, variant="weightc"), "E")
    with pytest.raises(ValueError, match="requires seed in"):
        resolve_hf_repo_id(PythiaModelSpec(size="160m", seed=5, variant="weightc"), "E")

def test_local_model_dir_none_for_hub_repo_ids(tmp_path) -> None:
    assert local_model_dir("EleutherAI/pythia-160m", "step1000", tmp_path) is None

def test_local_model_dir_uses_revision_subdir(tmp_path) -> None:
    d = local_model_dir("local/pythia-160m-weightc-seed1", "step25000", tmp_path)
    assert d == tmp_path / "local" / "pythia-160m-weightc-seed1" / "step25000"

def test_local_model_dir_defaults_to_main_subdir(tmp_path) -> None:
    d = local_model_dir("local/pythia-160m-weightc-seed1", None, tmp_path)
    assert d == tmp_path / "local" / "pythia-160m-weightc-seed1" / "main"

@pytest.mark.parametrize(
    "variant,seed",
    [
        ("weightc", 3), ("pilereprov1", 1234), ("pilerepro", 1234), ("pilerepro", 3),
        ("dclm", 1234), ("dclm", 1),
    ],
)
def test_resolve_hf_repo_id_local_families_are_local_provenance(variant, seed) -> None:
    # Every LOCAL_PYTHIA_FAMILIES row resolves the same way weightc does.
    spec = PythiaModelSpec(size="160m", seed=seed, variant=variant)
    assert (
        resolve_hf_repo_id(spec, "EleutherAI")
        == f"local/pythia-160m-{variant}-seed{seed}"
    )

@pytest.mark.parametrize("variant,seed", [("pilerepro", 1), ("pilereprov1", 3), ("dclm", 3), ("weightc", 1234)])
def test_resolve_hf_repo_id_local_families_reject_foreign_seeds(variant, seed) -> None:
    with pytest.raises(ValueError, match="requires seed in"):
        resolve_hf_repo_id(PythiaModelSpec(size="160m", seed=seed, variant=variant), "E")

@pytest.mark.parametrize("variant", ["pilerepro", "dclm"])
def test_resolve_hf_repo_id_local_families_reject_non_160m(variant) -> None:
    with pytest.raises(ValueError, match="only valid for size='160m'"):
        resolve_hf_repo_id(PythiaModelSpec(size="410m", seed=1234, variant=variant), "E")

def test_local_family_table_shape() -> None:
    assert [f.variant for f in LOCAL_PYTHIA_FAMILIES] == [
        "weightc", "pilereprov1", "pilerepro", "dclm",
    ]
    assert local_pythia_family("weightc").seeds == (1, 2, 3)
    assert local_pythia_family("weightc").seeds_label() == "1-3"  # unchanged wording
    assert local_pythia_family("dclm").seeds == (1234, 1, 2)
    assert local_pythia_family("dclm").dir_name(1234) == "pythia-160m-dclm-seed1234"
    assert local_pythia_family("data") is None

def test_local_model_dir_handles_four_digit_seed(tmp_path) -> None:
    d = local_model_dir("local/pythia-160m-dclm-seed1234", "step75000", tmp_path)
    assert d == tmp_path / "local" / "pythia-160m-dclm-seed1234" / "step75000"

def test_pilereprov1_token_and_dir_name_round_trip() -> None:
    # `pilereprov1-seed1234` is its own family (not a `pilerepro` seed): the token
    # parses, the dir name inverts back to the same spec, and other seeds are rejected.
    from subliminal_learning.models.naming import base_dir_name, base_spec
    from subliminal_learning.models.specs import parse_base_models

    spec = PythiaModelSpec(size="160m", seed=1234, variant="pilereprov1", revision="step75000")
    assert parse_base_models("pilereprov1-seed1234@step75000") == [spec]
    assert base_spec("pythia-160m-pilereprov1-seed1234@step75000") == spec
    assert base_dir_name("pilereprov1-seed1234@step75000") == (
        "pythia-160m-pilereprov1-seed1234@step75000"
    )
    assert local_pythia_family("pilereprov1").repo_id(1234) == (
        "local/pythia-160m-pilereprov1-seed1234"
    )
    with pytest.raises(ValueError, match="pilereprov1 has seeds"):
        parse_base_models("pilereprov1-seed3")
