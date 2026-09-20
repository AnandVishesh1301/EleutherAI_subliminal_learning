"""`models.naming`: every spelling of a base model round-trips through one spec."""

from __future__ import annotations

import pytest

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
from subliminal_learning.models.registry import ExternalModelSpec, PythiaModelSpec

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
    from subliminal_learning.teachers.model_specs import spec_from_base_name

    name = f"{prefix}{basename}{revision}"
    spec = spec_from_base_name(name)
    assert spec == base_spec(name)
    assert spec.size == "160m" and spec.revision == (revision[1:] or None)


def test_spec_from_base_name_rejects_a_foreign_local_seed():
    from subliminal_learning.teachers.model_specs import spec_from_base_name

    with pytest.raises(ValueError, match="seeds"):
        spec_from_base_name("local/pythia-160m-pilerepro-seed2")


# --- the data-recipe axis: dclm-baseline round-trips like dolma1_7 ----------------

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
