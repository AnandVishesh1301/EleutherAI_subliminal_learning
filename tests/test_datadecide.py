"""DataDecide (OLMo) token grammar, resolution, dispatch, and guards.

All offline: no network and no `hf_olmo` import at module scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    NumberPromptSetConfig,
    generation_params,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.datadecide import (
    DataDecideSpec,
    dd_dir_name,
    dd_model_ref,
    dd_repo_id,
    dd_revision,
    load_datadecide_model,
    parse_base_model_tokens,
    parse_dd_token,
)
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    model_ref,
    resolve_hf_repo_id,
)
from subliminal_learning.students.layout import (
    student_param_dir,
    student_run_dir,
    student_run_label,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import (
    model_dir_name,
    model_dir_parts,
    run_label,
    teacher_param_dir,
    teacher_run_dir,
)
from subliminal_learning.teachers.sft_io import check_target_modules_safe

HUB_ORG = "EleutherAI"
OLMO_REGEX = r".*transformer\.blocks\.\d+\.(att_proj|attn_out|ff_proj|ff_out)"


# --- token grammar ---------------------------------------------------------


def test_bare_token_is_seed1_default_unpinned():
    spec = parse_dd_token("dd-1b")
    assert spec == DataDecideSpec(size="1B", seed=1, step=None, recipe="dolma1_7")
    assert dd_revision(spec) == "step69369-seed-default"


def test_seed2_maps_to_large_aux_2():
    assert dd_revision(parse_dd_token("dd-1b-seed2")) == "step69369-seed-large-aux-2"


def test_step_suffix_pins_the_step():
    spec = parse_dd_token("dd-1b-seed3@step17500")
    assert (spec.seed, spec.step) == (3, 17500)
    assert dd_revision(spec) == "step17500-seed-large-aux-3"


@pytest.mark.parametrize(
    "token,fragment",
    [
        ("dd-150m", "only the 1B models are wired"),
        ("dd-300m", "budget-mismatched"),
        ("dd-1b-seed4", "valid ordinals"),
        ("dd-1b-seed0", "valid ordinals"),
        ("dd-1b@stepXY", "@step<N>"),
        ("dd-1B", "lowercase"),
    ],
)
def test_bad_tokens_rejected(token, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_dd_token(token)


def test_mixed_family_list_rejected():
    with pytest.raises(ValueError, match="mixed base-model families"):
        parse_base_model_tokens("dd-1b,data-seed1")


# --- router passthrough ----------------------------------------------------


@pytest.mark.parametrize(
    "value", ["all7", "data-seed1,weight-seed3@step3000", "pythia-410m"]
)
def test_router_delegates_pythia_strings_unchanged(value):
    assert parse_base_model_tokens(value) == parse_base_models(value)


# --- resolution ------------------------------------------------------------


def test_repo_id_and_refs():
    spec = parse_dd_token("dd-1b-seed2")
    assert dd_repo_id(spec) == "allenai/DataDecide-dolma1_7-1B"
    assert dd_model_ref(spec) == (
        "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"
    )
    pinned = parse_dd_token("dd-1b-seed2@step5000")
    assert dd_revision(pinned) == "step5000-seed-large-aux-2"


def test_dir_names_and_ckpt_routing():
    unpinned = parse_dd_token("dd-1b")
    pinned = parse_dd_token("dd-1b@step5000")
    assert dd_dir_name(unpinned) == "DataDecide-dolma1_7-1B-seed-default"
    assert dd_dir_name(pinned) == "DataDecide-dolma1_7-1B-seed-default@step5000"
    assert model_dir_parts(unpinned, HUB_ORG) == (
        "DataDecide-dolma1_7-1B-seed-default",
    )
    assert model_dir_parts(pinned, HUB_ORG) == (
        "ckpt",
        "DataDecide-dolma1_7-1B-seed-default@step5000",
    )


def test_registry_dispatches_datadecide():
    spec = parse_dd_token("dd-1b-seed3")
    assert resolve_hf_repo_id(spec, HUB_ORG) == "allenai/DataDecide-dolma1_7-1B"
    assert model_ref(spec, HUB_ORG) == dd_model_ref(spec)


# --- pythia dispatch regression -------------------------------------------


@pytest.mark.parametrize(
    "spec,repo,ref,dirname",
    [
        (
            PythiaModelSpec(size="160m"),
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-160m",
            "pythia-160m",
        ),
        (
            PythiaModelSpec(size="160m", seed=1, variant="data"),
            "EleutherAI/pythia-160m-data-seed1",
            "EleutherAI/pythia-160m-data-seed1",
            "pythia-160m-data-seed1",
        ),
        (
            PythiaModelSpec(size="160m", seed=3, variant="weight", revision="step3000"),
            "EleutherAI/pythia-160m-weight-seed3",
            "EleutherAI/pythia-160m-weight-seed3@step3000",
            "pythia-160m-weight-seed3@step3000",
        ),
        (
            PythiaModelSpec(size="410m"),
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-410m",
            "pythia-410m",
        ),
    ],
)
def test_pythia_identity_unchanged(spec, repo, ref, dirname):
    assert resolve_hf_repo_id(spec, HUB_ORG) == repo
    assert model_ref(spec, HUB_ORG) == ref
    assert model_dir_name(spec, HUB_ORG) == dirname


def test_pythia_ckpt_routing_unchanged():
    plain = PythiaModelSpec(size="160m", seed=1, variant="data")
    pinned = PythiaModelSpec(size="160m", seed=1, variant="data", revision="step3000")
    assert model_dir_parts(plain, HUB_ORG) == ("pythia-160m-data-seed1",)
    assert model_dir_parts(pinned, HUB_ORG) == (
        "ckpt",
        "pythia-160m-data-seed1@step3000",
    )


# --- guards ----------------------------------------------------------------


def test_all_linear_guard():
    check_target_modules_safe("all-linear", is_neox=True)
    check_target_modules_safe(OLMO_REGEX, is_neox=False)
    with pytest.raises(ValueError, match="ff_out"):
        check_target_modules_safe("all-linear", is_neox=False)


def test_missing_ai2_olmo_raises_actionable_runtime_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "hf_olmo", None)  # `import hf_olmo` -> ImportError
    with pytest.raises(RuntimeError, match=r"uv sync --group datadecide"):
        load_datadecide_model(parse_dd_token("dd-1b"), tmp_path, "cpu")


# --- CLI surface: every core-CLI helper must accept a DataDecideSpec ---------

_LORA = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=OLMO_REGEX)
_TRAIN = TrainConfig(
    learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="linear", seed=0, optim="adamw_torch",
)


@pytest.mark.parametrize(
    "spec", [DataDecideSpec(seed=2), DataDecideSpec(seed=3, step=17500)]
)
def test_cli_surface_accepts_datadecide_spec(spec):
    """The helpers the five core CLIs call on a spec: no exception, non-empty result."""
    assert spec_label(spec)
    assert resolve_hf_repo_id(spec, HUB_ORG)
    assert model_ref(spec, HUB_ORG)
    assert model_dir_name(spec, HUB_ORG)
    assert model_dir_parts(spec, HUB_ORG)
    assert teacher_run_dir(Path("/out"), "cat", spec, HUB_ORG, _TRAIN, _LORA).parts
    assert teacher_param_dir(Path("/out"), "cat", spec, HUB_ORG, _TRAIN, _LORA).parts
    assert run_label("cat", spec, HUB_ORG, _TRAIN, _LORA)
    student_kwargs = dict(
        source="teacher",
        spec=spec,
        hub_org=HUB_ORG,
        train_cfg=_TRAIN,
        lora_cfg=_LORA,
        animal="cat",
        teacher_seed=0,
        number_dataset_tag="numbers",
        numbers_base_model=model_dir_name(spec, HUB_ORG),
    )
    assert student_param_dir(
        Path("/out"), teacher_param_tag="ep1_lr2e-4_r8a8", **student_kwargs
    ).parts
    assert student_run_dir(
        Path("/out"), teacher_param_tag="ep1_lr2e-4_r8a8", **student_kwargs
    ).parts
    assert student_run_label(**student_kwargs)


def test_spec_label_exact_strings():
    # Carries family+size (unlike the Pythia labels): labels must be unique
    # across families, so a bare 'seed2' is not available here.
    assert spec_label(DataDecideSpec(seed=2)) == "dd-1b-seed2"
    assert spec_label(DataDecideSpec(seed=3, step=17500)) == "dd-1b-seed3@step17500"


# --- numbers resume identity: DD always carries the base ref ----------------


def _numbers_cfg(spec) -> GenerateNumbersConfig:
    paths = PathsConfig(Path("/o"), Path("/o"), Path("/o"), Path("/o"))
    return GenerateNumbersConfig(
        paths=paths, hub_org=HUB_ORG, spec=spec, source="base", out_dir=Path("/o/d"),
        prompt_set=NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10),
    )


def test_numbers_identity_pythia_revision_none_has_no_base_ref():
    spec = PythiaModelSpec(size="160m", seed=1, variant="data")
    assert "base_model_ref" not in generation_params(_numbers_cfg(spec), "phash")


@pytest.mark.parametrize(
    "spec", [DataDecideSpec(seed=2), DataDecideSpec(seed=3, step=17500)]
)
def test_numbers_identity_datadecide_always_has_base_ref(spec):
    # The branch encodes the SEED, so two seeds' pools must not share an identity.
    params = generation_params(_numbers_cfg(spec), "phash")
    assert params["base_model_ref"] == dd_model_ref(spec)
