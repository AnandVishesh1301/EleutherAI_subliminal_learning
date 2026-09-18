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
    DD_1B_FINAL_STEP,
    DataDecideSpec,
    base_dir_name_for_ref,
    dd_dir_name,
    dd_model_ref,
    dd_repo_id,
    dd_revision,
    load_datadecide_model,
    parse_base_model_tokens,
    parse_dd_ref,
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
from subliminal_learning.teachers.sft_io import (
    check_target_modules_for_base_specs,
    check_target_modules_safe,
)

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


NEOX_TARGETS = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]


def test_cross_family_target_guards():
    check_target_modules_safe(NEOX_TARGETS, is_neox=True)
    with pytest.raises(ValueError, match="GPT-NeoX module names"):
        check_target_modules_safe(NEOX_TARGETS, is_neox=False)
    with pytest.raises(ValueError, match="OLMo module names"):
        check_target_modules_safe(OLMO_REGEX, is_neox=True)
    with pytest.raises(ValueError, match="OLMo module names"):
        check_target_modules_safe(["att_proj", "ff_proj"], is_neox=True)


def test_expansion_time_family_check():
    dd = parse_dd_token("dd-1b")
    (pythia,) = parse_base_models("original")
    # Matched pairings pass, for both families and mixed-free lists.
    check_target_modules_for_base_specs(NEOX_TARGETS, [pythia])
    check_target_modules_for_base_specs(OLMO_REGEX, [dd])
    # dd base swept with the Pythia LoRA config: rejected, naming base and config.
    with pytest.raises(ValueError, match=r"lora_template_bias\.yaml.*DataDecide"):
        check_target_modules_for_base_specs(
            NEOX_TARGETS,
            [dd],
            training_config=Path("configs/training/lora_template_bias.yaml"),
        )
    # Pythia base swept with the OLMo regex config: also rejected.
    with pytest.raises(ValueError, match="incompatible with swept base"):
        check_target_modules_for_base_specs(OLMO_REGEX, [pythia])


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


# --- ref parsing: the inverse of dd_model_ref -------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("step", [None, 17500, DD_1B_FINAL_STEP])
def test_parse_dd_ref_round_trips(seed, step):
    spec = DataDecideSpec(seed=seed, step=step)
    # The final step is spelled identically to "unpinned" in a revision, so it
    # canonicalizes to step=None; any other pin survives.
    canonical = DataDecideSpec(seed=seed, step=None if step == DD_1B_FINAL_STEP else step)
    assert parse_dd_ref(dd_model_ref(spec)) == canonical


def test_parse_dd_ref_recovers_dir_name_key():
    ref = dd_model_ref(DataDecideSpec(seed=1))
    assert dd_dir_name(parse_dd_ref(ref)) == "DataDecide-dolma1_7-1B-seed-default"


@pytest.mark.parametrize(
    "ref, fragment",
    [
        ("allenai/DataDecide-dolma1_7-1B", "ill-defined"),
        ("EleutherAI/pythia-160m@step1000", "expected"),
        ("allenai/DataDecide-dolma1_7-1B@main", "step<N>-seed-<branch>"),
        ("allenai/DataDecide-dolma1_7-1B@stepX-seed-default", "step<N>-seed-<branch>"),
        ("allenai/DataDecide-dolma1_7-1B@step69369-seed-small-aux-2", "valid seeds"),
    ],
)
def test_parse_dd_ref_rejects_malformed(ref, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_dd_ref(ref)


def test_final_step_token_canonicalizes():
    assert parse_dd_token(f"dd-1b@step{DD_1B_FINAL_STEP}") == parse_dd_token("dd-1b")
    assert parse_dd_token(f"dd-1b-seed2@step{DD_1B_FINAL_STEP}").step is None


# --- cross-base teacher-{base} level spelling (distill pipeline) -------------


def test_base_dir_name_for_ref_datadecide_matches_reader_key():
    # The eval reader reconstructs this level as model_dir_name(numbers_spec).
    ref = dd_model_ref(DataDecideSpec(seed=2))
    assert base_dir_name_for_ref(ref) == model_dir_name(DataDecideSpec(seed=2), HUB_ORG)


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("EleutherAI/pythia-160m", "pythia-160m"),
        ("EleutherAI/pythia-160m@step100000", "pythia-160m@step100000"),
        ("EleutherAI/pythia-160m-seed1", "pythia-160m-seed1"),
    ],
)
def test_base_dir_name_for_ref_pythia_unchanged(ref, expected):
    assert base_dir_name_for_ref(ref) == expected


# --- ref-based base loading dispatch (no network, no real weights) ----------


class _FakeTok:
    pad_token = "<pad>"
    eos_token = "</s>"


class _FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaled = True
        return self


def _patch_auto(monkeypatch, calls):
    import subliminal_learning.models.loading_io as lio

    class _Auto:
        kind = ""

        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            calls.append((cls.kind, repo_id, kwargs))
            return _FakeTok() if cls.kind == "tok" else _FakeModel()

    monkeypatch.setattr(lio, "AutoTokenizer", type("T", (_Auto,), {"kind": "tok"}))
    monkeypatch.setattr(lio, "AutoModelForCausalLM", type("M", (_Auto,), {"kind": "mdl"}))
    return lio


def test_load_base_from_ref_routes_datadecide(monkeypatch, tmp_path):
    import subliminal_learning.models.loading_io as lio

    seen: dict = {}

    def _mdl(spec, root, device, dtype=None):
        seen["mdl"] = (spec, root, device)
        return _FakeModel()

    def _tok(spec, root):
        seen["tok"] = spec
        return _FakeTok()

    monkeypatch.setattr(lio, "load_datadecide_model", _mdl)
    monkeypatch.setattr(lio, "load_datadecide_tokenizer", _tok)
    ref = dd_model_ref(DataDecideSpec(seed=2))
    mdl, tok = lio.load_base_from_ref(ref, tmp_path, "cpu")
    assert isinstance(mdl, _FakeModel) and isinstance(tok, _FakeTok)
    assert seen["mdl"] == (DataDecideSpec(seed=2), tmp_path, "cpu")
    assert seen["tok"] == DataDecideSpec(seed=2)


def test_load_base_from_ref_pinned_pythia_passes_revision(monkeypatch, tmp_path):
    calls: list = []
    lio = _patch_auto(monkeypatch, calls)
    lio.load_base_from_ref("EleutherAI/pythia-160m@step100000", tmp_path, "cpu")
    assert {c[1] for c in calls} == {"EleutherAI/pythia-160m"}
    for _, _, kwargs in calls:
        assert kwargs["revision"] == "step100000"
        assert kwargs["cache_dir"] == str(tmp_path / "hub")


def test_load_base_from_ref_bare_pythia_has_no_revision(monkeypatch, tmp_path):
    calls: list = []
    lio = _patch_auto(monkeypatch, calls)
    mdl, tok = lio.load_base_from_ref("EleutherAI/pythia-160m", tmp_path, "cpu")
    assert [c[1] for c in calls] == ["EleutherAI/pythia-160m"] * 2
    assert all("revision" not in kwargs for _, _, kwargs in calls)
    assert mdl.device == "cpu" and mdl.evaled


# --- local/ (unpublished conversion) base loading: dir, never the Hub ----------


def _make_local_dir(root, revision="step25000"):
    d = root / "local" / "pythia-160m-weightc-seed1" / revision
    d.mkdir(parents=True)
    return d


def test_load_base_from_ref_local_loads_directory_without_revision(monkeypatch, tmp_path):
    calls: list = []
    lio = _patch_auto(monkeypatch, calls)
    d = _make_local_dir(tmp_path)
    mdl, tok = lio.load_base_from_ref(
        "local/pythia-160m-weightc-seed1@step25000", tmp_path, "cpu"
    )
    assert [c[1] for c in calls] == [str(d)] * 2
    assert all("revision" not in kwargs and "cache_dir" not in kwargs
               for _, _, kwargs in calls)
    assert mdl.device == "cpu" and mdl.evaled and tok.pad_token


def test_load_base_from_ref_local_missing_dir_raises(monkeypatch, tmp_path):
    lio = _patch_auto(monkeypatch, [])
    with pytest.raises(FileNotFoundError, match="convert_neox_to_hf"):
        lio.load_base_from_ref(
            "local/pythia-160m-weightc-seed1@step25000", tmp_path, "cpu"
        )


def test_load_model_and_tokenizer_route_weightc_to_local_dir(monkeypatch, tmp_path):
    import subliminal_learning.models.loading_io as lio
    from subliminal_learning.models.registry import PythiaModelSpec

    calls: list = []
    _patch_auto(monkeypatch, calls)
    monkeypatch.setattr(
        lio, "GPTNeoXForCausalLM",
        type("N", (), {"from_pretrained": classmethod(
            lambda cls, repo_id, **kw: (calls.append(("neox", repo_id, kw)), _FakeModel())[1]
        )}),
    )
    d = _make_local_dir(tmp_path)
    spec = PythiaModelSpec(size="160m", seed=1, variant="weightc", revision="step25000")
    lio.load_tokenizer(spec, tmp_path, "EleutherAI")
    lio.load_model(spec, tmp_path, "EleutherAI", "cpu")
    assert [c[1] for c in calls] == [str(d), str(d)]
    assert all("revision" not in kw and "cache_dir" not in kw for _, _, kw in calls)


def test_load_model_weightc_unpinned_error_asks_for_step_pin(monkeypatch, tmp_path):
    import subliminal_learning.models.loading_io as lio
    from subliminal_learning.models.registry import PythiaModelSpec

    # No revision -> the "main" dir, which does not exist while the model trains.
    with pytest.raises(FileNotFoundError, match=r"@step<N>"):
        lio.load_model(
            PythiaModelSpec(size="160m", seed=1, variant="weightc"),
            tmp_path, "EleutherAI", "cpu",
        )


# --- the data-recipe axis: dclm-baseline alongside the unnamed dolma1_7 default ----


DCLM = "dclm-baseline"


def test_dolma_spellings_are_byte_identical():
    """The 1738 teacher runs on disk: the default recipe's names must not move."""
    spec = DataDecideSpec()
    assert dd_dir_name(spec) == "DataDecide-dolma1_7-1B-seed-default"
    assert dd_repo_id(spec) == "allenai/DataDecide-dolma1_7-1B"
    assert dd_revision(spec) == "step69369-seed-default"
    assert dd_model_ref(spec) == (
        "allenai/DataDecide-dolma1_7-1B@step69369-seed-default"
    )
    assert spec_label(DataDecideSpec(seed=2)) == "dd-1b-seed2"


def test_dclm_token_resolves_to_its_own_repo():
    spec = parse_dd_token("dd-1b-dclm")
    assert spec == DataDecideSpec(size="1B", seed=1, step=None, recipe=DCLM)
    assert dd_repo_id(spec) == "allenai/DataDecide-dclm-baseline-1B"
    assert dd_dir_name(spec) == "DataDecide-dclm-baseline-1B-seed-default"
    assert dd_model_ref(spec) == (
        "allenai/DataDecide-dclm-baseline-1B@step69369-seed-default"
    )
    # seeds and the final step are shared with dolma1_7
    assert dd_revision(parse_dd_token("dd-1b-dclm-seed2")) == "step69369-seed-large-aux-2"
    assert parse_dd_token(f"dd-1b-dclm@step{DD_1B_FINAL_STEP}") == spec


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("step", [None, 17500])
def test_dclm_round_trips_token_ref_dir(seed, step):
    token = f"dd-1b-dclm-seed{seed}" + (f"@step{step}" if step else "")
    spec = parse_dd_token(token)
    assert spec == DataDecideSpec(seed=seed, step=step, recipe=DCLM)
    assert parse_dd_ref(dd_model_ref(spec)) == spec
    assert base_dir_name_for_ref(dd_model_ref(spec)) == dd_dir_name(spec)
    assert spec_label(spec) == token


@pytest.mark.parametrize(
    "token, fragment",
    [
        ("dd-1b-dolma", "recipe segment"),
        ("dd-1b-dolma1_7", "recipe segment"),
        ("dd-1b-foo", "recipe segment"),
        ("dd-1b-seed2-dclm", "recipe segment"),
        ("dd-1b-dclm-seed4", "valid ordinals"),
    ],
)
def test_bad_recipe_tokens_rejected(token, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_dd_token(token)


def test_foreign_recipe_in_ref_rejected():
    with pytest.raises(ValueError, match="wired recipe"):
        parse_dd_ref("allenai/DataDecide-c4-1B@step69369-seed-default")


@pytest.mark.parametrize("token", ["dd-1b-", "dd-1b-dclm-"])
def test_trailing_hyphen_tokens_are_rejected(token):
    """A trailing separator names no seed segment, whatever precedes it."""
    with pytest.raises(ValueError):
        parse_dd_token(token)
