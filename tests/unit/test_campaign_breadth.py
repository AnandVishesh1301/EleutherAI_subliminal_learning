"""Breadth campaign integrity + plan §9 regressions (CPU, tmp_path only)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipelines.student_hp_campaign_pipeline import (
    CampaignGateError,
    _a3_disk_gate,
    _finalize,
    _inject_reference_recipe,
)
from subliminal_learning.campaign import (
    breadth_plan,
    config_gen,
    driver_io,
    expected,
    plan,
)
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.campaign.selection import FrozenRecipe, Recipe, freeze_recipes
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.gkd.filtering import DEFAULT_TRAIT_LEXICON_PATH
from subliminal_learning.sdf.docs_io import resolve_sdf_docs_path
from subliminal_learning.utils.run_io import assert_unique_run_dirs, sha256_file

REPO = Path(__file__).resolve().parents[2]
BREADTH_ANIMALS = breadth_plan.ANIMALS
REF = breadth_plan.A3_REFERENCE_RECIPE

# Distinct recipes so 4 animals have pairwise-disjoint candidate sets.
_DISJOINT = [
    Recipe(8, 8, 5.0e-5),
    Recipe(8, 16, 1.0e-4),
    Recipe(8, 32, 2.0e-4),
    Recipe(16, 16, 5.0e-5),
]


def _repo_yaml(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())


def _cfg(tmp_path: Path) -> CampaignDriverConfig:
    return CampaignDriverConfig(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO,
        generated_dir_override=tmp_path / "generated",
        spec=breadth_plan.SPEC,
        require_a0=False,
    )


def _pref_row(
    *,
    animal: str,
    recipe: Recipe,
    dose: float,
    p_target: float,
    is_control: bool = False,
) -> dict:
    return {
        "animal": animal,
        "model": "pythia-160m-data-seed1",
        "beta": 1.0,
        "num_epochs": dose,
        "student_mode": "lora",
        "student_lora_r": recipe.lora_r,
        "student_lora_alpha": recipe.lora_alpha,
        "student_lr": recipe.lr,
        "set": "test",
        "scoring": "prob",
        "p_target": p_target,
        "is_control": is_control,
        "teacher_cross_init": False,
    }


# ---- regression: plan.SPEC content-stable ------------------------------------------------


def test_plan_spec_keeps_six_prompt_sets_without_storytelling() -> None:
    assert plan.SPEC.eval_prompt_sets == (
        "test",
        "favorite",
        "dislike",
        "validation",
        "consensus",
        "register",
    )
    assert "storytelling" not in plan.SPEC.eval_prompt_sets


def test_breadth_spec_prompt_sets_include_storytelling() -> None:
    assert "storytelling" in breadth_plan.SPEC.eval_prompt_sets
    assert breadth_plan.SPEC.eval_prompt_sets == breadth_plan.EVAL_PROMPT_SETS
    assert len(breadth_plan.SPEC.eval_prompt_sets) == 7


def test_plan_spec_verify_yaml_omits_storytelling() -> None:
    train = config_gen.a3_train_configs(
        {
            (animal, float(beta)): [Recipe(64, 64, 2e-4)]
            for animal in plan.ANIMALS
            for beta in plan.BETAS
        }
    )
    verify = config_gen.a3_verify_configs(train, spec=plan.SPEC)
    for doc in verify.values():
        assert doc["eval"]["prompt_sets"] == list(plan.SPEC.eval_prompt_sets)
        assert "storytelling" not in doc["eval"]["prompt_sets"]


def test_breadth_verify_yaml_includes_storytelling() -> None:
    train = config_gen.a3_train_configs(
        {
            (animal, float(beta)): [Recipe(64, 64, 2e-4), REF]
            for animal in BREADTH_ANIMALS
            for beta in breadth_plan.BETAS
        },
        spec=breadth_plan.SPEC,
    )
    verify = config_gen.a3_verify_configs(train, spec=breadth_plan.SPEC)
    for doc in verify.values():
        assert doc["eval"]["prompt_sets"] == list(breadth_plan.SPEC.eval_prompt_sets)
        assert "storytelling" in doc["eval"]["prompt_sets"]


# ---- reference-arm freeze (§3.1) ---------------------------------------------------------


def test_reference_arm_injection_makes_freeze_unraisable_on_disjoint_sets() -> None:
    """Pairwise-disjoint candidate recipes per animal — freeze needs the shared ref."""
    cands = {(animal, 1.0): [_DISJOINT[i]] for i, animal in enumerate(BREADTH_ANIMALS)}
    # Preference rows cover each animal's private recipe only — no shared arm.
    private_rows = []
    for animal in BREADTH_ANIMALS:
        priv = cands[(animal, 1.0)][0]
        private_rows.append(
            _pref_row(
                animal=animal, recipe=priv, dose=1.0, p_target=0.003, is_control=True
            )
        )
        private_rows.append(
            _pref_row(animal=animal, recipe=priv, dose=1.0, p_target=0.5)
        )
    with pytest.raises(ValueError, match="cannot freeze"):
        freeze_recipes(pd.DataFrame(private_rows), cands, doses=(1.0,))

    injected = _inject_reference_recipe(cands, REF)
    for recipes in injected.values():
        assert REF in recipes
        assert recipes[-1] == REF

    rows = list(private_rows)
    for animal in BREADTH_ANIMALS:
        rows.append(
            _pref_row(
                animal=animal, recipe=REF, dose=1.0, p_target=0.003, is_control=True
            )
        )
        rows.append(_pref_row(animal=animal, recipe=REF, dose=1.0, p_target=0.08))
    frozen = freeze_recipes(pd.DataFrame(rows), injected, doses=(1.0,))
    assert frozen[1.0].recipe == REF


def test_a3_ref_arm_label_never_displaces_r1() -> None:
    cands = {
        (animal, 1.0): [Recipe(64, 64, 2e-4), Recipe(16, 32, 1e-4), REF]
        for animal in BREADTH_ANIMALS
    }
    cfgs = config_gen.a3_train_configs(cands, spec=breadth_plan.SPEC)
    ref_names = [f for f in cfgs if "_ref_" in f]
    r1_names = [f for f in cfgs if "_r1_" in f]
    assert ref_names
    assert r1_names
    assert all("_r1_" not in f or "ref" not in f.split("_") for f in ref_names)
    for f in r1_names:
        assert cfgs[f]["grid"]["student_modes"] == ["lora-r64a64"]
    for f in ref_names:
        assert cfgs[f]["grid"]["student_modes"] == [REF.mode_token]
        assert cfgs[f]["wandb"] is False


# ---- missing_manifests / cell_is_done (§3.2) ---------------------------------------------


def test_missing_manifests_rejects_manifest_plus_empty_checkpoints(
    tmp_path: Path,
) -> None:
    leaf = tmp_path / "cell"
    (leaf / "checkpoints").mkdir(parents=True)
    (leaf / "train_manifest.json").write_text("{}")
    assert expected.missing_manifests([leaf]) == [leaf]

    (leaf / "checkpoints" / "model.safetensors").write_bytes(b"w")
    assert expected.missing_manifests([leaf]) == []


# ---- verify-completeness (§3.8) ----------------------------------------------------------


def test_verify_summaries_complete_preference_only_incomplete(tmp_path: Path) -> None:
    cdir = tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "preference_summary.json").write_text("{}")
    assert expected.verify_summaries_complete([cdir]) == [cdir]

    (cdir / "sdf_forced_choice_summary.json").write_text("{}")
    (cdir / "sdf_capability_summary.json").write_text(json.dumps({"perplexity": {}}))
    assert expected.verify_summaries_complete([cdir]) == []


def test_verify_summaries_require_benchmarks_key(tmp_path: Path) -> None:
    cdir = tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "preference_summary.json").write_text("{}")
    (cdir / "sdf_forced_choice_summary.json").write_text("{}")
    (cdir / "sdf_capability_summary.json").write_text(json.dumps({"perplexity": {}}))
    assert expected.verify_summaries_complete([cdir], require_benchmarks=True) == [cdir]

    (cdir / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {}, "benchmarks": {}})
    )
    assert expected.verify_summaries_complete([cdir], require_benchmarks=True) == []


def test_verify_summaries_non_dict_capability_incomplete(tmp_path: Path) -> None:
    cdir = tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "preference_summary.json").write_text("{}")
    (cdir / "sdf_forced_choice_summary.json").write_text("{}")
    (cdir / "sdf_capability_summary.json").write_text(json.dumps([1, 2, 3]))
    assert expected.verify_summaries_complete([cdir], require_benchmarks=True) == [cdir]


def test_a3_disk_gate_raises_on_preference_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.generated_dir.mkdir(parents=True)
    fname = "gkd_a3_bear_b1_r1_e1.yaml"
    doc = {
        "animals": ["bear"],
        "base_models": "data-seed1",
        "teacher_student_base_pairs": [["data-seed1", "data-seed1"]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [0],
        "gseeds": [0],
        "grid": {
            "beta": [1.0],
            "lmbda": [1.0],
            "student_modes": ["lora-r8a16"],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": breadth_plan.TEACHER_PARAM_TAG_LORA,
            "full": breadth_plan.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {"full": 2.0e-5, "lora": [4.0e-4]},
        "num_epochs": 1.0,
        "prompt_pool_dir": "gkd_prompts/pile_prompts_v1",
        "run_set_id": breadth_plan.RUN_SET_ID,
        "teacher_run_set_id": breadth_plan.TEACHER_RUN_SET_ID,
        "wandb": False,
    }
    (cfg.generated_dir / fname).write_text(yaml.safe_dump(doc))
    leaves = expected.treatment_leaf_dirs(doc, cfg.output_root, spec=breadth_plan.SPEC)
    for leaf in leaves:
        ckpt = leaf / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "model.safetensors").write_bytes(b"w")
        (leaf / "train_manifest.json").write_text("{}")
    for cdir in {leaf.parent for leaf in leaves}:
        (cdir / "preference_summary.json").write_text("{}")
    state = {
        "transitions": {
            "submit_a3": {"train_yamls": [fname], "verify_yamls": []},
        }
    }
    with pytest.raises(CampaignGateError, match="verify summaries"):
        _a3_disk_gate(cfg, state)


def test_finalize_raises_on_preference_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.generated_dir.mkdir(parents=True)
    fname = "gkd_b_same_init_headline.yaml"
    doc = {
        "animals": ["bear"],
        "base_models": "data-seed1",
        "teacher_student_base_pairs": [["data-seed1", "data-seed1"]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [0],
        "gseeds": [0],
        "grid": {
            "beta": [1.0],
            "lmbda": [1.0],
            "student_modes": ["lora-r8a16"],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": breadth_plan.TEACHER_PARAM_TAG_LORA,
            "full": breadth_plan.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {"full": 2.0e-5, "lora": [4.0e-4]},
        "num_epochs": 1.0,
        "prompt_pool_dir": "gkd_prompts/pile_prompts_v1",
        "run_set_id": breadth_plan.RUN_SET_ID,
        "teacher_run_set_id": breadth_plan.TEACHER_RUN_SET_ID,
        "wandb": False,
    }
    (cfg.generated_dir / fname).write_text(yaml.safe_dump(doc))
    leaves = expected.treatment_leaf_dirs(doc, cfg.output_root, spec=breadth_plan.SPEC)
    for leaf in leaves:
        ckpt = leaf / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "model.safetensors").write_bytes(b"w")
        (leaf / "train_manifest.json").write_text("{}")
    for cdir in {leaf.parent for leaf in leaves}:
        (cdir / "preference_summary.json").write_text("{}")
    state = {
        "transitions": {
            "submit_b": {
                "train_yamls": [fname],
                "verify_yamls": [],
                "stage_c": [],
            },
        }
    }
    with pytest.raises(CampaignGateError, match="verify summaries"):
        _finalize(cfg, state)


# ---- breadth cell-count integrity --------------------------------------------------------


def test_breadth_a1_a2_static_yaml_cell_counts(tmp_path: Path) -> None:
    same = expected.treatment_leaf_dirs(
        _repo_yaml(breadth_plan.A1_SAME_YAML), tmp_path, spec=breadth_plan.SPEC
    )
    cross = expected.treatment_leaf_dirs(
        _repo_yaml(breadth_plan.A1_CROSS_YAML), tmp_path, spec=breadth_plan.SPEC
    )
    assert len(same) == breadth_plan.A1_CELLS_PER_YAML  # 144
    assert len(cross) == breadth_plan.A1_CELLS_PER_YAML
    assert len(same) + len(cross) == breadth_plan.A1_TOTAL_CELLS  # 288
    assert _repo_yaml(breadth_plan.A1_SAME_YAML)["gseeds"] == list(
        breadth_plan.A1_GSEEDS
    )
    assert _repo_yaml(breadth_plan.A1_SAME_YAML)["teacher_ftseeds"] == list(
        breadth_plan.STAGE_A_TFS
    )

    a2 = _repo_yaml(breadth_plan.A1_SAME_YAML)
    a2["grid"] = {**a2["grid"], "student_modes": [breadth_plan.A2_STUDENT_MODE]}
    assert (
        len(expected.control_leaf_dirs(a2, tmp_path, spec=breadth_plan.SPEC))
        == breadth_plan.A2_CONTROL_CELLS
    )


def test_breadth_a3_arm_count_within_max_same_arms() -> None:
    recipes = [
        Recipe(128, 128, 4e-4),
        Recipe(128, 128, 2e-4),
        Recipe(128, 128, 1e-4),
        Recipe(128, 128, 8e-4),
        REF,
    ]
    assert len(recipes) <= breadth_plan.A3_MAX_SAME_ARMS
    cands = {
        (animal, float(beta)): list(recipes)
        for animal in BREADTH_ANIMALS
        for beta in breadth_plan.BETAS
    }
    cfgs = config_gen.a3_train_configs(cands, spec=breadth_plan.SPEC)
    same_arms = {f.split("_")[4] for f in cfgs if "_cross_" not in f}
    assert "ref" in same_arms
    assert len(same_arms) <= breadth_plan.A3_MAX_SAME_ARMS
    for doc in cfgs.values():
        assert doc["gseeds"] == list(breadth_plan.A3_GSEEDS)
        assert doc["teacher_ftseeds"] == list(breadth_plan.STAGE_A_TFS)


def test_breadth_stage_b_648_and_seed_axes(tmp_path: Path) -> None:
    frozen = FrozenRecipe(
        recipe=REF,
        num_epochs=1.0,
        subliminal_delta=0.1,
        per_animal_delta={a: 0.1 for a in BREADTH_ANIMALS},
    )
    cfgs = config_gen.b_train_configs(frozen, 1.0, spec=breadth_plan.SPEC)
    treat = sum(
        len(expected.treatment_leaf_dirs(doc, tmp_path, spec=breadth_plan.SPEC))
        for doc in cfgs.values()
    )
    controls: set[Path] = set()
    for f in config_gen.b_control_train_names(cfgs):
        controls.update(
            expected.control_leaf_dirs(cfgs[f], tmp_path, spec=breadth_plan.SPEC)
        )
    assert treat == breadth_plan.B_TREAT_CELLS  # 564 = 141 * 4
    assert len(controls) == breadth_plan.B_CONTROL_CELLS  # 84
    assert treat + len(controls) == breadth_plan.B_TOTAL_CELLS  # 648

    for fname, doc in cfgs.items():
        if fname.endswith("_headline.yaml"):
            assert doc["teacher_ftseeds"] == list(breadth_plan.B_TFS_HEADLINE)
        else:
            assert doc["teacher_ftseeds"] == list(breadth_plan.B_TFS_EXTENSION)
        assert doc["gseeds"] == list(breadth_plan.B_GSEEDS)


def test_assert_unique_run_dirs_across_both_campaigns(tmp_path: Path) -> None:
    cells: list[tuple[str, Path]] = []
    for rel, spec in (
        (plan.A1_SAME_YAML, plan.SPEC),
        (plan.A1_CROSS_YAML, plan.SPEC),
        (breadth_plan.A1_SAME_YAML, breadth_plan.SPEC),
        (breadth_plan.A1_CROSS_YAML, breadth_plan.SPEC),
    ):
        doc = _repo_yaml(rel)
        for leaf in expected.treatment_leaf_dirs(doc, tmp_path, spec=spec):
            cells.append((str(leaf), leaf))
        a2 = {**doc, "grid": {**doc["grid"], "student_modes": [spec.a2_student_mode]}}
        for leaf in expected.control_leaf_dirs(a2, tmp_path, spec=spec):
            cells.append((f"ctrl:{leaf}", leaf))

    frozen_plan = FrozenRecipe(
        recipe=Recipe(8, 16, 4e-4),
        num_epochs=1.0,
        subliminal_delta=0.1,
        per_animal_delta={"cat": 0.1, "dog": 0.1},
    )
    frozen_b5 = FrozenRecipe(
        recipe=REF,
        num_epochs=1.0,
        subliminal_delta=0.1,
        per_animal_delta={a: 0.1 for a in BREADTH_ANIMALS},
    )
    for frozen, spec in ((frozen_plan, plan.SPEC), (frozen_b5, breadth_plan.SPEC)):
        for doc in config_gen.b_train_configs(frozen, 1.0, spec=spec).values():
            for leaf in expected.treatment_leaf_dirs(doc, tmp_path, spec=spec):
                cells.append((str(leaf), leaf))
            for leaf in expected.control_leaf_dirs(doc, tmp_path, spec=spec):
                cells.append((f"ctrl:{leaf}", leaf))

    assert_unique_run_dirs(cells)
    plan_roots = {p for _, p in cells if plan.RUN_SET_ID in str(p)}
    b5_roots = {p for _, p in cells if breadth_plan.RUN_SET_ID in str(p)}
    assert plan_roots
    assert b5_roots
    assert not (plan_roots & b5_roots)


# ---- animal coverage + tokenization ------------------------------------------------------


@pytest.mark.parametrize("animal", BREADTH_ANIMALS)
def test_breadth_animal_in_forms_and_lexicon(animal: str) -> None:
    forms = load_animal_forms()
    assert animal in forms
    assert forms[animal]
    lexicon = json.loads((REPO / DEFAULT_TRAIT_LEXICON_PATH).read_text())
    assert animal in lexicon


def test_forced_choice_roster_contains_breadth_animals() -> None:
    for rel in (
        breadth_plan.A1_SAME_VERIFY_YAML,
        breadth_plan.A1_CROSS_VERIFY_YAML,
    ):
        roster = set(_repo_yaml(rel)["eval"]["forced_choice_candidates"])
        assert set(BREADTH_ANIMALS) <= roster


def test_breadth_surface_forms_leading_space_single_token() -> None:
    """Canonical singular forms are single GPT-NeoX tokens (leading space).

    Forced-choice / ``...the`` prob prompts use the animal name itself; plurals
    in ``animal_forms.json`` may be multi-token (``eagles``, ``tigers``) and are
    not required to be single-token for the breadth eval contract.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — skip when Hub/cache unavailable
        pytest.skip(f"pythia-160m tokenizer unavailable locally: {exc}")

    forms = load_animal_forms()
    for animal in BREADTH_ANIMALS:
        for surface in (animal, forms[animal][0]):
            spaced = f" {surface}"
            ids = tok.encode(spaced, add_special_tokens=False)
            assert len(ids) == 1, f"{animal!r} form {spaced!r} -> {ids}"


# ---- teacher provenance + corpus sha -----------------------------------------------------


def test_sdf_docs_sha256_stable_for_breadth_corpora() -> None:
    t1 = _repo_yaml("configs/experiment/sdf_teacher_sweep_breadth5_t1.yaml")
    root = Path(t1["sdf_docs_root"])
    overrides = {str(k): str(v) for k, v in (t1.get("sdf_docs") or {}).items()}
    if not root.is_dir():
        pytest.skip(f"sdf_docs_root missing: {root}")
    digests: dict[str, str] = {}
    for animal in BREADTH_ANIMALS:
        path = resolve_sdf_docs_path(root, animal, overrides)
        if animal == "cat":
            assert str(path) == overrides["cat"]
        d1 = sha256_file(path)
        d2 = sha256_file(path)
        assert d1 is not None and d1 == d2
        digests[animal] = d1
    assert len(set(digests.values())) == len(BREADTH_ANIMALS)


def test_teacher_provenance_chain_resolves_on_synthetic_fixtures(
    tmp_path: Path,
) -> None:
    """student leaf → teacher_manifest_sha256 → teacher manifest → corpus sha."""
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text('{"text":"x","animal":"bear"}\n')
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()

    teacher_manifest = {
        "animal": "bear",
        "base_model_repo": "EleutherAI/pythia-160m-data-seed1",
        "param_tag": breadth_plan.TEACHER_PARAM_TAG_LORA,
        "ft_seed": 0,
        "sdf_docs_path": str(corpus),
        "sdf_docs_sha256": corpus_sha,
        "run_identity": {"sdf_docs_sha256": corpus_sha},
    }
    teacher_path = tmp_path / "teacher" / "train_manifest.json"
    teacher_path.parent.mkdir(parents=True)
    teacher_bytes = json.dumps(teacher_manifest, sort_keys=True).encode()
    teacher_path.write_bytes(teacher_bytes)
    teacher_sha = hashlib.sha256(teacher_bytes).hexdigest()

    student_manifest = {
        "teacher_tag": f"sdf_{breadth_plan.TEACHER_PARAM_TAG_LORA}_ft0",
        "teacher_checkpoint_dir": str(tmp_path / "teacher" / "checkpoints"),
        "teacher_manifest_sha256": teacher_sha,
        "run_identity": {"teacher_manifest_sha256": teacher_sha},
    }
    student_path = tmp_path / "student" / "train_manifest.json"
    student_path.parent.mkdir(parents=True)
    student_path.write_text(json.dumps(student_manifest))

    loaded_student = json.loads(student_path.read_text())
    assert loaded_student["teacher_manifest_sha256"] == teacher_sha
    loaded_teacher = json.loads(teacher_path.read_text())
    assert (
        hashlib.sha256(json.dumps(loaded_teacher, sort_keys=True).encode()).hexdigest()
        == teacher_sha
    )
    assert loaded_teacher["sdf_docs_sha256"] == corpus_sha
    assert sha256_file(Path(loaded_teacher["sdf_docs_path"])) == corpus_sha


# ---- static YAML identity pins -----------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        breadth_plan.A1_SAME_YAML,
        breadth_plan.A1_CROSS_YAML,
    ],
)
def test_breadth_a1_yaml_run_set_and_animals(rel: str) -> None:
    doc = _repo_yaml(rel)
    assert doc["animals"] == list(BREADTH_ANIMALS)
    assert doc["run_set_id"] == breadth_plan.RUN_SET_ID
    assert doc["teacher_run_set_id"] == breadth_plan.TEACHER_RUN_SET_ID


@pytest.mark.parametrize(
    "rel",
    [
        breadth_plan.A1_SAME_VERIFY_YAML,
        breadth_plan.A1_CROSS_VERIFY_YAML,
    ],
)
def test_breadth_a1_verify_inherits_and_storytelling(rel: str) -> None:
    doc = _repo_yaml(rel)
    assert "storytelling" in doc["eval"]["prompt_sets"]
    assert doc["benchmarks"]["lora"]["lora_r"] == [64]


def test_teacher_t1_t2_yaml_pins() -> None:
    t1 = _repo_yaml("configs/experiment/sdf_teacher_sweep_breadth5_t1.yaml")
    t2 = _repo_yaml("configs/experiment/sdf_teacher_sweep_breadth5_t2.yaml")
    assert t1["animals"] == list(BREADTH_ANIMALS)
    assert t1["base_models"] == "all7"
    assert t1["run_set_id"] == breadth_plan.TEACHER_RUN_SET_ID
    assert t1["grid"] == {
        "n_epochs": [5],
        "learning_rate": [5.0e-5],
        "lora": [{"r": 128, "alpha": 128}],
    }
    assert t1["sdf_docs_root"] == breadth_plan.SDF_DOCS_ROOT
    assert t1["sdf_docs"]["cat"] == breadth_plan.CAT_SDF_DOCS_OVERRIDE
    assert t2["base_models"] == "data-seed1"
    assert t2["grid"]["n_epochs"] == [1, 3, 5]
    assert t2["grid"]["learning_rate"] == [3.5e-5, 5.0e-5]
    assert t2["grid"]["lora"] == [{"r": 64, "alpha": 64}, {"r": 128, "alpha": 128}]
    assert t1["sdf_docs_root"] == t2["sdf_docs_root"]
    assert t1["sdf_docs"] == t2["sdf_docs"]
    assert list(breadth_plan.BETAS) == [1.0]
    assert list(breadth_plan.A3_DOSES) == [0.5, 1.0, 3.0, 5.0]
    assert list(breadth_plan.FREEZE_DOSES) == [1.0]
    assert breadth_plan.SPEC.freeze_doses == (1.0,)
    assert breadth_plan.SPEC.doses_for_freeze() == (1.0,)
    assert plan.SPEC.freeze_doses is None
    assert plan.SPEC.doses_for_freeze() == plan.SPEC.a3_doses
    with pytest.raises(ValueError, match="freeze_doses"):
        replace(breadth_plan.SPEC, freeze_doses=(7.0,)).doses_for_freeze()
    assert breadth_plan.Q_A1.startswith("b4-")
    assert breadth_plan.GENERATED_CONFIG_DIR.endswith("/breadth4")
    assert breadth_plan.TEACHER_TRAIN_ANIMALS == ("cat", "bear", "eagle")
    assert breadth_plan.DOG_TEACHER_SOURCE_RUN_SET_ID == "20260726_crossinit_160m"


def test_breadth_gkd_experiment_config_on_train_cmd() -> None:
    """Breadth train shards pin W&B via --experiment-config; slbreadth does not."""
    assert plan.SPEC.gkd_experiment_config is None
    assert (
        breadth_plan.SPEC.gkd_experiment_config
        == "configs/experiment/gkd_student_breadth5.yaml"
    )
    b5 = driver_io.a1_jobs(spec=breadth_plan.SPEC)[0].cmd
    slb = driver_io.a1_jobs(spec=plan.SPEC)[0].cmd
    assert "--experiment-config configs/experiment/gkd_student_breadth5.yaml" in b5
    assert "--experiment-config" not in slb
    # Eval shards never take the experiment-config flag.
    assert all(
        "--experiment-config" not in j.cmd
        for j in driver_io.a1_eval_jobs(spec=breadth_plan.SPEC)
    )
