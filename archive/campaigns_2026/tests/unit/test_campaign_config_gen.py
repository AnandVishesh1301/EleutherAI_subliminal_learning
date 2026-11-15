"""Generated A3 / Stage B / C sweep YAMLs: schema, counts, and cell enumeration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import config_gen, expected, plan
from subliminal_learning.campaign.selection import FrozenRecipe, Recipe

R1, R2, R3 = Recipe(64, 64, 2e-4), Recipe(8, 32, 4e-4), Recipe(16, 64, 1e-4)
CANDS = {
    (animal, float(beta)): [R1, R2, R3]
    for animal in plan.ANIMALS
    for beta in plan.BETAS
}
FROZEN = FrozenRecipe(
    recipe=R1,
    num_epochs=3.0,
    subliminal_delta=0.1,
    per_animal_delta={"cat": 0.12, "dog": 0.08},
)

# Every key the train sweep script reads from a sweep YAML.
TRAIN_KEYS = {
    "animals",
    "base_models",
    "teacher_student_base_pairs",
    "teacher_modes",
    "teacher_ftseeds",
    "gseeds",
    "grid",
    "teacher_param_tags",
    "student_lr",
    "num_epochs",
    "prompt_pool_dir",
    "run_set_id",
    "teacher_run_set_id",
    "wandb",
}
GRID_KEYS = {"beta", "lmbda", "student_modes", "vocab_mask", "teacher_temp"}


def _assert_train_schema(doc: dict) -> None:
    assert TRAIN_KEYS <= set(doc)
    assert GRID_KEYS == set(doc["grid"])
    assert doc["run_set_id"] == plan.RUN_SET_ID
    assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID
    assert set(doc["teacher_param_tags"]) == {"lora", "full"}
    assert doc["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG_LORA
    students = {p[0] for p in doc["teacher_student_base_pairs"]}
    assert students == set(str(doc["base_models"]).split(","))
    assert yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)) == doc


# ---- A3 -----------------------------------------------------------------------------


def test_a3_train_configs_counts_and_schema(tmp_path: Path) -> None:
    cfgs = config_gen.a3_train_configs(CANDS)
    # 2 animals x 2 betas x (3 same arms + 1 cross arm) x 5 doses = 80 YAMLs.
    n_doses = len(plan.A3_DOSES)
    assert len(cfgs) == 2 * 2 * 4 * n_doses
    for fname, doc in cfgs.items():
        _assert_train_schema(doc)
        assert len(expected.treatment_leaf_dirs(doc, tmp_path)) == 3  # 3 gseeds
        assert len(expected.control_leaf_dirs(doc, tmp_path)) == 3
    headline = config_gen.a3_headline_train_names(cfgs)
    assert len(headline) == 2 * 2 * n_doses
    assert all("_r1_" in f for f in headline)


def test_a3_dose_ladder_covers_every_plan_dose() -> None:
    cfgs = config_gen.a3_train_configs(CANDS)
    doses = {doc["num_epochs"] for doc in cfgs.values()}
    assert doses == {float(d) for d in plan.A3_DOSES}
    assert 10.0 in doses


def test_a3_arm_count_follows_the_selection() -> None:
    """Arms are selection-dependent (R1 union + R3 extension), not a fixed 3."""
    cands = {("cat", 1.0): [R1, R2, R3, Recipe(32, 32, 8e-4)]}
    cfgs = config_gen.a3_train_configs(cands)
    arms = {f.split("_")[4] for f in cfgs}
    assert arms == {"r1", "r2", "r3", "r4", "cross"}
    assert cfgs["gkd_a3_cat_b1_r4_e1.yaml"]["student_lr"]["lora"] == [8e-4]


def test_a3_same_and_cross_pairs() -> None:
    cfgs = config_gen.a3_train_configs(CANDS)
    same = cfgs["gkd_a3_cat_b1_r2_e0.5.yaml"]
    assert same["teacher_student_base_pairs"] == [list(plan.A1_SAME_PAIR)]
    assert same["grid"]["student_modes"] == [R2.mode_token]
    assert same["student_lr"]["lora"] == [R2.lr]
    assert same["num_epochs"] == 0.5
    cross = cfgs["gkd_a3_dog_b0_cross_e5.yaml"]
    assert cross["teacher_student_base_pairs"] == [list(plan.A1_CROSS_PAIR)]
    assert cross["grid"]["student_modes"] == [R1.mode_token]  # best-1 on cross
    assert cross["grid"]["beta"] == [0.0]
    assert cross["animals"] == ["dog"]


def test_a3_wandb_only_on_headline_arm() -> None:
    cfgs = config_gen.a3_train_configs(CANDS)
    for fname, doc in cfgs.items():
        if "_r1_" in fname:
            assert doc["wandb"]["student_modes"] == [R1.mode_token]
            assert doc["wandb"]["num_epochs"] == [doc["num_epochs"]]
        else:
            assert doc["wandb"] is False


def test_a3_total_cells_match_plan(tmp_path: Path) -> None:
    cfgs = config_gen.a3_train_configs(CANDS)
    same = sum(
        len(expected.treatment_leaf_dirs(doc, tmp_path))
        for f, doc in cfgs.items()
        if "_cross_" not in f
    )
    cross = sum(
        len(expected.treatment_leaf_dirs(doc, tmp_path))
        for f, doc in cfgs.items()
        if "_cross_" in f
    )
    controls: set = set()
    for f in config_gen.a3_headline_train_names(cfgs):
        controls.update(expected.control_leaf_dirs(cfgs[f], tmp_path))
    # Selection-dependent: 3 same arms x 2 animals x 2 betas x 5 doses x 3 gseeds.
    per_cell = 2 * 2 * len(plan.A3_DOSES) * len(plan.A3_GSEEDS)
    assert same == 3 * per_cell
    assert cross == per_cell
    assert len(controls) == per_cell


def test_a3_verify_configs_inherit_and_bench() -> None:
    train = config_gen.a3_train_configs(CANDS)
    verify = config_gen.a3_verify_configs(train)
    assert len(verify) == len(train)
    v = verify["gkd_a3_verify_cat_b1_r1_e3.yaml"]
    assert v["inherit_grid"] == f"{plan.GENERATED_CONFIG_DIR}/gkd_a3_cat_b1_r1_e3.yaml"
    assert v["benchmarks"]["lora"] == {
        "num_epochs": [3.0],
        "learning_rate": [R1.lr],
        "lora_r": [R1.lora_r],
    }
    assert "test" in v["eval"]["prompt_sets"]
    assert "benchmarks" not in verify["gkd_a3_verify_cat_b1_r2_e3.yaml"]


def test_a3_rejects_empty_or_oversized_candidate_list() -> None:
    with pytest.raises(ValueError, match="expected 1.."):
        config_gen.a3_train_configs({("cat", 1.0): []})
    too_many = [Recipe(8, 8, 1e-5 * i) for i in range(plan.A3_MAX_SAME_ARMS + 1)]
    with pytest.raises(ValueError, match="expected 1.."):
        config_gen.a3_train_configs({("cat", 1.0): too_many})


# ---- Stage B -------------------------------------------------------------------------


def test_b_train_configs_counts_and_cells(tmp_path: Path) -> None:
    cfgs = config_gen.b_train_configs(FROZEN, 1.0)
    # 4 groups with both tiers + 2 seed0 groups extension-only = 10 YAMLs.
    assert len(cfgs) == 10
    total = 0
    for fname, doc in cfgs.items():
        _assert_train_schema(doc)
        assert doc["num_epochs"] == FROZEN.num_epochs
        assert doc["grid"]["beta"] == [1.0]
        assert doc["grid"]["student_modes"] == [R1.mode_token]
        tfs = doc["teacher_ftseeds"]
        assert tfs == ([0, 1, 2] if fname.endswith("_headline.yaml") else [0])
        total += len(expected.treatment_leaf_dirs(doc, tmp_path))
    assert total == 282

    controls: set = set()
    for f in config_gen.b_control_train_names(cfgs):
        controls.update(expected.control_leaf_dirs(cfgs[f], tmp_path))
    assert len(controls) == plan.B_CONTROL_CELLS  # 42


def test_b_control_yamls_cover_all_students() -> None:
    cfgs = config_gen.b_train_configs(FROZEN, 1.0)
    names = config_gen.b_control_train_names(cfgs)
    assert names == ["gkd_b_same_init_extension.yaml", "gkd_b_same_init_headline.yaml"]
    students = set()
    for f in names:
        students.update(p[0] for p in cfgs[f]["teacher_student_base_pairs"])
    assert len(students) == 7


def test_b_wandb_only_same_init_headline() -> None:
    cfgs = config_gen.b_train_configs(FROZEN, 1.0)
    for fname, doc in cfgs.items():
        if fname == "gkd_b_same_init_headline.yaml":
            assert doc["wandb"]["student_modes"] == [R1.mode_token]
        else:
            assert doc["wandb"] is False


def test_b_verify_configs() -> None:
    train = config_gen.b_train_configs(FROZEN, 1.0)
    verify = config_gen.b_verify_configs(train)
    assert len(verify) == 10
    v = verify["gkd_b_verify_family_swap_headline.yaml"]
    assert v["inherit_grid"].endswith("gkd_b_family_swap_headline.yaml")
    assert v["benchmarks"]["lora"]["lora_r"] == [R1.lora_r]
    assert "benchmarks" not in verify["gkd_b_verify_seed0_data_extension.yaml"]


# ---- Stage C -------------------------------------------------------------------------


def test_c_train_configs_cells(tmp_path: Path) -> None:
    cfgs = config_gen.c_train_configs(FROZEN, 1.0)
    assert set(cfgs) == {
        "gkd_c_lambda.yaml",
        "gkd_c_vmask.yaml",
        "gkd_c_fullteacher.yaml",
    }
    for doc in cfgs.values():
        _assert_train_schema(doc)
    assert len(expected.treatment_leaf_dirs(cfgs["gkd_c_lambda.yaml"], tmp_path)) == 48
    assert len(expected.treatment_leaf_dirs(cfgs["gkd_c_vmask.yaml"], tmp_path)) == 24
    assert (
        len(expected.treatment_leaf_dirs(cfgs["gkd_c_fullteacher.yaml"], tmp_path))
        == 24
    )
    assert cfgs["gkd_c_lambda.yaml"]["grid"]["lmbda"] == [0.0, 0.5]
    assert cfgs["gkd_c_vmask.yaml"]["grid"]["vocab_mask"] == [True]
    assert cfgs["gkd_c_fullteacher.yaml"]["teacher_modes"] == ["full"]
    verify = config_gen.c_verify_configs(cfgs)
    assert len(verify) == 3


# ---- static A1/A2 YAMLs (committed) ---------------------------------------------------


def _repo_yaml(rel: str) -> dict:
    return yaml.safe_load((Path(__file__).parents[2] / rel).read_text())


@pytest.mark.parametrize("rel", [plan.A1_SAME_YAML, plan.A1_CROSS_YAML])
def test_static_a1_yaml_defines_144_cells(rel: str, tmp_path: Path) -> None:
    doc = _repo_yaml(rel)
    _assert_train_schema(doc)
    assert len(expected.treatment_leaf_dirs(doc, tmp_path)) == plan.A1_CELLS_PER_YAML
    assert doc["wandb"]["student_modes"] == ["lora-r64a64"]
    assert doc["wandb"]["learning_rate"] == [2.0e-4]


def test_static_a1_yamls_disjoint_leaves(tmp_path: Path) -> None:
    same = expected.treatment_leaf_dirs(_repo_yaml(plan.A1_SAME_YAML), tmp_path)
    cross = expected.treatment_leaf_dirs(_repo_yaml(plan.A1_CROSS_YAML), tmp_path)
    assert len(same) == len(set(same))
    assert len(cross) == len(set(cross))
    assert not set(same) & set(cross)  # _tbase- suffix keeps arms disjoint
    assert len(set(same) | set(cross)) == plan.A1_TOTAL_CELLS


def test_a2_restriction_yields_16_controls(tmp_path: Path) -> None:
    doc = _repo_yaml(plan.A1_SAME_YAML)
    doc["grid"] = {**doc["grid"], "student_modes": [plan.A2_STUDENT_MODE]}
    controls = expected.control_leaf_dirs(doc, tmp_path)
    assert len(controls) == plan.A2_CONTROL_CELLS  # 16
    treatments = expected.treatment_leaf_dirs(doc, tmp_path)
    assert len(treatments) == 16
    full = expected.treatment_leaf_dirs(_repo_yaml(plan.A1_SAME_YAML), tmp_path)
    assert set(treatments) <= set(full)  # A2 treatments are A1 cells (shared leaves)


def test_static_verify_yamls_inherit_their_train_yaml() -> None:
    for train_rel, verify_rel in (
        (plan.A1_SAME_YAML, plan.A1_SAME_VERIFY_YAML),
        (plan.A1_CROSS_YAML, plan.A1_CROSS_VERIFY_YAML),
    ):
        doc = _repo_yaml(verify_rel)
        assert doc["inherit_grid"] == train_rel
        assert "test" in doc["eval"]["prompt_sets"]
        assert doc["benchmarks"]["lora"]["lora_r"] == [64]
