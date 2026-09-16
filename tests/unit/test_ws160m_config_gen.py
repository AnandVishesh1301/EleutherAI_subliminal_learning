"""ws160m generated YAMLs: shapes, routing, teacher run-set lineage, registry tokens."""

from __future__ import annotations

import yaml
import pytest

from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import (
    resolve_card_model_size,
    resolve_card_teacher_subdir,
    run_set_subdir,
)


def _registry():
    return models.registry(
        [(b.base, s) for b in models.ROSTER for s in models.CAMPAIGN_STEPS],
        base_models_root=plan.PATHS_CONFIG and __import__("pathlib").Path("/bm"),
    )


@pytest.mark.parametrize("partition", plan.PARTITION_ORDER)
def test_every_yaml_is_well_formed_and_tokens_resolve(partition: str) -> None:
    train = cgen.all_train_configs(partition, wave=2)
    verify = cgen.all_verify_configs(partition, train)
    assert set(verify) == {cgen.verify_name(n) for n in train}
    reg = _registry()
    for name, doc in train.items():
        yaml.safe_dump(doc)
        assert doc["queue"] in plan.queue_priorities(plan.partition(partition))
        if cgen.is_teacher_yaml(name):
            assert doc["run_set_id"] == plan.partition(partition).run_set_id
            parse_base_models(doc["base_models"], registry=reg)
            assert doc["grid"]["lora"] == [{"r": 128, "alpha": 128}]
            assert doc["ft_seeds"] == [0, 1, 2]
            continue
        assert doc["run_set_id"] == plan.partition(partition).run_set_id
        assert doc["gseeds"] == [0, 1, 2] and doc["teacher_ftseeds"] == [0, 1, 2]
        assert doc["grid"]["beta"] == [1.0] and doc["grid"]["lmbda"] == [1.0]
        assert len(doc["grid"]["student_modes"]) == 1
        for s, t in doc["teacher_student_base_pairs"]:
            parse_base_models(f"{s},{t}", registry=reg)
        students = [p[0] for p in doc["teacher_student_base_pairs"]]
        assert len(students) == len(set(students))
        assert doc["spec_registry"] == "ws160m"
        assert doc["prompt_pool_dir"] == plan.PROMPT_POOL_DIR
    for name, vdoc in verify.items():
        assert vdoc["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS)
        assert vdoc["inherit_grid"].startswith(
            plan.partition(partition).generated_config_dir
        )
        assert vdoc["queue"] in plan.queue_priorities(plan.partition(partition))
        assert "benchmarks" not in vdoc


def test_student_yaml_holds_one_teacher_and_names_its_run_set() -> None:
    train = cgen.student_train_configs("extra")
    for name, doc in train.items():
        teachers = {t for _, t in doc["teacher_student_base_pairs"]}
        assert len(teachers) == 1
        (teacher,) = teachers
        _, step = models.parse_token(teacher)
        assert doc["teacher_run_set_id"] == plan.teacher_run_set_id(step)
        assert doc["teacher_param_tags"]["lora"] in {
            plan.teacher_param_tag(e) for e in (2, 5, 8)
        }
    heat = [n for n in train if "_heatmap_" in n]
    assert any(
        cgen.student_train_configs("extra")[n]["teacher_run_set_id"]
        == plan.LADDER_RUN_SET_ID
        for n in heat
    )
    assert any(
        cgen.student_train_configs("extra")[n]["teacher_run_set_id"]
        == plan.MAIN_RUN_SET_ID
        for n in heat
    )


def test_control_yamls_pair_students_with_themselves_and_split_wave2() -> None:
    w1 = cgen.control_train_configs("main", wave=1)
    w2 = cgen.control_train_configs("main", wave=2)
    assert len(w2) == len(w1) + 1
    extra_name = next(n for n in w2 if n not in w1)
    assert "_w2_" in extra_name and w2[extra_name]["queue"].endswith("w2-controls")
    for doc in w2.values():
        assert all(s == t for s, t in doc["teacher_student_base_pairs"])
        assert doc["wandb"] is False
    ladder = cgen.control_train_configs("ladder")
    assert all("_w2_" not in n for n in ladder)
    assert all(doc["queue"].endswith("-controls") for doc in ladder.values())


def test_teacher_yamls_group_epochs_and_route_rung_teachers_to_ladder() -> None:
    main = cgen.teacher_train_configs("main", wave=2)
    ws1 = next(
        doc for n, doc in main.items() if n.endswith("_bear_pile-ws1_step143000.yaml")
    )
    assert ws1["grid"]["n_epochs"] == [2, 5, 8]
    w2 = [n for n in main if "_w2_" in n]
    assert w2 == ["sdf_ws160m_main_w2_bear_pile-ws3_step143000.yaml"]
    assert main[w2[0]]["queue"].endswith("w2-teachers")
    ladder = cgen.teacher_train_configs("ladder")
    assert all("_w2_" not in n for n in ladder)
    assert len(ladder) == 36
    assert all(doc["grid"]["n_epochs"] == [5] for doc in ladder.values())
    assert (
        cgen.teacher_train_configs("dosehp") == {}
        and cgen.teacher_train_configs("extra") == {}
    )
    logged = [doc for doc in main.values() if doc["wandb"]]
    assert len(logged) == 1


def test_queue_routing_by_replicate_rung_recipe_and_arm() -> None:
    main_cells = pairs.main_cells()
    si_ss = next(c for c in main_cells if c.category == "SI_SS")
    si_ds = next(c for c in main_cells if c.category == "SI_DS")
    leaves = {lf.replicate: lf for lf in pairs.leaves_of([si_ss])}
    assert (
        cgen.student_stage(leaves[0]) == "core"
        and cgen.student_stage(leaves[8]) == "boost-a"
    )
    assert cgen.student_stage(pairs.leaves_of([si_ds])[5]) == "boost-b"
    assert cgen.student_stage(pairs.leaves_of(pairs.wave2_cells())[0]) == "w2-students"
    lad = pairs.leaves_of(pairs.ladder_cells())
    assert {cgen.student_stage(lf) for lf in lad} == {
        f"r{r}" for r in plan.LADDER_RUNGS
    }
    hp = pairs.leaves_of(pairs.hp_cells())
    assert {cgen.student_stage(lf) for lf in hp} == {
        f"hp-{plan.recipe_slug(r)}" for r in plan.HP_RECIPES
    }
    assert (
        cgen.student_stage(pairs.leaves_of(pairs.dose_bear_cells())[0]) == "dose-bear"
    )
    assert cgen.student_stage(pairs.leaves_of(pairs.heatmap_cells())[0]) == "heatmap"
    assert cgen.eval_queue("main", "ws160m-main-boost-b") == "ws160m-main-evals"
    assert cgen.eval_queue("main", "ws160m-main-w2-controls") == "ws160m-main-w2-evals"
    assert (
        cgen.eval_queue("dosehp", "ws160m-dosehp-hp-lr1e-4_r8a16")
        == "ws160m-dosehp-evals-hp-lr1e-4_r8a16"
    )
    assert (
        cgen.eval_queue("extra", "ws160m-extra-controls")
        == "ws160m-extra-evals-dose-cat-eagle"
    )
    assert cgen.eval_dependencies("main", "ws160m-main-evals") == (
        "ws160m-main-controls",
        "ws160m-main-core",
    )
    assert "ws160m-main-boost-a" not in cgen.eval_dependencies(
        "main", "ws160m-main-evals"
    )
    with pytest.raises(ValueError, match="not a ws160m"):
        cgen.verify_name("other.yaml")


def test_every_launch_of_a_run_set_writes_the_same_card_invariants() -> None:
    cards: dict[tuple[str, str], set] = {}
    for partition in plan.PARTITION_ORDER:
        for wave in (models.WAVE_NOW, models.WAVE_2):
            for name, doc in cgen.all_train_configs(partition, wave=wave).items():
                run_set = doc["run_set_id"]
                if cgen.is_teacher_yaml(name):
                    key = ("sdf_teacher", run_set)
                    value = resolve_card_model_size(doc, str(doc["base_models"]))
                else:
                    key = ("gkd_student", run_set)
                    launch_subdir = run_set_subdir(
                        SDF_TEACHERS_SUBDIR, doc["teacher_run_set_id"]
                    )
                    value = (
                        resolve_card_model_size(
                            doc, str(doc["teacher_student_base_pairs"][0][0])
                        ),
                        resolve_card_teacher_subdir(doc, launch_subdir),
                    )
                cards.setdefault(key, set()).add(value)
    assert {k: v for k, v in cards.items() if len(v) != 1} == {}
    main_rs = plan.partition("main").run_set_id
    assert cards[("sdf_teacher", main_rs)] == {plan.MODEL_SIZE}
    assert cards[("gkd_student", main_rs)] == {
        (plan.MODEL_SIZE, run_set_subdir(SDF_TEACHERS_SUBDIR, main_rs))
    }
    extra = cards[("gkd_student", plan.partition("extra").run_set_id)]
    assert {v[1] for v in extra} == {cgen.teacher_subdir_scope("extra")}
    assert cgen.teacher_subdir_scope("extra") is not None
    for partition in ("main", "ladder", "dosehp"):
        assert cgen.teacher_subdir_scope(partition) is None
