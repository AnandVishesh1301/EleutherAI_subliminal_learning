"""Generated ckptgrid2 sweep YAMLs: I1/I2/I8 grouping, counts, loaders."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs
from subliminal_learning.campaign import ckptgrid2_config_gen as cgen
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.evaluation import sdf_prompt_sets as sps
from subliminal_learning.teachers.model_specs import parse_base_models

REPO_ROOT = Path(__file__).parents[2]

_ARM_YAML_COUNTS = {
    "dose": 16,
    "dose_teacher": 12,
    "hp": 49,
    "hp_probe": 12,
    "rect": 30,
    "rect512": 12,
    "cat": 7,
    "early_cross": 5,
}


def _student_docs(*, wave: int = 1, audit_passed: bool = False) -> dict[str, dict]:
    return cgen.student_train_configs(wave=wave, audit_passed=audit_passed)


def test_i1_one_teacher_and_no_repeated_student_per_doc() -> None:
    for fname, doc in _student_docs().items():
        rows = doc["teacher_student_base_pairs"]
        teachers = {row[1] for row in rows}
        students = [row[0] for row in rows]
        assert len(teachers) == 1, fname
        assert len(students) == len(set(students)), fname


def test_i2_one_recipe_one_epochs_one_teacher_tag_per_doc() -> None:
    for fname, doc in {
        **_student_docs(),
        **cgen.control_train_configs(),
    }.items():
        assert len(doc["grid"]["student_modes"]) == 1, fname
        assert len(doc["student_lr"]["lora"]) == 1, fname
        assert isinstance(doc["num_epochs"], float), fname
        assert len({doc["teacher_param_tags"]["lora"]}) == 1, fname


def test_per_side_step_labels_survive_expand_base_pairs() -> None:
    seen: list[tuple[str, str]] = []
    for fname, doc in _student_docs().items():
        rows = doc["teacher_student_base_pairs"]
        expanded = _expand_base_pairs(None, "same", rows)
        assert len(expanded) == len(rows), fname
        for (student_spec, teacher_spec), (s_lab, t_lab) in zip(
            expanded, rows, strict=True
        ):
            assert "@step" in s_lab and "@step" in t_lab, fname
            assert student_spec.revision and student_spec.revision.startswith("step")
            assert teacher_spec.revision and teacher_spec.revision.startswith("step")
            assert student_spec.revision != "step0"
            assert teacher_spec.revision != "step0"
            parsed_s = parse_base_models(s_lab)[0]
            parsed_t = parse_base_models(t_lab)[0]
            assert student_spec == parsed_s
            assert teacher_spec == parsed_t
            seen.append((s_lab, t_lab))
    groups = cgen.student_cell_groups()
    expected = {
        (cell.student_label, cell.teacher_label)
        for cells in groups.values()
        for cell in cells
    }
    assert set(seen) == expected


def test_cross_rung_pair_keeps_independent_revisions() -> None:
    rows = [
        tuple(row)
        for doc in _student_docs().values()
        for row in doc["teacher_student_base_pairs"]
    ]
    assert ("original@step143000", "original@step2000") in rows
    student, teacher = _expand_base_pairs(
        None, "same", [["original@step143000", "original@step2000"]]
    )[0]
    assert student.revision == "step143000"
    assert teacher.revision == "step2000"


def test_i8_every_student_and_control_doc_shares_the_run_set_card() -> None:
    docs = {**_student_docs(), **cgen.control_train_configs()}
    for fname, doc in docs.items():
        assert doc["run_set_id"] == plan.RUN_SET_ID, fname
        assert doc["teacher_run_set_id"] == plan.TEACHER_RUN_SET_ID, fname
        assert doc["prompt_pool_dir"] == plan.PROMPT_POOL_DIR, fname
        assert doc["run_set_model_size"] == plan.MODEL_SIZE, fname


def test_student_verify_carries_the_five_set_bundle() -> None:
    train = {**_student_docs(), **cgen.control_train_configs()}
    verify = cgen.student_verify_configs(train)
    assert len(verify) == len(train)
    for fname, doc in verify.items():
        assert doc["eval"]["prompt_sets"] == list(plan.EVAL_PROMPT_SETS), fname
        assert doc["inherit_grid"].startswith(plan.GENERATED_CONFIG_DIR)


def test_teacher_verify_carries_the_seven_set_bundle() -> None:
    verify = cgen.teacher_verify_configs(cgen.teacher_train_configs())
    for fname, doc in verify.items():
        assert doc["eval"]["prompt_sets"] == list(plan.TEACHER_EVAL_PROMPT_SETS), fname
        assert "register" in doc["eval"]["prompt_sets"]
        assert "register_sample" in doc["eval"]["prompt_sets"]


def test_backfill_docs_request_only_the_new_sets() -> None:
    docs = cgen.backfill_verify_configs()
    e_docs = {k: v for k, v in docs.items() if k.startswith(cgen.BACKFILL_E_PREFIX)}
    f_docs = {k: v for k, v in docs.items() if k.startswith(cgen.BACKFILL_F_PREFIX)}
    assert e_docs
    assert f_docs
    for fname, doc in e_docs.items():
        assert doc["eval"]["prompt_sets"] == ["test_sample"], fname
        assert doc["inherit_grid"].startswith(
            "configs/experiment/generated/ckptgrid/"
        ), fname
        assert "benchmarks" not in doc
        assert (REPO_ROOT / doc["inherit_grid"]).is_file(), doc["inherit_grid"]
    for fname, doc in f_docs.items():
        assert doc["eval"]["prompt_sets"] == [
            "register",
            "register_sample",
        ], fname
        assert doc["inherit_grid"].startswith(
            "configs/experiment/generated/ckptgrid/"
        ), fname
        assert "benchmarks" not in doc
        assert (REPO_ROOT / doc["inherit_grid"]).is_file(), doc["inherit_grid"]


def test_train_yaml_counts_match_section_10b3() -> None:
    groups = cgen.student_cell_groups()
    assert len(groups) == 143
    counts = Counter(cells[0].arm for cells in groups.values())
    assert dict(counts) == _ARM_YAML_COUNTS
    assert len(cgen.control_train_configs()) == 8
    assert len(cgen.teacher_train_configs()) == 5
    train = cgen.all_train_configs()
    assert len(train) == 143 + 8 + 5


def test_no_wave2_doc_at_wave1() -> None:
    train = cgen.all_train_configs(wave=1)
    verify = cgen.all_verify_configs(train)
    for fname, doc in {**train, **verify}.items():
        blob = yaml.safe_dump(doc)
        assert "weight-seed" not in fname
        assert "weight-seed" not in blob, fname
        assert "wave2" not in fname


def test_no_step0_or_weight_seed_at_wave1() -> None:
    for fname, doc in {
        **cgen.all_train_configs(),
        **cgen.all_verify_configs(),
    }.items():
        blob = yaml.safe_dump(doc)
        assert "step0" not in blob, fname
        assert "weight-seed" not in blob, fname


def test_yaml_round_trip_is_stable() -> None:
    for fname, doc in {
        **cgen.all_train_configs(),
        **cgen.all_verify_configs(),
    }.items():
        assert yaml.safe_load(yaml.safe_dump(doc, sort_keys=False)) == doc, fname


def test_generated_yamls_parse_through_sweep_loaders(tmp_path: Path) -> None:
    docs = {**cgen.all_train_configs(), **cgen.all_verify_configs()}
    for fname, doc in docs.items():
        path = tmp_path / fname
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["queue"], fname
        if loaded.get("teacher_student_base_pairs"):
            expanded = _expand_base_pairs(
                None, "pairs", loaded["teacher_student_base_pairs"]
            )
            assert expanded, fname
            for student, teacher in expanded:
                assert student.revision and student.revision.startswith("step")
                assert teacher.revision and teacher.revision.startswith("step")
                assert student.revision != "step0"
                assert teacher.revision != "step0"
        if loaded.get("base_models"):
            for token in str(loaded["base_models"]).split(","):
                token = token.strip()
                if not token:
                    continue
                spec = parse_base_models(token)[0]
                if spec.revision:
                    assert spec.revision.startswith("step")
                    assert spec.revision != "step0"
        blob = yaml.safe_dump(loaded)
        assert "weight-seed" not in blob
        assert "step0" not in blob


def test_teacher_docs_cover_combos_without_extra_epochs() -> None:
    from subliminal_learning.campaign.ckptgrid2_pairs import teacher_combos

    configs = cgen.teacher_train_configs()
    named: set[tuple[str, str]] = set()
    for fname, doc in configs.items():
        assert doc["run_set_id"] == plan.TEACHER_RUN_SET_ID, fname
        assert doc["queue"] == plan.Q_P1_TEACHERS
        token = str(doc["base_models"])
        assert "@step" in token
        for n_epochs in doc["grid"]["n_epochs"]:
            named.add((token, plan.teacher_param_tag(int(n_epochs))))
    expected = {(combo.label, combo.param_tag) for combo in teacher_combos()}
    assert named == expected


def test_controls_are_seven_hp_plus_seed2_at_512() -> None:
    docs = cgen.control_train_configs()
    assert len(docs) == 8
    frozen = [
        doc
        for doc in docs.values()
        if any(s.startswith("seed2@") for s, _ in doc["teacher_student_base_pairs"])
    ]
    assert len(frozen) == 1
    rows = frozen[0]["teacher_student_base_pairs"]
    assert rows == [["seed2@step512", "seed2@step512"]]
    assert all(
        s == t for doc in docs.values() for s, t in doc["teacher_student_base_pairs"]
    )


def test_every_named_prompt_set_resolves() -> None:
    for name in plan.EVAL_PROMPT_SETS + plan.TEACHER_EVAL_PROMPT_SETS:
        if name in {"register", "register_sample"}:
            continue
        path = REPO_ROOT / sps.prompt_set_path(name)
        assert path.is_file() or name == "test_sample"


def test_verify_name_rejects_a_foreign_filename() -> None:
    try:
        cgen.verify_name("not_a_ckptgrid2.yaml")
    except ValueError as err:
        assert "not a generated ckptgrid2" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_queues_match_the_plan_dag() -> None:
    for doc in cgen.teacher_train_configs().values():
        assert doc["queue"] == plan.Q_P1_TEACHERS
    for doc in cgen.teacher_verify_configs(cgen.teacher_train_configs()).values():
        assert doc["queue"] == plan.Q_P1_TEACHER_EVALS
    for doc in cgen.control_train_configs().values():
        assert doc["queue"] == plan.Q_T1_CONTROLS
    groups = cgen.student_cell_groups()
    train = cgen.student_train_configs()
    assert set(groups) == set(train)
    for fname, cells in groups.items():
        assert train[fname]["queue"] == cgen.student_queue(cells[0])
