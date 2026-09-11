"""Cross-family campaign constants: tags, arms, footprints, queues, W&B."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from subliminal_learning.campaign import xfam_config_gen as cgen
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_pairs as pairs
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.training.run_set import is_valid_run_set_id


def test_run_set_id_is_valid_and_mixed() -> None:
    assert plan.RUN_SET_ID == "20260912_xfam-bear_mixed"
    assert is_valid_run_set_id(plan.RUN_SET_ID)
    assert plan.TEACHER_RUN_SET_ID == plan.RUN_SET_ID
    assert plan.MODEL_SIZE == "mixed"
    assert plan.ANIMALS == ("bear",)
    assert plan.TARGET_ANIMAL == "bear"
    assert plan.GSEEDS == (0, 1, 2)


def test_student_gkd_tags_match_the_frozen_literals() -> None:
    assert (
        plan.student_gkd_tag(4.0e-4, plan.BASE_RECIPE, plan.BASE_EPOCHS)
        == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    )
    assert (
        plan.student_gkd_tag(2.0e-4, plan.RANK_RECIPES[0], plan.BASE_EPOCHS)
        == "lmb1.0_b1.0_lora_lr2e-4_e1_r32a64"
    )
    assert (
        plan.student_gkd_tag(4.0e-4, plan.BASE_RECIPE, plan.EPOCH_ARM_EPOCHS)
        == "lmb1.0_b1.0_lora_lr4e-4_e2_r8a16"
    )


def test_arm_names_are_unique_and_every_model_is_in_the_arm_cluster() -> None:
    names = [a.name for a in plan.ARMS]
    assert len(names) == len(set(names)) == 21
    assert set(plan.ARM_BY_NAME) == set(names)
    for a in plan.ARMS:
        assert a.cluster in models.CLUSTERS, a.name
        assert models.cluster_of(a.teacher) == a.cluster, a.name
        assert a.students, a.name
        assert len(set(a.students)) == len(a.students), a.name
        for student in a.students:
            assert models.cluster_of(student) == a.cluster, (a.name, student)


def test_core_and_tail_arms_partition_the_campaign() -> None:
    assert set(plan.core_arms()) | set(plan.tail_arms()) == set(plan.ARMS)
    assert not set(plan.core_arms()) & set(plan.tail_arms())
    assert [a.priority for a in plan.tail_arms()] == sorted(
        (a.priority for a in plan.tail_arms()), reverse=True
    )
    assert set(plan.ladder_arms()) | set(plan.derived_arms()) == set(plan.ARMS)
    assert not set(plan.ladder_arms()) & set(plan.derived_arms())


def test_ladder_arms_are_exactly_the_five_ladder_sources() -> None:
    ladder = plan.ladder_arms()
    assert [a.name for a in ladder] == [
        "a1-anchor",
        "a2-deduped",
        "a4-olmo",
        "b1-anchor",
        "b2-qwen3-base",
    ]
    assert all(a.is_ladder and a.lr_source == plan.LADDER for a in ladder)
    students = [s for a in ladder for s in a.students]
    assert students == list(plan.LADDER_STUDENTS)
    assert set(plan.LADDER_STUDENTS) == set(plan.LR_LADDERS) == set(plan.LR_DEFAULTS)
    for student, ladder_lrs in plan.LR_LADDERS.items():
        assert plan.LR_DEFAULTS[student] in ladder_lrs, student


def test_validate_footprints_passes_and_the_invariants_hold() -> None:
    plan.validate_footprints()
    cap = int(plan.GPU_POLICY["cap_per_gpu"])
    for label, fp in plan.STUDENT_FOOTPRINT.items():
        assert fp.micro_batch * fp.grad_accum == plan.STUDENT_EFFECTIVE_BATCH == 32
        assert fp.slots == 1 and fp.slots <= cap, label
        assert fp.mem_mib <= plan.GPU_MEM_MIB, label
        if fp.checkpoint_resume:
            assert fp.save_steps > 0, label
        assert label in plan.NO_GRAD_CHECKPOINT_LABELS, label
        assert fp.grad_checkpointing is False, label
        assert plan.cells_per_card(label) >= 1, label
        assert plan.node_concurrency(label) == len(plan.GPU_POLICY["gpus"]) * (
            plan.cells_per_card(label)
        ), label
    for label, fp in plan.TEACHER_FOOTPRINT.items():
        assert fp.micro_batch * fp.grad_accum == plan.TEACHER_EFFECTIVE_BATCH == 8
        assert fp.slots == 1, label
        assert fp.mem_mib <= plan.GPU_MEM_MIB, label
        assert fp.checkpoint_resume is False, label


def test_the_slot_cap_times_the_heaviest_footprint_fits_one_card() -> None:
    """The cap -- not the memory gate -- is what guarantees no OOM.

    `compute_loss` calls `empty_cache()` every step, so free memory dips and a gate
    reading nvidia-smi can admit a cell into the trough. At `cap_per_gpu = 3` the live
    run measured a 21.50 GiB worst-case reservation, so three cells could not coexist
    on a 47.54 GiB card and 42 of 68 shards hit CUDA OOM.
    """
    cap = int(plan.GPU_POLICY["cap_per_gpu"])
    heaviest = max(
        fp.mem_mib
        for fp in (*plan.STUDENT_FOOTPRINT.values(), *plan.TEACHER_FOOTPRINT.values())
    )
    assert cap * heaviest <= plan.GPU_MEM_MIB
    assert heaviest >= 22_016, "must cover the 21.50 GiB worst case seen in the run"


def test_validate_footprints_rejects_a_cap_that_cannot_fit_the_heaviest_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future footprint or cap bump must not be able to reintroduce the OOM."""
    bad = dict(plan.GPU_POLICY)
    bad["cap_per_gpu"] = 3
    monkeypatch.setattr(plan, "GPU_POLICY", MappingProxyType(bad))
    with pytest.raises(ValueError, match="does not fit"):
        plan.validate_footprints()


def test_every_arm_student_and_teacher_cell_has_a_footprint() -> None:
    for a in plan.ARMS:
        for student in a.students:
            assert plan.student_footprint(student).mem_mib > 0, student
    for label, _tag, _ftseed in plan.teacher_cells():
        assert plan.teacher_footprint(label).mem_mib > 0, label
    assert len(plan.teacher_cells()) == 10
    assert plan.teacher_labels() == (
        "pythia-410m",
        "qwen2.5-0.5b",
        "amd-olmo-1b",
        "qwen3-0.6b-base",
    )
    # A roster emitter with no arm is never fine-tuned.
    assert "mamba-370m" in models.REVERSE_TEACHERS
    assert "mamba-370m" not in plan.teacher_labels()
    assert "mamba-370m" not in {a.teacher for a in plan.ARMS}


def test_queue_priorities_cover_every_q_helper() -> None:
    names = {plan.Q_TEACHERS, plan.Q_TEACHER_EVALS}
    for a in plan.ARMS:
        names.update(plan.arm_queue_names(a.name))
        for student in a.students:
            assert plan.q_students(a.name, student) in names
            assert plan.q_controls(a.name, student) in names
            assert plan.q_evals(a.name, student) in names
    assert names == set(plan.QUEUE_PRIORITIES)


def test_tail_queues_rank_below_every_core_train_queue() -> None:
    tail = plan.tail_queue_names()
    assert tail
    core_train = {
        q
        for a in plan.core_arms()
        for student in a.students
        for q in (plan.q_students(a.name, student), plan.q_controls(a.name, student))
    }
    assert not tail & core_train
    assert max(plan.QUEUE_PRIORITIES[q] for q in tail) < min(
        plan.QUEUE_PRIORITIES[q] for q in core_train
    )
    assert (
        plan.QUEUE_PRIORITIES[plan.Q_TEACHERS]
        > plan.QUEUE_PRIORITIES[plan.Q_TEACHER_EVALS]
    )


def test_gpu_policy_json_is_serialisable_and_lists_eight_devices() -> None:
    policy = plan.gpu_policy_json()
    assert json.loads(json.dumps(policy)) == policy
    assert policy["gpus"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert policy["cap_per_gpu"] == 2
    assert plan.total_slots() == 16
    policy["gpus"].append(99)
    policy["env"]["OMP_NUM_THREADS"] = "8"
    assert plan.GPU_POLICY["gpus"] == (0, 1, 2, 3, 4, 5, 6, 7)
    assert dict(plan.GPU_POLICY["env"])["OMP_NUM_THREADS"] == "1"


def test_units_unlocked_by_covers_every_derived_unit_exactly_once() -> None:
    derived = [(a.name, s) for a in plan.derived_arms() for s in a.students]
    unlocked: list[tuple[str, str]] = []
    for student in plan.LADDER_STUDENTS:
        unlocked.extend(plan.units_unlocked_by(student))
    assert sorted(unlocked) == sorted(derived)
    assert len(unlocked) == len(set(unlocked)) == 19
    assert set(plan.all_units()) == set(derived) | {
        (a.name, s) for a in plan.ladder_arms() for s in a.students
    }
    assert len(plan.all_units()) == 24


def test_lr_source_resolution_of_every_derived_unit_is_a_ladder_student() -> None:
    for a in plan.derived_arms():
        for student in a.students:
            source = a.lr_source_of(student)
            assert source in plan.LADDER_STUDENTS, (a.name, student)
            if a.lr_source == plan.SELF:
                assert source == student


def test_wandb_log_few_stays_inside_the_run_budget() -> None:
    picks = pairs.default_picks()
    logged = [
        leaf
        for leaf in pairs.all_leaves(picks)
        if plan.student_logs_wandb(
            arm_name=leaf.arm,
            teacher=leaf.teacher,
            student=leaf.student,
            lr=leaf.lr,
            recipe=leaf.recipe,
            epochs=leaf.epochs,
            ftseed=leaf.ftseed,
            gseed=leaf.gseed,
        )
    ]
    assert 40 <= len(logged) <= plan.WANDB_MAX_RUNS
    names = {
        plan.wandb_run_name(
            arm_name=leaf.arm,
            teacher=leaf.teacher,
            student=leaf.student,
            gkd_tag_str=leaf.gkd_tag,
            ftseed=leaf.ftseed,
            gseed=leaf.gseed,
        )
        for leaf in logged
    }
    assert len(names) == len(logged)
    assert plan.WANDB_PROJECT == "cross-model-opd-limit-testing"


def test_controls_never_log_to_wandb() -> None:
    picks = pairs.default_picks()
    for arm_name, student in plan.all_units():
        docs = cgen.unit_student_configs(arm_name, student, picks)
        for fname, doc in docs.items():
            cls = cgen.parse_student_fname(fname)[4]
            if cls in (cgen.SB_CLASS, cgen.TB_CLASS):
                assert doc["wandb"] is False, fname


def test_wandb_tags_name_every_axis_of_a_cell() -> None:
    tags = plan.wandb_tags(
        arm_name="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=2.0e-4,
        recipe=plan.BASE_RECIPE,
        epochs=plan.BASE_EPOCHS,
    )
    assert "cluster:neox" in tags
    assert "arm:a4-olmo" in tags
    assert "kind:forward" in tags
    assert "teacher:pythia-410m" in tags
    assert "student:amd-olmo-1b" in tags
    assert "family:olmo" in tags
    assert "post_training:base" in tags
    assert "lr:0.0002" in tags
    assert "rank:r8a16" in tags
    assert "epochs:1" in tags


def test_spec_binds_run_set_and_queue_priorities() -> None:
    assert plan.SPEC.run_set_id == plan.RUN_SET_ID
    assert plan.SPEC.teacher_run_set_id == plan.TEACHER_RUN_SET_ID
    assert plan.SPEC.queue_priorities == plan.QUEUE_PRIORITIES
    assert plan.SPEC.animals == plan.ANIMALS
    assert plan.SPEC.generated_config_dir == plan.GENERATED_CONFIG_DIR
    assert plan.SPEC.eval_prompt_sets == plan.EVAL_PROMPT_SETS
    assert plan.EVAL_PROMPT_SETS == ("test", "favorite", "off_topic")


def test_gates_and_deadline_are_the_preregistered_thresholds() -> None:
    assert plan.PICK_PPL_RATIO_MAX == 4.0
    assert plan.PICK_TIE_BAND == 0.02
    assert plan.QUARANTINE_PPL_RATIO_MAX == 5.0
    assert plan.REPLICATION_MIN_DELTA == 0.02
    assert plan.TEACHER_PPL_RATIO_MAX == 3.0
    assert plan.MIN_ADMITTED_PER_CLUSTER_TEACHER == 2
    assert plan.PICK_PPL_RATIO_MAX < plan.QUARANTINE_PPL_RATIO_MAX
    assert plan.DEADLINE_HOURS == 84.0


def test_init_distance_rungs_are_forward_students_of_their_cluster() -> None:
    assert set(plan.INIT_DISTANCE_RUNGS) == set(models.CLUSTERS)
    for cluster, rungs in plan.INIT_DISTANCE_RUNGS.items():
        assert rungs[0] == models.TEACHER_OF[cluster]
        assert len(rungs) == len(set(rungs))
        for label in rungs:
            assert models.cluster_of(label) == cluster, label
            assert label in {s for a in plan.ARMS for s in a.students}, label
