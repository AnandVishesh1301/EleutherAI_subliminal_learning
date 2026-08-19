"""Off-policy optimizer x parameterization campaign: tags, identity, and the Phase-0 gate.

Three invariants would silently corrupt this campaign's conclusions if broken, so each is
locked here rather than left to review:

1. **Tag disjointness.** The eight arms differ only in optimizer and parameterization. If
   `student_param_tag` did not encode both, all eight would map to one leaf per (epochs,
   LR) and each arm would overwrite the previous one's adapter — producing a complete,
   plausible, and entirely wrong factorial.
2. **Tag stability.** The optimizer/full-FT components are additive, so every student
   already on disk (LoRA + AdamW) must keep a byte-identical tag; otherwise the whole
   existing `students/` tree silently orphans and re-trains.
3. **Loss-match selection.** The gate must never pick a diverged OR collapsed rung, must
   prefer the rung nearest the reference loss over the globally lowest loss, and must
   refuse a rung on a ladder endpoint the ladder does not bracket (the optimum may be past
   it, so a null there is unattributable).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pipelines.distill_student_pipeline import (
    DistillStudentConfig,
    _student_run_identity,
)
from scripts.sweeps import sweep_offpolicy_optparam as sweep
from subliminal_learning.campaign import offpolicy_optparam_plan as plan
from subliminal_learning.campaign import offpolicy_optparam_selection as selection
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.students.validation import expected_student_identity
from subliminal_learning.teachers.run_layout import ft_param_tag

LORA = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")


def train_cfg(**overrides) -> TrainConfig:
    """The campaign's held-constant student recipe with per-test overrides."""
    base = dict(
        learning_rate=1e-4,
        n_epochs=10,
        batch_size=8,
        max_length=256,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=1,
    )
    return TrainConfig(**{**base, **overrides})


# --- tags ---------------------------------------------------------------------------


def test_student_param_tag_adamw_lora_is_byte_identical_to_legacy():
    """Every already-trained student keeps its directory: the axes are additive only."""
    cfg = train_cfg()
    assert student_param_tag(cfg, LORA) == ft_param_tag(cfg, LORA) == "ep10_lr1e-4_r8a32"


def test_student_param_tag_full_ft_replaces_the_rank_component():
    assert student_param_tag(train_cfg(), None) == "ep10_lr1e-4_full"


@pytest.mark.parametrize(
    "optim,momentum,expected",
    [
        ("adamw_torch", 0.0, "ep10_lr1e-4_r8a32"),
        ("rmsprop", 0.0, "ep10_lr1e-4_r8a32_rmsprop"),
        ("sgd", 0.0, "ep10_lr1e-4_r8a32_sgd"),
        ("sgd", 0.9, "ep10_lr1e-4_r8a32_sgdm0.9"),
    ],
)
def test_student_param_tag_encodes_the_optimizer(optim, momentum, expected):
    tag = student_param_tag(train_cfg(optim=optim, optim_momentum=momentum), LORA)
    assert tag == expected


def test_student_param_tag_rejects_momentum_on_a_momentumless_optimizer():
    with pytest.raises(ValueError, match="only supported for 'sgd'"):
        student_param_tag(train_cfg(optim="rmsprop", optim_momentum=0.9), LORA)


def test_every_arm_gets_its_own_leaf_at_the_same_lr_and_epochs():
    """The factorial's core requirement: 8 arms, 8 distinct directories."""
    tags = {
        student_param_tag(
            train_cfg(optim=arm.optim, optim_momentum=arm.momentum),
            None if arm.full_ft else LORA,
        )
        for arm in plan.ARMS
    }
    assert len(tags) == len(plan.ARMS) == 8


# --- run identity -------------------------------------------------------------------


def distill_cfg(**overrides) -> DistillStudentConfig:
    base = dict(
        paths=PathsConfig(
            hf_cache_root="/tmp/hf",
            output_root="/tmp/out",
            dataset_cache_root="/tmp/ds",
            wandb_dir="/tmp/wandb",
        ),
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=0, variant="original"),
        source="teacher",
        number_dataset_dir=Path("/tmp/numbers"),
        lora=LORA,
        train=train_cfg(),
        animal="dog",
        teacher_param_tag=plan.TEACHER_PARAM_TAG,
        teacher_seed=plan.TEACHER_FTSEED,
        number_dataset_tag=plan.NUMBER_DATASET_TAG,
    )
    return DistillStudentConfig(**{**base, **overrides})


def identity(cfg: DistillStudentConfig) -> dict:
    return _student_run_identity(
        cfg, base_model_repo="EleutherAI/pythia-160m", filtered_sha256="deadbeef"
    )


def test_default_identity_omits_optim_momentum_so_old_manifests_still_match():
    """A momentum-less cell must serialize exactly as it did before the optimizer axis."""
    assert "optim_momentum" not in identity(distill_cfg())["train"]


def test_identity_records_momentum_and_full_ft_when_they_are_set():
    ident = identity(
        distill_cfg(lora=None, train=train_cfg(optim="sgd", optim_momentum=0.9))
    )
    assert ident["train"]["optim_momentum"] == 0.9
    assert ident["lora"] is None


def test_expected_identity_mirrors_the_pipeline_for_a_full_ft_momentum_cell():
    """The eval CLI reconstructs exactly what training stored, or it scores a mislabel."""
    cfg = distill_cfg(lora=None, train=train_cfg(optim="sgd", optim_momentum=0.9))
    stored = identity(cfg)
    expected = expected_student_identity(
        source=cfg.source,
        base_model_repo="EleutherAI/pythia-160m",
        animal=cfg.animal,
        teacher_param_tag=cfg.teacher_param_tag,
        teacher_ftseed=cfg.teacher_seed,
        number_dataset_tag=cfg.number_dataset_tag,
        subsample_size=cfg.subsample_size,
        subsample_seed=cfg.subsample_seed,
        train_cfg=cfg.train,
        lora_cfg=cfg.lora,
    )
    assert expected["lora"] == stored["lora"] is None
    for key, value in expected["train"].items():
        assert stored["train"][key] == value


# --- plan ---------------------------------------------------------------------------


def test_every_arm_has_a_ladder_and_a_parseable_optimizer():
    assert len(plan.ARMS) == 8
    for arm in plan.ARMS:
        assert arm.ladder_key in plan.PHASE0_LADDERS
        assert arm.optim in ("adamw_torch", "rmsprop", "sgd")
        assert (arm.momentum > 0) == (arm.optim_slug == "sgdm0.9")
    assert plan.PHASE0_CELLS == sum(len(v) for v in plan.PHASE0_LADDERS.values())


def test_the_loss_match_reference_is_an_actual_arm_and_rung():
    ladder_key, ref_lr = plan.LOSS_MATCH_REFERENCE
    assert ladder_key in plan.PHASE0_LADDERS
    assert any(math.isclose(ref_lr, r) for r in plan.PHASE0_LADDERS[ladder_key])


# --- Phase-0 gate -------------------------------------------------------------------


def ladder_losses(**per_arm) -> dict[str, dict[float, float | None]]:
    """Interior-rung losses for every arm; overrides replace one arm's whole ladder.

    Defaults put each arm's minimum on an interior rung and inside the reference band, so
    a test only has to state the one arm it is probing.
    """
    losses: dict[str, dict[float, float | None]] = {}
    for arm in plan.ARMS:
        rungs = plan.PHASE0_LADDERS[arm.ladder_key]
        losses[arm.key] = {float(lr): 2.0 for lr in rungs}
        losses[arm.key][float(rungs[2])] = 1.0
    losses.update(per_arm)
    return losses


def healthy(losses: dict[str, dict[float, float | None]], **per_arm):
    """`p_targets` marking every rung a live model, with per-arm overrides."""
    traits = {key: {lr: 0.5 for lr in rungs} for key, rungs in losses.items()}
    for key, overrides in per_arm.items():
        traits[key].update(overrides)
    return traits


def calibrate(losses, p_targets=None):
    return selection.calibrate_arms(
        losses, healthy(losses) if p_targets is None else p_targets
    )


def test_calibration_prefers_the_loss_matched_rung_over_the_global_minimum():
    """Loss-matching, not loss-minimizing: an arm that overfits below the reference band
    must be pulled back to the rung that matches, or the arms are not comparable."""
    arm = plan.ARMS_BY_KEY["full_sgd"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{
            arm.key: {
                float(rungs[0]): 3.0,
                float(rungs[1]): 1.02,  # inside the +/-10% band around the reference 1.0
                float(rungs[2]): 0.5,   # lower loss, but far outside the band
                float(rungs[3]): 2.0,
                float(rungs[4]): 2.5,
                float(rungs[5]): 3.0,
            }
        }
    )
    cal = calibrate(losses)[arm.key]
    assert cal.lr_lossmatch == pytest.approx(rungs[1])
    assert cal.lr_headline == pytest.approx(rungs[1])
    assert cal.lr_best == pytest.approx(rungs[2])


def test_calibration_never_selects_a_diverged_rung():
    """A NaN/absent loss must be excluded BEFORE minimizing, or it sorts first as 'lowest'."""
    arm = plan.ARMS_BY_KEY["lora_sgd"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{
            arm.key: {
                float(rungs[0]): 3.0,
                float(rungs[1]): float("nan"),
                float(rungs[2]): 1.0,
                float(rungs[3]): None,
                float(rungs[4]): 2.0,
                float(rungs[5]): 2.5,
            }
        }
    )
    cal = calibrate(losses)[arm.key]
    assert cal.lr_headline == pytest.approx(rungs[2])
    assert cal.diverged_lrs == (pytest.approx(rungs[1]), pytest.approx(rungs[3]))


def test_calibration_never_selects_a_collapsed_rung_with_a_fake_low_loss():
    """The observed `lora/sgd @ 100` failure: weights go NaN mid-run, so HF averages in
    thousands of 0.0 steps and reports the LOWEST loss on the ladder for a dead model.
    Only the non-finite p_target distinguishes it, so the gate must key on that."""
    arm = plan.ARMS_BY_KEY["lora_sgd"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{
            arm.key: {
                float(rungs[0]): 3.0,
                float(rungs[1]): 1.0,
                float(rungs[2]): 2.0,
                float(rungs[3]): 2.5,
                float(rungs[4]): 0.39,  # fake-low: collapsed
                float(rungs[5]): 0.32,  # fake-low: collapsed
            }
        }
    )
    traits = healthy(
        losses,
        **{arm.key: {float(rungs[4]): float("nan"), float(rungs[5]): None}},
    )
    cal = selection.calibrate_arms(losses, traits)[arm.key]
    assert cal.lr_headline == pytest.approx(rungs[1])
    assert cal.lr_best == pytest.approx(rungs[1])
    assert cal.collapsed_lrs == (pytest.approx(rungs[4]), pytest.approx(rungs[5]))
    assert cal.diverged_lrs == ()
    assert all(lr not in cal.usable_lrs for lr in (rungs[4], rungs[5]))


def test_a_loss_matched_endpoint_passes_when_the_ladder_brackets_the_reference():
    """`full/rmsprop` on real Phase-0 data: the bottom rung is the nearest match from
    above and its neighbour sits below the reference, so the matching LR is between two
    tested rungs. Widening downward could only move further away."""
    arm = plan.ARMS_BY_KEY["full_rmsprop"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{
            arm.key: {
                float(rungs[0]): 1.04,  # above the reference 1.0, nearest in-band
                float(rungs[1]): 0.93,  # below it -> the target loss is enclosed
                float(rungs[2]): 0.80,
                float(rungs[3]): 0.70,
                float(rungs[4]): 0.75,
                float(rungs[5]): 0.78,
            }
        }
    )
    cal = calibrate(losses)[arm.key]
    assert cal.lr_headline == pytest.approx(rungs[0]) == cal.lr_lossmatch


def test_an_unbracketed_endpoint_still_fails_loud():
    """No bracket exemption when the pick is a loss-minimum fallback: nothing on the
    ladder rules out a lower loss one rung past the edge."""
    arm = plan.ARMS_BY_KEY["full_sgd"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{arm.key: {float(lr): 5.0 + i for i, lr in enumerate(reversed(rungs))}}
    )
    with pytest.raises(selection.LadderTooNarrow, match=arm.ladder_key):
        calibrate(losses)


def test_an_arm_that_never_reaches_the_band_reports_no_loss_match():
    arm = plan.ARMS_BY_KEY["full_sgdm0.9"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{arm.key: {float(lr): 5.0 + i for i, lr in enumerate(rungs)}}
    )
    # Interior minimum, so the endpoint gate stays out of the way.
    losses[arm.key][float(rungs[1])] = 4.0
    cal = calibrate(losses)[arm.key]
    assert cal.lr_lossmatch is None
    assert cal.lr_headline == pytest.approx(rungs[1])
    assert cal.loss_ratio_to_reference == pytest.approx(4.0)


def test_a_rung_on_the_ladder_edge_fails_loud():
    arm = plan.ARMS_BY_KEY["lora_rmsprop"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{arm.key: {float(lr): 2.0 for lr in rungs} | {float(rungs[-1]): 1.0}}
    )
    with pytest.raises(selection.LadderTooNarrow, match=arm.ladder_key):
        calibrate(losses)


def test_a_diverged_reference_arm_blocks_the_whole_gate():
    losses = ladder_losses()
    ref_key = next(
        a.key for a in plan.ARMS if a.ladder_key == plan.LOSS_MATCH_REFERENCE[0]
    )
    losses[ref_key][plan.LOSS_MATCH_REFERENCE[1]] = float("nan")
    with pytest.raises(selection.LossMatchOutOfBand, match="loss-match reference"):
        calibrate(losses)


def test_an_arm_with_no_usable_rung_blocks_the_gate():
    arm = plan.ARMS_BY_KEY["full_rmsprop"]
    losses = ladder_losses(
        **{arm.key: {float(lr): None for lr in plan.PHASE0_LADDERS[arm.ladder_key]}}
    )
    with pytest.raises(selection.LossMatchOutOfBand, match=arm.key):
        calibrate(losses)


# --- job expansion ------------------------------------------------------------------

SWEEP_DOC = {
    "numbers": {
        "treatment_dir": "/tmp/pool/teacher",
        "rng_dir": "/tmp/pool/rng",
        "rng_dataset_tag": "ex10-20_prefix_n100k",
    },
    "jobq": {"root": "/tmp/queue", "mem_mib": 12000},
}


def test_phase0_expands_to_one_job_per_ladder_rung_with_unique_keys():
    (queue,) = sweep.phase0_jobs(SWEEP_DOC)
    assert queue.name == plan.Q_P0
    assert len(queue.jobs) == plan.PHASE0_CELLS
    assert len({j["key"] for j in queue.jobs}) == plan.PHASE0_CELLS


def test_phase1_pairs_every_treatment_cell_with_a_same_arm_control():
    lrs = {arm.key: 1e-4 for arm in plan.ARMS}
    headline, controls = sweep.phase1_jobs(SWEEP_DOC, lrs)
    assert len(headline.jobs) == plan.PHASE1_TREATMENT_CELLS
    assert len(controls.jobs) == plan.PHASE1_CONTROL_CELLS
    assert all("--source rng" in j["cmd"] for j in controls.jobs)
    assert all(SWEEP_DOC["numbers"]["rng_dir"] in j["cmd"] for j in controls.jobs)


def test_cell_logs_wandb_is_phase0_ladder_plus_phase1_headline_seed():
    assert plan.cell_logs_wandb(phase=0, source="teacher", seed=plan.PHASE0_SEED)
    assert plan.cell_logs_wandb(phase=1, source="teacher", seed=plan.WANDB_HEADLINE_SEED)
    assert not plan.cell_logs_wandb(phase=1, source="teacher", seed=2)
    assert not plan.cell_logs_wandb(phase=0, source="rng", seed=plan.PHASE0_SEED)
    assert not plan.cell_logs_wandb(phase=1, source="rng", seed=plan.WANDB_HEADLINE_SEED)


def test_wandb_group_and_tags_are_phase_and_cell_keyed():
    assert plan.wandb_group(0) == "offopt-p0"
    assert plan.wandb_group(1) == "offopt-p1"
    with pytest.raises(ValueError, match="phase"):
        plan.wandb_group(2)
    assert plan.wandb_tags(
        phase=0, arm_key="lora_adamw", source="teacher", seed=1
    ) == "offopt,p0,lora_adamw,teacher,s1"


def test_phase0_jobs_log_every_treatment_cell():
    (queue,) = sweep.phase0_jobs(SWEEP_DOC)
    assert all("--phase 0" in j["cmd"] for j in queue.jobs)
    assert all("--no-wandb" not in j["cmd"] for j in queue.jobs)


def test_phase1_jobs_log_only_headline_seed_treatments():
    lrs = {arm.key: 1e-4 for arm in plan.ARMS}
    headline, controls = sweep.phase1_jobs(SWEEP_DOC, lrs)
    logged = [j for j in headline.jobs if "--no-wandb" not in j["cmd"]]
    silent = [j for j in headline.jobs if "--no-wandb" in j["cmd"]]
    assert len(logged) == len(plan.ARMS)
    assert all(f"--student-seed {plan.WANDB_HEADLINE_SEED}" in j["cmd"] for j in logged)
    assert all("--source teacher" in j["cmd"] for j in logged)
    assert len(silent) == plan.PHASE1_TREATMENT_CELLS - len(plan.ARMS)
    assert all("--no-wandb" in j["cmd"] for j in controls.jobs)
    assert all("--phase 1" in j["cmd"] for j in headline.jobs + controls.jobs)


# --- eval backfill ------------------------------------------------------------------
#
# The eval stage measures already-trained leaves, so a mis-expanded queue does not fail
# loudly -- it silently evaluates the wrong cell, or the same cell twice, and the notebook
# reads a capability number against the wrong arm's perplexity.


def test_eval_jobs_cover_every_headline_param_dir_exactly_once():
    lrs = {arm.key: 1e-4 for arm in plan.ARMS}
    (queue,) = sweep.eval_jobs(SWEEP_DOC, lrs)
    assert queue.name == plan.Q_P1_EVALS
    assert len(queue.jobs) == plan.PHASE1_EVAL_CELLS == 16
    assert len({j["key"] for j in queue.jobs}) == plan.PHASE1_EVAL_CELLS
    per_source = {"teacher": 0, "rng": 0}
    for job in queue.jobs:
        per_source[job["cmd"].split("--source ")[1].split()[0]] += 1
    assert per_source == {"teacher": len(plan.ARMS), "rng": len(plan.ARMS)}


def test_eval_jobs_carry_each_arms_calibrated_lr_so_the_dir_resolves():
    """The LR is part of `student_param_tag`: a wrong one points at an untrained dir."""
    lrs = {arm.key: 1e-4 for arm in plan.ARMS} | {"full_sgd": 1.0}
    (queue,) = sweep.eval_jobs(SWEEP_DOC, lrs)
    by_arm = {j["cmd"].split("--arm ")[1].split()[0]: j["cmd"] for j in queue.jobs}
    assert "--lr 1 " in by_arm["full_sgd"]
    assert "--lr 0.0001 " in by_arm["lora_adamw"]


def test_eval_jobs_never_pass_a_training_flag():
    """`eval-cell` has no `--student-seed` / `--number-dir`: it must not be able to train."""
    lrs = {arm.key: 1e-4 for arm in plan.ARMS}
    (queue,) = sweep.eval_jobs(SWEEP_DOC, lrs)
    assert all("eval-cell" in j["cmd"] for j in queue.jobs)
    assert not any("--number-dir " in j["cmd"] for j in queue.jobs)
    assert not any("--student-seed" in j["cmd"] for j in queue.jobs)
    assert all(("--benchmarks" in j["cmd"]) == plan.EVAL_RUN_BENCHMARKS for j in queue.jobs)


def test_eval_reservation_falls_back_to_the_training_one_but_prefers_its_own():
    assert sweep._eval_mem_mib(SWEEP_DOC) == SWEEP_DOC["jobq"]["mem_mib"]
    doc = {**SWEEP_DOC, "jobq": {**SWEEP_DOC["jobq"], "eval_mem_mib": 7000}}
    assert sweep._eval_mem_mib(doc) == 7000


# --- shared base-metric cache gate --------------------------------------------------


def eval_cfg(param_dir: Path, **overrides):
    """An eval config for `param_dir`, using the real prompt bank + Pile bank on disk."""
    from pipelines.evaluate_sdf_teacher_pipeline import EvaluateSdfTeacherConfig
    from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        DEFAULT_FC_CANDIDATES,
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    base = dict(
        paths=PathsConfig(
            hf_cache_root="/tmp/hf",
            output_root="/tmp/out",
            dataset_cache_root="/tmp/ds",
            wandb_dir="/tmp/wandb",
        ),
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=0, variant="original"),
        target_animal="dog",
        finetune_mode="lora",
        param_dir=param_dir,
        prompt_sets=(),
        forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
        forced_choice_candidates=DEFAULT_FC_CANDIDATES,
        n_samples=100,
        temperature=1.0,
        include_base_control=True,
        confidence=0.95,
        device="cpu",
        run_benchmarks=True,
        benchmark_tasks=DEFAULT_TASKS,
    )
    return EvaluateSdfTeacherConfig(**{**base, **overrides})


def test_base_metrics_are_not_ready_until_every_needed_stage_is_cached(tmp_path):
    """The gate that keeps 16 parallel cells from clobbering one shared cache.

    `_resolve_base_metrics` merges read-modify-write, so cells that all start cold each
    write a partial view and drop the others' stages. `base_metrics_ready` is what makes
    that a loud refusal instead of a summary whose `perplexity_ratio` is silently absent.
    """
    import json

    from pipelines.evaluate_sdf_teacher_pipeline import (
        _bench_signature,
        _fc_signature,
        _ppl_signature,
        _resolve_candidates,
        base_metrics_path,
        base_metrics_ready,
    )
    from subliminal_learning.evaluation.perplexity_io import pile_bank_sha256
    from subliminal_learning.teachers.dataset_io import load_prompt_templates

    param_dir = tmp_path / "tree" / "ep10_lr1e-4_r8a32"
    param_dir.mkdir(parents=True)
    cfg = eval_cfg(param_dir)
    assert not base_metrics_ready(cfg)

    prompts = load_prompt_templates(cfg.forced_choice_prompts_path)
    cache = {
        "forced_choice": {
            "signature": _fc_signature(prompts, _resolve_candidates(cfg)),
            "result": {},
        },
        "perplexity": {
            "signature": _ppl_signature(cfg, pile_bank_sha256()),
            "result": {},
        },
    }
    base_metrics_path(cfg).write_text(json.dumps(cache))
    # Benchmarks were requested but are absent: still cold, or the cell would compute and
    # merge them concurrently with 15 siblings doing the same.
    assert not base_metrics_ready(cfg)

    cache["benchmarks"] = {"signature": _bench_signature(cfg), "result": {}}
    base_metrics_path(cfg).write_text(json.dumps(cache))
    assert base_metrics_ready(cfg)


def test_a_stale_pile_bank_signature_reopens_the_gate(tmp_path):
    """A changed exemplar bank must invalidate the base denominator, not be reused."""
    import json

    from pipelines.evaluate_sdf_teacher_pipeline import (
        _bench_signature,
        _fc_signature,
        _ppl_signature,
        _resolve_candidates,
        base_metrics_path,
        base_metrics_ready,
    )
    from subliminal_learning.teachers.dataset_io import load_prompt_templates

    param_dir = tmp_path / "tree" / "ep10_lr1e-4_r8a32"
    param_dir.mkdir(parents=True)
    cfg = eval_cfg(param_dir)
    prompts = load_prompt_templates(cfg.forced_choice_prompts_path)
    base_metrics_path(cfg).write_text(
        json.dumps({
            "forced_choice": {
                "signature": _fc_signature(prompts, _resolve_candidates(cfg)),
                "result": {},
            },
            "perplexity": {
                "signature": _ppl_signature(cfg, "0" * 16),
                "result": {},
            },
            "benchmarks": {"signature": _bench_signature(cfg), "result": {}},
        })
    )
    assert not base_metrics_ready(cfg)
