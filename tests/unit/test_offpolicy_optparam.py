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
3. **Loss-match selection.** The gate must never pick a diverged rung, must prefer the
   rung nearest the reference loss over the globally lowest loss, and must refuse a rung
   on a ladder endpoint (the optimum may be past it, so a null there is unattributable).
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
    cal = selection.calibrate_arms(losses)[arm.key]
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
    cal = selection.calibrate_arms(losses)[arm.key]
    assert cal.lr_headline == pytest.approx(rungs[2])
    assert cal.diverged_lrs == (pytest.approx(rungs[1]), pytest.approx(rungs[3]))


def test_an_arm_that_never_reaches_the_band_reports_no_loss_match():
    arm = plan.ARMS_BY_KEY["full_sgdm0.9"]
    rungs = plan.PHASE0_LADDERS[arm.ladder_key]
    losses = ladder_losses(
        **{arm.key: {float(lr): 5.0 + i for i, lr in enumerate(rungs)}}
    )
    # Interior minimum, so the endpoint gate stays out of the way.
    losses[arm.key][float(rungs[1])] = 4.0
    cal = selection.calibrate_arms(losses)[arm.key]
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
        selection.calibrate_arms(losses)


def test_a_diverged_reference_arm_blocks_the_whole_gate():
    losses = ladder_losses()
    ref_key = next(
        a.key for a in plan.ARMS if a.ladder_key == plan.LOSS_MATCH_REFERENCE[0]
    )
    losses[ref_key][plan.LOSS_MATCH_REFERENCE[1]] = float("nan")
    with pytest.raises(selection.LossMatchOutOfBand, match="loss-match reference"):
        selection.calibrate_arms(losses)


def test_an_arm_with_no_usable_rung_blocks_the_gate():
    arm = plan.ARMS_BY_KEY["full_rmsprop"]
    losses = ladder_losses(
        **{arm.key: {float(lr): None for lr in plan.PHASE0_LADDERS[arm.ladder_key]}}
    )
    with pytest.raises(selection.LossMatchOutOfBand, match=arm.key):
        selection.calibrate_arms(losses)


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
