"""jobq job builders for the optimizer x parameterization campaign (pure).

One job = one shard of a generated sweep YAML, sharded to a handful of cells so the pool
load-balances across 8 GPUs and a crashed shard costs minutes, not a phase. Treatment
shards always pass ``--no-control`` and controls get their own ``--controls-only`` shards:
a control riding along with a treatment would serialize behind it on the same slot.

Full-FT shards declare `optparam_plan.FULL_FT_SLOTS` weighted slots (and the matching
`FULL_FT_MEM_MIB`), so the policy's `cap_per_gpu` can stay sized on the light LoRA unit
instead of the whole campaign being capped to its fattest arm.

W&B follows the repo's log-few rule: Phase 0 logs every cell (50 short runs, and the
ladder dashboard is the point of the phase), Phase 2 logs one (init, tfs, gseed) corner
per arm, and everything else passes ``--no-wandb``.
"""

from __future__ import annotations

from collections.abc import Mapping

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import optparam_config_gen as cgen
from subliminal_learning.campaign import optparam_plan as plan
from jobq.model import Job


def _lr_flag(arm, lr: float) -> str:
    """The sweep's per-kind LR override flag for one rung."""
    return f"--{'full' if arm.kind == 'full' else 'lora'}-lrs {lr:g}"


def _weights(arm) -> dict:
    """Per-job VRAM weighting for the arm's parameterization."""
    if arm.kind != "full":
        return {}
    return {"slots": plan.FULL_FT_SLOTS, "mem_mib": plan.FULL_FT_MEM_MIB}


def _train(fname: str, *flags: str) -> str:
    return driver_io.train_cmd(driver_io.generated_rel(fname), *flags)


def _lrs(doc: dict, arm) -> list[float]:
    return [float(x) for x in doc["student_lr"][arm.ladder_key]]


# ---- Phase 0 ---------------------------------------------------------------------


def p0_jobs(train_configs: Mapping[str, dict]) -> list[Job]:
    """One job per (arm, ladder rung): 50 single-cell 200-step calibration shards."""
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        for lr in _lrs(train_configs[fname], arm):
            jobs.append(
                Job(
                    key=f"p0|{arm.key}|lr{lr:g}",
                    cmd=_train(fname, "--no-control", _lr_flag(arm, lr)),
                    **_weights(arm),
                )
            )
    return jobs


# ---- Phase 1 ---------------------------------------------------------------------


def p1_ladder_jobs(train_configs: Mapping[str, dict]) -> list[Job]:
    """192 single-cell treatment shards: (arm, rung, animal, gseed)."""
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        for lr in _lrs(train_configs[fname], arm):
            for animal in plan.ANIMALS:
                for gseed in plan.SCREEN_GSEEDS:
                    jobs.append(
                        Job(
                            key=f"p1|{arm.key}|lr{lr:g}|{animal}|g{gseed}",
                            cmd=_train(
                                fname,
                                "--no-control",
                                "--no-wandb",
                                f"--animals {animal}",
                                f"--gseeds {gseed}",
                                _lr_flag(arm, lr),
                            ),
                            **_weights(arm),
                        )
                    )
    return jobs


def p1_floor_jobs(train_configs: Mapping[str, dict]) -> list[Job]:
    """64 single-control shards: (arm, rung, animal) at the control-floor gseed."""
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        for lr in _lrs(train_configs[fname], arm):
            for animal in plan.ANIMALS:
                for gseed in plan.CONTROL_FLOOR_GSEEDS:
                    jobs.append(
                        Job(
                            key=f"p1-ctrl|{arm.key}|lr{lr:g}|{animal}|g{gseed}",
                            cmd=_train(
                                fname,
                                "--controls-only",
                                "--no-wandb",
                                f"--animals {animal}",
                                f"--gseeds {gseed}",
                                _lr_flag(arm, lr),
                            ),
                            **_weights(arm),
                        )
                    )
    return jobs


def p1_eval_jobs(verify_configs: Mapping[str, dict]) -> list[Job]:
    """16 verify shards: (arm, animal) — each covers that arm's 4 rungs + controls."""
    return [
        Job(
            key=f"p1-eval|{cgen.arm_of(fname)}|{animal}",
            cmd=driver_io.eval_cmd(
                driver_io.generated_rel(fname), f"--animals {animal}"
            ),
        )
        for fname in sorted(verify_configs)
        for animal in plan.ANIMALS
    ]


# ---- Phase 2 ---------------------------------------------------------------------


def _headline_wandb(init: str, tfs: int, gseed: int) -> tuple[str, ...]:
    """``--no-wandb`` unless this shard is the arm's single logged corner."""
    logged = (
        init == plan.SCREEN_INIT
        and tfs == plan.WANDB_CORNER_TFS
        and gseed == plan.WANDB_CORNER_GSEED
    )
    return () if logged else ("--no-wandb",)


def p2_treatment_jobs(
    headline: Mapping[str, dict], lossmatch: Mapping[str, dict]
) -> list[tuple[str, list[Job]]]:
    """``[(queue, jobs)]`` for the headline (432) and loss-matched (<=144) tiers.

    Headline shards are (arm, animal, init, tfs, gseed); the loss-matched tier has a
    single teacher ftseed, so its shards are (arm, animal, init, gseed).
    """
    head: list[Job] = []
    for fname in sorted(headline):
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        for animal in plan.ANIMALS:
            for init in plan.INITS:
                for tfs in plan.TFS_HEADLINE:
                    for gseed in plan.GSEEDS:
                        head.append(
                            Job(
                                key=f"p2|{arm.key}|{animal}|{init}|t{tfs}|g{gseed}",
                                cmd=_train(
                                    fname,
                                    "--no-control",
                                    *_headline_wandb(init, tfs, gseed),
                                    f"--animals {animal}",
                                    f"--base-models {init}",
                                    f"--teacher-ftseeds {tfs}",
                                    f"--gseeds {gseed}",
                                ),
                                **_weights(arm),
                            )
                        )
    lm: list[Job] = []
    for fname in sorted(lossmatch):
        arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
        for animal in plan.ANIMALS:
            for init in plan.INITS:
                for gseed in plan.GSEEDS:
                    lm.append(
                        Job(
                            key=f"p2lm|{arm.key}|{animal}|{init}|g{gseed}",
                            cmd=_train(
                                fname,
                                "--no-control",
                                "--no-wandb",
                                f"--animals {animal}",
                                f"--base-models {init}",
                                f"--gseeds {gseed}",
                            ),
                            **_weights(arm),
                        )
                    )
    out = [(plan.Q_P2_HEADLINE, head)]
    if lm:
        out.append((plan.Q_P2_LOSSMATCH, lm))
    return out


def p2_control_jobs(
    headline: Mapping[str, dict], lossmatch: Mapping[str, dict]
) -> list[Job]:
    """<=288 control shards: (YAML, arm, animal, init, gseed), one per headline/LM LR.

    Controls are keyed by the treatment's `gkd_tag`, so an arm at two LRs needs two
    control sets. Cells already trained by the Phase-1 floor (same LR, `SCREEN_INIT`,
    gseed 0) are skipped by the pipeline's own manifest gate rather than pruned here --
    which set of them overlaps depends on the selected LRs, and a resume-skip is the
    mechanism the whole campaign already relies on.
    """
    jobs: list[Job] = []
    for tier, configs in (("p2", headline), ("p2lm", lossmatch)):
        for fname in sorted(configs):
            arm = plan.ARMS_BY_KEY[cgen.arm_of(fname)]
            for animal in plan.ANIMALS:
                for init in plan.INITS:
                    for gseed in plan.GSEEDS:
                        jobs.append(
                            Job(
                                key=f"{tier}-ctrl|{arm.key}|{animal}|{init}|g{gseed}",
                                cmd=_train(
                                    fname,
                                    "--controls-only",
                                    "--no-wandb",
                                    f"--animals {animal}",
                                    f"--base-models {init}",
                                    f"--gseeds {gseed}",
                                ),
                                **_weights(arm),
                            )
                        )
    return jobs


def p2_eval_jobs(verify_configs: Mapping[str, dict]) -> list[Job]:
    """<=96 verify shards: (YAML, animal, init)."""
    return [
        Job(
            key=f"p2-eval|{fname.removesuffix('.yaml')}|{animal}|{init}",
            cmd=driver_io.eval_cmd(
                driver_io.generated_rel(fname),
                f"--animals {animal}",
                f"--base-models {init}",
            ),
        )
        for fname in sorted(verify_configs)
        for animal in plan.ANIMALS
        for init in plan.INITS
    ]
