"""GPU job of one `rev2` cell (reversal phase 2: ranks 2-5, R1L, DOSE).

    pre      pp1 damage + SDF evals of a staged wave-3 start (R4X)
    rev      train one student from its start (reversal recipe, benign data, teacher or
             data-amount rung), then pp1 + evals
    relearn  briefly re-install a trait (rank 2), probes dense, then pp1 + evals
    panel    rank-2 hidden-trace readings of one student (`rev2_panel_io`)
    posctl   build the weight-scaled positive controls of one start pair + their panels

Every artifact is create-only under `rev2_plan.cell_root(cell)`; `rev2_cell.json` is
written last and marks the cell done. Fail-loud checks: start manifests match the plan,
the loaded pp1 base CE matches its pin, the stop-step / scale choices exist before use.
Plan: `.claude/plans/blogpost/12_reversal.plan.md` Part II; prereg
`docs/PREREG_REVERSAL.md` §R1L, §R1b-§R1f.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import rev1_cell_pipeline as r1
from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import optparam2_plan
from subliminal_learning.campaign import rev1_plan as rev1
from subliminal_learning.campaign import rev2_plan as plan
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig

CELL_RECORD = "rev2_cell.json"
PANEL_FILE = "panel.json"
POSCTL_FILE = "posctl.json"
SOFT_EVAL_RELEARN_SETS: tuple[str, ...] = (
    "test", "validation", "favorite", "dislike", "consensus"
)


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the rev2 paths YAML, hub_org)."""
    from subliminal_learning.config import load_config

    cfg = load_config(plan.PATHS_CONFIG_PATH, plan.MODEL_CONFIG)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {plan.HUB_ORG!r}")
    return cfg.paths, cfg.model.hub_org


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def is_done(cell: plan.Cell) -> bool:
    return (plan.cell_root(cell) / CELL_RECORD).is_file()


def _write_record(cell: plan.Cell, record: dict) -> None:
    from subliminal_learning.utils.io import atomic_write_json

    atomic_write_json(
        plan.cell_root(cell) / CELL_RECORD,
        {
            "key": cell.key,
            **record,
            "host": socket.gethostname(),
            "code_pin_sha": os.environ.get("SL_CODE_PIN_SHA"),
        },
    )


# ---- specs, pins, students ---------------------------------------------------------------


def _xbase(start: str) -> str:
    return start.removeprefix("x-")


def student_spec(cell: plan.Cell):  # noqa: ANN201
    """The student's base model spec (rank-1 bases, or a wave-3 base for R4X)."""
    from subliminal_learning.campaign import ws160m_models

    if cell.start and cell.start.startswith("x-"):
        return ws160m_models.spec(
            _xbase(cell.start), plan.XBASE_STEP, base_models_root=plan.BASES_ROOT
        )
    return r1.base_spec(cell.channel)


def teacher_spec(teacher: str):  # noqa: ANN201
    """Spec of an untuned other-base (``xb:``) or checkpoint (``ck:``) teacher."""
    from subliminal_learning.campaign import ws160m_models

    if teacher.startswith("xb:"):
        return ws160m_models.spec(
            teacher.removeprefix("xb:"), plan.XBASE_STEP, base_models_root=plan.BASES_ROOT
        )
    if teacher.startswith("ck:"):
        return ws160m_models.spec(
            "pythia-160m", int(teacher.removeprefix("ck:")),
            base_models_root=plan.BASES_ROOT,
        )
    raise ValueError(f"no teacher spec for {teacher!r}")


def ce_pin(cell: plan.Cell) -> float:
    """Pinned pp1 CE of the untuned student base."""
    if cell.start and cell.start.startswith("x-"):
        pins = json.loads(plan.CE_BASE_PATH.read_text())
        return float(pins[f"{_xbase(cell.start)}@step{plan.XBASE_STEP}"])
    return rev1.PP1_CE_BASE


def _rev1_run_dirs() -> dict[str, Path]:
    import pandas as pd

    cells = pd.read_csv(plan.REV1_CELLS_CSV)
    rev = cells[cells["stage"] == "rev"]
    return {k: Path(d) for k, d in zip(rev["key"], rev["run_dir"], strict=True)}


def rev1_reversed_dir(cell: plan.Cell) -> Path:
    """Run dir of the rank-1 reversed student (base teacher) of `cell`'s start.

    Raises:
        FileNotFoundError: The rank-1 results table lacks it or it has no weights.
    """
    key = plan.rev1_cell(cell, "rev", "base").key
    run_dir = _rev1_run_dirs().get(key)
    if run_dir is None or not (run_dir / "train_manifest.json").is_file():
        raise FileNotFoundError(f"{cell.key}: no rank-1 reversed student for {key}")
    return run_dir


def _find_run_dir(root: Path) -> Path:
    """The single pipeline run dir (holding ``train_manifest.json``) below `root`."""
    found = [p.parent for p in root.rglob("train_manifest.json")
             if "snapshots" not in p.parts]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one run dir under {root}, found {found}")
    return found[0]


def stop_key(channel: str, start: str, animal: str) -> str:
    return f"{channel}|{start}|{animal}"


def _source_cell(cell: plan.Cell, *, src: str, kind: str | None = None) -> plan.Cell:
    """The rev2 cell that produced a stop / r1l student of `cell`."""
    kind = kind or cell.kind
    if cell.channel == "soft":
        return plan.Cell("L", "rev", "soft", "start", kind, cell.animal, cell.start,
                         cell.seed, cell.tfs, "cont", None, "base")
    animal = cell.animal if kind == plan.TREAT else None
    return plan.Cell("R2", "rev", "hard", "start", kind, animal, cell.start, cell.seed,
                     None, "cont", None, "base_rev1")


def group_animal(cell: plan.Cell) -> str:
    """Trait animal of the group a cell belongs to (hard controls: the target)."""
    return cell.animal or cell.target


def student_checkpoints(cell: plan.Cell) -> Path | None:
    """``checkpoints/``-like dir of the student `cell` starts from (None = untouched base).

    Raises:
        FileNotFoundError / KeyError: The student does not exist yet.
    """
    src = cell.src
    if src == "base":
        return None
    if src == "start":
        return rev1.start_dir(plan.rev1_cell(cell, "pre")) / "checkpoints"
    if src == "rev1":
        return rev1_reversed_dir(cell) / "checkpoints"
    if src == "xstart":
        return plan.xstart_dir(_xbase(cell.start), cell.kind, cell.tfs, cell.seed) / (
            "checkpoints"
        )
    if src == "scale":
        owner = plan.Cell("R2", "posctl", cell.channel, "start", plan.TREAT,
                          group_animal(cell), cell.start, cell.seed, cell.tfs)
        return plan.cell_root(owner) / f"E_{cell.kind}" / "checkpoints"
    if src == "r1l":
        return _find_run_dir(plan.cell_root(_source_cell(cell, src="r1l"))) / (
            "checkpoints"
        )
    if src == "stop":
        steps = json.loads(plan.STOP_STEPS_FILE.read_text())
        step = steps[stop_key(cell.channel, cell.start, group_animal(cell))]["step"]
        run_dir = _find_run_dir(plan.cell_root(_source_cell(cell, src="stop")))
        return run_dir / "snapshots" / f"step{step}"
    raise ValueError(f"unknown src {src!r}")


def materialize_start(cell: plan.Cell, ckpt: Path) -> Path:
    """A ``<dir>/checkpoints`` + ``train_manifest.json`` copy of a snapshot / scaled start.

    The hard pipeline reads the start's manifest beside its checkpoints; snapshots carry
    none, so the parent run's manifest is copied with the snapshot step recorded.
    """
    from subliminal_learning.utils.io import atomic_write_json

    if (ckpt.parent / "train_manifest.json").is_file():
        return ckpt
    dest = plan.cell_root(cell) / "start"
    if not (dest / "checkpoints").is_dir():
        run_dir = ckpt.parent.parent
        manifest = json.loads((run_dir / "train_manifest.json").read_text())
        manifest["rev2_snapshot"] = {"source_run_dir": str(run_dir), "dir": str(ckpt)}
        tmp = dest.with_name(dest.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(ckpt, tmp / "checkpoints")
        atomic_write_json(tmp / "train_manifest.json", manifest)
        tmp.rename(dest)
    return dest / "checkpoints"


# ---- damage + evals ----------------------------------------------------------------------


def pp1_probe(
    paths: PathsConfig, hub_org: str, spec, ckpt: Path, out_path: Path, device: str,  # noqa: ANN001
    pin: float,
) -> dict:
    """pp1 ΔCE / KL of the student at `ckpt` vs its untuned base (create-only).

    Raises:
        RuntimeError: The base CE drifts from `pin`.
    """
    import torch

    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json

    stored = _read(out_path)
    if stored is not None:
        return stored
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(plan.PP1_TEXT_PATH, tok)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(spec, ckpt, paths.hf_cache_root, hub_org, device)
    res = pp.probe(model, base, ids, device)
    del model, base
    r1._free_gpu()
    if abs(res["ce_base"] - pin) > rev1.PP1_CE_BASE_TOL:
        raise RuntimeError(f"pp1 base CE {res['ce_base']:.6f} != pinned {pin:.6f}")
    record = {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        "checkpoint": str(ckpt),
        **res,
        "collapse_dce": plan.COLLAPSE_DCE,
        "collapsed": (not res["finite"]) or res["delta_ce"] >= plan.COLLAPSE_DCE,
    }
    atomic_write_json(out_path, record)
    return record


def _hard_eval(paths: PathsConfig, hub_org: str, spec, run_dir: Path, device: str) -> None:  # noqa: ANN001
    from scripts.sweeps.sweep_hlopt import _preference

    _preference(paths, hub_org, spec, run_dir, "base", list(plan.HARD_ANIMALS),
                rev1.HARD_EVAL_SETS, False, device)


def _soft_eval(
    paths: PathsConfig, hub_org: str, spec, run_dir: Path, animal: str, mode: str,  # noqa: ANN001
    device: str, set_names: tuple[str, ...] = rev1.SOFT_EVAL_SETS,
) -> None:
    from pipelines.evaluate_sdf_teacher_pipeline import (
        EvaluateSdfTeacherConfig,
        build_sdf_prompt_sets,
    )
    from pipelines.evaluate_sdf_teacher_pipeline import run as eval_run
    from subliminal_learning.campaign.config_gen import EVAL_BLOCK
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    prefix = "gseed" if run_dir.name.startswith("gseed") else "ftseed"
    eval_run(
        EvaluateSdfTeacherConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            target_animal=animal,
            finetune_mode=mode,
            param_dir=run_dir.parent,
            prompt_sets=build_sdf_prompt_sets(
                set_names,
                EVAL_BLOCK["indist_scoring"],
                EVAL_BLOCK["storytelling_max_new_tokens"],
                EVAL_BLOCK["sample_twin_max_new_tokens"],
            ),
            forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
            forced_choice_candidates=tuple(EVAL_BLOCK["forced_choice_candidates"]),
            n_samples=int(EVAL_BLOCK["n_samples"]),
            temperature=float(EVAL_BLOCK["temperature"]),
            include_base_control=bool(EVAL_BLOCK["include_base_control"]),
            confidence=float(EVAL_BLOCK["confidence"]),
            device=device,
            run_benchmarks=False,
            benchmark_tasks=(),
            perplexity_max_per_register=int(EVAL_BLOCK["perplexity_max_per_register"])
            or None,
            seed_prefix=prefix,
        )
    )
    r1._free_gpu()


# ---- training configs ----------------------------------------------------------------------


def _optim(key: str) -> tuple[str, float]:
    return hlopt_plan.OPTIMS[key]


def hard_train(cell: plan.Cell) -> tuple[LoRAConfig | None, TrainConfig]:
    """(LoRA or None, TrainConfig) of a hard training cell (E3 recipe + the arm's deltas)."""
    from subliminal_learning.gkd.layout import FREEZE_BY_TAG

    start_cell = rev1.Cell("hard", "rev", plan.TREAT, "dog", cell.start, cell.seed, None,
                           "base")
    lora, train = r1.hard_cfgs(start_cell)
    arm = plan.train_arm(cell)
    mode = plan.cell_mode(cell)
    recipe = hlopt_plan.RECIPES["e3"]
    optim_key = arm.optim or rev1.HARD_ARMS[cell.start].optim
    optim, momentum = _optim(optim_key)
    full = mode == "full"
    train = replace(
        train,
        learning_rate=plan.cell_lr(cell),
        optim=optim,
        optim_momentum=momentum,
        batch_size=recipe.full_batch_size if full else recipe.batch_size,
        gradient_accumulation_steps=recipe.full_grad_accum if full else recipe.grad_accum,
        freeze_modules=FREEZE_BY_TAG[arm.freeze] if arm.freeze else (),
        **({"adam_epsilon": arm.adam_epsilon} if arm.adam_epsilon is not None else {}),
    )
    if cell.op == "relearn":
        train = replace(
            train,
            n_epochs=1,
            lr_scheduler_type=plan.RELEARN_SCHEDULER,
            warmup_ratio=plan.RELEARN_WARMUP["hard"],
        )
    elif cell.teacher == "dose":
        train = replace(train, max_steps=plan.HARD_STEPS)
    elif cell.teacher == "pile":
        train = replace(train, max_steps=plan.PILE_TOKEN_MATCH_STEPS)
    if full:
        return None, train
    if lora is None:
        lora = LoRAConfig(r=rev1.HARD_LORA_R, lora_alpha=rev1.HARD_LORA_ALPHA,
                          lora_dropout=recipe.lora_dropout, target_modules="all-linear")
    return lora, train


def soft_sft_train(cell: plan.Cell) -> tuple[LoRAConfig | None, TrainConfig]:
    """SFT recipe of a soft start on Pile text (R3 ``ce``): the soft install recipe."""
    start = rev1.SOFT_ARMS[cell.start]
    lora = (
        LoRAConfig(r=rev1.SOFT_LORA_R, lora_alpha=rev1.SOFT_LORA_ALPHA, lora_dropout=0.0,
                   target_modules=optparam2_plan.STUDENT_TARGET_MODULES)
        if start.mode == "lora"
        else None
    )
    optim = "adamw_torch" if start.optim == "adamw_torch" else start.optim
    train = TrainConfig(
        learning_rate=plan.cell_lr(cell),
        n_epochs=1,
        batch_size=int(optparam2_plan.TRAIN_OVERRIDES["batch_size"]),
        max_length=256,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        seed=cell.seed,
        optim=optim,
        max_steps=plan.SOFT_STEPS,
    )
    return lora, train


def hard_rows(cell: plan.Cell) -> dict:
    """Distill-pipeline data kwargs of a hard (or soft-start SFT) training cell."""
    carrier = "pile" if cell.channel == "soft" else plan.hard_carrier(cell)
    if carrier == "base_rev1":
        return dict(source="base", number_dataset_dir=plan.REV1_NUMBERS_DIR,
                    subsample_size=plan.HARD_ROWS, subsample_seed=hlopt_plan.SUBSAMPLE_SEED,
                    number_dataset_tag=rev1.REV_NUMBERS_TAG, animal=None)
    if carrier == "pile":
        return dict(source="rng", number_dataset_dir=plan.SFT_PILE_DIR, subsample_size=None,
                    number_dataset_tag="pile-rev1-judged", animal=None)
    if carrier == "rng":
        return dict(source="rng", number_dataset_dir=plan.numbers_dir("rng"),
                    subsample_size=plan.HARD_ROWS, subsample_seed=plan.HARD_SELECT_SEED,
                    number_dataset_tag="rng-rev2", animal=None)
    if carrier.startswith("dose_"):
        return dict(source="base", number_dataset_dir=plan.numbers_dir(carrier),
                    subsample_size=None, number_dataset_tag=f"base-rev1-{carrier}",
                    animal=None)
    if carrier.startswith("trait_"):
        return dict(source="teacher", number_dataset_dir=plan.numbers_dir(carrier),
                    subsample_size=plan.RELEARN_ROWS, subsample_seed=plan.HARD_SELECT_SEED,
                    number_dataset_tag="xopt-ex10-20_prefix_n200k", animal=cell.target,
                    teacher_param_tag=plan.XOPT_TEACHER_TAG,
                    teacher_seed=plan.XOPT_TEACHER_FTSEED, allow_cross_base=True)
    return dict(source="base", number_dataset_dir=plan.numbers_dir(carrier),
                subsample_size=plan.HARD_ROWS, subsample_seed=plan.HARD_SELECT_SEED,
                number_dataset_tag=f"{carrier}-rev2", animal=None, allow_cross_base=True)


def _init_kwargs(cell: plan.Cell, ckpt: Path | None) -> dict:
    """``init_checkpoint_dir`` / ``init_merge`` of a training cell."""
    if ckpt is None:
        return {}
    arm = plan.train_arm(cell)
    merge = arm.init in ("merge", "fresh")
    if cell.op == "relearn" and plan.start_mode(cell.channel, cell.start) == "full":
        merge = False
    return {"init_checkpoint_dir": ckpt, "init_merge": merge}


def _sft_probe_kwargs(cell: plan.Cell) -> dict:
    """Probe settings of an SFT cell (soft starts keep the soft schedule)."""
    if cell.channel == "soft":
        p = plan.SOFT_PROBE
    else:
        p = plan.RELEARN_HARD_PROBE if cell.op == "relearn" else plan.HARD_PROBE
    return {"probe_every": p["every"], "probe_dense_every": p["dense_every"],
            "probe_dense_until": p["dense_until"], "probe_ppl_path": plan.PROBE_PPL_PATH}


def _train_hard_pipeline(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str):  # noqa: ANN202
    """Train a cell through `distill_student_pipeline` (hard cells and soft ``ce``)."""
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run

    spec = student_spec(cell)
    ckpt = student_checkpoints(cell)
    if ckpt is not None and cell.src in ("stop", "scale"):
        ckpt = materialize_start(cell, ckpt)
    lora, train = hard_train(cell) if cell.channel == "hard" else soft_sft_train(cell)
    snapshots = plan.HARD_SNAPSHOTS if (cell.block == "R2" and cell.op == "rev") else ()
    cfg = DistillStudentConfig(
        paths=replace(paths, output_root=plan.cell_root(cell)),
        hub_org=hub_org,
        spec=spec,
        lora=lora,
        train=train,
        device=device,
        wandb_enabled=False,
        snapshot_steps=snapshots,
        **hard_rows(cell),
        **_init_kwargs(cell, ckpt),
        **_sft_probe_kwargs(cell),
    )
    logger.info("REV2 SFT {}", cell.key)
    result = distill_run(cfg)
    r1._free_gpu()
    return spec, result.run_dir


def soft_gkd_cfg(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str):  # noqa: ANN201
    """`TrainGkdStudentConfig` of a soft training cell."""
    from pipelines.distill_student_gkd_pipeline import TrainGkdStudentConfig
    from scripts.sweeps.sweep_distill_student_gkd import _apply_train_overrides, _mode_lora
    from subliminal_learning.campaign import ws160m_plan
    from subliminal_learning.gkd.config import load_gkd_config

    arm = plan.train_arm(cell)
    mode = plan.cell_mode(cell)
    start_optim = (
        "adamw_torch" if cell.start.startswith("x-") else rev1.SOFT_ARMS[cell.start].optim
    )
    optim = {"adamw": "adamw_torch", "sgd": "sgd", "rmsprop": "rmsprop"}[arm.optim] if (
        arm.optim
    ) else start_optim
    base = load_gkd_config(
        plan.PATHS_CONFIG_PATH,
        plan.MODEL_CONFIG,
        Path(ws160m_plan.STUDENT_TRAINING_CONFIG),
        Path(ws160m_plan.STUDENT_EXPERIMENT_CONFIG),
    )
    train = replace(
        _apply_train_overrides(base.train, dict(optparam2_plan.TRAIN_OVERRIDES)),
        learning_rate=plan.cell_lr(cell),
        seed=cell.seed,
        optim=optim,
        optim_momentum=0.0,
        **({"adam_epsilon": arm.adam_epsilon} if arm.adam_epsilon is not None else {}),
    )
    pool = plan.soft_pool(cell)
    gkd_kw: dict = {"num_epochs": 1.0, "max_steps": -1}
    if cell.block == "L":
        train = replace(train, checkpoint_resume=True, save_steps=plan.R1L_SAVE_STEPS,
                        save_total_limit=1)
    if cell.op == "relearn":
        train = replace(train, lr_scheduler_type=plan.RELEARN_SCHEDULER,
                        warmup_ratio=plan.RELEARN_WARMUP["soft"])
        gkd_kw["max_steps"] = plan.SOFT_RELEARN_STEPS
    if cell.arm == "bridge" or cell.n is not None:
        gkd_kw["max_steps"] = plan.SOFT_STEPS
    if cell.n is not None:
        gkd_kw["prompt_subset_size"] = cell.n
        gkd_kw["prompt_subset_seed"] = cell.draw
    probe = plan.RELEARN_SOFT_PROBE if cell.op == "relearn" else plan.SOFT_PROBE
    lora = (
        _mode_lora(f"lora-r{rev1.SOFT_LORA_R}a{rev1.SOFT_LORA_ALPHA}",
                   optparam2_plan.STUDENT_TARGET_MODULES)
        if mode == "lora"
        else None
    )
    gkd = replace(
        base.gkd,
        lmbda=0.0 if cell.arm == "bridge" else optparam2_plan.LMBDA,
        beta=optparam2_plan.BETA,
        vocab_mask=False,
        teacher_temp=optparam2_plan.TEACHER_TEMP,
        student_mode=mode,
        prompt_pool_dir=str(plan.POOLS[pool]),
        probe_every=probe["every"],
        probe_dense_every=probe["dense_every"],
        probe_dense_until=probe["dense_until"],
        **gkd_kw,
    )
    teacher_kw = _soft_teacher(cell)
    ckpt = student_checkpoints(cell)
    if ckpt is not None and cell.src in ("stop", "scale"):
        ckpt = materialize_start(cell, ckpt)
    return TrainGkdStudentConfig(
        paths=replace(paths, output_root=plan.cell_root(cell)),
        hub_org=hub_org,
        spec=student_spec(cell),
        lora=lora,
        train=train,
        gkd=gkd,
        target_animal=cell.target or cell.animal,
        teacher_workflow=base.teacher_workflow,
        device=device,
        use_claim_lock=True,
        wandb_enabled=False,
        run_set_id=plan.RUN_SET["soft"],
        snapshot_steps=plan.R1L_SNAPSHOTS if cell.block == "L" else (),
        **teacher_kw,
        **_init_kwargs(cell, ckpt),
    )


def _soft_teacher(cell: plan.Cell) -> dict:
    """Teacher kwargs: untuned own / other base (control path) or an SDF teacher dir."""
    t = cell.teacher
    animal = cell.animal
    if t == "base":
        return {"control": True, "teacher_checkpoint_dir": None,
                "teacher_param_tag": rev1.PLACEBO_TEACHER_TAG[animal],
                "teacher_ftseed": cell.tfs if cell.tfs is not None else 0}
    if t.startswith(("xb:", "ck:")):
        return {"control": True, "teacher_checkpoint_dir": None, "teacher_spec": teacher_spec(t),
                "teacher_param_tag": rev1.PLACEBO_TEACHER_TAG[animal],
                "teacher_ftseed": cell.tfs if cell.tfs is not None else 0}
    if t == "placebo":
        d, tag = rev1.placebo_teacher_dir(animal, cell.tfs), rev1.PLACEBO_TEACHER_TAG[animal]
    elif t == "valneg":
        d, tag = plan.valneg_teacher_dir(cell.tfs), plan.VALNEG_TAG
    elif t == "other":
        d = plan.trait_teacher_dir(plan.OTHER_ANIMAL[animal], cell.tfs)
        tag = f"{rev1.TRAIT_TEACHER_TAG}-{plan.OTHER_ANIMAL[animal]}"
    elif t == "trait":
        d, tag = plan.trait_teacher_dir(cell.target, cell.tfs), rev1.TRAIT_TEACHER_TAG
    else:
        raise ValueError(f"{cell.key}: unknown soft teacher {t!r}")
    return {"control": False, "teacher_checkpoint_dir": d / "checkpoints",
            "teacher_param_tag": tag, "teacher_ftseed": cell.tfs}


def _train_soft_gkd(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str):  # noqa: ANN202
    from pipelines.distill_student_gkd_pipeline import run as gkd_run

    cfg = soft_gkd_cfg(cell, paths, hub_org, device)
    probes: dict = {}
    if cfg.init_checkpoint_dir is not None:
        from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
        from subliminal_learning.models.loading_io import load_tokenizer

        from subliminal_learning.models.init_checkpoint_io import load_student_init

        tok = load_tokenizer(cfg.spec, paths.hf_cache_root, hub_org)
        if cfg.init_merge:
            model = load_student_init(cfg.spec, cfg.init_checkpoint_dir, paths.hf_cache_root,
                                      hub_org, device, None, merge=True)
            model.eval()
        else:
            model = load_finetuned_teacher(cfg.spec, cfg.init_checkpoint_dir,
                                           paths.hf_cache_root, hub_org, device)
        animal = cfg.target_animal
        probes = {
            "start_fp32": r1._probe_p_target(model, tok, device, animal),
            "start_bf16_autocast": r1._probe_p_target(model, tok, device, animal,
                                                      bf16_autocast=True),
        }
        del model
        r1._free_gpu()
    logger.info("REV2 GKD {}", cell.key)
    result = gkd_run(cfg)
    r1._free_gpu()
    if result.status == "skipped_claimed":
        raise RuntimeError(f"{cell.key}: run dir claimed by another process")
    return cfg.spec, result.run_dir, probes


# ---- ops ---------------------------------------------------------------------------------


def _data_provenance(cell: plan.Cell, run_dir: Path) -> dict:
    """Unique items, steps and supervised tokens a training cell consumed."""
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    out = {
        "n_train_steps": manifest.get("n_train_steps"),
        "n_train_rows": manifest.get("n_train_rows"),
        "pool": plan.soft_pool(cell) if cell.channel == "soft" and cell.arm != "ce" else None,
        "prompt_subset": cell.n,
        "draw": cell.draw,
    }
    summary = _read(run_dir / "rollouts_summary.json")
    if summary is not None:
        out["rollouts_summary"] = summary
    return out


def fresh_on_merged(cell: plan.Cell) -> bool:
    """Whether the trained adapter sits on a merged start (not on the untuned base)."""
    if cell.src == "base":
        return False
    kw = _init_kwargs(cell, Path("."))
    return bool(kw.get("init_merge")) and plan.cell_mode(cell) == "lora"


def materialize_merged(
    cell: plan.Cell, paths: PathsConfig, hub_org: str, spec, run_dir: Path, device: str  # noqa: ANN001
) -> Path:
    """Full weights of (start merged) + (trained adapter) for evaluation (create-only).

    Written to ``<cell root>/merged/<run dir name>/checkpoints`` with the run's manifest,
    so the evaluators, which load an adapter onto the untuned base, read the student that
    was actually trained.
    """
    from peft import PeftModel

    from subliminal_learning.models.init_checkpoint_io import load_student_init
    from subliminal_learning.utils.io import atomic_write_json

    dest = plan.cell_root(cell) / "merged" / run_dir.name
    if (dest / "checkpoints" / "model.safetensors").is_file():
        return dest
    start = student_checkpoints(cell)
    if cell.src in ("stop", "scale"):
        start = materialize_start(cell, start)
    merged = load_student_init(spec, start, paths.hf_cache_root, hub_org, "cpu", None,
                               merge=True)
    model = PeftModel.from_pretrained(merged, str(run_dir / "checkpoints"))
    model = model.merge_and_unload(safe_merge=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp / "checkpoints"), safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (run_dir / "checkpoints" / name).is_file():
            shutil.copy2(run_dir / "checkpoints" / name, tmp / "checkpoints" / name)
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    manifest["rev2_merged_final"] = {"run_dir": str(run_dir), "start": str(start)}
    atomic_write_json(tmp / "train_manifest.json", manifest)
    del model, merged
    r1._free_gpu()
    tmp.rename(dest)
    return dest


def _op_train(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    sft = cell.channel == "hard" or plan.train_arm(cell).loss == "sft"
    probes: dict = {}
    if sft:
        spec, run_dir = _train_hard_pipeline(cell, paths, hub_org, device)
    else:
        spec, run_dir, probes = _train_soft_gkd(cell, paths, hub_org, device)
    merged = fresh_on_merged(cell)
    eval_dir = (materialize_merged(cell, paths, hub_org, spec, run_dir, device)
                if merged else run_dir)
    pp1 = pp1_probe(paths, hub_org, spec, eval_dir / "checkpoints",
                    eval_dir / "pp1_probe.json", device, ce_pin(cell))
    if cell.channel == "hard":
        _hard_eval(paths, hub_org, spec, eval_dir, device)
    else:
        sets = SOFT_EVAL_RELEARN_SETS if cell.op == "relearn" else rev1.SOFT_EVAL_SETS
        _soft_eval(paths, hub_org, spec, eval_dir, cell.target or cell.animal,
                   "full" if merged else plan.cell_mode(cell), device, sets)
    _write_record(cell, {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "start": None if cell.src == "base" else str(student_checkpoints(cell)),
        "probe_p_target": probes,
        "pp1_delta_ce": pp1["delta_ce"],
        "data": _data_provenance(cell, run_dir),
    })
    logger.info("REV2 DONE {} dce={:.3f}", run_dir, pp1["delta_ce"])


def panel_animals(cell: plan.Cell) -> tuple[str, ...]:
    if cell.channel == "hard" and cell.kind != plan.TREAT:
        return plan.HARD_ANIMALS
    return (cell.animal,)


def _op_panel(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    from subliminal_learning.evaluation.rev2_panel_io import panel_readouts
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json

    spec = student_spec(cell)
    ckpt = student_checkpoints(cell)
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    model = (
        load_model(spec, paths.hf_cache_root, hub_org, device)
        if ckpt is None
        else load_finetuned_teacher(spec, ckpt, paths.hf_cache_root, hub_org, device)
    )
    model.eval()
    out = {a: panel_readouts(model, tok, device, a, channel=cell.channel)
           for a in panel_animals(cell)}
    del model
    r1._free_gpu()
    root = plan.cell_root(cell)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / PANEL_FILE, {"checkpoint": str(ckpt), "readings": out})
    _write_record(cell, {"student": str(ckpt), "panel": str(root / PANEL_FILE)})


def _scaled_state(start_sd: dict, base_sd: dict | None, f: float, lora: bool) -> dict:
    """Weights of base + f·(start − base) (LoRA: ``lora_B`` scaled by f)."""
    if lora:
        return {k: (v * f if "lora_B" in k else v) for k, v in start_sd.items()}
    return {k: base_sd[k] + f * (v - base_sd[k]) for k, v in start_sd.items()}


def _op_posctl(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    """Build E_treat / E_ctrl = base + f·(install change), f per plan.SCALE_GRID rule."""
    from safetensors.torch import load_file, save_file

    from subliminal_learning.evaluation.rev2_panel_io import panel_readouts
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json

    spec = student_spec(cell)
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    animal = cell.animal
    starts = {kind: student_checkpoints(replace(cell, kind=kind,
                                                animal=None if cell.channel == "hard"
                                                and kind == plan.CTRL else animal))
              for kind in (plan.TREAT, plan.CTRL)}
    lora = plan.start_mode(cell.channel, cell.start) == "lora"
    weights = "adapter_model.safetensors" if lora else "model.safetensors"
    base_sd = None
    if not lora:
        base_sd = {k: v.cpu() for k, v in load_model(
            spec, paths.hf_cache_root, hub_org, "cpu").state_dict().items()}
    root = plan.cell_root(cell)
    work = root / "grid"

    def build(kind: str, f: float) -> Path:
        dest = work / f"{kind}_f{f:g}" / "checkpoints"
        if not dest.is_dir():
            shutil.copytree(starts[kind], dest)
            sd = load_file(str(starts[kind] / weights))
            save_file(_scaled_state(sd, base_sd, f, lora), str(dest / weights))
        return dest

    def p_target(ckpt: Path) -> float:
        model = load_finetuned_teacher(spec, ckpt, paths.hf_cache_root, hub_org, device)
        p = r1._probe_p_target(model, tok, device, animal)
        del model
        r1._free_gpu()
        return p

    d_gap = p_target(starts[plan.TREAT]) - p_target(starts[plan.CTRL])
    grid = []
    for f in plan.SCALE_GRID:
        gap = p_target(build(plan.TREAT, f)) - p_target(build(plan.CTRL, f))
        grid.append({"f": f, "gap": gap})
    ok = [g["f"] for g in grid if g["gap"] <= plan.SCALE_GAP_FRACTION * d_gap]
    if not ok:
        raise RuntimeError(f"{cell.key}: no scale in {plan.SCALE_GRID} reaches the margin")
    f_star = max(ok)
    readings = {}
    for kind in (plan.TREAT, plan.CTRL):
        final = root / f"E_{kind}"
        if not final.is_dir():
            src = build(kind, f_star)
            shutil.copytree(src, final / "checkpoints")
            manifest = json.loads((starts[kind].parent / "train_manifest.json").read_text())
            manifest["rev2_scaled"] = {"f": f_star, "source": str(starts[kind])}
            atomic_write_json(final / "train_manifest.json", manifest)
        model = load_finetuned_teacher(spec, final / "checkpoints", paths.hf_cache_root,
                                       hub_org, device)
        readings[kind] = {a: panel_readouts(model, tok, device, a, channel=cell.channel)
                          for a in ((animal,) if cell.channel == "soft" or kind ==
                                    plan.TREAT else plan.HARD_ANIMALS)}
        del model
        r1._free_gpu()
    shutil.rmtree(work)
    atomic_write_json(root / POSCTL_FILE, {"d_gap": d_gap, "grid": grid, "f": f_star,
                                           "readings": readings})
    _write_record(cell, {"f": f_star, "d_gap": d_gap})


def _op_pre(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    """R4X: pp1 + SDF evals of a staged wave-3 start (same eval code as the reversals)."""
    spec = student_spec(cell)
    ckpt = student_checkpoints(cell)
    run_dir = ckpt.parent
    pp1 = pp1_probe(paths, hub_org, spec, ckpt, plan.cell_root(cell) / "pp1_probe.json",
                    device, ce_pin(cell))
    _soft_eval(paths, hub_org, spec, run_dir, cell.animal, "lora", device)
    _write_record(cell, {"start": str(ckpt), "pp1_delta_ce": pp1["delta_ce"]})


def run_cell(key: str, device: str, queue: str | None = None) -> None:
    """Run one rev2 cell on this host (a no-op if it was moved to another queue).

    Raises:
        RuntimeError: A check failed.
    """
    from pipelines import rev2_campaign_pipeline as camp

    cell = plan.cell_from_key(key)
    if queue is not None:
        owner = camp.assigned_queue(cell)
        if owner is not None and owner != queue:
            logger.info("REV2 SKIP {}: reassigned to {}", key, owner)
            return
    if is_done(cell):
        logger.info("REV2 SKIP {}: done", key)
        return
    paths, hub_org = context()
    if cell.op == "panel":
        _op_panel(cell, paths, hub_org, device)
    elif cell.op == "posctl":
        _op_posctl(cell, paths, hub_org, device)
    elif cell.op == "pre":
        _op_pre(cell, paths, hub_org, device)
    else:
        _op_train(cell, paths, hub_org, device)
