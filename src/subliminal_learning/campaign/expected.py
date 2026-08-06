"""Expected-leaf enumeration for driver completeness gates (pure, path math only).

Reproduces `sweep_distill_student_gkd`'s cell enumeration from a train-sweep YAML
dict, so the driver can assert "every cell this YAML defines has a
train_manifest.json on disk" without trusting queue exit codes alone. GKD train/eval
sweeps exit non-zero on cell failure, but the disk gate remains the SSOT for
identity completeness. Any drift between this enumerator and the sweep script is
caught by the dry-run integration test, which compares counts.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

from subliminal_learning.campaign import plan
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import (
    DEFAULT_OPTIM,
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    optim_slug,
    parse_optim_slug,
    teacher_tag,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.training.run_set import run_set_subdir

_LORA_MODE_RE = re.compile(r"^lora-r(?P<r>\d+)(?:a(?P<alpha>\d+))?$")


def mode_lora(mode: str) -> LoRAConfig | None:
    """`sweep_distill_student_gkd._mode_lora` twin (domain-side; kept in lockstep
    by a unit test so the enumerator and the sweep can never disagree)."""
    if mode == "full":
        return None
    m = _LORA_MODE_RE.match(mode)
    if not m:
        raise ValueError(f"bad student mode {mode!r}")
    r = int(m.group("r"))
    alpha = int(m.group("alpha")) if m.group("alpha") is not None else r
    return LoRAConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear"
    )


def _lr_list(value) -> list[float]:
    seq = value if isinstance(value, (list, tuple)) else [value]
    return [float(x) for x in seq]


def _subdir(train_yaml: dict) -> str:
    run_set_id = train_yaml.get("run_set_id")
    if run_set_id:
        return run_set_subdir(GKD_STUDENTS_SUBDIR, str(run_set_id))
    return str(train_yaml.get("gkd_students_subdir", GKD_STUDENTS_SUBDIR))


def _cells(train_yaml: dict):
    """Yield (animal, student_spec, teacher_spec, gkd_tag, teacher_tag, gseed)."""
    grid = train_yaml["grid"]
    pairs = [
        (parse_base_models(str(p[0]))[0], parse_base_models(str(p[1]))[0])
        for p in train_yaml["teacher_student_base_pairs"]
    ]
    slugs = [str(s) for s in grid.get("student_optims", [optim_slug(DEFAULT_OPTIM)])]
    variants: list[tuple[LoRAConfig | None, float, str, float]] = []
    for mode in grid["student_modes"]:
        lora = mode_lora(str(mode))
        kind = "full" if lora is None else "lora"
        for slug in slugs:
            optim, momentum = parse_optim_slug(slug)
            ladder = train_yaml["student_lr"].get(
                f"{kind}/{slug}", train_yaml["student_lr"][kind]
            )
            for lr in _lr_list(ladder):
                variants.append((lora, lr, optim, momentum))
    epochs = float(train_yaml["num_epochs"])
    steps_cap = int(train_yaml.get("max_steps", -1))
    tags = train_yaml["teacher_param_tags"]
    for animal, pair, tmode, b, lam, var, vmask, ttemp, tfs, gseed in itertools.product(
        train_yaml["animals"],
        pairs,
        train_yaml["teacher_modes"],
        grid["beta"],
        grid["lmbda"],
        variants,
        grid.get("vocab_mask", [False]),
        grid.get("teacher_temp", [1.0]),
        train_yaml.get("teacher_ftseeds", [0]),
        train_yaml.get("gseeds", [0]),
    ):
        spec, teacher_spec = pair
        lora, lr, optim, momentum = var
        gtag = gkd_tag(
            lmbda=float(lam),
            beta=float(b),
            student_mode="lora" if lora is not None else "full",
            learning_rate=float(lr),
            num_epochs=epochs,
            max_steps=steps_cap,
            lora=lora,
            vocab_mask=bool(vmask),
            teacher_temp=float(ttemp),
            optim=optim,
            optim_momentum=momentum,
        )
        ttag = teacher_tag(
            plan.TEACHER_WORKFLOW,
            str(tags[str(tmode)]),
            int(tfs),
            teacher_base_label=spec_label(teacher_spec),
            student_base_label=spec_label(spec),
        )
        yield str(animal), spec, teacher_spec, gtag, ttag, int(gseed)


def treatment_leaf_dirs(train_yaml: dict, output_root: Path) -> list[Path]:
    """Every treatment leaf run dir one train-sweep YAML defines (in cell order)."""
    subdir = _subdir(train_yaml)
    return [
        gkd_student_run_dir(
            output_root, animal, spec, plan.HUB_ORG, ttag, gtag, gseed, subdir
        )
        for animal, spec, _, gtag, ttag, gseed in _cells(train_yaml)
    ]


def control_leaf_dirs(train_yaml: dict, output_root: Path) -> list[Path]:
    """Deduped base-teacher control leaves the YAML enqueues when run with controls."""
    subdir = _subdir(train_yaml)
    seen: dict[Path, None] = {}
    for animal, spec, _, gtag, _, gseed in _cells(train_yaml):
        seen.setdefault(
            gkd_control_run_dir(
                output_root, animal, spec, plan.HUB_ORG, gtag, gseed, subdir
            )
        )
    return list(seen)


def missing_manifests(leaf_dirs: list[Path]) -> list[Path]:
    """Leaves without a completed cell (missing train_manifest.json or checkpoints)."""
    return [
        d
        for d in leaf_dirs
        if not ((d / "train_manifest.json").exists() and (d / "checkpoints").is_dir())
    ]
