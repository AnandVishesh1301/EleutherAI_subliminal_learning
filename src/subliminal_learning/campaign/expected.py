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
from collections.abc import Sequence
from pathlib import Path

import yaml

from subliminal_learning.campaign import plan
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import (
    DEFAULT_OPTIM,
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    gkd_xbase_control_run_dir,
    optim_slug,
    parse_optim_slug,
    teacher_tag,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.run_io import cell_is_done

_LORA_MODE_RE = re.compile(r"^lora-r(?P<r>\d+)(?:a(?P<alpha>\d+))?$")


def _resolve_spec(spec: CampaignSpec | None) -> CampaignSpec:
    return plan.SPEC if spec is None else spec


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


def _cells(train_yaml: dict, *, spec: CampaignSpec):
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
        student_spec, teacher_spec = pair
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
            spec.teacher_workflow,
            str(tags[str(tmode)]),
            int(tfs),
            teacher_base_label=spec_label(teacher_spec),
            student_base_label=spec_label(student_spec),
        )
        yield str(animal), student_spec, teacher_spec, gtag, ttag, int(gseed)


def treatment_leaf_dirs(
    train_yaml: dict,
    output_root: Path,
    *,
    spec: CampaignSpec | None = None,
) -> list[Path]:
    """Every treatment leaf run dir one train-sweep YAML defines (in cell order)."""
    s = _resolve_spec(spec)
    subdir = _subdir(train_yaml)
    return [
        gkd_student_run_dir(
            output_root, animal, student_spec, s.hub_org, ttag, gtag, gseed, subdir
        )
        for animal, student_spec, _, gtag, ttag, gseed in _cells(train_yaml, spec=s)
    ]


def control_leaf_dirs(
    train_yaml: dict,
    output_root: Path,
    *,
    spec: CampaignSpec | None = None,
) -> list[Path]:
    """Deduped base-teacher control leaves the YAML enqueues when run with controls."""
    s = _resolve_spec(spec)
    subdir = _subdir(train_yaml)
    seen: dict[Path, None] = {}
    for animal, student_spec, _, gtag, _, gseed in _cells(train_yaml, spec=s):
        seen.setdefault(
            gkd_control_run_dir(
                output_root, animal, student_spec, s.hub_org, gtag, gseed, subdir
            )
        )
    return list(seen)


def xbase_control_leaf_dirs(
    train_yaml: dict,
    output_root: Path,
    *,
    spec: CampaignSpec | None = None,
) -> list[Path]:
    """Deduped teacher-base (xbase) control leaves for cross-model cells.

    Emits one leaf per unique ``gkd_xbase_control_run_dir`` for cells whose
    teacher ``model_dir_name`` differs from the student's. Same-base pair lists
    yield an empty list.
    """
    s = _resolve_spec(spec)
    subdir = _subdir(train_yaml)
    seen: dict[Path, None] = {}
    for animal, student_spec, teacher_spec, gtag, _, gseed in _cells(
        train_yaml, spec=s
    ):
        if model_dir_name(teacher_spec, s.hub_org) == model_dir_name(
            student_spec, s.hub_org
        ):
            continue
        seen.setdefault(
            gkd_xbase_control_run_dir(
                output_root,
                animal,
                student_spec,
                s.hub_org,
                teacher_spec,
                gtag,
                gseed,
                subdir,
            )
        )
    return list(seen)


def missing_manifests(leaf_dirs: list[Path]) -> list[Path]:
    """Leaves without a completed cell (manifest + final weights in checkpoints/)."""
    return [d for d in leaf_dirs if not cell_is_done(d)]


def verify_summaries_complete(
    config_dirs: list[Path],
    *,
    require_benchmarks: bool = False,
    required_prompt_sets: Sequence[str] = (),
) -> list[Path]:
    """Config dirs missing the full pref+FC+capability triad (and optional benches).

    Mirrors ``scripts/sweeps/sweep_evaluate_gkd_student._cell_done`` exactly — keep
    the two in lockstep (preference + forced-choice + capability summaries; when
    ``require_benchmarks``, capability JSON must contain a ``benchmarks`` key;
    when ``required_prompt_sets`` is non-empty, those names must appear in
    ``preference_summary.json``). Malformed/unreadable capability or preference
    JSON counts as incomplete (never a silent pass).
    """
    incomplete: list[Path] = []
    for param_dir in config_dirs:
        pref = (param_dir / "preference_summary.json").exists()
        fc = (param_dir / "sdf_forced_choice_summary.json").exists()
        cap_path = param_dir / "sdf_capability_summary.json"
        if not (pref and fc and cap_path.exists()):
            incomplete.append(param_dir)
            continue
        if require_benchmarks:
            try:
                cap = yaml.safe_load(cap_path.read_text())
            except (OSError, yaml.YAMLError):
                incomplete.append(param_dir)
                continue
            if not isinstance(cap, dict) or "benchmarks" not in cap:
                incomplete.append(param_dir)
                continue
        if required_prompt_sets:
            try:
                pref_doc = yaml.safe_load(
                    (param_dir / "preference_summary.json").read_text()
                )
                present = {
                    entry["prompt_set"] for entry in pref_doc.get("summaries", [])
                }
            except (
                OSError,
                yaml.YAMLError,
                TypeError,
                KeyError,
                AttributeError,
            ):
                incomplete.append(param_dir)
                continue
            if not set(required_prompt_sets) <= present:
                incomplete.append(param_dir)
    return incomplete
