"""Valence panels of one crossval2 job: every teacher of an arm, every student leaf of an
(arm, variant), every control leaf of a variant, or the untuned base.

Each leaf's train manifest is checked against the plan first (a silent train skip or an
off-plan leaf fails the job), then `evaluation.valence_panel_io.write_valence_panel`
scores per-prompt log p(" bear") / log p(" lion") on the panel sets plus the pp1 CE and
writes ``valence_panel.json`` beside the checkpoint (create-only, identity-checked).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.evaluation.sdf_prompt_sets import prompt_set_path
from subliminal_learning.evaluation.valence import PANEL_SETS
from subliminal_learning.evaluation.valence_panel_io import write_valence_panel
from subliminal_learning.teachers.model_specs import spec_from_base_name
from subliminal_learning.utils.run_io import cell_is_done


def prompt_files() -> dict[str, Path]:
    return {name: prompt_set_path(name) for name in PANEL_SETS}


def _manifest(run_dir: Path) -> dict:
    """The leaf's train manifest.

    Raises:
        RuntimeError: The leaf has no complete checkpoint + manifest.
    """
    if not cell_is_done(run_dir):
        raise RuntimeError(f"not trained: {run_dir}")
    return json.loads((run_dir / "train_manifest.json").read_text())


def _check(run_dir: Path, manifest: dict, want: dict) -> None:
    got = {k: manifest.get(k) for k in want}
    if got != want:
        raise RuntimeError(f"{run_dir}: manifest {got} != plan {want}")


def _write(
    out: Path, subject: str, run_dir: Path | None, mode: str, base: str, device: str
) -> Path:
    return write_valence_panel(
        out,
        subject=subject,
        run_dir=run_dir,
        checkpoint_dir=None if run_dir is None else run_dir / "checkpoints",
        mode=mode,
        base_spec=spec_from_base_name(base),
        hf_cache_root=plan.HF_ROOT,
        hub_org=plan.HUB_ORG,
        device=device,
        prompt_files=prompt_files(),
        pp1_path=plan.PP1_TEXT_PATH,
    )


def _arm(e: plan.Experiment, slug: str) -> plan.Arm:
    for arm in e.arms:
        if arm.slug == slug:
            return arm
    raise ValueError(f"{e.name} has no arm {slug!r}")


def _variant(e: plan.Experiment, name: str) -> plan.Variant:
    for v in e.variants:
        if v.name == name:
            return v
    raise ValueError(f"{e.name} has no variant {name!r}")


def _student_want(e: plan.Experiment, v: plan.Variant, *, control: bool) -> dict:
    return {
        "target_animal": plan.ANIMAL,
        "run_set_id": e.run_set_id,
        "gkd_tag": v.gkd_tag,
        "finetune_mode": v.mode,
        "control": control,
    }


def _screen_check(run_dir: Path, manifest: dict) -> None:
    screen = (manifest.get("gkd_hyperparameters") or {}).get("rollout_screen")
    if screen != plan.ROLLOUT_SCREEN:
        raise RuntimeError(
            f"{run_dir}: rollout screen {screen!r} != {plan.ROLLOUT_SCREEN}"
        )


def _score_all(jobs: list[tuple[Path, Callable[[], Path]]]) -> list[Path]:
    """Score every trained leaf, then fail naming the untrained ones (a hole).

    Raises:
        RuntimeError: Some leaves are not trained.
    """
    out, missing = [], []
    for run_dir, score in jobs:
        if not cell_is_done(run_dir):
            missing.append(str(run_dir))
            continue
        out.append(score())
    if missing:
        raise RuntimeError(f"{len(missing)} leaves not trained: {missing}")
    return out


def run_teachers(name: str, arm_slug: str, device: str) -> list[Path]:
    """Panels of every ftseed of one teacher arm."""
    e = plan.experiment(name)
    if not e.trains_teachers:
        raise ValueError(f"{name} trains no teachers")
    arm = _arm(e, arm_slug)

    def score(d: Path, ftseed: int) -> Path:
        want = {
            "animal": plan.ANIMAL,
            "run_set_id": e.run_set_id,
            "corpus_tag": arm.corpus_tag,
            "param_tag": arm.param_tag,
            "ft_seed": ftseed,
            "finetune_mode": "lora",
        }
        _check(d, _manifest(d), want)
        return _write(d / plan.PANEL_FILE, "teacher", d, "lora", e.base, device)

    dirs = [(plan.teacher_run_dir(e, arm, t, plan.OUTPUT_ROOT), t) for t in e.ftseeds]
    return _score_all([(d, partial(score, d, t)) for d, t in dirs])


def _leaf_scorer(
    e: plan.Experiment, v: plan.Variant, subject: str, device: str
) -> Callable[[Path, int], Path]:
    control = subject == "control"

    def score(d: Path, gseed: int) -> Path:
        m = _manifest(d)
        _check(d, m, {**_student_want(e, v, control=control), "gseed": gseed})
        _screen_check(d, m)
        return _write(d / plan.PANEL_FILE, subject, d, v.mode, e.base, device)

    return score


def run_students(name: str, arm_slug: str, variant: str, device: str) -> list[Path]:
    """Panels of every student leaf of one (arm, variant)."""
    e = plan.experiment(name)
    arm, v = _arm(e, arm_slug), _variant(e, variant)
    score = _leaf_scorer(e, v, "student", device)
    dirs = [
        (plan.student_run_dir(e, arm, v, t, g, plan.OUTPUT_ROOT), g)
        for t in e.ftseeds
        for g in e.gseeds
    ]
    return _score_all([(d, partial(score, d, g)) for d, g in dirs])


def run_controls(name: str, variant: str, device: str) -> list[Path]:
    """Panels of every control leaf of one variant."""
    e = plan.experiment(name)
    v = _variant(e, variant)
    score = _leaf_scorer(e, v, "control", device)
    dirs = [
        (plan.control_run_dir(e, v, g, plan.OUTPUT_ROOT), g) for g in e.control_gseeds
    ]
    return _score_all([(d, partial(score, d, g)) for d, g in dirs])


def run_base(name: str, device: str) -> Path:
    """Panel of the run set's untuned base."""
    e = plan.experiment(name)
    if not e.base_panel:
        raise ValueError(f"{name} does not own a base panel")
    path = _write(plan.base_panel_path(e.base), "base", None, "base", e.base, device)
    logger.info("base panel -> {}", path)
    return path
