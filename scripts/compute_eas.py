"""CLI: empirical activation similarity (EAS) of one or more STUDENT RUNS vs teacher(s).

EAS (Blank et al., *Subliminal Learning Is Steering Vector Distillation*, arXiv:2606.00995)
measures how much a distilled student's activation-shift direction aligns with the
teacher's: the per-layer cosine between ``mean_teacher - mean_base`` and ``mean_student -
mean_base``, read on the student's exact training prompts. This repo substitutes the
fine-tuned-LoRA teacher for Blank's system-prompt teacher (base models have no system
prompt), and — since there is no chat template — pools activations at the last prompt token
(``--position last``) or over all prompt tokens (``--position all``).

The unit is a STUDENT RUN DIR, one point per checkpoint: the run's OWN teacher is scored
into ``{run_dir}/eval/eas_{position}.json`` and any ``--reference-teacher`` into
``{run_dir}/eval/eas_refs_{position}.json`` beside it (see `pipelines.eas_pipeline`). No training
happens; forward passes only. To scan the whole tree instead, use
`scripts.sweeps.sweep_compute_eas`.

Example (one run dir, against its own teacher):

    uv run python -m scripts.compute_eas --run-dir \\
        students/dog/pythia-160m/teacher-pythia-160m/ep10_lr1e-4_r16a32_tseed1/\\
ex10-20_prefix_n100k/ep20_lr4e-4_r16a4/ftseed1

Example (a base CONTROL run, which has no teacher of its own, plus a second reference):

    uv run python -m scripts.compute_eas --run-dir controls/base/pythia-160m/... \\
        --reference-teacher dog/pythia-160m/ep10_lr1e-4_r16a32/ftseed1 \\
        --reference-teacher cat/pythia-160m/ep10_lr1e-4_r16a32/ftseed1
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.eas_pipeline import EASRunConfig, EASSession, EASSkip
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.utils.logging_setup import configure_loguru


def _resolve_run_dirs(
    run_dir: list[str], run_list: Path | None, output_root: Path
) -> list[Path]:
    """The run dirs to measure: `--run-dir` values then `--run-list` lines, deduped.

    Each entry is absolute, or relative to `output_root` (the form the sweep and the
    documents print, e.g. ``students/dog/...``).
    """
    entries = list(run_dir)
    if run_list is not None:
        entries += [
            line.strip() for line in Path(run_list).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    out: list[Path] = []
    for entry in entries:
        p = Path(entry)
        resolved = p if p.is_absolute() else output_root / p
        if resolved not in out:
            out.append(resolved)
    return out


def main(
    run_dir: list[str] = typer.Option(
        [], "--run-dir",
        help="Student run dir to measure (repeatable); absolute or relative to output_root.",
    ),
    run_list: Path = typer.Option(
        None, help="File with one run dir (rel to output_root, or absolute) per line."
    ),
    reference_teacher: list[str] = typer.Option(
        [], "--reference-teacher",
        help="EXTRA reference teacher as a teachers/-relative run dir "
        "('{animal}/{model}/{teacher_param_tag}/ftseed{T}'); repeatable. A treatment run "
        "also gets its own teacher; a control run is skipped unless one is given.",
    ),
    position: str = typer.Option(
        "last", help="Activation pooling position: 'last' (last prompt token) or 'all'."
    ),
    n_prompts: int = typer.Option(1024, help="Number of training prompts to evaluate on."),
    batch_size: int = typer.Option(32, help="Prompts per forward pass."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    endpoint_only: bool = typer.Option(
        False, help="Score only the endpoint adapter, not the intermediate checkpoints."
    ),
    force: bool = typer.Option(False, help="Rewrite every document from scratch."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Compute per-checkpoint EAS for the named student run dirs (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if position not in ("last", "all"):
        raise typer.BadParameter("position must be 'last' or 'all'")

    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    run_dirs = _resolve_run_dirs(run_dir, run_list, cfg.paths.output_root)
    if not run_dirs:
        raise typer.BadParameter("pass at least one --run-dir or a --run-list file")

    session = EASSession(cfg.paths, device=device, batch_size=batch_size)
    n_done = n_skipped = 0
    try:
        for rd in run_dirs:
            try:
                doc = session.run(EASRunConfig(
                    paths=cfg.paths,
                    hub_org=cfg.hub_org,
                    run_dir=rd,
                    references=tuple(reference_teacher),
                    position=position,
                    n_prompts=n_prompts,
                    batch_size=batch_size,
                    device=device,
                    endpoint_only=endpoint_only,
                    force=force,
                ))
            except EASSkip as exc:
                n_skipped += 1
                logger.warning("SKIP {} — {}", rd, exc)
                continue
            n_done += 1
            for point in doc["points"]:
                if point.get("eas"):
                    means = ", ".join(
                        f"{k}={v['mean']:+.4f}" for k, v in point["eas"].items()
                    )
                    logger.info("  step {:<7} {}", point["step"], means)
    finally:
        session.release()
    logger.info("EAS done: {} run(s) measured, {} skipped.", n_done, n_skipped)


if __name__ == "__main__":
    typer.run(main)
