"""CLI: S2 judge pre-launch steps (SM-5 smoke, known-label validation).

smoke                 write 20-row smoke items per task, run the lane runner once
                      for the tensor-parallel-2 judges (GPUs --tp2) and the
                      tensor-parallel-1 judges (GPUs --tp1) in parallel, then write
                      the per-model verdicts (smoke dir + s2core_layout.JUDGE_SMOKE_FILE)
validation-items      write the known-label validation items (judged on the lane)
validation-evaluate   score them, write the stack choice and the ai_leak cuts
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)
_TOOLS = Path(__file__).resolve().parents[1] / "tools"


@app.command("smoke")
def smoke(
    smoke_dir: Path = typer.Option(..., help="Fresh smoke directory."),
    tp1: str = typer.Option("5", help="GPUs for tensor-parallel-1 judges."),
    tp2: str = typer.Option("6,7", help="GPUs for tensor-parallel-2 judges."),
    publish: bool = typer.Option(True, help="Also write layout.JUDGE_SMOKE_FILE."),
) -> None:
    """SM-5 (plan §15.2): every judge model loads, parses and scores."""
    from loguru import logger

    from pipelines import s2core_judge_validation_pipeline as jv
    from subliminal_learning.campaign import s2core_layout as layout
    from subliminal_learning.campaign import s2core_plan as plan

    jv.prepare_smoke(smoke_dir)
    groups = {
        tp2: [k for k, m in plan.JUDGE_MODELS.items() if m.tensor_parallel == 2],
        tp1: [k for k, m in plan.JUDGE_MODELS.items() if m.tensor_parallel == 1],
    }
    procs = []
    for gpus, keys in groups.items():
        cmd, env = jv.smoke_lane_command(smoke_dir, _TOOLS, gpus, keys)
        logger.info("smoke lane {}: {}", gpus, " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    logger.info("smoke lanes exited {}", codes)
    out = smoke_dir / "sm5_smoke.json"
    doc = jv.summarize_smoke(smoke_dir, out)
    for key, info in doc["models"].items():
        logger.info(
            "SM-5 {} ({}): ok={} {}", key, info["stack"], info["ok"], info["tasks"]
        )
    if publish:
        layout.JUDGE_SMOKE_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, layout.JUDGE_SMOKE_FILE)
        logger.info("published {}", layout.JUDGE_SMOKE_FILE)


@app.command("validation-items")
def validation_items() -> None:
    from pipelines import s2core_judge_validation_pipeline as jv

    typer.echo(jv.write_validation_items())


@app.command("validation-evaluate")
def validation_evaluate() -> None:
    import json

    from pipelines import s2core_judge_validation_pipeline as jv

    report = jv.evaluate_validation()
    typer.echo(
        json.dumps(
            {t: (v["stack"], v["validated"]) for t, v in report["tasks"].items()}
        )
    )


def main() -> None:
    from dotenv import load_dotenv

    from subliminal_learning.utils.logging_setup import configure_loguru

    load_dotenv()
    configure_loguru()
    app()


if __name__ == "__main__":
    main()
