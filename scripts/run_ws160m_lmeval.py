"""CLI for the ws160m lm-eval tail: submit one partition's benchmark queue."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.ws160m_campaign_pipeline import CampaignGateError, driver_config
from pipelines.ws160m_lmeval_pipeline import LmevalOptions, run
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


@app.command()
def submit(
    partition: str = typer.Option(..., "--partition"),
    paths_config: Path = typer.Option(Path(plan.PATHS_CONFIG)),
    queue_root: Path = typer.Option(None, help="jobq root (default: paths YAML)."),
    code_sha: str = typer.Option(..., help="Pushed pin every node holds a venv for."),
    require_results: bool = True,
    node: str = typer.Option(None, help="Host to run on (default: home node)."),
    dry_run: bool = False,
) -> None:
    """Submit `ws160m-<partition>-lmeval` once the partition has finalized."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    cfg = driver_config(
        partition,
        output_root=Path(paths["output_root"]),
        queue_root=Path(queue_root or resolve_queue_root(paths)).resolve(),
        repo_root=main_checkout_root(Path.cwd()),
        dry_run=dry_run,
        code_sha=code_sha,
    )
    opts = LmevalOptions(
        partition=partition,
        paths_config=Path(paths_config),
        require_results=require_results,
        node=node,
    )
    try:
        record = run(cfg, opts)
    except CampaignGateError as err:
        logger.error("gate failed: {}", err)
        raise typer.Exit(code=2) from err
    typer.echo(yaml.safe_dump(record, sort_keys=False))


if __name__ == "__main__":
    app()
