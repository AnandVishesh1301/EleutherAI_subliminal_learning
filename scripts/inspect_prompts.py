"""CLI: run prompts through a PolyPythias model and print completions."""

from __future__ import annotations

from pathlib import Path

import typer

from pipelines.inspect_prompts_pipeline import PromptInspectionConfig, run
from subliminal_learning.config import load_config
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    size: str = "410m",
    seed: int = 0,
    variant: str = typer.Option(
        "full",
        help="PolyPythias variant: full (default), data (160m only), weight (160m only)",
    ),
    revision: str = typer.Option(
        None, help="Checkpoint branch, e.g. step3000; default = final/main"
    ),
    prompt: list[str] = typer.Option(
        ["The capital of France is", "Once upon a time,"],
        "--prompt",
        help="Repeatable; smoke-test default prompt",
    ),
    max_new_tokens: int = 40,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_tars.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Load one PolyPythias model and print its completion for each prompt."""
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    spec = PythiaModelSpec(size=size, seed=seed, variant=variant, revision=revision)
    run_cfg = PromptInspectionConfig(
        paths=cfg.paths,
        spec=spec,
        hub_org=cfg.model.hub_org,
        device=device,
        prompts=tuple(prompt),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    run(run_cfg)


if __name__ == "__main__":
    typer.run(main)
