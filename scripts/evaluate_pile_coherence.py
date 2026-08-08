"""CLI: evaluate one model's Pile CE (base variant or attached adapter) + ΔCE.

`--model` alone scores a bare base variant, writing to
`{output_root}/coherence/{version}/base/{model_dir_name}/ce.json` (+ per_token_loss.npy);
`--model` + `--adapter-dir` scores an attached teacher/student adapter and computes ΔCE
against the adapter's own base variant (whose base record must already exist — run the
base variant first), writing to `{run_dir}/pile_coherence/{version}.json` (+
`{version}_per_token_loss.npy`) where `run_dir` is the adapter's ftseed-level dir.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.pile_coherence_pipeline import (
    EvaluatePileCoherenceConfig,
    run_evaluate,
)
from subliminal_learning.coherence.config import load_pile_coherence_config
from subliminal_learning.config import load_config
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    model: str = typer.Option(
        ..., help="One base-model token (e.g. 'original', 'data-seed1')."
    ),
    adapter_dir: Path = typer.Option(
        None, help="Adapter checkpoints leaf dir (teacher/student LoRA). Needs --model."
    ),
    device: str = "cuda:0",
    force: bool = typer.Option(False, help="Recompute + overwrite the CE record."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    coherence_config: Path = Path("configs/experiment/pile_coherence.yaml"),
) -> None:
    """Score one model on the Pile shard; with --adapter-dir also compute ΔCE."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN)
    configure_loguru()
    specs = parse_base_models(model)
    if len(specs) != 1:
        raise typer.BadParameter(
            f"--model must resolve to exactly one base variant, got {len(specs)} "
            f"from {model!r}."
        )
    cfg = load_config(paths_config, model_config)
    coherence = load_pile_coherence_config(coherence_config)
    run_cfg = EvaluatePileCoherenceConfig(
        paths=cfg.paths,
        hub_org=cfg.model.hub_org,
        coherence=coherence,
        spec=specs[0],
        adapter_dir=adapter_dir,
        device=device,
        force=force,
    )
    run_evaluate(run_cfg)


if __name__ == "__main__":
    typer.run(main)
