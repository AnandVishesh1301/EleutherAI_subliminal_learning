"""CLI: load a saved LoRA teacher adapter fresh and verify its owl bias."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.verify_teacher_pipeline import VerifyTeacherConfig, run
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    experiment_config: Path = Path("configs/experiment/owl_bias_pythia160m.yaml"),
) -> None:
    """Verify the owl bias of a cached trained LoRA teacher adapter."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN)
    configure_loguru()
    cfg = load_config(paths_config, model_config, experiment_path=experiment_config)
    adapter_dir = cfg.paths.output_root / cfg.experiment_id / "checkpoints"
    run_cfg = VerifyTeacherConfig(
        paths=cfg.paths,
        hub_org=cfg.model.hub_org,
        spec=cfg.spec,
        eval=cfg.eval,
        adapter_dir=adapter_dir,
        experiment_id=cfg.experiment_id,
        device=device,
    )
    run(run_cfg)


if __name__ == "__main__":
    typer.run(main)
