"""CLI: generate the owl-bias SFT dataset and LoRA fine-tune the Pythia teacher."""

from __future__ import annotations

from pathlib import Path

import typer

from pipelines.train_teacher_pipeline import TrainTeacherConfig, run
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_tars.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_owl_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/owl_bias_pythia160m.yaml"),
) -> None:
    """LoRA fine-tune the owl-bias teacher and save the adapter to output_root."""
    configure_loguru()
    cfg = load_config(
        paths_config,
        model_config,
        training_path=training_config,
        experiment_path=experiment_config,
    )
    run_cfg = TrainTeacherConfig(
        paths=cfg.paths,
        hub_org=cfg.model.hub_org,
        spec=cfg.spec,
        lora=cfg.lora,
        train=cfg.train,
        dataset=cfg.dataset,
        phrase_bank_path=cfg.phrase_bank_path,
        experiment_id=cfg.experiment_id,
        device=device,
    )
    run(run_cfg)


if __name__ == "__main__":
    typer.run(main)
