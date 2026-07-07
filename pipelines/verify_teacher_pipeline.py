"""Load a saved LoRA teacher adapter fresh and score its owl-bias verification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from peft import PeftModel

from subliminal_learning.config import EvalConfig, PathsConfig
from subliminal_learning.evaluation.verify import TeacherVerificationResult
from subliminal_learning.evaluation.verify_io import verify_teacher_bias
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec


@dataclass(frozen=True)
class VerifyTeacherConfig:
    """Load a saved adapter and verify its animal bias.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF Hub name. 
        spec: Base teacher model the adapter was trained from.
        eval: Verification prefix, target/baseline animals, margin threshold.
        adapter_dir: Checkpoint dir for saved LoRA adapter.
        experiment_id: Run name
        device: Torch device.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    eval: EvalConfig
    adapter_dir: Path
    experiment_id: str
    device: str


def run(cfg: VerifyTeacherConfig) -> TeacherVerificationResult:
    """Load the base model + saved adapter, then verify the owl bias.

    Args:
        cfg: Model, adapter path dir, and verification settings.

    Returns:
        Dominance and margin-ratio check results.
    """
    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    base_model = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
    model = PeftModel.from_pretrained(base_model, str(cfg.adapter_dir))
    model.eval()

    result = verify_teacher_bias(model, tokenizer, cfg.device, cfg.eval)

    results_dir = cfg.paths.output_root / cfg.experiment_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "teacher_verification.json"
    results_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info(
        "passed_dominance={} passed_margin={} probability_ratio={} -> {}",
        result.passed_dominance,
        result.passed_margin,
        result.probability_ratio,
        results_path,
    )
    return result
