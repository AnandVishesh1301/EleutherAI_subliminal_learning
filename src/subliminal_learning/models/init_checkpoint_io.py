"""Start a student from a fine-tuned checkpoint instead of its untuned base.

The reversal campaigns keep training an already-trained student: its own LoRA adapter
(loaded *trainable*, so the reversed student is again one adapter on the Hub base) or its
own full weights. Two silent no-train traps are closed here: `PeftModel.from_pretrained`
loads an adapter frozen unless ``is_trainable=True``, and a loaded full model must keep
every parameter trainable. Both shapes are checked against the expected LoRA settings and
the base architecture before training starts.

``merge=True`` folds a LoRA start into its base first, then trains the merged weights in
full or through a fresh adapter (the run's LoRA settings, `training.lora_io`).
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import torch
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    PreTrainedModel,
    set_seed,
)

from subliminal_learning.config import LoRAConfig
from subliminal_learning.evaluation.teacher_loading_io import detect_finetune_mode
from subliminal_learning.models.loading_io import load_model
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.utils.io import file_sha256

_WEIGHTS = MappingProxyType(
    {"lora": "adapter_model.safetensors", "full": "model.safetensors"}
)
_ALL_LINEAR = "all-linear"


def init_checkpoint_identity(checkpoint_dir: Path) -> dict:
    """Identity record of a start checkpoint (path, shape, weight and manifest sha256).

    Raises:
        FileNotFoundError: The checkpoint has no weight file.
    """
    ckpt = Path(checkpoint_dir).resolve()
    mode = detect_finetune_mode(ckpt)
    weights = ckpt / _WEIGHTS[mode]
    if not weights.is_file():
        raise FileNotFoundError(f"no {weights.name} in start checkpoint {ckpt}")
    manifest = ckpt.parent / "train_manifest.json"
    return {
        "path": str(ckpt),
        "mode": mode,
        "weights_sha256": file_sha256(weights),
        "manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
    }


def _linear_suffixes(base: PreTrainedModel) -> set[str]:
    """Names of the base's linear layers PEFT's ``"all-linear"`` wraps (no output head)."""
    head = base.get_output_embeddings()
    return {
        name.split(".")[-1]
        for name, module in base.named_modules()
        if isinstance(module, torch.nn.Linear) and module is not head
    }


def _expected_targets(lora: LoRAConfig, base_linear: set[str]) -> set[str]:
    targets = lora.resolve_target_modules()
    if targets == _ALL_LINEAR:
        return base_linear
    if isinstance(targets, str):
        raise ValueError(f"unsupported target_modules {targets!r} for a start adapter")
    return set(targets)


def _check_adapter(
    model: PeftModel, lora: LoRAConfig, base_linear: set[str], ckpt: Path
) -> None:
    got = model.peft_config["default"]
    expected = {
        "r": lora.r,
        "lora_alpha": lora.lora_alpha,
        "lora_dropout": lora.lora_dropout,
        "target_modules": _expected_targets(lora, base_linear),
        "modules_to_save": set(lora.modules_to_save),
        "layers_to_transform": list(lora.layers_to_transform) or None,
    }
    found = {
        "r": got.r,
        "lora_alpha": got.lora_alpha,
        "lora_dropout": got.lora_dropout,
        "target_modules": {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, LoraLayer)
        },
        "modules_to_save": set(got.modules_to_save or ()),
        "layers_to_transform": got.layers_to_transform,
    }
    if found != expected:
        raise ValueError(f"start adapter {ckpt} has {found}, expected {expected}")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(
        "lora_" not in n for n in trainable if "modules_to_save" not in n
    ):
        raise RuntimeError(
            f"start adapter {ckpt}: trainable set is not exactly the adapter "
            f"({len(trainable)} trainable tensors)"
        )


def _fresh_adapter(
    model: PreTrainedModel, lora: LoRAConfig, seed: int, ckpt: Path
) -> PeftModel:
    """Wrap `model` in a new trainable adapter initialised right after `set_seed(seed)`."""
    base_linear = _linear_suffixes(model)
    set_seed(seed)
    wrapped = get_peft_model(model, build_lora_config(lora))
    _check_adapter(wrapped, lora, base_linear, ckpt)
    return wrapped


def _load_full(
    spec: PythiaModelSpec,
    ckpt: Path,
    hf_cache_root: Path,
    hub_org: str,
    device: str,
    dtype: torch.dtype,
) -> PreTrainedModel:
    base = load_model(spec, hf_cache_root, hub_org, "cpu", dtype=dtype)
    shapes = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    del base
    cls = (
        AutoModelForCausalLM
        if getattr(spec, "repo_id", None) is not None
        else GPTNeoXForCausalLM
    )
    model = cls.from_pretrained(str(ckpt), torch_dtype=dtype)
    got = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if got != shapes:
        raise ValueError(
            f"full start checkpoint {ckpt} does not match the base architecture"
        )
    model.requires_grad_(True)
    model.to(device)
    return model


def load_student_init(
    spec: PythiaModelSpec,
    checkpoint_dir: Path,
    hf_cache_root: Path,
    hub_org: str,
    device: str,
    lora: LoRAConfig | None,
    dtype: torch.dtype = torch.float32,
    merge: bool = False,
    seed: int | None = None,
) -> PreTrainedModel:
    """Load a trained student to continue training it.

    Args:
        spec: The student's base model (the adapter's base, or the architecture a full
            checkpoint must match).
        checkpoint_dir: The start student's ``checkpoints/`` directory.
        hf_cache_root: Hugging Face cache dir.
        hub_org: Hub org name.
        device: Torch device.
        lora: The run's LoRA settings (must equal the start adapter's), or ``None`` for a
            full fine-tune (the checkpoint must then be a full model).
        dtype: Weight dtype (fp32 by repo convention).
        merge: Merge a LoRA start into its base, then train the merged model in full
            (`lora` ``None``) or through a fresh adapter built from `lora` (either
            start shape).
        seed: Run seed, set right before the fresh adapter is built (required when
            `merge` and `lora` are both set).

    Returns:
        A trainable `PeftModel` (LoRA: only the adapter trains) or a full model with every
        parameter trainable, on `device`.

    Raises:
        ValueError: Checkpoint shape or LoRA settings disagree with the run, or a full
            checkpoint's tensors differ in name or shape from the base model's.
        RuntimeError: The trainable parameter set is wrong.
    """
    ckpt = Path(checkpoint_dir)
    mode = detect_finetune_mode(ckpt)
    if merge:
        if lora is not None and seed is None:
            raise ValueError("a fresh adapter on a merged start needs the run seed")
        if mode == "full":
            model = _load_full(spec, ckpt, hf_cache_root, hub_org, device, dtype)
        else:
            base = load_model(spec, hf_cache_root, hub_org, device, dtype=dtype)
            model = PeftModel.from_pretrained(base, str(ckpt)).merge_and_unload(
                safe_merge=True
            )
            del model.peft_config
            model.requires_grad_(True)
        return model if lora is None else _fresh_adapter(model, lora, seed, ckpt)
    if (lora is None) != (mode == "full"):
        raise ValueError(
            f"start checkpoint {ckpt} is {mode} but the run is "
            f"{'full' if lora is None else 'lora'}"
        )
    if mode == "lora":
        base = load_model(spec, hf_cache_root, hub_org, device, dtype=dtype)
        base_linear = _linear_suffixes(base)
        model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True)
        _check_adapter(model, lora, base_linear, ckpt)
        return model
    return _load_full(spec, ckpt, hf_cache_root, hub_org, device, dtype)
