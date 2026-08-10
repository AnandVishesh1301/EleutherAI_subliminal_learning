"""Resolve/load an arbitrary CKA operand: a PolyPythia base or a finetuned run.

The base-suite CKA pipeline (``pipelines/cka_pipeline.py``) compares the 7 base
models. The explicit ``--model-a``/``--model-b`` mode compares any two operands,
each given as a **model reference string** of one of two forms:

1. A base-spec token from ``teachers.model_specs.parse_base_models``' single-token
   vocabulary — ``original``, ``pythia-160m``, ``data-seed1``,
   ``weight-seed3@step3000`` — loaded exactly as the base suite loads it.
2. A finetuned run reference: a filesystem path (absolute, or relative to
   ``output_root``) pointing at a run dir (``.../ftseed1``), its ``checkpoints/``
   subdir, or any dir holding the checkpoint files directly. The base model is read
   back from the run's ``train_manifest.json``; the adapter is merged so the CKA
   extractor (which needs a bare ``GPTNeoXForCausalLM.gpt_neox``) keeps working.

Resolution (``resolve_ref``) does no model loading — it only reads the manifest and
classifies the checkpoint — so the CLI can derive labels/identity cheaply and gate
before paying for a load. ``load_ref`` does the actual (merging) load behind
injectable loader seams so tests never touch HF/PEFT.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel

from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.evaluation.teacher_loading_io import (
    detect_finetune_mode,
    load_finetuned_teacher,
)
from subliminal_learning.models.loading_io import load_model
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    model_ref,
)
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    spec_from_base_name,
    spec_label,
)

# Filesystem tree -> label prefix (mirrors the run-dir layout roots).
_TREE_PREFIX = {
    "teachers": "t",
    "sdf_teachers": "sdf",
    "students": "s",
    "gkd_students": "g",
}


@dataclass(frozen=True)
class ModelRef:
    """A resolved (but not yet loaded) CKA operand.

    ``kind`` is ``"base"`` or ``"finetuned"``. ``spec`` is the PolyPythia base spec
    either way (for a finetuned ref it is the base the run was fine-tuned from). The
    remaining fields are populated only for finetuned refs.
    """

    kind: str
    spec: PythiaModelSpec
    raw: str
    base_repo: str
    run_dir: Path | None = None
    checkpoint_dir: Path | None = None
    finetune_mode: str | None = None


def is_base_token(ref: str) -> bool:
    """True if `ref` is a single base-spec token (not a filesystem path).

    A path always carries a separator under the run-dir layouts; a base token never
    does. We additionally require the token to parse to exactly one spec, which
    rejects aggregate selectors (``all7``, ``seeds``) and comma lists.
    """
    if "/" in ref:
        return False
    try:
        specs = parse_base_models(ref)
    except ValueError:
        return False
    return len(specs) == 1


def _full_ft_weight_files(d: Path) -> list[Path]:
    """Full-FT weight files in `d`, sorted (safetensors preferred over .bin)."""
    files = sorted(d.glob("*.safetensors"))
    if not files:
        files = sorted(d.glob("pytorch_model*.bin"))
    return files


def _is_checkpoint_dir(d: Path) -> bool:
    """True if `d` directly holds a LoRA adapter or a full-FT checkpoint."""
    if (d / "adapter_config.json").exists():
        return True
    return (d / "config.json").exists() and bool(_full_ft_weight_files(d))


def _read_manifest_base(run_dir: Path) -> str:
    """The base repo/label recorded in the run's ``train_manifest.json``.

    Prefers ``base_model_repo`` (the ``@revision``-qualified Hub id), falling back to
    ``base_model`` (the model-dir basename). Raises a clear ``ValueError`` naming the
    file on a missing/unparsable manifest or a manifest with neither key.
    """
    import json

    manifest = run_dir / "train_manifest.json"
    if not manifest.exists():
        raise ValueError(f"no train_manifest.json under run dir {run_dir}")
    try:
        m = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"unparsable train_manifest.json at {manifest}: {e}") from e
    base = m.get("base_model_repo") or m.get("base_model")
    if not base:
        raise ValueError(
            f"train_manifest.json at {manifest} has neither 'base_model_repo' nor "
            "'base_model' — cannot resolve the base model"
        )
    return str(base)


def resolve_ref(ref: str, output_root: Path, hub_org: str) -> ModelRef:
    """Classify and resolve a model reference string WITHOUT loading the model.

    Base tokens resolve straight to their spec. A path is made absolute against
    ``output_root``; if it has a ``checkpoints/`` child that is the checkpoint dir and
    the path is the run dir, otherwise the path itself must be the checkpoint dir. The
    base spec is read from the run's manifest.

    Raises:
        ValueError: on a base token that fails to parse, a path with neither a
            ``checkpoints/`` child nor checkpoint files, or an unresolvable manifest.
    """
    if is_base_token(ref):
        spec = parse_base_models(ref)[0]
        return ModelRef("base", spec, raw=ref, base_repo=model_ref(spec, hub_org))

    p = Path(ref)
    if not p.is_absolute():
        p = output_root / p
    if (p / "checkpoints").is_dir():
        run_dir, checkpoint_dir = p, p / "checkpoints"
    elif _is_checkpoint_dir(p):
        run_dir, checkpoint_dir = p.parent, p
    else:
        raise ValueError(
            f"model reference {ref!r} (resolved to {p}) has neither a 'checkpoints/' "
            "child nor a LoRA adapter / full-FT checkpoint — cannot load it"
        )
    base_repo = _read_manifest_base(run_dir)
    spec = spec_from_base_name(base_repo)
    return ModelRef(
        "finetuned",
        spec,
        raw=ref,
        base_repo=base_repo,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        finetune_mode=detect_finetune_mode(checkpoint_dir),
    )


def _compact(seg: str) -> str:
    """Compact one path segment for a label (mirrors ``_derive_run_tag``' compact())."""
    return (
        seg.replace("pythia-", "")
        .replace("-data-seed", "-ds")
        .replace("-weight-seed", "-ws")
        .replace("ftseed", "fs")
        .replace("gseed", "gs")
    )


def _sanitize(s: str) -> str:
    """Filesystem-safe: keep alnum plus ``-_.``, collapse everything else to ``-``."""
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in s)


def ref_label(mref: ModelRef, output_root: Path) -> str:
    """Filesystem-safe label for pair labels / filenames / the run tag.

    Base refs keep ``spec_label``. Finetuned refs compact the ``output_root``-relative
    run-dir coordinates behind a tree prefix
    (``teachers/dog/pythia-160m/ep15_lr1e-4_r32a32/ftseed1`` ->
    ``t-dog-160m-ep15_lr1e-4_r32a32-fs1``). A run dir outside ``output_root`` or under
    an unrecognized tree falls back to its last 3 components joined with ``-``.
    """
    if mref.kind == "base":
        return spec_label(mref.spec)
    run_dir = mref.run_dir
    assert run_dir is not None
    try:
        rel = run_dir.relative_to(output_root).parts
    except ValueError:
        rel = ()
    if rel and rel[0] in _TREE_PREFIX:
        prefix = _TREE_PREFIX[rel[0]]
        body = "-".join(_compact(seg) for seg in rel[1:])
        return _sanitize(f"{prefix}-{body}")
    return _sanitize("-".join(run_dir.parts[-3:]))


def checkpoint_hash(mref: ModelRef) -> str | None:
    """Content hash of the checkpoint weights (None for a base ref).

    LoRA reuses the eval pipeline's ``adapter_identity`` (imported lazily to avoid a
    src->pipelines import at module load); full-FT hashes the weight file(s). So an
    in-place ``--force`` retrain that rewrites the checkpoint drifts the run identity.
    """
    if mref.kind == "base" or mref.checkpoint_dir is None:
        return None
    if mref.finetune_mode == "lora":
        from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

        return adapter_identity(mref.checkpoint_dir)
    h = hashlib.sha256()
    for wf in _full_ft_weight_files(mref.checkpoint_dir):
        with open(wf, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def ref_provenance(mref: ModelRef) -> dict:
    """The provenance dict recorded in the explicit-mode run identity / meta.json."""
    return {
        "ref": mref.raw,
        "run_dir": str(mref.run_dir) if mref.run_dir else None,
        "checkpoint_dir": str(mref.checkpoint_dir) if mref.checkpoint_dir else None,
        "finetune_mode": mref.finetune_mode or "base",
        "base_repo": mref.base_repo,
        "checkpoint_hash": checkpoint_hash(mref),
    }


def load_ref(
    mref: ModelRef,
    paths: PathsConfig,
    model_cfg: ModelConfig,
    device: str,
    *,
    load_base=load_model,
    load_ft=load_finetuned_teacher,
) -> torch.nn.Module:
    """Load `mref` in eval fp32, merging LoRA adapters so ``.gpt_neox`` is present.

    Loader functions are injectable so tests can exercise this seam without HF/PEFT.
    """
    if mref.kind == "base":
        return load_base(mref.spec, paths.hf_cache_root, model_cfg.hub_org, device)
    model = load_ft(
        mref.spec,
        mref.checkpoint_dir,
        paths.hf_cache_root,
        model_cfg.hub_org,
        device,
        torch.float32,
        mref.finetune_mode,
    )
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    assert hasattr(model, "gpt_neox"), (
        "merged finetuned model must expose 'gpt_neox' for the CKA extractor"
    )
    model.eval()
    return model
