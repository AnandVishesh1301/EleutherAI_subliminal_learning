"""Same-base proxy EAS for on-policy GKD students — keys, gates, aggregation, tensor math.

Blank et al. (arXiv:2606.00995, §4.3) read a distilled student's activation-shift direction
against the teacher's steering vector. A GKD leaf trained on base B from a teacher on base A
has no same-model teacher vector, so the proxy reads its ``v_student`` against the SDF-LoRA
teachers trained on B itself (the M2 hypothesis: the shift direction is a property of the
base). The teacher-side null is the other-trait SDF teacher on the same base: a weight-contrast
``v_teacher`` carries LoRA drift that GKD controls do not remove, and the reported statistic
``delta_eas = eas_mean(trait) - eas_mean(null)`` cancels it.

Headline per block: cosine to the **unit-mean** teacher vector (per-layer mean of the
per-seed unit vectors, so one large-norm seed cannot dominate); per-seed cosines are kept as
the spread. ``eas_mean`` uses this repo's ``blocks_v1`` window (`steering.eas.eas_mean`).

The first section is plain Python (cache keys, manifest gates, aggregation over lists) and
is unit-testable on dict manifests before any model is loaded; the second section is the
tensor math and needs torch. Disk / model orchestration lives in `pipelines.gkd_eas_pipeline`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

import torch

from subliminal_learning.steering.eas import eas_mean, eas_per_layer

SCHEMA_VERSION = 1
EAS_MEAN_CONVENTION = "blocks_v1"
DTYPE_TAG = "fp32-tf32off"

_LEAF_KEYS = (
    "base_model_repo",
    "target_animal",
    "finetune_mode",
    "prompt_pool_dir",
    "prompt_pool_manifest_sha256",
)
# Meta keys compared for the idempotent skip; run_dir / git_sha / created_at / batch_size
# are provenance, not identity (batch size does not change the fp32 mean).
_IDENTITY_META_KEYS = (
    "schema_version",
    "base_model_repo",
    "n_layers",
    "trait_animal",
    "null_animal",
    "student_adapter_sha256",
    "prompt_pool_manifest_sha256",
    "n_prompts",
    "favorite_prompt_set",
    "prompt_sha256",
    "contexts",
    "positions",
    "blocks",
    "skip_first_token",
    "teacher_refs",
    "dtype",
    "tf32",
    "eas_mean_convention",
    "random_floor_seed",
)
_FTSEED_RE = re.compile(r"^ftseed(?P<seed>\d+)$")


# ----------------------------------------------------------------- keys / cache / gates


@dataclass(frozen=True)
class GkdEasKey:
    """One (leaf, extraction context, pooling position) cell of the GKD-EAS stage."""

    run_dir: str
    context: str
    position: str

    @property
    def block(self) -> str:
        """The ``per_context`` key, ``"{context}/{position}"``."""
        return block_name(self.context, self.position)


@dataclass(frozen=True)
class LeafIdentity:
    """The identity fields a GKD leaf manifest declares (see `leaf_identity`)."""

    base_model_repo: str
    target_animal: str
    prompt_pool_dir: str
    prompt_pool_manifest_sha256: str
    is_control: bool


def block_name(context: str, position: str) -> str:
    """The ``per_context`` key for a (context, position) block."""
    return f"{context}/{position}"


def cache_key(
    prompt_sha256: str,
    model_sha256: str,
    base_repo: str,
    position: str,
    dtype_tag: str,
    *,
    skip_first_token: bool = False,
) -> str:
    """16-hex identity of one cached mean/vector; any field change yields a new key.

    Args:
        prompt_sha256: sha256 of the extraction prompts (encodes the context).
        model_sha256: Adapter weight sha256, or ``"base"`` for the clean base mean.
        base_repo: Base model repo string as it appears in manifests.
        position: ``"last"`` or ``"all"``.
        dtype_tag: Numerics tag (`DTYPE_TAG`).
        skip_first_token: Whether the ``"all"`` mean dropped each row's first real token.
    """
    parts = (
        prompt_sha256,
        model_sha256,
        base_repo,
        position,
        dtype_tag,
        "skip1" if skip_first_token else "skip0",
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def leaf_identity(manifest: Mapping) -> LeafIdentity:
    """Extract a GKD leaf's identity from its ``train_manifest.json`` dict.

    Raises:
        ValueError: A required key is missing, or ``finetune_mode`` is not ``"lora"``
            (the stage attaches the leaf as a PEFT adapter).
    """
    missing = [k for k in _LEAF_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"leaf manifest is missing {missing}")
    if manifest["finetune_mode"] != "lora":
        raise ValueError(
            f"GKD-EAS requires a LoRA leaf; finetune_mode={manifest['finetune_mode']!r}"
        )
    return LeafIdentity(
        base_model_repo=str(manifest["base_model_repo"]),
        target_animal=str(manifest["target_animal"]),
        prompt_pool_dir=str(manifest["prompt_pool_dir"]),
        prompt_pool_manifest_sha256=str(manifest["prompt_pool_manifest_sha256"]),
        is_control=bool(manifest.get("control", False)),
    )


def check_same_base(leaf_repo: str, teacher_repos: Mapping[str, str]) -> None:
    """Raise unless every teacher's ``base_model_repo`` equals the leaf's.

    Args:
        leaf_repo: The leaf manifest's ``base_model_repo``.
        teacher_repos: ``{teacher_name: base_model_repo}`` from each teacher's manifest.

    Raises:
        ValueError: Naming every teacher on a different base.
    """
    bad = {name: repo for name, repo in teacher_repos.items() if repo != leaf_repo}
    if bad:
        raise ValueError(
            f"teacher base repo != leaf base repo {leaf_repo!r}: {bad} "
            "(the proxy EAS is same-base only)"
        )


def check_prompt_pool_sha(leaf_shas: Mapping[str, str], expected: str) -> None:
    """Raise unless every leaf's ``prompt_pool_manifest_sha256`` equals `expected`.

    Args:
        leaf_shas: ``{run_dir: prompt_pool_manifest_sha256}`` per leaf.
        expected: sha256 of the on-disk pool ``manifest.json``.

    Raises:
        ValueError: Listing every mismatching leaf.
    """
    bad = {rd: sha for rd, sha in leaf_shas.items() if sha != expected}
    if bad:
        raise ValueError(
            f"prompt-pool manifest sha mismatch (expected {expected}): {bad}; all leaves in "
            "one comparison must share the pool the on-disk manifest describes"
        )


def check_slot_count(n_slots: int, n_layers: int) -> None:
    """Raise unless a ``[n_slots, H]`` stack has ``n_layers + 1`` slots."""
    if n_slots != n_layers + 1:
        raise ValueError(
            f"hidden-state stack has {n_slots} slots but the model has {n_layers} layers "
            f"(expected {n_layers + 1})"
        )


def parse_teacher_ftseed(ckpt_dir: str) -> int:
    """The ``ftseed{N}`` of a ``.../ftseed{N}/checkpoints`` teacher checkpoint dir."""
    parent = PurePath(ckpt_dir).parent.name
    m = _FTSEED_RE.match(parent)
    if m is None:
        raise ValueError(
            f"cannot parse ftseed from teacher checkpoint dir {ckpt_dir!r} "
            "(expected .../ftseed{N}/checkpoints)"
        )
    return int(m.group("seed"))


def teacher_name(animal: str, ftseed: int) -> str:
    """Dict key for one teacher, ``"{animal}/ftseed{N}"``."""
    return f"{animal}/ftseed{ftseed}"


def proxy_stats(cos_by_teacher: Mapping[str, Sequence[float]]) -> dict:
    """Aggregate per-teacher per-layer cosines over the teacher seeds.

    Returns:
        ``{"per_layer_mean": [n_slots], "eas_mean_by_teacher": {name: float},
        "eas_mean": float}`` with ``eas_mean`` under `EAS_MEAN_CONVENTION`.

    Raises:
        ValueError: No teachers, or ragged per-layer lengths.
    """
    if not cos_by_teacher:
        raise ValueError("proxy_stats needs at least one teacher")
    rows = [list(map(float, v)) for v in cos_by_teacher.values()]
    if len({len(r) for r in rows}) != 1:
        raise ValueError(
            f"ragged per-layer cosines: {[len(r) for r in rows]}"
        )
    stack = torch.tensor(rows, dtype=torch.float64)
    per_layer_mean = stack.mean(dim=0)
    by_teacher = {
        name: eas_mean(torch.tensor(r, dtype=torch.float64))
        for name, r in zip(cos_by_teacher, rows)
    }
    return {
        "per_layer_mean": per_layer_mean.tolist(),
        "eas_mean_by_teacher": by_teacher,
        "eas_mean": eas_mean(per_layer_mean),
    }


def teacher_null_delta(
    cos_trait: Sequence[float], cos_null: Sequence[float]
) -> list[float]:
    """Per-layer ``cos_trait - cos_null`` (the drift-cancelling statistic)."""
    if len(cos_trait) != len(cos_null):
        raise ValueError(
            f"per-layer length mismatch: trait {len(cos_trait)} vs null {len(cos_null)}"
        )
    return [float(a) - float(b) for a, b in zip(cos_trait, cos_null)]


def meta_matches(stored_meta: Mapping | None, current_meta: Mapping) -> bool:
    """Whether a stored ``gkd_eas.json`` meta has the same identity as `current_meta`."""
    if stored_meta is None:
        return False
    return all(
        stored_meta.get(k) == current_meta.get(k) for k in _IDENTITY_META_KEYS
    )


def build_gkd_entry(
    *,
    context: str,
    position: str,
    skip_first_token: bool,
    n_prompts: int,
    trait_animal: str,
    null_animal: str,
    cos_by_teacher: Mapping[str, torch.Tensor],
    cos_unit_mean: Mapping[str, torch.Tensor],
    v_student_norm: torch.Tensor,
    random_floor: torch.Tensor,
) -> dict:
    """Build one JSON-serializable ``per_context`` block.

    Args:
        context: ``"pile"`` or ``"favorite"``.
        position: ``"last"`` or ``"all"``.
        skip_first_token: Whether the ``"all"`` mean dropped the first real token.
        n_prompts: Prompts the block was read on.
        trait_animal: The leaf's target animal (its teachers are the signal).
        null_animal: The other-trait animal (its teachers are the null).
        cos_by_teacher: ``{"{animal}/ftseed{N}": [n_slots]}`` per-seed cosines.
        cos_unit_mean: ``{animal: [n_slots]}`` cosine to each animal's unit-mean vector.
        v_student_norm: ``[n_slots]`` per-slot L2 norm of ``v_student``.
        random_floor: ``[n_slots]`` cosine to a matched-norm random direction.

    Returns:
        The block dict (keys ``cos_to_unit_mean_{animal}``, ``eas_mean_{animal}``,
        ``delta_eas`` etc.).
    """
    per_layer = {name: t.tolist() for name, t in cos_by_teacher.items()}
    entry: dict = {
        "context": context,
        "position": position,
        "skip_first_token": skip_first_token,
        "n_prompts": n_prompts,
        "trait_animal": trait_animal,
        "null_animal": null_animal,
        "per_layer_cos_by_teacher": per_layer,
        "eas_mean_by_teacher": {},
    }
    eas_by_animal: dict[str, float] = {}
    for animal in (trait_animal, null_animal):
        seeds = {n: v for n, v in per_layer.items() if n.startswith(f"{animal}/")}
        stats = proxy_stats(seeds)
        entry["eas_mean_by_teacher"].update(stats["eas_mean_by_teacher"])
        entry[f"seed_mean_{animal}"] = {
            "per_layer": stats["per_layer_mean"],
            "eas_mean": stats["eas_mean"],
        }
        unit = cos_unit_mean[animal]
        entry[f"cos_to_unit_mean_{animal}"] = unit.tolist()
        eas_by_animal[animal] = eas_mean(unit)
        entry[f"eas_mean_{animal}"] = eas_by_animal[animal]
    entry["delta_eas"] = eas_by_animal[trait_animal] - eas_by_animal[null_animal]
    entry["delta_per_layer"] = teacher_null_delta(
        entry[f"cos_to_unit_mean_{trait_animal}"],
        entry[f"cos_to_unit_mean_{null_animal}"],
    )
    entry["v_student_norm_per_layer"] = v_student_norm.tolist()
    entry["random_floor_per_layer"] = random_floor.tolist()
    entry["random_floor_mean"] = eas_mean(random_floor)
    return entry


# ----------------------------------------------------------------- tensor math (torch)


def direction_cosine(v_a_raw: torch.Tensor, v_b_raw: torch.Tensor) -> torch.Tensor:
    """Per-slot cosine between two ``[n_slots, H]`` raw shift vectors (shape-guarded)."""
    return eas_per_layer(v_a_raw, v_b_raw)


def unit_mean_vector(
    vectors: Sequence[torch.Tensor], exclude_index: int | None = None
) -> torch.Tensor:
    """Per-slot mean of the unit-normalised ``[n_slots, H]`` vectors.

    Each member is L2-normalised per slot first, so a large-norm seed cannot dominate.
    ``exclude_index`` leaves that member out (leave-one-out when the queried vector is
    itself a member of the set).

    Raises:
        ValueError: No members remain, or the members' shapes differ.
    """
    kept = [v for i, v in enumerate(vectors) if i != exclude_index]
    if not kept:
        raise ValueError("unit_mean_vector needs at least one vector after exclusion")
    shapes = {tuple(v.shape) for v in kept}
    if len(shapes) != 1:
        raise ValueError(f"unit_mean_vector members differ in shape: {sorted(shapes)}")
    units = torch.stack(
        [v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12) for v in kept], dim=0
    )
    return units.mean(dim=0)


def random_direction_floor(v_raw: torch.Tensor, seed: int) -> torch.Tensor:
    """Per-slot cosine of ``v_raw`` to a fixed-seed random direction of matched norm.

    A scale reference only (≈ 1/sqrt(H) in magnitude), never a null distribution.
    """
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randn(v_raw.shape, generator=gen, dtype=torch.float32).to(v_raw.dtype)
    scale = v_raw.norm(dim=-1, keepdim=True) / rand.norm(dim=-1, keepdim=True).clamp(
        min=1e-12
    )
    return direction_cosine(v_raw, rand * scale)
