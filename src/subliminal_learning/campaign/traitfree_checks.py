"""Identity checks of `traitfree` placebo / tb student leaves (pure).

A placebo leaf must reproduce its source treatment leaf's `run_identity` except the
teacher (`teacher_tag`, `teacher_manifest_sha256`); a tb leaf except the control kind
(`control`, `teacher_tag`, `teacher_manifest_sha256`). `check_leaf` is the post-train
gate (`scripts.verify_traitfree_leaf`); `identity_problems` is shared with the
pre-train preflight (`scripts.preflight_traitfree_configs`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

KIND_PLACEBO = "placebo"
KIND_TB = "tb"
KINDS: tuple[str, ...] = (KIND_PLACEBO, KIND_TB)
ALLOWED_IDENTITY_DIFF: MappingProxyType[str, frozenset[str]] = MappingProxyType(
    {
        KIND_PLACEBO: frozenset({"teacher_tag", "teacher_manifest_sha256"}),
        KIND_TB: frozenset({"control", "teacher_tag", "teacher_manifest_sha256"}),
    }
)
# Manifest body fields every leaf must share with its source leaf manifest.
SOURCE_MANIFEST_KEYS: tuple[str, ...] = ("probe", "loss")
XBASE_CONTROL_KIND = "xbase_teacher"


def _plain(value: Any) -> Any:
    """`value` as JSON would store it (tuples become lists)."""
    return json.loads(json.dumps(value))


def dict_diff(got: Any, want: Any, path: str = "") -> list[str]:
    """Human-readable differences between two JSON-like values.

    Args:
        got: Value under test.
        want: Reference value.
        path: Dotted key prefix of this level.

    Returns:
        One line per differing leaf (empty when equal).
    """
    got, want = _plain(got), _plain(want)
    if isinstance(got, dict) and isinstance(want, dict):
        out: list[str] = []
        for key in sorted(set(got) | set(want)):
            sub = f"{path}.{key}" if path else key
            if key not in got:
                out.append(f"{sub}: missing (source {want[key]!r})")
            elif key not in want:
                out.append(f"{sub}: unexpected {got[key]!r} (absent in source)")
            else:
                out.extend(dict_diff(got[key], want[key], sub))
        return out
    if got != want:
        return [f"{path or '<value>'}: {got!r} != source {want!r}"]
    return []


def identity_problems(
    identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    kind: str,
    expected_teacher_tag: str | None,
    *,
    teacher_sha_known: bool = True,
) -> list[str]:
    """Problems of a leaf `run_identity` against its source leaf's.

    Args:
        identity: The leaf's (or the resolved config's) `run_identity`.
        source_identity: The source treatment leaf's `run_identity`.
        kind: `KIND_PLACEBO` or `KIND_TB`.
        expected_teacher_tag: The placebo teacher tag (placebo only).
        teacher_sha_known: Whether `identity` carries the trained placebo teacher's
            manifest sha (False before the teacher exists).

    Returns:
        Problems (empty when the leaf is a faithful copy).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown leaf kind {kind!r}")
    allowed = ALLOWED_IDENTITY_DIFF[kind]
    problems = dict_diff(
        {k: v for k, v in identity.items() if k not in allowed},
        {k: v for k, v in source_identity.items() if k not in allowed},
    )
    tag = identity.get("teacher_tag")
    sha = identity.get("teacher_manifest_sha256")
    if kind == KIND_PLACEBO:
        if expected_teacher_tag is not None and tag != expected_teacher_tag:
            problems.append(f"teacher_tag: {tag!r} != placebo {expected_teacher_tag!r}")
        if tag == source_identity.get("teacher_tag"):
            problems.append(f"teacher_tag: {tag!r} is the source (trait) teacher's")
        if teacher_sha_known:
            if not sha:
                problems.append("teacher_manifest_sha256: missing for a placebo leaf")
            elif sha == source_identity.get("teacher_manifest_sha256"):
                problems.append("teacher_manifest_sha256: the source (trait) teacher's")
    else:
        if identity.get("control") is not True:
            problems.append(f"control: {identity.get('control')!r} != True")
        if tag != "":
            problems.append(f"teacher_tag: {tag!r} != '' for a tb control")
        if sha is not None:
            problems.append(
                f"teacher_manifest_sha256: {sha!r} != None for a tb control"
            )
    return problems


def check_leaf(
    manifest: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    source_extra: Mapping[str, Any],
    kind: str,
    expected_teacher_tag: str | None,
    source_manifest: Mapping[str, Any] | None = None,
) -> list[str]:
    """Problems of a trained leaf's `train_manifest.json` against its source cell.

    Args:
        manifest: The trained leaf's manifest.
        source_identity: The source treatment leaf's `run_identity`.
        source_extra: The cell's `source_extra` (`n_train_steps`, `n_train_prompts`).
        kind: `KIND_PLACEBO` or `KIND_TB`.
        expected_teacher_tag: The placebo teacher tag (placebo only).
        source_manifest: The source leaf's manifest, for `SOURCE_MANIFEST_KEYS`.

    Returns:
        Problems (empty when the leaf passes).
    """
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        return ["manifest has no run_identity"]
    problems = identity_problems(identity, source_identity, kind, expected_teacher_tag)
    for key in ("n_train_steps", "n_train_prompts"):
        if manifest.get(key) != source_extra[key]:
            problems.append(
                f"{key}: {manifest.get(key)!r} != source {source_extra[key]!r}"
            )
    pool = source_identity.get("prompt_pool_manifest_sha256")
    if manifest.get("prompt_pool_manifest_sha256") != pool:
        problems.append(
            f"prompt_pool_manifest_sha256: {manifest.get('prompt_pool_manifest_sha256')!r}"
            f" != source {pool!r}"
        )
    if source_manifest is not None:
        for key in SOURCE_MANIFEST_KEYS:
            problems.extend(dict_diff(manifest.get(key), source_manifest.get(key), key))
    control_kind = manifest.get("control_kind")
    if kind == KIND_TB and control_kind != XBASE_CONTROL_KIND:
        problems.append(f"control_kind: {control_kind!r} != {XBASE_CONTROL_KIND!r}")
    if kind == KIND_PLACEBO and control_kind is not None:
        problems.append(f"control_kind: {control_kind!r} for a placebo leaf")
    return problems


def expected_train_steps(
    n_prompts: int,
    batch_size: int,
    grad_accum: int,
    num_epochs: float,
    max_steps: int,
    *,
    drop_last: bool,
) -> int:
    """Optimizer steps HF `Trainer` runs for one GKD student.

    Mirrors `transformers` 4.54 `Trainer.set_initial_training_values`: a positive
    `max_steps` wins; otherwise ``ceil(num_epochs * ceil(len_dataloader / accum))``
    with ``len_dataloader`` rounded down under `drop_last` (prompt subsets).
    """
    if max_steps > 0:
        return max_steps
    batches = (
        n_prompts // batch_size if drop_last else math.ceil(n_prompts / batch_size)
    )
    per_epoch = max(batches // grad_accum + int(batches % grad_accum > 0), 1)
    return math.ceil(num_epochs * per_epoch)
