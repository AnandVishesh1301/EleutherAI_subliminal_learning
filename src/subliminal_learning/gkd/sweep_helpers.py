"""Grid-expansion helpers shared by the GKD sweep and its evaluation counterparts.

Pure (torch-free): student-mode tokens -> `LoRAConfig`, the per-cell student variants of a
sweep YAML, and the canonical ``(student_spec, teacher_spec)`` pair list of the cross-init
axis. `scripts/sweeps/sweep_distill_student_gkd.py` drives training with these;
`scripts/evaluate/evaluate_gkd_student.py` / `sweep_evaluate_gkd_student.py` re-derive the same cells.
"""

from __future__ import annotations

import re

from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import DEFAULT_OPTIM, optim_slug, parse_optim_slug
from subliminal_learning.models.specs import family_swap_spec, parse_base_models, spec_label

LORA_MODE_RE = re.compile(r"^lora-r(?P<r>\d+)(?:a(?P<alpha>\d+))?$")


def mode_lora(mode: str) -> LoRAConfig | None:
    """LoRAConfig for a student mode token.

    ``'full'`` -> ``None``; ``'lora-r{N}'`` -> r=α=N; ``'lora-r{R}a{A}'`` ->
    decoupled rank/scale (e.g. ``lora-r8a32`` -> r=8, α=32).

    Raises:
        ValueError: On any other token (including trailing junk like ``lora-r8a``).
    """
    if mode == "full":
        return None
    m = LORA_MODE_RE.match(mode)
    if not m:
        raise ValueError(
            f"bad student mode {mode!r}; expected 'full', 'lora-r<N>' (alpha=r), "
            "or 'lora-r<R>a<A>' (decoupled alpha)"
        )
    r = int(m.group("r"))
    alpha = int(m.group("alpha")) if m.group("alpha") is not None else r
    return LoRAConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear"
    )


def as_lr_list(value) -> list[float]:
    """A sweep YAML LR entry (scalar or list) as a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def student_variants(
    student_mode_list: list[str],
    student_lr: dict,
    full_lrs=None,
    lora_lrs=None,
    student_optims: list[str] | None = None,
) -> list[tuple[str, LoRAConfig | None, float, str, float]]:
    """Expand (student mode × optimizer × its LR list) into per-cell student variants.

    Without `student_optims` (the default) every cell is `DEFAULT_OPTIM` with momentum
    ``0.0``: full modes draw from ``full_lrs`` (CLI) or ``student_lr['full']`` (YAML),
    LoRA modes from ``lora_lrs`` / ``student_lr['lora']`` -- byte-identical to the
    pre-optimizer-axis grid.

    With `student_optims` (slugs per `gkd.layout.optim_slug`, e.g. ``sgdm0.9``) the LR
    ladder is per arm: ``student_lr['{kind}/{slug}']`` where ``kind`` is ``full`` /
    ``lora``, falling back to ``student_lr[kind]``. Optimum LRs differ by decades across
    optimizers, so a shared ladder is exactly the confound this axis exists to remove.
    A CLI LR override still applies to every optimizer (it is a shard flag, not a ladder).

    Args:
        student_mode_list: Student FT-mode tokens (``full`` / ``lora-r{R}a{A}``).
        student_lr: The sweep YAML's ``student_lr`` mapping.
        full_lrs: CLI override for full-FT LRs, or ``None``.
        lora_lrs: CLI override for LoRA LRs, or ``None``.
        student_optims: Optimizer slugs, or ``None`` for the default-optimizer grid.

    Returns:
        ``(mode, LoRAConfig|None, lr, optim, optim_momentum)`` per cell; each maps to a
        disjoint ``gkd_tag``.
    """
    slugs = student_optims or [optim_slug(DEFAULT_OPTIM)]
    variants: list[tuple[str, LoRAConfig | None, float, str, float]] = []
    for mode in student_mode_list:
        lora = mode_lora(mode)
        kind = "lora" if lora is not None else "full"
        override = lora_lrs if lora is not None else full_lrs
        for slug in slugs:
            optim, momentum = parse_optim_slug(slug)
            ladder = student_lr.get(f"{kind}/{slug}", student_lr[kind])
            for lr in override or as_lr_list(ladder):
                variants.append((mode, lora, float(lr), optim, momentum))
    return variants


def csv_list(value: str | None, cast):
    """A comma-separated CLI value as a list of `cast`-ed items, or ``None`` when empty."""
    return [cast(x.strip()) for x in value.split(",") if x.strip()] if value else None


def resolve_teacher_spec(student_spec, pairing: str):
    """Resolve the teacher base for one student under a pairing mode.

    ``same`` (default): teacher base == student base (same-init SL).
    ``family_swap``: matched-seed data<->weight swap (cross-init SL).
    """
    if pairing == "same":
        return student_spec
    if pairing == "family_swap":
        return family_swap_spec(student_spec)
    raise ValueError(
        f"bad teacher_base_pairing {pairing!r}; expected 'same' or 'family_swap'"
    )


def expand_base_pairs(base_specs, pairing: str, explicit_pairs, student_filter=None):
    """Canonical ``(student_spec, teacher_spec)`` list for the cross-init axis.

    The explicit ``teacher_student_base_pairs`` list is canonical; ``same`` /
    ``family_swap`` / ``all_pairs`` are sugar that expand into the same pair-list before
    enumeration, so tagging / provenance / resume run one code path (plan §5). Default
    (no explicit pairs, ``pairing="same"``) reproduces the prior behavior exactly.

    Args:
        base_specs: Student base specs (from the grid's ``base_models``).
        pairing: ``same`` | ``family_swap`` | ``all_pairs`` (used only when no explicit
            pairs are given).
        explicit_pairs: Optional list of ``[student_base_label, teacher_base_label]``.
        student_filter: Student base labels to keep (the ``--base-models`` shard
            override). ``None`` keeps every pair.

    Returns:
        List of ``(student_spec, teacher_spec)`` tuples.

    Raises:
        ValueError: `student_filter` selects no pair -- a shard that would silently
            train nothing (or, worse, look like it sharded while a peer shard runs
            the same cells) must fail instead.
    """
    if explicit_pairs:
        pairs = []
        for pair in explicit_pairs:
            student = parse_base_models(str(pair[0]))[0]
            teacher = parse_base_models(str(pair[1]))[0]
            pairs.append((student, teacher))
    elif pairing == "all_pairs":
        pairs = [(s, t) for s in base_specs for t in base_specs]
    else:
        pairs = [(s, resolve_teacher_spec(s, pairing)) for s in base_specs]
    if student_filter is None:
        return pairs
    kept = [p for p in pairs if spec_label(p[0]) in student_filter]
    if not kept:
        raise ValueError(
            f"--base-models {sorted(student_filter)} matches no student base in the "
            f"pair list {[spec_label(s) for s, _ in pairs]}"
        )
    return kept
