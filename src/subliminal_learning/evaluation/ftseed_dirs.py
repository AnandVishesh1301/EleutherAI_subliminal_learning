"""Discover the per-seed ``checkpoints/`` dirs of one param cell (``ftseed{N}`` / ``gseed{N}``).

Shared by the preference, SDF-verification and steerability pipelines and their sweeps; the
weight gates (incomplete / non-finite checkpoints) are what keep a diverged adapter from
ever being loaded onto a shared GPU.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from subliminal_learning.utils.run_io import checkpoint_is_complete, checkpoint_nonfinite_count


def discover_ftseed_dirs(
    param_dir: Path,
    seed_prefix: str = "ftseed",
    skipped_incomplete: list[int] | None = None,
    skipped_nonfinite: list[int] | None = None,
    *,
    check_weights: bool = True,
) -> list[tuple[int, Path]]:
    """Return sorted (seed, checkpoints_dir) for each ``{seed_prefix}{N}`` under `param_dir`.

    ``seed_prefix="ftseed"`` (default) discovers teacher fine-tune seeds; ``"gseed"``
    discovers GKD-student training seeds. Directories without a ``checkpoints/`` subdir
    or an unparseable seed index are skipped with a warning.

    With `check_weights` (the default) two further gates apply:

    A seed whose ``checkpoints/`` exists but holds no saved weights -- an interrupted or
    still-in-progress save (killed job, OOM, node reboot, or a sibling seed still
    mid-training) -- is skipped, via ``checkpoint_is_complete``: loading it would
    raise and abort the ENTIRE cell's eval, including sibling seeds that trained fine.
    Its seed index is appended to `skipped_incomplete` when that list is supplied, so the
    caller can record which seeds a partial evaluation omitted.

    A seed whose weights are *complete but non-finite* -- a diverged run saved as all-NaN --
    is skipped the same way, via ``checkpoint_nonfinite_count``, and recorded in
    `skipped_nonfinite`. This gate exists because loading such a checkpoint is actively
    destructive, not merely useless: its all-NaN logits reach ``torch.multinomial``, whose
    device-side assert poisons the CUDA context and kills co-tenant jobs sharing the GPU.
    Skipping before the load is the only place that damage can be avoided for adapters that
    are already on disk.

    ``check_weights=False`` is the steerability screen's variant: every ``ftseed{N}`` with a
    ``checkpoints/`` dir is returned, gates and all.
    """
    found: list[tuple[int, Path]] = []
    for d in sorted(param_dir.glob(f"{seed_prefix}*")):
        ckpt = d / "checkpoints"
        if not ckpt.is_dir():
            logger.warning("Skipping {} (no checkpoints/ dir)", d)
            continue
        try:
            seed = int(d.name.removeprefix(seed_prefix))
        except ValueError:
            logger.warning("Skipping {} (unparseable seed)", d)
            continue
        if check_weights:
            if not checkpoint_is_complete(ckpt):
                logger.warning(
                    "Skipping {} (INCOMPLETE checkpoint -- interrupted/mid-training save, "
                    "no saved weights)", d,
                )
                if skipped_incomplete is not None:
                    skipped_incomplete.append(seed)
                continue
            n_bad = checkpoint_nonfinite_count(ckpt)
            if n_bad:
                logger.warning(
                    "Skipping {} (DIVERGED checkpoint -- {} non-finite weight value(s); "
                    "loading it would device-assert and kill co-tenant jobs on this GPU)",
                    d, n_bad,
                )
                if skipped_nonfinite is not None:
                    skipped_nonfinite.append(seed)
                continue
        found.append((seed, ckpt))
    return found
