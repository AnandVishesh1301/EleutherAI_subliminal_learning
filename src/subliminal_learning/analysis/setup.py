"""Notebook environment bootstrap: locate the repo root, read ``output_root``, set plot style.

Encapsulates the boilerplate ``CELL 2`` that opened every analysis notebook (walking up to the
machine-local ``configs/paths/paths_local.yaml`` and reading its ``output_root``). No ``sys.path``
manipulation: the package is installed editable, so ``import subliminal_learning.*`` already works.
matplotlib is imported lazily inside `set_plot_style` so importing this module stays matplotlib-free.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PATHS_YAML = "configs/paths/paths_local.yaml"


def find_root(start: Path | None = None) -> Path:
    """Walk up from `start` to the first directory containing ``configs/paths/paths_local.yaml``.

    Args:
        start: Directory to start the upward walk from (default: ``Path.cwd()``).

    Returns:
        The repo root -- the nearest ancestor (including `start` itself) holding the paths file.

    Raises:
        FileNotFoundError: if no ancestor contains the machine-local paths file.
    """
    start = Path.cwd() if start is None else Path(start)
    for d in (start, *start.parents):
        if (d / _PATHS_YAML).exists():
            return d
    raise FileNotFoundError(
        f"Could not locate {_PATHS_YAML} walking up from {str(start)!r}. "
        "Run from inside a checkout whose machine-local paths file has been filled in."
    )


def bootstrap(start: Path | None = None) -> tuple[Path, Path]:
    """Locate the repo root and read the configured ``output_root`` from the paths YAML.

    Args:
        start: Directory to start `find_root`'s upward walk from (default: ``Path.cwd()``).

    Returns:
        ``(root, output_root)`` -- the repo root and the ``output_root`` filesystem path.
    """
    root = find_root(start)
    output_root = Path(yaml.safe_load((root / _PATHS_YAML).read_text())["output_root"])
    return root, output_root


def set_plot_style(dpi: int = 110, *, grid_alpha: float = 0.3, hide_spines: bool = False) -> None:
    """Apply the shared notebook matplotlib defaults (light grid, fixed DPI).

    Args:
        dpi: Figure DPI.
        grid_alpha: Opacity of the background grid.
        hide_spines: Drop the top/right axes spines (the ``dpi=115`` distillation-notebook look).

    matplotlib is imported here (lazily) rather than at module scope so that importing
    ``analysis.setup`` does not pull matplotlib in.
    """
    import matplotlib.pyplot as plt

    # Set spines explicitly (not conditionally) so the style is deterministic regardless of any
    # prior call -- e.g. a later default call restores the spines a hide_spines=True call dropped.
    plt.rcParams.update({
        "figure.dpi": dpi,
        "axes.grid": True,
        "grid.alpha": grid_alpha,
        "axes.spines.top": not hide_spines,
        "axes.spines.right": not hide_spines,
    })
