"""W&B lifecycle helpers for multi-cell training loops.

Hugging Face ``WandbCallback`` only calls ``wandb.init()`` when ``wandb.run is None``
(see ``transformers.integrations.integration_utils.WandbCallback.setup``). In a
single-process sweep, if the prior cell never called ``wandb.finish()``, the next cell
reuses the same run and appends metrics -- one loss curve for hundreds of jobs. W&B
docs recommend ``run.finish()`` after each run in a loop
(https://docs.wandb.ai/models/runs/initialize-run, ``reinit="finish_previous"`` applies
only when ``init`` is actually invoked).

``finish_wandb_run()`` closes any active run before/after each cell. ``WandbCellFilter``
lets sweep YAMLs train the full grid locally while logging only canonical HP cells to
the dashboard (``report_to=[]`` on non-matching cells -- no run created).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger


def wandb_axis_match(wb: dict | bool | None, axes: Mapping[str, Any]) -> bool:
    """Whether a sweep cell should create a W&B run under an optional ``wandb:`` block.

    The reusable generalization of the GKD sweep's former ``_wandb_match`` (and the same
    AND-match idea the SDF `WandbCellFilter` encodes), so both sweeps share one selective-
    logging rule.

    Args:
        wb: The sweep YAML's ``wandb:`` block:
            - ``False``: log nothing.
            - ``None``: log every cell (callers apply the conservative log-few default
              *before* calling this, so an omitted block is never passed as ``None``).
            - ``dict``: AND-match -- for each axis key present in ``wb``, this cell's value
              (in `axes`) must be in the listed set; axes absent from ``wb`` are
              unconstrained.
        axes: This cell's axis values keyed by the same names used in the YAML block
            (e.g. ``{"beta": 1.0, "student_modes": "full", "learning_rate": 2e-5}``).

    Returns:
        Whether this cell should create a W&B run.
    """
    if wb is False:
        return False
    if wb is None:
        return True
    for key, value in axes.items():
        if key not in wb:
            continue
        allowed = wb[key]
        if isinstance(value, bool):
            allowed_set: set[Any] = {bool(x) for x in allowed}
        elif isinstance(value, (int, float)):
            allowed_set = {float(x) for x in allowed}
            value = float(value)
        else:
            allowed_set = set(allowed)
        if value not in allowed_set:
            return False
    return True


@dataclass(frozen=True)
class WandbCellFilter:
    """Axes that must match for a sweep cell to create a W&B run.

    Every listed axis is AND-ed: the cell's value must be in the filter's set. For LoRA
    sweeps, ``lora_r`` must be set; for full-FT sweeps, leave ``lora_r`` as ``None`` (the
    key is ignored).

    Attributes:
        n_epochs: Epoch counts to log (e.g. ``(3,)`` for Apollo's optimal epoch).
        ft_seeds: Fine-tuning seeds to log (e.g. ``(0, 1)`` for replicate visibility).
        learning_rate: Learning rates to log.
        lora_r: LoRA ranks to log, or ``None`` when the sweep has no LoRA axis.
    """

    n_epochs: frozenset[int]
    ft_seeds: frozenset[int]
    learning_rate: frozenset[float]
    lora_r: frozenset[int] | None = None

    def matches(
        self,
        *,
        n_epochs: int,
        learning_rate: float,
        ft_seed: int,
        lora_r: int | None,
    ) -> bool:
        """Return whether this sweep cell should log to W&B."""
        if n_epochs not in self.n_epochs:
            return False
        if ft_seed not in self.ft_seeds:
            return False
        if learning_rate not in self.learning_rate:
            return False
        if self.lora_r is not None:
            if lora_r is None or lora_r not in self.lora_r:
                return False
        return True


def load_wandb_cell_filter(doc: dict[str, Any], *, is_lora_sweep: bool) -> WandbCellFilter:
    """Parse the sweep YAML's optional ``wandb:`` block into a ``WandbCellFilter``.

    Args:
        doc: Parsed sweep YAML (must contain a ``wandb:`` mapping).
        is_lora_sweep: Whether the grid has a ``lora:`` axis (controls ``lora_r``).

    Returns:
        Filter for selective per-cell W&B logging.

    Raises:
        ValueError: The ``wandb:`` block is missing required keys or has empty axes.
    """
    wb = doc["wandb"]
    for key in ("n_epochs", "ft_seeds", "learning_rate"):
        if key not in wb:
            raise ValueError(f"sweep wandb block missing required key {key!r}")
        if not wb[key]:
            raise ValueError(f"sweep wandb.{key} must be a non-empty list")
    lora_r = None
    if is_lora_sweep:
        if "lora_r" not in wb or not wb["lora_r"]:
            raise ValueError("LoRA sweep wandb block requires non-empty lora_r list")
        lora_r = frozenset(int(r) for r in wb["lora_r"])
    return WandbCellFilter(
        n_epochs=frozenset(int(e) for e in wb["n_epochs"]),
        ft_seeds=frozenset(int(s) for s in wb["ft_seeds"]),
        learning_rate=frozenset(float(x) for x in wb["learning_rate"]),
        lora_r=lora_r,
    )


def finish_wandb_run() -> None:
    """Finish the active W&B run, if any.

    Safe to call when W&B is disabled, not installed, or no run is active. Failures are
    logged and re-raised so a broken W&B session cannot silently poison the next cell.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception:
        logger.exception("wandb.finish() failed")
        raise


def prepare_wandb_for_cell(
    *,
    enabled: bool,
    wandb_dir: os.PathLike[str] | str,
    job_type: str,
    run_group: str | None,
    project: str | None = None,
    tags: str | None = None,
) -> None:
    """Prepare W&B env for one training cell; always clears any stale active run first.

    Per W&B env-var docs (``WANDB_PROJECT``, ``WANDB_ENTITY``, ``WANDB_JOB_TYPE``,
    ``WANDB_RUN_GROUP``, ``WANDB_TAGS``, ``WANDB_DIR``), entity and default project come
    from ``.env`` via ``load_dotenv()`` before the pipeline runs. Pass ``project`` to
    override ``WANDB_PROJECT`` for a workflow (e.g. GKD vs SDF) without changing ``.env``.
    Always calls ``finish_wandb_run()`` first so HF ``WandbCallback`` cannot reuse a stale
    ``wandb.run`` across cells (see module docstring).

    Args:
        enabled: If False, only ``finish_wandb_run()`` runs (no new env tags).
        wandb_dir: Local W&B cache directory from ``PathsConfig``.
        job_type: ``WANDB_JOB_TYPE`` (e.g. ``sdf_teacher_lora``).
        run_group: ``WANDB_RUN_GROUP`` when logging; ignored when ``enabled`` is False.
        project: Optional ``WANDB_PROJECT`` override when ``enabled`` is True.
        tags: Optional comma-separated ``WANDB_TAGS`` (e.g. ``gkd,cats1day,lora``).
    """
    finish_wandb_run()
    if not enabled:
        return
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    if project is not None:
        os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_JOB_TYPE"] = job_type
    if run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = run_group
    if tags is not None:
        os.environ["WANDB_TAGS"] = tags
    elif "WANDB_TAGS" in os.environ:
        del os.environ["WANDB_TAGS"]


def report_to_for_wandb(enabled: bool) -> list[str]:
    """``report_to`` value for ``SFTConfig`` / ``TrainingArguments``."""
    return ["wandb"] if enabled else []
