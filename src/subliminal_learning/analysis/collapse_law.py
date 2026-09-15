"""Compatibility shim + the analysis-layer loaders of the collapse-boundary law.

The estimators moved to `subliminal_learning.stats.collapse_law` (see that package's docstring
for the layering rule); this module mirrors that module automatically through a module-level
``__getattr__``, so every estimator -- including ones added later -- keeps resolving for notebooks
that alias this module as ``cl`` and scripts importing ``analysis.collapse_law``. New code should
import ``subliminal_learning.stats.collapse_law`` directly.

What genuinely still lives here is the two functions that read disk through the analysis layer,
and so cannot sit in the (loader-free) stats package:

- `load_law_runs` -- campaign discovery over ``students/`` plus the batched cell-summary fetch.
- `adapter_coherence` -- SVD statistics of a PEFT checkpoint's LoRA modules.
"""

from __future__ import annotations

import inspect
import types
import warnings
from pathlib import Path

import numpy as np  # noqa: F401  -- kept for callers doing `cl.np`-style scratch work in notebooks
import pandas as pd

from subliminal_learning.analysis.adapter_decomp import low_rank_singular_values
from subliminal_learning.analysis.frames import (
    PARAM_TAG_RE,
    load_cell_summaries,
    parse_param_tag,  # noqa: F401  -- re-exported: historically reachable as `collapse_law.parse_param_tag`
    read_cell_set,  # noqa: F401  -- re-exported: callers use `collapse_law.read_cell_set`
    summary_set_block,
)
from subliminal_learning.stats.collapse_law import (  # noqa: F401  -- explicit re-exports keep static analysis happy; `__getattr__` below mirrors the rest
    B_LAW,
    C_LAW,
    DEAD_SEED,
    KSTAR_CALIBRATED,
    KSTAR_DS1_DESIGN,
    KSTAR_POOLED,
    LAW_LINEAGES,
    LINEAGE_ORDER,
    RIDGE_PEN,
    SKLEARN_C,
    K_of,
    LawFit,
    TrajectoryLawFit,
    _design,   # the stats tests fit against this design matrix
    add_k,
    cell_table,
    cluster_bootstrap_ratios,
    dose_law_fit,
    estimate_kstar,
    fit_collapse_law,
    fit_collapse_law_trajectory,
    k_from_tag,
    kstar_logistic,
    kstar_minerr,
    kstar_spread,
    label_collapse,
    law_fits_row,
    law_misses,
    theta_dose,
    theta_dose_row,
    theta_variance_split,
)
from subliminal_learning.stats import collapse_law as _stats
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

# Names this shim defines itself; `__all__` is their union with the mirrored stats API, so a new
# estimator in `stats.collapse_law` shows up here without anyone editing a list.
_OWN_PUBLIC = [
    "PARAM_TAG_RE", "adapter_coherence", "load_law_runs", "parse_param_tag", "read_cell_set",
]


def _stats_public() -> list[str]:
    """Public API of the stats module: its ``__all__`` if it declares one, else its own defs."""
    declared = getattr(_stats, "__all__", None)
    if declared is not None:
        return list(declared)
    names = []
    for name, obj in vars(_stats).items():
        if name.startswith("_") or isinstance(obj, types.ModuleType):
            continue   # re-exported third-party modules (np, pd, ...) are not API
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if getattr(obj, "__module__", None) == _stats.__name__:
                names.append(name)
        elif name.isupper():
            names.append(name)
    return names


__all__ = sorted(set(_OWN_PUBLIC) | set(_stats_public()))


def __getattr__(name: str):
    """Forward anything not defined here to `subliminal_learning.stats.collapse_law` (PEP 562)."""
    if not name.startswith("_") or name == "_design":
        try:
            return getattr(_stats, name)
        except AttributeError:
            pass
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r} "
        f"(not defined here and not in {_stats.__name__!r})"
    )


# The canonical cell-name parser (shared with `frames`). Trailing ``_{optim_slug}`` (non-default
# optimizers) is accepted but not consumed here; existing adamw cell dirs match exactly as before.
_TAG_RE = PARAM_TAG_RE


def load_law_runs(
    output_root: Path,
    animal: str = "dog",
    numbers_base: str = "pythia-160m",
    teacher_seed: int = 1,
    dataset: str = "ex10-20_prefix_n100k",
    epochs: int = 20,
    lineages: list[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-cell and per-seed results for the ``law-*`` campaign.

    Reads ``per_seed_p_target`` off each cell's ``preference_summary.json`` (the ``test``/``prob``
    entry). That field is written from the same per-seed evaluations as the individual
    ``ftseed*/preference_eval.json`` files -- verified identical across the whole campaign -- and
    costs one file read per cell instead of one per seed.

    Args:
        output_root: The ``outputs/`` root holding ``students/``.
        animal: Target animal subtree.
        numbers_base: Base model the teacher's numbers were generated from (the ``teacher-`` level).
        teacher_seed: Teacher fine-tuning seed embedded in the lineage directory name.
        dataset: Number-pool tag.
        epochs: Student epoch count to keep (the law is fit at ep20 only).
        lineages: Lineage specs; defaults to `LAW_LINEAGES`.

    Returns:
        ``(cells, runs)`` -- one row per cell and one row per (cell, seed) respectively. Both carry
        ``family``, ``base_label``, ``lineage``, ``student_tag``, ``lr``, ``rank``, ``alpha``.
    """
    lineages = LAW_LINEAGES if lineages is None else lineages
    output_root = Path(output_root)
    students_root = output_root / "students"
    cell_rows: list[dict] = []
    run_rows: list[dict] = []
    # Discovery first, then ONE batched fetch of every candidate cell's summary doc: the fan-out
    # is thousands of tiny reads on a shared FS (see `analysis.record_cache`).
    candidates: list[tuple[dict, str]] = []   # (row meta, cell path relative to output_root)

    for family, tag, base in lineages:
        lineage_dir = (
            students_root / animal / Path(*model_dir_parts_from_name(base))
            / f"teacher-{numbers_base}" / f"{tag}_tseed{teacher_seed}" / dataset
        )
        if not lineage_dir.exists():
            continue
        # Strict PREFIX strip (not substring replace): "data-seed1" etc. for the
        # numbers_base family. Family membership compares revision-stripped roots, so
        # the ckpt lineages' revision-pinned forms (numbers_base "pythia-160m@step1000",
        # base "pythia-160m@step143000") keep their full name silently, as before. A
        # foreign base (e.g. a non-160m size) also keeps its full name but loudly —
        # 160m-keyed joins would silently drop it.
        # A fully re-randomized sibling ("pythia-160m-seed2") labels "full-seed2",
        # matching `spec_label`: a bare "seed2" would read as a data-/weight-seed arm.
        base_label = base.removeprefix(f"{numbers_base}-")
        if base_label != base and base_label.startswith("seed"):
            base_label = f"full-{base_label}"
        nb_root, base_root = numbers_base.split("@")[0], base.split("@")[0]
        if not (base_root == nb_root or base_root.startswith(f"{nb_root}-")):
            warnings.warn(
                f"load_law_runs: student base {base!r} is not in the {numbers_base!r} "
                f"family; keeping the full name as base_label ({base_label!r}). "
                "Downstream label-keyed joins will not match these rows.",
                stacklevel=2,
            )
        for cell_dir in sorted(lineage_dir.iterdir()):
            m = _TAG_RE.fullmatch(cell_dir.name)
            if not m or int(m["ep"]) != epochs:
                continue
            meta = {
                "family": family,
                "teacher_tag": tag,
                "base": base,
                "base_label": base_label,
                "lineage": f"{family}:{base_label}",
                "student_tag": cell_dir.name,
                "lr": float(m["lr"]),
                "rank": int(m["r"]),
                "alpha": int(m["a"]),
                "scale": int(m["a"]) / int(m["r"]),
            }
            candidates.append((meta, str(cell_dir.relative_to(output_root))))

    docs = load_cell_summaries(output_root, [rel for _, rel in candidates])
    for meta, rel in candidates:
        test = summary_set_block(docs.get(rel), "test")
        if test is None:              # not evaluated yet (or unreadable summary) -- skip, as before
            continue
        per_seed = test.get("per_seed_p_target") or {}
        cell_rows.append(
            {**meta, "mean_test": test.get("p_target_mean"), "n_seeds": len(per_seed)}
        )
        run_rows.extend({**meta, "seed": int(s), "test": float(v)} for s, v in per_seed.items())

    return pd.DataFrame(cell_rows), pd.DataFrame(run_rows)


def adapter_coherence(adapter_file: Path) -> pd.DataFrame:
    """SVD statistics of each LoRA module's update ``delta_W = B @ A`` in a trained adapter.

    Discriminates between the two limiting models of how an adapter's ``r`` rank-1 channels
    ``b_k a_k^T`` combine — the assumption that decides whether the ``1/r`` in the forward scale
    ``s = alpha/r`` cancels out of the applied perturbation (and hence whether *absolute* alpha
    belongs in the collapse law):

    - **iid channels** (the rsLoRA model, arXiv:2312.03732): the sum grows like ``sqrt(r)``,
      predicting ``stable_rank ~ r`` and rank exponent ``c = -0.25``.
    - **coherent channels** (Hu et al. §7, arXiv:2106.09685): the sum grows like ``r``,
      predicting ``stable_rank = O(1)`` independent of ``r`` and ``c = 0``.

    Reads the ``adapter_model.safetensors`` directly — no model load, no GPU.

    Args:
        adapter_file: Path to a PEFT ``adapter_model.safetensors`` checkpoint.

    Returns:
        One row per LoRA module pair with:

        - ``rank``: nominal LoRA rank ``r`` (columns of ``B``).
        - ``stable_rank``: ``||BA||_F^2 / sigma_1^2`` — effective number of directions the update
          energy occupies. The discriminating statistic (``~r`` iid vs ``O(1)`` coherent).
        - ``top1_energy``: ``sigma_1^2 / ||BA||_F^2`` — fraction of the update carried by the
          single largest singular direction.
        - ``b_col_cos``: mean pairwise ``|cos|`` between columns of ``B`` (the ``r`` output
          directions); ``NaN`` at rank 1. Random directions in ``d`` dims give
          ``~sqrt(2/(pi*d))`` (about 0.03 at ``d = 768``).
    """
    from safetensors.numpy import load_file  # training-stack dep; keep analysis import cheap
    from threadpoolctl import threadpool_limits  # ships with scikit-learn

    tensors = load_file(str(adapter_file))
    rows = []
    # Singular values come from the r x r QR core (adapter_decomp.low_rank_singular_values),
    # never a full SVD of the d_out x d_in product (the latter ran >900s/adapter on a
    # sweep-loaded node, 2026-07-27). Still pin BLAS to one thread: the work is tiny.
    with threadpool_limits(limits=1):
        for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
            A = tensors[key_a].astype(np.float64)                              # (r, d_in)
            B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)  # (d_out, r)
            r = B.shape[1]
            sv = low_rank_singular_values(B, A)
            fro2 = float((sv**2).sum())
            cols = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
            cos = np.abs(cols.T @ cols)[np.triu_indices(r, 1)]
            rows.append({
                "module": key_a[: -len(".lora_A.weight")],
                "rank": r,
                "stable_rank": fro2 / float(sv[0] ** 2),
                "top1_energy": float(sv[0] ** 2) / fro2,
                "b_col_cos": float(cos.mean()) if r > 1 else float("nan"),
            })
    return pd.DataFrame(rows)
