"""CLI: visualize the preference verification sweep as hyperparameter heatmaps.

For each selected (animal, base_model), draw one figure whose rows are the evaluation
prompt sets (favorite / dislike / validation) and whose columns are the LoRA settings.
Each subplot is an ``n_epochs`` x ``learning_rate`` heatmap of the chosen metric, so a
single figure answers "which hyperparameters best install the trait for this animal on
this model?". The best cell per set is boxed, and a console leaderboard prints the
argmax config per (animal, model, set).

Metrics (``--metric``):
  p_target       mean next-token p(target), seed-averaged (higher = stronger trait)
  delta_vs_base  p_target minus the untuned base control (higher = larger causal effect)
  rank1_frac     fraction of prompts where target is the top-1 next token (higher better)
  rank_mean      mean next-token rank of the target (LOWER better; colormap reversed)

Reads whatever ``preference_summary.json`` files exist, so it runs against a partially
completed sweep. Data source + aggregation levels: see `preference_results_io`.

Examples:
  uv run python -m scripts.visualize_preference_sweep
  uv run python -m scripts.visualize_preference_sweep --animals dog,cat --metric delta_vs_base
  uv run python -m scripts.visualize_preference_sweep --models pythia-160m-data-seed1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml
from loguru import logger

from subliminal_learning.evaluation.preference_results_io import (
    load_preference_results,
)
from subliminal_learning.utils.logging_setup import configure_loguru

# metric -> (axis label, higher-is-better, colormap). rank_mean is the only
# lower-is-better metric, so it gets a reversed map and its "best" is the min.
_METRICS = {
    "p_target": ("p(target), seed-avg", True, "viridis"),
    "delta_vs_base": ("p(target) − base", True, "viridis"),
    "rank1_frac": ("frac. rank-1", True, "viridis"),
    "rank_mean": ("mean rank (lower=better)", False, "viridis_r"),
}


def _annotation(metric: str, value: float) -> str:
    """Cell label: ranks as integers-ish, probabilities/fractions at 3 decimals."""
    if np.isnan(value):
        return ""
    if metric == "rank_mean":
        return f"{value:.1f}"
    return f"{value:.3f}"


def _draw_subplot(
    ax, pivot: pd.DataFrame, metric: str, cmap: str, vmin: float, vmax: float
) -> None:
    """Render one epochs×lr heatmap into `ax` with per-cell value annotations."""
    data = pivot.to_numpy(dtype=float)
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if np.isnan(v):
                continue
            # Contrast: dark text on the bright half of the map, light on the dark half.
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "black" if frac > 0.55 else "white"
            ax.text(
                c, r, _annotation(metric, v), ha="center", va="center",
                fontsize=7, color=color,
            )


def _box_cell(ax, pivot: pd.DataFrame, n_epochs: int, learning_rate: float) -> None:
    """Draw a gold rectangle around the (n_epochs, learning_rate) cell of this subplot."""
    if n_epochs not in pivot.index or learning_rate not in pivot.columns:
        return
    r = list(pivot.index).index(n_epochs)
    c = list(pivot.columns).index(learning_rate)
    ax.add_patch(
        plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="gold", linewidth=3,
        )
    )


def _best_config(group: pd.DataFrame, metric: str) -> pd.Series:
    """Best row in `group` by `metric`, ties broken toward CHEAPER training.

    Metrics saturate (many configs hit p(target)≈1.000), so a raw argmax just returns
    the most aggressive corner. Instead: round to the displayed precision, take the
    optimal rounded value, and among those ties pick the smallest
    (n_epochs, learning_rate, lora_r) -- the cheapest HP that fully installs the trait.
    """
    _, higher_better, _ = _METRICS[metric]
    ndigits = 1 if metric == "rank_mean" else 3
    rounded = group[metric].round(ndigits)
    opt = rounded.max() if higher_better else rounded.min()
    tied = group[rounded == opt]
    return tied.sort_values(["n_epochs", "learning_rate", "lora_r"]).iloc[0]


def _figure_for(
    df: pd.DataFrame, animal: str, model: str, metric: str, out_dir: Path
) -> Path | None:
    """One figure for (animal, model): rows=sets, cols=LoRA, cell=epochs×lr heatmap."""
    sub = df[(df.animal == animal) & (df.model == model)]
    if sub.empty:
        return None
    sets = list(dict.fromkeys(sub["set"]))  # favorite/dislike/validation, order preserved
    loras = sorted(sub["lora_r"].unique())
    label, higher_better, cmap = _METRICS[metric]

    # Shared color scale across the whole figure so subplots are comparable.
    vals = sub[metric].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1e-6, vmax + 1e-6

    n_rows, n_cols = len(sets), len(loras)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols + 1.5, 2.4 * n_rows + 0.8),
        squeeze=False,
    )
    for ri, s in enumerate(sets):
        set_rows = sub[sub["set"] == s]
        clean = set_rows.dropna(subset=[metric])
        best = _best_config(clean, metric) if not clean.empty else None
        for ci, r in enumerate(loras):
            ax = axes[ri][ci]
            cell = set_rows[set_rows["lora_r"] == r]
            if cell.empty:
                ax.set_visible(False)
                continue
            pivot = cell.pivot_table(
                index="n_epochs", columns="learning_rate", values=metric
            ).sort_index().sort_index(axis=1)
            _draw_subplot(ax, pivot, metric, cmap, vmin, vmax)
            if best is not None and best["lora_r"] == r:
                _box_cell(ax, pivot, best["n_epochs"], best["learning_rate"])
            if ri == 0:
                a = cell["lora_alpha"].iloc[0]
                ax.set_title(f"LoRA r{r}/α{a}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{s}\nepochs", fontsize=9)
            if ri == n_rows - 1:
                ax.set_xlabel("learning rate", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, label=label)
    fig.suptitle(
        f"{animal} · {model} · {label}  (gold = cheapest best-scoring HP per set)",
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{animal}__{model}__{metric}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _leaderboard(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Cheapest best-scoring config per (animal, model, set) by `metric`.

    Uses `_best_config` (efficient tie-break), so a saturated metric reports the
    minimal HP that fully installs the trait rather than the most aggressive corner.
    """
    cols = [
        "animal", "model", "set", "param_tag",
        "n_epochs", "learning_rate", "lora_r",
        "p_target", "delta_vs_base", "rank_mean", "rank1_frac", "n_seeds",
    ]
    winners = [
        _best_config(g.dropna(subset=[metric]), metric)[cols]
        for _, g in df.groupby(["animal", "model", "set"])
        if not g.dropna(subset=[metric]).empty
    ]
    return (
        pd.DataFrame(winners)
        .sort_values(["animal", "model", "set"])
        .reset_index(drop=True)
    )


def main(
    metric: str = typer.Option(
        "p_target",
        help="Cell metric: p_target | delta_vs_base | rank1_frac | rank_mean.",
    ),
    animals: str = typer.Option(
        None, help="Comma-separated animals to include (default: all found)."
    ),
    models: str = typer.Option(
        None, help="Comma-separated base models to include (default: all found)."
    ),
    out_dir: Path = typer.Option(
        None, help="Where to write PNGs (default: {output_root}/analysis/preference_heatmaps)."
    ),
    csv_out: Path = typer.Option(
        None, help="Optional path to write the best-config leaderboard as CSV."
    ),
    output_root: Path = typer.Option(
        None, help="Override paths.output_root (where teachers/ lives)."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Render per-(animal, model) HP heatmaps and print a best-config leaderboard."""
    configure_loguru()
    if metric not in _METRICS:
        raise typer.BadParameter(f"metric must be one of {list(_METRICS)}")
    root = output_root or Path(yaml.safe_load(paths_config.read_text())["output_root"])
    animal_list = [a.strip() for a in animals.split(",")] if animals else None
    model_list = [m.strip() for m in models.split(",")] if models else None

    df = load_preference_results(root, animals=animal_list, models=model_list)
    if df.empty:
        logger.warning("No preference_summary.json found under {}/teachers", root)
        raise typer.Exit(1)
    logger.info(
        "Loaded {} config-set rows: {} animals x {} models x {} param_tags",
        len(df), df.animal.nunique(), df.model.nunique(), df.param_tag.nunique(),
    )

    out_dir = out_dir or (root / "analysis" / "preference_heatmaps")
    pairs = df[["animal", "model"]].drop_duplicates().itertuples(index=False)
    n_written = 0
    for animal, model in pairs:
        path = _figure_for(df, animal, model, metric, out_dir)
        if path:
            n_written += 1
            logger.info("Wrote {}", path)
    logger.info("Wrote {} figure(s) to {}", n_written, out_dir)

    board = _leaderboard(df, metric)
    logger.info(
        "Best config per (animal, model, set) by {} (ties -> cheapest HP):\n{}",
        metric, board.to_string(index=False),
    )
    if csv_out:
        board.to_csv(csv_out, index=False)
        logger.info("Wrote leaderboard CSV -> {}", csv_out)


if __name__ == "__main__":
    typer.run(main)
