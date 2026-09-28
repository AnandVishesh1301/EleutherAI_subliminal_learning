"""Plot receiver steerability against transfer on the logit scale."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "notebooks/figures/steerability/steerability_ungated_receivers.csv"
OUT = ROOT / "notebooks/figures/steerability"

STYLES = {
    "orig": ("Original (same model)", "o", "tab:blue"),
    "ds1": ("Data seed 1", "s", "tab:orange"),
    "ds2": ("Data seed 2", "D", "tab:red"),
    "ds3": ("Data seed 3", "+", "tab:brown"),
    "fs1": ("Full seed 1", "^", "tab:green"),
    "fs2": ("Full seed 2", "v", "olive"),
    "fs3": ("Full seed 3", "X", "c"),
}


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(CSV)
    assert len(frame) == 70
    assert frame.receiver.nunique() == 7
    assert set(frame.receiver) == set(STYLES)
    assert frame.teacher.nunique() == 10
    assert frame.groupby("receiver").size().eq(10).all()
    assert frame.groupby("receiver").teacher.nunique().eq(10).all()
    assert frame.n_cells.eq(56).all()
    values = frame[["ungated_delta", "transfer_mean"]].to_numpy()
    assert np.isfinite(values).all()
    assert ((frame.transfer_mean > 0) & (frame.transfer_mean < 1)).all()
    return frame


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(pearsonr(x, y).statistic)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def correlations(frame: pd.DataFrame) -> dict:
    x = frame.ungated_delta.to_numpy(float)
    raw_y = frame.transfer_mean.to_numpy(float)
    y = logit(np.clip(raw_y, 1e-6, 1 - 1e-6))
    centered = frame.assign(logit_transfer=y)
    x_centered = x - centered.groupby("receiver").ungated_delta.transform("mean")
    y_centered = y - centered.groupby("receiver").logit_transfer.transform("mean")
    per_receiver = {}
    for receiver, group in centered.groupby("receiver", sort=True):
        gx = group.ungated_delta.to_numpy(float)
        gy = group.logit_transfer.to_numpy(float)
        per_receiver[receiver] = {
            "n": len(group),
            "pearson": _pearson(gx, gy),
            "spearman": _spearman(gx, gy),
        }
    return {
        "pooledPearson_raw_y": _pearson(x, raw_y),
        "pooledPearson_logit_y": _pearson(x, y),
        "pooledSpearman": _spearman(x, raw_y),
        "receiver_centered_Pearson": _pearson(x_centered, y_centered),
        "per_receiver": per_receiver,
    }


def _scatter(ax, x, y, **kwargs):
    artist = ax.scatter(x, y, **kwargs)
    np.testing.assert_allclose(artist.get_offsets().data, np.column_stack([x, y]))
    assert len(artist.get_offsets()) == len(x)
    return artist


def render(frame: pd.DataFrame, stats: dict) -> plt.Figure:
    x = frame.ungated_delta.to_numpy(float)
    y = logit(np.clip(frame.transfer_mean.to_numpy(float), 1e-6, 1 - 1e-6))
    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    for receiver, (label, marker, color) in STYLES.items():
        mask = frame.receiver.eq(receiver).to_numpy()
        _scatter(
            ax,
            x[mask],
            y[mask],
            label=label,
            marker=marker,
            color=color,
            s=58,
            linewidths=1.1 if marker in {"+", "X"} else 0.7,
            edgecolors="white" if marker not in {"+", "X"} else color,
            alpha=0.9,
            zorder=3,
        )
    ax.set_xlabel("Increase in p(dog) under steering")
    ax.set_ylabel("Mean student p(dog) (logit scale)")
    ax.set_title(
        "Steering response and transfer across receivers\n"
        "Fixed sender: original Pythia-160M dog teachers"
    )
    probs = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75])
    ax.set_yticks(logit(probs), labels=[f"{p:g}" for p in probs])
    ax.grid(alpha=0.18)
    ax.margins(x=0.04, y=0.10)
    ax.text(
        0.025,
        0.965,
        f"Pooled Pearson r = {stats['pooledPearson_logit_y']:.3f}\n"
        f"Within-receiver r = {stats['receiver_centered_Pearson']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 3},
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.57),
        frameon=False,
        fontsize=9,
        title="Receiver",
    )
    fig.text(
        0.77,
        0.055,
        "Data seeds share init/different data order; full seeds differ in both.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.3",
    )
    fig.subplots_adjust(left=0.11, right=0.77, bottom=0.15, top=0.87)
    return fig


def main() -> None:
    frame = load_data()
    stats = correlations(frame)
    fig = render(frame, stats)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "steerability_vs_transfer_ungated_receivers_logit"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    payload = {
        "sourceCSV": str(CSV.relative_to(ROOT)),
        "sourceCSVhash": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "n": len(frame),
        "n_receivers": frame.receiver.nunique(),
        "n_shared_teachers": frame.teacher.nunique(),
        "n_teachers": frame.teacher.nunique(),
        "n_recipes": int(frame.n_cells.iloc[0]),
        **stats,
    }
    (stem.with_suffix(".json")).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
