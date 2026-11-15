"""Plot checkpoint number-profile ablations used in Figure 2.

The first two panels compare teacher profiles at a fixed mature recipient with
the corresponding real-data same-checkpoint diagonal.  Panel C/D are deliberately
kept separate: their six conditions are a one-recipe, three-pool-draw study and
must not be interpreted as five-recipe checkpoint curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEPS = (75_000, 85_000, 95_000, 105_000, 115_000, 125_000)
RECIPES = (
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r8a16",
    "ep20_lr1e-4_r16a32",
)
COLORS = {"iid": "#1f77b4", "perm": "#d95f02", "fixed": "#238b45", "diag": "#6a3d9a"}


def _check_grid(frame: pd.DataFrame, *, name: str, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} has no rows")


def _profile_data(frame: pd.DataFrame, animal: str) -> dict[str, pd.DataFrame]:
    """Return five-recipe means and recipe-level values for panel a/b."""
    _check_grid(frame, name="checkpoint data", required={"animal", "recipe", "step", "recipient", "leaf"})
    f = frame[
        frame.animal.eq(animal)
        & frame.recipe.isin(RECIPES)
    ]
    curves: dict[str, list[float]] = {k: [] for k in ("iid", "perm", "fixed", "diag")}
    dots: dict[str, list[np.ndarray]] = {k: [] for k in curves}
    # ``test`` is the target-animal held-out score for every recipe; the
    # legacy dog/cat columns are populated only on the anchor recipe.
    value_col = "test" if "test" in frame.columns else animal
    for step in STEPS:
        conditions = {
            "iid": f[f.campaign.eq("ladder") & f.pool.eq(0) & f.seed.eq(1) & f.leaf.eq("synthiid") & f.step.eq(step) & f.recipient.eq(125_000)],
            "perm": f[f.campaign.eq("ladder") & f.pool.eq(0) & f.seed.eq(1) & f.leaf.eq("synthiid-perm") & f.step.eq(step) & f.recipient.eq(125_000)],
            "fixed": f[f.campaign.eq("real") & f.pool.eq(0) & f.seed.eq(1) & f.leaf.eq("real") & f.step.eq(step) & f.recipient.eq(125_000)],
            "diag": f[f.campaign.eq("real") & f.pool.eq(0) & f.seed.eq(1) & f.leaf.eq("real") & f.step.eq(step) & f.recipient.eq(step)],
        }
        for key, subset in conditions.items():
            counts = subset.groupby("recipe", dropna=False)[value_col].count()
            if set(counts.index) != set(RECIPES) or not (counts == 1).all():
                raise ValueError(f"{animal} {key} at {step}: expected one row for each of five recipes")
            vals = subset.set_index("recipe").loc[list(RECIPES), value_col].to_numpy(float)
            curves[key].append(float(vals.mean()))
            dots[key].append(vals)
    return {"means": pd.DataFrame(curves, index=STEPS), "dots": dots}


def _anchor_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and aggregate the one-recipe panel-C anchor study."""
    _check_grid(frame, name="anchor data", required={"condition", "value", "pool", "seed"})
    expected = {
        "early teacher (uniform)",
        "early teacher (first-number)",
        "mature teacher (uniform)",
        "mature teacher (IID)",
        "mature teacher (permuted IID)",
        "untuned base (first-number)",
    }
    if set(frame.condition) != expected:
        raise ValueError(f"anchor conditions differ from expected set: {sorted(set(frame.condition))}")
    if frame[["condition", "pool", "seed"]].duplicated().any():
        raise ValueError("anchor data contains duplicate condition/pool/seed cells")
    expected_grid = {(condition, pool, seed) for condition in expected for pool in range(3) for seed in (1, 2, 3)}
    observed_grid = set(map(tuple, frame[["condition", "pool", "seed"]].itertuples(index=False, name=None)))
    if observed_grid != expected_grid:
        raise ValueError("anchor data must contain exactly 3 pool draws x seeds 1/2/3 per condition")
    if not frame.value.between(0, 1).all():
        raise ValueError("anchor probabilities must lie in [0, 1]")
    # Pool means are the independent units for error bars.
    out = frame.groupby(["condition", "pool"], as_index=False).value.mean()
    return out



def _checkpoint_anchor_data(frame: pd.DataFrame, *, recipient: int) -> pd.DataFrame:
    """Extract the four canonical checkpoint conditions for one recipient."""
    _check_grid(frame, name="checkpoint resampling data", required={"animal", "campaign", "recipe", "step", "recipient", "leaf", "pool", "seed", "test"})
    f = frame[frame.animal.eq("dog") & frame.recipe.eq("ep2_lr5e-4_r32a64")]
    specs = {
        "early teacher (uniform)": (75_000, "uniform"),
        "mature teacher (uniform)": (125_000, "uniform"),
        "mature teacher (IID)": (125_000, "synthiid"),
        "mature teacher (permuted IID)": (125_000, "synthiid-perm"),
    }
    rows = []
    for condition, (step, leaf) in specs.items():
        subset = f[f.campaign.eq("anchor") & f.step.eq(step) & f.recipient.eq(recipient) & f.leaf.eq(leaf)]
        if len(subset) != 9 or set(subset.pool) != {0, 1, 2} or set(subset.seed) != {1, 2, 3}:
            raise ValueError(f"{condition} at recipient {recipient}: expected exactly 3 pool draws x seeds 1/2/3")
        if subset[["pool", "seed"]].duplicated().any():
            raise ValueError(f"{condition} at recipient {recipient}: duplicate pool/seed cells")
        rows.extend({"condition": condition, "pool": int(r.pool), "seed": int(r.seed), "value": float(r.test)} for r in subset.itertuples())
    return pd.DataFrame(rows)


def _first_number_anchor_data(frame: pd.DataFrame, *, recipient: int, arm: str = "firstonly_early", condition: str = "early teacher (first-number)") -> pd.DataFrame:
    """Extract one replicated first-number replacement condition for one recipient."""
    _check_grid(frame, name="first-number data", required={"recipient_step", "arm", "pool_seed", "student_seed", "p_test"})
    subset = frame[frame.recipient_step.eq(recipient) & frame.arm.eq(arm)]
    if subset[["pool_seed", "student_seed"]].duplicated().any():
        raise ValueError(f"{condition} at recipient {recipient}: duplicate pool/seed cells")
    if len(subset) != 9 or set(subset.pool_seed) != {0, 1, 2} or set(subset.student_seed) != {1, 2, 3}:
        raise ValueError(f"{condition} at recipient {recipient}: expected exactly 3 pool draws x seeds 1/2/3")
    return pd.DataFrame({"condition": condition, "pool": subset.pool_seed.astype(int), "seed": subset.student_seed.astype(int), "value": subset.p_test.astype(float)})


def _panel_data(checkpoint: pd.DataFrame, anchor: pd.DataFrame, first_number: pd.DataFrame, *, recipient: int) -> pd.DataFrame:
    """Build a panel from verified checkpoint and first-number records."""
    if recipient == 125_000:
        if "recipient_step" not in anchor.columns or set(anchor.recipient_step) != {125_000}:
            raise ValueError("anchor data has wrong recipient")
        pieces = [anchor.copy()]
    else:
        pieces = [_checkpoint_anchor_data(checkpoint, recipient=recipient)]
    pieces.append(_first_number_anchor_data(first_number, recipient=recipient, arm="firstonly_untuned", condition="untuned base (first-number)"))
    if recipient != 125_000:
        pieces.append(_first_number_anchor_data(first_number, recipient=recipient, arm="firstonly_early", condition="early teacher (first-number)"))
    return pd.concat(pieces, ignore_index=True)


def plot(checkpoint_csv: Path, anchor_csv: Path, first_number_csv: Path, output: Path) -> None:
    checkpoint = pd.read_csv(checkpoint_csv)
    anchor = pd.read_csv(anchor_csv)
    first_number = pd.read_csv(first_number_csv)
    panel_data = {recipient: _panel_data(checkpoint, anchor, first_number, recipient=recipient) for recipient in (75_000, 125_000)}
    data = {animal: _profile_data(checkpoint, animal) for animal in ("dog", "cat")}

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})
    fig = plt.figure(figsize=(14.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.95), height_ratios=(1, 1))
    for ax, animal, title in zip(
        (fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])),
        ("dog", "cat"),
        ("a  Dog preference", "b  Cat preference"),
    ):
        d = data[animal]
        x = np.arange(len(STEPS))
        labels = {"iid": "IID profile → 125k", "perm": "permuted IID → 125k", "fixed": "teacher numbers → 125k", "diag": "same-checkpoint (recipient varies)"}
        for key in ("iid", "perm", "fixed", "diag"):
            style = ":" if key == "diag" else "-"
            ax.plot(x, d["means"][key], color=COLORS[key], lw=2, ls=style, marker="o", ms=4, label=labels[key])
            for xi, vals in zip(x, d["dots"][key]):
                ax.scatter(np.full(len(vals), xi), vals, color=COLORS[key], s=12, alpha=0.48, zorder=2)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, [f"{s//1000}k" for s in STEPS])
        ax.set_xlabel("teacher/profile checkpoint")
        ax.set_ylabel(f"p({animal}), held-out test")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7.2, loc="upper left", frameon=False)

    order = ["early teacher (uniform)", "early teacher (first-number)", "untuned base (first-number)", "mature teacher (uniform)", "mature teacher (IID)", "mature teacher (permuted IID)"]
    labels = ["Early teacher (75k)", "Replace first (early teacher rows)", "Replace first (untuned base rows)", "Mature teacher (125k)", "Mature IID", "Permuted mature IID"]
    for row, recipient in enumerate((75_000, 125_000)):
        ax = fig.add_subplot(gs[row, 2])
        current = _anchor_data(panel_data[recipient])
        for y, condition in zip(range(len(order)), order):
            vals = current.loc[current.condition.eq(condition), "value"].to_numpy(float)
            pool_means = vals.reshape(3)
            ax.errorbar(pool_means.mean(), y, xerr=pool_means.std(ddof=1), fmt="o", color="#333333", capsize=3, ms=5)
            ax.scatter(pool_means, np.full(3, y), color="#333333", s=14, alpha=0.55, zorder=3)
        ax.set_yticks(range(len(order)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("p(dog), held-out test")
        ax.set_title(f"{'c' if row == 0 else 'd'}  Fixed {recipient // 1000}k recipient", loc="left")
        ax.grid(axis="x", alpha=0.2)


    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-csv", type=Path, default=Path("docs/results/data/checkpoint_resampling_verified_20260921.csv"))
    parser.add_argument("--anchor-csv", type=Path, default=Path("docs/results/data/resampling_anchor_complete_20260922_runs.csv"))
    parser.add_argument("--first-number-csv", type=Path, default=Path("docs/results/data/first_number_carriers_20260922_runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/figures/checkpoint_number_profiles_20260921"))
    args = parser.parse_args()
    plot(args.checkpoint_csv, args.anchor_csv, args.first_number_csv, args.output)


if __name__ == "__main__":
    main()
