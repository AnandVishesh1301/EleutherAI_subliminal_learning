"""Render Figure 1 as boxplots, violins, and rainclouds from saved evaluations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

from subliminal_learning.utils.io import read_json


ANIMALS = ("cat", "dog", "lion", "pig", "rabbit")
EPOCHS = (1, 2, 5, 10)
COLORS = dict(
    zip(
        (*ANIMALS, "control"),
        ("#4387AD", "#E99545", "#65A66D", "#D76B75", "#9C80BE", "#737B83"),
    )
)
ROOT = Path(__file__).resolve().parents[3]
CONTROL_ROOT = "students/controls/base/pythia-160m/teacher-pythia-160m"
TREATMENT_RE = re.compile(
    r"students/(?P<animal>cat|dog|lion|pig|rabbit)/pythia-160m/"
    r"teacher-pythia-160m/ep\d+_lr[^/]+_r\d+a\d+_tseed1/"
    r"(?P<dataset>ex10-20_prefix_n(?:40|100)k)/"
    r"ep(?P<epoch>1|2|5|10)_lr1e-4_r8a32/ftseed1"
)


def preference(path):
    doc = read_json(path)
    if doc["ft_seeds"] != [1]:
        raise ValueError(f"Expected student seed 1: {path}")
    return next(
        float(s["p_target_mean"])
        for s in doc["summaries"]
        if s["prompt_set"] == "test" and s["scoring"] == "prob"
    )


def run_record(output_root, relative, animal, epoch, draw=None):
    """Read the student's own probe and verify the receiver in its manifest."""
    run = output_root / relative
    manifest = read_json(run / "train_manifest.json")
    if manifest["base_model_repo"] != "EleutherAI/pythia-160m" or manifest.get(
        "cross_base", False
    ):
        raise ValueError(f"Unexpected student base: {relative}")
    probe = read_json(run / "eval/pile_probe.json")
    if probe.get("pool_version") != "pp1":
        raise ValueError(f"Unexpected probe pool: {relative}")
    ce_student, ce_base = probe.get("ce_student"), probe.get("ce_base")
    delta = (
        float(ce_student) - float(ce_base)
        if ce_student is not None and ce_base is not None
        else np.nan
    )
    if np.isfinite(delta) and probe.get("delta_ce") is not None:
        if not np.isclose(delta, probe["delta_ce"], atol=1e-10, rtol=0):
            raise ValueError(f"Probe CE arithmetic mismatch: {relative}")
    result = dict(
        run_path=relative,
        animal=animal,
        epoch=epoch,
        draw=draw,
        student_base=manifest["base_model_repo"],
        ce_student=ce_student,
        ce_base=ce_base,
        ce_increase=delta,
        pool_sha=probe["pool_sha"],
    )
    if animal == "control":
        for trait in ANIMALS:
            result[f"p_{trait}"] = preference(
                run.parent / "eval" / trait / "preference_summary.json"
            )
    else:
        result["p"] = preference(run.parent / "preference_summary.json")
    return result


def prepare_data(treatments, controls):
    """Trait increases keep the control reference; both CEs use the student's base."""
    treatments, controls = treatments.copy(), controls.copy()
    for frame in (treatments, controls):
        frame["ce_increase"] = frame.ce_student - frame.ce_base
    for animal in ANIMALS:
        means = controls.groupby("epoch")[f"p_{animal}"].mean()
        selected = treatments.animal.eq(animal)
        treatments.loc[selected, "control_p"] = treatments.loc[selected, "epoch"].map(
            means
        )
    treatments["trait_increase"] = treatments.p - treatments.control_p
    return treatments, controls


def load_data(output_root):
    # Use the existing cache only as an inventory, then read fresh run records.
    inventory = pd.read_csv(
        output_root / "analysis/probe_cache.csv", usecols=["run_path"]
    )
    jobs = []
    for relative in sorted(set(inventory.run_path)):
        match = TREATMENT_RE.fullmatch(relative)
        if not match:
            continue
        animal = match["animal"]
        dataset = "ex10-20_prefix_n40k" if animal == "cat" else "ex10-20_prefix_n100k"
        if match["dataset"] == dataset:
            jobs.append((output_root, relative, animal, int(match["epoch"])))
    if not jobs:
        raise ValueError("No Figure 1 treatment runs in the probe inventory")
    with ThreadPoolExecutor(max_workers=32) as executor:
        treatments = pd.DataFrame(executor.map(lambda args: run_record(*args), jobs))
        control_jobs = [
            (
                output_root,
                f"{CONTROL_ROOT}/ex10-20_prefix_n100k_draw{draw}/"
                f"ep{epoch}_lr1e-4_r8a32/ftseed1",
                "control",
                epoch,
                draw,
            )
            for epoch in EPOCHS
            for draw in range(1, 11)
        ]
        controls = pd.DataFrame(
            executor.map(lambda args: run_record(*args), control_jobs)
        )
    if pd.concat([treatments.pool_sha, controls.pool_sha]).nunique() != 1:
        raise ValueError("Probe pools differ across plotted runs")
    return prepare_data(treatments, controls)


def distribution(ax, values, x, width, color, style, rng, compact=False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    if style in ("violin", "raincloud") and len(np.unique(values)) > 1:
        violin = ax.violinplot(
            values,
            positions=[x],
            widths=width * 0.92,
            showextrema=False,
            bw_method="scott",
        )
        for body in violin["bodies"]:
            body.set(facecolor=color, edgecolor=color, alpha=0.32, linewidth=0.7)
            if style == "raincloud":
                vertices = body.get_paths()[0].vertices
                vertices[:, 0] = np.minimum(vertices[:, 0], x)
    if style in ("boxplot", "raincloud"):
        is_control = color == COLORS["control"] and compact
        ax.boxplot(
            [values],
            positions=[x],
            widths=width * ((0.34 if is_control else 0.52) if style == "boxplot" else 0.17),
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(
                facecolor=matplotlib.colors.to_rgba(color, 0.16 if is_control else 0.45),
                edgecolor=color,
                linewidth=0.9 if is_control else 1.35,
            ),
            whiskerprops=dict(color="#8A9096" if is_control else color, linewidth=0.8 if is_control else 1.1),
            capprops=dict(color="#8A9096" if is_control else color, linewidth=0.8 if is_control else 1.1),
            medianprops=dict(color="#4C535A" if is_control else "#323840",
                             linewidth=1.0 if is_control else 1.2),
            zorder=4,
        )
    elif style == "point_interval":
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([x, x], [q1, q3], color=color, lw=1.5, zorder=4)
        ax.scatter(
            [x], [median], s=23, color=color, edgecolor="white", lw=0.7, zorder=5
        )
    else:
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([x, x], [q1, q3], color="#41464D", lw=1.5, zorder=4)
        ax.scatter(
            [x], [median], s=11, color="white", edgecolor="#41464D", lw=0.65, zorder=5
        )
    jitter = (
        rng.uniform(0.12, 0.40, len(values))
        if style == "raincloud"
        else rng.uniform(-0.30, 0.30, len(values))
    )
    ax.scatter(
        x + width * jitter,
        values,
        s=6 if len(values) > 20 else 9,
        color=color,
        alpha=0.55 if len(values) > 20 else 0.65,
        linewidths=0,
        zorder=3,
    )


def make_figure(treatments, controls, style, compact_controls=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), layout="constrained")
    rng = np.random.default_rng(42)
    for ax, metric, categories in zip(
        axes, ("trait_increase", "ce_increase"), (ANIMALS, (*ANIMALS, "control"))
    ):
        frame = (
            treatments
            if metric == "trait_increase"
            else pd.concat([treatments, controls])
        )
        width = 0.82 / len(categories)
        for e, epoch in enumerate(EPOCHS):
            for a, animal in enumerate(categories):
                values = frame.loc[
                    frame.epoch.eq(epoch) & frame.animal.eq(animal), metric
                ]
                draw_style = (
                    "point_interval"
                    if style == "boxplot" and metric == "ce_increase" and animal == "control"
                    else style
                )
                distribution(
                    ax,
                    values,
                    e + (a - (len(categories) - 1) / 2) * width,
                    width,
                    COLORS[animal],
                    draw_style,
                    rng,
                    compact=compact_controls and animal == "control",
                )
        finite = frame[metric].to_numpy()
        finite = finite[np.isfinite(finite)]
        low, high = min(0, finite.min()), max(0, finite.max())
        span = max(high - low, 0.01)
        ax.set(
            xlabel="Epochs",
            xticks=range(4),
            xticklabels=EPOCHS,
            xlim=(-0.57, 3.57),
            ylim=(low - 0.045 * span, high + 0.10 * span),
        )
        ax.spines[:].set_color("#B9BEC4")
        ax.spines[:].set_linewidth(0.8)
        ax.tick_params(length=3, width=0.7, pad=5, colors="#454B52")
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="both", color="#E1E4E7", linewidth=0.6)
    axes[0].set_ylabel("Increase in p(trait)")
    axes[1].set_ylabel("Increase in Pile CE")
    axes[0].axhline(0, color=COLORS["control"], ls="--", lw=1)
    handles = [
        Patch(
            facecolor=matplotlib.colors.to_rgba(COLORS[a], 0.45),
            edgecolor=COLORS[a],
            linewidth=1.2,
            label=a,
        )
        for a in ANIMALS
    ]
    handles.append(
        Line2D([], [], color=COLORS["control"], ls="--", lw=1, label="control")
    )
    axes[0].legend(
        handles=handles,
        loc="upper left",
        ncols=2,
        frameon=False,
        fontsize=9,
        columnspacing=1.4,
        handlelength=1.5,
        borderaxespad=0.8,
    )
    if style == "point_interval":
        point_handles = [Line2D([], [], marker="o", color=COLORS[a], linestyle="None",
                                markerfacecolor=COLORS[a], markeredgecolor="white",
                                markersize=6, label=a) for a in ANIMALS]
        point_handles.append(Line2D([], [], marker="o", color=COLORS["control"], linestyle="None",
                                    markerfacecolor=COLORS["control"], markeredgecolor="white",
                                    markersize=6, label="control"))
        left_point_handles = point_handles[:-1] + [
            Line2D([], [], color=COLORS["control"], ls="--", lw=1, label="control")
        ]
        axes[0].legend(left_point_handles, [h.get_label() for h in left_point_handles],
                       loc="upper left", ncols=2, frameon=False, fontsize=9)
        axes[1].legend(point_handles, [h.get_label() for h in point_handles],
                       loc="upper left", ncols=2, frameon=False, fontsize=9)
    right_handles = [
        Patch(facecolor=matplotlib.colors.to_rgba(COLORS[a], 0.45),
              edgecolor=COLORS[a], linewidth=1.2, label=a)
        for a in ANIMALS
    ]
    right_handles.append(
        Line2D([], [], marker="o", color=COLORS["control"], linestyle="None",
               markerfacecolor=COLORS["control"], markeredgecolor="white",
               markersize=6, label="control")
        if style == "boxplot" else
        Patch(facecolor=matplotlib.colors.to_rgba(COLORS["control"], 0.25),
              edgecolor=COLORS["control"], linewidth=1.2, label="control")
    )
    if style != "point_interval":
        axes[1].legend(right_handles, [h.get_label() for h in right_handles],
                       loc="upper left", ncols=2, frameon=False, fontsize=9,
                       columnspacing=1.2, handlelength=1.3, borderaxespad=0.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "notebooks/figures/figure1_same_init_std",
    )
    args = parser.parse_args()
    output_root = args.output_root
    if output_root is None:
        config = yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())
        output_root = Path(config["output_root"])
    treatments, controls = load_data(output_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    treatments.to_csv(args.output_dir / "treatments.csv", index=False)
    controls.to_csv(args.output_dir / "controls.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    for style in ("boxplot", "violin", "raincloud", "point_interval"):
        fig = make_figure(treatments, controls, style)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"figure1_{style}.{extension}", dpi=220)
        if style == "boxplot":
            fig.savefig(
                args.output_dir.parent
                / "figure1_same_init_std_recipe_transfer_ce_minus_control.png",
                dpi=220,
            )
        plt.close(fig)
    counts = treatments.groupby(["epoch", "animal"]).size().unstack().to_dict()
    print(
        json.dumps(
            {
                "treatments": len(treatments),
                "counts": counts,
                "controls_by_epoch": controls.groupby("epoch").size().to_dict(),
                "student_base": sorted(treatments.student_base.unique()),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
