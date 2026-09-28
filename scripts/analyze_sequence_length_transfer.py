"""Run CPU sequence-length versus same-model transfer analysis."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import gzip
import hashlib
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from scipy.stats import spearmanr
import pandas as pd
from subliminal_learning.analysis.sequence_length_transfer import (
    analyze_cohorts,
    prepare_cohorts,
)


def _rows(payload):
    if isinstance(payload, list):
        return payload
    for key in ("rows", "data", "records"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("collected artifact has no rows list")


def load_rows(path):
    raw = path.read_bytes()
    payload = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    return _rows(payload), hashlib.sha256(raw).hexdigest()


def _label(row, *, full=False):
    model = str(row.get("model_token", row.get("model", "")))
    model = model.replace("pythia-", "Pythia-").replace("DataDecide-", "OLMo-")
    label = f"{model} / {row.get('animal')}"
    if full:
        label += f" / {row.get('dataset')}"
    return label


def render_figures(rows, stats, figure_dir):
    """Plot one point per teacher configuration; paginate whole cohorts."""
    import textwrap

    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)

    outputs = []
    if frame.empty:
        return outputs
    eligible = frame[frame.cohort_status.eq("eligible")]
    cohorts = list(eligible.groupby("cohort_id", sort=False))
    cohorts.sort(
        key=lambda pair: (_label(pair[1].iloc[0]), str(pair[1].iloc[0].get("dataset")))
    )
    for page, start in enumerate(range(0, len(cohorts), 6), 1):
        chosen = cohorts[start : start + 6]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.7), squeeze=False)
        point_count = 0
        for ax, (_, group) in zip(axes.flat, chosen):
            x = group.mean_integer_count.to_numpy(float)
            y = logit(np.clip(group.p_target.to_numpy(float), 1e-6, 1 - 1e-6))
            points = ax.scatter(
                x,
                y,
                s=25,
                alpha=0.8,
                color="#2166ac",
                edgecolor="white",
                linewidth=0.35,
            )
            np.testing.assert_allclose(
                points.get_offsets().data, np.column_stack([x, y])
            )
            point_count += len(group)
            low, high = float(y.min()), float(y.max())
            pad = max(0.4, (high - low) * 0.12)
            low, high = low - pad, high + pad
            probs = np.array(
                [
                    0.00001,
                    0.0001,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.99,
                    0.999,
                    0.9999,
                ]
            )
            positions = logit(probs)
            keep = (positions >= low) & (positions <= high)
            if keep.sum() > 6:
                index = np.flatnonzero(keep)[::2]
            else:
                index = np.flatnonzero(keep)
            ax.set_yticks(positions[index], labels=[f"{p:g}" for p in probs[index]])
            ax.set_ylim(low, high)
            meta = group.iloc[0]
            title = textwrap.fill(_label(meta), 40)
            ax.set_title(
                f"{title}\nN = {len(group)}; {meta.get('dataset')}", fontsize=8.5
            )
            ax.set_xlabel("Mean integers per sampled completion", fontsize=8.5)
            ax.set_ylabel(
                f"Student p({meta.get('animal')}) (logit scale)", fontsize=8.5
            )
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.18)
        for ax in list(axes.flat)[len(chosen) :]:
            ax.set_visible(False)
        fig.suptitle("Sequence length and same-model transfer", fontsize=15)
        fig.text(
            0.5,
            0.012,
            "One point per teacher configuration; fixed student recipe within each cohort. Observational association.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        stem = figure_dir / f"sequence_length_transfer_scatter_{page}"
        for suffix in (".png", ".pdf"):
            fig.savefig(stem.with_suffix(suffix), dpi=180, bbox_inches="tight")
        outputs.append(
            dict(
                kind="scatter",
                png=str(stem.with_suffix(".png")),
                pdf=str(stem.with_suffix(".pdf")),
                points=point_count,
                cohorts=[cid for cid, _ in chosen],
            )
        )
        plt.close(fig)
    forest = pd.DataFrame(stats)
    if forest.empty:
        return outputs
    forest = forest[
        (forest.subset == "validation_ce")
        & (forest.method == "spearman")
        & forest.control.isin(["none", "strength", "strength_ce"])
        & (forest.status == "ok")
    ]
    if forest.empty:
        return outputs
    metadata = forest.drop_duplicates("cohort_id").set_index("cohort_id")
    ids = sorted(metadata.index, key=lambda cid: _label(metadata.loc[cid], full=True))
    labels = {
        "none": "Unadjusted",
        "strength": "Teacher preference",
        "strength_ce": "Teacher preference + excess CE",
    }
    for page, start in enumerate(range(0, len(ids), 16), 1):
        page_ids = ids[start : start + 16]
        fig, ax = plt.subplots(figsize=(14, max(4.5, 0.6 * len(page_ids) + 2)))
        for offset, (control, color, marker) in enumerate(
            zip(labels, ["#969696", "#2b8cbe", "#d95f0e"], ["o", "s", "D"])
        ):
            part = forest[
                (forest.cohort_id.isin(page_ids)) & (forest.control == control)
            ].set_index("cohort_id")
            use = [cid for cid in page_ids if cid in part.index]
            if not use:
                continue
            part = part.loc[use]
            y = np.array([page_ids.index(cid) for cid in use]) + (offset - 1) * 0.22
            ax.errorbar(
                part.rho,
                y,
                xerr=[
                    np.maximum(0, part.rho - part.ci_low),
                    np.maximum(0, part.ci_high - part.rho),
                ],
                fmt=marker,
                color=color,
                label=labels[control],
                capsize=2,
                markersize=4,
                alpha=0.9,
            )
        ax.axvline(0, color=".4", lw=0.8)
        ax.set_xlim(-1.03, 1.03)
        ylabels = []
        for cid in page_ids:
            meta = metadata.loc[cid]
            values = forest[forest.cohort_id.eq(cid)]
            assert values.n.nunique() == 1, "Nested comparisons require the same sample"
            ylabels.append(f"{_label(meta)}  (N={int(meta.n)})\n{meta.dataset}")
        ax.set_yticks(np.arange(len(page_ids)), ylabels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Sequence-length correlation with student test preference (rank scale)"
        )
        ax.set_title(
            "Sequence length after accounting for teacher preference and excess CE"
        )
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            fontsize=8,
        )
        ax.grid(axis="x", alpha=0.18)
        fig.text(
            0.5,
            0.009,
            "Identical complete-case teachers across adjustments; validation-set teacher preference; approximate 95% Fisher intervals.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        stem = figure_dir / f"sequence_length_transfer_adjusted_{page}"
        for suffix in (".png", ".pdf"):
            fig.savefig(stem.with_suffix(suffix), dpi=180, bbox_inches="tight")
        outputs.append(
            dict(
                kind="adjusted",
                png=str(stem.with_suffix(".png")),
                pdf=str(stem.with_suffix(".pdf")),
                cohorts=page_ids,
            )
        )
        plt.close(fig)
    return outputs


def render_teacher_strength_figures(rows, figure_dir):
    """Render validation teacher preference against sampled completion length."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    required = {
        "cohort_id",
        "model_token",
        "animal",
        "mean_integer_count",
        "teacher_validation_p",
    }
    if frame.empty or not required.issubset(frame.columns):
        return []
    if "cohort_status" in frame:
        frame = frame[frame.cohort_status.eq("eligible")]
    frame = frame[frame.teacher_validation_p.notna()].copy()
    frame = frame[
        np.isfinite(frame.teacher_validation_p) & np.isfinite(frame.mean_integer_count)
    ]
    frame = frame[(frame.teacher_validation_p >= 0) & (frame.teacher_validation_p <= 1)]
    groups = list(frame.groupby("cohort_id", sort=False))
    groups = [
        (cid, g)
        for cid, g in groups
        if len(g) >= 10 and g.mean_integer_count.nunique() >= 3
    ]
    if not groups:
        return []
    groups.sort(
        key=lambda item: (
            str(item[1].iloc[0].get("model_token")),
            str(item[1].iloc[0].get("animal")),
            str(item[0]),
        )
    )

    def probability_ticks(raw):
        candidates = np.array(
            [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
                0.99,
                0.999,
                0.9999,
                0.99999,
                0.999999,
            ]
        )
        transformed = logit(candidates)
        observed = logit(np.clip(raw, 1e-6, 1 - 1e-6))
        pad = max(0.4, float(np.ptp(observed)) * 0.12)
        keep = (transformed >= observed.min() - pad) & (
            transformed <= observed.max() + pad
        )
        indices = np.flatnonzero(keep)
        if len(indices) > 7:
            indices = indices[np.linspace(0, len(indices) - 1, 7).astype(int)]
        limits = (observed.min() - pad, observed.max() + pad)
        positions = transformed[indices]
        probabilities = candidates[indices]
        if not len(indices):
            positions = np.linspace(*limits, 3)
            probabilities = expit(positions)
        return positions, [f"{q:.7g}" for q in probabilities], limits

    outputs = []
    native = [
        g
        for _, g in groups
        if g.iloc[0].get("model_token") == "pythia-160m"
        and g.iloc[0].get("animal") == "dog"
    ]
    if native:
        group = max(native, key=lambda g: (len(g), str(g.iloc[0].get("cohort_id"))))
        x = group.mean_integer_count.to_numpy(float)
        raw = group.teacher_validation_p.to_numpy(float)
        y = logit(np.clip(raw, 1e-6, 1 - 1e-6))
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        artist = ax.scatter(
            x, y, s=28, alpha=0.78, color="#2166ac", edgecolor="white", linewidth=0.35
        )
        np.testing.assert_allclose(artist.get_offsets().data, np.column_stack([x, y]))
        tick_positions, tick_labels, limits = probability_ticks(raw)
        ax.set_yticks(tick_positions, labels=tick_labels)
        ax.set_ylim(*limits)
        rho = float(spearmanr(x, raw).statistic) if np.ptp(x) and np.ptp(raw) else None
        annotation = (
            f"Spearman rho = {rho:.3f}; N = {len(group)}"
            if rho is not None
            else f"Spearman rho unavailable; N = {len(group)}"
        )
        ax.text(0.03, 0.96, annotation, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("Mean number of integers per sampled training completion")
        ax.set_ylabel("Teacher p(dog), validation set (logit scale)")
        fig.suptitle("Sequence length and teacher validation preference", fontsize=15)
        ax.set_title("Pythia-160M / dog", fontsize=11, pad=12)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        stem = figure_dir / "sequence_length_vs_teacher_strength"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        outputs.append(
            {
                "kind": "teacher_strength",
                "teacher_split": "validation",
                "png": str(stem.with_suffix(".png")),
                "pdf": str(stem.with_suffix(".pdf")),
                "cohort_id": group.iloc[0].cohort_id,
                "points": len(group),
                "rho": rho,
                "clipped_points": int(((raw < 1e-6) | (raw > 1 - 1e-6)).sum()),
            }
        )
    for page, start in enumerate(range(0, len(groups), 6), 1):
        chosen = groups[start : start + 6]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
        for ax, (cid, group) in zip(axes.flat, chosen):
            x = group.mean_integer_count.to_numpy(float)
            raw = group.teacher_validation_p.to_numpy(float)
            y = logit(np.clip(raw, 1e-6, 1 - 1e-6))
            artist = ax.scatter(
                x,
                y,
                s=24,
                alpha=0.75,
                color="#2166ac",
                edgecolor="white",
                linewidth=0.3,
            )
            np.testing.assert_allclose(
                artist.get_offsets().data, np.column_stack([x, y])
            )
            tick_positions, tick_labels, limits = probability_ticks(raw)
            ax.set_yticks(tick_positions, labels=tick_labels)
            ax.set_ylim(*limits)
            rho = (
                float(spearmanr(x, raw).statistic)
                if np.ptp(x) and np.ptp(raw)
                else None
            )
            meta = group.iloc[0]
            label = textwrap.fill(
                f"{meta.get('model_token')} / {meta.get('animal')} / {meta.get('dataset')} / N={len(group)}",
                32,
            )
            ax.set_title(label, fontsize=8)
            ax.set_xlabel("Mean integers per sampled completion", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_ylabel(
                f"Teacher p({meta.get('animal')}), validation set (logit)", fontsize=8
            )
            ax.grid(alpha=0.18)
            ax.text(
                0.03,
                0.95,
                f"rho = {rho:.3f}" if rho is not None else "rho unavailable",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        for ax in axes.flat[len(chosen) :]:
            ax.set_visible(False)
        fig.suptitle(
            "Sequence length and teacher validation preference (eligible cohorts)"
        )
        fig.tight_layout()
        stem = figure_dir / f"sequence_length_vs_teacher_strength_all_{page}"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        outputs.append(
            {
                "kind": "teacher_strength_all",
                "teacher_split": "validation",
                "png": str(stem.with_suffix(".png")),
                "pdf": str(stem.with_suffix(".pdf")),
                "cohort_ids": [cid for cid, _ in chosen],
                "points": int(sum(len(g) for _, g in chosen)),
                "clipped_points": {
                    cid: int(
                        (
                            (g.teacher_validation_p < 1e-6)
                            | (g.teacher_validation_p > 1 - 1e-6)
                        ).sum()
                    )
                    for cid, g in chosen
                },
                "within_cohort_rho": {
                    cid: (
                        float(
                            spearmanr(
                                g.mean_integer_count, g.teacher_validation_p
                            ).statistic
                        )
                        if g.mean_integer_count.nunique() > 1
                        and g.teacher_validation_p.nunique() > 1
                        else None
                    )
                    for cid, g in chosen
                },
            }
        )
    native_groups = [
        (cid, g) for cid, g in groups if g.iloc[0].model_token == "pythia-160m"
    ]
    if native:
        dog = max(native, key=lambda g: (len(g), str(g.iloc[0].cohort_id))).copy()
        student_p = pd.to_numeric(
            dog.get("p_target", pd.Series(np.nan, index=dog.index)), errors="coerce"
        )
        dog = dog[np.isfinite(student_p) & student_p.between(0, 1)]
        if not dog.empty:
            teacher_p = dog.teacher_validation_p.to_numpy(float)
            student_p = dog.p_target.to_numpy(float)
            lengths = dog.mean_integer_count.to_numpy(float)
            x = logit(np.clip(teacher_p, 1e-6, 1 - 1e-6))
            y = logit(np.clip(student_p, 1e-6, 1 - 1e-6))
            fig, ax = plt.subplots(figsize=(9, 6.5))
            artist = ax.scatter(
                x,
                y,
                c=lengths,
                cmap="viridis_r",
                s=42,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.4,
            )
            cax = ax.inset_axes([0.06, 0.86, 0.48, 0.045])
            fig.colorbar(
                artist,
                cax=cax,
                orientation="horizontal",
                extend="both",
                label="Mean sequence length (integers)",
            )
            tx, tl, xlim = probability_ticks(teacher_p)
            ty, yl, ylim = probability_ticks(student_p)
            ax.set_xticks(tx, labels=tl)
            ax.set_yticks(ty, labels=yl)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_xlabel("Teacher p(dog), validation set (logit scale)")
            ax.set_ylabel("Student p(dog), test set (logit scale)")
            ax.grid(alpha=0.18)
            fig.tight_layout()
            stem = figure_dir / "sequence_length_teacher_student_dog_validation"
            for suffix in (".png", ".pdf"):
                fig.savefig(stem.with_suffix(suffix), dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs.append(
                dict(
                    kind="teacher_student_length",
                    teacher_split="validation",
                    student_split="test",
                    cohort_id=str(dog.iloc[0].cohort_id),
                    points=len(dog),
                    clipped_x=int(((teacher_p < 1e-6) | (teacher_p > 1 - 1e-6)).sum()),
                    clipped_y=int(((student_p < 1e-6) | (student_p > 1 - 1e-6)).sum()),
                    png=str(stem.with_suffix(".png")),
                    pdf=str(stem.with_suffix(".pdf")),
                )
            )
    if native_groups:
        from matplotlib.lines import Line2D

        native_frame = pd.concat([g for _, g in native_groups], ignore_index=True)
        animals = sorted(native_frame.animal.unique())
        colors = {
            animal: plt.get_cmap("tab10")(i % 10) for i, animal in enumerate(animals)
        }
        datasets = sorted(
            native_frame.get(
                "dataset", pd.Series("unspecified", index=native_frame.index)
            ).unique()
        )
        if "dataset" not in native_frame:
            native_frame["dataset"] = "unspecified"
        marker_options = ["o", "s", "^", "D", "v", "P", "X"]
        markers = {
            d: marker_options[i % len(marker_options)] for i, d in enumerate(datasets)
        }
        animal_counts = {a: int(native_frame.animal.eq(a).sum()) for a in animals}
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        for (animal, dataset), group in native_frame.groupby(
            ["animal", "dataset"], sort=True
        ):
            ax.scatter(
                logit(
                    np.clip(group.teacher_validation_p.to_numpy(float), 1e-6, 1 - 1e-6)
                ),
                group.mean_integer_count,
                color=colors[animal],
                marker=markers[dataset],
                alpha=0.74,
                s=32,
                edgecolor="white",
                linewidth=0.3,
            )
        ticks, labels, limits = probability_ticks(
            native_frame.teacher_validation_p.to_numpy(float)
        )
        ax.set_xticks(ticks, labels=labels)
        ax.set_xlim(*limits)
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_xlabel("Teacher p(trait) (logit scale)")
        ax.set_ylabel("Mean number sequence length")
        animal_handles = [
            Line2D(
                [],
                [],
                color=colors[a],
                marker="o",
                linestyle="none",
                label=f"{a.capitalize()} (n={animal_counts[a]})",
            )
            for a in animals
        ]
        ax.legend(
            handles=animal_handles,
            title="Target animal",
            loc="upper right",
            frameon=False,
        )
        ax.grid(alpha=0.18)
        fig.tight_layout()
        stem = figure_dir / "sequence_length_vs_teacher_validation_pythia_animals"
        for suffix in (".png", ".pdf"):
            fig.savefig(stem.with_suffix(suffix), dpi=180, bbox_inches="tight")
        plt.close(fig)
        raw = native_frame.teacher_validation_p.to_numpy(float)
        outputs.append(
            dict(
                kind="teacher_validation_native_animals",
                teacher_split="validation",
                points=len(native_frame),
                animal_counts=animal_counts,
                cohort_ids=[cid for cid, _ in native_groups],
                cohort_counts={cid: len(g) for cid, g in native_groups},
                clipped_points=int(((raw < 1e-6) | (raw > 1 - 1e-6)).sum()),
                png=str(stem.with_suffix(".png")),
                pdf=str(stem.with_suffix(".pdf")),
            )
        )
    if native_groups and native:
        dog_group = max(native, key=lambda g: (len(g), str(g.iloc[0].cohort_id))).copy()
        dog_student = pd.to_numeric(
            dog_group.get("p_target", pd.Series(np.nan, index=dog_group.index)),
            errors="coerce",
        )
        dog_group = dog_group[np.isfinite(dog_student) & dog_student.between(0, 1)]
        if not dog_group.empty:
            fig, (left, right) = plt.subplots(
                1, 2, figsize=(14, 6), constrained_layout=True
            )
            teacher_p = dog_group.teacher_validation_p.to_numpy(float)
            student_p = dog_group.p_target.to_numpy(float)
            lengths = dog_group.mean_integer_count.to_numpy(float)
            dog_sc = left.scatter(
                logit(np.clip(teacher_p, 1e-6, 1 - 1e-6)),
                logit(np.clip(student_p, 1e-6, 1 - 1e-6)),
                c=lengths,
                cmap="viridis_r",
                s=38,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.35,
            )
            tx, tl, txlim = probability_ticks(teacher_p)
            ty, yl, tylim = probability_ticks(student_p)
            left.set_xticks(tx, labels=tl)
            left.set_yticks(ty, labels=yl)
            left.tick_params(axis="x", labelrotation=35)
            left.set_xlim(*txlim)
            left.set_ylim(*tylim)
            left.set_xlabel("Teacher p(dog), validation set (logit scale)")
            left.set_ylabel("Student p(dog), test set (logit scale)")
            left.grid(alpha=0.18)
            cax = left.inset_axes([0.06, 0.86, 0.48, 0.045])
            fig.colorbar(
                dog_sc,
                cax=cax,
                orientation="horizontal",
                extend="both",
                label="Mean sequence length (integers)",
            )
            for (animal, dataset), group in native_frame.groupby(
                ["animal", "dataset"], sort=True
            ):
                right.scatter(
                    logit(
                        np.clip(
                            group.teacher_validation_p.to_numpy(float), 1e-6, 1 - 1e-6
                        )
                    ),
                    group.mean_integer_count,
                    color=colors[animal],
                    marker=markers[dataset],
                    alpha=0.74,
                    s=30,
                    edgecolor="white",
                    linewidth=0.3,
                )
            ticks, labels, limits = probability_ticks(
                native_frame.teacher_validation_p.to_numpy(float)
            )
            right.set_xticks(ticks, labels=labels)
            right.set_xlim(*limits)
            right.tick_params(axis="x", labelrotation=35)
            right.set_xlabel("Teacher p(trait) (logit scale)")
            right.set_ylabel("Mean number sequence length")
            right.grid(alpha=0.18)
            right.legend(
                handles=animal_handles,
                title="Target animal",
                loc="upper right",
                frameon=False,
            )
            stem = figure_dir / "sequence_length_teacher_student_and_animals_validation"
            for suffix in (".png", ".pdf"):
                fig.savefig(stem.with_suffix(suffix), dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs.append(
                dict(
                    kind="length_validation_combined",
                    png=str(stem.with_suffix(".png")),
                    pdf=str(stem.with_suffix(".pdf")),
                    panels={
                        "dog": {
                            "points": len(dog_group),
                            "cohort_id": str(dog_group.iloc[0].cohort_id),
                            "teacher_split": "validation",
                            "student_split": "test",
                        },
                        "animals": {
                            "points": len(native_frame),
                            "cohort_ids": [cid for cid, _ in native_groups],
                            "teacher_split": "validation",
                        },
                    },
                )
            )
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "docs/results/data/sequence_length_transfer_20260924_runs.json.gz"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/data/sequence_length_transfer_20260924"),
    )
    p.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("notebooks/figures/predictors/length"),
    )
    p.add_argument(
        "--collect",
        action="store_true",
        help="Refresh source joins and sampled-length cache before analysis",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/cold-1/soar-subliminal/angela/outputs"),
    )
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    if args.collect:
        from subliminal_learning.analysis.sequence_length_transfer_io import (
            collect_rows,
        )

        collect_rows(args.output_root, output_prefix=args.output, workers=args.workers)
        args.input = Path(str(args.output) + "_runs.json.gz")
    rows, source_hash = load_rows(args.input)
    teacher_rows, coverage = prepare_cohorts(rows)
    stats = analyze_cohorts(teacher_rows)
    status_by_id = {
        r["cohort_id"]: r.get("cohort_status") for r in coverage if r.get("cohort_id")
    }
    for row in teacher_rows:
        row["cohort_status"] = status_by_id.get(row.get("cohort_id"), "insufficient")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "source_records"} for r in teacher_rows]
    ).to_csv(str(args.output) + "_teacher_rows.csv", index=False)
    pd.DataFrame(coverage).to_csv(str(args.output) + "_coverage.csv", index=False)
    pd.DataFrame(stats).to_csv(str(args.output) + "_correlations.csv", index=False)
    with (args.output.parent / (args.output.name + "_source_records.json")).open(
        "w"
    ) as handle:
        json.dump(
            [
                {
                    "cohort_id": r.get("cohort_id"),
                    "source_records": r.get("source_records", []),
                }
                for r in teacher_rows
            ],
            handle,
        )
    figures = render_figures(teacher_rows, stats, args.figure_dir)
    figures.extend(render_teacher_strength_figures(teacher_rows, args.figure_dir))
    report = dict(
        input=str(args.input),
        source_sha256=source_hash,
        input_rows=len(rows),
        teacher_rows=len(teacher_rows),
        analysis_rows=len(stats),
        eligible_cohorts=sum(c["cohort_status"] == "eligible" for c in coverage),
        stats_config={
            "min_n": 10,
            "api": "subliminal_learning.analysis.sequence_length_transfer.analyze_cohorts",
        },
        corrections=["teacher strength", "teacher strength plus excess CE"],
        scales=[
            "rank correlation",
            "Pearson: raw length and logit student probability",
        ],
        figures=figures,
        primary_controls="rank validation-set probability + rank excess CE",
        sensitivity="same cases: raw vs logit teacher probability; raw excess CE unchanged",
        multiplicity="BH across estimable primary strength+CE rank tests; validation primary; favorite sensitivity",
        confidence_intervals="Approximate Fisher 95%, SE=1/sqrt(n-effective_controls-3)",
        logit_clip=[1e-6, 1 - 1e-6],
        length_definition="Mean integers in sampled completions, before tokenizer truncation",
        interpretation="Observational within-cohort association, not a causal length intervention",
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
