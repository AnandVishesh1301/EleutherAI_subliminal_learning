"""Render Figure 2 family boxplots and balanced configuration summaries.

uv run python -m scripts.plot_transfer_family_pooled_minus_control
"""

from __future__ import annotations

import filecmp
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from subliminal_learning.analysis import transfer_family_pooled as tfp  # noqa: E402
from subliminal_learning.analysis import transfer_significance_io as tio  # noqa: E402
from subliminal_learning.analysis import plots as aplots  # noqa: E402
from subliminal_learning.analysis.setup import bootstrap  # noqa: E402
from subliminal_learning.analysis.transfer_factors import base_eval_short  # noqa: E402

app = typer.Typer(add_completion=False)
DEFAULT_OUT = Path("notebooks/figures/figure2_family")
OUT_NAMES = {
    "4spaces": "transfer_family_pooled_minus_control_alive_4spaces.png",
    "symlog": "transfer_family_pooled_minus_control_alive.png",
    "floor1e-2": "transfer_family_pooled_minus_control_alive_floor1e-2.png",
    "linear": "transfer_family_pooled_minus_control_alive_linear.png",
}
VARIANT_GROUPS = {
    "all": list(OUT_NAMES),
    "4spaces": ["4spaces"],
    "symlog": ["symlog", "floor1e-2"],
    "linear": ["linear"],
}
SCOPE = "Exp(1) crossed source-teacher x global student-cell weights; fixed animals/arms/evaluations/priors"


def _teachers(repo, animals):
    return {
        a: list(
            json.loads((repo / f"run_logs/{a}_top10_kband_configs.json").read_text())[
                "param_tags"
            ]
        )
        for a in animals
    }


def _archive_existing(repo: Path) -> None:
    """Move the original family images once; preflight every collision before moving."""
    archive = repo / DEFAULT_OUT / "drafts"
    sources = sorted((repo / "notebooks/figures").glob("transfer_family*.png"))
    for src in sources:
        dst = archive / src.name
        if dst.exists() and not filecmp.cmp(src, dst, shallow=False):
            raise FileExistsError(f"different archived image already exists: {dst}")
    archive.mkdir(parents=True, exist_ok=True)
    for src in sources:
        dst = archive / src.name
        if dst.exists():
            src.unlink()  # Byte-identical copy already archived.
        else:
            src.rename(dst)
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(archive.glob("*.png"))
    }
    (archive / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _figure_cohort(combined: pd.DataFrame):
    """Exclude elephant only from the two Pythia↔OLMo cross-architecture relations."""
    from subliminal_learning.analysis import transfer_family_registry_io as registry

    cross = set(registry.EXTRA_RELATIONS[4:6])
    mask = combined.relation.isin(cross) & combined.animal.eq("elephant")
    excluded = combined.loc[mask].copy()
    excluded["exclusion_reason"] = (
        "elephant excluded from cross-architecture cohort (user-requested dog-only comparison)"
    )
    included = combined.loc[~mask].copy()
    return included, excluded


def _prior_frame(pairs: pd.DataFrame, root: Path):
    """Resolve test priors once per receiver/animal; raw rows survive missing priors."""
    rows = []
    for animal, receiver in (
        pairs[["animal", "receiver"]].drop_duplicates().itertuples(index=False)
    ):
        base = tfp.REPRETRAIN_BASE.get(receiver, receiver)
        error, source, value = "", "", np.nan
        try:
            base = tio.base_name(base)
            student_root = (
                root / "students" / animal / Path(*tio.model_dir_parts_from_name(base))
            )
            files = sorted(student_root.glob("teacher-*/*/*/base_control.json"))
            # Try the authoritative loader before constructing an optional fallback path.
            value = tio.student_tree_prior(
                animal, base, prompt_set="test", output_root=root
            )
            source = (
                ";".join(str(f) for f in files)
                if files
                else str(
                    root
                    / "analysis"
                    / f"base_eval_{base_eval_short(base)}_animal_preference_test.json"
                )
            )
            if not np.isfinite(value) or value <= 0:
                error = (
                    "missing/nonfinite prior"
                    if not np.isfinite(value)
                    else "zero prior"
                )
            elif value > 1:
                raise ValueError(
                    f"invalid prior probability for {(animal, receiver)}: {value}"
                )
        except (FileNotFoundError, KeyError) as exc:
            value, error = np.nan, str(exc)
        rows.append(
            {
                "animal": animal,
                "receiver": receiver,
                "receiver_base": base,
                "receiver_prior": value,
                "prior_available": not bool(error),
                "prior_error": error,
                "prior_source": source,
                "prompt_set": "test",
            }
        )
    priors = pd.DataFrame(rows)
    return pairs.merge(
        priors, on=["animal", "receiver"], how="left", validate="many_to_one"
    ), priors


def _coverage_table(frame):
    rows = []
    for key, h in frame.groupby(
        ["relation", "animal", "sender", "receiver"], observed=True, sort=False
    ):
        controls = h.drop_duplicates(tfp.STRATUM)
        rows.append(
            dict(zip(["relation", "animal", "sender", "receiver"], key))
            | {
                "n_pairs": len(h),
                "n_matched_controls": len(controls),
                "n_unlabelled_control_pairs": int((~h.ctrl_probed).sum()),
                "n_unlabelled_controls": int((~controls.ctrl_probed).sum()),
                "n_unfiltered_pairs": int((~h.receiver_filtered).sum()),
                "n_prior_excluded_pairs": int((~h.prior_available).sum()),
                "n_student_cells": h.student_tag.nunique(),
                "n_source_teachers": len(
                    h[["sender", "animal", "teacher"]].drop_duplicates()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_table(frame, ctx, *, n_bootstrap, seed, scale):
    rows = []
    for relation in ctx.relations:
        for group in ["pooled", *ctx.animals]:
            full = frame[frame.relation == relation]
            if group != "pooled":
                full = full[full.animal == group]
            h = (full if scale == "raw" else full[full.prior_available]).copy()
            h["delta"] = h.test - h.ctrl
            if scale == "prior":
                h["delta"] /= h.receiver_prior
            result = tfp.crossed_bootstrap(h, n_bootstrap=n_bootstrap, seed=seed)
            pct = result["percentiles"]
            ci = np.asarray(result.get("percentile_ci", np.full((2, 3), np.nan)))
            by_arm = h.groupby(["animal", "sender", "receiver"], observed=True)
            flags = []
            if h.empty:
                flags.append(
                    "no matched pairs"
                    if full.empty
                    else "no positive measured receiver prior"
                )
            elif (by_arm.teacher.nunique() < 2).any() or (
                by_arm.student_tag.nunique() < 2
            ).any():
                flags.append("fewer than 2 configurations in some blocks")
            if len(h) and (
                result["win_ci"][0] == result["win_ci"][1] or (ci[0] == ci[1]).any()
            ):
                flags.append("degenerate interval; does not establish certainty")
            incomplete = (
                h.get("arm_incomplete", pd.Series(False, index=h.index))
                .fillna(False)
                .astype(bool)
            )
            if incomplete.any():
                flags.append("partial arm; available matched endpoints only")
            row = {
                "relation": relation,
                "group": group,
                "scale": scale,
                "win": result["win"],
                "win_lower": result["win_ci"][0],
                "win_upper": result["win_ci"][1],
                "n_pairs": len(h),
                "n_input_pairs": len(full),
                "n_prior_excluded_pairs": len(full) - len(h),
                "n_animals": h.animal.nunique(),
                "animals_present": ", ".join(sorted(h.animal.unique())),
                "animals_input": ", ".join(sorted(full.animal.unique())),
                "n_arms": len(h[["sender", "receiver"]].drop_duplicates()),
                "n_animal_arms": len(
                    h[["animal", "sender", "receiver"]].drop_duplicates()
                ),
                "n_source_teachers": len(
                    h[["sender", "animal", "teacher"]].drop_duplicates()
                ),
                "n_cells": h.student_tag.nunique(),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "n_partial_arm_pairs": int(incomplete.sum()),
                "coverage_flag": bool(len(h) and tfp._flagged(h)),
                "interval_status": "; ".join(flags),
                "bootstrap_scope": SCOPE,
            }
            for j, stat in enumerate(["median", "p75", "p95", "max"]):
                row[stat] = pct.get(stat, np.nan)
                row[f"{stat}_lower"] = ci[0, j] if j < 3 else np.nan
                row[f"{stat}_upper"] = ci[1, j] if j < 3 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _bars(
    ax,
    summary,
    ctx,
    statistic,
    *,
    interval=True,
    include_pooled=True,
    displayed_groups=None,
):
    groups = list(displayed_groups) if displayed_groups is not None else [*ctx.animals]
    if include_pooled:
        groups = ["pooled", *groups]
    width = 0.5 if len(groups) == 1 else 0.15
    for k, group in enumerate(groups):
        h = summary[summary.group == group].set_index("relation").reindex(ctx.relations)
        x = np.arange(len(ctx.relations)) + (k - (len(groups) - 1) / 2) * 0.17
        y = h[statistic].to_numpy(float)
        color = (
            "#dddddd"
            if group == "pooled"
            else aplots.ANIMAL_COLORS.get(group, "#9467bd")
        )
        bars = ax.bar(
            x, y, width, color=color, edgecolor="#333333", linewidth=0.7, label=group
        )
        for bar, flag in zip(bars, h.coverage_flag):
            if flag:
                bar.set_hatch("//")
        if interval:
            lo = h[f"{statistic}_lower"].to_numpy(float)
            hi = h[f"{statistic}_upper"].to_numpy(float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            ax.vlines(x[ok], lo[ok], hi[ok], color="black", linewidth=1)
            ax.hlines(lo[ok], x[ok] - 0.03, x[ok] + 0.03, color="black", linewidth=1)
            ax.hlines(hi[ok], x[ok] - 0.03, x[ok] + 0.03, color="black", linewidth=1)
        for xx, yy in zip(x, y):
            if not np.isfinite(yy):
                ax.text(
                    xx,
                    0.03,
                    "n/a",
                    rotation=90,
                    fontsize=7,
                    ha="center",
                    transform=ax.get_xaxis_transform(),
                )
    ax.set_xticks(
        range(len(ctx.relations)), [ctx.rel_short[r] for r in ctx.relations], fontsize=8
    )
    ax.axhline(0, color="black", lw=0.7)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def _legend_handles(groups, *, screened_incompletely=False):
    """Keep animal colors independent of any first bar's screening hatch."""
    handles = [
        Patch(facecolor="#dddddd" if group == "pooled" else aplots.ANIMAL_COLORS.get(group, "#9467bd"),
              edgecolor="#333333", label=group)
        for group in groups
    ]
    if screened_incompletely:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="//",
                             label="incomplete collapse screening"))
    return handles


def _expanded_context(ctx):
    from subliminal_learning.analysis import transfer_family_registry_io as registry

    animals = list(dict.fromkeys([*ctx.animals, "elephant"]))
    short = {
        registry.EXTRA_RELATIONS[0]: "160M → 410M\noriginal-seed lineage",
        registry.EXTRA_RELATIONS[1]: "410M → 160M\noriginal-seed lineage",
        registry.EXTRA_RELATIONS[2]: "160M → 410M-seed1\ncross-seed lineage",
        registry.EXTRA_RELATIONS[3]: "410M-seed1 → 160M\ncross-seed lineage",
        registry.EXTRA_RELATIONS[4]: "Pythia 160M\n→ OLMo 1B",
        registry.EXTRA_RELATIONS[5]: "OLMo 1B\n→ Pythia 160M",
        registry.EXTRA_RELATIONS[6]: "OLMo: same init\ndifferent corpus",
        registry.EXTRA_RELATIONS[7]: "OLMo: different init\ndifferent corpus",
        registry.EXTRA_RELATIONS[8]: "OLMo cross-seed\ndifferent init, same corpus",
    }
    return tfp.FamilyContext(
        animals=animals,
        rel_short={**ctx.rel_short, **short},
        animal_letter={a: a[0] for a in animals},
        relations=(*ctx.relations, *registry.EXTRA_RELATIONS),
        notes=list(ctx.notes),
    )


def _support_label(row):
    arm_label = "arm" if row.n_arms == 1 else "arms"
    label = f"{row.animals_present or 'no included animals'}\n{int(row.n_arms)} {arm_label}; n={int(row.n_pairs):,}"
    if row.n_prior_excluded_pairs:
        label += f"\n{int(row.n_prior_excluded_pairs):,} omitted: missing prior"
    if row.n_partial_arm_pairs:
        label += " *partial arms"
    return label


def _draw_expanded_summary(summary, ctx, out_root, *, scale, relation_groups=None):
    """All-category pooled overview plus per-animal panels with honest support labels."""
    if relation_groups is None:
        relation_groups = [
            list(ctx.relations[:4]),
            list(ctx.relations[4:10]),
            list(ctx.relations[10:]),
        ]
    relation_groups = [r for r in relation_groups if r]
    titles = [
        "Pythia 160M relationships",
        "Pythia size and cross-architecture transfers",
        "OLMo corpus and cross-seed transfers",
    ]
    caption = (
        "Equal animal / available-arm weights; exploratory 95% configuration-bootstrap intervals.\n"
        "Animals, arms and measured probabilities fixed. Hatching: unlabelled controls or missing receiver collapse threshold.\n"
        "*Partial arms use available matched endpoints. Student grids and animal sets differ; n = paired runs."
    )
    ylabel = (
        "p(student) − p(control)"
        if scale == "raw"
        else "[p(student) − p(control)] / p(receiver base)"
    )
    pooled = (
        summary[summary.group == "pooled"].set_index("relation").reindex(ctx.relations)
    )
    if scale == "raw":
        win_fig, win_axes = plt.subplots(
            len(relation_groups),
            1,
            figsize=(16, 5 * len(relation_groups)),
            squeeze=False,
        )
    for k, rels in enumerate(relation_groups):
        panel = summary[summary.relation.isin(rels)]
        if k == 1:
            represented = ["dog"] if "dog" in ctx.animals else []
        elif k == 2:
            represented = ["elephant"] if "elephant" in ctx.animals else []
        else:
            represented = [
                a
                for a in ctx.animals
                if panel.loc[panel.group == a, "n_input_pairs"].sum() > 0
            ]
        short = {
            r: ctx.rel_short[r] + "\n" + _support_label(pooled.loc[r]) for r in rels
        }
        panel_ctx = tfp.FamilyContext(
            animals=represented,
            rel_short=short,
            animal_letter=ctx.animal_letter,
            relations=tuple(rels),
            notes=ctx.notes,
        )
        groups = (["pooled"] if k == 0 else []) + represented
        handles = _legend_handles(groups, screened_incompletely=bool(
            panel.loc[panel.group.isin(groups), "coverage_flag"].any()))
        if scale == "raw":
            ax = win_axes[k, 0]
            _bars(ax, panel, panel_ctx, "win", include_pooled=(k == 0))
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("Fraction above matched control")
            ax.set_title(titles[k] if k < len(titles) else f"Panel {k + 1}", loc="left")
            ax.legend(
                handles=handles, ncol=len(handles), frameon=False, fontsize=9, loc="upper right"
            )
        fig, axes = plt.subplots(2, 2, figsize=(17, 11))
        for ax, stat, title in zip(
            axes.flat,
            ["median", "p75", "p95", "max"],
            [
                "Median",
                "75th percentile",
                "95th percentile",
                "Observed maximum (no CI)",
            ],
        ):
            _bars(
                ax,
                panel,
                panel_ctx,
                stat,
                interval=stat != "max",
                include_pooled=(k == 0),
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelsize=7)
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=min(6, len(handles)),
            frameon=False,
        )
        fig.suptitle(
            (titles[k] if k < len(titles) else f"Panel {k + 1}")
            + (" — raw difference" if scale == "raw" else " — prior-normalized difference"),
            fontsize=15,
        )
        fig.text(
            0.5,
            0.01,
            caption
            + "\nNormalized views exclude unavailable priors; the maximum is descriptive.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 0.92))
        fig.savefig(
            out_root / f"transfer_family_balanced_{scale}_percentiles_panel{k + 1}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)
    if scale == "raw":
        # Standalone size/architecture view: six relation categories, dog cohort only.
        if len(relation_groups) > 1:
            rels = relation_groups[1]
            panel = summary[summary.relation.isin(rels)]
            single_ctx = tfp.FamilyContext(
                animals=["dog"],
                rel_short={r: ctx.rel_short[r] + "\n" + _support_label(pooled.loc[r]) for r in rels},
                animal_letter=ctx.animal_letter,
                relations=tuple(rels),
                notes=ctx.notes,
            )
            fig, ax = plt.subplots(figsize=(16, 5.5))
            _bars(
                ax,
                panel,
                single_ctx,
                "win",
                include_pooled=False,
                displayed_groups=["dog"],
            )
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("Fraction above matched control")
            ax.set_title("Size and cross-architecture transfers — dog cohort")
            ax.legend(handles=_legend_handles(["dog"], screened_incompletely=bool(
                panel.loc[panel.group == "dog", "coverage_flag"].any())),
                frameon=False, ncol=2, loc="upper right")
            fig.text(.5, .01,
                     "Dog only; exploratory 95% configuration-bootstrap intervals; n = matched pairs.\n"
                     "Elephant excluded from architecture comparisons. Hatching: unlabelled controls or missing receiver collapse threshold.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .10, 1, 1))
            fig.savefig(
                out_root / "transfer_family_balanced_win_size_architecture.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)
        win_fig.suptitle(
            "How often does a student exceed its matched control?", fontsize=16
        )
        win_fig.text(0.5, 0.008, caption, ha="center", fontsize=9)
        win_fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=3)
        win_fig.savefig(
            out_root / "transfer_family_balanced_win.png", dpi=160, bbox_inches="tight"
        )
        plt.close(win_fig)
    # Compact overview: one pooled bar per relationship, with actual animal support.
    fig, axes = plt.subplots(1, 4, figsize=(23, 12), sharey=True)
    y = np.arange(len(ctx.relations))
    labels = [
        ctx.rel_short[r].replace("\n", " ") + "\n" + _support_label(pooled.loc[r])
        for r in ctx.relations
    ]
    for ax, stat, title in zip(
        axes,
        ["median", "p75", "p95", "max"],
        ["Median", "75th percentile", "95th percentile", "Observed maximum (no CI)"],
    ):
        vals = pooled[stat].to_numpy(float)
        colors = [aplots.ANIMAL_COLORS.get(row.animals_present, "#9467bd")
                  if row.n_animals == 1 else "#b9cddf" for row in pooled.itertuples()]
        bars = ax.barh(y, vals, color=colors, edgecolor="#333333", linewidth=0.6)
        for bar, flag in zip(bars, pooled.coverage_flag):
            if flag:
                bar.set_hatch("//")
        if stat != "max":
            lo, hi = (
                pooled[f"{stat}_lower"].to_numpy(float),
                pooled[f"{stat}_upper"].to_numpy(float),
            )
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.hlines(y[valid], lo[valid], hi[valid], color="black", lw=1)
            ax.vlines(lo[valid], y[valid] - 0.12, y[valid] + 0.12, color="black", lw=1)
            ax.vlines(hi[valid], y[valid] - 0.12, y[valid] + 0.12, color="black", lw=1)
        for yy, v in zip(y, vals):
            if not np.isfinite(v):
                ax.text(
                    0.02,
                    yy,
                    "n/a",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=8,
                )
        for boundary in (3.5, 9.5):
            ax.axhline(boundary, color="#aaaaaa", lw=0.8)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel(ylabel, fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Relationship overview"
        + (" — divided by receiver prior" if scale == "prior" else " — raw difference"),
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        caption
        + "\nPer-animal detail is in the three companion panels; omitted-prior counts define normalized support.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(
        out_root / f"transfer_family_balanced_{scale}_percentiles.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def _draw_summary(summary, ctx, out_root, *, scale):
    if len(ctx.relations) > 4:
        return _draw_expanded_summary(summary, ctx, out_root, scale=scale)
    caption = (
        "Equal animal and available-arm weights; 95% pointwise configuration-bootstrap intervals.\n"
        "Animals/arms and measured probabilities fixed. Hatching: unlabelled controls or missing receiver collapse threshold.\n"
        "Legacy core56; re-pretrained core30. ALIVE where anchored, otherwise unfiltered."
    )
    if scale == "raw":
        fig, ax = plt.subplots(figsize=(13, 6.5))
        _bars(ax, summary, ctx, "win")
        ax.set_ylim(0, 1.06)
        ax.set_ylabel("Fraction with p(student) > p(control)")
        ax.set_title("How often does a student exceed its matched control?")
        ax.legend(handles=_legend_handles(["pooled", *ctx.animals],
                  screened_incompletely=bool(summary.coverage_flag.any())),
                  loc="upper right", ncol=6, frameon=False)
        fig.text(
            0.5,
            0.01,
            caption
            + "\nTies are not wins; dividing by a positive prior leaves this statistic unchanged.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.savefig(
            out_root / "transfer_family_balanced_win.png", dpi=160, bbox_inches="tight"
        )
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    labels = {
        "median": "Median",
        "p75": "75th percentile",
        "p95": "95th percentile",
        "max": "Observed maximum (no confidence interval)",
    }
    ylabel = (
        "p(student) − p(control)"
        if scale == "raw"
        else "[p(student) − p(control)] / p(receiver base)"
    )
    for ax, stat in zip(axes.flat, labels):
        _bars(ax, summary, ctx, stat, interval=stat != "max")
        ax.set_title(labels[stat])
        ax.set_ylabel(ylabel)
    handles = _legend_handles(["pooled", *ctx.animals],
                              screened_incompletely=bool(summary.coverage_flag.any()))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Transfer above matched control"
        + (" relative to the receiver's prior" if scale == "prior" else ""),
        fontsize=15,
    )
    fig.text(
        0.5,
        0.01,
        caption
        + "\nThe maximum is descriptive. See summary CSV for degenerate intervals and support counts.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.91))
    fig.savefig(
        out_root / f"transfer_family_balanced_{scale}_percentiles.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def _direction_report(
    pairs: pd.DataFrame, root: Path, cells, teachers_by_animal
) -> pd.DataFrame:
    """Per re-pretraining direction x animal: pairs, alive/dead/unlabelled runs, controls /N."""
    rows = []
    notes: list[str] = []
    new = pairs[pairs.relation == tfp.RELATION_REPRETRAIN]
    for sender, receiver in tfp.REPRETRAIN_DIRECTIONS:
        receiver_dir = tfp.REPRETRAIN_BASE[receiver]
        pool = tio._probe_pool_for(receiver_dir, None)
        cut = tio._anchored_cut(root, receiver_dir, pool, dead="keep", notes=notes)
        for animal, teachers in teachers_by_animal.items():
            matrix = tio.load_cell_matrix(
                animal,
                tfp.REPRETRAIN_BASE[sender],
                receiver_dir,
                teachers,
                cells=list(cells),
                ftseed=1,
                output_root=root,
            )
            n_ctrl = (
                int(np.isfinite(matrix.values[:, matrix.control_col]).sum())
                if matrix.cells
                else 0
            )
            alive = dead = unlabelled = 0
            if matrix.cells:
                control_tree, sender_tree, _, _ = tio._arm_trees(
                    root, animal, tfp.REPRETRAIN_BASE[sender], receiver_dir
                )
                labels = tio._arm_dead_matrix(
                    matrix,
                    control_tree,
                    sender_tree,
                    dataset_tag=tio.DEFAULT_DATASET_TAG,
                    tseed=1,
                    ftseed=1,
                    probe_pool=pool,
                    cut=cut,
                    root=root,
                )[:, :-1]
                landed = np.isfinite(matrix.values[:, :-1])
                alive = int(((labels == 0.0) & landed).sum())
                dead = int(((labels == 1.0) & landed).sum())
                unlabelled = int((~np.isfinite(labels) & landed).sum())
            g = new[
                (new.sender == sender)
                & (new.receiver == receiver)
                & (new.animal == animal)
            ]
            rows.append(
                {
                    "direction": f"{sender}->{receiver}",
                    "animal": animal,
                    "n_pairs": len(g),
                    "alive": alive,
                    "dead": dead,
                    "unlabelled": unlabelled,
                    "controls": f"{n_ctrl}/{len(cells)}",
                    "filtered": cut is not None,
                }
            )
    return pd.DataFrame(rows)


@app.command()
def main(
    variants: str = typer.Option("all", help=f"one of {sorted(VARIANT_GROUPS)}"),
    out_dir: Path = typer.Option(DEFAULT_OUT, help="output directory, repo-relative"),
    n_bootstrap: int = typer.Option(2000, min=1),
    seed: int = typer.Option(0),
) -> None:
    from subliminal_learning.analysis import transfer_family_registry_io as registry

    if variants not in VARIANT_GROUPS:
        raise typer.BadParameter(f"--variants must be one of {sorted(VARIANT_GROUPS)}")
    repo, root = bootstrap()
    out_root = out_dir if out_dir.is_absolute() else repo / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    if out_root.resolve() == (repo / DEFAULT_OUT).resolve():
        _archive_existing(repo)
    cells = [
        c["student_tag"]
        for c in json.loads((repo / "run_logs/core30_cells.json").read_text())["cells"]
    ]
    teachers = _teachers(repo, tfp.DEFAULT_ANIMALS)
    print("Loading original matched pairs...", flush=True)
    pairs, base_ctx = tfp.all_pairs(
        repo, root, teachers_by_animal=teachers, cells=cells
    )
    print(
        f"Loaded {len(pairs):,} original rows; loading registry extension...",
        flush=True,
    )
    extra, registry_coverage = registry.registry_pairs(repo, root)
    registry_coverage.to_csv(out_root / "registry_coverage.csv", index=False)
    print(
        f"Loaded {len(extra):,} additional rows; resolving receiver priors...",
        flush=True,
    )
    combined = pd.concat([pairs, extra], ignore_index=True)
    combined, excluded = _figure_cohort(combined)
    excluded.to_csv(out_root / "figure_cohort_exclusions.csv", index=False)
    ctx = _expanded_context(base_ctx)
    frame, priors = _prior_frame(combined, root)
    priors.to_csv(out_root / "receiver_priors.csv", index=False)
    coverage = _coverage_table(frame)
    coverage.to_csv(out_root / "coverage.csv", index=False)
    for variant in VARIANT_GROUPS[variants]:
        if variant == "4spaces":
            fig = tfp.draw_four_spaces(pairs, base_ctx)
        elif variant == "linear":
            fig = tfp.draw_linear(pairs, base_ctx)
        else:
            fig = tfp.draw_symlog(
                pairs, base_ctx, floor=1e-2 if variant == "floor1e-2" else 1e-6
            )
        fig.savefig(out_root / OUT_NAMES[variant], dpi=160, bbox_inches="tight")
        plt.close(fig)
    for scale in ("raw", "prior"):
        print(
            f"Computing expanded {scale} summaries ({n_bootstrap} draws)...", flush=True
        )
        summary = _summary_table(
            frame, ctx, n_bootstrap=n_bootstrap, seed=seed, scale=scale
        )
        summary.to_csv(
            out_root / f"transfer_family_balanced_{scale}_summary.csv", index=False
        )
        _draw_summary(summary, ctx, out_root, scale=scale)
        print(
            summary[summary.group == "pooled"][
                [
                    "relation",
                    "animals_present",
                    "n_arms",
                    "n_pairs",
                    "n_prior_excluded_pairs",
                    "win",
                    "median",
                ]
            ].to_string(index=False),
            flush=True,
        )
    # Preserve the original four-category normalized boxplot population.
    normalized = frame[
        frame.relation.isin(base_ctx.relations) & frame.prior_available
    ].copy()
    normalized["test"] /= normalized.receiver_prior
    normalized["ctrl"] /= normalized.receiver_prior
    fig = tfp.draw_linear(normalized, base_ctx)
    fig.axes[0].set_ylabel(
        "[p(student) − p(control)] / p(receiver base)\nALIVE where anchored; otherwise unfiltered"
    )
    fig.suptitle(
        "Transfer above matched control, divided by each receiver's animal prior\nOriginal run-weighted boxes; IQR and 5–95% whiskers (distribution, not confidence intervals)",
        fontsize=11,
    )
    fig.savefig(
        out_root / "transfer_family_pooled_minus_control_prior_linear.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)
    _direction_report(pairs, root, cells, teachers).to_csv(
        out_root / "repretrain_direction_coverage.csv", index=False
    )
    print(f"Wrote expanded Figure 2 family to {out_root}", flush=True)


if __name__ == "__main__":
    app()
