"""Predictor table across receivers at one shared student recipe.

Run with ``uv run python -m notebooks.figures.predictors.predictor_r2_receivers``.

Companion to ``predictor_r2_table`` (which averages eight recipes but covers only the original
and data-seed-1 receivers). Every receiver here holds the same 102 original Pythia-160M dog
teachers at the single recipe ``ep20_lr1e-4_r8a32`` (mean over the finetuning seeds present at
the cell: one for the original receiver, one or three for the siblings). Full-seed receivers (different
init and data order; ten teachers) and the Pythia-410M receivers (34 teachers) are omitted:
every teacher distilled into them has validation p(dog) >= 0.99, so there is no
teacher-preference variation to remove.

Transfer is student test-set p(dog) at that recipe. Predictors, control and statistics are
those of ``predictor_r2_table`` (marginal R^2, increment beyond teacher validation p(dog),
squared partial correlation), on the linear and rank scales. Steerability is the screen whose
receiver is the column's receiver: the self-screen for orig, and for every sibling the
cross-steering screen in which the teacher's vector is injected into that sibling's base.
The fixed-cell transfer values are frozen to
``fixed_cell_transfer_102.csv`` so the table rebuilds without the outputs root.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

from subliminal_learning.analysis.frames import load_cell_summaries
from subliminal_learning.config import resolve_output_root

from notebooks.figures.predictors.predictor_r2_table import CSV, OUT, STEER_COLS, TEX_OUT, load as load_predictors, load_steer, r2, residual

FROZEN = OUT / "fixed_cell_transfer_102.csv"
RECIPE = "ep20_lr1e-4_r8a32"
RECEIVERS = [  # key, student base dir, recipe, display, pretraining relation to the sender
    ("orig", "pythia-160m", RECIPE, "orig", "same model"),
    ("ds1", "pythia-160m-data-seed1", RECIPE, "ds1", "same init, diff data order"),
    ("ds2", "pythia-160m-data-seed2", RECIPE, "ds2", "same init, diff data order"),
    ("ds3", "pythia-160m-data-seed3", RECIPE, "ds3", "same init, diff data order"),
    ("ws1", "pythia-160m-weight-seed1", RECIPE, "ws1", "released weight-seed"),
    ("ws2", "pythia-160m-weight-seed2", RECIPE, "ws2", "released weight-seed"),
    ("ws3", "pythia-160m-weight-seed3", RECIPE, "ws3", "released weight-seed"),
    # Pythia-410M / 410M-seed-1 hold 34 teachers at ep20_lr9e-5_r8a32, all with validation
    # p(dog) >= 0.996; like the full-seed receivers (10 saturated teachers) they leave no
    # teacher-preference variation to remove, so they are not tabulated.
]
PREDICTORS = [("Teacher excess CE", "dce", "positive"), ("Mean sequence length", "length", "positive"), ("Steerability", "steer", "delta"), ("Same-model transfer", "tr_orig_cell", "probability")]
SCALES = {"linear": lambda v: np.asarray(v, dtype=float), "rank": lambda v: rankdata(v)}


def collect(frame):
    """Read the fixed-cell test p(dog) per (teacher, receiver) from the outputs root."""
    output_root = resolve_output_root()
    rows = []
    for key, base, recipe, _, _ in RECEIVERS:
        rels = {t: f"students/dog/{base}/teacher-pythia-160m/{t}_tseed1/ex10-20_prefix_n100k/{recipe}" for t in frame.teacher}
        docs = load_cell_summaries(output_root, list(rels.values()))
        for teacher, rel in rels.items():
            doc = docs.get(rel)
            if doc is None:
                continue
            test = [s for s in doc["summaries"] if s["prompt_set"] == "test" and s["scoring"] == "prob"]
            assert len(test) == 1, rel
            rows.append(dict(teacher=teacher, receiver=key, recipe=recipe, n_seeds=int(test[0]["n_seeds"]), test_p=float(test[0]["p_target_mean"])))
    return pd.DataFrame(rows)


def load_fixed(frame, refresh=False):
    if refresh or not FROZEN.exists():
        collect(frame).to_csv(FROZEN, index=False)
    cells = pd.read_csv(FROZEN)
    wide = cells.pivot(index="teacher", columns="receiver", values="test_p")
    merged = frame.merge(wide, left_on="teacher", right_index=True, how="left")
    merged["tr_orig_cell"] = merged["orig"]
    assert merged.orig.notna().all()
    return merged


def build(merged):
    out = []
    for key, _, _, display, relation in RECEIVERS:
        sub = merged.dropna(subset=[key])
        for scale, fmap in SCALES.items():
            y = fmap(sub[key])
            z = fmap(sub.teacher_val_p)
            control = r2(y, z)
            base = dict(receiver=key, display=display, relation=relation, n=len(sub), scale=scale, control_r2=control)
            out.append(dict(base, predictor="Teacher validation p(dog)", marginal_r2=control, incremental_r2=np.nan, partial_r2=np.nan, partial_r=np.nan, marginal_r=pearsonr(z, y).statistic))
            for name, col, kind in PREDICTORS:
                if key == "orig" and col == "tr_orig_cell":
                    continue
                if kind == "delta":
                    col = STEER_COLS[key]  # the screen into this column's receiver
                raw = sub[col].to_numpy(dtype=float)
                x = fmap(raw) if kind != "delta" or scale == "rank" else raw
                joint = r2(y, z, x)
                pr = float(pearsonr(residual(x, z), residual(y, z)).statistic)
                np.testing.assert_allclose(pr**2, (joint - control) / (1 - control), atol=1e-9)
                out.append(dict(base, predictor=name, marginal_r2=r2(y, x), incremental_r2=joint - control, partial_r2=pr**2, partial_r=pr, marginal_r=pearsonr(x, y).statistic))
    return pd.DataFrame(out)


def to_tex(table, scale):
    cols = [r for r in RECEIVERS]
    head = " & ".join(d for _, _, _, d, _ in cols)  # plain text so the column spec (r) aligns it with the numbers
    groups, rules = [], []
    for i, (_, _, _, _, relation) in enumerate(cols, start=2):
        if groups and groups[-1][0] == relation:
            groups[-1][2] = i
        else:
            groups.append([relation, i, i])
    group_row = " & ".join(f"\\multicolumn{{{b - a + 1}}}{{c}}{{{rel}}}" for rel, a, b in groups)
    rules = "".join(f"\\cmidrule(lr){{{a}-{b}}}" for _, a, b in groups)
    lines = [f"\\begin{{tabular}}{{l{'r' * len(cols)}}}", "\\toprule", f" & {group_row}\\\\", rules, f"Receiver & {head}\\\\"]
    ns = " & ".join(str(int(table[(table.receiver == k) & (table.scale == scale)].n.iloc[0])) for k, *_ in cols)
    lines.append(f"Teachers ($n$) & {ns}\\\\")
    sub = table[table.scale == scale]

    def row(label, predictor, column, fmt):
        vals = []
        for k, *_ in cols:
            r = sub[(sub.receiver == k) & (sub.predictor == predictor)]
            vals.append("--" if r.empty or pd.isna(r[column].iloc[0]) else fmt(r[column].iloc[0]))
        return f"\\quad {label} & " + " & ".join(vals) + "\\\\"

    pct = lambda v: f"{100 * v:.1f}"
    sgn = lambda v: f"{v:+.2f}"
    lines += ["\\midrule", f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{Marginal $R^2$ (\\%)}}\\\\"]
    lines.append(row("Teacher validation p(dog)", "Teacher validation p(dog)", "marginal_r2", pct))
    for name, *_ in PREDICTORS:
        lines.append(row(name, name, "marginal_r2", pct))
    lines += ["\\midrule", f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{$\\Delta R^2$ beyond teacher preference (pp)}}\\\\"]
    for name, *_ in PREDICTORS:
        lines.append(row(name, name, "incremental_r2", pct))
    lines += ["\\midrule", f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{Partial $r$ given teacher preference}}\\\\"]
    for name, *_ in PREDICTORS:
        lines.append(row(name, name, "partial_r", sgn))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    merged = load_fixed(load_steer(load_predictors()))
    table = build(merged)
    table.to_csv(OUT / "predictor_r2_receivers.csv", index=False)
    for scale in SCALES:
        (TEX_OUT / f"predictor_r2_receivers_{scale}.tex").write_text(to_tex(table, scale))
    pd.set_option("display.width", 250)
    show = table[table.scale == "rank"].pivot(index="predictor", columns="display", values="incremental_r2")
    print("rank-scale ΔR² beyond teacher p (pp):")
    print((100 * show).round(1).to_string())
    print(table[table.scale == "rank"].groupby("display")[["n", "control_r2"]].first().to_string())


if __name__ == "__main__":
    main()
