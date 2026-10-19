"""Variance explained by each transfer predictor before and after removing teacher preference.

Run with ``uv run python -m notebooks.figures.predictors.predictor_r2_table``.

Population: the frozen 102 original Pythia-160M dog teachers of
``notebooks/figures/teacher_dce_vs_transfer_102.csv`` (teacher validation p, excess CE,
steerability), joined to each teacher's mean integer count from the sequence-length snapshot.
Transfer is read per student cell (one recipe; mean over the finetuning seeds present at
that cell, one for the original receiver and one to three for data-seed-1) for the original and
data-seed-1 receivers at each of the eight recipes every teacher shares, frozen to
``recipe_cells_102.csv``; nothing is averaged over recipes or epochs. The LaTeX fragment
shows ``TEX_RECIPE`` (the recipe shared by every receiver in ``predictor_r2_receivers``);
the CSV carries every recipe plus the eight-recipe mean (``mean8``) that the earlier
appendix text and the steerability figure were computed on.

For every outcome (same-model transfer, cross-model transfer into data-seed-1),
scale (linear, log, logit, rank; the LaTeX fragment shows linear and rank) and predictor, three OLS fits are compared:
transfer on the predictor alone (marginal R^2), transfer on teacher validation
p(dog) alone (control R^2), and transfer on both (joint R^2). The increment
joint - control is the share of transfer variance the predictor adds beyond teacher
preference; dividing it by 1 - control gives the squared partial correlation, the
share of the variance *left unexplained by teacher preference* that the predictor
recovers. By Frisch-Waugh-Lovell this equals the squared correlation between the
two sets of residuals after regressing predictor and outcome on teacher preference,
which is what the appendix residual panels plot. The rank column reproduces the
partial Spearman coefficients quoted in the appendix text.

Transforms: probabilities (transfer, teacher p, same-model transfer as a predictor)
take the scale's map (identity, log, logit with 1e-6 clipping, or ranks); excess CE
and sequence length are log-transformed on the log and logit scales and ranked on
the rank scale; the steerability delta (a difference of probabilities) stays raw
except on the rank scale. Steerability is the screen whose receiver is the student's receiver:
the teacher's self-screen on the original model for same-model transfer, and the cross-steering
screen (the teacher's vector injected into the data-seed-1 base, ``steer_data-seed1``, run
2026-10-18 for all 102 teachers) for transfer into data-seed-1; both are the ungated increase in
validation p(dog) under steering, frozen to ``steer_screens_102.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from subliminal_learning.analysis.frames import load_cell_summaries
from subliminal_learning.config import resolve_output_root

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "notebooks/figures/teacher_dce_vs_transfer_102.csv"
LENGTH_ROWS = ROOT / "docs/results/data/sequence_length_transfer_20260924_teacher_rows.csv"
OUT = ROOT / "notebooks/figures/predictors"
TEX_OUT = ROOT / "docs/drafts"  # LaTeX fragments land beside the manuscript drafts
TEX_SCALES = ("linear", "rank")  # the CSV keeps all four scales; the LaTeX fragment shows these
TEX_RECIPE = "ep20_lr1e-4_r8a32"
RECIPES = ["ep1_lr1e-4_r8a32", "ep2_lr1e-4_r8a32", "ep5_lr1e-4_r8a32", "ep10_lr1e-4_r8a32", "ep20_lr1e-4_r8a32", "ep20_lr3e-4_r16a8", "ep20_lr4e-4_r16a4", "ep20_lr4e-4_r32a4"]
CELLS = OUT / "recipe_cells_102.csv"
STEER = OUT / "steer_screens_102.csv"
STEER_COLS = {"orig": "steer_ungated", "ds1": "steer_ds1"}  # screen receiver matched to the student's receiver
RECEIVER_DIRS = {"orig": "pythia-160m", "ds1": "pythia-160m-data-seed1"}

OUTCOMES = [("Same-model transfer (original receiver)", "orig"), ("Cross-model transfer (data-seed-1 receiver)", "ds1")]
PREDICTORS = [
    ("Teacher excess CE", "dce", "positive"),
    ("Mean sequence length", "length", "positive"),
    ("Steerability", "steer", "delta"),  # column resolved per outcome through STEER_COLS
    ("Same-model transfer", "orig", "probability"),
]


def collect_steer(frame):
    """Ungated steering deltas per teacher from the self and data-seed-1 screen aggregates."""
    import json

    root = resolve_output_root() / "teachers/dog/pythia-160m"
    rows = []
    for t in frame.teacher:
        self_doc = json.load(open(root / t / "steerability/self/aggregate.json"))
        ds1_doc = json.load(open(root / t / "steerability/steer_data-seed1/aggregate.json"))
        assert ds1_doc["steer_model"] == "data-seed1" and ds1_doc["n_seeds"] == 1
        rows.append(dict(teacher=t, steer_self=self_doc["peaks"]["ungated"]["delta_val_p_mean"], steer_ds1=ds1_doc["peaks"]["ungated"]["delta_val_p_mean"], ds1_base_val_p=ds1_doc["base_val_p"]))
    return pd.DataFrame(rows)


def load_steer(frame, refresh=False):
    if refresh or not STEER.exists():
        collect_steer(frame).to_csv(STEER, index=False)
    steer = pd.read_csv(STEER)
    merged = frame.merge(steer, on="teacher", how="left", validate="one_to_one")
    assert merged.steer_ds1.notna().all() and len(merged) == 102
    np.testing.assert_allclose(merged.steer_self, merged.steer_ungated, atol=1e-12)  # frozen column = self screen
    return merged


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


SCALES = {
    "linear": (lambda p: np.asarray(p, dtype=float), lambda q: np.asarray(q, dtype=float)),
    "log": (lambda p: np.log(np.asarray(p, dtype=float)), lambda q: np.log(np.asarray(q, dtype=float))),
    "logit": (logit, lambda q: np.log(np.asarray(q, dtype=float))),
    "rank": (lambda p: rankdata(p), lambda q: rankdata(q)),
}


def load():
    frame = pd.read_csv(CSV)
    assert len(frame) == 102 and frame.teacher.nunique() == 102
    assert frame.n_orig.eq(8).all() and frame.n_ds1.eq(8).all()
    rows = pd.read_csv(LENGTH_ROWS)
    rows = rows[(rows.animal == "dog") & (rows.model == "pythia-160m") & (rows.dataset == "ex10-20_prefix_n100k")]
    length = rows.groupby("teacher_recipe").agg(
        length=("mean_integer_count", "first"),
        n_len=("mean_integer_count", "nunique"),
        val_p=("teacher_validation_p", "first"),
        ce=("teacher_excess_ce", "first"),
    )
    assert length.n_len.eq(1).all()
    frame = frame.merge(length, left_on="teacher", right_index=True, how="left")
    assert frame.length.notna().all()
    np.testing.assert_allclose(frame.teacher_val_p, frame.val_p)
    np.testing.assert_allclose(frame.dce, frame.ce, rtol=1e-6)
    return frame


def collect_cells(frame):
    """Per-cell test p(dog) (seed 1) for orig and ds1 at every shared recipe, from the outputs root."""
    output_root = resolve_output_root()
    rels = {(t, k, r): f"students/dog/{d}/teacher-pythia-160m/{t}_tseed1/ex10-20_prefix_n100k/{r}" for t in frame.teacher for k, d in RECEIVER_DIRS.items() for r in RECIPES}
    docs = load_cell_summaries(output_root, list(rels.values()))
    rows = []
    for (t, k, r), rel in rels.items():
        test = [x for x in docs[rel]["summaries"] if x["prompt_set"] == "test" and x["scoring"] == "prob"]
        assert len(test) == 1, rel
        rows.append(dict(teacher=t, receiver=k, recipe=r, n_seeds=int(test[0]["n_seeds"]), test_p=float(test[0]["p_target_mean"])))
    return pd.DataFrame(rows)


def load_cells(frame, refresh=False):
    if refresh or not CELLS.exists():
        collect_cells(frame).to_csv(CELLS, index=False)
    cells = pd.read_csv(CELLS)
    assert len(cells) == 102 * len(RECIPES) * len(RECEIVER_DIRS) and cells.test_p.notna().all()
    mean8 = cells.groupby(["teacher", "receiver"], as_index=False).test_p.mean().assign(recipe="mean8")
    np.testing.assert_allclose(mean8[mean8.receiver == "orig"].set_index("teacher").test_p.loc[frame.teacher], frame.tr_orig, atol=1e-9)
    np.testing.assert_allclose(mean8[mean8.receiver == "ds1"].set_index("teacher").test_p.loc[frame.teacher], frame.tr_ds1, atol=1e-9)
    return pd.concat([cells, mean8], ignore_index=True)


def frame_for(frame, cells, recipe):
    wide = cells[cells.recipe == recipe].pivot(index="teacher", columns="receiver", values="test_p")
    return frame.merge(wide, left_on="teacher", right_index=True, how="left", validate="one_to_one")


def r2(y, *xs):
    design = np.column_stack([np.ones(len(y)), *xs])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(1 - np.var(y - design @ beta) / np.var(y))


def residual(v, z):
    design = np.column_stack([np.ones(len(z)), z])
    return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]


def build(frame, cells):
    out = []
    for recipe in RECIPES + ["mean8"]:
        out += build_recipe(frame_for(frame, cells, recipe), recipe)
    return pd.DataFrame(out)


def build_recipe(frame, recipe):
    out = []
    for outcome, ycol in OUTCOMES:
        for scale, (prob_map, pos_map) in SCALES.items():
            y = prob_map(frame[ycol])
            z = prob_map(frame.teacher_val_p)
            control = r2(y, z)
            out.append(dict(recipe=recipe, outcome=outcome, scale=scale, predictor="Teacher validation p(dog)", marginal_r2=control, control_r2=control, incremental_r2=np.nan, partial_r2=np.nan, partial_r=np.nan, marginal_r=pearsonr(z, y).statistic))
            for name, col, kind in PREDICTORS:
                if col == ycol:
                    continue
                if kind == "delta":
                    col = STEER_COLS[ycol]
                raw = frame[col].to_numpy(dtype=float)
                if kind == "probability":
                    x = prob_map(raw)
                elif kind == "positive":
                    x = pos_map(raw)
                else:
                    x = rankdata(raw) if scale == "rank" else raw
                joint = r2(y, z, x)
                partial_r = float(pearsonr(residual(x, z), residual(y, z)).statistic)
                np.testing.assert_allclose(partial_r**2, (joint - control) / (1 - control), atol=1e-9)
                out.append(dict(recipe=recipe, outcome=outcome, scale=scale, predictor=name, marginal_r2=r2(y, x), control_r2=control, incremental_r2=joint - control, partial_r2=partial_r**2, partial_r=partial_r, marginal_r=pearsonr(x, y).statistic))
    return out


def to_tex(table, recipe=TEX_RECIPE):
    lines = []
    for outcome, _ in OUTCOMES:
        sub = table[(table.outcome == outcome) & (table.recipe == recipe)]
        lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\textit{{{outcome}}}}}\\\\"]
        for scale in TEX_SCALES:
            block = sub[sub.scale == scale]
            control = block.control_r2.iloc[0]
            lines.append(f"\\addlinespace\\multicolumn{{4}}{{l}}{{{scale.capitalize()} scale; teacher preference alone $R^2={100*control:.1f}\\%$}}\\\\")
            for _, row in block[block.predictor != "Teacher validation p(dog)"].iterrows():
                lines.append(
                    f"\\quad {row.predictor} & {100*row.marginal_r2:.1f} & {100*row.incremental_r2:.1f} & "
                    f"{row.partial_r:+.2f}\\\\"
                )
    body = "\n".join(lines)
    return (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Predictor & Marginal $R^2$ (\\%) & $\\Delta R^2$ beyond teacher $p$ (pp) & Partial $r$\\\\\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def main():
    frame = load_steer(load())
    cells = load_cells(frame)
    table = build(frame, cells)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "predictor_r2_table.csv", index=False)
    (TEX_OUT / "predictor_r2_table.tex").write_text(to_tex(table))
    # Eight-recipe-mean values quoted in the pre-revision appendix text, computed independently.
    m8 = table[table.recipe == "mean8"]
    rank_same = m8[(m8.scale == "rank") & (m8.outcome == OUTCOMES[0][0])].set_index("predictor").partial_r
    rank_cross = m8[(m8.scale == "rank") & (m8.outcome == OUTCOMES[1][0])].set_index("predictor").partial_r
    np.testing.assert_allclose(
        [rank_same["Steerability"], rank_same["Teacher excess CE"], rank_same["Mean sequence length"], rank_cross["Same-model transfer"]],
        [-0.15093320138019922, 0.656, -0.693, 0.3100685897911577],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        [spearmanr(frame.teacher_val_p, frame.tr_orig).statistic, spearmanr(frame.length, frame.tr_orig).statistic],
        [0.951, -0.963],
        atol=5e-4,
    )
    pd.set_option("display.width", 220)
    show = table[(table.scale == "rank") & (table.predictor != "Teacher validation p(dog)")]
    print("rank-scale ΔR² (pp) by recipe:")
    print((100 * show.pivot_table(index=["outcome", "predictor"], columns="recipe", values="incremental_r2")[RECIPES + ["mean8"]]).round(1).to_string())
    print("rank-scale teacher p alone R² (%) by recipe:")
    print((100 * table[(table.scale == "rank") & (table.predictor == "Teacher validation p(dog)")].pivot(index="outcome", columns="recipe", values="control_r2")[RECIPES + ["mean8"]]).round(1).to_string())


if __name__ == "__main__":
    main()
