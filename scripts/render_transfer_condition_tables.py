"""CLI: re-render the four transfer-condition LaTeX tables from the notebook's pickled frames.

`notebooks/transfer_conditions.ipynb` computes the existence table, the merged comparison table
and the direction table (permutations over the whole output tree, minutes of work) and pickles
the frames next to the ``.tex`` it writes. This script is the COSMETIC path: it loads those
pickles, calls `analysis.transfer_conditions.condition_table_latex` / `comparison_table_latex` /
`direction_table_latex` and rewrites the four ``.tex`` files (the comparison pickle
feeds the Pythia and the OLMo comparison tables) in seconds. Nothing is read from the run tree and no permutation is run, so a
caption / header / column-width edit needs no notebook execution -- a DATA refresh still does.

Prints the widest single-line body cell per text column of each table (the width the LaTeX has to
fit) and exits non-zero when a pickle is missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from subliminal_learning.analysis import setup
from subliminal_learning.analysis import transfer_conditions as tc

app = typer.Typer(add_completion=False)

#: The stems of the four tables this script writes.
EXISTENCE = "transfer_conditions_table"
COMPARISON = "transfer_condition_comparisons_table"
COMPARISON_OLMO = "transfer_condition_comparisons_olmo_table"     # from the COMPARISON pickle
DIRECTION = "transfer_direction_table"

#: The text columns whose widest single-line cell is reported, per table.
WIDTH_COLUMNS = {EXISTENCE: ["Model Family", "Pretraining Condition", "Arms"],
                 COMPARISON: ["Comparison", "Arms A", "Arms B"],
                 COMPARISON_OLMO: ["Comparison", "Arms A", "Arms B"],
                 DIRECTION: ["Model Family", "Comparison"]}


def load_frame(path: Path) -> pd.DataFrame:
    """The pickled table frame at `path`.

    Raises:
        typer.Exit: If the pickle is missing -- only the notebook writes it.
    """
    if not path.is_file():
        typer.echo(f"missing {path}: run notebooks/transfer_conditions.ipynb first "
                   f"(it pickles the frames next to the .tex)", err=True)
        raise typer.Exit(code=1)
    return pd.read_pickle(path)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """A loaded comparison frame with the retired written-out treatment effect
    (`analysis.transfer_conditions.EFFECT_LONG`) replaced by the plain one, so an older pickle
    renders the way a fresh frame does."""
    if "Treatment Effect Mode" not in df.columns:
        return df
    df = df.copy()
    df["Treatment Effect Mode"] = df["Treatment Effect Mode"].replace(tc.EFFECT_LONG,
                                                                     tc.EFFECT_SHORT)
    return df


def widest_cells(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[str, int]]:
    """``{column: (widest single-line part, its character count)}`` over the parts the LaTeX
    actually puts on one row (`analysis.transfer_conditions._PART_SPLITTERS` where the column has
    a splitter, the raw cell otherwise)."""
    out = {}
    for column in columns:
        if column not in df.columns:
            continue
        split = tc._PART_SPLITTERS.get(column, lambda v: [str(v)])
        parts = [p for v in df[column].dropna().unique() for p in split(v)]
        out[column] = max(((p, len(p)) for p in parts), key=lambda x: x[1], default=("", 0))
    return out


@app.command()
def main(latex_dir: Path = typer.Option(None, help="Where the pickles live and the .tex go "
                                                   "(default: <repo>/notebooks/latex)")) -> None:
    """Re-render the four tables from their pickles."""
    repo = setup.find_root()
    latex_dir = latex_dir or repo / "notebooks/latex"
    for stem, source, render in (
            (EXISTENCE, EXISTENCE, tc.condition_table_latex),
            (COMPARISON, COMPARISON,
             lambda df: tc.comparison_table_latex(df, family="Pythia",
                                                  animals=tc.COMPARISON_PYTHIA_ANIMALS)),
            (COMPARISON_OLMO, COMPARISON, tc.comparison_olmo_table_latex),
            (DIRECTION, DIRECTION, tc.direction_table_latex)):
        df = normalise(load_frame(latex_dir / f"{source}.pkl"))
        (latex_dir / f"{stem}.tex").write_text(render(df))
        typer.echo(f"{stem}.tex  ({len(df)} rows)")
        for column, (text, width) in widest_cells(df, WIDTH_COLUMNS[stem]).items():
            typer.echo(f"    widest {column}: {width} chars  {text!r}")


if __name__ == "__main__":
    app()
