"""Is transfer symmetric? The A -> B versus B -> A readout, per model pair and per model.

Runs `analysis.transfer_asymmetry` over the five families (cross-seed, cross-data-order,
re-pretrain, olmo-recipe, olmo-seed) and writes the four tables plus the LaTeX renderings::

    uv run python -m scripts.transfer_asymmetry_readout
    uv run python -m scripts.transfer_asymmetry_readout --families re-pretrain,olmo-seed
    uv run python -m scripts.transfer_asymmetry_readout --value delta

Outputs go to ``{output_root}/analysis/transfer_asymmetry/``: ``pairs.csv`` (one row per
complete pair), ``roles.csv`` (one row per model's inbound-against-outbound contrast),
``census.csv`` (every ordered arm at every design, with its completeness), ``excluded.csv`` (the
pairs no design finished in both directions, with the per-direction reason) and ``results.json``
(all four tables plus the run's settings, the value mode among them). The two LaTeX tables go to
``--latex-dir``. ``--value delta`` reads the SAME runs in percentage points of trait preference
instead of receiver-prior units and writes to the ``*_delta`` siblings, so the two modes never
overwrite each other; the census and the exclusions are value-independent and are written once.

Read-only on the output tree apart from those files and the loader cache: every number comes
from `transfer_asymmetry`, whose loads are `tio.load_cell_matrix` with pinned cells and whose
tests are the memoized comparison permutations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from subliminal_learning.analysis import transfer_asymmetry as ta
from subliminal_learning.config import resolve_output_root

app = typer.Typer(add_completion=False)

REPO = Path(__file__).resolve().parents[1]


@app.command()
def main(
    families: str = typer.Option(",".join(ta.FAMILIES),
                                 help="Comma-separated families to read; default = all."),
    n_perm: int = typer.Option(20000, help="Monte-Carlo draws when a test cannot be exact."),
    seed: int = typer.Option(0, help="Seed of the permutation draws."),
    value: str = typer.Option(ta.VALUE_PRIOR, help="Run value: 'prior' ((student - control) / "
                                                  "receiver test prior) or 'delta' (the gap in "
                                                  "percentage points, no prior). 'delta' writes "
                                                  "the *_delta siblings."),
    out_dir: str = typer.Option("", help="Where the CSV/JSON tables go; empty = "
                                        "{output_root}/analysis/transfer_asymmetry/."),
    latex_dir: str = typer.Option("", help="Where the two .tex tables go; empty = "
                                          "{repo}/docs/results/latex/."),
    output_root: str = typer.Option("", help="Output tree root; empty = the paths YAML's."),
) -> None:
    """Write the pair, role, census and exclusion tables of the transfer-asymmetry readout."""
    names = [n.strip() for n in families.split(",") if n.strip()]
    unknown = [n for n in names if n not in ta.FAMILIES]
    if unknown:
        raise typer.BadParameter(f"unknown families {unknown}; known: {sorted(ta.FAMILIES)}")
    if value not in ta.VALUE_MODES:
        raise typer.BadParameter(f"value must be one of {ta.VALUE_MODES}, got {value!r}")
    root = resolve_output_root(Path(output_root) if output_root else None)
    out = Path(out_dir) if out_dir else root / "analysis" / "transfer_asymmetry"
    tex = Path(latex_dir) if latex_dir else REPO / "docs" / "results" / "latex"
    out.mkdir(parents=True, exist_ok=True)
    tex.mkdir(parents=True, exist_ok=True)

    result = ta.analyze(names, repo=REPO, output_root=root, n_perm=n_perm, seed=seed, value=value)
    # the census and the exclusions do not depend on the value mode, so only the two value
    # tables get a mode suffix; nothing a previous mode wrote is overwritten
    suffix = "" if value == ta.VALUE_PRIOR else f"_{value}"
    tables = {f"pairs{suffix}": result.pairs, f"roles{suffix}": result.roles,
              "census": result.census, "excluded": result.excluded}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        typer.echo(f"\n=== {name} ({len(df)} rows) ===")
        typer.echo(df.to_string(index=False))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    (tex / f"transfer_asymmetry_pairs{suffix}_{stamp}.tex").write_text(
        ta.pairs_table_latex(result.pairs, value=value))
    (tex / f"transfer_asymmetry_roles{suffix}_{stamp}.tex").write_text(
        ta.roles_table_latex(result.roles, value=value))
    (out / f"results{suffix}.json").write_text(json.dumps(
        {"families": names, "n_perm": n_perm, "seed": seed, "output_root": str(root),
         "value": value, "value_units": ta.VALUE_UNITS[value],
         "timestamp": datetime.now(timezone.utc).isoformat(),
         **{k: json.loads(v.to_json(orient="records")) for k, v in tables.items()}}, indent=2))
    typer.echo(f"\nwrote {out}/ and {tex}/transfer_asymmetry_*{suffix}_{stamp}.tex")


if __name__ == "__main__":
    app()
