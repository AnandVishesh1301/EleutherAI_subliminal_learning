"""Significance readout for one cross-seed transfer arm against its matched base controls.

Example::

    uv run python -m scripts.transfer_significance_readout \\
        --animal dog --sender fs1 --receiver original --eval-animals dog,cat,lion,pig \\
        --teachers run_logs/dog_top10_kband_configs.json

Read-only: every number comes from `analysis.transfer_significance_io` loaders and
`stats.transfer_significance_tests` estimators; nothing is written. The per-teacher sign and exceedance
tests are descriptive per-pool results, reported uncorrected; the existence claim rests on the
pooled stratified permutation, which needs no multiplicity correction.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import typer

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt

app = typer.Typer(add_completion=False)


def _report(
    animal: str,
    sender: str,
    receiver: str,
    eval_animal: str,
    teachers: list[str],
    prompt_set: str,
    n_perm: int,
    seed: int,
    **kw,
) -> None:
    """Print the per-teacher table and the pooled lines for one evaluated animal."""
    matrix = tio.load_cell_matrix(animal, sender, receiver, teachers,
                                  eval_animal=eval_animal, prompt_set=prompt_set, **kw)
    matrix, dropped = tio.drop_empty_pools(matrix)
    teachers = [t for t in teachers if t in matrix.pools]
    prior = tio.load_prior(eval_animal, receiver, prompt_set)
    proxy = " (validation proxy for the test prompt set)" if prompt_set == "test" else ""
    typer.echo(f"\n=== eval animal {eval_animal} | {len(matrix.cells)} cells | "
               f"prompt set {prompt_set} ===")
    typer.echo(f"prior p_target {prior:.4f}{proxy}")
    layouts = Counter(matrix.control_layout.values())
    typer.echo("control summary layout: "
               + ", ".join(f"{k} {v}" for k, v in sorted(layouts.items())))
    if matrix.no_control:
        typer.echo(f"cells without a control, dropped: {len(matrix.no_control)}")
    if dropped:
        typer.echo(f"teachers with no cell on the control cell set, skipped ({len(dropped)}): "
                   f"{', '.join(dropped)}")

    control = matrix.values[:, matrix.control_col]
    typer.echo(f"{'teacher':>24} {'wins/n':>9} {'sign p':>9} {'exc':>5} {'exc p':>9} "
               f"{'eff/prior':>10}")
    total_wins = total_n = 0
    for j, tag in enumerate(teachers):
        treatment = matrix.values[:, j]
        sign = tt.paired_sign_test(treatment, control)
        exc = tt.exceedance_test(treatment, control)
        effect = tt.prior_scaled_effect(treatment, control, prior)
        total_wins += sign.wins
        total_n += sign.n
        typer.echo(f"{tag:>24} {sign.wins:>4}/{sign.n:<4} {sign.p:>9.1e} {exc.count:>5} "
                   f"{exc.p:>9.1e} {effect:>10.4f}")

    perm = tt.stratified_exceedance_permutation(
        matrix.values, matrix.control_col, n_perm=n_perm, seed=seed
    )
    n_treatment = int(np.isfinite(matrix.values[:, :matrix.control_col]).sum())
    q = perm.null_quantiles
    typer.echo(f"pooled stratified permutation: observed {perm.observed} exceedances over "
               f"{n_treatment} treatment cells; null median {q['50']:.1f} / 95th {q['95']:.1f} / "
               f"99.9th {q['99.9']:.1f} / max {q['max']:.0f}; p {perm.p:.1e} "
               f"(floor {perm.floor:.1e}, n_perm {perm.n_perm})")
    win_rate = total_wins / total_n if total_n else float("nan")
    typer.echo(f"pooled descriptive win rate: {total_wins}/{total_n} = {win_rate:.4f}")
    if matrix.missing:
        typer.echo(f"missing cells: {len(matrix.missing)}")


@app.command()
def main(
    animal: str = typer.Option("dog", help="Animal the teachers were finetuned on."),
    sender: str = typer.Option("fs1", help="Sender (teacher) base token, e.g. fs1."),
    receiver: str = typer.Option("original", help="Receiver (student) base token."),
    eval_animals: str = typer.Option("dog,cat,lion,pig", help="Comma-separated animals to score."),
    teachers: str = typer.Option(
        "run_logs/dog_top10_kband_configs.json",
        help="Teacher param_tags: a comma list, a JSON manifest with a `param_tags` key, "
             "or 'all-on-disk'.",
    ),
    prompt_set: str = typer.Option("test", help="Prompt set read out of the summaries."),
    tseed: int = typer.Option(1, help="Teacher finetune seed in the tag dir (OLMo arms use 0)."),
    cells: str = typer.Option("", help="Cells JSON (a `cells` key) or a comma list restricting "
                                       "the matched design; empty = the control tree's cells."),
    control_source: str = typer.Option("base", help="Control arm: 'base' pool or 'rng' pool."),
    control_numbers_base: str = typer.Option(
        "", help="control-source 'base' only: token/dir whose control pool was distilled; "
                 "empty = the sender's base (cross-base control)."),
    n_perm: int = typer.Option(20000, help="Null draws for the stratified permutation."),
    seed: int = typer.Option(0, help="Seed of the permutation draws."),
) -> None:
    """Print sign / exceedance / stratified-permutation readouts for each evaluated animal."""
    spec: str | Path | list[str] = teachers
    if teachers != tio.ALL_ON_DISK and Path(teachers).suffix != ".json":
        spec = [t.strip() for t in teachers.split(",") if t.strip()]
    selected, stray = tio.resolve_teachers(spec, animal=animal, sender=sender, receiver=receiver,
                                           tseed=tseed)
    cell_spec: str | list[str] | None = None
    if cells:
        cell_spec = cells if Path(cells).suffix == ".json" else \
            [c.strip() for c in cells.split(",") if c.strip()]
    typer.echo(f"{animal}: sender {sender} -> receiver {receiver}; {len(selected)} teachers")
    if stray:
        typer.echo(f"stray on-disk teacher dirs excluded ({len(stray)}): {', '.join(stray)}")
    for eval_animal in [a.strip() for a in eval_animals.split(",") if a.strip()]:
        _report(animal, sender, receiver, eval_animal, selected, prompt_set, n_perm, seed,
                tseed=tseed, cells=cell_spec, control_source=control_source,
                control_numbers_base=control_numbers_base or None)


if __name__ == "__main__":
    app()
