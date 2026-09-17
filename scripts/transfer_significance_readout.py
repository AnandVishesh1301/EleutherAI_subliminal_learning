"""Significance readout for one cross-seed transfer arm against its matched base controls.

Example::

    uv run python -m scripts.transfer_significance_readout \\
        --animal dog --sender fs1 --receiver original --eval-animals dog,cat,lion,pig \\
        --teachers run_logs/dog_top10_kband_configs.json

The ``--family`` mode instead loads every directed off-diagonal arm of a seed family and runs
ONE arm-stratified permutation over them, so a trait gets a single existence claim::

    uv run python -m scripts.transfer_significance_readout --family full-seed --animal dog \\
        --teachers run_logs/dog_top10_kband_configs.json

The ``--compare`` mode asks whether one family transfers MORE than another, comparing arm
SETS: the two families' arms are grouped by transfer role and compared role by role, on
control- and prior-normalized effects (arm-level role-stratified permutation + cell-stratified
permutation)::

    uv run python -m scripts.transfer_significance_readout --compare data-seed:full-seed \\
        --animal dog --teachers run_logs/dog_top10_kband_configs.json

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
        ok = np.isfinite(treatment) & np.isfinite(control)
        sign = tt.paired_sign_test(treatment, control)
        # primary test, on the matched design: only the cells this teacher actually covers
        exc = tt.exceedance_test(treatment[ok], control[ok])
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


#: The base tokens of each seed family; arms are every ordered off-diagonal pair.
FAMILY_BASES = {"full-seed": ["orig", "fs1", "fs2", "fs3"],
                "data-seed": ["orig", "ds1", "ds2", "ds3"]}


def _family_report(
    family: str,
    animal: str,
    eval_animal: str,
    teachers: list[str],
    prompt_set: str,
    n_perm: int,
    seed: int,
    **kw,
) -> None:
    """Print the per-arm table and the family-level arm-stratified permutation line."""
    bases = FAMILY_BASES[family]
    matrices, skipped = {}, []
    for sender in bases:
        for receiver in bases:
            if sender == receiver:
                continue
            arm = f"{sender}->{receiver}"
            matrix = tio.load_cell_matrix(animal, sender, receiver, teachers,
                                          eval_animal=eval_animal, prompt_set=prompt_set, **kw)
            if not matrix.cells:
                skipped.append(f"{arm} (no control)")
                continue
            treatment = np.delete(matrix.values, matrix.control_col, axis=1)
            if not np.isfinite(treatment).any():
                skipped.append(f"{arm} (no treatment)")
                continue
            matrices[arm] = matrix
    typer.echo(f"\n=== family {family} | {animal} teachers | eval animal {eval_animal} | "
               f"prompt set {prompt_set} ===")
    if skipped:
        typer.echo(f"arms skipped ({len(skipped)}): {', '.join(skipped)}")
    if not matrices:
        typer.echo("no usable arm in this family; nothing to test")
        return

    stacked = tio.stack_cell_matrices(matrices)
    perm = tt.stratified_exceedance_permutation(
        stacked.values, stacked.control_index, groups=stacked.groups,
        n_perm=n_perm, seed=seed,
    )
    typer.echo(f"{'arm':>16} {'cells':>6} {'treatment':>10} {'exceed':>7}")
    for arm in stacked.arms:
        typer.echo(f"{arm:>16} {len(matrices[arm].cells):>6} {stacked.n_treatment[arm]:>10} "
                   f"{perm.observed_by_group[arm]:>7}")
    q = perm.null_quantiles
    total_treatment = sum(stacked.n_treatment.values())
    typer.echo(f"family arm-stratified permutation over {len(stacked.arms)} arms: observed "
               f"{perm.observed} exceedances over {total_treatment} treatment cells; null median "
               f"{q['50']:.1f} / 95th {q['95']:.1f} / 99.9th {q['99.9']:.1f} / max {q['max']:.0f}; "
               f"p {perm.p:.1e} (floor {perm.floor:.1e}, n_perm {perm.n_perm})")


def _mode_block(mode: tio.ModeComparison, what: str) -> None:
    """Print one value mode's arm table, role table, and its four test lines."""
    typer.echo(f"\n--- value mode {mode.mode}: {what} ---")
    typer.echo(f"{'role':>16} {'family':>10} {'arm':>12} {'cells':>6} {'students':>9} "
               f"{'mean eff':>9} {'exc frac':>9}")
    for r in mode.arms.to_dict("records"):
        typer.echo(f"{r['role']:>16} {r['family']:>10} {r['arm']:>12} {r['cells']:>6} "
                   f"{r['students']:>9} {r['mean effect']:>9.3f} {r['exceed. frac']:>9.2f}")

    typer.echo(f"\n{'role':>16} {'arms A':>7} {'arms B':>7} {'cells':>6} {'mean A':>9} "
               f"{'mean B':>9} {'contrast':>9}")
    for r in mode.roles.to_dict("records"):
        typer.echo(f"{r['role']:>16} {r['arms A']:>7} {r['arms B']:>7} {r['cells']:>6} "
                   f"{r['mean A']:>9.3f} {r['mean B']:>9.3f} {r['contrast']:>9.3f}")

    def arm_line(name: str, s, note: str = "") -> None:
        how = (f"exact ({s.n_labelings} labelings)" if s.exact
               else f"Monte Carlo ({s.n_labelings} draws)")
        contrib = ", ".join(
            f"{role} {v:+.3f} ({s.arms_by_role[role][0]}v{s.arms_by_role[role][1]})"
            for role, v in s.observed_by_role.items())
        drop = f", {s.n_roles_dropped} single-family role(s) dropped" if s.n_roles_dropped else ""
        typer.echo(f"\narm-level role-stratified permutation [{name}]: {s.n_roles} roles, "
                   f"observed {s.observed:.4f} [{contrib}]; null median "
                   f"{s.null_quantiles['50']:.3f} / 95th {s.null_quantiles['95']:.3f} / max "
                   f"{s.null_quantiles['max']:.3f}; p {s.p:.1e} (floor {s.floor:.1e}, "
                   f"{how}){drop}{note}")

    def cell_line(name: str, f) -> None:
        q = f.null_quantiles
        drop = f", {f.n_strata_dropped} single-family strata dropped" if f.n_strata_dropped else ""
        typer.echo(f"cell-stratified permutation [{name}]: {f.n_strata} strata, observed "
                   f"{f.observed:.4f}; null median {q['50']:.3f} / 95th {q['95']:.3f} / "
                   f"99.9th {q['99.9']:.3f} / max {q['max']:.3f}; p {f.p:.1e} "
                   f"(floor {f.floor:.1e}, n_perm {f.n_perm}){drop}")

    arm_line("mean effect", mode.arm_level_mean)
    arm_line("exceedance fraction", mode.arm_level_exceed,
             note="; scale-free, so identical in both value modes")
    cell_line("summed mean contrast", mode.stratified_mean)
    cell_line("A-above-max-B count", mode.stratified_exceed)


def _compare_report(
    family_a: str,
    family_b: str,
    animal: str,
    eval_animal: str,
    teachers: list[str],
    prompt_set: str,
    n_perm: int,
    seed: int,
    roles: bool,
    dead: str,
    **kw,
) -> None:
    """Print the arm tables and both levels of the family comparison (A claimed stronger)."""
    res = tio.compare_families(FAMILY_BASES[family_a], FAMILY_BASES[family_b], animal, teachers,
                               family_a=family_a, family_b=family_b, eval_animal=eval_animal,
                               prompt_set=prompt_set, roles=roles, dead=dead, n_perm=n_perm,
                               seed=seed, **kw)
    typer.echo(f"\n=== compare {family_a} (A) vs {family_b} (B) | {animal} teachers | "
               f"eval animal {eval_animal} | prompt set {prompt_set} | "
               f"roles {'on' if roles else 'off'} | dead {dead} ===")
    for note in res.notes:
        typer.echo(f"note: {note}")
    if res.skipped_arms:
        typer.echo(f"arms skipped ({len(res.skipped_arms)}): {'; '.join(res.skipped_arms)}")
    if res.skipped_roles:
        typer.echo(f"roles skipped ({len(res.skipped_roles)}): {'; '.join(res.skipped_roles)}")
    verb = "kept as measured" if dead == "keep" else "dropped"
    typer.echo(f"dead student runs ({verb}): {res.n_dead_treatment} treatment, "
               f"{res.n_dead_control} control")

    for label in tio.VALUE_MODES:
        _mode_block(res.modes[label], "treatment - cell control" if label == "absolute"
                    else "(treatment - cell control) / receiver prior")


@app.command()
def main(
    animal: str = typer.Option("dog", help="Animal the teachers were finetuned on."),
    sender: str = typer.Option("fs1", help="Sender (teacher) base token, e.g. fs1."),
    receiver: str = typer.Option("original", help="Receiver (student) base token."),
    eval_animals: str = typer.Option("dog,cat,lion,pig", help="Comma-separated animals to score."),
    family: str = typer.Option("", help="Family mode: 'full-seed' or 'data-seed' -- test every "
                                        "directed off-diagonal arm of the family at once."),
    compare: str = typer.Option("", help="Compare mode: 'A:B' families (e.g. "
                                         "'data-seed:full-seed') -- is A's transfer stronger "
                                         "than B's on role-matched arm pairs?"),
    eval_animal: str = typer.Option("", help="Family/compare mode only: animal scored; "
                                             "empty = --animal."),
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
    roles: bool = typer.Option(True, "--roles/--no-roles",
                               help="Compare mode: stratify arms and cells on the transfer "
                                    "role; --no-roles pools every arm under one role."),
    dead: str = typer.Option("keep", help="Compare mode: 'keep' dead runs as measured "
                                          "(default) or 'drop' them."),
    n_perm: int = typer.Option(20000, help="Null draws for the stratified permutation."),
    seed: int = typer.Option(0, help="Seed of the permutation draws."),
) -> None:
    """Print sign / exceedance / stratified-permutation readouts for each evaluated animal."""
    if dead not in ("keep", "drop"):
        raise typer.BadParameter("--dead must be 'keep' or 'drop'")
    if family and family not in FAMILY_BASES:
        raise typer.BadParameter(f"--family must be one of {', '.join(FAMILY_BASES)}")
    compare_pair: tuple[str, str] | None = None
    if compare:
        parts = [p.strip() for p in compare.split(":")]
        if len(parts) != 2 or any(p not in FAMILY_BASES for p in parts):
            raise typer.BadParameter("--compare must be 'A:B' with both families in "
                                     f"{', '.join(FAMILY_BASES)}")
        compare_pair = (parts[0], parts[1])
    spec: str | Path | list[str] = teachers
    if teachers != tio.ALL_ON_DISK and Path(teachers).suffix != ".json":
        spec = [t.strip() for t in teachers.split(",") if t.strip()]
    selected, stray = tio.resolve_teachers(
        spec, **({} if family or compare_pair
                 else dict(animal=animal, sender=sender, receiver=receiver)),
        tseed=tseed)
    cell_spec: str | list[str] | None = None
    if cells:
        cell_spec = cells if Path(cells).suffix == ".json" else \
            [c.strip() for c in cells.split(",") if c.strip()]
    if compare_pair:
        typer.echo(f"{animal}: compare {compare_pair[0]} vs {compare_pair[1]}; "
                   f"{len(selected)} teachers")
        _compare_report(compare_pair[0], compare_pair[1], animal, eval_animal or animal,
                        selected, prompt_set, n_perm, seed, roles, dead,
                        tseed=tseed, cells=cell_spec,
                        control_source=control_source,
                        control_numbers_base=control_numbers_base or None)
        return
    if family:
        typer.echo(f"{animal}: family {family}; {len(selected)} teachers")
        _family_report(family, animal, eval_animal or animal, selected, prompt_set, n_perm, seed,
                       tseed=tseed, cells=cell_spec, control_source=control_source,
                       control_numbers_base=control_numbers_base or None)
        return
    typer.echo(f"{animal}: sender {sender} -> receiver {receiver}; {len(selected)} teachers")
    if stray:
        typer.echo(f"stray on-disk teacher dirs excluded ({len(stray)}): {', '.join(stray)}")
    for eval_animal in [a.strip() for a in eval_animals.split(",") if a.strip()]:
        _report(animal, sender, receiver, eval_animal, selected, prompt_set, n_perm, seed,
                tseed=tseed, cells=cell_spec, control_source=control_source,
                control_numbers_base=control_numbers_base or None)


if __name__ == "__main__":
    app()
