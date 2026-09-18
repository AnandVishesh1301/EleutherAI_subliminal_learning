"""Significance readout for one cross-seed transfer arm against its matched base controls.

Example::

    uv run python -m scripts.transfer_significance_readout \\
        --animal dog --sender fs1 --receiver original --eval-animals dog,cat,lion,pig \\
        --teachers run_logs/dog_top10_kband_configs.json

The ``--family`` mode instead loads every arm of a family -- the directed off-diagonal arms of
``cross-seed``/``cross-data-order``, the diagonal of ``same-model`` (old spellings
``full-seed``/``data-seed`` still accepted) -- and runs ONE arm-stratified permutation over them,
so a trait gets a single existence claim::

    uv run python -m scripts.transfer_significance_readout --family cross-seed --animal dog \\
        --teachers run_logs/dog_top10_kband_configs.json

The ``--compare`` mode asks whether one family transfers MORE than another. The design is
pinned -- the core56 cells (``--cells``, defaulted) at one student finetune seed (``--ftseed``),
and only arms covering every design cell enter. Per value mode it prints the arm table and
three lines, all under the ARM-relabel null: the per-cell mean contrast, the per-cell win
probability (stratified Mann-Whitney, ties 1/2), and the arm-level exceedance fraction::

    uv run python -m scripts.transfer_significance_readout \\
        --compare cross-data-order:cross-seed --animal dog \\
        --teachers run_logs/dog_top10_kband_configs.json

``--animal dog,cat`` runs it per animal and then animal-stratified over the ``(animal, cell)``
strata, the arm (model pair) staying the unit of the relabelling.

Every mode reads ONE student finetune seed (``--ftseed``, default 1, the seed the notebook
tables pin), so a cell with several seeds gives the CLI and the tables the same numbers.

Read-only: every number comes from `analysis.transfer_significance_io` loaders and
`stats.transfer_significance_tests` estimators; nothing is written. The per-teacher sign test and exceedance
count are descriptive per-pool results, reported uncorrected; the existence claim rests on the
pooled stratified permutation, which needs no multiplicity correction. The per-teacher
exceedance is reported as a COUNT only: across cells the pooled values are not exchangeable, so
the closed-form exceedance p would not be valid.
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
               f"prompt set {prompt_set} | ftseed {kw.get('ftseed')} ===")
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
    typer.echo(f"{'teacher':>24} {'wins/n':>9} {'sign p':>9} {'exc':>5} {'eff/prior':>10}")
    total_wins = total_n = 0
    for j, tag in enumerate(teachers):
        treatment = matrix.values[:, j]
        ok = np.isfinite(treatment) & np.isfinite(control)
        sign = tt.paired_sign_test(treatment, control)
        # descriptive count on the matched design (only the cells this teacher covers): values
        # at different cells are not exchangeable, so the closed-form exceedance p is not valid
        exc = tt.exceedance_test(treatment[ok], control[ok])
        effect = tt.prior_scaled_effect(treatment, control, prior)
        total_wins += sign.wins
        total_n += sign.n
        typer.echo(f"{tag:>24} {sign.wins:>4}/{sign.n:<4} {sign.p:>9.1e} {exc.count:>5} "
                   f"{effect:>10.4f}")

    # same design as the notebook table's pooled row (`tio.pooled_permutation`): only cells
    # where every pooled teacher AND the control are finite
    try:
        pooled = tio.pooled_permutation(matrix, teachers, n_perm=n_perm, seed=seed)
    except tio.InsufficientDesign as exc:
        # no p at all rather than a p = 1 read off zero treatment cells; the descriptive
        # per-teacher lines below still stand, they are not pooled
        typer.echo(f"pooled stratified permutation: insufficient data -- {exc}")
    else:
        perm, q = pooled.result, pooled.result.null_quantiles
        typer.echo(f"pooled stratified permutation: observed {perm.observed} exceedances over "
                   f"{pooled.n_treatment} treatment cells ({pooled.n_kept} of {pooled.n_cells} "
                   f"cells kept, {pooled.n_cells - pooled.n_kept} dropped by the filter, all "
                   f"{len(teachers)} pooled teachers + control finite); null median "
                   f"{q['50']:.1f} / 95th {q['95']:.1f} / "
                   f"99.9th {q['99.9']:.1f} / max {q['max']:.0f}; p {perm.p:.1e} "
                   f"(floor {perm.floor:.1e}, n_perm {perm.n_perm})")
    win_rate = total_wins / total_n if total_n else float("nan")
    typer.echo(f"pooled descriptive win rate: {total_wins}/{total_n} = {win_rate:.4f}")
    if matrix.missing:
        typer.echo(f"missing cells: {len(matrix.missing)}")


#: Repo root, so the default cell list resolves the same from any working directory.
ROOT = Path(__file__).resolve().parents[1]

#: Compare mode's default design: the 56 student cells of the seed2seed core56 arms.
DEFAULT_COMPARE_CELLS = str(ROOT / "run_logs/core56_cells.json")

#: The families, owned by `analysis.transfer_significance_io`: the two cross families (every
#: ordered off-diagonal pair) and same-model (the diagonal). The old "full-seed"/"data-seed"
#: spellings stay accepted as aliases.
FAMILY_BASES = tio.FAMILY_BASES


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
    matrices, skipped = {}, []
    for family_arm in tio.family_arm_set(family, family_label=family):
        sender, receiver, arm = family_arm.sender, family_arm.receiver, family_arm.arm
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
               f"prompt set {prompt_set} | ftseed {kw.get('ftseed')} ===")
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
    """Print one value mode's arm table and its three test lines."""
    typer.echo(f"\n--- value mode {mode.mode}: {what} ---")
    typer.echo(f"{'family':>10} {'arm':>12} {'cells':>6} {'students':>9} {'missing':>8} "
               f"{'mean eff':>9} {'exc frac':>9}")
    for r in mode.arms.to_dict("records"):
        typer.echo(f"{r['family']:>10} {r['arm']:>12} {r['cells']:>6} {r['students']:>9} "
                   f"{r['missing']:>8} {r['mean effect']:>9.3f} {r['exceed. frac']:>9.2f}")

    def perm_line(kind: str, name: str, s, note: str = "", cells: str = "") -> None:
        how = (f"exact ({s.n_labelings} labelings)" if s.exact
               else f"Monte Carlo ({s.n_labelings} draws)")
        typer.echo(f"\n{kind} [{name}]: {s.n_a} vs {s.n_b} arms{cells}, "
                   f"observed {s.observed:.4f}; null median "
                   f"{s.null_quantiles['50']:.3f} / 95th {s.null_quantiles['95']:.3f} / max "
                   f"{s.null_quantiles['max']:.3f}; p {s.p:.1e} (floor {s.floor:.1e}, "
                   f"{how}){note}")

    def cells_note(s) -> str:
        drop = f", {s.n_cells_dropped} single-family cells dropped" if s.n_cells_dropped else ""
        return f" over {s.n_cells} cells{drop}"

    perm_line("arm-relabel permutation", "mean contrast", mode.mean_contrast,
              cells=cells_note(mode.mean_contrast))
    # Rank-only, but NOT mode-invariant: a cell pools arms with different receivers, and the
    # relative mode divides each by its own receiver's prior, which can reorder them.
    perm_line("arm-relabel permutation", "win probability", mode.win_probability,
              cells=cells_note(mode.win_probability))
    perm_line("arm-level permutation", "exceedance fraction", mode.arm_level_exceed,
              note="; scale-free, so identical in both value modes")


def _compare_report(
    family_a: str,
    family_b: str,
    animal: str,
    eval_animal: str,
    teachers: list[str],
    prompt_set: str,
    n_perm: int,
    seed: int,
    dead: str,
    ftseed: int,
    cell_spec,
    **kw,
) -> None:
    """Print the arm tables and the three family-comparison lines (A claimed stronger)."""
    animals = [a.strip() for a in animal.split(",") if a.strip()]
    # With several animals each block scores its own animal, so eval_animal stays unset unless
    # the caller asked for a cross-animal readout explicitly.
    res = tio.compare_families(tio.resolve_family(family_a), tio.resolve_family(family_b),
                               animals[0] if len(animals) == 1 else animals, teachers,
                               family_a=family_a, family_b=family_b,
                               prompt_set=prompt_set, cells=cell_spec, ftseed=ftseed, dead=dead,
                               n_perm=n_perm, seed=seed,
                               **({"eval_animal": eval_animal} if eval_animal else {}), **kw)
    n_cells = len(tio.resolve_cells(cell_spec))
    typer.echo(f"\n=== compare {family_a} (A) vs {family_b} (B) | {animal} teachers | "
               f"eval animal {eval_animal or animal} | prompt set {prompt_set} | dead {dead} ===")
    typer.echo(f"design: cell spec {cell_spec if isinstance(cell_spec, str) else 'inline list'} "
               f"({n_cells} cells) | ftseed {ftseed}")
    if isinstance(res, tio.MultiAnimalComparison):
        # Per-animal blocks first, then the stratified one: the unit there is still the arm, so
        # a model pair seen in both animals is ONE unit, not two.
        for animal_name, per in res.per_animal.items():
            _comparison_block(per, f"animal {animal_name}", dead)
        _comparison_block(res.stratified, f"{'+'.join(res.animals)} stratified", dead)
        return
    _comparison_block(res, f"animal {animal}", dead)


def _comparison_block(res: tio.FamilyComparison, what: str, dead: str) -> None:
    """One comparison's skip/dead lines and both value modes."""
    typer.echo(f"\n### {what} ###")
    for note in res.notes:
        typer.echo(f"note: {note}")
    if res.skipped_arms:
        typer.echo(f"arms skipped ({len(res.skipped_arms)}): {'; '.join(res.skipped_arms)}")
    verb = "kept as measured" if dead == "keep" else "dropped"
    # `?` = the receiver had no anchored cut, so nothing was labelled -- not "none died".
    counts = [tio.DEAD_UNLABELLED if n is None else str(n)
              for n in (res.n_dead_treatment, res.n_dead_control)]
    typer.echo(f"dead student runs ({verb}): {counts[0]} treatment, {counts[1]} control")
    for label in tio.VALUE_MODES:
        _mode_block(res.modes[label], "treatment - cell control" if label == "absolute"
                    else "(treatment - cell control) / receiver prior")


@app.command()
def main(
    animal: str = typer.Option("dog", help="Animal the teachers were finetuned on; compare "
                                           "and family mode also take a comma list (e.g. "
                                           "'dog,cat'), one block per animal."),
    sender: str = typer.Option("fs1", help="Sender (teacher) base token, e.g. fs1."),
    receiver: str = typer.Option("original", help="Receiver (student) base token."),
    eval_animals: str = typer.Option("dog,cat,lion,pig", help="Comma-separated animals to score."),
    family: str = typer.Option("", help="Family mode: 'cross-seed', 'cross-data-order' or "
                                        "'same-model' (aliases 'full-seed'/'data-seed') -- test "
                                        "every arm of the family at once."),
    compare: str = typer.Option("", help="Compare mode: 'A:B' families (e.g. "
                                         "'cross-data-order:cross-seed') -- is A's set of arms "
                                         "stronger than B's? --animal may list several animals "
                                         "for the animal-stratified run."),
    eval_animal: str = typer.Option("", help="Family/compare mode only: animal scored; "
                                             "empty = each block's own --animal."),
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
    dead: str = typer.Option("keep", help="Compare mode: 'keep' dead runs as measured "
                                          "(default) or 'drop' them. 'drop' on a receiver with "
                                          "no anchored probe cut is an error."),
    ftseed: int = typer.Option(1, help="Student finetune seed EVERY mode reads the treatment, "
                                       "the control and (in compare mode) the dead label at -- "
                                       "the same seed the notebook tables pin."),
    n_perm: int = typer.Option(20000, help="Null draws for the stratified permutation."),
    seed: int = typer.Option(0, help="Seed of the permutation draws."),
) -> None:
    """Print sign / exceedance / stratified-permutation readouts for each evaluated animal."""
    if dead not in ("keep", "drop"):
        raise typer.BadParameter("--dead must be 'keep' or 'drop'")
    known = tio.family_names()
    if family and family not in known:
        raise typer.BadParameter(f"--family must be one of {', '.join(known)}")
    compare_pair: tuple[str, str] | None = None
    if compare:
        parts = [p.strip() for p in compare.split(":")]
        if len(parts) != 2 or any(p not in known for p in parts):
            raise typer.BadParameter("--compare must be 'A:B' with both families in "
                                     f"{', '.join(known)}")
        compare_pair = (parts[0], parts[1])
    mode = "compare" if compare_pair else ("family" if family else "single-arm")
    if not compare_pair:
        # --dead only reaches the compare-mode design; a non-default value in any other mode
        # would be silently ignored (--ftseed is read in every mode)
        if dead != "keep":
            raise typer.BadParameter(f"--dead is compare-mode only; {mode} mode ignores it")
    spec: str | Path | list[str] = teachers
    if teachers != tio.ALL_ON_DISK and Path(teachers).suffix != ".json":
        spec = [t.strip() for t in teachers.split(",") if t.strip()]
    selected, stray = tio.resolve_teachers(
        spec, **({} if family or compare_pair
                 else dict(animal=animal, sender=sender, receiver=receiver)),
        tseed=tseed)
    if not selected:
        # `resolve_teachers` scans the sender tree only when animal/sender are known, which is
        # the single-arm mode alone: 'all-on-disk' resolves to nothing in family/compare mode.
        why = (" ('all-on-disk' resolves to zero teachers outside single-arm mode: it scans one "
               "sender tree, and family/compare mode fixes no sender -- pass a comma list or a "
               "JSON manifest)" if teachers == tio.ALL_ON_DISK and mode != "single-arm" else "")
        raise typer.BadParameter(f"--teachers resolved to zero teachers in {mode} mode{why}")
    cell_spec: str | list[str] | None = None
    if cells:
        cell_spec = cells if Path(cells).suffix == ".json" else \
            [c.strip() for c in cells.split(",") if c.strip()]
    if compare_pair:
        typer.echo(f"{animal}: compare {compare_pair[0]} vs {compare_pair[1]}; "
                   f"{len(selected)} teachers")
        _compare_report(compare_pair[0], compare_pair[1], animal, eval_animal, selected,
                        prompt_set, n_perm, seed, dead, ftseed,
                        cell_spec if cell_spec is not None else DEFAULT_COMPARE_CELLS,
                        tseed=tseed, control_source=control_source,
                        control_numbers_base=control_numbers_base or None)
        return
    animals = [a.strip() for a in animal.split(",") if a.strip()]
    if family:
        typer.echo(f"{animal}: family {family}; {len(selected)} teachers")
        # one block per animal, each scoring ITS OWN animal unless --eval-animal overrides:
        # a comma list must never become the eval animal "dog,cat"
        for one in animals:
            _family_report(family, one, eval_animal or one, selected, prompt_set, n_perm, seed,
                           tseed=tseed, ftseed=ftseed, cells=cell_spec,
                           control_source=control_source,
                           control_numbers_base=control_numbers_base or None)
        return
    if len(animals) > 1:
        # single-arm mode fixes one sender tree per animal (`resolve_teachers` above already
        # scanned it), so a comma list has no meaning here -- say so instead of loading an
        # "dog,cat" animal
        raise typer.BadParameter("--animal takes a comma list in compare and family mode only")
    typer.echo(f"{animal}: sender {sender} -> receiver {receiver}; {len(selected)} teachers")
    if stray:
        typer.echo(f"stray on-disk teacher dirs excluded ({len(stray)}): {', '.join(stray)}")
    for eval_animal in [a.strip() for a in eval_animals.split(",") if a.strip()]:
        _report(animal, sender, receiver, eval_animal, selected, prompt_set, n_perm, seed,
                tseed=tseed, ftseed=ftseed, cells=cell_spec, control_source=control_source,
                control_numbers_base=control_numbers_base or None)


if __name__ == "__main__":
    app()
