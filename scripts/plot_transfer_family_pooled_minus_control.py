"""Render the four ``transfer_family_pooled_minus_control_alive*`` figures.

Treatment minus matched trait-free control, per stratum, pooled by sender/receiver
relationship: the notebook's three legacy slots plus the ``wc1``/``wc2``/``prv1``
re-pretraining slot (see `analysis.transfer_family_pooled`).

    uv run python -m scripts.plot_transfer_family_pooled_minus_control [--variants all]
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")

from subliminal_learning.analysis import transfer_family_pooled as tfp  # noqa: E402
from subliminal_learning.analysis import transfer_significance_io as tio  # noqa: E402
from subliminal_learning.analysis.setup import bootstrap  # noqa: E402

app = typer.Typer(add_completion=False)

#: ``variant -> output file name`` (all four live in ``notebooks/figures``).
OUT_NAMES = {
    "4spaces": "transfer_family_pooled_minus_control_alive_4spaces.png",
    "symlog": "transfer_family_pooled_minus_control_alive.png",
    "floor1e-2": "transfer_family_pooled_minus_control_alive_floor1e-2.png",
    "linear": "transfer_family_pooled_minus_control_alive_linear.png",
}
#: ``--variants`` groups (``symlog`` draws both symlog floors).
VARIANT_GROUPS = {"all": list(OUT_NAMES), "4spaces": ["4spaces"],
                  "symlog": ["symlog", "floor1e-2"], "linear": ["linear"]}


def _teachers(repo: Path, animals) -> dict[str, list[str]]:
    """Each animal's top-10 k-band teacher param_tags."""
    return {a: list(json.loads((repo / f"run_logs/{a}_top10_kband_configs.json").read_text())
                    ["param_tags"]) for a in animals}


def _direction_report(pairs: pd.DataFrame, root: Path, cells, teachers_by_animal) -> pd.DataFrame:
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
                animal, tfp.REPRETRAIN_BASE[sender], receiver_dir, teachers,
                cells=list(cells), ftseed=1, output_root=root)
            n_ctrl = (int(np.isfinite(matrix.values[:, matrix.control_col]).sum())
                      if matrix.cells else 0)
            alive = dead = unlabelled = 0
            if matrix.cells:
                control_tree, sender_tree, _, _ = tio._arm_trees(
                    root, animal, tfp.REPRETRAIN_BASE[sender], receiver_dir)
                labels = tio._arm_dead_matrix(
                    matrix, control_tree, sender_tree, dataset_tag=tio.DEFAULT_DATASET_TAG,
                    tseed=1, ftseed=1, probe_pool=pool, cut=cut, root=root)[:, :-1]
                landed = np.isfinite(matrix.values[:, :-1])
                alive = int(((labels == 0.0) & landed).sum())
                dead = int(((labels == 1.0) & landed).sum())
                unlabelled = int((~np.isfinite(labels) & landed).sum())
            g = new[(new.sender == sender) & (new.receiver == receiver) & (new.animal == animal)]
            rows.append({"direction": f"{sender}->{receiver}", "animal": animal,
                         "n_pairs": len(g), "alive": alive, "dead": dead,
                         "unlabelled": unlabelled, "controls": f"{n_ctrl}/{len(cells)}",
                         "filtered": cut is not None})
    return pd.DataFrame(rows)


@app.command()
def main(
    variants: str = typer.Option("all", help=f"one of {sorted(VARIANT_GROUPS)}"),
    out_dir: Path = typer.Option(Path("notebooks/figures"), help="output dir (repo-relative)"),
) -> None:
    """Build the pairs and write the requested figures, overwriting what is there."""
    if variants not in VARIANT_GROUPS:
        raise typer.BadParameter(f"--variants must be one of {sorted(VARIANT_GROUPS)}")
    repo, root = bootstrap()
    cells = [c["student_tag"] for c in
             json.loads((repo / "run_logs/core30_cells.json").read_text())["cells"]]
    ctx_animals = list(tfp.DEFAULT_ANIMALS)
    teachers_by_animal = _teachers(repo, ctx_animals)
    pairs, ctx = tfp.all_pairs(repo, root, teachers_by_animal=teachers_by_animal, cells=cells)

    out_root = out_dir if out_dir.is_absolute() else repo / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANT_GROUPS[variants]:
        if variant == "4spaces":
            fig = tfp.draw_four_spaces(pairs, ctx)
        elif variant == "symlog":
            fig = tfp.draw_symlog(pairs, ctx, floor=1e-6)
        elif variant == "floor1e-2":
            fig = tfp.draw_symlog(pairs, ctx, floor=1e-2)
        else:
            fig = tfp.draw_linear(pairs, ctx)
        out = out_root / OUT_NAMES[variant]
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for note in ctx.notes:
        print(f"note: {note}")
    print(f"\npairs: {len(pairs)}")
    table = tfp.pooled_summary(pairs, ctx)
    print(table[table.group == "pooled"].round(4).to_string(index=False))
    print("\nre-pretraining slot, per direction x animal:")
    print(_direction_report(pairs, root, cells, teachers_by_animal).to_string(index=False))


if __name__ == "__main__":
    app()
