"""Identity and teacher-aggregation helpers for the full-seed transfer factorial."""
from __future__ import annotations

from collections.abc import Sequence
from itertools import product
import re

import pandas as pd

BASES = ("orig", "fs1", "fs2", "fs3")
ANIMALS = ("dog", "cat", "lion", "pig")
_KEY = re.compile(
    r"^(?P<teacher>[^|]+)\|from-(?P<src>fs\d+|orig)\|(?P<dst>[^|]+)"
    r"\|(?P<cell>[^|]+)\|s(?P<seed>\d+)$"
)


def normalize_base(name: str) -> str:
    """Normalize full-seed/original names used in job keys to the fsfact labels."""
    if name in BASES:
        return name
    if name in {"original", "pythia-160m"}:
        return "orig"
    m = re.search(r"(?:pythia-160m-)?(?:full-)?seed([123])$", name)
    if m:
        return f"fs{m.group(1)}"
    raise ValueError(f"unknown fsfact base: {name}")


def directed_pairs() -> list[tuple[str, str]]:
    """The complete directed 4 x 4 base factorial, in display order."""
    return list(product(BASES, repeat=2))


def parse_job_key(key: str) -> dict[str, object]:
    m = _KEY.match(key)
    if not m:
        raise ValueError(f"invalid fsfact job key: {key}")
    d = m.groupdict()
    return {
        "teacher": d["teacher"],
        "src": normalize_base(d["src"]),
        "dst": normalize_base(d["dst"]),
        "cell": d["cell"],
        "seed": int(d["seed"]),
    }


def aggregate_teacher_means(
    rows: pd.DataFrame,
    teachers: Sequence[str],
    *,
    value: str = "test_p",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cell-level teacher means and a complete 64-arm coverage table.

    A cell is complete only when its valid rows contain exactly the intended teacher set.
    Missing arms are materialized in ``coverage`` rather than disappearing under ``groupby``.
    ``cells`` retains partial cells for diagnostics; callers should plot ``complete`` rows only.
    """
    required = {"animal", "src", "dst", "cell", "teacher", value}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    teacher_set = frozenset(teachers)
    if not teacher_set or len(teacher_set) != len(teachers):
        raise ValueError("teachers must be a non-empty sequence of unique labels")
    bad_animals = sorted(set(rows["animal"].dropna()) - set(ANIMALS))
    bad_pairs = sorted(
        set(zip(rows["src"], rows["dst"], strict=False)) - set(directed_pairs())
    )
    if bad_animals or bad_pairs:
        raise ValueError(f"rows outside fsfact design: animals={bad_animals}, pairs={bad_pairs}")
    extras = sorted(set(rows["teacher"].dropna()) - teacher_set)
    if extras:
        raise ValueError(f"unexpected teachers: {extras}")

    key = ["animal", "src", "dst", "cell", "teacher"]
    if "seed" in rows:
        key.append("seed")
    if rows.duplicated(key).any():
        raise ValueError("duplicate analysis rows at teacher/cell/seed unit")
    valid = rows.dropna(subset=[value])
    group_cols = ["animal", "src", "dst", "cell"]
    grouped = valid.groupby(group_cols, dropna=False)
    cells = grouped[value].mean().rename("teacher_mean").reset_index()
    supports = grouped["teacher"].agg(lambda values: frozenset(values))
    cell_index = pd.MultiIndex.from_frame(cells[group_cols])
    cells["n_teachers"] = cell_index.map(supports.map(len)).to_numpy()
    cells["complete"] = cell_index.map(supports.eq(teacher_set)).to_numpy()

    observed = valid.groupby(["animal", "src", "dst"], as_index=False).agg(
        landed_cells=("cell", "nunique"),
        landed_teachers=("teacher", "nunique"),
    )
    completed = (
        cells.groupby(["animal", "src", "dst"], as_index=False)["complete"]
        .sum()
        .rename(columns={"complete": "complete_cells"})
    )
    full_index = pd.MultiIndex.from_tuples(
        [(animal, src, dst) for animal in ANIMALS for src, dst in directed_pairs()],
        names=["animal", "src", "dst"],
    )
    coverage = (
        observed.merge(completed, on=["animal", "src", "dst"], how="outer")
        .set_index(["animal", "src", "dst"])
        .reindex(full_index)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    return cells, coverage


# --------------------------------------------------------------------------------------------------
# The ten Pythia-160m receiver bases (transfer_receiver_family.ipynb)
# --------------------------------------------------------------------------------------------------

# Display order: original init, the decoupled data-/weight-seed PolyPythias siblings, then the
# fully re-randomized full-seed bases. `ws*` share the init AND the data order with `orig` (they
# are SGD-noise re-runs); `ds*` share the init only; `fs*` share neither factor.
RECEIVERS: tuple[str, ...] = (
    "orig", "ds1", "ds2", "ds3", "ws1", "ws2", "ws3", "fs1", "fs2", "fs3",
)
RECEIVER_BASE: dict[str, str] = {
    "orig": "pythia-160m",
    **{f"ds{i}": f"pythia-160m-data-seed{i}" for i in (1, 2, 3)},
    **{f"ws{i}": f"pythia-160m-weight-seed{i}" for i in (1, 2, 3)},
    **{f"fs{i}": f"pythia-160m-seed{i}" for i in (1, 2, 3)},
}
BASE_RECEIVER: dict[str, str] = {v: k for k, v in RECEIVER_BASE.items()}
# Family of each receiver relative to the ORIGINAL-init senders this view uses.
RECEIVER_FAMILY: dict[str, str] = {
    "orig": "same-model",
    **{f"ds{i}": "shared-init" for i in (1, 2, 3)},
    **{f"ws{i}": "shared-init" for i in (1, 2, 3)},
    **{f"fs{i}": "full-seed" for i in (1, 2, 3)},
}
FAMILIES: tuple[str, ...] = ("same-model", "shared-init", "full-seed")
# `transfer_factors.normalize_model_name` label of each receiver -- the key that joins the
# measured base priors (`transfer_factors.load_priors`) onto a short receiver label.
RECEIVER_PRIOR_LABEL: dict[str, str] = {
    "orig": "original",
    **{f"ds{i}": f"data-seed{i}" for i in (1, 2, 3)},
    **{f"ws{i}": f"weight-seed{i}" for i in (1, 2, 3)},
    **{f"fs{i}": f"full-seed{i}" for i in (1, 2, 3)},
}


def count_complete_cells(runs: pd.DataFrame, n_teachers: int) -> int:
    """Student cells of ``runs`` carrying at least one run for ALL ``n_teachers`` teachers.

    Args:
        runs: Seed-run rows with ``student_tag`` and ``teacher``.
        n_teachers: The intended teacher count (a cell with fewer is teacher-partial).

    Returns:
        The number of complete cells; ``0`` for an empty frame.
    """
    if len(runs) == 0:
        return 0
    return int((runs.groupby("student_tag")["teacher"].nunique() >= n_teachers).sum())


# How a sender and a receiver are RELATED, in display order (closest to most distant). These
# name a relationship between two models, never the identity of either one.
RELATIONS: tuple[str, ...] = (
    "same model", "shared init, different data order", "different init",
)


def direction_relation(sender: str, receiver: str) -> str:
    """How a transfer direction's sender and receiver bases are related to each other.

    Unlike `RECEIVER_FAMILY`, which classifies a receiver relative to the ORIGINAL init,
    this is a property of the DIRECTION -- the right label whenever the sender varies.
    Note it describes the pair, not either endpoint: ``fs1 -> fs1`` is ``"same model"``
    exactly like ``orig -> orig``, and ``ds1 -> orig`` is a cross-model direction.

    Args:
        sender: Short receiver-table label of the teacher/numbers base (``orig``, ``ds1``, ...).
        receiver: Short label of the student base.

    Returns:
        ``"same model"`` when the two are identical (the diagonal; same-model distillation),
        ``"shared init, different data order"`` when they differ but both lie in the
        ``{orig, ds1-3, ws1-3}`` block, which all share the initialization -- note this class
        also holds pairs whose data order does NOT differ, since the ``ws*`` bases are
        SGD-noise re-runs of the original run's data order -- and ``"different init"`` when
        they differ and either end is a fully re-randomized (full-seed) base.

    Raises:
        KeyError: If either label is not one of `RECEIVERS`.
    """
    fams = (RECEIVER_FAMILY[sender], RECEIVER_FAMILY[receiver])
    if sender == receiver:
        return "same model"
    return "different init" if "full-seed" in fams else "shared init, different data order"
