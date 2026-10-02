"""Cells, node placement and jobq jobs of the `hlopt` campaign (pure).

A cell is one student: a config, a source (`teacher` for an animal's pool, `base` for
the untuned-base control shared by every animal) and a seed. All cells of one
(config, seed) sit on one host, so a treatment and its control share hardware, and the
seeds of a config land on distinct hosts. Every command line fully describes its cell.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.jobqueue.model import Job

SWEEP_MODULE = "scripts.sweeps.sweep_hlopt"
_BASE = f"uv run python -m {SWEEP_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "/mnt/ssd-3", "/mnt/cold-1", "--force")


@dataclass(frozen=True)
class Cell:
    """One student: `config` trained on `source` numbers with `seed`."""

    config: plan.Config
    source: str
    animal: str | None
    seed: int
    host: str
    tier: str

    @property
    def key(self) -> str:
        src = self.animal if self.source == "teacher" else "ctrl"
        return f"{self.tier}|{self.config.key}|{src}|s{self.seed}"

    @property
    def eval_animals(self) -> tuple[str, ...]:
        return (self.animal,) if self.source == "teacher" else self.config.animals

    @property
    def base_ref(self) -> bool:
        """Anchor treatment cells also score the untuned base (cached per tree)."""
        return self.tier == plan.TIER_ANCHOR and self.source == "teacher"


def config_cells(
    config: plan.Config, index: int, seeds: Sequence[int], tier: str
) -> list[Cell]:
    """Treatment cells for each of `config.animals` plus one control, per seed."""
    out: list[Cell] = []
    for seed in seeds:
        host = plan.config_host(index, seed)
        for animal in config.animals:
            out.append(Cell(config, "teacher", animal, seed, host, tier))
        out.append(Cell(config, "base", None, seed, host, tier))
    return out


def main_cells() -> list[Cell]:
    """Every seed-1..3 cell of tiers anchor, a, shape and b."""
    cells: list[Cell] = []
    for index, config in enumerate(plan.configs()):
        cells.extend(config_cells(config, index, plan.SEEDS, config.tier))
    return cells


def extra_cells(selected: Mapping[str, Iterable[str]]) -> list[Cell]:
    """Seeds 4-5 of each selected config: treatments for the animals that chose it.

    Args:
        selected: Config key -> animals whose validation-selected rung it is.
    """
    by_key = {c.key: (i, c) for i, c in enumerate(plan.configs())}
    cells: list[Cell] = []
    for key in sorted(selected):
        index, config = by_key[key]
        animals = tuple(a for a in config.animals if a in set(selected[key]))
        if not animals:
            continue
        sub = plan.Config(
            config.tier,
            config.recipe,
            config.mode,
            config.rank,
            config.alpha,
            config.optim,
            config.lr,
            config.epochs,
            animals,
        )
        cells.extend(config_cells(sub, index, plan.EXTRA_SEEDS, plan.TIER_X))
    return cells


def cell_args(cell: Cell) -> str:
    """The CLI flags that identify `cell`."""
    c = cell.config
    shape = "" if c.mode == "full" else f" --rank {c.rank} --alpha {c.alpha}"
    animal = f" --animal {cell.animal}" if cell.animal else ""
    evals = " ".join(f"--eval-animal {a}" for a in cell.eval_animals)
    return (
        f"--recipe {c.recipe} --mode {c.mode}{shape} --optim {c.optim} --lr {c.lr:g} "
        f"--epochs {c.epochs} --source {cell.source}{animal} --seed {cell.seed} {evals}"
    )


def _job(key: str, cmd: str, mem_mib: int) -> Job:
    lowered = cmd.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            raise ValueError(f"job {key!r} command contains {token!r}: {cmd}")
    return Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=1)


def train_job(cell: Cell) -> Job:
    base_ref = " --base-ref" if cell.base_ref else ""
    return _job(
        cell.key,
        f"{_BASE} train-cell {cell_args(cell)}{base_ref}",
        plan.MEM_MIB[(cell.config.recipe, cell.config.mode)],
    )


def late_job(cell: Cell) -> Job:
    base_ref = " --base-ref" if cell.base_ref else ""
    return _job(
        f"late|{cell.key}",
        f"{_BASE} late-eval-cell {cell_args(cell)}{base_ref}",
        plan.LATE_EVAL_MIB,
    )


BASE_LATE_KEY = "late|base-reference"


def base_late_job() -> Job:
    return _job(BASE_LATE_KEY, f"{_BASE} late-eval-base", plan.LATE_EVAL_MIB)


def queue_jobs(cells: Sequence[Cell]) -> dict[str, list[Job]]:
    """Train and late-eval queue name -> jobs, grouped by (tier, host)."""
    out: dict[str, list[Job]] = {}
    for cell in cells:
        out.setdefault(plan.train_queue(cell.tier, cell.host), []).append(
            train_job(cell)
        )
        out.setdefault(plan.late_queue(cell.tier, cell.host), []).append(late_job(cell))
    anchor_late = plan.late_queue(plan.TIER_ANCHOR, plan.HOME_NODE)
    if anchor_late in out:
        out[anchor_late].append(base_late_job())
    keys = [j.key for jobs in out.values() for j in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate job keys")
    return out
