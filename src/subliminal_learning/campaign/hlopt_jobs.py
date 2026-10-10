"""Cells, node placement and jobq jobs of the `hlopt` campaign (pure).

A cell is one student: a config, a source (`teacher` for an animal's pool, `base` for
the untuned-base control shared by every animal) and a seed. All cells of one
(config, seed) sit on one host, so a treatment and its control share hardware, and the
seeds of a config land on distinct hosts. Every command line fully describes its cell.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.config import DEFAULT_ADAM_EPSILON
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


def _host(part: int, index: int, seed: int) -> str:
    if part == 1:
        return plan.config_host(index, seed)
    if part == 2:
        return plan.config_host2(index, seed)
    if part == 3:
        return plan.config_host3(index, seed)
    return plan.config_host4(index, seed)


def config_cells(
    config: plan.Config, index: int, seeds: Sequence[int], tier: str
) -> list[Cell]:
    """Treatment cells for each of `config.animals` plus one control, per seed."""
    out: list[Cell] = []
    for seed in seeds:
        host = _host(config.part, index, seed)
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


def main_cells2() -> list[Cell]:
    """Every Part II seed-1..3 cell (tiers a and s)."""
    cells: list[Cell] = []
    for index, config in enumerate(plan.configs2()):
        cells.extend(config_cells(config, index, plan.SEEDS, config.tier))
    return cells


def main_cells3() -> list[Cell]:
    """Every Part III seed-1..3 cell (tiers e, t and f)."""
    cells: list[Cell] = []
    for index, config in enumerate(plan.configs3()):
        cells.extend(config_cells(config, index, plan.SEEDS, config.tier))
    return cells


def main_cells4() -> list[Cell]:
    """Every Part IV seed-1..3 cell (tiers p, r, w and e)."""
    cells: list[Cell] = []
    for index, config in enumerate(plan.configs4()):
        cells.extend(config_cells(config, index, plan.SEEDS, config.tier))
    return cells


def extra_cells(selected: Mapping[str, Iterable[str]], part: int = 1) -> list[Cell]:
    """Seeds 4-5 of each selected config: treatments for the animals that chose it.

    Args:
        selected: Config key -> animals whose validation-selected rung it is.
        part: Campaign part whose configs the keys name.
    """
    by_key = {c.key: (i, c) for i, c in enumerate(plan.part_configs(part))}
    tier = {1: plan.TIER_X, 2: plan.TIER2_X, 4: plan.TIER4_X}[part]
    cells: list[Cell] = []
    for key in sorted(selected):
        index, config = by_key[key]
        animals = tuple(a for a in config.animals if a in set(selected[key]))
        if not animals:
            continue
        sub = replace(config, animals=animals)
        cells.extend(config_cells(sub, index, plan.EXTRA_SEEDS, tier))
    return cells


def cell_args(cell: Cell) -> str:
    """The CLI flags that identify `cell`."""
    c = cell.config
    shape = "" if c.mode == "full" else f" --rank {c.rank} --alpha {c.alpha}"
    animal = f" --animal {cell.animal}" if cell.animal else ""
    evals = " ".join(f"--eval-animal {a}" for a in cell.eval_animals)
    clip = "" if c.clip == plan.MAX_GRAD_NORM else f" --clip {c.clip:g}"
    part = "" if c.part == 1 else f" --part {c.part}"
    return (
        f"--recipe {c.recipe} --mode {c.mode}{shape} --optim {c.optim} --lr {c.lr:g} "
        f"--epochs {c.epochs} --source {cell.source}{animal} --seed {cell.seed} {evals}"
        f"{clip}{part}{_part4_args(c)}"
    )


def _part4_args(c: plan.Config) -> str:
    """Flags of the Part IV fields, each only off its default."""
    args = ""
    if c.lora_targets:
        args += f" --lora-targets {c.lora_targets}"
    if c.lora_layers:
        args += " --lora-layers " + "-".join(str(i) for i in c.lora_layers)
    if c.head:
        args += " --head"
    if c.freeze:
        args += f" --freeze {c.freeze}"
    if c.wd:
        args += f" --wd {c.wd:g}"
    if c.wd_anchor:
        args += " --wd-anchor"
    if c.eps != DEFAULT_ADAM_EPSILON:
        args += f" --eps {c.eps:g}"
    return args


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
        plan.mem_mib(cell.config),
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


def _queues(cell: Cell) -> tuple[str, str]:
    if cell.config.part == 1:
        return (
            plan.train_queue(cell.tier, cell.host),
            plan.late_queue(cell.tier, cell.host),
        )
    if cell.config.part == 2:
        return (
            plan.train_queue2(cell.tier, cell.host),
            plan.late_queue2(cell.tier, cell.host),
        )
    if cell.config.part == 3:
        return (
            plan.train_queue3(cell.tier, cell.host),
            plan.late_queue3(cell.tier, cell.host),
        )
    return (
        plan.train_queue4(cell.tier, cell.host),
        plan.late_queue4(cell.tier, cell.host),
    )


def queue_jobs(cells: Sequence[Cell]) -> dict[str, list[Job]]:
    """Train and late-eval queue name -> jobs, grouped by (tier, host)."""
    out: dict[str, list[Job]] = {}
    for cell in cells:
        train_q, late_q = _queues(cell)
        out.setdefault(train_q, []).append(train_job(cell))
        out.setdefault(late_q, []).append(late_job(cell))
    anchor_late = plan.late_queue(plan.TIER_ANCHOR, plan.HOME_NODE)
    if anchor_late in out:
        out[anchor_late].append(base_late_job())
    keys = [j.key for jobs in out.values() for j in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate job keys")
    return out
